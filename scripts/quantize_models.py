"""Make fp16 and int8 variants of the browser models (web/models/*.onnx).

    uv run --no-project --with onnx --with onnxruntime --with sympy --with packaging \
        --with opencv-python --with numpy python scripts/quantize_models.py

Writes <out>/{detector,classifier}_{fp16,int8dyn,int8qdq}.onnx. With
--install fp16, also copies that variant next to the fp32 models and points
<models>/meta.json at it (the browser app loads whatever meta.json names). int8qdq is
static quantization calibrated on synthetic *training* images/crops (never the
evaluation data); int8dyn quantizes weights only (activations stay float).
"""

from __future__ import annotations

import argparse
import random
import re
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.transformers.float16 import convert_float_to_float16


def letterbox(img: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255


class Reader(CalibrationDataReader):
    def __init__(self, name: str, batches: list[np.ndarray]):
        self.it = iter([{name: b} for b in batches])

    def get_next(self):
        return next(self.it, None)


def float_head(model_path: Path) -> list[str]:
    """Detector nodes to keep in float for static int8: the output packs box
    coords (0-640), scores (0-1) and keypoints into one tensor, and one 8-bit
    scale over that range rounds every score to 0. Keep the head's decode ops and
    the final prediction convs feeding them in float (like Ultralytics' own int8
    exports)."""
    names = [n.name for n in onnx.load(str(model_path)).graph.node]
    decode = re.compile(r"^/model\.23/[A-Za-z]+(_\d+)?$")
    final_convs = re.compile(r"^/model\.23/(cv2\.\d/cv2\.\d\.2|cv3\.\d/cv3\.\d\.2|cv4_kpts\.\d)/Conv$")
    return [n for n in names if decode.match(n) or final_convs.match(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("web/models"))
    ap.add_argument("--out", type=Path, default=Path("web/models/variants"))
    ap.add_argument("--synth", type=Path, default=Path("data/synth/train/images"))
    ap.add_argument("--crops", type=Path, default=Path("data/crops/train/images"))
    ap.add_argument("--n-images", type=int, default=128)
    ap.add_argument("--n-crops", type=int, default=512)
    ap.add_argument("--install", choices=["fp32", "fp16", "int8qdq"], default=None,
                    help="make the browser app use this variant (fp32 = the original export)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(0)

    # skip hidden files (macOS copies leave ._* AppleDouble files next to the images)
    listing = lambda d: sorted(q for q in d.iterdir() if q.suffix.lower() in (".jpg", ".png") and not q.name.startswith("."))  # noqa: E731
    imgs = listing(a.synth)
    det_cal = [letterbox(cv2.imread(str(p))) for p in rnd.sample(imgs, a.n_images)]
    crops = listing(a.crops)
    c = [cv2.imread(str(p))[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255
         for p in rnd.sample(crops, a.n_crops)]
    cls_cal = [np.stack(c[i:i + 32]) for i in range(0, len(c), 32)]

    for name, cal in (("detector", det_cal), ("classifier", cls_cal)):
        src = a.models / f"{name}.onnx"
        inp = onnx.load(str(src)).graph.input[0].name
        # onnxruntime's converter (onnxconverter-common's mistypes the detector's Resize casts).
        # GlobalAveragePool stays fp32: WebGPU sums in fp16, and the classifier's
        # squeeze-excite pools over up to 128x80 values -> 153/155 real cards misread.
        m16 = convert_float_to_float16(onnx.load(str(src)), keep_io_types=True, op_block_list=["GlobalAveragePool"])
        onnx.save(m16, str(a.out / f"{name}_fp16.onnx"))
        pre = a.out / f"{name}_pre.onnx"
        quant_pre_process(str(src), str(pre))
        quantize_dynamic(str(pre), str(a.out / f"{name}_int8dyn.onnx"), weight_type=QuantType.QInt8)
        exclude = float_head(pre) if name == "detector" else []
        quantize_static(str(pre), str(a.out / f"{name}_int8qdq.onnx"), Reader(inp, cal), nodes_to_exclude=exclude,
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax)
        pre.unlink()
        print(name, "done")

    if a.install:
        meta_path = a.models / "meta.json"
        meta = json.loads(meta_path.read_text())
        for name in ("detector", "classifier"):
            file = f"{name}.onnx" if a.install == "fp32" else f"{name}_{a.install}.onnx"
            if a.install != "fp32":
                shutil.copy(a.out / file, a.models / file)
            meta[name].update(file=file, precision=a.install, size_mb=round((a.models / file).stat().st_size / 1e6, 2))
        meta_path.write_text(json.dumps(meta, indent=1))
        print("app now uses", a.install, {n: meta[n]["file"] for n in ("detector", "classifier")})


if __name__ == "__main__":
    main()

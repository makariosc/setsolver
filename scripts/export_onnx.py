"""Export the detector and classifier to ONNX for the browser app (web/), and
check that ONNX Runtime reproduces PyTorch on real photos.

    .venv-train/bin/python scripts/export_onnx.py \
        --detector runs/pose/runs/detector/y26n_640/weights/best.pt \
        --classifier runs/classify/effb0/weights/best.pt \
        --photos data/real/setchecker_samples --out web/models

Writes <out>/detector.onnx, <out>/classifier.onnx and <out>/meta.json.

The classifier graph takes RGB crops scaled to 0..1 ([N, 3, 256, 160]) and
returns per-attribute probabilities ([N, 4, 3]): ImageNet normalization and
softmax are baked in, so the browser does no extra math.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from ultralytics import YOLO

from setsolver.classifier import ATTRS, MEAN, STD, load
from setsolver.crops import warp_card


class ClassifierForExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):  # x: RGB in 0..1
        return self.model((x - self.mean) / self.std).softmax(-1)


def export_detector(weights: Path, out: Path, imgsz: int) -> dict:
    model = YOLO(str(weights))
    path = Path(model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False, half=False))
    shutil.copy(path, out / "detector.onnx")
    sess = ort.InferenceSession(str(out / "detector.onnx"), providers=["CPUExecutionProvider"])
    o = sess.get_outputs()[0]
    return {"file": "detector.onnx", "imgsz": imgsz, "keypoints": 4, "conf": 0.5,
            "input": sess.get_inputs()[0].name, "output": o.name, "output_shape": o.shape}


def export_classifier(weights: Path, out: Path) -> dict:
    model = ClassifierForExport(load(str(weights), "cpu")).eval()
    dummy = torch.rand(2, 3, 256, 160)
    torch.onnx.export(
        model, dummy, str(out / "classifier.onnx"), opset_version=17, dynamo=False,
        input_names=["crops"], output_names=["probs"],
        dynamic_axes={"crops": {0: "n"}, "probs": {0: "n"}},
    )
    return {"file": "classifier.onnx", "input": [256, 160], "attrs": [[n, list(v)] for n, v in ATTRS]}


def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return x, r, px, py


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def parse_detections(out: np.ndarray, conf: float, nk: int = 4) -> np.ndarray:
    """Mirror of parseDetections() in web/js/pipeline.js -> (n, nk, 3) keypoints."""
    _, a, b = out.shape
    if b == 6 + nk * 3:  # end-to-end: [1, N, x1 y1 x2 y2 score cls kpts]
        rows = out[0][out[0][:, 4] >= conf]
        return rows[:, 6:].reshape(-1, nk, 3)
    if a == 5 + nk * 3:  # raw: [1, 5 + 3k, anchors] (cx cy w h score kpts), greedy NMS at IoU 0.7
        d = out[0]
        keep = np.nonzero(d[4] >= conf)[0]
        keep = keep[np.argsort(-d[4, keep])]
        boxes = np.stack([d[0] - d[2] / 2, d[1] - d[3] / 2, d[0] + d[2] / 2, d[1] + d[3] / 2])
        chosen = []
        for j in keep:
            if all(_iou(boxes[:, j], boxes[:, c]) < 0.7 for c in chosen):
                chosen.append(j)
        return d[5:, chosen].T.reshape(-1, nk, 3)
    raise SystemExit(f"unexpected detector output shape {out.shape}")


def check(det_pt: Path, cls_pt: Path, out: Path, photos: list[Path], imgsz: int):
    """Compare ONNX Runtime against PyTorch on real photos: same cards, same attributes."""
    yolo = YOLO(str(det_pt))
    clf_torch = ClassifierForExport(load(str(cls_pt), "cpu")).eval()
    det = ort.InferenceSession(str(out / "detector.onnx"), providers=["CPUExecutionProvider"])
    cls = ort.InferenceSession(str(out / "classifier.onnx"), providers=["CPUExecutionProvider"])
    max_corner_err, max_prob_err, n_cards, disagreements = 0.0, 0.0, 0, 0
    for p in photos:
        img = cv2.imread(str(p))
        ref = yolo.predict(img, imgsz=imgsz, conf=0.5, device="cpu", verbose=False)[0]
        ref_k = ref.keypoints.data.cpu().numpy()[..., :2] if ref.keypoints is not None else np.zeros((0, 4, 2))
        x, r, px, py = letterbox(img, imgsz)
        out = det.run(None, {det.get_inputs()[0].name: x})[0]
        onnx_k = (parse_detections(out, 0.5)[..., :2] - [px, py]) / r
        if len(onnx_k) != len(ref_k):
            print(f"  {p.name}: count differs (onnx {len(onnx_k)} vs torch {len(ref_k)})")
            disagreements += 1
            continue
        for q in onnx_k:  # nearest reference card
            d = np.abs(ref_k - q).max(axis=(1, 2)).min() if len(ref_k) else 0
            max_corner_err = max(max_corner_err, float(d))
        if len(onnx_k):
            crops = np.stack([warp_card(img, q)[..., ::-1] for q in onnx_k]).transpose(0, 3, 1, 2).astype(np.float32) / 255
            po = cls.run(None, {"crops": crops})[0]
            with torch.no_grad():
                pt = clf_torch(torch.from_numpy(crops)).numpy()
            max_prob_err = max(max_prob_err, float(np.abs(po - pt).max()))
            disagreements += int((po.argmax(-1) != pt.argmax(-1)).any(axis=1).sum())
            n_cards += len(onnx_k)
    print(f"parity on {len(photos)} photos / {n_cards} cards: max corner diff {max_corner_err:.2f} px "
          f"(letterbox resize differs slightly from Ultralytics'), max probability diff {max_prob_err:.2e}, "
          f"cards with a different prediction: {disagreements}")
    return {"photos": len(photos), "cards": n_cards, "max_corner_px": round(max_corner_err, 3),
            "max_prob_diff": max_prob_err, "prediction_disagreements": disagreements}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--photos", type=Path, default=None, help="folder of real photos for the parity check")
    ap.add_argument("--out", type=Path, default=Path("web/models"))
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detector": {**export_detector(a.detector, a.out, a.imgsz), "source": str(a.detector)},
            "classifier": {**export_classifier(a.classifier, a.out), "source": str(a.classifier)}}
    for k in ("detector", "classifier"):
        meta[k]["size_mb"] = round((a.out / meta[k]["file"]).stat().st_size / 1e6, 2)
    if a.photos:
        photos = sorted(q for q in a.photos.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
        meta["parity_check"] = check(a.detector, a.classifier, a.out, photos, a.imgsz)
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("detector", "classifier")}, indent=1))


if __name__ == "__main__":
    main()

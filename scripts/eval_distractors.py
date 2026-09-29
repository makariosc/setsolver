"""How often does the detector mistake a distractor (envelope, book, playing
card...) for a SET card? Runs an ONNX detector over generated scenes whose
labels list distractors with their image corners (scenes made with
--distractor-prob > 0).

    uv run --no-project --with onnxruntime --with opencv-python --with numpy \
        python scripts/eval_distractors.py data/synth/distractor_eval --detector web/models/detector.onnx

A detection is a false positive if it matches no labeled card (box IoU < 0.5,
any visibility); it is blamed on a distractor if its center lies inside one.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from compare_precision import detect, match, quad_box  # noqa: E402  (scripts/ is on sys.path when run directly)

THRESHOLDS = (0.5, 0.85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", type=Path)
    ap.add_argument("--detector", type=Path, default=Path("web/models/detector.onnx"))
    ap.add_argument("--out", type=Path, default=None, help="write a JSON summary here")
    a = ap.parse_args()
    det = ort.InferenceSession(str(a.detector), providers=["CPUExecutionProvider"])

    present = Counter()
    fooled = {t: Counter() for t in THRESHOLDS}  # distractor type -> #distractors with a FP on them
    fp_other = Counter()
    tp = Counter()
    n_gt = 0
    labels = sorted((a.scenes / "labels").iterdir())
    for lp in labels:
        lab = json.loads(lp.read_text())
        img = cv2.imread(str(a.scenes / "images" / (lp.stem + ".jpg")))
        dets = detect(det, img)
        gt = [np.array(c["corners"]) for c in lab["cards"]]
        n_gt += sum(c["visible_fraction"] >= 0.35 for c in lab["cards"])
        polys = [(d["type"], np.array(d["corners"], np.float32)) for d in lab.get("distractors", [])]
        for t, _ in polys:
            present[t] += 1
        for thr in THRESHOLDS:
            ds = [q for s, q in dets if s >= thr]
            m = match(ds, gt)
            tp[thr] += sum(lab["cards"][j]["visible_fraction"] >= 0.35 for _, j in m)
            matched = {i for i, _ in m}
            hit = set()
            for i, q in enumerate(ds):
                if i in matched:
                    continue
                cx, cy = q.mean(0)
                owner = [k for k, (_, poly) in enumerate(polys) if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(cx), float(cy)), False) >= 0]
                if owner:
                    hit.add(owner[-1])  # topmost
                else:
                    fp_other[thr] += 1
            for k in hit:
                fooled[thr][polys[k][0]] += 1

    summary = {"scenes": len(labels), "cards": n_gt, "detector": str(a.detector)}
    for thr in THRESHOLDS:
        summary[f"@{thr}"] = {
            "recall": round(tp[thr] / max(n_gt, 1), 4),
            "distractors_fooling_detector": sum(fooled[thr].values()),
            "other_false_positives": fp_other[thr],
            "by_type": {t: f"{fooled[thr][t]}/{present[t]}" for t in sorted(present, key=lambda t: -present[t])},
        }
    summary["distractors"] = sum(present.values())
    print(json.dumps(summary, indent=1))
    if a.out:
        a.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

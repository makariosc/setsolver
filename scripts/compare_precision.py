"""Compare the fp32 browser models against their fp16 / int8 variants
(made by scripts/quantize_models.py): size, speed, and how much the answers move.

    uv run --no-project --with onnxruntime --with opencv-python --with numpy \
        python scripts/compare_precision.py --photos ../setchecker/vibe/samples

Real photos have no labels here, so fp32 (151/151 correct on them) is the
reference; synthetic validation scenes/crops give true accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from setsolver.crops import warp_card  # noqa: E402

VARIANTS = ["fp32", "fp16", "int8dyn", "int8qdq"]
CONF, GATE, IMGSZ = 0.5, 0.85, 640


def model_path(models: Path, name: str, v: str) -> Path:
    return models / f"{name}.onnx" if v == "fp32" else models / "variants" / f"{name}_{v}.onnx"


def letterbox(img):
    h, w = img.shape[:2]
    r = min(IMGSZ / w, IMGSZ / h)
    nw, nh = round(w * r), round(h * r)
    px, py = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    c = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    c[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return c[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255, r, px, py


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def detect(sess, img):
    """-> list of (score, corners[4,2] in image px); same parse + NMS as the browser."""
    x, r, px, py = letterbox(img)
    d = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    keep = np.nonzero(d[4] >= CONF)[0]
    keep = keep[np.argsort(-d[4, keep])]
    boxes = np.stack([d[0] - d[2] / 2, d[1] - d[3] / 2, d[0] + d[2] / 2, d[1] + d[3] / 2])
    chosen = []
    for j in keep:
        if all(box_iou(boxes[:, j], boxes[:, c]) < 0.7 for c in chosen):
            chosen.append(j)
    return [(float(d[4, j]), (d[5:, j].reshape(4, 3)[:, :2] - [px, py]) / r) for j in chosen]


def quad_box(q):
    return [q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]


def corner_err(a, b):
    """Max corner distance, allowing any cyclic relabeling (order conventions differ)."""
    return min(np.linalg.norm(np.roll(a, s, 0) - b, axis=1).max() for s in range(4))


def match(dets, refs):
    """Greedy box-IoU matching -> list of (i_det, i_ref)."""
    pairs = sorted(((box_iou(quad_box(d), quad_box(r)), i, j) for i, d in enumerate(dets)
                    for j, r in enumerate(refs)), reverse=True)
    used_i, used_j, out = set(), set(), []
    for iou, i, j in pairs:
        if iou >= 0.5 and i not in used_i and j not in used_j:
            used_i.add(i), used_j.add(j), out.append((i, j))
    return out


def card_size(q):
    return float(np.sqrt(abs(cv2.contourArea(q.astype(np.float32)))))


def to_batch(crops_bgr):
    return np.stack([c[..., ::-1] for c in crops_bgr]).transpose(0, 3, 1, 2).astype(np.float32) / 255


def timeit(sess, feed, n=20):
    sess.run(None, feed)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, feed)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("web/models"))
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--synth", type=Path, default=Path("data/synth/val"))
    ap.add_argument("--crops", type=Path, default=Path("data/crops/val"))
    ap.add_argument("--n-scenes", type=int, default=300)
    ap.add_argument("--n-crops", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("web/models/variants/comparison.json"))
    a = ap.parse_args()
    meta = json.loads((a.models / "meta.json").read_text())
    attrs = meta["classifier"]["attrs"]

    so = ort.SessionOptions()
    so.log_severity_level = 3
    det = {v: ort.InferenceSession(str(model_path(a.models, "detector", v)), so, providers=["CPUExecutionProvider"]) for v in VARIANTS}
    cls = {v: ort.InferenceSession(str(model_path(a.models, "classifier", v)), so, providers=["CPUExecutionProvider"]) for v in VARIANTS}
    report = {v: {"size_mb": {n: round(model_path(a.models, n, v).stat().st_size / 1e6, 2) for n in ("detector", "classifier")}}
              for v in VARIANTS}

    # ---- real photos: everything relative to fp32 ----
    photos = sorted(p for p in a.photos.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    real = {v: {"count_diff_photos": 0, "gate_diff_photos": 0, "unmatched": 0, "corner_rel": [], "score_diff": [],
                "cls_disagree": 0, "cls_prob_diff": 0.0, "cls_gate_flips": 0, "e2e_diff_photos": 0} for v in VARIANTS}
    n_real_cards = 0
    for p in photos:
        img = cv2.imread(str(p))
        ref = detect(det["fp32"], img)
        ref_crops = [warp_card(img, q) for _, q in ref]
        ref_probs = cls["fp32"].run(None, {"crops": to_batch(ref_crops)})[0] if ref else np.zeros((0, 4, 3))
        n_real_cards += len(ref)

        def final(dets, probs):  # what the app would count: sorted readings of cards passing both gates
            return sorted(tuple(pr.argmax(-1)) for (s, _), pr in zip(dets, probs) if s >= GATE and pr.max(-1).min() >= GATE)

        ref_final = final(ref, ref_probs)
        for v in VARIANTS[1:]:
            r = real[v]
            dets = detect(det[v], img)
            r["count_diff_photos"] += len(dets) != len(ref)
            r["gate_diff_photos"] += sum(s >= GATE for s, _ in dets) != sum(s >= GATE for s, _ in ref)
            m = match([q for _, q in dets], [q for _, q in ref])
            r["unmatched"] += len(dets) + len(ref) - 2 * len(m)
            for i, j in m:
                r["corner_rel"].append(corner_err(dets[i][1], ref[j][1]) / card_size(ref[j][1]))
                r["score_diff"].append(abs(dets[i][0] - ref[j][0]))
            if ref:  # classifier alone, on fp32's crops
                pv = cls[v].run(None, {"crops": to_batch(ref_crops)})[0]
                r["cls_disagree"] += int((pv.argmax(-1) != ref_probs.argmax(-1)).any(1).sum())
                r["cls_prob_diff"] = max(r["cls_prob_diff"], float(np.abs(pv - ref_probs).max()))
                r["cls_gate_flips"] += int(((pv.max(-1).min(-1) >= GATE) != (ref_probs.max(-1).min(-1) >= GATE)).sum())
            # end to end: this variant's detector + classifier
            probs = cls[v].run(None, {"crops": to_batch([warp_card(img, q) for _, q in dets])})[0] if dets else np.zeros((0, 4, 3))
            r["e2e_diff_photos"] += final(dets, probs) != ref_final
    for v in VARIANTS[1:]:
        r = real[v]
        report[v]["real"] = {
            "photos": len(photos), "fp32_cards": n_real_cards,
            "photos_with_different_detection_count": r["count_diff_photos"],
            "photos_with_different_count_at_0.85": r["gate_diff_photos"],
            "unmatched_detections": r["unmatched"],
            "corner_diff_pct_of_card_mean": round(100 * float(np.mean(r["corner_rel"])), 3),
            "corner_diff_pct_of_card_max": round(100 * float(np.max(r["corner_rel"])), 3),
            "det_score_diff_max": round(float(np.max(r["score_diff"])), 4),
            "cls_cards_different_prediction": r["cls_disagree"],
            "cls_max_prob_diff": round(r["cls_prob_diff"], 4),
            "cls_cards_crossing_0.85": r["cls_gate_flips"],
            "photos_with_different_final_answer": r["e2e_diff_photos"],
        }

    # ---- synthetic validation scenes: detector vs ground truth ----
    labels = sorted((a.synth / "labels").iterdir())[:a.n_scenes]
    for v in VARIANTS:
        tp, fn, fp, errs = {CONF: 0, GATE: 0}, {CONF: 0, GATE: 0}, {CONF: 0, GATE: 0}, []
        for lp in labels:
            lab = json.loads(lp.read_text())
            gt = [np.array(c["corners"]) for c in lab["cards"] if c["visible_fraction"] >= 0.35]
            img = cv2.imread(str(a.synth / "images" / (lp.stem + ".jpg")))
            dets = detect(det[v], img)
            for thr in (CONF, GATE):
                ds = [q for s, q in dets if s >= thr]
                m = match(ds, gt)
                tp[thr] += len(m); fn[thr] += len(gt) - len(m); fp[thr] += len(ds) - len(m)
                if thr == CONF:
                    errs += [corner_err(ds[i], gt[j]) / card_size(gt[j]) for i, j in m]
        report[v]["synth_detector"] = {
            "scenes": len(labels),
            **{f"recall@{t}": round(tp[t] / (tp[t] + fn[t]), 4) for t in (CONF, GATE)},
            **{f"false_pos@{t}": fp[t] for t in (CONF, GATE)},
            "corner_err_pct_of_card_mean": round(100 * float(np.mean(errs)), 3),
        }

    # ---- synthetic validation crops: classifier vs ground truth ----
    rows = [json.loads(l) for l in (a.crops / "index.jsonl").read_text().splitlines()]
    rows = random.Random(0).sample(rows, min(a.n_crops, len(rows)))
    truth = np.array([[vals.index(r[k]) for k, vals in attrs] for r in rows])
    imgs = [cv2.imread(str(a.crops / "images" / r["file"])) for r in rows]
    for v in VARIANTS:
        probs = np.concatenate([cls[v].run(None, {"crops": to_batch(imgs[i:i + 64])})[0] for i in range(0, len(imgs), 64)])
        pred = probs.argmax(-1)
        report[v]["synth_classifier"] = {
            "crops": len(rows),
            "card_acc": round(float((pred == truth).all(1).mean()), 4),
            **{f"{k}_acc": round(float((pred[:, i] == truth[:, i]).mean()), 4) for i, (k, _) in enumerate(attrs)},
            "below_0.85": int((probs.max(-1).min(-1) < GATE).sum()),
        }

    # ---- CPU speed (ONNX Runtime CPU on this machine; not the browser) ----
    x_det = {det["fp32"].get_inputs()[0].name: np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32)}
    x_cls = {"crops": np.random.rand(12, 3, 256, 160).astype(np.float32)}
    for v in VARIANTS:
        report[v]["cpu_ms"] = {"detector": round(timeit(det[v], x_det), 1), "classifier_x12": round(timeit(cls[v], x_cls), 1)}
        o1 = ort.SessionOptions(); o1.intra_op_num_threads = 1; o1.log_severity_level = 3
        d1 = ort.InferenceSession(str(model_path(a.models, "detector", v)), o1, providers=["CPUExecutionProvider"])
        c1 = ort.InferenceSession(str(model_path(a.models, "classifier", v)), o1, providers=["CPUExecutionProvider"])
        report[v]["cpu_ms_1thread"] = {"detector": round(timeit(d1, x_det, 10), 1), "classifier_x12": round(timeit(c1, x_cls, 10), 1)}

    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

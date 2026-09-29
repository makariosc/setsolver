"""Run the card detector (and optionally the attribute classifier) on real
photos: draw each card's outline and predicted attributes, and print per-image
card counts (optionally against known counts / known attributes).

    .venv-train/bin/python scripts/predict_detector.py DET.pt photos/ --out preds/NAME
    .venv-train/bin/python scripts/predict_detector.py DET.pt photos/ --classifier CLS.pt \
        --truth counts.json --card-truth cards.json --out preds/NAME

--truth       JSON {filename: number_of_cards}
--card-truth  JSON {filename: [{"cx", "cy", "number", "color", "shape", "shading"}, ...]}
              (cx/cy: card center as a fraction of the image; the setchecker label format)

Writes <out>/orig/*.jpg, <out>/pred/*.jpg (downscaled to 1600 px), with
--classifier also <out>/cards/*.jpg (each card's straightened crop + prediction),
and <out>/summary.json, which the dashboard's Predictions page reads.

Cards touching the photo edge are marked "cut off" and not classified: their
crops are missing part of the card, so the app asks for a retake instead.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
PALETTE = [(0, 200, 0), (0, 160, 255), (255, 120, 0), (200, 0, 200), (0, 220, 220), (255, 0, 80)]
ATTR_KEYS = ("number", "color", "shape", "shading")


def list_images(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT and not q.name.startswith("."))
        elif p.suffix.lower() in IMG_EXT:
            out.append(p)
    return out


def is_cut_off(quad: np.ndarray, w: int, h: int, margin_frac: float = 0.004) -> bool:
    m = margin_frac * min(w, h)
    x, y = quad[:, 0], quad[:, 1]
    return bool((x < m).any() or (y < m).any() or (x > w - 1 - m).any() or (y > h - 1 - m).any())


def card_text(c: dict) -> str:
    if c.get("cut_off"):
        return "cut off"
    if "number" not in c:
        return f"{c['conf']:.2f}"
    return f"{c['number']} {c['color']} {c['shape']} {c['shading']}"


def put_label(img, text, center, scale, t):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
    x, y = int(center[0] - tw / 2), int(center[1] + th / 2)
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t, cv2.LINE_AA)


def draw(img: np.ndarray, cards: list[dict]) -> np.ndarray:
    out = img.copy()
    t = max(2, round(min(img.shape[:2]) / 400))
    for i, c in enumerate(cards):
        col = PALETTE[i % len(PALETTE)]
        pts = np.round(np.array(c["corners"])).astype(np.int32)
        cv2.polylines(out, [pts], True, col, t, cv2.LINE_AA)
        for j, (x, y) in enumerate(pts):
            cv2.circle(out, (int(x), int(y)), t * (4 if j == 0 else 2), col, -1, cv2.LINE_AA)
        side = min(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1]))
        scale = max(0.35, min(1.2, side / 260))
        put_label(out, card_text(c), pts.mean(0), scale, max(1, t // 2))
    return out


def card_sheet(crops: list[np.ndarray], cards: list[dict], cols: int = 8) -> np.ndarray:
    cells = []
    for i, (crop, c) in enumerate(zip(crops, cards)):
        cap = np.full((40, crop.shape[1], 3), 255, np.uint8)
        cv2.putText(cap, f"#{i + 1} {card_text(c)}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        if "min_p" in c:
            cv2.putText(cap, f"lowest p {c['min_p']:.2f} ({c['least_sure']})", (3, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (90, 90, 90), 1, cv2.LINE_AA)
        cells.append(cv2.copyMakeBorder(np.vstack([crop, cap]), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 255))
    return np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])


def match_truth(cards: list[dict], truth: list[dict], w: int, h: int) -> list[tuple[int, dict]]:
    """Greedy nearest-center matching of predicted cards to truth cards."""
    centers = [np.mean(c["corners"], axis=0) / [w, h] for c in cards]
    pairs, used = [], set()
    for t in truth:
        best, bd = None, 0.25  # max normalized distance
        for i, ce in enumerate(centers):
            d = float(np.hypot(ce[0] - t["cx"], ce[1] - t["cy"]))
            if i not in used and d < bd:
                best, bd = i, d
        if best is not None:
            used.add(best)
            pairs.append((best, t))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--classifier", default=None, help="classifier checkpoint (weights/best.pt)")
    ap.add_argument("--out", type=Path, default=Path("preds"))
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--card-truth", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    truth = json.loads(a.truth.read_text()) if a.truth else {}
    card_truth = json.loads(a.card_truth.read_text()) if a.card_truth else {}
    model = YOLO(a.weights)
    clf = None
    if a.classifier:
        import torch

        from setsolver.classifier import decode, load, preprocess
        from setsolver.crops import warp_card
        dev = "cpu" if a.device == "cpu" or not torch.cuda.is_available() else "cuda"
        clf = load(a.classifier, dev)
    for sub in ("orig", "pred") + (("cards",) if clf else ()):
        (a.out / sub).mkdir(parents=True, exist_ok=True)

    found_total = truth_total = 0
    attr_ok = {k: 0 for k in ATTR_KEYS}
    attr_n = card_ok = 0
    items = []
    for path in list_images(a.images):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        det_ms = (time.perf_counter() - t0) * 1000
        kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 4, 3))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        cards = [{"conf": round(float(c), 3), "corners": [[round(float(x), 1), round(float(y), 1)] for x, y, _ in q],
                  "cut_off": is_cut_off(q[:, :2], w, h)} for q, c in zip(kpts, confs)]

        cls_ms = 0.0
        if clf is not None and cards:
            crops = [warp_card(img, np.array(c["corners"])) for c in cards]
            idx = [i for i, c in enumerate(cards) if not c["cut_off"]]
            if idx:
                t1 = time.perf_counter()
                with torch.no_grad():
                    preds = decode(clf(preprocess([crops[i] for i in idx], dev)))
                cls_ms = (time.perf_counter() - t1) * 1000
                for i, p in zip(idx, preds):
                    cards[i].update(p)
                    ps = {k: p[f"{k}_p"] for k in ATTR_KEYS}
                    least = min(ps, key=ps.get)
                    cards[i]["min_p"], cards[i]["least_sure"] = round(ps[least], 3), least
            cv2.imwrite(str(a.out / "cards" / f"{path.stem}.jpg"), card_sheet(crops, cards), [cv2.IMWRITE_JPEG_QUALITY, 90])

        # attribute scoring against known cards
        scored = []
        if clf is not None and path.name in card_truth:
            for i, t in match_truth(cards, card_truth[path.name], w, h):
                c = cards[i]
                if c["cut_off"]:
                    continue
                ok = {k: c[k] == t[k] for k in ATTR_KEYS}
                for k in ATTR_KEYS:
                    attr_ok[k] += ok[k]
                attr_n += 1
                card_ok += all(ok.values())
                c["truth"] = {k: t[k] for k in ATTR_KEYS}
                c["correct"] = all(ok.values())
                scored.append(all(ok.values()))

        s = min(1.0, 1600 / max(h, w))
        small = lambda im: cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im
        cv2.imwrite(str(a.out / "orig" / f"{path.stem}.jpg"), small(img), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(a.out / "pred" / f"{path.stem}.jpg"), small(draw(img, cards)), [cv2.IMWRITE_JPEG_QUALITY, 90])
        n = len(cards)
        items.append({
            "file": path.name, "stem": path.stem, "n": n, "truth": truth.get(path.name),
            "confs": [c["conf"] for c in cards], "ms": round(det_ms, 1), "cls_ms": round(cls_ms, 1),
            "size": [w, h], "cards": cards, "cut_off": sum(c["cut_off"] for c in cards),
            "attr_scored": len(scored), "attr_correct": sum(scored),
        })
        line = f"{path.name:32s} detected {n:2d}"
        if path.name in truth:
            found_total += min(n, truth[path.name])
            truth_total += truth[path.name]
            line += f"  actual {truth[path.name]:2d}"
        if clf is not None:
            line += "  | " + ", ".join(card_text(c) for c in cards[:4]) + (" ..." if n > 4 else "")
        print(line)
    if truth_total:
        print(f"\ncards found (capped at actual): {found_total}/{truth_total} ({found_total / truth_total:.0%})")
    if attr_n:
        print(f"cards with known attributes: {card_ok}/{attr_n} fully correct; "
              + ", ".join(f"{k} {attr_ok[k]}/{attr_n}" for k in ATTR_KEYS))
    (a.out / "summary.json").write_text(json.dumps({
        "weights": str(a.weights), "classifier": a.classifier, "conf": a.conf, "imgsz": a.imgsz,
        "created": time.time(), "found": found_total, "truth_total": truth_total,
        "attr_n": attr_n, "attr_card_ok": card_ok, "attr_ok": attr_ok, "images": items,
    }, indent=1))


if __name__ == "__main__":
    main()

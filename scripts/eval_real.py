"""Compare two model pairs (detector + classifier ONNX) on real photos.

    .venv-train/bin/python scripts/eval_real.py \
        --a web/models/detector.onnx web/models/classifier.onnx --a-name current \
        --b web/models/candidates/detector.onnx web/models/candidates/classifier.onnx --b-name new \
        --photos data/real/setchecker_samples data/real/glare --out eval/real

Checks, per model pair, the way the app counts cards (detector score >= 0.85
and every attribute probability >= 0.85; cards touching the photo edge count
only if they also pass setsolver/edge_gate.py's stricter check):
- photos named *full_deck*: all 81 cards, each exactly once -> every misread
  shows up as a duplicate + a missing card, no hand labels needed;
- --counts JSON {photo: n}: detected card count;
- --cards JSON {photo: [{number, color, shape, shading}, ...]}: readings;
- cards the two pairs read differently.
Writes <out>/summary.json, <out>/report.txt and crop sheets for every card
that is wrong, uncertain, or read differently (<out>/<photo>_cards.jpg).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_precision import detect, match, to_batch  # noqa: E402
from setsolver.crops import warp_card  # noqa: E402
from setsolver.edge_gate import edge_card_ok  # noqa: E402

ATTRS = [("number", [1, 2, 3]), ("color", ["red", "green", "purple"]),
         ("shape", ["diamond", "oval", "squiggle"]), ("shading", ["solid", "striped", "open"])]
GATE = 0.85
EDGE = 0.004  # same cut-off rule as web/js/pipeline.js


def read_photo(det, cls, img):
    h, w = img.shape[:2]
    margin = EDGE * min(w, h)
    out = []
    for s, q in detect(det, img):
        cut = bool(((q < margin) | (q > [w - 1 - margin, h - 1 - margin])).any())
        out.append({"score": s, "corners": q, "cut": cut})
    todo = out   # edge cards too: they count if they pass edge_gate's stricter check (as web/js/pipeline.js)
    if todo:
        crops = [warp_card(img, c["corners"]) for c in todo]
        probs = cls.run(None, {"crops": to_batch(crops)})[0]
        for c, p, cr in zip(todo, probs, crops):
            c["probs"], c["crop"] = p, cr
            c["label"] = tuple(vals[i] for (_, vals), i in zip(ATTRS, p.argmax(-1)))
            c["minp"] = float(p.max(-1).min())
    whole = [c["corners"] for c in out if not c["cut"]]
    for c in out:
        c["edge_ok"] = c["cut"] and edge_card_ok(c["corners"], c["minp"], w, h, whole)
        c["counted"] = (not c["cut"] or c["edge_ok"]) and c["score"] >= GATE and c["minp"] >= GATE
    return out


def lab(c) -> str:
    return " ".join(map(str, c["label"])) if "label" in c else "cut off"


def why(c) -> str:
    if c["cut"] and not c["edge_ok"]:
        return "cut off" if c["minp"] >= 0.95 else f"cut off (min p {c['minp']:.2f})"
    bits = [f"det {c['score']:.2f}"] if c["score"] < GATE else []
    for (name, vals), p in zip(ATTRS, c["probs"]):
        o = np.argsort(-p)
        if p[o[0]] < GATE:
            bits.append(f"{name} {vals[o[0]]} {p[o[0]]:.2f}/{vals[o[1]]} {p[o[1]]:.2f}")
    return ", ".join(bits)


def tile(crop, lines, color):
    t = cv2.copyMakeBorder(cv2.resize(crop, (160, 256)), 0, 16 * len(lines) + 4, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for i, (text, col) in enumerate(lines):
        cv2.putText(t, text[:36], (3, 270 + 16 * i), 0, 0.31, col, 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, required=True, metavar=("DETECTOR", "CLASSIFIER"))
    ap.add_argument("--b", nargs=2, required=True, metavar=("DETECTOR", "CLASSIFIER"))
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--photos", type=Path, nargs="+", required=True)
    ap.add_argument("--counts", type=Path, default=Path("data/real/setchecker_counts.json"))
    ap.add_argument("--cards", type=Path, default=Path("data/real/setchecker_cards.json"))
    ap.add_argument("--out", type=Path, default=Path("eval/real"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    counts = json.loads(a.counts.read_text()) if a.counts.exists() else {}
    truth = json.loads(a.cards.read_text()) if a.cards.exists() else {}
    names = [a.a_name, a.b_name]
    sess = {n: [ort.InferenceSession(str(f), providers=["CPUExecutionProvider"]) for f in pair]
            for n, pair in zip(names, (a.a, a.b))}
    deck = set(itertools.product(*[v for _, v in ATTRS]))

    photos = sorted(p for d in a.photos for p in d.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith("."))
    S = {n: Counter() for n in names}
    lines = []
    for p in photos:
        img = cv2.imread(str(p))
        R = {n: read_photo(*sess[n], img) for n in names}
        head = f"{p.name}"
        for n in names:
            cards = R[n]
            counted = [c for c in cards if c["counted"]]
            S[n]["photos"] += 1
            S[n]["detections"] += len(cards)
            S[n]["counted"] += len(counted)
            S[n]["uncertain"] += sum(not c["cut"] and not c["counted"] for c in cards)
            bits = [f"{len(cards)} det, {len(counted)} counted"]
            if p.name in counts:
                ok = len(cards) == counts[p.name]
                S[n]["count_ok"] += ok
                S[n]["count_photos"] += 1
                bits.append(f"count {'ok' if ok else 'WRONG'} ({counts[p.name]})")
            if p.name in truth:
                want = Counter(tuple(t[k] for k, _ in ATTRS) for t in truth[p.name])
                got = Counter(c["label"] for c in cards if "label" in c)
                right = sum((want & got).values())
                S[n]["labeled_cards"] += sum(want.values())
                S[n]["labeled_right"] += right
                bits.append(f"labels {right}/{sum(want.values())}")
            if "full_deck" in p.name:
                read = [c for c in cards if "label" in c]
                cnt = Counter(c["label"] for c in read)
                dup = {l for l, k in cnt.items() if k > 1}
                missing = deck - set(cnt)
                wrong = sum(k - 1 for k in cnt.values() if k > 1)
                wrong_counted = sum(1 for c in read if c["counted"] and c["label"] in dup)
                S[n]["deck_cards"] += len(read)
                S[n]["deck_distinct"] += len(cnt)
                S[n]["deck_misread"] += wrong
                S[n]["deck_uncertain"] += sum(not c["counted"] for c in read)
                bits.append(f"full deck: {len(cnt)}/81 distinct, {wrong} misread, {len(missing)} missing, "
                            f"{sum(not c['counted'] for c in read)} uncertain, {wrong_counted} wrong-but-counted(max)")
                for c in read:
                    c["dup"] = c["label"] in dup
            head += f"\n    {n:>10}: " + "; ".join(bits)
        # pair cards across models; sheet of every card that is wrong/uncertain/different
        A, B = R[names[0]], R[names[1]]
        m = match([c["corners"] for c in B], [c["corners"] for c in A])
        tiles = []
        for ib, ia in m:
            ca, cb = A[ia], B[ib]
            if "label" not in ca or "label" not in cb:
                continue
            differ = ca["label"] != cb["label"]
            flag = differ or not ca["counted"] or not cb["counted"] or ca.get("dup") or cb.get("dup")
            if differ:
                S[names[0]]["disagreements"] += 1
            if flag:
                x, y = ca["corners"].mean(0)
                rows = [(f"({x:.0f},{y:.0f})", (0, 0, 0))]
                for n, c in ((names[0], ca), (names[1], cb)):
                    col = (0, 0, 200) if c.get("dup") else ((0, 120, 200) if not c["counted"] else (0, 140, 0))
                    rows.append((f"{n}: {lab(c)}", col))
                    if not c["counted"]:
                        rows.append((f"  ? {why(c)}", col))
                    elif c.get("dup"):
                        rows.append(("  same reading as another card", col))
                tiles.append(tile(ca["crop"], rows, (0, 0, 200) if differ else (200, 200, 200)))
        if tiles:
            per = 8
            hmax = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
            tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % per)
            sheet = np.vstack([np.hstack(tiles[i:i + per]) for i in range(0, len(tiles), per)])
            cv2.imwrite(str(a.out / f"{p.stem}_cards.jpg"), sheet)
            head += f"\n    sheet: {p.stem}_cards.jpg ({len(tiles)} cards)"
        lines.append(head)

    report = "\n".join(lines) + "\n\nTOTALS\n" + "\n".join(
        f"  {n:>10}: " + ", ".join(f"{k} {v}" for k, v in S[n].items()) for n in names)
    (a.out / "report.txt").write_text(report)
    (a.out / "summary.json").write_text(json.dumps({n: dict(S[n]) for n in names}, indent=1))
    print(report)


if __name__ == "__main__":
    main()

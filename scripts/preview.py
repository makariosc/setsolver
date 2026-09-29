"""Draw labels over generated images and build contact sheets for eyeballing.

    uv run scripts/preview.py data/synth/preview
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ABBR = {"red": "R", "green": "G", "purple": "P", "diamond": "D", "oval": "O", "squiggle": "S",
        "solid": "s", "striped": "t", "open": "o"}
COLOR_BGR = {"red": (40, 40, 230), "green": (60, 190, 40), "purple": (180, 50, 140)}


def overlay(img: np.ndarray, labels: dict) -> np.ndarray:
    out = img.copy()
    for c in labels["cards"]:
        pts = np.array(c["corners"], np.float32)
        col = COLOR_BGR[c["color"]]
        cv2.polylines(out, [np.round(pts).astype(np.int32)], True, col, 2, cv2.LINE_AA)
        cv2.circle(out, tuple(np.round(pts[0]).astype(int)), 5, (0, 255, 255), -1)  # TL corner
        text = f'{c["number"]}{ABBR[c["color"]]}{ABBR[c["shape"]]}{ABBR[c["shading"]]}'
        if c["visible_fraction"] < 0.98:
            text += f' {c["visible_fraction"]:.0%}'
        org = tuple(np.round(pts.mean(0) - [30, 0]).astype(int))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def caption(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((36, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def sheet(tiles: list[np.ndarray], cols: int, cell: int) -> np.ndarray:
    """Letterbox each tile into a square cell and grid them."""
    cells = []
    for t in tiles:
        s = cell / max(t.shape[:2])
        r = cv2.resize(t, (int(t.shape[1] * s), int(t.shape[0] * s)), interpolation=cv2.INTER_AREA)
        pad = np.full((cell, cell, 3), 255, np.uint8)
        y, x = (cell - r.shape[0]) // 2, (cell - r.shape[1]) // 2
        pad[y : y + r.shape[0], x : x + r.shape[1]] = r
        cells.append(pad)
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 255))
    return np.vstack([np.hstack(cells[i : i + cols]) for i in range(0, len(cells), cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()
    out_dir = a.root / "preview"
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    raw, ann = [], []
    for jp in sorted((a.root / "images").glob("*.jpg")):
        labels = json.loads((a.root / "labels" / f"{jp.stem}.json").read_text())
        img = cv2.imread(str(jp))
        info = f'{jp.stem} {labels["background"]["name"]} / {labels["lighting"]["rig"]} / tilt {labels["camera"]["tilt_deg"]:.0f}'
        o = overlay(img, labels)
        cv2.imwrite(str(out_dir / "overlays" / jp.name), o)
        raw.append(caption(img, info))
        ann.append(caption(o, info))
    for i in range(0, len(raw), a.per_sheet):
        n = i // a.per_sheet
        cv2.imwrite(str(out_dir / f"sheet_{n:02d}.jpg"), sheet(raw[i : i + a.per_sheet], a.cols, 700))
        cv2.imwrite(str(out_dir / f"sheet_{n:02d}_labels.jpg"), sheet(ann[i : i + a.per_sheet], a.cols, 700))
    print(f"wrote {len(raw)} overlays and sheets to {out_dir}")


if __name__ == "__main__":
    main()

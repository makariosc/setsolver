"""Convert generator output (images/ + labels/*.json) into an Ultralytics pose
dataset: one class ("card"), a box, and 4 corner keypoints per card.

    uv run scripts/export_yolo.py data/synth/train data/synth/val --out data/yolo/cards
    uv run scripts/export_yolo.py data/synth/train,data/synth/train_distractors \
        data/synth/val,data/synth/val_distractors --out data/yolo/cards_v2   # several sources per split

Corner order. A card looks the same rotated 180 degrees, so its own TL/TR/BR/BL
labels are ambiguous to a network. We pick, for each card, whichever of the two
equivalent orderings (TL,TR,BR,BL) or (BR,BL,TL,TR) has the card's "up" axis
pointing in the half-plane of angles [-135, 45) degrees (image coords, y down).
The ordering then depends only on what is visible, and the unavoidable
discontinuity sits at diagonal orientations, away from the common upright and
sideways grid shots. Because of this, train with fliplr=0 and degrees=0: those
augmentations would reorder corners inconsistently.

Visibility (Ultralytics / COCO convention): 2 = visible, 1 = occluded but
position known (still trained), 0 = outside the frame (Ultralytics requires
in-image coordinates, so these are zeroed and not trained).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

VIS = {"visible": 2, "occluded": 1, "outside": 0}


def canonical_order(corners: np.ndarray, vis: list[int]) -> tuple[np.ndarray, list[int]]:
    up = (corners[0] + corners[1]) / 2 - (corners[3] + corners[2]) / 2
    theta = np.degrees(np.arctan2(up[1], up[0]))
    if -135 <= theta < 45:
        return corners, vis
    idx = [2, 3, 0, 1]
    return corners[idx], [vis[i] for i in idx]


def convert_labels(L: dict, min_visible: float) -> list[str]:
    w, h = L["image_size"]
    lines = []
    for c in L["cards"]:
        if c["visible_fraction"] < min_visible:
            continue
        pts = np.array(c["corners"], float)
        vis = [VIS[v] for v in c["corner_visibility"]]
        pts, vis = canonical_order(pts, vis)
        lo = np.clip(pts.min(0), 0, [w, h])
        hi = np.clip(pts.max(0), 0, [w, h])
        if (hi - lo).min() < 2:
            continue
        box = [(lo[0] + hi[0]) / 2 / w, (lo[1] + hi[1]) / 2 / h, (hi[0] - lo[0]) / w, (hi[1] - lo[1]) / h]
        kp = []
        for (x, y), v in zip(pts, vis):
            if v == 0 or not (0 <= x < w and 0 <= y < h):
                kp += [0.0, 0.0, 0]
            else:
                kp += [x / w, y / h, v]
        lines.append("0 " + " ".join(f"{v:.6f}" for v in box) + " " + " ".join(
            f"{v:.6f}" if i % 3 != 2 else str(int(v)) for i, v in enumerate(kp)))
    return lines


def export_split(src: Path, dst: Path, split: str, min_visible: float) -> tuple[int, int]:
    (dst / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_img = n_obj = 0
    for jp in sorted((src / "labels").glob("*.json")):
        L = json.loads(jp.read_text())
        img = (src / "images" / f"{jp.stem}.jpg").resolve()
        name = f"{src.name}_{jp.stem}"
        link = dst / "images" / split / f"{name}.jpg"
        if not link.exists():
            os.symlink(img, link)
        lines = convert_labels(L, min_visible)
        (dst / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_obj += len(lines)
    return n_img, n_obj


def main():
    ap = argparse.ArgumentParser()
    paths = lambda s: [Path(p) for p in s.split(",")]  # noqa: E731
    ap.add_argument("train", type=paths, help="scene folder(s), comma-separated")
    ap.add_argument("val", type=paths, help="scene folder(s), comma-separated")
    ap.add_argument("--out", type=Path, default=Path("data/yolo/cards"))
    ap.add_argument("--min-visible", type=float, default=0.35, help="drop cards less visible than this")
    a = ap.parse_args()
    for split, srcs in (("train", a.train), ("val", a.val)):
        for src in srcs:
            n_img, n_obj = export_split(src, a.out, split, a.min_visible)
            print(f"{split} <- {src}: {n_img} images, {n_obj} cards")
    (a.out / "data.yaml").write_text(
        f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\n"
        "kpt_shape: [4, 3]\nnames:\n  0: card\n"
    )
    print(f"wrote {a.out / 'data.yaml'}")


if __name__ == "__main__":
    main()

"""Extract straightened card crops (+ labels) from generated scenes, in parallel.

    uv run scripts/extract_crops.py data/synth/train --out data/crops/train
    uv run scripts/extract_crops.py data/synth/val   --out data/crops/val --jitter 0
    uv run scripts/extract_crops.py data/synth/train --out /tmp/c --limit 200   # quick look

Each crop is a 160x256 portrait JPEG. Cards cut off by the photo edge and cards
less than 70% visible are skipped by default. Corners come from the generator labels,
perturbed by Gaussian noise (sigma = --jitter x the card's short side) to mimic
detector imprecision; use --jitter 0 for clean evaluation crops.

Writes <out>/images/<scene>_<k>.jpg and <out>/index.jsonl (one row per crop:
attributes, class index 0-80, and scene metadata for slicing results).
Work is split by scene across processes (--workers, default: all cores).
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from setsolver.crops import CROP_H, CROP_W, jitter_corners, warp_card


def _init_worker():
    cv2.setNumThreads(1)  # one process per core; no nested threading


def extract_scene(args) -> list[dict]:
    label_path, image_dir, out_dir, jitter, min_visible, exclude_clipped, seed = args
    L = json.loads(Path(label_path).read_text())
    stem = Path(label_path).stem
    img = cv2.imread(str(Path(image_dir) / f"{stem}.jpg"))
    if img is None:
        return []
    # per-scene RNG: reproducible regardless of worker count
    rng = np.random.default_rng([seed, int(stem) if stem.isdigit() else abs(hash(stem)) % 2**31])
    rows = []
    for k, c in enumerate(L["cards"]):
        if c["visible_fraction"] < min_visible:
            continue
        if exclude_clipped and c["clipped"]:
            continue  # cut off by the frame: the app asks for a retake instead of classifying
        corners = np.array(c["corners"], np.float64)
        if jitter > 0:
            corners = jitter_corners(corners, rng, jitter)
        crop = warp_card(img, corners, (CROP_W, CROP_H))
        name = f"{stem}_{k:02d}.jpg"
        cv2.imwrite(str(Path(out_dir) / "images" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        vis = c.get("corner_visibility", [])
        rows.append({
            "file": name,
            "scene": stem,
            "number": c["number"], "color": c["color"], "shape": c["shape"], "shading": c["shading"],
            "index": c["index"],
            "visible_fraction": c["visible_fraction"],
            "clipped": c["clipped"],
            "corners_occluded": sum(v == "occluded" for v in vis),
            "px_per_mm": c["px_per_mm"],
            "rig": L["lighting"]["rig"],
            "background": L["background"]["name"],
            "palette": L["deck_style"]["palette"],
            "occluders": [o["type"] for o in L.get("occluders", [])],
            "framing": L.get("framing", {}).get("mode"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="generator output dir (images/ + labels/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jitter", type=float, default=0.015, help="corner noise, fraction of card short side")
    ap.add_argument("--min-visible", type=float, default=0.7,
                    help="skip cards less visible than this (below ~0.7 whole symbols can be hidden)")
    ap.add_argument("--include-clipped", action="store_true",
                    help="keep cards cut off by the photo edge (default: skip them)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N scenes")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    labels = sorted((a.src / "labels").glob("*.json"))[: a.limit]
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(a.src / "images"), str(a.out), a.jitter, a.min_visible, not a.include_clipped, a.seed)
            for p in labels]
    t0 = time.time()
    n = 0
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as ex, (a.out / "index.jsonl").open("w") as idx:
        for i, rows in enumerate(ex.map(extract_scene, jobs, chunksize=16), 1):
            for r in rows:
                idx.write(json.dumps(r) + "\n")
            n += len(rows)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)} scenes, {n} crops, {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"{n} crops from {len(jobs)} scenes in {dt:.1f}s ({len(jobs) / dt:.0f} scenes/s, {a.workers} workers) -> {a.out}")


if __name__ == "__main__":
    main()

"""Generate synthetic SET photos + JSON labels.

    uv run scripts/generate.py -n 48 --out data/synth/preview
    uv run scripts/generate.py -n 10 --only-backgrounds rug,wood --seed 7

Image i is generated from seed (seed, i), so any single image can be
reproduced exactly regardless of worker count.
"""

import argparse
import json
import os
import time

# One thread per worker process: the pool already uses every core, and nested
# OpenCV/BLAS threads oversubscribe the CPU (~3x slower on a 16-core Linux box).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from setsolver.synth import SceneConfig, generate_scene

cv2.setNumThreads(1)


def _one(args):
    i, seed, out, cfg = args
    rng = np.random.default_rng([seed, i])
    img, labels = generate_scene(rng, cfg)
    labels["seed"] = [seed, i]
    name = f"{i:06d}"
    cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 97])
    (out / "labels" / f"{name}.json").write_text(json.dumps(labels, indent=1))
    clipped = any(c["clipped"] for c in labels["cards"])
    return (name, labels["background"]["name"], labels["lighting"]["rig"], len(labels["cards"]), clipped,
            labels["framing"]["mode"], [d["type"] for d in labels["distractors"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("--out", type=Path, default=Path("data/synth/preview"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="first image index")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--only-backgrounds", type=lambda s: s.split(","), default=None)
    ap.add_argument("--only-rigs", type=lambda s: s.split(","), default=None)
    ap.add_argument("--only-layouts", type=lambda s: s.split(","), default=None)
    ap.add_argument("--only-distractors", type=lambda s: s.split(","), default=None)
    ap.add_argument("--distractor-prob", type=float, default=None, help="override SceneConfig.distractor_prob")
    ap.add_argument("--supersample", type=int, default=2,
                    help="render at N x the output size (2: crisper fine detail for classifier crops; "
                         "1: ~3.5x faster, plenty for detector training at 640 px)")
    a = ap.parse_args()

    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "labels").mkdir(parents=True, exist_ok=True)
    cfg = SceneConfig(
        width=a.width,
        height=a.height,
        backgrounds=a.only_backgrounds,
        rigs=a.only_rigs,
        layouts=a.only_layouts,
        distractors=a.only_distractors,
        supersample=a.supersample,
    )
    if a.distractor_prob is not None:
        cfg.distractor_prob = a.distractor_prob
    jobs = [(i, a.seed, a.out, cfg) for i in range(a.start, a.start + a.n)]
    t0 = time.time()
    n_clipped = 0
    rigs: dict[str, int] = {}
    framings: dict[str, int] = {}
    distractors: dict[str, int] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for name, bg, rig, n, clipped, framing, ds in ex.map(_one, jobs):
            for d in ds:
                distractors[d] = distractors.get(d, 0) + 1
            n_clipped += clipped
            rigs[rig] = rigs.get(rig, 0) + 1
            framings[framing] = framings.get(framing, 0) + 1
            print(f"{name}  {bg:<12} {rig:<20} {n:2d} cards{'  (clipped)' if clipped else ''}"
                  f"{'  + ' + ', '.join(ds) if ds else ''}", flush=True)
    print(f"{a.n} images in {time.time() - t0:.1f}s -> {a.out}")
    print(f"images with clipped cards: {n_clipped}/{a.n} ({n_clipped / a.n:.0%})")
    print("framing:", ", ".join(f"{k} {v}" for k, v in sorted(framings.items(), key=lambda kv: -kv[1])))
    print("distractors:", ", ".join(f"{k} {v}" for k, v in sorted(distractors.items(), key=lambda kv: -kv[1])) or "none")
    print("lighting:", ", ".join(f"{k} {v}" for k, v in sorted(rigs.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()

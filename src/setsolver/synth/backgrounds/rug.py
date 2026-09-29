"""Patterned (Persian-style) rug in ink-like reds, blues and purples.

This is deliberately a hard negative: busy, saturated shapes in the same hues
as the card inks. Motifs are drawn once into a repeat tile, then tiled.
"""

import cv2
import numpy as np

from ..color import jitter_hsv_srgb, srgb_to_linear
from ..noise import fractal_noise
from .base import BACKGROUNDS, Background, PlaneRegion, SurfaceMaps

FIELDS = [(0.78, 0.74, 0.66), (0.62, 0.16, 0.16), (0.16, 0.20, 0.38), (0.85, 0.82, 0.76)]
MOTIFS = [
    (0.70, 0.20, 0.18),
    (0.18, 0.28, 0.55),
    (0.35, 0.40, 0.62),
    (0.40, 0.18, 0.35),
    (0.80, 0.60, 0.25),
    (0.12, 0.12, 0.18),
    (0.30, 0.45, 0.35),
]


def _motif_tile(size: int, rng: np.random.Generator, colors: list[np.ndarray], field: np.ndarray) -> np.ndarray:
    """Seamlessly repeating motif tile: every primitive is drawn at all 9
    wrap-around offsets on a 3x canvas, and the center is cropped."""
    img = np.empty((3 * size, 3 * size, 3), np.float32)
    img[:] = field
    offsets = [(dx, dy) for dx in (0, size, 2 * size) for dy in (0, size, 2 * size)]

    def pt(x, y, dx, dy):
        return int(x + dx), int(y + dy)

    # vines: smooth random curves (random walk with smoothed heading)
    for _ in range(rng.integers(4, 9)):
        heading = np.cumsum(np.convolve(rng.normal(0, 0.35, 80), np.ones(9) / 9, mode="same"))
        heading += rng.uniform(0, 2 * np.pi)
        step = size * 0.012
        pts = np.cumsum(np.stack([np.cos(heading), np.sin(heading)], 1) * step, 0) + rng.uniform(0, size, 2)
        c = colors[rng.integers(len(colors))].tolist()
        for dx, dy in offsets:
            cv2.polylines(img, [np.round(pts + (dx, dy)).astype(np.int32)], False, c, max(1, size // 120), cv2.LINE_AA)
    # flowers / medallions: petals around a center
    for _ in range(rng.integers(12, 30)):
        cx, cy = rng.uniform(0, size, 2)
        r = rng.uniform(0.02, 0.07) * size
        c1, c2 = colors[rng.integers(len(colors))].tolist(), colors[rng.integers(len(colors))].tolist()
        n = int(rng.integers(4, 9))
        for dx, dy in offsets:
            for k in range(n):
                a = 2 * np.pi * k / n
                center = pt(cx + r * np.cos(a), cy + r * np.sin(a), dx, dy)
                cv2.ellipse(img, center, (int(r * 0.7), int(r * 0.35)), np.degrees(a), 0, 360, c1, -1, cv2.LINE_AA)
            cv2.circle(img, pt(cx, cy, dx, dy), int(r * 0.45), c2, -1, cv2.LINE_AA)
    # leaves
    for _ in range(rng.integers(25, 70)):
        cx, cy = rng.uniform(0, size, 2)
        c = colors[rng.integers(len(colors))].tolist()
        ax = (int(rng.uniform(0.02, 0.05) * size), int(rng.uniform(0.008, 0.02) * size))
        ang = rng.uniform(0, 180)
        for dx, dy in offsets:
            cv2.ellipse(img, pt(cx, cy, dx, dy), ax, ang, 0, 360, c, -1, cv2.LINE_AA)
    return img[size : 2 * size, size : 2 * size].copy()


@BACKGROUNDS.register("rug", weight=1.5)
class Rug(Background):
    def render(self, region: PlaneRegion, rng: np.random.Generator) -> SurfaceMaps:
        h, w = region.shape
        field = srgb_to_linear(jitter_hsv_srgb(FIELDS[rng.integers(len(FIELDS))], rng, dh=0.02, ds=0.15, dv=0.1))
        idx = rng.choice(len(MOTIFS), size=rng.integers(3, 6), replace=False)
        colors = [srgb_to_linear(jitter_hsv_srgb(MOTIFS[i], rng, dh=0.03, ds=0.15, dv=0.15)) for i in idx]

        repeat_mm = rng.uniform(250, 600)
        tile_px = max(64, int(region.px(repeat_mm)))
        draw_px = min(tile_px, 1024)  # motifs don't need more detail than this
        tile = _motif_tile(draw_px, rng, colors, field)
        if draw_px != tile_px:
            tile = cv2.resize(tile, (tile_px, tile_px), interpolation=cv2.INTER_CUBIC)
        reps = (h // tile_px + 2, w // tile_px + 2)
        big = np.tile(tile, (reps[0], reps[1], 1))
        oy, ox = rng.integers(0, tile_px, 2)
        albedo = big[oy : oy + h, ox : ox + w]

        # pile: soften motif edges, add fuzz and wear
        albedo = cv2.GaussianBlur(albedo, (0, 0), max(0.5, region.px(1.0)))
        fuzz = fractal_noise((h, w), max(1.0, region.px(1.2)), rng, octaves=2)
        wear = fractal_noise((h, w), region.px(120), rng, octaves=3)
        albedo = albedo * (1 + 0.18 * (fuzz - 0.5) + 0.12 * (wear - 0.5))[..., None]
        return SurfaceMaps(
            np.clip(albedo, 0, 1).astype(np.float32),
            np.zeros((h, w), np.float32),
            shininess=5.0,
            info={"repeat_mm": repeat_mm},
        )

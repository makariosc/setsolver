"""Stone / ceramic floor tiles with grout lines (strong straight false edges)."""

import numpy as np

from ..color import jitter_hsv_srgb, srgb_to_linear
from ..noise import fractal_noise
from .base import BACKGROUNDS, Background, PlaneRegion, SurfaceMaps

TILE_COLORS = [(0.82, 0.76, 0.66), (0.90, 0.89, 0.86), (0.55, 0.53, 0.50), (0.70, 0.60, 0.50), (0.30, 0.30, 0.32)]
GROUT_COLORS = [(0.35, 0.33, 0.30), (0.85, 0.85, 0.82), (0.20, 0.20, 0.20)]


@BACKGROUNDS.register("tile", weight=1.0)
class Tile(Background):
    def render(self, region: PlaneRegion, rng: np.random.Generator) -> SurfaceMaps:
        h, w = region.shape
        tile = rng.uniform(150, 600)
        grout = rng.uniform(2, 8)
        base = srgb_to_linear(jitter_hsv_srgb(TILE_COLORS[rng.integers(len(TILE_COLORS))], rng, dh=0.03, ds=0.2, dv=0.08))
        grout_c = srgb_to_linear(GROUT_COLORS[rng.integers(len(GROUT_COLORS))])

        X, Y = region.grid_mm()
        ox, oy = rng.uniform(0, tile, 2)
        ix, iy = np.floor((X + ox) / tile), np.floor((Y + oy) / tile)
        fx, fy = (X + ox) - ix * tile, (Y + oy) - iy * tile
        in_grout = (np.minimum(fx, tile - fx) < grout / 2) | (np.minimum(fy, tile - fy) < grout / 2)

        # marble-ish veining + per-tile tone
        veins = fractal_noise((h, w), region.px(rng.uniform(40, 120)), rng, octaves=5)
        veins = np.abs(veins - 0.5) * 2  # ridges near 0
        speck = fractal_noise((h, w), max(1.0, region.px(1.5)), rng, octaves=2)
        tile_tone = ((ix * 7919 + iy * 104729) % 97) / 97.0
        shade = 1 + 0.08 * (tile_tone - 0.5) - 0.15 * np.clip(0.15 - veins, 0, 1) / 0.15 + 0.05 * (speck - 0.5)
        albedo = base[None, None] * shade[..., None]
        albedo[in_grout] = grout_c * (0.9 + 0.2 * speck[in_grout, None])

        gloss = np.where(in_grout, 0.0, rng.uniform(0.03, 0.09)).astype(np.float32)
        return SurfaceMaps(
            np.clip(albedo, 0, 1).astype(np.float32),
            gloss,
            shininess=float(rng.uniform(80, 400)),
            info={"tile_mm": tile, "grout_mm": grout},
        )

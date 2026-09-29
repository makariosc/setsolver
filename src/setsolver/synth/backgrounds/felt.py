"""Felt / neoprene game mat: saturated, matte, fuzzy."""

import numpy as np

from ..color import jitter_hsv_srgb, srgb_to_linear
from ..noise import fractal_noise
from .base import BACKGROUNDS, Background, PlaneRegion, SurfaceMaps

COLORS = [(0.08, 0.40, 0.20), (0.10, 0.20, 0.45), (0.50, 0.08, 0.10), (0.06, 0.06, 0.07), (0.35, 0.10, 0.40)]


@BACKGROUNDS.register("felt", weight=0.7)
class Felt(Background):
    def render(self, region: PlaneRegion, rng: np.random.Generator) -> SurfaceMaps:
        h, w = region.shape
        base = srgb_to_linear(jitter_hsv_srgb(COLORS[rng.integers(len(COLORS))], rng, dh=0.03, ds=0.15, dv=0.2))
        fuzz = fractal_noise((h, w), max(1.0, region.px(0.8)), rng, octaves=3)
        blotch = fractal_noise((h, w), region.px(rng.uniform(30, 90)), rng, octaves=3)
        albedo = base[None, None] * (1 + 0.15 * (fuzz - 0.5) + 0.1 * (blotch - 0.5))[..., None]
        return SurfaceMaps(
            np.clip(albedo, 0, 1).astype(np.float32),
            np.zeros((h, w), np.float32),
            shininess=5.0,
            info={},
        )

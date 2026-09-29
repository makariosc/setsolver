"""Plain laminate / painted / plastic table top: one color with subtle texture."""

import numpy as np

from ..color import jitter_hsv_srgb, srgb_to_linear
from ..noise import fractal_noise
from .base import BACKGROUNDS, Background, PlaneRegion, SurfaceMaps

COLORS = {
    "white": (0.88, 0.88, 0.86),
    "light_grey": (0.70, 0.70, 0.70),
    "dark_grey": (0.25, 0.25, 0.26),
    "black": (0.08, 0.08, 0.09),
    "beige": (0.80, 0.72, 0.60),
    "navy": (0.12, 0.16, 0.30),
    "sage": (0.55, 0.62, 0.52),
}


@BACKGROUNDS.register("solid_table", weight=1.0)
class SolidTable(Background):
    def render(self, region: PlaneRegion, rng: np.random.Generator) -> SurfaceMaps:
        h, w = region.shape
        name = str(rng.choice(list(COLORS)))
        base = srgb_to_linear(jitter_hsv_srgb(COLORS[name], rng, dh=0.02, ds=0.2, dv=0.1))
        mottling = fractal_noise((h, w), region.px(rng.uniform(40, 150)), rng, octaves=3)
        speckle = fractal_noise((h, w), max(1.0, region.px(0.6)), rng, octaves=1)
        albedo = base[None, None] * (1 + 0.06 * (mottling - 0.5) + 0.04 * (speckle - 0.5))[..., None]
        glossy = rng.random() < 0.5
        gloss = np.full((h, w), rng.uniform(0.04, 0.1) if glossy else rng.uniform(0.005, 0.03), np.float32)
        return SurfaceMaps(
            np.clip(albedo, 0, 1).astype(np.float32),
            gloss,
            shininess=float(rng.uniform(150, 500) if glossy else rng.uniform(10, 40)),
            info={"color": name, "glossy": glossy},
        )

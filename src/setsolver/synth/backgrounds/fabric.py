"""Bedsheet / duvet / tablecloth: woven micro-texture with soft wrinkle shading."""

import numpy as np

from ..color import jitter_hsv_srgb, srgb_to_linear
from ..noise import anisotropic_noise, fractal_noise
from .base import BACKGROUNDS, Background, PlaneRegion, SurfaceMaps

COLORS = [
    (0.22, 0.22, 0.22),  # charcoal
    (0.12, 0.14, 0.22),  # navy
    (0.88, 0.88, 0.86),  # white
    (0.70, 0.78, 0.86),  # pale blue
    (0.80, 0.70, 0.66),  # blush
    (0.45, 0.50, 0.40),  # olive
]


@BACKGROUNDS.register("fabric", weight=1.0)
class Fabric(Background):
    def render(self, region: PlaneRegion, rng: np.random.Generator) -> SurfaceMaps:
        h, w = region.shape
        base = srgb_to_linear(jitter_hsv_srgb(COLORS[rng.integers(len(COLORS))], rng, dh=0.03, ds=0.2, dv=0.15))
        X, Y = region.grid_mm()
        period = rng.uniform(0.5, 1.4)
        weave = 0.5 + 0.25 * (np.sin(2 * np.pi * X / period) + np.sin(2 * np.pi * Y / period))
        fibers = fractal_noise((h, w), max(1.0, region.px(2)), rng, octaves=2)

        # Wrinkles: long soft ridges. Baked into albedo as shading (the plane
        # geometry stays flat, which is fine for a first pass).
        ridges = anisotropic_noise((h, w), region.px(rng.uniform(150, 400)), region.px(rng.uniform(20, 60)), rng)
        ridges = 1 - np.abs(ridges - 0.5) * 2
        strength = rng.uniform(0.1, 0.35)
        shade = 1 - strength * (1 - ridges) ** 2 + 0.06 * (weave - 0.5) + 0.05 * (fibers - 0.5)
        albedo = base[None, None] * shade[..., None]
        return SurfaceMaps(
            np.clip(albedo, 0, 1).astype(np.float32),
            np.full((h, w), rng.uniform(0.0, 0.01), np.float32),
            shininess=float(rng.uniform(3, 10)),
            info={"wrinkle_strength": strength},
        )

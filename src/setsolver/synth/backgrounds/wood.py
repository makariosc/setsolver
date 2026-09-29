"""Wooden table / floor made of planks, optionally varnished (glossy)."""

import numpy as np

from ..color import jitter_hsv_srgb, srgb_to_linear
from ..noise import anisotropic_noise, fractal_noise
from .base import BACKGROUNDS, Background, PlaneRegion, SurfaceMaps

# (light, dark) grain colors in sRGB
WOODS = {
    "oak": ((0.78, 0.60, 0.40), (0.55, 0.38, 0.22)),
    "pine": ((0.85, 0.70, 0.48), (0.66, 0.48, 0.28)),
    "walnut": ((0.45, 0.30, 0.20), (0.24, 0.15, 0.10)),
    "ash_grey": ((0.70, 0.66, 0.60), (0.48, 0.44, 0.40)),
    "cherry": ((0.66, 0.38, 0.24), (0.42, 0.22, 0.13)),
}


@BACKGROUNDS.register("wood", weight=2.0)
class Wood(Background):
    def render(self, region: PlaneRegion, rng: np.random.Generator) -> SurfaceMaps:
        h, w = region.shape
        kind = str(rng.choice(list(WOODS)))
        light, dark = (jitter_hsv_srgb(c, rng, dh=0.02, ds=0.15, dv=0.12) for c in WOODS[kind])
        light, dark = srgb_to_linear(light), srgb_to_linear(dark)

        plank_w = rng.uniform(70, 220)  # mm
        plank_len = rng.uniform(600, 2000)
        seam_w = rng.uniform(0.6, 2.5)

        X, Y = region.grid_mm()
        row = np.floor(Y / plank_w).astype(np.int32)
        # per-row random end-joint offsets
        r0 = row.min()
        n_rows = row.max() - r0 + 1
        off_map = rng.uniform(0, plank_len, n_rows).astype(np.float32)[row - r0]
        tone_map = rng.uniform(-0.12, 0.12, n_rows).astype(np.float32)[row - r0]
        col = np.floor((X + off_map) / plank_len)

        # grain: stretched noise + wavy "ring" lines
        grain = anisotropic_noise((h, w), region.px(rng.uniform(80, 250)), region.px(rng.uniform(1.5, 5)), rng)
        warp = anisotropic_noise((h, w), region.px(300), region.px(40), rng)
        rings = 0.5 + 0.5 * np.sin((Y + 25 * warp) * 2 * np.pi / rng.uniform(4, 12) + col * 1.7)
        t = np.clip(0.55 * grain + 0.3 * rings**3 + tone_map + 0.15 * (col % 2), 0, 1)
        albedo = light[None, None] * (1 - t[..., None]) + dark[None, None] * t[..., None]

        # seams between planks (along x) and end joints
        yin = Y - row * plank_w
        xin = (X + off_map) - col * plank_len
        seam = (np.minimum(yin, plank_w - yin) < seam_w) | (np.minimum(xin, plank_len - xin) < seam_w)
        albedo[seam] *= rng.uniform(0.35, 0.6)

        # a few scratches / dents
        wear = fractal_noise((h, w), region.px(20), rng, octaves=4)
        albedo *= (1 - 0.08 * np.clip(wear - 0.7, 0, 1) / 0.3)[..., None]

        varnished = rng.random() < 0.6
        gloss = np.full((h, w), rng.uniform(0.04, 0.12) if varnished else rng.uniform(0.0, 0.02), np.float32)
        gloss[seam] *= 0.3
        return SurfaceMaps(
            albedo.astype(np.float32),
            gloss,
            shininess=float(rng.uniform(80, 400) if varnished else rng.uniform(5, 20)),
            info={"wood": kind, "plank_w_mm": plank_w, "varnished": varnished},
        )

"""Card surface finish: how glossy the cards are and their fine texture.

Real SET cards are coated card stock, often with an embossed "linen" finish.
Under a lamp at the mirror angle the coating reflects a broad glare that
bleaches the ink toward white, and the linen bumps break that highlight into
a grid of bright specks. Both confuse the classifier (a glare-washed symbol
loses its fill or disappears; speckled solid ink looks striped), so scenes
sample a finish per deck:

- `gloss`/`shininess`: Blinn-Phong specular strength and tightness. Matte
  decks (most scenes) keep the old faint sheen; glossy decks glare.
- a height field -> per-pixel surface slopes (nx, ny) in the card's texture
  frame: crossed sine "weave" at a sub-millimeter pitch plus paper grain. The
  slopes tilt the normal in the specular term only (the bumps are far too
  small to change diffuse shading).
- ink is slightly more or less glossy than the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .noise import fractal_noise


@dataclass
class CardFinish:
    glossy: bool
    gloss: float  # specular albedo
    shininess: float  # Blinn-Phong exponent: low = broad wash, high = tight hot spot
    linen_pitch_mm: float  # weave period
    linen_slope: float  # max surface slope from the weave (tan of tilt angle)
    grain_slope: float  # random bumps on top
    ink_gloss: float  # ink gloss relative to paper

    @classmethod
    def sample(cls, rng: np.random.Generator, glossy: bool) -> "CardFinish":
        if glossy:
            gloss = rng.uniform(0.12, 0.5)
            shininess = float(np.exp(rng.uniform(np.log(25), np.log(350))))
        else:
            gloss = rng.uniform(0.02, 0.08)
            shininess = rng.uniform(15, 80)
        linen = rng.random() < 0.7  # plain smooth stock otherwise
        return cls(
            glossy=glossy,
            gloss=gloss,
            shininess=shininess,
            linen_pitch_mm=rng.uniform(0.3, 0.6),
            linen_slope=rng.uniform(0.04, 0.15) if linen else 0.0,
            grain_slope=rng.uniform(0.01, 0.06),
            ink_gloss=rng.uniform(0.7, 1.3),
        )

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def render_finish(finish: CardFinish, ink: np.ndarray, ppm: float, rng: np.random.Generator):
    """Per-texel (gloss, nx, ny) for a card texture of ink.shape at ppm px/mm.
    nx/ny are surface slopes along the texture's x (right) and y (down) axes."""
    h, w = ink.shape
    gloss = finish.gloss * (1 + (finish.ink_gloss - 1) * ink)
    gloss = gloss * (1 + 0.25 * (fractal_noise((h, w), 20 * ppm, rng, octaves=2) - 0.5))  # uneven coating
    nx = np.zeros((h, w), np.float32)
    ny = np.zeros((h, w), np.float32)
    if finish.linen_slope > 0:
        # crossed weave; each thread's strength wanders so the pattern isn't a perfect grid
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / ppm
        k = 2 * np.pi / finish.linen_pitch_mm
        ph = rng.uniform(0, 2 * np.pi, 2)
        wx = 0.6 + 0.8 * fractal_noise((h, w), 3 * ppm, rng, octaves=2)
        wy = 0.6 + 0.8 * fractal_noise((h, w), 3 * ppm, rng, octaves=2)
        nx += finish.linen_slope * wx * np.cos(k * xx + ph[0])
        ny += finish.linen_slope * wy * np.cos(k * yy + ph[1])
    # paper grain: gradient of a fine random height field
    hgt = fractal_noise((h, w), max(1.0, 0.15 * ppm), rng, octaves=2).astype(np.float32)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8
    s = finish.grain_slope / max(float(np.abs(np.stack([gx, gy])).std()) * 3, 1e-6)
    nx += gx * s
    ny += gy * s
    return gloss.astype(np.float32), nx, ny

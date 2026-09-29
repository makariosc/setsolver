"""Color helpers. All rendering happens in linear RGB; sRGB only at the edges."""

from __future__ import annotations

import colorsys

import numpy as np


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float32), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


def hex_to_linear(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return srgb_to_linear([int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def jitter_hsv_srgb(rgb_srgb, rng, dh=0.0, ds=0.0, dv=0.0) -> np.ndarray:
    """Jitter an sRGB (0-1) color in HSV space. dh in hue turns (0-1)."""
    h, s, v = colorsys.rgb_to_hsv(*rgb_srgb)
    h = (h + rng.uniform(-dh, dh)) % 1.0
    s = float(np.clip(s * (1 + rng.uniform(-ds, ds)), 0, 1))
    v = float(np.clip(v * (1 + rng.uniform(-dv, dv)), 0, 1))
    return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)


def kelvin_to_linear_rgb(kelvin: float) -> np.ndarray:
    """Approximate blackbody color (Tanner Helland fit), normalized so max channel = 1."""
    t = kelvin / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    rgb = np.clip([r, g, b], 1, 255) / 255.0
    lin = srgb_to_linear(rgb)
    return (lin / lin.max()).astype(np.float32)

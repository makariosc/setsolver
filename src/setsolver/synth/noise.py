"""Cheap procedural noise built from upsampled random grids (value noise)."""

from __future__ import annotations

import cv2
import numpy as np


def value_noise(shape: tuple[int, int], cell_px: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth noise in [0, 1] with features roughly `cell_px` pixels across."""
    h, w = shape
    gh = max(2, int(np.ceil(h / cell_px)) + 2)
    gw = max(2, int(np.ceil(w / cell_px)) + 2)
    grid = rng.random((gh, gw), dtype=np.float32)
    big = cv2.resize(grid, (int(gw * cell_px), int(gh * cell_px)), interpolation=cv2.INTER_CUBIC)
    oy = rng.integers(0, max(1, big.shape[0] - h))
    ox = rng.integers(0, max(1, big.shape[1] - w))
    out = big[oy : oy + h, ox : ox + w]
    if out.shape != (h, w):  # tiny rounding mismatch
        out = cv2.resize(out, (w, h))
    return np.clip(out, 0, 1)


def fractal_noise(
    shape: tuple[int, int],
    cell_px: float,
    rng: np.random.Generator,
    octaves: int = 4,
    persistence: float = 0.5,
) -> np.ndarray:
    """Sum of value-noise octaves, normalized to [0, 1]."""
    total = np.zeros(shape, np.float32)
    amp, norm = 1.0, 0.0
    for i in range(octaves):
        cell = cell_px / (2**i)
        if cell < 1.0:
            break
        total += amp * value_noise(shape, cell, rng)
        norm += amp
        amp *= persistence
    total /= norm
    lo, hi = np.percentile(total[::5, ::5], [1, 99])  # subsampled: fast on big canvases
    return np.clip((total - lo) / max(hi - lo, 1e-6), 0, 1)


def anisotropic_noise(
    shape: tuple[int, int], cell_along: float, cell_across: float, rng: np.random.Generator, octaves: int = 3
) -> np.ndarray:
    """Noise stretched along x (e.g. wood grain / fibers)."""
    h, w = shape
    sx = cell_along / cell_across
    small_w = max(4, int(np.ceil(w / sx)))
    n = fractal_noise((h, small_w), cell_across, rng, octaves=octaves)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)

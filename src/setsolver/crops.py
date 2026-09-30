"""Straighten a card from its 4 image corners into a fixed-size portrait crop.

Used both to build classifier training data (from generator labels) and at
inference time (from detector corners), so the classifier always sees crops
made the same way.
"""

from __future__ import annotations

import cv2
import numpy as np

CROP_W, CROP_H = 160, 256  # portrait, ~2.8 px/mm for a 57 x 89 mm card


def portrait_order(corners: np.ndarray) -> np.ndarray:
    """Reorder 4 corners (any cyclic order, either direction) so that
    corners[0]->corners[1] is a *short* edge and the sequence runs clockwise in
    image coordinates (y down). The result maps onto a portrait crop as
    TL, TR, BR, BL. The card's 180-degree ambiguity is left unresolved (the
    card's attributes don't depend on it).
    """
    c = np.asarray(corners, np.float64).reshape(4, 2)
    # clockwise in image coords <=> positive signed area with y pointing down
    x, y = c[:, 0], c[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) < 0:
        c = c[::-1]
    edges = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    # edge i goes c[i] -> c[i+1]; pick the start so edge 0 and edge 2 are the short pair
    start = 0 if edges[0] + edges[2] <= edges[1] + edges[3] else 1
    return np.roll(c, -start, axis=0)


def jitter_corners(corners: np.ndarray, rng: np.random.Generator, frac: float) -> np.ndarray:
    """Perturb corners by Gaussian noise with sigma = frac x the card's short side,
    mimicking detector imprecision."""
    c = np.asarray(corners, np.float64).reshape(4, 2)
    short = min(np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1]))
    return c + rng.normal(0, frac * short, c.shape)


def warp_card(image: np.ndarray, corners: np.ndarray, size: tuple[int, int] = (CROP_W, CROP_H)) -> np.ndarray:
    """Perspective-warp the card to a `size` (w, h) portrait crop.

    Parts of the card beyond the photo's edge (clipped cards) are filled with
    black: an unambiguous "no data" signal. (Replicating the edge pixels
    instead smears symbols into fake shapes.)"""
    w, h = size
    src = portrait_order(corners)
    # Work on the card's neighbourhood only (much cheaper than the whole photo).
    shrink = np.linalg.norm(src[1] - src[0]) / w
    sigma = 0.5 * shrink if shrink > 1.5 else 0.0
    pad = int(np.ceil(3 * sigma)) + 2
    H, W = image.shape[:2]
    x0 = int(np.clip(np.floor(src[:, 0].min()) - pad, 0, W - 1))
    y0 = int(np.clip(np.floor(src[:, 1].min()) - pad, 0, H - 1))
    x1 = int(np.clip(np.ceil(src[:, 0].max()) + pad + 1, x0 + 1, W))
    y1 = int(np.clip(np.ceil(src[:, 1].max()) + pad + 1, y0 + 1, H))
    region = image[y0:y1, x0:x1]
    # warpPerspective has no INTER_AREA; when shrinking a lot, pre-blur so
    # stripes don't alias into moire.
    if sigma > 0:
        region = cv2.GaussianBlur(region, (0, 0), sigma)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform((src - [x0, y0]).astype(np.float32), dst)
    # The region is clamped to the photo, so anything outside it is outside the photo.
    return cv2.warpPerspective(region, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def add_glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Training augmentation: paint a specular glare patch onto a card crop.

    Real photos under a lamp at the mirror angle get a broad highlight that
    bleaches the ink toward white (fills vanish, whole symbols can fade out),
    often broken into fine specks by the card's linen texture. The patch is a
    soft ellipse or a long streak (tube light), optionally modulated by fine
    speckle or a crosshatch, blended toward a near-white light color.
    img: HxWx3 uint8 (any channel order). Returns a new uint8 image.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
    ang = rng.uniform(0, np.pi)
    if rng.random() < 0.3:  # streak
        sa, sb = rng.uniform(1.0, 3.0) * h, rng.uniform(0.06, 0.25) * w
    else:
        sa, sb = rng.uniform(0.15, 0.8) * w, rng.uniform(0.15, 0.8) * w
    c, s = np.cos(ang), np.sin(ang)
    u, v = (xx - cx) * c + (yy - cy) * s, -(xx - cx) * s + (yy - cy) * c
    m = np.exp(-0.5 * ((u / sa) ** 2 + (v / sb) ** 2))
    m = np.clip(m * rng.uniform(1.0, 2.5), 0, 1)  # >1 flattens the top: a blown-out core
    kind = rng.choice(["smooth", "speckle", "crosshatch"], p=[0.35, 0.4, 0.25])
    if kind == "speckle":
        cell = rng.uniform(1.0, 2.5)
        n = cv2.resize(rng.random((int(h / cell) + 2, int(w / cell) + 2), dtype=np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
        # sparse bright flecks inside the highlight (linen bumps catching the light)
        m = m * (rng.uniform(0.3, 0.7) + 0.9 * np.clip(n, 0, 1) ** rng.uniform(4, 8))
    elif kind == "crosshatch":
        p, t = rng.uniform(2.0, 4.5), rng.uniform(0, np.pi)
        g = (0.5 + 0.5 * np.cos(2 * np.pi * (xx * np.cos(t) + yy * np.sin(t)) / p)) * \
            (0.5 + 0.5 * np.cos(2 * np.pi * (-xx * np.sin(t) + yy * np.cos(t)) / p * rng.uniform(0.8, 1.25)))
        m = m * (rng.uniform(0.3, 0.7) + 0.6 * g ** 2)
    m = np.clip(m * rng.uniform(0.5, 1.0), 0, 1)[..., None]
    light = 255 * np.clip(1 + rng.normal(0, 0.03, 3), 0.9, 1.0).astype(np.float32)
    out = img.astype(np.float32) * (1 - m) + light * m
    return np.clip(out, 0, 255).astype(np.uint8)

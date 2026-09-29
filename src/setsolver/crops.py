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

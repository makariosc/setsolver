"""When to trust a card that touches the photo edge (mirrors web/js/pipeline.js).

The app reads edge cards but counts one only if every attribute is at least
EDGE_CONFIDENCE sure AND its corners still describe a real card by either
check: card_shape_error (a 57x88 mm rectangle under some perspective, assumed
~70 deg phone FOV), aspect_dev (same proportions as the photo's whole cards) or
real_card_dev (the proportions of a real card, 57:88: cutting a card off always
moves its outline away from that). Corners squashed against the edge fail all
three. On synthetic clipped cards (eval/clipped_gate.py): ~56% kept at 98.9%
right with the YOLO detector, ~98% at 99.7% with CardCornerNet.
"""
from __future__ import annotations

import cv2
import numpy as np

EDGE_CONFIDENCE = 0.95
EDGE_SHAPE_TOL = 0.15
EDGE_ASPECT_TOL = 0.15
EDGE_REAL_TOL = 0.10    # side ratio within 10% of a real card's: only a (nearly) whole card has it
CARD_MM = (57.0, 88.0)


def card_shape_error(q, w: int, h: int, fov: float = 70.0) -> float:
    """0 = exactly a 57x88 rectangle seen through a pinhole camera: non-orthogonality
    + scale mismatch of the first two columns of K^-1 H (H: card rectangle -> quad)."""
    f = (np.hypot(w, h) / 2) / np.tan(np.radians(fov) / 2)
    Ki = np.linalg.inv(np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]]))
    best = np.inf
    for a, b in (CARD_MM, CARD_MM[::-1]):
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [a, 0], [a, b], [0, b]]), np.float32(q))
        A = Ki @ H
        c1, c2 = A[:, 0], A[:, 1]
        n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
        best = min(best, abs(np.dot(c1, c2)) / (n1 * n2) + abs(n1 - n2) / max(n1, n2))
    return float(best)


def side_ratio(q) -> float:
    """Short / long side of a quad (means of opposite sides)."""
    q = np.asarray(q, np.float64)
    e = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1)
    a, b = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
    return float(min(a, b) / max(a, b))


def aspect_dev(q, whole_cards) -> float:
    """|side ratio / reference - 1|; reference: median of the photo's whole (not
    edge) cards, else a real card's 57/88."""
    ref = float(np.median([side_ratio(o) for o in whole_cards])) if len(whole_cards) else CARD_MM[0] / CARD_MM[1]
    return abs(side_ratio(q) / ref - 1)


def real_card_dev(q) -> float:
    """|side ratio / 57:88 - 1|."""
    return abs(side_ratio(q) / (CARD_MM[0] / CARD_MM[1]) - 1)


def edge_card_ok(q, minp: float, w: int, h: int, whole_cards) -> bool:
    return minp >= EDGE_CONFIDENCE and (card_shape_error(q, w, h) <= EDGE_SHAPE_TOL or aspect_dev(q, whole_cards) <= EDGE_ASPECT_TOL
                                        or real_card_dev(q) <= EDGE_REAL_TOL)

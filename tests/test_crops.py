"""Card straightening: corner ordering and crop orientation."""

import cv2
import numpy as np
import pytest

from setsolver.crops import portrait_order, warp_card


def _card(angle_deg: float, center=(300.0, 300.0), w=57.0 * 3, h=89.0 * 3) -> np.ndarray:
    """Portrait card corners TL, TR, BR, BL (image coords), rotated by angle."""
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return local @ R.T + center


@pytest.mark.parametrize("angle", [0, 17, 45, 90, 133, 180, 260, 315])
@pytest.mark.parametrize("start", [0, 1, 2, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_portrait_order_any_input_order(angle, start, reverse):
    c = np.roll(_card(angle), -start, axis=0)
    if reverse:
        c = c[::-1]
    p = portrait_order(c)
    e = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    assert e[0] < e[1] and e[2] < e[3], "edge 0->1 must be a short edge"
    x, y = p[:, 0], p[:, 1]
    assert np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) > 0, "must be clockwise (y down)"
    # same corner set, and equal to the card's TL..BR order up to a 180-degree turn
    truth = _card(angle)
    assert any(np.allclose(p, np.roll(truth, s, axis=0)) for s in (0, 2))


def test_crop_not_mirrored():
    """A mark in the card's top-left quadrant must stay top-left (or bottom-right,
    for the allowed 180-degree turn), never top-right/bottom-left."""
    img = np.full((600, 600, 3), 40, np.uint8)
    c = _card(30)
    cv2.fillPoly(img, [np.round(c).astype(np.int32)], (255, 255, 255))
    mark = c[0] + 0.2 * (c[1] - c[0]) + 0.15 * (c[3] - c[0])  # near the TL corner, inside
    cv2.circle(img, tuple(np.round(mark).astype(int)), 12, (0, 0, 255), -1)
    crop = warp_card(img, np.roll(c, -1, axis=0)[::-1])  # scrambled input order
    red = (crop[..., 2] > 200) & (crop[..., 1] < 80)
    ys, xs = np.nonzero(red)
    cx, cy = xs.mean() / crop.shape[1], ys.mean() / crop.shape[0]
    assert (cx < 0.5 and cy < 0.5) or (cx > 0.5 and cy > 0.5)


def test_clipped_card_outside_is_black():
    """The part of a card beyond the photo edge must be filled black, not with
    replicated edge pixels (which smear symbols into fake shapes)."""
    img = np.full((400, 400, 3), 200, np.uint8)
    c = _card(0, center=(40.0, 200.0))  # left half of the card lies at x < 0
    crop = warp_card(img, c)
    left, right = crop[:, : crop.shape[1] // 4], crop[:, -crop.shape[1] // 4 :]
    assert left.max() == 0, "out-of-frame area should be black"
    assert right.min() > 150, "in-frame area should be image content"

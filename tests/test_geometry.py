"""Geometry invariants of the scene generator."""

import numpy as np
import pytest

from setsolver.synth.camera import sample_camera
from setsolver.synth.cards import CARD_H_MM, CARD_W_MM
from setsolver.synth.layout import Placement
from setsolver.synth.scene import _card_to_plane


def _texture_to_image(seed: int):
    rng = np.random.default_rng(seed)
    p = Placement(rng.uniform(-100, 100, 2), rng.uniform(0, 2 * np.pi))
    cam = sample_camera(rng, 960, 1280, p.corners(), 0.8)
    ppm = 10.0
    M = cam.H @ _card_to_plane(p, ppm)
    return p, cam, ppm, lambda u, v: (lambda q: q[:2] / q[2])(M @ np.array([u, v, 1.0]))


@pytest.mark.parametrize("seed", range(20))
def test_cards_are_not_mirrored(seed):
    """Texture -> image must preserve handedness, or squiggles render mirrored."""
    _, _, _, f = _texture_to_image(seed)
    J = np.column_stack([f(286, 445) - f(285, 445), f(285, 446) - f(285, 445)])
    assert np.linalg.det(J) > 0


@pytest.mark.parametrize("seed", range(20))
def test_texture_corners_match_labels(seed):
    """The rendered card's TL/TR/BR/BL must land on Placement.corners()."""
    p, cam, ppm, f = _texture_to_image(seed)
    w, h = CARD_W_MM * ppm, CARD_H_MM * ppm
    tex = [(-0.5, -0.5), (w - 0.5, -0.5), (w - 0.5, h - 0.5), (-0.5, h - 0.5)]
    for t, q in zip(tex, cam.project_plane(p.corners())):
        assert np.linalg.norm(f(*t) - q) < 1e-6

"""Distractors (non-card rectangles) render, land in view, and never get labels."""

import numpy as np
import pytest

from setsolver.synth import SceneConfig, generate_scene
from setsolver.synth.distractors import DISTRACTORS
from setsolver.synth.scene import _rect_corners, _tex_to_plane


@pytest.mark.parametrize("name", DISTRACTORS.names())
def test_distractor_renders(name):
    rng = np.random.default_rng(0)
    for ppm in (2.0, 8.0):
        d = DISTRACTORS.get(name).obj(rng)
        alb, a = d.render(ppm, rng)
        w, h = d.size_mm
        assert alb.shape[:2] == a.shape and abs(a.shape[1] - w * ppm) <= 1 and abs(a.shape[0] - h * ppm) <= 1
        assert alb.dtype == np.float32 and 0 <= alb.min() and alb.max() <= 1
        assert a.mean() > 0.7  # mostly opaque rectangle (round coasters ~0.78)


def test_rect_corners_match_texture_mapping():
    """_rect_corners and _tex_to_plane must agree (same convention as cards)."""
    c, ang, w, h, ppm = np.array([12.0, -30.0]), 0.7, 150.0, 90.0, 5.0
    A = _tex_to_plane(c, ang, w, h, ppm)
    tex = [(-0.5, -0.5), (w * ppm - 0.5, -0.5), (w * ppm - 0.5, h * ppm - 0.5), (-0.5, h * ppm - 0.5)]
    for t, q in zip(tex, _rect_corners(c, ang, w, h)):
        p = A @ np.array([*t, 1.0])
        assert np.allclose(p[:2] / p[2], q)


@pytest.mark.parametrize("seed", range(6))
def test_scenes_with_distractors(seed):
    cfg = SceneConfig(width=480, height=640, distractor_prob=1.0)
    img, labels = generate_scene(np.random.default_rng([99, seed]), cfg)
    assert img.shape[:2] in ((640, 480), (480, 640))
    for d in labels["distractors"]:
        assert d["type"] in DISTRACTORS.names() and d["layer"] in ("under", "top")
    # labels are only ever SET cards
    assert all({"number", "color", "shape", "shading", "corners"} <= set(c) for c in labels["cards"])

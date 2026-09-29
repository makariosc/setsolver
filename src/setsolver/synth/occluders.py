"""Objects lying on the table on top of the cards: fingers, coins, phones, pens,
paper. They occlude card corners/edges (so the detector learns to infer hidden
corners) and act as hard negatives (paper and sticky notes are white-ish
rectangles that are *not* SET cards).

Occluders are flat shapes on the table plane defined by a signed distance
function (mm), so they go through the same camera, shadows and lighting as
everything else. Fingers are not really flat, but they rest close to the table
where they touch a card, and the drop shadow sells the height.

To add one: a function `(rng, ctx) -> Occluder` decorated with
`@OCCLUDERS.register("name", weight=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .color import jitter_hsv_srgb, srgb_to_linear
from .layout import Placement
from .registry import Registry

OCCLUDERS: Registry = Registry("occluder")


@dataclass
class OccluderContext:
    placements: list[Placement]
    toward: np.ndarray  # unit plane vector from the photographer into the scene
    near_edge_pts: np.ndarray  # plane points just beyond the image's bottom edge


@dataclass
class Occluder:
    name: str
    sdf: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (X, Y) mm -> signed distance, <0 inside
    color: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (X, Y) -> HxWx3 linear albedo
    gloss: float
    shininess: float
    lift_mm: float  # height above the table, for the drop shadow
    bounds: tuple[np.ndarray, np.ndarray]  # (min_xy, max_xy) in mm, loose
    info: dict = field(default_factory=dict)


# --- SDF primitives (plane mm) ----------------------------------------------------


def sdf_capsule(a: np.ndarray, b: np.ndarray, r: float):
    ab = b - a
    L2 = float(ab @ ab)

    def f(X, Y):
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0, 1)
        return np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1])) - r

    return f


def sdf_circle(c: np.ndarray, r: float):
    return lambda X, Y: np.hypot(X - c[0], Y - c[1]) - r


def sdf_rounded_rect(c: np.ndarray, half: tuple[float, float], radius: float, angle: float):
    cs, sn = np.cos(angle), np.sin(angle)

    def f(X, Y):
        u = np.abs((X - c[0]) * cs + (Y - c[1]) * sn) - (half[0] - radius)
        v = np.abs(-(X - c[0]) * sn + (Y - c[1]) * cs) - (half[1] - radius)
        outside = np.hypot(np.maximum(u, 0), np.maximum(v, 0))
        return outside + np.minimum(np.maximum(u, v), 0) - radius

    return f


def _solid(rgb_lin):
    return lambda X, Y: np.broadcast_to(rgb_lin, X.shape + (3,)).astype(np.float32)


def _rand_card_point(rng, ctx: OccluderContext, corner_bias: float = 0.6) -> np.ndarray:
    """A point on a random card, often close to one of its corners."""
    p = ctx.placements[rng.integers(len(ctx.placements))]
    corners = p.corners()
    if rng.random() < corner_bias:
        c = corners[rng.integers(4)]
        return c + (p.center - c) * rng.uniform(0.0, 0.25) + rng.normal(0, 3, 2)
    w = rng.uniform(0.1, 0.9, 2)
    return corners[0] + (corners[1] - corners[0]) * w[0] + (corners[3] - corners[0]) * w[1]


# --- occluders ----------------------------------------------------------------------

SKIN = [(0.95, 0.80, 0.70), (0.88, 0.68, 0.55), (0.76, 0.55, 0.40), (0.55, 0.38, 0.27), (0.36, 0.24, 0.17)]


@OCCLUDERS.register("finger", weight=2.0)
def finger(rng, ctx):
    """A fingertip reaching in from the photographer's side and resting on a card."""
    tip = _rand_card_point(rng, ctx, corner_bias=0.7)
    base = ctx.near_edge_pts[rng.integers(len(ctx.near_edge_pts))]
    d = tip - base
    d = d / (np.linalg.norm(d) + 1e-9)
    d = d * np.cos(rng.normal(0, 0.3)) + np.array([-d[1], d[0]]) * np.sin(rng.normal(0, 0.3))
    length = rng.uniform(70, 160)
    r = rng.uniform(7, 10)
    a = tip - d * length
    skin = srgb_to_linear(jitter_hsv_srgb(SKIN[rng.integers(len(SKIN))], rng, dh=0.02, ds=0.1, dv=0.08))
    nail_c = tip - d * r * 0.9
    ab = tip - a

    def color(X, Y):
        # darker toward the finger's sides (it is round), slightly pink nail at the tip
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / float(ab @ ab), 0, 1)
        dist = np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1]))
        shade = 0.65 + 0.35 * np.sqrt(np.clip(1 - (dist / r) ** 2, 0, 1))
        out = skin[None, None, :] * shade[..., None]
        nail = np.hypot(X - nail_c[0], Y - nail_c[1]) < r * 0.55
        out[nail] = out[nail] * 0.85 + 0.15 * np.array([0.9, 0.7, 0.7], np.float32)
        return out.astype(np.float32)

    pts = np.array([a, tip])
    return Occluder("finger", sdf_capsule(a, tip, r), color, 0.03, 20, rng.uniform(4, 15),
                    (pts.min(0) - r, pts.max(0) + r), {"radius_mm": r})


@OCCLUDERS.register("coin", weight=1.0)
def coin(rng, ctx):
    c = _rand_card_point(rng, ctx, corner_bias=0.4)
    r = rng.uniform(9, 14)
    base = [(0.75, 0.75, 0.75), (0.72, 0.45, 0.30), (0.80, 0.68, 0.35)][rng.integers(3)]
    col = srgb_to_linear(jitter_hsv_srgb(base, rng, dv=0.1))

    def color(X, Y):
        rr = np.hypot(X - c[0], Y - c[1]) / r
        rim = np.where(rr > 0.85, 0.8, 1.0)  # raised rim reads darker
        return (col[None, None, :] * rim[..., None]).astype(np.float32)

    return Occluder("coin", sdf_circle(c, r), color, 0.15, rng.uniform(30, 120), 1.5,
                    (c - r, c + r), {"radius_mm": r})


@OCCLUDERS.register("phone", weight=0.8)
def phone(rng, ctx):
    """A phone lying partly over the cards (dark, glossy)."""
    edge_pt = _rand_card_point(rng, ctx, corner_bias=0.6)
    half = (rng.uniform(34, 40), rng.uniform(70, 80))
    angle = rng.uniform(0, np.pi)
    # shift the phone so that its edge (not center) lands on the card point
    out_dir = np.array([np.cos(angle), np.sin(angle)])
    c = edge_pt + out_dir * half[0] * rng.uniform(0.6, 0.95)
    col = srgb_to_linear(jitter_hsv_srgb([(0.05, 0.05, 0.06), (0.25, 0.25, 0.27), (0.75, 0.75, 0.78)][rng.integers(3)], rng, dv=0.1))
    R = max(half)
    return Occluder("phone", sdf_rounded_rect(c, half, rng.uniform(6, 10), angle), _solid(col), 0.12,
                    rng.uniform(200, 600), 8.0, (c - R, c + R))


@OCCLUDERS.register("pen", weight=0.8)
def pen(rng, ctx):
    mid = _rand_card_point(rng, ctx, corner_bias=0.3)
    ang = rng.uniform(0, np.pi)
    d = np.array([np.cos(ang), np.sin(ang)])
    L = rng.uniform(120, 150)
    r = rng.uniform(4, 5.5)
    a, b = mid - d * L / 2, mid + d * L / 2
    col = srgb_to_linear(jitter_hsv_srgb([(0.1, 0.1, 0.12), (0.1, 0.2, 0.6), (0.8, 0.1, 0.1), (0.9, 0.9, 0.9), (0.95, 0.8, 0.1)][rng.integers(5)], rng, dv=0.1))
    pts = np.array([a, b])
    return Occluder("pen", sdf_capsule(a, b, r), _solid(col), 0.08, 80, r * 2, (pts.min(0) - r, pts.max(0) + r))


@OCCLUDERS.register("paper", weight=1.0)
def paper(rng, ctx):
    """Sticky note / receipt / business card: rectangle, often white. Hard negative."""
    kind = rng.choice(["sticky", "receipt", "business_card"])
    if kind == "sticky":
        half = (38.0, 38.0)
        base = [(0.98, 0.93, 0.45), (0.98, 0.75, 0.80), (0.70, 0.90, 0.95)][rng.integers(3)]
    elif kind == "receipt":
        half = (rng.uniform(28, 40), rng.uniform(60, 120))
        base = (0.95, 0.95, 0.93)
    else:
        half = (44.5, 25.5)  # 89 x 51 mm
        base = (0.94, 0.94, 0.92)
    col = srgb_to_linear(jitter_hsv_srgb(base, rng, dv=0.03))
    # half the time lying on a card, otherwise nearby on the table
    if rng.random() < 0.5:
        c = _rand_card_point(rng, ctx, corner_bias=0.6) + rng.normal(0, 25, 2)
    else:
        p = ctx.placements[rng.integers(len(ctx.placements))]
        c = p.center + rng.normal(0, 1, 2) / np.sqrt(2) * rng.uniform(80, 160)
    R = max(half) * 1.5
    return Occluder("paper", sdf_rounded_rect(c, half, rng.uniform(0.3, 1.5), rng.uniform(0, np.pi)), _solid(col),
                    0.02, 20, 0.3, (c - R, c + R), {"kind": str(kind)})

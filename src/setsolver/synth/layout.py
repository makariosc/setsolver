"""Card arrangements on the table plane (millimeters).

To add a layout: a function `(rng) -> list[Placement]` decorated with
`@LAYOUTS.register("name", weight=...)`. The scene draws the cards and centers
the camera on the layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cards import CARD_H_MM, CARD_W_MM
from .registry import Registry

LAYOUTS: Registry = Registry("layout")


@dataclass
class Placement:
    center: np.ndarray  # (x, y) mm on the table
    angle: float  # radians; 0 = portrait card aligned with the table axes

    def corners(self) -> np.ndarray:
        """Card corners on the table in card order TL, TR, BR, BL (portrait frame).

        Table coordinates have y pointing away from a viewer looking down, so the
        card's top edge is at +y (matches the texture mapping in scene.py)."""
        hw, hh = CARD_W_MM / 2, CARD_H_MM / 2
        local = np.array([[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]])
        c, s = np.cos(self.angle), np.sin(self.angle)
        return local @ np.array([[c, s], [-s, c]]) + self.center


@LAYOUTS.register("grid", weight=4.0)
def grid(rng: np.random.Generator) -> list[Placement]:
    """The usual tableau: rows x cols with small jitter."""
    rows, cols = [(3, 4), (4, 3), (3, 3), (3, 5), (5, 3), (4, 4), (3, 6), (2, 3)][
        rng.choice(8, p=[0.3, 0.15, 0.1, 0.15, 0.1, 0.05, 0.05, 0.1])
    ]
    landscape = rng.random() < 0.5
    base = np.pi / 2 if landscape else 0.0
    cw, ch = (CARD_H_MM, CARD_W_MM) if landscape else (CARD_W_MM, CARD_H_MM)
    gap = rng.uniform(4, 25, 2)
    pos_jit, ang_jit = rng.uniform(0.5, 5), np.radians(rng.uniform(0.5, 5))
    out = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2) * (cw + gap[0])
            y = (r - (rows - 1) / 2) * (ch + gap[1])
            flip = np.pi if rng.random() < 0.5 else 0.0  # cards are 180-degree symmetric
            out.append(Placement(np.array([x, y]) + rng.normal(0, pos_jit, 2), base + flip + rng.normal(0, ang_jit)))
    return out


@LAYOUTS.register("scatter", weight=1.5)
def scatter(rng: np.random.Generator) -> list[Placement]:
    """Loosely scattered cards at arbitrary angles; occasional overlaps."""
    n = int(rng.integers(3, 16))
    radius = np.sqrt(n) * rng.uniform(45, 70)
    min_dist = rng.choice([70.0, 50.0], p=[0.8, 0.2])  # 50 mm allows partial overlaps
    out: list[Placement] = []
    tries = 0
    while len(out) < n and tries < 2000:
        tries += 1
        p = rng.uniform(-radius, radius, 2)
        if np.linalg.norm(p) > radius or any(np.linalg.norm(p - q.center) < min_dist for q in out):
            continue
        out.append(Placement(p, rng.uniform(0, 2 * np.pi)))
    return out


@LAYOUTS.register("few", weight=1.0)
def few(rng: np.random.Generator) -> list[Placement]:
    """1-4 cards, usually in a row (close-up shots of a candidate set)."""
    n = int(rng.integers(1, 5))
    landscape = rng.random() < 0.5
    base = np.pi / 2 if landscape else 0.0
    step = (CARD_H_MM if landscape else CARD_W_MM) + rng.uniform(5, 30)
    return [
        Placement(np.array([(i - (n - 1) / 2) * step, 0.0]) + rng.normal(0, 3, 2), base + rng.normal(0, np.radians(4)))
        for i in range(n)
    ]

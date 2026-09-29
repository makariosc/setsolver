"""Background (table surface) interface.

A background renders *material maps* for a rectangle of the table plane, in
millimeters, at a requested resolution. It knows nothing about the camera or
lighting: the scene warps these maps through the camera and lights them.

To add a background: create a module in this package that defines a class with
a `render(region, rng) -> SurfaceMaps` method, decorated with
`@BACKGROUNDS.register("name", weight=...)`. It is picked up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..registry import Registry


@dataclass
class PlaneRegion:
    x0: float  # mm
    y0: float  # mm
    width_mm: float
    height_mm: float
    px_per_mm: float

    @property
    def shape(self) -> tuple[int, int]:
        return (
            max(2, int(np.ceil(self.height_mm * self.px_per_mm))),
            max(2, int(np.ceil(self.width_mm * self.px_per_mm))),
        )

    def px(self, mm: float) -> float:
        """Convert a size in mm to pixels of this region's canvas."""
        return mm * self.px_per_mm

    def grid_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Plane coordinates (X, Y) in mm of every canvas pixel center."""
        h, w = self.shape
        xs = self.x0 + (np.arange(w, dtype=np.float32) + 0.5) / self.px_per_mm
        ys = self.y0 + (np.arange(h, dtype=np.float32) + 0.5) / self.px_per_mm
        return np.meshgrid(xs, ys)


@dataclass
class SurfaceMaps:
    albedo: np.ndarray  # HxWx3 linear RGB, 0-1
    gloss: np.ndarray  # HxW specular strength (~0-0.15); 0 = matte
    shininess: float  # Blinn-Phong exponent: higher = tighter highlight
    info: dict  # sampled parameters, stored in labels for debugging


class Background:
    name = "base"

    def render(self, region: PlaneRegion, rng: np.random.Generator) -> SurfaceMaps:
        raise NotImplementedError


BACKGROUNDS: Registry[type[Background]] = Registry("background")

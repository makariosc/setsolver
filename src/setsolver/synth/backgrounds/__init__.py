"""Table surfaces. Every module in this package is imported so its
`@BACKGROUNDS.register(...)` decorators run; see base.py for the interface."""

import importlib
import pkgutil

from .base import BACKGROUNDS, Background, PlaneRegion, SurfaceMaps

for _m in pkgutil.iter_modules(__path__):
    if _m.name != "base":
        importlib.import_module(f"{__name__}.{_m.name}")

__all__ = ["BACKGROUNDS", "Background", "PlaneRegion", "SurfaceMaps"]

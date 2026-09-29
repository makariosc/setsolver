"""A tiny plugin registry so new backgrounds / lighting rigs / layouts are drop-in.

Each component module registers a class or function with a weight. Scenes sample a
component by weight, so adding variety is just adding a new registered class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class Entry(Generic[T]):
    name: str
    obj: T  # a class (backgrounds) or a function (layouts, lighting rigs)
    weight: float


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, Entry[T]] = {}

    def register(self, name: str, weight: float = 1.0):
        """Class/function decorator: `@BACKGROUNDS.register("wood", weight=2)`."""

        def deco(obj):
            if name in self._entries:
                raise ValueError(f"duplicate {self.kind} {name!r}")
            self._entries[name] = Entry(name, obj, weight)
            return obj

        return deco

    def names(self) -> list[str]:
        return list(self._entries)

    def get(self, name: str) -> Entry[T]:
        return self._entries[name]

    def sample(self, rng: np.random.Generator, only: list[str] | None = None) -> Entry[T]:
        entries = [e for e in self._entries.values() if only is None or e.name in only]
        if not entries:
            raise ValueError(f"no {self.kind} to sample from (filter={only})")
        w = np.array([e.weight for e in entries], dtype=float)
        return entries[rng.choice(len(entries), p=w / w.sum())]

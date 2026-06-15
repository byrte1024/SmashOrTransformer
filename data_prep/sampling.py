from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .config import DataConfig


class SamplingStrategy(ABC):
    @abstractmethod
    def epoch_plan(self, rows_by_pokemon: dict[int, list[int]],
                   rng: np.random.Generator) -> list[int]:
        """Return a list of row indices (with repeats) for one epoch, shuffled."""
        ...


def _shuffled(plan: list[int], rng: np.random.Generator) -> list[int]:
    arr = np.array(plan, dtype=np.int64)
    rng.shuffle(arr)
    return arr.tolist()


class FlatVariations(SamplingStrategy):
    def __init__(self, n: int):
        self.n = n

    def epoch_plan(self, rows_by_pokemon, rng):
        plan: list[int] = []
        for rows in rows_by_pokemon.values():
            for r in rows:
                plan.extend([r] * self.n)
        return _shuffled(plan, rng)


class FillSo(SamplingStrategy):
    def __init__(self, target: int | None):
        self.target = target

    def epoch_plan(self, rows_by_pokemon, rng):
        target = self.target
        if target is None:
            target = max((len(r) for r in rows_by_pokemon.values()), default=0)
        plan: list[int] = []
        for rows in rows_by_pokemon.values():
            if not rows:
                continue
            k = len(rows)
            # Every image used equally `target // k` times; the leftover
            # `target % k` slots go to a random subset (re-rolled each epoch)
            # so no image is systematically favored.
            base, rem = divmod(target, k)
            plan.extend(rows * base)
            if rem:
                extra = rng.choice(k, size=rem, replace=False)
                plan.extend(rows[j] for j in extra)
        return _shuffled(plan, rng)


def build_sampling(cfg: DataConfig) -> SamplingStrategy:
    v = cfg.variations
    if v.mode == "flat":
        return FlatVariations(v.n)
    return FillSo(v.target)

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .config import DataConfig


class SplitStrategy(ABC):
    @abstractmethod
    def split(self, row_pokemon_ids: np.ndarray, val_frac: float,
              seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_row_indices, val_row_indices)."""
        ...


class PokemonLevelSplit(SplitStrategy):
    def split(self, row_pokemon_ids, val_frac, seed):
        rng = np.random.default_rng(seed)
        pids = np.unique(row_pokemon_ids)
        rng.shuffle(pids)
        n_val = int(round(len(pids) * val_frac))
        val_pids = set(pids[:n_val].tolist())
        rows = np.arange(len(row_pokemon_ids))
        val_mask = np.array([p in val_pids for p in row_pokemon_ids])
        return rows[~val_mask], rows[val_mask]


class ImageLevelSplit(SplitStrategy):
    def split(self, row_pokemon_ids, val_frac, seed):
        rng = np.random.default_rng(seed)
        rows = np.arange(len(row_pokemon_ids))
        rng.shuffle(rows)
        n_val = int(round(len(rows) * val_frac))
        val = rows[:n_val]; train = rows[n_val:]
        return np.sort(train), np.sort(val)


def build_split(cfg: DataConfig) -> SplitStrategy:
    return PokemonLevelSplit() if cfg.split.strategy == "pokemon" else ImageLevelSplit()

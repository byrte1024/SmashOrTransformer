from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .config import DataConfig
from .augmentations import build_augmentations
from .sampling import build_sampling


class DataSampler:
    """Reads a prepared dataset dir and yields augmented (image, label) samples.

    Deterministic: sample i in a given epoch uses default_rng([seed, epoch, i]).
    Framework-agnostic; wrap into a torch Dataset later.
    """

    def __init__(self, dataset_dir, split: str = "train", epoch: int = 0):
        self.dir = Path(dataset_dir)
        self.cfg = DataConfig.from_dict(json.loads((self.dir / "config.json").read_text()))
        with np.load(self.dir / "data.npz", allow_pickle=True) as data:
            self._images = data["images"]
            self._row_pid = data["pokemon_id"]
            self._smash = data["smash_pct"]
            # total_votes is kept available for optional confidence-weighting by
            # callers (e.g. weighting the loss by vote volume); not used internally.
            self._votes = data["total_votes"]

        split_info = json.loads((self.dir / "split.json").read_text())
        self._rows = list(split_info[split])
        self.split = split
        self.epoch = epoch

        self._compose = build_augmentations(self.cfg)
        self._sampling = build_sampling(self.cfg)
        self._plan = self._build_plan(epoch)

    def _build_plan(self, epoch: int) -> list[int]:
        rows_by_pokemon: dict[int, list[int]] = {}
        for r in self._rows:
            rows_by_pokemon.setdefault(int(self._row_pid[r]), []).append(r)
        rng = np.random.default_rng([self.cfg.seed, epoch])
        return self._sampling.epoch_plan(rows_by_pokemon, rng)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._plan = self._build_plan(epoch)

    def pokemon_ids(self) -> set[int]:
        return {int(self._row_pid[r]) for r in self._rows}

    def votes(self, i: int) -> int:
        """Total smash+pass votes for sample i (for optional confidence weighting)."""
        return int(self._votes[self._plan[i]])

    def __len__(self) -> int:
        return len(self._plan)

    def __getitem__(self, i: int):
        row = self._plan[i]
        rng = np.random.default_rng([self.cfg.seed, self.epoch, i])
        sprite = Image.fromarray(self._images[row], "RGBA")
        img = self._compose.apply(sprite, rng, self.cfg.resolution)
        return np.asarray(img, dtype=np.uint8), float(self._smash[row])

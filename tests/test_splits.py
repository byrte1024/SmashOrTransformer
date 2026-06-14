import numpy as np
from data_prep.config import DataConfig
from data_prep.splits import PokemonLevelSplit, ImageLevelSplit, build_split


def test_pokemon_level_no_leak():
    row_pid = np.array([1, 1, 1, 2, 2, 3, 3, 4])
    train, val = PokemonLevelSplit().split(row_pid, val_frac=0.5, seed=0)
    train_pids = set(row_pid[train]); val_pids = set(row_pid[val])
    assert train_pids.isdisjoint(val_pids)
    assert set(train) | set(val) == set(range(len(row_pid)))
    assert set(train) & set(val) == set()


def test_image_level_partitions_rows():
    row_pid = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3])
    train, val = ImageLevelSplit().split(row_pid, val_frac=0.4, seed=1)
    assert len(val) == 4
    assert set(train) | set(val) == set(range(10))
    assert set(train) & set(val) == set()


def test_build_split_dispatch():
    cfg_p = DataConfig.from_dict({"name": "d", "resolution": 8,
                                  "split": {"strategy": "pokemon"}})
    cfg_i = DataConfig.from_dict({"name": "d", "resolution": 8,
                                  "split": {"strategy": "image"}})
    assert isinstance(build_split(cfg_p), PokemonLevelSplit)
    assert isinstance(build_split(cfg_i), ImageLevelSplit)


def test_deterministic():
    row_pid = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    a = PokemonLevelSplit().split(row_pid, 0.5, seed=7)
    b = PokemonLevelSplit().split(row_pid, 0.5, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

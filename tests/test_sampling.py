import numpy as np
from data_prep.config import DataConfig
from data_prep.sampling import FlatVariations, FillSo, build_sampling


def _rows_by_pokemon():
    return {1: [0, 1, 2], 2: [3]}


def test_flat_repeats_each_row():
    plan = FlatVariations(2).epoch_plan(_rows_by_pokemon(), np.random.default_rng(0))
    assert sorted(plan) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_fill_so_balances_pokemon():
    plan = FillSo(4).epoch_plan(_rows_by_pokemon(), np.random.default_rng(0))
    assert plan.count(3) == 4
    assert sum(1 for r in plan if r in (0, 1, 2)) == 4
    assert len(plan) == 8


def test_fill_so_default_target_is_max_count():
    plan = FillSo(None).epoch_plan(_rows_by_pokemon(), np.random.default_rng(0))
    assert sum(1 for r in plan if r == 3) == 3
    assert len(plan) == 6


def test_build_sampling_dispatch():
    flat = build_sampling(DataConfig.from_dict({"name": "d", "resolution": 8,
                                                "variations": 3}))
    fill = build_sampling(DataConfig.from_dict({"name": "d", "resolution": 8,
                                                "variations": {"fill_so": 10}}))
    assert isinstance(flat, FlatVariations) and flat.n == 3
    assert isinstance(fill, FillSo) and fill.target == 10

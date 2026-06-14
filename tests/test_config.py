import pytest
from data_prep.config import DataConfig


def test_defaults_filled():
    cfg = DataConfig.from_dict({"name": "d", "resolution": 128})
    assert cfg.seed == 0
    assert cfg.minimages == 1
    assert cfg.selection.categories == []
    assert cfg.split.strategy == "pokemon"
    assert cfg.split.val_frac == 0.1
    assert cfg.augmentations.scale_method == "bilinear"
    assert cfg.augmentations.scale.w == (0.65, 1.10)
    assert cfg.augmentations.rotation == (-10.0, 10.0)
    assert cfg.augmentations.background.mode == "white"


def test_variations_flat():
    cfg = DataConfig.from_dict({"name": "d", "resolution": 64, "variations": 5})
    assert cfg.variations.mode == "flat"
    assert cfg.variations.n == 5


def test_variations_fill_so():
    cfg = DataConfig.from_dict(
        {"name": "d", "resolution": 64, "variations": {"fill_so": 30}}
    )
    assert cfg.variations.mode == "fill_so"
    assert cfg.variations.target == 30


def test_variations_fill_so_default_target_none():
    cfg = DataConfig.from_dict(
        {"name": "d", "resolution": 64, "variations": {"fill_so": None}}
    )
    assert cfg.variations.mode == "fill_so"
    assert cfg.variations.target is None


def test_roundtrip_to_dict():
    raw = {
        "name": "d", "resolution": 96, "seed": 7,
        "selection": {"categories": ["portrait"], "gens": {"exclude": [1, 2]}},
        "variations": {"fill_so": 12},
    }
    cfg = DataConfig.from_dict(raw)
    again = DataConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_validate_rejects_bad_split():
    with pytest.raises(ValueError):
        DataConfig.from_dict({"name": "d", "resolution": 64,
                              "split": {"strategy": "nonsense"}})


def test_validate_rejects_bad_resolution():
    with pytest.raises(ValueError):
        DataConfig.from_dict({"name": "d", "resolution": 0})


def test_validate_rejects_bad_scale_method():
    with pytest.raises(ValueError):
        DataConfig.from_dict({"name": "d", "resolution": 8,
                              "augmentations": {"scale_method": "wat"}})


def test_validate_rejects_unknown_category():
    with pytest.raises(ValueError):
        DataConfig.from_dict({"name": "d", "resolution": 8,
                              "selection": {"categories": ["bogus"]}})


def test_validate_rejects_flat_below_one():
    with pytest.raises(ValueError):
        DataConfig.from_dict({"name": "d", "resolution": 8, "variations": 0})


def test_validate_rejects_negative_minimages():
    with pytest.raises(ValueError):
        DataConfig.from_dict({"name": "d", "resolution": 8, "minimages": -1})

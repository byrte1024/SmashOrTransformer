import pytest
from data_prep.config import DataConfig


def test_defaults():
    cfg = DataConfig.from_dict({"name": "d", "resolution": 128})
    assert cfg.seed == 0 and cfg.minimages == 1
    assert cfg.selection.categories == []
    assert cfg.split.strategy == "pokemon"
    s = cfg.augmentations.sprite
    assert s.scale_method == "bilinear"
    assert s.scale.w == (0.65, 1.10) and s.rotation == (-10.0, 10.0)
    assert s.flip == 0.5
    assert s.background.prob == 0.8
    assert s.background.dirs == ["backgrounds/real", "backgrounds/pokemon_battle"]
    p = cfg.augmentations.photo
    assert p.crop_scale == (0.6, 1.0)
    assert p.color.brightness == 0.2 and p.color.hue == 0.05
    assert p.flip == 0.5


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


def test_nested_aug_parsed():
    cfg = DataConfig.from_dict({"name": "d", "resolution": 64, "augmentations": {
        "sprite": {"rotation": [-5, 5], "flip": 0.0,
                   "background": {"prob": 0.5, "dirs": ["backgrounds/real"]}},
        "photo": {"crop_scale": [0.4, 0.9], "flip": 1.0,
                  "color": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.0}}}})
    assert cfg.augmentations.sprite.rotation == (-5.0, 5.0)
    assert cfg.augmentations.sprite.flip == 0.0
    assert cfg.augmentations.sprite.background.prob == 0.5
    assert cfg.augmentations.sprite.background.dirs == ["backgrounds/real"]
    assert cfg.augmentations.photo.crop_scale == (0.4, 0.9)
    assert cfg.augmentations.photo.flip == 1.0


def test_back_compat_flat_aug_is_sprite():
    # old flat schema (pre-source-aware) parses as the sprite config
    cfg = DataConfig.from_dict({"name": "d", "resolution": 64, "augmentations": {
        "scale": {"w": [0.7, 1.0], "h": [0.7, 1.0]}, "scale_method": "nearest",
        "rotation": [-3, 3], "background": {"mode": "white"}}})
    s = cfg.augmentations.sprite
    assert s.scale_method == "nearest"
    assert s.scale.w == (0.7, 1.0) and s.rotation == (-3.0, 3.0)
    assert s.background.prob == 0.0           # mode:white -> prob 0


def test_booru_is_a_valid_category():
    cfg = DataConfig.from_dict({"name": "d", "resolution": 64,
                                "selection": {"categories": ["booru"]}})
    assert cfg.selection.categories == ["booru"]


def test_roundtrip_nested(tmp_path=None):
    cfg = DataConfig.from_dict({"name": "d", "resolution": 96,
                                "selection": {"categories": ["portrait", "booru"]},
                                "augmentations": {"sprite": {"flip": 0.25}}})
    again = DataConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()

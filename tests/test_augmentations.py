import numpy as np
from PIL import Image
from data_prep.config import DataConfig
from data_prep.augmentations import (
    Scale, Rotate, Position, CompositeBackground, Compose, build_augmentations,
)


def _sprite(w=20, h=20):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[2:h - 2, 2:w - 2] = (255, 0, 0, 255)
    return Image.fromarray(arr, "RGBA")


def test_scale_within_bounds():
    rng = np.random.default_rng(0)
    out = Scale((0.5, 0.5), (0.25, 0.25), "bilinear").apply(_sprite(), rng, 100)
    assert out.size == (50, 25)


def test_rotate_expands_and_keeps_alpha():
    rng = np.random.default_rng(0)
    out = Rotate((90.0, 90.0)).apply(_sprite(20, 10), rng, 100)
    assert out.mode == "RGBA"
    assert out.size == (10, 20)


def test_position_returns_square_rgba_canvas():
    rng = np.random.default_rng(0)
    out = Position((0.5, 0.5), (0.5, 0.5)).apply(_sprite(20, 20), rng, 64)
    assert out.size == (64, 64)
    assert out.mode == "RGBA"
    assert np.asarray(out)[32, 32, 3] == 255


def test_composite_white_removes_alpha():
    rng = np.random.default_rng(0)
    canvas = Position((0.5, 0.5), (0.5, 0.5)).apply(_sprite(20, 20), rng, 64)
    out = CompositeBackground("white").apply(canvas, rng, 64)
    assert out.mode == "RGB"
    a = np.asarray(out)
    assert (a[0, 0] == 255).all()
    assert (a[32, 32] == (255, 0, 0)).all()


def test_compose_end_to_end_shape():
    cfg = DataConfig.from_dict({"name": "d", "resolution": 48})
    comp = build_augmentations(cfg)
    rng = np.random.default_rng(1)
    out = comp.apply(_sprite(20, 20), rng, 48)
    assert np.asarray(out).shape == (48, 48, 3)


def test_determinism_same_seed_same_pixels():
    cfg = DataConfig.from_dict({"name": "d", "resolution": 48})
    comp = build_augmentations(cfg)
    a = np.asarray(comp.apply(_sprite(), np.random.default_rng(5), 48))
    b = np.asarray(comp.apply(_sprite(), np.random.default_rng(5), 48))
    assert np.array_equal(a, b)

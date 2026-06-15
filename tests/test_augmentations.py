import numpy as np
from PIL import Image
from data_prep.config import DataConfig
from data_prep.augmentations import (
    Scale, Rotate, Position, CompositeBackground, Compose, build_augmentations,
    HorizontalFlip, RandomResizedCrop, ColorJitter, ToRGB, BackgroundPool,
    ScaleFit, CompositeRandomBackground, build_sprite_aug, build_photo_aug,
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


def test_composite_white_when_no_pool():
    rng = np.random.default_rng(0)
    canvas = Position((0.5, 0.5), (0.5, 0.5)).apply(_sprite(20, 20), rng, 64)
    out = CompositeBackground(pool=None, prob=0.0).apply(canvas, rng, 64)
    assert out.mode == "RGB"
    a = np.asarray(out)
    assert (a[0, 0] == 255).all()          # white background
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


def test_horizontal_flip():
    arr = np.zeros((1, 4, 4), dtype=np.uint8)
    # build a left/right asymmetric RGBA image
    im = np.zeros((4, 4, 4), dtype=np.uint8); im[:, 0] = (255, 0, 0, 255)
    img = Image.fromarray(im, "RGBA")
    flipped = np.asarray(HorizontalFlip(1.0).apply(img, np.random.default_rng(0), 4))
    assert (flipped[:, 3, 0] == 255).all()       # red column moved left->right
    same = np.asarray(HorizontalFlip(0.0).apply(img, np.random.default_rng(0), 4))
    assert np.array_equal(same, np.asarray(img))  # prob 0 -> identity


def test_random_resized_crop_shape():
    img = _sprite(40, 30)
    out = RandomResizedCrop((0.5, 1.0)).apply(img, np.random.default_rng(0), 24)
    assert out.size == (24, 24)


def test_color_jitter_changes_preserves_alpha_deterministic():
    img = _sprite(20, 20)
    a = np.asarray(ColorJitter(0.4, 0.4, 0.4, 0.1).apply(img, np.random.default_rng(3), 20))
    b = np.asarray(ColorJitter(0.4, 0.4, 0.4, 0.1).apply(img, np.random.default_rng(3), 20))
    assert a.shape == (20, 20, 4)                # alpha preserved (RGBA in -> RGBA out)
    assert np.array_equal(a, b)                  # deterministic for fixed rng
    plain = np.asarray(img)
    assert not np.array_equal(a[..., :3], plain[..., :3])   # RGB jittered
    assert np.array_equal(a[..., 3], plain[..., 3])         # alpha untouched


def test_scalefit_preserves_aspect():
    out = ScaleFit((1.0, 1.0)).apply(_sprite(40, 20), np.random.default_rng(0), 64)
    assert out.size == (64, 32)                  # longest side -> 64, 2:1 kept


def test_composite_random_background_white_or_black():
    rng = np.random.default_rng(0)
    canvas = Position((0.5, 0.5), (0.5, 0.5)).apply(_sprite(16, 16), rng, 64)
    w = np.asarray(CompositeRandomBackground(pool=None, use_white=True, use_black=False)
                   .apply(canvas, rng, 64))
    assert tuple(w[0, 0]) == (255, 255, 255) and tuple(w[32, 32]) == (255, 0, 0)
    blk = np.asarray(CompositeRandomBackground(pool=None, use_white=False, use_black=True)
                     .apply(canvas, rng, 64))
    assert tuple(blk[0, 0]) == (0, 0, 0)


def test_background_pool_composites_non_white(tmp_path):
    bgdir = tmp_path / "bg"; bgdir.mkdir()
    Image.new("RGB", (50, 50), (10, 120, 200)).save(bgdir / "b.png")
    pool = BackgroundPool([str(bgdir)])
    assert len(pool) == 1 and not pool.empty()
    rng = np.random.default_rng(0)
    canvas = Position((0.5, 0.5), (0.5, 0.5)).apply(_sprite(16, 16), rng, 64)
    out = np.asarray(CompositeBackground(pool=pool, prob=1.0).apply(canvas, rng, 64))
    assert tuple(out[0, 0]) == (10, 120, 200)    # corner shows the background
    assert tuple(out[32, 32]) == (255, 0, 0)     # sprite still on top


def test_empty_background_pool_falls_back_to_white(tmp_path):
    pool = BackgroundPool([str(tmp_path / "missing")])
    assert pool.empty()
    rng = np.random.default_rng(0)
    canvas = Position((0.5, 0.5), (0.5, 0.5)).apply(_sprite(16, 16), rng, 64)
    out = np.asarray(CompositeBackground(pool=pool, prob=1.0).apply(canvas, rng, 64))
    assert tuple(out[0, 0]) == (255, 255, 255)


def test_build_photo_aug_outputs_rgb():
    from data_prep.config import DataConfig
    cfg = DataConfig.from_dict({"name": "d", "resolution": 48})
    comp = build_photo_aug(cfg)
    out = comp.apply(_sprite(60, 40), np.random.default_rng(1), 48)
    assert np.asarray(out).shape == (48, 48, 3)

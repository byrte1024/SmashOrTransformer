# Source-Aware Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real-background compositing for base Pokemon sprites and an opt-in opaque "booru" image source with crop/scale/color augmentation, routed per-image by category.

**Architecture:** Augmentation is split into a `sprite` pipeline (geometric + background composite) and a `photo` pipeline (crop/color/flip), selected per sample by mapping its `category` to a kind (`booru`->photo, else sprite). Selection gains an opt-in `booru` category; config nests `augmentations.sprite`/`augmentations.photo` with back-compat for the old flat schema; the sampler routes each row to the right pipeline.

**Tech Stack:** Python, numpy, Pillow (PIL.ImageEnhance for color jitter). Tests run with `uv run pytest` (no network/GPU).

Spec: `docs/superpowers/specs/2026-06-15-source-aware-augmentation-design.md`

---

## File Structure

```
data_prep/config.py        # rewrite: nested sprite/photo aug + back-compat + validation
data_prep/selection.py     # booru opt-in category, read booru/meta.csv, relaxation excludes booru
data_prep/augmentations.py # rewrite: new ops (HFlip/RandomResizedCrop/ColorJitter/ToRGB),
                           #          BackgroundPool, CompositeBackground(pool), sprite/photo builders
data_prep/sampler.py       # route per category->kind; build both pipelines
configs/example_mixed.json # new example with booru + backgrounds
tests/test_config.py       # update for nested schema + back-compat
tests/test_selection.py    # update booru-exclusion test + add inclusion test
tests/test_augmentations.py# update composite test + new op/pool tests
tests/test_sampler.py      # add routing test
```

Conventions: categories `portrait`/`in-game`/`animated` are kind `sprite`; `booru` is kind `photo`. `load_sprite` stores all images RGBA (booru opaque). Pipelines output RGB `[res,res,3]`.

---

## Task 1: Config - nested sprite/photo augmentation schema

**Files:**
- Modify: `data_prep/config.py` (full rewrite below)
- Test: `tests/test_config.py`

- [ ] **Step 1: Update tests/test_config.py for the new schema**

Replace the body of `test_defaults` and add new tests. Open `tests/test_config.py` and replace the `test_defaults` function and append the new ones:

```python
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
```

Keep the existing `test_variations_*`, `test_validate_rejects_bad_split`, `test_validate_rejects_bad_resolution`, `test_validate_rejects_bad_scale_method`, `test_validate_rejects_unknown_category`, `test_validate_rejects_flat_below_one`, `test_validate_rejects_negative_minimages`, and `test_roundtrip` tests as-is (they still pass: `scale_method`/`category` validation works through back-compat, and `test_validate_rejects_unknown_category` uses `"bogus"` which is still invalid).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL (new attribute paths like `cfg.augmentations.sprite` don't exist yet).

- [ ] **Step 3: Rewrite data_prep/config.py**

```python
from __future__ import annotations
from dataclasses import dataclass, field, asdict

CATEGORIES = ("portrait", "in-game", "animated", "booru")
SCALE_METHODS = ("nearest", "bilinear", "bicubic", "lanczos")
SPLIT_STRATEGIES = ("pokemon", "image")

DEFAULT_SCALE_W = (0.65, 1.10)
DEFAULT_SCALE_H = (0.65, 1.10)
DEFAULT_POS_X = (0.10, 0.90)
DEFAULT_POS_Y = (0.10, 0.90)
DEFAULT_ROTATION = (-10.0, 10.0)
DEFAULT_BG_DIRS = ["backgrounds/real", "backgrounds/pokemon_battle"]
DEFAULT_BG_PROB = 0.8
DEFAULT_FLIP = 0.5
DEFAULT_CROP_SCALE = (0.6, 1.0)
_FLAT_AUG_KEYS = {"scale", "scale_method", "position", "rotation", "background", "flip"}


def _pair(v, default):
    if v is None:
        return tuple(float(x) for x in default)
    return (float(v[0]), float(v[1]))


@dataclass
class NamesCfg:
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class GensCfg:
    include: list[int] = field(default_factory=list)
    exclude: list[int] = field(default_factory=list)


@dataclass
class SelectionCfg:
    categories: list[str] = field(default_factory=list)
    names: NamesCfg = field(default_factory=NamesCfg)
    gens: GensCfg = field(default_factory=GensCfg)


@dataclass
class SplitCfg:
    strategy: str = "pokemon"
    val_frac: float = 0.1


@dataclass
class VariationsCfg:
    mode: str = "flat"
    n: int | None = 1
    target: int | None = None


@dataclass
class ScaleAugCfg:
    w: tuple[float, float] = DEFAULT_SCALE_W
    h: tuple[float, float] = DEFAULT_SCALE_H


@dataclass
class PositionAugCfg:
    x: tuple[float, float] = DEFAULT_POS_X
    y: tuple[float, float] = DEFAULT_POS_Y


@dataclass
class BackgroundCfg:
    prob: float = DEFAULT_BG_PROB
    dirs: list[str] = field(default_factory=lambda: list(DEFAULT_BG_DIRS))
    weights: list[float] | None = None


@dataclass
class SpriteAugCfg:
    scale: ScaleAugCfg = field(default_factory=ScaleAugCfg)
    scale_method: str = "bilinear"
    position: PositionAugCfg = field(default_factory=PositionAugCfg)
    rotation: tuple[float, float] = DEFAULT_ROTATION
    flip: float = DEFAULT_FLIP
    background: BackgroundCfg = field(default_factory=BackgroundCfg)


@dataclass
class ColorCfg:
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05


@dataclass
class PhotoAugCfg:
    crop_scale: tuple[float, float] = DEFAULT_CROP_SCALE
    color: ColorCfg = field(default_factory=ColorCfg)
    flip: float = DEFAULT_FLIP


@dataclass
class AugmentationsCfg:
    sprite: SpriteAugCfg = field(default_factory=SpriteAugCfg)
    photo: PhotoAugCfg = field(default_factory=PhotoAugCfg)


def _parse_sprite(sp: dict) -> SpriteAugCfg:
    sc = sp.get("scale", {}) or {}
    po = sp.get("position", {}) or {}
    bg = sp.get("background", {}) or {}
    if "prob" in bg:
        prob = float(bg["prob"])
    elif bg.get("mode") == "white":           # back-compat: old white stub
        prob = 0.0
    else:
        prob = DEFAULT_BG_PROB
    return SpriteAugCfg(
        scale=ScaleAugCfg(w=_pair(sc.get("w"), DEFAULT_SCALE_W),
                          h=_pair(sc.get("h"), DEFAULT_SCALE_H)),
        scale_method=sp.get("scale_method", "bilinear"),
        position=PositionAugCfg(x=_pair(po.get("x"), DEFAULT_POS_X),
                                y=_pair(po.get("y"), DEFAULT_POS_Y)),
        rotation=_pair(sp.get("rotation"), DEFAULT_ROTATION),
        flip=float(sp.get("flip", DEFAULT_FLIP)),
        background=BackgroundCfg(prob=prob,
                                dirs=list(bg.get("dirs", DEFAULT_BG_DIRS)),
                                weights=bg.get("weights")),
    )


def _parse_photo(ph: dict) -> PhotoAugCfg:
    col = ph.get("color", {}) or {}
    return PhotoAugCfg(
        crop_scale=_pair(ph.get("crop_scale"), DEFAULT_CROP_SCALE),
        color=ColorCfg(brightness=float(col.get("brightness", 0.2)),
                       contrast=float(col.get("contrast", 0.2)),
                       saturation=float(col.get("saturation", 0.2)),
                       hue=float(col.get("hue", 0.05))),
        flip=float(ph.get("flip", DEFAULT_FLIP)),
    )


@dataclass
class DataConfig:
    name: str
    resolution: int
    seed: int = 0
    minimages: int = 1
    selection: SelectionCfg = field(default_factory=SelectionCfg)
    variations: VariationsCfg = field(default_factory=VariationsCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    augmentations: AugmentationsCfg = field(default_factory=AugmentationsCfg)

    @staticmethod
    def from_dict(d: dict) -> "DataConfig":
        sel = d.get("selection", {}) or {}
        names = sel.get("names", {}) or {}
        gens = sel.get("gens", {}) or {}
        selection = SelectionCfg(
            categories=list(sel.get("categories", []) or []),
            names=NamesCfg(include=list(names.get("include", []) or []),
                           exclude=list(names.get("exclude", []) or [])),
            gens=GensCfg(include=list(gens.get("include", []) or []),
                         exclude=list(gens.get("exclude", []) or [])),
        )

        var_raw = d.get("variations", 1)
        if isinstance(var_raw, dict):
            if "mode" in var_raw:
                variations = VariationsCfg(mode=var_raw.get("mode", "flat"),
                                           n=var_raw.get("n"), target=var_raw.get("target"))
            else:
                variations = VariationsCfg(mode="fill_so", n=None,
                                           target=var_raw.get("fill_so", None))
        else:
            variations = VariationsCfg(mode="flat", n=int(var_raw), target=None)

        sp = d.get("split", {}) or {}
        split = SplitCfg(strategy=sp.get("strategy", "pokemon"),
                         val_frac=float(sp.get("val_frac", 0.1)))

        au = d.get("augmentations", {}) or {}
        sprite_raw = au.get("sprite")
        if sprite_raw is None:                    # back-compat: old flat aug == sprite
            sprite_raw = au if (set(au) & _FLAT_AUG_KEYS) else {}
        aug = AugmentationsCfg(sprite=_parse_sprite(sprite_raw),
                               photo=_parse_photo(au.get("photo", {}) or {}))

        cfg = DataConfig(
            name=d["name"], resolution=int(d["resolution"]),
            seed=int(d.get("seed", 0)), minimages=int(d.get("minimages", 1)),
            selection=selection, variations=variations, split=split, augmentations=aug,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.minimages < 0:
            raise ValueError("minimages must be >= 0")
        if self.split.strategy not in SPLIT_STRATEGIES:
            raise ValueError(f"split.strategy must be one of {SPLIT_STRATEGIES}")
        if self.augmentations.sprite.scale_method not in SCALE_METHODS:
            raise ValueError(f"scale_method must be one of {SCALE_METHODS}")
        if not (0.0 <= self.augmentations.sprite.background.prob <= 1.0):
            raise ValueError("background.prob must be in [0,1]")
        for c in self.selection.categories:
            if c not in CATEGORIES:
                raise ValueError(f"unknown category {c!r}")
        if self.variations.mode == "flat" and (self.variations.n or 0) < 1:
            raise ValueError("flat variations must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS (all config tests).

- [ ] **Step 5: Commit**

```bash
git add data_prep/config.py tests/test_config.py
git commit -m "feat: nested sprite/photo augmentation config with back-compat"
```

---

## Task 2: Selection - opt-in booru category

**Files:**
- Modify: `data_prep/selection.py`
- Test: `tests/test_selection.py`

- [ ] **Step 1: Update the existing booru-exclusion test and add an inclusion test**

In `tests/test_selection.py`, REPLACE `test_booru_subfolder_images_are_excluded` with the following two functions (the old one asserted `load_records` ignored booru; now `load_records` returns booru records tagged `category="booru"`, and `select_pokemon` excludes/includes them based on config):

```python
def _add_booru(images_dir, pid):
    import csv as _csv
    from PIL import Image as _Image
    booru = images_dir / str(pid) / "booru"
    booru.mkdir()
    _Image.new("RGB", (8, 8), (1, 2, 3)).save(booru / "00_999.jpg")
    with open(booru / "meta.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rank", "post_id", "score", "rating", "file_url"])
        w.writeheader(); w.writerow({"rank": 0, "post_id": 999, "score": 9,
                                     "rating": "safe", "file_url": "http://x/00_999.jpg"})


def test_load_records_includes_booru_tagged(mini_repo):
    _add_booru(mini_repo["images"], 1)
    recs = load_records(mini_repo["images"], 1, load_labels(mini_repo["labels"]))
    booru = [r for r in recs if r.category == "booru"]
    assert len(booru) == 1 and booru[0].source_name == "00_999"
    assert booru[0].path.parent.name == "booru"


def test_booru_excluded_by_default_included_on_opt_in(mini_repo):
    _add_booru(mini_repo["images"], 1)
    recs = load_records(mini_repo["images"], 1, load_labels(mini_repo["labels"]))
    # default (categories empty = all sprite categories) -> no booru
    default_cfg = DataConfig.from_dict({"name": "d", "resolution": 16})
    kept, _ = select_pokemon(default_cfg, recs)
    assert all(r.category != "booru" for r in kept)
    # sprite-only config -> no booru
    sprite_cfg = DataConfig.from_dict({"name": "d", "resolution": 16,
                                       "selection": {"categories": ["portrait"]}})
    kept, _ = select_pokemon(sprite_cfg, recs)
    assert all(r.category != "booru" for r in kept)
    # opt-in -> booru present
    booru_cfg = DataConfig.from_dict({"name": "d", "resolution": 16,
                                      "selection": {"categories": ["booru"]}})
    kept, _ = select_pokemon(booru_cfg, recs)
    assert [r.category for r in kept] == ["booru"]


def test_relaxation_never_adds_booru(mini_repo):
    _add_booru(mini_repo["images"], 1)
    recs = load_records(mini_repo["images"], 1, load_labels(mini_repo["labels"]))
    # impossible minimages with a name filter that matches nothing; relaxation
    # may re-add sprites but must never pull in the booru record
    cfg = DataConfig.from_dict({"name": "d", "resolution": 16, "minimages": 99,
                                "selection": {"names": {"include": ["nope"]}}})
    kept, _ = select_pokemon(cfg, recs)
    assert all(r.category != "booru" for r in kept)
```

(Keep all other existing selection tests unchanged.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_selection.py -q`
Expected: FAIL (booru not yet read / not yet opt-in-gated).

- [ ] **Step 3: Edit data_prep/selection.py**

(a) After the `meta`-reading loop in `load_records` (before `return recs`), also read the booru subfolder. Replace the end of `load_records`:

```python
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            if Path(fname).suffix.lower() in _UNSUPPORTED_SUFFIXES:
                continue
            name = fname.rsplit(".", 1)[0]
            recs.append(ImageRecord(
                pokemon_id=pokemon_id, source_name=name,
                category=row["category"], gen=gen_of(name),
                path=folder / fname, smash_pct=smash, total_votes=votes,
            ))

    booru_meta = folder / "booru" / "meta.csv"
    if booru_meta.exists():
        with open(booru_meta, newline="") as f:
            for row in csv.DictReader(f):
                fp = next(iter((folder / "booru").glob(f"*_{row['post_id']}.*")), None)
                if fp is None or fp.name == "meta.csv":
                    continue
                recs.append(ImageRecord(
                    pokemon_id=pokemon_id, source_name=fp.stem, category="booru",
                    gen=0, path=fp, smash_pct=smash, total_votes=votes,
                ))
    return recs
```

(b) Gate booru in `_passes` (it is opt-in only). Replace `_passes`:

```python
def _passes(rec: "ImageRecord", cfg: DataConfig) -> bool:
    s = cfg.selection
    if rec.category == "booru":
        return "booru" in s.categories          # opt-in only, even when categories is empty
    if s.categories and rec.category not in s.categories:
        return False
    if s.names.include and rec.source_name not in s.names.include:
        return False
    if rec.source_name in s.names.exclude:
        return False
    if rec.gen != 0:
        if s.gens.include and rec.gen not in s.gens.include:
            return False
        if rec.gen in s.gens.exclude:
            return False
    return True
```

(c) Exclude booru from relaxation. In `select_pokemon`, change the relaxation source line:

```python
    excluded = [r for r in records if r not in kept and r.category != "booru"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_selection.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_prep/selection.py tests/test_selection.py
git commit -m "feat: opt-in booru category in selection"
```

---

## Task 3: Augmentations - new ops, BackgroundPool, sprite/photo builders

**Files:**
- Modify: `data_prep/augmentations.py` (full rewrite below)
- Test: `tests/test_augmentations.py`

- [ ] **Step 1: Update tests/test_augmentations.py**

Replace `test_composite_white_removes_alpha` with the new-API version and append new tests. Edit the imports line at top to:

```python
from data_prep.augmentations import (
    Scale, Rotate, Position, CompositeBackground, Compose, build_augmentations,
    HorizontalFlip, RandomResizedCrop, ColorJitter, ToRGB, BackgroundPool,
    build_sprite_aug, build_photo_aug,
)
```

Replace `test_composite_white_removes_alpha`:

```python
def test_composite_white_when_no_pool():
    rng = np.random.default_rng(0)
    canvas = Position((0.5, 0.5), (0.5, 0.5)).apply(_sprite(20, 20), rng, 64)
    out = CompositeBackground(pool=None, prob=0.0).apply(canvas, rng, 64)
    assert out.mode == "RGB"
    a = np.asarray(out)
    assert (a[0, 0] == 255).all()          # white background
    assert (a[32, 32] == (255, 0, 0)).all()
```

Append:

```python
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


def test_color_jitter_changes_and_deterministic():
    img = _sprite(20, 20)
    a = np.asarray(ColorJitter(0.4, 0.4, 0.4, 0.1).apply(img, np.random.default_rng(3), 20))
    b = np.asarray(ColorJitter(0.4, 0.4, 0.4, 0.1).apply(img, np.random.default_rng(3), 20))
    assert a.shape == (20, 20, 3)
    assert np.array_equal(a, b)                  # deterministic for fixed rng
    plain = np.asarray(img.convert("RGB"))
    assert not np.array_equal(a, plain)          # actually jittered


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
```

(Keep `test_scale_within_bounds`, `test_rotate_expands_and_keeps_alpha`,
`test_position_returns_square_rgba_canvas`, `test_compose_end_to_end_shape`,
`test_determinism_same_seed_same_pixels` unchanged — `build_augmentations`
remains an alias for the sprite pipeline.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_augmentations.py -q`
Expected: FAIL (new symbols not defined).

- [ ] **Step 3: Rewrite data_prep/augmentations.py**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
from .config import DataConfig

_RESAMPLE = {
    "nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS,
}
_BG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class Augmentation(ABC):
    @abstractmethod
    def apply(self, img: Image.Image, rng: np.random.Generator,
              resolution: int) -> Image.Image:
        ...


class Scale(Augmentation):
    def __init__(self, w_range, h_range, method):
        self.w_range, self.h_range, self.method = w_range, h_range, method

    def apply(self, img, rng, resolution):
        w = max(1, round(rng.uniform(*self.w_range) * resolution))
        h = max(1, round(rng.uniform(*self.h_range) * resolution))
        return img.resize((w, h), _RESAMPLE[self.method])


class Rotate(Augmentation):
    def __init__(self, deg_range):
        self.deg_range = deg_range

    def apply(self, img, rng, resolution):
        return img.rotate(rng.uniform(*self.deg_range), resample=Image.BILINEAR, expand=True)


class Position(Augmentation):
    def __init__(self, x_range, y_range):
        self.x_range, self.y_range = x_range, y_range

    def apply(self, img, rng, resolution):
        canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
        cx = rng.uniform(*self.x_range) * resolution
        cy = rng.uniform(*self.y_range) * resolution
        canvas.paste(img, (round(cx - img.width / 2), round(cy - img.height / 2)), img)
        return canvas


class HorizontalFlip(Augmentation):
    def __init__(self, prob):
        self.prob = prob

    def apply(self, img, rng, resolution):
        if self.prob and rng.random() < self.prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomResizedCrop(Augmentation):
    def __init__(self, scale_range, ratio=(0.8, 1.25)):
        self.scale_range, self.ratio = scale_range, ratio

    def apply(self, img, rng, resolution):
        W, H = img.size
        area = W * H
        crop = img
        for _ in range(10):
            a = rng.uniform(*self.scale_range) * area
            ar = rng.uniform(*self.ratio)
            w, h = int(round((a * ar) ** 0.5)), int(round((a / ar) ** 0.5))
            if 0 < w <= W and 0 < h <= H:
                x = int(rng.integers(0, W - w + 1))
                y = int(rng.integers(0, H - h + 1))
                crop = img.crop((x, y, x + w, y + h))
                break
        return crop.resize((resolution, resolution), Image.BILINEAR)


class ColorJitter(Augmentation):
    def __init__(self, brightness, contrast, saturation, hue):
        self.b, self.c, self.s, self.h = brightness, contrast, saturation, hue

    def apply(self, img, rng, resolution):
        im = img.convert("RGB")
        if self.b:
            im = ImageEnhance.Brightness(im).enhance(1 + rng.uniform(-self.b, self.b))
        if self.c:
            im = ImageEnhance.Contrast(im).enhance(1 + rng.uniform(-self.c, self.c))
        if self.s:
            im = ImageEnhance.Color(im).enhance(1 + rng.uniform(-self.s, self.s))
        if self.h:
            hsv = np.asarray(im.convert("HSV")).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-self.h, self.h) * 255)) % 256
            im = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
        return im


class ToRGB(Augmentation):
    def apply(self, img, rng, resolution):
        return img.convert("RGB")


class BackgroundPool:
    """Lazily lists background images across dirs; caches each chosen image
    resized + center-cropped to the canvas. Missing dirs are skipped."""
    def __init__(self, dirs, weights=None):
        self.groups, gw = [], []
        for i, d in enumerate(dirs or []):
            p = Path(d)
            files = sorted(f for f in p.glob("*") if f.suffix.lower() in _BG_EXTS) \
                if p.is_dir() else []
            if files:
                self.groups.append(files)
                gw.append(float(weights[i]) if weights and i < len(weights) else 1.0)
        self._gw = np.array(gw, dtype=float) if self.groups else None
        self._cache = {}

    def __len__(self):
        return sum(len(g) for g in self.groups)

    def empty(self):
        return not self.groups

    def sample(self, rng, resolution):
        if not self.groups:
            return None
        gi = int(rng.choice(len(self.groups), p=self._gw / self._gw.sum()))
        group = self.groups[gi]
        path = group[int(rng.integers(len(group)))]
        key = (str(path), resolution)
        if key not in self._cache:
            self._cache[key] = self._load(path, resolution)
        return self._cache[key]

    @staticmethod
    def _load(path, res):
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = res / min(w, h)
        nw, nh = max(res, round(w * s)), max(res, round(h * s))
        im = im.resize((nw, nh), Image.BILINEAR)
        l, t = (nw - res) // 2, (nh - res) // 2
        return im.crop((l, t, l + res, t + res))


class CompositeBackground(Augmentation):
    """Flatten an RGBA res x res canvas onto a background (real with prob, else
    white) -> RGB."""
    def __init__(self, pool=None, prob=0.0):
        self.pool, self.prob = pool, prob

    def apply(self, img, rng, resolution):
        bg = None
        if self.pool is not None and self.prob > 0 and not self.pool.empty() \
                and rng.random() < self.prob:
            sampled = self.pool.sample(rng, resolution)
            bg = sampled.copy() if sampled is not None else None
        if bg is None:
            bg = Image.new("RGB", (resolution, resolution), (255, 255, 255))
        bg.paste(img, (0, 0), img)
        return bg


class Compose(Augmentation):
    def __init__(self, steps: list[Augmentation]):
        self.steps = steps

    def apply(self, img, rng, resolution):
        for step in self.steps:
            img = step.apply(img, rng, resolution)
        return img


def build_sprite_aug(cfg: DataConfig) -> Compose:
    s = cfg.augmentations.sprite
    pool = BackgroundPool(s.background.dirs, s.background.weights)
    return Compose([
        Scale(s.scale.w, s.scale.h, s.scale_method),
        Rotate(s.rotation),
        Position(s.position.x, s.position.y),
        HorizontalFlip(s.flip),
        CompositeBackground(pool=pool, prob=s.background.prob),
    ])


def build_photo_aug(cfg: DataConfig) -> Compose:
    p = cfg.augmentations.photo
    return Compose([
        RandomResizedCrop(p.crop_scale),
        ColorJitter(p.color.brightness, p.color.contrast, p.color.saturation, p.color.hue),
        HorizontalFlip(p.flip),
        ToRGB(),
    ])


def build_augmentations(cfg: DataConfig) -> Compose:
    """Back-compat alias: the sprite pipeline."""
    return build_sprite_aug(cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_augmentations.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_prep/augmentations.py tests/test_augmentations.py
git commit -m "feat: photo aug ops, BackgroundPool, sprite/photo aug builders"
```

---

## Task 4: Sampler - route per category->kind

**Files:**
- Modify: `data_prep/sampler.py`
- Test: `tests/test_sampler.py`

- [ ] **Step 1: Add a routing test to tests/test_sampler.py**

Append:

```python
def test_sampler_routes_booru_through_photo_aug(mini_repo, tmp_path):
    # give pokemon 1 a booru image; build a booru+portrait dataset
    import csv as _csv
    from PIL import Image as _Image
    booru = mini_repo["images"] / "1" / "booru"
    booru.mkdir()
    _Image.new("RGB", (40, 40), (10, 200, 30)).save(booru / "00_777.jpg")
    with open(booru / "meta.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rank", "post_id", "score", "rating", "file_url"])
        w.writeheader(); w.writerow({"rank": 0, "post_id": 777, "score": 9,
                                     "rating": "safe", "file_url": "x"})
    cfg = DataConfig.from_dict({"name": "mix", "resolution": 32, "minimages": 1,
                                "variations": 2,
                                "selection": {"categories": ["portrait", "booru"]},
                                "split": {"strategy": "image", "val_frac": 0.0},
                                "augmentations": {"sprite": {"background": {"prob": 0.0}},
                                                  "photo": {"flip": 0.0}}})
    out = prepare(cfg, mini_repo["images"], mini_repo["labels"], tmp_path / "datasets")
    ds = DataSampler(out, split="train", epoch=0)
    # both categories made it into the dataset
    cats = set(__import__("numpy").load(out / "data.npz", allow_pickle=True)["category"].tolist())
    assert "booru" in cats and "portrait" in cats
    img, label = ds[0]
    assert img.shape == (32, 32, 3) and img.dtype.name == "uint8"
    assert 0.0 <= label <= 1.0
```

(Add `from data_prep.prepare import prepare` and `from data_prep.config import DataConfig` to the test imports if not already present.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sampler.py::test_sampler_routes_booru_through_photo_aug -q`
Expected: FAIL (sampler doesn't load category / doesn't route yet; or KeyError on category).

- [ ] **Step 3: Edit data_prep/sampler.py**

Change the import and the `__init__`/`__getitem__` to route by category. Replace the import line:

```python
from .augmentations import build_sprite_aug, build_photo_aug
```

In `__init__`, load the `category` array (inside the `with np.load(...)` block, after `self._votes = ...`):

```python
            self._cat = data["category"]
```

Replace the pipeline build line `self._compose = build_augmentations(self.cfg)` with:

```python
        self._sprite_aug = build_sprite_aug(self.cfg)
        self._photo_aug = build_photo_aug(self.cfg)
```

Replace `__getitem__`:

```python
    def __getitem__(self, i: int):
        row = self._plan[i]
        rng = np.random.default_rng([self.cfg.seed, self.epoch, i])
        sprite = Image.fromarray(self._images[row], "RGBA")
        aug = self._photo_aug if str(self._cat[row]) == "booru" else self._sprite_aug
        img = aug.apply(sprite, rng, self.cfg.resolution)
        return np.asarray(img, dtype=np.uint8), float(self._smash[row])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sampler.py -q`
Expected: PASS (routing test + existing sampler tests).

- [ ] **Step 5: Commit**

```bash
git add data_prep/sampler.py tests/test_sampler.py
git commit -m "feat: sampler routes booru->photo aug, sprites->sprite aug"
```

---

## Task 5: Example config + full suite

**Files:**
- Create: `configs/example_mixed.json`
- Test: full suite

- [ ] **Step 1: Create configs/example_mixed.json**

```json
{
  "name": "mixed_v1",
  "resolution": 224,
  "seed": 0,
  "minimages": 3,
  "selection": {"categories": ["portrait", "in-game", "booru"]},
  "variations": {"fill_so": null},
  "split": {"strategy": "pokemon", "val_frac": 0.1},
  "augmentations": {
    "sprite": {
      "scale": {"w": [0.65, 1.10], "h": [0.65, 1.10]},
      "scale_method": "bilinear",
      "position": {"x": [0.10, 0.90], "y": [0.10, 0.90]},
      "rotation": [-10, 10],
      "flip": 0.5,
      "background": {"prob": 0.8, "dirs": ["backgrounds/real", "backgrounds/pokemon_battle"]}
    },
    "photo": {
      "crop_scale": [0.6, 1.0],
      "color": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
      "flip": 0.5
    }
  }
}
```

- [ ] **Step 2: Verify the example config parses**

Run:
```bash
uv run python -c "import json; from data_prep.config import DataConfig; c=DataConfig.from_dict(json.load(open('configs/example_mixed.json'))); print('booru' in c.selection.categories, c.augmentations.sprite.background.prob, c.augmentations.photo.crop_scale)"
```
Expected: `True 0.8 (0.6, 1.0)`

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (all tests green).

- [ ] **Step 4: Commit**

```bash
git add configs/example_mixed.json
git commit -m "feat: example mixed (sprites+booru+backgrounds) dataset config"
```

---

## Self-Review

**Spec coverage:**
- Category->kind routing (booru=photo, else sprite) -> Task 4. ✓
- Booru opt-in selection + read booru/meta.csv + relaxation excludes booru -> Task 2. ✓
- New ops (HFlip, RandomResizedCrop, ColorJitter, ToRGB) + BackgroundPool + CompositeBackground pool + sprite/photo builders -> Task 3. ✓
- Nested config + back-compat (flat->sprite, mode:white->prob 0) + validation -> Task 1. ✓
- Sampler loads category, builds both pipelines, routes -> Task 4. ✓
- No npz schema change (category already stored); prepare unchanged -> implicit (Task 4 reads existing `category`). ✓
- Tests: selection opt-in, augmentation ops/pool, routing, back-compat -> Tasks 1-4. ✓
- Example config -> Task 5. ✓

**Placeholder scan:** none — every step has complete code.

**Type consistency:** `DataConfig.augmentations.{sprite,photo}`, `SpriteAugCfg.{scale,scale_method,position,rotation,flip,background}`, `BackgroundCfg.{prob,dirs,weights}`, `PhotoAugCfg.{crop_scale,color,flip}`, `ColorCfg.{brightness,contrast,saturation,hue}`; `build_sprite_aug`/`build_photo_aug`/`build_augmentations`; `BackgroundPool(dirs, weights)` with `.empty()/.sample()/__len__`; `CompositeBackground(pool, prob)`; sampler `self._cat`, `self._sprite_aug`, `self._photo_aug`. Consistent across tasks.

**Note:** existing model/data-prep tests build tiny datasets and run the sampler; after Task 4 the sprite pipeline includes a `BackgroundPool` over the real `backgrounds/` dirs (which exist in the repo), so sprite samples may show a real background — tests assert shapes/determinism only, which still hold. `build_augmentations` stays an alias so `tests/test_augmentations.py` end-to-end/determinism tests are unaffected.

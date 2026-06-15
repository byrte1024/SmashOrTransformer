# Source-Aware Augmentation: Backgrounds + Booru Images - Design

Date: 2026-06-15
Status: Approved design, pending implementation plan

## Goal

Extend the data pipeline with two source-specific behaviors:

1. **Backgrounds** apply only to the base Pokemon sprites (transparent RGBA):
   composite the augmented sprite onto a real background sampled from the
   `backgrounds/` folders (instead of the white stub).
2. **Booru fan-art** becomes an opt-in alternative image source. Booru images
   are opaque, so they get crop / scale / color / flip augmentation and NO
   background compositing.

Both sources can coexist in one dataset; every image for a Pokemon shares that
Pokemon's `smash_pct` label. Builds on the data-prep pipeline
(`2026-06-14-data-prep-design.md`) and the booru scraper
(`data_prep/booru.py`, which fills `images/{id}/booru/`).

## Core concept: augmentation "kind" routed by category

Each image already carries a `category`. Augmentation is routed by mapping
category -> kind:

- `sprite` kind: `portrait`, `in-game`, `animated` (transparent RGBA) ->
  geometric augmentation + background compositing.
- `photo` kind: `booru` (opaque) -> random-resized-crop + color jitter +
  horizontal flip; no background.

No npz schema change: `category` is already stored; the sampler derives the kind
and selects the matching pipeline. (`load_sprite` converts every image to RGBA,
so booru images are stored RGBA-opaque alongside the sprites.)

## Selection (`data_prep/selection.py`)

- `CATEGORIES` gains `"booru"`.
- `load_records` additionally reads `images/{id}/booru/meta.csv` when that file
  exists, emitting `ImageRecord`s with `category="booru"`, `gen=0`,
  `source_name` = the booru filename stem, `path` = the booru image.
- **Booru is opt-in.** `_passes()`:
  - a `booru` record passes only if `"booru"` is explicitly in
    `selection.categories` (so the default empty=all and any sprite-only config
    exclude it);
  - non-booru records use the existing logic (empty categories = all sprite
    categories; name/gen filters apply).
- minimages relaxation re-adds only non-booru (sprite) records, so booru never
  sneaks in as a fallback when not requested.

## Augmentation (`data_prep/augmentations.py`)

New ops (each an `Augmentation` with `apply(img, rng, resolution)`):

- `HorizontalFlip(prob)` - random horizontal flip.
- `RandomResizedCrop(scale_range)` - crop a random area fraction (in
  `scale_range`) at a random location, resize to `resolution` (square).
- `ColorJitter(brightness, contrast, saturation, hue)` - random per-sample color
  perturbation.
- `BackgroundPool(dirs, weights)` - lazily lists image files across `dirs`, and
  caches each chosen background resized+center-cropped to the canvas. `sample(rng,
  resolution)` returns an RGB background surface (or None if the pool is empty).
- `CompositeBackground` gains pool support: with probability `prob` it composites
  the sprite over a `BackgroundPool` sample; otherwise (or if the pool is empty)
  white. Output is RGB `[res,res,3]`.

Builders:

- `build_sprite_aug(cfg)` -> `Compose([Scale, Rotate, Position, HorizontalFlip,
  CompositeBackground(pool)])`.
- `build_photo_aug(cfg)` -> `Compose([RandomResizedCrop, ColorJitter,
  HorizontalFlip, ToRGB])` where `ToRGB` flattens RGBA-opaque to RGB at
  `resolution`.

Both pipelines output RGB uint8 `[res,res,3]` so batches stay uniform.

## Config (`data_prep/config.py`)

Nested augmentation config:

```
augmentations:
  sprite:
    scale:        {w: [0.65, 1.10], h: [0.65, 1.10]}
    scale_method: bilinear
    position:     {x: [0.10, 0.90], y: [0.10, 0.90]}
    rotation:     [-10, 10]
    flip:         0.5
    background:
      prob:    0.8
      dirs:    ["backgrounds/real", "backgrounds/pokemon_battle"]
      weights: null            # optional per-dir weights; null = equal
  photo:
    crop_scale: [0.6, 1.0]
    color:      {brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.05}
    flip:       0.5
```

**Back-compatibility:** if `augmentations` contains the old flat keys
(`scale`/`scale_method`/`position`/`rotation`/`background`) and no `sprite` key,
those are parsed as the `sprite` config, and `photo` takes defaults. So existing
`datasets/portraits_v1/config.json` and `configs/example.json` load unchanged.
`background.mode: white` (old) maps to `background.prob: 0.0`.

Dataclasses: `SpriteAugCfg` (existing scale/position/rotation + `flip` +
`BackgroundCfg`), `BackgroundCfg` (`prob`, `dirs`, `weights`), `PhotoAugCfg`
(`crop_scale`, `ColorCfg`, `flip`), `AugmentationsCfg` (`sprite`, `photo`).

## Sampler (`data_prep/sampler.py`)

- Also load the npz `category` array.
- Build both pipelines once: `build_sprite_aug(cfg)` (with its `BackgroundPool`)
  and `build_photo_aug(cfg)`.
- `__getitem__(i)`: look up the row's category -> kind (`booru` -> photo, else
  sprite) -> apply that pipeline with the per-sample RNG
  `default_rng([seed, epoch, i])`. The RNG also drives background choice and
  color jitter, so results stay reproducible.
- Output unchanged: `(RGB uint8 [res,res,3], label float)`.

## Data flow

```
images/{id}/{meta.csv sprites, booru/meta.csv} --selection(categories incl. booru)-->
  manifest --prepare--> data.npz (images RGBA, category, ...) --DataSampler-->
    per row: category->kind-> sprite-aug(+bg pool) | photo-aug --> RGB sample
```

## Testing

Selection:
- booru excluded by default (categories empty) and for sprite-only configs;
  included only when `categories=["booru"]`; mixed config includes both.
- relaxation never adds a booru record.

Augmentations:
- `HorizontalFlip(1.0)` mirrors; `(0.0)` is identity.
- `RandomResizedCrop` returns `resolution` square; cropped region within bounds.
- `ColorJitter` changes pixels and is deterministic for a fixed rng.
- `BackgroundPool`: returns a resized-to-canvas surface; `CompositeBackground`
  with `prob=1` over a non-white pool yields non-white corners; `prob=0` (or
  empty pool) yields white.
- `build_photo_aug` output is RGB `[res,res,3]`.

Sampler / integration:
- A mixed dataset (a Pokemon with sprites + a `booru/` folder, `categories`
  including `booru`) routes booru rows through photo aug and sprite rows through
  sprite aug (e.g. a sprite sample can show a background; a booru sample fills
  the frame). Determinism preserved.

Back-compat:
- An old flat-aug `config.json` loads via `from_dict` and the sampler produces
  samples without error.

## Out of scope

- Per-source `smash_pct` weighting or separate labels (all sources share the
  Pokemon's label).
- Training-side changes (the model/training pipeline is unaffected; it still
  consumes `(image, label)` from the sampler).
- `backgrounds/illustrated` is empty, so it is omitted from the default pool
  until populated.

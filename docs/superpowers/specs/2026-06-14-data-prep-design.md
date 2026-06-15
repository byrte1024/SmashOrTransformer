# Data Preparation Pipeline - Design

Date: 2026-06-14
Status: Approved design, pending implementation plan

## Goal

Turn the on-disk assets (`images/{id}/*` sprites + `meta.csv`, `pokesmash_votes.csv`
labels) into a self-contained, reproducible dataset artifact under
`datasets/{name}/`. The dataset stores **clean base sprites + a resolved config**;
augmentation happens **online** in a sampler at read time. A shared set of ABCs
guarantees the preparer and the sampler interpret the config identically.

Backgrounds are out of scope for this iteration (folders exist but are not wired
in). The compositing step is stubbed to a flat white background, designed so real
background sampling drops in later without changing the dataset format.

## Architecture

Shared core (used by BOTH `prepare.py` and `sampler.py`):

- `config.py` - dataclasses for the full config; parsing, defaults, validation.
- `selection.py` - filtering logic: given config, produce the per-Pokemon image
  manifest (which `images/{id}/*` survive class/gen/name filters + minimages
  relaxation).
- `augmentations.py` - `Augmentation` ABC + `Compose`; concrete ops `Scale`,
  `Rotate`, `Position`, `CompositeBackground`.
- `splits.py` - `SplitStrategy` ABC -> `PokemonLevelSplit`, `ImageLevelSplit`.
- `sampling.py` - `SamplingStrategy` ABC -> `FlatVariations`, `FillSo`.

Entry points:

- `prepare.py` - builds `datasets/{name}/` (selection -> pack npz -> write
  config/manifest/split/stats).
- `sampler.py` - `DataSampler` reads the dataset dir and yields augmented
  `(image, label)` samples. Standalone now; wraps into a torch `Dataset` later.
  Built and tested in this iteration.

## Config schema

Resolved config is written to `datasets/{name}/config.json`.

```
name                    # dataset folder name
seed                    # global RNG seed for splits + augmentation
resolution              # square int; what the model is fed

selection:
  categories: [...]     # subset of {portrait, in-game, animated}; [] = all
  names:                # specific source names (official-artwork, home, ...)
    include: [...]      #   [] = all
    exclude: [...]
  gens:                 # national-dex generations 1..9
    include: [...]      #   [] = all
    exclude: [...]

minimages               # floor on distinct images per Pokemon
variations              # int N  OR  {fill_so: <target|null>}

split:
  strategy: pokemon | image   # default: pokemon
  val_frac: 0.1

augmentations:
  scale:        {w: [0.65, 1.10], h: [0.65, 1.10]}  # ratios of resolution
  scale_method: nearest | bilinear | bicubic | lanczos
  position:     {x: [0.10, 0.90], y: [0.10, 0.90]}  # center, fraction of canvas
  rotation:     [-10, 10]                            # degrees
  background:   {mode: white}                        # placeholder; bg later
```

## Storage format (`datasets/{name}/`)

Sprites have heterogeneous native sizes (96x96 pixel art up to 512x512 renders).
They are stored **raw at native resolution** and scaled to the target resolution
at training time (single resample, best fidelity).

- `data.npz` (saved with `allow_pickle=True`):
  - `images` - object array `[N]`, each entry a uint8 RGBA array `HxWx4` at
    native size.
  - `pokemon_id` - int `[N]`
  - `category` - str `[N]`  (portrait | in-game | animated)
  - `gen` - int `[N]`
  - `source_name` - str `[N]`  (e.g. official-artwork, gen3_emerald)
  - `smash_pct` - float `[N]` in [0, 1]
  - `total_votes` - int `[N]`
- `config.json` - the resolved config (all defaults filled).
- `manifest.csv` - human-readable: row -> pokemon_id, source_name, category,
  gen, smash_pct, total_votes.
- `split.json` - train/val assignment for the chosen strategy + seed.
- `stats.json` - N, per-Pokemon image counts, label histogram, and a
  selection report (how many images each Pokemon had before/after filtering,
  which were relaxed back in, which Pokemon were augmentation-padded).

Animated sources (showdown, gen5 animated GIFs): take the **first frame** as a
static RGBA image.

## Selection, minimages, variations

Selection per Pokemon `p`:

1. Apply category + gen + name filters -> distinct image set `S_p`.
2. **minimages relaxation**: if `|S_p| < minimages`, re-add excluded images in
   priority order until `minimages` reached or images exhausted:
   portraits -> newest in-game gen -> older in-game gens -> animated.
3. If the Pokemon *physically* owns fewer than `minimages` images, keep all of
   them; the per-Pokemon sample count is padded by augmentation (see variations).

Variations (how an epoch is enumerated by the sampler; NOT materialized pixels):

- flat `N`: each base image yields `N` augmented samples per epoch ->
  `samples_per_pokemon = |S_p| * N`.
- `fill_so: T`: every Pokemon emits exactly `T` samples per epoch regardless of
  `|S_p|`, cycling through `S_p` with fresh augmentations. Default
  `T = max_p(|S_p|)` (richest Pokemon sets the bar; the rest augment up to match),
  giving perfectly balanced per-Pokemon exposure.

## Train/val split (ABC)

`SplitStrategy.split(manifest, val_frac, seed) -> (train_idx, val_idx)`.

- `PokemonLevelSplit` (default): a Pokemon's images all land in train OR val.
  Tests generalization to unseen species - the right target for this task.
- `ImageLevelSplit`: split individual rows; a Pokemon may appear in both sets.

## Augmentation pipeline (ABC)

`Augmentation.apply(sprite_rgba, rng, resolution) -> image`. `Compose` chains
them. Per-sample RNG is seeded by `hash(seed, epoch, index)` for reproducibility.

Order, per sample:

1. `Scale` - resize raw sprite to `(w_ratio*res, h_ratio*res)`, ratios drawn from
   `scale.w` / `scale.h`, using `scale_method`.
2. `Rotate` - rotate by `theta` drawn from `rotation`, expand canvas, keep alpha.
3. `Position` - paste onto a transparent `res x res` canvas at a center drawn from
   `position` ranges (partial off-canvas clipping is a valid augmentation).
4. `CompositeBackground` - flatten onto white now (RGBA -> RGB `[res,res,3]`
   uint8). Real background sampling replaces this step later.

Sampler output: `(image RGB uint8 [res,res,3], label smash_pct float)`.
`total_votes` is exposed alongside for optional confidence weighting.

## Testing

Unit:
- Each augmentation: output shape/dtype; scale within ratio bounds; rotation
  applied; position center within range; white composite has no transparency.
- `selection`: category/gen/name filters; minimages relaxation priority;
  physical-shortfall fallthrough.
- `sampling`: flat count math; `fill_so` balances all Pokemon to `T`.
- `splits`: pokemon-level has zero Pokemon leakage across train/val; image-level
  partitions rows.

Integration:
- Build a tiny dataset (a handful of ids) via `prepare.py`, instantiate
  `DataSampler`, pull a batch; assert shapes, label alignment, and determinism
  (same seed -> identical pixels).

## Out of scope (this iteration)

- Background compositing (real/illustrated/battle) - format is ready; only the
  white stub is implemented.
- torch `Dataset`/`DataLoader` wrapper - sampler is framework-agnostic now.
- Multi-frame animation handling beyond first-frame extraction.

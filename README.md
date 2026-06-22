# Smash or Transformer

A Vision Transformer that looks at a Pokemon and predicts the crowd's
"smash" fraction (0-100) aka how attractive the internet finds it.
Ground-truth labels come from aggregate vote counts on
[pokesmash.xyz](https://pokesmash.xyz); images come from official artwork,
in-game sprites, and (optionally) Safebooru fan-art.

Take it away claude.

The model is a pretrained `timm` ViT-Small fine-tuned with a soft-label BCE
objective, selected on validation Spearman correlation, and post-hoc
isotonically calibrated.

> Trained weights and their scores are tracked in [MODELS.md](MODELS.md).

---

## Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). Everything
runs through `uv run`, which resolves the environment on first use.

```bash
uv sync                 # create the environment from pyproject.toml + uv.lock
uv run pytest -q        # sanity-check the install (CPU-only, no GPU needed)
```

Training/inference default to CUDA (`--device cuda`); pass `--device cpu` to
force CPU. The pinned PyTorch is a CUDA 13.0 build.

### Repo layout

| Path | What |
|------|------|
| `data_prep/` | scraping, selection, packing, augmentation, sampling |
| `model/` | architecture, training, calibration, scoring, "dreaming" |
| `gimmicks/` | a pygame scoring GUI and a Discord bot |
| `configs/` | dataset-prep and training configs (JSON) |
| `images/` | per-Pokemon image folders (gitignored, regenerable) |
| `backgrounds/` | background pools used to composite transparent sprites |
| `datasets/` | packed dataset artifacts (gitignored) |
| `runs/` | training runs: checkpoints, history, predictions (gitignored) |
| `results/` | rendered scorecards and CSVs (gitignored) |

---

## End-to-end pipeline

The flow is: **gather labels + images -> prepare a dataset -> train ->
calibrate -> score / visualize**. Each stage is an independent CLI.

### 1. Gather labels and images

```bash
# Crowd vote counts -> pokesmash_votes.csv
uv run python scrape_pokesmash.py

# Official artwork + sprites into images/{id}/, with a per-folder meta.csv
uv run python download_images.py

# Optional: top Safebooru fan-art into images/{id}/booru/ (filters out
# human-only and group pics). --top sets how many to keep per Pokemon.
uv run python -m data_prep.booru --top 75
```

Backgrounds (for compositing transparent sprites) live under
`backgrounds/real/` and `backgrounds/pokemon_battle/`. The battle backgrounds
can be sliced from sprite sheets with `uv run python slice_battle_bgs.py`.

### 2. Prepare a dataset

`prepare` reads a JSON config and packs the selected images into
`datasets/{name}/` as a memory-bounded blob (`images.bin` + `data.npz`) plus a
Pokemon-level train/val split.

```bash
uv run python -m data_prep.prepare configs/mixed_v2.json
```

Config knobs (see `configs/example_mixed.json` for an annotated example):

- **`selection.categories`** -- which image sources to include
  (`portrait`, `in-game`, `booru`). Booru is opt-in.
- **`minimages`** -- minimum images per Pokemon; below this, augmentation
  relaxation kicks in so every Pokemon still contributes.
- **`variations`** -- samples per epoch. `{"fill_so": N}` gives every Pokemon
  exactly `N` augmented samples per epoch (balancing rare vs popular ones),
  cycling its images equally; a bare integer means N samples per image.
- **`augmentations`** -- two source-aware pipelines:
  - `sprite` (official/in-game): scale, rotate, position, flip, then
    composite onto a real background with some probability.
  - `photo` (booru): color jitter, flip, scale 0.9-1.2x, position jitter,
    composited onto a random white / black / real background.

Training-time decoding is accelerated by a one-time capped-image cache
(`cache_s*.bin`) built automatically on first use and reused across runs.

### 3. Train

```bash
uv run python -m model.train configs/train_mixed_v2.json
# override on the fly:
uv run python -m model.train configs/train_mixed_v2.json --epochs 40 --device cuda
```

Writes to `runs/{run_name}/`:

- `checkpoints/best.pt` (best val Spearman), `last.pt`, and `epoch_NNN.pt`
- `history.csv` / `history.jsonl` -- per-epoch loss + correlations
- `predictions/` -- per-epoch validation predictions
- `config.json` -- full train + dataset config snapshot

Training warms up a fresh head with the backbone frozen for `freeze_epochs`,
then unfreezes and fine-tunes end-to-end with a cosine schedule.

### 4. Calibrate

The raw sigmoid output is monotonic but not on the true smash-fraction scale.
Isotonic calibration fits a mapping and writes `calibration.json` next to the
checkpoint (with `train` / `val` / `combined` maps; `val` is the default for
unseen inputs).

```bash
uv run python -m model.calibrate --checkpoint runs/vit_small_mixed_v2/checkpoints/best.pt
```

---

## Scoring and inference

```bash
# Score arbitrary image files (0-100), calibrated
uv run python -m model.infer --checkpoint runs/.../best.pt path/to/img.png

# Score a whole folder, render labeled scorecards + scores.csv
uv run python -m model.score_folder --checkpoint runs/.../best.pt path/to/folder

# Re-score the full dataset and render per-Pokemon cards + rankings.csv
uv run python -m model.results --checkpoint runs/.../best.pt

# Compare scores across image sources (portrait vs in-game vs booru)
uv run python -m model.compare_sources --checkpoint runs/.../best.pt

# Score the scraped booru fan-art per Pokemon
uv run python -m model.score_booru --checkpoint runs/.../best.pt
```

All scorers accept `--calibration {val,train,combined,none}` and
`--no-stretch` (aspect-fit instead of the default stretch-to-square input).

---

## Gimmicks

### Scoring GUI

```bash
uv run python -m gimmicks.score_image_app
```

A pygame app that auto-discovers runs under `runs/` and lets you score images
interactively. A sidebar lists models and their checkpoints (click to load).
Drag-and-drop images, whole folders, or a `.pt` checkpoint onto the window.

- **Left / Right** -- previous / next image
- **o** -- open a file browser, **s** -- save the scored card to
  `results/shared/<name>_scored.png`, **Esc** -- quit

Each card shows raw %, calibrated %, and a colored SMASH/PASS banner. Flags:
`--checkpoint`, `--runs runs`, `--device`, `--threshold 0.5`, `--no-stretch`.

### Discord bot

A joke "smash or pass" bot. Reply to / attach an image, say `me` to rate your
own avatar, or @mention someone to rate theirs.

```bash
uv run python -m gimmicks.discord_bot
```

Needs two files (paths overridable via flags):

- `gimmicks/secret.txt` -- the Discord bot token (one line)
- `gimmicks/thebestofthebest.txt` -- the checkpoint path to use (one line),
  e.g. `runs/vit_small_mixed_v2/checkpoints/best.pt`

Optional `gimmicks/settings.json` (`{"light_llm": bool, "llm_model": "claude-haiku-4-5"}`):
with `light_llm` on (or `--light-llm`), the bot shells out to the local
`claude` CLI for a one-line quip per verdict, cached in
`gimmicks/llm_cache.json`. Other flags: `--checkpoint`, `--model-file`,
`--settings`, `--device`, `--threshold`.

> **Leave `light_llm` off.** In practice it's unreliable -- most LLMs
> (Anthropic's included) refuse with a "can't fulfill that request" roughly
> 1 in 10 times, so the quips are inconsistent. It also relies on prompting the
> model into rating images, which isn't something to lean on: do not do this
> with API keys you care about, as that kind of use can get a key banned. The
> plain model verdict works fine on its own.

---

## "Dreaming": what the model finds attractive

Two ways to visualize what maximizes (or minimizes) the predicted score, each
sweeping target brackets `1.0 ... 0.0` and writing labeled images, grids, and
a manifest CSV.

### Diffusion dreaming -- plausible, score-targeted images

Steers a frozen Stable Diffusion 1.5 prior with the frozen smash model, so the
results stay image-like. SD weights download on first run.

```bash
uv run python -m model.dream_diffusion --checkpoint runs/.../best.pt
```

Three guidance strategies (`--methods xhat,doodl,sds`):

- **xhat** -- per-step universal guidance on the predicted clean image (x0-hat)
- **doodl** -- optimize the initial noise by backprop through the DDIM chain
- **sds** -- score-distillation sampling (DreamFusion-style)

Key flags: `--brackets`, `--n-per 4`, `--steps 30`, `--guidance-scale 200`,
`--sd-guidance-scale 7.5`, `--model-id`, `--seed0`. Resume-safe (skips images
already on disk). Output: `results/dream_diffusion/<method>/<cond>/bracket_*/`
+ grids + `manifest.csv`.

### Activation maximization -- the raw "ideal" per bracket

Optimizes images directly against the model (no diffusion prior), cycling six
feature-visualization techniques (`pixel_tv`, `robust`, `blur`, `fourier`,
`maco`, and `deepdream` which starts from a real official-artwork seed).

```bash
uv run python -m model.acti_maxim --checkpoint runs/.../best.pt --steps 256
```

Flags: `--n-per 10`, `--steps 256`, `--res`, `--images images`. Output:
`results/acti_maxim/bracket_*/` + grids + `summary.csv`.

---

## Testing

```bash
uv run pytest -q              # full suite (CPU)
uv run pytest tests/test_imagestore.py -q
```

Tests run on CPU with a tiny untrained backbone, so they're fast and need no
GPU or downloaded weights.

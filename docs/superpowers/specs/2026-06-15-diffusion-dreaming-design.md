# Diffusion Dreaming: Plausible Score-Maximizing Images - Design

Date: 2026-06-15
Status: Approved design, pending implementation plan

## Goal

Generate **plausible images** (not the texture-y output of pixel activation
maximization) that the frozen smash model scores at a target level, by using a
pretrained diffusion model as a natural-image prior steered by our model. This
complements `model/acti_maxim.py` (kept as the "pure model introspection" view);
this module is the "what would a plausible high/low-smash creature look like?"
view.

Builds on the trained model + calibration (`model/infer.py`, `model/calibrate.py`)
and reuses the canonical render / scoring conventions.

## Approach

A **frozen Stable Diffusion 1.5** prior (HuggingFace `diffusers`) is steered by
the frozen smash model. All guidance shares one differentiable objective:

```
loss(latent, target) = (sigmoid(smash_model(norm(resize(vae.decode(latent), R)))) - target)^2
```

i.e. VAE-decode the latent -> differentiably resize to the model resolution R
(224) -> normalize (model's mean/std) -> smash model -> push the score toward
the bracket target. R, mean, std come from the loaded checkpoint.

## Guidance strategies (`GuidanceStrategy` ABC, pluggable like AM techniques)

Each implements `generate(pipe, smash_model, prompt, target, seed, cfg) -> PIL.Image`:

- **`xhat`** (Universal x0 guidance, the workhorse): run the normal DDIM denoise
  loop; at each step compute the predicted clean latent x0-hat, evaluate the
  loss on it, and nudge the latent by `-guidance_scale * grad`. Frozen SD,
  moderate memory.
- **`doodl`** (End-to-End latent optimization): optimize the initial noise latent
  by backpropagating the final-image loss through a short deterministic DDIM
  chain (few steps + gradient checkpointing to fit 12 GB). Highest fidelity,
  heaviest.
- **`sds`** (Score Distillation): optimize a latent with the SD score-distillation
  gradient plus the smash-loss gradient (DreamFusion-style). Flexible, noisier.

`xhat` is implemented first as a complete vertical slice; `doodl` and `sds` are
added as separate strategies.

## Conditioning (text prompts)

Three conditioning modes, run for each cell:
- `""` (unconditional) - score guidance fully shapes the image.
- `"a pokemon creature, full body, plain background"` - coherent creature.
- `"a pokemon creature, furry, anthro"` - anthro/creature variation.

The prompt list is configurable; these three are the defaults.

## Targets

Brackets `[1.0, 0.8, 0.6, 0.4, 0.2, 0.0]` (reverse order, highest-smash first).
Configurable.

## Generation matrix & output

For each cell `(method x conditioning x bracket)`, generate `n_per` images
(default 4) with distinct seeds. After generation, **score each image with the
calibrated smash model** (canonical render -> model -> sigmoid -> calibration
'combined' map) to label it with the achieved %.

Outputs under `results/dream_diffusion/`:
- `{method}/{cond_tag}/bracket_{b}/{seed}.png` - each labeled with achieved
  raw -> calibrated %.
- `grid_{method}_{cond_tag}_{b}.png` - contact sheet per cell (title = target +
  prompt).
- `manifest.csv` - one row per image: method, conditioning, prompt, target,
  seed, achieved_raw_pct, achieved_calibrated_pct.

Default matrix = 3 methods x 3 conditioning x 6 brackets = 54 cells x n_per;
all of methods / prompts / brackets / n_per / steps / guidance_scale are config
so a run can subset (e.g. `--methods xhat --brackets 1.0,0.0`).

## Config / CLI

`DreamConfig` (or CLI args): `checkpoint`, `model_id` (default
`sd-legacy/stable-diffusion-v1-5`, since the original runwayml repo was removed from HF), `methods`, `prompts`, `brackets`, `n_per`,
`steps`, `guidance_scale` (smash-guidance strength), `sd_guidance_scale` (CFG),
`device`, `out_dir`, `seed0`. The smash model + its calibration are loaded from
`checkpoint` via the existing `model.infer.load_model` / `load_calibration`.

## Dependencies & compute

- `diffusers`, `transformers`, `accelerate`, `safetensors` (added to `uv` /
  pyproject). SD1.5 weights (~4 GB) download on first pipeline load (cached by
  HuggingFace).
- fp16 on CUDA. SD1.5 @ 512 + guidance gradients (incl. VAE-decode backprop)
  fits the 12 GB RTX 5070 for `xhat`. `doodl`/`sds` use few steps + gradient
  checkpointing; flagged as memory-risky.
- Full matrix at `n_per=4` is ~144 guided generations (~30 min-2 hr); subset via
  config.

## Structure (`model/dream_diffusion.py`)

- `load_pipeline(model_id, device)` -> frozen `StableDiffusionPipeline` (eval,
  fp16 on cuda; safety checker disabled for speed).
- `smash_loss(smash_model, image_or_latent, target, ...)` -> scalar loss + the
  differentiable score helper, reused by all strategies.
- `GuidanceStrategy` ABC + `XHatGuidance`, `DoodlGuidance`, `SdsGuidance`.
- `score_image(smash_model, cfg, pil_image, calib)` -> achieved raw/calibrated
  (reuses `model.infer` scoring).
- `run(...)` -> drives the matrix, writes images/grids/manifest, prints a summary
  (mean achieved calibrated score per method, so we can see which method best
  hits its targets).
- `main()` CLI.

## Testing

SD cannot run in CI (no GPU, ~4 GB download). Tests **mock the pipeline** and
cover the pure logic:
- `smash_loss`: gradient flows to the input; loss decreases as the input's score
  approaches the target; direction correct for high vs low targets (use a tiny
  smash model, no SD).
- cell/bracket enumeration: reverse bracket order (1.0 first); the
  methods x prompts x brackets matrix is built correctly.
- `run()` end-to-end with a **stub pipeline** (returns a fixed small image and a
  fake latent): writes the expected folder tree, per-cell grids, and a
  `manifest.csv` with the right columns; achieved scores are recorded.
- `score_image` labeling: returns calibrated % in [0,100].

A real SD smoke test is a manual GPU step (documented in the plan), not in the
automated suite.

## Out of scope

- Fine-tuning SD or training any model (everything frozen).
- A Pokemon-specific diffusion prior (we use stock SD1.5 + prompts).
- Realtime/interactive generation (batch CLI only).
- Multi-GPU.

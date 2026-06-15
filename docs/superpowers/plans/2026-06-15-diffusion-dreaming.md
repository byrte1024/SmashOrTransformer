# Diffusion Dreaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate plausible images that the frozen smash model scores at target levels, using a frozen Stable Diffusion 1.5 prior steered by the smash model (three guidance strategies, three prompts, reverse bracket sweep).

**Architecture:** `model/dream_diffusion.py` loads a frozen SD1.5 (`diffusers`) and a frozen smash model. A shared differentiable `smash_score` (VAE-decode latent -> resize 224 -> normalize -> model -> sigmoid) drives three pluggable `GuidanceStrategy` implementations (xhat / doodl / sds). `run()` sweeps method x prompt x bracket, scores each generated image with the calibrated model, and writes labeled images + grids + a manifest.

**Tech Stack:** PyTorch (cu130), diffusers 0.38, transformers, accelerate, safetensors, Pillow. Heavy SD code is GPU-only; tests use a tiny fake pipe + stub strategies (no SD download, CPU).

Spec: `docs/superpowers/specs/2026-06-15-diffusion-dreaming-design.md`

---

## File Structure

```
model/dream_diffusion.py    # helpers, GuidanceStrategy ABC + 3 strategies, load_pipeline, run, CLI
tests/test_dream_diffusion.py   # pure helpers + run-orchestration (stub) + strategy wiring (fake pipe)
```

Conventions: smash model + calibration loaded via existing `model.infer.load_model` /
`load_calibration`. Strategies return a 512x512 RGB PIL image. `run()` accepts injected
`pipe` and `strategies` so tests never touch real SD. Deps already in `pyproject.toml`.

---

## Task 1: Differentiable scoring + image helpers

**Files:**
- Create: `model/dream_diffusion.py`
- Test: `tests/test_dream_diffusion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dream_diffusion.py
import numpy as np
import torch
from PIL import Image
from model.model import SmashRanker
from model import dream_diffusion as dd


def _tiny_model():
    return SmashRanker("vit_tiny_patch16_224", resolution=32, dropout=0.0, pretrained=False).eval()


def test_score_tensor_shape_and_range():
    m = _tiny_model()
    img01 = torch.rand(2, 3, 96, 96)            # B,3,H,W in [0,1]
    s = dd._score_tensor(m, img01, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 32)
    assert s.shape == (2,) and float(s.min()) >= 0 and float(s.max()) <= 1


def test_smash_loss_grad_flows_and_direction():
    m = _tiny_model()
    img = torch.rand(1, 3, 96, 96, requires_grad=True)
    loss = dd.smash_loss(m, img, target=1.0, mean=(0.5,) * 3, std=(0.5,) * 3, resolution=32)
    loss.backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()


def test_to_pil_roundtrip():
    img01 = torch.zeros(1, 3, 8, 8); img01[:, 0] = 1.0   # red
    pil = dd.to_pil(img01)
    assert isinstance(pil, Image.Image) and pil.size == (8, 8)
    assert np.asarray(pil)[0, 0, 0] == 255
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_dream_diffusion.py -q`
Expected: FAIL (`No module named 'model.dream_diffusion'`).

- [ ] **Step 3: Create model/dream_diffusion.py with the helpers**

```python
"""Diffusion dreaming: plausible score-maximizing images.

A frozen Stable Diffusion 1.5 prior (diffusers) steered by the frozen smash
model. Three guidance strategies (xhat / doodl / sds), three prompts, and a
reverse bracket sweep. Each generated image is scored back through the
calibrated model and labeled with its achieved %.

The heavy SD code runs on GPU only; the pure helpers and the run() orchestration
are unit-tested with a tiny model + a fake pipe (no SD download).
"""
from __future__ import annotations
import argparse
import csv
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from .infer import load_model, load_calibration
from .dataset import render_input, to_tensor
from .calibrate import apply_calibration

DEFAULT_PROMPTS = [
    ("uncond", ""),
    ("creature", "a pokemon creature, full body, plain background"),
    ("anthro", "a pokemon creature, furry, anthro"),
]
DEFAULT_BRACKETS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]   # reverse: highest-smash first
DEFAULT_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"


def _score_tensor(smash_model, img01, mean, std, resolution):
    """Differentiable smash score in [0,1] for a [B,3,H,W] image in [0,1]."""
    x = F.interpolate(img01, size=resolution, mode="bilinear", align_corners=False)
    m = torch.tensor(mean, device=img01.device, dtype=img01.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=img01.device, dtype=img01.dtype).view(1, 3, 1, 1)
    return torch.sigmoid(smash_model((x - m) / s).reshape(-1))


def smash_loss(smash_model, img01, target, mean, std, resolution):
    return ((_score_tensor(smash_model, img01, mean, std, resolution) - target) ** 2).mean()


def to_pil(img01) -> Image.Image:
    arr = (img01[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_dream_diffusion.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add model/dream_diffusion.py tests/test_dream_diffusion.py
git commit -m "feat: differentiable smash scoring + image helpers for diffusion dreaming"
```

---

## Task 2: GuidanceStrategy ABC, calibrated scoring, run() orchestration

**Files:**
- Modify: `model/dream_diffusion.py`
- Test: `tests/test_dream_diffusion.py`

- [ ] **Step 1: Append the failing test**

```python
# tests/test_dream_diffusion.py  (append)
import csv as _csv
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run as train_run
from model.calibrate import run as calib_run
from model.infer import load_model as _load_model, load_calibration as _load_calib


class _StubStrategy(dd.GuidanceStrategy):
    def generate(self, pipe, smash_model, prompt, target, seed, mean, std):
        # ignore SD; return a flat image whose brightness encodes the target
        v = int(target * 255)
        return Image.fromarray(np.full((32, 32, 3), v, np.uint8), "RGB")


def _trained(mini_repo, tmp_path):
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    run_dir = train_run(TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "dd", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"}), pretrained=False)
    calib_run(run_dir / "checkpoints" / "best.pt", device="cpu", batch_size=4, num_workers=0)
    return run_dir / "checkpoints" / "best.pt"


def test_run_orchestration_with_stub(mini_repo, tmp_path):
    ckpt = _trained(mini_repo, tmp_path)
    out = dd.run(ckpt, out_dir=str(tmp_path / "dream"), device="cpu",
                 methods=["stub"], prompts=[("uncond", ""), ("creature", "x")],
                 brackets=[1.0, 0.0], n_per=2, pipe=object(),
                 strategies={"stub": _StubStrategy()})
    # folder tree: method/cond/bracket/seed.png
    pngs = list((out).rglob("*.png"))
    # 1 method x 2 prompts x 2 brackets x (2 imgs + 1 grid) = 12 png
    assert len(pngs) == 12
    rows = list(_csv.DictReader(open(out / "manifest.csv")))
    assert len(rows) == 1 * 2 * 2 * 2                      # method*prompt*bracket*n_per
    assert {"method", "conditioning", "prompt", "target", "seed",
            "achieved_raw_pct", "achieved_calibrated_pct"} == set(rows[0].keys())
    # brackets recorded in reverse order (1.0 before 0.0) within a cell grouping
    assert rows[0]["target"] == "1.0"


def test_score_pil_in_range(mini_repo, tmp_path):
    ckpt = _trained(mini_repo, tmp_path)
    model, cfg = _load_model(ckpt, device="cpu", pretrained=False)
    calib = _load_calib(ckpt, fit="combined")
    raw, cal = dd.score_pil(model, cfg, Image.new("RGB", (40, 40), (200, 100, 50)),
                            calib, device="cpu")
    assert 0 <= raw <= 100 and 0 <= cal <= 100
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_dream_diffusion.py -q`
Expected: FAIL (`GuidanceStrategy` / `run` / `score_pil` not defined).

- [ ] **Step 3: Append to model/dream_diffusion.py**

```python
# model/dream_diffusion.py  (append)
class GuidanceStrategy(ABC):
    """Generate one 512x512 RGB image steered toward `target` smash score."""
    @abstractmethod
    def generate(self, pipe, smash_model, prompt, target, seed, mean, std) -> Image.Image:
        ...


def score_pil(smash_model, cfg, pil, calib, device="cuda") -> tuple[float, float]:
    """Score a generated image with the calibrated model -> (raw%, calibrated%)."""
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    arr = np.asarray(pil.convert("RGBA"), dtype=np.uint8)
    img = render_input(arr, cfg.resolution, True)   # stretch-to-square, matches eval
    t = to_tensor(img, smash_model.data_config["mean"], smash_model.data_config["std"])
    t = t.unsqueeze(0).to(dev)
    with torch.no_grad():
        raw = float(torch.sigmoid(smash_model(t).reshape(-1)[0]))
    cal = float(apply_calibration([raw], calib[0], calib[1])[0]) if calib else raw
    return raw * 100, cal * 100


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _label(pil, text):
    w, h = pil.size
    bar = 28
    out = Image.new("RGB", (w, h + bar), (20, 20, 20))
    out.paste(pil, (0, 0))
    ImageDraw.Draw(out).text((4, h + 4), text, fill=(235, 235, 235), font=_font(15))
    return out


def _grid(tiles, cols=4):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    g = Image.new("RGB", (cols * w, rows * h), (10, 10, 10))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        g.paste(t, (c * w, r * h))
    return g


def load_pipeline(model_id=DEFAULT_MODEL_ID, device="cuda"):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    use_cuda = device == "cuda" and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype,
                                                   safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda" if use_cuda else "cpu")
    pipe.set_progress_bar_config(disable=True)
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)
    return pipe


def run(checkpoint_path, out_dir="results/dream_diffusion", device="cuda",
        methods=None, prompts=None, brackets=None, n_per=4, steps=30,
        guidance_scale=200.0, sd_guidance_scale=7.5, model_id=DEFAULT_MODEL_ID,
        seed0=0, pipe=None, strategies=None) -> Path:
    methods = methods or ["xhat", "doodl", "sds"]
    prompts = prompts or DEFAULT_PROMPTS
    brackets = brackets if brackets is not None else DEFAULT_BRACKETS
    smash_model, cfg = load_model(checkpoint_path, device=device, pretrained=False)
    calib = load_calibration(checkpoint_path, fit="combined")
    mean, std = smash_model.data_config["mean"], smash_model.data_config["std"]
    if strategies is None:
        strategies = _build_strategies(steps, guidance_scale, sd_guidance_scale)
    if pipe is None:
        pipe = load_pipeline(model_id, device)

    out = Path(out_dir)
    rows = []
    for method in methods:
        strat = strategies[method]
        for cond_tag, prompt in prompts:
            for b in brackets:
                cell = out / method / cond_tag / f"bracket_{b:.1f}"
                cell.mkdir(parents=True, exist_ok=True)
                tiles = []
                for k in range(n_per):
                    seed = seed0 + k
                    img = strat.generate(pipe, smash_model, prompt, b, seed, mean, std)
                    raw, cal = score_pil(smash_model, cfg, img, calib, device)
                    _label(img, f"raw {raw:.0f}% -> {cal:.0f}%").save(cell / f"{seed:03d}.png")
                    tiles.append(_label(img, f"{cal:.0f}%"))
                    rows.append({"method": method, "conditioning": cond_tag, "prompt": prompt,
                                 "target": b, "seed": seed,
                                 "achieved_raw_pct": round(raw, 2),
                                 "achieved_calibrated_pct": round(cal, 2)})
                _grid(tiles).save(out / f"grid_{method}_{cond_tag}_{b:.1f}.png")

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "conditioning", "prompt", "target",
                                          "seed", "achieved_raw_pct", "achieved_calibrated_pct"])
        w.writeheader(); w.writerows(rows)

    print(f"\nGenerated {len(rows)} images -> {out}/")
    for method in methods:
        for b in brackets:
            cal_vals = [r["achieved_calibrated_pct"] for r in rows
                        if r["method"] == method and r["target"] == b]
            if cal_vals:
                print(f"  {method:6s} target {b:.1f}: mean achieved {np.mean(cal_vals):.1f}%")
    return out


def _build_strategies(steps, guidance_scale, sd_guidance_scale):
    return {"xhat": XHatGuidance(steps, guidance_scale, sd_guidance_scale),
            "doodl": DoodlGuidance(steps, guidance_scale, sd_guidance_scale),
            "sds": SdsGuidance(steps, guidance_scale, sd_guidance_scale)}
```

(Forward-references `XHatGuidance` / `DoodlGuidance` / `SdsGuidance` are defined in
Tasks 3-5. `_build_strategies` is only called when `strategies` is not injected, so
Task 2's tests — which inject a stub — pass before those classes exist. Do NOT call
`run()` without `strategies=` until Task 5.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_dream_diffusion.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add model/dream_diffusion.py tests/test_dream_diffusion.py
git commit -m "feat: dream run() orchestration, calibrated scoring, output/manifest"
```

---

## Task 3: XHatGuidance (universal x0 guidance) + fake-pipe wiring test

**Files:**
- Modify: `model/dream_diffusion.py`
- Test: `tests/test_dream_diffusion.py`

- [ ] **Step 1: Append a fake-pipe wiring test**

```python
# tests/test_dream_diffusion.py  (append)
import torch.nn as _nn
from types import SimpleNamespace


class _FakeUNet(_nn.Module):
    def __init__(self): super().__init__(); self.c = _nn.Conv2d(4, 4, 3, padding=1)
    def forward(self, x, t, encoder_hidden_states=None): return SimpleNamespace(sample=self.c(x))
    __call__ = forward


class _FakeVAE(_nn.Module):
    def __init__(self):
        super().__init__(); self.w = _nn.Conv2d(4, 3, 1)
        self.config = SimpleNamespace(scaling_factor=0.18215)
    def decode(self, z):
        img = _nn.functional.interpolate(self.w(z), size=512, mode="nearest")
        return SimpleNamespace(sample=torch.tanh(img))


class _FakeScheduler:
    def __init__(self, n=1000):
        self.alphas_cumprod = torch.linspace(0.9999, 0.02, n)
        self.init_noise_sigma = 1.0
        self.config = SimpleNamespace(num_train_timesteps=n)
    def set_timesteps(self, steps, device=None):
        self.timesteps = torch.linspace(self.config.num_train_timesteps - 1, 0, steps).long()
    def scale_model_input(self, x, t): return x
    def step(self, noise, t, latents): return SimpleNamespace(prev_sample=latents - 0.01 * noise)


class _FakePipe:
    def __init__(self):
        self.unet = _FakeUNet(); self.vae = _FakeVAE(); self.scheduler = _FakeScheduler()
        self.device = torch.device("cpu")
    def encode_prompt(self, prompt, device, n, cfg, negative_prompt=""):
        e = torch.zeros(1, 4, 8)
        return e, e   # (prompt_embeds, negative_prompt_embeds)


def test_xhat_generate_returns_image_and_uses_model():
    pipe = _FakePipe()
    m = _tiny_model()
    calls = {"n": 0}
    orig = dd._score_tensor
    def counting(*a, **k):
        calls["n"] += 1; return orig(*a, **k)
    dd._score_tensor = counting
    try:
        img = dd.XHatGuidance(steps=3, guidance_scale=10.0, sd_guidance_scale=7.5).generate(
            pipe, m, "a creature", target=1.0, seed=0, mean=(0.5,) * 3, std=(0.5,) * 3)
    finally:
        dd._score_tensor = orig
    assert isinstance(img, Image.Image) and img.size == (512, 512)
    assert calls["n"] >= 3            # guidance evaluated the smash model each step
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_dream_diffusion.py::test_xhat_generate_returns_image_and_uses_model -q`
Expected: FAIL (`XHatGuidance` not defined).

- [ ] **Step 3: Append XHatGuidance to model/dream_diffusion.py**

```python
# model/dream_diffusion.py  (append)
def _embed(pipe, prompt):
    """Concatenated [uncond, cond] text embeddings for classifier-free guidance."""
    prompt_embeds, neg_embeds = pipe.encode_prompt(prompt, pipe.device, 1, True, negative_prompt="")
    return torch.cat([neg_embeds, prompt_embeds]).to(next(pipe.unet.parameters()).dtype)


def _decode01(pipe, latents):
    """VAE-decode latents -> image in [0,1]."""
    img = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    return (img / 2 + 0.5).clamp(0, 1)


class XHatGuidance(GuidanceStrategy):
    """Universal guidance: each DDIM step, nudge the latent by the gradient of the
    smash loss evaluated on the predicted clean image x0-hat."""
    def __init__(self, steps, guidance_scale, sd_guidance_scale):
        self.steps, self.guidance_scale, self.sd = steps, guidance_scale, sd_guidance_scale

    def generate(self, pipe, smash_model, prompt, target, seed, mean, std):
        dev = pipe.device
        dtype = next(pipe.unet.parameters()).dtype
        emb = _embed(pipe, prompt)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(1, 4, 64, 64, generator=gen).to(dev, dtype)
        latents = latents * pipe.scheduler.init_noise_sigma
        pipe.scheduler.set_timesteps(self.steps, device=dev)
        res = _smash_res(smash_model)
        for t in pipe.scheduler.timesteps:
            latents = latents.detach().requires_grad_(True)
            lat_in = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
            noise = pipe.unet(lat_in, t, encoder_hidden_states=emb).sample
            n_unc, n_cond = noise.chunk(2)
            noise = n_unc + self.sd * (n_cond - n_unc)
            a = pipe.scheduler.alphas_cumprod[int(t)].to(dev, dtype)
            x0 = (latents - (1 - a).sqrt() * noise) / a.sqrt()
            img01 = _decode01(pipe, x0)
            loss = ((_score_tensor(smash_model, img01, mean, std, res) - target) ** 2).mean()
            grad = torch.autograd.grad(loss, latents)[0]
            with torch.no_grad():
                latents = pipe.scheduler.step(noise, t, latents).prev_sample
                latents = latents - self.guidance_scale * grad
        with torch.no_grad():
            return to_pil(_decode01(pipe, latents))


def _smash_res(smash_model):
    """Model input resolution from its timm data config."""
    return int(smash_model.data_config["input_size"][-1])
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_dream_diffusion.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add model/dream_diffusion.py tests/test_dream_diffusion.py
git commit -m "feat: xhat (universal x0) diffusion guidance strategy"
```

---

## Task 4: DoodlGuidance (initial-latent optimization) + wiring test

**Files:**
- Modify: `model/dream_diffusion.py`
- Test: `tests/test_dream_diffusion.py`

- [ ] **Step 1: Append the wiring test**

```python
# tests/test_dream_diffusion.py  (append)
def test_doodl_generate_returns_image():
    pipe = _FakePipe()
    m = _tiny_model()
    img = dd.DoodlGuidance(steps=3, guidance_scale=10.0, sd_guidance_scale=7.5).generate(
        pipe, m, "a creature", target=1.0, seed=1, mean=(0.5,) * 3, std=(0.5,) * 3)
    assert isinstance(img, Image.Image) and img.size == (512, 512)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_dream_diffusion.py::test_doodl_generate_returns_image -q`
Expected: FAIL (`DoodlGuidance` not defined).

- [ ] **Step 3: Append DoodlGuidance**

```python
# model/dream_diffusion.py  (append)
class DoodlGuidance(GuidanceStrategy):
    """Optimize the initial noise latent so the final decoded image hits the
    target, backpropagating through a short deterministic DDIM chain."""
    def __init__(self, steps, guidance_scale, sd_guidance_scale, opt_iters=2, lr=0.1):
        self.steps, self.sd = steps, sd_guidance_scale
        self.opt_iters, self.lr = opt_iters, lr

    def _sample(self, pipe, emb, latents, dtype, dev):
        pipe.scheduler.set_timesteps(self.steps, device=dev)
        for t in pipe.scheduler.timesteps:
            lat_in = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
            noise = pipe.unet(lat_in, t, encoder_hidden_states=emb).sample
            n_unc, n_cond = noise.chunk(2)
            noise = n_unc + self.sd * (n_cond - n_unc)
            latents = pipe.scheduler.step(noise, t, latents).prev_sample
        return latents

    def generate(self, pipe, smash_model, prompt, target, seed, mean, std):
        dev = pipe.device
        dtype = next(pipe.unet.parameters()).dtype
        emb = _embed(pipe, prompt)
        res = _smash_res(smash_model)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        z0 = torch.randn(1, 4, 64, 64, generator=gen).to(dev, dtype) * pipe.scheduler.init_noise_sigma
        z0 = z0.detach().requires_grad_(True)
        opt = torch.optim.Adam([z0], lr=self.lr)
        for _ in range(self.opt_iters):
            opt.zero_grad()
            latents = self._sample(pipe, emb, z0, dtype, dev)
            img01 = _decode01(pipe, latents)
            loss = ((_score_tensor(smash_model, img01, mean, std, res) - target) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return to_pil(_decode01(pipe, self._sample(pipe, emb, z0.detach(), dtype, dev)))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_dream_diffusion.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add model/dream_diffusion.py tests/test_dream_diffusion.py
git commit -m "feat: doodl (initial-latent optimization) diffusion guidance strategy"
```

---

## Task 5: SdsGuidance (score distillation) + wiring test

**Files:**
- Modify: `model/dream_diffusion.py`
- Test: `tests/test_dream_diffusion.py`

- [ ] **Step 1: Append the wiring test**

```python
# tests/test_dream_diffusion.py  (append)
def test_sds_generate_returns_image():
    pipe = _FakePipe()
    m = _tiny_model()
    img = dd.SdsGuidance(steps=4, guidance_scale=10.0, sd_guidance_scale=7.5).generate(
        pipe, m, "a creature", target=1.0, seed=2, mean=(0.5,) * 3, std=(0.5,) * 3)
    assert isinstance(img, Image.Image) and img.size == (512, 512)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_dream_diffusion.py::test_sds_generate_returns_image -q`
Expected: FAIL (`SdsGuidance` not defined).

- [ ] **Step 3: Append SdsGuidance**

```python
# model/dream_diffusion.py  (append)
class SdsGuidance(GuidanceStrategy):
    """Score-distillation: optimize a latent with the SD prior's SDS gradient plus
    the smash-loss gradient (DreamFusion-style, in latent space)."""
    def __init__(self, steps, guidance_scale, sd_guidance_scale, sds_weight=1.0):
        self.iters, self.guidance_scale, self.sd = steps, guidance_scale, sd_guidance_scale
        self.sds_weight = sds_weight

    def generate(self, pipe, smash_model, prompt, target, seed, mean, std):
        dev = pipe.device
        dtype = next(pipe.unet.parameters()).dtype
        emb = _embed(pipe, prompt)
        res = _smash_res(smash_model)
        n_train = pipe.scheduler.config.num_train_timesteps
        gen = torch.Generator(device="cpu").manual_seed(seed)
        latents = (torch.randn(1, 4, 64, 64, generator=gen).to(dev, dtype)
                   * pipe.scheduler.init_noise_sigma).detach().requires_grad_(True)
        opt = torch.optim.Adam([latents], lr=0.05)
        for i in range(self.iters):
            opt.zero_grad()
            # --- SDS gradient (no grad through UNet) ---
            with torch.no_grad():
                tg = torch.randint(20, n_train - 20, (1,), generator=gen).item()
                a = pipe.scheduler.alphas_cumprod[tg].to(dev, dtype)
                noise = torch.randn(latents.shape, generator=gen).to(dev, dtype)
                noisy = a.sqrt() * latents + (1 - a).sqrt() * noise
                pred = pipe.unet(torch.cat([noisy] * 2), tg, encoder_hidden_states=emb).sample
                n_unc, n_cond = pred.chunk(2)
                pred = n_unc + self.sd * (n_cond - n_unc)
                sds_grad = self.sds_weight * (1 - a) * (pred - noise)
            latents.backward(sds_grad, retain_graph=True)
            # --- smash-score gradient (through VAE) ---
            img01 = _decode01(pipe, latents)
            loss = ((_score_tensor(smash_model, img01, mean, std, res) - target) ** 2).mean()
            (self.guidance_scale * loss).backward()
            opt.step()
        with torch.no_grad():
            return to_pil(_decode01(pipe, latents.detach()))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_dream_diffusion.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add model/dream_diffusion.py tests/test_dream_diffusion.py
git commit -m "feat: sds (score distillation) diffusion guidance strategy"
```

---

## Task 6: CLI + full suite + manual GPU smoke

**Files:**
- Modify: `model/dream_diffusion.py`
- Test: full suite

- [ ] **Step 1: Append the CLI**

```python
# model/dream_diffusion.py  (append)
def main(argv=None):
    p = argparse.ArgumentParser(description="Diffusion dreaming: plausible score-targeted images.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="results/dream_diffusion")
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--methods", default="xhat,doodl,sds")
    p.add_argument("--brackets", default="1.0,0.8,0.6,0.4,0.2,0.0")
    p.add_argument("--n-per", type=int, default=4)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=200.0)
    p.add_argument("--sd-guidance-scale", type=float, default=7.5)
    p.add_argument("--seed0", type=int, default=0)
    args = p.parse_args(argv)
    run(args.checkpoint, out_dir=args.out, device=args.device, model_id=args.model_id,
        methods=args.methods.split(","),
        brackets=[float(b) for b in args.brackets.split(",")],
        n_per=args.n_per, steps=args.steps, guidance_scale=args.guidance_scale,
        sd_guidance_scale=args.sd_guidance_scale, seed0=args.seed0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (all tests green, including the 8 dream_diffusion tests).

- [ ] **Step 3: Verify CLI help**

Run: `uv run python -m model.dream_diffusion --help`
Expected: prints usage, no error.

- [ ] **Step 4: Manual GPU smoke (document result, do not add to suite)**

This needs the GPU + the ~4 GB SD1.5 download. Run a tiny real generation to confirm the SD path works end to end (one method, one prompt, two brackets, 1 image each):

```bash
uv run python -m model.dream_diffusion \
  --checkpoint runs/vit_small_mixed_v1/checkpoints/best.pt \
  --methods xhat --brackets 1.0,0.0 --n-per 1 --steps 20 \
  --out results/dream_smoke --device cuda
```
Expected: writes `results/dream_smoke/xhat/{uncond,creature,anthro}/bracket_{1.0,0.0}/000.png`,
per-cell grids, and `manifest.csv`. Confirm the high-bracket images score higher
(in manifest `achieved_calibrated_pct`) than the low-bracket ones. If `guidance_scale`
is too weak/strong (no effect / artifacts), tune it; if `encode_prompt` raises on the
installed diffusers version, adjust the `_embed` call to that version's signature.
Report the outcome; do NOT commit `results/`.

- [ ] **Step 5: Commit**

```bash
git add model/dream_diffusion.py
git commit -m "feat: diffusion dreaming CLI"
```

---

## Self-Review

**Spec coverage:**
- Frozen SD1.5 prior + frozen smash model, shared differentiable loss (VAE-decode -> resize 224 -> normalize -> sigmoid) -> Task 1 (`_score_tensor`/`smash_loss`) + Task 3 (`_decode01`). ✓
- Three pluggable strategies xhat/doodl/sds (GuidanceStrategy ABC) -> Tasks 2-5. ✓
- Three conditioning prompts (uncond + creature + anthro), configurable -> Task 2 (`DEFAULT_PROMPTS`). ✓
- Reverse bracket sweep 1.0->0.0, configurable -> Task 2 (`DEFAULT_BRACKETS`). ✓
- Generate n_per per cell, score each with the calibrated model, label + grid + manifest -> Task 2 (`run`). ✓
- model_id default `sd-legacy/stable-diffusion-v1-5`, fp16 cuda, safety checker off, frozen -> Task 2 (`load_pipeline`). ✓
- CLI with all knobs -> Task 6. ✓
- Tests: pure helpers (Task 1), run orchestration via stub (Task 2), strategy wiring via fake pipe (Tasks 3-5), manual GPU smoke (Task 6). ✓
- Deps already added to uv/pyproject (prior step). ✓

**Placeholder scan:** none — every code step is complete. The only "fill-in-on-GPU" note is the manual smoke step (guidance_scale tuning / diffusers version), which is inherent to research SD code, not a plan placeholder.

**Type consistency:** `_score_tensor(model, img01, mean, std, resolution) -> [B]`; `smash_loss(...)`; `to_pil(img01)`; `GuidanceStrategy.generate(pipe, smash_model, prompt, target, seed, mean, std) -> Image`; `score_pil(model, cfg, pil, calib, device) -> (raw,cal)`; `load_calibration(ckpt, fit="combined")` returns `(xs,ys)`; `_embed`/`_decode01`/`_smash_res` shared by all strategies; `run(..., pipe=None, strategies=None)` injectable. Consistent across tasks.

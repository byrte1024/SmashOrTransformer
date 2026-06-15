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
    """Model input resolution from the backbone's actual patch embed size.

    timm's resolve_model_data_config returns the canonical backbone size (e.g.
    224 for vit_tiny_patch16_224) even when the model was built with a custom
    img_size, so we read it from backbone.patch_embed.img_size instead.
    """
    return int(smash_model.backbone.patch_embed.img_size[0])

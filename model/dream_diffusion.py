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

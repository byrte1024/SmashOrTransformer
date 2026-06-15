"""Activation maximization: synthesize the model's "ideal" Pokemon for each
score bracket using several feature-visualization techniques.

For each target bracket in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} we optimize an input
image so sigmoid(model(x)) -> target, producing 10 images per bracket. Slots
cycle through techniques (Olah et al., Distill 2017 / MACO 2023):

  pixel_tv   pixel params + total-variation + L2          (from noise)
  robust     pixel params + transform robustness (the key trick, from noise)
  blur       pixel params + periodic Gaussian blur        (from noise)
  fourier    Fourier-parameterized + transform robustness (from noise)
  maco       phase-only Fourier, fixed 1/f magnitude      (from noise; good for ViT)
  deepdream  start from a real Pokemon, nudge to target   (from real seed)

Each saved image is labeled with the achieved raw score and the calibrated
"correction" (what that raw score actually maps to in true crowd %).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from data_prep.prepare import load_sprite
from .dataset import canonical_render
from .infer import load_model
from .calibrate import apply_calibration

BRACKETS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TECHNIQUES = ("pixel_tv", "robust", "blur", "fourier", "maco", "deepdream")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _normalize(img01, mean, std):
    m = torch.tensor(mean, device=img01.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=img01.device).view(1, 3, 1, 1)
    return (img01 - m) / s


def _tv(img01):
    dx = (img01[:, :, :, 1:] - img01[:, :, :, :-1]).abs().mean()
    dy = (img01[:, :, 1:, :] - img01[:, :, :-1, :]).abs().mean()
    return dx + dy


def _gaussian_kernel(sigma, device):
    r = max(1, int(3 * sigma))
    xs = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    k = (k / k.sum())
    k2 = (k[:, None] * k[None, :])
    return k2.view(1, 1, 2 * r + 1, 2 * r + 1)


def _blur(img01, sigma):
    k = _gaussian_kernel(sigma, img01.device).repeat(3, 1, 1, 1)
    pad = k.shape[-1] // 2
    return F.conv2d(F.pad(img01, [pad] * 4, mode="reflect"), k, groups=3)


def _random_affine(x, gen, max_rot=12.0, max_scale=0.12, max_jitter=0.08):
    """Robustness transform: random rotate/scale/translate via grid_sample."""
    n, _, h, w = x.shape
    ang = (torch.rand(1, generator=gen, device=x.device) * 2 - 1) * math.radians(max_rot)
    sc = 1.0 + (torch.rand(1, generator=gen, device=x.device) * 2 - 1) * max_scale
    tx = (torch.rand(1, generator=gen, device=x.device) * 2 - 1) * max_jitter
    ty = (torch.rand(1, generator=gen, device=x.device) * 2 - 1) * max_jitter
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)


# --------------------------------------------------------------------------- #
# image parameterizations (each is an nn.Module; .image() -> [1,3,H,W] in [0,1])
# --------------------------------------------------------------------------- #
class PixelParam(torch.nn.Module):
    def __init__(self, res, gen, device, init01=None):
        super().__init__()
        if init01 is None:
            init01 = 0.5 + 0.01 * torch.randn(1, 3, res, res, generator=gen, device=device)
        logit = torch.logit(init01.clamp(1e-3, 1 - 1e-3))
        self.p = torch.nn.Parameter(logit)

    def image(self):
        return torch.sigmoid(self.p)


class FourierParam(torch.nn.Module):
    def __init__(self, res, gen, device, decay=1.0):
        super().__init__()
        wf = res // 2 + 1
        fy = torch.fft.fftfreq(res, device=device)[:, None]
        fx = torch.fft.rfftfreq(res, device=device)[None, :]
        freq = torch.sqrt(fy ** 2 + fx ** 2)
        self.scale = (1.0 / torch.clamp(freq, 1.0 / res) ** decay).view(1, 1, res, wf)
        self.re = torch.nn.Parameter(0.01 * torch.randn(1, 3, res, wf, generator=gen, device=device))
        self.im = torch.nn.Parameter(0.01 * torch.randn(1, 3, res, wf, generator=gen, device=device))
        self.res = res

    def image(self):
        spec = torch.complex(self.re, self.im) * self.scale
        img = torch.fft.irfft2(spec, s=(self.res, self.res))
        img = img / (img.std() + 1e-6)
        return torch.sigmoid(img)


class MacoParam(torch.nn.Module):
    """Phase-only: magnitude fixed to a 1/f natural-image prior (MACO, 2023)."""
    def __init__(self, res, gen, device):
        super().__init__()
        wf = res // 2 + 1
        fy = torch.fft.fftfreq(res, device=device)[:, None]
        fx = torch.fft.rfftfreq(res, device=device)[None, :]
        freq = torch.sqrt(fy ** 2 + fx ** 2)
        self.mag = (1.0 / torch.clamp(freq, 1.0 / res)).view(1, 1, res, wf)
        self.phase = torch.nn.Parameter(
            (torch.rand(1, 3, res, wf, generator=gen, device=device) * 2 - 1) * math.pi)
        self.res = res

    def image(self):
        spec = self.mag * torch.complex(torch.cos(self.phase), torch.sin(self.phase))
        img = torch.fft.irfft2(spec, s=(self.res, self.res))
        img = img / (img.std() + 1e-6)
        return torch.sigmoid(img)


_CONFIG = {
    "pixel_tv":  dict(param="pixel",   transform=False, blur=False, tv=0.10, l2=1e-3),
    "robust":    dict(param="pixel",   transform=True,  blur=False, tv=0.05, l2=0.0),
    "blur":      dict(param="pixel",   transform=False, blur=True,  tv=0.02, l2=0.0),
    "fourier":   dict(param="fourier", transform=True,  blur=False, tv=0.0,  l2=0.0),
    "maco":      dict(param="maco",    transform=True,  blur=False, tv=0.0,  l2=0.0),
    "deepdream": dict(param="pixel",   transform=True,  blur=False, tv=0.05, l2=0.0),
}


def maximize(model, mean, std, target, technique, seed, device, res,
             steps=256, lr=0.05, seed_img01=None):
    """Optimize an input image so sigmoid(model(x)) -> target. Returns a
    (res,res,3) uint8 array and the achieved raw score (float)."""
    cfg = _CONFIG[technique]
    gen = torch.Generator(device=device).manual_seed(seed)
    if cfg["param"] == "pixel":
        param = PixelParam(res, gen, device, init01=seed_img01)
    elif cfg["param"] == "fourier":
        param = FourierParam(res, gen, device)
    else:
        param = MacoParam(res, gen, device)
    param.to(device)
    opt = torch.optim.Adam(param.parameters(), lr=lr)
    t = torch.tensor(float(target), device=device)

    for step in range(steps):
        img01 = param.image()
        x = _normalize(img01, mean, std)
        if cfg["transform"]:
            x = _random_affine(x, gen)
        score = torch.sigmoid(model(x).squeeze())
        loss = (score - t) ** 2
        if cfg["tv"]:
            loss = loss + cfg["tv"] * _tv(img01)
        if cfg["l2"]:
            loss = loss + cfg["l2"] * ((img01 - 0.5) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if cfg["blur"] and step % 16 == 0 and step > 0:
            with torch.no_grad():
                blurred = _blur(param.image(), sigma=1.0)
                param.p.copy_(torch.logit(blurred.clamp(1e-3, 1 - 1e-3)))

    with torch.no_grad():
        img01 = param.image()
        achieved = float(torch.sigmoid(model(_normalize(img01, mean, std)).squeeze()))
        arr = (img01.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return arr, achieved


# --------------------------------------------------------------------------- #
# rendering / driver
# --------------------------------------------------------------------------- #
def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _label(arr, technique, achieved_pct, calibrated_pct, bar_h=40):
    img = Image.fromarray(arr)
    W, H = img.size
    canvas = Image.new("RGB", (W, H + bar_h), (20, 20, 20))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    f = _font(int(bar_h * 0.4))
    draw.text((6, H + 4), f"{technique}", fill=(180, 180, 180), font=f)
    draw.text((6, H + 4 + int(bar_h * 0.45)),
              f"raw {achieved_pct:.0f}% -> {calibrated_pct:.0f}%",
              fill=(240, 240, 240), font=f)
    return canvas


def _real_seeds(images_dir, n, res, gen):
    """Random official-artwork portraits (canonical-rendered, [0,1] tensors)."""
    paths = sorted(Path(images_dir).glob("*/official-artwork.png"))
    if not paths:
        return [None] * n
    idx = torch.randint(0, len(paths), (n,), generator=gen).tolist()
    seeds = []
    for i in idx:
        rgb = canonical_render(load_sprite(paths[i]), res).astype(np.float32) / 255.0
        seeds.append(torch.from_numpy(rgb).permute(2, 0, 1)[None])
    return seeds


def run(checkpoint_path, images_dir="images", device="cuda", out_dir="results/acti_maxim",
        steps=256, res=None, n_per=10, seed0=0) -> Path:
    model, cfg = load_model(checkpoint_path, device=device, pretrained=False)
    dev = next(model.parameters()).device
    res = res or cfg.resolution
    mean, std = model.data_config["mean"], model.data_config["std"]

    cpath = Path(checkpoint_path).parent / "calibration.json"
    cmap = (json.loads(cpath.read_text())["maps"].get("combined")
            if cpath.exists() else None)

    def correct(raw):  # raw fraction -> calibrated "true" fraction
        if cmap is None:
            return raw
        return float(apply_calibration([raw], cmap["xs"], cmap["ys"])[0])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_gen = torch.Generator(device=dev).manual_seed(seed0)
    rows = []

    for b in BRACKETS:
        bdir = out / f"bracket_{b:.1f}"
        bdir.mkdir(exist_ok=True)
        tiles = []
        for slot in tqdm(range(n_per), desc=f"bracket {b:.1f}", unit="img"):
            tech = TECHNIQUES[slot % len(TECHNIQUES)]
            seed = seed0 + b.__hash__() % 1000 + slot
            seed_img = (_real_seeds(images_dir, 1, res, seed_gen)[0]
                        if tech == "deepdream" else None)
            if seed_img is not None:
                seed_img = seed_img.to(dev)
            arr, achieved = maximize(model, mean, std, b, tech, seed, dev, res,
                                     steps=steps, seed_img01=seed_img)
            cal = correct(achieved)
            tile = _label(arr, tech, achieved * 100, cal * 100)
            tile.save(bdir / f"{slot:02d}_{tech}.png")
            tiles.append(tile)
            rows.append({"bracket": b, "slot": slot, "technique": tech,
                         "target_raw": b, "achieved_raw_pct": round(achieved * 100, 2),
                         "calibrated_pct": round(cal * 100, 2)})
        _grid(tiles, out / f"grid_bracket_{b:.1f}.png", target_pct=b * 100,
              target_cal=correct(b) * 100)

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} images across {len(BRACKETS)} brackets to {out}/")
    return out


def _grid(tiles, path, target_pct, target_cal, cols=5):
    if not tiles:
        return
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    header = 34
    grid = Image.new("RGB", (cols * w, header + rows * h), (10, 10, 10))
    d = ImageDraw.Draw(grid)
    d.text((8, 8), f"target raw {target_pct:.0f}%  (true ~{target_cal:.0f}%)",
           fill=(255, 255, 255), font=_font(22))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid.paste(t, (c * w, header + r * h))
    grid.save(path)


def main(argv=None):
    p = argparse.ArgumentParser(description="Activation-maximize ideal Pokemon per score bracket.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", default="images")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/acti_maxim")
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--res", type=int, default=None)
    p.add_argument("--n-per", type=int, default=10)
    args = p.parse_args(argv)
    run(args.checkpoint, images_dir=args.images, device=args.device, out_dir=args.out,
        steps=args.steps, res=args.res, n_per=args.n_per)


if __name__ == "__main__":
    main()

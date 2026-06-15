#!/usr/bin/env python3
"""Slice downloaded battle-background sprite sheets into individual images
under backgrounds/pokemon_battle/. Empty (near-black) cells are skipped.

Sheets handled:
  - usable-battle-background-...webp : flush 5x5 grid (23 filled scenes)
  - 62308.png                       : single column, 34 tiles of 320x240,
                                      separated by 2px gaps
"""
import os
import numpy as np
from PIL import Image

DL = os.path.expanduser("~/Downloads")
OUT = "backgrounds/pokemon_battle"
EMPTY_MEAN = 30  # cells dimmer than this are treated as empty padding


def save_if_content(crop: Image.Image, path: str) -> bool:
    arr = np.asarray(crop.convert("RGB"), dtype=float)
    if arr.mean() < EMPTY_MEAN:
        return False
    crop.save(path)
    return True


def slice_grid(fname: str, cols: int, rows: int, prefix: str) -> int:
    im = Image.open(os.path.join(DL, fname)).convert("RGB")
    W, H = im.size
    xs = np.linspace(0, W, cols + 1).round().astype(int)
    ys = np.linspace(0, H, rows + 1).round().astype(int)
    n = 0
    for r in range(rows):
        for c in range(cols):
            crop = im.crop((xs[c], ys[r], xs[c + 1], ys[r + 1]))
            n += save_if_content(crop, os.path.join(OUT, f"{prefix}_{n:03d}.png"))
    return n


def slice_column(fname: str, tile_h: int, gap: int, prefix: str) -> int:
    im = Image.open(os.path.join(DL, fname)).convert("RGBA")
    W, H = im.size
    step = tile_h + gap
    n = 0
    for k in range((H + gap) // step):
        top = k * step
        crop = im.crop((0, top, W, min(top + tile_h, H)))
        n += save_if_content(crop, os.path.join(OUT, f"{prefix}_{n:03d}.png"))
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    a = slice_grid("usable-battle-background-v0-mlhc8ty8e6ng1.webp",
                   cols=5, rows=5, prefix="scene")
    print(f"grid sheet  -> {a} scenes")
    b = slice_column("62308.png", tile_h=240, gap=2, prefix="platform")
    print(f"column sheet -> {b} platforms")
    print(f"total in {OUT}: {a + b}")


if __name__ == "__main__":
    main()

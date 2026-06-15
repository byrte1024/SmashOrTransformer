"""Evaluate the scraped Safebooru fan-art (images/{id}/booru/) with the
calibrated model.

For each Pokemon that has fetched fan-art it writes:
  results/booru/all_avg/{id}.png  - official portrait + banner with the average
                                    score over the booru images + a side panel
                                    listing each image's score and the spread
  results/booru/matrix/{id}.png   - a contact sheet: every booru image as a tile
                                    labeled with its own SMASH/PASS score
plus per-image (booru_scores.csv) and per-Pokemon (booru_rankings.csv) tables.

Fan-art is a different visual domain than the training portraits, so the default
calibration is the 'val' (generalization) map.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from data_prep.prepare import load_sprite
from .dataset import canonical_render, render_input, to_tensor
from .infer import load_model
from .calibrate import apply_calibration
from .results import annotate_avg, annotate_portrait, _font
from PIL import ImageDraw

_IMG_GLOB = "[0-9]*.*"


def _gather(images_dir):
    """-> list of (pokemon_id, [booru image paths], official_artwork path|None)."""
    out = []
    for folder in sorted(Path(images_dir).iterdir(), key=lambda p: int(p.name)
                         if p.name.isdigit() else 1 << 30):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        booru = folder / "booru"
        if not booru.is_dir():
            continue
        imgs = sorted(booru.glob(_IMG_GLOB))
        if not imgs:
            continue
        oa = folder / "official-artwork.png"
        out.append((int(folder.name), imgs, oa if oa.exists() else None))
    return out


def _score_paths(model, paths, res, mean, std, device, batch_size, stretch=True):
    raws = np.empty(len(paths), dtype=float)
    for start in tqdm(range(0, len(paths), batch_size), desc="scoring", unit="batch"):
        batch = paths[start:start + batch_size]
        x = torch.stack([to_tensor(render_input(load_sprite(p), res, stretch), mean, std)
                         for p in batch]).to(device)
        with torch.no_grad():
            raws[start:start + len(x)] = torch.sigmoid(model(x).reshape(-1)).cpu().numpy()
    return raws


def _matrix(tiles, header, cols=5):
    if not tiles:
        return None
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    head = 30
    grid = Image.new("RGB", (cols * w, head + rows * h), (245, 245, 245))
    ImageDraw.Draw(grid).text((8, 6), header, fill=(20, 20, 20), font=_font(20))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid.paste(t, (c * w, head + r * h))
    return grid


def run(checkpoint_path, images_dir="images", out_dir="results/booru", device="cuda",
        calibration="val", threshold=0.5, display_res=384, tile_res=180,
        names=None, batch_size=64, stretch=True) -> Path:
    model, cfg = load_model(checkpoint_path, device=device, pretrained=False)
    dev = next(model.parameters()).device
    mean, std = model.data_config["mean"], model.data_config["std"]
    cpath = Path(checkpoint_path).parent / "calibration.json"
    cmap = (json.loads(cpath.read_text())["maps"].get(calibration)
            if (calibration != "none" and cpath.exists()) else None)
    name_map = {}
    if names and Path(names).exists():
        with open(names, newline="") as f:
            name_map = {int(r["id"]): r["name"] for r in csv.DictReader(f)}

    entries = _gather(images_dir)
    if not entries:
        raise ValueError(f"no images/*/booru/ folders found under {images_dir!r}")

    flat = [(pid, p) for pid, imgs, _ in entries for p in imgs]
    raw = _score_paths(model, [p for _, p in flat], cfg.resolution, mean, std, dev,
                       batch_size, stretch)
    cal = apply_calibration(raw, cmap["xs"], cmap["ys"]) if cmap else raw
    by_pid = {}
    for (pid, path), c in zip(flat, cal):
        by_pid.setdefault(pid, []).append((path, float(c) * 100))

    out = Path(out_dir)
    (out / "all_avg").mkdir(parents=True, exist_ok=True)
    (out / "matrix").mkdir(parents=True, exist_ok=True)
    thr_pct = threshold * 100
    img_rows, agg_rows = [], []

    for pid, imgs, oa in tqdm(entries, desc="rendering", unit="pkmn"):
        scored = sorted(by_pid[pid], key=lambda t: -t[1])      # (path, pct) high->low
        scores = np.array([s for _, s in scored])
        avg, spread = float(scores.mean()), float(scores.std())
        lo, hi = float(scores.min()), float(scores.max())
        nm = name_map.get(pid, "")
        tag = f"#{pid} {nm}".strip()

        # all_avg: official portrait (fallback to top booru image) + booru sidebar
        portrait_src = oa if oa is not None else scored[0][0]
        portrait = canonical_render(load_sprite(portrait_src), display_res)
        per_image = [(p.name, s) for p, s in scored]
        annotate_avg(portrait, avg, avg >= thr_pct, per_image, spread, lo, hi,
                     thr_pct).save(out / "all_avg" / f"{pid:04d}.png")

        # matrix: each booru image as a labeled tile
        tiles = [annotate_portrait(canonical_render(load_sprite(p), tile_res),
                                   s, s >= thr_pct, banner_h=34) for p, s in scored]
        grid = _matrix(tiles, f"{tag}   avg {avg:.0f}%   spread +/-{spread:.0f}%")
        if grid is not None:
            grid.save(out / "matrix" / f"{pid:04d}.png")

        agg_rows.append({"pokemon_id": pid, "name": nm, "n_booru": len(scored),
                         "avg_pct": round(avg, 2), "spread_pct": round(spread, 2),
                         "min_pct": round(lo, 2), "max_pct": round(hi, 2),
                         "decision": "SMASH" if avg >= thr_pct else "PASS"})

    # per-image rows (raw score looked up per (pid, path))
    raw_by = {(pid, path): rw for (pid, path), rw in zip(flat, raw)}
    for pid, imgs, _ in entries:
        for path, pct in sorted(by_pid[pid], key=lambda t: -t[1]):
            img_rows.append({"pokemon_id": pid, "name": name_map.get(pid, ""),
                             "file": path.name,
                             "raw_pct": round(float(raw_by[(pid, path)]) * 100, 2),
                             "calibrated_pct": round(pct, 2),
                             "decision": "SMASH" if pct >= thr_pct else "PASS"})

    with open(out / "booru_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pokemon_id", "name", "file", "raw_pct",
                                          "calibrated_pct", "decision"])
        w.writeheader(); w.writerows(img_rows)
    agg_rows.sort(key=lambda r: -r["avg_pct"])
    with open(out / "booru_rankings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pokemon_id", "name", "n_booru", "avg_pct",
                                          "spread_pct", "min_pct", "max_pct", "decision"])
        w.writeheader(); w.writerows(agg_rows)

    n_smash = sum(r["decision"] == "SMASH" for r in agg_rows)
    print(f"\nScored {len(img_rows)} booru images across {len(agg_rows)} pokemon | "
          f"SMASH {n_smash} (avg >{thr_pct:.0f}%) | calibration={calibration}")
    print("\nTop 10 by avg booru score:")
    for r in agg_rows[:10]:
        print(f"  {r['avg_pct']:5.1f}%  #{r['pokemon_id']} {r['name']}  (n={r['n_booru']})")
    print(f"\nWrote all_avg/ + matrix/ + 2 CSVs to {out}/")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate scraped booru fan-art per pokemon.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", default="images")
    p.add_argument("--out", default="results/booru")
    p.add_argument("--device", default="cuda")
    p.add_argument("--calibration", default="val",
                   choices=["val", "train", "combined", "none"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--display-res", type=int, default=384)
    p.add_argument("--tile-res", type=int, default=180)
    p.add_argument("--names", default="pokemon_names.csv")
    p.add_argument("--stretch", action=argparse.BooleanOptionalAction, default=True,
                   help="stretch-to-square model input (default) vs aspect-fit (--no-stretch)")
    args = p.parse_args(argv)
    run(args.checkpoint, images_dir=args.images, out_dir=args.out, device=args.device,
        calibration=args.calibration, threshold=args.threshold,
        display_res=args.display_res, tile_res=args.tile_res, names=args.names,
        stretch=args.stretch)


if __name__ == "__main__":
    main()

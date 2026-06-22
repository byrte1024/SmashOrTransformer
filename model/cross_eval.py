"""Cross-evaluate several checkpoints on a common, fair footing.

Each model was trained on its own dataset, but the Pokemon-level val split is
seeded identically, so the held-out Pokemon are the same across runs. Their
*reported* val numbers still aren't comparable: training-time validation
averaged over each dataset's own image set per Pokemon (e.g. portraits has no
booru) and the eval render changed over time.

This scores the SAME images with every model on the intersection of all the
models' val Pokemon (so no model has trained on any evaluated Pokemon), and
reports three views per model:

  portrait    - official-artwork.png only (one canonical image per Pokemon)
  sprites_avg - mean over official + in-game sprites (images/{id}/*.*)
  all_avg     - mean over sprites + booru fan-art (images/{id}/**)

Spearman/Pearson are rank metrics (calibration-invariant); MAE uses each
checkpoint's own 'val' calibration map. Writes results/cross_eval.csv.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from data_prep.prepare import load_sprite
from .dataset import render_input, to_tensor
from .infer import load_model, load_calibration
from .calibrate import apply_calibration
from .metrics import spearman, pearson

_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif"}


def _val_pids(run_dir: Path) -> set[int]:
    cfg = json.loads((run_dir / "config.json").read_text())
    ds = Path(cfg["train_config"]["dataset_dir"])
    pid = np.asarray(np.load(ds / "data.npz", allow_pickle=True)["pokemon_id"])
    rows = json.loads((ds / "split.json").read_text())["val"]
    return {int(pid[i]) for i in rows}


def _images_for(pid: int, images_dir: Path):
    folder = images_dir / str(pid)
    sprites = sorted(f for f in folder.glob("*") if f.suffix.lower() in _EXTS)
    booru = sorted(f for f in (folder / "booru").glob("*") if f.suffix.lower() in _EXTS)
    portrait = folder / "official-artwork.png"
    return (portrait if portrait.exists() else None), sprites, booru


def _score_paths(model, paths, res, mean, std, dev, bs=64) -> dict:
    """path -> raw sigmoid score, scored once each."""
    uniq = list(dict.fromkeys(paths))
    out = {}
    for i in range(0, len(uniq), bs):
        batch = uniq[i:i + bs]
        x = torch.stack([to_tensor(render_input(load_sprite(p), res, True), mean, std)
                         for p in batch]).to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                             enabled=dev.type == "cuda"):
            s = torch.sigmoid(model(x).reshape(-1)).float().cpu().numpy()
        out.update(zip(batch, s))
    return out


def _metrics(pred, true, cmap):
    pred, true = np.asarray(pred), np.asarray(true)
    sp, pe = spearman(pred, true), pearson(pred, true)
    cal = apply_calibration(pred, cmap[0], cmap[1]) if cmap else pred
    mae = float(np.mean(np.abs(cal - true)))
    return sp, pe, mae


def run(run_dirs, images="images", labels="pokesmash_votes.csv",
        out="results/cross_eval.csv", device="cuda") -> Path:
    runs = [Path(r) for r in run_dirs]
    images_dir = Path(images)
    true_pct = {int(r["id"]): float(r["smash_pct"]) / 100.0
                for r in csv.DictReader(open(labels))}

    # common eval set: Pokemon held out by EVERY model (clean for all)
    val_sets = {r.name: _val_pids(r) for r in runs}
    common = sorted(set.intersection(*val_sets.values()))
    print("val pokemon per model:", {k: len(v) for k, v in val_sets.items()})
    print(f"common (eval) set: {len(common)} pokemon held out by all models\n")

    # gather the image lists once (shared across models)
    per_pid = {pid: _images_for(pid, images_dir) for pid in common}

    rows = []
    for r in runs:
        ckpt = r / "checkpoints" / "best.pt"
        model, cfg = load_model(ckpt, device=device, pretrained=False)
        dev = next(model.parameters()).device
        mean, std = model.data_config["mean"], model.data_config["std"]
        cmap = load_calibration(ckpt, fit="val")

        all_paths = []
        for portrait, sprites, booru in per_pid.values():
            if portrait is not None:
                all_paths.append(portrait)
            all_paths.extend(sprites)
            all_paths.extend(booru)
        scores = _score_paths(model, all_paths, cfg.resolution, mean, std, dev)

        views = {"portrait": [], "sprites_avg": [], "all_avg": []}
        truth = []
        for pid in common:
            portrait, sprites, booru = per_pid[pid]
            truth.append(true_pct[pid])
            views["portrait"].append(scores[portrait] if portrait else np.nan)
            views["sprites_avg"].append(np.mean([scores[p] for p in sprites]) if sprites else np.nan)
            allimg = sprites + booru
            views["all_avg"].append(np.mean([scores[p] for p in allimg]) if allimg else np.nan)

        print(f"=== {r.name} ===")
        for view, pred in views.items():
            pred = np.array(pred); m = ~np.isnan(pred)
            sp, pe, mae = _metrics(pred[m], np.array(truth)[m], cmap)
            print(f"  {view:12s}  spearman {sp:.3f}  pearson {pe:.3f}  mae {mae:.3f}  (n={m.sum()})")
            rows.append({"model": r.name, "view": view, "spearman": round(sp, 4),
                         "pearson": round(pe, 4), "mae": round(mae, 4), "n": int(m.sum())})
        print()

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "view", "spearman", "pearson", "mae", "n"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")
    return Path(out)


def main(argv=None):
    p = argparse.ArgumentParser(description="Cross-evaluate checkpoints on a common val set.")
    p.add_argument("runs", nargs="+", help="run directories (each with checkpoints/best.pt)")
    p.add_argument("--images", default="images")
    p.add_argument("--labels", default="pokesmash_votes.csv")
    p.add_argument("--out", default="results/cross_eval.csv")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    run(args.runs, images=args.images, labels=args.labels, out=args.out, device=args.device)


if __name__ == "__main__":
    main()

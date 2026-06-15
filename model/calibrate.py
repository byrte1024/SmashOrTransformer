"""Post-hoc calibration of the smash-ranker's predicted probabilities.

The model is a strong ranker but its outputs are compressed toward the mean
(the high tail is under-predicted). A monotonic isotonic-regression map fit on
predicted->true corrects the numbers WITHOUT changing the ranking (so Spearman
is preserved exactly), improving MAE and making the displayed 0-100 score read
closer to the real crowd percentage.

We fit on each split (train / val / combined) and report all three, then save
every map to `calibration.json` next to the checkpoint. `infer` applies one.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import EvalDataset
from .metrics import spearman, pearson, mae, aggregate_per_pokemon
from .infer import load_model

SPLITS = ("train", "val", "combined")


def fit_isotonic(pred, true) -> tuple[list[float], list[float]]:
    """Monotonic non-decreasing fit (pool-adjacent-violators). Returns (xs, ys)
    knots usable with numpy.interp. Identity if fewer than 2 points."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if len(pred) < 2:
        return [0.0, 1.0], [0.0, 1.0]
    order = np.argsort(pred, kind="stable")
    x = pred[order]
    y = true[order]
    # PAVA: blocks of [mean, weight, count]
    blocks: list[list[float]] = []
    for yi in y:
        blocks.append([float(yi), 1.0, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            w = w1 + w2
            blocks.append([(m1 * w1 + m2 * w2) / w, w, int(c1 + c2)])
    fitted = np.empty(len(y))
    i = 0
    for m, _w, c in blocks:
        fitted[i:i + c] = m
        i += c
    # collapse duplicate x to one knot (take the largest fitted value -> monotone)
    xs, ys = [], []
    for ux in np.unique(x):
        xs.append(float(ux))
        ys.append(float(fitted[x == ux].max()))
    return xs, ys


def apply_calibration(pred, xs, ys) -> np.ndarray:
    """Map predictions through the isotonic knots; clamps outside [0,1]."""
    out = np.interp(np.asarray(pred, dtype=float), xs, ys)
    return np.clip(out, 0.0, 1.0)


def _score_split(model, dataset_dir, split, mean, std, resolution, device,
                 batch_size, num_workers):
    """Per-image predictions for a split: (preds, trues, pokemon_ids, categories).
    Uses parallel workers so the GPU isn't starved decoding hi-res images."""
    cat_all = np.asarray(np.load(Path(dataset_dir) / "data.npz",
                                 allow_pickle=True)["category"])
    ds = EvalDataset(dataset_dir, split, mean, std, resolution)
    cats = cat_all[ds._rows]                     # aligned to dataset (shuffle=False) order
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    model.eval()
    preds, trues, pids = [], [], []
    with torch.no_grad():
        for t, y, pid in tqdm(loader, desc=f"scoring {split}", unit="batch"):
            preds.append(torch.sigmoid(model(t.to(device))).cpu().numpy())
            trues.append(y.numpy())
            pids.append(np.asarray(pid))
    return (np.concatenate(preds), np.concatenate(trues), np.concatenate(pids), cats)


def _per_category(preds, trues, pids, cats) -> dict:
    """Per-pokemon-aggregated metrics split by source kind (sprite vs booru):
    'if you only had this kind of image, how well does the model rank/score?'"""
    out = {}
    for kind, mask in (("sprite", cats != "booru"), ("booru", cats == "booru")):
        if not mask.any():
            continue
        _, mp, tt = aggregate_per_pokemon(pids[mask], preds[mask], trues[mask])
        out[kind] = {"n_images": int(mask.sum()), "n_pokemon": len(tt),
                     "spearman": round(spearman(mp, tt), 4),
                     "pearson": round(pearson(mp, tt), 4),
                     "mae": round(mae(mp, tt), 4)}
    return out


def run(checkpoint_path, dataset_dir=None, device="cuda", batch_size=64,
        num_workers=0) -> Path:
    model, cfg = load_model(checkpoint_path, device=device, pretrained=False)
    dev = next(model.parameters()).device
    dataset_dir = dataset_dir or cfg.dataset_dir
    mean, std = model.data_config["mean"], model.data_config["std"]

    perimg = {sp: _score_split(model, dataset_dir, sp, mean, std, cfg.resolution,
                               dev, batch_size, num_workers)
              for sp in ("train", "val")}

    # per-pokemon predictions (all categories) for fitting the maps
    preds, trues = {}, {}
    for sp in ("train", "val"):
        p, t, ids, _ = perimg[sp]
        _, preds[sp], trues[sp] = aggregate_per_pokemon(ids, p, t)
    preds["combined"] = np.concatenate([preds["train"], preds["val"]])  # splits disjoint
    trues["combined"] = np.concatenate([trues["train"], trues["val"]])

    maps, report = {}, {}
    for fit_sp in SPLITS:
        xs, ys = fit_isotonic(preds[fit_sp], trues[fit_sp])
        maps[fit_sp] = {"xs": xs, "ys": ys}
        per_eval = {}
        for ev_sp in SPLITS:
            cal = apply_calibration(preds[ev_sp], xs, ys)
            per_eval[ev_sp] = {
                "mae_raw": round(mae(preds[ev_sp], trues[ev_sp]), 4),
                "mae_calibrated": round(mae(cal, trues[ev_sp]), 4),
                "spearman": round(spearman(preds[ev_sp], trues[ev_sp]), 4)}
        report[fit_sp] = per_eval

    per_category = {sp: _per_category(*perimg[sp]) for sp in ("val", "train")}

    out = Path(checkpoint_path).parent / "calibration.json"
    out.write_text(json.dumps(
        {"method": "isotonic", "default": "val", "maps": maps, "report": report,
         "per_category": per_category}, indent=2))

    print("\nCalibration report (Spearman is unchanged by monotonic mapping):")
    for fit_sp in SPLITS:
        print(f"\n  fit on {fit_sp}:")
        for ev_sp in SPLITS:
            r = report[fit_sp][ev_sp]
            print(f"    eval {ev_sp:8s}  MAE {r['mae_raw']:.4f} -> "
                  f"{r['mae_calibrated']:.4f}  (spearman {r['spearman']:.3f})")
    print("\nPer-source performance (sprite vs booru, per-pokemon aggregated):")
    for sp in ("val", "train"):
        print(f"  {sp}:")
        for kind, m in per_category[sp].items():
            print(f"    {kind:6s}  spearman {m['spearman']:.3f}  pearson {m['pearson']:.3f}  "
                  f"mae {m['mae']:.4f}   (n_img {m['n_images']}, n_pkmn {m['n_pokemon']})")
    print(f"\nSaved maps (train/val/combined) to {out}; infer applies 'val' by default.")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Fit isotonic calibration on the model.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers for parallel image decode (speeds up scoring)")
    args = p.parse_args(argv)
    run(args.checkpoint, dataset_dir=args.dataset_dir, device=args.device,
        batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()

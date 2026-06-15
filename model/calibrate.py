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
from .dataset import EvalDataset
from .metrics import evaluate, spearman, mae
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


def _predict_split(model, dataset_dir, split, mean, std, resolution, device, batch_size):
    ds = EvalDataset(dataset_dir, split, mean, std, resolution)
    loader = DataLoader(ds, batch_size=batch_size)
    out = evaluate(model, loader, device)
    return np.asarray(out["y_pred"]), np.asarray(out["y_true"])


def run(checkpoint_path, dataset_dir=None, device="cuda", batch_size=64) -> Path:
    model, cfg = load_model(checkpoint_path, device=device, pretrained=False)
    dev = next(model.parameters()).device
    dataset_dir = dataset_dir or cfg.dataset_dir
    mean, std = model.data_config["mean"], model.data_config["std"]

    # per-pokemon predictions for each base split
    preds, trues = {}, {}
    for sp in ("train", "val"):
        preds[sp], trues[sp] = _predict_split(
            model, dataset_dir, sp, mean, std, cfg.resolution, dev, batch_size)
    preds["combined"] = np.concatenate([preds["train"], preds["val"]])
    trues["combined"] = np.concatenate([trues["train"], trues["val"]])

    # fit a map on each split, evaluate every map on every split
    maps, report = {}, {}
    for fit_sp in SPLITS:
        xs, ys = fit_isotonic(preds[fit_sp], trues[fit_sp])
        maps[fit_sp] = {"xs": xs, "ys": ys}
        per_eval = {}
        for ev_sp in SPLITS:
            raw_mae = mae(preds[ev_sp], trues[ev_sp])
            cal = apply_calibration(preds[ev_sp], xs, ys)
            per_eval[ev_sp] = {
                "mae_raw": round(raw_mae, 4),
                "mae_calibrated": round(mae(cal, trues[ev_sp]), 4),
                "spearman": round(spearman(preds[ev_sp], trues[ev_sp]), 4)}
        report[fit_sp] = per_eval

    out = Path(checkpoint_path).parent / "calibration.json"
    out.write_text(json.dumps(
        {"method": "isotonic", "default": "val", "maps": maps, "report": report},
        indent=2))

    print(f"\nCalibration report (Spearman is unchanged by monotonic mapping):")
    for fit_sp in SPLITS:
        print(f"\n  fit on {fit_sp}:")
        for ev_sp in SPLITS:
            r = report[fit_sp][ev_sp]
            print(f"    eval {ev_sp:8s}  MAE {r['mae_raw']:.4f} -> "
                  f"{r['mae_calibrated']:.4f}  (spearman {r['spearman']:.3f})")
    print(f"\nSaved maps (train/val/combined) to {out}; infer applies '{maps and 'val'}' by default.")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Fit isotonic calibration on the model.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    run(args.checkpoint, dataset_dir=args.dataset_dir, device=args.device,
        batch_size=args.batch_size)


if __name__ == "__main__":
    main()

"""Score every Pokemon with the (calibrated) model and render labeled portraits.

Two variations are written:
  results/portrait/{id}.png  - official artwork, score from the PORTRAIT only
  results/all_avg/{id}.png   - official artwork, score AVERAGED over all the
                               Pokemon's images, with a side panel listing each
                               image's score and the spread between them.

Also writes:
  results/rankings.csv        - per-Pokemon aggregate (true / portrait / all-avg
                                %, spread, range, n_images, split, decisions)
  results/per_image_scores.csv- one row per image (raw + calibrated %)

A Pokemon is SMASH when its calibrated predicted crowd-smash exceeds the
threshold (default 50%), else PASS.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from .dataset import canonical_render, to_tensor
from .metrics import spearman
from .infer import load_model
from .calibrate import apply_calibration

GREEN = (40, 170, 70)
RED = (200, 60, 60)
DARK = (60, 60, 60)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # older Pillow without size arg
        return ImageFont.load_default()


def _banner(canvas, x, y, w, h, text, smash):
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x, y, x + w, y + h], fill=(GREEN if smash else RED))
    font = _font(int(h * 0.5))
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((x + (w - tw) // 2, y + (h - th) // 2 - tb[1]), text,
              fill=(255, 255, 255), font=font)


def annotate_portrait(portrait_rgb, score_pct, smash, banner_h=60):
    img = Image.fromarray(portrait_rgb)
    W, H = img.size
    canvas = Image.new("RGB", (W, H + banner_h), (255, 255, 255))
    canvas.paste(img, (0, banner_h))
    _banner(canvas, 0, 0, W, banner_h,
            f"{'SMASH' if smash else 'PASS'}  {score_pct:.0f}%", smash)
    return canvas


def annotate_avg(portrait_rgb, avg_pct, smash, per_image, spread_pct, lo_pct,
                 hi_pct, threshold_pct, banner_h=60, sidebar_w=300):
    """per_image: list of (name, score_pct) sorted high->low."""
    img = Image.fromarray(portrait_rgb)
    D, _ = img.size
    H = banner_h + D
    canvas = Image.new("RGB", (D + sidebar_w, H), (255, 255, 255))
    canvas.paste(img, (0, banner_h))
    _banner(canvas, 0, 0, D, banner_h,
            f"{'SMASH' if smash else 'PASS'}  {avg_pct:.0f}%", smash)

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([D, 0, D + sidebar_w - 1, H - 1], outline=(200, 200, 200))
    hf, lf = _font(17), _font(15)
    x = D + 12
    draw.text((x, 8), "per-image", fill=DARK, font=hf)
    y, line_h = 34, 20
    max_lines = (H - 90 - 34) // line_h
    shown = per_image[:max_lines]
    for name, sc in shown:
        col = GREEN if sc >= threshold_pct else DARK
        draw.text((x, y), name[:22], fill=col, font=lf)
        draw.text((D + sidebar_w - 46, y), f"{sc:.0f}%", fill=col, font=lf)
        y += line_h
    if len(per_image) > len(shown):
        draw.text((x, y), f"... +{len(per_image) - len(shown)} more", fill=DARK, font=lf)
        y += line_h
    fy = H - 76
    draw.line([D + 8, fy - 6, D + sidebar_w - 8, fy - 6], fill=(200, 200, 200))
    draw.text((x, fy), f"avg {avg_pct:.0f}%", fill=DARK, font=hf)
    draw.text((x, fy + 22), f"spread +/-{spread_pct:.0f}%", fill=DARK, font=lf)
    draw.text((x, fy + 42), f"range {lo_pct:.0f}-{hi_pct:.0f}%", fill=DARK, font=lf)
    return canvas


def _score_rows(model, images, res, mean, std, device, batch_size):
    preds = np.empty(len(images), dtype=float)
    for start in tqdm(range(0, len(images), batch_size), desc="scoring", unit="batch"):
        rows = range(start, min(len(images), start + batch_size))
        x = torch.stack([to_tensor(canonical_render(images[r], res), mean, std)
                         for r in rows]).to(device)
        with torch.no_grad():
            preds[start:start + len(x)] = torch.sigmoid(model(x)).cpu().numpy()
    return preds


def _calibrate_per_split(raw, pid, val_set, maps, mode):
    """Return calibrated predictions. 'per-split' uses the train map for train
    Pokemon and the val map for val Pokemon (each map only covers the regime it
    was fit on -- applying one global map to the other regime saturates at its
    domain edges). Other modes apply a single named map; 'none' is identity."""
    if maps is None or mode == "none":
        return raw.copy()
    if mode == "per-split":
        is_val = np.array([int(p) in val_set for p in pid])
        cal = raw.copy()
        cal[is_val] = apply_calibration(raw[is_val], maps["val"]["xs"], maps["val"]["ys"])
        cal[~is_val] = apply_calibration(raw[~is_val], maps["train"]["xs"], maps["train"]["ys"])
        return cal
    return apply_calibration(raw, maps[mode]["xs"], maps[mode]["ys"])


def run(checkpoint_path, dataset_dir=None, device="cuda", out_dir="results",
        threshold=0.5, display_res=384, batch_size=64, calibration="per-split") -> Path:
    model, cfg = load_model(checkpoint_path, device=device, pretrained=False)
    dev = next(model.parameters()).device
    dataset_dir = dataset_dir or cfg.dataset_dir
    mean, std = model.data_config["mean"], model.data_config["std"]
    cpath = Path(checkpoint_path).parent / "calibration.json"
    maps = json.loads(cpath.read_text())["maps"] if cpath.exists() else None

    d = np.load(Path(dataset_dir) / "data.npz", allow_pickle=True)
    images, pid, name, smash = (d["images"], d["pokemon_id"], d["source_name"],
                                d["smash_pct"])
    val_set = set(int(pid[i]) for i in
                  json.loads((Path(dataset_dir) / "split.json").read_text())["val"])

    raw = _score_rows(model, images, cfg.resolution, mean, std, dev, batch_size)
    cal = _calibrate_per_split(raw, pid, val_set, maps, calibration)   # per-image [0,1]

    out = Path(out_dir)
    (out / "portrait").mkdir(parents=True, exist_ok=True)
    (out / "all_avg").mkdir(parents=True, exist_ok=True)
    thr_pct = threshold * 100
    rank_rows, per_image_rows = [], []

    for u in tqdm(np.unique(pid), desc="rendering", unit="pkmn"):
        rows = np.where(pid == u)[0]
        oa = np.where((pid == u) & (name == "official-artwork"))[0]
        disp_row = int(oa[0]) if len(oa) else int(rows[0])
        portrait_rgb = canonical_render(images[disp_row], display_res)

        per_img = sorted(((str(name[r]), float(cal[r]) * 100) for r in rows),
                         key=lambda t: -t[1])
        scores = np.array([s for _, s in per_img])
        avg = float(scores.mean()); spread = float(scores.std())
        lo, hi = float(scores.min()), float(scores.max())
        portrait_pct = float(cal[disp_row]) * 100
        true_pct = float(smash[disp_row]) * 100
        split = "val" if int(u) in val_set else "train"

        annotate_portrait(portrait_rgb, portrait_pct, portrait_pct >= thr_pct
                          ).save(out / "portrait" / f"{int(u):04d}.png")
        annotate_avg(portrait_rgb, avg, avg >= thr_pct, per_img, spread, lo, hi,
                     thr_pct).save(out / "all_avg" / f"{int(u):04d}.png")

        rank_rows.append({"pokemon_id": int(u), "split": split,
                          "true_pct": round(true_pct, 2),
                          "portrait_pct": round(portrait_pct, 2),
                          "allavg_pct": round(avg, 2), "spread_pct": round(spread, 2),
                          "min_pct": round(lo, 2), "max_pct": round(hi, 2),
                          "n_images": len(rows),
                          "portrait_decision": "SMASH" if portrait_pct >= thr_pct else "PASS",
                          "allavg_decision": "SMASH" if avg >= thr_pct else "PASS"})
        for r in rows:
            per_image_rows.append({"pokemon_id": int(u), "source_name": str(name[r]),
                                   "raw_pct": round(float(raw[r]) * 100, 2),
                                   "calibrated_pct": round(float(cal[r]) * 100, 2),
                                   "smash": bool(float(cal[r]) * 100 >= thr_pct)})

    with open(out / "rankings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
        w.writeheader(); w.writerows(rank_rows)
    with open(out / "per_image_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pokemon_id", "source_name", "raw_pct",
                                          "calibrated_pct", "smash"])
        w.writeheader(); w.writerows(per_image_rows)

    _report(rank_rows, val_set, thr_pct)
    print(f"\nWrote {len(rank_rows)} portraits x2 + rankings.csv + "
          f"per_image_scores.csv to {out}/")
    return out


def _report(rows, val_set, thr_pct):
    by_pred = sorted(rows, key=lambda r: -r["allavg_pct"])
    by_true = sorted(rows, key=lambda r: -r["true_pct"])
    pred = np.array([r["allavg_pct"] for r in rows])
    true = np.array([r["true_pct"] for r in rows])
    vmask = np.array([r["split"] == "val" for r in rows])
    print(f"\nSpearman (all-avg vs true): full {spearman(pred, true):.3f}  "
          f"val-only {spearman(pred[vmask], true[vmask]):.3f}")
    n_smash = sum(r["allavg_decision"] == "SMASH" for r in rows)
    print(f"AI SMASH verdicts (all-avg, >{thr_pct:.0f}%): {n_smash}/{len(rows)}")
    print("\nAI's top 10 (all-avg %  /  true %):")
    for r in by_pred[:10]:
        print(f"  #{r['pokemon_id']:4d}  AI {r['allavg_pct']:5.1f}  true {r['true_pct']:5.1f}  [{r['split']}]")
    print("\nActually-best top 10 (true %  /  AI %):")
    for r in by_true[:10]:
        print(f"  #{r['pokemon_id']:4d}  true {r['true_pct']:5.1f}  AI {r['allavg_pct']:5.1f}  [{r['split']}]")


def main(argv=None):
    p = argparse.ArgumentParser(description="Score all Pokemon and render labeled portraits.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--display-res", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--calibration", default="per-split",
                   choices=["per-split", "val", "train", "combined", "none"],
                   help="per-split (default): train map for train mons, val map "
                        "for val mons -- avoids the domain-saturation artifact")
    args = p.parse_args(argv)
    run(args.checkpoint, dataset_dir=args.dataset_dir, device=args.device,
        out_dir=args.out, threshold=args.threshold, display_res=args.display_res,
        batch_size=args.batch_size, calibration=args.calibration)


if __name__ == "__main__":
    main()

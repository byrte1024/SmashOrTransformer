"""Score a prepared dataset with a model and aggregate each Pokemon's
calibrated score by image source, so source disagreements are scannable.

For every Pokemon it computes the per-Pokemon average score over:
  portrait  - portrait images only (official-artwork / home ...)
  ingame    - in-game sprites only
  sprite    - portrait + in-game (all non-booru)
  booru     - booru fan-art only
  all       - everything available (incl. booru)

and a `disagreement` = max-min across {portrait, ingame, all}. The CSV is sorted
by disagreement (biggest first) so you can scan where sources diverge. Also
prints val Spearman-vs-true for each grouping (which source ranks best).

Scoring uses canonical (non-augmented) renders + per-split calibration, the same
honest eval view used by calibration. Reuses the parallel per-image scorer.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .infer import load_model
from .calibrate import _score_split
from .results import _calibrate_per_split
from .metrics import spearman

_GROUP_COLS = ("portrait_pct", "ingame_pct", "sprite_pct", "booru_pct", "all_pct")


def run(checkpoint_path, dataset_dir=None, out_dir="results", device="cuda",
        calibration="per-split", batch_size=64, num_workers=0, names=None,
        threshold=0.5, top=20) -> Path:
    model, cfg = load_model(checkpoint_path, device=device, pretrained=False)
    dev = next(model.parameters()).device
    dataset_dir = dataset_dir or cfg.dataset_dir
    mean, std = model.data_config["mean"], model.data_config["std"]

    perimg = {sp: _score_split(model, dataset_dir, sp, mean, std, cfg.resolution,
                               dev, batch_size, num_workers) for sp in ("train", "val")}
    preds = np.concatenate([perimg[s][0] for s in ("train", "val")])
    trues = np.concatenate([perimg[s][1] for s in ("train", "val")])
    pids = np.concatenate([perimg[s][2] for s in ("train", "val")])
    cats = np.concatenate([perimg[s][3] for s in ("train", "val")])
    val_set = set(int(x) for x in perimg["val"][2])

    cpath = Path(checkpoint_path).parent / "calibration.json"
    maps = (json.loads(cpath.read_text())["maps"]
            if (calibration != "none" and cpath.exists()) else None)
    cal = _calibrate_per_split(preds, pids, val_set, maps, calibration)  # [0,1] per image

    name_map = {}
    if names and Path(names).exists():
        with open(names, newline="") as f:
            name_map = {int(r["id"]): r["name"] for r in csv.DictReader(f)}

    thr = threshold * 100
    rows = []
    for u in np.unique(pids):
        m = pids == u

        def avg(mask):
            return round(float(cal[mask].mean()) * 100, 2) if mask.any() else ""

        port = avg(m & (cats == "portrait"))
        game = avg(m & (cats == "in-game"))
        booru = avg(m & (cats == "booru"))
        sprite = avg(m & (cats != "booru"))
        allv = round(float(cal[m].mean()) * 100, 2)
        variants = [v for v in (port, game, allv) if v != ""]
        disagree = round(max(variants) - min(variants), 2) if len(variants) > 1 else 0.0
        rows.append({"pokemon_id": int(u), "name": name_map.get(int(u), ""),
                     "split": "val" if int(u) in val_set else "train",
                     "true_pct": round(float(trues[m][0]) * 100, 2),
                     "portrait_pct": port, "ingame_pct": game, "sprite_pct": sprite,
                     "booru_pct": booru, "all_pct": allv, "disagreement": disagree,
                     "decision_portrait": "SMASH" if port != "" and port >= thr else "PASS",
                     "decision_all": "SMASH" if allv >= thr else "PASS"})
    rows.sort(key=lambda r: -r["disagreement"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "source_comparison.csv"
    fields = ["pokemon_id", "name", "split", "true_pct", "portrait_pct", "ingame_pct",
              "sprite_pct", "booru_pct", "all_pct", "disagreement",
              "decision_portrait", "decision_all"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    print("\nVal Spearman vs true crowd %, by source grouping:")
    val_rows = [r for r in rows if r["split"] == "val"]
    for col in _GROUP_COLS:
        pairs = [(r[col], r["true_pct"]) for r in val_rows if r[col] != ""]
        if len(pairs) > 1:
            sp = spearman(np.array([p for p, _ in pairs]), np.array([t for _, t in pairs]))
            print(f"  {col:13s} spearman {sp:.3f}  (n={len(pairs)})")
    print(f"\nTop {top} source disagreements (max-min across portrait/ingame/all):")
    for r in rows[:top]:
        print(f"  #{r['pokemon_id']:4d} {(r['name'] or '')[:14]:14s} true {r['true_pct']:5.1f}  "
              f"portrait {r['portrait_pct']}  ingame {r['ingame_pct']}  all {r['all_pct']}  "
              f"(disagree {r['disagreement']}) [{r['split']}]")
    print(f"\nWrote {path}  ({len(rows)} pokemon)")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare per-Pokemon scores across image sources.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--calibration", default="per-split",
                   choices=["per-split", "val", "train", "combined", "none"])
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--names", default="pokemon_names.csv")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args(argv)
    run(args.checkpoint, dataset_dir=args.dataset_dir, out_dir=args.out, device=args.device,
        calibration=args.calibration, batch_size=args.batch_size,
        num_workers=args.num_workers, names=args.names, threshold=args.threshold, top=args.top)


if __name__ == "__main__":
    main()

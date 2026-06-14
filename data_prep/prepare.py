from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageSequence
from .config import DataConfig
from .selection import load_labels, load_records, select_pokemon
from .splits import build_split


def load_sprite(path) -> np.ndarray:
    """Load an image as uint8 RGBA at native size; first frame for animations."""
    im = Image.open(path)
    if getattr(im, "is_animated", False):
        im = next(ImageSequence.Iterator(im)).copy()
    return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def prepare(cfg: DataConfig, images_dir, labels_csv, out_root) -> Path:
    labels = load_labels(labels_csv)
    out = Path(out_root) / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    images, pid_arr, cat_arr, gen_arr, name_arr = [], [], [], [], []
    smash_arr, votes_arr = [], []
    manifest_rows, reports = [], {}

    for pid in sorted(labels.keys()):
        recs = load_records(images_dir, pid, labels)
        if not recs:
            continue
        kept, report = select_pokemon(cfg, recs)
        reports[pid] = report
        for r in kept:
            images.append(load_sprite(r.path))
            pid_arr.append(pid); cat_arr.append(r.category); gen_arr.append(r.gen)
            name_arr.append(r.source_name)
            smash_arr.append(r.smash_pct); votes_arr.append(r.total_votes)
            manifest_rows.append({"pokemon_id": pid, "source_name": r.source_name,
                                  "category": r.category, "gen": r.gen,
                                  "smash_pct": r.smash_pct, "total_votes": r.total_votes})

    if not images:
        raise ValueError(
            f"No images selected for dataset {cfg.name!r}; "
            "check selection filters and that images_dir/labels_csv are correct."
        )

    pid_np = np.array(pid_arr, dtype=np.int64)
    images_obj = np.empty(len(images), dtype=object)
    for i, a in enumerate(images):
        images_obj[i] = a

    np.savez(out / "data.npz",
             images=images_obj,
             pokemon_id=pid_np,
             category=np.array(cat_arr),
             gen=np.array(gen_arr, dtype=np.int64),
             source_name=np.array(name_arr),
             smash_pct=np.array(smash_arr, dtype=np.float32),
             total_votes=np.array(votes_arr, dtype=np.int64))

    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pokemon_id", "source_name", "category",
                                          "gen", "smash_pct", "total_votes"])
        w.writeheader(); w.writerows(manifest_rows)

    train, val = build_split(cfg).split(pid_np, cfg.split.val_frac, cfg.seed)
    (out / "split.json").write_text(json.dumps(
        {"strategy": cfg.split.strategy, "val_frac": cfg.split.val_frac,
         "seed": cfg.seed, "train": train.tolist(), "val": val.tolist()}, indent=2))

    counts: dict[int, int] = {}
    for p in pid_arr:
        counts[p] = counts.get(p, 0) + 1
    (out / "stats.json").write_text(json.dumps(
        {"n_images": len(images), "n_pokemon": len(counts),
         "per_pokemon_counts": counts, "selection_reports": reports}, indent=2))

    return out

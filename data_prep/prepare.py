from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from .config import DataConfig
from .selection import load_labels, load_records, select_pokemon
from .splits import build_split
from .imagestore import ImageStoreWriter, decode_rgba


def load_sprite(path) -> np.ndarray:
    """Load an image as uint8 RGBA at native size; first frame for animations."""
    return decode_rgba(path)


def prepare(cfg: DataConfig, images_dir, labels_csv, out_root) -> Path:
    labels = load_labels(labels_csv)
    out = Path(out_root) / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    pid_arr, cat_arr, gen_arr, name_arr = [], [], [], []
    smash_arr, votes_arr = [], []
    manifest_rows, reports, skipped = [], {}, []

    # stream image bytes to a packed blob, freeing each (O(1) memory)
    writer = ImageStoreWriter(out / "images.bin")
    for pid in tqdm(sorted(labels.keys()), desc="Building dataset", unit="pkmn"):
        recs = load_records(images_dir, pid, labels)
        if not recs:
            continue
        kept, report = select_pokemon(cfg, recs)
        reports[pid] = report
        for r in kept:
            try:
                data = Path(r.path).read_bytes()
                decode_rgba(data)        # validate it decodes (transient); skip if not
            except Exception as e:
                # a few downloaded sprites are corrupt/undecodable; skip them
                # rather than aborting the whole build (recorded in stats).
                skipped.append({"path": str(r.path), "error": type(e).__name__})
                continue
            writer.add_bytes(data)
            pid_arr.append(pid); cat_arr.append(r.category); gen_arr.append(r.gen)
            name_arr.append(r.source_name)
            smash_arr.append(r.smash_pct); votes_arr.append(r.total_votes)
            manifest_rows.append({"pokemon_id": pid, "source_name": r.source_name,
                                  "category": r.category, "gen": r.gen,
                                  "smash_pct": r.smash_pct, "total_votes": r.total_votes})
    offsets, lengths = writer.close()

    if skipped:
        print(f"warning: skipped {len(skipped)} unreadable image(s); "
              "see stats.json 'skipped_unreadable'")

    if not pid_arr:
        (out / "images.bin").unlink(missing_ok=True)
        raise ValueError(
            f"No images selected for dataset {cfg.name!r}; "
            "check selection filters and that images_dir/labels_csv are correct."
        )

    pid_np = np.array(pid_arr, dtype=np.int64)

    np.savez(out / "data.npz",
             offsets=offsets,
             lengths=lengths,
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
    smash_hist_np = np.array(smash_arr, dtype=np.float64)
    hist_counts, hist_edges = np.histogram(smash_hist_np, bins=10, range=(0.0, 1.0))
    (out / "stats.json").write_text(json.dumps(
        {"n_images": len(pid_arr), "n_pokemon": len(counts),
         "per_pokemon_counts": counts,
         "label_histogram": {"bins": hist_edges.tolist(), "counts": hist_counts.tolist()},
         "n_skipped_unreadable": len(skipped), "skipped_unreadable": skipped,
         "selection_reports": reports}, indent=2))

    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Build a datasets/{name}/ artifact.")
    p.add_argument("config", help="path to a dataset config JSON")
    p.add_argument("--images", default="images")
    p.add_argument("--labels", default="pokesmash_votes.csv")
    p.add_argument("--out", default="datasets")
    args = p.parse_args(argv)
    cfg = DataConfig.from_dict(json.loads(Path(args.config).read_text()))
    out = prepare(cfg, args.images, args.labels, args.out)
    print(f"Wrote dataset to {out}")


if __name__ == "__main__":
    main()

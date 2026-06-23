#!/usr/bin/env python3
"""Pack distributable release artifacts into dist/.

- Strips each checkpoint to inference essentials (model_state + config + metrics),
  dropping the optimizer/scheduler/RNG state that only matters for resuming
  training. ~249 MB -> ~87 MB per checkpoint.
- Copies the calibration map alongside each (if present).
- Tars the reference dataset (images.bin is already-compressed image bytes, so a
  plain tar is used -- gzip/zip would not meaningfully shrink it).

Upload the dist/ contents to Hugging Face (see docs/RELEASE.md); consumers fetch
them with `download_models.py`.
"""
from __future__ import annotations
import argparse
import shutil
import tarfile
from pathlib import Path
import torch

MODELS = ["vit_small_portraits_v1", "vit_small_mixed_v1", "vit_small_mixed_v2"]
DATASET = "mixed_v1"
_KEEP = ("model_state", "config", "metrics", "epoch")   # inference-only essentials


def strip_checkpoint(src: Path, dst: Path) -> tuple[float, float]:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    slim = {k: ckpt[k] for k in _KEEP if k in ckpt}
    torch.save(slim, dst)
    return src.stat().st_size / 1e6, dst.stat().st_size / 1e6


def pack(out="dist", runs="runs", datasets="datasets") -> Path:
    out = Path(out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        src = Path(runs) / name / "checkpoints" / "best.pt"
        if not src.exists():
            print(f"  skip {name}: no best.pt"); continue
        dst = out / "checkpoints" / f"{name}.pt"
        big, small = strip_checkpoint(src, dst)
        print(f"  {name}.pt: {big:.0f} MB -> {small:.0f} MB")
        calib = src.parent / "calibration.json"
        if calib.exists():
            shutil.copy(calib, out / "checkpoints" / f"{name}.calibration.json")
            print(f"    + calibration.json")
        else:
            print(f"    ! no calibration.json (run model.calibrate)")

    ds_dir = Path(datasets) / DATASET
    if ds_dir.is_dir():
        tar_path = out / f"{DATASET}_dataset.tar"
        print(f"  taring {ds_dir} + backgrounds/ -> {tar_path} ...")
        # arcnames are repo-root-relative; the tar extracts at the repo root so
        # datasets/{name}/ and backgrounds/ both land where config paths expect.
        with tarfile.open(tar_path, "w") as tf:
            tf.add(ds_dir, arcname=f"{datasets}/{DATASET}")
            bg = Path("backgrounds")
            if bg.is_dir():
                tf.add(bg, arcname="backgrounds")
                print(f"    + backgrounds/ (sprite augmentation needs these)")
            else:
                print(f"    ! backgrounds/ not found -- sprite aug will fall back to white")
        print(f"  {tar_path.name}: {tar_path.stat().st_size / 1e9:.1f} GB")
    print(f"\nDone. Upload dist/ contents per docs/RELEASE.md")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Pack release artifacts into dist/.")
    p.add_argument("--out", default="dist")
    p.add_argument("--runs", default="runs")
    p.add_argument("--datasets", default="datasets")
    p.add_argument("--no-dataset", action="store_true", help="checkpoints only")
    args = p.parse_args(argv)
    if args.no_dataset:
        global DATASET
        DATASET = ""
    pack(out=args.out, runs=args.runs, datasets=args.datasets)


if __name__ == "__main__":
    main()

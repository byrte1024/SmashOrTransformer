#!/usr/bin/env python3
"""Download pretrained Smash or Transformer checkpoints (and optionally the
dataset) from Hugging Face into the layout the rest of the tooling expects.

  uv run python download_models.py                 # recommended model (mixed_v1)
  uv run python download_models.py --all           # all three checkpoints
  uv run python download_models.py --dataset       # + the mixed_v1 dataset (~11 GB)
  uv run python download_models.py --list          # show what's available

Checkpoints land in runs/{name}/checkpoints/best.pt (+ calibration.json) so
`model.infer`, `model.results`, the GUI, etc. find them with no extra flags.

Repo IDs default to the published locations but can be overridden with
--model-repo / --dataset-repo or the SOT_MODEL_REPO / SOT_DATASET_REPO env vars
(handy before the repos are public, or for forks).
"""
from __future__ import annotations
import argparse
import os
import tarfile
from pathlib import Path

# Published Hugging Face repos (override via flags/env if needed).
MODEL_REPO = os.environ.get("SOT_MODEL_REPO", "supernovayuli/smash-or-transformer")
DATASET_REPO = os.environ.get("SOT_DATASET_REPO", "supernovayuli/smash-or-transformer-data")

# name -> (checkpoint file in the model repo, calibration file or None)
MODELS = {
    "vit_small_mixed_v1":    ("vit_small_mixed_v1.pt",    "vit_small_mixed_v1.calibration.json"),
    "vit_small_portraits_v1":("vit_small_portraits_v1.pt","vit_small_portraits_v1.calibration.json"),
    "vit_small_mixed_v2":    ("vit_small_mixed_v2.pt",    None),
}
RECOMMENDED = "vit_small_mixed_v1"
DATASET_FILE = "mixed_v1_dataset.tar"      # extracts to datasets/mixed_v1/


def _hf_download(repo_id, filename, repo_type="model"):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def fetch_model(name, model_repo, runs="runs") -> None:
    ckpt_file, calib_file = MODELS[name]
    dst = Path(runs) / name / "checkpoints"
    dst.mkdir(parents=True, exist_ok=True)
    print(f"  {name}: downloading {ckpt_file} ...")
    src = _hf_download(model_repo, ckpt_file)
    Path(dst / "best.pt").write_bytes(Path(src).read_bytes())
    if calib_file:
        try:
            csrc = _hf_download(model_repo, calib_file)
            Path(dst / "calibration.json").write_bytes(Path(csrc).read_bytes())
            print(f"    + calibration.json")
        except Exception as e:
            print(f"    ! calibration unavailable ({e})")
    print(f"    -> {dst / 'best.pt'}")


def fetch_dataset(dataset_repo, root=".") -> None:
    print(f"  dataset: downloading {DATASET_FILE} (~11 GB) ...")
    tar = _hf_download(dataset_repo, DATASET_FILE, repo_type="dataset")
    out = Path(root)
    print(f"  extracting -> {out}/ (datasets/mixed_v1/ + backgrounds/) ...")
    with tarfile.open(tar) as tf:
        tf.extractall(out, filter="data")        # tar holds repo-root-relative paths
    print(f"    -> {out}/datasets/mixed_v1/  and  {out}/backgrounds/")


def main(argv=None):
    p = argparse.ArgumentParser(description="Download pretrained checkpoints + dataset.")
    p.add_argument("models", nargs="*", help=f"model names (default: {RECOMMENDED})")
    p.add_argument("--all", action="store_true", help="all checkpoints")
    p.add_argument("--dataset", action="store_true", help="also fetch the mixed_v1 dataset (~11 GB)")
    p.add_argument("--list", action="store_true", help="list available artifacts and exit")
    p.add_argument("--model-repo", default=MODEL_REPO)
    p.add_argument("--dataset-repo", default=DATASET_REPO)
    p.add_argument("--runs", default="runs")
    p.add_argument("--root", default=".", help="where to extract the dataset tar (default: repo root)")
    args = p.parse_args(argv)

    if args.list:
        print("models:", *MODELS, sep="\n  ")
        print(f"dataset: {DATASET_FILE} (-> datasets/mixed_v1/)")
        print(f"\nmodel repo:   {args.model_repo}\ndataset repo: {args.dataset_repo}")
        return

    if "TODO" in args.model_repo:
        print("Set the repo first: --model-repo <user>/<repo> or $SOT_MODEL_REPO\n"
              "(repos not published yet -- see docs/RELEASE.md)")

    names = list(MODELS) if args.all else (args.models or [RECOMMENDED])
    bad = [n for n in names if n not in MODELS]
    if bad:
        p.error(f"unknown model(s): {bad}. choose from {list(MODELS)}")

    for n in names:
        fetch_model(n, args.model_repo, runs=args.runs)
    if args.dataset:
        fetch_dataset(args.dataset_repo, root=args.root)
    print("\nDone.")


if __name__ == "__main__":
    main()

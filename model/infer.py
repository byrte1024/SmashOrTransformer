from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from data_prep.prepare import load_sprite
from .config import TrainConfig
from .model import SmashRanker
from .dataset import render_input, to_tensor


def load_calibration(checkpoint_path, fit: str = "auto"):
    """Return (xs, ys) isotonic knots from calibration.json next to the
    checkpoint, or None if absent/disabled. fit selects which map to use;
    'auto' uses the file's stored default."""
    if fit == "none":
        return None
    path = Path(checkpoint_path).parent / "calibration.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    key = data.get("default", "val") if fit == "auto" else fit
    m = data["maps"][key]
    return m["xs"], m["ys"]


def load_model(checkpoint_path, device="cuda", pretrained: bool = False):
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = TrainConfig.from_dict(ckpt["config"])
    model = SmashRanker(cfg.backbone, cfg.resolution, cfg.dropout, pretrained=pretrained)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).eval()
    return model, cfg


def score_image(model, cfg: TrainConfig, image_path, device="cuda", calib=None,
                stretch=True) -> float:
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    sprite = load_sprite(image_path)
    img = render_input(sprite, cfg.resolution, stretch)
    t = to_tensor(img, model.data_config["mean"], model.data_config["std"])
    t = t.unsqueeze(0).to(dev)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).item()
    if calib is not None:
        prob = float(np.clip(np.interp(prob, calib[0], calib[1]), 0.0, 1.0))
    return prob * 100.0


def main(argv=None):
    p = argparse.ArgumentParser(description="Score image(s) for smash-ability (0-100).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--calibration", default="auto",
                   choices=["auto", "none", "train", "val", "combined"],
                   help="which calibration map to apply (auto=file default; none=raw)")
    p.add_argument("--stretch", action=argparse.BooleanOptionalAction, default=True,
                   help="stretch-to-square model input (default) vs aspect-fit (--no-stretch)")
    p.add_argument("images", nargs="+")
    args = p.parse_args(argv)
    model, cfg = load_model(args.checkpoint, device=args.device, pretrained=False)
    calib = load_calibration(args.checkpoint, fit=args.calibration)
    tag = "raw" if calib is None else f"calibrated:{args.calibration}"
    for path in args.images:
        score = score_image(model, cfg, path, device=args.device, calib=calib,
                            stretch=args.stretch)
        print(f"{path}: {score:.1f} [{tag}]")


if __name__ == "__main__":
    main()

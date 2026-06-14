from __future__ import annotations
import argparse
import torch
from data_prep.prepare import load_sprite
from .config import TrainConfig
from .model import SmashRanker
from .dataset import canonical_render, to_tensor


def load_model(checkpoint_path, device="cuda", pretrained: bool = False):
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = TrainConfig.from_dict(ckpt["config"])
    model = SmashRanker(cfg.backbone, cfg.resolution, cfg.dropout, pretrained=pretrained)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).eval()
    return model, cfg


def score_image(model, cfg: TrainConfig, image_path, device="cuda") -> float:
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    sprite = load_sprite(image_path)
    img = canonical_render(sprite, cfg.resolution)
    t = to_tensor(img, model.data_config["mean"], model.data_config["std"])
    t = t.unsqueeze(0).to(dev)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).item()
    return prob * 100.0


def main(argv=None):
    p = argparse.ArgumentParser(description="Score image(s) for smash-ability (0-100).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("images", nargs="+")
    args = p.parse_args(argv)
    model, cfg = load_model(args.checkpoint, device=args.device, pretrained=False)
    for path in args.images:
        print(f"{path}: {score_image(model, cfg, path, device=args.device):.1f}")


if __name__ == "__main__":
    main()

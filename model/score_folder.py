"""Score an arbitrary folder of images with the (calibrated) model and render
labeled portraits -- the 'results' treatment for fresh, out-of-distribution
entries that have no ground-truth label (e.g. Palworld Pals).

For each image: canonical-render -> model -> calibrated score -> a SMASH/PASS
banner. Writes labeled portraits to the output dir plus scores.csv (ranked),
and prints the top/bottom and the SMASH count.

Default calibration is the 'val' map -- the generalization regime, correct for
unseen inputs (the train map would be wrong; per-split doesn't apply with no
split).
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from data_prep.prepare import load_sprite
from .dataset import canonical_render, to_tensor
from .infer import load_model
from .calibrate import apply_calibration
from .results import _banner          # reuse the green/red banner drawer

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _gather(source) -> list[Path]:
    p = Path(source)
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in _EXTS)
    return sorted(Path(".").glob(source))   # treat as a glob pattern


def _annotate(portrait_rgb, text, smash, banner_h=56):
    img = Image.fromarray(portrait_rgb)
    W, H = img.size
    canvas = Image.new("RGB", (W, H + banner_h), (255, 255, 255))
    canvas.paste(img, (0, banner_h))
    _banner(canvas, 0, 0, W, banner_h, text, smash)
    return canvas


def _load_map(checkpoint_path, calibration):
    if calibration == "none":
        return None
    cpath = Path(checkpoint_path).parent / "calibration.json"
    if not cpath.exists():
        return None
    return json.loads(cpath.read_text())["maps"].get(calibration)


def _load_names(names_csv) -> dict:
    if not names_csv:
        return {}
    out = {}
    with open(names_csv, newline="") as f:
        for r in csv.DictReader(f):
            out[str(r["stem"])] = r["name"]
    return out


def run(checkpoint_path, source, out_dir="results/external", device="cuda",
        calibration="val", threshold=0.5, display_res=384, names=None) -> Path:
    model, cfg = load_model(checkpoint_path, device=device, pretrained=False)
    dev = next(model.parameters()).device
    mean, std = model.data_config["mean"], model.data_config["std"]
    cmap = _load_map(checkpoint_path, calibration)
    name_map = _load_names(names)

    paths = _gather(source)
    if not paths:
        raise ValueError(f"no images found at {source!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    thr_pct = threshold * 100
    rows = []

    for p in tqdm(paths, desc="scoring", unit="img"):
        sprite = load_sprite(p)
        disp = canonical_render(sprite, display_res)
        t = to_tensor(canonical_render(sprite, cfg.resolution), mean, std).unsqueeze(0).to(dev)
        with torch.no_grad():
            raw = float(torch.sigmoid(model(t).reshape(-1)[0]))
        cal = float(apply_calibration([raw], cmap["xs"], cmap["ys"])[0]) if cmap else raw
        pct = cal * 100
        smash = pct >= thr_pct
        name = name_map.get(p.stem, "")
        label = (f"{name}  " if name else "") + f"{'SMASH' if smash else 'PASS'} {pct:.0f}%"
        _annotate(disp, label, smash).save(out / f"{p.stem}.png")
        rows.append({"file": p.name, "name": name, "raw_pct": round(raw * 100, 2),
                     "calibrated_pct": round(pct, 2),
                     "decision": "SMASH" if smash else "PASS"})

    rows.sort(key=lambda r: -r["calibrated_pct"])
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "name", "raw_pct", "calibrated_pct", "decision"])
        w.writeheader(); w.writerows(rows)

    n_smash = sum(r["decision"] == "SMASH" for r in rows)
    print(f"\nScored {len(rows)} images | SMASH {n_smash} (>{thr_pct:.0f}%) | calibration={calibration}")
    label = lambda r: f"{r['name'] or r['file']}"
    print("\nTop 10:")
    for r in rows[:10]:
        print(f"  {r['calibrated_pct']:5.1f}%  {label(r)}")
    print("\nBottom 10:")
    for r in rows[-10:]:
        print(f"  {r['calibrated_pct']:5.1f}%  {label(r)}")
    print(f"\nWrote labeled portraits + scores.csv to {out}/")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Score any folder of images and render labeled portraits.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("source", help="folder of images (scanned recursively) or a glob pattern")
    p.add_argument("--out", default="results/external")
    p.add_argument("--device", default="cuda")
    p.add_argument("--calibration", default="val",
                   choices=["val", "train", "combined", "none"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--display-res", type=int, default=384)
    p.add_argument("--names", default=None, help="optional CSV with columns stem,name")
    args = p.parse_args(argv)
    run(args.checkpoint, args.source, out_dir=args.out, device=args.device,
        calibration=args.calibration, threshold=args.threshold,
        display_res=args.display_res, names=args.names)


if __name__ == "__main__":
    main()

"""Tiny Tkinter GUI: pick an image, see the model's smash/pass verdict.

Loads a trained checkpoint (+ its calibration), lets you open any image, and
shows the picture with a green SMASH / red PASS banner and the score.

    uv run python -m model.score_image_app \
        --checkpoint runs/vit_small_portraits_v1/checkpoints/best.pt

Tkinter is imported lazily inside the GUI builder so this module (and its pure
`score_file` helper) import fine on headless machines.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
from data_prep.prepare import load_sprite
from .dataset import canonical_render
from .infer import load_model, load_calibration, score_image


def score_file(model, cfg, calib, path, device="cuda", threshold=0.5):
    """-> (raw_pct, calibrated_pct, smash_bool) for one image."""
    cal = score_image(model, cfg, path, device=device, calib=calib)
    raw = score_image(model, cfg, path, device=device, calib=None)
    return raw, cal, cal >= threshold * 100


def _run_gui(model, cfg, calib, device, threshold, preview_res=360):
    import tkinter as tk
    from tkinter import filedialog
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Smash or Pass - Pokemon scorer")
    root.geometry(f"{preview_res + 80}x{preview_res + 200}")
    root.configure(bg="#1e1e1e")

    state = {"imgref": None}
    tk.Button(root, text="Open image...", font=("Helvetica", 14),
              command=lambda: _open(root, state, model, cfg, calib, device, threshold,
                                    preview_res, preview, verdict, detail)).pack(pady=12)
    preview = tk.Label(root, bg="#1e1e1e")
    preview.pack(pady=6)
    verdict = tk.Label(root, text="choose an image", font=("Helvetica", 30, "bold"),
                       bg="#1e1e1e", fg="#cccccc")
    verdict.pack(pady=10)
    detail = tk.Label(root, text="", font=("Helvetica", 12), bg="#1e1e1e", fg="#999999")
    detail.pack()
    root.mainloop()


def _open(root, state, model, cfg, calib, device, threshold, preview_res,
          preview, verdict, detail):
    from tkinter import filedialog
    from PIL import ImageTk
    path = filedialog.askopenfilename(
        title="Pick an image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"), ("All files", "*.*")])
    if not path:
        return
    disp = canonical_render(load_sprite(path), preview_res)
    state["imgref"] = ImageTk.PhotoImage(Image.fromarray(disp))
    preview.config(image=state["imgref"])
    raw, cal, smash = score_file(model, cfg, calib, path, device=device, threshold=threshold)
    verdict.config(text=f"{'SMASH' if smash else 'PASS'}  {cal:.0f}%",
                   fg=("#34c759" if smash else "#ff3b30"))
    detail.config(text=f"raw {raw:.1f}%   calibrated {cal:.1f}%   -   {Path(path).name}")


def main(argv=None):
    p = argparse.ArgumentParser(description="GUI to score an image for smash/pass.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--calibration", default="auto",
                   choices=["auto", "none", "train", "val", "combined"])
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)
    model, cfg = load_model(args.checkpoint, device=args.device, pretrained=False)
    calib = load_calibration(args.checkpoint, fit=args.calibration)
    _run_gui(model, cfg, calib, args.device, args.threshold)


if __name__ == "__main__":
    main()

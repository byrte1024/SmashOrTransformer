"""Tkinter GUI to score images for smash/pass.

Features:
  - pick a model (a run under runs/) and a specific checkpoint from dropdowns
    (auto-discovered), or browse to any .pt
  - open a single image, or a whole folder and step through it with Prev/Next
    (or the Left/Right arrow keys)
  - drag-and-drop image files onto the window (if tkinterdnd2 is available)
  - Share: save the generated labeled (banner) image to disk

All non-GUI logic lives in pure helpers (find_models / find_checkpoints /
list_images / build_result_image / _parse_drop) that are unit-tested; tkinter is
imported lazily inside the GUI builder so this module imports headlessly.

    uv run python -m model.score_image_app            # auto-discovers runs/
    uv run python -m model.score_image_app --checkpoint runs/<run>/checkpoints/best.pt
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from PIL import Image
from data_prep.prepare import load_sprite
from .dataset import canonical_render
from .infer import load_model, load_calibration, score_image
from .results import annotate_portrait

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


# --------------------------------------------------------------------------- #
# pure helpers (tested)
# --------------------------------------------------------------------------- #
def find_models(runs_dir="runs") -> list[tuple[str, Path]]:
    """Runs (model dirs) under runs_dir that contain at least one checkpoint."""
    base = Path(runs_dir)
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "checkpoints").is_dir() and any((d / "checkpoints").glob("*.pt")):
            out.append((d.name, d))
    return out


def find_checkpoints(model_dir) -> list[tuple[str, Path]]:
    """Checkpoints for a model: best, last, then epochs (latest first)."""
    cdir = Path(model_dir) / "checkpoints"
    pts = {p.name: p for p in cdir.glob("*.pt")} if cdir.is_dir() else {}
    ordered = []
    for special in ("best.pt", "last.pt"):
        if special in pts:
            ordered.append((special, pts.pop(special)))
    epochs = sorted((p for n, p in pts.items() if n.startswith("epoch_")),
                    key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
                    reverse=True)
    ordered += [(p.name, p) for p in epochs]
    ordered += [(n, p) for n, p in pts.items() if not n.startswith("epoch_")]
    return ordered


def list_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS)


def _parse_drop(data: str) -> list[Path]:
    """Parse a tkdnd <<Drop>> data string into image paths (handles {braced}
    paths with spaces and multiple files)."""
    parts = re.findall(r"\{[^}]*\}|\S+", data or "")
    out = []
    for part in parts:
        p = Path(part.strip("{}"))
        if p.suffix.lower() in _IMG_EXTS:
            out.append(p)
    return out


def score_file(model, cfg, calib, path, device="cuda", threshold=0.5):
    """-> (raw_pct, calibrated_pct, smash_bool) for one image."""
    cal = score_image(model, cfg, path, device=device, calib=calib)
    raw = score_image(model, cfg, path, device=device, calib=None)
    return raw, cal, cal >= threshold * 100


def build_result_image(model, cfg, calib, path, device="cuda", threshold=0.5,
                       display_res=384):
    """-> (labeled PIL image with SMASH/PASS banner, raw_pct, cal_pct, smash)."""
    raw, cal, smash = score_file(model, cfg, calib, path, device=device, threshold=threshold)
    disp = canonical_render(load_sprite(path), display_res)
    return annotate_portrait(disp, cal, smash), raw, cal, smash


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def _run_gui(runs_dir, device, threshold, initial_checkpoint=None, display_res=384):
    import tkinter as tk
    from tkinter import filedialog, ttk
    from PIL import ImageTk
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        root = TkinterDnD.Tk()
        has_dnd = True
    except Exception:
        root = tk.Tk()
        has_dnd = False

    root.title("Smash or Pass - Pokemon scorer")
    root.configure(bg="#1e1e1e")
    st = {"model": None, "cfg": None, "calib": None, "images": [], "idx": -1,
          "result": None, "imgref": None, "cks": []}
    models = find_models(runs_dir)

    def status(msg):
        detail.config(text=msg)

    def do_load(ckpt_path):
        status(f"loading {Path(ckpt_path).parent.parent.name}/{Path(ckpt_path).name} ...")
        root.update_idletasks()
        st["model"], st["cfg"] = load_model(ckpt_path, device=device, pretrained=False)
        st["calib"] = load_calibration(ckpt_path, fit="auto")
        status(f"loaded {Path(ckpt_path).parent.parent.name}/{Path(ckpt_path).name}")
        if st["images"]:
            show_current()

    def on_model_change(*_):
        md = dict(models).get(model_var.get())
        st["cks"] = find_checkpoints(md) if md else []
        ckpt_menu["values"] = [c[0] for c in st["cks"]]
        if st["cks"]:
            ckpt_var.set(st["cks"][0][0])
            do_load(st["cks"][0][1])

    def on_ckpt_change(*_):
        ck = dict(st["cks"]).get(ckpt_var.get())
        if ck:
            do_load(ck)

    def browse_checkpoint():
        p = filedialog.askopenfilename(title="Pick a checkpoint",
                                       filetypes=[("Checkpoints", "*.pt"), ("All", "*.*")])
        if p:
            do_load(p)

    def set_images(paths):
        paths = [Path(p) for p in paths if Path(p).suffix.lower() in _IMG_EXTS]
        if not paths:
            return
        st["images"], st["idx"] = paths, 0
        show_current()

    def open_image():
        p = filedialog.askopenfilename(filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"), ("All", "*.*")])
        if p:
            set_images([p])

    def open_folder():
        d = filedialog.askdirectory()
        if d:
            imgs = list_images(d)
            if imgs:
                set_images(imgs)
            else:
                status("no images in that folder")

    def step(delta):
        if st["images"]:
            st["idx"] = (st["idx"] + delta) % len(st["images"])
            show_current()

    def show_current():
        if st["model"] is None:
            status("select a model first")
            return
        path = st["images"][st["idx"]]
        img, raw, cal, smash = build_result_image(st["model"], st["cfg"], st["calib"],
                                                  path, device, threshold, display_res)
        st["result"] = img
        st["imgref"] = ImageTk.PhotoImage(img)
        preview.config(image=st["imgref"])
        n = len(st["images"])
        status(f"raw {raw:.1f}%   calibrated {cal:.1f}%   "
               f"[{st['idx'] + 1}/{n}]   {path.name}")

    def share():
        if st["result"] is None:
            return
        name = st["images"][st["idx"]].stem if st["images"] else "result"
        f = filedialog.asksaveasfilename(defaultextension=".png", initialfile=f"{name}_scored.png",
                                         filetypes=[("PNG", "*.png")])
        if f:
            st["result"].save(f)
            status(f"saved {f}")

    def on_drop(event):
        set_images(_parse_drop(event.data))

    # --- layout ---
    top = tk.Frame(root, bg="#1e1e1e"); top.pack(pady=8, padx=8, fill="x")
    tk.Label(top, text="Model", bg="#1e1e1e", fg="#ccc").pack(side="left")
    model_var = tk.StringVar()
    model_menu = ttk.Combobox(top, textvariable=model_var, state="readonly", width=24,
                              values=[m[0] for m in models])
    model_menu.pack(side="left", padx=4)
    model_menu.bind("<<ComboboxSelected>>", on_model_change)
    tk.Label(top, text="Checkpoint", bg="#1e1e1e", fg="#ccc").pack(side="left")
    ckpt_var = tk.StringVar()
    ckpt_menu = ttk.Combobox(top, textvariable=ckpt_var, state="readonly", width=22)
    ckpt_menu.pack(side="left", padx=4)
    ckpt_menu.bind("<<ComboboxSelected>>", on_ckpt_change)
    tk.Button(top, text="Browse...", command=browse_checkpoint).pack(side="left", padx=4)

    btns = tk.Frame(root, bg="#1e1e1e"); btns.pack(pady=4)
    for text, cmd in [("Open image", open_image), ("Open folder", open_folder),
                      ("< Prev", lambda: step(-1)), ("Next >", lambda: step(1)),
                      ("Share", share)]:
        tk.Button(btns, text=text, command=cmd, font=("Helvetica", 12)).pack(side="left", padx=4)

    preview = tk.Label(root, bg="#1e1e1e"); preview.pack(pady=8)
    detail = tk.Label(root, text="select a model and open an image", bg="#1e1e1e",
                      fg="#999", font=("Helvetica", 12))
    detail.pack(pady=4)
    hint = ("drag & drop images here  |  arrow keys to navigate" if has_dnd
            else "arrow keys to navigate  (install tkinterdnd2 for drag & drop)")
    tk.Label(root, text=hint, bg="#1e1e1e", fg="#666", font=("Helvetica", 10)).pack(pady=2)

    root.bind("<Left>", lambda e: step(-1))
    root.bind("<Right>", lambda e: step(1))
    if has_dnd:
        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", on_drop)

    # initial model selection
    if initial_checkpoint:
        do_load(initial_checkpoint)
    elif models:
        model_var.set(models[0][0])
        on_model_change()

    root.geometry(f"{display_res + 160}x{display_res + 260}")
    root.mainloop()


def main(argv=None):
    p = argparse.ArgumentParser(description="GUI to score images for smash/pass.")
    p.add_argument("--checkpoint", default=None, help="preload a specific .pt")
    p.add_argument("--runs", default="runs", help="directory of model runs to discover")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)
    _run_gui(args.runs, args.device, args.threshold, initial_checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()

"""Pygame GUI to score images for smash/pass.

Features:
  - pick a model (a run under runs/) and a specific checkpoint from the sidebar
    (auto-discovered); drop a .pt to load one from anywhere
  - drag-and-drop image file(s) or a whole folder; step with Prev/Next or the
    Left/Right arrow keys
  - Share (button or 's'): save the generated labeled (banner) image to
    results/shared/

All non-GUI logic lives in pure helpers (find_models / find_checkpoints /
list_images / build_result_image / _parse_drop) that are unit-tested; pygame is
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
    import pygame
    pygame.init()
    SW = display_res + 250
    SH = display_res + 240
    screen = pygame.display.set_mode((SW, SH), pygame.RESIZABLE)
    pygame.display.set_caption("Smash or Pass - Pokemon scorer")
    clock = pygame.time.Clock()
    F = lambda s, b=False: pygame.font.SysFont("Arial", s, bold=b)
    font, big, small, hdr = F(17), F(26, True), F(13), F(14, True)
    BG, PANEL, SELc = (28, 28, 30), (44, 44, 48), (58, 88, 140)
    TXT, MUT = (222, 222, 222), (140, 140, 142)
    GREEN, RED, BTN, BTNH = (52, 199, 89), (255, 59, 48), (66, 66, 72), (96, 96, 104)
    SIDEBAR = 230

    st = {"model": None, "cfg": None, "calib": None, "images": [], "idx": -1,
          "result": None, "surf": None, "status": "select a model (sidebar) "
          "and drop an image", "models": find_models(runs_dir),
          "cks": [], "sel_model": -1, "sel_ck": -1}

    def to_surface(pil):
        return pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode)

    def render_loading(msg):
        screen.fill(BG)
        screen.blit(big.render(msg, True, TXT), (SIDEBAR + 24, screen.get_height() // 2))
        pygame.display.flip()

    def do_load(ckpt_path):
        render_loading("loading model...")
        st["model"], st["cfg"] = load_model(ckpt_path, device=device, pretrained=False)
        st["calib"] = load_calibration(ckpt_path, fit="auto")
        st["status"] = f"loaded {Path(ckpt_path).parent.parent.name}/{Path(ckpt_path).name}"
        if st["images"]:
            show_current()

    def select_model(i):
        st["sel_model"] = i
        st["cks"] = find_checkpoints(st["models"][i][1])
        st["sel_ck"] = 0 if st["cks"] else -1
        if st["cks"]:
            do_load(st["cks"][0][1])

    def select_ck(j):
        st["sel_ck"] = j
        do_load(st["cks"][j][1])

    def set_images(paths):
        paths = [Path(p) for p in paths if Path(p).suffix.lower() in _IMG_EXTS]
        if paths:
            st["images"], st["idx"] = paths, 0
            show_current()

    def handle_drop(paths):
        if len(paths) == 1 and Path(paths[0]).is_dir():
            imgs = list_images(paths[0])
            set_images(imgs) if imgs else st.update(status="no images in that folder")
        elif len(paths) == 1 and Path(paths[0]).suffix.lower() == ".pt":
            do_load(paths[0])
        else:
            set_images(paths)

    def step(delta):
        if st["images"]:
            st["idx"] = (st["idx"] + delta) % len(st["images"])
            show_current()

    def show_current():
        if st["model"] is None:
            st["status"] = "select a model first"
            return
        path = st["images"][st["idx"]]
        img, raw, cal, smash = build_result_image(st["model"], st["cfg"], st["calib"],
                                                  path, device, threshold, display_res)
        st["result"] = img
        st["surf"] = to_surface(img)
        st["status"] = (f"raw {raw:.1f}%   calibrated {cal:.1f}%   "
                        f"[{st['idx'] + 1}/{len(st['images'])}]   {path.name}")

    def share():
        if st["result"] is None:
            return
        out = Path("results/shared")
        out.mkdir(parents=True, exist_ok=True)
        name = st["images"][st["idx"]].stem if st["images"] else "result"
        dest = out / f"{name}_scored.png"
        st["result"].save(dest)
        st["status"] = f"saved {dest}"

    def button(rect, label, clicks, mouse):
        hover = rect.collidepoint(mouse)
        pygame.draw.rect(screen, BTNH if hover else BTN, rect, border_radius=6)
        t = font.render(label, True, TXT)
        screen.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))
        clicks.append((rect, label))

    if initial_checkpoint:
        do_load(initial_checkpoint)
    elif st["models"]:
        select_model(0)

    drop_buf = []
    running = True
    while running:
        W, H = screen.get_size()
        mouse = pygame.mouse.get_pos()
        clicks = []          # (rect, action_key) registered this frame
        row_actions = []     # (rect, callable)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_LEFT:
                    step(-1)
                elif e.key == pygame.K_RIGHT:
                    step(1)
                elif e.key == pygame.K_s:
                    share()
                elif e.key == pygame.K_ESCAPE:
                    running = False
            elif e.type == pygame.DROPFILE:
                drop_buf.append(e.file)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for rect, fn in st.get("_rows", []):
                    if rect.collidepoint(e.pos):
                        fn(); break
                else:
                    for rect, key in st.get("_btns", []):
                        if rect.collidepoint(e.pos):
                            {"prev": lambda: step(-1), "next": lambda: step(1),
                             "share": share}[key](); break
        if drop_buf:
            handle_drop(drop_buf); drop_buf = []

        # --- render ---
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, (0, 0, SIDEBAR, H))
        y = 12
        screen.blit(hdr.render("MODELS", True, MUT), (12, y)); y += 22
        for i, (name, _) in enumerate(st["models"]):
            r = pygame.Rect(8, y, SIDEBAR - 16, 24)
            if i == st["sel_model"]:
                pygame.draw.rect(screen, SELc, r, border_radius=4)
            screen.blit(font.render(name[:26], True, TXT), (14, y + 3))
            row_actions.append((r, (lambda k: (lambda: select_model(k)))(i)))
            y += 26
        y += 10
        screen.blit(hdr.render("CHECKPOINTS", True, MUT), (12, y)); y += 22
        for j, (label, _) in enumerate(st["cks"]):
            r = pygame.Rect(8, y, SIDEBAR - 16, 22)
            if j == st["sel_ck"]:
                pygame.draw.rect(screen, SELc, r, border_radius=4)
            screen.blit(small.render(label, True, TXT), (14, y + 3))
            row_actions.append((r, (lambda k: (lambda: select_ck(k)))(j)))
            y += 24
        st["_rows"] = row_actions

        # main area: image
        if st["surf"] is not None:
            iw, ih = st["surf"].get_size()
            avail_w, avail_h = W - SIDEBAR - 40, H - 110
            scale = min(avail_w / iw, avail_h / ih, 1.0)
            surf = pygame.transform.smoothscale(st["surf"], (int(iw * scale), int(ih * scale)))
            screen.blit(surf, (SIDEBAR + (W - SIDEBAR - surf.get_width()) // 2, 20))

        # buttons row + status
        st["_btns"] = []
        bw, bh, by = 110, 36, H - 78
        bx = SIDEBAR + 24
        for key, label in [("prev", "< Prev"), ("next", "Next >"), ("share", "Share (s)")]:
            r = pygame.Rect(bx, by, bw, bh)
            button(r, label, st["_btns"], mouse)
            st["_btns"][-1] = (r, key)
            bx += bw + 12
        screen.blit(font.render(st["status"], True, TXT), (SIDEBAR + 24, H - 34))
        screen.blit(small.render("drop image(s)/folder/.pt here  |  arrows navigate",
                                 True, MUT), (SIDEBAR + 24, H - 14))

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


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

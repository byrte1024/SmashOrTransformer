"""A gimmick Discord bot that rates images for smash/pass with the model.

When @mentioned, it picks what to rate by priority:
  1. an image attached to the message
  2. else, if the message is a REPLY to a message with an image, that image
  3. else, if the message says "me", the author's profile picture
  4. else, if it @mentions another member, that member's profile picture
  5. else, a little help reply

The bot token is read from a gitignored secret.txt, and the model checkpoint
path from a gitignored thebestofthebest.txt -- both next to this file.

    echo "<your-bot-token>" > gimmicks/secret.txt
    echo "runs/<run>/checkpoints/best.pt" > gimmicks/thebestofthebest.txt
    uv run python -m gimmicks.discord_bot

discord is imported lazily inside run() so the module (and its pure helpers,
which are unit-tested) import without it / headlessly.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageSequence
from model.dataset import canonical_render, to_tensor
from model.results import annotate_portrait
from model.infer import load_model, load_calibration
from model.calibrate import apply_calibration


# --------------------------------------------------------------------------- #
# pure helpers (tested)
# --------------------------------------------------------------------------- #
def pick_source(has_attachment, has_reply_image, said_me, has_other_mention) -> str:
    """Resolve which image to rate, by priority. Returns one of:
    'attachment' | 'reply' | 'me' | 'mention' | 'none'."""
    if has_attachment:
        return "attachment"
    if has_reply_image:
        return "reply"
    if said_me:
        return "me"
    if has_other_mention:
        return "mention"
    return "none"


def _first_line(path, what, hint) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found ({what}).\n    {hint.format(p=p)}")
    for line in p.read_text().splitlines():
        if line.strip():
            return line.strip()
    raise ValueError(f"{p} is empty")


def read_token(path) -> str:
    """Read the bot token from secret.txt (first non-empty line)."""
    return _first_line(path, "discord bot token",
                       'echo "<your-bot-token>" > {p}')


def read_model_path(path) -> str:
    """Read the model checkpoint path from thebestofthebest.txt (first line)."""
    return _first_line(path, "model checkpoint path",
                       'echo "runs/<run>/checkpoints/best.pt" > {p}')


def _rgba_from_bytes(data: bytes) -> np.ndarray:
    im = Image.open(BytesIO(data))
    if getattr(im, "is_animated", False):              # gif/webp -> first frame
        im = next(ImageSequence.Iterator(im)).copy()
    return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def rate_bytes(model, cfg, calib, data, device="cuda", threshold=0.5, display_res=384):
    """Score raw image bytes. -> (raw_pct, cal_pct, smash_bool, labeled_png_bytes)."""
    rgba = _rgba_from_bytes(data)
    x = to_tensor(canonical_render(rgba, cfg.resolution),
                  model.data_config["mean"], model.data_config["std"]).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = float(torch.sigmoid(model(x).reshape(-1)[0]))
    cal = float(apply_calibration([raw], calib[0], calib[1])[0]) if calib else raw
    raw_pct, cal_pct = raw * 100, cal * 100
    smash = cal_pct >= threshold * 100
    img = annotate_portrait(canonical_render(rgba, display_res), cal_pct, smash)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return raw_pct, cal_pct, smash, buf.getvalue()


def verdict_text(label, cal_pct, smash) -> str:
    verdict = "SMASH" if smash else "PASS"
    if cal_pct >= 70:
        flair = "down BAD"
    elif cal_pct >= 50:
        flair = "yeah alright"
    elif cal_pct >= 30:
        flair = "respectfully, no"
    else:
        flair = "absolutely not"
    return f"**{label}** -> **{verdict}** ({cal_pct:.0f}% smashable) - {flair}"


HELP = ("ping me with an image, reply to one, say `me` for your own pfp, or "
        "@ someone to rate theirs.")


# --------------------------------------------------------------------------- #
# settings + "light LLM" explanation (local Haiku via the claude CLI, cached)
# --------------------------------------------------------------------------- #
_DEFAULT_SETTINGS = {"light_llm": False, "llm_model": "claude-haiku-4-5"}


def load_settings(path) -> dict:
    """Load bot settings (defaults merged). 'light_llm' toggles the Haiku
    explanation; 'llm_model' picks the model."""
    s = dict(_DEFAULT_SETTINGS)
    p = Path(path)
    if p.exists():
        s.update(json.loads(p.read_text()))
    return s


def ahash(data: bytes) -> str:
    """8x8 average hash -> 16-hex string. Near-identical images share a hash,
    so the explanation cache reuses across 'similar' images."""
    im = Image.open(BytesIO(data)).convert("L").resize((8, 8))
    arr = np.asarray(im, dtype=np.float64).flatten()
    avg = arr.mean()
    bits = "".join("1" if p > avg else "0" for p in arr)
    return f"{int(bits, 2):016x}"


class ExplanationCache:
    """Tiny JSON-backed cache: ahash(+verdict) -> explanation text."""
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def build_prompt(image_path, cal_pct, smash) -> str:
    gut = "SMASH" if smash else "PASS"
    return (
        f"Look at the image at {image_path}. On instinct your gut blurted out "
        f"{cal_pct:.0f}% smashable ({gut}) -- treat that score as your gut/senses, "
        f"not a fact. Now actually study the image and react with your MIND. "
        f"If your gut was right, lean in: a witty compliment if the score is high, "
        f"a playful diss if it is low. But if your gut was clearly WRONG -- e.g. it "
        f"is static/noise, an adversarial mess, a random object, or just nothing "
        f"attractive -- openly CONTRADICT the score and call your gut out (e.g. "
        f"\"82%? what was I thinking?\"). "
        f"Rules: ONE short sentence, ASCII only, no emoji. Output ONLY the sentence "
        f"wrapped exactly as ~? your sentence here ?~ with nothing before or after.")


_REPLY_RE = re.compile(r"~\?(.*?)\?~", re.S)


def parse_reply(raw):
    """Extract the message between ~? and ?~ (fall back to the whole text if the
    markers are missing), strip non-ASCII, and return it (or None if empty)."""
    if not raw:
        return None
    m = _REPLY_RE.search(raw)
    msg = (m.group(1) if m else raw).strip()
    msg = msg.encode("ascii", "ignore").decode().strip()
    return msg or None


def _run_claude(cmd, timeout) -> str:
    if shutil.which(cmd[0]) is None:
        raise FileNotFoundError(f"{cmd[0]} CLI not found on PATH")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "claude failed").strip()[:200])
    return r.stdout


def explain(image_path, cal_pct, smash, model="claude-haiku-4-5", timeout=90):
    """Fire a quick local Haiku (claude CLI) to diss/compliment the image.
    Best-effort: returns the parsed one-liner, or None on any failure."""
    cmd = ["claude", "-p", build_prompt(image_path, cal_pct, smash),
           "--model", model, "--allowedTools", "Read"]
    try:
        out = _run_claude(cmd, timeout)
    except Exception:
        return None
    return parse_reply(out)


def get_explanation(cache, data, cal_pct, smash, model):
    """Cached explanation for the image bytes (writes a temp file for the CLI)."""
    key = f"{ahash(data)}:{int(bool(smash))}"
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        Path(tmp).write_bytes(data)
        exp = explain(tmp, cal_pct, smash, model=model)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if exp and cache is not None:
        cache.put(key, exp)
    return exp


# --------------------------------------------------------------------------- #
# discord wiring (lazy import; needs a token + gateway)
# --------------------------------------------------------------------------- #
def run(checkpoint_path, token_path="gimmicks/secret.txt", device="cuda", threshold=0.5,
        settings_path="gimmicks/settings.json", light_llm=None,
        cache_path="gimmicks/llm_cache.json"):
    import discord

    token = read_token(token_path)
    settings = load_settings(settings_path)
    use_llm = settings["light_llm"] if light_llm is None else light_llm
    llm_model = settings["llm_model"]
    cache = ExplanationCache(cache_path) if use_llm else None
    model, cfg = load_model(checkpoint_path, device=device, pretrained=False)
    calib = load_calibration(checkpoint_path, fit="auto")
    dev = next(model.parameters()).device

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    async def first_image_attachment(msg):
        for a in msg.attachments:
            if (a.content_type or "").startswith("image"):
                return a
        return None

    @client.event
    async def on_ready():
        print(f"logged in as {client.user} - mention me with an image to rate it")

    @client.event
    async def on_message(message):
        if message.author.bot or not client.user.mentioned_in(message) or message.mention_everyone:
            return

        attach = await first_image_attachment(message)
        reply_attach = None
        if message.reference is not None:
            ref = message.reference.resolved
            if ref is None and message.reference.message_id:
                try:
                    ref = await message.channel.fetch_message(message.reference.message_id)
                except discord.DiscordException:
                    ref = None
            if ref is not None:
                reply_attach = await first_image_attachment(ref)

        words = message.clean_content.lower().replace("@", " ").split()
        said_me = "me" in words
        others = [u for u in message.mentions if u.id != client.user.id and not u.bot]

        src = pick_source(bool(attach), bool(reply_attach), said_me, bool(others))
        if src == "attachment":
            data, label = await attach.read(), "this"
        elif src == "reply":
            data, label = await reply_attach.read(), "this"
        elif src == "me":
            data, label = await message.author.display_avatar.read(), message.author.display_name
        elif src == "mention":
            data, label = await others[0].display_avatar.read(), others[0].display_name
        else:
            await message.reply(HELP)
            return

        async with message.channel.typing():
            raw, cal, smash, png = await asyncio.to_thread(
                rate_bytes, model, cfg, calib, data, dev, threshold)
            text = verdict_text(label, cal, smash)
            if use_llm:
                exp = await asyncio.to_thread(get_explanation, cache, data, cal, smash, llm_model)
                if exp:
                    text += f"\n> {exp}"
        await message.reply(text, file=discord.File(BytesIO(png), filename="rating.png"))

    client.run(token)


def main(argv=None):
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description="Gimmick Discord image-rating bot.")
    p.add_argument("--model-file", default=str(here / "thebestofthebest.txt"),
                   help="file holding the checkpoint path to use")
    p.add_argument("--checkpoint", default=None,
                   help="override the path in thebestofthebest.txt")
    p.add_argument("--token", default=str(here / "secret.txt"))
    p.add_argument("--settings", default=str(here / "settings.json"))
    p.add_argument("--light-llm", dest="light_llm", action="store_true", default=None,
                   help="force-enable the Haiku explanation (overrides settings.json)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)
    checkpoint = args.checkpoint or read_model_path(args.model_file)
    run(checkpoint, token_path=args.token, device=args.device, threshold=args.threshold,
        settings_path=args.settings, light_llm=args.light_llm,
        cache_path=str(here / "llm_cache.json"))


if __name__ == "__main__":
    main()

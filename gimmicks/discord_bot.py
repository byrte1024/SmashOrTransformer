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
# discord wiring (lazy import; needs a token + gateway)
# --------------------------------------------------------------------------- #
def run(checkpoint_path, token_path="gimmicks/secret.txt", device="cuda", threshold=0.5):
    import discord

    token = read_token(token_path)
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
        await message.reply(verdict_text(label, cal, smash),
                            file=discord.File(BytesIO(png), filename="rating.png"))

    client.run(token)


def main(argv=None):
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description="Gimmick Discord image-rating bot.")
    p.add_argument("--model-file", default=str(here / "thebestofthebest.txt"),
                   help="file holding the checkpoint path to use")
    p.add_argument("--checkpoint", default=None,
                   help="override the path in thebestofthebest.txt")
    p.add_argument("--token", default=str(here / "secret.txt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)
    checkpoint = args.checkpoint or read_model_path(args.model_file)
    run(checkpoint, token_path=args.token, device=args.device, threshold=args.threshold)


if __name__ == "__main__":
    main()

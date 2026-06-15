from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run as train_run
from model.calibrate import run as calib_run
from model.infer import load_model, load_calibration
from gimmicks import discord_bot as bot


def test_module_imports_without_discord():
    import importlib
    m = importlib.import_module("gimmicks.discord_bot")
    for fn in ("pick_source", "rate_bytes", "verdict_text", "read_token", "run", "main"):
        assert hasattr(m, fn)


def test_pick_source_priority():
    # attachment beats everything
    assert bot.pick_source(True, True, True, True) == "attachment"
    # reply image beats me / mention
    assert bot.pick_source(False, True, True, True) == "reply"
    # "me" beats other mention
    assert bot.pick_source(False, False, True, True) == "me"
    # other mention when nothing else
    assert bot.pick_source(False, False, False, True) == "mention"
    # nothing applicable
    assert bot.pick_source(False, False, False, False) == "none"


def test_read_token(tmp_path):
    f = tmp_path / "secret.txt"
    f.write_text("\n  my-token-123  \nignored\n")
    assert bot.read_token(f) == "my-token-123"      # first non-empty, stripped
    with pytest.raises(FileNotFoundError):
        bot.read_token(tmp_path / "missing.txt")
    (tmp_path / "empty.txt").write_text("\n  \n")
    with pytest.raises(ValueError):
        bot.read_token(tmp_path / "empty.txt")


def test_verdict_text_has_decision_and_score():
    hi = bot.verdict_text("Gardevoir", 82.0, True)
    lo = bot.verdict_text("Caterpie", 7.0, False)
    assert "SMASH" in hi and "82%" in hi
    assert "PASS" in lo and "7%" in lo


def test_rate_bytes_scores_and_returns_png(mini_repo, tmp_path):
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    tcfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "bot", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    run_dir = train_run(tcfg, pretrained=False)
    calib_run(run_dir / "checkpoints" / "best.pt", device="cpu", batch_size=4)
    model, cfg = load_model(run_dir / "checkpoints" / "best.pt", device="cpu", pretrained=False)
    calib = load_calibration(run_dir / "checkpoints" / "best.pt", fit="auto")

    data = (mini_repo["images"] / "1" / "official-artwork.png").read_bytes()
    raw, cal, smash, png = bot.rate_bytes(model, cfg, calib, data, device="cpu", display_res=64)
    assert 0.0 <= raw <= 100.0 and 0.0 <= cal <= 100.0
    assert isinstance(smash, (bool, np.bool_))
    img = Image.open(BytesIO(png))                  # valid PNG, portrait + banner
    assert img.size[0] == 64 and img.size[1] > 64


def test_rate_bytes_handles_gif_first_frame(mini_repo, tmp_path):
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    tcfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "bot2", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    run_dir = train_run(tcfg, pretrained=False)
    model, cfg = load_model(run_dir / "checkpoints" / "best.pt", device="cpu", pretrained=False)
    data = (mini_repo["images"] / "1" / "showdown.gif").read_bytes()
    raw, cal, smash, png = bot.rate_bytes(model, cfg, None, data, device="cpu", display_res=48)
    assert Image.open(BytesIO(png)).size[0] == 48   # gif first frame scored, no crash


def test_read_model_path(tmp_path):
    f = tmp_path / "thebestofthebest.txt"
    f.write_text("runs/vit_small/checkpoints/best.pt\n")
    assert bot.read_model_path(f) == "runs/vit_small/checkpoints/best.pt"
    with pytest.raises(FileNotFoundError):
        bot.read_model_path(tmp_path / "nope.txt")


def _png_bytes(color):
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def test_load_settings_default_and_override(tmp_path):
    assert bot.load_settings(tmp_path / "nope.json")["light_llm"] is False
    f = tmp_path / "settings.json"
    f.write_text('{"light_llm": true}')
    s = bot.load_settings(f)
    assert s["light_llm"] is True and s["llm_model"] == "claude-haiku-4-5"  # default kept


def test_ahash_same_and_different():
    a, a2 = _png_bytes((200, 30, 30)), _png_bytes((200, 30, 30))
    b = _png_bytes((30, 30, 200))
    assert bot.ahash(a) == bot.ahash(a2)            # identical -> same hash
    # flat-color images all hash to 0; use structured images to show difference
    from PIL import Image, ImageDraw
    def structured(side):
        im = Image.new("L", (32, 32), 0); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 15, 31] if side else [16, 0, 31, 31], fill=255)
        buf = BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()
    assert bot.ahash(structured(True)) != bot.ahash(structured(False))


def test_build_prompt_frames_gut_vs_mind_and_markers():
    s = bot.build_prompt("/tmp/x.png", 82, True)
    p = bot.build_prompt("/tmp/x.png", 7, False)
    assert "82%" in s and "SMASH" in s
    assert "7%" in p and "PASS" in p
    for txt in (s, p):
        assert "ASCII" in txt and "~?" in txt and "?~" in txt
        assert "CONTRADICT" in txt          # the mind may disagree with the gut/score


def test_parse_reply_extracts_strips_and_falls_back():
    assert bot.parse_reply("blah blah ~? you look great ?~ trailing") == "you look great"
    assert bot.parse_reply("no markers here, just text") == "no markers here, just text"
    assert bot.parse_reply("~? cafe ☕ vibes ?~") == "cafe  vibes"   # non-ascii stripped
    assert bot.parse_reply("") is None
    assert bot.parse_reply("~?   ?~") is None


def test_explain_uses_cli_output_and_handles_failure(monkeypatch):
    monkeypatch.setattr(bot, "_run_claude", lambda cmd, timeout: "junk ~? lookin sharp ?~ end")
    assert bot.explain("/tmp/x.png", 80, True) == "lookin sharp"
    def boom(cmd, timeout):
        raise RuntimeError("no cli")
    monkeypatch.setattr(bot, "_run_claude", boom)
    assert bot.explain("/tmp/x.png", 80, True) is None


def test_get_explanation_caches(tmp_path, monkeypatch):
    cache = bot.ExplanationCache(tmp_path / "c.json")
    calls = {"n": 0}
    def fake_explain(path, cal, smash, model="x"):
        calls["n"] += 1
        return "cached line"
    monkeypatch.setattr(bot, "explain", fake_explain)
    data = _png_bytes((10, 120, 200))
    assert bot.get_explanation(cache, data, 60, True, "m") == "cached line"
    assert bot.get_explanation(cache, data, 60, True, "m") == "cached line"  # 2nd from cache
    assert calls["n"] == 1                          # explain invoked once
    assert (tmp_path / "c.json").exists()           # persisted


def test_broken_image_raises_and_has_message():
    import pytest as _pytest
    # BROKEN is a usable ASCII string
    assert bot.BROKEN and bot.BROKEN.isascii()
    # garbage and truncated PNG both raise (so on_message can catch -> BROKEN)
    with _pytest.raises(Exception):
        bot._rgba_from_bytes(b"definitely not an image")
    with _pytest.raises(Exception):
        bot._rgba_from_bytes(b"\x89PNG\r\n\x1a\n\x00\x00@Sbroken")
    # rate_bytes propagates the error (model never reached for a broken image)
    with _pytest.raises(Exception):
        bot.rate_bytes(None, None, None, b"not an image", device="cpu")


def test_is_explicit_mention_ignores_reply_pings():
    bid = 123456
    assert bot.is_explicit_mention(f"<@{bid}> rate this", bid)      # typed @mention
    assert bot.is_explicit_mention(f"hey <@!{bid}> me", bid)        # nickname-form mention
    assert not bot.is_explicit_mention("just replying, no @", bid)  # reply ping (no token)
    assert not bot.is_explicit_mention("<@999> rate this", bid)     # someone else @'d
    assert not bot.is_explicit_mention("", bid)


def test_model_loader_reloads_on_mtime_and_path_change(tmp_path):
    import os
    f = tmp_path / "best.pt"; f.write_bytes(b"x"); os.utime(f, (1000, 1000))
    n = {"c": 0}
    def load(p):
        n["c"] += 1
        return ("model", p)
    loader = bot.ModelLoader(lambda: str(f), load)

    bundle, path, reloaded = loader.get()
    assert reloaded and n["c"] == 1 and path == str(f)
    _, _, reloaded = loader.get()
    assert not reloaded and n["c"] == 1                  # unchanged -> cached

    os.utime(f, (2000, 2000))                            # best.pt rewritten
    _, _, reloaded = loader.get()
    assert reloaded and n["c"] == 2                      # picks up the newer file

    g = tmp_path / "best2.pt"; g.write_bytes(b"y"); os.utime(g, (1000, 1000))
    cur = {"p": str(f)}
    loader2 = bot.ModelLoader(lambda: cur["p"], load)
    loader2.get()
    cur["p"] = str(g)                                    # re-pointed to a new model
    _, path, reloaded = loader2.get()
    assert reloaded and path == str(g)


def test_model_tag():
    assert bot.model_tag("runs/vit_small_mixed_v1/checkpoints/best.pt") == "vit_small_mixed_v1/best"
    assert bot.model_tag("runs/foo/checkpoints/epoch_012.pt") == "foo/epoch_012"
    assert bot.model_tag("/tmp/whatever.pt") == "whatever"     # non-standard path


def test_model_loader_reloads_on_mtime_and_path_change(tmp_path):
    import os
    f = tmp_path / "best.pt"; f.write_bytes(b"x"); os.utime(f, (1000, 1000))
    n = {"c": 0}
    def load(p):
        n["c"] += 1
        return ("model", p)
    loader = bot.ModelLoader(lambda: str(f), load)

    _, path, reloaded = loader.get()
    assert reloaded and n["c"] == 1 and path == str(f)
    _, _, reloaded = loader.get()
    assert not reloaded and n["c"] == 1                  # unchanged -> cached
    os.utime(f, (2000, 2000))                            # best.pt rewritten
    _, _, reloaded = loader.get()
    assert reloaded and n["c"] == 2                      # picks up the newer file

    g = tmp_path / "best2.pt"; g.write_bytes(b"y"); os.utime(g, (1000, 1000))
    cur = {"p": str(f)}
    loader2 = bot.ModelLoader(lambda: cur["p"], load)
    loader2.get()
    cur["p"] = str(g)                                    # re-pointed to a new model
    _, path, reloaded = loader2.get()
    assert reloaded and path == str(g)


def test_model_loader_reloads_on_stamp_change_same_path():
    # custom stamp (e.g. calibration.json mtime) forces reload even if path is fixed
    n = {"c": 0}
    s = {"v": 1}
    loader = bot.ModelLoader(lambda: "x",
                             lambda p: (n.__setitem__("c", n["c"] + 1) or n["c"]),
                             stamp=lambda p: s["v"])
    loader.get()
    _, _, r = loader.get()
    assert not r and n["c"] == 1            # same stamp -> cached
    s["v"] = 2                              # e.g. calibration.json regenerated
    _, _, r = loader.get()
    assert r and n["c"] == 2                # reloaded -> picks up new calibration


def test_stretch_square_fills_square_vs_canonical_pads():
    from model.dataset import canonical_render
    # tall opaque sprite: stretch fills the whole square; canonical pads with white
    arr = np.zeros((40, 10, 4), dtype=np.uint8); arr[:, :] = (200, 50, 50, 255)
    stretched = bot.stretch_square(arr, 16)
    assert stretched.shape == (16, 16, 3)
    assert (stretched == (200, 50, 50)).all()          # no white bars -> distorted to fill
    canon = canonical_render(arr, 16)
    assert (canon == 255).any()                        # canonical leaves white padding

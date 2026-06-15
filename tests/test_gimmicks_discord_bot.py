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

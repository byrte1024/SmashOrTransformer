from pathlib import Path
import numpy as np
from PIL import Image
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run as train_run
from model.calibrate import run as calib_run
from model.infer import load_model, load_calibration
from model import score_image_app as app


def test_module_imports_without_tk():
    import importlib
    m = importlib.import_module("model.score_image_app")
    for fn in ("score_file", "build_result_image", "find_models", "find_checkpoints",
               "list_images", "_parse_drop", "main"):
        assert hasattr(m, fn)


def test_find_models_and_checkpoints(tmp_path):
    runs = tmp_path / "runs"
    for run in ("vit_a", "vit_b"):
        cdir = runs / run / "checkpoints"
        cdir.mkdir(parents=True)
        for f in ("best.pt", "last.pt", "epoch_001.pt", "epoch_012.pt", "epoch_003.pt"):
            (cdir / f).write_bytes(b"x")
    (runs / "empty_run").mkdir()                       # no checkpoints -> ignored
    models = app.find_models(str(runs))
    assert [m[0] for m in models] == ["vit_a", "vit_b"]
    cks = app.find_checkpoints(dict(models)["vit_a"])
    labels = [c[0] for c in cks]
    assert labels[:2] == ["best.pt", "last.pt"]        # specials first
    assert labels[2:] == ["epoch_012.pt", "epoch_003.pt", "epoch_001.pt"]  # epochs latest-first


def test_list_images_filters_and_sorts(tmp_path):
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("nope")
    assert [p.name for p in app.list_images(str(tmp_path))] == ["a.jpg", "b.png"]


def test_parse_drop_handles_braces_and_multiple():
    data = "{/tmp/a b/img one.png} /tmp/two.jpg /tmp/skip.txt"
    out = app._parse_drop(data)
    assert [p.name for p in out] == ["img one.png", "two.jpg"]   # txt dropped


def _trained(mini_repo, tmp_path):
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    tcfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "app", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    run_dir = train_run(tcfg, pretrained=False)
    calib_run(run_dir / "checkpoints" / "best.pt", device="cpu", batch_size=4)
    return run_dir


def test_score_file_and_build_result_image(mini_repo, tmp_path):
    run_dir = _trained(mini_repo, tmp_path)
    ckpt = run_dir / "checkpoints" / "best.pt"
    model, cfg = load_model(ckpt, device="cpu", pretrained=False)
    calib = load_calibration(ckpt, fit="auto")
    img_path = mini_repo["images"] / "1" / "official-artwork.png"

    raw, cal, smash = app.score_file(model, cfg, calib, img_path, device="cpu", threshold=0.5)
    assert 0.0 <= raw <= 100.0 and 0.0 <= cal <= 100.0
    assert smash == (cal >= 50.0)

    result, r2, c2, s2 = app.build_result_image(model, cfg, calib, img_path, device="cpu",
                                                display_res=64)
    assert isinstance(result, Image.Image)
    assert result.size[0] == 64 and result.size[1] > 64   # portrait + banner
    assert (r2, c2, s2) == (raw, cal, smash)

    # also discoverable as a model by find_models/find_checkpoints
    models = app.find_models(str(tmp_path / "runs"))
    assert any(name == "app" for name, _ in models)

import numpy as np
from PIL import Image
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run as train_run
from model.calibrate import run as calib_run
from model.infer import load_model, load_calibration
from model.score_image_app import score_file


def test_score_file_returns_raw_cal_decision(mini_repo, tmp_path):
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

    model, cfg = load_model(run_dir / "checkpoints" / "best.pt", device="cpu", pretrained=False)
    calib = load_calibration(run_dir / "checkpoints" / "best.pt", fit="auto")
    img = mini_repo["images"] / "1" / "official-artwork.png"
    raw, cal, smash = score_file(model, cfg, calib, img, device="cpu", threshold=0.5)
    assert 0.0 <= raw <= 100.0 and 0.0 <= cal <= 100.0
    assert isinstance(smash, (bool, np.bool_))
    assert smash == (cal >= 50.0)


def test_module_imports_without_tk():
    # importing the app (and its helper) must not require a display / tkinter
    import importlib
    m = importlib.import_module("model.score_image_app")
    assert hasattr(m, "score_file") and hasattr(m, "main")

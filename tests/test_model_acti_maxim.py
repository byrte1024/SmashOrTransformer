import csv
import torch
import numpy as np
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.model import SmashRanker
from model.train import run as train_run
from model.calibrate import run as calib_run
from model.acti_maxim import (maximize, run as am_run, FourierParam, MacoParam,
                              _tv, _random_affine, TECHNIQUES, BRACKETS)

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def _tiny_model():
    return SmashRanker("vit_tiny_patch16_224", resolution=32, dropout=0.0,
                       pretrained=False).eval()


def test_tv_and_affine_shapes():
    x = torch.rand(1, 3, 16, 16)
    assert _tv(x).ndim == 0 and float(_tv(x)) >= 0
    gen = torch.Generator().manual_seed(0)
    assert _random_affine(x, gen).shape == x.shape


def test_param_images_in_range():
    gen = torch.Generator().manual_seed(0)
    for P in (FourierParam, MacoParam):
        img = P(32, gen, torch.device("cpu")).image().detach()
        assert img.shape == (1, 3, 32, 32)
        assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_maximize_returns_image_for_every_technique():
    model = _tiny_model()
    for tech in TECHNIQUES:
        seed_img = torch.rand(1, 3, 32, 32) if tech == "deepdream" else None
        arr, score = maximize(model, MEAN, STD, target=0.8, technique=tech, seed=1,
                              device=torch.device("cpu"), res=32, steps=3,
                              seed_img01=seed_img)
        assert arr.shape == (32, 32, 3) and arr.dtype == np.uint8
        assert 0.0 <= score <= 1.0


def test_maximize_respects_target_ordering():
    model = _tiny_model()
    _, hi = maximize(model, MEAN, STD, target=1.0, technique="pixel_tv", seed=0,
                     device=torch.device("cpu"), res=32, steps=60)
    _, lo = maximize(model, MEAN, STD, target=0.0, technique="pixel_tv", seed=0,
                     device=torch.device("cpu"), res=32, steps=60)
    assert hi > lo                       # pushing toward 1.0 scores higher than toward 0.0


def _trained(mini_repo, tmp_path):
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    tcfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "am", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    run_dir = train_run(tcfg, pretrained=False)
    calib_run(run_dir / "checkpoints" / "best.pt", device="cpu", batch_size=4)
    return run_dir


def test_am_run_writes_brackets_and_summary(mini_repo, tmp_path):
    run_dir = _trained(mini_repo, tmp_path)
    out = am_run(run_dir / "checkpoints" / "best.pt", images_dir=str(mini_repo["images"]),
                 device="cpu", out_dir=str(tmp_path / "am_out"), steps=2, res=32, n_per=2)
    for b in BRACKETS:
        bdir = out / f"bracket_{b:.1f}"
        assert len(list(bdir.glob("*.png"))) == 2
        assert (out / f"grid_bracket_{b:.1f}.png").exists()
    rows = list(csv.DictReader(open(out / "summary.csv")))
    assert len(rows) == 2 * len(BRACKETS)
    assert {"bracket", "slot", "technique", "target_raw", "achieved_raw_pct",
            "calibrated_pct"} == set(rows[0].keys())

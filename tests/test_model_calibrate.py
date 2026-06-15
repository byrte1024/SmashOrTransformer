import json
import numpy as np
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run as train_run
from model.calibrate import fit_isotonic, apply_calibration, run as calib_run
from model.infer import load_model, score_image, load_calibration


def test_fit_isotonic_is_monotonic_and_corrects_compression():
    # compressed predictions vs spread-out truth
    pred = np.array([0.10, 0.20, 0.30, 0.40])
    true = np.array([0.10, 0.30, 0.60, 0.90])
    xs, ys = fit_isotonic(pred, true)
    assert ys == sorted(ys)                       # non-decreasing
    cal = apply_calibration(pred, xs, ys)
    # calibration pulls predictions toward truth -> lower MAE
    assert np.mean(np.abs(cal - true)) < np.mean(np.abs(pred - true))


def test_apply_calibration_clamps_outside_range():
    xs, ys = [0.2, 0.8], [0.0, 1.0]
    out = apply_calibration([0.0, 0.5, 1.0], xs, ys)
    assert out[0] == 0.0 and out[2] == 1.0       # flat extrapolation, clamped
    assert 0.0 <= out[1] <= 1.0


def test_fit_isotonic_few_points_is_identity():
    xs, ys = fit_isotonic([0.5], [0.9])
    assert xs == [0.0, 1.0] and ys == [0.0, 1.0]


def _trained(mini_repo, tmp_path):
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    tcfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "cal", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    return train_run(tcfg, pretrained=False)


def test_calibrate_run_writes_all_three_maps(mini_repo, tmp_path):
    run_dir = _trained(mini_repo, tmp_path)
    ckpt = run_dir / "checkpoints" / "best.pt"
    out = calib_run(ckpt, device="cpu", batch_size=4)
    data = json.loads(out.read_text())
    assert data["method"] == "isotonic"
    assert set(data["maps"]) == {"train", "val", "combined"}
    # report has before/after MAE for every (fit, eval) pair
    for fit in ("train", "val", "combined"):
        for ev in ("train", "val", "combined"):
            r = data["report"][fit][ev]
            assert "mae_raw" in r and "mae_calibrated" in r


def test_infer_applies_calibration(mini_repo, tmp_path):
    run_dir = _trained(mini_repo, tmp_path)
    ckpt = run_dir / "checkpoints" / "best.pt"
    calib_run(ckpt, device="cpu", batch_size=4)
    model, cfg = load_model(ckpt, device="cpu", pretrained=False)
    img = mini_repo["images"] / "1" / "official-artwork.png"
    raw = score_image(model, cfg, img, device="cpu", calib=None)
    cal = load_calibration(ckpt, fit="auto")
    calibrated = score_image(model, cfg, img, device="cpu", calib=cal)
    assert cal is not None
    assert 0.0 <= raw <= 100.0 and 0.0 <= calibrated <= 100.0

import csv
import numpy as np
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run as train_run
from model.calibrate import run as calib_run
from model.results import annotate_portrait, annotate_avg, run as results_run


def _portrait():
    return np.full((64, 64, 3), 200, dtype=np.uint8)


def test_annotate_portrait_adds_banner():
    out = annotate_portrait(_portrait(), 80.0, smash=True, banner_h=20)
    assert out.size == (64, 64 + 20)            # banner stacked above portrait
    assert out.mode == "RGB"


def test_annotate_avg_has_sidebar_and_banner():
    per = [("official-artwork", 78.0), ("home", 60.0), ("gen1_red-blue", 41.0)]
    out = annotate_avg(_portrait(), 59.7, smash=True, per_image=per,
                       spread_pct=15.0, lo_pct=41.0, hi_pct=78.0,
                       threshold_pct=50.0, banner_h=20, sidebar_w=120)
    assert out.size == (64 + 120, 20 + 64)      # portrait+banner left, sidebar right


def _trained(mini_repo, tmp_path):
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    tcfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "res", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    run_dir = train_run(tcfg, pretrained=False)
    calib_run(run_dir / "checkpoints" / "best.pt", device="cpu", batch_size=4)
    return run_dir, out


def test_results_run_writes_folders_and_csvs(mini_repo, tmp_path):
    run_dir, ds = _trained(mini_repo, tmp_path)
    out = results_run(run_dir / "checkpoints" / "best.pt", dataset_dir=str(ds),
                      device="cpu", out_dir=str(tmp_path / "results"),
                      display_res=48, batch_size=4)
    n_pokemon = len(np.unique(np.load(f"{ds}/data.npz", allow_pickle=True)["pokemon_id"]))
    assert len(list((out / "portrait").glob("*.png"))) == n_pokemon
    assert len(list((out / "all_avg").glob("*.png"))) == n_pokemon

    rank = list(csv.DictReader(open(out / "rankings.csv")))
    assert len(rank) == n_pokemon
    assert {"pokemon_id", "true_pct", "portrait_pct", "allavg_pct", "spread_pct",
            "min_pct", "max_pct", "n_images", "split", "portrait_decision",
            "allavg_decision"} <= set(rank[0].keys())
    assert rank[0]["allavg_decision"] in ("SMASH", "PASS")

    per = list(csv.DictReader(open(out / "per_image_scores.csv")))
    total_imgs = len(np.load(f"{ds}/data.npz", allow_pickle=True)["pokemon_id"])
    assert len(per) == total_imgs               # one row per image
    assert {"pokemon_id", "source_name", "raw_pct", "calibrated_pct", "smash"} == set(per[0].keys())


def test_per_split_calibration_avoids_saturation():
    from model.results import _calibrate_per_split
    # val map: narrow domain [0.1,0.4] -> [0.1,0.8]; train map: wide identity-ish
    maps = {"val": {"xs": [0.1, 0.4], "ys": [0.1, 0.8]},
            "train": {"xs": [0.1, 0.8], "ys": [0.1, 0.8]}}
    raw = np.array([0.7])           # a TRAIN pokemon whose raw exceeds the val domain
    pid = np.array([1]); val_set = set()
    # a global val map clamps 0.7 to its 0.8 ceiling (the saturation bug)
    assert abs(_calibrate_per_split(raw, pid, val_set, maps, "val")[0] - 0.8) < 1e-9
    # per-split routes it through the train map -> ~0.7, no saturation
    assert abs(_calibrate_per_split(raw, pid, val_set, maps, "per-split")[0] - 0.7) < 1e-9


def test_per_split_uses_val_map_for_val_pokemon():
    from model.results import _calibrate_per_split
    maps = {"val": {"xs": [0.1, 0.4], "ys": [0.1, 0.8]},
            "train": {"xs": [0.1, 0.8], "ys": [0.1, 0.8]}}
    raw = np.array([0.25]); pid = np.array([9]); val_set = {9}
    # val pokemon raw 0.25 -> val map midpoint ~0.45, not the train-map's ~0.25
    cal = _calibrate_per_split(raw, pid, val_set, maps, "per-split")[0]
    assert 0.4 < cal < 0.5


def test_calibration_none_is_identity():
    from model.results import _calibrate_per_split
    raw = np.array([0.3, 0.7])
    out = _calibrate_per_split(raw, np.array([1, 2]), set(), None, "none")
    assert np.allclose(out, raw)

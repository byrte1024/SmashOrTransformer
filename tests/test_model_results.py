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

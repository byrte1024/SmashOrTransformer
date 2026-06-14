import csv
import json
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run
from model.infer import load_model, score_image


def _trained(mini_repo, tmp_path):
    cfg_d = DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                                  "variations": 2,
                                  "split": {"strategy": "pokemon", "val_frac": 0.34}})
    out = prepare(cfg_d, mini_repo["images"], mini_repo["labels"],
                  mini_repo["root"] / "datasets")
    tcfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "inf", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    return run(tcfg, pretrained=False), mini_repo


def test_infer_scores_in_range(mini_repo, tmp_path):
    run_dir, repo = _trained(mini_repo, tmp_path)
    model, cfg = load_model(run_dir / "checkpoints" / "best.pt",
                            device="cpu", pretrained=False)
    img = repo["images"] / "1" / "official-artwork.png"
    score = score_image(model, cfg, img, device="cpu")
    assert 0.0 <= score <= 100.0

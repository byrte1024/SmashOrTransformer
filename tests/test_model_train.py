import csv
import json
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run


def _tiny_dataset(mini_repo, resolution=32):
    cfg = DataConfig.from_dict({"name": "ds", "resolution": resolution,
                                "minimages": 1, "variations": 2,
                                "split": {"strategy": "pokemon", "val_frac": 0.34}})
    return prepare(cfg, mini_repo["images"], mini_repo["labels"],
                   mini_repo["root"] / "datasets")


def test_train_smoke(mini_repo, tmp_path):
    out = _tiny_dataset(mini_repo, 32)
    cfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "smoke", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 2, "batch_size": 4, "freeze_epochs": 1, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    run_dir = run(cfg, pretrained=False)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "checkpoints" / "epoch_001.pt").exists()
    assert (run_dir / "checkpoints" / "epoch_002.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    with open(run_dir / "history.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    for r in rows:
        assert r["train_loss"] not in ("", "nan")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "best_epoch" in summary and "best_spearman" in summary
    assert isinstance(summary["best_spearman"], float)

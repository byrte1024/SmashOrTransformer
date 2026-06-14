import json
import pytest
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig


def _dataset(mini_repo, resolution=32):
    cfg = DataConfig.from_dict({"name": "ds", "resolution": resolution,
                                "minimages": 1, "variations": 2})
    return prepare(cfg, mini_repo["images"], mini_repo["labels"],
                   mini_repo["root"] / "datasets")


def test_defaults(mini_repo):
    out = _dataset(mini_repo, 32)
    cfg = TrainConfig.from_dict({"dataset_dir": str(out), "run_name": "r",
                                 "resolution": 32})
    assert cfg.backbone == "vit_small_patch16_224"
    assert cfg.epochs == 30 and cfg.batch_size == 64
    assert cfg.freeze_epochs == 3
    assert cfg.save_every_epoch is True
    assert cfg.amp is True


def test_resolution_must_match_dataset(mini_repo):
    out = _dataset(mini_repo, 32)
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"dataset_dir": str(out), "run_name": "r",
                               "resolution": 64})


def test_roundtrip(mini_repo):
    out = _dataset(mini_repo, 32)
    cfg = TrainConfig.from_dict({"dataset_dir": str(out), "run_name": "r",
                                 "resolution": 32, "epochs": 5})
    again = TrainConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_validate_rejects_bad_freeze(mini_repo):
    out = _dataset(mini_repo, 32)
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"dataset_dir": str(out), "run_name": "r",
                               "resolution": 32, "epochs": 2, "freeze_epochs": 5})

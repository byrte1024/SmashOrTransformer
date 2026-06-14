import numpy as np
import torch
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from data_prep.sampler import DataSampler
from model.dataset import canonical_render, to_tensor, TrainDataset, EvalDataset

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def test_canonical_render_centers_on_white():
    sprite = np.zeros((20, 10, 4), dtype=np.uint8)
    sprite[:, :] = (255, 0, 0, 255)
    out = canonical_render(sprite, 16)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    assert (out[0, 0] == 255).all()
    assert (out[8, 8] == (255, 0, 0)).all()


def test_canonical_render_deterministic():
    sprite = np.zeros((12, 12, 4), dtype=np.uint8)
    sprite[3:9, 3:9] = (0, 255, 0, 255)
    a = canonical_render(sprite, 24)
    b = canonical_render(sprite, 24)
    assert np.array_equal(a, b)


def test_to_tensor_normalizes():
    img = np.full((8, 8, 3), 255, dtype=np.uint8)
    t = to_tensor(img, MEAN, STD)
    assert t.shape == (3, 8, 8) and t.dtype == torch.float32
    assert torch.allclose(t, torch.ones_like(t))


def _dataset(mini_repo, resolution=32):
    cfg = DataConfig.from_dict({"name": "ds", "resolution": resolution,
                                "minimages": 1, "variations": 2,
                                "split": {"strategy": "pokemon", "val_frac": 0.34}})
    return prepare(cfg, mini_repo["images"], mini_repo["labels"],
                   mini_repo["root"] / "datasets")


def test_train_dataset_item(mini_repo):
    out = _dataset(mini_repo, 32)
    sampler = DataSampler(out, split="train", epoch=0)
    ds = TrainDataset(sampler, MEAN, STD)
    assert len(ds) == len(sampler)
    t, y = ds[0]
    assert t.shape == (3, 32, 32) and t.dtype == torch.float32
    assert 0.0 <= float(y) <= 1.0


def test_eval_dataset_item_has_pokemon_id(mini_repo):
    out = _dataset(mini_repo, 32)
    ds = EvalDataset(out, split="val", mean=MEAN, std=STD, resolution=32)
    assert len(ds) > 0
    t, y, pid = ds[0]
    assert t.shape == (3, 32, 32)
    assert isinstance(pid, int)
    assert 0.0 <= float(y) <= 1.0

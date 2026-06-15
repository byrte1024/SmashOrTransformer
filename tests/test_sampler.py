import numpy as np
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from data_prep.sampler import DataSampler


def _build(mini_repo, **over):
    base = {"name": "s", "resolution": 32, "seed": 3, "minimages": 1,
            "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}
    base.update(over)
    cfg = DataConfig.from_dict(base)
    return prepare(cfg, mini_repo["images"], mini_repo["labels"],
                   mini_repo["root"] / "datasets")


def test_sampler_shapes_and_label_range(mini_repo):
    ds = DataSampler(_build(mini_repo), split="train", epoch=0)
    img, label = ds[0]
    assert img.shape == (32, 32, 3) and img.dtype == np.uint8
    assert 0.0 <= label <= 1.0
    assert len(ds) > 0


def test_sampler_determinism(mini_repo):
    out = _build(mini_repo)
    a = DataSampler(out, split="train", epoch=0)
    b = DataSampler(out, split="train", epoch=0)
    ia, la = a[1]; ib, lb = b[1]
    assert np.array_equal(ia, ib) and la == lb


def test_sampler_epoch_changes_augmentation(mini_repo):
    out = _build(mini_repo)
    a = DataSampler(out, split="train", epoch=0)
    b = DataSampler(out, split="train", epoch=1)
    assert a._plan != b._plan          # epoch changes the plan order/content
    assert not np.array_equal(a[0][0], b[0][0])


def test_sampler_split_isolation(mini_repo):
    out = _build(mini_repo)
    train = DataSampler(out, split="train", epoch=0)
    val = DataSampler(out, split="val", epoch=0)
    assert set(train.pokemon_ids()).isdisjoint(val.pokemon_ids())


def test_fill_so_balances_samples(mini_repo):
    out = _build(mini_repo, variations={"fill_so": 5}, minimages=1)
    ds = DataSampler(out, split="train", epoch=0)
    counts = {}
    for pid in (ds._row_pid[r] for r in ds._plan):
        counts[int(pid)] = counts.get(int(pid), 0) + 1
    assert len(set(counts.values())) == 1


def test_sampler_votes_accessible(mini_repo):
    ds = DataSampler(_build(mini_repo), split="train", epoch=0)
    v = ds.votes(0)
    assert isinstance(v, int) and v > 0


def test_sampler_routes_booru_through_photo_aug(mini_repo, tmp_path):
    # give pokemon 1 a booru image; build a booru+portrait dataset
    import csv as _csv
    from PIL import Image as _Image
    booru = mini_repo["images"] / "1" / "booru"
    booru.mkdir()
    _Image.new("RGB", (40, 40), (10, 200, 30)).save(booru / "00_777.jpg")
    with open(booru / "meta.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rank", "post_id", "score", "rating", "file_url"])
        w.writeheader(); w.writerow({"rank": 0, "post_id": 777, "score": 9,
                                     "rating": "safe", "file_url": "x"})
    cfg = DataConfig.from_dict({"name": "mix", "resolution": 32, "minimages": 1,
                                "variations": 2,
                                "selection": {"categories": ["portrait", "booru"]},
                                "split": {"strategy": "image", "val_frac": 0.0},
                                "augmentations": {"sprite": {"background": {"prob": 0.0}},
                                                  "photo": {"flip": 0.0}}})
    out = prepare(cfg, mini_repo["images"], mini_repo["labels"], tmp_path / "datasets")
    ds = DataSampler(out, split="train", epoch=0)
    # both categories made it into the dataset
    cats = set(__import__("numpy").load(out / "data.npz", allow_pickle=True)["category"].tolist())
    assert "booru" in cats and "portrait" in cats
    img, label = ds[0]
    assert img.shape == (32, 32, 3) and img.dtype.name == "uint8"
    assert 0.0 <= label <= 1.0

import json
import pytest
import numpy as np
from data_prep.config import DataConfig
from data_prep.prepare import prepare, load_sprite


def test_load_sprite_gif_first_frame(mini_repo):
    arr = load_sprite(mini_repo["images"] / "1" / "showdown.gif")
    assert arr.dtype == np.uint8 and arr.shape[2] == 4
    assert arr[..., :3].sum() > 0 and not (arr[..., 0] == 0).all()


def test_prepare_writes_all_outputs(mini_repo):
    cfg = DataConfig.from_dict({"name": "tiny", "resolution": 32, "seed": 1,
                                "minimages": 1, "variations": 2,
                                "split": {"strategy": "pokemon", "val_frac": 0.5}})
    out = prepare(cfg, mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    assert (out / "data.npz").exists()
    assert (out / "config.json").exists()
    assert (out / "manifest.csv").exists()
    assert (out / "split.json").exists()
    assert (out / "stats.json").exists()

    data = np.load(out / "data.npz", allow_pickle=True)
    n = len(data["images"])
    assert n == len(data["pokemon_id"]) == len(data["smash_pct"]) == len(data["total_votes"])
    assert data["images"][0].dtype == np.uint8 and data["images"][0].shape[2] == 4
    assert set(np.unique(data["pokemon_id"]).tolist()) == {1, 4, 7}
    assert float(data["smash_pct"].max()) <= 1.0

    split = json.loads((out / "split.json").read_text())
    assert set(split["train"]).isdisjoint(split["val"])


def test_prepare_respects_category_filter(mini_repo):
    cfg = DataConfig.from_dict({"name": "p2", "resolution": 16,
                                "selection": {"categories": ["portrait"]}})
    out = prepare(cfg, mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    data = np.load(out / "data.npz", allow_pickle=True)
    assert set(np.unique(data["category"]).tolist()) == {"portrait"}


def test_prepare_raises_on_empty_selection(mini_repo):
    cfg = DataConfig.from_dict({"name": "empty", "resolution": 16, "minimages": 0,
                                "selection": {"names": {"include": ["does-not-exist"]}}})
    with pytest.raises(ValueError):
        prepare(cfg, mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")

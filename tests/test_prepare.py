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

    # packed-blob storage: pixels live in images.bin, npz holds offsets + metadata
    assert (out / "images.bin").exists()
    data = np.load(out / "data.npz", allow_pickle=True)
    assert "images" not in data.files
    n = len(data["offsets"])
    assert n == len(data["lengths"]) == len(data["pokemon_id"]) == len(data["smash_pct"])
    assert set(np.unique(data["pokemon_id"]).tolist()) == {1, 4, 7}
    assert float(data["smash_pct"].max()) <= 1.0

    # images decode via DatasetImages -> RGBA uint8
    from data_prep.imagestore import DatasetImages
    store = DatasetImages(out, data)
    assert len(store) == n
    assert store[0].dtype == np.uint8 and store[0].shape[2] == 4

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


def test_stats_has_label_histogram(mini_repo):
    cfg = DataConfig.from_dict({"name": "h", "resolution": 16})
    out = prepare(cfg, mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    stats = json.loads((out / "stats.json").read_text())
    assert "label_histogram" in stats
    assert len(stats["label_histogram"]["counts"]) == 10
    assert len(stats["label_histogram"]["bins"]) == 11


def test_prepare_skips_unreadable_image(mini_repo):
    # corrupt one selected image; prepare must skip it, not crash, and record it
    bad = mini_repo["images"] / "1" / "gen1_red-blue.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n garbage not a real png")
    cfg = DataConfig.from_dict({"name": "skip", "resolution": 16})
    out = prepare(cfg, mini_repo["images"], mini_repo["labels"],
                  mini_repo["root"] / "datasets")
    stats = json.loads((out / "stats.json").read_text())
    assert stats["n_skipped_unreadable"] == 1
    assert stats["skipped_unreadable"][0]["path"].endswith("gen1_red-blue.png")
    # the corrupt file is absent from the dataset; the rest are present
    data = np.load(out / "data.npz", allow_pickle=True)
    assert "gen1_red-blue" not in set(data["source_name"].tolist())
    assert stats["n_images"] > 0

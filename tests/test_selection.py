from data_prep.config import DataConfig
from data_prep.selection import load_labels, load_records, gen_of, select_pokemon, relax_priority


def test_gen_of():
    assert gen_of("gen1_red-blue") == 1
    assert gen_of("gen9_scarlet-violet") == 9
    assert gen_of("official-artwork") == 0
    assert gen_of("default") == 0


def test_load_labels(mini_repo):
    labels = load_labels(mini_repo["labels"])
    assert abs(labels[1][0] - 444142 / (444142 + 1566929)) < 1e-6
    assert labels[1][1] == 444142 + 1566929


def test_load_records(mini_repo):
    recs = load_records(mini_repo["images"], 1, load_labels(mini_repo["labels"]))
    names = {r.source_name for r in recs}
    assert names == {"official-artwork", "home", "gen1_red-blue",
                     "gen9_scarlet-violet", "showdown"}
    by_name = {r.source_name: r for r in recs}
    assert by_name["gen9_scarlet-violet"].gen == 9
    assert by_name["official-artwork"].gen == 0
    assert by_name["showdown"].category == "animated"
    assert by_name["official-artwork"].total_votes == 444142 + 1566929


def _cfg(**sel):
    from data_prep.config import DataConfig
    return DataConfig.from_dict({"name": "d", "resolution": 64,
                                 "minimages": sel.pop("minimages", 1),
                                 "selection": sel})


def test_filter_by_category(mini_repo):
    cfg = _cfg(categories=["portrait"])
    recs = load_records(mini_repo["images"], 1, load_labels(mini_repo["labels"]))
    kept, report = select_pokemon(cfg, recs)
    assert {r.source_name for r in kept} == {"official-artwork", "home"}
    assert report["n_filtered"] == 2


def test_filter_gen_exclude_keeps_nongen(mini_repo):
    cfg = _cfg(gens={"exclude": [1]})
    recs = load_records(mini_repo["images"], 1, load_labels(mini_repo["labels"]))
    kept, _ = select_pokemon(cfg, recs)
    names = {r.source_name for r in kept}
    assert "gen1_red-blue" not in names
    assert "official-artwork" in names
    assert "gen9_scarlet-violet" in names


def test_minimages_relaxation_readds_excluded(mini_repo):
    cfg = _cfg(categories=["in-game"], minimages=4)
    recs = load_records(mini_repo["images"], 1, load_labels(mini_repo["labels"]))
    kept, report = select_pokemon(cfg, recs)
    assert len(kept) == 4
    assert "official-artwork" in {r.source_name for r in kept}
    assert report["n_relaxed"] == 2


def test_physical_shortfall_keeps_all(mini_repo):
    cfg = _cfg(minimages=5)
    recs = load_records(mini_repo["images"], 7, load_labels(mini_repo["labels"]))
    kept, report = select_pokemon(cfg, recs)
    assert len(kept) == 1
    assert report["padded"] is True


def test_relax_priority_orders_portrait_then_newgen_then_animated():
    from data_prep.selection import ImageRecord
    from pathlib import Path
    mk = lambda n, c, g: ImageRecord(1, n, c, g, Path("/x"), 0.0, 0)
    recs = [mk("showdown", "animated", 0), mk("gen1_x", "in-game", 1),
            mk("gen9_y", "in-game", 9), mk("home", "portrait", 0)]
    ordered = sorted(recs, key=relax_priority)
    assert [r.source_name for r in ordered] == ["home", "gen9_y", "gen1_x", "showdown"]


def test_load_records_skips_svg(tmp_path):
    import csv as _csv
    from PIL import Image as _Image
    from data_prep.selection import load_records
    folder = tmp_path / "images" / "1"
    folder.mkdir(parents=True)
    _Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(folder / "official-artwork.png")
    (folder / "dream_world.svg").write_text("<svg></svg>")
    with open(folder / "meta.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["filename", "category", "version", "source_url"])
        w.writeheader()
        w.writerow({"filename": "official-artwork.png", "category": "portrait",
                    "version": "official-artwork", "source_url": "x"})
        w.writerow({"filename": "dream_world.svg", "category": "portrait",
                    "version": "dream_world", "source_url": "x"})
    recs = load_records(tmp_path / "images", 1, {1: (0.5, 100)})
    names = {r.source_name for r in recs}
    assert "official-artwork" in names
    assert "dream_world" not in names


def _add_booru(images_dir, pid):
    import csv as _csv
    from PIL import Image as _Image
    booru = images_dir / str(pid) / "booru"
    booru.mkdir()
    _Image.new("RGB", (8, 8), (1, 2, 3)).save(booru / "00_999.jpg")
    with open(booru / "meta.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rank", "post_id", "score", "rating", "file_url"])
        w.writeheader(); w.writerow({"rank": 0, "post_id": 999, "score": 9,
                                     "rating": "safe", "file_url": "http://x/00_999.jpg"})


def test_load_records_includes_booru_tagged(mini_repo):
    _add_booru(mini_repo["images"], 1)
    recs = load_records(mini_repo["images"], 1, load_labels(mini_repo["labels"]))
    booru = [r for r in recs if r.category == "booru"]
    assert len(booru) == 1 and booru[0].source_name == "00_999"
    assert booru[0].path.parent.name == "booru"


def test_booru_excluded_by_default_included_on_opt_in(mini_repo):
    _add_booru(mini_repo["images"], 1)
    recs = load_records(mini_repo["images"], 1, load_labels(mini_repo["labels"]))
    # default (categories empty = all sprite categories) -> no booru
    default_cfg = DataConfig.from_dict({"name": "d", "resolution": 16})
    kept, _ = select_pokemon(default_cfg, recs)
    assert all(r.category != "booru" for r in kept)
    # sprite-only config -> no booru
    sprite_cfg = DataConfig.from_dict({"name": "d", "resolution": 16,
                                       "selection": {"categories": ["portrait"]}})
    kept, _ = select_pokemon(sprite_cfg, recs)
    assert all(r.category != "booru" for r in kept)
    # opt-in -> booru present
    booru_cfg = DataConfig.from_dict({"name": "d", "resolution": 16,
                                      "selection": {"categories": ["booru"]}})
    kept, _ = select_pokemon(booru_cfg, recs)
    assert [r.category for r in kept] == ["booru"]


def test_relaxation_never_adds_booru(mini_repo):
    _add_booru(mini_repo["images"], 1)
    recs = load_records(mini_repo["images"], 1, load_labels(mini_repo["labels"]))
    # impossible minimages with a name filter that matches nothing; relaxation
    # may re-add sprites but must never pull in the booru record
    cfg = DataConfig.from_dict({"name": "d", "resolution": 16, "minimages": 99,
                                "selection": {"names": {"include": ["nope"]}}})
    kept, _ = select_pokemon(cfg, recs)
    assert all(r.category != "booru" for r in kept)

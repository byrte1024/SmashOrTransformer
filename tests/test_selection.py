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

from data_prep.config import DataConfig
from data_prep.selection import load_labels, load_records, gen_of


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

import csv
from pathlib import Path
import data_prep.booru as booru
from data_prep.booru import (base_of, search_candidates, build_poke_index,
                             filter_posts, process_pokemon)


def test_base_of_collapses_forms_keeps_real_hyphens():
    assert base_of("charizard") == "charizard"
    assert base_of("deoxys-normal") == "deoxys"       # form suffix stripped
    assert base_of("giratina-altered") == "giratina"
    assert base_of("ho-oh") == "ho-oh"                # genuine name kept
    assert base_of("porygon-z") == "porygon-z"


def test_search_candidates_specials_and_fallbacks():
    assert search_candidates("mr-mime")[0] == "mr._mime"
    c = search_candidates("deoxys-normal")
    assert "deoxys-normal" in c and "deoxys" in c     # tries form then base


def _post(pid, score, tags, rating="safe", url="http://x/img.jpg"):
    return {"id": pid, "score": score, "tags": tags, "rating": rating, "file_url": url}


def test_filter_rejects_humans_groups_and_unsafe():
    idx = build_poke_index([(6, "charizard"), (4, "charmander"), (282, "gardevoir")])
    posts = [
        _post(1, 100, "charizard 1girl"),                  # human -> reject
        _post(2, 90, "charizard charmander"),              # 2 species -> reject
        _post(3, 80, "gardevoir", rating="questionable"),  # unsafe -> reject
        _post(4, 70, "charizard fire flying"),             # clean -> keep
        _post(5, 60, "charizard humanization"),            # humanized -> reject
        _post(6, 50, "gardevoir solo"),                    # clean single species -> keep
    ]
    kept = filter_posts(posts, idx, top=10)
    assert [p["id"] for p in kept] == [4, 6]


def test_filter_respects_top_and_score_order():
    idx = build_poke_index([(6, "charizard")])
    posts = [_post(i, 100 - i, "charizard") for i in range(20)]  # pre-sorted desc
    kept = filter_posts(posts, idx, top=5)
    assert len(kept) == 5 and [p["id"] for p in kept] == [0, 1, 2, 3, 4]


def test_process_pokemon_downloads_and_resumes(tmp_path, monkeypatch):
    idx = build_poke_index([(6, "charizard"), (4, "charmander")])
    fake_posts = [
        _post(11, 99, "charizard fire", url="http://x/a.jpg"),
        _post(12, 98, "charizard charmander"),                 # group -> filtered out
        _post(13, 97, "charizard flying", url="http://x/b.png"),
    ]
    calls = {"search": 0, "download": 0}

    def fake_search(session, tag, buffer):
        calls["search"] += 1
        return fake_posts

    def fake_download(session, url, dest):
        calls["download"] += 1
        Path(dest).write_bytes(b"\x89PNG fake")

    monkeypatch.setattr(booru, "search", fake_search)
    monkeypatch.setattr(booru, "download", fake_download)

    n = process_pokemon(None, 6, "charizard", str(tmp_path), idx, top=10,
                        buffer=100, min_score=0, sleep_dl=0, force=False)
    folder = tmp_path / "6" / "booru"
    assert n == 2                                    # group pic filtered, 2 kept
    assert len(list(folder.glob("[0-9]*.*"))) == 2
    rows = list(csv.DictReader(open(folder / "meta.csv")))
    assert len(rows) == 2 and rows[0]["post_id"] == "11"

    # resume: enough images present -> returns -1, no new downloads
    before = calls["download"]
    n2 = process_pokemon(None, 6, "charizard", str(tmp_path), idx, top=2,
                         buffer=100, min_score=0, sleep_dl=0, force=False)
    assert n2 == -1 and calls["download"] == before

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
    page0 = [
        _post(11, 99, "charizard fire", url="http://x/a.jpg"),
        _post(12, 98, "charizard charmander"),                 # group -> filtered out
        _post(13, 97, "charizard flying", url="http://x/b.png"),
    ]
    calls = {"search": 0, "fetch": 0}

    def fake_search(session, tag, limit, pid=0):
        calls["search"] += 1
        return page0 if pid == 0 else []                       # page 1 empty -> stop

    def fake_fetch(session, url):
        calls["fetch"] += 1
        return b"\x89PNG fake"

    monkeypatch.setattr(booru, "search", fake_search)
    monkeypatch.setattr(booru, "fetch_bytes", fake_fetch)

    n = process_pokemon(None, 6, "charizard", str(tmp_path), idx, top=10,
                        page_size=100, max_pages=10, min_score=0,
                        sleep_page=0, force=False, download_workers=1)
    folder = tmp_path / "6" / "booru"
    assert n == 2                                    # group pic filtered, 2 kept
    assert len(list(folder.glob("[0-9]*.*"))) == 2
    rows = list(csv.DictReader(open(folder / "meta.csv")))
    assert len(rows) == 2 and rows[0]["post_id"] == "11"

    before = calls["fetch"]
    n2 = process_pokemon(None, 6, "charizard", str(tmp_path), idx, top=2,
                         page_size=100, max_pages=10, min_score=0,
                         sleep_page=0, force=False, download_workers=1)
    assert n2 == -1 and calls["fetch"] == before     # resume skip, no new fetches


def test_collect_clean_auto_grows_across_pages(monkeypatch):
    from data_prep.booru import collect_clean
    idx = build_poke_index([(6, "charizard")])
    # page 0 is all humans (rejected); the clean ones live on page 1
    pages = {
        0: [_post(i, 100 - i, "charizard 1girl") for i in range(5)],
        1: [_post(100 + i, 50 - i, "charizard fire") for i in range(5)],
    }

    def fake_search(session, tag, limit, pid=0):
        return pages.get(pid, [])

    monkeypatch.setattr(booru, "search", fake_search)
    kept = collect_clean(None, ["charizard"], idx, top=3, page_size=5,
                         max_pages=10, sleep_page=0)
    assert len(kept) == 3                            # found by digging to page 1
    assert [p["id"] for p in kept] == [100, 101, 102]


def test_collect_clean_stops_at_max_pages(monkeypatch):
    from data_prep.booru import collect_clean
    idx = build_poke_index([(6, "charizard")])
    seen_pages = []

    def fake_search(session, tag, limit, pid=0):
        seen_pages.append(pid)
        return [_post(pid * 10, 5, "charizard 1boy")]   # always rejected (human)

    monkeypatch.setattr(booru, "search", fake_search)
    kept = collect_clean(None, ["charizard"], idx, top=10, page_size=1,
                         max_pages=3, sleep_page=0)
    assert kept == [] and seen_pages == [0, 1, 2]       # capped at 3 pages


def test_is_human_covers_count_and_humanoid_variants():
    from data_prep.booru import is_human
    for t in ["1girl", "1boy", "2girls", "3boys", "6+girls", "10+boys", "1other", "6+others"]:
        assert is_human({"charizard", t}), t            # all people-count variants
    for t in ["humanization", "humanized", "gijinka", "personification", "cosplay",
              "humanoid", "human", "multiple_girls"]:
        assert is_human({"charizard", t}), t            # humanoid keywords
    assert not is_human({"charizard", "fire", "flying", "solo"})   # solo pokemon kept
    assert not is_human({"gardevoir", "green_hair", "red_eyes"})   # humanoid POKEMON kept
    assert not is_human({"charizard", "furry", "anthro"})          # anthro art kept now


def test_delete_booru_removes_only_selected(tmp_path):
    from data_prep.booru import delete_booru
    for pid in (1, 2):
        d = tmp_path / str(pid) / "booru"
        d.mkdir(parents=True)
        (d / "00_1.jpg").write_bytes(b"x")
    # delete only pokemon 1's booru folder
    assert delete_booru(str(tmp_path), [1]) == 1
    assert not (tmp_path / "1" / "booru").exists()
    assert (tmp_path / "2" / "booru").exists()        # untouched
    assert (tmp_path / "1").exists()                  # only booru/ removed, sprites kept
    # deleting again is a no-op (folder already gone)
    assert delete_booru(str(tmp_path), [1]) == 0


def test_process_pokemon_parallel_downloads_preserve_order(tmp_path, monkeypatch):
    import threading
    idx = build_poke_index([(6, "charizard")])
    posts = [_post(10 + i, 50 - i, "charizard fire", url=f"http://x/{i}.jpg") for i in range(6)]

    def fake_search(session, tag, limit, pid=0):
        return posts if pid == 0 else []

    lock = threading.Lock(); seen = []
    def fake_fetch(session, url):
        with lock:
            seen.append(url)
        return b"x"

    monkeypatch.setattr(booru, "search", fake_search)
    monkeypatch.setattr(booru, "fetch_bytes", fake_fetch)
    n = process_pokemon(None, 6, "charizard", str(tmp_path), idx, top=6, page_size=100,
                        max_pages=10, min_score=0, sleep_page=0, force=False,
                        download_workers=4)
    assert n == 6 and len(seen) == 6
    rows = list(csv.DictReader(open(tmp_path / "6" / "booru" / "meta.csv")))
    # meta is rank-sorted even though downloads ran concurrently
    assert [int(r["rank"]) for r in rows] == [0, 1, 2, 3, 4, 5]
    assert [int(r["post_id"]) for r in rows] == [10, 11, 12, 13, 14, 15]


def test_process_pokemon_backfills_dead_links(tmp_path, monkeypatch):
    idx = build_poke_index([(6, "charizard")])
    # 8 clean candidates; even-numbered urls are dead (404), odd succeed
    posts = [_post(10 + i, 50 - i, "charizard fire", url=f"http://x/{i}.jpg") for i in range(8)]

    def fake_search(session, tag, limit, pid=0):
        return posts if pid == 0 else []

    def fake_fetch(session, url):
        i = int(url.rsplit("/", 1)[1].split(".")[0])
        if i % 2 == 0:
            raise RuntimeError("404 dead link")
        return b"x"

    monkeypatch.setattr(booru, "search", fake_search)
    monkeypatch.setattr(booru, "fetch_bytes", fake_fetch)
    # want top=3 -> must skip the dead even-url posts and backfill from odd ones
    n = process_pokemon(None, 6, "charizard", str(tmp_path), idx, top=3, page_size=100,
                        max_pages=10, min_score=0, sleep_page=0, force=False,
                        download_workers=1)
    assert n == 3
    rows = list(csv.DictReader(open(tmp_path / "6" / "booru" / "meta.csv")))
    # surviving = odd-url posts 11(url1), 13(url3), 15(url5); contiguous ranks
    assert [int(r["post_id"]) for r in rows] == [11, 13, 15]
    assert [int(r["rank"]) for r in rows] == [0, 1, 2]

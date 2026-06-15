"""Download the top-ranked Safebooru fan-art for each Pokemon into
images/{id}/booru/, filtering out human-implying tags and group pictures.

For each Pokemon we query Safebooru sorted by score, pull a buffer, then keep
the top N posts that: are safe-rated, contain no human tag (1girl, 1boy,
humanization, ...), and tag only ONE Pokemon species (no group/crossover art).

Network calls (`search`, `download`) are module-level so tests can monkeypatch
them; everything else is pure filtering logic. Be polite: the driver sleeps
between requests and resumes (skips Pokemon already populated).
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import requests
from tqdm import tqdm

API = "https://safebooru.org/index.php"
UA = {"User-Agent": "Mozilla/5.0 (SmashOrTransformer dataset builder)"}

# tags that imply a human/humanized subject in the picture
HUMAN_TAGS = {
    "1girl", "1boy", "2girls", "2boys", "3girls", "3boys", "4girls", "4boys",
    "5girls", "5boys", "6+girls", "6+boys", "multiple_girls", "multiple_boys",
    "human", "humanization", "personification", "gijinka", "cosplay", "1other",
}
SAFE_RATINGS = {"safe", "general"}

# PokeAPI default-form suffixes -> stripped to the base species for the booru tag
_FORM_SUFFIXES = {
    "normal", "altered", "land", "plant", "incarnate", "standard", "ordinary",
    "aria", "average", "male", "female", "red", "shield", "disguised", "amped",
    "full-belly", "family-of-four", "single-strike", "zero", "curly", "two-segment",
    "ice", "midday", "solo", "baile", "natural",
}
# genuinely hyphenated/punctuated names -> their Safebooru tag
_SPECIAL = {
    "ho-oh": "ho-oh", "porygon-z": "porygon-z", "jangmo-o": "jangmo-o",
    "hakamo-o": "hakamo-o", "kommo-o": "kommo-o", "mr-mime": "mr._mime",
    "mime-jr": "mime_jr.", "mr-rime": "mr._rime", "type-null": "type:_null",
    "farfetchd": "farfetch'd", "sirfetchd": "sirfetch'd",
    "nidoran-f": "nidoran", "nidoran-m": "nidoran", "tapu-koko": "tapu_koko",
    "tapu-lele": "tapu_lele", "tapu-bulu": "tapu_bulu", "tapu-fini": "tapu_fini",
}


def base_of(name: str) -> str:
    """Canonical species key for grouping (collapses alternate forms)."""
    if name in _SPECIAL:
        return name
    head, _, tail = name.partition("-")
    return name if (tail and tail not in _FORM_SUFFIXES and "-" in name) else head


def search_candidates(name: str) -> list[str]:
    """Booru tag candidates to try, in priority order."""
    cands = []
    if name in _SPECIAL:
        cands.append(_SPECIAL[name])
    cands.append(name)
    if "-" in name:
        cands.append(name.partition("-")[0])   # strip form suffix
        cands.append(name.replace("-", "_"))
    out = []
    for c in cands:
        if c and c not in out:
            out.append(c)
    return out


def load_names(path) -> list[tuple[int, str]]:
    with open(path, newline="") as f:
        return [(int(r["id"]), r["name"]) for r in csv.DictReader(f)]


def build_poke_index(names) -> dict[str, str]:
    """Map every recognizable Pokemon tag -> its base species key."""
    idx: dict[str, str] = {}
    for _id, name in names:
        b = base_of(name)
        for tag in search_candidates(name):
            idx.setdefault(tag, b)
    return idx


def filter_posts(posts, poke_index, top, min_score=0) -> list[dict]:
    """Keep the top `top` posts that are safe, human-free, and single-species.
    `posts` must already be score-sorted (Safebooru sort:score)."""
    kept = []
    for p in posts:
        if (p.get("rating") or "safe") not in SAFE_RATINGS:
            continue
        if not p.get("file_url"):
            continue
        if int(p.get("score") or 0) < min_score:
            continue
        tags = set((p.get("tags") or "").split())
        if tags & HUMAN_TAGS:
            continue
        species = {poke_index[t] for t in tags if t in poke_index}
        if len(species) > 1:                      # crossover / group picture
            continue
        kept.append(p)
        if len(kept) >= top:
            break
    return kept


# --- network (monkeypatched in tests) ------------------------------------- #
def search(session, tag, buffer) -> list[dict]:
    params = {"page": "dapi", "s": "post", "q": "index", "json": "1",
              "limit": buffer, "tags": f"{tag} sort:score"}
    r = session.get(API, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    if not r.text.strip():
        return []
    data = r.json()
    return data if isinstance(data, list) else data.get("post", [])


def download(session, url, dest) -> None:
    r = session.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    Path(dest).write_bytes(r.content)


def process_pokemon(session, pid, name, images_dir, poke_index, top, buffer,
                    min_score, sleep_dl, force) -> int:
    folder = Path(images_dir) / str(pid) / "booru"
    existing = list(folder.glob("[0-9]*.*")) if folder.exists() else []
    if existing and len(existing) >= top and not force:
        return -1                                  # already done -> resume skip
    folder.mkdir(parents=True, exist_ok=True)

    posts = []
    for tag in search_candidates(name):
        posts = search(session, tag, buffer)
        if posts:
            break
    kept = filter_posts(posts, poke_index, top, min_score)

    rows = []
    for rank, p in enumerate(kept):
        url = p["file_url"]
        ext = url.rsplit(".", 1)[-1].split("?")[0][:4]
        dest = folder / f"{rank:02d}_{p['id']}.{ext}"
        try:
            download(session, url, dest)
        except Exception as e:
            print(f"  [{pid}] download failed {url}: {e}")
            continue
        rows.append({"rank": rank, "post_id": p["id"], "score": p.get("score"),
                     "rating": p.get("rating"), "file_url": url})
        if sleep_dl:
            time.sleep(sleep_dl)
    with open(folder / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "post_id", "score", "rating", "file_url"])
        w.writeheader(); w.writerows(rows)
    return len(rows)


def run(names_csv="pokemon_names.csv", images_dir="images", top=10, buffer=100,
        min_score=0, ids=None, limit=None, sleep=1.0, sleep_dl=0.25, force=False):
    names = load_names(names_csv)
    poke_index = build_poke_index(names)
    if ids:
        idset = set(ids)
        names = [(i, n) for i, n in names if i in idset]
    if limit:
        names = names[:limit]

    session = requests.Session()
    got, skipped, empty = 0, 0, 0
    for pid, name in tqdm(names, desc="booru", unit="pkmn"):
        n = process_pokemon(session, pid, name, images_dir, poke_index, top,
                            buffer, min_score, sleep_dl, force)
        if n == -1:
            skipped += 1
            continue
        got += n
        if n == 0:
            empty += 1
            print(f"  [{pid}] {name}: no clean results")
        if sleep:
            time.sleep(sleep)
    print(f"\nDone. downloaded {got} images | {skipped} skipped (resume) | "
          f"{empty} pokemon with 0 clean results")


def main(argv=None):
    p = argparse.ArgumentParser(description="Download top Safebooru fan-art per Pokemon.")
    p.add_argument("--names", default="pokemon_names.csv")
    p.add_argument("--images", default="images")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--buffer", type=int, default=100)
    p.add_argument("--min-score", type=int, default=0)
    p.add_argument("--ids", default=None, help="comma-separated dex ids (e.g. 6,282)")
    p.add_argument("--limit", type=int, default=None, help="only the first N pokemon")
    p.add_argument("--sleep", type=float, default=1.0, help="seconds between queries")
    p.add_argument("--sleep-dl", type=float, default=0.25, help="seconds between downloads")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args(argv)
    ids = [int(x) for x in args.ids.split(",")] if args.ids else None
    run(names_csv=args.names, images_dir=args.images, top=args.top, buffer=args.buffer,
        min_score=args.min_score, ids=ids, limit=args.limit, sleep=args.sleep,
        sleep_dl=args.sleep_dl, force=args.force)


if __name__ == "__main__":
    main()

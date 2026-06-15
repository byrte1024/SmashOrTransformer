#!/usr/bin/env python3
"""Download per-pokemon images from PokeAPI into images/{id}/ and write a
meta.csv in each folder cataloguing what was fetched.

Selection: front-facing, true-color sprites only -- the images that match
what smash/pass voters actually judged. We skip back views, shiny recolors,
tiny menu icons, and the gray/transparent gen-1/2 duplicates.

Per pokemon we keep:
  - official-artwork  (Sugimori box art)      [portrait]
  - home              (Pokemon HOME 3D render) [portrait]
  - dream_world       (vector art, .svg)       [portrait]
  - showdown          (battle animation, .gif) [animated]
  - default           (current default sprite) [in-game]
  - one front_default per game/generation      [in-game]

Each folder gets meta.csv: filename,category,version,source_url
Re-running is safe: existing files are skipped (resumable).
"""
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://pokeapi.co/api/v2/pokemon"
IMG_DIR = "images"
WORKERS = 8
MAX_ID = 1025
UA = {"User-Agent": "Mozilla/5.0 (SmashOrTransformer dataset builder)"}

GEN_MAP = {
    "generation-i": "gen1", "generation-ii": "gen2", "generation-iii": "gen3",
    "generation-iv": "gen4", "generation-v": "gen5", "generation-vi": "gen6",
    "generation-vii": "gen7", "generation-viii": "gen8", "generation-ix": "gen9",
}


def http_get(url: str, binary: bool = False, retries: int = 3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            return data if binary else json.loads(data.decode())
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def collect_sprites(sprites: dict) -> list[dict]:
    """Walk the sprite tree, return chosen entries as
    {path, url, filename, category, version}."""
    leaves = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + [k])
        elif isinstance(node, str) and node.startswith("http"):
            leaves.append((path, node))

    walk(sprites, [])

    chosen = []
    seen_names = set()
    for path, url in leaves:
        p = "/".join(path)
        # keep only front-facing, non-shiny, real-color, non-icon images
        if not p.endswith("front_default"):
            continue
        if "shiny" in p or "back" in p or "icons" in p or "female" in p:
            continue

        ext = os.path.splitext(url.split("?")[0])[1] or ".png"

        if p == "front_default":
            name, category, version = "default", "in-game", "default"
        elif p.startswith("other/"):
            kind = path[1]  # official-artwork | home | dream_world | showdown
            name = kind
            category = "animated" if kind == "showdown" else "portrait"
            version = kind
        elif p.startswith("versions/"):
            gen = GEN_MAP.get(path[1], path[1])
            game = path[2]
            animated = "animated" in path
            name = f"{gen}_{game}" + ("_animated" if animated else "")
            category = "animated" if animated else "in-game"
            version = game
        else:
            continue

        if name in seen_names:
            continue
        seen_names.add(name)
        chosen.append({
            "filename": name + ext,
            "category": category,
            "version": version,
            "source_url": url,
        })
    return chosen


def process(pid: int) -> tuple[int, int]:
    folder = os.path.join(IMG_DIR, str(pid))
    os.makedirs(folder, exist_ok=True)
    try:
        data = http_get(f"{API}/{pid}")
    except Exception as e:
        print(f"  [{pid}] API error: {e}", file=sys.stderr)
        return pid, 0

    entries = collect_sprites(data["sprites"])
    saved = 0
    for e in entries:
        dest = os.path.join(folder, e["filename"])
        if os.path.exists(dest):
            saved += 1
            continue
        try:
            blob = http_get(e["source_url"], binary=True)
            with open(dest, "wb") as f:
                f.write(blob)
            saved += 1
        except Exception as ex:
            print(f"  [{pid}] failed {e['filename']}: {ex}", file=sys.stderr)

    # write meta.csv for this pokemon
    with open(os.path.join(folder, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["filename", "category", "version", "source_url"]
        )
        w.writeheader()
        w.writerows(entries)

    return pid, saved


def main() -> int:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_ID
    ids = range(start, end + 1)
    total_imgs = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, (pid, saved) in enumerate(pool.map(process, ids), 1):
            total_imgs += saved
            if i % 25 == 0 or pid == end:
                print(f"  {pid} done ({i}/{len(ids)}), {total_imgs} imgs so far",
                      file=sys.stderr)
    print(f"Done. {total_imgs} images across ids {start}-{end}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

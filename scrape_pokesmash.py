#!/usr/bin/env python3
"""Scrape per-pokemon smash/pass aggregate counts from pokesmash.xyz's
public Firebase RTDB and write them to a CSV.

Data model (Firebase Realtime Database, public per-node read):
    /pokemon/{id}/smashCount
    /pokemon/{id}/passCount

The label of interest is smash_pct = smashCount / (smashCount + passCount).
We read the count-only paths so each request is a few bytes (the full node
also contains every voter's username, which we don't need).
"""
import csv
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DB = "https://poke-smash-default-rtdb.firebaseio.com"
MAX_ID = 1025          # national dex through Gen 9 (Pecharunt)
WORKERS = 8            # be polite to a hobby project's free-tier DB
OUT = "pokesmash_votes.csv"


def get_int(path: str) -> int | None:
    """GET a single RTDB value; return int or None (for null/missing)."""
    url = f"{DB}/{path}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        val = json.loads(resp.read().decode())
    if val is None:
        return None
    return int(val)


def fetch(pid: int) -> dict:
    smash = get_int(f"pokemon/{pid}/smashCount")
    passc = get_int(f"pokemon/{pid}/passCount")
    smash = smash or 0
    passc = passc or 0
    total = smash + passc
    pct = round(100 * smash / total, 2) if total else ""
    return {
        "id": pid,
        "smash_count": smash,
        "pass_count": passc,
        "total_votes": total,
        "smash_pct": pct,
    }


def main() -> int:
    ids = range(1, MAX_ID + 1)
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, row in enumerate(pool.map(fetch, ids), 1):
            rows.append(row)
            if i % 50 == 0 or i == MAX_ID:
                print(f"  {i}/{MAX_ID} fetched", file=sys.stderr)

    rows.sort(key=lambda r: r["id"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "smash_count", "pass_count", "total_votes", "smash_pct"],
        )
        w.writeheader()
        w.writerows(rows)

    voted = [r for r in rows if r["total_votes"] > 0]
    print(f"Wrote {len(rows)} rows to {OUT} ({len(voted)} with votes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

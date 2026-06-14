from __future__ import annotations
import csv
import re
from dataclasses import dataclass
from pathlib import Path

_GEN_RE = re.compile(r"^gen(\d+)_")


def gen_of(source_name: str) -> int:
    m = _GEN_RE.match(source_name)
    return int(m.group(1)) if m else 0


@dataclass
class ImageRecord:
    pokemon_id: int
    source_name: str
    category: str
    gen: int
    path: Path
    smash_pct: float
    total_votes: int


def load_labels(labels_csv) -> dict[int, tuple[float, int]]:
    out: dict[int, tuple[float, int]] = {}
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            pid = int(row["id"])
            smash = int(row["smash_count"])
            total = int(row["total_votes"])
            out[pid] = (smash / total, total)
    return out


def load_records(images_dir, pokemon_id: int,
                 labels: dict[int, tuple[float, int]]) -> list[ImageRecord]:
    folder = Path(images_dir) / str(pokemon_id)
    meta = folder / "meta.csv"
    if not meta.exists():
        return []
    smash, votes = labels.get(pokemon_id, (0.0, 0))
    recs: list[ImageRecord] = []
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            name = row["filename"].rsplit(".", 1)[0]
            recs.append(ImageRecord(
                pokemon_id=pokemon_id, source_name=name,
                category=row["category"], gen=gen_of(name),
                path=folder / row["filename"], smash_pct=smash, total_votes=votes,
            ))
    return recs

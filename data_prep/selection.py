from __future__ import annotations
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .config import DataConfig

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
    _UNSUPPORTED = {".svg"}
    recs: list[ImageRecord] = []
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            if Path(fname).suffix.lower() in _UNSUPPORTED:
                continue
            name = fname.rsplit(".", 1)[0]
            recs.append(ImageRecord(
                pokemon_id=pokemon_id, source_name=name,
                category=row["category"], gen=gen_of(name),
                path=folder / fname, smash_pct=smash, total_votes=votes,
            ))
    return recs


def relax_priority(rec: "ImageRecord") -> tuple[int, int]:
    """Lower sorts first: portraits, then newest in-game gen, then animated."""
    if rec.category == "portrait":
        return (0, 0)
    if rec.category == "in-game":
        return (1, -rec.gen)
    return (2, 0)


def _passes(rec: "ImageRecord", cfg: DataConfig) -> bool:
    s = cfg.selection
    if s.categories and rec.category not in s.categories:
        return False
    if s.names.include and rec.source_name not in s.names.include:
        return False
    if rec.source_name in s.names.exclude:
        return False
    if rec.gen != 0:
        if s.gens.include and rec.gen not in s.gens.include:
            return False
        if rec.gen in s.gens.exclude:
            return False
    return True


def select_pokemon(cfg: DataConfig,
                   records: list["ImageRecord"]) -> tuple[list["ImageRecord"], dict]:
    kept = [r for r in records if _passes(r, cfg)]
    excluded = [r for r in records if r not in kept]
    n_filtered = len(kept)
    n_relaxed = 0
    if len(kept) < cfg.minimages and excluded:
        for r in sorted(excluded, key=relax_priority):
            if len(kept) >= cfg.minimages:
                break
            kept.append(r)
            n_relaxed += 1
    padded = len(kept) < cfg.minimages
    report = {"n_source": len(records), "n_filtered": n_filtered,
              "n_relaxed": n_relaxed, "n_kept": len(kept), "padded": padded}
    return kept, report

import csv
import numpy as np
import pytest
from PIL import Image


def _write_sprite(path, size, color):
    """Write an RGBA sprite: opaque colored disk on transparent background."""
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    cx, cy, r = w / 2, h / 2, min(w, h) / 2 - 1
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    arr[mask] = (*color, 255)
    Image.fromarray(arr, "RGBA").save(path)


# id -> list of (filename, category, version, size, color)
_SPEC = {
    1: [
        ("official-artwork.png", "portrait", "official-artwork", (60, 60), (50, 200, 50)),
        ("home.png", "portrait", "home", (48, 48), (40, 180, 40)),
        ("gen1_red-blue.png", "in-game", "red-blue", (32, 32), (30, 160, 30)),
        ("gen9_scarlet-violet.png", "in-game", "scarlet-violet", (40, 40), (60, 210, 60)),
        ("showdown.gif", "animated", "showdown", (36, 36), (45, 190, 45)),
    ],
    4: [
        ("official-artwork.png", "portrait", "official-artwork", (60, 60), (210, 90, 40)),
        ("gen3_emerald.png", "in-game", "emerald", (34, 34), (200, 80, 30)),
    ],
    7: [
        ("official-artwork.png", "portrait", "official-artwork", (60, 60), (40, 120, 220)),
    ],
}

_LABELS = {  # id -> (smash_count, pass_count)
    1: (444142, 1566929),
    4: (361681, 1517691),
    7: (300000, 700000),
}


@pytest.fixture
def mini_repo(tmp_path):
    images = tmp_path / "images"
    for pid, files in _SPEC.items():
        folder = images / str(pid)
        folder.mkdir(parents=True)
        rows = []
        for fname, cat, ver, size, color in files:
            p = folder / fname
            if fname.endswith(".gif"):
                # animated: 2-frame gif, first frame is the sprite
                w, h = size
                a = np.zeros((h, w, 4), dtype=np.uint8); a[:, :] = (*color, 255)
                b = np.zeros((h, w, 4), dtype=np.uint8); b[:, :] = (0, 0, 0, 255)
                frames = [Image.fromarray(a, "RGBA").convert("P"),
                          Image.fromarray(b, "RGBA").convert("P")]
                frames[0].save(p, save_all=True, append_images=frames[1:], loop=0)
            else:
                _write_sprite(p, size, color)
            rows.append({"filename": fname, "category": cat, "version": ver,
                         "source_url": "http://example/" + fname})
        with open(folder / "meta.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "category", "version", "source_url"])
            w.writeheader(); w.writerows(rows)

    labels = tmp_path / "pokesmash_votes.csv"
    with open(labels, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "smash_count", "pass_count", "total_votes", "smash_pct"])
        for pid, (s, p) in _LABELS.items():
            tot = s + p
            w.writerow([pid, s, p, tot, round(100 * s / tot, 2)])

    return {"root": tmp_path, "images": images, "labels": labels}

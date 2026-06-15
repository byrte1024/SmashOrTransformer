import csv
import numpy as np
from PIL import Image
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run as train_run
from model.calibrate import run as calib_run
from model.compare_sources import run as compare_run


def _add_booru(images_dir, pid, n):
    folder = images_dir / str(pid) / "booru"
    folder.mkdir(parents=True)
    rows = []
    for i in range(n):
        post = 2000 + i
        Image.fromarray(np.full((30, 30, 3), (30, 90 + i, 150), np.uint8), "RGB").save(
            folder / f"{i:02d}_{post}.jpg")
        rows.append({"rank": i, "post_id": post, "score": 9, "rating": "safe",
                     "file_url": f"http://x/{i:02d}_{post}.jpg"})
    with open(folder / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "post_id", "score", "rating", "file_url"])
        w.writeheader(); w.writerows(rows)


def test_compare_sources_writes_csv(mini_repo, tmp_path):
    _add_booru(mini_repo["images"], 1, 4)
    _add_booru(mini_repo["images"], 4, 3)
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "selection": {"categories": ["portrait", "in-game", "booru"]},
                  "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    run_dir = train_run(TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "cmp", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"}), pretrained=False)
    calib_run(run_dir / "checkpoints" / "best.pt", device="cpu", batch_size=4, num_workers=0)

    path = compare_run(run_dir / "checkpoints" / "best.pt", dataset_dir=str(out),
                       out_dir=str(tmp_path / "res"), device="cpu", batch_size=4,
                       num_workers=0, names=None)
    rows = list(csv.DictReader(open(path)))
    assert {r["pokemon_id"] for r in rows} == {"1", "4", "7"}
    cols = set(rows[0].keys())
    assert {"portrait_pct", "ingame_pct", "sprite_pct", "booru_pct", "all_pct",
            "disagreement", "true_pct", "split"} <= cols
    # pokemon 1 has all three sources -> portrait/ingame/booru all populated
    r1 = {r["pokemon_id"]: r for r in rows}["1"]
    assert r1["portrait_pct"] and r1["ingame_pct"] and r1["booru_pct"] and r1["all_pct"]
    # pokemon 7 has no booru -> booru_pct empty, but portrait/all present
    r7 = {r["pokemon_id"]: r for r in rows}["7"]
    assert r7["booru_pct"] == "" and r7["all_pct"]
    # sorted by disagreement descending
    dis = [float(r["disagreement"]) for r in rows]
    assert dis == sorted(dis, reverse=True)

import csv
import numpy as np
from PIL import Image
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run as train_run
from model.calibrate import run as calib_run
from model.score_booru import run as booru_run, _gather


def _trained(mini_repo, tmp_path):
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    tcfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "sb", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    run_dir = train_run(tcfg, pretrained=False)
    calib_run(run_dir / "checkpoints" / "best.pt", device="cpu", batch_size=4)
    return run_dir


def _add_booru(images_dir, pid, n):
    folder = images_dir / str(pid) / "booru"
    folder.mkdir(parents=True)
    for i in range(n):
        arr = np.zeros((30, 30, 3), dtype=np.uint8)
        arr[i:25, i:25] = (40 * i % 255, 90, 150)
        Image.fromarray(arr, "RGB").save(folder / f"{i:02d}_{1000 + i}.jpg")


def test_gather_finds_booru_folders(mini_repo):
    _add_booru(mini_repo["images"], 1, 3)
    _add_booru(mini_repo["images"], 4, 2)
    entries = _gather(str(mini_repo["images"]))
    found = {pid: len(imgs) for pid, imgs, _ in entries}
    assert found == {1: 3, 4: 2}                       # only pokemon with booru/
    # official-artwork path resolved for pokemon 1
    oa = dict((pid, oa) for pid, _, oa in entries)[1]
    assert oa is not None and oa.name == "official-artwork.png"


def test_score_booru_writes_views_and_csvs(mini_repo, tmp_path):
    run_dir = _trained(mini_repo, tmp_path)
    _add_booru(mini_repo["images"], 1, 4)
    _add_booru(mini_repo["images"], 7, 2)
    out = booru_run(run_dir / "checkpoints" / "best.pt", images_dir=str(mini_repo["images"]),
                    out_dir=str(tmp_path / "bres"), device="cpu", display_res=48,
                    tile_res=32, names=None)
    # all_avg + matrix image per pokemon with booru art
    assert (out / "all_avg" / "0001.png").exists() and (out / "all_avg" / "0007.png").exists()
    assert (out / "matrix" / "0001.png").exists() and (out / "matrix" / "0007.png").exists()

    imgs = list(csv.DictReader(open(out / "booru_scores.csv")))
    assert len(imgs) == 6                              # 4 + 2 booru images
    assert {"pokemon_id", "name", "file", "raw_pct", "calibrated_pct", "decision"} == set(imgs[0].keys())

    agg = list(csv.DictReader(open(out / "booru_rankings.csv")))
    assert {int(r["pokemon_id"]) for r in agg} == {1, 7}
    assert int(dict((int(r["pokemon_id"]), r) for r in agg)[1]["n_booru"]) == 4
    pcts = [float(r["avg_pct"]) for r in agg]
    assert pcts == sorted(pcts, reverse=True)          # ranked by avg

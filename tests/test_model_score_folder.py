import csv
import numpy as np
from PIL import Image
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run as train_run
from model.calibrate import run as calib_run
from model.score_folder import run as score_run, _gather


def _trained(mini_repo, tmp_path):
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    tcfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "sf", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    run_dir = train_run(tcfg, pretrained=False)
    calib_run(run_dir / "checkpoints" / "best.pt", device="cpu", batch_size=4)
    return run_dir


def _make_folder(tmp_path):
    folder = tmp_path / "ext"
    folder.mkdir()
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        arr = np.zeros((40, 40, 4), dtype=np.uint8)
        arr[5:35, 5:35] = (*color, 255)
        Image.fromarray(arr, "RGBA").save(folder / f"thing_{i}.png")
    return folder


def test_gather_finds_images(tmp_path):
    folder = _make_folder(tmp_path)
    assert len(_gather(str(folder))) == 3


def test_score_folder_writes_portraits_and_csv(mini_repo, tmp_path):
    run_dir = _trained(mini_repo, tmp_path)
    folder = _make_folder(tmp_path)
    out = score_run(run_dir / "checkpoints" / "best.pt", str(folder),
                    out_dir=str(tmp_path / "out"), device="cpu", display_res=48)
    assert len(list(out.glob("*.png"))) == 3
    rows = list(csv.DictReader(open(out / "scores.csv")))
    assert len(rows) == 3
    assert {"file", "name", "raw_pct", "calibrated_pct", "decision"} == set(rows[0].keys())
    # ranked descending and decisions valid
    pcts = [float(r["calibrated_pct"]) for r in rows]
    assert pcts == sorted(pcts, reverse=True)
    assert all(r["decision"] in ("SMASH", "PASS") for r in rows)


def test_names_csv_labels(mini_repo, tmp_path):
    run_dir = _trained(mini_repo, tmp_path)
    folder = _make_folder(tmp_path)
    names = tmp_path / "names.csv"
    with open(names, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem", "name"]); w.writerow(["thing_0", "Reddo"])
    out = score_run(run_dir / "checkpoints" / "best.pt", str(folder),
                    out_dir=str(tmp_path / "out2"), device="cpu", display_res=48,
                    names=str(names))
    rows = {r["file"]: r for r in csv.DictReader(open(out / "scores.csv"))}
    assert rows["thing_0.png"]["name"] == "Reddo"

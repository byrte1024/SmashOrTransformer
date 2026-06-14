import csv
import json
from model.stats import RunLogger


def _record(epoch, spear):
    return {"epoch": epoch, "phase": "frozen", "train_loss": 0.5,
            "val_loss": 0.4, "val_spearman": spear, "val_mae": 0.1,
            "lr_head": 1e-3, "lr_backbone": 2e-5, "grad_norm": 1.2,
            "epoch_seconds": 3.0, "n_train_samples": 10, "n_val_pokemon": 1}


def test_runlogger_writes_all_artifacts(tmp_path):
    rl = RunLogger(out_dir=tmp_path, run_name="run1")
    rl.write_config({"backbone": "vit_small_patch16_224"})
    for e, s in [(1, 0.2), (2, 0.6)]:
        rl.log_epoch(_record(e, s))
        rl.save_predictions(e, [1, 2], [0.3, 0.7], [0.4, 0.6])
        rl.save_checkpoint({"epoch": e}, e, is_best=(e == 2))
    rl.finalize({"best_epoch": 2, "best_spearman": 0.6})

    run = tmp_path / "run1"
    assert (run / "config.json").exists()
    assert (run / "summary.json").exists()
    assert (run / "checkpoints" / "epoch_001.pt").exists()
    assert (run / "checkpoints" / "epoch_002.pt").exists()
    assert (run / "checkpoints" / "best.pt").exists()
    assert (run / "checkpoints" / "last.pt").exists()
    assert (run / "predictions" / "epoch_001.csv").exists()
    rows = list(csv.DictReader(open(run / "history.csv")))
    assert len(rows) == 2
    assert rows[1]["val_spearman"] == "0.6"
    lines = [json.loads(x) for x in open(run / "history.jsonl") if x.strip()]
    assert len(lines) == 2


def test_predictions_csv_content(tmp_path):
    rl = RunLogger(out_dir=tmp_path, run_name="r")
    rl.save_predictions(1, [5, 9], [0.1, 0.2], [0.15, 0.25])
    rows = list(csv.DictReader(open(tmp_path / "r" / "predictions" / "epoch_001.csv")))
    assert rows[0]["pokemon_id"] == "5"
    assert rows[0]["y_true"] == "0.1"
    assert rows[0]["y_pred"] == "0.15"


def test_save_checkpoint_can_skip_epoch_file(tmp_path):
    rl = RunLogger(out_dir=tmp_path, run_name="s")
    rl.save_checkpoint({"e": 1}, 1, is_best=True, save_epoch=False)
    cd = tmp_path / "s" / "checkpoints"
    assert not (cd / "epoch_001.pt").exists()
    assert (cd / "last.pt").exists() and (cd / "best.pt").exists()

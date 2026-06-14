from __future__ import annotations
import csv
import json
from pathlib import Path
import torch

_HISTORY_FIELDS = ["epoch", "phase", "train_loss", "val_loss", "val_spearman",
                   "val_mae", "lr_head", "lr_backbone", "grad_norm",
                   "epoch_seconds", "n_train_samples", "n_val_pokemon"]


class RunLogger:
    """Writes config, per-epoch history/predictions/checkpoints, and summary.

    All epoch checkpoints are kept and never auto-deleted.
    """

    def __init__(self, out_dir, run_name: str):
        self.run = Path(out_dir) / run_name
        self.ckpt_dir = self.run / "checkpoints"
        self.pred_dir = self.run / "predictions"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self._history_started = False

    def write_config(self, config: dict) -> None:
        (self.run / "config.json").write_text(json.dumps(config, indent=2))

    def log_epoch(self, record: dict) -> None:
        csv_path = self.run / "history.csv"
        write_header = not self._history_started and not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow({k: record.get(k, "") for k in _HISTORY_FIELDS})
        with open(self.run / "history.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        self._history_started = True

    def save_predictions(self, epoch: int, ids, y_true, y_pred) -> None:
        path = self.pred_dir / f"epoch_{epoch:03d}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pokemon_id", "y_true", "y_pred"])
            for i, yt, yp in zip(ids, y_true, y_pred):
                w.writerow([int(i), float(yt), float(yp)])

    def save_checkpoint(self, state: dict, epoch: int, is_best: bool,
                        save_epoch: bool = True) -> None:
        if save_epoch:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pt")
        torch.save(state, self.ckpt_dir / "last.pt")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pt")

    def finalize(self, summary: dict) -> None:
        (self.run / "summary.json").write_text(json.dumps(summary, indent=2))

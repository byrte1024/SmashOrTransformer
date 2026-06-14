from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainConfig:
    dataset_dir: str
    run_name: str
    resolution: int
    out_dir: str = "runs"
    backbone: str = "vit_small_patch16_224"
    epochs: int = 30
    batch_size: int = 64
    freeze_epochs: int = 3
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    dropout: float = 0.1
    amp: bool = True
    grad_clip: float = 1.0
    num_workers: int = 8
    seed: int = 0
    device: str = "cuda"
    save_every_epoch: bool = True

    @staticmethod
    def from_dict(d: dict) -> "TrainConfig":
        known = TrainConfig.__dataclass_fields__.keys()
        cfg = TrainConfig(**{k: v for k, v in d.items() if k in known})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not (0 <= self.freeze_epochs <= self.epochs):
            raise ValueError("freeze_epochs must be in [0, epochs]")
        ds_cfg = json.loads((Path(self.dataset_dir) / "config.json").read_text())
        if int(ds_cfg["resolution"]) != int(self.resolution):
            raise ValueError(
                f"resolution {self.resolution} != dataset resolution "
                f"{ds_cfg['resolution']}")

    def to_dict(self) -> dict:
        return asdict(self)

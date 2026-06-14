from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from data_prep.sampler import DataSampler
from .config import TrainConfig
from .model import SmashRanker
from .dataset import TrainDataset, EvalDataset
from .loss import soft_bce
from .metrics import evaluate
from .stats import RunLogger


def _build_scheduler(optimizer, cfg: TrainConfig):
    cos_epochs = max(1, cfg.epochs - cfg.warmup_epochs)
    if cfg.warmup_epochs > 0:
        warm = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
        cos = CosineAnnealingLR(optimizer, T_max=cos_epochs)
        return SequentialLR(optimizer, [warm, cos], milestones=[cfg.warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=cos_epochs)


def run(cfg: TrainConfig, pretrained: bool = True) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available())
                          else "cpu")

    model = SmashRanker(cfg.backbone, cfg.resolution, cfg.dropout, pretrained=pretrained)
    model.to(device)
    mean, std = model.data_config["mean"], model.data_config["std"]

    train_sampler = DataSampler(cfg.dataset_dir, split="train", epoch=0)
    train_ds = TrainDataset(train_sampler, mean, std)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=False)
    val_ds = EvalDataset(cfg.dataset_dir, "val", mean, std, cfg.resolution)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    optimizer = AdamW(
        [{"params": model.head.parameters(), "lr": cfg.lr_head},
         {"params": model.backbone.parameters(), "lr": cfg.lr_backbone}],
        weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)

    logger = RunLogger(cfg.out_dir, cfg.run_name)
    ds_cfg = json.loads((Path(cfg.dataset_dir) / "config.json").read_text())
    logger.write_config({"train_config": cfg.to_dict(), "dataset_config": ds_cfg,
                         "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                         "device": str(device)})

    use_amp = cfg.amp and device.type == "cuda"
    best_spearman = -2.0
    best_epoch = -1
    t_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        phase = "frozen" if epoch < cfg.freeze_epochs else "finetune"
        if epoch == 0 and cfg.freeze_epochs > 0:
            model.freeze_backbone()
        if epoch == cfg.freeze_epochs and cfg.freeze_epochs > 0:
            model.unfreeze_backbone()

        train_ds.set_epoch(epoch)
        model.train()
        ep_start = time.perf_counter()
        loss_sum, n_samples, grad_norm_sum, n_batches = 0.0, 0, 0.0, 0
        for t, y in train_loader:
            t = t.to(device); y = y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                logits = model(t)
                loss = soft_bce(logits, y)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss_sum += loss.detach().item() * len(y)
            n_samples += len(y)
            grad_norm_sum += float(gn); n_batches += 1
        scheduler.step()

        val = evaluate(model, val_loader, device)
        lrs = [g["lr"] for g in optimizer.param_groups]
        # grad_norm logged is the per-epoch mean of pre-clip batch gradient norms
        record = {
            "epoch": epoch + 1, "phase": phase,
            "train_loss": loss_sum / max(1, n_samples),
            "val_loss": val["val_loss"], "val_spearman": val["spearman"],
            "val_mae": val["mae"], "lr_head": lrs[0], "lr_backbone": lrs[1],
            "grad_norm": grad_norm_sum / max(1, n_batches),
            "epoch_seconds": time.perf_counter() - ep_start,
            "n_train_samples": n_samples, "n_val_pokemon": val["n_pokemon"]}
        logger.log_epoch(record)

        logger.save_predictions(epoch + 1, val["ids"], val["y_true"], val["y_pred"])

        is_best = val["spearman"] > best_spearman
        if is_best:
            best_spearman = val["spearman"]; best_epoch = epoch + 1
        state = {"model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(),
                 "epoch": epoch + 1, "config": cfg.to_dict(), "metrics": val,
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state()}
        logger.save_checkpoint(state, epoch + 1, is_best)

    logger.finalize({"best_epoch": best_epoch, "best_spearman": best_spearman,
                     "total_seconds": time.perf_counter() - t_start,
                     "epochs": cfg.epochs})
    return logger.run


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the smash-ranker model.")
    p.add_argument("config", help="path to a TrainConfig JSON")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    raw = json.loads(Path(args.config).read_text())
    if args.epochs is not None:
        raw["epochs"] = args.epochs
    if args.device is not None:
        raw["device"] = args.device
    cfg = TrainConfig.from_dict(raw)
    run_dir = run(cfg)
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()

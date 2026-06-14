from __future__ import annotations
import numpy as np
import torch
from scipy.stats import spearmanr
from .loss import soft_bce


def aggregate_per_pokemon(ids, preds, targets):
    ids = np.asarray(ids); preds = np.asarray(preds); targets = np.asarray(targets)
    uniq = np.unique(ids)
    mean_pred = np.array([preds[ids == u].mean() for u in uniq])
    true = np.array([targets[ids == u][0] for u in uniq])
    return uniq, mean_pred, true


def spearman(pred, true) -> float:
    pred = np.asarray(pred); true = np.asarray(true)
    if len(pred) < 2:
        return 0.0
    r = spearmanr(pred, true).correlation
    return float(r) if r == r else 0.0


def mae(pred, true) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_ids, all_pred, all_true = [], [], []
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for t, y, pid in loader:
            t = t.to(device); y = y.to(device)
            logit = model(t)
            loss_sum += float(soft_bce(logit, y)) * len(y)
            n += len(y)
            all_pred.append(torch.sigmoid(logit).cpu().numpy())
            all_true.append(y.cpu().numpy())
            all_ids.append(np.asarray(pid))
    preds = np.concatenate(all_pred)
    trues = np.concatenate(all_true)
    ids = np.concatenate(all_ids)
    uniq, mean_pred, true = aggregate_per_pokemon(ids, preds, trues)
    return {"spearman": spearman(mean_pred, true), "mae": mae(mean_pred, true),
            "n_pokemon": len(true), "val_loss": loss_sum / max(1, n),
            "ids": uniq, "y_true": true, "y_pred": mean_pred}

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.metrics import aggregate_per_pokemon, spearman, pearson, mae, evaluate


def test_aggregate_per_pokemon_means():
    ids = np.array([1, 1, 2])
    preds = np.array([0.2, 0.4, 0.9])
    targets = np.array([0.5, 0.5, 0.8])
    uid, mp, tt = aggregate_per_pokemon(ids, preds, targets)
    assert list(uid) == [1, 2]
    assert np.allclose(mp, [0.3, 0.9])
    assert np.allclose(tt, [0.5, 0.8])


def test_spearman_monotonic_and_reversed():
    assert spearman(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])) == 1.0
    assert spearman(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])) == -1.0


def test_spearman_single_point_is_zero():
    assert spearman(np.array([0.5]), np.array([0.5])) == 0.0


def test_pearson_perfect_and_guard():
    assert abs(pearson(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])) - 1.0) < 1e-9
    assert pearson(np.array([0.5]), np.array([0.5])) == 0.0


def test_mae():
    assert abs(mae(np.array([0.1, 0.5]), np.array([0.2, 0.5])) - 0.05) < 1e-9


class _ConstModel(torch.nn.Module):
    def forward(self, x):
        return x.flatten(1).mean(1)


def test_evaluate_returns_metrics():
    xs, ys, pids = [], [], []
    for pid, level in enumerate([0.1, 0.5, 0.9], start=1):
        for _ in range(2):
            xs.append(torch.full((3, 4, 4), level))
            ys.append(torch.tensor(level, dtype=torch.float32))
            pids.append(pid)
    loader = DataLoader(list(zip(xs, ys, pids)), batch_size=2)
    out = evaluate(_ConstModel(), loader, torch.device("cpu"))
    assert {"spearman", "pearson", "mae", "n_pokemon", "val_loss",
            "ids", "y_true", "y_pred"} == set(out)
    assert len(out["ids"]) == 3 and len(out["y_pred"]) == 3
    assert out["n_pokemon"] == 3
    assert out["spearman"] > 0.9
    assert out["pearson"] > 0.9

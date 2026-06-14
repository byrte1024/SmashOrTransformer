# Model Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a pretrained ViT-Small to predict a Pokemon's crowd "smash" fraction from an image and rank Pokemon by it, with a full training pipeline (per-epoch checkpoints + rich stats) and an inference CLI.

**Architecture:** A `model/` package: `SmashRanker` (timm backbone + scalar regression head) fed by torch datasets that wrap the data-prep `DataSampler` (train, augmented) and a canonical renderer (eval/inference, deterministic). Soft-label BCE loss, AdamW with head-warmup-then-unfreeze, model selection by per-Pokemon Spearman. A `RunLogger` writes every epoch's checkpoint and statistics.

**Tech Stack:** PyTorch (CUDA 12.8 build for the RTX 5070 / Blackwell), timm, scipy. Tests run on CPU with `pretrained=False` and a tiny backbone (no network, no GPU).

Spec: `docs/superpowers/specs/2026-06-14-model-architecture-design.md`

---

## File Structure

```
model/
  __init__.py
  config.py        # TrainConfig dataclass + parse/validate/to_dict
  model.py         # SmashRanker (timm backbone + head), freeze/unfreeze, data_config
  dataset.py       # canonical_render, to_tensor, TrainDataset, EvalDataset
  loss.py          # soft_bce
  metrics.py       # aggregate_per_pokemon, spearman, mae, evaluate
  stats.py         # RunLogger (config/history/predictions/summary/checkpoints)
  train.py         # run(cfg) training loop + CLI
  infer.py         # load_model, score_image + CLI
tests/
  test_model_config.py
  test_model_net.py
  test_model_dataset.py
  test_model_loss.py
  test_model_metrics.py
  test_model_stats.py
  test_model_train.py
  test_model_infer.py
```

Conventions:
- `model/` modules use relative imports (`from .config import TrainConfig`) and reuse data-prep code (`from data_prep.sampler import DataSampler`, `from data_prep.prepare import load_sprite`).
- All forward passes return a **logit** (pre-sigmoid); sigmoid is applied only at metric/inference time.
- Tests reuse the existing `mini_repo` fixture (tests/conftest.py) and build a tiny dataset via `data_prep.prepare.prepare`.
- Run tests with `uv run pytest`.

---

## Task 1: Dependencies and package scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `model/__init__.py`

- [ ] **Step 1: Install torch (cu128), timm, scipy**

Run:
```bash
cd /home/drore/repos/SmashOrTransformer
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install timm scipy
```
Note: the cu128 torch wheel is large (~2.5 GB) and enables the RTX 5070. It also runs on CPU, which the tests use. If the cu128 index is unreachable, fall back to `uv pip install torch` (default CPU/cuda build) and report it.

- [ ] **Step 2: Add timm + scipy to pyproject and include `model*` in package discovery**

Edit `pyproject.toml`: add `"timm>=1.0"` and `"scipy>=1.11"` to `[project].dependencies`, and update the setuptools find include to also discover the `model` package:
```toml
[tool.setuptools.packages.find]
include = ["data_prep*", "model*"]
```
(Leave `torch` out of `dependencies` — it is installed via the cu128 index above, not from PyPI.)

- [ ] **Step 3: Create the package + gitignore runs/**

`model/__init__.py`:
```python
"""Smash-ranker model: ViT-based crowd-smash-fraction predictor."""
```
Run:
```bash
printf '%s\n' 'runs/' >> .gitignore
```

- [ ] **Step 4: Verify imports**

Run:
```bash
uv run python -c "import torch, timm, scipy, model; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
uv run pytest -q
```
Expected: prints torch version line; existing 45 tests still pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml model/__init__.py .gitignore
git commit -m "chore: add torch/timm/scipy and model package scaffold"
```

---

## Task 2: TrainConfig

**Files:**
- Create: `model/config.py`
- Test: `tests/test_model_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_config.py
import json
import pytest
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig


def _dataset(mini_repo, resolution=32):
    cfg = DataConfig.from_dict({"name": "ds", "resolution": resolution,
                                "minimages": 1, "variations": 2})
    return prepare(cfg, mini_repo["images"], mini_repo["labels"],
                   mini_repo["root"] / "datasets")


def test_defaults(mini_repo):
    out = _dataset(mini_repo, 32)
    cfg = TrainConfig.from_dict({"dataset_dir": str(out), "run_name": "r",
                                 "resolution": 32})
    assert cfg.backbone == "vit_small_patch16_224"
    assert cfg.epochs == 30 and cfg.batch_size == 64
    assert cfg.freeze_epochs == 3
    assert cfg.save_every_epoch is True
    assert cfg.amp is True


def test_resolution_must_match_dataset(mini_repo):
    out = _dataset(mini_repo, 32)
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"dataset_dir": str(out), "run_name": "r",
                               "resolution": 64})  # dataset is 32


def test_roundtrip(mini_repo):
    out = _dataset(mini_repo, 32)
    cfg = TrainConfig.from_dict({"dataset_dir": str(out), "run_name": "r",
                                 "resolution": 32, "epochs": 5})
    again = TrainConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_validate_rejects_bad_freeze(mini_repo):
    out = _dataset(mini_repo, 32)
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"dataset_dir": str(out), "run_name": "r",
                               "resolution": 32, "epochs": 2, "freeze_epochs": 5})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/config.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_config.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add model/config.py tests/test_model_config.py
git commit -m "feat: TrainConfig with dataset-resolution validation"
```

---

## Task 3: SmashRanker model

**Files:**
- Create: `model/model.py`
- Test: `tests/test_model_net.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_net.py
import torch
from model.model import SmashRanker


def _tiny():
    # vit_tiny at img_size=32: small + offline (pretrained=False)
    return SmashRanker(backbone="vit_tiny_patch16_224", resolution=32,
                       dropout=0.0, pretrained=False)


def test_forward_returns_logit_per_sample():
    m = _tiny()
    x = torch.randn(2, 3, 32, 32)
    out = m(x)
    assert out.shape == (2,)
    assert out.dtype == torch.float32


def test_data_config_has_mean_std():
    m = _tiny()
    assert len(m.data_config["mean"]) == 3
    assert len(m.data_config["std"]) == 3


def test_freeze_unfreeze_toggles_grad():
    m = _tiny()
    m.freeze_backbone()
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
    m.unfreeze_backbone()
    assert all(p.requires_grad for p in m.backbone.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_net.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/model.py
from __future__ import annotations
import timm
import torch
import torch.nn as nn
from timm.data import resolve_model_data_config


class SmashRanker(nn.Module):
    """timm backbone (ViT) + scalar regression head. forward -> logit [B]."""

    def __init__(self, backbone: str, resolution: int, dropout: float = 0.1,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, img_size=resolution)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))
        self.data_config = resolve_model_data_config(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)          # [B, feat_dim]
        return self.head(feats).squeeze(-1)   # [B]

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_net.py -q`
Expected: PASS (3 passed). First run downloads no weights (pretrained=False).

- [ ] **Step 5: Commit**

```bash
git add model/model.py tests/test_model_net.py
git commit -m "feat: SmashRanker timm backbone + regression head"
```

---

## Task 4: Datasets and rendering

**Files:**
- Create: `model/dataset.py`
- Test: `tests/test_model_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_dataset.py
import numpy as np
import torch
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from data_prep.sampler import DataSampler
from model.dataset import canonical_render, to_tensor, TrainDataset, EvalDataset

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def test_canonical_render_centers_on_white():
    # 10x20 RGBA opaque red sprite -> 16x16 white canvas, centered
    sprite = np.zeros((20, 10, 4), dtype=np.uint8)
    sprite[:, :] = (255, 0, 0, 255)
    out = canonical_render(sprite, 16)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    assert (out[0, 0] == 255).all()           # corner is white background
    assert (out[8, 8] == (255, 0, 0)).all()   # center is the sprite


def test_canonical_render_deterministic():
    sprite = np.zeros((12, 12, 4), dtype=np.uint8)
    sprite[3:9, 3:9] = (0, 255, 0, 255)
    a = canonical_render(sprite, 24)
    b = canonical_render(sprite, 24)
    assert np.array_equal(a, b)


def test_to_tensor_normalizes():
    img = np.full((8, 8, 3), 255, dtype=np.uint8)
    t = to_tensor(img, MEAN, STD)
    assert t.shape == (3, 8, 8) and t.dtype == torch.float32
    # (1.0 - 0.5) / 0.5 == 1.0
    assert torch.allclose(t, torch.ones_like(t))


def _dataset(mini_repo, resolution=32):
    cfg = DataConfig.from_dict({"name": "ds", "resolution": resolution,
                                "minimages": 1, "variations": 2,
                                "split": {"strategy": "pokemon", "val_frac": 0.34}})
    return prepare(cfg, mini_repo["images"], mini_repo["labels"],
                   mini_repo["root"] / "datasets")


def test_train_dataset_item(mini_repo):
    out = _dataset(mini_repo, 32)
    sampler = DataSampler(out, split="train", epoch=0)
    ds = TrainDataset(sampler, MEAN, STD)
    assert len(ds) == len(sampler)
    t, y = ds[0]
    assert t.shape == (3, 32, 32) and t.dtype == torch.float32
    assert 0.0 <= float(y) <= 1.0


def test_eval_dataset_item_has_pokemon_id(mini_repo):
    out = _dataset(mini_repo, 32)
    ds = EvalDataset(out, split="val", mean=MEAN, std=STD, resolution=32)
    assert len(ds) > 0
    t, y, pid = ds[0]
    assert t.shape == (3, 32, 32)
    assert isinstance(pid, int)
    assert 0.0 <= float(y) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/dataset.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def canonical_render(sprite_rgba: np.ndarray, resolution: int) -> np.ndarray:
    """Scale longest side to fit, center on a white res x res canvas -> RGB uint8."""
    im = Image.fromarray(sprite_rgba, "RGBA")
    w, h = im.size
    scale = resolution / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    canvas.paste(im, ((resolution - nw) // 2, (resolution - nh) // 2), im)
    bg = Image.new("RGB", (resolution, resolution), (255, 255, 255))
    bg.paste(canvas, (0, 0), canvas)
    return np.asarray(bg, dtype=np.uint8)


def to_tensor(img_rgb_uint8: np.ndarray, mean, std) -> torch.Tensor:
    arr = img_rgb_uint8.astype(np.float32) / 255.0          # HWC
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (t - m) / s


class TrainDataset(Dataset):
    """Augmented training samples from a DataSampler -> (tensor, label)."""

    def __init__(self, sampler, mean, std):
        self.sampler = sampler
        self.mean = mean
        self.std = std

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, i: int):
        img, label = self.sampler[i]
        return to_tensor(img, self.mean, self.std), torch.tensor(label, dtype=torch.float32)


class EvalDataset(Dataset):
    """Canonical (non-augmented) val/eval renders -> (tensor, label, pokemon_id)."""

    def __init__(self, dataset_dir, split, mean, std, resolution):
        d = Path(dataset_dir)
        with np.load(d / "data.npz", allow_pickle=True) as data:
            self._images = data["images"]
            self._pid = data["pokemon_id"]
            self._smash = data["smash_pct"]
        self._rows = list(json.loads((d / "split.json").read_text())[split])
        self.mean, self.std, self.resolution = mean, std, resolution

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int):
        r = self._rows[i]
        img = canonical_render(self._images[r], self.resolution)
        t = to_tensor(img, self.mean, self.std)
        return t, torch.tensor(float(self._smash[r]), dtype=torch.float32), int(self._pid[r])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_dataset.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add model/dataset.py tests/test_model_dataset.py
git commit -m "feat: train/eval torch datasets and canonical render"
```

---

## Task 5: Loss

**Files:**
- Create: `model/loss.py`
- Test: `tests/test_model_loss.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_loss.py
import torch
from model.loss import soft_bce


def test_soft_bce_finite_scalar():
    logits = torch.tensor([0.0, 2.0, -1.0])
    targets = torch.tensor([0.5, 0.9, 0.1])
    loss = soft_bce(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_soft_bce_gradients_flow():
    logits = torch.tensor([0.3, -0.4], requires_grad=True)
    targets = torch.tensor([0.7, 0.2])
    soft_bce(logits, targets).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_perfect_prediction_low_loss():
    # logits far in the right direction -> small loss
    logits = torch.tensor([10.0, -10.0])
    targets = torch.tensor([1.0, 0.0])
    assert soft_bce(logits, targets) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_loss.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.loss'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/loss.py
from __future__ import annotations
import torch
import torch.nn.functional as F


def soft_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with soft (fractional) targets in [0,1]. Mean scalar."""
    return F.binary_cross_entropy_with_logits(logits, targets)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_loss.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add model/loss.py tests/test_model_loss.py
git commit -m "feat: soft-label BCE loss"
```

---

## Task 6: Metrics

**Files:**
- Create: `model/metrics.py`
- Test: `tests/test_model_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_metrics.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.metrics import aggregate_per_pokemon, spearman, mae, evaluate


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


def test_mae():
    assert abs(mae(np.array([0.1, 0.5]), np.array([0.2, 0.5])) - 0.05) < 1e-9


class _ConstModel(torch.nn.Module):
    def forward(self, x):
        # logit proportional to mean pixel -> monotonic with a fake label
        return x.flatten(1).mean(1)


def test_evaluate_returns_metrics():
    # 3 "pokemon", 2 images each; image brightness correlates with target
    xs, ys, pids = [], [], []
    for pid, level in enumerate([0.1, 0.5, 0.9], start=1):
        for _ in range(2):
            xs.append(torch.full((3, 4, 4), level))
            ys.append(torch.tensor(level, dtype=torch.float32))
            pids.append(pid)
    loader = DataLoader(list(zip(xs, ys, pids)), batch_size=2)
    out = evaluate(_ConstModel(), loader, torch.device("cpu"))
    assert set(out) == {"spearman", "mae", "n_pokemon", "val_loss"}
    assert out["n_pokemon"] == 3
    assert out["spearman"] > 0.9     # brightness monotonic with target
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/metrics.py
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
    return float(r) if r == r else 0.0   # guard NaN (constant input)


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
    _, mean_pred, true = aggregate_per_pokemon(ids, preds, trues)
    return {"spearman": spearman(mean_pred, true), "mae": mae(mean_pred, true),
            "n_pokemon": len(true), "val_loss": loss_sum / max(1, n)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_metrics.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add model/metrics.py tests/test_model_metrics.py
git commit -m "feat: per-pokemon Spearman/MAE metrics and evaluate"
```

---

## Task 7: RunLogger (statistics + checkpoints)

**Files:**
- Create: `model/stats.py`
- Test: `tests/test_model_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_stats.py
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
    # both per-epoch checkpoints kept + best + last
    assert (run / "checkpoints" / "epoch_001.pt").exists()
    assert (run / "checkpoints" / "epoch_002.pt").exists()
    assert (run / "checkpoints" / "best.pt").exists()
    assert (run / "checkpoints" / "last.pt").exists()
    # predictions per epoch
    assert (run / "predictions" / "epoch_001.csv").exists()
    # history.csv has header + 2 rows
    rows = list(csv.DictReader(open(run / "history.csv")))
    assert len(rows) == 2
    assert rows[1]["val_spearman"] == "0.6"
    # history.jsonl has 2 records
    lines = [json.loads(x) for x in open(run / "history.jsonl") if x.strip()]
    assert len(lines) == 2


def test_predictions_csv_content(tmp_path):
    rl = RunLogger(out_dir=tmp_path, run_name="r")
    rl.save_predictions(1, [5, 9], [0.1, 0.2], [0.15, 0.25])
    rows = list(csv.DictReader(open(tmp_path / "r" / "predictions" / "epoch_001.csv")))
    assert rows[0]["pokemon_id"] == "5"
    assert rows[0]["y_true"] == "0.1"
    assert rows[0]["y_pred"] == "0.15"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_stats.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.stats'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/stats.py
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

    def save_checkpoint(self, state: dict, epoch: int, is_best: bool) -> None:
        if state.get("config", None) is not None or True:
            pass  # state is the caller's full checkpoint dict
        epoch_path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(state, epoch_path)
        torch.save(state, self.ckpt_dir / "last.pt")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pt")

    def finalize(self, summary: dict) -> None:
        (self.run / "summary.json").write_text(json.dumps(summary, indent=2))
```

Note: remove the no-op `if state.get(...)` block when implementing — it is a leftover. Implement `save_checkpoint` as just the three `torch.save` calls plus the `is_best` branch:
```python
    def save_checkpoint(self, state: dict, epoch: int, is_best: bool) -> None:
        torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pt")
        torch.save(state, self.ckpt_dir / "last.pt")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pt")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_stats.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add model/stats.py tests/test_model_stats.py
git commit -m "feat: RunLogger for checkpoints and statistics"
```

---

## Task 8: Training loop + CLI

**Files:**
- Create: `model/train.py`
- Test: `tests/test_model_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_train.py
import csv
import json
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run


def _tiny_dataset(mini_repo, resolution=32):
    cfg = DataConfig.from_dict({"name": "ds", "resolution": resolution,
                                "minimages": 1, "variations": 2,
                                "split": {"strategy": "pokemon", "val_frac": 0.34}})
    return prepare(cfg, mini_repo["images"], mini_repo["labels"],
                   mini_repo["root"] / "datasets")


def test_train_smoke(mini_repo, tmp_path):
    out = _tiny_dataset(mini_repo, 32)
    cfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "smoke", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 2, "batch_size": 4, "freeze_epochs": 1, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    # run must construct the model with pretrained=False in CPU/test contexts
    run_dir = run(cfg, pretrained=False)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "checkpoints" / "epoch_001.pt").exists()
    assert (run_dir / "checkpoints" / "epoch_002.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    rows = list(csv.DictReader(open(run_dir / "history.csv")))
    assert len(rows) == 2
    for r in rows:
        assert r["train_loss"] not in ("", "nan")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "best_epoch" in summary and "best_spearman" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_train.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.train'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/train.py
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
            loss_sum += float(loss) * len(y); n_samples += len(y)
            grad_norm_sum += float(gn); n_batches += 1
        scheduler.step()

        val = evaluate(model, val_loader, device)
        lrs = [g["lr"] for g in optimizer.param_groups]
        record = {
            "epoch": epoch + 1, "phase": phase,
            "train_loss": loss_sum / max(1, n_samples),
            "val_loss": val["val_loss"], "val_spearman": val["spearman"],
            "val_mae": val["mae"], "lr_head": lrs[0], "lr_backbone": lrs[1],
            "grad_norm": grad_norm_sum / max(1, n_batches),
            "epoch_seconds": time.perf_counter() - ep_start,
            "n_train_samples": n_samples, "n_val_pokemon": val["n_pokemon"]}
        logger.log_epoch(record)

        ids, y_true, y_pred = _val_predictions(model, val_loader, device)
        logger.save_predictions(epoch + 1, ids, y_true, y_pred)

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


def _val_predictions(model, loader, device):
    model.eval()
    ids, y_true, y_pred = [], [], []
    with torch.no_grad():
        for t, y, pid in loader:
            t = t.to(device)
            p = torch.sigmoid(model(t)).cpu().numpy()
            for a, b, c in zip(np.asarray(pid), y.numpy(), p):
                ids.append(int(a)); y_true.append(float(b)); y_pred.append(float(c))
    return ids, y_true, y_pred


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_train.py -q`
Expected: PASS (1 passed). Runs 2 CPU epochs on the tiny ViT; takes a few seconds.

- [ ] **Step 5: Commit**

```bash
git add model/train.py tests/test_model_train.py
git commit -m "feat: training loop with head-warmup, per-epoch checkpoints, stats"
```

---

## Task 9: Inference CLI

**Files:**
- Create: `model/infer.py`
- Test: `tests/test_model_infer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_infer.py
import csv
import json
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run
from model.infer import load_model, score_image


def _trained(mini_repo, tmp_path):
    cfg_d = DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                                  "variations": 2,
                                  "split": {"strategy": "pokemon", "val_frac": 0.34}})
    out = prepare(cfg_d, mini_repo["images"], mini_repo["labels"],
                  mini_repo["root"] / "datasets")
    tcfg = TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "inf", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"})
    return run(tcfg, pretrained=False), mini_repo


def test_infer_scores_in_range(mini_repo, tmp_path):
    run_dir, repo = _trained(mini_repo, tmp_path)
    model, cfg = load_model(run_dir / "checkpoints" / "best.pt",
                            device="cpu", pretrained=False)
    img = repo["images"] / "1" / "official-artwork.png"
    score = score_image(model, cfg, img, device="cpu")
    assert 0.0 <= score <= 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_infer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.infer'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/infer.py
from __future__ import annotations
import argparse
import torch
from data_prep.prepare import load_sprite
from .config import TrainConfig
from .model import SmashRanker
from .dataset import canonical_render, to_tensor


def load_model(checkpoint_path, device="cuda", pretrained: bool = False):
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = TrainConfig.from_dict(ckpt["config"])
    model = SmashRanker(cfg.backbone, cfg.resolution, cfg.dropout, pretrained=pretrained)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).eval()
    return model, cfg


def score_image(model, cfg: TrainConfig, image_path, device="cuda") -> float:
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    sprite = load_sprite(image_path)                       # uint8 RGBA
    img = canonical_render(sprite, cfg.resolution)         # uint8 RGB on white
    t = to_tensor(img, model.data_config["mean"], model.data_config["std"])
    t = t.unsqueeze(0).to(dev)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).item()
    return prob * 100.0


def main(argv=None):
    p = argparse.ArgumentParser(description="Score image(s) for smash-ability (0-100).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("images", nargs="+")
    args = p.parse_args(argv)
    model, cfg = load_model(args.checkpoint, device=args.device, pretrained=False)
    for path in args.images:
        print(f"{path}: {score_image(model, cfg, path, device=args.device):.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_infer.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add model/infer.py tests/test_model_infer.py
git commit -m "feat: inference CLI scoring images 0-100"
```

---

## Task 10: Example train config + full suite

**Files:**
- Create: `configs/train_example.json`
- Test: full suite

- [ ] **Step 1: Create `configs/train_example.json`**

```json
{
  "dataset_dir": "datasets/portraits_v1",
  "run_name": "vit_small_portraits_v1",
  "resolution": 224,
  "out_dir": "runs",
  "backbone": "vit_small_patch16_224",
  "epochs": 30,
  "batch_size": 64,
  "freeze_epochs": 3,
  "lr_head": 1e-3,
  "lr_backbone": 2e-5,
  "weight_decay": 0.05,
  "warmup_epochs": 1,
  "dropout": 0.1,
  "amp": true,
  "grad_clip": 1.0,
  "num_workers": 8,
  "seed": 0,
  "device": "cuda"
}
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS (all tests green: 45 data-prep + ~24 model = ~69).

- [ ] **Step 3: Verify the train CLI help works**

Run: `uv run python -m model.train --help` and `uv run python -m model.infer --help`
Expected: both print usage without error.

- [ ] **Step 4: Commit**

```bash
git add configs/train_example.json
git commit -m "feat: example training config"
```

---

## Self-Review

**Spec coverage:**
- ViT-Small pretrained fine-tune backbone + scalar head -> Task 3. ✓
- TrainConfig (all fields, resolution match) -> Task 2. ✓
- SmashTorchDataset train (DataSampler) + eval canonical render + normalization -> Task 4. ✓
- soft-label BCE, unweighted -> Task 5. ✓
- per-Pokemon aggregation + Spearman (primary) + MAE + evaluate -> Task 6. ✓
- RunLogger: config/history.csv/history.jsonl/predictions/summary + per-epoch checkpoints (all kept) + best/last with full state -> Task 7. ✓
- Training loop: AdamW two groups, cosine+warmup, head-warmup-then-unfreeze, AMP bf16, grad clip, model selection by Spearman, CLI -> Task 8. ✓
- Inference CLI (canonical render -> sigmoid -> 0-100) -> Task 9. ✓
- Tests CPU + pretrained=False + tiny backbone, smoke train + infer -> Tasks 3-9. ✓
- cu128 torch + timm + scipy + package discovery -> Task 1. ✓
- Example config -> Task 10. ✓

**Placeholder scan:** One intentional callout in Task 7 Step 3 flags a leftover no-op block and gives the exact clean `save_checkpoint` to use; otherwise no placeholders.

**Type consistency:** `SmashRanker(backbone, resolution, dropout, pretrained)`, `.data_config["mean"|"std"]`, `.backbone`/`.head`; `TrainDataset(sampler, mean, std)` -> `(tensor, label)`; `EvalDataset(dir, split, mean, std, resolution)` -> `(tensor, label, pid)`; `soft_bce(logits, targets)`; `evaluate(model, loader, device) -> {spearman,mae,n_pokemon,val_loss}`; `RunLogger(out_dir, run_name)` with `write_config/log_epoch/save_predictions/save_checkpoint/finalize`; `run(cfg, pretrained=True) -> Path`; `load_model(path, device, pretrained) -> (model, cfg)`; `score_image(model, cfg, path, device) -> float`. Consistent across tasks and tests.

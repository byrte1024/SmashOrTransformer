# Model Architecture - Design

Date: 2026-06-14
Status: Approved design, pending implementation plan

## Goal

Train a Vision Transformer to predict a Pokemon's crowd "smash" fraction from an
image, and rank Pokemon by smash-ability. A pretrained ViT-Small is fine-tuned
with a scalar regression head on the dataset produced by the data-prep pipeline
(`datasets/{name}/`, read via `DataSampler`). Ships with a training pipeline and
an inference CLI. Per-epoch checkpoints and rich statistics are saved for every
run.

Depends on the data-prep pipeline (`data_prep/`, spec
`2026-06-14-data-prep-design.md`): `DataSampler` yields augmented
`(uint8 HWC RGB, label)` samples; labels are the smash fraction in [0,1].

## Hardware / environment

- Target GPU: RTX 5070, 12 GB (Blackwell, sm_120). Requires a PyTorch build with
  CUDA 12.8 (`--index-url https://download.pytorch.org/whl/cu128`).
- Tests run on CPU with a tiny backbone and the data-prep mini fixture.

## Dependencies

- `torch` (cu128), `timm` (backbones + pretrained weights + data config),
  `scipy` (Spearman). numpy/pillow already present.

## Module structure (`model/` package)

- `config.py` - `TrainConfig` dataclass + parse/validate/to_dict.
- `model.py` - `SmashRanker` (timm backbone + regression head).
- `dataset.py` - `SmashTorchDataset` (train via DataSampler; eval via canonical
  render) + normalization.
- `loss.py` - soft-label BCE.
- `metrics.py` - per-Pokemon aggregation + Spearman + MAE.
- `stats.py` - `RunLogger`: writes history, predictions, summary, checkpoints.
- `train.py` - training loop + CLI (`python -m model.train`).
- `infer.py` - inference CLI (`python -m model.infer`).
- `tests/` - unit + smoke tests (CPU).

### config.py

`TrainConfig` fields (with defaults):

```
dataset_dir                         # path to datasets/{name}/
run_name                            # output subfolder name
out_dir            = "runs"
backbone           = "vit_small_patch16_224"   # any timm model
resolution         = 224            # must match dataset config.resolution
epochs             = 30
batch_size         = 64
freeze_epochs      = 3              # head-only warmup before unfreezing backbone
lr_head            = 1e-3
lr_backbone        = 2e-5
weight_decay       = 0.05
warmup_epochs      = 1             # LR warmup
dropout            = 0.1
amp                = true          # bf16 mixed precision
grad_clip          = 1.0
num_workers        = 8
seed               = 0
device             = "cuda"        # falls back to cpu if unavailable
save_every_epoch   = true          # per-epoch checkpoint; ALL epochs kept, never auto-deleted
```

`validate()`: resolution must equal the dataset's `config.json` resolution
(loaded and checked); epochs/batch_size > 0; freeze_epochs in [0, epochs].

### model.py

`SmashRanker(nn.Module)`:
- `timm.create_model(backbone, pretrained=True, num_classes=0)` -> feature vector
  of size `feat_dim` (read from the model).
- Head: `Dropout(dropout)` -> `Linear(feat_dim, 1)`.
- `forward(x) -> Tensor[B]` returns a **logit** (pre-sigmoid). Sigmoid is applied
  only at inference / metric time.
- `freeze_backbone()` / `unfreeze_backbone()` toggle `requires_grad` on backbone
  params (head always trainable).
- Exposes `data_config` (timm's resolve_data_config: mean/std/input_size) so the
  dataset can normalize correctly for the chosen backbone.

### dataset.py

`SmashTorchDataset(torch.utils.data.Dataset)`:
- Constructed from a `DataSampler` (train) OR a dataset dir + split (eval).
- Normalization: ImageNet-style mean/std from the backbone's timm data config,
  passed in at construction.
- `train` mode: `__getitem__(i)` pulls `(img_uint8 HWC RGB, label)` from the
  `DataSampler`, converts to normalized CHW float tensor; returns
  `(tensor, float_label)`. (No `pokemon_id` — training does not aggregate
  per Pokemon; only eval needs the id.)
- `eval` mode: iterates the val split's rows once (no random augmentation).
  Renders each sprite **canonically**: scale longest side to fit `resolution`,
  center on a white `resolution x resolution` canvas, normalize. Returns
  `(tensor, float_label, pokemon_id)`. `pokemon_id` is required for per-Pokemon
  metric aggregation.
- `set_epoch(e)` forwards to the underlying `DataSampler` (train only).
- Canonical render is a small self-contained helper (`canonical_render(sprite,
  resolution)`) in this module; it does not depend on the random augmentation
  pipeline.

### loss.py

`soft_bce(logits, targets) -> Tensor` wrapping `BCEWithLogitsLoss` with float
targets in [0,1] (no vote weighting - all Pokemon have >170k votes, so the
fraction is statistically near-exact). Returns the mean scalar loss.

### metrics.py

- `aggregate_per_pokemon(pokemon_ids, preds, targets) -> (ids, mean_pred, true)`:
  averages predicted probabilities over each Pokemon's val images; the target is
  that Pokemon's single label.
- `spearman(pred, true) -> float` (scipy.stats.spearmanr).
- `mae(pred, true) -> float`.
- `evaluate(model, val_loader, device) -> dict`: runs the model over the val
  loader, sigmoids logits, aggregates per Pokemon, returns
  `{spearman, mae, n_pokemon, val_loss}`.

### stats.py - RunLogger

Writes everything under `out_dir/run_name/`:

```
runs/{run_name}/
  config.json                 # resolved TrainConfig + dataset config snapshot + env (torch/cuda/gpu)
  checkpoints/
    epoch_001.pt ...          # every epoch (all kept, never auto-deleted)
    best.pt                   # best val Spearman so far
    last.pt                   # most recent
  history.csv                 # one row/epoch, columns below
  history.jsonl               # one JSON record/epoch (superset of csv)
  predictions/
    epoch_001.csv ...         # per-Pokemon: pokemon_id, y_true, y_pred (val)
  summary.json                # best_epoch, best_spearman, final metrics, total_seconds
```

`history.csv` columns: `epoch, phase(frozen|finetune), train_loss, val_loss,
val_spearman, val_mae, lr_head, lr_backbone, grad_norm, epoch_seconds,
n_train_samples, n_val_pokemon`.

Each checkpoint (`epoch_*.pt`, `best.pt`, `last.pt`) bundles: `model_state`,
`optimizer_state`, `scheduler_state`, `epoch`, `config` (TrainConfig dict),
`metrics` (that epoch), and RNG states (torch/numpy) for resumability.

`RunLogger` API: `log_epoch(record: dict)`, `save_checkpoint(state, epoch,
is_best)`, `save_predictions(epoch, ids, y_true, y_pred)`, `finalize(summary)`.

Disk note: ALL epoch checkpoints are kept and never auto-deleted (per user
requirement). ViT-Small epoch checkpoints (weights + optimizer states) are
roughly ~260 MB each; 30 epochs ~= 8 GB. `runs/` is gitignored.

### train.py

Loop:
1. Load `TrainConfig`; load dataset config; build `DataSampler(train)` and
   `DataSampler(val)`; build `SmashRanker` (reads data_config for normalization);
   wrap in `SmashTorchDataset` + `DataLoader`s.
2. Optimizer: AdamW, two param groups (head `lr_head`, backbone `lr_backbone`),
   weight_decay. Cosine schedule with `warmup_epochs`. AMP via
   `torch.autocast` + `GradScaler` (or bf16). Grad clip `grad_clip`.
3. Epochs `0..epochs-1`:
   - If `epoch < freeze_epochs`: backbone frozen (phase "frozen"); at
     `epoch == freeze_epochs`: unfreeze (phase "finetune").
   - `train_dataset.set_epoch(epoch)` (new augmentations each epoch).
   - Train pass: accumulate mean loss, track grad norm.
   - Eval pass via `metrics.evaluate`.
   - `RunLogger.log_epoch(...)`, `save_predictions(...)`,
     `save_checkpoint(..., is_best = val_spearman > best)`.
4. `finalize(summary)`; print best epoch + Spearman.

CLI: `python -m model.train <config.json>` (config path positional; CLI flags may
override key fields like `--epochs`, `--device`).

### infer.py

`python -m model.infer --checkpoint runs/{run}/checkpoints/best.pt image.png [...]`:
- Load checkpoint -> rebuild `SmashRanker` from saved config; load weights; eval.
- For each image: `canonical_render` -> normalize -> forward -> sigmoid -> print
  `path: <score 0-100>` (fraction * 100). Supports multiple image paths.

## Data flow

```
datasets/{name}/ --DataSampler(train)--> SmashTorchDataset(train) --DataLoader--> SmashRanker --soft_bce--> backward
datasets/{name}/ --DataSampler(val)----> SmashTorchDataset(eval) --DataLoader--> SmashRanker --sigmoid--> per-Pokemon Spearman/MAE
```

## Testing

Unit:
- `model.py`: forward `(B,3,R,R)` -> shape `(B,)` logits; feat_dim wired; freeze/
  unfreeze toggles `requires_grad`.
- `dataset.py`: train item is normalized float `(3,R,R)` + label + id; eval
  `canonical_render` centers a non-square sprite on white and is deterministic.
- `loss.py`: `soft_bce` on soft targets returns a finite scalar; gradients flow.
- `metrics.py`: `aggregate_per_pokemon` averages correctly; `spearman` returns
  1.0 for a monotonic synthetic pair, near 0 for shuffled; `mae` correct.
- `stats.py`: `RunLogger` writes config.json/history.csv/history.jsonl/
  predictions/summary.json with expected fields; per-epoch checkpoints accumulate
  (all kept, none deleted) and `best.pt`/`last.pt` are maintained.

Integration / smoke (CPU, tiny):
- Build a tiny dataset from the data-prep mini fixture via `prepare`. Run
  `train` for 2 epochs with a tiny timm backbone (e.g. `vit_tiny_patch16_224` or
  a small test model, `pretrained=False` to avoid network in tests). Assert:
  losses finite, `history.csv` has 2 rows, per-epoch checkpoints + best.pt exist,
  summary.json written.
- `infer`: load the smoke checkpoint, score one image, assert output parses to a
  float in [0,100].

Tests must not require network or GPU: use `pretrained=False` and CPU in tests.

## Out of scope (this iteration)

- Background compositing (still the white stub from data-prep).
- Hyperparameter search / multi-GPU / distributed.
- Export to ONNX / deployment.
- Resume-from-checkpoint CLI (checkpoints store enough state to add it later).

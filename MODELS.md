# Models

Index of trained checkpoints for Smash or Transformer. See [README.md](README.md)
for the workflow; click a model for its full reproduction guide.

All models are a `timm` **ViT-Small/16 @ 224** backbone with a scalar regression
head, fine-tuned with soft-label BCE.

## Leaderboard (apples-to-apples)

Numbers below are from a **common cross-evaluation** (`model.cross_eval`): every
model scores the **same images** on the **102 Pokemon held out by all models**
(the seed-0 split is identical across datasets, so the held-out Pokemon match).
This is the fair comparison -- each run's *training-time* val number is **not**
comparable, because it averaged over that dataset's own per-Pokemon image set
(portraits has no booru) and the eval render changed over time.

Headline metric: **Spearman on `all_avg`** (mean score over every image of a
Pokemon -- the real-world ensemble use). Higher is better.

| Model | Sources | Spearman (all_avg) | Status | Details |
|-------|---------|:---:|--------|---------|
| **`vit_small_mixed_v1`** | portrait + in-game + booru | **0.770** | **best / reference** | [guide](docs/vit_small_mixed_v1.md) |
| `vit_small_mixed_v2` | + 75 booru/mon, heavy aug | 0.734 | regression (see below) | -- |
| `vit_small_portraits_v1` | portrait + in-game | 0.690 | sprite-only baseline | [guide](docs/vit_small_portraits_v1.md) |

### Full cross-eval

Spearman / Pearson / MAE on the common 102-Pokemon set (n=102). `portrait` =
official art only; `sprites_avg` = official + in-game; `all_avg` = incl. booru.

| Model | portrait | sprites_avg | all_avg |
|-------|----------|-------------|---------|
| `vit_small_mixed_v1`     | 0.657 / 0.794 / 0.056 | 0.728 / 0.810 / 0.050 | **0.770 / 0.835 / 0.045** |
| `vit_small_mixed_v2`     | 0.657 / 0.730 / 0.059 | 0.694 / 0.777 / 0.056 | 0.734 / 0.797 / 0.058 |
| `vit_small_portraits_v1` | 0.640 / 0.772 / 0.056 | **0.737 / 0.792 / 0.050** | 0.690 / 0.751 / 0.081 |

Reproduce:

```bash
uv run python -m model.cross_eval \
  runs/vit_small_portraits_v1 runs/vit_small_mixed_v1 runs/vit_small_mixed_v2
```

### Takeaways

- **`vit_small_mixed_v1` is the model to use.** Its `all_avg` ensemble (0.770)
  is the best result anywhere -- learning from booru fan-art *and* averaging a
  Pokemon's images across sources both help.
- **`portraits_v1` is a solid sprite-only model** (best `sprites_avg`, 0.737)
  but can't use fan-art -- booru *lowers* its score (all_avg 0.690 <
  sprites_avg 0.737), since it never trained on that domain.
- **`mixed_v2` regressed on every view** despite more booru data and heavier
  augmentation. Likely causes: a train/eval mismatch from the heavy booru aug
  (random white/black/real backgrounds + 0.9-1.2 scale at train time vs
  stretch-to-fit at eval), noisier fan-art at 75/Pokemon, and `fill_so` target
  250 oversampling the augmentation. Kept for the record, not recommended.

## Using a checkpoint

```bash
# score images
uv run python -m model.infer --checkpoint <path/to/best.pt> img.png
# render per-Pokemon scorecards + rankings
uv run python -m model.results --checkpoint <path/to/best.pt>
```

A checkpoint is self-describing (it stores its own train + dataset config).
Calibration lives in `calibration.json` next to it; regenerate with
`python -m model.calibrate --checkpoint <path>` if missing.

## Pretrained weights

Download links will be added here as checkpoints are published.

<!--
| Model | Spearman (all_avg) | Download |
|-------|:---:|----------|
| vit_small_mixed_v1 | 0.770 | <link> |
-->

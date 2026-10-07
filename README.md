# Multimodal Clinical AI with HAM10000

A compact multimodal-learning project combining dermoscopic images with structured clinical metadata.

This repository compares four models on the same leakage-safe held-out test split:

1. **Image-only** — ResNet-18
2. **Tabular-only** — age + sex + anatomical localization
3. **Late fusion** — concatenated image and tabular embeddings
4. **Cross-attention** — clinical tokens query spatial image tokens

## Dataset and task

The project uses HAM10000 dermoscopic images and associated metadata.

Educational binary target:

- Positive: `mel`, `bcc`, `akiec`
- Negative: `nv`, `bkl`, `df`, `vasc`

This grouping is for a multimodal-learning exercise only and is **not** a clinical diagnostic definition.

## Leakage-safe split

HAM10000 can contain multiple images from the same lesion. To avoid train/test leakage, the dataset is split by `lesion_id`, not by image.

Tabular preprocessing is fit on the training split only:
- age imputation uses training-set median
- age standardization uses training-set statistics
- categorical vocabularies are learned from training data

## Architectures

### Image-only

```text
Dermoscopy image
      ↓
ResNet-18
      ↓
image embedding
      ↓
classifier
```

### Tabular-only

```text
age ───────────────┐
sex embedding ─────┼→ MLP → classifier
site embedding ────┘
```

### Late fusion

```text
image → ResNet-18 → image embedding ─────┐
                                         ├→ concatenate → MLP → prediction
clinical metadata → tabular encoder ─────┘
```

### Cross-attention

```text
image → ResNet-18 layer4 → 7×7 spatial tokens → Key / Value
                                                    ↑
age + sex + site → 3 clinical tokens → Query ──────┘
                                                    ↓
                                             cross-attention
                                                    ↓
                                                classifier
```

## Held-out test results

| Model | AUROC | AUPRC | F1 | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Image-only | **0.9205** | 0.7341 | **0.6735** | **0.8327** | 0.8077 | 0.8577 |
| Tabular-only | 0.7629 | 0.4158 | 0.4756 | 0.7078 | 0.7657 | 0.6498 |
| Late fusion | 0.9132 | **0.7366** | 0.6718 | 0.8219 | 0.7692 | **0.8746** |
| Cross-attention | 0.9140 | 0.7194 | 0.6501 | 0.8258 | **0.8217** | 0.8300 |

The tabular-only model shows that the clinical metadata contains independent predictive signal. However, neither fusion strategy clearly outperformed the image-only model.

## Paired bootstrap comparison

The same held-out samples are used for each pairwise comparison, so differences are assessed with paired nonparametric bootstrap resampling.

| Comparison | ΔAUROC | 95% CI | ΔAUPRC | 95% CI |
|---|---:|---:|---:|---:|
| Late − Image | -0.0073 | -0.0187 to +0.0041 | +0.0025 | -0.0330 to +0.0348 |
| Cross − Image | -0.0065 | -0.0175 to +0.0047 | -0.0147 | -0.0549 to +0.0221 |
| Cross − Late | +0.0008 | -0.0120 to +0.0125 | -0.0172 | -0.0577 to +0.0223 |

All confidence intervals cross zero, so the experiment does not provide clear evidence that either fusion strategy improves AUROC or AUPRC relative to the competing model.

## Main conclusion

> Clinical metadata contained independent predictive information, but neither late fusion nor clinical-to-image cross-attention demonstrated a clear improvement over the image-only baseline on the held-out test set.

This illustrates an important multimodal-learning principle:

> More modalities and more sophisticated fusion do not automatically produce better predictive performance.

## Reproduce

Prepare the leakage-safe split:

```bash
python scripts/prepare_ham10000.py --dataset-root data/raw
```

Run tests:

```bash
pytest -q
```

Train/evaluate Day 4 baselines:

```bash
python train.py --model image --epochs 8 --batch-size 32
python evaluate.py --checkpoint outputs/image_best.pt --split test

python train.py --model tabular --epochs 20 --batch-size 64 --lr 1e-3
python evaluate.py --checkpoint outputs/tabular_best.pt --split test

python train.py --model late --epochs 8 --batch-size 32 \
  --init-image-checkpoint outputs/image_best.pt
python evaluate.py --checkpoint outputs/late_best.pt --split test
```

Train/evaluate cross-attention:

```bash
python train_cross.py \
  --epochs 8 \
  --batch-size 32 \
  --lr 1e-4 \
  --init-image-checkpoint outputs/image_best.pt

python evaluate_cross.py \
  --checkpoint outputs/cross_best.pt \
  --split test
```

Generate pairwise bootstrap JSON files:

```bash
python paired_bootstrap.py \
  --a outputs/image_test_predictions.csv \
  --b outputs/late_test_predictions.csv \
  --name-a image \
  --name-b late \
  --bootstrap 2000 \
  --output outputs/bootstrap_late_vs_image.json

python paired_bootstrap.py \
  --a outputs/image_test_predictions.csv \
  --b outputs/cross_test_predictions.csv \
  --name-a image \
  --name-b cross \
  --bootstrap 2000 \
  --output outputs/bootstrap_cross_vs_image.json

python paired_bootstrap.py \
  --a outputs/late_test_predictions.csv \
  --b outputs/cross_test_predictions.csv \
  --name-a late \
  --name-b cross \
  --bootstrap 2000 \
  --output outputs/bootstrap_cross_vs_late.json
```

Generate final tables and figures:

```bash
python finalize_day45.py
```

Outputs:
- `outputs/final_model_comparison.csv`
- `outputs/bootstrap_comparison.csv`
- `outputs/FINAL_RESULTS.md`
- `figures/01_discrimination_comparison.png`
- `figures/02_operating_metrics.png`
- `figures/03_bootstrap_delta_auroc.png`
- `figures/04_bootstrap_delta_auprc.png`

## Tests

Current project state:

```text
5 passed
```
## Disclaimer

This repository is an educational/research implementation and is not intended for clinical diagnosis or treatment decisions.

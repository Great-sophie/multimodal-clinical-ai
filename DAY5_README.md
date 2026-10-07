# Day 5 — Cross-Attention

Clinical metadata (age, sex, localization) becomes 3 query tokens. The ResNet18 layer-4 map becomes 49 image tokens. Multi-head cross-attention lets clinical context query spatial image features.

## Train
```bash
python train_cross.py --epochs 8 --batch-size 32 --lr 1e-4 --init-image-checkpoint outputs/image_best.pt
```

## Evaluate
```bash
python evaluate_cross.py --checkpoint outputs/cross_best.pt --split test
```

## Four-model comparison
```bash
python compare_day5.py
```

## Paired bootstrap: cross vs image
```bash
python paired_bootstrap.py --a outputs/image_test_predictions.csv --b outputs/cross_test_predictions.csv --name-a image --name-b cross --bootstrap 2000
```

Interpretation: Δ = cross − image. A 95% CI entirely above 0 supports improvement; a CI crossing 0 means no clear evidence of improvement.

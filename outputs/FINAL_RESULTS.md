# Day 4–5 Final Results

## Held-out test performance

| model           |   AUROC |   AUPRC |     F1 |   Balanced accuracy |   Sensitivity |   Specificity |
|:----------------|--------:|--------:|-------:|--------------------:|--------------:|--------------:|
| Image-only      |  0.9205 |  0.7341 | 0.6735 |              0.8327 |        0.8077 |        0.8577 |
| Tabular-only    |  0.7629 |  0.4158 | 0.4756 |              0.7078 |        0.7657 |        0.6498 |
| Late fusion     |  0.9132 |  0.7366 | 0.6718 |              0.8219 |        0.7692 |        0.8746 |
| Cross-attention |  0.9140 |  0.7194 | 0.6501 |              0.8258 |        0.8217 |        0.8300 |

## Paired bootstrap comparisons

| comparison    |   delta_AUROC |   AUROC_CI_low |   AUROC_CI_high |   delta_AUPRC |   AUPRC_CI_low |   AUPRC_CI_high |
|:--------------|--------------:|---------------:|----------------:|--------------:|---------------:|----------------:|
| late - image  |       -0.0073 |        -0.0187 |          0.0041 |        0.0025 |        -0.0330 |          0.0348 |
| cross - image |       -0.0065 |        -0.0175 |          0.0047 |       -0.0147 |        -0.0549 |          0.0221 |
| cross - late  |        0.0007 |        -0.0120 |          0.0125 |       -0.0172 |        -0.0577 |          0.0223 |

## Interpretation

Clinical metadata showed independent predictive signal, but neither late fusion nor cross-attention demonstrated a clear improvement over the image-only baseline. All paired-bootstrap 95% confidence intervals for ΔAUROC and ΔAUPRC crossed zero.

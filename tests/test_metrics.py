import numpy as np
from src.metrics import binary_metrics

def test_perfect_binary_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.1, 0.9, 0.99])
    metrics = binary_metrics(y_true, y_prob, threshold=0.5)
    assert abs(metrics["auroc"] - 1.0) < 1e-8
    assert abs(metrics["auprc"] - 1.0) < 1e-8
    assert abs(metrics["f1"] - 1.0) < 1e-8

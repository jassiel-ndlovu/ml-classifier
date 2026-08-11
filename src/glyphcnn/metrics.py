"""Pure-numpy classification metrics (no scikit-learn dependency needed).

Kept dependency-light and unit-tested: accuracy, per-class precision/recall/F1,
macro averages and a confusion matrix.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def per_class_metrics(cm: np.ndarray) -> dict:
    """Return precision/recall/F1/support per class plus macro averages."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.astype(int).tolist(),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }

"""Evaluation metrics for multi-label ECG classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from ecg_explain.data.labels import SUPERCLASSES


def per_class_auroc(
    y_true: np.ndarray, y_score: np.ndarray, class_names: list[str] | None = None
) -> dict[str, float]:
    """Per-class AUROC. NaN if a class has no positives in y_true."""
    names = class_names or SUPERCLASSES
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        if y_true[:, i].sum() == 0:
            out[name] = float("nan")
        else:
            out[name] = float(roc_auc_score(y_true[:, i], y_score[:, i]))
    return out


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged AUROC across all classes (skips classes with no positives)."""
    per_class = per_class_auroc(y_true, y_score)
    valid = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def per_class_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Per-class F1 at the given threshold."""
    names = class_names or SUPERCLASSES
    y_pred = (y_score >= threshold).astype(int)
    return {
        name: float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        for i, name in enumerate(names)
    }


def macro_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_all_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float | dict[str, float]]:
    """One-shot metric computation for logging or eval JSON dumps."""
    return {
        "macro_auroc": macro_auroc(y_true, y_score),
        "per_class_auroc": per_class_auroc(y_true, y_score),
        "macro_f1": macro_f1(y_true, y_score, threshold=threshold),
        "per_class_f1": per_class_f1(y_true, y_score, threshold=threshold),
    }
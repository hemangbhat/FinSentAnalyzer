"""
Shared evaluation statistics.

``bootstrap_macro_f1`` is the single implementation behind every confidence
interval the project reports (baselines, FinBERT and the LLM benchmark), so
all intervals are computed the same way and stay comparable.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

LABEL_IDS = [0, 1, 2]


def macro_f1(y_true, y_pred) -> float:
    """Macro-F1 over the three sentiment classes; predictions outside them (e.g. -1 for invalid) count as wrong."""
    return float(f1_score(y_true, y_pred, average="macro", labels=LABEL_IDS, zero_division=0))


def bootstrap_macro_f1(
    y_true, y_pred, n_boot: int = 2000, seed: int = 42, confidence: float = 0.95, rng: np.random.Generator | None = None
) -> dict:
    """
    Macro-F1 with a percentile bootstrap interval.

    Resamples the evaluation rows with replacement ``n_boot`` times and takes
    the (1 - confidence) / 2 and 1 - (1 - confidence) / 2 percentiles. Pass a
    shared ``rng`` to draw several intervals from one random stream.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    rng = rng if rng is not None else np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[i] = macro_f1(y_true[idx], y_pred[idx])
    tail = round((1 - confidence) / 2 * 100, 10)  # avoid 2.500000000000002
    return {
        "f1_macro": macro_f1(y_true, y_pred),
        "ci_low": float(np.percentile(scores, tail)),
        "ci_high": float(np.percentile(scores, 100 - tail)),
        "n": int(n),
    }


def pairs_from_confusion(cm) -> tuple[np.ndarray, np.ndarray]:
    """Expand a confusion matrix (rows = true, columns = predicted) into (y_true, y_pred) arrays."""
    y_true, y_pred = [], []
    for t, row in enumerate(cm):
        for p, count in enumerate(row):
            y_true += [t] * int(count)
            y_pred += [p] * int(count)
    return np.array(y_true), np.array(y_pred)

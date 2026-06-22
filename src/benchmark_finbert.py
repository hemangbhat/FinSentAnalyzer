"""
Benchmark pre-trained FinBERT on the same test set used for baseline models.

Produces metrics (accuracy, F1 macro/weighted, precision, recall) and
appends them to results/evaluation_results.json so that the README
comparison table includes all claimed models.

Usage:
    python src/benchmark_finbert.py
"""

import json
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from utils import get_project_root, get_results_dir, setup_logging

logger = setup_logging(__name__)


def load_test_set() -> pd.DataFrame:
    """Load the processed test split."""
    test_path = get_project_root() / "data" / "processed" / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test set not found at {test_path}. Run preprocessing first.")
    return pd.read_csv(test_path)


def run_finbert_benchmark() -> dict:
    """Evaluate pre-trained FinBERT on the held-out test set."""
    from finbert_pretrained import predict_with_finbert

    df = load_test_set()
    texts = df["sentence"].tolist()
    y_true = df["label"].tolist()

    logger.info("Running FinBERT on %d test samples...", len(texts))
    predictions = predict_with_finbert(texts)

    y_pred = [p["prediction"] for p in predictions]

    metrics = {
        "name": "finbert_pretrained",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "y_true": y_true,
        "y_pred": y_pred,
    }

    logger.info("FinBERT Pre-trained Results:")
    logger.info("  Accuracy:       %.4f", metrics["accuracy"])
    logger.info("  F1 (macro):     %.4f", metrics["f1_macro"])
    logger.info("  F1 (weighted):  %.4f", metrics["f1_weighted"])
    logger.info("  Precision (macro): %.4f", metrics["precision_macro"])
    logger.info("  Recall (macro):    %.4f", metrics["recall_macro"])

    return metrics


def save_to_evaluation_results(finbert_metrics: dict) -> None:
    """Append or update FinBERT metrics in evaluation_results.json."""
    results_path = get_results_dir() / "evaluation_results.json"

    existing: list = []
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)

    # Remove any existing finbert_pretrained entry to avoid duplicates
    existing = [r for r in existing if r.get("name") != "finbert_pretrained"]
    existing.append(finbert_metrics)

    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info("Saved FinBERT metrics to %s", results_path)


if __name__ == "__main__":
    metrics = run_finbert_benchmark()
    save_to_evaluation_results(metrics)

    print("\n" + "=" * 60)
    print("FinBERT Pre-trained Benchmark")
    print("=" * 60)
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):       {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):    {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro):{metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):   {metrics['recall_macro']:.4f}")
    print("\n✅ Results appended to results/evaluation_results.json")

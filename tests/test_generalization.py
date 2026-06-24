"""
Test the real out-of-distribution generalization evaluation.

Skips cleanly when the real dataset hasn't been fetched or the model is absent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from integrate_news import REAL_NEWS_PATH, evaluate_generalization  # noqa: E402
from utils import get_model_dir  # noqa: E402


@pytest.mark.skipif(
    not REAL_NEWS_PATH.exists(), reason="real dataset not fetched (run scripts/fetch_real_news_dataset.py)"
)
def test_generalization_metrics_are_valid():
    if not (get_model_dir() / "baseline_svm.joblib").exists():
        pytest.skip("baseline_svm model not present")

    result = evaluate_generalization("svm")
    assert result["n_samples"] > 0
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1_macro"] <= 1.0
    assert 0.0 <= result["majority_class_accuracy"] <= 1.0
    assert "MIT" in result["dataset"]

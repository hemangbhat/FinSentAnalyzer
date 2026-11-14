"""
Tests for the SHAP explainability module (shap_explain.py).
"""

import json
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


class TestShapExplain:
    """Tests for the shap_explain module."""

    def test_import_module(self):
        """Module can be imported without errors."""
        import shap_explain  # noqa: F401

    def test_shap_available(self):
        """SHAP library is installed and importable."""
        try:
            import shap  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("SHAP not installed — skipping SHAP tests")

    def test_get_shap_explanation_gradient_boosting(self):
        """SHAP explanation runs for gradient_boosting model."""
        model_path = Path(__file__).parent.parent / "models" / "baseline_gradient_boosting.joblib"
        if not model_path.exists():
            pytest.skip("Gradient Boosting model not trained yet")

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("SHAP not installed")

        from shap_explain import get_shap_explanation

        result = get_shap_explanation("gradient_boosting", max_samples=20)

        assert isinstance(result, dict)
        assert "model" in result
        assert result["model"] == "gradient_boosting"
        assert "top_features" in result
        assert len(result["top_features"]) > 0
        assert "feature" in result["top_features"][0]
        assert "importance" in result["top_features"][0]

    def test_get_shap_explanation_logreg(self):
        """SHAP explanation (linear coefficients) runs for logreg model."""
        model_path = Path(__file__).parent.parent / "models" / "baseline_logreg.joblib"
        if not model_path.exists():
            pytest.skip("Logistic Regression model not trained yet")

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("SHAP not installed")

        from shap_explain import get_shap_explanation

        result = get_shap_explanation("logreg", max_samples=20)

        assert isinstance(result, dict)
        assert result["model"] == "logreg"
        assert len(result["top_features"]) > 0

    def test_shap_results_contain_class_features(self):
        """SHAP results include per-class feature drivers."""
        model_path = Path(__file__).parent.parent / "models" / "baseline_gradient_boosting.joblib"
        if not model_path.exists():
            pytest.skip("Gradient Boosting model not trained yet")

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("SHAP not installed")

        from shap_explain import get_shap_explanation

        result = get_shap_explanation("gradient_boosting", max_samples=20)

        class_features = result.get("class_features", {})
        # Tree models should produce per-class features
        if class_features:
            assert "positive" in class_features or "negative" in class_features
            for cls_name, features in class_features.items():
                assert len(features) > 0
                assert "feature" in features[0]
                assert "shap_value" in features[0]
                assert "direction" in features[0]

    def test_save_shap_results(self):
        """SHAP results can be saved to JSON file."""
        model_path = Path(__file__).parent.parent / "models" / "baseline_gradient_boosting.joblib"
        if not model_path.exists():
            pytest.skip("Gradient Boosting model not trained yet")

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("SHAP not installed")

        from shap_explain import save_shap_results
        from utils import get_results_dir

        result = save_shap_results("gradient_boosting", "test")
        assert result is not None

        # Check file was saved
        output_path = get_results_dir() / "shap_gradient_boosting.json"
        assert output_path.exists(), f"SHAP results file not saved: {output_path}"

        # Verify JSON is valid
        with open(output_path) as f:
            data = json.load(f)
        assert data["model"] == "gradient_boosting"
        assert len(data["top_features"]) > 0


def _linear_case():
    np = pytest.importorskip("numpy")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = ["profit rose sharply", "loss widened badly", "shares were flat", "sales grew strongly"] * 5
    labels = [2, 0, 1, 2] * 5
    tfidf = TfidfVectorizer().fit(texts)
    X = tfidf.transform(texts).toarray()
    clf = LogisticRegression(max_iter=500).fit(X, labels)
    return np, clf, X


def test_linear_shap_is_exact_and_additive():
    """sum_j phi_j(x) + base must equal the model's score for every sample (base = score at the mean)."""
    from shap_explain import compute_shap_values

    np, clf, X = _linear_case()
    mean = X.mean(axis=0)
    values, method = compute_shap_values("logreg", clf, X, mean)
    assert method == "linear"
    raw = clf.decision_function(X)
    base = clf.decision_function(mean.reshape(1, -1))[0]
    for k in range(3):
        assert np.allclose(values[k].sum(axis=1) + base[k], raw[:, k])


def test_gradient_boosting_shap_is_exact_and_additive():
    """Multi-class GradientBoosting is explained tree by tree; contributions must add up to the model score."""
    pytest.importorskip("shap")
    from sklearn.ensemble import GradientBoostingClassifier

    from shap_explain import compute_shap_values

    np, _, X = _linear_case()
    labels = [2, 0, 1, 2] * 5
    gb = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X, labels)
    values, method = compute_shap_values("gradient_boosting", gb, X, X.mean(axis=0))
    assert method == "gb_trees"
    raw = gb.decision_function(X)
    for k in range(3):
        residual = raw[:, k] - values[k].sum(axis=1)
        assert np.ptp(residual) < 1e-8  # the same base value for every sample

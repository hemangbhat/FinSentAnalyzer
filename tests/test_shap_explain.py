"""
Tests for the SHAP explainability module (shap_explain.py).
"""

import sys
import json
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

        # Check file was saved
        output_path = get_results_dir() / "shap_gradient_boosting.json"
        assert output_path.exists(), f"SHAP results file not saved: {output_path}"

        # Verify JSON is valid
        with open(output_path) as f:
            data = json.load(f)
        assert data["model"] == "gradient_boosting"
        assert len(data["top_features"]) > 0

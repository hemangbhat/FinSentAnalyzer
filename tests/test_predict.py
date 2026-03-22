"""Tests for src/predict.py"""

import pytest
from pathlib import Path

from predict import get_available_models, SentimentPredictor


class TestGetAvailableModels:
    def test_returns_list(self):
        models = get_available_models()
        assert isinstance(models, list)

    def test_finds_trained_models(self, models_dir):
        """Should find at least one model if models/ has .joblib files."""
        models = get_available_models()
        joblib_files = list(models_dir.glob("*.joblib"))
        if joblib_files:
            assert len(models) > 0


class TestSentimentPredictor:
    @pytest.fixture
    def predictor(self):
        """Load a baseline predictor if models exist."""
        models = get_available_models()
        baseline_models = [m for m in models if m.startswith("baseline_")]
        if not baseline_models:
            pytest.skip("No trained baseline models found")
        return SentimentPredictor(model_type=baseline_models[0])

    def test_predict_single(self, predictor):
        result = predictor.predict("Revenue increased significantly.")
        assert isinstance(result, dict)
        assert "label" in result
        assert result["label"] in ["positive", "negative", "neutral"]

    def test_predict_returns_probabilities(self, predictor):
        result = predictor.predict("Revenue increased significantly.")
        if "probabilities" in result:
            assert isinstance(result["probabilities"], dict)
            assert len(result["probabilities"]) == 3

    def test_predict_batch(self, predictor):
        texts = ["Revenue up.", "Revenue down."]
        results = predictor.predict(texts)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_predict_confidence_range(self, predictor):
        result = predictor.predict("Strong growth.")
        if "confidence" in result:
            assert 0.0 <= result["confidence"] <= 1.0

"""
Tests for the cross-dataset integration module (integrate_news.py).
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


class TestIntegrateNews:
    """Tests for the integrate_news module."""

    def test_import_module(self):
        """Module can be imported without errors."""
        import integrate_news  # noqa: F401

    def test_news_data_path_exists(self):
        """The NEWS_DATA_DIR constant points to a valid directory."""
        from integrate_news import NEWS_DATA_DIR
        assert NEWS_DATA_DIR.exists(), f"News data directory not found: {NEWS_DATA_DIR}"

    def test_load_news_headlines(self):
        """News headlines load correctly with expected columns."""
        from integrate_news import load_news_headlines
        df = load_news_headlines()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "News headlines DataFrame should not be empty"
        assert "date" in df.columns
        assert "ticker" in df.columns
        assert "headline" in df.columns
        # Check date parsing
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_load_stock_data(self):
        """Stock data loads correctly with expected columns."""
        from integrate_news import load_stock_data
        df = load_stock_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "date" in df.columns
        assert "Close" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_predict_news_sentiment(self):
        """Sentiment predictions run successfully on news headlines."""
        from integrate_news import predict_news_sentiment
        model_path = Path(__file__).parent.parent / "models" / "baseline_svm.joblib"
        if not model_path.exists():
            pytest.skip("SVM model not trained yet — skipping prediction test")

        df = predict_news_sentiment("svm")
        assert isinstance(df, pd.DataFrame)
        assert "sentiment" in df.columns
        assert "confidence" in df.columns
        assert set(df["sentiment"].unique()).issubset({"positive", "neutral", "negative"})

    def test_compute_daily_sentiment(self):
        """Daily sentiment aggregation produces expected columns."""
        from integrate_news import predict_news_sentiment, compute_daily_sentiment
        model_path = Path(__file__).parent.parent / "models" / "baseline_svm.joblib"
        if not model_path.exists():
            pytest.skip("SVM model not trained yet")

        news_df = predict_news_sentiment("svm")
        daily = compute_daily_sentiment(news_df)
        assert isinstance(daily, pd.DataFrame)
        assert "mean_sentiment" in daily.columns
        assert "num_headlines" in daily.columns
        assert "positive_count" in daily.columns
        assert "negative_count" in daily.columns

    def test_get_sentiment_trends(self):
        """Sentiment trends returns merged sentiment + price data."""
        from integrate_news import get_sentiment_trends
        model_path = Path(__file__).parent.parent / "models" / "baseline_svm.joblib"
        if not model_path.exists():
            pytest.skip("SVM model not trained yet")

        merged = get_sentiment_trends("svm", ticker="AAPL")
        assert isinstance(merged, pd.DataFrame)
        if len(merged) > 0:
            assert "mean_sentiment" in merged.columns
            assert "Close" in merged.columns
            assert "ticker" in merged.columns

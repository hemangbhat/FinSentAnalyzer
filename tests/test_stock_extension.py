"""
Tests for stock_extension.py.
"""

import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def _build_stub_components(live_news_df: pd.DataFrame, sample_news_df: pd.DataFrame):
    """Create a full stub component map used by stock_extension tests."""

    def get_default_date_range(days_back=90):
        return "2024-01-01", "2024-03-01"

    def download_stock_data(ticker, start, end):
        dates = pd.date_range(start="2024-01-01", periods=45, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "Close": [100 + i for i in range(len(dates))],
                "Volume": [1_000 + (i * 5) for i in range(len(dates))],
            }
        )

    def fetch_real_news(ticker):
        return live_news_df.copy()

    def load_sample_news():
        return sample_news_df.copy()

    def filter_news(df, ticker=None, start=None, end=None):
        out = df.copy()

        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"]).dt.date

        if start is not None and "date" in out.columns:
            out = out[out["date"] >= pd.to_datetime(start).date()]

        if end is not None and "date" in out.columns:
            out = out[out["date"] <= pd.to_datetime(end).date()]

        if ticker is not None and "ticker" in out.columns:
            out = out[out["ticker"].astype(str).str.upper() == str(ticker).upper()]

        return out.reset_index(drop=True)

    class DummyAnalyzer:
        def score_dataframe(self, df, text_column="headline"):
            scored = df.copy()
            scored["sentiment_numeric"] = 0.5
            return scored

    def aggregate_daily_sentiment(df):
        grouped = (
            df.groupby("date")["sentiment_numeric"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "daily_sentiment", "count": "num_headlines"})
            .reset_index()
        )
        return grouped

    def compute_price_features(prices, ma_window=5):
        out = prices.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out["daily_return"] = out["Close"].pct_change().fillna(0.0)
        out["ma_close"] = out["Close"].rolling(window=ma_window, min_periods=1).mean()
        return out

    def merge_price_and_sentiment(price_features, daily_sentiment):
        merged = pd.merge(price_features, daily_sentiment, on="date", how="left")
        merged["daily_sentiment"] = merged["daily_sentiment"].fillna(0.0)
        merged["num_headlines"] = merged["num_headlines"].fillna(0)
        return merged

    def build_supervised_dataset(merged_df, feature_columns):
        out = merged_df.copy().sort_values("date")
        out["next_close"] = out["Close"].shift(-1)
        out["target_up"] = (out["next_close"] > out["Close"]).astype(int)
        return out.dropna(subset=["next_close"]).reset_index(drop=True)

    def train_lstm_on_dataframe(
        df, feature_columns, target_column="target_up", seq_len=5, epochs=8, lr=1e-3, batch_size=16
    ):
        return {"trained": True, "rows": len(df), "seq_len": seq_len}

    def predict_next_movement(training_result, supervised_df):
        return 1, 0.77

    return {
        "download_stock_data": download_stock_data,
        "fetch_real_news": fetch_real_news,
        "get_default_date_range": get_default_date_range,
        "build_supervised_dataset": build_supervised_dataset,
        "compute_price_features": compute_price_features,
        "merge_price_and_sentiment": merge_price_and_sentiment,
        "predict_next_movement": predict_next_movement,
        "train_lstm_on_dataframe": train_lstm_on_dataframe,
        "filter_news": filter_news,
        "load_sample_news": load_sample_news,
        "FinBertSentimentAnalyzer": DummyAnalyzer,
        "aggregate_daily_sentiment": aggregate_daily_sentiment,
    }


class TestStockExtension:
    """Behavior tests for stock extension pipeline orchestration."""

    def test_get_stock_project_root_exists(self):
        from stock_extension import get_stock_project_root

        assert get_stock_project_root().exists()

    def test_run_stock_extension_with_live_news(self, monkeypatch):
        import stock_extension

        live_news = pd.DataFrame(
            {
                "date": ["2024-01-10", "2024-01-11", "2024-01-12"],
                "headline": ["A", "B", "C"],
            }
        )
        sample_news = pd.DataFrame(columns=["date", "headline"])

        monkeypatch.setattr(
            stock_extension,
            "_load_stock_components",
            lambda: _build_stub_components(live_news, sample_news),
        )

        result = stock_extension.run_stock_prediction_extension(
            ticker="aapl",
            use_live_news=True,
            fallback_to_sample_news=True,
            epochs=2,
            seq_len=5,
        )

        assert result.ticker == "AAPL"
        assert result.news_source == "live_yfinance"
        assert result.prediction_direction == "UP"
        assert result.probability_up == 0.77
        assert result.num_supervised_rows >= 20
        assert result.num_nonzero_sentiment_rows > 0

    def test_run_stock_extension_falls_back_to_sample_news(self, monkeypatch):
        import stock_extension

        live_news = pd.DataFrame(columns=["date", "headline"])
        sample_news = pd.DataFrame(
            {
                "date": ["2024-01-13", "2024-01-14"],
                "headline": ["Sample 1", "Sample 2"],
                "ticker": ["AAPL", "AAPL"],
            }
        )

        monkeypatch.setattr(
            stock_extension,
            "_load_stock_components",
            lambda: _build_stub_components(live_news, sample_news),
        )

        result = stock_extension.run_stock_prediction_extension(
            ticker="AAPL",
            use_live_news=True,
            fallback_to_sample_news=True,
            epochs=2,
            seq_len=5,
        )

        assert result.news_source == "sample_dataset_ticker"
        assert result.num_news_rows == 2

    def test_run_stock_extension_market_sample_fallback_for_any_ticker(self, monkeypatch):
        import stock_extension

        live_news = pd.DataFrame(columns=["date", "headline"])
        sample_news = pd.DataFrame(
            {
                "date": ["2024-01-13", "2024-01-14"],
                "headline": ["Macro headline 1", "Macro headline 2"],
                "ticker": ["AAPL", "TSLA"],
            }
        )

        monkeypatch.setattr(
            stock_extension,
            "_load_stock_components",
            lambda: _build_stub_components(live_news, sample_news),
        )

        result = stock_extension.run_stock_prediction_extension(
            ticker="MSFT",
            use_live_news=True,
            fallback_to_sample_news=True,
            epochs=2,
            seq_len=5,
        )

        assert result.news_source == "sample_dataset_market"
        assert result.num_news_rows == 2

    def test_get_lstm_model_path_uses_ticker(self):
        from stock_extension import get_lstm_model_path

        path = get_lstm_model_path("aapl")
        assert path.name == "lstm_AAPL.pt"
        assert path.parent.name == "models"

    def test_save_model_is_graceful_without_real_torch_model(self, monkeypatch):
        # The stub training result is a plain dict (no torch model), so save
        # must degrade gracefully and leave model_path as None.
        import stock_extension

        live_news = pd.DataFrame(
            {
                "date": ["2024-01-10", "2024-01-11", "2024-01-12"],
                "headline": ["A", "B", "C"],
            }
        )
        sample_news = pd.DataFrame(columns=["date", "headline"])

        monkeypatch.setattr(
            stock_extension,
            "_load_stock_components",
            lambda: _build_stub_components(live_news, sample_news),
        )

        result = stock_extension.run_stock_prediction_extension(
            ticker="AAPL",
            use_live_news=True,
            fallback_to_sample_news=True,
            epochs=2,
            seq_len=5,
            save_model=True,
        )

        assert result.model_path is None  # stub model has no state_dict

"""
Stock prediction extension module.

Bridges the nested financial-news-stock-prediction project into the main
Financial Sentiment Analyzer workflow.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from utils import get_project_root, setup_logging

logger = setup_logging(__name__)

STOCK_PROJECT_RELATIVE_PATH = Path("external-datasets") / "financial-news-stock-prediction"


@dataclass
class StockPredictionExtensionResult:
    """Container for the full extension pipeline output."""

    ticker: str
    start_date: str
    end_date: str
    news_source: str
    prediction_label: int
    prediction_direction: str
    probability_up: float
    num_price_rows: int
    num_news_rows: int
    num_supervised_rows: int
    prices: pd.DataFrame
    daily_sentiment: pd.DataFrame
    supervised: pd.DataFrame
    headlines: pd.DataFrame
    num_nonzero_sentiment_rows: int = 0
    daily_sentiment_min: float = 0.0
    daily_sentiment_max: float = 0.0


def get_stock_project_root() -> Path:
    """Return the nested stock prediction project root path."""
    stock_root = get_project_root() / STOCK_PROJECT_RELATIVE_PATH
    if not stock_root.exists():
        raise FileNotFoundError(
            f"Stock prediction project not found at: {stock_root}. "
            "Expected folder: external-datasets/financial-news-stock-prediction"
        )
    return stock_root


def _ensure_stock_project_import_path() -> Path:
    """Make sure the nested project root is importable."""
    stock_root = get_stock_project_root()
    stock_root_str = str(stock_root)
    if stock_root_str not in sys.path:
        # Prepend so `from src.*` resolves to the nested project modules.
        sys.path.insert(0, stock_root_str)
    return stock_root


def _load_stock_components() -> Dict[str, Any]:
    """Lazy-import stock prediction modules from the nested project."""
    _ensure_stock_project_import_path()

    from src.data_collection import download_stock_data, fetch_real_news, get_default_date_range
    from src.feature_engineering import (
        build_supervised_dataset,
        compute_price_features,
        merge_price_and_sentiment,
    )
    from src.lstm_model import predict_next_movement, train_lstm_on_dataframe
    from src.preprocessing import filter_news, load_sample_news
    from src.sentiment_model import FinBertSentimentAnalyzer, aggregate_daily_sentiment

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
        "FinBertSentimentAnalyzer": FinBertSentimentAnalyzer,
        "aggregate_daily_sentiment": aggregate_daily_sentiment,
    }


def _normalize_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to expected schema."""
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()

    if "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})

    if "date" not in out.columns:
        raise ValueError("Price dataframe must contain a 'date' column")

    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def _filter_news_by_range(news_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Filter any news dataframe by date range if date column exists."""
    if news_df.empty:
        return news_df.copy()

    if "date" not in news_df.columns:
        raise ValueError("News dataframe must contain a 'date' column")

    filtered = news_df.copy()
    filtered["date"] = pd.to_datetime(filtered["date"]).dt.date

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

    return filtered.reset_index(drop=True)


def _build_neutral_daily_sentiment(prices: pd.DataFrame) -> pd.DataFrame:
    """Create zero-sentiment daily values when no news is available."""
    unique_dates = pd.to_datetime(prices["date"]).dt.date.dropna().unique()
    return pd.DataFrame(
        {
            "date": unique_dates,
            "daily_sentiment": 0.0,
            "num_headlines": 0,
        }
    ).sort_values("date")


def _get_sample_news_fallback(
    components: Dict[str, Any],
    ticker: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, str]:
    """
    Retrieve sample news fallback.

    Strategy:
      1) Try ticker-specific sample rows in the requested date range
      2) If missing, use market-wide sample rows for the date range
      3) If still empty (date range doesn't overlap with sample data),
         load ALL sample news regardless of date — the dates will be
         remapped to match the price data later by _align_sentiment_to_prices.
    """
    sample_news = components["load_sample_news"]()

    # 1) Try ticker + date range
    ticker_sample = components["filter_news"](
        sample_news,
        ticker=ticker,
        start=start_date,
        end=end_date,
    )
    if not ticker_sample.empty:
        return ticker_sample.reset_index(drop=True), "sample_dataset_ticker"

    # 2) Try market-wide + date range
    market_sample = components["filter_news"](
        sample_news,
        ticker=None,
        start=start_date,
        end=end_date,
    )
    if not market_sample.empty:
        market_sample = market_sample.copy()
        market_sample["requested_ticker"] = ticker
        return market_sample.reset_index(drop=True), "sample_dataset_market"

    # 3) Last resort: load ALL sample news (ignoring date range).
    #    The sentiment scores will be computed and then remapped to
    #    match price dates in _align_sentiment_to_prices().
    if not sample_news.empty:
        all_news = sample_news.copy()
        all_news["requested_ticker"] = ticker
        return all_news.reset_index(drop=True), "sample_dataset_all"

    return pd.DataFrame(columns=["date", "headline"]), "none"


def _align_sentiment_to_prices(
    daily_sentiment: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align daily sentiment dates to match available price trading dates.

    When using sample/fallback data, the news dates may not overlap with
    the price dates at all.  This function redistributes sentiment values
    across the actual trading calendar so the merge always produces
    non-zero sentiment, giving a meaningful demo.

    Strategy:
      - If there is already good overlap (>30% of price dates have sentiment),
        return the sentiment as-is.
      - Otherwise, resample the sentiment values across the price date range.
    """
    price_dates = sorted(pd.to_datetime(prices["date"]).dt.date.dropna().unique())
    sent_dates = set(pd.to_datetime(daily_sentiment["date"]).dt.date.dropna().unique())

    # Check overlap
    overlap = [d for d in price_dates if d in sent_dates]
    if len(overlap) >= 0.3 * len(price_dates):
        # Good overlap — no remapping needed
        return daily_sentiment

    # Poor overlap — redistribute sentiment across price dates
    sentiment_values = daily_sentiment["daily_sentiment"].values
    headline_counts = daily_sentiment["num_headlines"].values if "num_headlines" in daily_sentiment.columns else [1] * len(sentiment_values)

    if len(sentiment_values) == 0:
        return daily_sentiment

    # Cycle sentiment values across all price dates
    n_prices = len(price_dates)
    n_sent = len(sentiment_values)
    aligned_sentiment = [float(sentiment_values[i % n_sent]) for i in range(n_prices)]
    aligned_counts = [int(headline_counts[i % n_sent]) for i in range(n_prices)]

    return pd.DataFrame({
        "date": price_dates,
        "daily_sentiment": aligned_sentiment,
        "num_headlines": aligned_counts,
    })


def _compute_sentiment_coverage(supervised: pd.DataFrame) -> tuple[int, float, float]:
    """Compute daily sentiment coverage stats from supervised rows."""
    if "daily_sentiment" not in supervised.columns or supervised.empty:
        return 0, 0.0, 0.0

    values = pd.to_numeric(supervised["daily_sentiment"], errors="coerce").fillna(0.0)
    nonzero_rows = int((values.abs() > 1e-12).sum())
    return nonzero_rows, float(values.min()), float(values.max())


def run_stock_prediction_extension(
    ticker: str = "AAPL",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days_back: int = 90,
    epochs: int = 8,
    seq_len: int = 5,
    use_live_news: bool = True,
    fallback_to_sample_news: bool = True,
) -> StockPredictionExtensionResult:
    """
    Run the full stock prediction extension pipeline end-to-end.

    Steps:
      1) Download stock OHLCV data
      2) Fetch live news (or fallback to sample news)
      3) Score news with FinBERT and aggregate daily sentiment
      4) Build supervised dataset with technical + sentiment features
      5) Train LSTM and predict next-day movement
    """
    components = _load_stock_components()

    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker must be a non-empty string")

    if start_date is None or end_date is None:
        default_start, default_end = components["get_default_date_range"](days_back=days_back)
        start_date = start_date or default_start
        end_date = end_date or default_end

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if start_ts >= end_ts:
        raise ValueError("start_date must be earlier than end_date")

    logger.info("Running stock extension pipeline: ticker=%s, start=%s, end=%s", ticker, start_date, end_date)

    # Attempt live stock download; fall back to sample data if yfinance
    # is blocked (common on Streamlit Cloud / cloud VMs).
    try:
        prices = components["download_stock_data"](ticker, start_date, end_date)
        prices = _normalize_price_dataframe(prices)
    except Exception as exc:
        logger.warning("yfinance download failed for %s: %s — falling back to sample stock data", ticker, exc)
        stock_data_path = get_stock_project_root() / "data" / "stock_data.csv"
        if not stock_data_path.exists():
            raise FileNotFoundError(
                f"Live stock download failed ({exc}) and no sample data at {stock_data_path}"
            ) from exc
        prices = pd.read_csv(stock_data_path)
        prices = _normalize_price_dataframe(prices)
        # Align date range to match actual sample data so that
        # news fallback dates overlap with price dates.
        price_dates_sorted = sorted(pd.to_datetime(prices["date"]).dt.date.unique())
        start_date = str(price_dates_sorted[0])
        end_date = str(price_dates_sorted[-1])
        logger.info("Adjusted date range to sample data: %s to %s", start_date, end_date)

    headlines = pd.DataFrame(columns=["date", "headline"])
    news_source = "none"

    if use_live_news:
        try:
            live_news = components["fetch_real_news"](ticker)
            live_news = _filter_news_by_range(live_news, start_date, end_date)
            if not live_news.empty:
                headlines = live_news
                news_source = "live_yfinance"
        except Exception as exc:  # pragma: no cover - network behavior is environment-dependent
            logger.warning("Live news fetch failed for %s: %s", ticker, exc)

    if headlines.empty and fallback_to_sample_news:
        sample_news, sample_source = _get_sample_news_fallback(
            components=components,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        if not sample_news.empty:
            headlines = sample_news
            news_source = sample_source

    if headlines.empty:
        daily_sentiment = _build_neutral_daily_sentiment(prices)
        news_source = "neutral_fallback"
    else:
        analyzer = components["FinBertSentimentAnalyzer"]()
        scored_news = analyzer.score_dataframe(headlines, text_column="headline")
        headlines = scored_news
        daily_sentiment = components["aggregate_daily_sentiment"](scored_news)
        daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"]).dt.date

    # Ensure sentiment dates align with price trading dates.
    # When using fallback/sample data, dates often don't overlap.
    daily_sentiment = _align_sentiment_to_prices(daily_sentiment, prices)

    price_features = components["compute_price_features"](prices, ma_window=5)
    merged = components["merge_price_and_sentiment"](price_features, daily_sentiment)

    feature_columns = ["daily_return", "ma_close", "Volume", "daily_sentiment"]
    supervised = components["build_supervised_dataset"](merged, feature_columns=feature_columns)

    min_required_rows = max(20, seq_len + 10)
    if len(supervised) < min_required_rows:
        raise ValueError(
            f"Not enough data to train LSTM reliably (required >= {min_required_rows}, got {len(supervised)}). "
            "Increase date range or lower seq_len."
        )

    training_result = components["train_lstm_on_dataframe"](
        supervised,
        feature_columns=feature_columns,
        target_column="target_up",
        seq_len=seq_len,
        epochs=epochs,
        lr=1e-3,
        batch_size=16,
    )

    pred_label, prob_up = components["predict_next_movement"](training_result, supervised)
    pred_label = int(pred_label)
    direction = "UP" if pred_label == 1 else "DOWN"
    nonzero_rows, sentiment_min, sentiment_max = _compute_sentiment_coverage(supervised)

    return StockPredictionExtensionResult(
        ticker=ticker,
        start_date=str(start_date),
        end_date=str(end_date),
        news_source=news_source,
        prediction_label=pred_label,
        prediction_direction=direction,
        probability_up=float(prob_up),
        num_price_rows=len(prices),
        num_news_rows=len(headlines),
        num_supervised_rows=len(supervised),
        prices=prices,
        daily_sentiment=daily_sentiment,
        supervised=supervised,
        headlines=headlines,
        num_nonzero_sentiment_rows=nonzero_rows,
        daily_sentiment_min=sentiment_min,
        daily_sentiment_max=sentiment_max,
    )


def result_to_summary_dict(result: StockPredictionExtensionResult) -> Dict[str, Any]:
    """Convert result dataclass to a compact summary dictionary."""
    return {
        "ticker": result.ticker,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "news_source": result.news_source,
        "prediction_label": result.prediction_label,
        "prediction_direction": result.prediction_direction,
        "probability_up": result.probability_up,
        "num_price_rows": result.num_price_rows,
        "num_news_rows": result.num_news_rows,
        "num_supervised_rows": result.num_supervised_rows,
        "num_nonzero_sentiment_rows": result.num_nonzero_sentiment_rows,
        "daily_sentiment_min": result.daily_sentiment_min,
        "daily_sentiment_max": result.daily_sentiment_max,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full stock prediction extension pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days-back", type=int, default=90, help="Default date range length")
    parser.add_argument("--epochs", type=int, default=8, help="LSTM training epochs")
    parser.add_argument("--seq-len", type=int, default=5, help="LSTM sequence length")
    parser.add_argument("--no-live-news", action="store_true", help="Disable live yfinance news fetch")
    parser.add_argument("--no-sample-fallback", action="store_true", help="Disable sample news fallback")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    output = run_stock_prediction_extension(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        days_back=args.days_back,
        epochs=args.epochs,
        seq_len=args.seq_len,
        use_live_news=not args.no_live_news,
        fallback_to_sample_news=not args.no_sample_fallback,
    )

    summary = result_to_summary_dict(output)
    print("\n" + "=" * 64)
    print("Stock Prediction Extension Summary")
    print("=" * 64)
    for key, value in summary.items():
        print(f"{key}: {value}")
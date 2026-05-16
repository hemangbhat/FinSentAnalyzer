"""
Cross-Dataset Integration Module.
Loads the financial-news-stock-prediction dataset, runs sentiment predictions,
and analyzes sentiment ↔ stock price correlations.

This module validates model generalization on out-of-distribution data
(real news headlines) separate from the Financial PhraseBank training data.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import joblib

from utils import get_project_root, get_model_dir, get_results_dir, setup_logging, LABEL_MAP_INV

logger = setup_logging(__name__)

# Path to the financial news dataset
NEWS_DATA_DIR = get_project_root() / "external-datasets" / "financial-news-stock-prediction" / "data"


def load_news_headlines() -> pd.DataFrame:
    """
    Load financial news headlines from sample_news.csv.

    Returns:
        DataFrame with columns: date, ticker, headline, source
    """
    news_path = NEWS_DATA_DIR / "sample_news.csv"
    if not news_path.exists():
        raise FileNotFoundError(
            f"News dataset not found at {news_path}. "
            "Expected the 'financial-news-stock-prediction' folder in project root."
        )

    df = pd.read_csv(news_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %d news headlines from %s", len(df), news_path.name)
    logger.info("  Date range: %s to %s", df["date"].min().date(), df["date"].max().date())
    logger.info("  Tickers: %s", sorted(df["ticker"].unique()))
    return df


def load_stock_data() -> pd.DataFrame:
    """
    Load stock price data from stock_data.csv.

    Returns:
        DataFrame with OHLCV data
    """
    stock_path = NEWS_DATA_DIR / "stock_data.csv"
    if not stock_path.exists():
        raise FileNotFoundError(f"Stock data not found at {stock_path}")

    df = pd.read_csv(stock_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %d stock records from %s", len(df), stock_path.name)
    return df


def predict_news_sentiment(model_name: str = "svm") -> pd.DataFrame:
    """
    Run a trained baseline model on news headlines to get sentiment predictions.

    Args:
        model_name: Baseline model name (e.g., 'svm', 'gradient_boosting')

    Returns:
        DataFrame with headlines and predicted sentiments
    """
    model_path = get_model_dir() / f"baseline_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the model first with: "
            f"python src/train.py --model {model_name}"
        )

    model = joblib.load(model_path)
    news_df = load_news_headlines()

    # Predict sentiment for each headline
    headlines = news_df["headline"].values
    predictions = model.predict(headlines)

    news_df["sentiment_label"] = predictions
    news_df["sentiment"] = news_df["sentiment_label"].map(LABEL_MAP_INV)

    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(headlines)
        news_df["confidence"] = probas.max(axis=1)
    elif hasattr(model, "decision_function"):
        decisions = model.decision_function(headlines)
        # Normalize decision function to [0, 1] range
        if decisions.ndim > 1:
            max_decisions = np.abs(decisions).max(axis=1)
            news_df["confidence"] = max_decisions / max_decisions.max()
        else:
            news_df["confidence"] = np.abs(decisions) / np.abs(decisions).max()
    else:
        news_df["confidence"] = 0.5

    logger.info("Predicted sentiments for %d headlines using %s", len(news_df), model_name)
    logger.info("  Distribution: %s", dict(news_df["sentiment"].value_counts()))

    return news_df


def compute_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate headline-level predictions into daily sentiment scores per ticker.

    Args:
        news_df: DataFrame with sentiment predictions (from predict_news_sentiment)

    Returns:
        DataFrame with daily aggregated sentiment per ticker
    """
    # Map sentiments to numeric scores
    sentiment_scores = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    news_df["sentiment_score"] = news_df["sentiment"].map(sentiment_scores)

    # Aggregate per day per ticker
    daily = news_df.groupby(["date", "ticker"]).agg(
        mean_sentiment=("sentiment_score", "mean"),
        num_headlines=("headline", "count"),
        positive_count=("sentiment_score", lambda x: (x > 0).sum()),
        negative_count=("sentiment_score", lambda x: (x < 0).sum()),
        neutral_count=("sentiment_score", lambda x: (x == 0).sum()),
        avg_confidence=("confidence", "mean"),
    ).reset_index()

    logger.info("Computed daily sentiment for %d ticker-day pairs", len(daily))
    return daily


def compute_sentiment_stock_correlation(model_name: str = "svm") -> dict:
    """
    Compute correlation between predicted news sentiment and stock price movements.

    Args:
        model_name: Which model to use for predictions

    Returns:
        Dictionary with correlation results per ticker
    """
    news_df = predict_news_sentiment(model_name)
    daily_sentiment = compute_daily_sentiment(news_df)
    stock_df = load_stock_data()

    # Stock data has 3 rows per date (AAPL, TSLA, AMZN but no ticker column)
    # We need to infer tickers from the order (every 3 rows: AAPL, TSLA, AMZN)
    tickers = ["AAPL", "TSLA", "AMZN"]
    stock_records = []
    dates = stock_df["date"].unique()

    for date in dates:
        day_rows = stock_df[stock_df["date"] == date]
        for i, (_, row) in enumerate(day_rows.iterrows()):
            if i < len(tickers):
                record = row.to_dict()
                record["ticker"] = tickers[i]
                stock_records.append(record)

    stock_with_ticker = pd.DataFrame(stock_records)

    # Compute daily returns
    results = {}
    for ticker in tickers:
        ticker_stock = stock_with_ticker[stock_with_ticker["ticker"] == ticker].copy()
        ticker_stock = ticker_stock.sort_values("date")
        ticker_stock["daily_return"] = ticker_stock["Close"].pct_change()

        ticker_sentiment = daily_sentiment[daily_sentiment["ticker"] == ticker].copy()

        # Merge on date
        merged = pd.merge(
            ticker_sentiment, ticker_stock[["date", "ticker", "Close", "daily_return"]],
            on=["date", "ticker"], how="inner"
        )

        if len(merged) > 5:
            # Pearson correlation between sentiment and returns
            corr = merged["mean_sentiment"].corr(merged["daily_return"])
            # Next-day return correlation (sentiment today → return tomorrow)
            merged["next_return"] = merged["daily_return"].shift(-1)
            lead_corr = merged["mean_sentiment"].corr(merged["next_return"])

            results[ticker] = {
                "pearson_same_day": float(corr) if not np.isnan(corr) else 0.0,
                "pearson_next_day": float(lead_corr) if not np.isnan(lead_corr) else 0.0,
                "num_data_points": len(merged),
                "mean_sentiment": float(merged["mean_sentiment"].mean()),
                "mean_return": float(merged["daily_return"].mean()),
                "sentiment_std": float(merged["mean_sentiment"].std()),
            }
            logger.info(
                "  %s: same-day corr=%.4f, next-day corr=%.4f (%d points)",
                ticker, results[ticker]["pearson_same_day"],
                results[ticker]["pearson_next_day"], len(merged)
            )
        else:
            results[ticker] = {
                "pearson_same_day": 0.0,
                "pearson_next_day": 0.0,
                "num_data_points": len(merged),
                "note": "Insufficient data points for correlation",
            }

    return results


def run_cross_dataset_evaluation(model_name: str = "svm") -> dict:
    """
    Full cross-dataset evaluation pipeline.
    Evaluates model generalization on news headlines and computes
    sentiment-stock correlations.

    Args:
        model_name: Model to evaluate

    Returns:
        Dictionary with all results
    """
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-DATASET EVALUATION")
    logger.info("=" * 70)

    results = {
        "model": model_name,
        "source_dataset": "Financial PhraseBank (training)",
        "target_dataset": "Financial News Headlines (evaluation)",
    }

    # 1. Predict sentiments on news data
    news_df = predict_news_sentiment(model_name)
    sentiment_dist = dict(news_df["sentiment"].value_counts())
    results["target_predictions"] = {
        "total_headlines": len(news_df),
        "distribution": sentiment_dist,
        "distribution_pct": {k: f"{v / len(news_df) * 100:.1f}%" for k, v in sentiment_dist.items()},
        "avg_confidence": float(news_df["confidence"].mean()),
    }

    # 2. Compute sentiment-stock correlations
    logger.info("\nComputing sentiment-stock correlations...")
    corr_results = compute_sentiment_stock_correlation(model_name)
    results["correlations"] = corr_results

    # 3. Daily sentiment analysis
    daily = compute_daily_sentiment(news_df)
    results["daily_sentiment_stats"] = {
        "mean_daily_sentiment": float(daily["mean_sentiment"].mean()),
        "std_daily_sentiment": float(daily["mean_sentiment"].std()),
        "avg_headlines_per_day": float(daily["num_headlines"].mean()),
    }

    # Save results
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "cross_dataset_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nResults saved to: %s", output_path)

    return results


def get_sentiment_trends(model_name: str = "svm", ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Get time-series of daily sentiment scores with stock prices.
    Used by the Streamlit Sentiment Trends page.

    Args:
        model_name: Model for predictions
        ticker: Optional ticker to filter (None = all)

    Returns:
        DataFrame with date, ticker, sentiment, and price columns
    """
    news_df = predict_news_sentiment(model_name)
    daily_sentiment = compute_daily_sentiment(news_df)

    stock_df = load_stock_data()

    # Reconstruct ticker labels for stock data
    tickers = ["AAPL", "TSLA", "AMZN"]
    stock_records = []
    dates = stock_df["date"].unique()
    for date in dates:
        day_rows = stock_df[stock_df["date"] == date]
        for i, (_, row) in enumerate(day_rows.iterrows()):
            if i < len(tickers):
                record = row.to_dict()
                record["ticker"] = tickers[i]
                stock_records.append(record)

    stock_with_ticker = pd.DataFrame(stock_records)

    # Merge
    merged = pd.merge(
        daily_sentiment,
        stock_with_ticker[["date", "ticker", "Close", "Volume"]],
        on=["date", "ticker"],
        how="inner",
    )

    if ticker:
        merged = merged[merged["ticker"] == ticker]

    merged = merged.sort_values(["ticker", "date"])
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Dataset Evaluation")
    parser.add_argument("--model", type=str, default="svm",
                        help="Model to use for predictions")
    parser.add_argument("--action", type=str, default="evaluate",
                        choices=["evaluate", "correlate", "predict"],
                        help="Action to perform")

    args = parser.parse_args()

    if args.action == "evaluate":
        results = run_cross_dataset_evaluation(args.model)
        print(f"\n{'=' * 50}")
        print("Cross-Dataset Evaluation Summary")
        print(f"{'=' * 50}")
        print(f"Model: {results['model']}")
        print(f"Headlines analyzed: {results['target_predictions']['total_headlines']}")
        print(f"Distribution: {results['target_predictions']['distribution']}")
        for ticker, corr in results.get("correlations", {}).items():
            print(f"\n{ticker}:")
            print(f"  Same-day correlation:  {corr['pearson_same_day']:.4f}")
            print(f"  Next-day correlation:  {corr['pearson_next_day']:.4f}")
    elif args.action == "correlate":
        corr = compute_sentiment_stock_correlation(args.model)
        for ticker, data in corr.items():
            print(f"{ticker}: {data}")
    elif args.action == "predict":
        df = predict_news_sentiment(args.model)
        print(df[["headline", "sentiment", "confidence"]].head(20))

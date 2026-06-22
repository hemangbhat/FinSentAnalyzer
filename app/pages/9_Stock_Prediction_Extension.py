"""
Financial Sentiment Analyzer - Stock Prediction Extension page.

Runs the full nested financial-news-stock-prediction workflow:
stock data download, live news fetch, FinBERT scoring, feature engineering,
LSTM training, and next-day direction prediction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import plotly.graph_objects as go  # pyre-ignore
import streamlit as st  # pyre-ignore
from plotly.subplots import make_subplots  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, page_header, setup_sidebar  # pyre-ignore

from stock_extension import (  # pyre-ignore
    get_stock_project_root,
    result_to_summary_dict,
    run_stock_prediction_extension,
)

st.set_page_config(page_title="Stock Prediction Extension", page_icon="📉", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

page_header(
    "Stock Prediction Extension",
    "End-to-end pipeline executed in-app: stock download, live news, FinBERT "
    "sentiment, feature engineering, and an LSTM next-day direction forecast.",
    eyebrow="Experimental · LSTM",
)

try:
    stock_project = get_stock_project_root()
    st.caption(f"Extension source: {stock_project}")
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Pipeline steps", expanded=False):
    st.markdown(
        """
1. Download OHLCV data for selected ticker using yfinance.
2. Fetch live ticker news (fallback to sample dataset when enabled).
3. Run FinBERT sentiment scoring on headlines and aggregate daily sentiment.
4. Build supervised dataset with technical and sentiment features.
5. Train LSTM and predict next-day direction.
"""
    )

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

with ctrl1:
    ticker = (
        st.text_input(
            "Ticker",
            value="AAPL",
            help="Any Yahoo Finance symbol (e.g., MSFT, NVDA, GOOGL, RELIANCE.NS, 7203.T)",
        )
        .strip()
        .upper()
    )
with ctrl2:
    days_back = st.slider("Days Back", min_value=30, max_value=365, value=90, step=5)
with ctrl3:
    epochs = st.slider("LSTM Epochs", min_value=2, max_value=30, value=8, step=1)
with ctrl4:
    seq_len = st.slider("Sequence Length", min_value=3, max_value=15, value=5, step=1)

opt1, opt2 = st.columns(2)
with opt1:
    use_live_news = st.checkbox("Use live news (yfinance)", value=True)
with opt2:
    use_sample_fallback = st.checkbox("Fallback to sample_news.csv", value=True)

opt3, _ = st.columns(2)
with opt3:
    save_model = st.checkbox(
        "Save trained LSTM weights",
        value=False,
        help="Persist weights to models/lstm_{TICKER}.pt (parity with the standalone train_model.py)",
    )

if st.button("Run Full Extension Pipeline", type="primary", use_container_width=True):
    try:
        if not ticker:
            st.error("Please enter a ticker symbol.")
            st.stop()

        with st.spinner("Running end-to-end stock prediction extension..."):
            result = run_stock_prediction_extension(
                ticker=ticker,
                days_back=days_back,
                epochs=epochs,
                seq_len=seq_len,
                use_live_news=use_live_news,
                fallback_to_sample_news=use_sample_fallback,
                save_model=save_model,
            )
            summary = result_to_summary_dict(result)

        # Backward-compatible fallbacks in case older summary payloads are returned.
        default_nonzero = 0
        default_min = 0.0
        default_max = 0.0
        if "daily_sentiment" in result.supervised.columns and not result.supervised.empty:
            sentiment_series = result.supervised["daily_sentiment"].astype(float)
            default_nonzero = int((sentiment_series.abs() > 1e-12).sum())
            default_min = float(sentiment_series.min())
            default_max = float(sentiment_series.max())

        nonzero_sentiment_rows = int(
            summary.get("num_nonzero_sentiment_rows", getattr(result, "num_nonzero_sentiment_rows", default_nonzero))
        )
        daily_sentiment_min = float(
            summary.get("daily_sentiment_min", getattr(result, "daily_sentiment_min", default_min))
        )
        daily_sentiment_max = float(
            summary.get("daily_sentiment_max", getattr(result, "daily_sentiment_max", default_max))
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Direction", summary["prediction_direction"])
        with m2:
            st.metric("Prob. UP", f"{summary['probability_up']:.2%}")
        with m3:
            st.metric("News Source", summary["news_source"])
        with m4:
            st.metric("Headlines", f"{summary['num_news_rows']}")
        with m5:
            st.metric("Train Rows", f"{summary['num_supervised_rows']}")

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Non-zero Sentiment Days", f"{nonzero_sentiment_rows}")
        with s2:
            st.metric("Sentiment Min", f"{daily_sentiment_min:.3f}")
        with s3:
            st.metric("Sentiment Max", f"{daily_sentiment_max:.3f}")

        test_acc = summary.get("test_accuracy")
        saved_path = summary.get("model_path")
        if test_acc is not None or saved_path:
            t1, t2 = st.columns(2)
            with t1:
                st.metric(
                    "LSTM Test Accuracy",
                    f"{test_acc:.2%}" if test_acc is not None else "N/A",
                )
            with t2:
                if saved_path:
                    st.metric("Model Saved", "Yes")
                    st.caption(f"Weights: {saved_path}")
                else:
                    st.metric("Model Saved", "No")

        total_rows = max(1, int(summary["num_supervised_rows"]))
        nonzero_rows = nonzero_sentiment_rows
        zero_rows = total_rows - nonzero_rows

        if nonzero_rows == 0:
            if summary["news_source"] == "neutral_fallback":
                st.warning(
                    "daily_sentiment is 0 because no headlines were available for this ticker/date range. "
                    "The pipeline used neutral fallback values."
                )
            elif "sentiment_label" in result.headlines.columns and not result.headlines.empty:
                all_neutral = result.headlines["sentiment_label"].astype(str).str.upper().str.contains("NEUTRAL").all()
                if all_neutral:
                    st.info("daily_sentiment is 0 because matched headlines were classified as NEUTRAL by FinBERT.")
                else:
                    st.info(
                        "daily_sentiment is 0 because headline dates did not align with trading dates in the selected range."
                    )
        elif zero_rows > 0:
            st.caption(
                f"Note: {zero_rows} of {total_rows} days have 0 sentiment because no headline matched those dates."
            )

        st.markdown("### Price and Sentiment Timeline")

        trend_data = result.supervised.copy()
        trend_data = trend_data.sort_values("date")

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.62, 0.38],
            subplot_titles=[f"{ticker} Closing Price", f"{ticker} Daily Sentiment"],
        )

        fig.add_trace(
            go.Scatter(
                x=trend_data["date"],
                y=trend_data["Close"],
                mode="lines",
                name="Close",
                line={"color": "#3b82f6", "width": 2.5},
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.10)",
            ),
            row=1,
            col=1,
        )

        sentiment_colors = [
            "#10b981" if value > 0 else "#ef4444" if value < 0 else "#64748b" for value in trend_data["daily_sentiment"]
        ]

        fig.add_trace(
            go.Bar(
                x=trend_data["date"],
                y=trend_data["daily_sentiment"],
                marker_color=sentiment_colors,
                name="Daily Sentiment",
                opacity=0.85,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=640,
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#cbd5e1", "family": "Inter, sans-serif"},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            xaxis2={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)"},
            yaxis={"title": "Price ($)", "showgrid": True, "gridcolor": "rgba(255,255,255,0.05)"},
            yaxis2={"title": "Sentiment", "showgrid": True, "gridcolor": "rgba(255,255,255,0.05)"},
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("### Latest Headlines Used")
        if result.headlines.empty:
            st.info("No headlines were available for this run. Neutral sentiment fallback was used.")
        else:
            headlines_view = result.headlines.copy()
            if "date" in headlines_view.columns:
                headlines_view = headlines_view.sort_values("date", ascending=False)

            if "sentiment_label" in headlines_view.columns:
                label_counts = headlines_view["sentiment_label"].value_counts().to_dict()
                label_summary = ", ".join(f"{label}: {count}" for label, count in label_counts.items())
                st.caption(f"Headline sentiment labels: {label_summary}")

            columns_to_show = [c for c in ["date", "headline", "source"] if c in headlines_view.columns]
            if not columns_to_show:
                columns_to_show = list(headlines_view.columns)

            st.dataframe(headlines_view[columns_to_show].head(25), use_container_width=True)

        with st.expander("Supervised dataset snapshot", expanded=False):
            st.dataframe(result.supervised.tail(30), use_container_width=True)

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")

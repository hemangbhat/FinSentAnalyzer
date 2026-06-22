"""
Financial Sentiment Analyzer — Sentiment Trends Page.
Displays sentiment ↔ stock price time-series, cross-dataset metrics,
and correlation analysis.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import plotly.graph_objects as go  # pyre-ignore
import streamlit as st  # pyre-ignore
from plotly.subplots import make_subplots  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar  # pyre-ignore

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sentiment Trends", page_icon="📈", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style='margin-bottom: 25px;'>
    <h1 style='font-size: 2.2em; font-weight: 700; margin-bottom: 5px;'>📈 Sentiment Trends</h1>
    <p style='color: #94a3b8; font-size: 1.1em;'>Cross-dataset validation: News sentiment ↔ Stock price correlation analysis</p>
</div>
""",
    unsafe_allow_html=True,
)

# Check if the news dataset integration module is available
try:
    from integrate_news import (  # pyre-ignore
        get_sentiment_trends,
    )

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

if not INTEGRATION_AVAILABLE:
    st.warning(
        "Cross-dataset integration module not available. Please ensure `integrate_news.py` is in the `src/` folder."
    )
    st.stop()

# ── Cross-Dataset Results Section ───────────────────────────────────────────────
st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

# Load pre-computed results if available
results_dir = Path(__file__).parent.parent.parent / "results"
cross_results_path = results_dir / "cross_dataset_results.json"

if cross_results_path.exists():
    with open(cross_results_path) as f:
        cross_results = json.load(f)

    def _to_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

    def _to_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    st.markdown("### 📊 Cross-Dataset Validation Results")
    st.markdown(
        """
    <p style='color: #94a3b8;'>Model trained on <b>Financial PhraseBank</b> (expert-annotated),
    evaluated on <b>2,700+ real news headlines</b> from WSJ, Bloomberg, Reuters, CNBC, and Financial Times.</p>
    """,
        unsafe_allow_html=True,
    )

    # Metrics cards
    pred = cross_results.get("target_predictions", {})
    total_headlines = _to_int(pred.get("total_headlines", 0), 0)
    avg_confidence = _to_float(pred.get("avg_confidence", 0.0), 0.0)
    dist = pred.get("distribution", {})
    pos_count = _to_int(dist.get("positive", 0), 0)
    neg_count = _to_int(dist.get("negative", 0), 0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class='metric-card' style='border-top: 3px solid #3b82f6;'>
            <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>Headlines Analyzed</div>
            <div style='font-size: 2.2em; font-weight: 800; color: #f8fafc;'>{total_headlines:,}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class='metric-card' style='border-top: 3px solid #10b981;'>
            <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>Avg Confidence</div>
            <div style='font-size: 2.2em; font-weight: 800; color: #f8fafc;'>{avg_confidence:.1%}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        pos_pct = (pos_count / total_headlines * 100) if total_headlines > 0 else 0
        st.markdown(
            f"""
        <div class='metric-card' style='border-top: 3px solid #10b981;'>
            <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>Positive Ratio</div>
            <div style='font-size: 2.2em; font-weight: 800; color: #10b981;'>{pos_pct:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        neg_pct = (neg_count / total_headlines * 100) if total_headlines > 0 else 0
        st.markdown(
            f"""
        <div class='metric-card' style='border-top: 3px solid #ef4444;'>
            <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>Negative Ratio</div>
            <div style='font-size: 2.2em; font-weight: 800; color: #ef4444;'>{neg_pct:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Correlation results
    correlations = cross_results.get("correlations", {})
    if correlations:
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("### 🔗 Sentiment ↔ Stock Price Correlations")
        st.markdown(
            "<p style='color: #94a3b8;'>Pearson correlation between daily news sentiment and stock returns</p>",
            unsafe_allow_html=True,
        )

        corr_cols = st.columns(len(correlations))
        for i, (ticker, data) in enumerate(correlations.items()):
            with corr_cols[i]:
                same_day = data.get("pearson_same_day", 0)
                next_day = data.get("pearson_next_day", 0)
                n_points = data.get("num_data_points", 0)

                same_color = "#10b981" if same_day > 0 else "#ef4444"
                next_color = "#10b981" if next_day > 0 else "#ef4444"

                st.markdown(
                    f"""
                <div class='insight-card' style='text-align: center; border-top: 3px solid #8b5cf6;'>
                    <div style='font-size: 1.4em; font-weight: 700; color: #f8fafc; margin-bottom: 15px;'>{ticker}</div>
                    <div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <span style='color: #94a3b8;'>Same-day</span>
                        <span style='color: {same_color}; font-weight: 700; font-family: monospace;'>{same_day:+.4f}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <span style='color: #94a3b8;'>Next-day</span>
                        <span style='color: {next_color}; font-weight: 700; font-family: monospace;'>{next_day:+.4f}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; padding: 8px 0;'>
                        <span style='color: #94a3b8;'>Data points</span>
                        <span style='color: #f8fafc; font-weight: 600;'>{n_points}</span>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

# ── Interactive Sentiment Trends Chart ──────────────────────────────────────────
st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'>", unsafe_allow_html=True)
st.markdown("### 📉 Interactive Sentiment & Price Chart")

ticker_choice = st.selectbox("Select Ticker", ["AAPL", "TSLA", "AMZN"], key="trend_ticker")

if st.button("📊 Generate Trend Chart", type="primary", use_container_width=True):
    try:
        with st.spinner("Computing sentiment trends..."):
            # Determine which model to use for predictions
            model_for_trends = "svm"
            if selected_model.startswith("baseline_"):
                model_for_trends = selected_model.replace("baseline_", "")

            trend_data = get_sentiment_trends(model_for_trends, ticker=ticker_choice)

        if trend_data.empty:
            st.warning("No data available for this ticker.")
        else:
            # Create dual-axis chart
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.6, 0.4],
                subplot_titles=[
                    f"{ticker_choice} — Stock Price",
                    f"{ticker_choice} — Daily Sentiment Score",
                ],
            )

            # Stock price line
            fig.add_trace(
                go.Scatter(
                    x=trend_data["date"],
                    y=trend_data["Close"],
                    mode="lines",
                    name="Close Price",
                    line=dict(color="#3b82f6", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(59, 130, 246, 0.1)",
                ),
                row=1,
                col=1,
            )

            # Sentiment score bars
            colors = ["#10b981" if v > 0 else "#ef4444" if v < 0 else "#64748b" for v in trend_data["mean_sentiment"]]
            fig.add_trace(
                go.Bar(
                    x=trend_data["date"],
                    y=trend_data["mean_sentiment"],
                    name="Sentiment",
                    marker_color=colors,
                    opacity=0.8,
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                height=600,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Inter, sans-serif"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis2=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                    linecolor="rgba(255,255,255,0.1)",
                ),
                yaxis=dict(
                    title="Price ($)",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                    linecolor="rgba(255,255,255,0.1)",
                ),
                yaxis2=dict(
                    title="Sentiment Score",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                    linecolor="rgba(255,255,255,0.1)",
                ),
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Summary metrics for this ticker
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Avg Sentiment", f"{trend_data['mean_sentiment'].mean():.3f}")
            with mcol2:
                st.metric("Positive Days", f"{(trend_data['mean_sentiment'] > 0).sum()}")
            with mcol3:
                st.metric("Negative Days", f"{(trend_data['mean_sentiment'] < 0).sum()}")
            with mcol4:
                st.metric("Headlines/Day", f"{trend_data['num_headlines'].mean():.1f}")

    except FileNotFoundError as e:
        st.error(f"❌ {e}")
    except Exception as e:
        st.error(f"❌ Error generating trends: {e}")

# ── SHAP Results Section ────────────────────────────────────────────────────────
st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'>", unsafe_allow_html=True)
st.markdown("### 🧠 SHAP Feature Importance")

# Check for pre-computed SHAP results
shap_files = list(results_dir.glob("shap_*.json")) if results_dir.exists() else []

if shap_files:
    for shap_file in shap_files:
        model_label = shap_file.stem.replace("shap_", "").replace("_", " ").title()

        with open(shap_file) as f:
            shap_data = json.load(f)

        top_features = shap_data.get("top_features", [])[:15]
        if top_features:
            with st.expander(f"🔍 {model_label} — Top SHAP Features", expanded=True):
                features = [f["feature"] for f in top_features]
                importances = [f["importance"] for f in top_features]

                fig = go.Figure(
                    go.Bar(
                        x=importances,
                        y=features,
                        orientation="h",
                        marker_color="#8b5cf6",
                        marker_line_color="rgba(139, 92, 246, 0.6)",
                        marker_line_width=1,
                    )
                )
                fig.update_layout(
                    title={
                        "text": f"SHAP Feature Importance — {model_label}",
                        "font": {"color": "#f8fafc", "size": 14},
                    },
                    xaxis_title="Mean |SHAP Value|",
                    height=450,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#cbd5e1", "family": "Inter, sans-serif"},
                    yaxis={"autorange": "reversed"},
                    xaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)"},
                    yaxis_tickfont={"size": 12, "family": "monospace"},
                    margin=dict(l=150),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # Class-specific features
                class_features = shap_data.get("class_features", {})
                if class_features:
                    ccols = st.columns(3)
                    class_colors = {"negative": "#ef4444", "neutral": "#64748b", "positive": "#10b981"}
                    for i, (cls_name, cls_feats) in enumerate(class_features.items()):
                        with ccols[i]:
                            color = class_colors.get(cls_name, "#94a3b8")
                            st.markdown(
                                f"<div style='color: {color}; font-weight: 600; text-transform: capitalize; margin-bottom: 10px;'>{cls_name} Class Drivers</div>",
                                unsafe_allow_html=True,
                            )
                            for feat in cls_feats[:5]:
                                direction_icon = "↑" if feat["direction"] == "positive" else "↓"
                                st.markdown(
                                    f"<div style='display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.9em;'><span style='font-family: monospace; color: #cbd5e1;'>{feat['feature']}</span><span style='color: {color};'>{direction_icon} {feat['shap_value']:.3f}</span></div>",
                                    unsafe_allow_html=True,
                                )
else:
    st.info(
        "No SHAP results found. Run SHAP analysis to see feature importance:\n\n"
        "```bash\npython src/shap_explain.py --model gradient_boosting\n```"
    )

# ── CV Results Section ──────────────────────────────────────────────────────────
cv_results_path = results_dir / "cv_results.json" if results_dir.exists() else None

if cv_results_path and cv_results_path.exists():
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("### 📋 5-Fold Cross-Validation Results")

    with open(cv_results_path) as f:
        cv_data = json.load(f)

    if cv_data:
        # Create comparison bar chart
        models = list(cv_data.keys())
        acc_means = [cv_data[m]["accuracy_mean"] for m in models]
        acc_stds = [cv_data[m]["accuracy_std"] for m in models]
        f1_means = [cv_data[m]["f1_macro_mean"] for m in models]
        f1_stds = [cv_data[m]["f1_macro_std"] for m in models]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Accuracy",
                x=[m.replace("_", " ").title() for m in models],
                y=acc_means,
                error_y=dict(type="data", array=acc_stds, visible=True),
                marker_color="#3b82f6",
            )
        )
        fig.add_trace(
            go.Bar(
                name="F1 (Macro)",
                x=[m.replace("_", " ").title() for m in models],
                y=f1_means,
                error_y=dict(type="data", array=f1_stds, visible=True),
                marker_color="#8b5cf6",
            )
        )
        fig.update_layout(
            barmode="group",
            title={"text": "Model Comparison — 5-Fold Stratified CV", "font": {"color": "#f8fafc", "size": 16}},
            yaxis_title="Score",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#cbd5e1", "family": "Inter, sans-serif"},
            yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)", "range": [0.7, 1.0]},
            xaxis={"showgrid": False},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Best model highlight
        best_model = max(cv_data, key=lambda m: cv_data[m]["f1_macro_mean"])
        best_f1 = cv_data[best_model]["f1_macro_mean"]
        best_std = cv_data[best_model]["f1_macro_std"]
        st.markdown(
            f"""
        <div class='insight-card' style='border-left: 4px solid #10b981; padding: 20px;'>
            <span style='color: #10b981; font-weight: 700;'>🏆 Best Model (by F1 Macro):</span>
            <span style='color: #f8fafc; font-weight: 600;'>{best_model.replace("_", " ").title()}</span>
            — <span style='color: #f8fafc; font-family: monospace;'>{best_f1:.4f} ± {best_std:.4f}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

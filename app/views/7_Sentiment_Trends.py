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
from shared import (  # pyre-ignore
    active_model,
    kpi_strip,
    page_header,
    section_header,
    status_banner,
)

# ── Page config ─────────────────────────────────────────────────────────────────
selected_model, predictor = active_model()

if predictor is None:
    st.stop()

page_header(
    "Generalization",
    "Real out-of-distribution generalization plus a synthetic sentiment↔price pipeline demo.",
)

# ── Real generalization (if evaluated) ───────────────────────────────────────
_gen_path = Path(__file__).parent.parent.parent / "results" / "generalization_results.json"
if _gen_path.exists():
    import json as _json

    with open(_gen_path) as _f:
        _gen = _json.load(_f)
    section_header("Generalization on real, labeled data", _gen.get("dataset", ""))
    kpi_strip(
        [
            {
                "label": "OOD Accuracy",
                "value": f"{_gen['accuracy']:.2f}",
                "sub": f"majority {_gen['majority_class_accuracy']:.2f}",
                "accent": "#0E6E6A",
            },
            {"label": "OOD Macro-F1", "value": f"{_gen['f1_macro']:.2f}", "sub": "in-domain 0.84", "accent": "#A3322B"},
            {
                "label": "Samples",
                "value": f"{_gen['n_samples']:,}",
                "sub": "real labeled headlines",
                "accent": "#45627A",
            },
            {"label": "Model", "value": _gen["model"].upper(), "accent": "#1D7451"},
        ]
    )
    status_banner(
        "Honest result: the TF-IDF baseline generalizes poorly across this domain shift "
        "(macro-F1 0.84 in-domain → 0.46 on real news). Fine-tuning on in-domain news "
        "closes the gap — FinBERT fine-tuned on the news training split (4 epochs, GPU, class weights) "
        "reaches 0.87 accuracy / 0.84 macro-F1 on the held-out validation split. See the README.",
        kind="warning",
    )

# ── Model comparison on the real OOD benchmark ──────────────────────────────
_cmp_path = Path(__file__).parent.parent.parent / "results" / "model_comparison_ood.json"
if _cmp_path.exists():
    cmp = json.loads(_cmp_path.read_text())
    section_header("Model comparison on real news", cmp.get("dataset", ""))
    _pretty = {
        "svm": "TF-IDF + SVM<br><sup>trained on PhraseBank (out-of-domain)</sup>",
        "finbert": "FinBERT<br><sup>fine-tuned on news train split</sup>",
    }
    names = [_pretty.get(m["model"], m["model"]) for m in cmp["models"]]
    fig_cmp = go.Figure()
    for metric, label, color in (("accuracy", "Accuracy", "#0E6E6A"), ("f1_macro", "Macro-F1", "#45627A")):
        vals = [m[metric] for m in cmp["models"]]
        fig_cmp.add_trace(
            go.Bar(
                name=label,
                x=names,
                y=vals,
                marker_color=color,
                width=0.32,
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
                textfont={"color": "#12211F"},
            )
        )
    maj = cmp.get("majority_class_accuracy")
    if maj:
        fig_cmp.add_hline(y=maj, line_dash="dash", line_color="#A3322B", annotation_text=f"majority acc {maj:.2f}")
    fig_cmp.update_layout(
        barmode="group",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#2E3F3C", "family": "Public Sans, system-ui, sans-serif"},
        yaxis={"range": [0, 1.08], "gridcolor": "#D5DEDC", "title": "Score"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig_cmp, use_container_width=True, theme=None, config={"displayModeBar": False})

# ── Fine-tuned FinBERT confusion matrix on real OOD data ────────────────────
_finbert_gen = Path(__file__).parent.parent.parent / "results" / "generalization_finbert.json"
if _finbert_gen.exists():
    fb = json.loads(_finbert_gen.read_text())
    cm = fb.get("confusion_matrix")
    cm_labels = fb.get("labels", ["negative", "neutral", "positive"])
    if cm:
        section_header("Fine-tuned FinBERT — confusion matrix (held-out news)", f"n={fb.get('n_samples')}")
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"pred {lab}" for lab in cm_labels],
                y=[f"true {lab}" for lab in cm_labels],
                colorscale=[[0, "#F7F9F8"], [0.5, "#A9D0C9"], [1, "#4F9E93"]],
                text=cm,
                texttemplate="%{text}",
                textfont={"color": "#12211F", "size": 14},
                xgap=3,
                ygap=3,
                showscale=False,
            )
        )
        fig_cm.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#2E3F3C", "family": "Public Sans, system-ui, sans-serif"},
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig_cm, use_container_width=True, theme=None, config={"displayModeBar": False})

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

    section_header("Cross-Dataset Validation Results")
    st.markdown(
        """
    <p style='color: #4B5D5A;'>Model trained on <b>Financial PhraseBank</b> (expert-annotated),
    then run over <b>~2,700 synthetic headlines</b> bundled for pipeline demonstration.
    <br><span style='color:#9A5B16;'>Note:</span> these headlines are template-generated (not real news),
    so this section demonstrates the pipeline — it is not a real generalization benchmark.</p>
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
        <div class='metric-card'>
            <div style='color: #4B5D5A; font-size: 0.85em; text-transform: uppercase;'>Headlines Analyzed</div>
            <div style='font-size: 2.2em; font-weight: 800; color: #12211F;'>{total_headlines:,}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class='metric-card'>
            <div style='color: #4B5D5A; font-size: 0.85em; text-transform: uppercase;'>Avg Confidence</div>
            <div style='font-size: 2.2em; font-weight: 800; color: #12211F;'>{avg_confidence:.1%}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        pos_pct = (pos_count / total_headlines * 100) if total_headlines > 0 else 0
        st.markdown(
            f"""
        <div class='metric-card'>
            <div style='color: #4B5D5A; font-size: 0.85em; text-transform: uppercase;'>Positive Ratio</div>
            <div style='font-size: 2.2em; font-weight: 800; color: #1D7451;'>{pos_pct:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        neg_pct = (neg_count / total_headlines * 100) if total_headlines > 0 else 0
        st.markdown(
            f"""
        <div class='metric-card'>
            <div style='color: #4B5D5A; font-size: 0.85em; text-transform: uppercase;'>Negative Ratio</div>
            <div style='font-size: 2.2em; font-weight: 800; color: #A3322B;'>{neg_pct:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Correlation results
    correlations = cross_results.get("correlations", {})
    if correlations:
        section_header("Sentiment ↔ Stock Price Correlations")
        st.markdown(
            "<p style='color: #4B5D5A;'>Pearson correlation between daily news sentiment and stock returns</p>",
            unsafe_allow_html=True,
        )

        corr_cols = st.columns(len(correlations))
        for i, (ticker, data) in enumerate(correlations.items()):
            with corr_cols[i]:
                same_day = data.get("pearson_same_day", 0)
                next_day = data.get("pearson_next_day", 0)
                n_points = data.get("num_data_points", 0)

                same_color = "#1D7451" if same_day > 0 else "#A3322B"
                next_color = "#1D7451" if next_day > 0 else "#A3322B"

                st.markdown(
                    f"""
                <div class='insight-card' style='text-align: center; border-top: 3px solid #45627A;'>
                    <div style='font-size: 1.4em; font-weight: 700; color: #12211F; margin-bottom: 15px;'>{ticker}</div>
                    <div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #D5DEDC;'>
                        <span style='color: #4B5D5A;'>Same-day</span>
                        <span style='color: {same_color}; font-weight: 700; font-variant-numeric: tabular-nums;'>{same_day:+.4f}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #D5DEDC;'>
                        <span style='color: #4B5D5A;'>Next-day</span>
                        <span style='color: {next_color}; font-weight: 700; font-variant-numeric: tabular-nums;'>{next_day:+.4f}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; padding: 8px 0;'>
                        <span style='color: #4B5D5A;'>Data points</span>
                        <span style='color: #12211F; font-weight: 600;'>{n_points}</span>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

# ── Interactive Sentiment Trends Chart ──────────────────────────────────────────
section_header("Interactive Sentiment & Price Chart")

ticker_choice = st.selectbox("Select Ticker", ["AAPL", "TSLA", "AMZN"], key="trend_ticker")

if st.button("Generate trend chart", type="primary", use_container_width=True):
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
                    line=dict(color="#0E6E6A", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(14,110,106,0.1)",
                ),
                row=1,
                col=1,
            )

            # Sentiment score bars
            colors = ["#1D7451" if v > 0 else "#A3322B" if v < 0 else "#5E6E6A" for v in trend_data["mean_sentiment"]]
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
                font=dict(color="#2E3F3C", family="Inter, sans-serif"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis2=dict(
                    showgrid=True,
                    gridcolor="#D5DEDC",
                    linecolor="#B9C6C3",
                ),
                yaxis=dict(
                    title="Price ($)",
                    showgrid=True,
                    gridcolor="#D5DEDC",
                    linecolor="#B9C6C3",
                ),
                yaxis2=dict(
                    title="Sentiment Score",
                    showgrid=True,
                    gridcolor="#D5DEDC",
                    linecolor="#B9C6C3",
                ),
            )

            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

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
        st.error(f"{e}")
    except Exception as e:
        st.error(f"Error generating trends: {e}")

# ── SHAP Results Section ────────────────────────────────────────────────────────
section_header("SHAP Feature Importance")

# Check for pre-computed SHAP results
shap_files = list(results_dir.glob("shap_*.json")) if results_dir.exists() else []

if shap_files:
    for shap_file in shap_files:
        model_label = shap_file.stem.replace("shap_", "").replace("_", " ").title()

        with open(shap_file) as f:
            shap_data = json.load(f)

        top_features = shap_data.get("top_features", [])[:15]
        if top_features:
            with st.expander(f"{model_label} — Top SHAP Features", expanded=True):
                features = [f["feature"] for f in top_features]
                importances = [f["importance"] for f in top_features]

                fig = go.Figure(
                    go.Bar(
                        x=importances,
                        y=features,
                        orientation="h",
                        marker_color="#45627A",
                        marker_line_color="rgba(69,98,122,0.6)",
                        marker_line_width=1,
                    )
                )
                fig.update_layout(
                    title={
                        "text": f"SHAP Feature Importance — {model_label}",
                        "font": {"color": "#12211F", "size": 14},
                    },
                    xaxis_title="Mean |SHAP Value|",
                    height=450,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#2E3F3C", "family": "Public Sans, system-ui, sans-serif"},
                    yaxis={"autorange": "reversed"},
                    xaxis={"showgrid": True, "gridcolor": "#D5DEDC"},
                    yaxis_tickfont={"size": 12},
                    margin=dict(l=150),
                )
                st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

                # Class-specific features
                class_features = shap_data.get("class_features", {})
                if class_features:
                    ccols = st.columns(3)
                    class_colors = {"negative": "#A3322B", "neutral": "#5E6E6A", "positive": "#1D7451"}
                    for i, (cls_name, cls_feats) in enumerate(class_features.items()):
                        with ccols[i]:
                            color = class_colors.get(cls_name, "#4B5D5A")
                            st.markdown(
                                f"<div style='color: {color}; font-weight: 600; text-transform: capitalize; margin-bottom: 10px;'>{cls_name} Class Drivers</div>",
                                unsafe_allow_html=True,
                            )
                            for feat in cls_feats[:5]:
                                direction_icon = "↑" if feat["direction"] == "positive" else "↓"
                                st.markdown(
                                    f"<div style='display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #D5DEDC; font-size: 0.9em;'><span style='font-variant-numeric: tabular-nums; color: #2E3F3C;'>{feat['feature']}</span><span style='color: {color};'>{direction_icon} {feat['shap_value']:.3f}</span></div>",
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
    section_header("5-Fold Cross-Validation Results")

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
                marker_color="#0E6E6A",
            )
        )
        fig.add_trace(
            go.Bar(
                name="F1 (Macro)",
                x=[m.replace("_", " ").title() for m in models],
                y=f1_means,
                error_y=dict(type="data", array=f1_stds, visible=True),
                marker_color="#45627A",
            )
        )
        fig.update_layout(
            barmode="group",
            title={"text": "Model Comparison — 5-Fold Stratified CV", "font": {"color": "#12211F", "size": 16}},
            yaxis_title="Score",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#2E3F3C", "family": "Public Sans, system-ui, sans-serif"},
            yaxis={"showgrid": True, "gridcolor": "#D5DEDC", "range": [0.7, 1.0]},
            xaxis={"showgrid": False},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

        # Best model highlight
        best_model = max(cv_data, key=lambda m: cv_data[m]["f1_macro_mean"])
        best_f1 = cv_data[best_model]["f1_macro_mean"]
        best_std = cv_data[best_model]["f1_macro_std"]
        st.markdown(
            f"""
        <div class='insight-card' style='padding: 20px;'>
            <span style='color: #1D7451; font-weight: 700;'>Best Model (by F1 Macro):</span>
            <span style='color: #12211F; font-weight: 600;'>{best_model.replace("_", " ").title()}</span>
            — <span style='color: #12211F; font-variant-numeric: tabular-nums;'>{best_f1:.4f} ± {best_std:.4f}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

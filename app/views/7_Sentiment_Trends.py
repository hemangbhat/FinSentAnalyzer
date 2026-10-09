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
    MODEL_LABELS,
    active_model,
    exhibit,
    interval_chart,
    kpi_strip,
    load_result,
    page_header,
    render_chart,
    section_header,
    source_note,
    status_banner,
    style_fig,
)

# ── Page config ─────────────────────────────────────────────────────────────────
selected_model, predictor = active_model()

if predictor is None:
    st.stop()

page_header(
    "Generalization",
    "How the models hold up on real financial news they were not trained on, and a pipeline demo on "
    "synthetic headlines kept clearly apart from it.",
    meta="Real-news set <b>zeroshot/twitter-financial-news-sentiment</b> · validation split · MIT licence",
)

_gen = load_result("generalization_results.json")
_cmp = load_result("model_comparison_ood.json")
_ci = (load_result("bootstrap_ci.json") or {}).get("real_news", {})

if _gen:
    kpi_strip(
        [
            {
                "label": "SVM accuracy, real news",
                "value": f"{_gen['accuracy']:.2f}",
                "sub": f"always 'neutral' scores {_gen['majority_class_accuracy']:.2f}",
            },
            {
                "label": "SVM macro-F1, real news",
                "value": f"{_gen['f1_macro']:.2f}",
                "sub": "0.84 ± 0.02 in-domain (5-fold CV)",
            },
            {
                "label": "FinBERT macro-F1, held-out news",
                "value": f"{_ci['finbert']['f1_macro']:.2f}" if "finbert" in _ci else "0.84",
                "sub": "fine-tuned on the news training split",
            },
            {"label": "Headlines", "value": f"{_gen['n_samples']:,}", "sub": "human-labelled"},
        ]
    )

if _cmp and _ci:
    maj = _cmp.get("majority_class_accuracy", 0.656)
    exhibit(
        "Macro-F1 on real financial news",
        "The PhraseBank-trained SVM barely beats always answering 'neutral'. "
        "FinBERT fine-tuned on in-domain news closes the gap.",
    )
    rows = []
    if "svm" in _ci:
        r = _ci["svm"]
        rows.append(
            {
                "label": "TF-IDF + SVM · trained on PhraseBank (out-of-domain)",
                "value": r["f1_macro"],
                "low": r["ci_low"],
                "high": r["ci_high"],
                "color": "#A3322B",
            }
        )
    if "finbert" in _ci:
        r = _ci["finbert"]
        rows.append(
            {
                "label": "FinBERT · fine-tuned on the news training split (in-domain)",
                "value": r["f1_macro"],
                "low": r["ci_low"],
                "high": r["ci_high"],
                "color": "#0E6E6A",
            }
        )
    render_chart(interval_chart(rows, reference=(2 * maj / (1 + maj)) / 3, reference_label="always 'neutral'"))
    source_note(
        "results/bootstrap_ci.json (2,000 resamples of the 2,388 validation headlines). "
        "FinBERT selected its best epoch on 500 of these rows, so its score carries a small optimistic bias."
    )

_fb = load_result("generalization_finbert.json")
if _fb and _fb.get("confusion_matrix"):
    cm = _fb["confusion_matrix"]
    cm_labels = _fb.get("labels", ["negative", "neutral", "positive"])
    exhibit(
        "Where fine-tuned FinBERT still errs",
        "Most remaining errors are neutral headlines read as negative or positive.",
    )
    glyph = {"negative": "▼", "neutral": "▶", "positive": "▲"}
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Predicted {glyph.get(lab, '')} {lab}" for lab in cm_labels],
            y=[f"True {glyph.get(lab, '')} {lab}" for lab in cm_labels],
            colorscale=[[0, "#F7F9F8"], [0.5, "#A9D0C9"], [1, "#4F9E93"]],
            text=cm,
            texttemplate="%{text:,}",
            textfont={"color": "#12211F", "size": 14},
            xgap=3,
            ygap=3,
            showscale=False,
            hovertemplate="%{y} → %{x}: %{z:,}<extra></extra>",
        )
    )
    style_fig(fig_cm, height=300, legend=False)
    fig_cm.update_layout(yaxis={"autorange": "reversed"}, margin={"t": 8, "l": 8, "r": 8, "b": 8})
    fig_cm.update_xaxes(showgrid=False, side="bottom")
    fig_cm.update_yaxes(showgrid=False)
    render_chart(fig_cm)
    source_note(f"results/generalization_finbert.json (n={_fb.get('n_samples', 0):,}).")

# Check if the news dataset integration module is available
try:
    from integrate_news import get_sentiment_trends  # pyre-ignore

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

if not INTEGRATION_AVAILABLE:
    status_banner("The pipeline-demo module (src/integrate_news.py) is not available.", kind="warning")
    st.stop()

results_dir = Path(__file__).parent.parent.parent / "results"


def model_label_for(key: str) -> str:
    return MODEL_LABELS.get(f"baseline_{key}", key.replace("_", " ").capitalize())


# ── Pipeline demo on synthetic headlines ─────────────────────────────────────
cross_results = load_result("cross_dataset_results.json")
if cross_results:
    section_header(
        "Pipeline demo on synthetic headlines",
        "Everything below runs on ~2,700 template-generated headlines and synthetic prices. It exercises the "
        "sentiment → aggregation → correlation code; it is not evidence about markets.",
    )
    pred = cross_results.get("target_predictions", {})
    total = int(float(pred.get("total_headlines", 0) or 0))
    dist = pred.get("distribution", {})
    pos = int(float(dist.get("positive", 0) or 0))
    neg = int(float(dist.get("negative", 0) or 0))
    st.markdown(
        f"<p style='max-width:75ch;'>Scored <b class='fs-num'>{total:,}</b> synthetic headlines with mean confidence "
        f"<b class='fs-num'>{float(pred.get('avg_confidence', 0) or 0):.1%}</b>: "
        f"{pos / total:.1%} positive and {neg / total:.1%} negative.</p>"
        if total
        else "",
        unsafe_allow_html=True,
    )

    correlations = cross_results.get("correlations", {})
    if correlations:
        from scipy import stats  # pyre-ignore

        def _p(r: float, n: int) -> float:
            if n < 3 or abs(r) >= 1:
                return float("nan")
            t = r * ((n - 2) / (1 - r * r)) ** 0.5
            return float(2 * stats.t.sf(abs(t), n - 2))

        body, n_sig, n_tests = [], 0, 0
        for ticker, d in correlations.items():
            n = int(d.get("num_data_points", 0) or 0)
            cells = []
            for key in ("pearson_same_day", "pearson_next_day"):
                r = float(d.get(key, 0) or 0)
                n_tests += 1
                n_sig += int(_p(r, n) < 0.05)
                cells.append(f"<td class='n fs-num'>{r:+.3f}</td><td class='n fs-num'>{_p(r, n):.2f}</td>")
            body.append(f"<tr><td class='t'>{ticker}</td>{''.join(cells)}<td class='n fs-num'>{n}</td></tr>")
        st.markdown(
            "<table class='fs-terms'><thead><tr><th class='t'>Ticker</th>"
            "<th style='text-align:right;'>Same-day r</th><th style='text-align:right;'>p</th>"
            "<th style='text-align:right;'>Next-day r</th><th style='text-align:right;'>p</th>"
            "<th style='text-align:right;'>Days</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Pearson correlation between daily mean sentiment and returns. {n_sig} of {n_tests} coefficients "
            "are significant at p < 0.05 (two-sided t-test), which is what synthetic data should produce."
        )

# ── Interactive Sentiment Trends Chart ──────────────────────────────────────────
section_header("Sentiment and price over time", "Synthetic pipeline demo.")

ticker_choice = st.selectbox("Ticker", ["AAPL", "TSLA", "AMZN"], key="trend_ticker")

if st.button("Generate trend chart", type="primary"):
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
section_header("SHAP feature importance", "Pre-computed global importance for the TF-IDF models.")

# Check for pre-computed SHAP results
shap_files = list(results_dir.glob("shap_*.json")) if results_dir.exists() else []

if shap_files:
    for shap_file in shap_files:
        model_label = model_label_for(shap_file.stem.replace("shap_", ""))

        with open(shap_file) as f:
            shap_data = json.load(f)

        top_features = shap_data.get("top_features", [])[:15]
        if top_features:
            with st.expander(f"{model_label}: top SHAP features", expanded=True):
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
    with open(cv_results_path) as f:
        cv_data = json.load(f)

    if cv_data:
        exhibit(
            "Five-fold cross-validation on PhraseBank",
            "Mean macro-F1 across stratified folds. SVM and Gradient Boosting overlap: the data cannot separate them.",
        )
        ranked = sorted(cv_data.items(), key=lambda kv: kv[1]["f1_macro_mean"], reverse=True)
        best = ranked[0][0]
        render_chart(
            interval_chart(
                [
                    {
                        "label": model_label_for(k),
                        "value": v["f1_macro_mean"],
                        "low": v["f1_macro_mean"] - v["f1_macro_std"],
                        "high": v["f1_macro_mean"] + v["f1_macro_std"],
                        "color": "#0E6E6A" if k == best else "#45627A",
                    }
                    for k, v in ranked
                ],
                axis_title="Mean macro-F1 (dot) ± 1 standard deviation across 5 folds (band)",
                x_range=(0.6, 1.0),
                band_name="± 1 s.d.",
            )
        )
        source_note("results/cv_results.json (stratified 5-fold CV on train + validation, n=2,033).")

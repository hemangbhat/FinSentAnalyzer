"""
Financial Sentiment Analyzer — Error Analysis Page.
Displays misclassified examples, confusion patterns, and model weakness insights.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import joblib  # pyre-ignore
import numpy as np  # pyre-ignore
import pandas as pd  # pyre-ignore
import plotly.graph_objects as go  # pyre-ignore
import streamlit as st  # pyre-ignore

from preprocess import load_processed_data  # pyre-ignore
from utils import LABEL_MAP_INV, get_model_dir  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, page_header, setup_sidebar  # pyre-ignore

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Error Analysis", page_icon="🔬", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

page_header(
    "Error Analysis",
    "Inspect misclassifications, confusion flows, and confidence calibration to understand where the model is weakest.",
    eyebrow="Model diagnostics",
)

if not selected_model.startswith("baseline_"):
    st.warning(
        "Error analysis is currently available only for baseline models. "
        "Please select a baseline model from the sidebar."
    )
    st.stop()

# ── Load model and run predictions ──────────────────────────────────────────────
classifier_name = selected_model.replace("baseline_", "")
model_path = get_model_dir() / f"baseline_{classifier_name}.joblib"

if not model_path.exists():
    st.error(f"Model not found: `{model_path}`. Please train the model first.")
    st.stop()

split = st.selectbox("Data Split", ["test", "val"], key="error_split")


@st.cache_data(show_spinner=False)
def get_error_analysis(model_name, data_split):
    """Load model, predict, and identify errors."""
    model = joblib.load(get_model_dir() / f"baseline_{model_name}.joblib")

    # The ensemble model is stored as a dict with separate 'tfidf' and
    # 'ensemble' components, unlike the other baselines which are Pipelines.
    if isinstance(model, dict):
        tfidf = model["tfidf"]
        clf = model["ensemble"]
        df = load_processed_data(data_split)
        X = df["sentence"].values
        X_transformed = tfidf.transform(X)
        y_true = df["label"].values
        y_pred = clf.predict(X_transformed)
    else:
        df = load_processed_data(data_split)
        X = df["sentence"].values
        y_true = df["label"].values
        y_pred = model.predict(X)

    # Get probabilities — handle dict (ensemble) and Pipeline models
    if isinstance(model, dict):
        clf = model["ensemble"]
        if hasattr(clf, "predict_proba"):
            probas = clf.predict_proba(X_transformed)
        elif hasattr(clf, "decision_function"):
            from scipy.special import softmax  # pyre-ignore

            probas = softmax(clf.decision_function(X_transformed), axis=1)
        else:
            probas = np.zeros((len(X), 3))
    elif hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        decisions = model.decision_function(X)
        from scipy.special import softmax  # pyre-ignore

        probas = softmax(decisions, axis=1)
    else:
        probas = np.zeros((len(X), 3))

    # Build results dataframe
    results = pd.DataFrame(
        {
            "sentence": X,
            "true_label": [LABEL_MAP_INV[int(lbl)] for lbl in y_true],
            "pred_label": [LABEL_MAP_INV[int(lbl)] for lbl in y_pred],
            "true_num": y_true,
            "pred_num": y_pred,
            "correct": y_true == y_pred,
            "confidence": probas.max(axis=1),
        }
    )

    # Add per-class probabilities
    for i, cls in enumerate(["negative", "neutral", "positive"]):
        results[f"prob_{cls}"] = probas[:, i]

    return results


with st.spinner("Analyzing predictions..."):
    results_df = get_error_analysis(classifier_name, split)

# ── Summary metrics ─────────────────────────────────────────────────────────────
total = len(results_df)
correct = results_df["correct"].sum()
errors = total - correct
accuracy = correct / total

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
    <div class='metric-card' style='border-top: 3px solid #3b82f6;'>
        <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>Total Samples</div>
        <div style='font-size: 2.2em; font-weight: 800; color: #f8fafc;'>{total}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
    <div class='metric-card' style='border-top: 3px solid #10b981;'>
        <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>Correct</div>
        <div style='font-size: 2.2em; font-weight: 800; color: #10b981;'>{correct}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
    <div class='metric-card' style='border-top: 3px solid #ef4444;'>
        <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>Errors</div>
        <div style='font-size: 2.2em; font-weight: 800; color: #ef4444;'>{errors}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
    <div class='metric-card' style='border-top: 3px solid #8b5cf6;'>
        <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>Accuracy</div>
        <div style='font-size: 2.2em; font-weight: 800; color: #f8fafc;'>{accuracy:.1%}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── Confusion Pattern Analysis ──────────────────────────────────────────────────
st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'>", unsafe_allow_html=True)
st.markdown("### 🎯 Confusion Pattern Analysis")

errors_df = results_df[~results_df["correct"]].copy()

if len(errors_df) == 0:
    st.success("🎉 Perfect accuracy! No misclassifications found on this split.")
else:
    # Confusion pattern breakdown
    confusion_patterns = errors_df.groupby(["true_label", "pred_label"]).size().reset_index(name="count")
    confusion_patterns = confusion_patterns.sort_values("count", ascending=False)

    # Visualize confusion patterns as a Sankey-style chart
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                label=[
                    "True: negative",
                    "True: neutral",
                    "True: positive",
                    "Pred: negative",
                    "Pred: neutral",
                    "Pred: positive",
                ],
                color=["#ef4444", "#64748b", "#10b981", "#ef4444", "#64748b", "#10b981"],
            ),
            link=dict(
                source=[],
                target=[],
                value=[],
                color=[],
            ),
        )
    )

    # Build links from confusion patterns
    label_to_source = {"negative": 0, "neutral": 1, "positive": 2}
    label_to_target = {"negative": 3, "neutral": 4, "positive": 5}
    link_colors = {
        "negative": "rgba(239,68,68,0.3)",
        "neutral": "rgba(100,116,139,0.3)",
        "positive": "rgba(16,185,129,0.3)",
    }

    sources, targets, values, colors = [], [], [], []
    for _, row in confusion_patterns.iterrows():
        sources.append(label_to_source[row["true_label"]])
        targets.append(label_to_target[row["pred_label"]])
        values.append(row["count"])
        colors.append(link_colors[row["true_label"]])

    fig.data[0].link.source = sources
    fig.data[0].link.target = targets
    fig.data[0].link.value = values
    fig.data[0].link.color = colors

    fig.update_layout(
        title={"text": f"Misclassification Flow ({len(errors_df)} errors)", "font": {"color": "#f8fafc", "size": 16}},
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1", "family": "Inter, sans-serif"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Top confusion pairs
    st.markdown("#### Most Common Confusion Pairs")
    for _, row in confusion_patterns.head(5).iterrows():
        true_color = {"negative": "#ef4444", "neutral": "#64748b", "positive": "#10b981"}[row["true_label"]]
        pred_color = {"negative": "#ef4444", "neutral": "#64748b", "positive": "#10b981"}[row["pred_label"]]
        pct = row["count"] / len(errors_df) * 100
        st.markdown(
            f"""
        <div style='display: flex; align-items: center; padding: 10px 15px; margin: 5px 0;
                    background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);'>
            <span style='color: {true_color}; font-weight: 600; min-width: 90px; text-transform: capitalize;'>{row["true_label"]}</span>
            <span style='color: #475569; margin: 0 12px;'>→</span>
            <span style='color: {pred_color}; font-weight: 600; min-width: 90px; text-transform: capitalize;'>{row["pred_label"]}</span>
            <span style='color: #94a3b8; margin-left: auto; font-family: monospace;'>{row["count"]} errors ({pct:.1f}%)</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Confidence distribution for errors vs correct ────────────────────────────
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("### 📊 Confidence Distribution: Correct vs Errors")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Histogram(
            x=results_df[results_df["correct"]]["confidence"],
            name="Correct",
            marker_color="#10b981",
            opacity=0.7,
            nbinsx=20,
        )
    )
    fig2.add_trace(
        go.Histogram(
            x=results_df[~results_df["correct"]]["confidence"],
            name="Errors",
            marker_color="#ef4444",
            opacity=0.7,
            nbinsx=20,
        )
    )
    fig2.update_layout(
        barmode="overlay",
        title={"text": "Model Confidence: Correct Predictions vs Errors", "font": {"color": "#f8fafc", "size": 14}},
        xaxis_title="Confidence",
        yaxis_title="Count",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1", "family": "Inter, sans-serif"},
        xaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)"},
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Key insight
    avg_conf_correct = results_df[results_df["correct"]]["confidence"].mean()
    avg_conf_error = results_df[~results_df["correct"]]["confidence"].mean()
    st.markdown(
        f"""
    <div class='insight-card' style='border-left: 4px solid #f59e0b; padding: 15px 20px;'>
        <span style='color: #f59e0b; font-weight: 700;'>💡 Insight:</span>
        Average confidence on <span style='color: #10b981; font-weight: 600;'>correct</span> predictions:
        <span style='font-family: monospace; color: #f8fafc;'>{avg_conf_correct:.3f}</span> |
        Average confidence on <span style='color: #ef4444; font-weight: 600;'>errors</span>:
        <span style='font-family: monospace; color: #f8fafc;'>{avg_conf_error:.3f}</span>
        {"— The model is well-calibrated (lower confidence on errors)." if avg_conf_error < avg_conf_correct else "— The model is overconfident on errors (calibration needed)."}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Misclassified Examples Table ─────────────────────────────────────────────
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("### 📋 Misclassified Examples")

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        true_filter = st.multiselect(
            "Filter by True Label",
            options=["negative", "neutral", "positive"],
            default=["negative", "neutral", "positive"],
            key="err_true_filter",
        )
    with filter_col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Lowest Confidence", "Highest Confidence"],
            key="err_sort",
        )

    filtered = errors_df[errors_df["true_label"].isin(true_filter)].copy()
    if sort_by == "Lowest Confidence":
        filtered = filtered.sort_values("confidence", ascending=True)
    else:
        filtered = filtered.sort_values("confidence", ascending=False)

    # Display errors
    display_count = min(20, len(filtered))
    for idx, (_, row) in enumerate(filtered.head(display_count).iterrows()):
        true_color = {"negative": "#ef4444", "neutral": "#64748b", "positive": "#10b981"}[row["true_label"]]
        pred_color = {"negative": "#ef4444", "neutral": "#64748b", "positive": "#10b981"}[row["pred_label"]]

        st.markdown(
            f"""
        <div style='padding: 15px 20px; margin: 8px 0; background: rgba(255,255,255,0.02);
                    border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);'>
            <div style='color: #e2e8f0; font-size: 0.95em; line-height: 1.5; margin-bottom: 10px;'>
                "{row["sentence"][:200]}{"..." if len(row["sentence"]) > 200 else ""}"
            </div>
            <div style='display: flex; gap: 20px; align-items: center;'>
                <span style='font-size: 0.8em; color: #94a3b8;'>True:
                    <span style='color: {true_color}; font-weight: 700; text-transform: capitalize;'>{row["true_label"]}</span>
                </span>
                <span style='font-size: 0.8em; color: #94a3b8;'>Predicted:
                    <span style='color: {pred_color}; font-weight: 700; text-transform: capitalize;'>{row["pred_label"]}</span>
                </span>
                <span style='font-size: 0.8em; color: #94a3b8; margin-left: auto;'>Confidence:
                    <span style='font-family: monospace; color: #f8fafc;'>{row["confidence"]:.3f}</span>
                </span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    if len(filtered) > display_count:
        st.info(f"Showing {display_count} of {len(filtered)} misclassified examples.")

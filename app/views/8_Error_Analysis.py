"""
Financial Sentiment Analyzer — Error Analysis Page.
Displays misclassified examples, confusion patterns, and model weakness insights.
"""

import html
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
from shared import (  # pyre-ignore
    SENTIMENT_COLORS,
    active_model,
    empty_state,
    exhibit,
    kpi_strip,
    page_header,
    section_header,
    sentiment_tag,
    source_note,
    status_banner,
)

# ── Page config ─────────────────────────────────────────────────────────────────
selected_model, predictor = active_model()

if predictor is None:
    st.stop()

page_header(
    "Error Analysis",
    "Inspect misclassifications, confusion flows, and confidence calibration to understand where the model is weakest.",
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

_split_col, _ = st.columns([1, 3])
with _split_col:
    split = st.selectbox(
        "Evaluation split",
        ["test", "val"],
        format_func=lambda v: {"test": "Test (n=226)", "val": "Validation (n=226)"}[v],
        key="error_split",
    )


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

kpi_strip(
    [
        {"label": f"Sentences ({split} split)", "value": f"{total:,}"},
        {"label": "Correct", "value": f"{correct:,}"},
        {"label": "Errors", "value": f"{errors:,}", "accent": SENTIMENT_COLORS["negative"]},
        {"label": "Accuracy", "value": f"{accuracy:.1%}"},
    ]
)

# ── Confusion Pattern Analysis ──────────────────────────────────────────────────
exhibit(
    "Where the errors flow",
    "Each band is a group of misclassified sentences, from true label (left) to predicted label (right).",
)

errors_df = results_df[~results_df["correct"]].copy()

if len(errors_df) == 0:
    status_banner("No misclassifications on this split.", kind="success")
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
                line=dict(color="#B9C6C3", width=0.5),
                label=[
                    "True ▼ negative",
                    "True ▶ neutral",
                    "True ▲ positive",
                    "Predicted ▼ negative",
                    "Predicted ▶ neutral",
                    "Predicted ▲ positive",
                ],
                color=["#A3322B", "#4F6470", "#1D7451", "#A3322B", "#4F6470", "#1D7451"],
            ),
            link=dict(
                source=[],
                target=[],
                value=[],
                color=[],
            ),
            textfont=dict(color="#12211F", size=12, shadow="none"),
        )
    )

    # Build links from confusion patterns
    label_to_source = {"negative": 0, "neutral": 1, "positive": 2}
    label_to_target = {"negative": 3, "neutral": 4, "positive": 5}
    link_colors = {
        "negative": "rgba(163,50,43,0.3)",
        "neutral": "rgba(79,100,112,0.3)",
        "positive": "rgba(29,116,81,0.3)",
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
        height=300,
        margin={"t": 10, "l": 8, "r": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#2E3F3C", "family": "Public Sans, system-ui, sans-serif"},
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    # Top confusion pairs
    section_header("Most common confusion pairs")
    rows_html = []
    for _, row in confusion_patterns.head(5).iterrows():
        pct = row["count"] / len(errors_df) * 100
        noun = "error" if row["count"] == 1 else "errors"
        rows_html.append(
            f"<tr><td class='t'>{sentiment_tag(row['true_label'])}</td><td>read as</td>"
            f"<td>{sentiment_tag(row['pred_label'])}</td>"
            f"<td class='n fs-num'>{row['count']} {noun}</td><td class='n fs-num'>{pct:.0f}%</td></tr>"
        )
    st.markdown(
        "<table class='fs-terms'><thead><tr><th class='t'>True label</th><th></th><th>Predicted</th>"
        "<th style='text-align:right;'>Count</th><th style='text-align:right;'>Share of errors</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>",
        unsafe_allow_html=True,
    )

    # ── Confidence distribution for errors vs correct ────────────────────────────
    exhibit(
        "Confidence on correct predictions vs errors",
        "A calibrated model is less sure when it is wrong, so low confidence can route a sentence to human review.",
    )

    fig2 = go.Figure()
    fig2.add_trace(
        go.Histogram(
            x=results_df[results_df["correct"]]["confidence"],
            name="Correct",
            marker_color="#45627A",
            opacity=0.7,
            nbinsx=20,
        )
    )
    fig2.add_trace(
        go.Histogram(
            x=results_df[~results_df["correct"]]["confidence"],
            name="Errors",
            marker_color="#A3322B",
            opacity=0.7,
            nbinsx=20,
        )
    )
    fig2.update_layout(
        barmode="overlay",
        xaxis_title="Confidence",
        yaxis_title="Count",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#2E3F3C", "family": "Public Sans, system-ui, sans-serif"},
        xaxis={"showgrid": True, "gridcolor": "#E6ECEA"},
        yaxis={"showgrid": True, "gridcolor": "#E6ECEA"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig2, use_container_width=True, theme=None, config={"displayModeBar": False})

    avg_conf_correct = results_df[results_df["correct"]]["confidence"].mean()
    avg_conf_error = results_df[~results_df["correct"]]["confidence"].mean()
    direction = "lower" if avg_conf_error < avg_conf_correct else "higher"
    st.markdown(
        f"<p style='max-width:75ch;'>Mean confidence is <b class='fs-num'>{avg_conf_correct:.2f}</b> on correct "
        f"predictions and <b class='fs-num'>{avg_conf_error:.2f}</b> on errors: {direction} when the model is wrong. "
        "Mean confidence alone does not establish calibration; a reliability diagram would.</p>",
        unsafe_allow_html=True,
    )
    source_note(f"{classifier_name} predictions on the {split} split.")

    # ── Misclassified Examples Table ─────────────────────────────────────────────
    section_header("Misclassified sentences", "The sentences behind the numbers above.")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        true_filter = st.multiselect(
            "True label",
            options=["negative", "neutral", "positive"],
            default=["negative", "neutral", "positive"],
            key="err_true_filter",
        )
    with filter_col2:
        sort_by = st.selectbox("Sort by", ["Lowest confidence", "Highest confidence"], key="err_sort")

    filtered = errors_df[errors_df["true_label"].isin(true_filter)].copy()
    filtered = filtered.sort_values("confidence", ascending=sort_by == "Lowest confidence")

    display_count = min(20, len(filtered))
    if display_count == 0:
        empty_state(
            "No errors match these labels", "Add a label above to see its misclassified sentences.", icon="search"
        )
    else:
        body = []
        for _, row in filtered.head(display_count).iterrows():
            text = html.escape(row["sentence"][:220]) + ("…" if len(row["sentence"]) > 220 else "")
            body.append(
                f"<tr><td class='t' style='font-weight:400;line-height:1.5;'>{text}</td>"
                f"<td class='c'>{sentiment_tag(row['true_label'])}</td><td class='c'>{sentiment_tag(row['pred_label'])}</td>"
                f"<td class='n fs-num'>{row['confidence']:.2f}</td></tr>"
            )
        st.markdown(
            "<table class='fs-terms'><thead><tr><th class='t'>Sentence</th><th class='c'>True</th>"
            "<th class='c'>Predicted</th><th style='text-align:right;'>Confidence</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>",
            unsafe_allow_html=True,
        )

    if len(filtered) > display_count:
        st.caption(f"Showing {display_count} of {len(filtered)} misclassified sentences.")

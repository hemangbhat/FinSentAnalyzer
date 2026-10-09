"""
Financial Sentiment Analyzer — Word Insights Page.

The vocabulary each TF-IDF model weights most heavily for each class.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import plotly.graph_objects as go  # pyre-ignore
import streamlit as st  # pyre-ignore
from plotly.subplots import make_subplots  # pyre-ignore

from explain import get_feature_importance_summary  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import (  # pyre-ignore
    SENTIMENT_GLYPHS,
    active_model,
    exhibit,
    get_sentiment_color,
    model_label,
    page_header,
    render_chart,
    source_note,
    status_banner,
    style_fig,
    term_table,
)

STOPWORDS = {"is", "be", "and", "the", "to", "of", "in", "a", "from", "by", "has", "will", "our", "its", "will be"}
CLASSES = ["negative", "neutral", "positive"]

selected_model, predictor = active_model()

page_header(
    "Word Insights",
    "The terms each model weights most heavily for every sentiment class. "
    "Weights are the classifier's learned coefficients over TF-IDF features.",
)

if not selected_model.startswith("baseline_"):
    status_banner(
        f"{model_label(selected_model)} has no per-term weights: transformers read context, not a vocabulary list. "
        "Choose a TF-IDF model in the sidebar, or open Explainability for token attribution.",
        kind="info",
    )
    st.stop()

with st.spinner("Loading learned vocabulary…"):
    importance = get_feature_importance_summary(selected_model)

if not isinstance(importance, dict) or "error" in importance:
    status_banner(f"Could not read this model's weights: {importance.get('error', 'unknown error')}", kind="error")
    st.stop()

top = {c: importance.get(f"top_{c}", [])[:12] for c in CLASSES}

exhibit(
    "Highest-weighted terms by class",
    f"Top 8 terms for {model_label(selected_model)}. Longer bars move a prediction further toward that class.",
)
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[f"{SENTIMENT_GLYPHS[c]} {c.capitalize()}" for c in CLASSES],
    horizontal_spacing=0.12,
)
for i, cls in enumerate(CLASSES, start=1):
    words = top[cls][:8][::-1]
    fig.add_trace(
        go.Bar(
            x=[w for _, w in words],
            y=[t for t, _ in words],
            orientation="h",
            marker_color=get_sentiment_color(cls),
            text=[f"{w:.2f}" for _, w in words],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ),
        row=1,
        col=i,
    )
style_fig(fig, height=360, legend=False)
fig.update_layout(margin={"t": 36, "l": 8, "r": 30, "b": 8})
fig.update_xaxes(showticklabels=False, showgrid=False)
fig.update_yaxes(showgrid=False, ticks="")
for ann in fig.layout.annotations:
    ann.update(x=ann.x, xanchor="center", font={"size": 13, "color": "#12211F"})
render_chart(fig)
source_note(f"models/{selected_model}.joblib (classifier coefficients over TF-IDF unigrams and bigrams).")

flagged = sorted({t for c in CLASSES for t, _ in top[c] if t.lower() in STOPWORDS})
if flagged:
    status_banner(
        "Function words carry real weight here: "
        + ", ".join(f"<b>{t}</b>" for t in flagged)
        + ". The model has partly learned the house style of Financial PhraseBank's annual-report sentences, "
        "one reason it drops to 0.46 macro-F1 on real news headlines.",
        kind="warning",
    )

cols = st.columns(3, gap="large")
for col, cls in zip(cols, CLASSES):
    with col:
        st.markdown(
            f"<div class='fs-section' style='margin-top:18px;'><h3 style='color:{get_sentiment_color(cls)};'>"
            f"{SENTIMENT_GLYPHS[cls]} {cls.capitalize()}</h3><p class='cap'>Top {len(top[cls])} terms</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(term_table([(t, w, cls) for t, w in top[cls]], show_class=False), unsafe_allow_html=True)

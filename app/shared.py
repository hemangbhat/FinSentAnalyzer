"""
Shared utilities for the Financial Sentiment Analyzer multipage app.
Contains helpers, cached loaders, and sidebar setup used across all pages.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import plotly.graph_objects as go

from predict import SentimentPredictor, get_available_models


# ── Custom CSS ──────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .sentiment-neutral { color: #007bff; font-weight: bold; }
    .big-font { font-size: 24px !important; }
    .metric-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
"""


def inject_css():
    """Inject the shared custom CSS into the page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Cached Model Loader ────────────────────────────────────────────────────────

@st.cache_resource
def load_predictor(model_type: str) -> SentimentPredictor:
    """Load and cache the sentiment predictor."""
    return SentimentPredictor(model_type)


# ── Sidebar Setup ──────────────────────────────────────────────────────────────

def setup_sidebar():
    """
    Render the sidebar with model selection and about section.

    Returns
    -------
    tuple[str, SentimentPredictor | None]
        (selected_model_key, predictor) — predictor is None when loading fails.
    """
    st.sidebar.title("📈 Financial Sentiment Analyzer")
    st.sidebar.markdown("---")

    available_models = get_available_models()
    if not available_models:
        st.error("No trained models found. Please train a model first.")
        st.code("python src/train.py --model baselines")
        return None, None

    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=0,
        help="Choose the model for sentiment analysis",
    )

    try:
        predictor = load_predictor(selected_model)
    except Exception as e:
        st.error(f"❌ Failed to load model **{selected_model}**: {e}")
        return selected_model, None

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app analyzes financial text sentiment using ML models trained on "
        "the Financial PhraseBank dataset. It classifies text as **Positive**, "
        "**Negative**, or **Neutral**."
    )

    return selected_model, predictor


# ── Chart Helpers ───────────────────────────────────────────────────────────────

def get_sentiment_color(label: str) -> str:
    """Return the hex colour associated with a sentiment label."""
    colors = {
        "positive": "#28a745",
        "negative": "#dc3545",
        "neutral": "#007bff",
    }
    return colors.get(label, "#6c757d")


def create_gauge_chart(probabilities: dict, prediction: str) -> go.Figure:
    """Create a gauge chart showing prediction confidence."""
    confidence = probabilities[prediction]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Confidence: {prediction.upper()}"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": get_sentiment_color(prediction)},
            "steps": [
                {"range": [0, 33], "color": "#ffebee"},
                {"range": [33, 66], "color": "#fff3e0"},
                {"range": [66, 100], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_probability_chart(probabilities: dict) -> go.Figure:
    """Create a horizontal bar chart for class probabilities."""
    labels = list(probabilities.keys())
    values = [probabilities[l] * 100 for l in labels]
    colors = [get_sentiment_color(l) for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
    ))

    fig.update_layout(
        title="Sentiment Probabilities",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

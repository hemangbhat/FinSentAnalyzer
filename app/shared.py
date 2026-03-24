"""
Shared utilities for the Financial Sentiment Analyzer multipage app.
Contains helpers, cached loaders, and sidebar setup used across all pages.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st  # pyre-ignore
import plotly.graph_objects as go  # pyre-ignore

from predict import SentimentPredictor, get_available_models  # pyre-ignore


# ── Custom CSS ──────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Backgrounds */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp div:not([class*="material-icons"]):not([class*="icon"]):not(.material-symbols-rounded) {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main App Background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1e212b 0%, #121418 100%);
        color: #e2e8f0;
    }
    
    /* Header (Top bar) transparency */
    [data-testid="stHeader"] {
        background: rgba(18, 20, 24, 0.7);
        backdrop-filter: blur(10px);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1a1d24;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] .css-17lntkn {
        color: #cbd5e1;
    }
    
    /* Sidebar Navigation Links Active State Overrides (if using relative paths) */
    .st-emotion-cache-16ctxm4 {
        border-radius: 8px;
    }

    /* Primary Buttons */
    .stButton > button {
        background: linear-gradient(180deg, #2c313c 0%, #232730 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #f8fafc;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(180deg, #353b47 0%, #2a2f39 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        color: #fff;
    }
    .stButton > button:active {
        transform: scale(0.98);
    }

    /* Input & Select fields */
    .stTextInput > div > div > input, 
    .stTextArea > div > textarea, 
    .stSelectbox > div > div {
        background-color: #232730;
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    .stTextInput > div > div > input:focus, 
    .stTextArea > div > textarea:focus,
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }

    /* Cards / Containers */
    .metric-card, .feature-card, .insight-card {
        background-color: rgba(26, 29, 36, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border-color: rgba(255,255,255,0.1);
    }
    
    /* Card Headers */
    .card-title {
        font-size: 1.1em;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-text {
        color: #94a3b8;
        font-size: 0.95em;
        line-height: 1.5;
    }

    /* Sentiment Colors & Chips */
    .sentiment-positive { color: #10b981; font-weight: 600; }
    .sentiment-negative { color: #ef4444; font-weight: 600; }
    .sentiment-neutral { color: #3b82f6; font-weight: 600; }
    .sentiment-warning { color: #f59e0b; font-weight: 600; }
    
    .chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 500;
        margin: 2px;
    }
    .chip.positive { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .chip.negative { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .chip.neutral { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    
    /* Hero Section */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 60px 20px 40px;
        background: radial-gradient(ellipse at top, rgba(59, 130, 246, 0.08) 0%, transparent 70%);
        border-radius: 24px;
        margin-bottom: 30px;
    }
    .hero-title {
        font-size: 3em;
        font-weight: 700;
        background: linear-gradient(90deg, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.2em;
        color: #94a3b8;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.05);
        margin: 2rem 0;
    }
    
    /* Streamlit Metric specific */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* DataFrame & Table overrides */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        overflow: hidden;
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
    st.sidebar.markdown("""
    <div style='padding: 10px 0 20px 0'>
        <h2 style='margin: 0; font-size: 1.5em; font-weight: 700; background: linear-gradient(90deg, #f8fafc, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            📈 FinSent
        </h2>
        <div style='font-size: 0.85em; color: #64748b; margin-top: 5px; font-weight: 500;'>
            ANALYTICS DASHBOARD
        </div>
    </div>
    """, unsafe_allow_html=True)

    available_models = get_available_models()
    if not available_models:
        st.sidebar.error("No trained models found.")
        return None, None

    st.sidebar.markdown("<div style='font-size: 0.85em; color: #94a3b8; margin-bottom: 5px; font-weight: 500'>AI ENGINE</div>", unsafe_allow_html=True)
    selected_model = st.sidebar.selectbox(
        "Model Selection",
        available_models,
        index=0,
        label_visibility="collapsed",
        help="Choose the model for sentiment analysis",
    )

    predictor = None
    try:
        predictor = load_predictor(selected_model)
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

    st.sidebar.markdown("<hr style='margin: 20px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    
    # Compact About/Status card
    st.sidebar.markdown("""
    <div style='background: rgba(26, 29, 36, 0.5); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 15px;'>
        <div style='font-size: 0.8em; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;'>
            System Status
        </div>
        <div style='display: flex; align-items: center; gap: 8px; font-size: 0.9em; color: #e2e8f0; margin-bottom: 6px;'>
            <div style='width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);'></div>
            Engine Active
        </div>
        <div style='display: flex; align-items: center; gap: 8px; font-size: 0.9em; color: #e2e8f0;'>
            <div style='width: 8px; height: 8px; border-radius: 50%; background: #3b82f6; box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);'></div>
            """ + str(selected_model) + """ Loaded
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style='margin-top: 15px; font-size: 0.8em; color: #64748b; line-height: 1.5;'>
        FinSent uses ML models trained on the Financial PhraseBank to analyze institutional sentiment.
    </div>
    """, unsafe_allow_html=True)

    return selected_model, predictor


# ── Chart Helpers ───────────────────────────────────────────────────────────────

def get_sentiment_color(label: str) -> str:
    """Return the hex colour associated with a sentiment label."""
    colors = {
        "positive": "#10b981", # Emerald 500
        "negative": "#ef4444", # Red 500
        "neutral": "#3b82f6",  # Blue 500
    }
    return colors.get(label, "#64748b")


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

    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#e2e8f0", family="Inter, sans-serif")
    )
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
        textposition="outside",
        textfont={"color": "#e2e8f0", "size": 12},
        orientation='v' # Vertical bars feels slightly more modern usually, but depends. Let's stick with vertical
    ))

    fig.update_layout(
        title={"text": "Sentiment Probabilities", "font": {"color": "#f8fafc", "size": 16}},
        yaxis_title="Probability (%)",
        yaxis_range=[0, 110], # padding for text
        height=300,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={"color": "#cbd5e1", "family": "Inter, sans-serif"},
        xaxis={"showgrid": False, "linecolor": 'rgba(255,255,255,0.1)'},
        yaxis={"showgrid": True, "gridcolor": 'rgba(255,255,255,0.05)', "linecolor": 'rgba(255,255,255,0.1)'},
    )
    return fig

    fig.update_layout(
        title="Sentiment Probabilities",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

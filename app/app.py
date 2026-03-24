"""
Financial Sentiment Analyzer — Home Page (Streamlit Multipage App Entrypoint).
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st  # pyre-ignore
import plotly.graph_objects as go  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent))
from shared import inject_css, setup_sidebar  # pyre-ignore

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
selected_model, predictor = setup_sidebar()

# ── Home page content ──────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-container'>
    <h1 class='hero-title'>Financial Sentiment Analyzer</h1>
    <p class='hero-subtitle'>
        Uncover institutional grade insights with ML-powered sentiment classification, 
        transparent explainability, and deep linguistic reasoning for financial markets.
    </p>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-card'>
        <div class='card-title'>
            <span style='font-size: 1.4em;'>📝</span> Single Analysis
        </div>
        <div class='card-text'>
            Instantly evaluate market sentiment from news excerpts or reports with high-confidence predictive modelling and AI-driven explainability.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
        <div class='card-title'>
            <span style='font-size: 1.4em;'>📁</span> Batch Processing
        </div>
        <div class='card-text'>
            Scale your textual analysis by uploading datasets. Automate bulk sentiment extraction, view aggregated trends, and generate comprehensive intelligence reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
        <div class='card-title'>
            <span style='font-size: 1.4em;'>🔍</span> Explainability
        </div>
        <div class='card-text'>
            Understand the 'why' behind the prediction. Visualise precise token attributions and key indicator highlights that drive the model's decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class='feature-card'>
        <div class='card-title'>
            <span style='font-size: 1.4em;'>💡</span> Word Insights
        </div>
        <div class='card-text'>
            Examine the model's vocabulary and learned correlations. Browse high-impact financial lexicon terms categorised by their sentiment influence.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class='feature-card'>
        <div class='card-title'>
            <span style='font-size: 1.4em;'>🧠</span> Deep Analysis
        </div>
        <div class='card-text'>
            Dive into advanced linguistic decomposition, Chain-of-Thought reasoning, and entity-specific metrics for a meticulous market perspective.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class='feature-card'>
        <div class='card-title'>
            <span style='font-size: 1.4em;'>📊</span> Model Info
        </div>
        <div class='card-text'>
            Review complete registries of all trained algorithms. Access granular performance KPIs, structural architectures, and training corpus details.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)

col_footer, _ = st.columns([1, 0.01])
with col_footer:
    st.markdown("""
    <div style='text-align: center; opacity: 0.5; padding: 20px; font-size: 0.9em;'>
        <p>Use the <b>sidebar</b> to select advanced models or navigate modules.</p>
    </div>
    """, unsafe_allow_html=True)

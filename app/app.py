"""
Financial Sentiment Analyzer — Home Page (Streamlit Multipage App Entrypoint).
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from shared import inject_css, setup_sidebar

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
<div style='text-align: center; padding: 40px 20px;'>
    <h1 style='font-size: 2.5em; margin-bottom: 10px;'>📈 Financial Sentiment Analyzer</h1>
    <p style='font-size: 1.2em; opacity: 0.8; max-width: 700px; margin: 0 auto;'>
        Analyze financial texts with ML-powered sentiment classification,
        explainability insights, and advanced NLP reasoning.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background-color: #262730; padding: 25px; border-radius: 12px; 
                border-top: 4px solid #28a745; min-height: 180px;'>
        <h3>📝 Single Analysis</h3>
        <p>Paste any financial text and instantly see the predicted sentiment 
        with confidence scores and AI explanations.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background-color: #262730; padding: 25px; border-radius: 12px;
                border-top: 4px solid #007bff; min-height: 180px;'>
        <h3>📁 Batch Processing</h3>
        <p>Upload CSV or TXT files to analyze hundreds of texts at once.
        Get sentiment distributions, trends, and downloadable results.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background-color: #262730; padding: 25px; border-radius: 12px;
                border-top: 4px solid #dc3545; min-height: 180px;'>
        <h3>🔍 Explainability</h3>
        <p>See which words influenced the model's decision with highlighted 
        text and word-importance charts.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div style='background-color: #262730; padding: 25px; border-radius: 12px;
                border-top: 4px solid #fd7e14; min-height: 180px;'>
        <h3>💡 Word Insights</h3>
        <p>Discover the most important positive, negative, and neutral 
        indicator words learned by the trained model.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div style='background-color: #262730; padding: 25px; border-radius: 12px;
                border-top: 4px solid #6f42c1; min-height: 180px;'>
        <h3>🧠 Deep Analysis</h3>
        <p>Chain-of-thought reasoning, financial lexicon scoring, entity 
        extraction, and linguistic analysis in one place.</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div style='background-color: #262730; padding: 25px; border-radius: 12px;
                border-top: 4px solid #17a2b8; min-height: 180px;'>
        <h3>📊 Model Info</h3>
        <p>View model metrics, comparison tables, and dataset details 
        for all available trained models.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div style='text-align: center; opacity: 0.6; padding: 20px;'>
    <p>👈 Use the <b>sidebar</b> to navigate between pages and select a model.</p>
</div>
""", unsafe_allow_html=True)

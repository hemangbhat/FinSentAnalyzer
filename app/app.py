"""
Financial Sentiment Analyzer — Home (Streamlit multipage app entrypoint).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # pyre-ignore
from shared import inject_css, kpi_strip, section_header, setup_sidebar  # pyre-ignore

st.set_page_config(
    page_title="FinSight · Sentiment Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
selected_model, predictor = setup_sidebar()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class='fs-hero'>
    <div class='fs-eyebrow'>Financial NLP · Explainable ML</div>
    <h1>Financial Sentiment Intelligence</h1>
    <p>
        Classify market sentiment from financial text, see exactly which words drove each
        decision, and validate the model against real-world news — all in one transparent dashboard.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── KPI strip (honest, verifiable facts) ─────────────────────────────────────
kpi_strip(
    [
        {"label": "Trained Models", "value": "8", "sub": "6 baselines + ensemble + FinBERT", "accent": "#3b82f6"},
        {"label": "Sentiment Classes", "value": "3", "sub": "positive · neutral · negative", "accent": "#10b981"},
        {"label": "Validation Headlines", "value": "2.7K+", "sub": "WSJ, Bloomberg, Reuters…", "accent": "#8b5cf6"},
        {"label": "Explainability", "value": "SHAP", "sub": "+ per-token attribution", "accent": "#f59e0b"},
    ]
)

# ── Capabilities ─────────────────────────────────────────────────────────────
section_header("Capabilities", "Each page is focused on a single task")

CARDS = [
    (
        "📝",
        "Single Analysis",
        "Classify any financial excerpt and get confidence, probabilities, and a plain-language rationale in one flow.",
    ),
    ("📁", "Batch Processing", "Upload a CSV or TXT file, validate it, analyze at scale, and export a results report."),
    ("🔍", "Explainability", "Per-token attribution plus SHAP global feature importance for baseline models."),
    ("💡", "Word Insights", "Browse the highest-impact vocabulary the models learned, grouped by sentiment."),
    ("🧠", "Deep Analysis", "Rule-based reasoning, NER, and Loughran-McDonald lexicon decomposition."),
    ("📊", "Model Registry", "Compare every model's metrics, architecture, and training data."),
    ("📈", "Sentiment Trends", "News sentiment vs. stock price with Pearson correlation for AAPL, TSLA, AMZN."),
    ("🔬", "Error Analysis", "Confusion flows, confidence calibration, and inspectable misclassifications."),
    ("📉", "Stock Extension", "End-to-end pipeline: live news → FinBERT → LSTM next-day direction forecast."),
]

for row_start in range(0, len(CARDS), 3):
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, CARDS[row_start : row_start + 3]):
        with col:
            st.markdown(
                f"""
                <div class='feature-card' style='height:100%;'>
                    <div class='card-title'><span style='font-size:1.3em;'>{icon}</span> {title}</div>
                    <div class='card-text'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── How it works ─────────────────────────────────────────────────────────────
section_header("How it works", "From raw text to an explainable verdict")

steps = st.columns(4)
STEPS = [
    ("01", "Select a model", "Pick a baseline or the FinBERT transformer from the sidebar."),
    ("02", "Provide text", "Type an excerpt or upload a file for batch scoring."),
    ("03", "Get a verdict", "Sentiment, confidence, and a probability breakdown."),
    ("04", "Understand why", "Token attribution and SHAP reveal the drivers."),
]
for col, (num, title, desc) in zip(steps, STEPS):
    with col:
        st.markdown(
            f"""
            <div class='app-card' style='height:100%;'>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;color:#60a5fa;font-weight:600;'>{num}</div>
                <div style='font-weight:600;color:#f8fafc;margin:6px 0 4px 0;'>{title}</div>
                <div class='card-text'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<div style='text-align:center;opacity:0.55;padding:30px 0 6px 0;font-size:0.88rem;'>"
    "Use the <b>sidebar</b> to choose a model and navigate between modules.</div>",
    unsafe_allow_html=True,
)

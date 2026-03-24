"""
Financial Sentiment Analyzer — Single Text Analysis Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st  # pyre-ignore
import plotly.graph_objects as go  # pyre-ignore
from explain import explain_prediction_baseline  # pyre-ignore
from llm_explain import get_llm_explanation  # pyre-ignore
import os
import sys

# Shared helpers
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar, get_sentiment_color, create_probability_chart  # pyre-ignore

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Single Analysis", page_icon="📝", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 25px;'>
    <h1 style='font-size: 2.2em; font-weight: 700; margin-bottom: 5px;'>📝 Single Text Analysis</h1>
    <p style='color: #94a3b8; font-size: 1.1em;'>Evaluate precise market sentiment from financial excerpts or reports.</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for example text
if "example_text" not in st.session_state:
    st.session_state.example_text = ""

# Example texts
with st.expander("📌 Example Texts"):
    examples = [
        "The company reported a 25% increase in quarterly revenue.",
        "Stock prices fell sharply after disappointing earnings report.",
        "The firm announced it will maintain its current dividend policy.",
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn.",
        "The company's market share remained unchanged at 15%.",
    ]
    for i, example in enumerate(examples):
        if st.button(f"Use Example {i+1}", key=f"example_{i}"):
            st.session_state.example_text = example

# Text input
text_input = st.text_area(
    "Enter financial text:",
    value=st.session_state.example_text,
    height=150,
    placeholder="e.g., The company reported strong Q3 earnings, beating analyst expectations...",
)

if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if text_input.strip():
        try:
            with st.spinner("Analyzing..."):
                if predictor is not None:
                    result = predictor.predict(text_input)  # pyre-ignore
                else:
                    st.error("Predictor not loaded.")
                    st.stop()
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        # Results
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("### Analysis Results")
        
        col1, col2 = st.columns([1, 1.2])

        with col1:
            sentiment = result["label"]
            color = get_sentiment_color(sentiment)
            conf = result.get('confidence', 0.0)

            st.markdown(f"""
            <div class='insight-card' style='text-align: center; display: flex; flex-direction: column; justify-content: center; height: 100%; border-top: 4px solid {color};'>
                <div style='color: #94a3b8; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>Predicted Sentiment</div>
                <div style='font-size: 3.5em; font-weight: 800; color: {color}; line-height: 1.2; text-transform: capitalize;'>
                    {sentiment}
                </div>
                <div style='margin-top: 15px; font-size: 1.1em; color: #cbd5e1;'>
                    Confidence Score: <span style='font-weight: 600; color: #f8fafc;'>{conf:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if "probabilities" in result:
                st.markdown("<div class='insight-card' style='padding: 10px;'>", unsafe_allow_html=True)
                fig = create_probability_chart(result["probabilities"])
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)

        # LLM-powered Natural Language Explanation
        if selected_model.startswith("baseline_"):
            st.markdown("### 💬 AI Explanation")
            try:
                explanation_data = explain_prediction_baseline(text_input, selected_model)
                llm_explanation = get_llm_explanation(
                    text=text_input,
                    prediction=result["label"],
                    probabilities=result.get("probabilities", {}),
                    word_importance=explanation_data.get("word_importance", []),
                )
                st.markdown(
                    f"""
                    <div class='insight-card' style='border-left: 4px solid {color}; background: rgba(26, 29, 36, 0.95);'>
                        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 15px;'>
                            <span style='font-size: 1.5em;'>🤖</span>
                            <span style='font-weight: 600; font-size: 1.1em; color: #f8fafc;'>AI Reasoning Overview</span>
                        </div>
                        <div style='line-height: 1.6; color: #cbd5e1; font-size: 1.05em;'>
                            {llm_explanation}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception:
                st.info("Enable the Explainability page for detailed AI-powered explanations.")
    else:
        st.warning("Please enter some text to analyze.")

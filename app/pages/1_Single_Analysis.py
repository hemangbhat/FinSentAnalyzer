"""
Financial Sentiment Analyzer — Single Text Analysis Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st

from explain import explain_prediction_baseline
from llm_explain import get_llm_explanation

# Shared helpers
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar, get_sentiment_color, create_probability_chart

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Single Analysis", page_icon="📝", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.header("📝 Single Text Analysis")
st.markdown("Enter financial text to analyze its sentiment.")

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
                result = predictor.predict(text_input)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        # Results
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Result")
            sentiment = result["label"]
            color = get_sentiment_color(sentiment)

            st.markdown(
                f"<h2 style='color: {color};'>{sentiment.upper()}</h2>",
                unsafe_allow_html=True,
            )

            if "confidence" in result:
                st.metric("Confidence", f"{result['confidence']:.1%}")

        with col2:
            if "probabilities" in result:
                fig = create_probability_chart(result["probabilities"])
                st.plotly_chart(fig, use_container_width=True)

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
                    f"<div style='background-color: #262730; padding: 15px; border-radius: 10px; "
                    f"border-left: 4px solid {color};'>{llm_explanation}</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                st.info("Enable the Explainability page for detailed AI-powered explanations.")
    else:
        st.warning("Please enter some text to analyze.")

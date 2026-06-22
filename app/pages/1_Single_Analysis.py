"""
Financial Sentiment Analyzer — Single Text Analysis (flagship page).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st  # pyre-ignore

from explain import explain_prediction_baseline  # pyre-ignore
from llm_explain import get_llm_explanation  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import (  # pyre-ignore
    create_probability_chart,
    inject_css,
    page_header,
    render_chart,
    section_header,
    setup_sidebar,
    status_banner,
    verdict_card,
)

st.set_page_config(page_title="Single Analysis", page_icon="📝", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

page_header(
    "Single Text Analysis",
    "Classify market sentiment from a financial excerpt and see the confidence, "
    "probability breakdown, and the words behind the decision.",
    eyebrow="Real-time inference",
)

# ── Examples ─────────────────────────────────────────────────────────────────
if "example_text" not in st.session_state:
    st.session_state.example_text = ""

EXAMPLES = [
    "The company reported a 25% increase in quarterly revenue.",
    "Stock prices fell sharply after a disappointing earnings report.",
    "The firm announced it will maintain its current dividend policy.",
    "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn.",
    "The company's market share remained unchanged at 15%.",
]

with st.expander("Try an example", expanded=False):
    ex_cols = st.columns(len(EXAMPLES))
    for i, (col, example) in enumerate(zip(ex_cols, EXAMPLES)):
        with col:
            if st.button(f"Example {i + 1}", key=f"example_{i}", use_container_width=True):
                st.session_state.example_text = example

# ── Input ────────────────────────────────────────────────────────────────────
text_input = st.text_area(
    "Financial text",
    value=st.session_state.example_text,
    height=150,
    placeholder="e.g., The company reported strong Q3 earnings, beating analyst expectations…",
)

analyze = st.button("Analyze Sentiment", type="primary", use_container_width=True)

if analyze and not text_input.strip():
    status_banner("Enter some financial text to analyze.", kind="warning")

if analyze and text_input.strip():
    try:
        with st.spinner("Analyzing…"):
            result = predictor.predict(text_input)  # pyre-ignore
    except Exception as exc:  # noqa: BLE001
        status_banner(f"Prediction failed: {exc}", kind="error")
        st.stop()

    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Result")

    sentiment = result["label"]
    conf = result.get("confidence", 0.0)

    col1, col2 = st.columns([1, 1.25])
    with col1:
        verdict_card(sentiment, conf)
    with col2:
        if "probabilities" in result:
            render_chart(create_probability_chart(result["probabilities"]))
        else:
            status_banner("This model does not expose class probabilities.", kind="info")

    # ── Plain-language rationale (baseline models only) ──────────────────────
    if selected_model.startswith("baseline_"):
        section_header("Why this verdict", "Template rationale from the model's word weights")
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
                <div class='insight-card' style='border-left:3px solid #3b82f6;'>
                    <div style='line-height:1.65;color:#cbd5e1;font-size:1.02em;'>{llm_explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:  # noqa: BLE001
            status_banner("Open the Explainability page for detailed token-level attribution.", kind="info")
    else:
        status_banner(
            "Token-level rationale is available for baseline models. "
            "Switch to a baseline model in the sidebar to see it.",
            kind="info",
        )

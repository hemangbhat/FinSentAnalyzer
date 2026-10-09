"""
Financial Sentiment Analyzer — Single Text Analysis (flagship page).
"""

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st  # pyre-ignore

from explain import explain_prediction_baseline  # pyre-ignore
from llm_explain import get_llm_explanation  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import (  # pyre-ignore
    active_model,
    create_probability_chart,
    empty_state,
    page_header,
    render_chart,
    section_header,
    status_banner,
    verdict_card,
)

selected_model, predictor = active_model()

if predictor is None:
    st.stop()

page_header(
    "Single Text Analysis",
    "Classify market sentiment from a financial excerpt and see the confidence, "
    "probability breakdown, and the words behind the decision.",
    eyebrow="Real-time inference",
)


def _md_bold(text: str) -> str:
    """Render **bold** markdown inside an HTML block (escaping everything else)."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))


# ── Examples ─────────────────────────────────────────────────────────────────
EXAMPLES = {
    "Revenue up 25%": "The company reported a 25% increase in quarterly revenue.",
    "Shares fall on earnings": "Stock prices fell sharply after a disappointing earnings report.",
    "Dividend unchanged": "The firm announced it will maintain its current dividend policy.",
    "Profit EUR 8.7m → 13.1m": "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn.",
    "Plant closure, 300 jobs cut": "The company will close its Tampere plant, cutting 300 jobs.",
}


def _use_example() -> None:
    choice = st.session_state.get("sa_example")
    if choice:
        st.session_state.sa_text = EXAMPLES[choice]
        st.session_state.sa_run = True


st.pills(
    "Try an example",
    list(EXAMPLES),
    key="sa_example",
    on_change=_use_example,
)

# ── Input ────────────────────────────────────────────────────────────────────
text_input = st.text_area(
    "Financial text",
    key="sa_text",
    height=130,
    placeholder="e.g., The company reported strong Q3 earnings, beating analyst expectations…",
)

analyze = st.button("Analyze sentiment", type="primary", use_container_width=True)
analyze = analyze or st.session_state.pop("sa_run", False)

if not analyze:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    empty_state(
        "Paste a sentence from an earnings release, filing, or headline",
        "You'll get the predicted sentiment, the model's confidence, the full probability "
        "breakdown, and the words that pushed it there. Change the model in the sidebar to compare.",
        icon="✎",
    )

if analyze and not text_input.strip():
    status_banner("Enter some financial text to analyze.", kind="warning")

if analyze and text_input.strip():
    try:
        with st.spinner("Analyzing…"):
            result = predictor.predict(text_input)  # pyre-ignore
    except Exception as exc:  # noqa: BLE001
        status_banner(f"Prediction failed: {exc}", kind="error")
        st.stop()

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
                    <div style='line-height:1.65;color:#cbd5e1;font-size:1.02em;'>{_md_bold(llm_explanation)}</div>
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

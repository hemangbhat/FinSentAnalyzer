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
    empty_state,
    model_label,
    page_header,
    section_header,
    status_banner,
    term_table,
    verdict_card,
)

selected_model, predictor = active_model()

if predictor is None:
    st.stop()

page_header(
    "Single Text Analysis",
    "Classify market sentiment from a financial excerpt and see the confidence, "
    "probability breakdown, and the words behind the decision.",
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
        icon="edit",
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

    section_header("Result", f"{model_label(selected_model)} · {len(text_input.split())} words")

    sentiment = result["label"]
    conf = result.get("confidence", 0.0)
    probs = result.get("probabilities")

    explanation = None
    if selected_model.startswith("baseline_"):
        try:
            explanation = explain_prediction_baseline(text_input, selected_model)
        except Exception:  # noqa: BLE001
            explanation = None

    col1, col2 = st.columns([1, 1.15], gap="large")
    with col1:
        verdict_card(sentiment, conf, probs)
        if not probs:
            status_banner("This model does not expose class probabilities.", kind="info")
    with col2:
        st.markdown("<div class='fs-verdict'><div class='label'>Key drivers</div></div>", unsafe_allow_html=True)
        if explanation and explanation.get("word_importance"):
            st.markdown(term_table(explanation["word_importance"][:6]), unsafe_allow_html=True)
            st.caption("TF-IDF weight × model coefficient for each term in the text, and the class it pushes toward.")
        else:
            st.caption(
                "Term-level drivers come from the TF-IDF models' weights. Switch to a baseline model in the "
                "sidebar to see them; for FinBERT, open Explainability."
            )

    if explanation:
        section_header("Why this verdict", "Template rationale built from the model's word weights (not an LLM).")
        rationale = get_llm_explanation(
            text=text_input,
            prediction=sentiment,
            probabilities=probs or {},
            word_importance=explanation.get("word_importance", []),
        )
        st.markdown(f"<p style='max-width:75ch;'>{_md_bold(rationale)}</p>", unsafe_allow_html=True)

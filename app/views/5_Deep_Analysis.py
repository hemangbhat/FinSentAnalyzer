"""
Financial Sentiment Analyzer — Deep Analysis (Advanced NLP) Page.
"""

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st  # pyre-ignore

# Import advanced NLP modules (optional dependency)
try:
    from llm_enhanced import ChainOfThoughtReasoner  # pyre-ignore
    from nlp_advanced import FinancialTextAnalyzer  # pyre-ignore

    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import (  # pyre-ignore
    active_model,
    empty_state,
    page_header,
    section_header,
    sentiment_tag,
    status_banner,
    verdict_card,
)

# ── Page config ─────────────────────────────────────────────────────────────────
selected_model, predictor = active_model()

if predictor is None:
    st.stop()

page_header(
    "Deep Analysis",
    "Entities, figures and Loughran-McDonald lexicon hits extracted from one sentence, with a step-by-step "
    "rule-based reasoning trace (deterministic rules, not a language model).",
)

if not ADVANCED_NLP_AVAILABLE:
    status_banner("The NLP modules (src/nlp_advanced.py, src/llm_enhanced.py) are not available.", kind="warning")
    st.stop()

DEEP_EXAMPLES = {
    "Earnings beat, EUR figures": "Nokia reported Q3 operating profit of EUR 1.2 billion, up 25% year on year, beating analyst expectations.",
    "Guidance cut": "The company lowered its full-year outlook after weak demand in Europe, and shares fell 8%.",
    "Routine update": "The board will hold its annual general meeting in Helsinki on 15 March.",
}


def _use_deep_example() -> None:
    choice = st.session_state.get("deep_example")
    if choice:
        st.session_state.deep_analysis_input = DEEP_EXAMPLES[choice]
        st.session_state.deep_run = True


st.pills("Try an example", list(DEEP_EXAMPLES), key="deep_example", on_change=_use_deep_example)
deep_text = st.text_area(
    "Financial text",
    height=120,
    placeholder="e.g., The company reported strong earnings growth of 25%, beating analyst expectations significantly...",
    key="deep_analysis_input",
)

run_deep = st.button("Run deep analysis", type="primary") or st.session_state.pop("deep_run", False)
if not run_deep:
    empty_state(
        "Pick an example or paste a sentence",
        "You'll get the companies, amounts, percentages and dates it mentions, the finance-specific sentiment words "
        "it contains, and each reasoning step with its confidence.",
        icon="search",
    )
if run_deep:
    if deep_text.strip():
        nlp_result = {}
        cot_result = None
        try:
            with st.spinner("Running comprehensive analysis..."):
                nlp_analyzer = FinancialTextAnalyzer()
                cot_reasoner = ChainOfThoughtReasoner()
                nlp_result = nlp_analyzer.analyze(deep_text)
                cot_result = cot_reasoner.analyze(deep_text)
        except Exception as e:
            st.error(f"Deep analysis failed: {e}")
            st.stop()

        if cot_result is None or nlp_result is None:
            st.stop()

        # Add explicit asserts for strict static analysis
        assert cot_result is not None
        assert nlp_result is not None

        f = nlp_result["features"]
        ents = nlp_result["entities"]

        def _ruled(rows) -> str:
            body = "".join(f"<tr><td>{k}</td><td class='n fs-num'>{v}</td></tr>" for k, v in rows)
            return f"<table class='fs-terms'><tbody>{body}</tbody></table>"

        def _tags(items, kind: str = "") -> str:
            if not items:
                return "<span style='color:#5E6E6A;'>None found</span>"
            if kind in ("positive", "negative"):
                return " ".join(sentiment_tag(kind, "").replace(kind.capitalize(), html.escape(str(w))) for w in items)
            return " ".join(f"<span class='chip'>{html.escape(str(w))}</span>" for w in items)

        # ── Verdict ───────────────────────────────────────────────────────
        section_header("Summary", "Rule-based synthesis of lexicon, entities and context.")
        v1, v2 = st.columns([1, 1.2], gap="large")
        with v1:
            verdict_card(cot_result.final_sentiment, cot_result.final_confidence)
        with v2:
            st.markdown(cot_result.explanation)

        # ── Measurements ──────────────────────────────────────────────────
        section_header("Linguistic measurements")
        m1, m2, m3 = st.columns(3, gap="large")
        with m1:
            st.markdown(
                "<div class='fs-verdict'><div class='label'>Polarity</div></div>"
                + _ruled(
                    [
                        ("Lexicon score", f"{f['sentiment_score']:.3f}"),
                        ("Uncertainty", f"{f['uncertainty_score']:.3f}"),
                        ("Subjectivity", f"{f['subjectivity_score']:.3f}"),
                    ]
                ),
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                "<div class='fs-verdict'><div class='label'>Lexicon hits</div></div>"
                + _ruled(
                    [
                        ("▲ Positive words", f["positive_word_count"]),
                        ("▼ Negative words", f["negative_word_count"]),
                        ("Uncertainty words", f["uncertainty_word_count"]),
                    ]
                ),
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                "<div class='fs-verdict'><div class='label'>Structure</div></div>"
                + _ruled(
                    [
                        ("Words", f["word_count"]),
                        ("Sentences", f["sentence_count"]),
                        ("Mean word length", f"{f['avg_word_length']:.1f}"),
                    ]
                ),
                unsafe_allow_html=True,
            )

        # ── What it found ─────────────────────────────────────────────────
        section_header("What it found", "Loughran-McDonald lexicon words and extracted entities.")
        st.markdown(
            "<table class='fs-terms'><tbody>"
            f"<tr><td class='t'>Positive words</td><td>{_tags(f['positive_words'][:8], 'positive')}</td></tr>"
            f"<tr><td class='t'>Negative words</td><td>{_tags(f['negative_words'][:8], 'negative')}</td></tr>"
            f"<tr><td class='t'>Uncertainty words</td><td>{_tags(f['uncertainty_words'][:8])}</td></tr>"
            f"<tr><td class='t'>Percentages</td><td>{_tags(ents['percentages'])}</td></tr>"
            f"<tr><td class='t'>Amounts</td><td>{_tags(ents['currencies'][:5])}</td></tr>"
            f"<tr><td class='t'>Companies and tickers</td><td>{_tags(ents['companies'][:5])}</td></tr>"
            "</tbody></table>",
            unsafe_allow_html=True,
        )

        # ── Reasoning trace ───────────────────────────────────────────────
        section_header("Reasoning trace", "Each deterministic rule step with its confidence.")
        body = []
        for i, step in enumerate(cot_result.steps, 1):
            body.append(
                f"<tr><td class='t' style='vertical-align:top;'>{i}. {step.step.value.replace('_', ' ').capitalize()}</td>"
                f"<td style='line-height:1.5;'>{html.escape(str(step.observation))}<br>"
                f"<span style='color:#4B5D5A;'>{html.escape(str(step.conclusion))}</span></td>"
                f"<td class='n fs-num' style='vertical-align:top;'>{step.confidence:.2f}</td></tr>"
            )
        st.markdown(
            "<table class='fs-terms'><thead><tr><th class='t'>Step</th><th>Observation and conclusion</th>"
            "<th style='text-align:right;'>Confidence</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>",
            unsafe_allow_html=True,
        )
        if cot_result.key_factors:
            st.markdown("**Driving factors**")
            st.markdown("\n".join(f"- {k}" for k in cot_result.key_factors))

        # About Loughran-McDonald
        with st.expander("About the Loughran-McDonald lexicon"):
            st.markdown("""
            The **Loughran-McDonald Dictionary** (2011) is the gold standard for financial sentiment analysis.
            Unlike general sentiment lexicons, it was specifically designed for financial text.

            **Key Categories:**
            - **Positive Words:** Terms indicating favorable outcomes (e.g., growth, profit, gain)
            - **Negative Words:** Terms indicating unfavorable outcomes (e.g., loss, decline, risk)
            - **Uncertainty Words:** Terms indicating speculation or uncertainty (e.g., may, could, expect)
            - **Litigious Words:** Legal/regulatory terms (e.g., lawsuit, litigation, regulatory)

            **Why it matters:**
            - General lexicons often misclassify financial terms (e.g., "liability" is negative in general, but neutral in finance)
            - Built from analysis of 10-K filings from 1994-2008
            - Contains 2,707 negative words and 354 positive words specific to finance

            **Reference:** Loughran, T. and McDonald, B. (2011). "When is a Liability not a Liability?"
            """)
    else:
        status_banner("Enter some financial text to analyze.", kind="warning")

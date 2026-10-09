"""
Financial Sentiment Analyzer — Deep Analysis (Advanced NLP) Page.
"""

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
from shared import active_model, get_sentiment_color, page_header, section_header  # pyre-ignore

# ── Page config ─────────────────────────────────────────────────────────────────
selected_model, predictor = active_model()

if predictor is None:
    st.stop()

page_header(
    "Deep Analysis",
    "Rule-based chain-of-thought reasoning, named-entity extraction, and Loughran-McDonald lexicon decomposition.",
)

if not ADVANCED_NLP_AVAILABLE:
    st.warning(
        "Advanced NLP modules not available. "
        "Please ensure `nlp_advanced.py` and `llm_enhanced.py` are in the `src/` folder."
    )
    st.stop()

deep_text = st.text_area(
    "Enter financial text for deep analysis:",
    height=120,
    placeholder="e.g., The company reported strong earnings growth of 25%, beating analyst expectations significantly...",
    key="deep_analysis_input",
)

if st.button("Run Deep Analysis", type="primary", use_container_width=True):
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

        # ── Summary Card ────────────────────────────────────────────────────
        section_header("Executive Summary")

        sentiment = cot_result.final_sentiment
        confidence = cot_result.final_confidence
        color = get_sentiment_color(sentiment)

        st.markdown(
            f"""
        <div class='insight-card' style='background: linear-gradient(135deg, #F7F9F8 0%, #F7F9F8 100%);
                    padding: 30px; border-top: 1px solid #D5DEDC; margin-bottom: 25px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <div style='color: #4B5D5A; font-size: 0.95em; text-transform: uppercase; letter-spacing: 1px;'>Final Synthesized Sentiment</div>
                    <div style='margin: 5px 0 0 0; color: {color}; font-size: 2.5em; font-weight: 800; text-transform: capitalize;'>{sentiment}</div>
                </div>
                <div style='text-align: right;'>
                    <div style='color: #4B5D5A; font-size: 0.95em; text-transform: uppercase; letter-spacing: 1px;'>Reliability Metrics</div>
                    <div style='font-size: 1.2em; color: #2E3F3C; margin-top: 5px;'>
                        Confidence: <span style='font-weight: 700; color: #12211F;'>{confidence:.1%}</span>
                    </div>
                    <div style='font-size: 1.2em; color: #2E3F3C;'>
                        Lexicon Baseline: <span style='font-weight: 700; color: #12211F;'>{nlp_result["features"]["sentiment_score"]:.2f}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ── Key metrics ─────────────────────────────────────────────────────
        section_header("Linguistic Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
            <div class='metric-card'>
                <div style='color: #4B5D5A; font-size: 0.85em; text-transform: uppercase;'>Polarity Profile</div>
                <div style='display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #D5DEDC;'><span>Lexicon Score</span> <span style='color: #12211F; font-weight: 600;'>{nlp_result["features"]["sentiment_score"]:.3f}</span></div>
                <div style='display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #D5DEDC;'><span>Uncertainty</span> <span style='color: #12211F; font-weight: 600;'>{nlp_result["features"]["uncertainty_score"]:.3f}</span></div>
                <div style='display: flex; justify-content: space-between; padding: 5px 0;'><span>Subjectivity</span> <span style='color: #12211F; font-weight: 600;'>{nlp_result["features"]["subjectivity_score"]:.3f}</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class='metric-card'>
                <div style='color: #4B5D5A; font-size: 0.85em; text-transform: uppercase;'>Vocabulary Spread</div>
                <div style='display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #D5DEDC;'><span>Positive Tokens</span> <span style='color: #1D7451; font-weight: 600;'>{nlp_result["features"]["positive_word_count"]}</span></div>
                <div style='display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #D5DEDC;'><span>Negative Tokens</span> <span style='color: #A3322B; font-weight: 600;'>{nlp_result["features"]["negative_word_count"]}</span></div>
                <div style='display: flex; justify-content: space-between; padding: 5px 0;'><span>Uncertainty Tokens</span> <span style='color: #9A5B16; font-weight: 600;'>{nlp_result["features"]["uncertainty_word_count"]}</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class='metric-card'>
                <div style='color: #4B5D5A; font-size: 0.85em; text-transform: uppercase;'>Structural Blueprint</div>
                <div style='display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #D5DEDC;'><span>Total Words</span> <span style='color: #12211F; font-weight: 600;'>{nlp_result["features"]["word_count"]}</span></div>
                <div style='display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #D5DEDC;'><span>Sentences</span> <span style='color: #12211F; font-weight: 600;'>{nlp_result["features"]["sentence_count"]}</span></div>
                <div style='display: flex; justify-content: space-between; padding: 5px 0;'><span>Avg Word Length</span> <span style='color: #12211F; font-weight: 600;'>{nlp_result["features"]["avg_word_length"]:.1f}</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # ── Detected words ──────────────────────────────────────────────────
        section_header("Detected Sentiment Tokens")
        word_col1, word_col2, word_col3 = st.columns(3)

        with word_col1:
            st.markdown(
                "<div style='color: #1D7451; font-weight: 600; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2em;'></span> <span>Positive Drivers</span></div>",
                unsafe_allow_html=True,
            )
            pos_words = nlp_result["features"]["positive_words"]
            if pos_words:
                words_html = "".join(
                    [
                        f"<span class='chip positive' style='margin-right: 6px; margin-bottom: 6px; display: inline-block;'>{w}</span>"
                        for w in pos_words[:8]
                    ]
                )
                st.markdown(f"<div>{words_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<span style='color: #5E6E6A; font-style: italic;'>None detected</span>", unsafe_allow_html=True
                )

        with word_col2:
            st.markdown(
                "<div style='color: #A3322B; font-weight: 600; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2em;'></span> <span>Negative Drivers</span></div>",
                unsafe_allow_html=True,
            )
            neg_words = nlp_result["features"]["negative_words"]
            if neg_words:
                words_html = "".join(
                    [
                        f"<span class='chip negative' style='margin-right: 6px; margin-bottom: 6px; display: inline-block;'>{w}</span>"
                        for w in neg_words[:8]
                    ]
                )
                st.markdown(f"<div>{words_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<span style='color: #5E6E6A; font-style: italic;'>None detected</span>", unsafe_allow_html=True
                )

        with word_col3:
            st.markdown(
                "<div style='color: #9A5B16; font-weight: 600; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2em;'></span> <span>Uncertainty Markers</span></div>",
                unsafe_allow_html=True,
            )
            unc_words = nlp_result["features"]["uncertainty_words"]
            if unc_words:
                words_html = "".join(
                    [
                        f"<span class='chip' style='background: rgba(154,91,22,0.15); color: #9A5B16; border: 1px solid rgba(154,91,22,0.3); margin-right: 6px; margin-bottom: 6px; display: inline-block;'>{w}</span>"
                        for w in unc_words[:8]
                    ]
                )
                st.markdown(f"<div>{words_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<span style='color: #5E6E6A; font-style: italic;'>None detected</span>", unsafe_allow_html=True
                )

        # ── Entities ────────────────────────────────────────────────────────
        section_header("Named Entities & Variables")
        entity_col1, entity_col2, entity_col3 = st.columns(3)

        with entity_col1:
            st.markdown(
                "<div style='color: #2E3F3C; font-weight: 600; margin-bottom: 10px;'>Percentages & Ratios</div>",
                unsafe_allow_html=True,
            )
            if nlp_result["entities"]["percentages"]:
                for pct in nlp_result["entities"]["percentages"]:
                    st.markdown(
                        f"<div style='background: #D5DEDC; padding: 4px 10px; border-radius: 6px; font-variant-numeric: tabular-nums; display: inline-block; margin: 2px;'>{pct}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<span style='color: #5E6E6A; font-style: italic;'>None found</span>", unsafe_allow_html=True
                )

        with entity_col2:
            st.markdown(
                "<div style='color: #2E3F3C; font-weight: 600; margin-bottom: 10px;'>Financial Figures</div>",
                unsafe_allow_html=True,
            )
            if nlp_result["entities"]["currencies"]:
                for curr in nlp_result["entities"]["currencies"][:5]:
                    st.markdown(
                        f"<div style='background: #D5DEDC; padding: 4px 10px; border-radius: 6px; font-variant-numeric: tabular-nums; display: inline-block; margin: 2px;'>{curr}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<span style='color: #5E6E6A; font-style: italic;'>None found</span>", unsafe_allow_html=True
                )

        with entity_col3:
            st.markdown(
                "<div style='color: #2E3F3C; font-weight: 600; margin-bottom: 10px;'>Corporate Entities</div>",
                unsafe_allow_html=True,
            )
            if nlp_result["entities"]["companies"]:
                for comp in nlp_result["entities"]["companies"][:5]:
                    st.markdown(
                        f"<div style='background: #D5DEDC; padding: 4px 10px; border-radius: 6px; font-variant-numeric: tabular-nums; display: inline-block; margin: 2px;'>{comp}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<span style='color: #5E6E6A; font-style: italic;'>None found</span>", unsafe_allow_html=True
                )

        # ── Chain of Thought Reasoning ──────────────────────────────────────
        section_header("Rule-based Reasoning Steps")
        st.markdown(
            "<p style='color: #4B5D5A;'>Step-by-step reasoning algorithm breakdown:</p>", unsafe_allow_html=True
        )

        with st.expander("View Full Execution Trace", expanded=False):
            for i, step in enumerate(cot_result.steps, 1):
                step_name = step.step.value.replace("_", " ").title()

                step_colors = {
                    "comprehension": "#4F9E93",
                    "entity_extraction": "#45627A",
                    "sentiment_detection": "#1D7451",
                    "context_analysis": "#9A5B16",
                    "confidence_calibration": "#9A5B16",
                    "final_synthesis": "#0E6E6A",
                }
                step_color = step_colors.get(step.step.value, "#5E6E6A")

                st.markdown(
                    f"""
                <div class='insight-card' style='position: relative; padding: 20px; background: #F7F9F8;
                            margin-bottom: 20px; border-left: 0; border: 1px solid #D5DEDC;'>
                    <div style='position: absolute; left: -32px; top: 25px; width: 22px; height: 22px; background: {step_color}; border-radius: 50%; box-shadow: 0 0 10px {step_color}; border: 4px solid #F7F9F8;'></div>
                    <h4 style='margin: 0 0 15px 0; color: {step_color}; display: flex; align-items: center; justify-content: space-between;'>
                        <span>Step {i}: {step_name}</span>
                        <span style='font-size: 0.8em; color: #B9C6C3; font-weight: normal; font-variant-numeric: tabular-nums;'>CONF: {step.confidence:.2f}</span>
                    </h4>
                    <div style='display: grid; grid-template-columns: 100px 1fr; gap: 10px; margin-bottom: 8px;'>
                        <div style='color: #4B5D5A; font-weight: 600; text-transform: uppercase; font-size: 0.8em; margin-top: 3px;'>Observation</div>
                        <div style='color: #2E3F3C;'>{step.observation}</div>
                    </div>
                    <div style='display: grid; grid-template-columns: 100px 1fr; gap: 10px; margin-bottom: 8px;'>
                        <div style='color: #4B5D5A; font-weight: 600; text-transform: uppercase; font-size: 0.8em; margin-top: 3px;'>Reasoning</div>
                        <div style='color: #2E3F3C;'>{step.reasoning}</div>
                    </div>
                    <div style='display: grid; grid-template-columns: 100px 1fr; gap: 10px;'>
                        <div style='color: {step_color}; font-weight: 600; text-transform: uppercase; font-size: 0.8em; margin-top: 3px;'>Conclusion</div>
                        <div style='color: #12211F; font-weight: 500;'>{step.conclusion}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        colA, colB = st.columns([1, 1])
        with colA:
            # Key Factors
            section_header("Driving Factors")
            if cot_result.key_factors:
                for factor in cot_result.key_factors:
                    st.markdown(
                        f"<div style='margin-bottom: 8px; display: flex; align-items: start; gap: 10px;'><span style='color: #0E6E6A;'>•</span> <span style='color: #12211F; line-height: 1.5;'>{factor}</span></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<span style='color: #5E6E6A; font-style: italic;'>No specific key factors identified</span>",
                    unsafe_allow_html=True,
                )

        with colB:
            # Final Explanation
            section_header("Final Executive Synthesis")
            st.markdown(
                f"""
            <div class='insight-card' style='background: #F7F9F8; height: 100%;'>
                <p style='color: #12211F; line-height: 1.6; font-size: 1.05em; margin: 0;'>{cot_result.explanation}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # About Loughran-McDonald
        with st.expander("About the Loughran-McDonald Financial Lexicon"):
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
        st.warning("Please enter some text to analyze.")

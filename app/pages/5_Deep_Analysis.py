"""
Financial Sentiment Analyzer — Deep Analysis (Elite NLP) Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st

# Import advanced NLP modules (optional dependency)
try:
    from nlp_advanced import FinancialTextAnalyzer
    from llm_enhanced import ChainOfThoughtReasoner
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar, get_sentiment_color

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Deep Analysis", page_icon="🧠", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.header("🧠 Deep Analysis")
st.markdown("""
**Elite NLP Analysis** with chain-of-thought reasoning, financial lexicon scoring,
and comprehensive linguistic analysis.
""")

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

if st.button("🔬 Run Deep Analysis", type="primary", use_container_width=True):
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
            st.error(f"❌ Deep analysis failed: {e}")
            st.stop()
        
        if cot_result is None or nlp_result is None:
            st.stop()
        
        # Add explicit asserts for strict static analysis
        assert cot_result is not None
        assert nlp_result is not None

        # ── Summary Card ────────────────────────────────────────────────────
        st.markdown("---")
        sentiment = cot_result.final_sentiment
        confidence = cot_result.final_confidence
        color = get_sentiment_color(sentiment)

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                    padding: 25px; border-radius: 15px; border-left: 5px solid {color}; margin-bottom: 20px;'>
            <h2 style='margin: 0; color: {color};'>🎯 {sentiment.upper()}</h2>
            <p style='margin: 10px 0 0 0; font-size: 18px;'>
                Confidence: <strong>{confidence:.1%}</strong> |
                Lexicon Score: <strong>{nlp_result['features']['sentiment_score']:.2f}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Key metrics ─────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📊 Sentiment Scores")
            st.metric("Lexicon Sentiment", f"{nlp_result['features']['sentiment_score']:.3f}")
            st.metric("Uncertainty", f"{nlp_result['features']['uncertainty_score']:.3f}")
            st.metric("Subjectivity", f"{nlp_result['features']['subjectivity_score']:.3f}")

        with col2:
            st.markdown("### 📈 Word Counts")
            st.metric("Positive Words", nlp_result["features"]["positive_word_count"])
            st.metric("Negative Words", nlp_result["features"]["negative_word_count"])
            st.metric("Uncertainty Words", nlp_result["features"]["uncertainty_word_count"])

        with col3:
            st.markdown("### 📝 Text Stats")
            st.metric("Word Count", nlp_result["features"]["word_count"])
            st.metric("Sentence Count", nlp_result["features"]["sentence_count"])
            st.metric("Avg Word Length", f"{nlp_result['features']['avg_word_length']:.1f}")

        st.markdown("---")

        # ── Detected words ──────────────────────────────────────────────────
        st.markdown("### 🔤 Detected Sentiment Words")
        word_col1, word_col2, word_col3 = st.columns(3)

        with word_col1:
            st.markdown("**✅ Positive Words**")
            pos_words = nlp_result["features"]["positive_words"]
            if pos_words:
                for word in pos_words[:8]:
                    st.markdown(
                        f"<span style='background-color: rgba(40, 167, 69, 0.2); padding: 3px 10px; "
                        f"border-radius: 15px; margin: 2px; display: inline-block;'>{word}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("*None detected*")

        with word_col2:
            st.markdown("**❌ Negative Words**")
            neg_words = nlp_result["features"]["negative_words"]
            if neg_words:
                for word in neg_words[:8]:
                    st.markdown(
                        f"<span style='background-color: rgba(220, 53, 69, 0.2); padding: 3px 10px; "
                        f"border-radius: 15px; margin: 2px; display: inline-block;'>{word}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("*None detected*")

        with word_col3:
            st.markdown("**⚠️ Uncertainty Words**")
            unc_words = nlp_result["features"]["uncertainty_words"]
            if unc_words:
                for word in unc_words[:8]:
                    st.markdown(
                        f"<span style='background-color: rgba(255, 193, 7, 0.2); padding: 3px 10px; "
                        f"border-radius: 15px; margin: 2px; display: inline-block;'>{word}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("*None detected*")

        st.markdown("---")

        # ── Entities ────────────────────────────────────────────────────────
        st.markdown("### 🏢 Extracted Entities")
        entity_col1, entity_col2, entity_col3 = st.columns(3)

        with entity_col1:
            st.markdown("**📊 Percentages**")
            if nlp_result["entities"]["percentages"]:
                for pct in nlp_result["entities"]["percentages"]:
                    st.code(pct)
            else:
                st.markdown("*None found*")

        with entity_col2:
            st.markdown("**💰 Currency**")
            if nlp_result["entities"]["currencies"]:
                for curr in nlp_result["entities"]["currencies"][:5]:
                    st.code(curr)
            else:
                st.markdown("*None found*")

        with entity_col3:
            st.markdown("**🏛️ Companies**")
            if nlp_result["entities"]["companies"]:
                for comp in nlp_result["entities"]["companies"][:5]:
                    st.code(comp)
            else:
                st.markdown("*None found*")

        st.markdown("---")

        # ── Chain of Thought Reasoning ──────────────────────────────────────
        st.markdown("### 🔗 Chain-of-Thought Reasoning")
        st.markdown("*Step-by-step reasoning process used to determine sentiment:*")

        with st.expander("📖 View Full Reasoning Trace", expanded=True):
            for i, step in enumerate(cot_result.steps, 1):
                step_name = step.step.value.replace("_", " ").title()

                step_colors = {
                    "comprehension": "#17a2b8",
                    "entity_extraction": "#6f42c1",
                    "sentiment_detection": "#28a745",
                    "context_analysis": "#fd7e14",
                    "confidence_calibration": "#20c997",
                    "final_synthesis": "#007bff",
                }
                step_color = step_colors.get(step.step.value, "#6c757d")

                st.markdown(f"""
                <div style='background-color: #262730; padding: 15px; border-radius: 10px;
                            margin-bottom: 10px; border-left: 4px solid {step_color};'>
                    <h4 style='margin: 0 0 10px 0; color: {step_color};'>Step {i}: {step_name}</h4>
                    <p style='margin: 5px 0;'><strong>Observation:</strong> {step.observation}</p>
                    <p style='margin: 5px 0;'><strong>Reasoning:</strong> {step.reasoning}</p>
                    <p style='margin: 5px 0;'><strong>Conclusion:</strong> {step.conclusion}</p>
                    <p style='margin: 5px 0; color: #6c757d;'><em>Confidence: {step.confidence:.2f}</em></p>
                </div>
                """, unsafe_allow_html=True)

        # Key Factors
        st.markdown("### 🔑 Key Factors")
        if cot_result.key_factors:
            for factor in cot_result.key_factors:
                st.markdown(f"• {factor}")
        else:
            st.markdown("*No specific key factors identified*")

        # Final Explanation
        st.markdown("### 💬 AI Explanation")
        st.markdown(f"""
        <div style='background-color: #262730; padding: 20px; border-radius: 10px;
                    border-left: 4px solid #007bff;'>
            {cot_result.explanation}
        </div>
        """, unsafe_allow_html=True)

        # About Loughran-McDonald
        with st.expander("ℹ️ About the Loughran-McDonald Financial Lexicon"):
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

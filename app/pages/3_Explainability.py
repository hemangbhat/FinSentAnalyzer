"""
Financial Sentiment Analyzer — Explainability Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import plotly.graph_objects as go

from explain import explain_prediction_baseline, highlight_text

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar, get_sentiment_color, create_probability_chart

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.header("🔍 Explainability")
st.markdown("Understand why the model made its prediction.")

if not selected_model.startswith("baseline_"):
    st.warning(
        "Explainability is currently available only for baseline models. "
        "Please select a baseline model from the sidebar."
    )
    st.stop()

explain_text = st.text_area(
    "Enter text to explain:",
    height=100,
    placeholder="Enter financial text...",
    key="explain_text",
)

if st.button("🔍 Explain Prediction", type="primary", use_container_width=True):
    if explain_text.strip():
        try:
            with st.spinner("Analyzing..."):
                explanation = explain_prediction_baseline(explain_text, selected_model)
        except Exception as e:
            st.error(f"❌ Explainability analysis failed: {e}")
            st.stop()

        # Show prediction
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Prediction")
            sentiment = explanation["prediction"]
            color = get_sentiment_color(sentiment)
            st.markdown(
                f"<h2 style='color: {color};'>{sentiment.upper()}</h2>",
                unsafe_allow_html=True,
            )

            if explanation["probabilities"]:
                st.markdown("**Probabilities:**")
                for label, prob in explanation["probabilities"].items():
                    st.write(f"- {label}: {prob:.1%}")

        with col2:
            if explanation["probabilities"]:
                fig = create_probability_chart(explanation["probabilities"])
                st.plotly_chart(fig, use_container_width=True)

        # Highlighted text
        st.markdown("### Text Analysis")
        st.markdown("Words highlighted by their sentiment contribution:")
        highlighted = highlight_text(explain_text, explanation["word_importance"])
        st.markdown(
            f"<div style='background-color: #262730; padding: 15px; border-radius: 8px; "
            f"line-height: 2;'>{highlighted}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("""
        <div style='margin-top: 10px; font-size: 0.9em;'>
        <span style='background-color: rgba(40, 167, 69, 0.2); padding: 2px 8px; border-radius: 3px;'>Positive</span>
        <span style='background-color: rgba(220, 53, 69, 0.2); padding: 2px 8px; border-radius: 3px; margin-left: 10px;'>Negative</span>
        <span style='background-color: rgba(0, 123, 255, 0.2); padding: 2px 8px; border-radius: 3px; margin-left: 10px;'>Neutral</span>
        </div>
        """, unsafe_allow_html=True)

        # Word importance
        st.markdown("### Key Words")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Positive Indicators**")
            for word, score in explanation["positive_words"]:
                st.write(f"- {word} ({score:.3f})")

        with col2:
            st.markdown("**Negative Indicators**")
            for word, score in explanation["negative_words"]:
                st.write(f"- {word} ({score:.3f})")

        with col3:
            st.markdown("**Neutral Indicators**")
            for word, score in explanation["neutral_words"]:
                st.write(f"- {word} ({score:.3f})")

        # Top words chart
        if explanation["word_importance"]:
            st.markdown("### Word Importance Scores")
            top_words = explanation["word_importance"][:10]
            words = [w for w, s, d in top_words]
            scores = [s for w, s, d in top_words]
            directions = [d for w, s, d in top_words]
            colors = [get_sentiment_color(d) for d in directions]

            fig = go.Figure(go.Bar(
                x=scores,
                y=words,
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                title="Top 10 Important Words",
                xaxis_title="Importance Score",
                yaxis=dict(autorange="reversed"),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter text to explain.")

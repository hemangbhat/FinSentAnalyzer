"""
Financial Sentiment Analyzer — Explainability Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st  # pyre-ignore
import plotly.graph_objects as go  # pyre-ignore

from explain import explain_prediction_baseline, highlight_text  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar, get_sentiment_color, create_probability_chart  # pyre-ignore

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 25px;'>
    <h1 style='font-size: 2.2em; font-weight: 700; margin-bottom: 5px;'>🔍 Explainability</h1>
    <p style='color: #94a3b8; font-size: 1.1em;'>Deconstruct the AI reasoning to see exactly which words influenced the decision.</p>
</div>
""", unsafe_allow_html=True)

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
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction Results")
        
        col1, col2 = st.columns([1, 1.2])

        with col1:
            sentiment = explanation["prediction"]
            color = get_sentiment_color(sentiment)
            st.markdown(f"""
            <div class='insight-card' style='text-align: center; display: flex; flex-direction: column; justify-content: center; height: 100%; border-top: 4px solid {color};'>
                <div style='color: #94a3b8; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>Final Verdict</div>
                <div style='font-size: 3.5em; font-weight: 800; color: {color}; line-height: 1.2; text-transform: capitalize;'>
                    {sentiment}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if explanation["probabilities"]:
                st.markdown("<div class='insight-card' style='padding: 10px;'>", unsafe_allow_html=True)
                fig = create_probability_chart(explanation["probabilities"])
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)

        # Highlighted text
        st.markdown("<hr style='border-color: rgba(255,255,255,0.02); margin: 20px 0;'>", unsafe_allow_html=True)
        st.markdown("### 📝 Text Analysis")
        st.markdown("<p style='color: #94a3b8;'>Words highlighted by their sentiment contribution magnitude:</p>", unsafe_allow_html=True)
        
        highlighted = highlight_text(explain_text, explanation["word_importance"])
        st.markdown(
            f"<div class='insight-card' style='font-size: 1.1em; line-height: 1.8; color: #e2e8f0;'>"
            f"{highlighted}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("""
        <div style='margin-top: 15px; font-size: 0.9em; display: flex; gap: 15px;'>
            <div style='display: flex; align-items: center; gap: 5px;'><span style='display: inline-block; width: 12px; height: 12px; background: rgba(16, 185, 129, 0.4); border-radius: 2px;'></span> <span style='color: #94a3b8;'>Positive Driver</span></div>
            <div style='display: flex; align-items: center; gap: 5px;'><span style='display: inline-block; width: 12px; height: 12px; background: rgba(239, 68, 68, 0.4); border-radius: 2px;'></span> <span style='color: #94a3b8;'>Negative Driver</span></div>
            <div style='display: flex; align-items: center; gap: 5px;'><span style='display: inline-block; width: 12px; height: 12px; background: rgba(59, 130, 246, 0.4); border-radius: 2px;'></span> <span style='color: #94a3b8;'>Neutral / Uncertain</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Word importance
        st.markdown("<hr style='border-color: rgba(255,255,255,0.02); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("### 🧩 Extracted Features")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='insight-card' style='height: 100%; border-top: 4px solid #10b981;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #10b981; margin-top: 0;'>Positive Drivers</h4>", unsafe_allow_html=True)
            if explanation.get("lexicon_positive"):
                st.markdown("<div style='font-size: 0.85em; color: #94a3b8; margin-bottom: 5px; text-transform: uppercase;'>Lexicon Matches</div>", unsafe_allow_html=True)
                words_html = "".join([f"<span class='chip positive' style='margin-right: 6px; margin-bottom: 6px; display: inline-block;'>{w}</span>" for w in explanation["lexicon_positive"][:5]])
                st.markdown(f"<div>{words_html}</div>", unsafe_allow_html=True)
            if explanation["positive_words"]:
                st.markdown("<div style='font-size: 0.85em; color: #94a3b8; margin: 10px 0 5px 0; text-transform: uppercase;'>Top Model Weights</div>", unsafe_allow_html=True)
                for word, score in explanation["positive_words"]:
                    st.markdown(f"<div style='display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color: #cbd5e1;'><span style='font-family: monospace;'>{word}</span> <span>{score:.3f}</span></div>", unsafe_allow_html=True)
            if not explanation.get("lexicon_positive") and not explanation["positive_words"]:
                st.markdown("<span style='color: #64748b; font-style: italic;'>None detected</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='insight-card' style='height: 100%; border-top: 4px solid #ef4444;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #ef4444; margin-top: 0;'>Negative Drivers</h4>", unsafe_allow_html=True)
            if explanation.get("lexicon_negative"):
                st.markdown("<div style='font-size: 0.85em; color: #94a3b8; margin-bottom: 5px; text-transform: uppercase;'>Lexicon Matches</div>", unsafe_allow_html=True)
                words_html = "".join([f"<span class='chip negative' style='margin-right: 6px; margin-bottom: 6px; display: inline-block;'>{w}</span>" for w in explanation["lexicon_negative"][:5]])
                st.markdown(f"<div>{words_html}</div>", unsafe_allow_html=True)
            if explanation["negative_words"]:
                st.markdown("<div style='font-size: 0.85em; color: #94a3b8; margin: 10px 0 5px 0; text-transform: uppercase;'>Top Model Weights</div>", unsafe_allow_html=True)
                for word, score in explanation["negative_words"]:
                    st.markdown(f"<div style='display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color: #cbd5e1;'><span style='font-family: monospace;'>{word}</span> <span>{score:.3f}</span></div>", unsafe_allow_html=True)
            if not explanation.get("lexicon_negative") and not explanation["negative_words"]:
                st.markdown("<span style='color: #64748b; font-style: italic;'>None detected</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='insight-card' style='height: 100%; border-top: 4px solid #3b82f6;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #3b82f6; margin-top: 0;'>Neutral / Uncertain</h4>", unsafe_allow_html=True)
            if explanation.get("lexicon_uncertainty"):
                st.markdown("<div style='font-size: 0.85em; color: #94a3b8; margin-bottom: 5px; text-transform: uppercase;'>Lexicon Matches</div>", unsafe_allow_html=True)
                words_html = "".join([f"<span class='chip neutral' style='margin-right: 6px; margin-bottom: 6px; display: inline-block;'>{w}</span>" for w in explanation["lexicon_uncertainty"][:5]])
                st.markdown(f"<div>{words_html}</div>", unsafe_allow_html=True)
            if explanation["neutral_words"]:
                st.markdown("<div style='font-size: 0.85em; color: #94a3b8; margin: 10px 0 5px 0; text-transform: uppercase;'>Top Model Weights</div>", unsafe_allow_html=True)
                for word, score in explanation["neutral_words"]:
                    st.markdown(f"<div style='display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color: #cbd5e1;'><span style='font-family: monospace;'>{word}</span> <span>{score:.3f}</span></div>", unsafe_allow_html=True)
            if not explanation.get("lexicon_uncertainty") and not explanation["neutral_words"]:
                st.markdown("<span style='color: #64748b; font-style: italic;'>None detected</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

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
                title={"text": "Top 10 High Impact Words", "font": {"color": "#f8fafc", "size": 16}},
                xaxis_title="Influence Score",
                yaxis={"autorange": "reversed"},
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={"color": "#cbd5e1", "family": "Inter, sans-serif"},
                xaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)", "linecolor": "rgba(255,255,255,0.1)"},
                yaxis_tickfont={"size": 13, "family": "monospace"}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter text to explain.")

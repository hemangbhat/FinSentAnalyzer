"""
Financial Sentiment Analyzer — Word Insights Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st  # pyre-ignore
import pandas as pd  # pyre-ignore
import plotly.express as px  # pyre-ignore

from preprocess import LABEL_MAP_INV  # pyre-ignore
from explain import get_feature_importance_summary  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar, get_sentiment_color  # pyre-ignore

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Word Insights", page_icon="💡", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 25px;'>
    <h1 style='font-size: 2.2em; font-weight: 700; margin-bottom: 5px;'>💡 Word Insights</h1>
    <p style='color: #94a3b8; font-size: 1.1em;'>Explore the trained vocabulary and structural indicators learned by the core engine.</p>
</div>
""", unsafe_allow_html=True)

if not selected_model.startswith("baseline_"):
    st.warning(
        "Word insights are currently available only for baseline models. "
        "Please select a baseline model from the sidebar."
    )
    st.stop()

with st.spinner("Loading word insights..."):
    try:
        importance = get_feature_importance_summary(selected_model)
        assert isinstance(importance, dict), "Importance summary must be a dictionary"

        if "error" in importance:
            st.error(importance["error"])
            st.stop()

        # Top words for each sentiment
        classes = list(LABEL_MAP_INV.values())
        cols = st.columns(len(classes))

        emojis = {"positive": "🔥", "negative": "🧊", "neutral": "⚖️"}
        
        # We can map 'sentiment' string to the chip class directly if it matches 'positive', 'negative', 'neutral'
        class_map = {
            "positive": "chip positive",
            "negative": "chip negative",
            "neutral": "chip neutral"
        }

        for i, sentiment in enumerate(classes):
            with cols[i]:
                emoji = emojis.get(sentiment, "📊")
                chip_class = class_map.get(sentiment, "chip")
                color = "#10b981" if sentiment == "positive" else "#ef4444" if sentiment == "negative" else "#3b82f6"
                
                st.markdown(f"<div class='insight-card' style='height: 100%; border-top: 4px solid {color};'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='margin-top:0; color: {color}; display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2em;'>{emoji}</span> {sentiment.capitalize()} Drivers</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #94a3b8; font-size: 0.9em; margin-bottom: 15px;'>Top indicative tokens:</p>", unsafe_allow_html=True)

                key = f"top_{sentiment}"
                if key in importance:  # pyre-ignore
                    chips = []
                    for word, score in importance[key][:15]:  # pyre-ignore
                        chips.append(f"<span class='{chip_class}' style='margin-bottom:8px; display:inline-block;'>{word}</span> <span style='font-size:0.85em; color: #64748b; font-family: monospace; margin-left: 2px; margin-right: 12px;'>{score:.2f}</span>")
                    if chips:
                        st.markdown(f"<div style='display: flex; flex-wrap: wrap; line-height: 2.2;'>{''.join(chips)}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Visualization
        st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.05); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("### 📊 Cross-Sentiment Comparison")

        chart_data = []
        for sentiment in classes:
            key = f"top_{sentiment}"
            if key in importance:  # pyre-ignore
                for word, score in importance[key][:5]:  # pyre-ignore
                    chart_data.append({
                        "Word": word,
                        "Score": score,
                        "Sentiment": sentiment.capitalize(),
                    })

        if chart_data:
            chart_df = pd.DataFrame(chart_data)

            color_map: dict[str, str] = {
                "Positive": "#10b981",
                "Negative": "#ef4444",
                "Neutral": "#3b82f6",
            }
            for cls in classes:
                cls_str = str(cls)
                if cls_str.capitalize() not in color_map:
                    color_map[cls_str.capitalize()] = "#64748b"

            fig = px.bar(
                chart_df,
                x="Score",
                y="Word",
                color="Sentiment",
                orientation="h",
                color_discrete_map=color_map,
                text="Score",
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', textfont={"color": "#cbd5e1"})
            fig.update_layout(
                height=500,
                yaxis={"categoryorder": "total ascending", "tickfont": {"family": "monospace"}},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={"color": "#cbd5e1", "family": "Inter, sans-serif"},
                margin={"t": 40, "b": 10, "l": 10, "r": 10},
                xaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.05)", "title": "Relative Influence Score"},
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as e:
        st.error(f"❌ Error loading word insights: {e}")

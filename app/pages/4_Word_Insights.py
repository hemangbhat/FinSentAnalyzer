"""
Financial Sentiment Analyzer — Word Insights Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from preprocess import LABEL_MAP_INV
from explain import get_feature_importance_summary

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar, get_sentiment_color

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Word Insights", page_icon="💡", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.header("💡 Word Insights")
st.markdown("Discover which words most strongly indicate each sentiment in the trained model.")

if not selected_model.startswith("baseline_"):
    st.warning(
        "Word insights are currently available only for baseline models. "
        "Please select a baseline model from the sidebar."
    )
    st.stop()

with st.spinner("Loading word insights..."):
    try:
        importance = get_feature_importance_summary(selected_model)

        if "error" in importance:
            st.error(importance["error"])
            st.stop()

        # Top words for each sentiment
        classes = list(LABEL_MAP_INV.values())
        cols = st.columns(len(classes))

        emojis = {"positive": "📈", "negative": "📉", "neutral": "↔️"}
        bgs = {
            "positive": "rgba(40, 167, 69, 0.2)",
            "negative": "rgba(220, 53, 69, 0.2)",
            "neutral": "rgba(0, 123, 255, 0.2)",
        }

        for i, sentiment in enumerate(classes):
            with cols[i]:
                emoji = emojis.get(sentiment, "📊")
                bg_color = bgs.get(sentiment, "rgba(108, 117, 125, 0.2)")
                st.markdown(f"### {emoji} {sentiment.capitalize()} Words")
                st.markdown(f"Words that indicate {sentiment} sentiment:")

                key = f"top_{sentiment}"
                if key in importance:
                    for word, score in importance[key]:
                        st.markdown(
                            f"<span style='background-color: {bg_color}; padding: 3px 8px; "
                            f"border-radius: 4px; margin: 2px; display: inline-block;'>"
                            f"**{word}** ({score:.3f})</span>",
                            unsafe_allow_html=True,
                        )

        # Visualization
        st.markdown("---")
        st.markdown("### Word Importance Comparison")

        chart_data = []
        for sentiment in classes:
            key = f"top_{sentiment}"
            if key in importance:
                for word, score in importance[key][:5]:
                    chart_data.append({
                        "Word": word,
                        "Score": score,
                        "Sentiment": sentiment.capitalize(),
                    })

        if chart_data:
            chart_df = pd.DataFrame(chart_data)

            color_map = {
                "Positive": "#28a745",
                "Negative": "#dc3545",
                "Neutral": "#007bff",
            }
            for cls in classes:
                if cls.capitalize() not in color_map:
                    color_map[cls.capitalize()] = "#6c757d"

            fig = px.bar(
                chart_df,
                x="Score",
                y="Word",
                color="Sentiment",
                orientation="h",
                color_discrete_map=color_map,
                title="Top 5 Words for Each Sentiment",
            )
            fig.update_layout(
                height=500,
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading word insights: {e}")

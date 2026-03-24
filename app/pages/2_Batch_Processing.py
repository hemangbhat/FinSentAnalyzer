"""
Financial Sentiment Analyzer — Batch Processing Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

from llm_explain import generate_market_outlook

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar, get_sentiment_color

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Batch Processing", page_icon="📁", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.header("📁 Batch Processing")
st.markdown("Upload a CSV or TXT file to analyze multiple texts.")

uploaded_file = st.file_uploader(
    "Upload file",
    type=["csv", "txt"],
    help="CSV files should have a 'text' column. TXT files should have one text per line.",
)

if uploaded_file:
    # ── Read file ───────────────────────────────────────────────────────────
    texts = []
    df = pd.DataFrame()
    text_column = ""
    try:
        if uploaded_file.name.endswith(".csv"):
            csv_decoded = False
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    csv_decoded = True
                    break
                except UnicodeDecodeError:
                    continue

            if not csv_decoded:
                st.error("❌ Could not decode CSV file. Please use UTF-8 encoding.")
                st.stop()

            text_columns = df.select_dtypes(include=["object"]).columns.tolist()
            if not text_columns:
                st.error("❌ No text columns found in the CSV.")
                st.stop()

            text_column = st.selectbox("Select text column:", text_columns)
            texts = df[text_column].dropna().tolist()
        else:
            file_bytes = uploaded_file.getvalue()
            content = None
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    content = StringIO(file_bytes.decode(encoding))
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                st.error("❌ Could not decode file. Please use UTF-8 encoding.")
                st.stop()

            texts = [line.strip() for line in content if line.strip()]
            df = pd.DataFrame({"text": texts})
            text_column = "text"

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    st.info(f"Found {len(texts)} texts to analyze.")

    if st.button("🚀 Analyze All", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        results = []

        for i, text in enumerate(texts):
            try:
                result = predictor.predict(text)
            except Exception:
                result = {"label": "error", "confidence": 0.0}
            results.append(result)
            progress_bar.progress((i + 1) / len(texts))

        # Add results to dataframe
        assert df is not None
        df["Sentiment"] = [r["label"] for r in results]
        df["Confidence"] = [r.get("confidence", None) for r in results]

        # Filter out errors
        error_count = (df["Sentiment"] == "error").sum()
        if error_count > 0:
            st.warning(f"⚠️ {error_count} text(s) could not be analyzed and were marked as 'error'.")

        st.success(f"Analyzed {len(texts)} texts!")

        # ── Summary stats ───────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        sentiment_counts = df["Sentiment"].value_counts()

        with col1:
            pos_count = sentiment_counts.get("positive", 0)
            st.metric("Positive", pos_count, f"{pos_count/len(df)*100:.1f}%")

        with col2:
            neu_count = sentiment_counts.get("neutral", 0)
            st.metric("Neutral", neu_count, f"{neu_count/len(df)*100:.1f}%")

        with col3:
            neg_count = sentiment_counts.get("negative", 0)
            st.metric("Negative", neg_count, f"{neg_count/len(df)*100:.1f}%")

        # Distribution chart
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={
                "positive": "#28a745",
                "negative": "#dc3545",
                "neutral": "#007bff",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Sentiment Trend Analysis ────────────────────────────────────────
        st.markdown("### 📈 Sentiment Trend Analysis")
        pos_pct = pos_count / len(df) * 100
        neg_pct = neg_count / len(df) * 100
        neu_pct = neu_count / len(df) * 100

        if pos_pct > neg_pct + 10:
            trend, trend_color, trend_icon = "bullish", "#28a745", "📈"
        elif neg_pct > pos_pct + 10:
            trend, trend_color, trend_icon = "bearish", "#dc3545", "📉"
        else:
            trend, trend_color, trend_icon = "mixed", "#007bff", "↔️"

        avg_confidence = df["Confidence"].mean() if df["Confidence"].notna().any() else 0
        dominant = df["Sentiment"].mode().iloc[0].capitalize() if len(df) > 0 else "N/A"

        st.markdown(
            f"""
            <div style='background-color: #262730; padding: 20px; border-radius: 10px;
                        border-left: 5px solid {trend_color};'>
                <h4 style='margin: 0;'>{trend_icon} Overall Sentiment: <span style='color: {trend_color};'>{trend.upper()}</span></h4>
                <p style='margin-top: 10px; margin-bottom: 0;'>
                    <b>Summary:</b> Out of {len(df)} texts analyzed, {pos_pct:.1f}% are positive,
                    {neu_pct:.1f}% are neutral, and {neg_pct:.1f}% are negative.<br>
                    <b>Average Confidence:</b> {avg_confidence:.1%}<br>
                    <b>Dominant Sentiment:</b> {dominant}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")  # Spacer

        # ── Results table ───────────────────────────────────────────────────
        st.markdown("### Detailed Results")
        st.dataframe(
            df[[text_column, "Sentiment", "Confidence"]].head(100),
            use_container_width=True,
        )

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="sentiment_results.csv",
            mime="text/csv",
        )

        # ── Market Outlook Report ───────────────────────────────────────────
        with st.expander("📊 Generate Market Outlook Report", expanded=False):
            st.markdown("*AI-generated market sentiment analysis based on your batch data*")
            if st.button("🔮 Generate Outlook", key="generate_outlook"):
                with st.spinner("Generating market outlook..."):
                    try:
                        sentiment_counts_dict = {
                            "positive": sentiment_counts.get("positive", 0),
                            "neutral": sentiment_counts.get("neutral", 0),
                            "negative": sentiment_counts.get("negative", 0),
                        }
                        outlook = generate_market_outlook(
                            sentiment_counts=sentiment_counts_dict,
                            total_texts=len(df),
                            avg_confidence=avg_confidence,
                        )
                        st.markdown(outlook)
                    except Exception as e:
                        st.error(f"❌ Could not generate market outlook: {e}")

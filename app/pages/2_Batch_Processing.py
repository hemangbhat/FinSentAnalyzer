"""
Financial Sentiment Analyzer — Batch Processing Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st  # pyre-ignore
import pandas as pd  # pyre-ignore
import plotly.express as px  # pyre-ignore
import os
from io import StringIO
from llm_explain import generate_market_outlook  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar, get_sentiment_color  # pyre-ignore

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Batch Processing", page_icon="📁", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 25px;'>
    <h1 style='font-size: 2.2em; font-weight: 700; margin-bottom: 5px;'>📁 Batch Processing</h1>
    <p style='color: #94a3b8; font-size: 1.1em;'>Analyze hundreds of transcripts or news streams rapidly at scale.</p>
</div>
""", unsafe_allow_html=True)

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
            # In pandas, sum() on boolean returns int. Pyre doesn't recognize this.
            total_empty = int(df[text_column].isna().sum())
            if total_empty > 0:
                st.warning(f"⚠️ Found {total_empty} empty texts. These will be skipped.")
            clean_df = df.dropna(subset=[text_column]).copy()
            texts = clean_df[text_column].tolist()
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
            assert content is not None

            texts = [line.strip() for line in content if line.strip()]
            df = pd.DataFrame({"text": texts})
            text_column = "text"
            clean_df = df.copy() # For consistency with CSV path

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    st.info(f"Found {len(texts)} texts to analyze.")

    if st.button("🚀 Analyze All", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        results = []

        assert predictor is not None, "Predictor is required"
        for i, text in enumerate(texts):
            try:
                result = predictor.predict(text)
            except Exception:
                result = {"label": "error", "confidence": 0.0}
            results.append(result)
            progress_bar.progress((i + 1) / len(texts))

        # Add results to dataframe
        assert clean_df is not None
        clean_df["Sentiment"] = [r["label"] for r in results]
        clean_df["Confidence"] = [r.get("confidence", None) for r in results]

        # Filter out errors
        error_count = clean_df["Sentiment"].tolist().count("error")
        if error_count > 0:
            st.warning(f"⚠️ {error_count} text(s) could not be analyzed and were marked as 'error'.")

        st.success(f"Analyzed {len(texts)} texts!")

        # ── Summary stats ───────────────────────────────────────────────────
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("### 📊 Summary Statistics")

        col1, col2, col3 = st.columns(3)
        sentiment_counts = clean_df["Sentiment"].value_counts()

        with col1:
            pos_count = sentiment_counts.get("positive", 0)
            st.markdown(f"""
            <div class='metric-card' style='border-top: 4px solid #10b981;'>
                <div style='color: #94a3b8; font-size: 0.9em; text-transform: uppercase;'>Positive Signals</div>
                <div style='font-size: 2.2em; font-weight: 700; color: #10b981;'>{pos_count}</div>
                <div style='color: #cbd5e1; font-size: 0.85em; opacity: 0.8;'>{pos_count/len(clean_df)*100:.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            neu_count = sentiment_counts.get("neutral", 0)
            st.markdown(f"""
            <div class='metric-card' style='border-top: 4px solid #3b82f6;'>
                <div style='color: #94a3b8; font-size: 0.9em; text-transform: uppercase;'>Neutral Signals</div>
                <div style='font-size: 2.2em; font-weight: 700; color: #3b82f6;'>{neu_count}</div>
                <div style='color: #cbd5e1; font-size: 0.85em; opacity: 0.8;'>{neu_count/len(clean_df)*100:.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            neg_count = sentiment_counts.get("negative", 0)
            st.markdown(f"""
            <div class='metric-card' style='border-top: 4px solid #ef4444;'>
                <div style='color: #94a3b8; font-size: 0.9em; text-transform: uppercase;'>Negative Signals</div>
                <div style='font-size: 2.2em; font-weight: 700; color: #ef4444;'>{neg_count}</div>
                <div style='color: #cbd5e1; font-size: 0.85em; opacity: 0.8;'>{neg_count/len(clean_df)*100:.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Distribution chart
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Distribution",
            color=sentiment_counts.index,
            color_discrete_map={
                "positive": "#10b981",
                "negative": "#ef4444",
                "neutral": "#3b82f6",
                "error": "#f59e0b",
            },
            hole=0.4 # Donut chart looks more modern
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#cbd5e1", family="Inter, sans-serif"),
            margin=dict(t=40, b=10, l=10, r=10),
            title={"font": {"color": "#f8fafc", "size": 18}}
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Sentiment Trend Analysis ────────────────────────────────────────
        st.markdown("### 📈 Sentiment Trend Analysis")
        pos_pct = pos_count / len(clean_df) * 100
        neg_pct = neg_count / len(clean_df) * 100
        neu_pct = neu_count / len(clean_df) * 100

        if pos_pct > neg_pct + 10:
            trend, trend_color, trend_icon = "Bullish", "#10b981", "🔥"
        elif neg_pct > pos_pct + 10:
            trend, trend_color, trend_icon = "Bearish", "#ef4444", "🧊"
        else:
            trend, trend_color, trend_icon = "Mixed / Sideways", "#3b82f6", "⚖️"

        avg_confidence = clean_df["Confidence"].mean() if clean_df["Confidence"].notna().any() else 0
        dominant = clean_df["Sentiment"].mode().iloc[0].capitalize() if len(clean_df) > 0 else "N/A"

        st.markdown(
            f"""
            <div class='insight-card' style='border-left: 5px solid {trend_color};'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <div style='color: #94a3b8; font-size: 0.9em; text-transform: uppercase;'>Current Market Trend</div>
                        <h3 style='margin: 5px 0 0 0; color: {trend_color}; font-size: 1.8em;'>{trend_icon} {trend}</h3>
                    </div>
                    <div style='text-align: right;'>
                        <div style='color: #94a3b8; font-size: 0.9em; text-transform: uppercase;'>Model Confidence</div>
                        <div style='font-size: 1.5em; font-weight: 700; color: #f8fafc;'>{avg_confidence:.1%}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")  # Spacer

        # ── Results table ───────────────────────────────────────────────────
        st.markdown("### Detailed Results")
        st.dataframe(
            clean_df[[text_column, "Sentiment", "Confidence"]].head(100),
            use_container_width=True,
        )

        # Download button
        csv = clean_df.to_csv(index=False)
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
                    assert predictor is not None, "Predictor is required"
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

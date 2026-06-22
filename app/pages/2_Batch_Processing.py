"""
Financial Sentiment Analyzer — Batch Processing.
Enterprise flow: upload → validate → preview → analyze → summarize → export.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from io import StringIO

import pandas as pd  # pyre-ignore
import plotly.express as px  # pyre-ignore
import streamlit as st  # pyre-ignore

from llm_explain import generate_market_outlook  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import (  # pyre-ignore
    COLORS,
    empty_state,
    inject_css,
    kpi_strip,
    page_header,
    render_chart,
    section_header,
    setup_sidebar,
    status_banner,
    style_fig,
)

st.set_page_config(page_title="Batch Processing", page_icon="📁", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

page_header(
    "Batch Processing",
    "Score thousands of headlines or transcripts in one pass, review aggregate signals, and export a report.",
    eyebrow="Bulk inference",
)

uploaded_file = st.file_uploader(
    "Upload a dataset",
    type=["csv", "txt"],
    help="CSV: include a text column. TXT: one text per line.",
)

if not uploaded_file:
    empty_state(
        "No file uploaded yet",
        "Drop a CSV (with a text column) or a TXT file (one text per line) to begin. "
        "You'll be able to preview and validate before running analysis.",
        icon="📁",
    )
    st.stop()

# ── Read + validate ──────────────────────────────────────────────────────────
texts: list[str] = []
df = pd.DataFrame()
text_column = ""
clean_df = pd.DataFrame()
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
            status_banner("Could not decode CSV file. Please save it as UTF-8.", kind="error")
            st.stop()

        text_columns = df.select_dtypes(include=["object"]).columns.tolist()
        if not text_columns:
            status_banner("No text columns found in the CSV.", kind="error")
            st.stop()

        text_column = st.selectbox("Text column", text_columns)
        total_empty = int(df[text_column].isna().sum())
        if total_empty > 0:
            status_banner(f"{total_empty} empty rows will be skipped.", kind="warning")
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
            status_banner("Could not decode file. Please use UTF-8 encoding.", kind="error")
            st.stop()
        texts = [line.strip() for line in content if line.strip()]
        df = pd.DataFrame({"text": texts})
        text_column = "text"
        clean_df = df.copy()
except Exception as exc:  # noqa: BLE001
    status_banner(f"Error reading file: {exc}", kind="error")
    st.stop()

# Validation KPIs + preview
kpi_strip(
    [
        {"label": "Rows Detected", "value": f"{len(df):,}", "accent": COLORS["primary"]},
        {"label": "Valid Texts", "value": f"{len(texts):,}", "accent": COLORS["positive"]},
        {"label": "Text Column", "value": text_column, "accent": COLORS["accent"]},
        {"label": "Source", "value": uploaded_file.name.split(".")[-1].upper(), "accent": COLORS["warning"]},
    ]
)

with st.expander("Preview first rows", expanded=True):
    st.dataframe(clean_df[[text_column]].head(8), use_container_width=True)

run = st.button("Analyze All", type="primary", use_container_width=True)
if not run:
    st.stop()

# ── Analyze ──────────────────────────────────────────────────────────────────
progress_bar = st.progress(0.0, text="Scoring texts…")
results = []
assert predictor is not None
for i, text in enumerate(texts):
    try:
        results.append(predictor.predict(text))
    except Exception:  # noqa: BLE001
        results.append({"label": "error", "confidence": 0.0})
    progress_bar.progress((i + 1) / len(texts), text=f"Scoring texts… {i + 1}/{len(texts)}")
progress_bar.empty()

clean_df["Sentiment"] = [r["label"] for r in results]
clean_df["Confidence"] = [r.get("confidence", None) for r in results]

error_count = clean_df["Sentiment"].tolist().count("error")
if error_count > 0:
    status_banner(f"{error_count} text(s) could not be analyzed and were marked as 'error'.", kind="warning")
status_banner(f"Analyzed {len(texts):,} texts successfully.", kind="success")

# ── Summary ──────────────────────────────────────────────────────────────────
sentiment_counts = clean_df["Sentiment"].value_counts()
pos_count = int(sentiment_counts.get("positive", 0))
neu_count = int(sentiment_counts.get("neutral", 0))
neg_count = int(sentiment_counts.get("negative", 0))
total = len(clean_df)
avg_confidence = clean_df["Confidence"].mean() if clean_df["Confidence"].notna().any() else 0.0

section_header("Summary", "Aggregate sentiment across the batch")
kpi_strip(
    [
        {
            "label": "Positive",
            "value": pos_count,
            "sub": f"{pos_count / total * 100:.1f}%",
            "accent": COLORS["positive"],
        },
        {"label": "Neutral", "value": neu_count, "sub": f"{neu_count / total * 100:.1f}%", "accent": COLORS["neutral"]},
        {
            "label": "Negative",
            "value": neg_count,
            "sub": f"{neg_count / total * 100:.1f}%",
            "accent": COLORS["negative"],
        },
        {"label": "Avg Confidence", "value": f"{avg_confidence:.1%}", "accent": COLORS["accent"]},
    ]
)

chart_col, trend_col = st.columns([1.1, 1])
with chart_col:
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map={"positive": "#10b981", "negative": "#ef4444", "neutral": "#3b82f6", "error": "#f59e0b"},
        hole=0.55,
    )
    fig.update_traces(textinfo="percent", textfont_size=13)
    style_fig(fig, height=320, title="Sentiment Distribution")
    render_chart(fig)

with trend_col:
    pos_pct = pos_count / total * 100
    neg_pct = neg_count / total * 100
    if pos_pct > neg_pct + 10:
        trend, trend_color, trend_icon = "Bullish", "#10b981", "▲"
    elif neg_pct > pos_pct + 10:
        trend, trend_color, trend_icon = "Bearish", "#ef4444", "▼"
    else:
        trend, trend_color, trend_icon = "Mixed / Sideways", "#3b82f6", "≈"
    st.markdown(
        f"""
        <div class='insight-card' style='border-left:4px solid {trend_color};height:100%;display:flex;flex-direction:column;justify-content:center;'>
            <div style='color:#94a3b8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;'>Aggregate Signal</div>
            <div style='color:{trend_color};font-size:2.1rem;font-weight:800;margin:6px 0;'>{trend_icon} {trend}</div>
            <div style='color:#94a3b8;font-size:0.92rem;line-height:1.5;'>
                Based on {total:,} scored texts, with mean model confidence of
                <b style='color:#f8fafc;'>{avg_confidence:.1%}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Results + export ─────────────────────────────────────────────────────────
section_header("Detailed Results")
st.dataframe(clean_df[[text_column, "Sentiment", "Confidence"]].head(200), use_container_width=True)

st.download_button(
    label="Download Results (CSV)",
    data=clean_df.to_csv(index=False),
    file_name="sentiment_results.csv",
    mime="text/csv",
)

# ── Market outlook (template-based) ──────────────────────────────────────────
with st.expander("Generate market outlook report", expanded=False):
    st.caption("Template-based summary derived from the batch sentiment distribution.")
    if st.button("Generate Outlook", key="generate_outlook"):
        with st.spinner("Generating outlook…"):
            try:
                outlook = generate_market_outlook(
                    sentiment_counts={"positive": pos_count, "neutral": neu_count, "negative": neg_count},
                    total_texts=len(df),
                    avg_confidence=avg_confidence,
                )
                st.markdown(outlook)
            except Exception as exc:  # noqa: BLE001
                status_banner(f"Could not generate market outlook: {exc}", kind="error")

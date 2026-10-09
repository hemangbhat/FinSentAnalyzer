"""
Financial Sentiment Analyzer — Batch Processing.
Enterprise flow: upload → validate → preview → analyze → summarize → export.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from io import StringIO

import pandas as pd  # pyre-ignore
import streamlit as st  # pyre-ignore

from llm_explain import generate_market_outlook  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import (  # pyre-ignore
    COLORS,
    SENTIMENT_GLYPHS,
    active_model,
    get_sentiment_color,
    kpi_strip,
    page_header,
    probability_band,
    section_header,
    status_banner,
)

selected_model, predictor = active_model()

if predictor is None:
    st.stop()

page_header(
    "Batch Processing",
    "Score thousands of headlines or transcripts in one pass, review aggregate signals, and export a report.",
)

uploaded_file = st.file_uploader(
    "Upload a CSV (with a text column) or a TXT file (one text per line)",
    type=["csv", "txt"],
)
use_sample = st.toggle(
    "Or try 50 sentences from the PhraseBank test split",
    key="batch_sample",
    disabled=uploaded_file is not None,
)

if not uploaded_file and not use_sample:
    st.caption("You can preview and choose the text column before anything is scored.")
    st.stop()

# ── Read + validate ──────────────────────────────────────────────────────────
texts: list[str] = []
df = pd.DataFrame()
text_column = ""
clean_df = pd.DataFrame()
try:
    if uploaded_file is None:
        from preprocess import load_processed_data  # pyre-ignore

        df = load_processed_data("test").head(50).rename(columns={"sentence": "text"})[["text"]]
        text_column = "text"
        clean_df = df.copy()
        texts = df["text"].tolist()
    elif uploaded_file.name.endswith(".csv"):
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

# Cap rows to protect the shared instance from OOM / timeouts.
MAX_ROWS = int(os.getenv("FINSIGHT_MAX_BATCH_ROWS", "2000"))
if len(texts) > MAX_ROWS:
    status_banner(
        f"Large file detected — analyzing the first {MAX_ROWS:,} of {len(texts):,} rows "
        "to keep the shared demo responsive. Run locally to process the full file.",
        kind="warning",
    )
    texts = texts[:MAX_ROWS]
    clean_df = clean_df.iloc[:MAX_ROWS].copy()

# Validation KPIs + preview
kpi_strip(
    [
        {"label": "Rows detected", "value": f"{len(df):,}", "accent": COLORS["primary"]},
        {"label": "Valid texts", "value": f"{len(texts):,}", "accent": COLORS["positive"]},
        {"label": "Text column", "value": text_column, "accent": COLORS["accent"]},
        {"label": "Source", "value": uploaded_file.name.split(".")[-1].upper() if uploaded_file else "Sample"},
    ]
)

with st.expander("Preview first rows", expanded=True):
    st.dataframe(clean_df[[text_column]].head(8), use_container_width=True)

run = st.button("Analyze all", type="primary")
if not run:
    st.stop()

# ── Analyze (chunked batch inference) ────────────────────────────────────────
progress_bar = st.progress(0.0, text="Scoring texts…")
results: list[dict] = []
assert predictor is not None
CHUNK = 64
for start in range(0, len(texts), CHUNK):
    chunk = texts[start : start + CHUNK]
    try:
        chunk_results = predictor.predict(chunk)
        if isinstance(chunk_results, dict):  # single-item safety
            chunk_results = [chunk_results]
    except Exception:  # noqa: BLE001
        chunk_results = [{"label": "error", "confidence": 0.0} for _ in chunk]
    results.extend(chunk_results)
    done = min(len(texts), start + CHUNK)
    progress_bar.progress(done / len(texts), text=f"Scoring texts… {done}/{len(texts)}")
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
        {"label": "Mean confidence", "value": f"{avg_confidence:.1%}", "accent": COLORS["accent"]},
    ]
)

pos_pct = pos_count / total * 100
neg_pct = neg_count / total * 100
mix = {k: float(sentiment_counts.get(k, 0)) / total for k in ("negative", "neutral", "positive")}
majority = max(mix, key=mix.get)
lean = "positive" if pos_pct > neg_pct + 10 else "negative" if neg_pct > pos_pct + 10 else None
if majority == "neutral":
    trend = f"Mostly neutral, leaning {lean}" if lean else "Mostly neutral, balanced"
else:
    trend = f"Mostly {majority}"
trend_key = lean if (majority == "neutral" and lean) else majority
st.markdown(
    f"<div class='fs-verdict' style='margin-top:18px;'><div class='label'>Batch sentiment mix</div>"
    f"<div class='word' style='font-size:1.7rem;color:{get_sentiment_color(trend_key)};'>"
    f"{SENTIMENT_GLYPHS[trend_key]} {trend}</div>"
    f"<div class='conf'>{total:,} texts · mean model confidence <b class='fs-num'>{avg_confidence:.1%}</b> · "
    "'leaning' means one side leads the other by more than 10 points</div>"
    f"{probability_band(mix, majority)}</div>",
    unsafe_allow_html=True,
)

# ── Results + export ─────────────────────────────────────────────────────────
section_header("Detailed results")
st.dataframe(clean_df[[text_column, "Sentiment", "Confidence"]].head(200), use_container_width=True)

st.download_button(
    label="Download results (CSV)",
    data=clean_df.to_csv(index=False),
    file_name="sentiment_results.csv",
    mime="text/csv",
)

# ── Market outlook (template-based) ──────────────────────────────────────────
with st.expander("Generate market outlook report", expanded=False):
    st.caption("Template-based summary derived from the batch sentiment distribution.")
    if st.button("Generate outlook", key="generate_outlook"):
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

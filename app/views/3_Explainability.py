"""
Financial Sentiment Analyzer — Explainability Page.
Supports both TF-IDF word-importance analysis and SHAP-based explanations.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import plotly.graph_objects as go  # pyre-ignore
import streamlit as st  # pyre-ignore

from explain import explain_prediction_baseline, highlight_text  # pyre-ignore

# Try importing SHAP module (optional dependency)
try:
    from shap_explain import get_shap_explanation  # pyre-ignore

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import (  # pyre-ignore
    active_model,
    create_probability_chart,
    get_sentiment_color,
    page_header,
    section_header,
)

# ── Page config ─────────────────────────────────────────────────────────────────
selected_model, predictor = active_model()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
page_header(
    "Explainability",
    "Deconstruct a prediction to see exactly which words drove it — locally per prediction and globally with SHAP.",
)

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

if st.button("Explain prediction", type="primary", use_container_width=True):
    if explain_text.strip():
        try:
            with st.spinner("Analyzing..."):
                explanation = explain_prediction_baseline(explain_text, selected_model)
        except Exception as e:
            st.error(f"Explainability analysis failed: {e}")
            st.stop()

        # Show prediction
        section_header("Prediction Results")

        col1, col2 = st.columns([1, 1.2])

        with col1:
            sentiment = explanation["prediction"]
            color = get_sentiment_color(sentiment)
            st.markdown(
                f"""
            <div class='insight-card' style='text-align: center; display: flex; flex-direction: column; justify-content: center; height: 100%; border-top: 4px solid {color};'>
                <div style='color: #4B5D5A; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>Final Verdict</div>
                <div style='font-size: 3.5em; font-weight: 800; color: {color}; line-height: 1.2; text-transform: capitalize;'>
                    {sentiment}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            if explanation["probabilities"]:
                fig = create_probability_chart(explanation["probabilities"])
                st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

        # Highlighted text
        section_header("Text Analysis")
        st.markdown(
            "<p style='color: #4B5D5A;'>Words highlighted by their sentiment contribution magnitude:</p>",
            unsafe_allow_html=True,
        )

        highlighted = highlight_text(explain_text, explanation["word_importance"])
        st.markdown(
            f"<div class='insight-card' style='font-size: 1.1em; line-height: 1.8; color: #12211F;'>"
            f"{highlighted}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div style='margin-top: 15px; font-size: 0.9em; display: flex; gap: 15px;'>
            <div style='display: flex; align-items: center; gap: 5px;'><span style='display: inline-block; width: 12px; height: 12px; background: rgba(29,116,81,0.4); border-radius: 2px;'></span> <span style='color: #4B5D5A;'>Positive Driver</span></div>
            <div style='display: flex; align-items: center; gap: 5px;'><span style='display: inline-block; width: 12px; height: 12px; background: rgba(163,50,43,0.4); border-radius: 2px;'></span> <span style='color: #4B5D5A;'>Negative Driver</span></div>
            <div style='display: flex; align-items: center; gap: 5px;'><span style='display: inline-block; width: 12px; height: 12px; background: rgba(14,110,106,0.4); border-radius: 2px;'></span> <span style='color: #4B5D5A;'>Neutral / Uncertain</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Word importance
        section_header("Extracted Features")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h4 style='color: #1D7451; margin-top: 0;'>Positive Drivers</h4>", unsafe_allow_html=True)
            if explanation.get("lexicon_positive"):
                st.markdown(
                    "<div style='font-size: 0.85em; color: #4B5D5A; margin-bottom: 5px; text-transform: uppercase;'>Lexicon Matches</div>",
                    unsafe_allow_html=True,
                )
                words_html = "".join(
                    [
                        f"<span class='chip positive' style='margin-right: 6px; margin-bottom: 6px; display: inline-block;'>{w}</span>"
                        for w in explanation["lexicon_positive"][:5]
                    ]
                )
                st.markdown(f"<div>{words_html}</div>", unsafe_allow_html=True)
            if explanation["positive_words"]:
                st.markdown(
                    "<div style='font-size: 0.85em; color: #4B5D5A; margin: 10px 0 5px 0; text-transform: uppercase;'>Top Model Weights</div>",
                    unsafe_allow_html=True,
                )
                for word, score in explanation["positive_words"]:
                    st.markdown(
                        f"<div style='display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #D5DEDC; color: #2E3F3C;'><span style='font-variant-numeric: tabular-nums;'>{word}</span> <span>{score:.3f}</span></div>",
                        unsafe_allow_html=True,
                    )
            if not explanation.get("lexicon_positive") and not explanation["positive_words"]:
                st.markdown(
                    "<span style='color: #5E6E6A; font-style: italic;'>None detected</span>", unsafe_allow_html=True
                )

        with col2:
            st.markdown("<h4 style='color: #A3322B; margin-top: 0;'>Negative Drivers</h4>", unsafe_allow_html=True)
            if explanation.get("lexicon_negative"):
                st.markdown(
                    "<div style='font-size: 0.85em; color: #4B5D5A; margin-bottom: 5px; text-transform: uppercase;'>Lexicon Matches</div>",
                    unsafe_allow_html=True,
                )
                words_html = "".join(
                    [
                        f"<span class='chip negative' style='margin-right: 6px; margin-bottom: 6px; display: inline-block;'>{w}</span>"
                        for w in explanation["lexicon_negative"][:5]
                    ]
                )
                st.markdown(f"<div>{words_html}</div>", unsafe_allow_html=True)
            if explanation["negative_words"]:
                st.markdown(
                    "<div style='font-size: 0.85em; color: #4B5D5A; margin: 10px 0 5px 0; text-transform: uppercase;'>Top Model Weights</div>",
                    unsafe_allow_html=True,
                )
                for word, score in explanation["negative_words"]:
                    st.markdown(
                        f"<div style='display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #D5DEDC; color: #2E3F3C;'><span style='font-variant-numeric: tabular-nums;'>{word}</span> <span>{score:.3f}</span></div>",
                        unsafe_allow_html=True,
                    )
            if not explanation.get("lexicon_negative") and not explanation["negative_words"]:
                st.markdown(
                    "<span style='color: #5E6E6A; font-style: italic;'>None detected</span>", unsafe_allow_html=True
                )

        with col3:
            st.markdown("<h4 style='color: #0E6E6A; margin-top: 0;'>Neutral / Uncertain</h4>", unsafe_allow_html=True)
            if explanation.get("lexicon_uncertainty"):
                st.markdown(
                    "<div style='font-size: 0.85em; color: #4B5D5A; margin-bottom: 5px; text-transform: uppercase;'>Lexicon Matches</div>",
                    unsafe_allow_html=True,
                )
                words_html = "".join(
                    [
                        f"<span class='chip neutral' style='margin-right: 6px; margin-bottom: 6px; display: inline-block;'>{w}</span>"
                        for w in explanation["lexicon_uncertainty"][:5]
                    ]
                )
                st.markdown(f"<div>{words_html}</div>", unsafe_allow_html=True)
            if explanation["neutral_words"]:
                st.markdown(
                    "<div style='font-size: 0.85em; color: #4B5D5A; margin: 10px 0 5px 0; text-transform: uppercase;'>Top Model Weights</div>",
                    unsafe_allow_html=True,
                )
                for word, score in explanation["neutral_words"]:
                    st.markdown(
                        f"<div style='display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #D5DEDC; color: #2E3F3C;'><span style='font-variant-numeric: tabular-nums;'>{word}</span> <span>{score:.3f}</span></div>",
                        unsafe_allow_html=True,
                    )
            if not explanation.get("lexicon_uncertainty") and not explanation["neutral_words"]:
                st.markdown(
                    "<span style='color: #5E6E6A; font-style: italic;'>None detected</span>", unsafe_allow_html=True
                )

        # Top words chart
        if explanation["word_importance"]:
            section_header("Word Importance Scores")
            top_words = explanation["word_importance"][:10]
            words = [w for w, s, d in top_words]
            scores = [s for w, s, d in top_words]
            directions = [d for w, s, d in top_words]
            colors = [get_sentiment_color(d) for d in directions]

            fig = go.Figure(
                go.Bar(
                    x=scores,
                    y=words,
                    orientation="h",
                    marker_color=colors,
                )
            )
            fig.update_layout(
                title={"text": "Top 10 High Impact Words", "font": {"color": "#12211F", "size": 16}},
                xaxis_title="Influence Score",
                yaxis={"autorange": "reversed"},
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#2E3F3C", "family": "Public Sans, system-ui, sans-serif"},
                xaxis={"showgrid": True, "gridcolor": "#D5DEDC", "linecolor": "#B9C6C3"},
                yaxis_tickfont={"size": 13},
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.warning("Please enter text to explain.")

# ── SHAP Global Explainability Section ──────────────────────────────────────────
section_header("SHAP Global Explainability", "Which features matter across the whole dataset, not just one prediction")

# Check for pre-computed SHAP results
results_dir = Path(__file__).parent.parent.parent / "results"
classifier_name = selected_model.replace("baseline_", "")
shap_results_path = results_dir / f"shap_{classifier_name}.json"

shap_data = None
if shap_results_path.exists():
    with open(shap_results_path) as f:
        shap_data = json.load(f)
    st.success(f"Loaded pre-computed SHAP results for **{classifier_name}**")
elif SHAP_AVAILABLE:
    st.info(f"No pre-computed SHAP results for **{classifier_name}**. Click below to generate them (may take ~30s).")
    if st.button("Run SHAP analysis", type="primary", use_container_width=True):
        with st.spinner("Computing SHAP values... this may take 30-60 seconds"):
            try:
                shap_data = get_shap_explanation(classifier_name, max_samples=100)
                # Auto-save
                results_dir.mkdir(parents=True, exist_ok=True)
                with open(shap_results_path, "w") as f:
                    json.dump(shap_data, f, indent=2)
                st.success("SHAP analysis complete and saved!")
            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")
else:
    st.warning("**SHAP** library is not installed. Install with: `pip install shap`")

if shap_data:
    top_features = shap_data.get("top_features", [])[:20]
    if top_features:
        features = [f["feature"] for f in top_features]
        importances = [f["importance"] for f in top_features]

        fig = go.Figure(
            go.Bar(
                x=importances,
                y=features,
                orientation="h",
                marker_color="#45627A",
                marker_line_color="rgba(69,98,122,0.6)",
                marker_line_width=1,
            )
        )
        fig.update_layout(
            title={
                "text": f"SHAP Feature Importance — {classifier_name.replace('_', ' ').title()}",
                "font": {"color": "#12211F", "size": 16},
            },
            xaxis_title="Mean |SHAP Value|",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#2E3F3C", "family": "Public Sans, system-ui, sans-serif"},
            yaxis={"autorange": "reversed"},
            xaxis={"showgrid": True, "gridcolor": "#D5DEDC"},
            yaxis_tickfont={"size": 12},
            margin=dict(l=150),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    # Per-class drivers
    class_features = shap_data.get("class_features", {})
    if class_features:
        section_header("Per-Class Feature Drivers")
        ccols = st.columns(3)
        class_colors = {"negative": "#A3322B", "neutral": "#5E6E6A", "positive": "#1D7451"}
        for i, (cls_name, cls_feats) in enumerate(class_features.items()):
            with ccols[i]:
                color = class_colors.get(cls_name, "#4B5D5A")
                st.markdown(
                    f"""
                    <h4 style='color: {color}; text-transform: capitalize; margin-top: 0;'>{cls_name}</h4>
                """,
                    unsafe_allow_html=True,
                )
                for feat in cls_feats[:7]:
                    icon = "↑" if feat["direction"] == "positive" else "↓"
                    st.markdown(
                        f"""
                    <div style='display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #D5DEDC; font-size: 0.9em;'>
                        <span style='font-variant-numeric: tabular-nums; color: #2E3F3C;'>{feat["feature"]}</span>
                        <span style='color: {color};'>{icon} {feat["shap_value"]:.3f}</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

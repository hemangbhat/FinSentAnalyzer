"""
Financial Sentiment Analyzer — Explainability Page.

Local: which terms in one sentence drove the verdict (TF-IDF weight × coefficient).
Global: SHAP feature importance across the dataset.
"""

import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import plotly.graph_objects as go  # pyre-ignore
import streamlit as st  # pyre-ignore

from explain import explain_prediction_baseline  # pyre-ignore

try:
    from shap_explain import get_shap_explanation  # pyre-ignore

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import (  # pyre-ignore
    COLORS,
    SENTIMENT_GLYPHS,
    SENTIMENT_TINTS,
    active_model,
    empty_state,
    exhibit,
    get_sentiment_color,
    model_label,
    page_header,
    render_chart,
    section_header,
    sentiment_tag,
    source_note,
    status_banner,
    style_fig,
    term_table,
    verdict_card,
)

EXAMPLES = {
    "Profit EUR 8.7m → 13.1m": "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn.",
    "Shares fall on earnings": "Stock prices fell sharply after a disappointing earnings report.",
    "Plant closure, 300 jobs cut": "The company will close its Tampere plant, cutting 300 jobs.",
}

selected_model, predictor = active_model()

page_header(
    "Explainability",
    "Which words drove one verdict, and which features matter across the whole dataset.",
)

if not selected_model.startswith("baseline_"):
    status_banner(
        f"{model_label(selected_model)} reads context rather than a weighted vocabulary, so per-term attribution "
        "and SHAP are computed for the TF-IDF models. Choose one in the sidebar.",
        kind="info",
    )
    st.stop()


def _use_example() -> None:
    choice = st.session_state.get("ex_example")
    if choice:
        st.session_state.explain_text = EXAMPLES[choice]
        st.session_state.ex_run = True


def _highlight(text: str, word_importance) -> str:
    """Tint each term by the class it pushes toward; the underline carries the class without colour."""
    weights = {}
    for term, _w, cls in word_importance[:15]:
        if " " not in term:
            weights.setdefault(term.lower(), cls)
    out = []
    for token in re.findall(r"\w[\w'.%-]*|\s+|[^\w\s]", text):
        cls = weights.get(token.lower())
        if cls:
            out.append(
                f"<mark style='background:{SENTIMENT_TINTS[cls]};color:{COLORS['text']};padding:1px 3px;"
                f"border-radius:3px;text-decoration:underline;text-decoration-color:{get_sentiment_color(cls)};"
                f"text-decoration-thickness:2px;text-underline-offset:4px;' title='pushes toward {cls}'>"
                f"{html.escape(token)}</mark>"
            )
        else:
            out.append(html.escape(token))
    return "".join(out)


st.pills("Try an example", list(EXAMPLES), key="ex_example", on_change=_use_example)
explain_text = st.text_area(
    "Financial text",
    height=100,
    placeholder="e.g., Operating profit rose to EUR 13.1 mn from EUR 8.7 mn.",
    key="explain_text",
)
run = st.button("Explain prediction", type="primary") or st.session_state.pop("ex_run", False)

if run and not explain_text.strip():
    status_banner("Enter some financial text to explain.", kind="warning")
elif run:
    try:
        with st.spinner("Explaining…"):
            explanation = explain_prediction_baseline(explain_text, selected_model)
    except Exception as e:  # noqa: BLE001
        status_banner(f"Explanation failed: {e}", kind="error")
        st.stop()

    section_header("This prediction", model_label(selected_model))
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        probs = explanation.get("probabilities") or {}
        verdict_card(explanation["prediction"], max(probs.values()) if probs else None, probs or None)
    with c2:
        st.markdown(
            "<div class='fs-verdict'><div class='label'>The sentence, by contribution</div>"
            "<p style='font-size:1.08rem;line-height:1.9;margin:10px 0 6px 0;'>"
            f"{_highlight(explain_text, explanation['word_importance'])}</p>"
            "<div style='font-size:0.82rem;color:#4B5D5A;'>Highlighted terms push toward: "
            f"{sentiment_tag('positive')} {sentiment_tag('neutral')} {sentiment_tag('negative')}</div></div>",
            unsafe_allow_html=True,
        )

    exhibit("Term attribution", "TF-IDF weight × classifier coefficient for each term in the sentence.")
    st.markdown(term_table(explanation["word_importance"][:10]), unsafe_allow_html=True)
    lex = [(c, w) for c in ("positive", "negative") for w in explanation.get(f"lexicon_{c}", [])]
    if lex:
        st.markdown(
            "<p style='margin-top:10px;'>Loughran-McDonald lexicon matches: "
            + " ".join(f"{sentiment_tag(c)} <b>{html.escape(w)}</b>" for c, w in lex)
            + "</p>",
            unsafe_allow_html=True,
        )
    source_note(f"models/{selected_model}.joblib; Loughran-McDonald financial sentiment lexicon.")
else:
    empty_state(
        "Pick an example or paste a sentence",
        "You'll see the verdict with its probability band, the sentence tinted by which terms pushed it, "
        "and a table of every term's contribution.",
        icon="search",
    )

# ── Global: SHAP ─────────────────────────────────────────────────────────────
classifier_name = selected_model.replace("baseline_", "")
results_dir = Path(__file__).parent.parent.parent / "results"
shap_path = results_dir / f"shap_{classifier_name}.json"

shap_data = None
if shap_path.exists():
    try:
        shap_data = json.loads(shap_path.read_text())
    except Exception:  # noqa: BLE001  (e.g. an un-fetched Git LFS pointer)
        shap_data = None

if shap_data and shap_data.get("top_features"):
    top = shap_data["top_features"][:20]
    exhibit(
        f"Global feature importance, {model_label(selected_model)}",
        "Mean absolute SHAP value across the dataset. Several of the strongest features are function words, "
        "a sign the model learned PhraseBank's phrasing rather than sentiment alone.",
    )
    fig = go.Figure(
        go.Bar(
            x=[f["importance"] for f in top][::-1],
            y=[f["feature"] for f in top][::-1],
            orientation="h",
            marker_color=COLORS["accent"],
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    style_fig(fig, height=520, legend=False)
    fig.update_layout(bargap=0.35, xaxis_title="Mean |SHAP value|")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=False, ticks="")
    render_chart(fig)
    source_note(f"results/shap_{classifier_name}.json (pre-computed with src/shap_explain.py).")

    class_features = shap_data.get("class_features", {})
    if class_features:
        section_header(
            "Per-class drivers",
            "Signed SHAP value: positive pushes a sentence toward the class, negative pushes it away.",
        )
        cols = st.columns(3, gap="large")
        for col, cls in zip(cols, ("negative", "neutral", "positive")):
            feats = class_features.get(cls, [])[:7]
            with col:
                body = "".join(
                    f"<tr><td class='t'>{html.escape(f['feature'])}</td>"
                    f"<td class='n fs-num' style='color:{COLORS['primary'] if f['shap_value'] >= 0 else COLORS['negative']};'>"
                    f"{f['shap_value']:+.3f}</td></tr>"
                    for f in feats
                )
                st.markdown(
                    f"<table class='fs-terms'><thead><tr><th class='t' style='color:{get_sentiment_color(cls)};'>"
                    f"{SENTIMENT_GLYPHS[cls]} {cls.capitalize()}</th><th style='text-align:right;'>SHAP</th></tr>"
                    f"</thead><tbody>{body}</tbody></table>",
                    unsafe_allow_html=True,
                )
elif SHAP_AVAILABLE:
    section_header("Global feature importance")
    empty_state(
        f"No SHAP results for {model_label(selected_model)} yet",
        "Computing them takes 30–60 seconds and saves to results/ for next time.",
        icon="chart",
    )
    if st.button("Compute SHAP values"):
        with st.spinner("Computing SHAP values…"):
            try:
                data = get_shap_explanation(classifier_name, max_samples=100)
                results_dir.mkdir(parents=True, exist_ok=True)
                shap_path.write_text(json.dumps(data, indent=2))
                st.rerun()
            except Exception as e:  # noqa: BLE001
                status_banner(f"SHAP computation failed: {e}", kind="error")
else:
    section_header("Global feature importance")
    status_banner("SHAP is not installed in this environment (<code>pip install shap</code>).", kind="info")

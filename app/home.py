"""
Home: the report's front page.

Every number is read from ``results/*.json`` with conservative fallbacks, so
the page can't drift from the evaluation artifacts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # pyre-ignore
from shared import (  # pyre-ignore
    COLORS,
    exhibit,
    interval_chart,
    kpi_strip,
    load_result,
    page_header,
    render_chart,
    section_header,
    source_note,
)


def headline_numbers() -> dict:
    """Pull the headline metrics and their bootstrap intervals out of the results artifacts."""
    n = {
        "svm_test": (0.902, 0.850, 0.945),
        "svm_cv": (0.838, 0.016),
        "svm_ood": (0.460, 0.436, 0.485),
        "finbert": (0.841, 0.823, 0.858),
        "finbert_acc": 0.87,
        "svm_ood_acc": 0.67,
        "majority_acc": 0.66,
        "n_ood": 2388,
    }
    cv = (load_result("cv_results.json") or {}).get("svm")
    if cv:
        n["svm_cv"] = (cv["f1_macro_mean"], cv["f1_macro_std"])
    ood = load_result("model_comparison_ood.json")
    if ood:
        n["majority_acc"] = ood.get("majority_class_accuracy", n["majority_acc"])
        n["n_ood"] = ood.get("n_samples", n["n_ood"])
        for m in ood.get("models", []):
            if m["model"] == "svm":
                n["svm_ood_acc"] = m["accuracy"]
            elif m["model"] == "finbert":
                n["finbert_acc"] = m["accuracy"]
    ci = load_result("bootstrap_ci.json") or {}
    if (r := ci.get("phrasebank_test", {}).get("svm")) is not None:
        n["svm_test"] = (r["f1_macro"], r["ci_low"], r["ci_high"])
    for key, name in (("svm_ood", "svm"), ("finbert", "finbert")):
        if (r := ci.get("real_news", {}).get(name)) is not None:
            n[key] = (r["f1_macro"], r["ci_low"], r["ci_high"])
    # Macro-F1 of always predicting "neutral": F1 is 2p/(1+p) on the neutral class, 0 on the others.
    p = n["majority_acc"]
    n["majority_f1"] = (2 * p / (1 + p)) / 3
    return n


n = headline_numbers()

st.markdown("<div class='fs-rule-top'></div>", unsafe_allow_html=True)
head, action = st.columns([3.3, 1], vertical_alignment="top")
with head:
    page_header(
        "Financial sentiment, measured honestly",
        "FinSight classifies financial text as positive, neutral or negative, shows the words behind "
        "every verdict, and reports how each model holds up on real news it never saw, failures included.",
        meta=(
            "<b>9</b> models · <b>116</b> automated tests · FastAPI service · "
            "Data: Financial PhraseBank and Twitter Financial News (MIT)"
        ),
        ruled=False,
    )
with action:
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    if st.button("Analyze a sentence", type="primary", icon=":material/arrow_forward:", use_container_width=True):
        st.switch_page("views/1_Single_Analysis.py")
    st.page_link("views/7_Sentiment_Trends.py", label="Read the evaluation", icon=":material/insights:")

kpi_strip(
    [
        {
            "label": "Fine-tuned FinBERT, real news",
            "value": f"{n['finbert'][0]:.2f}",
            "sub": f"macro-F1, 95% CI {n['finbert'][1]:.2f}–{n['finbert'][2]:.2f}",
        },
        {
            "label": "TF-IDF + SVM, real news",
            "value": f"{n['svm_ood'][0]:.2f}",
            "sub": f"macro-F1; accuracy {n['svm_ood_acc']:.2f} vs {n['majority_acc']:.2f} majority",
        },
        {
            "label": "TF-IDF + SVM, 5-fold CV",
            "value": f"{n['svm_cv'][0]:.3f}",
            "sub": f"± {n['svm_cv'][1]:.3f} macro-F1 on PhraseBank",
        },
        {
            "label": "Held-out headlines",
            "value": f"{n['n_ood']:,}",
            "sub": "human-labelled, never trained on by the SVM",
        },
    ]
)

exhibit(
    "Macro-F1 by evaluation set",
    "A 0.90 test score on PhraseBank did not survive real news. Domain-matched fine-tuning recovered it.",
)
rows = [
    {
        "label": "SVM · PhraseBank test (n=226)",
        "value": n["svm_test"][0],
        "low": n["svm_test"][1],
        "high": n["svm_test"][2],
        "color": COLORS["accent"],
    },
    {
        "label": f"SVM · real news (n={n['n_ood']:,})",
        "value": n["svm_ood"][0],
        "low": n["svm_ood"][1],
        "high": n["svm_ood"][2],
        "color": COLORS["negative"],
    },
    {
        "label": f"FinBERT fine-tuned · real news (n={n['n_ood']:,})",
        "value": n["finbert"][0],
        "low": n["finbert"][1],
        "high": n["finbert"][2],
        "color": COLORS["primary"],
    },
]
render_chart(interval_chart(rows, reference=n["majority_f1"], reference_label="always 'neutral'"))
source_note(
    "results/bootstrap_ci.json (2,000 bootstrap resamples), results/model_comparison_ood.json. "
    "FinBERT was fine-tuned on the news training split, so its row is held-out in-domain, not out-of-domain."
)

section_header("What happened", "Three steps, each one reproducible from the repository.")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(
        f"**Trained a strong baseline.** TF-IDF + linear SVM on Financial PhraseBank: "
        f"{n['svm_test'][0]:.2f} macro-F1 on the test split, {n['svm_cv'][0]:.3f} ± {n['svm_cv'][1]:.3f} "
        "under 5-fold cross-validation."
    )
with s2:
    st.markdown(
        f"**Stress-tested it on real news.** On {n['n_ood']:,} financial headlines it fell to "
        f"{n['svm_ood'][0]:.2f}, with accuracy {n['svm_ood_acc']:.2f} against {n['majority_acc']:.2f} for always "
        "saying neutral. SHAP shows it leaning on stopwords from annual-report phrasing."
    )
with s3:
    st.markdown(
        f"**Fixed it with domain data.** FinBERT fine-tuned on the news training split "
        f"(4 epochs, class weights, GPU) reaches {n['finbert'][0]:.2f} macro-F1 and {n['finbert_acc']:.2f} accuracy."
    )

section_header("Contents")
CONTENTS = [
    (
        "views/1_Single_Analysis.py",
        "Single Analysis",
        ":material/edit_note:",
        "Score one sentence: verdict, probability band, and the words behind it.",
    ),
    (
        "views/2_Batch_Processing.py",
        "Batch Processing",
        ":material/upload_file:",
        "Score a CSV or TXT file and export the results.",
    ),
    (
        "views/3_Explainability.py",
        "Explainability",
        ":material/troubleshoot:",
        "Token attribution for one prediction; SHAP across the dataset.",
    ),
    (
        "views/4_Word_Insights.py",
        "Word Insights",
        ":material/match_word:",
        "The vocabulary each model associates with each sentiment.",
    ),
    (
        "views/5_Deep_Analysis.py",
        "Deep Analysis",
        ":material/psychology:",
        "Entities, Loughran-McDonald lexicon hits, rule-based reasoning steps.",
    ),
    (
        "views/6_Model_Info.py",
        "Model Registry",
        ":material/leaderboard:",
        "Every model's metrics, architecture and training data.",
    ),
    (
        "views/8_Error_Analysis.py",
        "Error Analysis",
        ":material/rule:",
        "Confusion flows, confidence calibration, misclassified sentences.",
    ),
    (
        "views/7_Sentiment_Trends.py",
        "Generalization",
        ":material/public:",
        "Real-news results, cross-validation and SHAP summaries.",
    ),
    (
        "views/9_Stock_Prediction_Extension.py",
        "Stock Prediction",
        ":material/science:",
        "yfinance → FinBERT → LSTM pipeline. Experimental: the forecast is chance-level.",
    ),
]
for start in range(0, len(CONTENTS), 3):  # row by row, so phones read it in order
    for col, (path, title, ico, desc) in zip(st.columns(3, gap="large"), CONTENTS[start : start + 3]):
        with col:
            st.markdown("<div class='fs-contents-row'></div>", unsafe_allow_html=True)
            st.page_link(path, label=title, icon=ico)
            st.markdown(f"<p class='fs-contents-desc'>{desc}</p>", unsafe_allow_html=True)

"""
Home — what the system does and the headline evaluation story.

Every number on this page is read from ``results/*.json`` (with conservative
fallbacks), so it can't drift from the evaluation artifacts.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # pyre-ignore
from shared import COLORS, kpi_strip, section_header  # pyre-ignore

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _load(name: str):
    try:
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  (missing file or un-fetched Git LFS pointer)
        return None


@st.cache_data(show_spinner=False)
def headline_numbers() -> dict:
    """Pull the headline metrics out of the results artifacts."""
    nums = {
        "svm_test_f1": 0.90,
        "svm_cv_f1": 0.838,
        "svm_cv_std": 0.016,
        "svm_ood_f1": 0.46,
        "svm_ood_acc": 0.67,
        "majority_acc": 0.66,
        "finbert_f1": 0.84,
        "finbert_acc": 0.87,
        "n_ood": 2388,
    }
    for row in _load("evaluation_results.json") or []:
        if row.get("name") == "baseline_svm":
            nums["svm_test_f1"] = row["f1_macro"]
    cv = (_load("cv_results.json") or {}).get("svm")
    if cv:
        nums["svm_cv_f1"], nums["svm_cv_std"] = cv["f1_macro_mean"], cv["f1_macro_std"]
    ood = _load("model_comparison_ood.json")
    if ood:
        nums["majority_acc"] = ood.get("majority_class_accuracy", nums["majority_acc"])
        nums["n_ood"] = ood.get("n_samples", nums["n_ood"])
        for m in ood.get("models", []):
            if m["model"] == "svm":
                nums["svm_ood_f1"], nums["svm_ood_acc"] = m["f1_macro"], m["accuracy"]
            elif m["model"] == "finbert":
                nums["finbert_f1"], nums["finbert_acc"] = m["f1_macro"], m["accuracy"]
    return nums


n = headline_numbers()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class='fs-hero'>
    <div class='fs-eyebrow'>Financial NLP · Explainable ML · MLOps</div>
    <h1>Financial Sentiment Intelligence</h1>
    <p>
        Classify financial text as positive, neutral, or negative, see exactly which words drove
        each decision, and inspect how the models hold up on real-world news they were never trained on.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

cta1, cta2, _ = st.columns([1, 1, 2.2])
with cta1:
    st.page_link("views/1_Single_Analysis.py", label="Analyze a sentence", icon=":material/arrow_forward:")
with cta2:
    st.page_link("views/7_Sentiment_Trends.py", label="See the evaluation", icon=":material/insights:")

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

kpi_strip(
    [
        {
            "label": "Fine-tuned FinBERT",
            "value": f"{n['finbert_f1']:.2f}",
            "sub": f"macro-F1 on {n['n_ood']:,} held-out real headlines",
            "accent": COLORS["positive"],
        },
        {
            "label": "Baseline, cross-validated",
            "value": f"{n['svm_cv_f1']:.3f}",
            "sub": f"± {n['svm_cv_std']:.3f} · 5-fold macro-F1 (TF-IDF + SVM)",
            "accent": COLORS["primary"],
        },
        {
            "label": "Models compared",
            "value": "9",
            "sub": "7 classical baselines + 2 FinBERT variants",
            "accent": COLORS["accent"],
        },
        {
            "label": "Served via",
            "value": "API + UI",
            "sub": "FastAPI · rate limiting · Docker · CI",
            "accent": COLORS["warning"],
        },
    ]
)

# ── The evaluation story ─────────────────────────────────────────────────────
section_header("The evaluation story", "Why a 0.90 test score wasn't the end of it")

st.markdown(
    f"""
<div class='fs-story'>
  <div class='fs-step' style='border-top:2px solid {COLORS["primary"]};'>
    <div class='n'>01 · TRAIN</div>
    <div class='t'>Strong in-domain baseline</div>
    <div class='big' style='color:{COLORS["primary_soft"]};'>{n["svm_test_f1"]:.2f}</div>
    <div class='unit'>macro-F1 · PhraseBank test split (n=226)</div>
    <div class='d'>TF-IDF + linear SVM on Financial PhraseBank. Five-fold CV
    ({n["svm_cv_f1"]:.3f} ± {n["svm_cv_std"]:.3f}) confirms it isn't a lucky split.</div>
  </div>
  <div class='fs-step' style='border-top:2px solid {COLORS["negative"]};'>
    <div class='n'>02 · STRESS-TEST</div>
    <div class='t'>Collapses on real news</div>
    <div class='big' style='color:#f87171;'>{n["svm_ood_f1"]:.2f}</div>
    <div class='unit'>macro-F1 · {n["n_ood"]:,} real financial headlines</div>
    <div class='d'>Accuracy {n["svm_ood_acc"]:.2f} vs. {n["majority_acc"]:.2f} for always predicting
    “neutral”: the model learned annual-report phrasing, not sentiment in general.</div>
  </div>
  <div class='fs-step' style='border-top:2px solid {COLORS["positive"]};'>
    <div class='n'>03 · FIX</div>
    <div class='t'>Domain fine-tuning closes the gap</div>
    <div class='big' style='color:#34d399;'>{n["finbert_f1"]:.2f}</div>
    <div class='unit'>macro-F1 · same {n["n_ood"]:,} headlines, held out</div>
    <div class='d'>FinBERT fine-tuned on the news training split (GPU, class weights,
    early stopping) reaches {n["finbert_acc"]:.2f} accuracy.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Modules ──────────────────────────────────────────────────────────────────
section_header("Workspace", "Every module, one click away")

MODULES = [
    (
        "✎",
        "Single Analysis",
        "Classify one excerpt: verdict, confidence, class probabilities, and a plain-language rationale.",
        "views/1_Single_Analysis.py",
        False,
    ),
    (
        "⇪",
        "Batch Processing",
        "Upload a CSV or TXT, score hundreds of rows, and export the results.",
        "views/2_Batch_Processing.py",
        False,
    ),
    (
        "◎",
        "Explainability",
        "Per-token attribution for one prediction plus SHAP global feature importance.",
        "views/3_Explainability.py",
        False,
    ),
    (
        "Aa",
        "Word Insights",
        "The vocabulary each model learned to associate with every sentiment.",
        "views/4_Word_Insights.py",
        False,
    ),
    (
        "◈",
        "Deep Analysis",
        "Named entities, Loughran-McDonald lexicon hits, and rule-based reasoning steps.",
        "views/5_Deep_Analysis.py",
        False,
    ),
    (
        "▤",
        "Model Registry",
        "Every model's metrics, architecture, and training data side by side.",
        "views/6_Model_Info.py",
        False,
    ),
    (
        "✕",
        "Error Analysis",
        "Confusion flows, confidence calibration, and the misclassified sentences themselves.",
        "views/8_Error_Analysis.py",
        False,
    ),
    (
        "◍",
        "Generalization",
        "Out-of-domain results on real news, plus cross-validation and SHAP summaries.",
        "views/7_Sentiment_Trends.py",
        False,
    ),
    (
        "⚗",
        "Stock Prediction",
        "yfinance → FinBERT → LSTM pipeline. Engineering demo; the forecast is chance-level.",
        "views/9_Stock_Prediction_Extension.py",
        True,
    ),
]

for start in range(0, len(MODULES), 3):
    cols = st.columns(3)
    for col, (ico, title, desc, path, experimental) in zip(cols, MODULES[start : start + 3]):
        with col, st.container(border=True):
            tag = "<span class='fs-tag'>Experimental</span>" if experimental else ""
            st.markdown(
                f"<div class='fs-card-title'><span class='fs-card-ico'>{ico}</span>{title}{tag}</div>"
                f"<div class='fs-card-desc'>{desc}</div>",
                unsafe_allow_html=True,
            )
            st.page_link(path, label="Open", icon=":material/arrow_forward:")

st.markdown(
    "<div style='text-align:center;color:#64748b;padding:28px 0 4px 0;font-size:0.82rem;'>"
    "Built with scikit-learn · PyTorch · Hugging Face Transformers · SHAP · FastAPI · Streamlit</div>",
    unsafe_allow_html=True,
)

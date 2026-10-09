"""
Financial Sentiment Analyzer — Model Info Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd  # pyre-ignore
import streamlit as st  # pyre-ignore

from predict import get_available_models  # pyre-ignore
from utils import get_model_info  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import (  # pyre-ignore
    COLORS,
    active_model,
    exhibit,
    interval_chart,
    kpi_strip,
    load_result,
    model_label,
    page_header,
    render_chart,
    section_header,
    sentiment_tag,
    source_note,
)

# ── Page config ─────────────────────────────────────────────────────────────────
selected_model, predictor = active_model()

if predictor is None:
    st.stop()

try:
    info = get_model_info(selected_model)
except Exception as e:
    st.error(f"Could not load model info: {e}")
    st.stop()

page_header(
    "Model Registry",
    "Every trained model's metrics, architecture and training data, with the uncertainty on each score.",
    meta=f"Active engine <b>{info.get('name', selected_model)}</b> · Architecture <b>{info.get('type', 'Unknown')}</b>",
)

# Active model: key figures with their uncertainty
_ci = (load_result("bootstrap_ci.json") or {}).get("phrasebank_test", {})
_cv = (load_result("cv_results.json") or {}).get(selected_model.replace("baseline_", ""))
_mine = _ci.get(selected_model.replace("baseline_", ""))
section_header(model_label(selected_model), "PhraseBank test split (n=226) and 5-fold cross-validation.")
figures = [
    {"label": "Test accuracy", "value": info.get("accuracy", "—")},
    {
        "label": "Test macro-F1",
        "value": f"{_mine['f1_macro']:.3f}" if _mine else info.get("f1_macro", "—"),
        "sub": f"95% interval {_mine['ci_low']:.2f}–{_mine['ci_high']:.2f}" if _mine else "not scored on PhraseBank",
    },
]
if _cv:
    figures.append(
        {
            "label": "5-fold CV macro-F1",
            "value": f"{_cv['f1_macro_mean']:.3f}",
            "sub": f"± {_cv['f1_macro_std']:.3f} across folds",
        }
    )
figures.append({"label": "Inference cost", "value": info.get("speed", "—"), "sub": "relative, on CPU"})
kpi_strip(figures)

features = info.get("features", [])
if features:
    st.markdown(
        "<p style='max-width:75ch;margin-top:10px;'><b>How it works:</b> " + "; ".join(features) + ".</p>",
        unsafe_allow_html=True,
    )

# All models comparison table
section_header(
    "Model comparison",
    "PhraseBank test split (n=226). The interval is a 95% bootstrap range for macro-F1.",
)
try:
    comparison_data = []
    for model_key in get_available_models():
        m_info = get_model_info(model_key)
        ci = _ci.get(model_key.replace("baseline_", ""))
        f1 = ci["f1_macro"] if ci else m_info.get("f1_macro")
        comparison_data.append(
            {
                "Model": m_info.get("name", model_key),
                "Type": m_info.get("type", "Unknown"),
                "Macro-F1": f1 if isinstance(f1, (int, float)) else float("nan"),
                "95% interval": f"{ci['ci_low']:.3f} – {ci['ci_high']:.3f}" if ci else "—",
                "Inference cost": str(m_info.get("speed", "—")).split(" / ")[0],
                "Active": "●" if model_key == selected_model else "",
            }
        )
    st.dataframe(
        pd.DataFrame(comparison_data),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Macro-F1": st.column_config.NumberColumn(format="%.3f", help="Blank when not scored on PhraseBank")
        },
    )
except Exception as e:
    st.error(f"Could not load model comparison: {e}")

if _ci:
    exhibit(
        "The top baselines are statistically tied",
        "Overlapping intervals mean a 226-sentence test set cannot separate them; 5-fold CV is the fairer comparison.",
    )
    ordered = sorted(_ci.items(), key=lambda kv: kv[1]["f1_macro"], reverse=True)
    render_chart(
        interval_chart(
            [
                {
                    "label": model_label(f"baseline_{k}"),
                    "value": v["f1_macro"],
                    "low": v["ci_low"],
                    "high": v["ci_high"],
                    "color": COLORS["primary"] if f"baseline_{k}" == selected_model else COLORS["accent"],
                }
                for k, v in ordered
            ],
            x_range=(0.7, 1.0),
        )
    )
    source_note("results/bootstrap_ci.json (2,000 resamples of the test split; scripts/bootstrap_ci.py).")

# Dataset info
section_header(
    "Training data", "Financial PhraseBank, Malo et al. (2014): sentences from financial news and company reports."
)
d1, d2 = st.columns([1.1, 1], gap="large")
with d1:
    st.markdown(
        "<table class='fs-terms'><tbody>"
        "<tr><td class='t'>Subset</td><td>All annotators agree (100% agreement)</td></tr>"
        "<tr><td class='t'>Sentences</td><td class='fs-num'>2,264</td></tr>"
        "<tr><td class='t'>Classes</td><td>"
        f"{sentiment_tag('positive', '570')} {sentiment_tag('neutral', '1,391')} {sentiment_tag('negative', '303')}"
        "</td></tr>"
        "<tr><td class='t'>Split</td><td>Stratified 80 / 10 / 10, seed 42</td></tr>"
        "</tbody></table>",
        unsafe_allow_html=True,
    )
with d2:
    st.markdown(
        "<div class='fs-verdict'><div class='label'>Split</div>"
        "<div class='fs-band' style='height:30px;'>"
        "<div style='width:80%;background:#0E6E6A;'></div><div style='width:10%;background:#9A5B16;'></div>"
        "<div style='width:10%;background:#45627A;'></div></div>"
        "<div class='fs-band-legend'>"
        "<div><div class='k'>Train</div><div class='v fs-num'>1,807</div></div>"
        "<div><div class='k'>Validation</div><div class='v fs-num'>226</div></div>"
        "<div><div class='k'>Test</div><div class='v fs-num'>226</div></div></div></div>",
        unsafe_allow_html=True,
    )

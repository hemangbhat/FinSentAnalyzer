"""
Financial Sentiment Analyzer — Model Info Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import pandas as pd

from predict import get_available_models
from utils import get_model_info

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Info", page_icon="📊", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.header("📊 Model Information")

try:
    info = get_model_info(selected_model)
except Exception as e:
    st.error(f"❌ Could not load model info: {e}")
    st.stop()

# Current model card
st.markdown(f"""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;'>
    <h2 style='margin: 0; color: white;'>🎯 Current Model: {info.get('name', selected_model)}</h2>
    <p style='margin: 10px 0 0 0; opacity: 0.9;'>{info.get('type', 'Unknown')}</p>
</div>
""", unsafe_allow_html=True)

# Metrics in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", info.get("accuracy", "N/A"))
with col2:
    st.metric("F1 (Macro)", info.get("f1_macro", "N/A"))
with col3:
    st.metric("F1 (Weighted)", info.get("f1_weighted", "N/A"))
with col4:
    st.metric("Speed", info.get("speed", "N/A"))

# Features
st.markdown("### Model Features")
for feature in info.get("features", []):
    st.markdown(f"- {feature}")

st.markdown("---")

# All models comparison table
st.markdown("### All Available Models")

try:
    available_models = get_available_models()
    comparison_data = []
    for model_key in available_models:
        m_info = get_model_info(model_key)
        is_current = "✅" if model_key == selected_model else ""
        comparison_data.append({
            "Model": m_info.get("name", model_key),
            "Type": m_info.get("type", "Unknown"),
            "Accuracy": m_info.get("accuracy", "N/A"),
            "F1 Score": m_info.get("f1_macro", "N/A"),
            "Speed": m_info.get("speed", "N/A"),
            "Active": is_current,
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"❌ Could not load model comparison: {e}")

# Dataset info
st.markdown("---")
st.markdown("### Dataset Information")
st.markdown("""
**Financial PhraseBank** by Malo et al. (2014)
- **Source:** Financial news and reports
- **Labels:** Positive, Neutral, Negative
- **Samples:** 2,264 sentences (100% annotator agreement subset)
- **Split:** 70% train, 15% validation, 15% test
""")

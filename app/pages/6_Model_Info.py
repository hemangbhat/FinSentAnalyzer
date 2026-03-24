"""
Financial Sentiment Analyzer — Model Info Page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st  # pyre-ignore
import pandas as pd  # pyre-ignore

from predict import get_available_models  # pyre-ignore
from utils import get_model_info  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import inject_css, setup_sidebar  # pyre-ignore

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Info", page_icon="📊", layout="wide")
inject_css()
selected_model, predictor = setup_sidebar()

if predictor is None:
    st.stop()

# ── Page content ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 25px;'>
    <h1 style='font-size: 2.2em; font-weight: 700; margin-bottom: 5px;'>📊 Model Registry</h1>
    <p style='color: #94a3b8; font-size: 1.1em;'>Explore model architectures, evaluation metrics, and dataset properties.</p>
</div>
""", unsafe_allow_html=True)

try:
    info = get_model_info(selected_model)
except Exception as e:
    st.error(f"❌ Could not load model info: {e}")
    st.stop()

# Current model card
st.markdown(f"""
<div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(16, 185, 129, 0.05) 100%);
            padding: 30px; border-radius: 16px; border: 1px solid rgba(59, 130, 246, 0.3); margin-bottom: 30px; box-shadow: 0 4px 25px rgba(0,0,0,0.2);'>
    <div style='color: #60a5fa; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-size: 0.9em;'>Active Engine</div>
    <h2 style='margin: 0; color: #f8fafc; font-size: 2.2em; letter-spacing: -0.5px;'>{info.get('name', selected_model)}</h2>
    <p style='margin: 8px 0 0 0; color: #cbd5e1; font-size: 1.1em;'>Architecture: <span style='color: #94a3b8; font-family: monospace;'>{info.get('type', 'Unknown')}</span></p>
</div>
""", unsafe_allow_html=True)

# Metrics in columns
st.markdown("### 🏆 Performance Benchmarks")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card' style='border-top: 3px solid #3b82f6;'>
        <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>Accuracy Rating</div>
        <div style='font-size: 2em; font-weight: 700; color: #f8fafc; margin-top: 5px;'>{info.get("accuracy", "N/A")}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='metric-card' style='border-top: 3px solid #8b5cf6;'>
        <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>F1 Score (Macro)</div>
        <div style='font-size: 2em; font-weight: 700; color: #f8fafc; margin-top: 5px;'>{info.get("f1_macro", "N/A")}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='metric-card' style='border-top: 3px solid #f59e0b;'>
        <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>F1 Score (Weighted)</div>
        <div style='font-size: 2em; font-weight: 700; color: #f8fafc; margin-top: 5px;'>{info.get("f1_weighted", "N/A")}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    speed = info.get("speed", "N/A")
    speed_color = "#10b981" if "fast" in speed.lower() else "#f59e0b" if "medium" in speed.lower() else "#ef4444"
    st.markdown(f"""
    <div class='metric-card' style='border-top: 3px solid {speed_color};'>
        <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase;'>Processing Speed</div>
        <div style='font-size: 2em; font-weight: 700; color: {speed_color}; margin-top: 5px; text-transform: capitalize;'>{speed}</div>
    </div>
    """, unsafe_allow_html=True)

# Features
st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.05); margin: 30px 0;'>", unsafe_allow_html=True)
st.markdown("### 🧬 Key Capabilities")
st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
for feature in info.get("features", []):
    st.markdown(f"<div style='margin-bottom: 10px; display: flex; align-items: start; gap: 10px;'><span style='color: #3b82f6;'>✓</span> <span style='color: #cbd5e1;'>{feature}</span></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# All models comparison table
st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.05); margin: 30px 0;'>", unsafe_allow_html=True)
st.markdown("### 🗄️ Model Comparison Matrix")

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
st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.05); margin: 30px 0;'>", unsafe_allow_html=True)
st.markdown("### 📚 Foundational Dataset")

st.markdown("""
<div class='insight-card' style='border-top: 4px solid #8b5cf6;'>
    <div style='display: flex; gap: 20px; flex-wrap: wrap;'>
        <div style='flex: 1; min-width: 300px;'>
            <h3 style='margin-top: 0; color: #f8fafc;'>Financial PhraseBank</h3>
            <p style='color: #94a3b8;'>Formally created by Malo et al. (2014), heavily utilised for institutional testing.</p>
            <div style='display: grid; grid-template-columns: 120px 1fr; gap: 10px; margin-top: 20px;'>
                <div style='color: #94a3b8; font-weight: 600; font-size: 0.85em; text-transform: uppercase;'>Source</div> <div style='color: #cbd5e1;'>Financial news and reports</div>
                <div style='color: #94a3b8; font-weight: 600; font-size: 0.85em; text-transform: uppercase;'>Taxonomy</div> <div style='color: #cbd5e1;'><span class='chip positive'>Positive</span> <span class='chip neutral'>Neutral</span> <span class='chip negative'>Negative</span></div>
                <div style='color: #94a3b8; font-weight: 600; font-size: 0.85em; text-transform: uppercase;'>Samples</div> <div style='color: #cbd5e1;'>2,264 sentences (100% annotator agreement subset)</div>
            </div>
        </div>
        <div style='flex: 1; min-width: 250px; display: flex; flex-direction: column; justify-content: center; background: rgba(0,0,0,0.2); padding: 20px; border-radius: 12px; text-align: center;'>
            <div style='color: #94a3b8; font-size: 0.85em; text-transform: uppercase; margin-bottom: 10px;'>Dataset Training Split</div>
            <div style='display: flex; border-radius: 8px; overflow: hidden; height: 30px; margin-bottom: 10px;'>
                <div style='flex: 70; background: #3b82f6; display: flex; align-items: center; justify-content: center; font-size: 0.8em; font-weight: 600; color: white;'>70% Train</div>
                <div style='flex: 15; background: #f59e0b; display: flex; align-items: center; justify-content: center; font-size: 0.8em; font-weight: 600; color: white;'>15% Val</div>
                <div style='flex: 15; background: #10b981; display: flex; align-items: center; justify-content: center; font-size: 0.8em; font-weight: 600; color: white;'>15% Test</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

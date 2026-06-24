"""
FinSight design system — centralized UI tokens, theme, and reusable components.

This module is the single source of truth for the app's look and feel:
- Color + spacing tokens
- Global CSS (premium dark fintech theme)
- Plotly dark chart defaults
- Reusable components (page headers, KPI strips, metric cards, section
  headers, pills, status banners, chart containers, empty/loading states)
- Cached model loader and the sidebar

Pages import these helpers via ``shared`` (a thin compatibility re-export),
so every screen shares one consistent design language.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

# Ensure src/ is importable regardless of which page triggers the import.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import plotly.graph_objects as go  # pyre-ignore
import streamlit as st  # pyre-ignore

from predict import SentimentPredictor, get_available_models  # pyre-ignore

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "bg": "#0a0e17",
    "bg_alt": "#0e1320",
    "surface": "#111827",
    "card": "rgba(255,255,255,0.025)",
    "border": "rgba(255,255,255,0.07)",
    "border_strong": "rgba(255,255,255,0.14)",
    "text": "#e8eef7",
    "text_muted": "#94a3b8",
    "text_faint": "#64748b",
    "primary": "#3b82f6",
    "primary_soft": "#60a5fa",
    "accent": "#8b5cf6",
    "positive": "#10b981",
    "negative": "#ef4444",
    "neutral": "#3b82f6",
    "warning": "#f59e0b",
}

SENTIMENT_COLORS = {
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
    "neutral": COLORS["neutral"],
}

# Restrained categorical palette for charts.
CHART_SEQUENCE = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#14b8a6"]

# Friendly display names for the model selector.
MODEL_LABELS = {
    "baseline_logreg": "Logistic Regression",
    "baseline_naive_bayes": "Naive Bayes",
    "baseline_svm": "SVM (Linear)",
    "baseline_random_forest": "Random Forest",
    "baseline_gradient_boosting": "Gradient Boosting",
    "baseline_mlp": "Neural Network (MLP)",
    "baseline_ensemble": "Voting Ensemble",
    "finbert_pretrained": "FinBERT (Transformer)",
    "finbert": "FinBERT (Fine-tuned)",
    "distilbert": "DistilBERT (Fine-tuned)",
    "roberta": "RoBERTa (Fine-tuned)",
    "bert": "BERT (Fine-tuned)",
}


def model_label(key: str) -> str:
    """Return a human-friendly label for a model key."""
    return MODEL_LABELS.get(key, key.replace("baseline_", "").replace("_", " ").title())


def get_sentiment_color(label: str) -> str:
    """Return the hex colour associated with a sentiment label."""
    return SENTIMENT_COLORS.get(str(label).lower(), COLORS["text_faint"])


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --fs-primary: #3b82f6;
    --fs-accent: #8b5cf6;
    --fs-positive: #10b981;
    --fs-negative: #ef4444;
    --fs-neutral: #3b82f6;
    --fs-text: #e8eef7;
    --fs-muted: #94a3b8;
    --fs-faint: #64748b;
    --fs-border: rgba(255,255,255,0.07);
    --fs-radius: 16px;
}

/* Base typography — scoped to avoid overriding Material Symbols icon fonts */
.stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp label, .stApp li, .stApp td, .stApp th, .stMarkdown, .stMarkdown p {
    font-family: 'Inter', system-ui, sans-serif;
}
code, pre, .mono { font-family: 'JetBrains Mono', monospace; }

/* Never let the icon font be overridden (sidebar collapse, expander arrows, etc.) */
.material-symbols-rounded, .material-symbols-outlined, .material-icons,
[data-testid="stIconMaterial"], span[class*="material-symbols"], span[class*="material-icons"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}

/* App background — deep fintech gradient */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 600px at 12% -8%, rgba(59,130,246,0.10), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(139,92,246,0.10), transparent 55%),
        linear-gradient(180deg, #0a0e17 0%, #090c14 100%);
    color: var(--fs-text);
}
[data-testid="stHeader"] {
    background: rgba(10,14,23,0.55);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--fs-border);
}
[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1280px;
}

/* Headings */
h1, h2, h3, h4 { color: var(--fs-text); letter-spacing: -0.02em; font-weight: 700; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d121e 0%, #0a0e17 100%);
    border-right: 1px solid var(--fs-border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--fs-border);
    border-radius: 10px;
    color: var(--fs-text);
    font-weight: 500;
    padding: 0.55rem 1rem;
    transition: all 0.18s ease;
    width: 100%;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.08);
    border-color: var(--fs-border);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    border: none; color: #fff; font-weight: 600;
    box-shadow: 0 6px 18px rgba(59,130,246,0.32);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 10px 26px rgba(59,130,246,0.45); transform: translateY(-1px);
}
.stDownloadButton > button { border-radius: 10px; border: 1px solid var(--fs-border); }

/* ── Inputs ──────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput input {
    background-color: rgba(255,255,255,0.03);
    color: var(--fs-text);
    border: 1px solid var(--fs-border);
    border-radius: 10px;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus,
.stSelectbox > div > div:focus-within {
    border-color: var(--fs-primary);
    box-shadow: 0 0 0 2px rgba(59,130,246,0.25);
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.02);
    border: 1.5px dashed var(--fs-border); border-radius: var(--fs-radius);
}

/* ── Cards (legacy class names kept for backwards compatibility) ─────── */
.metric-card, .feature-card, .insight-card, .app-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.015) 100%);
    border: 1px solid var(--fs-border);
    border-radius: var(--fs-radius);
    padding: 22px 24px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.25);
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}
.feature-card:hover, .app-card:hover {
    transform: translateY(-3px);
    border-color: var(--fs-border-strong, rgba(255,255,255,0.16));
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.card-title { font-size: 1.05em; font-weight: 600; color: var(--fs-text); margin-bottom: 10px; display: flex; align-items: center; gap: 9px; }
.card-text { color: var(--fs-muted); font-size: 0.93em; line-height: 1.55; }

/* ── Page header / hero ──────────────────────────────────────────────── */
.fs-eyebrow {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--fs-primary-soft, #60a5fa);
    background: rgba(59,130,246,0.10); border: 1px solid rgba(59,130,246,0.22);
    padding: 5px 12px; border-radius: 999px; margin-bottom: 14px;
}
.fs-page-title { font-size: 2.05rem; font-weight: 800; margin: 0 0 6px 0; line-height: 1.15; }
.fs-page-sub { color: var(--fs-muted); font-size: 1.02rem; max-width: 760px; line-height: 1.55; margin: 0; }

.fs-hero {
    position: relative; overflow: hidden;
    border: 1px solid var(--fs-border); border-radius: 22px;
    padding: 54px 44px;
    background:
        radial-gradient(700px 320px at 8% -30%, rgba(59,130,246,0.20), transparent 60%),
        radial-gradient(600px 300px at 100% -10%, rgba(139,92,246,0.18), transparent 55%),
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    margin-bottom: 30px;
}
.fs-hero h1 {
    font-size: 3rem; font-weight: 800; margin: 0 0 14px 0; line-height: 1.08;
    background: linear-gradient(90deg, #f8fafc 10%, #93b4f5 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.fs-hero p { color: var(--fs-muted); font-size: 1.12rem; max-width: 640px; line-height: 1.6; margin: 0; }

/* ── Section header ──────────────────────────────────────────────────── */
.fs-section { display: flex; align-items: center; gap: 12px; margin: 30px 0 14px 0; }
.fs-section .bar { width: 4px; height: 26px; border-radius: 4px; background: linear-gradient(180deg, #3b82f6, #8b5cf6); }
.fs-section h3 { margin: 0; font-size: 1.22rem; font-weight: 700; }
.fs-section .cap { color: var(--fs-faint); font-size: 0.88rem; margin-left: 2px; }

/* ── KPI strip ───────────────────────────────────────────────────────── */
.fs-kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin: 6px 0 8px 0; }
.fs-kpi {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
    border: 1px solid var(--fs-border); border-radius: 14px; padding: 16px 18px;
    border-top: 2px solid var(--kpi-accent, #3b82f6);
}
.fs-kpi .label { color: var(--fs-muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.fs-kpi .value { color: var(--fs-text); font-size: 1.7rem; font-weight: 800; line-height: 1.25; margin-top: 4px; }
.fs-kpi .sub { color: var(--fs-faint); font-size: 0.8rem; margin-top: 2px; }

/* ── Pills & banners ─────────────────────────────────────────────────── */
.chip, .pill {
    display: inline-block; padding: 4px 11px; border-radius: 999px;
    font-size: 0.8em; font-weight: 600; margin: 2px;
}
.chip.positive, .pill.positive { background: rgba(16,185,129,0.14); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.chip.negative, .pill.negative { background: rgba(239,68,68,0.14); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.chip.neutral,  .pill.neutral  { background: rgba(59,130,246,0.14); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.pill.muted { background: rgba(148,163,184,0.12); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.25); }

.fs-banner { display: flex; gap: 12px; align-items: flex-start; border-radius: 12px; padding: 14px 18px; margin: 6px 0; border: 1px solid var(--fs-border); }
.fs-banner .ico { font-size: 1.15em; line-height: 1.4; }
.fs-banner .body { color: var(--fs-text); font-size: 0.94rem; line-height: 1.5; }
.fs-banner.info { background: rgba(59,130,246,0.08); border-color: rgba(59,130,246,0.25); }
.fs-banner.success { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.25); }
.fs-banner.warning { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.28); }
.fs-banner.error { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.28); }

/* ── Empty state ─────────────────────────────────────────────────────── */
.fs-empty {
    text-align: center; padding: 48px 28px; border: 1.5px dashed var(--fs-border);
    border-radius: var(--fs-radius); background: rgba(255,255,255,0.012);
}
.fs-empty .ico { font-size: 2.4rem; opacity: 0.7; }
.fs-empty h4 { margin: 12px 0 6px 0; color: var(--fs-text); }
.fs-empty p { color: var(--fs-muted); font-size: 0.92rem; max-width: 460px; margin: 0 auto; line-height: 1.55; }

/* ── Sentiment verdict (single analysis) ─────────────────────────────── */
.fs-verdict {
    border: 1px solid var(--fs-border); border-radius: var(--fs-radius); padding: 26px;
    text-align: center; display: flex; flex-direction: column; justify-content: center; height: 100%;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
}
.fs-verdict .ey { color: var(--fs-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.12em; }
.fs-verdict .lab { font-size: 3rem; font-weight: 800; line-height: 1.15; text-transform: capitalize; margin: 8px 0; }
.fs-verdict .conf { color: var(--fs-muted); font-size: 0.98rem; }
.fs-verdict .conf b { color: var(--fs-text); }
.fs-conf-track { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.08); margin-top: 14px; overflow: hidden; }
.fs-conf-fill { height: 100%; border-radius: 999px; }

/* ── Sidebar nav links ───────────────────────────────────────────────── */
[data-testid="stSidebarNav"] a { border-radius: 9px; }

/* ── Tables ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid var(--fs-border); border-radius: 12px; overflow: hidden; }
[data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: var(--fs-muted) !important; font-weight: 600 !important; }

/* Dividers */
hr { border-color: var(--fs-border); margin: 1.8rem 0; }

/* Tabs + expanders */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 9px 9px 0 0; }
[data-testid="stExpander"] { border: 1px solid var(--fs-border); border-radius: 12px; background: rgba(255,255,255,0.015); }

/* Responsive */
@media (max-width: 768px) {
    .fs-hero { padding: 34px 24px; }
    .fs-hero h1 { font-size: 2.1rem; }
    .fs-page-title { font-size: 1.6rem; }
}
</style>
"""


def inject_css() -> None:
    """Inject the global design-system CSS. Call once per page."""
    st.markdown(BASE_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────────────────────────────────────

_GRID = "rgba(255,255,255,0.06)"
_LINE = "rgba(255,255,255,0.12)"
_FONT = {"color": "#cbd5e1", "family": "Inter, sans-serif"}


def style_fig(fig: go.Figure, height: int | None = None, title: str | None = None, legend: bool = True) -> go.Figure:
    """Apply the consistent dark chart theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_FONT,
        margin=dict(l=16, r=16, t=46 if title else 18, b=16),
        colorway=CHART_SEQUENCE,
        title=({"text": title, "font": {"color": "#f8fafc", "size": 16}} if title else None),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#111827", bordercolor=_LINE, font_size=12),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, linecolor=_LINE, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, linecolor=_LINE, zeroline=False)
    return fig


def render_chart(fig: go.Figure) -> None:
    """Render a Plotly figure full-width with a clean config."""
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────


def page_header(title: str, subtitle: str = "", eyebrow: str | None = None) -> None:
    """Render a consistent page header with optional eyebrow tag."""
    eyebrow_html = f"<div class='fs-eyebrow'>{eyebrow}</div>" if eyebrow else ""
    sub_html = f"<p class='fs-page-sub'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div style='margin-bottom:22px;'>{eyebrow_html}<div class='fs-page-title'>{title}</div>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str = "") -> None:
    """Render a section header with an accent bar."""
    cap = f"<span class='cap'>· {caption}</span>" if caption else ""
    st.markdown(
        f"<div class='fs-section'><div class='bar'></div><h3>{title}</h3>{cap}</div>",
        unsafe_allow_html=True,
    )


def kpi_strip(items: list[dict]) -> None:
    """
    Render a horizontal KPI strip.

    Each item: {"label": str, "value": str|int, "sub": str?, "accent": hex?}.
    """
    cells = []
    for it in items:
        accent = it.get("accent", COLORS["primary"])
        sub = f"<div class='sub'>{it['sub']}</div>" if it.get("sub") else ""
        cells.append(
            f"<div class='fs-kpi' style='--kpi-accent:{accent};'>"
            f"<div class='label'>{it['label']}</div>"
            f"<div class='value'>{it['value']}</div>{sub}</div>"
        )
    st.markdown(f"<div class='fs-kpi-grid'>{''.join(cells)}</div>", unsafe_allow_html=True)


def metric_card(label: str, value, sub: str = "", accent: str | None = None) -> None:
    """Render a single metric card (use inside a column)."""
    accent = accent or COLORS["primary"]
    sub_html = f"<div style='color:{COLORS['text_faint']};font-size:0.82rem;margin-top:3px;'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='metric-card' style='border-top:2px solid {accent};'>"
        f"<div style='color:{COLORS['text_muted']};font-size:0.74rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;'>{label}</div>"
        f"<div style='font-size:1.8rem;font-weight:800;color:{accent};margin-top:4px;'>{value}</div>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "muted") -> str:
    """Return an inline pill HTML snippet (kind: positive|negative|neutral|muted)."""
    return f"<span class='pill {kind}'>{text}</span>"


def status_banner(text: str, kind: str = "info", icon: str | None = None) -> None:
    """Render a status banner (kind: info|success|warning|error)."""
    icons = {"info": "ℹ️", "success": "✓", "warning": "⚠️", "error": "✕"}
    ico = icon or icons.get(kind, "ℹ️")
    st.markdown(
        f"<div class='fs-banner {kind}'><div class='ico'>{ico}</div><div class='body'>{text}</div></div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str = "", icon: str = "📊") -> None:
    """Render a polished empty state."""
    st.markdown(
        f"<div class='fs-empty'><div class='ico'>{icon}</div><h4>{title}</h4><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def verdict_card(label: str, confidence: float | None = None) -> None:
    """Render a sentiment verdict card with a confidence bar."""
    color = get_sentiment_color(label)
    conf_html = ""
    if confidence is not None:
        pct = max(0.0, min(1.0, float(confidence))) * 100
        conf_html = (
            f"<div class='conf'>Confidence <b>{confidence:.1%}</b></div>"
            f"<div class='fs-conf-track'><div class='fs-conf-fill' style='width:{pct:.1f}%;background:{color};'></div></div>"
        )
    st.markdown(
        f"<div class='fs-verdict' style='border-top:3px solid {color};'>"
        f"<div class='ey'>Predicted Sentiment</div>"
        f"<div class='lab' style='color:{color};'>{label}</div>{conf_html}</div>",
        unsafe_allow_html=True,
    )


@contextmanager
def chart_container(title: str = "", caption: str = ""):
    """Context manager that renders an optional section header before a chart block."""
    if title:
        section_header(title, caption)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS (sentiment-specific)
# ─────────────────────────────────────────────────────────────────────────────


def create_probability_chart(probabilities: dict) -> go.Figure:
    """Vertical bar chart of class probabilities, dark-themed."""
    labels = list(probabilities.keys())
    values = [probabilities[lbl] * 100 for lbl in labels]
    colors = [get_sentiment_color(lbl) for lbl in labels]

    fig = go.Figure(
        go.Bar(
            x=[lbl.capitalize() for lbl in labels],
            y=values,
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont={"color": "#e2e8f0", "size": 12},
            width=0.55,
        )
    )
    style_fig(fig, height=300, title="Sentiment Probabilities", legend=False)
    fig.update_layout(yaxis_title="Probability (%)", yaxis_range=[0, 112])
    fig.update_xaxes(showgrid=False)
    return fig


def create_gauge_chart(probabilities: dict, prediction: str) -> go.Figure:
    """Confidence gauge for the predicted class, dark-themed."""
    confidence = probabilities[prediction]
    color = get_sentiment_color(prediction)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Confidence · {prediction.capitalize()}"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(255,255,255,0.03)"},
                    {"range": [33, 66], "color": "rgba(255,255,255,0.05)"},
                    {"range": [66, 100], "color": "rgba(255,255,255,0.07)"},
                ],
            },
        )
    )
    style_fig(fig, height=250, legend=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER + SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_predictor(model_type: str):
    """
    Load the predictor for a model.

    If FINSIGHT_API_URL is set, returns a RemotePredictor that calls the
    FastAPI inference service (UI/serving split). Otherwise loads the model
    in-process. Both expose the same ``.predict()`` interface.
    """
    api_url = os.getenv("FINSIGHT_API_URL")
    if api_url:
        from api_client import RemotePredictor  # local import; only needed in split mode

        return RemotePredictor(model_type, base_url=api_url, api_key=os.getenv("FINSIGHT_API_KEY"))
    return SentimentPredictor(model_type)


def _list_models() -> list[str]:
    """Model list — from the API in split mode (with local fallback), else local."""
    api_url = os.getenv("FINSIGHT_API_URL")
    if api_url:
        from api_client import remote_available_models

        remote = remote_available_models(api_url, api_key=os.getenv("FINSIGHT_API_KEY"))
        if remote:
            return remote
    return get_available_models()


def setup_sidebar():
    """
    Render the premium sidebar (brand, model selector, status).

    Returns
    -------
    tuple[str | None, SentimentPredictor | None]
    """
    st.sidebar.markdown(
        "<div style='padding:6px 2px 16px 2px;'>"
        "<div style='font-size:1.5em;font-weight:800;letter-spacing:-0.02em;"
        "background:linear-gradient(90deg,#f8fafc,#7aa7f5);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
        "◆ FinSight</div>"
        "<div style='font-size:0.72em;color:#64748b;font-weight:600;letter-spacing:0.16em;text-transform:uppercase;margin-top:2px;'>"
        "Sentiment Intelligence</div></div>",
        unsafe_allow_html=True,
    )

    available_models = _list_models()
    if not available_models:
        st.sidebar.error("No trained models found. Run `python src/train.py --model baselines`.")
        return None, None

    st.sidebar.markdown(
        "<div style='font-size:0.74em;color:#94a3b8;font-weight:600;letter-spacing:0.08em;"
        "text-transform:uppercase;margin-bottom:6px;'>Model Engine</div>",
        unsafe_allow_html=True,
    )
    selected_model = st.sidebar.selectbox(
        "Model",
        available_models,
        index=0,
        format_func=model_label,
        label_visibility="collapsed",
        help="Choose the model used across analysis pages.",
    )

    predictor = None
    load_error = None
    try:
        with st.spinner("Loading model…"):
            predictor = load_predictor(selected_model)
    except Exception as exc:  # noqa: BLE001
        load_error = str(exc)

    online = predictor is not None
    dot = "#10b981" if online else "#ef4444"
    status = "Engine ready" if online else "Engine unavailable"
    st.sidebar.markdown(
        "<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);"
        "border-radius:12px;padding:14px 16px;margin-top:16px;'>"
        "<div style='font-size:0.72em;color:#94a3b8;font-weight:600;letter-spacing:0.08em;"
        "text-transform:uppercase;margin-bottom:10px;'>System Status</div>"
        f"<div style='display:flex;align-items:center;gap:8px;font-size:0.9em;color:#e2e8f0;margin-bottom:7px;'>"
        f"<span style='width:8px;height:8px;border-radius:50%;background:{dot};box-shadow:0 0 8px {dot};'></span>{status}</div>"
        f"<div style='display:flex;align-items:center;gap:8px;font-size:0.9em;color:#e2e8f0;'>"
        f"<span style='width:8px;height:8px;border-radius:50%;background:#3b82f6;'></span>{model_label(selected_model)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if load_error:
        st.sidebar.error(f"Failed to load model: {load_error}")

    st.sidebar.markdown(
        "<div style='margin-top:14px;font-size:0.78em;color:#64748b;line-height:1.5;'>"
        "Models are trained on the Financial PhraseBank. Sentiment is classified as "
        "positive, neutral, or negative.</div>",
        unsafe_allow_html=True,
    )

    return selected_model, predictor

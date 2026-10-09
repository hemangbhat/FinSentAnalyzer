"""
FinSight design system: tokens, global CSS, chart theme and components.

The visual world is a central bank's quarterly report on its own models:
a white report ground, a deep-teal shell, hairline rules instead of card
shadows, numbered exhibits with a source line, and uncertainty drawn as
bands. Every text colour here meets WCAG AA on the white ground and on the
panel tone. Sentiment is never colour alone: each label carries a glyph
(▲ positive, ▶ neutral, ▼ negative) and a word.

Native widgets are themed in ``.streamlit/config.toml`` (including the
self-hosted Public Sans faces); this module styles the HTML Streamlit does
not draw itself. Pages import everything through ``shared``.
"""

from __future__ import annotations

import html
import json
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
    "bg": "#FFFFFF",
    "panel": "#F2F5F4",
    "panel_soft": "#F7F9F8",
    "border": "#D5DEDC",
    "border_strong": "#B9C6C3",
    "text": "#12211F",
    "text_body": "#2E3F3C",
    "text_muted": "#4B5D5A",
    "text_faint": "#5E6E6A",
    "primary": "#0E6E6A",
    "primary_soft": "#0E6E6A",
    "primary_tint": "#E3F0EE",
    "accent": "#45627A",
    "positive": "#1D7451",
    "negative": "#A3322B",
    "neutral": "#4F6470",
    "warning": "#9A5B16",
    "shell": "#0B3A3C",
}

SENTIMENT_COLORS = {
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
    "neutral": COLORS["neutral"],
}
SENTIMENT_TINTS = {"positive": "#E2F0E9", "negative": "#F6E4E2", "neutral": "#E6EBEE"}
SENTIMENT_GLYPHS = {"positive": "▲", "negative": "▼", "neutral": "▶"}

# Chart palette: teal first, then ochre for the one series that needs attention.
CHART_SEQUENCE = ["#0E6E6A", "#9A5B16", "#45627A", "#A3322B", "#1D7451", "#8FB8B2"]

# Friendly display names for the model selector.
MODEL_LABELS = {
    "baseline_logreg": "Logistic Regression",
    "baseline_naive_bayes": "Naive Bayes",
    "baseline_svm": "SVM (Linear)",
    "baseline_random_forest": "Random Forest",
    "baseline_gradient_boosting": "Gradient Boosting",
    "baseline_mlp": "Neural Network (MLP)",
    "baseline_ensemble": "Voting Ensemble",
    "finbert_pretrained": "FinBERT (zero-shot)",
    "finbert": "FinBERT (fine-tuned)",
    "distilbert": "DistilBERT (fine-tuned)",
    "roberta": "RoBERTa (fine-tuned)",
    "bert": "BERT (fine-tuned)",
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def model_label(key: str) -> str:
    """Return a human-friendly label for a model key."""
    return MODEL_LABELS.get(key, key.replace("baseline_", "").replace("_", " ").title())


def get_sentiment_color(label: str) -> str:
    """Return the hex colour associated with a sentiment label."""
    return SENTIMENT_COLORS.get(str(label).lower(), COLORS["text_faint"])


def sentiment_tag(label: str, extra: str = "") -> str:
    """Inline sentiment marker: glyph + word (+ optional figure), never colour alone."""
    key = str(label).lower()
    glyph = SENTIMENT_GLYPHS.get(key, "•")
    tail = f" <span class='fs-num'>{extra}</span>" if extra else ""
    return f"<span class='fs-tag {key}'><span aria-hidden='true'>{glyph}</span> {html.escape(key.capitalize())}{tail}</span>"


@st.cache_data(show_spinner=False)
def load_result(name: str):
    """Read ``results/<name>``; ``None`` when missing or an un-fetched Git LFS pointer."""
    try:
        return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ICONS (authored SVG, one stroke weight; used where Streamlit can't place
# Material Symbols, i.e. inside custom HTML)
# ─────────────────────────────────────────────────────────────────────────────

_ICON_PATHS = {
    "info": "<circle cx='12' cy='12' r='9'/><path d='M12 11v5'/><path d='M12 8h.01'/>",
    "success": "<circle cx='12' cy='12' r='9'/><path d='m8.5 12.5 2.5 2.5 4.5-5'/>",
    "warning": "<path d='M12 4 21 19.5H3L12 4z'/><path d='M12 10v4'/><path d='M12 17h.01'/>",
    "error": "<circle cx='12' cy='12' r='9'/><path d='m9 9 6 6'/><path d='m15 9-6 6'/>",
    "edit": "<path d='M4 20h4L19 9l-4-4L4 16v4z'/><path d='m13.5 6.5 4 4'/>",
    "upload": "<path d='M12 15V4'/><path d='m7 9 5-5 5 5'/><path d='M5 15v4h14v-4'/>",
    "chart": "<path d='M4 20V4'/><path d='M4 20h16'/><path d='M8 16v-4'/><path d='M12 16V8'/><path d='M16 16v-6'/>",
    "search": "<circle cx='11' cy='11' r='6.5'/><path d='m20 20-4.2-4.2'/>",
    "flask": "<path d='M9 3h6'/><path d='M10 3v6L4.5 19a1.3 1.3 0 0 0 1.1 2h12.8a1.3 1.3 0 0 0 1.1-2L14 9V3'/><path d='M7.5 14h9'/>",
}


def icon(name: str, size: int = 18) -> str:
    """Return an inline SVG icon (stroke follows ``currentColor``)."""
    paths = _ICON_PATHS.get(name, _ICON_PATHS["info"])
    return (
        f"<svg class='fs-ico' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' "
        "stroke='currentColor' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round' "
        f"aria-hidden='true'>{paths}</svg>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

BASE_CSS = """
<style>
:root {
    --fs-ink: #12211F; --fs-body: #2E3F3C; --fs-muted: #4B5D5A; --fs-faint: #5E6E6A;
    --fs-rule: #D5DEDC; --fs-rule-strong: #B9C6C3; --fs-panel: #F2F5F4; --fs-panel-soft: #F7F9F8;
    --fs-teal: #0E6E6A; --fs-teal-tint: #E3F0EE; --fs-shell: #0B3A3C;
    --fs-pos: #1D7451; --fs-neg: #A3322B; --fs-neu: #4F6470; --fs-ochre: #9A5B16;
    --fs-radius: 6px;
}

/* Browser surfaces carry the palette too */
::selection { background: #CDE6E1; color: var(--fs-ink); }
.stApp { caret-color: var(--fs-teal); }
.stApp * { scrollbar-width: thin; scrollbar-color: var(--fs-rule-strong) transparent; }
.stApp a { text-underline-offset: 3px; }
.stApp :focus-visible { outline: 2px solid var(--fs-teal); outline-offset: 2px; }
.fs-num, .fs-fig .value, [data-testid="stMetricValue"], [data-testid="stDataFrame"],
.stApp table { font-variant-numeric: tabular-nums lining-nums; }

/* Never let the icon font be overridden */
.material-symbols-rounded, [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* Page frame */
[data-testid="stHeader"] { background: rgba(255,255,255,0.92); border-bottom: 1px solid var(--fs-rule); }
[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 4.4rem; padding-bottom: 4rem; max-width: 1180px;
}
.stApp p, .stApp li { line-height: 1.6; }
h1, h2, h3, h4 { letter-spacing: -0.015em; color: var(--fs-ink); }

/* ── Shell (sidebar) ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] { border-right: none; }
[data-testid="stSidebarNav"] a { border-radius: 5px; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(143,214,201,0.14); box-shadow: inset 2px 0 0 #8FD6C9;
}
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: #FFFFFF; font-weight: 600; }
[data-testid="stNavSectionHeader"] {
    color: #A9C4C0 !important; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 12px;
}
[data-testid="stLogo"] { height: 1.9rem; max-width: 12rem; }
.fs-side-label { font-size: 0.72rem; color: #A9C4C0; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; margin: 6px 0 6px 0; }
.fs-side-status { border-top: 1px solid #24595A; border-bottom: 1px solid #24595A; padding: 10px 2px; margin-top: 12px; }
.fs-side-status .row { display: flex; align-items: center; gap: 8px; font-size: 0.84rem; color: #E3EEEC; }
.fs-side-status .row + .row { margin-top: 5px; }
.fs-side-status .dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.fs-side-note { margin-top: 12px; font-size: 0.78rem; color: #A9C4C0; line-height: 1.5; }

/* ── Controls ────────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    font-weight: 600; transition: background-color .15s ease, border-color .15s ease, box-shadow .15s ease;
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 2px 8px rgba(14,110,106,0.28); }
[data-testid="stFileUploaderDropzone"] { border-style: dashed; }

/* ── Report masthead (page header) ──────────────────────────────────── */
.fs-masthead { border-top: 3px solid var(--fs-teal); padding-top: 18px; margin-bottom: 26px; }
.fs-masthead .row { display: flex; gap: 24px; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; }
.fs-masthead h1 { font-size: 2.05rem; font-weight: 700; margin: 0 0 6px 0; line-height: 1.15; padding: 0; }
.fs-masthead .standfirst { color: var(--fs-muted); font-size: 1.04rem; max-width: 68ch; margin: 0; line-height: 1.55; }
.fs-masthead .meta { color: var(--fs-faint); font-size: 0.82rem; margin-top: 12px; }
.fs-masthead .meta b { color: var(--fs-body); font-weight: 600; }

/* ── Section and exhibit headings ───────────────────────────────────── */
.fs-section { border-top: 1px solid var(--fs-rule); padding-top: 14px; margin: 34px 0 12px 0; }
.fs-section h3 { margin: 0; padding: 0; font-size: 1.14rem; font-weight: 650; line-height: 1.3; }
.fs-section .cap { color: var(--fs-muted); font-size: 0.9rem; margin: 4px 0 0 0; max-width: 75ch; }
.fs-section .ex { color: var(--fs-teal); font-weight: 700; margin-right: 8px; }
.fs-source { color: var(--fs-faint); font-size: 0.78rem; margin: -4px 0 8px 0; }

/* ── Key figures (a ruled table row, not cards) ─────────────────────── */
.fs-figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    border-top: 1px solid var(--fs-ink); border-bottom: 1px solid var(--fs-rule); margin: 6px 0 10px 0; }
.fs-fig { padding: 12px 16px 12px 16px; border-left: 1px solid var(--fs-rule); }
.fs-fig:first-child { border-left: none; padding-left: 0; }
.fs-fig .label { color: var(--fs-muted); font-size: 0.8rem; font-weight: 600; }
.fs-fig .value { color: var(--fs-ink); font-size: 1.65rem; font-weight: 700; line-height: 1.25; margin-top: 2px; }
.fs-fig .sub { color: var(--fs-faint); font-size: 0.8rem; margin-top: 2px; line-height: 1.4; }

/* ── Legacy metric cards render as key figures ──────────────────────── */
.metric-card { background: none !important; border: none !important; border-top: 1px solid var(--fs-ink) !important;
    border-radius: 0 !important; padding: 12px 0 10px 0 !important; box-shadow: none !important; }
.metric-card > div:first-child { font-size: 0.8rem !important; text-transform: none !important; letter-spacing: 0 !important;
    color: var(--fs-muted) !important; font-weight: 600 !important; }
.metric-card > div:nth-child(2) { font-size: 1.65rem !important; font-weight: 700 !important; line-height: 1.25 !important;
    font-variant-numeric: tabular-nums; }

/* ── Panels (legacy class names used by pages) ──────────────────────── */
.insight-card, .app-card, .feature-card {
    background: var(--fs-panel-soft); border: 1px solid var(--fs-rule); border-radius: var(--fs-radius);
    padding: 18px 20px; color: var(--fs-body);
}
.card-title { font-weight: 650; color: var(--fs-ink); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.card-text { color: var(--fs-muted); font-size: 0.93em; line-height: 1.55; }

/* ── Sentiment tags and pills ───────────────────────────────────────── */
.fs-tag, .chip, .pill {
    display: inline-flex; align-items: center; gap: 5px; padding: 2px 9px; border-radius: 4px;
    font-size: 0.84em; font-weight: 600; margin: 2px; border: 1px solid transparent; white-space: nowrap;
}
.fs-tag.positive, .chip.positive, .pill.positive { background: #E2F0E9; color: var(--fs-pos); border-color: #BFDCCD; }
.fs-tag.negative, .chip.negative, .pill.negative { background: #F6E4E2; color: var(--fs-neg); border-color: #E6C3BF; }
.fs-tag.neutral,  .chip.neutral,  .pill.neutral  { background: #E6EBEE; color: var(--fs-neu); border-color: #CBD5DB; }
.chip, .pill.muted { background: var(--fs-panel); color: var(--fs-body); border-color: var(--fs-rule); }
.chip.positive::before { content: "▲"; } .chip.negative::before { content: "▼"; } .chip.neutral::before { content: "▶"; }

/* ── Banners ────────────────────────────────────────────────────────── */
.fs-banner { display: flex; gap: 10px; align-items: flex-start; border-radius: var(--fs-radius);
    padding: 12px 16px; margin: 8px 0; border: 1px solid var(--fs-rule); background: var(--fs-panel-soft); }
.fs-banner .ico { line-height: 0; margin-top: 2px; flex: none; }
.fs-banner .body { color: var(--fs-body); font-size: 0.93rem; line-height: 1.55; }
.fs-banner.info .ico { color: var(--fs-teal); }
.fs-banner.success { background: #EEF6F2; border-color: #BFDCCD; } .fs-banner.success .ico { color: var(--fs-pos); }
.fs-banner.warning { background: #FBF3E8; border-color: #E8CFA8; } .fs-banner.warning .ico { color: var(--fs-ochre); }
.fs-banner.error { background: #FBEEEC; border-color: #E6C3BF; } .fs-banner.error .ico { color: var(--fs-neg); }

/* ── Empty state ────────────────────────────────────────────────────── */
.fs-empty { padding: 34px 28px; border: 1px dashed var(--fs-rule-strong); border-radius: var(--fs-radius);
    background: var(--fs-panel-soft); display: flex; gap: 16px; align-items: flex-start; }
.fs-empty .ico { color: var(--fs-teal); line-height: 0; flex: none; margin-top: 2px; }
.fs-empty h4 { margin: 0 0 4px 0; padding: 0; font-size: 1.02rem; color: var(--fs-ink); }
.fs-empty p { color: var(--fs-muted); font-size: 0.92rem; max-width: 70ch; margin: 0; line-height: 1.55; }

/* ── Verdict with probability band (signature) ──────────────────────── */
.fs-verdict { border-top: 1px solid var(--fs-ink); padding-top: 14px; }
.fs-verdict .label { color: var(--fs-muted); font-size: 0.82rem; font-weight: 600; }
.fs-verdict .word { font-size: 2.6rem; font-weight: 700; line-height: 1.1; margin: 4px 0 2px 0; letter-spacing: -0.02em; }
.fs-verdict .conf { color: var(--fs-muted); font-size: 0.95rem; }
.fs-verdict .conf b { color: var(--fs-ink); }
.fs-band { display: flex; height: 26px; border-radius: 3px; overflow: hidden; margin-top: 16px; gap: 2px; }
.fs-band > div { height: 100%; min-width: 2px; }
.fs-band-legend { display: grid; grid-template-columns: repeat(3, 1fr); margin-top: 8px; gap: 8px; }
.fs-band-legend .k { font-size: 0.82rem; color: var(--fs-muted); }
.fs-band-legend .v { font-size: 1.02rem; font-weight: 650; color: var(--fs-ink); }

/* ── Term tables ────────────────────────────────────────────────────── */
table.fs-terms { width: 100%; border-collapse: collapse; font-size: 0.92rem; margin: 4px 0 6px 0; border: none; }
table.fs-terms th, table.fs-terms td { border: none; border-bottom: 1px solid var(--fs-rule); padding: 7px 8px; background: none; }
table.fs-terms thead th { color: var(--fs-muted); font-size: 0.78rem; font-weight: 600; text-align: left; border-bottom: 1px solid var(--fs-ink); }
table.fs-terms td.t { font-weight: 600; color: var(--fs-ink); padding-left: 0; }
table.fs-terms th.t { padding-left: 0; }
table.fs-terms td.b { width: 40%; }
table.fs-terms td.n { text-align: right; color: var(--fs-body); }
table.fs-terms .c { text-align: right; padding-right: 0; }

/* ── Contents list (home) ───────────────────────────────────────────── */
.fs-contents-row { border-top: 1px solid var(--fs-rule); padding: 10px 0 4px 0; }
.fs-contents-desc { color: var(--fs-muted); font-size: 0.88rem; margin: 2px 0 0 0; line-height: 1.5; }
.fs-tagline-exp { display: inline-block; font-size: 0.72rem; font-weight: 700; color: var(--fs-ochre);
    border: 1px solid #E8CFA8; background: #FBF3E8; padding: 0 6px; border-radius: 3px; margin-left: 6px; vertical-align: 2px; }

/* Page links as quiet actions */
[data-testid="stPageLink"] a { border-radius: 5px; padding: 0.3rem 0.4rem; }
[data-testid="stPageLink"] a:hover { background: var(--fs-teal-tint); }
[data-testid="stPageLink"] a p { font-weight: 600; color: var(--fs-teal); }

/* Tables, metrics, tabs, expanders */
[data-testid="stMetricValue"] { font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--fs-muted) !important; font-weight: 600 !important; }
[data-testid="stExpander"] details { border-radius: var(--fs-radius); }
hr { border-color: var(--fs-rule); margin: 1.8rem 0; }

@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"], .block-container { padding-top: 3.6rem; padding-left: 16px; padding-right: 16px; }
    .fs-masthead h1 { font-size: 1.6rem; }
    .fs-fig { border-left: none; padding-left: 0; border-top: 1px solid var(--fs-rule); }
    .fs-figs { border-bottom: none; }
    .fs-verdict .word { font-size: 2.1rem; }
    .fs-band-legend { grid-template-columns: 1fr; gap: 4px; }
}
</style>
"""


def inject_css() -> None:
    """Inject the global design-system CSS. Call once per run (the router does)."""
    st.markdown(BASE_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────

_GRID = "#E6ECEA"
_LINE = "#B9C6C3"
_FONT = {"color": "#2E3F3C", "family": "Public Sans, system-ui, sans-serif", "size": 12}


def _register_template() -> None:
    """Make the report style Plotly's default, so every chart (styled or not) shares it."""
    import plotly.io as pio  # pyre-ignore

    tpl = go.layout.Template(pio.templates["plotly_white"])
    tpl.layout.update(
        font=_FONT,
        colorway=CHART_SEQUENCE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": _LINE, "font": {"color": COLORS["text"]}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        title={"font": {"size": 14, "color": COLORS["text"]}, "x": 0, "xanchor": "left"},
    )
    tpl.layout.xaxis.update(gridcolor=_GRID, linecolor=_LINE, zerolinecolor=_GRID)
    tpl.layout.yaxis.update(gridcolor=_GRID, linecolor=_LINE, zerolinecolor=_GRID)
    pio.templates["finsight"] = tpl
    pio.templates.default = "finsight"


_register_template()


def style_fig(fig: go.Figure, height: int | None = None, title: str | None = None, legend: bool = True) -> go.Figure:
    """Apply the report chart theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_FONT,
        margin=dict(l=8, r=8, t=40 if title else 12, b=8),
        colorway=CHART_SEQUENCE,
        title=(
            {"text": title, "font": {"color": COLORS["text"], "size": 14}, "x": 0, "xanchor": "left"} if title else None
        ),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font={"size": 12}),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_LINE, font_size=12, font_color=COLORS["text"]),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=False, linecolor=_LINE, ticks="outside", tickcolor=_LINE, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, linecolor=_LINE, zeroline=False)
    return fig


def render_chart(fig: go.Figure) -> None:
    """Render a Plotly figure full-width with a clean config."""
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

_EXHIBIT_KEY = "_fs_exhibit_no"


def page_header(title: str, subtitle: str = "", eyebrow: str | None = None, meta: str = "") -> None:
    """
    Render the page masthead: title, standfirst and an optional metadata line.

    ``eyebrow`` is accepted for backwards compatibility and ignored; the title
    carries the page on its own. Resets the page's exhibit numbering.
    """
    st.session_state[_EXHIBIT_KEY] = 0
    sub_html = f"<p class='standfirst'>{subtitle}</p>" if subtitle else ""
    meta_html = f"<div class='meta'>{meta}</div>" if meta else ""
    st.markdown(
        f"<div class='fs-masthead'><h1>{title}</h1>{sub_html}{meta_html}</div>",
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str = "") -> None:
    """Render a ruled section heading with an optional one-line standfirst."""
    cap = f"<p class='cap'>{caption}</p>" if caption else ""
    st.markdown(f"<div class='fs-section'><h3>{title}</h3>{cap}</div>", unsafe_allow_html=True)


def exhibit(title: str, caption: str = "") -> None:
    """Render a numbered exhibit heading (numbering restarts on every page)."""
    n = st.session_state.get(_EXHIBIT_KEY, 0) + 1
    st.session_state[_EXHIBIT_KEY] = n
    cap = f"<p class='cap'>{caption}</p>" if caption else ""
    st.markdown(
        f"<div class='fs-section'><h3><span class='ex'>Exhibit {n}</span>{title}</h3>{cap}</div>",
        unsafe_allow_html=True,
    )


def source_note(text: str) -> None:
    """Render the source line under an exhibit."""
    st.markdown(f"<div class='fs-source'>Source: {text}</div>", unsafe_allow_html=True)


def kpi_strip(items: list[dict]) -> None:
    """
    Render key figures as one ruled row.

    Each item: {"label": str, "value": str|int, "sub": str?, "accent": hex?}.
    ``accent`` colours the value only when it is a sentiment colour.
    """
    semantic = set(SENTIMENT_COLORS.values())
    cells = []
    for it in items:
        colour = it.get("accent") if it.get("accent") in semantic else COLORS["text"]
        sub = f"<div class='sub'>{it['sub']}</div>" if it.get("sub") else ""
        cells.append(
            f"<div class='fs-fig'><div class='label'>{it['label']}</div>"
            f"<div class='value' style='color:{colour};'>{it['value']}</div>{sub}</div>"
        )
    st.markdown(f"<div class='fs-figs'>{''.join(cells)}</div>", unsafe_allow_html=True)


def metric_card(label: str, value, sub: str = "", accent: str | None = None) -> None:
    """Render a single key figure (use inside a column)."""
    kpi_strip([{"label": label, "value": value, "sub": sub, "accent": accent}])


def pill(text: str, kind: str = "muted") -> str:
    """Return an inline pill (kind: positive|negative|neutral|muted). Sentiment kinds carry a glyph."""
    if kind in SENTIMENT_GLYPHS:
        return f"<span class='pill {kind}'><span aria-hidden='true'>{SENTIMENT_GLYPHS[kind]}</span> {text}</span>"
    return f"<span class='pill {kind}'>{text}</span>"


def status_banner(text: str, kind: str = "info", icon_name: str | None = None, **_ignored) -> None:
    """Render a status banner (kind: info|success|warning|error)."""
    st.markdown(
        f"<div class='fs-banner {kind}' role='status'><div class='ico'>{icon(icon_name or kind)}</div>"
        f"<div class='body'>{text}</div></div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str = "", icon: str = "chart") -> None:  # noqa: A002  (kept name for callers)
    """Render an empty state. ``icon`` is a name from the SVG icon set."""
    name = icon if icon in _ICON_PATHS else "chart"
    st.markdown(
        f"<div class='fs-empty'><div class='ico'>{_svg(name, 26)}</div><div><h4>{title}</h4><p>{body}</p></div></div>",
        unsafe_allow_html=True,
    )


_svg = icon  # alias used where a parameter shadows ``icon``


def probability_band(probabilities: dict, predicted: str | None = None) -> str:
    """
    The signature: class probabilities as one segmented band.

    The predicted class is drawn solid, the others as tints, so the spread of
    the model's belief is visible at a glance.
    """
    order = [k for k in ("negative", "neutral", "positive") if k in probabilities]
    segs, legend = [], []
    for k in order:
        p = float(probabilities[k])
        solid = k == predicted
        fill = SENTIMENT_COLORS[k] if solid else SENTIMENT_TINTS[k]
        segs.append(f"<div title='{k} {p:.1%}' style='width:{max(p, 0.004) * 100:.2f}%;background:{fill};'></div>")
        weight = "700" if solid else "600"
        legend.append(
            f"<div><div class='k'>{SENTIMENT_GLYPHS[k]} {k.capitalize()}</div>"
            f"<div class='v fs-num' style='font-weight:{weight};color:{SENTIMENT_COLORS[k] if solid else COLORS['text']};'>{p:.1%}</div></div>"
        )
    return (
        f"<div class='fs-band' role='img' aria-label='Class probabilities'>{''.join(segs)}</div>"
        f"<div class='fs-band-legend'>{''.join(legend)}</div>"
    )


def verdict_card(label: str, confidence: float | None = None, probabilities: dict | None = None) -> None:
    """Render the sentiment verdict: glyph + word, confidence, and the probability band."""
    key = str(label).lower()
    color = get_sentiment_color(key)
    conf_html = (
        f"<div class='conf'>Model confidence <b class='fs-num'>{confidence:.1%}</b></div>"
        if confidence is not None
        else ""
    )
    band = probability_band(probabilities, key) if probabilities else ""
    st.markdown(
        f"<div class='fs-verdict'><div class='label'>Predicted sentiment</div>"
        f"<div class='word' style='color:{color};'><span aria-hidden='true'>{SENTIMENT_GLYPHS.get(key, '')}</span> "
        f"{html.escape(key.capitalize())}</div>{conf_html}{band}</div>",
        unsafe_allow_html=True,
    )


def term_table(rows, show_class: bool = True, weight_label: str = "Weight") -> str:
    """
    Vocabulary as a ruled table: term, weight bar, value and (optionally) the class it pushes toward.

    ``rows``: iterable of (term, weight, class).
    """
    rows = list(rows)
    top = max((abs(w) for _, w, _ in rows), default=1.0) or 1.0
    body = []
    for term, weight, cls in rows:
        bar = (
            f"<div style='height:6px;background:{get_sentiment_color(cls)};"
            f"width:{abs(weight) / top * 100:.0f}%;border-radius:2px;'></div>"
        )
        tag = f"<td class='c'>{sentiment_tag(cls)}</td>" if show_class else ""
        body.append(
            f"<tr><td class='t'>{html.escape(str(term))}</td><td class='b'>{bar}</td>"
            f"<td class='n fs-num'>{weight:.3f}</td>{tag}</tr>"
        )
    head_tag = "<th class='c'>Pushes toward</th>" if show_class else ""
    return (
        "<table class='fs-terms'><thead><tr><th class='t'>Term</th>"
        f"<th class='b'>{weight_label}</th><th></th>{head_tag}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


@contextmanager
def chart_container(title: str = "", caption: str = ""):
    """Context manager that renders an optional exhibit heading before a chart block."""
    if title:
        exhibit(title, caption)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS (sentiment-specific)
# ─────────────────────────────────────────────────────────────────────────────


def create_probability_chart(probabilities: dict) -> go.Figure:
    """Horizontal bar chart of class probabilities."""
    order = [k for k in ("positive", "neutral", "negative") if k in probabilities]
    values = [probabilities[k] * 100 for k in order]
    fig = go.Figure(
        go.Bar(
            y=[f"{SENTIMENT_GLYPHS[k]} {k.capitalize()}" for k in order],
            x=values,
            orientation="h",
            marker_color=[get_sentiment_color(k) for k in order],
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
            width=0.5,
        )
    )
    style_fig(fig, height=200, legend=False)
    fig.update_layout(xaxis_range=[0, 112], xaxis_title="Probability (%)")
    fig.update_xaxes(showgrid=True, gridcolor=_GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def create_gauge_chart(probabilities: dict, prediction: str) -> go.Figure:
    """Confidence gauge for the predicted class."""
    confidence = probabilities[prediction]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={"text": f"Confidence · {prediction.capitalize()}"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": _LINE},
                "bar": {"color": get_sentiment_color(prediction)},
                "bgcolor": COLORS["panel"],
                "borderwidth": 0,
            },
        )
    )
    style_fig(fig, height=250, legend=False)
    return fig


def interval_chart(rows: list[dict], reference: float | None = None, reference_label: str = "") -> go.Figure:
    """
    Dot-and-interval chart: one row per estimate, the point with its 95% band.

    Row labels sit above each band (not in a left gutter), so the chart keeps
    its full width on a phone. Each row: {"label", "value", "low", "high", "color"?}.
    """
    fig = go.Figure()
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - i
        c = r.get("color", COLORS["primary"])
        fig.add_trace(
            go.Scatter(
                x=[r["low"], r["high"]],
                y=[y, y],
                mode="lines",
                line={"color": c, "width": 9},
                opacity=0.25,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r["value"]],
                y=[y],
                mode="markers",
                marker={"color": c, "size": 12, "line": {"color": "#FFFFFF", "width": 2}},
                hovertemplate=f"{r['label']}<br>macro-F1 %{{x:.3f}}<br>95% interval {r['low']:.3f}–{r['high']:.3f}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=0,
            y=y + 0.38,
            xref="x",
            yref="y",
            xanchor="left",
            showarrow=False,
            text=f"{html.escape(r['label'])}",
            font={"size": 12, "color": COLORS["text_body"]},
        )
        fig.add_annotation(
            x=r["high"],
            y=y,
            xref="x",
            yref="y",
            xanchor="left",
            xshift=8,
            showarrow=False,
            text=f"<b>{r['value']:.2f}</b>",
            font={"size": 13, "color": COLORS["text"]},
        )
    if reference is not None:
        fig.add_vline(x=reference, line={"color": COLORS["text_faint"], "dash": "dot", "width": 1.5})
        fig.add_annotation(
            x=reference,
            y=0.35,
            xref="x",
            yref="y",
            xanchor="left",
            xshift=4,
            showarrow=False,
            text=reference_label,
            font={"size": 11, "color": COLORS["text_muted"]},
        )
    style_fig(fig, height=max(220, 62 * n + 70), legend=False)
    fig.update_layout(margin={"t": 10, "l": 8, "r": 40, "b": 8})
    fig.update_xaxes(
        range=[0, 1.04], showgrid=True, gridcolor=_GRID, title="Macro-F1 (dot) with 95% bootstrap interval (band)"
    )
    fig.update_yaxes(
        range=[0.2, n + 0.75], showticklabels=False, showgrid=False, zeroline=False, ticks="", showline=False
    )
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


ASSETS = Path(__file__).parent / "assets"

# Default engine: the fastest well-calibrated baseline (CV macro-F1 0.838).
DEFAULT_MODEL = "baseline_svm"

_SESSION_KEY = "_fs_active_model"


def setup_sidebar():
    """
    Render the persistent sidebar (brand, model selector, status).

    Called once per run from the navigation entrypoint (``app.py``), so the
    model selection survives page switches. Pages read it via ``active_model()``.

    Returns
    -------
    tuple[str | None, SentimentPredictor | None]
    """
    st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))

    available_models = _list_models()
    if not available_models:
        st.sidebar.error("No trained models found. Run `python src/train.py --model baselines`.")
        st.session_state[_SESSION_KEY] = (None, None)
        return None, None

    st.sidebar.markdown("<div class='fs-side-label'>Model engine</div>", unsafe_allow_html=True)
    default_index = available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0
    selected_model = st.sidebar.selectbox(
        "Model",
        available_models,
        index=default_index,
        key="fs_model_choice",
        format_func=model_label,
        label_visibility="collapsed",
        help="The model used across every analysis page.",
    )

    predictor = None
    load_error = None
    try:
        with st.spinner("Loading model…"):
            predictor = load_predictor(selected_model)
    except Exception as exc:  # noqa: BLE001
        load_error = str(exc)

    online = predictor is not None
    dot = "#6FD6A8" if online else "#F2948A"
    status = "Engine ready" if online else "Engine unavailable"
    is_transformer = not selected_model.startswith("baseline_")
    family = "Transformer · 110M params" if is_transformer else "TF-IDF · classical ML"
    st.sidebar.markdown(
        "<div class='fs-side-status'>"
        f"<div class='row'><span class='dot' style='background:{dot};'></span>{status}</div>"
        f"<div class='row'><span class='dot' style='background:#8FD6C9;'></span>{family}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if load_error:
        st.sidebar.error(f"Failed to load model: {load_error}")

    if selected_model == "finbert":
        note = "Fine-tuned on real financial news headlines (Twitter Financial News, MIT)."
    elif is_transformer:
        note = "ProsusAI/finbert, used zero-shot (no task-specific fine-tuning)."
    else:
        note = "Trained on Financial PhraseBank (2,264 sentences, 100% annotator agreement)."
    st.sidebar.markdown(f"<div class='fs-side-note'>{note}</div>", unsafe_allow_html=True)

    st.session_state[_SESSION_KEY] = (selected_model, predictor)
    return selected_model, predictor


def active_model():
    """
    Return ``(model_key, predictor)`` chosen in the sidebar.

    Stops the page with a friendly message if no model could be loaded.
    """
    selected_model, predictor = st.session_state.get(_SESSION_KEY, (None, None))
    if predictor is None:
        empty_state(
            "No model loaded",
            "Pick a model in the sidebar. If none are listed, train the baselines with "
            "<code>python src/train.py --model baselines</code>.",
            icon="warning",
        )
        st.stop()
    return selected_model, predictor

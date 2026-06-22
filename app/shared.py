"""
Shared utilities for the Financial Sentiment Analyzer multipage app.

This is a thin compatibility layer over the centralized design system in
``ui.py``. It preserves the historical import surface used by the pages
(``inject_css``, ``setup_sidebar``, ``get_sentiment_color``,
``create_probability_chart``, ``create_gauge_chart``) while exposing the new
reusable components (``page_header``, ``section_header``, ``kpi_strip``,
``metric_card``, ``pill``, ``status_banner``, ``empty_state``, ``verdict_card``,
``style_fig``, ``render_chart``) from a single place.
"""

import sys
from pathlib import Path

# Add src to path for imports (kept for backwards compatibility).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ui import (  # noqa: F401  (re-exported for pages)
    CHART_SEQUENCE,
    COLORS,
    SENTIMENT_COLORS,
    chart_container,
    create_gauge_chart,
    create_probability_chart,
    empty_state,
    get_sentiment_color,
    inject_css,
    kpi_strip,
    load_predictor,
    metric_card,
    model_label,
    page_header,
    pill,
    render_chart,
    section_header,
    setup_sidebar,
    status_banner,
    style_fig,
    verdict_card,
)

__all__ = [
    "CHART_SEQUENCE",
    "COLORS",
    "SENTIMENT_COLORS",
    "chart_container",
    "create_gauge_chart",
    "create_probability_chart",
    "empty_state",
    "get_sentiment_color",
    "inject_css",
    "kpi_strip",
    "load_predictor",
    "metric_card",
    "model_label",
    "page_header",
    "pill",
    "render_chart",
    "section_header",
    "setup_sidebar",
    "status_banner",
    "style_fig",
    "verdict_card",
]

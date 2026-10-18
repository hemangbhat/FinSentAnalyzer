"""
Financial Sentiment Analyzer — Streamlit entrypoint and router.

Uses ``st.navigation`` so the sidebar (brand + model selector) is rendered once
here and persists across pages; each page reads the selection via
``shared.active_model()``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # pyre-ignore

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="FinSight · Sentiment Intelligence",
    page_icon=str(APP_DIR / "assets" / "icon.svg"),
    layout="wide",
    initial_sidebar_state="auto",
)

from shared import inject_css, setup_sidebar  # pyre-ignore  # noqa: E402


def _page(file: str, title: str, icon: str, url_path: str, default: bool = False):
    return st.Page(str(APP_DIR / file), title=title, icon=icon, url_path=url_path, default=default)


PAGES = {
    "Overview": [
        _page("home.py", "Home", ":material/space_dashboard:", "home", default=True),
    ],
    "Analyze": [
        _page("views/1_Single_Analysis.py", "Single Analysis", ":material/edit_note:", "Single_Analysis"),
        _page("views/2_Batch_Processing.py", "Batch Processing", ":material/upload_file:", "Batch_Processing"),
        _page("views/10_Ask_the_News.py", "Ask the News", ":material/forum:", "Ask_the_News"),
    ],
    "Explain": [
        _page("views/3_Explainability.py", "Explainability", ":material/troubleshoot:", "Explainability"),
        _page("views/4_Word_Insights.py", "Word Insights", ":material/match_word:", "Word_Insights"),
        _page("views/5_Deep_Analysis.py", "Deep Analysis", ":material/psychology:", "Deep_Analysis"),
    ],
    "Evaluate": [
        _page("views/6_Model_Info.py", "Model Registry", ":material/leaderboard:", "Model_Info"),
        _page("views/8_Error_Analysis.py", "Error Analysis", ":material/rule:", "Error_Analysis"),
        _page("views/7_Sentiment_Trends.py", "Generalization", ":material/public:", "Sentiment_Trends"),
    ],
    "Labs": [
        _page(
            "views/9_Stock_Prediction_Extension.py",
            "Stock Prediction",
            ":material/science:",
            "Stock_Prediction_Extension",
        ),
    ],
}

nav = st.navigation(PAGES)
inject_css()
setup_sidebar()
nav.run()

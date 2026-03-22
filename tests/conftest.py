"""
Shared fixtures for Financial Sentiment Analyzer tests.
"""

import sys
import os
from pathlib import Path

import pytest

# Add src to path so imports work
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture
def positive_texts():
    """Sample positive financial texts."""
    return [
        "The company reported strong earnings growth of 25%, beating analyst expectations.",
        "Revenue increased significantly, driven by robust demand and innovation.",
        "Profits soared to record highs as the company delivered outstanding results.",
    ]


@pytest.fixture
def negative_texts():
    """Sample negative financial texts."""
    return [
        "Revenue declined sharply by 15% due to weak demand and supply chain disruptions.",
        "The company faces potential litigation risks that could negatively impact shareholder value.",
        "Losses mounted as the firm struggled with declining sales and rising costs.",
    ]


@pytest.fixture
def neutral_texts():
    """Sample neutral financial texts."""
    return [
        "The quarterly results were in line with expectations, with revenue remaining flat.",
        "The company maintained its dividend policy unchanged from the previous quarter.",
        "Operations continued as planned with no significant deviation from forecasts.",
    ]


@pytest.fixture
def mixed_texts(positive_texts, negative_texts, neutral_texts):
    """Mix of all sentiment types."""
    return positive_texts[:1] + negative_texts[:1] + neutral_texts[:1]


@pytest.fixture
def project_root():
    """Path to project root."""
    return Path(__file__).parent.parent


@pytest.fixture
def models_dir(project_root):
    """Path to models directory."""
    return project_root / "models"

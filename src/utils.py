"""
Shared utilities for Financial Sentiment Analysis.
Central module for constants, path helpers, logging, and shared functions.
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


# =============================================================================
# CONSTANTS
# =============================================================================

# Label mapping used across all modules
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

# Supported baseline classifier names
BASELINE_CLASSIFIERS = [
    "logreg", "naive_bayes", "svm",
    "random_forest", "gradient_boosting", "mlp", "ensemble",
]

# Supported transformer model names
TRANSFORMER_MODELS = ["finbert", "distilbert", "roberta", "bert"]


# =============================================================================
# PATH HELPERS
# =============================================================================

def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root (parent of the 'src' directory)
    """
    return Path(__file__).parent.parent


def get_model_dir() -> Path:
    """Get the models directory, creating it if needed."""
    model_dir = get_project_root() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_data_dir(subdir: str = "") -> Path:
    """
    Get a data directory path.

    Args:
        subdir: Optional subdirectory (e.g., 'raw', 'processed')

    Returns:
        Path to the data directory
    """
    data_dir = get_project_root() / "data"
    if subdir:
        data_dir = data_dir / subdir
    return data_dir


def get_results_dir() -> Path:
    """Get the results directory, creating it if needed."""
    results_dir = get_project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(
    name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (default: INFO)
        fmt: Log message format
        datefmt: Date format for timestamps

    Returns:
        Configured logger instance

    Example:
        logger = setup_logging(__name__)
        logger.info("Model training started")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt, datefmt=datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


# =============================================================================
# MODEL METRICS HELPERS
# =============================================================================

# Model descriptive metadata (no hardcoded metrics — those come from evaluation results only)
_MODEL_METADATA = {
    "baseline_logreg": {
        "name": "Logistic Regression",
        "type": "TF-IDF + Classifier",
        "speed": "Very Fast",
        "features": [
            "TF-IDF vectorization (unigrams + bigrams)",
            "Max 10,000 features",
            "Balanced class weights",
            "Linear decision boundary",
        ],
    },
    "baseline_naive_bayes": {
        "name": "Naive Bayes",
        "type": "TF-IDF + Classifier",
        "speed": "Very Fast",
        "features": [
            "TF-IDF vectorization",
            "Multinomial distribution",
            "Good for text classification",
            "Alpha smoothing = 0.1",
        ],
    },
    "baseline_svm": {
        "name": "Support Vector Machine (SVM)",
        "type": "TF-IDF + Classifier",
        "speed": "Fast",
        "features": [
            "TF-IDF vectorization",
            "Linear kernel",
            "Maximum margin classifier",
            "Balanced class weights",
            "Best for high-dimensional sparse data",
        ],
    },
    "baseline_random_forest": {
        "name": "Random Forest",
        "type": "TF-IDF + Ensemble",
        "speed": "Medium",
        "features": [
            "200 decision trees",
            "Max depth 50",
            "Ensemble voting",
            "Handles non-linear patterns",
        ],
    },
    "baseline_gradient_boosting": {
        "name": "Gradient Boosting",
        "type": "TF-IDF + Boosting",
        "speed": "Medium",
        "features": [
            "100 boosting iterations",
            "Sequential error correction",
            "Learning rate 0.1",
            "Strong on structured features",
        ],
    },
    "baseline_mlp": {
        "name": "Multi-Layer Perceptron (Neural Network)",
        "type": "TF-IDF + Deep Learning",
        "speed": "Medium",
        "features": [
            "2 hidden layers (256, 128 neurons)",
            "Early stopping",
            "Non-linear activation",
            "Learns complex patterns",
        ],
    },
    "baseline_ensemble": {
        "name": "Voting Ensemble",
        "type": "TF-IDF + Combined Models",
        "speed": "Slow",
        "features": [
            "Combines LogReg + SVM + Random Forest",
            "Soft voting (probability-based)",
            "More robust predictions",
        ],
    },
    "finbert_pretrained": {
        "name": "FinBERT (Pre-trained)",
        "type": "Transformer",
        "speed": "Slow (CPU) / Fast (GPU)",
        "features": [
            "Pre-trained on financial text",
            "ProsusAI/finbert from HuggingFace",
            "Understands financial context",
            "No training required",
            "110M parameters",
        ],
    },
}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get model information — tries dynamic evaluation results first,
    falls back to descriptive metadata only (no fake numbers).

    Args:
        model_name: Model identifier (e.g., 'baseline_svm')

    Returns:
        Dictionary with model info (name, type, accuracy, f1 scores, speed, features)
    """
    meta = _MODEL_METADATA.get(model_name, {})
    base_info = {
        "name": meta.get("name", model_name),
        "type": meta.get("type", "Unknown"),
        "accuracy": "N/A",
        "f1_macro": "N/A",
        "f1_weighted": "N/A",
        "speed": meta.get("speed", "N/A"),
        "features": meta.get("features", []),
    }

    # Try loading dynamic evaluation results
    results_path = get_results_dir() / "evaluation_results.json"
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
            for result in results:
                if result.get("name") == model_name:
                    base_info["accuracy"] = f"{result['accuracy']:.1%}"
                    base_info["f1_macro"] = f"{result['f1_macro']:.3f}"
                    base_info["f1_weighted"] = f"{result['f1_weighted']:.3f}"
                    break
        except (json.JSONDecodeError, KeyError):
            pass  # Keep N/A values

    return base_info


def get_all_model_info(available_models: list) -> Dict[str, Dict[str, Any]]:
    """
    Get info for all available models.

    Args:
        available_models: List of model name strings

    Returns:
        Dictionary mapping model names to their info dicts
    """
    return {name: get_model_info(name) for name in available_models}

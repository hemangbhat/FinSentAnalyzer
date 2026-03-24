"""
Explainability module for Financial Sentiment Analysis.
Provides word importance and attention-based explanations.
Uses financial lexicon for reliable sentiment direction detection.
"""

import numpy as np
import joblib
from typing import List, Tuple
import re

from utils import get_project_root, get_model_dir, setup_logging, LABEL_MAP_INV

# Import financial lexicons for reliable sentiment detection
try:
    from lexicons import FINANCIAL_POSITIVE, FINANCIAL_NEGATIVE, FINANCIAL_UNCERTAINTY
    LEXICON_AVAILABLE = True
except ImportError:
    LEXICON_AVAILABLE = False
    FINANCIAL_POSITIVE = set()
    FINANCIAL_NEGATIVE = set()
    FINANCIAL_UNCERTAINTY = set()

logger = setup_logging(__name__)


def get_word_importance_baseline(
    text: str,
    model_name: str = "baseline_svm"
) -> List[Tuple[str, float, str]]:
    """
    Get word importance for baseline TF-IDF models.
    Uses TF-IDF scores for importance ranking and financial lexicon for sentiment direction.

    Args:
        text: Input text
        model_name: Baseline model name

    Returns:
        List of (word, importance_score, sentiment_direction) tuples
    """
    model_path = get_model_dir() / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    pipeline = joblib.load(model_path)
    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    # Transform text
    tfidf_vector = tfidf.transform([text])

    # Get feature names and their TF-IDF values for this text
    feature_names = tfidf.get_feature_names_out()
    tfidf_values = tfidf_vector.toarray()[0]

    # Get non-zero features
    non_zero_idx = np.where(tfidf_values > 0)[0]

    # Get classifier coefficients
    if hasattr(classifier, "coef_"):
        coef = classifier.coef_
    else:
        coef = None

    # Calculate word importance
    word_scores = []
    for idx in non_zero_idx:
        word = feature_names[idx]
        tfidf_val = tfidf_values[idx]

        # Clean word for lexicon lookup (handle bigrams by checking each part)
        word_parts = word.lower().split()
        word_clean = word_parts[0] if word_parts else word.lower()

        # PRIORITY: Use financial lexicon for sentiment direction (more reliable)
        if LEXICON_AVAILABLE and word_clean in FINANCIAL_POSITIVE:
            direction = "positive"
        elif LEXICON_AVAILABLE and word_clean in FINANCIAL_NEGATIVE:
            direction = "negative"
        elif coef is not None and coef.shape[0] == 3:
            # Fallback to model coefficients for words not in lexicon
            pos_coef = coef[2, idx]  # positive class
            neg_coef = coef[0, idx]  # negative class
            neu_coef = coef[1, idx]  # neutral class

            # Determine dominant sentiment direction from model
            max_coef = max(pos_coef, neg_coef, neu_coef, key=abs)
            if max_coef == pos_coef:
                direction = "positive"
            elif max_coef == neg_coef:
                direction = "negative"
            else:
                direction = "neutral"
        else:
            direction = "neutral"

        # Calculate importance score
        if coef is not None and coef.shape[0] == 3:
            pos_coef = coef[2, idx]
            neg_coef = coef[0, idx]
            neu_coef = coef[1, idx]
            importance = tfidf_val * max(abs(pos_coef), abs(neg_coef), abs(neu_coef))
        else:
            importance = tfidf_val

        word_scores.append((word, float(importance), direction))

    # Sort by importance
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return word_scores[:20]  # Top 20 words


def explain_prediction_baseline(
    text: str,
    model_name: str = "baseline_svm"
) -> dict:
    """
    Explain a baseline model prediction with lexicon-enhanced word detection.

    Args:
        text: Input text
        model_name: Model name

    Returns:
        Dictionary with prediction and explanation
    """
    model_path = get_model_dir() / f"{model_name}.joblib"
    pipeline = joblib.load(model_path)

    # Get prediction
    pred = pipeline.predict([text])[0]
    pred_label = LABEL_MAP_INV[pred]

    # Get probabilities if available
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        probabilities = {LABEL_MAP_INV[i]: float(p) for i, p in enumerate(proba)}
    elif hasattr(pipeline.named_steps["classifier"], "decision_function"):
        decisions = pipeline.decision_function([text])[0]
        exp_decisions = np.exp(decisions - np.max(decisions))
        proba = exp_decisions / exp_decisions.sum()
        probabilities = {LABEL_MAP_INV[i]: float(p) for i, p in enumerate(proba)}
    else:
        probabilities = None

    # Get word importance (now uses lexicon for direction)
    word_importance = get_word_importance_baseline(text, model_name)

    # Also get lexicon-based words directly from text
    words_in_text = [re.sub(r'[^\w]', '', w.lower()) for w in text.split()]
    words_in_text = [w for w in words_in_text if w]

    lexicon_positive = [w for w in words_in_text if w in FINANCIAL_POSITIVE]
    lexicon_negative = [w for w in words_in_text if w in FINANCIAL_NEGATIVE]
    lexicon_uncertainty = [w for w in words_in_text if w in FINANCIAL_UNCERTAINTY]

    # Separate by direction
    positive_words = [(w, s) for w, s, d in word_importance if d == "positive"]
    negative_words = [(w, s) for w, s, d in word_importance if d == "negative"]
    neutral_words = [(w, s) for w, s, d in word_importance if d == "neutral"]

    # Add lexicon words that weren't captured by TF-IDF (might be out-of-vocabulary)
    existing_pos = {w for w, s in positive_words}
    existing_neg = {w for w, s in negative_words}
    for w in lexicon_positive:
        if w not in existing_pos:
            positive_words.append((w, 0.1))  # Small score for lexicon-only words
    for w in lexicon_negative:
        if w not in existing_neg:
            negative_words.append((w, 0.1))

    return {
        "text": text,
        "prediction": pred_label,
        "probabilities": probabilities,
        "word_importance": word_importance,
        "positive_words": positive_words[:5],
        "negative_words": negative_words[:5],
        "neutral_words": neutral_words[:5],
        "lexicon_positive": lexicon_positive,
        "lexicon_negative": lexicon_negative,
        "lexicon_uncertainty": lexicon_uncertainty,
    }


def highlight_text(text: str, word_importance: List[Tuple[str, float, str]]) -> str:
    """
    Create HTML with highlighted important words.

    Args:
        text: Original text
        word_importance: List of (word, score, direction) tuples

    Returns:
        HTML string with highlighted words
    """
    colors = {
        "positive": "rgba(40, 167, 69, 0.2)",  # dark mode positive
        "negative": "rgba(220, 53, 69, 0.2)",  # dark mode negative
        "neutral": "rgba(0, 123, 255, 0.2)",   # dark mode neutral
    }

    # Create word to color mapping
    word_colors = {}
    for word, score, direction in word_importance[:15]:  # Top 15
        # Handle both single words and bigrams
        word_colors[word.lower()] = colors.get(direction, "#ffffff")

    # Tokenize and highlight
    words = text.split()
    highlighted = []

    for word in words:
        clean_word = re.sub(r"[^\w]", "", word.lower())
        if clean_word in word_colors:
            color = word_colors[clean_word]
            highlighted.append(f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{word}</span>')
        else:
            highlighted.append(word)

    return " ".join(highlighted)


def get_feature_importance_summary(model_name: str = "baseline_svm") -> dict:
    """
    Get overall feature importance from the model.

    Args:
        model_name: Model name

    Returns:
        Dictionary with top positive and negative features
    """
    model_path = get_model_dir() / f"{model_name}.joblib"
    pipeline = joblib.load(model_path)

    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        return {"error": "Model doesn't support feature importance"}

    feature_names = tfidf.get_feature_names_out()
    coef = classifier.coef_

    results = {}

    if coef.shape[0] == 3:
        # Multi-class: get top features for each class
        for class_idx, class_name in LABEL_MAP_INV.items():
            class_coef = coef[class_idx]
            top_idx = np.argsort(class_coef)[-10:][::-1]
            top_features = [(feature_names[i], float(class_coef[i])) for i in top_idx]
            results[f"top_{class_name}"] = top_features

    return results


if __name__ == "__main__":
    # Test explainability
    test_text = "The company reported strong earnings growth this quarter."

    print("Testing explainability...")
    result = explain_prediction_baseline(test_text, "baseline_svm")

    print(f"\nText: {result['text']}")
    print(f"Prediction: {result['prediction']}")

    if result["probabilities"]:
        print("\nProbabilities:")
        for label, prob in result["probabilities"].items():
            print(f"  {label}: {prob:.2%}")

    print("\nTop Important Words:")
    for word, score, direction in result["word_importance"][:10]:
        print(f"  {word}: {score:.4f} ({direction})")

    print("\nPositive indicators:", [w for w, s in result["positive_words"]])
    print("Negative indicators:", [w for w, s in result["negative_words"]])

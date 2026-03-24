"""
Explainability module for Financial Sentiment Analysis.
Provides word importance and attention-based explanations.
"""

import numpy as np
import joblib
from typing import List, Tuple
import re

from utils import get_project_root, get_model_dir, setup_logging, LABEL_MAP_INV

logger = setup_logging(__name__)


def get_word_importance_baseline(
    text: str,
    model_name: str = "baseline_svm"
) -> List[Tuple[str, float, str]]:
    """
    Get word importance for baseline TF-IDF models.
    Uses feature coefficients to determine word importance.

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
        return []

    # Calculate word importance
    word_scores = []
    for idx in non_zero_idx:
        word = feature_names[idx]
        tfidf_val = tfidf_values[idx]

        # Get coefficient contribution for each class
        # coef shape: (n_classes, n_features) for multi-class
        if coef.shape[0] == 3:  # Multi-class
            pos_coef = coef[2, idx]  # positive class
            neg_coef = coef[0, idx]  # negative class
            neu_coef = coef[1, idx]  # neutral class

            # Determine dominant sentiment direction
            max_coef = max(pos_coef, neg_coef, neu_coef, key=abs)
            if max_coef == pos_coef:
                direction = "positive"
            elif max_coef == neg_coef:
                direction = "negative"
            else:
                direction = "neutral"

            importance = tfidf_val * abs(max_coef)
        else:
            importance = tfidf_val * abs(coef[0, idx])
            direction = "positive" if coef[0, idx] > 0 else "negative"

        word_scores.append((word, float(importance), direction))

    # Sort by importance
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return word_scores[:20]  # Top 20 words


def explain_prediction_baseline(
    text: str,
    model_name: str = "baseline_svm"
) -> dict:
    """
    Explain a baseline model prediction.

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

    # Get word importance
    word_importance = get_word_importance_baseline(text, model_name)

    # Separate by direction
    positive_words = [(w, s) for w, s, d in word_importance if d == "positive"]
    negative_words = [(w, s) for w, s, d in word_importance if d == "negative"]
    neutral_words = [(w, s) for w, s, d in word_importance if d == "neutral"]

    return {
        "text": text,
        "prediction": pred_label,
        "probabilities": probabilities,
        "word_importance": word_importance,
        "positive_words": positive_words[:5],
        "negative_words": negative_words[:5],
        "neutral_words": neutral_words[:5],
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

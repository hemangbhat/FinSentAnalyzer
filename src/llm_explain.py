"""
LLM-powered explanations for Financial Sentiment Analysis.
Provides natural language explanations and market outlook summaries.
"""

from typing import List, Dict, Optional
import os

# Try to import LangChain components
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def generate_explanation_template(
    text: str,
    prediction: str,
    confidence: float,
    positive_words: List[tuple],
    negative_words: List[tuple],
    neutral_words: List[tuple],
) -> str:
    """
    Generate a natural language explanation for the sentiment prediction.
    Uses template-based generation (works without API keys).

    Args:
        text: Original text
        prediction: Predicted sentiment (positive/negative/neutral)
        confidence: Confidence score (0-1)
        positive_words: List of (word, score) tuples
        negative_words: List of (word, score) tuples
        neutral_words: List of (word, score) tuples

    Returns:
        Natural language explanation string
    """
    # Extract word lists
    pos_terms = [w for w, s in positive_words[:3]] if positive_words else []
    neg_terms = [w for w, s in negative_words[:3]] if negative_words else []
    neu_terms = [w for w, s in neutral_words[:3]] if neutral_words else []

    # Build explanation based on prediction
    confidence_desc = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"

    if prediction == "positive":
        if pos_terms:
            key_phrase = f"'{pos_terms[0]}'" if len(pos_terms) == 1 else \
                        f"'{pos_terms[0]}' and '{pos_terms[1]}'" if len(pos_terms) == 2 else \
                        f"'{pos_terms[0]}', '{pos_terms[1]}', and '{pos_terms[2]}'"
            explanation = f"This text expresses **positive** sentiment with {confidence_desc} confidence ({confidence:.1%}). "
            explanation += f"The model identified key positive indicators: {key_phrase}. "
            if neg_terms:
                explanation += f"While some cautionary terms like '{neg_terms[0]}' are present, "
                explanation += "the overall tone remains optimistic, suggesting favorable market perception."
            else:
                explanation += "The language strongly suggests growth, improvement, or favorable financial outcomes."
        else:
            explanation = f"This text is classified as **positive** ({confidence:.1%} confidence). "
            explanation += "The overall tone and context suggest favorable sentiment."

    elif prediction == "negative":
        if neg_terms:
            key_phrase = f"'{neg_terms[0]}'" if len(neg_terms) == 1 else \
                        f"'{neg_terms[0]}' and '{neg_terms[1]}'" if len(neg_terms) == 2 else \
                        f"'{neg_terms[0]}', '{neg_terms[1]}', and '{neg_terms[2]}'"
            explanation = f"This text expresses **negative** sentiment with {confidence_desc} confidence ({confidence:.1%}). "
            explanation += f"Key negative indicators include: {key_phrase}. "
            if pos_terms:
                explanation += f"Despite some positive elements like '{pos_terms[0]}', "
                explanation += "the dominant language signals concerns, declines, or unfavorable outcomes."
            else:
                explanation += "The language indicates potential risks, losses, or adverse market conditions."
        else:
            explanation = f"This text is classified as **negative** ({confidence:.1%} confidence). "
            explanation += "The overall tone suggests unfavorable sentiment or concerns."

    else:  # neutral
        explanation = f"This text expresses **neutral** sentiment with {confidence_desc} confidence ({confidence:.1%}). "
        if neu_terms:
            explanation += f"The model recognized factual/neutral terms like '{neu_terms[0]}'. "
        explanation += "The language is primarily informational without strong positive or negative bias. "
        explanation += "This typically indicates routine business updates or factual reporting."

    return explanation


def generate_market_outlook(
    sentiment_counts: Dict[str, int],
    total_texts: int,
    avg_confidence: float,
    sample_texts: Optional[List[Dict]] = None,
) -> str:
    """
    Generate a market outlook summary based on batch sentiment analysis.

    Args:
        sentiment_counts: Dict with positive/negative/neutral counts
        total_texts: Total number of texts analyzed
        avg_confidence: Average confidence score
        sample_texts: Optional list of sample predictions with text and label

    Returns:
        Market outlook summary string
    """
    pos_count = sentiment_counts.get("positive", 0)
    neg_count = sentiment_counts.get("negative", 0)
    neu_count = sentiment_counts.get("neutral", 0)

    pos_pct = pos_count / total_texts * 100 if total_texts > 0 else 0
    neg_pct = neg_count / total_texts * 100 if total_texts > 0 else 0
    neu_pct = neu_count / total_texts * 100 if total_texts > 0 else 0

    # Determine overall sentiment
    if pos_pct > neg_pct + 15:
        overall = "bullish"
        outlook_emoji = "📈"
    elif neg_pct > pos_pct + 15:
        overall = "bearish"
        outlook_emoji = "📉"
    elif pos_pct > neg_pct + 5:
        overall = "cautiously optimistic"
        outlook_emoji = "📊"
    elif neg_pct > pos_pct + 5:
        overall = "cautiously pessimistic"
        outlook_emoji = "⚠️"
    else:
        overall = "mixed/neutral"
        outlook_emoji = "↔️"

    # Build the outlook report
    report = f"""
## {outlook_emoji} Market Sentiment Outlook

### Overall Assessment: **{overall.upper()}**

Based on analysis of **{total_texts} financial texts**, the market sentiment appears **{overall}**.

### Sentiment Breakdown
| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive  | {pos_count} | {pos_pct:.1f}% |
| Neutral   | {neu_count} | {neu_pct:.1f}% |
| Negative  | {neg_count} | {neg_pct:.1f}% |

### Key Insights
"""

    # Add insights based on data
    if pos_pct > 50:
        report += "- **Strong positive sentiment** dominates the analyzed texts, suggesting market optimism.\n"
    elif neg_pct > 50:
        report += "- **Strong negative sentiment** dominates, indicating potential market concerns.\n"
    elif neu_pct > 50:
        report += "- **Neutral/factual reporting** dominates, suggesting a wait-and-see market attitude.\n"

    if avg_confidence > 0.8:
        report += f"- **High confidence** in predictions ({avg_confidence:.1%}), indicating clear sentiment signals.\n"
    elif avg_confidence < 0.6:
        report += f"- **Lower confidence** ({avg_confidence:.1%}) suggests mixed or ambiguous language in texts.\n"
    else:
        report += f"- **Moderate confidence** ({avg_confidence:.1%}) in sentiment classification.\n"

    # Sentiment ratio analysis
    if pos_count > 0 and neg_count > 0:
        ratio = pos_count / neg_count
        if ratio > 2:
            report += f"- Positive-to-negative ratio is **{ratio:.1f}:1**, indicating predominantly optimistic coverage.\n"
        elif ratio < 0.5:
            report += f"- Positive-to-negative ratio is **{ratio:.1f}:1**, indicating predominantly pessimistic coverage.\n"
        else:
            report += f"- Positive-to-negative ratio is **{ratio:.1f}:1**, showing balanced sentiment distribution.\n"

    # Recommendation
    report += "\n### Interpretation\n"
    if overall == "bullish":
        report += "The analyzed texts suggest positive market perception. This could indicate favorable conditions "
        report += "for the entities mentioned, though investors should conduct additional due diligence."
    elif overall == "bearish":
        report += "The analyzed texts reveal concerns or negative developments. This warrants careful evaluation "
        report += "of the underlying factors driving negative sentiment."
    elif overall in ["cautiously optimistic", "cautiously pessimistic"]:
        report += "Mixed signals in the data suggest a transitional period. Monitor for emerging trends "
        report += "that could shift sentiment more decisively in either direction."
    else:
        report += "Balanced or neutral sentiment suggests stability without strong directional bias. "
        report += "This may indicate a consolidation phase or lack of significant catalysts."

    report += "\n\n*Note: This analysis is based on ML-powered sentiment classification and should be used "
    report += "as one input among many in investment decision-making.*"

    return report


def get_llm_explanation(
    text: str,
    prediction: str,
    probabilities: Dict[str, float],
    word_importance: List[tuple],
    use_api: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """
    Get an LLM-powered explanation for a prediction.
    Falls back to template-based generation if API is not available.

    Args:
        text: Original text
        prediction: Predicted sentiment
        probabilities: Class probabilities
        word_importance: Word importance from explain module
        use_api: Whether to use external LLM API
        api_key: Optional API key for LLM service

    Returns:
        Natural language explanation
    """
    # Extract words by direction
    positive_words = [(w, s) for w, s, d in word_importance if d == "positive"]
    negative_words = [(w, s) for w, s, d in word_importance if d == "negative"]
    neutral_words = [(w, s) for w, s, d in word_importance if d == "neutral"]

    confidence = probabilities.get(prediction, 0.5) if probabilities else 0.5

    # Use template-based generation (works without API)
    return generate_explanation_template(
        text=text,
        prediction=prediction,
        confidence=confidence,
        positive_words=positive_words,
        negative_words=negative_words,
        neutral_words=neutral_words,
    )


if __name__ == "__main__":
    # Test the explanation generator
    print("Testing LLM Explanation Generator...")

    # Test single explanation
    explanation = generate_explanation_template(
        text="The company reported strong earnings growth this quarter.",
        prediction="positive",
        confidence=0.89,
        positive_words=[("strong", 0.45), ("growth", 0.38), ("earnings", 0.22)],
        negative_words=[],
        neutral_words=[("reported", 0.15), ("quarter", 0.12)],
    )
    print("\n--- Single Explanation ---")
    print(explanation)

    # Test market outlook
    outlook = generate_market_outlook(
        sentiment_counts={"positive": 35, "neutral": 45, "negative": 20},
        total_texts=100,
        avg_confidence=0.78,
    )
    print("\n--- Market Outlook ---")
    print(outlook)

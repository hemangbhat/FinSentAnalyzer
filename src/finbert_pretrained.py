"""
Pre-trained FinBERT model for financial sentiment analysis.
Uses ProsusAI/finbert directly without fine-tuning.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Union
import numpy as np


class PretrainedFinBERT:
    """Pre-trained FinBERT model for instant predictions."""

    # FinBERT uses different label mapping
    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize pre-trained FinBERT.

        Args:
            model_name: HuggingFace model name
        """
        print(f"Loading pre-trained FinBERT from {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("FinBERT loaded successfully!")

    def predict(self, texts: Union[str, List[str]]) -> List[dict]:
        """
        Predict sentiment for text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            List of prediction dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]

        results = []

        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

            # Format results
            for j, (text, pred, prob) in enumerate(zip(batch_texts, preds, probs)):
                # Map FinBERT labels to our format
                label = self.LABEL_MAP[int(pred)]

                # Convert to our label format (negative=0, neutral=1, positive=2)
                our_label_map = {"negative": 0, "neutral": 1, "positive": 2}
                our_pred = our_label_map[label]

                # Reorder probabilities to match our format
                our_probs = {
                    "negative": float(prob[1]),  # FinBERT: 1=negative
                    "neutral": float(prob[2]),   # FinBERT: 2=neutral
                    "positive": float(prob[0]),  # FinBERT: 0=positive
                }

                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "prediction": our_pred,
                    "label": label,
                    "confidence": float(max(prob)),
                    "probabilities": our_probs,
                })

        return results


# Global instance for caching
_finbert_instance = None


def get_finbert():
    """Get or create FinBERT instance (cached)."""
    global _finbert_instance
    if _finbert_instance is None:
        _finbert_instance = PretrainedFinBERT()
    return _finbert_instance


def predict_with_finbert(texts: Union[str, List[str]]) -> Union[dict, List[dict]]:
    """
    Quick prediction with pre-trained FinBERT.

    Args:
        texts: Single text or list of texts

    Returns:
        Prediction dict or list of dicts
    """
    finbert = get_finbert()
    results = finbert.predict(texts)

    if isinstance(texts, str):
        return results[0]
    return results


if __name__ == "__main__":
    # Test the model
    test_texts = [
        "The company reported strong earnings growth this quarter.",
        "Stock prices fell sharply after disappointing results.",
        "Revenue remained unchanged from last year.",
    ]

    print("\nTesting Pre-trained FinBERT...")
    print("=" * 60)

    for text in test_texts:
        result = predict_with_finbert(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['label']} ({result['confidence']:.1%})")
        print(f"Probabilities: {result['probabilities']}")

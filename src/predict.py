"""
Prediction module for Financial Sentiment Analysis.
Single text and batch prediction support.
"""

import json
import joblib
import pandas as pd
import numpy as np
from typing import Union, List

from utils import get_project_root, get_model_dir, setup_logging, LABEL_MAP_INV
from preprocess import clean_text

logger = setup_logging(__name__)


class SentimentPredictor:
    """Unified predictor for baseline and transformer models."""

    def __init__(self, model_type: str = "baseline_svm"):
        """
        Initialize the predictor.

        Args:
            model_type: Model to use. Options:
                - 'baseline_logreg', 'baseline_naive_bayes', 'baseline_svm'
                - 'baseline_random_forest', 'baseline_gradient_boosting', 'baseline_mlp'
                - 'baseline_ensemble'
                - 'finbert_pretrained' (no training needed!)
                - 'finbert', 'distilbert', 'roberta', 'bert' (requires training)
        """
        self.model_type = model_type
        self.is_pretrained_finbert = model_type == "finbert_pretrained"
        self.is_transformer = not model_type.startswith("baseline_") and not self.is_pretrained_finbert
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model based on type."""
        model_dir = get_model_dir()

        # Handle pre-trained FinBERT (no training needed)
        if self.is_pretrained_finbert:
            from finbert_pretrained import get_finbert
            self.model = get_finbert()
        elif self.is_transformer:
            from model import FinancialSentimentModel
            model_path = model_dir / f"{self.model_type}_finetuned"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}. "
                    f"Train with: python src/train.py --model {self.model_type}"
                )
            self.model = FinancialSentimentModel.load(model_path)
        else:
            model_path = model_dir / f"{self.model_type}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}. "
                    f"Train with: python src/train.py --model baselines"
                )
            loaded = joblib.load(model_path)

            # Handle ensemble model (stored as dict with tfidf and ensemble)
            if isinstance(loaded, dict) and "ensemble" in loaded:
                self.model = loaded
                self.is_ensemble = True
            else:
                self.model = loaded
                self.is_ensemble = False

        logger.info("Loaded model: %s", self.model_type)

    def predict(self, text: Union[str, List[str]]) -> Union[dict, List[dict]]:
        """
        Predict sentiment for text(s).

        Args:
            text: Single text string or list of texts

        Returns:
            Single prediction dict or list of prediction dicts
        """
        single_input = isinstance(text, str)
        texts = [text] if single_input else text

        results = self._predict_batch(texts)

        return results[0] if single_input else results

    def _predict_batch(self, texts: List[str]) -> List[dict]:
        """Predict sentiment for a batch of texts."""
        if self.is_pretrained_finbert:
            return self.model.predict(texts)
        elif self.is_transformer:
            return self._predict_transformer(texts)
        else:
            return self._predict_baseline(texts)

    def _predict_baseline(self, texts: List[str]) -> List[dict]:
        """Predict using baseline model."""
        # Handle ensemble model separately
        if hasattr(self, 'is_ensemble') and self.is_ensemble:
            return self._predict_ensemble(texts)

        # Get predictions
        predictions = self.model.predict(texts)

        # Get probabilities if available
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(texts)
        elif hasattr(self.model.named_steps.get("classifier", None), "decision_function"):
            # For SVM, use decision function and softmax
            decisions = self.model.decision_function(texts)
            exp_decisions = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
            probas = exp_decisions / exp_decisions.sum(axis=1, keepdims=True)
        else:
            probas = None

        results = []
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "prediction": int(pred),
                "label": LABEL_MAP_INV[pred],
            }

            if probas is not None:
                result["confidence"] = float(probas[i][pred])
                result["probabilities"] = {
                    LABEL_MAP_INV[j]: float(p) for j, p in enumerate(probas[i])
                }

            results.append(result)

        return results

    def _predict_ensemble(self, texts: List[str]) -> List[dict]:
        """Predict using ensemble model."""
        tfidf = self.model["tfidf"]
        ensemble = self.model["ensemble"]

        # Transform texts
        X = tfidf.transform(texts)

        # Get predictions and probabilities
        predictions = ensemble.predict(X)
        probas = ensemble.predict_proba(X)

        results = []
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "prediction": int(pred),
                "label": LABEL_MAP_INV[pred],
                "confidence": float(probas[i][pred]),
                "probabilities": {
                    LABEL_MAP_INV[j]: float(p) for j, p in enumerate(probas[i])
                },
            }
            results.append(result)

        return results

    def _predict_transformer(self, texts: List[str]) -> List[dict]:
        """Predict using transformer model."""
        predictions, probas = self.model.predict(texts)

        results = []
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "prediction": int(pred),
                "label": LABEL_MAP_INV[pred],
                "confidence": float(probas[i][pred]),
                "probabilities": {
                    LABEL_MAP_INV[j]: float(p) for j, p in enumerate(probas[i])
                },
            }
            results.append(result)

        return results

    def predict_file(self, file_path: str, text_column: str = "text") -> pd.DataFrame:
        """
        Predict sentiment for texts in a file.

        Args:
            file_path: Path to CSV or text file
            text_column: Column name containing text (for CSV)

        Returns:
            DataFrame with predictions
        """
        file_path = Path(file_path)

        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            texts = df[text_column].tolist()
        elif file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({"text": texts})
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info("Predicting %d texts...", len(texts))
        results = self._predict_batch(texts)

        # Add predictions to dataframe
        df["predicted_label"] = [r["label"] for r in results]
        df["predicted_class"] = [r["prediction"] for r in results]
        df["confidence"] = [r.get("confidence", None) for r in results]

        return df


def predict_single(text: str, model_type: str = "baseline_svm") -> dict:
    """
    Quick prediction for a single text.

    Args:
        text: Text to analyze
        model_type: Model to use

    Returns:
        Prediction dictionary
    """
    predictor = SentimentPredictor(model_type)
    return predictor.predict(text)


def predict_batch(texts: List[str], model_type: str = "baseline_svm") -> List[dict]:
    """
    Quick prediction for multiple texts.

    Args:
        texts: List of texts to analyze
        model_type: Model to use

    Returns:
        List of prediction dictionaries
    """
    predictor = SentimentPredictor(model_type)
    return predictor.predict(texts)


def get_available_models() -> List[str]:
    """Get list of available trained models."""
    model_dir = get_model_dir()
    models = []

    # Check baseline models (including new ones)
    baseline_names = ["logreg", "naive_bayes", "svm", "random_forest", "gradient_boosting", "mlp", "ensemble"]
    for name in baseline_names:
        if (model_dir / f"baseline_{name}.joblib").exists():
            models.append(f"baseline_{name}")

    # Always add pre-trained FinBERT (no training needed)
    models.append("finbert_pretrained")

    # Check fine-tuned transformer models
    for name in ["finbert", "distilbert", "roberta", "bert"]:
        if (model_dir / f"{name}_finetuned").exists():
            models.append(name)

    return models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict Financial Sentiment")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to analyze")
    parser.add_argument("--file", type=str, default=None,
                        help="File to analyze (CSV or TXT)")
    parser.add_argument("--model", type=str, default="baseline_svm",
                        help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for batch predictions")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models")

    args = parser.parse_args()

    if args.list_models:
        logger.info("Available models:")
        for m in get_available_models():
            logger.info("  - %s", m)
    elif args.text:
        print(f"\nText: {args.text}")
        print(f"Sentiment: {result['label']}")
        if "confidence" in result:
            print(f"Confidence: {result['confidence']:.2%}")
        if "probabilities" in result:
            print("Probabilities:")
            for label, prob in result["probabilities"].items():
                print(f"  {label}: {prob:.2%}")
    elif args.file:
        predictor = SentimentPredictor(args.model)
        df = predictor.predict_file(args.file)

        print(f"\nResults:")
        print(df[["text", "predicted_label", "confidence"]].head(10))

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved to: {args.output}")
    else:
        # Interactive mode
        print("Financial Sentiment Analyzer")
        print(f"Model: {args.model}")
        print("Enter text to analyze (or 'quit' to exit):\n")

        predictor = SentimentPredictor(args.model)

        while True:
            text = input("> ").strip()
            if text.lower() in ["quit", "exit", "q"]:
                break
            if not text:
                continue

            result = predictor.predict(text)
            print(f"  Sentiment: {result['label']}", end="")
            if "confidence" in result:
                print(f" ({result['confidence']:.1%})")
            else:
                print()

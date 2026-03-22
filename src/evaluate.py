"""
Evaluation module for Financial Sentiment Analysis.
Comprehensive model evaluation with metrics, confusion matrix, and comparison.
"""

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from utils import get_project_root, get_model_dir, get_results_dir, setup_logging, LABEL_MAP_INV
from preprocess import load_processed_data

logger = setup_logging(__name__)


def evaluate_baseline(model_name: str, split: str = "test") -> dict:
    """
    Evaluate a baseline model on specified data split.

    Args:
        model_name: One of 'logreg', 'naive_bayes', 'svm'
        split: Data split to evaluate on ('val' or 'test')

    Returns:
        Dictionary with all metrics
    """
    model_path = get_model_dir() / f"baseline_{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train the model first.")

    model = joblib.load(model_path)
    df = load_processed_data(split)

    X = df["sentence"].values
    y_true = df["label"].values
    y_pred = model.predict(X)

    return compute_metrics(y_true, y_pred, model_name)


def evaluate_transformer(model_name: str, split: str = "test") -> dict:
    """
    Evaluate a transformer model on specified data split.

    Args:
        model_name: One of 'finbert', 'distilbert', etc.
        split: Data split to evaluate on ('val' or 'test')

    Returns:
        Dictionary with all metrics
    """
    from model import FinancialSentimentModel

    model_path = get_model_dir() / f"{model_name}_finetuned"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train the model first.")

    model = FinancialSentimentModel.load(model_path)
    df = load_processed_data(split)

    X = df["sentence"].values
    y_true = df["label"].values

    metrics = model.evaluate(X, y_true)
    metrics["name"] = model_name

    # Add additional metrics
    y_pred = metrics["y_pred"]
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro")
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")
    metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted")
    metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted")

    return metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    """
    Compute comprehensive metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Name of the model

    Returns:
        Dictionary with all metrics
    """
    return {
        "name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_evaluation_report(metrics: dict):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {metrics.get('name', 'Unknown Model')}")
    print(f"{'='*60}")

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  F1 Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"  F1 Score (weight):  {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")

    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]
    print(f"\nPer-Class Report:")
    print(classification_report(metrics["y_true"], metrics["y_pred"], target_names=target_names))


def plot_confusion_matrix(metrics: dict, save_path: str = None):
    """
    Plot confusion matrix.

    Args:
        metrics: Dictionary with y_true and y_pred
        save_path: Optional path to save the figure
    """
    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"])

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix: {metrics.get('name', 'Model')}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()
    return fig


def compare_models(model_names: list = None, split: str = "test") -> list:
    """
    Compare multiple models.

    Args:
        model_names: List of model names to compare (None = all available)
        split: Data split to evaluate on

    Returns:
        List of metrics dictionaries
    """
    if model_names is None:
        # Find all available models
        model_dir = get_model_dir()
        model_names = []

        # Check for baseline models
        for name in ["logreg", "naive_bayes", "svm"]:
            if (model_dir / f"baseline_{name}.joblib").exists():
                model_names.append(f"baseline_{name}")

        # Check for transformer models
        for name in ["finbert", "distilbert", "roberta", "bert"]:
            if (model_dir / f"{name}_finetuned").exists():
                model_names.append(name)

    results = []
    for name in model_names:
        try:
            if name.startswith("baseline_"):
                classifier = name.replace("baseline_", "")
                metrics = evaluate_baseline(classifier, split)
                metrics["name"] = name
            else:
                metrics = evaluate_transformer(name, split)

            results.append(metrics)
            logger.info("Evaluated: %s", name)
        except Exception as e:
            logger.error("Error evaluating %s: %s", name, e)

    # Print comparison table
    if results:
        print("\n" + "=" * 80)
        print(f"MODEL COMPARISON ({split.upper()} SET)")
        print("=" * 80)
        print(f"{'Model':<25} {'Accuracy':<10} {'F1 (macro)':<12} {'Precision':<12} {'Recall':<10}")
        print("-" * 80)

        for r in sorted(results, key=lambda x: x["f1_macro"], reverse=True):
            print(f"{r['name']:<25} {r['accuracy']:<10.4f} {r['f1_macro']:<12.4f} "
                  f"{r['precision_macro']:<12.4f} {r['recall_macro']:<10.4f}")

        best = max(results, key=lambda x: x["f1_macro"])
        print(f"\nBest model: {best['name']} (F1 macro: {best['f1_macro']:.4f})")

    return results


def error_analysis(model_name: str = "baseline_svm", split: str = "test", num_samples: int = 10) -> dict:
    """
    Analyze misclassified samples to identify patterns.

    Args:
        model_name: Model to analyze
        split: Data split to use

    Returns:
        Dictionary with error analysis results
    """
    from preprocess import load_processed_data

    # Get predictions
    if model_name.startswith("baseline_"):
        classifier = model_name.replace("baseline_", "")
        metrics = evaluate_baseline(classifier, split)
        metrics["name"] = model_name
    else:
        metrics = evaluate_transformer(model_name, split)

    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]

    # Load original texts
    df = load_processed_data(split)
    texts = df["sentence"].values

    # Find misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    total_errors = len(misclassified_idx)

    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS: {model_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(y_true)}")
    print(f"Misclassified: {total_errors} ({total_errors/len(y_true)*100:.1f}%)")

    # Analyze error patterns by class
    error_matrix = {}
    for true_label in range(3):
        for pred_label in range(3):
            if true_label != pred_label:
                key = f"{LABEL_MAP_INV[true_label]}_as_{LABEL_MAP_INV[pred_label]}"
                mask = (y_true == true_label) & (y_pred == pred_label)
                count = mask.sum()
                if count > 0:
                    error_matrix[key] = count

    print(f"\nError patterns:")
    for pattern, count in sorted(error_matrix.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}")

    # Show sample misclassifications
    print(f"\nSample misclassifications (first {num_samples}):")
    print("-" * 60)

    samples = []
    for i, idx in enumerate(misclassified_idx[:num_samples]):
        true_label = LABEL_MAP_INV[y_true[idx]]
        pred_label = LABEL_MAP_INV[y_pred[idx]]
        text = texts[idx][:80] + "..." if len(texts[idx]) > 80 else texts[idx]

        print(f"{i+1}. True: {true_label:<10} Pred: {pred_label:<10}")
        print(f"   Text: {text}")

        samples.append({
            "text": texts[idx],
            "true_label": true_label,
            "pred_label": pred_label,
        })

    # Identify common patterns
    print(f"\nKey insights:")
    if "neutral_as_positive" in error_matrix or "neutral_as_negative" in error_matrix:
        print("  - Neutral class is often confused with positive/negative")
    if "positive_as_neutral" in error_matrix:
        print("  - Some positive texts lack strong positive indicators")
    if "negative_as_neutral" in error_matrix:
        print("  - Some negative texts have subtle negative sentiment")

    return {
        "model": model_name,
        "total_errors": total_errors,
        "error_rate": total_errors / len(y_true),
        "error_matrix": error_matrix,
        "samples": samples,
    }


def save_results(results: list, filename: str = "evaluation_results.json"):
    """Save evaluation results to JSON."""
    output_dir = get_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k not in ["y_true", "y_pred"]}
        jr["y_true"] = r["y_true"].tolist() if hasattr(r["y_true"], "tolist") else r["y_true"]
        jr["y_pred"] = r["y_pred"].tolist() if hasattr(r["y_pred"], "tolist") else r["y_pred"]
        json_results.append(jr)

    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)

    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Financial Sentiment Models")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to evaluate (None = all available)")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split: val or test")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")
    parser.add_argument("--plot", action="store_true",
                        help="Plot confusion matrices")
    parser.add_argument("--errors", action="store_true",
                        help="Run error analysis")

    args = parser.parse_args()

    if args.errors and args.model:
        error_analysis(args.model, args.split)
    elif args.model:
        if args.model.startswith("baseline_"):
            classifier = args.model.replace("baseline_", "")
            metrics = evaluate_baseline(classifier, args.split)
        else:
            metrics = evaluate_transformer(args.model, args.split)

        print_evaluation_report(metrics)

        if args.plot:
            plot_confusion_matrix(metrics)
    else:
        results = compare_models(split=args.split)

        if args.save:
            save_results(results)

        if args.plot:
            for r in results:
                plot_confusion_matrix(r)

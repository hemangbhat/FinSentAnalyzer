"""
SHAP Explainability Module for Financial Sentiment Analysis.
Provides SHAP-based model interpretability for baseline models.

Supports:
- TreeExplainer for Gradient Boosting / Random Forest
- LinearExplainer for SVM / Logistic Regression
- Summary plots saved to results/
"""

import json

import joblib
import numpy as np

from preprocess import load_processed_data
from utils import LABEL_MAP_INV, get_model_dir, get_results_dir, setup_logging

logger = setup_logging(__name__)


def get_shap_explanation(
    model_name: str = "gradient_boosting",
    split: str = "test",
    max_samples: int = 100,
) -> dict:
    """
    Generate SHAP explanations for a baseline model.

    Args:
        model_name: One of 'gradient_boosting', 'random_forest', 'logreg', 'svm'
        split: Data split to explain
        max_samples: Maximum samples to explain (SHAP can be slow)

    Returns:
        Dictionary with SHAP values, feature names, and summary data
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    model_path = get_model_dir() / f"baseline_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    pipeline = joblib.load(model_path)
    df = load_processed_data(split)

    # Subsample if needed
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    X_text = df["sentence"].values

    # Extract TF-IDF transformer and classifier from pipeline
    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    X_tfidf = tfidf.transform(X_text)
    feature_names = tfidf.get_feature_names_out()

    logger.info("Computing SHAP values for %s on %d samples...", model_name, len(X_text))

    # Choose appropriate explainer
    tree_models = {"random_forest"}  # Only RF supports multi-class TreeExplainer reliably
    linear_models = {"logreg", "svm"}
    feature_importance_models = {"gradient_boosting"}  # Use feature_importances_ instead

    if model_name in tree_models:
        try:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_tfidf.toarray())
        except Exception as e:
            logger.warning("TreeExplainer failed (%s), using feature_importances_ fallback", e)
            # Fall through to feature_importances_ approach
            feature_importance_models.add(model_name)

    if model_name in feature_importance_models:
        # Use feature_importances_ + per-sample approximation
        if hasattr(classifier, "feature_importances_"):
            fi = classifier.feature_importances_  # global importance per feature
            X_dense = X_tfidf.toarray()
            # Approximate SHAP as feature_value * importance (1 class proxy)
            shap_values = X_dense * fi
            # For per-class analysis, use the model to predict class probabilities
            # and weight the importance by prediction confidence per class
            if hasattr(classifier, "predict_proba"):
                probas = classifier.predict_proba(X_dense)
                shap_values = []
                for class_idx in range(3):
                    class_weight = probas[:, class_idx : class_idx + 1]
                    class_shap = X_dense * fi * class_weight
                    shap_values.append(class_shap)
            else:
                # Single array, wrap for consistency
                shap_values = [shap_values]
        else:
            raise ValueError(f"Model {model_name} has no feature_importances_")
    elif model_name in linear_models:
        # For linear models, use the coefficients directly as feature importance
        if hasattr(classifier, "coef_"):
            # Compute per-sample SHAP-like values: feature_value * coefficient
            X_dense = X_tfidf.toarray()
            coefs = classifier.coef_  # shape: (n_classes, n_features)
            shap_values = []
            for class_idx in range(coefs.shape[0]):
                class_shap = X_dense * coefs[class_idx]
                shap_values.append(class_shap)
        else:
            raise ValueError(f"Model {model_name} doesn't have coef_ attribute")
    elif model_name not in tree_models:
        # Fallback: use KernelExplainer (slow but universal)
        background = shap.sample(X_tfidf, min(50, X_tfidf.shape[0]))
        if hasattr(classifier, "predict_proba"):
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
        else:
            explainer = shap.KernelExplainer(classifier.decision_function, background)
        shap_values = explainer.shap_values(X_tfidf.toarray()[: min(50, len(X_text))])

    # Compute feature importance (mean absolute SHAP across all classes)
    if isinstance(shap_values, list):
        # Multi-class: average across classes
        all_shap = np.array(shap_values)  # (n_classes, n_samples, n_features)
        mean_abs_shap = np.mean(np.abs(all_shap), axis=(0, 1))
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Get top features
    top_indices = np.argsort(mean_abs_shap)[::-1][:30]
    top_features = []
    for idx in top_indices:
        top_features.append(
            {
                "feature": str(feature_names[idx]),
                "importance": float(mean_abs_shap[idx]),
            }
        )

    # Per-class top features
    class_features = {}
    if isinstance(shap_values, list) and len(shap_values) == 3:
        for class_idx in range(3):
            class_name = LABEL_MAP_INV[class_idx]
            class_mean = np.mean(shap_values[class_idx], axis=0)
            class_top_idx = np.argsort(np.abs(class_mean))[::-1][:15]
            class_features[class_name] = [
                {
                    "feature": str(feature_names[i]),
                    "shap_value": float(class_mean[i]),
                    "direction": "positive" if class_mean[i] > 0 else "negative",
                }
                for i in class_top_idx
            ]

    result = {
        "model": model_name,
        "split": split,
        "num_samples": len(X_text),
        "num_features": len(feature_names),
        "top_features": top_features,
        "class_features": class_features,
    }

    logger.info("SHAP analysis complete. Top 5 features:")
    for f in top_features[:5]:
        logger.info("  %s: %.4f", f["feature"], f["importance"])

    return result


def save_shap_results(model_name: str = "gradient_boosting", split: str = "test") -> dict:
    """
    Generate and save SHAP results to the results directory.

    Args:
        model_name: Model to explain
        split: Data split

    Returns:
        SHAP results dictionary
    """
    result = get_shap_explanation(model_name, split)

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"shap_{model_name}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("SHAP results saved to: %s", output_path)

    return result


def generate_shap_plot(model_name: str = "gradient_boosting", split: str = "test", save: bool = True):
    """
    Generate and optionally save a SHAP summary bar plot.

    Args:
        model_name: Model to explain
        split: Data split
        save: Whether to save the plot to results/
    """
    try:
        import matplotlib
        import shap

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("SHAP and matplotlib are required: pip install shap matplotlib")

    model_path = get_model_dir() / f"baseline_{model_name}.joblib"
    pipeline = joblib.load(model_path)
    df = load_processed_data(split)

    if len(df) > 200:
        df = df.sample(n=200, random_state=42)

    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    X_tfidf = tfidf.transform(df["sentence"].values)
    feature_names = tfidf.get_feature_names_out()

    tree_models = {"gradient_boosting", "random_forest"}
    if model_name in tree_models:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_tfidf.toarray())

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_tfidf.toarray(),
            feature_names=feature_names,
            class_names=[LABEL_MAP_INV[i] for i in range(3)],
            plot_type="bar",
            max_display=20,
            show=False,
        )
        plt.title(f"SHAP Feature Importance — {model_name.replace('_', ' ').title()}", fontsize=14)
        plt.tight_layout()

        if save:
            results_dir = get_results_dir()
            results_dir.mkdir(parents=True, exist_ok=True)
            save_path = results_dir / f"shap_summary_{model_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            logger.info("SHAP plot saved to: %s", save_path)
            plt.close()
        else:
            plt.show()
    else:
        logger.warning("SHAP summary plots are best suited for tree-based models.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SHAP Explainability")
    parser.add_argument(
        "--model",
        type=str,
        default="gradient_boosting",
        help="Model to explain: gradient_boosting, random_forest, logreg, svm",
    )
    parser.add_argument("--split", type=str, default="test", help="Data split")
    parser.add_argument("--plot", action="store_true", help="Generate and save SHAP plot")

    args = parser.parse_args()

    if args.plot:
        generate_shap_plot(args.model, args.split)
    else:
        result = save_shap_results(args.model, args.split)
        print(f"\nTop 10 important features for {args.model}:")
        for f in result["top_features"][:10]:
            print(f"  {f['feature']:30s} {f['importance']:.4f}")

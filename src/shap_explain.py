"""
SHAP Explainability Module for Financial Sentiment Analysis.

Exact SHAP values for the TF-IDF baselines, in the model's own output space
(log-odds for Gradient Boosting and Logistic Regression, margin for the SVM):

- Linear models (SVM, Logistic Regression): linear SHAP,
  phi_j = coef_j * (x_j - mean_j), with the feature means taken from the
  training split. This is what ``shap.LinearExplainer`` computes for
  independent features, written out directly.
- Gradient Boosting: ``shap.TreeExplainer`` does not support multi-class
  ``GradientBoostingClassifier``, but the model's score for class k is a sum of
  regression trees, so each tree is explained exactly and the results are
  summed (scaled by the learning rate). Additivity is checked on every call.
- Random Forest: ``shap.TreeExplainer`` directly.
- Anything else: ``shap.KernelExplainer`` (sampling-based approximation).

Every result records the method in ``"method"``.
"""

import json

import joblib
import numpy as np

from preprocess import load_processed_data
from utils import LABEL_MAP_INV, get_model_dir, get_results_dir, setup_logging

logger = setup_logging(__name__)

METHODS = {
    "linear": "exact linear SHAP: coef * (x - training mean)",
    "gb_trees": "exact TreeSHAP, summed over the boosted trees of each class",
    "tree": "exact TreeSHAP (shap.TreeExplainer)",
    "kernel": "KernelSHAP (sampling approximation)",
}


def _linear_shap(classifier, X_dense: np.ndarray, background_mean: np.ndarray) -> list[np.ndarray]:
    """phi[k][i, j] = coef[k, j] * (x[i, j] - mean[j]); exact for a linear model with independent features."""
    centered = X_dense - background_mean
    return [centered * classifier.coef_[k] for k in range(classifier.coef_.shape[0])]


def _gradient_boosting_shap(classifier, X_dense: np.ndarray) -> list[np.ndarray]:
    """Exact SHAP for multi-class GradientBoostingClassifier by explaining each boosted tree."""
    import shap

    n_classes = classifier.estimators_.shape[1]
    values = [np.zeros_like(X_dense, dtype=float) for _ in range(n_classes)]
    for stage in classifier.estimators_:
        for k, tree in enumerate(stage):
            values[k] += classifier.learning_rate * shap.TreeExplainer(tree).shap_values(X_dense)
    # Additivity: score_k(x) - sum_j phi_kj(x) must be the same constant (the base value) for every x.
    raw = classifier.decision_function(X_dense)
    for k in range(n_classes):
        residual = raw[:, k] - values[k].sum(axis=1)
        if np.ptp(residual) > 1e-6:
            raise RuntimeError(f"SHAP additivity check failed for class {k} (spread {np.ptp(residual):.2e})")
    return values


def compute_shap_values(model_name: str, classifier, X_dense: np.ndarray, background_mean: np.ndarray):
    """Return (per-class SHAP arrays, method key) for a fitted baseline classifier."""
    if hasattr(classifier, "coef_") and model_name in {"logreg", "svm"}:
        return _linear_shap(classifier, X_dense, background_mean), "linear"
    if type(classifier).__name__ == "GradientBoostingClassifier":
        return _gradient_boosting_shap(classifier, X_dense), "gb_trees"

    import shap

    if type(classifier).__name__ == "RandomForestClassifier":
        values = shap.TreeExplainer(classifier).shap_values(X_dense)
        if isinstance(values, np.ndarray) and values.ndim == 3:  # newer shap: (samples, features, classes)
            values = [values[:, :, k] for k in range(values.shape[2])]
        return list(values), "tree"
    background = X_dense[: min(50, len(X_dense))]
    fn = classifier.predict_proba if hasattr(classifier, "predict_proba") else classifier.decision_function
    values = shap.KernelExplainer(fn, background).shap_values(X_dense[: min(50, len(X_dense))])
    if isinstance(values, np.ndarray) and values.ndim == 3:
        values = [values[:, :, k] for k in range(values.shape[2])]
    return list(values), "kernel"


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
        max_samples: Maximum samples to explain

    Returns:
        Dictionary with the method used, top features and per-class signed values
    """
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

    X_dense = tfidf.transform(X_text).toarray()
    feature_names = tfidf.get_feature_names_out()
    background_mean = np.asarray(tfidf.transform(load_processed_data("train")["sentence"].values).mean(axis=0)).ravel()

    logger.info("Computing SHAP values for %s on %d samples...", model_name, len(X_text))
    shap_values, method = compute_shap_values(model_name, classifier, X_dense, background_mean)

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
        "method": METHODS[method],
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

    X_dense = X_tfidf.toarray()
    background_mean = np.asarray(tfidf.transform(load_processed_data("train")["sentence"].values).mean(axis=0)).ravel()
    shap_values, _ = compute_shap_values(model_name, classifier, X_dense, background_mean)
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_dense,
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

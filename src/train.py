"""
Training module for Financial Sentiment Analysis.
Trains baseline models (TF-IDF + classifiers) and transformer models (FinBERT).
"""

import argparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils import get_project_root, get_model_dir, setup_logging, LABEL_MAP_INV
from preprocess import load_processed_data

logger = setup_logging(__name__)



def create_baseline_pipeline(classifier_name: str = "logreg") -> Pipeline:
    """
    Create a TF-IDF + classifier pipeline.

    Args:
        classifier_name: One of 'logreg', 'naive_bayes', 'svm', 'random_forest',
                        'gradient_boosting', 'mlp'

    Returns:
        sklearn Pipeline
    """
    classifiers = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "naive_bayes": MultinomialNB(alpha=0.1),
        "svm": LinearSVC(max_iter=1000, random_state=42, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=50,
            min_samples_split=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            early_stopping=True,
            random_state=42,
            verbose=False
        ),
    }

    if classifier_name not in classifiers:
        raise ValueError(f"Unknown classifier: {classifier_name}. Choose from {list(classifiers.keys())}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )),
        ("classifier", classifiers[classifier_name]),
    ])

    return pipeline


def train_baseline(classifier_name: str = "logreg", save: bool = True) -> dict:
    """
    Train a baseline model.

    Args:
        classifier_name: One of 'logreg', 'naive_bayes', 'svm'
        save: Whether to save the trained model

    Returns:
        Dictionary with model, metrics, and predictions
    """
    logger.info("=" * 50)
    logger.info("Training: TF-IDF + %s", classifier_name.upper())
    logger.info("=" * 50)

    # Load data
    train_df = load_processed_data("train")
    val_df = load_processed_data("val")

    X_train = train_df["sentence"].values
    y_train = train_df["label"].values
    X_val = val_df["sentence"].values
    y_val = val_df["label"].values

    logger.info("Train samples: %d", len(X_train))
    logger.info("Val samples: %d", len(X_val))

    # Create and train pipeline
    pipeline = create_baseline_pipeline(classifier_name)
    pipeline.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    logger.info("Validation Results:")
    logger.info("  Accuracy:    %.4f", accuracy)
    logger.info("  F1 (macro):  %.4f", f1_macro)
    logger.info("  F1 (weight): %.4f", f1_weighted)

    logger.info("Classification Report:")
    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]
    logger.info("\n%s", classification_report(y_val, y_pred, target_names=target_names))

    # Save model
    if save:
        model_dir = get_model_dir()
        model_path = model_dir / f"baseline_{classifier_name}.joblib"
        joblib.dump(pipeline, model_path)
        logger.info("Model saved to: %s", model_path)

    return {
        "model": pipeline,
        "classifier": classifier_name,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "y_val": y_val,
        "y_pred": y_pred,
    }


def train_all_baselines() -> list:
    """
    Train all baseline models and compare.

    Returns:
        List of result dictionaries
    """
    results = []
    models = ["logreg", "naive_bayes", "svm", "random_forest", "gradient_boosting", "mlp"]

    for name in models:
        result = train_baseline(name, save=True)
        results.append(result)

    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info("%-20s %-12s %-12s %-12s", "Model", "Accuracy", "F1 (macro)", "F1 (weight)")
    logger.info("-" * 60)
    for r in results:
        logger.info("%-20s %-12.4f %-12.4f %-12.4f", r['classifier'], r['accuracy'], r['f1_macro'], r['f1_weighted'])

    # Find best model
    best = max(results, key=lambda x: x["f1_macro"])
    logger.info("Best model: %s (F1 macro: %.4f)", best['classifier'], best['f1_macro'])

    return results


def train_ensemble(save: bool = True) -> dict:
    """
    Train a voting ensemble combining multiple models.

    Returns:
        Dictionary with model and metrics
    """
    logger.info("=" * 50)
    logger.info("Training: VOTING ENSEMBLE")
    logger.info("=" * 50)

    # Load data
    train_df = load_processed_data("train")
    val_df = load_processed_data("val")

    X_train = train_df["sentence"].values
    y_train = train_df["label"].values
    X_val = val_df["sentence"].values
    y_val = val_df["label"].values

    logger.info("Train samples: %d", len(X_train))
    logger.info("Val samples: %d", len(X_val))

    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    # Transform text data
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # Create calibrated SVM (for probability support)
    svm_calibrated = CalibratedClassifierCV(
        LinearSVC(max_iter=1000, random_state=42, class_weight="balanced"),
        cv=3
    )

    # Create ensemble with voting
    ensemble = VotingClassifier(
        estimators=[
            ('logreg', LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
            ('svm', svm_calibrated),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, class_weight="balanced", n_jobs=-1)),
        ],
        voting='soft'  # Use probabilities for voting
    )

    logger.info("Training ensemble (this may take a minute)...")
    ensemble.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = ensemble.predict(X_val_tfidf)

    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    logger.info("Validation Results:")
    logger.info("  Accuracy:    %.4f", accuracy)
    logger.info("  F1 (macro):  %.4f", f1_macro)
    logger.info("  F1 (weight): %.4f", f1_weighted)

    logger.info("Classification Report:")
    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]
    logger.info("\n%s", classification_report(y_val, y_pred, target_names=target_names))

    # Save model (save both tfidf and ensemble)
    if save:
        model_dir = get_model_dir()
        model_path = model_dir / "baseline_ensemble.joblib"
        joblib.dump({"tfidf": tfidf, "ensemble": ensemble}, model_path)
        logger.info("Model saved to: %s", model_path)

    return {
        "model": ensemble,
        "tfidf": tfidf,
        "classifier": "ensemble",
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "y_val": y_val,
        "y_pred": y_pred,
    }


def load_model(model_name: str = "baseline_logreg"):
    """
    Load a trained model.

    Args:
        model_name: Model filename without extension

    Returns:
        Loaded model/pipeline
    """
    model_path = get_model_dir() / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def train_transformer(
    model_name: str = "finbert",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    save: bool = True,
) -> dict:
    """
    Train a transformer model.

    Args:
        model_name: One of 'finbert', 'distilbert', 'roberta', 'bert'
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save: Whether to save the model

    Returns:
        Dictionary with model and metrics
    """
    from model import train_transformer as _train_transformer
    return _train_transformer(model_name, epochs, batch_size, learning_rate, save)


def train_all_models(include_transformers: bool = True, transformer_epochs: int = 3) -> dict:
    """
    Train all models (baselines + transformers) and compare.

    Args:
        include_transformers: Whether to train transformer models
        transformer_epochs: Number of epochs for transformers

    Returns:
        Dictionary with all results
    """
    results = {"baselines": [], "transformers": []}

    # Train baselines
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING BASELINE MODELS")
    logger.info("=" * 70)
    for name in ["logreg", "naive_bayes", "svm"]:
        result = train_baseline(name, save=True)
        results["baselines"].append({
            "name": f"tfidf_{name}",
            "accuracy": result["accuracy"],
            "f1_macro": result["f1_macro"],
            "f1_weighted": result["f1_weighted"],
        })

    # Train transformers
    if include_transformers:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING TRANSFORMER MODELS")
        logger.info("=" * 70)
        for name in ["finbert", "distilbert"]:
            try:
                result = train_transformer(name, epochs=transformer_epochs, save=True)
                results["transformers"].append({
                    "name": name,
                    "accuracy": result["metrics"]["accuracy"],
                    "f1_macro": result["metrics"]["f1_macro"],
                    "f1_weighted": result["metrics"]["f1_weighted"],
                })
            except Exception as e:
                logger.error("Error training %s: %s", name, e)

    # Final comparison
    logger.info("\n" + "=" * 70)
    logger.info("FINAL MODEL COMPARISON")
    logger.info("=" * 70)
    logger.info("%-25s %-12s %-12s %-12s", "Model", "Accuracy", "F1 (macro)", "F1 (weight)")
    logger.info("-" * 70)

    all_results = results["baselines"] + results["transformers"]
    for r in all_results:
        logger.info("%-25s %-12.4f %-12.4f %-12.4f", r['name'], r['accuracy'], r['f1_macro'], r['f1_weighted'])

    # Find best overall
    best = max(all_results, key=lambda x: x["f1_macro"])
    logger.info("Best overall: %s (F1 macro: %.4f)", best['name'], best['f1_macro'])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Financial Sentiment Models")
    parser.add_argument("--model", type=str, default="all",
                        help="Model to train: all, baselines, ensemble, finbert, distilbert, logreg, svm, naive_bayes, random_forest, gradient_boosting, mlp")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for transformer training")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()

    if args.model == "all":
        train_all_models(include_transformers=True, transformer_epochs=args.epochs)
    elif args.model == "baselines":
        train_all_baselines()
        train_ensemble()  # Also train ensemble
    elif args.model == "ensemble":
        train_ensemble()
    elif args.model in ["finbert", "distilbert", "roberta", "bert"]:
        train_transformer(args.model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    elif args.model in ["logreg", "svm", "naive_bayes", "random_forest", "gradient_boosting", "mlp"]:
        train_baseline(args.model)
    else:
        logger.error("Unknown model: %s", args.model)

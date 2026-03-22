"""
Training module for Financial Sentiment Analysis.
Trains baseline models (TF-IDF + classifiers) and transformer models (FinBERT).
"""

import argparse
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocess import load_processed_data, LABEL_MAP_INV


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


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
    print(f"\n{'='*50}")
    print(f"Training: TF-IDF + {classifier_name.upper()}")
    print(f"{'='*50}")

    # Load data
    train_df = load_processed_data("train")
    val_df = load_processed_data("val")

    X_train = train_df["sentence"].values
    y_train = train_df["label"].values
    X_val = val_df["sentence"].values
    y_val = val_df["label"].values

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")

    # Create and train pipeline
    pipeline = create_baseline_pipeline(classifier_name)
    pipeline.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    print(f"\nValidation Results:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weight): {f1_weighted:.4f}")

    print(f"\nClassification Report:")
    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]
    print(classification_report(y_val, y_pred, target_names=target_names))

    # Save model
    if save:
        model_dir = get_project_root() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"baseline_{classifier_name}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Model saved to: {model_path}")

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
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 (macro)':<12} {'F1 (weight)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['classifier']:<20} {r['accuracy']:<12.4f} {r['f1_macro']:<12.4f} {r['f1_weighted']:<12.4f}")

    # Find best model
    best = max(results, key=lambda x: x["f1_macro"])
    print(f"\nBest model: {best['classifier']} (F1 macro: {best['f1_macro']:.4f})")

    return results


def train_ensemble(save: bool = True) -> dict:
    """
    Train a voting ensemble combining multiple models.

    Returns:
        Dictionary with model and metrics
    """
    print(f"\n{'='*50}")
    print(f"Training: VOTING ENSEMBLE")
    print(f"{'='*50}")

    # Load data
    train_df = load_processed_data("train")
    val_df = load_processed_data("val")

    X_train = train_df["sentence"].values
    y_train = train_df["label"].values
    X_val = val_df["sentence"].values
    y_val = val_df["label"].values

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")

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

    print("Training ensemble (this may take a minute)...")
    ensemble.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = ensemble.predict(X_val_tfidf)

    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    print(f"\nValidation Results:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weight): {f1_weighted:.4f}")

    print(f"\nClassification Report:")
    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]
    print(classification_report(y_val, y_pred, target_names=target_names))

    # Save model (save both tfidf and ensemble)
    if save:
        model_dir = get_project_root() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "baseline_ensemble.joblib"
        joblib.dump({"tfidf": tfidf, "ensemble": ensemble}, model_path)
        print(f"Model saved to: {model_path}")

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
    model_path = get_project_root() / "models" / f"{model_name}.joblib"

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
    print("\n" + "=" * 70)
    print("TRAINING BASELINE MODELS")
    print("=" * 70)
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
        print("\n" + "=" * 70)
        print("TRAINING TRANSFORMER MODELS")
        print("=" * 70)
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
                print(f"Error training {name}: {e}")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1 (macro)':<12} {'F1 (weight)':<12}")
    print("-" * 70)

    all_results = results["baselines"] + results["transformers"]
    for r in all_results:
        print(f"{r['name']:<25} {r['accuracy']:<12.4f} {r['f1_macro']:<12.4f} {r['f1_weighted']:<12.4f}")

    # Find best overall
    best = max(all_results, key=lambda x: x["f1_macro"])
    print(f"\nBest overall: {best['name']} (F1 macro: {best['f1_macro']:.4f})")

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
        print(f"Unknown model: {args.model}")

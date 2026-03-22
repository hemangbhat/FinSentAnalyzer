"""
Model definitions for Financial Sentiment Analysis.
Includes FinBERT and other transformer models.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm
from pathlib import Path

from preprocess import LABEL_MAP_INV

# Available pre-trained models for financial sentiment
MODELS = {
    "finbert": "ProsusAI/finbert",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class FinancialSentimentModel:
    """Wrapper for transformer-based sentiment models."""

    def __init__(self, model_name: str = "finbert", num_labels: int = 3, device: str = None):
        """
        Initialize the model.

        Args:
            model_name: One of 'finbert', 'distilbert', 'roberta', 'bert'
            num_labels: Number of output classes
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}")

        self.model_name = model_name
        self.pretrained_name = MODELS[model_name]
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name} ({self.pretrained_name})...")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)

    def create_dataloader(self, texts, labels, batch_size=16, shuffle=True):
        """Create a DataLoader from texts and labels."""
        dataset = SentimentDataset(texts, labels, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(
        self,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
    ) -> dict:
        """
        Train the model.

        Returns:
            Dictionary with training history and final metrics
        """
        # Create dataloaders
        train_loader = self.create_dataloader(train_texts, train_labels, batch_size, shuffle=True)
        val_loader = self.create_dataloader(val_texts, val_labels, batch_size, shuffle=False)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

        print(f"\nTraining {self.model_name} for {epochs} epochs...")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_metrics = self.evaluate(val_texts, val_labels, batch_size)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1_macro"])

            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}, "
                  f"Val F1={val_metrics['f1_macro']:.4f}")

        return history

    def evaluate(self, texts, labels, batch_size: int = 16) -> dict:
        """Evaluate the model on a dataset."""
        self.model.eval()
        dataloader = self.create_dataloader(texts, labels, batch_size, shuffle=False)

        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=batch_labels)
                total_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
            "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
            "y_true": all_labels,
            "y_pred": all_preds,
        }

    def predict(self, texts, batch_size: int = 16) -> tuple:
        """
        Predict sentiment for texts.

        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()

        # Create dummy labels for dataloader
        dummy_labels = [0] * len(texts)
        dataloader = self.create_dataloader(texts, dummy_labels, batch_size, shuffle=False)

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)

    def predict_single(self, text: str) -> dict:
        """
        Predict sentiment for a single text.

        Returns:
            Dictionary with prediction, label, confidence, and all probabilities
        """
        preds, probs = self.predict([text])
        pred_idx = preds[0]
        confidence = probs[0][pred_idx]

        return {
            "prediction": pred_idx,
            "label": LABEL_MAP_INV[pred_idx],
            "confidence": float(confidence),
            "probabilities": {LABEL_MAP_INV[i]: float(p) for i, p in enumerate(probs[0])},
        }

    def save(self, path: str = None):
        """Save model and tokenizer."""
        if path is None:
            path = get_project_root() / "models" / f"{self.model_name}_finetuned"
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str, device: str = None):
        """Load a saved model."""
        path = Path(path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        instance = cls.__new__(cls)
        instance.device = device
        instance.model_name = path.name.replace("_finetuned", "")
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.model.to(device)
        instance.num_labels = instance.model.config.num_labels

        print(f"Loaded model from: {path}")
        return instance


def train_transformer(
    model_name: str = "finbert",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    save: bool = True,
) -> dict:
    """
    Train a transformer model on the financial sentiment dataset.

    Args:
        model_name: One of 'finbert', 'distilbert', 'roberta', 'bert'
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        save: Whether to save the model

    Returns:
        Dictionary with model, history, and metrics
    """
    from preprocess import load_processed_data

    print(f"\n{'='*60}")
    print(f"Training Transformer: {model_name.upper()}")
    print(f"{'='*60}")

    # Load data
    train_df = load_processed_data("train")
    val_df = load_processed_data("val")

    X_train = train_df["sentence"].values
    y_train = train_df["label"].values
    X_val = val_df["sentence"].values
    y_val = val_df["label"].values

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Create and train model
    model = FinancialSentimentModel(model_name)
    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )

    # Final evaluation
    print("\nFinal Validation Results:")
    metrics = model.evaluate(X_val, y_val)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  F1 (weight): {metrics['f1_weighted']:.4f}")

    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]
    print("\nClassification Report:")
    print(classification_report(metrics["y_true"], metrics["y_pred"], target_names=target_names))

    # Save model
    if save:
        model.save()

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
    }


if __name__ == "__main__":
    # Train FinBERT
    result = train_transformer("finbert", epochs=3)

"""
Data preprocessing module for Financial Sentiment Analysis.
Handles data loading, cleaning, and preparation for model training.
"""

import os
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Label mapping
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_raw_data(agreement_level: str = "AllAgree") -> pd.DataFrame:
    """
    Load raw Financial PhraseBank data.

    Args:
        agreement_level: One of '50Agree', '66Agree', '75Agree', 'AllAgree'

    Returns:
        DataFrame with 'sentence' and 'label' columns
    """
    root = get_project_root()
    data_path = root / f"data/raw/FinancialPhraseBank-v1.0/Sentences_{agreement_level}.txt"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    sentences = []
    labels = []

    with open(data_path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if "@" in line:
                parts = line.rsplit("@", 1)
                sentences.append(parts[0])
                labels.append(parts[1])

    return pd.DataFrame({"sentence": sentences, "label": labels})


def load_processed_data(split: str = "train") -> pd.DataFrame:
    """
    Load processed data from CSV.

    Args:
        split: One of 'train', 'val', 'test'

    Returns:
        DataFrame with 'sentence' and 'label' columns
    """
    root = get_project_root()
    data_path = root / f"data/processed/{split}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}. Run EDA notebook first.")

    return pd.read_csv(data_path)


def clean_text(text: str) -> str:
    """
    Clean financial text for model input.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep financial symbols
    text = re.sub(r"[^\w\s\.\,\%\$\-\+]", "", text)

    return text.strip()


def preprocess_dataframe(df: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
    """
    Preprocess a dataframe for model training.

    Args:
        df: DataFrame with 'sentence' and 'label' columns
        clean: Whether to apply text cleaning

    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()

    # Remove duplicates and empty rows
    df = df.drop_duplicates(subset=["sentence"])
    df = df[df["sentence"].str.strip() != ""]
    df = df.dropna(subset=["sentence"])

    # Clean text if requested
    if clean:
        df["sentence"] = df["sentence"].apply(clean_text)

    # Encode labels if they're strings
    if df["label"].dtype == object:
        df["label"] = df["label"].map(LABEL_MAP)

    return df.reset_index(drop=True)


def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42
) -> tuple:
    """
    Create train/val/test splits.

    Args:
        df: Preprocessed DataFrame
        test_size: Fraction for test+val combined
        val_size: Fraction of test_size for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    X = df["sentence"].values
    y = df["label"].values

    # First split: train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    train_df = pd.DataFrame({"sentence": X_train, "label": y_train})
    val_df = pd.DataFrame({"sentence": X_val, "label": y_val})
    test_df = pd.DataFrame({"sentence": X_test, "label": y_test})

    return train_df, val_df, test_df


def save_processed_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save processed splits to CSV files."""
    root = get_project_root()
    processed_dir = root / "data/processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    print(f"Saved processed data to {processed_dir}")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")


def prepare_data(clean: bool = True, save: bool = True) -> tuple:
    """
    Full data preparation pipeline.

    Args:
        clean: Whether to apply text cleaning
        save: Whether to save processed data to CSV

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load raw data
    df = load_raw_data()
    print(f"Loaded {len(df)} samples from raw data")

    # Preprocess
    df = preprocess_dataframe(df, clean=clean)
    print(f"After preprocessing: {len(df)} samples")

    # Create splits
    train_df, val_df, test_df = create_splits(df)

    # Save if requested
    if save:
        save_processed_data(train_df, val_df, test_df)

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Run data preparation pipeline
    train_df, val_df, test_df = prepare_data()

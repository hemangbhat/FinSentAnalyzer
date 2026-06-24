"""
Fetch a REAL, licensed financial-news sentiment dataset for out-of-distribution
generalization testing.

Source: `zeroshot/twitter-financial-news-sentiment` (Hugging Face) — real,
human-labeled financial news/social headlines, MIT licensed. We use its
held-out `validation` split (~2,388 rows) as a genuine out-of-domain test set,
distinct from the Financial PhraseBank used for training.

Label mapping (dataset → this project):
    0 Bearish  -> negative (0)
    1 Bullish  -> positive (2)
    2 Neutral  -> neutral  (1)

Output: data/external/real_financial_news.csv  (headline, label, label_name, source)

Usage:
    python scripts/fetch_real_news_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "external"
OUT_PATH = OUT_DIR / "real_financial_news.csv"

HF_DATASET = "zeroshot/twitter-financial-news-sentiment"
HF_SPLIT = "validation"
LICENSE = "MIT"

# dataset label id -> (project label id, project label name)
LABEL_MAP = {0: (0, "negative"), 1: (2, "positive"), 2: (1, "neutral")}


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("The 'datasets' package is required: pip install datasets")

    print(f"Downloading {HF_DATASET} [{HF_SPLIT}] …")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)

    rows = []
    for ex in ds:
        text = str(ex["text"]).strip()
        if not text:
            continue
        proj_label, proj_name = LABEL_MAP[int(ex["label"])]
        rows.append(
            {
                "headline": text,
                "label": proj_label,
                "label_name": proj_name,
                "source": f"{HF_DATASET} ({LICENSE})",
            }
        )

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"[ok] Wrote {len(df)} real labeled headlines to {OUT_PATH}")
    print("  Label distribution:")
    print(df["label_name"].value_counts().to_string())
    print(f"\n  License: {LICENSE}. Cite: {HF_DATASET} on Hugging Face.")
    print("  Next: python src/integrate_news.py --action generalization --model svm")


if __name__ == "__main__":
    main()

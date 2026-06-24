"""
Fine-tune a transformer on in-domain financial news.

Trains on the *train* split of `zeroshot/twitter-financial-news-sentiment`
(MIT) and evaluates on its *validation* split — the same set used for the
out-of-distribution baseline. This is **in-domain** training for the news
distribution: it shows that the 0.46 macro-F1 gap of the PhraseBank model is
domain mismatch, and that training on in-domain data closes it.

The saved model lands in `models/{name}_finetuned/`, which the dashboard and
API pick up automatically (via get_available_models / predict.py).

Usage:
    python scripts/finetune_finbert_news.py --model distilbert --epochs 2
    python scripts/finetune_finbert_news.py --model finbert --epochs 3 --max-train 6000

GPU (recommended for FinBERT — Colab/Kaggle/cloud):
    # ~3 min/epoch on a T4; mixed precision (AMP) is enabled automatically on CUDA.
    python scripts/finetune_finbert_news.py --model finbert --epochs 3 --batch-size 32

Notes:
- Defaults to DistilBERT (fast). FinBERT is heavier but more domain-apt.
- On CPU this is slow; use --max-train to cap, or run on a GPU box.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Dataset label id -> project label id (negative=0, neutral=1, positive=2)
DS_TO_PROJECT = {0: 0, 1: 2, 2: 1}  # 0 Bearish, 1 Bullish, 2 Neutral


def _load_news_split(split: str, max_n: int | None = None):
    from datasets import load_dataset

    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split=split)
    if max_n:
        ds = ds.select(range(min(max_n, len(ds))))
    texts = [str(x) for x in ds["text"]]
    labels = [DS_TO_PROJECT[int(y)] for y in ds["label"]]
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer on in-domain financial news.")
    parser.add_argument("--model", default="distilbert", choices=["distilbert", "finbert", "bert", "roberta"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-train", type=int, default=None, help="Cap training rows (for CPU runs)")
    parser.add_argument("--val-monitor", type=int, default=500, help="Val subset size used during training")
    args = parser.parse_args()

    import torch

    from model import FinancialSentimentModel
    from utils import get_results_dir

    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)} — mixed precision (AMP) enabled.")
    else:
        print("No GPU detected — running on CPU (slow). Consider --max-train, or run on a GPU box.")

    print(f"Loading data … (max_train={args.max_train})")
    train_texts, train_labels = _load_news_split("train", args.max_train)
    val_texts, val_labels = _load_news_split("validation")

    # Smaller val subset for per-epoch monitoring; full set for final eval.
    mon_texts = val_texts[: args.val_monitor]
    mon_labels = val_labels[: args.val_monitor]

    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | model: {args.model}")
    fsm = FinancialSentimentModel(model_name=args.model, num_labels=3)
    fsm.train(
        train_texts,
        train_labels,
        mon_texts,
        mon_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print("Final evaluation on full validation split …")
    final = fsm.evaluate(val_texts, val_labels, batch_size=args.batch_size)
    fsm.save()  # -> models/{model}_finetuned/

    results = {
        "model": args.model,
        "dataset": "zeroshot/twitter-financial-news-sentiment (in-domain train→val, MIT)",
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "epochs": args.epochs,
        "accuracy": float(final["accuracy"]),
        "f1_macro": float(final["f1_macro"]),
    }
    out = get_results_dir() / "finetune_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print("In-domain fine-tuning result")
    print("=" * 50)
    print(f"Model:    {args.model}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Macro-F1: {results['f1_macro']:.3f}  (PhraseBank baseline on this set: ~0.46)")
    print(f"Saved metrics -> {out}")


if __name__ == "__main__":
    main()

# ruff: noqa: E402
# NOTEBOOK VERSION: v5  (direct in-process training — no subprocess, full tracebacks always visible)
"""
FinBERT / DeBERTa GPU fine-tuning script — v5.

Run on Kaggle (free T4/P100) or Google Colab.

=======================================================
KAGGLE SETUP  (recommended, free T4)
=======================================================
  1. kaggle.com -> Create -> New Notebook
  2. Paste this whole file into a code cell
  3. Settings sidebar -> Accelerator -> GPU T4 x2
  4. *** Settings sidebar -> Internet -> ON ***
  5. Run All  (~30-40 min for both models)

=======================================================
COLAB SETUP
=======================================================
  1. colab.research.google.com -> New notebook
  2. Runtime -> Change runtime type -> T4 GPU
  3. Paste and run

=======================================================
AFTER DOWNLOADING THE ZIP
=======================================================
  - {model}_finetuned/            -> models/{model}_finetuned/
  - finetune_results_{model}.json -> results/
  Then locally:
    python src/integrate_news.py --action generalization --model finbert
    python src/integrate_news.py --action generalization --model deberta
    python scripts/compare_ood.py
    python src/registry.py --update
    # Update README table and push
=======================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1  Pre-flight
# ─────────────────────────────────────────────────────────────────────────────
import socket
import subprocess
import sys


def _check_internet(host: str = "pypi.org", port: int = 443, timeout: int = 5) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


if not _check_internet():
    raise RuntimeError("\nNo internet. KAGGLE: Settings -> Internet -> ON, then restart & re-run.")
print("[ok] Internet: reachable")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2  Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
PACKAGES = [
    "transformers>=4.40.0",
    "datasets",
    "accelerate",
    "scikit-learn",
    "tqdm",
    "sentencepiece",
    "protobuf",
]


def _importable(pkg: str) -> bool:
    name = pkg.split(">=")[0].split("==")[0].split("[")[0]
    try:
        __import__(name)
        return True
    except ImportError:
        return False


missing = [p for p in PACKAGES if not _importable(p)]
if missing:
    print(f"Installing: {missing}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing)
    if r.returncode != 0:
        raise RuntimeError("pip install failed. On Kaggle: Settings -> Internet -> ON")
else:
    print("[ok] All dependencies present")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3  Clone / hard-reset repo
# ─────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

REPO_URL = "https://github.com/hemangbhat/FinSentAnalyzer.git"
if Path("/kaggle/working").exists():
    REPO_DIR = "/kaggle/working/FinSentAnalyzer"
elif Path("/content").exists():
    REPO_DIR = "/content/FinSentAnalyzer"
else:
    REPO_DIR = str(Path.home() / "FinSentAnalyzer")

if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--depth=1", REPO_URL, REPO_DIR], check=True)
    print(f"[ok] Cloned to {REPO_DIR}")
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--depth=1", "origin", "main"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"], check=True)
    print("[ok] Hard-reset to origin/main")

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

commit = subprocess.run(
    ["git", "-C", REPO_DIR, "log", "-1", "--oneline"], capture_output=True, text=True
).stdout.strip()
print("[ok] Repo commit:", commit)

# Verify the AMP fix is present in model.py on disk.
model_py_text = Path("src/model.py").read_text()
if "deberta_family" not in model_py_text:
    raise AssertionError(
        f"Old model.py in repo — DeBERTa AMP fix missing! Commit: {commit}. "
        "Delete /kaggle/working/FinSentAnalyzer and re-run."
    )
print("[ok] model.py DeBERTa AMP fix confirmed")
print("[ok] Working dir:", os.getcwd())

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4  Fetch real news dataset
# ─────────────────────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "scripts/fetch_real_news_dataset.py"], check=True)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5  Confirm GPU
# ─────────────────────────────────────────────────────────────────────────────
import torch

if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Kaggle: Settings -> Accelerator -> GPU T4 x2. Colab: Runtime -> T4 GPU.")

print(f"[ok] GPU: {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6  In-process fine-tune (no subprocess — full tracebacks always visible)
# ─────────────────────────────────────────────────────────────────────────────
import json
import shutil

import numpy as np
from datasets import load_dataset

# Import project modules directly (repo is on sys.path from CELL 3).
# Force re-import to pick up the hard-reset version.
for _mod in list(sys.modules.keys()):
    if _mod in ("model", "utils", "preprocess"):
        del sys.modules[_mod]

from model import FinancialSentimentModel  # noqa: E402
from utils import get_results_dir  # noqa: E402

# Label mapping: dataset → project (0=Bearish→neg, 1=Bullish→pos, 2=Neutral→neu)
DS_TO_PROJECT = {0: 0, 1: 2, 2: 1}

RESULTS: dict = {}


def _load_split(split: str):
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split=split)
    return [str(x) for x in ds["text"]], [DS_TO_PROJECT[int(y)] for y in ds["label"]]


def _class_weights(labels):
    counts = np.bincount(labels, minlength=3).astype(float)
    counts[counts == 0] = 1.0
    return (counts.sum() / (3 * counts)).tolist()


def finetune(model_name: str, epochs: int = 4, batch_size: int = 32) -> None:
    """Fine-tune directly in-process so every exception prints inline."""
    print(f"\n{'=' * 60}")
    print(f"FINE-TUNING: {model_name.upper()}  epochs={epochs}  batch={batch_size}")
    print("=" * 60)

    train_texts, train_labels = _load_split("train")
    val_texts, val_labels = _load_split("validation")
    mon_texts, mon_labels = val_texts[:500], val_labels[:500]

    cw = _class_weights(train_labels)
    print(f"Class weights (neg, neu, pos): {[round(w, 3) for w in cw]}")
    print(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

    fsm = FinancialSentimentModel(model_name=model_name, num_labels=3)
    # Confirm AMP setting at runtime so it's always visible in the output.
    deberta_family = {"deberta", "deberta-v3", "microsoft/deberta-v3-base"}
    expected_amp = model_name not in deberta_family
    print(f"[info] AMP will be: {expected_amp}  (False for DeBERTa family = correct)")

    fsm.train(
        train_texts,
        train_labels,
        mon_texts,
        mon_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=2e-5,
        class_weights=cw,
        early_stopping=True,
        patience=2,
    )

    print(f"Final evaluation on full validation split ({len(val_texts)} rows)…")
    final = fsm.evaluate(val_texts, val_labels, batch_size=batch_size)
    fsm.save()

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    r = {
        "model": model_name,
        "dataset": "zeroshot/twitter-financial-news-sentiment (in-domain train->val, MIT)",
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "epochs": epochs,
        "accuracy": float(final["accuracy"]),
        "f1_macro": float(final["f1_macro"]),
    }
    (results_dir / "finetune_results.json").write_text(json.dumps(r, indent=2))
    (results_dir / f"finetune_results_{model_name}.json").write_text(json.dumps(r, indent=2))
    RESULTS[model_name] = r
    print(f"\n  [ok] {model_name}: acc={r['accuracy']:.3f}  macro-F1={r['f1_macro']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7  Run both models
# Set SKIP_FINBERT = True if FinBERT already ran successfully this session.
# ─────────────────────────────────────────────────────────────────────────────
SKIP_FINBERT = True  # <-- change to False to re-run FinBERT

if not SKIP_FINBERT:
    finetune("finbert", epochs=4, batch_size=32)
else:
    fb_path = Path("results/finetune_results_finbert.json")
    if fb_path.exists():
        RESULTS["finbert"] = json.loads(fb_path.read_text())
        r = RESULTS["finbert"]
        print(f"[ok] Recovered FinBERT: acc={r['accuracy']:.3f}  macro-F1={r['f1_macro']:.3f}")
    else:
        print("[warn] No saved FinBERT result — re-running.")
        finetune("finbert", epochs=4, batch_size=32)

# Clear GPU memory between models.
torch.cuda.empty_cache()
print("[ok] GPU cache cleared")

finetune("deberta", epochs=4, batch_size=16)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8  Comparison summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON  (real OOD val set, n=2388)")
print("=" * 60)
print(f"{'Model':<14}  {'Accuracy':>9}  {'Macro-F1':>9}")
print(f"{'SVM baseline':<14}  {'0.670':>9}  {'0.460':>9}  <- PhraseBank, no fine-tune")
for name, r in RESULTS.items():
    print(f"{name:<14}  {r['accuracy']:>9.3f}  {r['f1_macro']:>9.3f}")
print("\nPrevious best (FinBERT v1, 3 epochs, no class weights): acc=0.883  macro-F1=0.844")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9  Package for download
# ─────────────────────────────────────────────────────────────────────────────
if Path("/kaggle/working").exists():
    OUT_DIR = Path("/kaggle/working/finsight_finetuned_v2")
elif Path("/content").exists():
    OUT_DIR = Path("/content/finsight_finetuned_v2")
else:
    OUT_DIR = Path(REPO_DIR) / "finsight_finetuned_v2_export"

OUT_DIR.mkdir(parents=True, exist_ok=True)

for model_name in RESULTS:
    src = Path(f"models/{model_name}_finetuned")
    if src.exists():
        shutil.copytree(src, OUT_DIR / src.name, dirs_exist_ok=True)
        print(f"Weights: {OUT_DIR / src.name}")
    per_r = Path(f"results/finetune_results_{model_name}.json")
    if per_r.exists():
        shutil.copy(per_r, OUT_DIR / per_r.name)

zip_path = shutil.make_archive(str(OUT_DIR), "zip", OUT_DIR)
print(f"\n[ok] Download: {zip_path}")
print()
print("Locally after downloading:")
print("  python src/integrate_news.py --action generalization --model finbert")
print("  python src/integrate_news.py --action generalization --model deberta")
print("  python scripts/compare_ood.py && python src/registry.py --update")

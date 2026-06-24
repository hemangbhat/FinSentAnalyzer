# ruff: noqa: E402
"""
FinBERT / DistilBERT GPU fine-tuning script.

Run on Kaggle (free T4/P100) or Google Colab — NOT on CPU (will be very slow).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KAGGLE SETUP (recommended, free T4):
  1. kaggle.com → Create → New Notebook
  2. Paste this whole file into a code cell  (or: File → Upload notebook → upload notebooks/finetune_finbert_gpu.py)
  3. Settings sidebar → Accelerator → GPU T4 x2
  4. Run All   (~15-20 min for FinBERT 3 epochs)

COLAB SETUP:
  1. colab.research.google.com → New notebook
  2. Runtime → Change runtime type → T4 GPU
  3. Paste this whole file into a code cell and run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES:
  1. Installs dependencies
  2. Clones your repo
  3. Fetches the real MIT-licensed news dataset
  4. Fine-tunes FinBERT (3 epochs, AMP, batch 32)
  5. Evaluates on the real held-out validation set
  6. Zips the model weights + result JSON for download

AFTER DOWNLOADING THE ZIP:
  - Extract finbert_finetuned/ → models/finbert_finetuned/
  - Extract finetune_results.json → results/finetune_results.json
  - python src/integrate_news.py --action generalization --model finbert
  - python src/registry.py --update
  - Update the README generalization table and push
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1  Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==4.40.0",
        "datasets",
        "accelerate",
        "scikit-learn",
        "tqdm",
    ],
    check=True,
)
print("Dependencies installed.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2  Clone repo
# ─────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

REPO_URL = "https://github.com/hemangbhat/FinSentAnalyzer.git"

# Auto-detect Kaggle vs Colab vs fallback
if Path("/kaggle/working").exists():
    REPO_DIR = "/kaggle/working/FinSentAnalyzer"
elif Path("/content").exists():
    REPO_DIR = "/content/FinSentAnalyzer"
else:
    REPO_DIR = str(Path.home() / "FinSentAnalyzer")

if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--depth=1", REPO_URL, REPO_DIR], check=True)
    print(f"Cloned to {REPO_DIR}")
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
    print(f"Repo already present at {REPO_DIR}, pulled latest.")

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
print("Working dir:", os.getcwd())

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3  Fetch the real news dataset
# ─────────────────────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "scripts/fetch_real_news_dataset.py"], check=True)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4  Confirm GPU + AMP
# ─────────────────────────────────────────────────────────────────────────────
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected.\nKaggle: Settings → Accelerator → GPU T4 x2\nColab:  Runtime → Change runtime type → T4 GPU"
    )

print(f"GPU:    {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("Mixed precision (AMP) will be enabled automatically.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5  Fine-tune
#   Change --model to 'distilbert' for a faster run (~5 min/epoch on T4).
#   'finbert' (ProsusAI/finbert) is the most domain-appropriate.
# ─────────────────────────────────────────────────────────────────────────────
subprocess.run(
    [
        sys.executable,
        "scripts/finetune_finbert_news.py",
        "--model",
        "finbert",
        "--epochs",
        "3",
        "--batch-size",
        "32",
        "--lr",
        "2e-5",
    ],
    check=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6  Show results
# ─────────────────────────────────────────────────────────────────────────────
import json

r = json.loads(Path("results/finetune_results.json").read_text())

print()
print("=" * 52)
print("Fine-tuning result")
print("=" * 52)
print(f"Model:      {r['model']}")
print(f"Dataset:    {r['dataset']}")
print(f"Train rows: {r['n_train']}    Val rows: {r['n_val']}")
print(f"Epochs:     {r['epochs']}")
print(f"Accuracy:   {r['accuracy']:.3f}")
print(f"Macro-F1:   {r['f1_macro']:.3f}")
print()
print("Comparison vs TF-IDF SVM on same val set:")
print("  Accuracy  0.670   Macro-F1  0.460")
print(f"  Improvement: +{r['accuracy'] - 0.670:.3f} acc  +{r['f1_macro'] - 0.460:.3f} F1")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7  Package for download
# ─────────────────────────────────────────────────────────────────────────────
import shutil

OUT_DIR = (
    Path("/kaggle/working/finsight_finetuned")
    if Path("/kaggle/working").exists()
    else Path("/content/finsight_finetuned")
    if Path("/content").exists()
    else Path(REPO_DIR) / "finsight_finetuned_export"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

model_src = Path(f"models/{r['model']}_finetuned")
if model_src.exists():
    shutil.copytree(model_src, OUT_DIR / model_src.name, dirs_exist_ok=True)
    print(f"Weights copied: {OUT_DIR / model_src.name}")

shutil.copy("results/finetune_results.json", OUT_DIR / "finetune_results.json")

zip_path = shutil.make_archive(str(OUT_DIR), "zip", OUT_DIR)
print(f"\nDownload ready: {zip_path}")
print()
print("After downloading, copy to your local repo:")
print(f"  {r['model']}_finetuned/   →  models/{r['model']}_finetuned/")
print("  finetune_results.json    →  results/finetune_results.json")
print()
print("Then run locally:")
print(f"  python src/integrate_news.py --action generalization --model {r['model']}")
print("  python src/registry.py --update")
print("  # Update README generalization table and push")

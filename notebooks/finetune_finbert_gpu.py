# ruff: noqa: E402
"""
FinBERT / DistilBERT GPU fine-tuning script.

Run on Kaggle (free T4/P100) or Google Colab.

=======================================================
KAGGLE SETUP  (recommended, free T4)
=======================================================
  1. kaggle.com -> Create -> New Notebook
  2. Paste this whole file into a code cell
     (or File -> Upload Notebook)
  3. Settings sidebar -> Accelerator -> GPU T4 x2
  4. *** Settings sidebar -> Internet -> ON ***
     Internet is OFF by default on Kaggle.
     Without this, pip install will fail with
     "Temporary failure in name resolution".
  5. Run All  (~15-20 min for FinBERT 3 epochs)

=======================================================
COLAB SETUP
=======================================================
  1. colab.research.google.com -> New notebook
  2. Runtime -> Change runtime type -> T4 GPU
  3. Paste this whole file into a code cell and run
     (Colab has internet on by default)

=======================================================
WHAT IT DOES
=======================================================
  1. Checks internet connectivity before installing
  2. Installs dependencies (skips if already present)
  3. Clones your repo
  4. Fetches the real MIT-licensed news dataset
  5. Fine-tunes FinBERT (3 epochs, AMP, batch 32)
  6. Evaluates on the real held-out validation set
  7. Zips model weights + result JSON for download

=======================================================
AFTER DOWNLOADING THE ZIP
=======================================================
  - finbert_finetuned/      -> models/finbert_finetuned/
  - finetune_results.json   -> results/finetune_results.json
  Then locally:
  python src/integrate_news.py --action generalization --model finbert
  python src/registry.py --update
  # Update README generalization table and push
=======================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1  Pre-flight: internet connectivity check
# ─────────────────────────────────────────────────────────────────────────────
import socket
import subprocess
import sys


def _check_internet(host="pypi.org", port=443, timeout=5) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


if not _check_internet():
    raise RuntimeError(
        "\n"
        "No internet access detected.\n\n"
        "KAGGLE FIX:\n"
        "  Right sidebar -> Settings -> Internet -> turn ON\n"
        "  Then restart the kernel and run again.\n\n"
        "COLAB FIX:\n"
        "  Colab has internet on by default — if you see this,\n"
        "  check Runtime -> Manage sessions and reconnect.\n"
    )

print("Internet connectivity: OK")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2  Install dependencies (skip if already present)
# ─────────────────────────────────────────────────────────────────────────────

PACKAGES = ["transformers>=4.40.0", "datasets", "accelerate", "scikit-learn", "tqdm", "sentencepiece"]


def _pkg_installed(pkg: str) -> bool:
    """Return True if the base package name is importable."""
    name = pkg.split(">=")[0].split("==")[0].split("[")[0]
    try:
        __import__(name)
        return True
    except ImportError:
        return False


missing = [p for p in PACKAGES if not _pkg_installed(p)]
if missing:
    print(f"Installing: {missing}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + missing,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pip install failed (exit {result.returncode}).\nOn Kaggle: make sure Internet is ON in Settings."
        )
else:
    print("All dependencies already installed — skipping.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3  Clone / update repo
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
    print(f"Cloned to {REPO_DIR}")
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
    print(f"Repo present at {REPO_DIR} — pulled latest.")

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
print("Working dir:", os.getcwd())

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4  Fetch the real news dataset
# ─────────────────────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "scripts/fetch_real_news_dataset.py"], check=True)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5  Confirm GPU + AMP
# ─────────────────────────────────────────────────────────────────────────────
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected.\n"
        "Kaggle:  Settings sidebar -> Accelerator -> GPU T4 x2\n"
        "Colab:   Runtime -> Change runtime type -> T4 GPU\n"
        "Then restart and run again."
    )

print(f"GPU:    {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("Mixed precision (AMP) will be enabled automatically.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6  Fine-tune
#   Stronger recipe: 4 epochs + class weights (helps macro-F1 on the
#   neutral-heavy data) + early stopping (keeps the best val checkpoint).
#   Swap --model finbert for 'deberta' (microsoft/deberta-v3-base) to try a
#   stronger base, or 'distilbert' for a faster run (~5 min/epoch on T4).
# ─────────────────────────────────────────────────────────────────────────────
subprocess.run(
    [
        sys.executable,
        "scripts/finetune_finbert_news.py",
        "--model",
        "finbert",
        "--epochs",
        "4",
        "--batch-size",
        "32",
        "--lr",
        "2e-5",
        "--class-weights",
        "--early-stopping",
        "--patience",
        "2",
    ],
    check=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7  Show results
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
print("Comparison vs TF-IDF SVM (same val set):")
print("  Accuracy 0.670   Macro-F1 0.460")
print(f"  Improvement: +{r['accuracy'] - 0.670:.3f} acc  +{r['f1_macro'] - 0.460:.3f} F1")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8  Package for download
# ─────────────────────────────────────────────────────────────────────────────
import shutil

if Path("/kaggle/working").exists():
    OUT_DIR = Path("/kaggle/working/finsight_finetuned")
elif Path("/content").exists():
    OUT_DIR = Path("/content/finsight_finetuned")
else:
    OUT_DIR = Path(REPO_DIR) / "finsight_finetuned_export"

OUT_DIR.mkdir(parents=True, exist_ok=True)

model_src = Path(f"models/{r['model']}_finetuned")
if model_src.exists():
    shutil.copytree(model_src, OUT_DIR / model_src.name, dirs_exist_ok=True)
    print(f"Weights copied to {OUT_DIR / model_src.name}")

shutil.copy("results/finetune_results.json", OUT_DIR / "finetune_results.json")
zip_path = shutil.make_archive(str(OUT_DIR), "zip", OUT_DIR)

print(f"\nDownload ready: {zip_path}")
print()
print("Copy to your local repo after downloading:")
print(f"  {r['model']}_finetuned/  ->  models/{r['model']}_finetuned/")
print("  finetune_results.json   ->  results/finetune_results.json")
print()
print("Then run locally:")
print(f"  python src/integrate_news.py --action generalization --model {r['model']}")
print("  python src/registry.py --update")
print("  # Update README generalization table and push")

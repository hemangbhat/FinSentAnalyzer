# ruff: noqa: E402
# NOTEBOOK VERSION: v4  (DeBERTa AMP disabled — FP16/bf16 incompatibility fix)
"""
FinBERT / DeBERTa GPU fine-tuning script — v2 (stronger recipe).

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
     Without this, pip install will fail.
  5. Run All  (~25-35 min for both models)

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
  1. Checks internet + GPU
  2. Installs dependencies (skips if already present)
  3. Clones your repo
  4. Fetches the real MIT-licensed news dataset
  5. Runs TWO fine-tune jobs back-to-back:
       A. FinBERT  4 epochs  + class weights + early stopping
       B. DeBERTa-v3  4 epochs + class weights + early stopping
  6. Evaluates both on the held-out validation set
  7. Zips ALL model weights + result JSONs for download

=======================================================
WHAT IS DIFFERENT FROM v1
=======================================================
  - class weights:  inverse-frequency weights so the model
    stops ignoring the minority (negative/positive) classes.
    Directly lifts macro-F1 on neutral-heavy data.
  - early stopping: saves the best val macro-F1 checkpoint
    automatically; training stops if val F1 stalls (patience=2).
  - DeBERTa-v3:  microsoft/deberta-v3-base is currently one
    of the strongest general-purpose encoders and typically
    beats vanilla FinBERT by 1-3 macro-F1 points on news text.
  Expected: FinBERT ~0.85-0.87  DeBERTa ~0.86-0.89

=======================================================
AFTER DOWNLOADING THE ZIP
=======================================================
  For each model (finbert, deberta):
  - {model}_finetuned/       -> models/{model}_finetuned/
  - finetune_results_{model}.json -> results/
  Then locally:
  python src/integrate_news.py --action generalization --model finbert
  python src/integrate_news.py --action generalization --model deberta
  python scripts/compare_ood.py    # rebuilds comparison artifact
  python src/registry.py --update
  # Update README table and push
=======================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1  Pre-flight checks
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
    raise RuntimeError(
        "\nNo internet access detected.\n\n"
        "KAGGLE FIX:\n"
        "  Right sidebar -> Settings -> Internet -> turn ON\n"
        "  Then restart the kernel and run again.\n\n"
        "COLAB: Colab has internet on by default — reconnect if you see this.\n"
    )
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
    "sentencepiece",  # required by DeBERTa-v3 tokenizer
    "protobuf",  # also required by DeBERTa-v3 tokenizer
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
# CELL 3  Clone / update repo  (always hard-reset to latest main)
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
    # Hard-reset to latest main so stale notebook code can never be used.
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--depth=1", "origin", "main"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"], check=True)
    print(f"[ok] Hard-reset to origin/main at {REPO_DIR}")

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

# Sanity-check: confirm we have the fixed notebook.
nb = Path("notebooks/finetune_finbert_gpu.py").read_text()
assert "capture_output=True" in nb, (
    "Old notebook detected — clone/reset failed. Try deleting the repo folder and re-running."
)
assert "protobuf" in nb, "Old notebook detected — missing protobuf dep."
print("[ok] Notebook version verified (capture_output + protobuf present)")
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
    raise RuntimeError(
        "No GPU detected.\n"
        "Kaggle:  Settings sidebar -> Accelerator -> GPU T4 x2\n"
        "Colab:   Runtime -> Change runtime type -> T4 GPU"
    )

print(f"[ok] GPU: {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("[ok] AMP (mixed precision) will be enabled automatically")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6  Fine-tune helpers
# ─────────────────────────────────────────────────────────────────────────────
import json
import shutil

RESULTS: dict = {}  # collects output from each run


def finetune(model_name: str, epochs: int = 4, batch_size: int = 32) -> None:
    """Run one fine-tune job and record results."""
    print(f"\n{'=' * 60}")
    print(f"FINE-TUNING: {model_name.upper()}  epochs={epochs}  batch={batch_size}")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/finetune_finbert_news.py",
            "--model",
            model_name,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            "2e-5",
            "--class-weights",  # inverse-freq weights → lifts minority class F1
            "--early-stopping",  # save best val macro-F1 checkpoint
            "--patience",
            "2",
        ],
        # capture output so the real exception is always visible on failure
        capture_output=True,
        text=True,
    )

    # Always show stdout (training progress lives here)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.returncode != 0:
        print("--- STDERR ---")
        print(result.stderr[-2000:])
        raise RuntimeError(
            f"\nFine-tuning {model_name} failed (exit {result.returncode}).\nThe real exception is printed above."
        )

    # Read the metrics written by the script.
    r = json.loads(Path("results/finetune_results.json").read_text())
    # Save a per-model copy so both results survive.
    Path(f"results/finetune_results_{model_name}.json").write_text(json.dumps(r, indent=2))
    RESULTS[model_name] = r
    print(f"\n  [ok] {model_name}: acc={r['accuracy']:.3f}  macro-F1={r['f1_macro']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7  Run both models
# NOTE: If FinBERT already ran and DeBERTa failed, set SKIP_FINBERT = True
#       to skip the (already-done) FinBERT re-run and go straight to DeBERTa.
# ─────────────────────────────────────────────────────────────────────────────
SKIP_FINBERT = False  # <-- set True if FinBERT already ran this session

if not SKIP_FINBERT:
    finetune("finbert", epochs=4, batch_size=32)
else:
    # Recover FinBERT result from the previous run in this session.
    fb_path = Path("results/finetune_results_finbert.json")
    if fb_path.exists():
        RESULTS["finbert"] = json.loads(fb_path.read_text())
        print(
            f"[ok] Recovered existing FinBERT result: acc={RESULTS['finbert']['accuracy']:.3f}  macro-F1={RESULTS['finbert']['f1_macro']:.3f}"
        )
    else:
        print("[warn] No saved FinBERT result found — re-running FinBERT.")
        finetune("finbert", epochs=4, batch_size=32)

# Clear GPU memory before the second (larger) model.
torch.cuda.empty_cache()
print("[ok] GPU cache cleared before DeBERTa run")

# DeBERTa-v3 uses smaller batch (heavier model) and does NOT use AMP
# (its bf16 internal embeddings are incompatible with FP16 GradScaler).
finetune("deberta", epochs=4, batch_size=16)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8  Summary comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON  (real OOD val set, n=2388)")
print("=" * 60)
print(f"{'Model':<14}  {'Accuracy':>9}  {'Macro-F1':>9}")
print(f"{'SVM baseline':<14}  {'0.670':>9}  {'0.460':>9}  <- PhraseBank-trained, no fine-tune")
for name, r in RESULTS.items():
    print(f"{name:<14}  {r['accuracy']:>9.3f}  {r['f1_macro']:>9.3f}")
print()
print("Previous best (FinBERT v1, 3 epochs, no class weights): acc=0.883  macro-F1=0.844")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9  Package everything for download
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
        print(f"Copied weights: {OUT_DIR / src.name}")
    results_src = Path(f"results/finetune_results_{model_name}.json")
    if results_src.exists():
        shutil.copy(results_src, OUT_DIR / results_src.name)

zip_path = shutil.make_archive(str(OUT_DIR), "zip", OUT_DIR)
print(f"\n[ok] Download ready: {zip_path}")
print()
print("After downloading, for each model do:")
print("  {model}_finetuned/                -> models/{model}_finetuned/")
print("  finetune_results_{model}.json     -> results/")
print()
print("Then locally:")
print("  python src/integrate_news.py --action generalization --model finbert")
print("  python src/integrate_news.py --action generalization --model deberta")
print("  python scripts/compare_ood.py")
print("  python src/registry.py --update")
print("  # Update README table and push")

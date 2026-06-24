"""
Retraining orchestrator.

Runs the full reproducible pipeline end-to-end and refreshes the model
registry. Each step shells out to the existing CLI entry points so this stays
a thin, auditable wrapper rather than a parallel implementation.

Steps:
    1. Train baseline models (+ ensemble)
    2. 5-fold cross-validation
    3. Evaluate on the test set (saves results/)
    4. Refresh the model registry (models/registry.json)

Usage:
    python scripts/retrain.py                 # full run
    python scripts/retrain.py --skip-cv       # skip cross-validation
    python scripts/retrain.py --mlflow        # also log to MLflow
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_step(name: str, args: list[str]) -> None:
    print(f"\n{'=' * 64}\n▶ {name}\n{'=' * 64}")
    start = time.time()
    result = subprocess.run([PY, *args], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {name} (exit {result.returncode})")
    print(f"✓ {name} done in {time.time() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the FinSight models end-to-end.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 5-fold cross-validation")
    parser.add_argument("--mlflow", action="store_true", help="Log the registry to MLflow")
    args = parser.parse_args()

    run_step("Train baselines + ensemble", ["src/train.py", "--model", "baselines"])
    if not args.skip_cv:
        run_step("5-fold cross-validation", ["src/train.py", "--model", "cv"])
    run_step("Evaluate on test set", ["src/evaluate.py", "--save"])

    registry_args = ["src/registry.py", "--update"]
    if args.mlflow:
        registry_args.append("--mlflow")
    run_step("Refresh model registry", registry_args)

    print("\n✓ Retraining complete. Review results/ and models/registry.json before committing.")


if __name__ == "__main__":
    main()

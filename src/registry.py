"""
Lightweight model registry / version tracking.

Records a versioned snapshot of every trained model artifact (hash, size,
timestamp) joined with its latest evaluation metrics, written to
``models/registry.json``. This gives reproducible, reviewable model lineage
with zero external infrastructure.

If MLflow is installed, ``--mlflow`` additionally logs params + metrics to a
local MLflow tracking store (``mlruns/``) so the project integrates with the
standard tooling without requiring it.

Usage
-----
    python src/registry.py --update            # refresh models/registry.json
    python src/registry.py --update --mlflow   # also log to MLflow
    python src/registry.py --show              # print current registry
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from utils import get_model_dir, get_project_root, get_results_dir, setup_logging

logger = setup_logging(__name__)

REGISTRY_PATH = get_model_dir() / "registry.json"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _load_eval_metrics() -> Dict[str, Dict[str, Any]]:
    """Map model name -> metrics from results/evaluation_results.json."""
    path = get_results_dir() / "evaluation_results.json"
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in data if isinstance(data, list) else []:
        name = row.get("name")
        if name:
            out[name] = {
                "accuracy": row.get("accuracy"),
                "f1_macro": row.get("f1_macro"),
                "f1_weighted": row.get("f1_weighted"),
            }
    return out


def _discover_artifacts() -> List[Path]:
    """Find tracked model artifacts across the project."""
    model_dir = get_model_dir()
    artifacts = sorted(model_dir.glob("baseline_*.joblib"))
    stock_lstm = get_project_root() / "external-datasets" / "financial-news-stock-prediction" / "models"
    if stock_lstm.exists():
        artifacts += sorted(stock_lstm.glob("lstm_*.pt"))
    return artifacts


def build_registry() -> Dict[str, Any]:
    """Build a registry snapshot dict from current artifacts + metrics."""
    eval_metrics = _load_eval_metrics()
    entries = []
    for path in _discover_artifacts():
        stat = path.stat()
        key = path.stem  # e.g. baseline_svm, lstm_AAPL
        entries.append(
            {
                "model": key,
                "artifact": path.name,
                "sha256": _hash_file(path),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "metrics": eval_metrics.get(key, {}),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(entries),
        "models": entries,
    }


def update_registry() -> Dict[str, Any]:
    """Write the registry snapshot to models/registry.json and return it."""
    registry = build_registry()
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    logger.info("Wrote registry with %d model(s) to %s", registry["model_count"], REGISTRY_PATH)
    return registry


def log_to_mlflow(registry: Dict[str, Any]) -> bool:
    """Best-effort MLflow logging. Returns True if logged, False if unavailable."""
    try:
        import mlflow  # type: ignore
    except ImportError:
        logger.warning("MLflow not installed; skipping. Install with `pip install mlflow`.")
        return False

    mlflow.set_experiment("finsight-sentiment")
    for entry in registry["models"]:
        with mlflow.start_run(run_name=entry["model"]):
            mlflow.set_tag("artifact", entry["artifact"])
            mlflow.set_tag("sha256", entry["sha256"])
            mlflow.log_param("size_bytes", entry["size_bytes"])
            for metric, value in (entry.get("metrics") or {}).items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric, float(value))
    logger.info("Logged %d model(s) to MLflow.", registry["model_count"])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Model registry / version tracking")
    parser.add_argument("--update", action="store_true", help="Rebuild models/registry.json")
    parser.add_argument("--show", action="store_true", help="Print the current registry")
    parser.add_argument("--mlflow", action="store_true", help="Also log to MLflow (if installed)")
    args = parser.parse_args()

    if args.update or args.mlflow:
        registry = update_registry()
        if args.mlflow:
            log_to_mlflow(registry)
        print(json.dumps(registry, indent=2))
    elif args.show:
        if REGISTRY_PATH.exists():
            print(REGISTRY_PATH.read_text(encoding="utf-8"))
        else:
            print("No registry found. Run: python src/registry.py --update")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
Assemble a model-comparison artifact on the real OOD benchmark.

Reads every `results/generalization_<model>.json` produced by
`integrate_news.evaluate_generalization()` and writes a compact
`results/model_comparison_ood.json` used by the dashboard and README.

Usage:
    # First generate per-model results, e.g.:
    python src/integrate_news.py --action generalization --model svm
    python src/integrate_news.py --action generalization --model finbert
    # Then:
    python scripts/compare_ood.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import get_results_dir  # noqa: E402


def main() -> None:
    results_dir = get_results_dir()
    files = sorted(results_dir.glob("generalization_*.json"))
    # Exclude the "latest" alias file if present.
    files = [f for f in files if f.name != "generalization_results.json"]

    if not files:
        sys.exit("No generalization_<model>.json files found. Run the generalization action first.")

    models = []
    majority_acc = None
    dataset = None
    n_samples = None
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        majority_acc = data.get("majority_class_accuracy", majority_acc)
        dataset = data.get("dataset", dataset)
        n_samples = data.get("n_samples", n_samples)
        models.append(
            {
                "model": data["model"],
                "accuracy": round(data["accuracy"], 4),
                "f1_macro": round(data["f1_macro"], 4),
                "f1_weighted": round(data.get("f1_weighted", 0.0), 4),
            }
        )

    models.sort(key=lambda m: m["f1_macro"])

    out = {
        "dataset": dataset,
        "n_samples": n_samples,
        "majority_class_accuracy": majority_acc,
        "models": models,
    }
    out_path = results_dir / "model_comparison_ood.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} with {len(models)} model(s):")
    for m in models:
        print(f"  {m['model']:<28} acc={m['accuracy']:.3f}  macroF1={m['f1_macro']:.3f}")


if __name__ == "__main__":
    main()

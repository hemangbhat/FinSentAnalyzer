"""
Bootstrap 95% confidence intervals for macro-F1.

Single test-set scores hide how much they could move with a different sample:
the PhraseBank test split has only 226 rows. This script resamples each
evaluation set with replacement (2,000 times) and reports the 2.5th and 97.5th
percentiles of macro-F1.

- PhraseBank test split: every saved baseline is re-scored on
  ``data/processed/test.csv``.
- Real news (Twitter Financial News, n=2,388): resampled from the confusion
  matrices in ``results/generalization_{svm,finbert}.json``. Resampling the
  (true, predicted) pairs a confusion matrix encodes is equivalent to
  resampling the evaluation rows.

Writes ``results/bootstrap_ci.json``.

    python scripts/bootstrap_ci.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import load_processed_data  # noqa: E402
from stats import bootstrap_macro_f1, pairs_from_confusion  # noqa: E402
from utils import get_model_dir, get_results_dir  # noqa: E402

N_BOOT = 2000
SEED = 42
BASELINES = ["logreg", "naive_bayes", "svm", "random_forest", "gradient_boosting", "mlp", "ensemble"]


def _ci(y_true: np.ndarray, y_pred: np.ndarray, rng: np.random.Generator) -> dict:
    return bootstrap_macro_f1(y_true, y_pred, n_boot=N_BOOT, rng=rng)


def _predict(model, texts: np.ndarray) -> np.ndarray:
    if isinstance(model, dict) and "ensemble" in model:
        return model["ensemble"].predict(model["tfidf"].transform(texts))
    return model.predict(texts)


def main() -> None:
    import joblib

    rng = np.random.default_rng(SEED)
    out: dict = {"n_boot": N_BOOT, "seed": SEED, "confidence": 0.95, "phrasebank_test": {}, "real_news": {}}

    test = load_processed_data("test")
    texts, y_true = test["sentence"].values, test["label"].values
    for name in BASELINES:
        path = get_model_dir() / f"baseline_{name}.joblib"
        if path.exists():
            out["phrasebank_test"][name] = _ci(y_true, _predict(joblib.load(path), texts), rng)

    for name in ["svm", "finbert"]:
        path = get_results_dir() / f"generalization_{name}.json"
        if path.exists():
            yt, yp = pairs_from_confusion(json.loads(path.read_text())["confusion_matrix"])
            out["real_news"][name] = _ci(yt, yp, rng)

    dest = get_results_dir() / "bootstrap_ci.json"
    dest.write_text(json.dumps(out, indent=2))
    for split, rows in (("PhraseBank test", out["phrasebank_test"]), ("Real news", out["real_news"])):
        for name, r in rows.items():
            print(f"{split:16} {name:18} {r['f1_macro']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  n={r['n']}")
    print(f"Saved: {dest}")


if __name__ == "__main__":
    main()

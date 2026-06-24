# Retraining Guide

The models are trained on a static, curated dataset (Financial PhraseBank), so
they do **not** require frequent retraining. Retrain when the data changes, a
dependency major-version bumps, or you add a model. This guide covers both an
automated path and a manual checklist.

## Automated retraining

One command runs the full reproducible pipeline and refreshes the registry:

```bash
python scripts/retrain.py            # train baselines + CV + evaluate + registry
python scripts/retrain.py --skip-cv  # faster: skip cross-validation
python scripts/retrain.py --mlflow   # also log metrics to MLflow
```

### Scheduled (CI)

`.github/workflows/retrain.yml` runs monthly (and on manual dispatch). It
retrains, evaluates, runs the test suite, and uploads the regenerated
`results/` and `models/registry.json` as build artifacts for review. It does
**not** auto-commit model weights — promotion is a manual, reviewed step.

## Manual retraining checklist

1. [ ] **Sync data** — confirm `data/processed/{train,val,test}.csv` are current and match `data_manifest.toml`.
2. [ ] **Environment** — `pip install -r requirements.txt`; confirm versions are pinned.
3. [ ] **Train** — `python src/train.py --model baselines` (and `--model cv` for cross-validation).
4. [ ] **Tune (optional)** — `python src/train.py --model tune` for SVM + Gradient Boosting.
5. [ ] **Evaluate** — `python src/evaluate.py --save` and compare metrics against the previous `results/evaluation_results.json`.
6. [ ] **Generalization** — `python scripts/fetch_real_news_dataset.py` then `python src/integrate_news.py --action generalization --model svm`; confirm OOD macro-F1 hasn't regressed.
7. [ ] **GPU fine-tune (optional upgrade)** — run `notebooks/finetune_finbert_gpu.py` on Kaggle/Colab T4; download `finbert_finetuned/` + `finetune_results.json`; copy into `models/` + `results/`; re-run generalization eval; update README table.
6. [ ] **Guardrail** — reject the new model if macro-F1 drops more than 2 points versus the current registry entry.
7. [ ] **Explainability** — regenerate SHAP: `python src/shap_explain.py --model gradient_boosting`.
8. [ ] **Registry** — `python src/registry.py --update` to record new hashes + metrics.
9. [ ] **Tests** — `pytest -q` must pass.
10. [ ] **Docs** — update any metric tables in `README.md` so docs match code.
11. [ ] **Commit** — commit the new `.joblib` artifacts, `results/`, and `models/registry.json` together with a message noting the metric delta.

## Promotion criteria

A retrained model is promoted (committed + deployed) only when:

- macro-F1 is within or above the previous version's range, and
- cross-validation variance has not materially increased, and
- the full test suite passes.

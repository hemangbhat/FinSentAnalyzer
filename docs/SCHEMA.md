# Schema

## 1. Primary dataset schema
### Raw Financial PhraseBank sentence format
| Field | Type | Description |
|---|---|---|
| sentence | string | financial text sample |
| label | string | sentiment label before encoding |

### Encoded label mapping
| Encoded | Label |
|---|---|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

## 2. Processed split schema
| Field | Type | Description |
|---|---|---|
| sentence | string | cleaned financial text |
| label | int | encoded sentiment class |

Files:
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## 3. Model artifact schema
### Baseline artifacts
- `models/baseline_logreg.joblib`
- `models/baseline_naive_bayes.joblib`
- `models/baseline_svm.joblib`
- `models/baseline_random_forest.joblib`
- `models/baseline_gradient_boosting.joblib`
- `models/baseline_mlp.joblib`
- `models/baseline_ensemble.joblib`

### Result artifacts
- `results/evaluation_results.json`
- `results/cv_results.json`
- `results/cross_dataset_results.json`
- `results/shap_*.json`

## 4. External stock-extension schema
### Example inputs
| Field | Description |
|---|---|
| ticker | stock ticker |
| date | date of news / price alignment |
| headline | financial news text |
| sentiment | model-scored sentiment |
| direction | next-day price direction |

## 5. Notes
- Keep schema stable across preprocessing, training, and inference.
- Avoid renaming `sentence` to `text` in one script and not the others.
- If new fields are added, update all downstream loaders and docs.

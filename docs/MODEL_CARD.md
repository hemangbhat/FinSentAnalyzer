# Model Card — Financial Sentiment Models

This card covers the models shipped/produced by this project. It follows the
spirit of [Mitchell et al., 2019, "Model Cards for Model Reporting"].

## 1. Models at a glance

| Model | Type | Training data | Intended use |
|-------|------|---------------|--------------|
| `baseline_*` (logreg, svm, gradient_boosting, random_forest, naive_bayes, mlp, ensemble) | TF-IDF (1–2 grams) + classical classifier | Financial PhraseBank (AllAgree, 2,264 sentences) | Fast, interpretable in-domain sentiment on formal financial sentences |
| `finbert_pretrained` | ProsusAI/finbert, zero-shot | — (pre-trained) | General financial sentiment without training |
| `distilbert_finetuned` | distilbert-base-uncased, fine-tuned | Twitter Financial News (train split, 9,543) | Short financial-news / social headline sentiment |
| `finbert_finetuned` | ProsusAI/finbert, fine-tuned | Twitter Financial News (train split, 9,543) | Short financial-news / social headline sentiment (best) |

Labels (all models): `negative (0)`, `neutral (1)`, `positive (2)`.

## 2. Metrics

In-domain (Financial PhraseBank test, n=226) — best baselines:

| Model | Accuracy | Macro-F1 | CV Macro-F1 (5-fold) |
|-------|:--------:|:--------:|:--------------------:|
| Gradient Boosting | 0.94 | 0.92 | 0.835 ± 0.008 |
| SVM (Linear) | 0.92 | 0.90 | 0.838 ± 0.016 |

Out-of-distribution (real, MIT-licensed `zeroshot/twitter-financial-news-sentiment`,
validation, n=2,388) — this is the honest generalization benchmark:

| Model | Accuracy | Macro-F1 |
|-------|:--------:|:--------:|
| TF-IDF SVM (trained on PhraseBank) | 0.67 | 0.46 |
| Majority-class baseline | 0.66 | — |
| DistilBERT fine-tuned (1 epoch, CPU) | 0.81 | 0.73 |
| **FinBERT fine-tuned (4 epochs, GPU, class weights)** | **0.87** | **0.84** |

Per-class breakdowns and confusion matrices are in `results/generalization_*.json`.

## 3. Training data & provenance

- **Financial PhraseBank** (Malo et al., 2014): formal sentences from company
  financial reports; ~61% neutral. Used for the baselines.
- **Twitter Financial News Sentiment** (`zeroshot/...`, MIT): short, ticker-tagged
  headlines/tweets; ~66% neutral. Used to fine-tune the transformers and as the
  OOD benchmark. Fetch via `scripts/fetch_real_news_dataset.py`.
- The bundled `sample_news.csv` is **synthetic** (template-generated) and is used
  only to exercise the pipeline — never for metrics. See the data folder README.

## 4. Intended use & users

- Educational / portfolio demonstration of an end-to-end sentiment pipeline.
- Quick sentiment reads on financial text, with explainability.

## 5. Out-of-scope / limitations

- **Not financial advice and not a trading signal.** The stock-extension LSTM is
  an engineering demo and performs at roughly chance level.
- **Domain sensitivity.** The fine-tuned transformers are strongest on short,
  `$TICKER`-style headlines (their training domain). On long formal prose they are
  less reliable; on Reddit/Twitter slang they are out of scope.
- **Class imbalance.** Neutral dominates both datasets; macro-F1 (not accuracy) is
  the primary metric for this reason.
- **Small in-domain test set** (n=226): single-run test numbers carry ±~3–4%
  variance; prefer the 5-fold CV figures.

## 6. Ethical considerations

- Sentiment models can encode dataset biases; predictions should not be used for
  automated financial decisions without human review.
- No PII is used; all data is public/licensed.

## 7. How to reproduce

```bash
# Baselines + evaluation
python src/train.py --model baselines && python src/evaluate.py --save

# Real OOD benchmark
python scripts/fetch_real_news_dataset.py
python src/integrate_news.py --action generalization --model svm

# Fine-tune (GPU recommended) — see notebooks/finetune_finbert_gpu.py
python scripts/finetune_finbert_news.py --model finbert --epochs 3 --batch-size 32
```

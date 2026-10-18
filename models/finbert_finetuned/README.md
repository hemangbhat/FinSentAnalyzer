---
language: en
license: other
library_name: transformers
pipeline_tag: text-classification
base_model: ProsusAI/finbert
datasets:
  - zeroshot/twitter-financial-news-sentiment
tags:
  - finance
  - sentiment-analysis
  - finbert
metrics:
  - accuracy
  - f1
model-index:
  - name: finbert-financial-news
    results:
      - task:
          type: text-classification
          name: Financial news sentiment
        dataset:
          name: Twitter Financial News Sentiment (validation)
          type: zeroshot/twitter-financial-news-sentiment
          split: validation
        metrics:
          - type: accuracy
            value: 0.874
          - type: f1
            name: macro-F1
            value: 0.841
---

# FinBERT fine-tuned on financial news headlines

`ProsusAI/finbert` fine-tuned to classify short financial news headlines as
**negative**, **neutral** or **positive**. Part of the
[FinSight / FinSentAnalyzer](https://github.com/hemangbhat/FinSentAnalyzer) project.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="<this-repo-id>")
clf("$ALLY - Ally Financial pulls outlook")
# [{'label': 'negative', 'score': ...}]
```

Labels: `0 = negative`, `1 = neutral`, `2 = positive`.

## Training

| Setting | Value |
|---|---|
| Base model | ProsusAI/finbert (BERT-base, ~110M parameters) |
| Data | `zeroshot/twitter-financial-news-sentiment` train split, 9,543 headlines (MIT) |
| Epochs | 4, best epoch restored by validation macro-F1 |
| Optimiser | AdamW, learning rate 2e-5, linear schedule with 10% warmup |
| Batch size / max length | 32 / 128 tokens |
| Loss | Cross-entropy with inverse-frequency class weights |
| Precision | Mixed precision (fp16) on a T4 GPU |

## Evaluation

On the dataset's validation split (2,388 headlines, held out from training):

| Metric | Value |
|---|---|
| Accuracy | 0.874 |
| Macro-F1 | 0.841 (95% bootstrap interval 0.823–0.858) |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.72 | 0.86 | 0.78 | 347 |
| neutral | 0.94 | 0.89 | 0.91 | 1,566 |
| positive | 0.81 | 0.85 | 0.83 | 475 |

For comparison, a TF-IDF + linear SVM trained on Financial PhraseBank scores
0.46 macro-F1 on the same headlines.

## Limitations

- **In-domain, not out-of-domain.** The validation split comes from the same
  source as the training data. Performance on other text (filings, earnings
  calls, non-English news) is untested.
- **Small selection bias.** The best epoch was chosen on the first 500 rows of
  the same validation split that is reported, so the scores are slightly
  optimistic.
- **Most errors are neutral headlines read as negative** (92 of 2,388).
- Headline-level only; a headline about two companies gets one label.
- Not financial advice and not a trading signal on its own.

## Licence

Training data: `zeroshot/twitter-financial-news-sentiment` (MIT). This
fine-tune inherits the terms of its base model, `ProsusAI/finbert`; check that
model's licence before reuse or commercial use.

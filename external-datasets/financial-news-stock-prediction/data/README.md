# Data provenance — SYNTHETIC

> ⚠️ **These files are synthetic.** They are generated locally, not collected
> from real sources. Do not present them as real news or as a generalization
> benchmark.

| File | How it's produced | What it is |
|------|-------------------|------------|
| `sample_news.csv` | `generate_dummy_news.py` (template + random words) | Fake headlines. The `source` column (Reuters/Bloomberg/…) is a **randomly assigned label**, not the real publisher. |
| `stock_data.csv` | `generate_stock_data.py` | Synthetic OHLCV-style series. |
| `final_dataset.csv` | derived from the two above | Synthetic features + labels. |

## Why this exists

To exercise the end-to-end pipeline (FinBERT scoring → daily aggregation →
feature engineering → charts → LSTM) without depending on a paid news API.

## To make this a real benchmark

Replace `sample_news.csv` with a licensed, real headline dataset that includes a
true `date`, `ticker`, `headline`, and `source`, and real OHLCV data (e.g., via
`yfinance`). Then the "cross-dataset generalization" and sentiment↔price
correlation analyses become meaningful.

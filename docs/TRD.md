# Technical Requirements Document (TRD)

## 1. System overview
The system is a multipage Streamlit application backed by Python ML modules and saved artifacts.

## 2. Components
- `app/`: user interface and page routing
- `src/`: preprocessing, training, inference, explainability, stock extension
- `models/`: baseline `.joblib` artifacts
- `results/`: evaluation and explanation outputs
- `external-datasets/`: nested stock prediction workflow

## 3. Runtime requirements
- Python 3.10+
- Streamlit
- scikit-learn
- pandas / NumPy
- PyTorch + Transformers for FinBERT
- Plotly
- SHAP
- yfinance
- joblib

## 4. Data requirements
- Financial PhraseBank raw text files for the primary task
- processed train/val/test CSVs
- optional external news/stock datasets for the extension

## 5. Technical constraints
- Keep heavy model loading cached
- Avoid loading unnecessary dependencies on the home page
- Keep baseline and transformer paths isolated
- Use robust file-path resolution from project root
- Ensure pages can run independently without import collisions

## 6. Non-functional requirements
- reproducibility
- clear logging
- deterministic splits where possible
- acceptable response time for single prediction
- graceful fallback when optional packages are unavailable
- Docker compatibility

## 7. Engineering standards
- formatting via Ruff
- type hints where practical
- tests for critical code paths
- no silent overclaiming in docs or UI
- consistent naming for labels, files, and model artifacts

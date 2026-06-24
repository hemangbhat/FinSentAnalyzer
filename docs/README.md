\
\# FinSentAnalyzer / Financial Sentiment Analyzer

End-to-end financial sentiment analysis platform for classifying financial text into **positive**, **neutral**, or **negative** sentiment. The project combines classical ML baselines, a pre-trained FinBERT transformer, SHAP-style explainability for baseline models, rule-based reasoning, and a 9-page Streamlit dashboard.

## What’s included
- 8 model variants: Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, MLP, Voting Ensemble, and pre-trained FinBERT
- 5-fold stratified cross-validation and GridSearchCV tuning
- Batch processing, sentiment trends, error analysis, explainability, and stock-direction extension
- CI/CD, Docker, reusable model loading, and deployment-ready structure

## Core stack
Python, scikit-learn, pandas, NumPy, Streamlit, Plotly, SHAP, Hugging Face Transformers, PyTorch, yfinance, joblib, GitHub Actions, Docker

## Repository layout
```text
app/                     # Streamlit multipage UI
src/                     # Training, inference, explainability, utilities
models/                  # Saved baseline model artifacts
results/                 # Evaluation outputs and explainability summaries
data/                    # Raw and processed dataset files
external-datasets/       # Nested stock-prediction extension
tests/                   # Automated checks
```

## Main use cases
- Analyze a single financial headline or sentence
- Batch-score a CSV/TXT file of headlines
- Explain why the model predicted a sentiment
- Inspect model comparison, error patterns, and sentiment trends
- Run the stock-direction extension as a separate workflow

## Notes
- FinBERT is integrated as a **pre-trained** transformer in the current codebase.
- The “reasoning” module is **rule-based** and template-driven rather than an external LLM API.
- The project is designed to be honest, reproducible, and easy to demo.

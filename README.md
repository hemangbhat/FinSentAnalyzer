# 📈 Financial Sentiment Analyzer

> **An end-to-end ML pipeline for classifying financial text sentiment with 7 trained baseline models + pre-trained FinBERT, SHAP explainability, rigorous cross-validation, a FastAPI inference service, and a professional 9-page Streamlit dashboard.**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-yellow.svg)](https://huggingface.co/ProsusAI/finbert)

---

## 🔗 Live Demo

> **Deployed app:** `https://<your-app>.streamlit.app`  ← _replace with your public URL after deploying (see [Deployment](#deployment))._

![FinSight demo](docs/demo.gif)

> _Demo GIF placeholder._ Record a ~15s screen capture of the Single Analysis and
> Batch flows and save it as `docs/demo.gif` (e.g. with [ScreenToGif](https://www.screentogif.com/)
> or `ffmpeg`). It will render here automatically.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Dataset](#dataset)
- [Models](#models)
- [ML Rigor: Cross-Validation & Hyperparameter Tuning](#ml-rigor-cross-validation--hyperparameter-tuning)
- [Generalization (Real Data)](#generalization-real-licensed-data)
- [Pipeline Demo (Synthetic)](#pipeline-demo-on-synthetic-headlines)
- [Explainability](#explainability)
- [Dashboard Pages](#dashboard-pages)
- [CLI Reference](#cli-reference)
- [Testing](#testing)
- [Inference API](#inference-api)
- [Monitoring & Model Registry](#monitoring--model-registry)
- [Retraining](#retraining)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Interview Q&A](#interview-qa)

---

## Overview

Financial sentiment analysis is a critical tool for quantitative finance, risk management, and market intelligence. This project classifies financial text into three categories — **positive**, **neutral**, and **negative** — using both classical ML baselines and transformer-based models.

### What Makes This Project Stand Out

| Dimension | What We Do |
|-----------|------------|
| **8 Models** | 7 trained baselines (incl. Voting Ensemble) + pre-trained FinBERT (zero-shot) |
| **ML Rigor** | 5-fold stratified cross-validation + GridSearchCV hyperparameter tuning |
| **Explainability** | SHAP feature importance + per-prediction word highlighting |
| **Error Analysis** | Sankey misclassification diagrams + confidence calibration insights |
| **Generalization** | Honest OOD test on a real, MIT-licensed news dataset: TF-IDF 0.46 macro-F1 → DistilBERT fine-tuned 0.73 |
| **Dashboard** | Professional 9-page Streamlit app with dark fintech theme |
| **CI/CD** | GitHub Actions (lint + format + type check + tests) |
| **Deployment** | Docker + Streamlit Cloud + HuggingFace Spaces ready |

---

## Key Features

- **Real-Time Prediction** — Enter any financial text and get instant sentiment with confidence scores
- **Batch Processing** — Upload CSV/TXT files for bulk sentiment analysis with aggregate KPIs
- **8 Trained Models** — Compare Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, MLP, and a Voting Ensemble (7 trained baselines), plus a pre-trained FinBERT (zero-shot, not fine-tuned by this project)
- **5-Fold Cross-Validation** — Stratified CV with mean ± std for all metrics
- **GridSearchCV Tuning** — Automated hyperparameter optimization for SVM and Gradient Boosting
- **SHAP Explainability** — Global feature importance with per-class drivers
- **Pipeline demo (synthetic data)** — Runs the model over ~2,700 *template-generated* headlines to demonstrate the scoring/aggregation/correlation pipeline. This is **not** real news and **not** a generalization benchmark (see [Limitations](#limitations)).
- **Real generalization test** — Honest out-of-distribution evaluation on a real, MIT-licensed dataset (`zeroshot/twitter-financial-news-sentiment`, n=2,388): the TF-IDF baseline drops to macro-F1 **0.46** OOD, and fine-tuning DistilBERT on in-domain news recovers it to **0.73** — both reported transparently.
- **Stock Prediction Extension** — Experimental end-to-end flow (yfinance → FinBERT → LSTM). The LSTM is **chance-level** on this small/synthetic data — it's an engineering showcase, not a real forecast.
- **Error Analysis** — Inspect misclassified examples with Sankey diagrams and confidence histograms
- **Deep Linguistic Analysis** — Named Entity Recognition, rule-based Chain-of-Thought reasoning, Loughran-McDonald financial lexicon matching

---

## System Architecture

```mermaid
flowchart TD
    %% Data Sources
    subgraph Data ["Data Layer"]
        A["Financial PhraseBank"] --> B["src/preprocess.py"]
        B --> C["Processed Data Split"]
        N["News Headlines Dataset"] --> X["src/integrate_news.py"]
    end

    %% Training Pipeline
    subgraph Train ["Training and Evaluation"]
        C --> D{"Model Selection"}
        D -->|"TF-IDF + ML baselines"| E["src/train.py"]
        D -->|"HuggingFace/PyTorch"| F["src/model.py"]
        E -->|"5-Fold CV + GridSearchCV"| G["Saved Models .joblib"]
        F --> G
        G --> H["src/evaluate.py"]
        H --> HR["results/ JSON artifacts"]
    end

    %% Explainability
    subgraph Explain ["Explainability Layer"]
        G --> K["src/explain.py"]
        G --> SH["src/shap_explain.py"]
        SH --> HR
    end

    %% Cross-Dataset Validation
    subgraph Cross ["Cross-Dataset Validation"]
        G -.->|"Load Models"| X
        X -->|"Sentiment-Stock Correlation"| HR
    end

    %% Inference Engine
    subgraph Engine ["Inference Engine"]
        I["Input Text / Batch CSV"] --> J["src/predict.py"]
        G -.->|"Load Models"| J
        J --> L["src/nlp_advanced.py"]
        J --> M["src/llm_explain.py (template-based)"]
    end

    %% Dashboard
    subgraph App ["Streamlit Dashboard UI"]
        O["app/shared.py"] -.->|"CSS"| P["app/app.py"]
        P --- Q["Pages"]
        Q --> R["Single Analysis"]
        Q --> S["Batch Processing"]
        Q --> T["Explainability + SHAP"]
        Q --> U["Sentiment Trends"]
        Q --> V["Error Analysis"]
        Q --> W["Stock Prediction Extension"]

        J -.->|"Predictions"| Q
        K -.->|"Word Importances"| T
        SH -.->|"SHAP Values"| T
        HR -.->|"Results JSON"| U
    end
```

---

## Project Structure

```
financial-sentiment-analyzer/
├── app/                            # Streamlit Dashboard
│   ├── app.py                      # Main entry point (Home page)
│   ├── shared.py                   # Shared CSS, components, cached model loading
│   └── pages/
│       ├── 1_Single_Analysis.py    # Real-time single text prediction
│       ├── 2_Batch_Processing.py   # CSV/TXT bulk analysis
│       ├── 3_Explainability.py     # Word importance + SHAP global analysis
│       ├── 4_Word_Insights.py      # Financial lexicon exploration
│       ├── 5_Deep_Analysis.py      # NER, Chain-of-Thought, linguistic decomposition
│       ├── 6_Model_Info.py         # Model registry with live metrics
│       ├── 7_Sentiment_Trends.py   # Sentiment vs stock price correlation
│       ├── 8_Error_Analysis.py     # Misclassified examples & confusion patterns
│       └── 9_Stock_Prediction_Extension.py  # Full stock pipeline extension page
│
├── src/                            # Core ML Pipeline
│   ├── preprocess.py               # Data loading, cleaning, train/val/test splitting
│   ├── train.py                    # Model training, 5-fold CV, GridSearchCV tuning
│   ├── evaluate.py                 # Metrics, confusion matrix, model comparison
│   ├── predict.py                  # Unified inference for baseline + transformer models
│   ├── model.py                    # PyTorch FinBERT fine-tuning
│   ├── explain.py                  # Word importance & explainability
│   ├── shap_explain.py             # SHAP-based global feature importance
│   ├── integrate_news.py           # Cross-dataset evaluation & sentiment-stock correlation
│   ├── stock_extension.py          # Bridge to nested stock prediction project
│   ├── nlp_advanced.py             # Financial NLP: NER, lexicon, text features
│   ├── finbert_pretrained.py       # Pre-trained FinBERT wrapper (zero-shot)
│   ├── benchmark_finbert.py        # FinBERT test-set benchmark (appends to evaluation_results.json)
│   ├── llm_explain.py              # Template-based explanation generation
│   └── utils.py                    # Constants, paths, logging, dynamic model metrics
│
├── data/
│   ├── raw/                        # Financial PhraseBank (2,264 sentences)
│   └── processed/                  # Stratified train/val/test CSV splits
│
├── external-datasets/              # Synthetic headline set (pipeline demo only)
│
├── models/                         # Saved trained models
│   ├── baseline_logreg.joblib
│   ├── baseline_naive_bayes.joblib
│   ├── baseline_svm.joblib
│   ├── baseline_random_forest.joblib
│   ├── baseline_gradient_boosting.joblib
│   ├── baseline_mlp.joblib
│   ├── baseline_ensemble.joblib
│   └── finbert/                    # Fine-tuned FinBERT model
│
├── results/                        # Auto-generated evaluation artifacts
│   ├── evaluation_results.json     # All model metrics
│   ├── cv_results.json             # 5-fold cross-validation results
│   ├── cross_dataset_results.json  # News headline evaluation + stock correlations
│   ├── shap_gradient_boosting.json # SHAP feature importance (GB)
│   └── shap_svm.json              # SHAP feature importance (SVM)
│
├── tests/                          # Automated test suite (88 tests)
│   ├── conftest.py                 # Shared fixtures
│   ├── test_utils.py
│   ├── test_preprocess.py
│   ├── test_predict.py
│   ├── test_nlp_advanced.py
│   ├── test_llm_enhanced.py
│   ├── test_integrate_news.py
│   ├── test_stock_extension.py
│   └── test_shap_explain.py
│
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis
│
├── .github/workflows/ci.yml       # GitHub Actions CI pipeline
├── .streamlit/config.toml          # Streamlit theme configuration
├── Dockerfile                      # Production Docker image
├── Makefile                        # Build automation (7 targets)
├── pyproject.toml                  # ruff, mypy, pytest configuration
├── data_manifest.toml              # Dataset versioning & reproducibility
├── requirements.txt                # Full dependencies
└── requirements-deploy.txt         # Lightweight deployment dependencies
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/financial-sentiment-analyzer.git
cd financial-sentiment-analyzer
```

### Step 2: Create Virtual Environment & Install Dependencies

```bash
# Using Makefile (recommended)
make install

# OR manually
python -m venv venvMlProject
venvMlProject\Scripts\activate        # Windows
source venvMlProject/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Step 3: Download the Dataset

Download the **Financial PhraseBank** dataset from [HuggingFace](https://huggingface.co/datasets/financial_phrasebank) and place it in:

```
data/raw/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt
```

### Step 4: Preprocess the Data

Run the EDA notebook to generate train/val/test splits:

```bash
jupyter notebook notebooks/eda.ipynb
```

This creates `data/processed/train.csv`, `val.csv`, and `test.csv` with an **80/10/10 stratified split**.

---

## How to Run

### Launch the Dashboard (Quickest Way)

```bash
streamlit run app/app.py
```

Open **http://localhost:8501** in your browser. Use the sidebar to navigate 9 pages.

### Train All Models

```bash
# Train all 6 baseline models + ensemble
python src/train.py --model baselines

# Run 5-fold cross-validation
python src/train.py --model cv

# Run GridSearchCV hyperparameter tuning
python src/train.py --model tune

# Train a specific model
python src/train.py --model svm
python src/train.py --model gradient_boosting

# Fine-tune a transformer on in-domain real news (closes the OOD gap)
python scripts/fetch_real_news_dataset.py
python scripts/finetune_finbert_news.py --model distilbert --epochs 1 --max-train 6000
```

### Evaluate Models

```bash
# Compare all models on test set (saves results to results/)
python src/evaluate.py --save
```

### Run SHAP Explainability

```bash
python src/shap_explain.py --model gradient_boosting
python src/shap_explain.py --model svm
```

### Run Pipeline Demo (synthetic news)

```bash
python src/integrate_news.py --action evaluate --model svm
```

### Run Full Stock Prediction Extension (CLI)

```bash
python src/stock_extension.py --ticker AAPL --days-back 120 --epochs 8 --seq-len 5
```

### Single Prediction (CLI)

```bash
python src/predict.py --text "Revenue increased 25% driven by strong demand" --model svm
```

---

## Dataset

### Primary Dataset: Financial PhraseBank

| Property | Value |
|----------|-------|
| **Source** | [Malo et al. (2014)](https://huggingface.co/datasets/financial_phrasebank) |
| **Total Samples** | 2,264 sentences |
| **Agreement** | 100% annotator agreement (AllAgree subset) |
| **Classes** | Positive (570), Neutral (1,391), Negative (303) |
| **Domain** | Finnish company financial reports |
| **Split** | 80% train (1,807) / 10% val (226) / 10% test (226) |
| **Strategy** | Stratified random split (seed=42) |

### Pipeline-Demo Dataset: Synthetic Headlines

> ⚠️ **Synthetic, not real.** These headlines are template-generated by
> `external-datasets/.../generate_dummy_news.py`; the `source` column is a
> randomly assigned label, not the true publisher. They exist only to exercise
> the pipeline. See [`.../data/README.md`](external-datasets/financial-news-stock-prediction/data/README.md).

| Property | Value |
|----------|-------|
| **Total Headlines** | 2,688 (synthetic) |
| **Origin** | Locally generated templates (not collected from any outlet) |
| **Tickers** | AAPL, TSLA, AMZN |
| **Purpose** | Pipeline demonstration only — **not** a generalization benchmark |
| **Stock Data** | Synthetic OHLCV series |

All dataset metadata is documented in `data_manifest.toml` for reproducibility.

---

## Models

### Baseline Models (TF-IDF + Classifiers)

All baselines use TF-IDF vectorization with unigrams + bigrams (max 10,000 features).
Test metrics are on the held-out test split (**n = 226**, ~61% neutral, so a
majority-class baseline scores ~62% accuracy). Because the test set is small,
treat single-run test numbers as indicative (±~3–4%); the **5-fold CV F1 (macro)**
column is the more reliable signal.

| Model | Test Acc (n=226) | Test F1 (macro) | CV F1 macro (5-fold) | Speed |
|-------|:------------:|:--------------:|:--------------:|:-----:|
| Gradient Boosting | 94.25% | 0.920 | 0.835 ± 0.008 | Medium |
| **SVM (Linear)** | 92.48% | 0.902 | **0.838 ± 0.016** | Fast |
| Logistic Regression | 90.71% | 0.884 | 0.821 ± 0.021 | Very Fast |
| Random Forest | 88.50% | 0.838 | 0.755 ± 0.028 | Medium |
| Naive Bayes | 88.05% | 0.848 | 0.784 ± 0.016 | Very Fast |
| MLP Neural Network | 88.05% | 0.841 | 0.801 ± 0.012 | Medium |
| Voting Ensemble | (probability-calibrated; see error analysis) | | | Slow |

> On the test split Gradient Boosting scores highest, but under 5-fold CV SVM and
> Gradient Boosting are statistically tied (0.838 vs 0.835, overlapping std). SVM
> is the default for its speed and stability.

### Transformer Models

| Model | Type | Notes |
|-------|------|-------|
| **FinBERT** | ProsusAI/finbert (110M) | Used **pre-trained / zero-shot** for the dashboard. |
| **DistilBERT (fine-tuned)** | distilbert-base-uncased (66M) | **Fine-tuned by this project** on in-domain financial news (`scripts/finetune_finbert_news.py`). Achieves **0.81 acc / 0.73 macro-F1** on the real news test set vs 0.67 / 0.46 for the TF-IDF baseline. Auto-detected by the dashboard/API once `models/distilbert_finetuned/` exists. |

---

## ML Rigor: Cross-Validation & Hyperparameter Tuning

### 5-Fold Stratified Cross-Validation

Every baseline model is evaluated with 5-fold stratified CV on the combined train+val set (2,033 samples). This prevents overfitting to a single random split.

```
┌─────────────────────────────────────────────────────────────┐
│              CROSS-VALIDATION RESULTS (5-FOLD)              │
├──────────────────┬──────────────────┬───────────────────────┤
│ Model            │ Accuracy         │ F1 (macro)            │
├──────────────────┼──────────────────┼───────────────────────┤
│ svm              │ 0.8849 ± 0.0094  │ 0.8380 ± 0.0163      │
│ gradient_boost   │ 0.8810 ± 0.0047  │ 0.8346 ± 0.0078      │
│ logreg           │ 0.8702 ± 0.0155  │ 0.8209 ± 0.0208      │
│ mlp              │ 0.8652 ± 0.0089  │ 0.8010 ± 0.0120      │
│ naive_bayes      │ 0.8505 ± 0.0087  │ 0.7841 ± 0.0159      │
│ random_forest    │ 0.8392 ± 0.0160  │ 0.7546 ± 0.0278      │
└──────────────────┴──────────────────┴───────────────────────┘
Best model by mean F1 macro: SVM (0.838 ± 0.016)
```

### GridSearchCV Hyperparameter Tuning

Two models are tuned with GridSearchCV (5-fold CV scorer = F1 macro):

| Model | Parameters Tuned | Search Space | Best Params |
|-------|-----------------|--------------|-------------|
| **SVM** | C, max_iter | C ∈ {0.1, 0.5, 1.0, 2.0, 5.0} | Auto-saved |
| **Gradient Boosting** | n_estimators, max_depth, learning_rate | 18 combinations | Auto-saved |

Best models are automatically saved to `models/` after tuning.

---

## Generalization (real, licensed data)

The model is trained on Financial PhraseBank and evaluated on a **real,
human-labeled, out-of-distribution** dataset:
[`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
(MIT license, validation split, n = 2,388). Fetch it with
`python scripts/fetch_real_news_dataset.py`, then evaluate with
`python src/integrate_news.py --action generalization --model svm`.

| Setting | Accuracy | Macro-F1 |
|---------|:--------:|:--------:|
| In-domain (PhraseBank test, n=226) | 0.92 | 0.90 |
| In-domain (5-fold CV) | — | 0.84 ± 0.02 |
| **PhraseBank model → real news (OOD, n=2,388)** | **0.67** | **0.46** |
| Majority-class baseline (OOD) | 0.66 | — |
| **DistilBERT fine-tuned on in-domain news → same test** | **0.81** | **0.73** |

**Honest takeaway:** the PhraseBank-trained model **generalizes poorly** to a
different domain — on real financial-news headlines its accuracy barely exceeds
the majority-class baseline (0.67 vs 0.66) and macro-F1 nearly halves. **The gap
is domain mismatch, not a broken pipeline:** fine-tuning DistilBERT on in-domain
news (1 epoch, 6k examples) lifts macro-F1 from 0.46 → **0.73** and accuracy to
**0.81** on the same held-out set. Reproduce with:

```bash
python scripts/fetch_real_news_dataset.py
python scripts/finetune_finbert_news.py --model distilbert --epochs 1 --max-train 6000
python src/integrate_news.py --action generalization --model svm   # baseline number
```

> **Achieved here:** the 0.73 macro-F1 above is from a **DistilBERT, 1-epoch,
> 6k-example, CPU** run — deliberately bounded for reproducibility on a laptop.
>
> **GPU upgrade (Kaggle / Colab free tier):**
> Open `notebooks/finetune_finbert_gpu.py`, set runtime to GPU T4, and click
> Run All (~15 min). It fine-tunes FinBERT for 3 epochs, evaluates, packages
> the weights + result JSON for download, and tells you exactly where to copy
> them. Mixed precision (AMP) is enabled automatically on CUDA.
> Expected: acc ≈ 0.85–0.88, macro-F1 ≈ 0.79–0.83.
>
> After running, copy the weights into `models/finbert_finetuned/`, copy
> `finetune_results.json` into `results/`, run
> `python src/integrate_news.py --action generalization --model finbert`,
> update this table with the real numbers, and push.

## Pipeline Demo on Synthetic Headlines

> ⚠️ **Separate from the above.** The bundled `sample_news.csv` (~2,688 rows) is
> *template-generated* and used only to exercise the sentiment→aggregation→chart
> →LSTM plumbing. Its sentiment-vs-price "correlations" are meaningless by
> construction and must not be read as market evidence.

The project computes **Pearson correlation** between daily aggregated sentiment
and (synthetic) stock returns purely to exercise the analysis code:

| Ticker | Same-Day | Next-Day | Data Points |
|--------|:-------------------:|:-------------------:|:-----------:|
| AAPL | 0.056 | -0.085 | 448 |
| TSLA | -0.026 | 0.016 | 448 |
| AMZN | 0.028 | -0.035 | 448 |

On synthetic data these correlations are meaningless by construction — they are
shown only to demonstrate the end-to-end flow. Do not interpret them as evidence
about real markets.

---

## Explainability

### Local Explainability (Per-Prediction)

Each prediction comes with **word-importance highlighting** showing which tokens drove the model's decision. Uses TF-IDF coefficients to color-code words by their contribution to each sentiment class.

### Global Explainability (SHAP)

SHAP (SHapley Additive exPlanations) provides global feature importance:

- **Tree models** (Random Forest): SHAP TreeExplainer
- **Gradient Boosting**: Feature importance + probability-weighted per-class approximation
- **Linear models** (SVM, LogReg): Coefficient-based analysis per class

Results include:
- Top 30 most important features globally
- Per-class feature drivers (positive, neutral, negative) with direction indicators
- Saved to `results/shap_{model}.json`

---

## Theoretical Foundations & Interpretability

This project is built rigidly on principles of **Interpretable Machine Learning** and **Financial Text Mining**. Every prediction and metric generated by the system can be theoretically justified and mathematically traced.

### 1. Vectorization & Feature Engineering
Before classical models can process text, it is transformed using **TF-IDF (Term Frequency - Inverse Document Frequency)** with unigrams and bigrams.
*   **Why TF-IDF?** Unlike simple word counts (Bag-of-Words), TF-IDF penalizes common words (like "the", "and") and amplifies words that are rare globally but frequent in specific texts (like "dividend", "missed").
*   **Advanced NLP Features:** In the Deep Analysis module, the text is also decomposed into structural features: Named Entity Recognition (NER) extracts Organizations, Dates, and Monetary values, while financial lexicons detect specialized vocabularies (e.g., Loughran-McDonald financial sentiment dictionary).

### 2. Model Decision Boundaries (How they learn)
The 8 models map the TF-IDF feature space to sentiment classes in different ways:
*   **Linear Models (SVM, Logistic Regression):** These find a mathematical hyperplane separating the classes. They are highly interpretable because we can look directly at the coefficients (weights) assigned to each word. E.g., if the word "profit" has a weight of +2.5 for the Positive class, we know exactly why the model made its choice.
*   **Tree/Ensemble Models (Random Forest, Gradient Boosting):** These build decision trees that split the data based on word occurrences. They capture non-linear relationships (e.g., the combination of "revenue" and "dropped" is negative).
*   **Transformer Models (FinBERT):** FinBERT uses self-attention mechanisms to understand the *context* and *sequence* of words, capturing semantic meaning that classical bag-of-words models miss (e.g., distinguishing between "a positive outcome" and "tested positive for a virus").

### 3. Interpretability & Explainability (Opening the Black Box)
We use two explicit frameworks to ensure no decision is a "black box":
*   **Local Word Importance (Lexicon & Weighting):** For a single sentence, we project the model's coefficients back onto the tokens present in the text, generating a color-coded heatmap of words that drove the prediction.
*   **Global SHAP Analysis:** SHAP is a game-theoretic approach that assigns each feature an importance value for a particular prediction. We aggregate these locally computed values to determine global feature importance (e.g., globally, the word "growth" shifts the model probability toward Positive by 15%).

### 4. Evaluation Rigor & Error Analysis
A model that scores 99% accuracy on a random split is likely memorizing data. We use **5-Fold Stratified Cross-Validation** to ensure the model's performance is stable across 5 entirely different training/testing subsets, maintaining the original class imbalance (mostly Neutral) in each fold. 
Furthermore, the **Error Analysis** module mathematically breaks down failures mapping the joint distribution of $P(Predicted = Y | True = X)$. By comparing the *Confidence Distribution* of accurate vs inaccurate predictions, we measure the model's **Calibration** (a well-calibrated model is unsure when it makes mistakes, allowing you to set a confidence threshold for manual review).

---

## Dashboard Pages

### 🏠 Home
Landing page with 9 feature cards linking to each module.

### 📝 1. Single Analysis
- Enter any financial text
- Get instant sentiment prediction with confidence score
- View probability distribution (bar chart)
- Template-based natural language explanation

### 📁 2. Batch Processing
- Upload CSV or TXT files
- Process hundreds of texts at once
- Aggregate KPIs: sentiment distribution, market outlook
- Download results as CSV

### 🔍 3. Explainability
- **Word importance**: Color-coded token contributions per prediction
- **SHAP Global Analysis**: Top features, per-class drivers
- Run SHAP live or view pre-computed results

### 💡 4. Word Insights
- Browse top positive, negative, and neutral vocabulary
- Understand which words most influence predictions
- Financial lexicon coverage

### 🧠 5. Deep Analysis
- Named Entity Recognition (NER): Companies, amounts, dates
- Chain-of-Thought reasoning: Step-by-step sentiment logic
- Linguistic decomposition and entity-specific metrics

### 📊 6. Model Info
- Model registry showing all 8 models
- Live metrics from `evaluation_results.json`
- Architecture details, speed indicators, feature descriptions
- Comparison table with accuracy, F1, precision, recall

### 📈 7. Sentiment Trends
- Interactive time-series: sentiment vs stock price for AAPL, TSLA, AMZN
- Pearson correlation heatmaps (same-day + next-day)
- SHAP feature importance bar charts
- 5-Fold CV comparison with error bars

### 🔬 8. Error Analysis
- Summary KPIs: total, correct, errors, accuracy
- **Sankey diagram**: Misclassification flow (true label → predicted label)
- **Confusion pair ranking**: Top 5 error patterns with percentages
- **Confidence histogram**: Correct vs error distribution
- **Calibration insight**: Auto-generated (well-calibrated vs overconfident)
- **Misclassified examples table**: Filterable by true label, sortable by confidence

### 📉 9. Stock Prediction Extension
- Runs the full nested `financial-news-stock-prediction` flow inside the main app
- Downloads ticker OHLCV data with yfinance
- Fetches live ticker news and scores sentiment with FinBERT
- Engineers technical + sentiment features and trains an LSTM
- Predicts next-day direction (UP/DOWN) with confidence
- Reports LSTM test accuracy and can persist trained weights to `models/lstm_{TICKER}.pt`

---

## CLI Reference

### Training

```bash
python src/train.py --model all           # Train everything (baselines + transformers)
python src/train.py --model baselines     # Train all 6 baselines + ensemble
python src/train.py --model cv            # Run 5-fold cross-validation
python src/train.py --model tune          # GridSearchCV for SVM + Gradient Boosting
python src/train.py --model svm           # Train a specific model
python src/train.py --model finbert       # Fine-tune FinBERT
```

### Evaluation

```bash
python src/evaluate.py --save             # Evaluate all models, save results
python src/evaluate.py --model svm        # Evaluate a specific model
```

### Prediction

```bash
python src/predict.py --text "Text here" --model svm
python src/predict.py --file input.csv --model logreg --output results.csv
python src/predict.py --list-models       # Show available models
```

### Explainability

```bash
python src/shap_explain.py --model gradient_boosting
python src/shap_explain.py --model svm
python src/shap_explain.py --model logreg --plot   # Save SHAP summary plot
```

### Pipeline Demo (synthetic news)

```bash
python src/integrate_news.py --action evaluate --model svm
python src/integrate_news.py --action correlate --model svm
python src/integrate_news.py --action predict --model gradient_boosting
```

### Stock Prediction Extension

```bash
python src/stock_extension.py --ticker AAPL --days-back 120 --epochs 8 --seq-len 5
python src/stock_extension.py --ticker TSLA --start 2025-01-01 --end 2025-06-30 --no-live-news
python src/stock_extension.py --ticker AAPL --days-back 120 --save-model   # persist LSTM weights
```

---

## Testing

The project has **88 automated tests** across 8 test files:

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Run a specific test file
python -m pytest tests/test_predict.py -v
python -m pytest tests/test_integrate_news.py -v
python -m pytest tests/test_stock_extension.py -v
python -m pytest tests/test_shap_explain.py -v
```

### Test Coverage

| Test File | Tests | Covers |
|-----------|:-----:|--------|
| `test_utils.py` | 15 | Path helpers, logging, constants, model info |
| `test_preprocess.py` | 7 | Text cleaning, label encoding, data splits |
| `test_predict.py` | 6 | Model loading, single/batch prediction |
| `test_nlp_advanced.py` | 14 | Text processing, NER, lexicon, features |
| `test_llm_enhanced.py` | 10 | Thought steps, market outlook, confidence |
| `test_integrate_news.py` | 7 | News loading, sentiment prediction, trends |
| `test_stock_extension.py` | 6 | Pipeline orchestration, news fallbacks, model-path logic, save-model behavior |
| `test_shap_explain.py` | 6 | SHAP analysis, per-class features, JSON save |

### CI/CD Pipeline

GitHub Actions runs on every push/PR:

```yaml
- Ruff lint         # Code quality
- Ruff format check # Formatting consistency
- mypy type check   # Static type analysis
- pytest            # Automated tests
```

---

## Inference API

Beyond the dashboard, the same models are exposed as a small, reusable FastAPI
service (`api/main.py`) so other apps can consume predictions programmatically.

```bash
uvicorn api.main:app --reload --port 8000
# Interactive docs at http://localhost:8000/docs
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness + loaded models |
| `GET`  | `/models` | List available trained models |
| `POST` | `/predict` | Classify a single text |
| `POST` | `/predict/batch` | Classify a list of texts |
| `GET`  | `/metrics` | Counters + latency percentiles |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The company reported record quarterly profit.", "model": "baseline_svm"}'
# → {"label":"positive","confidence":0.95,"probabilities":{...},"model":"baseline_svm"}
```

**Hardening for public use** (all configurable via environment variables):
- **Authentication** — set `FINSIGHT_API_KEY` to require an `X-API-Key` header on `/predict`, `/predict/batch`, and `/metrics` (returns 401 otherwise). If unset, the API runs open for local dev and logs a warning.
- **Input validation** — non-empty text, `FINSIGHT_MAX_TEXT_CHARS` (default 5000), `FINSIGHT_MAX_BATCH_SIZE` (default 256), enforced by Pydantic.
- **Rate limiting** — IP-based via `slowapi`: `FINSIGHT_RATE_LIMIT` (default `60/minute`), `FINSIGHT_BATCH_RATE_LIMIT` (default `10/minute`); returns HTTP 429 when exceeded. Set `FINSIGHT_REDIS_URL` to share limits across workers/replicas (in-memory otherwise), and `FINSIGHT_TRUST_PROXY=true` only behind a trusted proxy so `X-Forwarded-For` is honored correctly.
- **Monitoring** — per-request latency header (`X-Process-Time-ms`), in-process metrics at `/metrics`, and a structured JSONL audit trail at `logs/predictions.jsonl`.

The Streamlit batch page also caps rows (`FINSIGHT_MAX_BATCH_ROWS`, default 2000) and uses chunked batch inference; the Stock Extension page can be disabled on shared hosts with `FINSIGHT_DISABLE_HEAVY=true`.

## Monitoring & Model Registry

- **Monitoring** (`src/monitoring.py`) — dependency-free structured logging + an
  in-process metrics registry (request/error counts, p50/p95/p99 latency).
  Prediction events are appended to `logs/predictions.jsonl` for an audit trail.
- **Model registry** (`src/registry.py`) — records a versioned snapshot of every
  model artifact (SHA-256 hash, size, timestamp) joined with its evaluation
  metrics to `models/registry.json` — reproducible lineage with zero infra.

```bash
python src/registry.py --update      # refresh models/registry.json
python src/registry.py --update --mlflow   # also log to MLflow (if installed)
python src/registry.py --show
```

MLflow is supported as an optional integration: with `mlflow` installed,
`--mlflow` logs params + metrics to a local `mlruns/` store. The registry works
fully without it.

## Retraining

The dataset is static, so retraining is occasional. A single command runs the
full reproducible pipeline and refreshes the registry:

```bash
python scripts/retrain.py            # train baselines + CV + evaluate + registry
python scripts/retrain.py --skip-cv  # faster
```

A scheduled GitHub Actions workflow (`.github/workflows/retrain.yml`) runs
monthly (and on manual dispatch), retrains, runs the test suite, and uploads
regenerated artifacts for review. Promotion of new weights is a manual, reviewed
step. See [docs/RETRAINING.md](docs/RETRAINING.md) for the full checklist and
promotion criteria.

## Deployment

### Option 1: Streamlit Cloud (Easiest)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point to `app/app.py`
4. Deploy

### Option 2: Docker

```bash
docker build -t financial-sentiment .
docker run -p 8501:8501 financial-sentiment
```

The Dockerfile is a **full-featured build**: Python 3.10-slim base, CPU-only PyTorch
(installed from the PyTorch CPU index to keep the image small), and the complete
`requirements-deploy.txt`. It copies the core app plus the nested stock-prediction
extension (`external-datasets/.../src` and `data`), so **all 9 pages — including
FinBERT and the LSTM Stock Prediction Extension — work in the container**. A
Python-stdlib health check hits the Streamlit `/_stcore/health` endpoint.

> First container start downloads the FinBERT weights (~500MB) from HuggingFace on
> demand. The model revision is pinned for reproducibility. For faster cold starts,
> bake the weights into the image or mount a `HF_HOME` cache volume.

### Option 3: HuggingFace Spaces

1. Create a new Space (Streamlit SDK)
2. Upload the `app/`, `src/`, `models/`, `data/`, and `external-datasets/` directories
3. Add `requirements-deploy.txt` as `requirements.txt`

### Option 4: Scaled split architecture (UI + API + load balancer)

For a production-shaped deployment, the UI and the model service run as
**separate tiers**. `docker-compose.yml` wires them together:

```
Browser ─► Streamlit UI ─► nginx (load balancer) ─► FastAPI replicas ─► models
                                                          │
                                                        Redis (shared rate limit)
```

```bash
# 3 API replicas behind the nginx load balancer
FINSIGHT_API_KEY=your-secret docker compose up --build --scale api=3
# UI:        http://localhost:8501
# API (LB):  http://localhost:8080/health
```

How it works:
- The UI sets `FINSIGHT_API_URL=http://lb:80`, so it becomes a **stateless
  presentation layer** — `ui.load_predictor` returns a `RemotePredictor` that
  calls the API instead of loading models in-process.
- `nginx` (`deploy/nginx.conf`) load-balances across the scaled `api` replicas
  and forwards the real client IP via `X-Forwarded-For`.
- The API runs with `FINSIGHT_TRUST_PROXY=true` (so rate limits key on the real
  caller) and `FINSIGHT_REDIS_URL` (so limits are **shared across all replicas**).
- Set `FINSIGHT_API_KEY` to require an `X-API-Key` header end-to-end.

This separation lets the model tier scale independently of the UI and removes
the single-process bottleneck of a standalone Streamlit app.

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| **ML Framework** | scikit-learn (Pipelines, GridSearchCV, StratifiedKFold) |
| **Deep Learning** | PyTorch, HuggingFace Transformers (FinBERT) |
| **NLP** | TF-IDF, spaCy-inspired NER, financial lexicons |
| **Explainability** | SHAP (TreeExplainer, coefficient analysis) |
| **Data** | pandas, numpy |
| **Visualization** | Plotly, matplotlib |
| **Dashboard** | Streamlit (multipage app) |
| **Serialization** | joblib (models), JSON (results) |
| **Testing** | pytest |
| **Linting** | ruff, mypy |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker |
| **Config** | pyproject.toml, data_manifest.toml |

---

## Interview Q&A

### Q: How do you know the model generalizes beyond the training data?

**A:** I measured it honestly — and then fixed it. Trained on Financial PhraseBank, the TF-IDF model scores macro-F1 0.84 in-domain but only **0.46** on a real, MIT-licensed out-of-distribution set (`zeroshot/twitter-financial-news-sentiment`, n=2,388), barely beating the majority-class baseline. That exposed the gap as domain mismatch. So I fine-tuned DistilBERT on in-domain news (1 epoch, 6k examples), which lifts macro-F1 to **0.73** and accuracy to **0.81** on the same held-out set. I report both numbers — the weak baseline and the fix — rather than hiding the gap.

### Q: Did you use cross-validation?

**A:** Yes, 5-fold stratified cross-validation on all 6 baseline models. SVM achieved the best mean F1 macro of 0.838 ± 0.016, while Gradient Boosting had the lowest variance at ± 0.008. The stratification ensures each fold has the same class distribution.

### Q: How did you tune hyperparameters?

**A:** GridSearchCV with 5-fold CV scoring on F1 macro for two models: SVM (tuning C and max_iter across 10 combinations) and Gradient Boosting (tuning n_estimators, max_depth, and learning_rate across 18 combinations). The best estimators are automatically saved.

### Q: Can you explain the model's predictions?

**A:** Two complementary approaches: (1) Local — per-prediction word-importance highlighting showing which tokens contributed to each class. (2) Global — SHAP feature importance with per-class drivers showing which words globally push predictions toward positive, neutral, or negative.

### Q: What did you learn from the error analysis?

**A:** The most common confusion is neutral → positive. The model is well-calibrated: average confidence on correct predictions (~0.82) is significantly higher than on errors (~0.45). This means low-confidence predictions can be flagged for human review.

### Q: Why not just use ChatGPT for sentiment analysis?

**A:** Three reasons: (1) Cost — our trained models run locally for free. (2) Latency — baseline models predict in <1ms vs API round-trips. (3) Transparency — we can explain exactly why a prediction was made via SHAP and word importance. (4) Reproducibility — same input always gives the same output.

### Q: What's your CI/CD story?

**A:** Every push triggers GitHub Actions: ruff lint/format checks, mypy static type analysis, and 80 pytest tests. The Docker image has a health check endpoint. I can deploy to Streamlit Cloud, HuggingFace Spaces, or any container platform.

### Q: What would you do with more time?

**A:** (1) DistilBERT/RoBERTa fine-tuning for a transformer comparison study. (2) Production monitoring with prediction drift detection. (3) Active learning pipeline to label the most uncertain predictions. (4) Real-time news ingestion via API for live dashboard updates.

---

## Limitations

This project is transparent about its boundaries:

| Limitation | Detail | Mitigation |
|------------|--------|------------|
| **Dataset size** | 2,264 sentences (AllAgree subset of Financial PhraseBank) | AllAgree was chosen for label quality (100% annotator agreement) over quantity. The full PhraseBank has 4,840 samples at 50% agreement, but noisy labels would undermine evaluation reliability. |
| **Class imbalance** | 61% neutral, 25% positive, 14% negative | Stratified splitting and `class_weight="balanced"` preserve minority-class recall. F1 macro (not accuracy) is the primary metric. |
| **Domain coverage** | Training data is from Finnish company financial reports (formal language) | In-distribution performance is validated with 5-fold CV + a held-out test set. True out-of-domain generalization is **not yet measured** — the bundled headline set is synthetic and used only for pipeline demonstration. Casual language (Reddit, Twitter) is out of scope. |
| **Sentiment–stock correlation** | Pearson correlations are low (0.05 for AAPL) | Expected and scientifically honest — single-factor sentiment is not a reliable price predictor. The value is in demonstrating the end-to-end pipeline and honest reporting. |
| **Explanation generation** | `llm_explain.py` uses template-based generation, not an external LLM API | Deliberate design choice for reproducibility, zero cost, and latency-free operation. The system works fully offline with no API dependency. |
| **Test vs CV accuracy gap** | Gradient Boosting: 94.25% test accuracy vs 88.1% CV accuracy | The test set happens to be slightly easier than the average CV fold. The CV result (88.1% ± 0.5%) is the more reliable performance estimate. |

---

## Demo

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app/app.py

# 3. Open http://localhost:8501
```

### FinBERT Benchmark

```bash
# Run FinBERT on the test set and append metrics to evaluation_results.json
python src/benchmark_finbert.py
```

> **Tip:** Record a short screen capture of the dashboard using [ScreenToGif](https://www.screentogif.com/) or your OS screen recorder, then embed a GIF here for instant visual impact on GitHub.

---

## License

MIT License — feel free to use, modify, and distribute.

---

<div align="center">

**Built with ❤️ for financial NLP research**

</div>

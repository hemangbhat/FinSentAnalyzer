# FinSight: technical interview guide

Everything an interviewer is likely to ask about this project, with answers
grounded in the code: motivation, architecture, data, models, evaluation,
explainability, serving and MLOps, challenges, limitations, trade-offs,
scaling, ethics and future work. There is also a demo script, a bank of
rapid-fire questions, and the numbers to know.

**Rule for using this guide:** only say what you can defend at a whiteboard.
Every answer here points at a file; open it and make sure you understand it
before the interview.

## Contents

0. [Readiness checklist](#0-readiness-checklist)
1. [The pitch: 30 seconds and 2 minutes](#1-the-pitch)
2. [Motivation and problem framing](#2-motivation-and-problem-framing)
3. [Architecture](#3-architecture)
4. [Data](#4-data)
5. [Models](#5-models)
6. [Evaluation methodology](#6-evaluation-methodology)
7. [Explainability](#7-explainability)
8. [Serving, MLOps and deployment](#8-serving-mlops-and-deployment)
9. [Dashboard and UX decisions](#9-dashboard-and-ux-decisions)
10. [Testing and code quality](#10-testing-and-code-quality)
11. [Challenges and how I solved them](#11-challenges-and-how-i-solved-them)
12. [Limitations: raise these yourself](#12-limitations-raise-these-yourself)
13. [Trade-offs and alternatives considered](#13-trade-offs-and-alternatives-considered)
14. [Scaling and system-design follow-ups](#14-scaling-and-system-design-follow-ups)
15. [Security, ethics and responsible use](#15-security-ethics-and-responsible-use)
16. [Future work](#16-future-work)
17. [Live demo script](#17-live-demo-script)
18. [Question bank](#18-question-bank)
19. [Behavioural questions](#19-behavioural-questions)
20. [Numbers cheat sheet](#20-numbers-cheat-sheet)
21. [Code walkthrough map](#21-code-walkthrough-map)
22. [Delivery tips](#22-delivery-tips)
23. [LLM layer: benchmark and RAG](#23-llm-layer-benchmark-and-rag)

---

## 0. Readiness checklist

Done:

- [x] ML pipeline, API, 11-page dashboard, Docker, and CI green on `main`
- [x] 139 automated tests; ruff and mypy in CI
- [x] Every headline number traces to a file in `results/`, with bootstrap
      confidence intervals (`scripts/bootstrap_ci.py`)
- [x] Limitations documented in the README, `docs/MODEL_CARD.md` and `docs/TRACKER.md`

Do before sending the repo to anyone:

- [ ] Deploy (README → Deployment → Option 1) and put the URL on README line 14
- [ ] Record a 15-second GIF of Home → Single Analysis as `docs/demo.gif`
- [ ] Open the live app an hour before the interview; free apps sleep and take
      about a minute to wake
- [ ] Run the LLM benchmark once with your key (section 23) so the
      Generalization page shows real LLM numbers, and commit `results/llm_benchmark.json`
- [ ] Rehearse the 30-second pitch and the demo script out loud
- [ ] Read sections 11 and 12 twice. Challenges and limitations are where
      interviews are won or lost.

---

## 1. The pitch

**The one idea to land:** you didn't stop at a good test score. You tested on
real data, found that the model failed, worked out why, and fixed it. Most
candidates show a 92% accuracy number. You can show 0.90 → 0.46 → 0.84 and
explain every arrow.

### 30 seconds

> "FinSight classifies financial text as positive, neutral or negative. I
> started with classical baselines, TF-IDF plus an SVM, which scored 0.90
> macro-F1 on Financial PhraseBank. Then I tested it on 2,400 real financial
> news headlines it had never seen, and it dropped to 0.46, barely better than
> always predicting 'neutral'. The model had learned annual-report phrasing,
> not sentiment. I fine-tuned FinBERT on in-domain news on a GPU and got back
> to 0.84. It's served through a FastAPI service with validation, auth and
> rate limiting, it has a Streamlit dashboard for explainability and error
> analysis, and it's covered by 119 tests in CI."

### 2 minutes (use the Home page as your slide)

1. **Problem.** Financial sentiment feeds trading signals, risk monitoring
   and news triage. Calling bad news neutral is expensive.
2. **Data.** Financial PhraseBank, the 2,264 sentences where every annotator
   agreed. Stratified 80/10/10 split. About 61% neutral, so always predicting
   neutral already scores about 62% accuracy. **That's why I report
   macro-F1.**
3. **Baselines.** Seven classical models on TF-IDF. 5-fold stratified CV,
   because a 226-row test set is noisy. SVM and Gradient Boosting tie
   (0.838 vs 0.835); I chose SVM for speed and stability.
4. **Stress test.** 2,388 real, human-labelled financial headlines. Macro-F1
   falls from 0.90 to 0.46.
5. **Diagnosis.** SHAP shows stopwords like `is`, `to`, `the` among the
   strongest features. The model learned the style of PhraseBank's sentences.
6. **Fix.** FinBERT, a BERT model pre-trained on financial text, fine-tuned
   on the news training split: 0.87 accuracy, 0.84 macro-F1, with a 95%
   bootstrap interval of 0.82–0.86.
7. **Production.** FastAPI with Pydantic validation, an optional API key,
   Redis-backed rate limits, nginx load balancing across replicas, a model
   registry with SHA-256 hashes, monthly retraining in CI, and Docker.
8. **LLM layer.** I benchmarked an OpenAI model against FinBERT on the same
   headlines with the same bootstrap, including cost and latency, and built
   a RAG page that answers questions from the headlines with validated
   citations (section 23).

---

## 2. Motivation and problem framing

**Why did you build this?** Make this answer yours. A truthful shape:
*"I wanted a project that goes past a notebook: a real model, honest
evaluation, and the engineering around it. Financial sentiment was a good fit
because the domain is tricky (generic sentiment tools misread finance
language) and the data is public."*

**Why is financial sentiment hard?**
- **Domain vocabulary.** "Liability", "tax" and "cost" are negative in general
  English but neutral in finance. That's why the Loughran-McDonald lexicon
  exists, and why FinBERT beats general BERT here.
- **Numbers carry the sentiment.** "Profit rose to EUR 13.1m from EUR 8.7m" is
  positive only because 13.1 > 8.7.
- **Neutral dominates.** Most financial sentences are factual, so naive
  models learn to say "neutral".
- **Domain shift.** Annual-report sentences, tweets and headlines are written
  very differently. That turned out to be the central finding of this project.

**Who would use it?** Analysts triaging news or filings, and quant teams who
want a sentiment feature. Framed honestly, it's a decision-support tool: it
ranks and explains, and a human decides.

**Why three classes, not a score?** The labelled datasets are three-class.
Class probabilities give a continuous signal anyway; the batch page turns the
mix into an aggregate lean.

**What does success look like?** High macro-F1 that survives a domain shift,
explanations a human can check, and a service that's safe to expose.

---

## 3. Architecture

```
                    OFFLINE (training / evaluation)
 Financial PhraseBank ─► preprocess.py ─► train/val/test CSVs
                                    │
             train.py (7 TF-IDF baselines, CV, GridSearch) ─► models/*.joblib
             notebooks/finetune_finbert_gpu.py (T4 GPU)    ─► models/finbert_finetuned/
                                    │
     evaluate.py · integrate_news.py · shap_explain.py · bootstrap_ci.py ─► results/*.json
     registry.py (SHA-256 + metrics) ─► models/registry.json

                    ONLINE (serving)
 Browser ─► Streamlit UI (app/) ──in-process──► predict.SentimentPredictor ─► model
               │
               └─split mode (FINSIGHT_API_URL)─► nginx ─► FastAPI replicas ─► predictor
                                                              │
                                                     Redis (shared rate limits)
```

**Key components**

| Layer | File(s) | Responsibility |
|---|---|---|
| Data | `src/preprocess.py` | Load PhraseBank, clean text, de-duplicate, stratified split |
| Training | `src/train.py`, `src/model.py` | Baselines, CV, GridSearch; transformer fine-tuning |
| Evaluation | `src/evaluate.py`, `src/integrate_news.py`, `scripts/bootstrap_ci.py` | Test metrics, real-news evaluation, intervals |
| Inference | `src/predict.py` | One `SentimentPredictor` interface for every model type |
| Explainability | `src/explain.py`, `src/shap_explain.py`, `src/nlp_advanced.py`, `src/llm_enhanced.py` | Term weights, SHAP, lexicon and entities, rule-based reasoning |
| API | `api/main.py`, `app/api_client.py` | REST service; UI client for split mode |
| Ops | `src/monitoring.py`, `src/registry.py`, `scripts/retrain.py`, `.github/workflows/` | Logging, metrics, registry, retraining, CI |
| UI | `app/app.py` (router), `app/ui.py` (design system), `app/views/*` | 11 pages |
| LLM layer | `src/llm_sentiment.py`, `scripts/llm_benchmark.py`, `src/rag_news.py`, `src/stats.py` | LangChain + OpenAI; optional, needs a key |

**Path of a single prediction.** The UI page calls `active_model()` to get the
cached predictor. `predictor.predict(text)` routes by model type:
- **TF-IDF pipeline:** `predict_proba` (or decision function).
- **Voting ensemble:** stored as `{tfidf, ensemble}`, so it transforms first.
- **Transformer:** tokenize, forward pass, softmax.

The result is `{label, confidence, probabilities}`. In split mode,
`RemotePredictor` sends the same call over HTTP to `/predict`, so the UI code
doesn't change.

**Design decisions to explain**
- **One predictor interface.** The UI, the API and the batch page all call
  `.predict()`, so adding a model means one branch in `predict.py`.
- **Results as JSON artifacts.** The UI reads `results/*.json` and never
  recomputes metrics, so what's displayed is exactly what was evaluated.
- **UI/API split is optional.** In-process is simplest for a demo. The split
  lets the model tier scale independently and keeps the UI stateless.
- **Model caching.** `@st.cache_resource` loads each model once per process.

---

## 4. Data

**Financial PhraseBank** (Malo et al., 2014)
- Sentences from financial news and company reports, labelled by 16
  annotators with finance backgrounds.
- I use the **AllAgree** subset: 2,264 sentences where every annotator agreed.
  Labels are cleaner, but these are also the easier sentences.
- Classes: 1,391 neutral, 570 positive, 303 negative.
- Split: stratified 80/10/10, seed 42 (1,807 / 226 / 226).

**Real-news evaluation set:** `zeroshot/twitter-financial-news-sentiment`
(MIT licence), validation split, 2,388 headlines. Fetched with
`scripts/fetch_real_news_dataset.py`, which also maps its label ids onto mine.
The training split (9,543 rows) was used only to fine-tune FinBERT.

**Preprocessing** (`clean_text`):
- lowercase;
- collapse whitespace;
- keep letters, digits and `. , % $ - +`, because numbers and signs carry
  financial meaning;
- drop duplicates and empty rows.

Stopwords are not removed. That's a decision you'll defend in section 12.

**Synthetic data.** `external-datasets/` holds about 2,700 template-generated
headlines and synthetic prices. They're used only to exercise the stock
pipeline, and the UI labels them as a demo, never as evidence.

**Likely questions**
- *Why AllAgree?* Cleaner labels, so differences between models aren't label
  noise. Trade-off: it's easier than the 50%-agree subsets, which inflates
  in-domain scores. Another reason to test out of domain.
- *Leakage?* Splits are stratified and de-duplicated before splitting. Test
  data is used only for final scoring, and CV runs on train + validation.
- *Imbalance?* Covered in section 5.

---

## 5. Models

### TF-IDF baselines (`src/train.py`)

TF-IDF settings: unigrams + bigrams, `max_features=10000`, `min_df=2`,
`max_df=0.95`.

| Model | Key settings |
|---|---|
| Logistic Regression | `class_weight='balanced'`, `max_iter=1000` |
| Naive Bayes | Multinomial, `alpha=0.1` |
| Linear SVM | `LinearSVC`, `class_weight='balanced'` |
| Random Forest | 200 trees, `max_depth=50`, balanced |
| Gradient Boosting | 100 trees, depth 5, lr 0.1 |
| MLP | (256, 128) hidden layers, early stopping |
| Voting ensemble | **soft** voting over LogReg + **calibrated** SVM (`CalibratedClassifierCV`, cv=3) + RF |

**GridSearchCV**, scored on macro-F1 with 5-fold stratified CV:
- SVM: `C ∈ {0.1, 0.5, 1, 2, 5}`.
- Gradient Boosting: 2 × 3 × 3 = 18 combinations of trees, depth and learning
  rate.

*Why calibrate the SVM in the ensemble?* `LinearSVC` has no
`predict_proba`, and soft voting needs probabilities. Platt-style calibration
supplies them.

### FinBERT

- **Zero-shot:** `ProsusAI/finbert` as published (`src/finbert_pretrained.py`).
- **Fine-tuned** (`src/model.py`, run via `notebooks/finetune_finbert_gpu.py`
  on a Kaggle/Colab T4):

| Setting | Value |
|---|---|
| Base model | ProsusAI/finbert (BERT-base, ~110M parameters) |
| Max length | 128 tokens |
| Optimiser | AdamW, lr 2e-5, linear schedule with 10% warmup |
| Batch / epochs | 32 / 4 |
| Loss | Cross-entropy with inverse-frequency class weights |
| Stability | Gradient clipping at 1.0; fp16 autocast + GradScaler on CUDA |
| Checkpointing | Best epoch by validation macro-F1, restored at the end; early stopping with patience 2 |

- **DistilBERT** (1 epoch, CPU, 6k examples) reached 0.73 macro-F1. It was a
  cheap first check that domain data helps.

### Stock extension (experimental)

yfinance prices → daily mean sentiment → features (daily return, 5-day moving
average, sentiment, headline count) → sequences of 5 days → a 1-layer LSTM
(hidden size 32) predicting next-day up/down. Adam, lr 1e-3. The split is
chronological. Performance is chance-level, and the UI says so.

**Likely questions**
- *Why TF-IDF + linear models first?* They're fast, strong on short texts,
  explainable through coefficients, and they set a bar. A transformer has to
  beat that bar to justify its cost.
- *Why FinBERT over BERT or RoBERTa?* It was pre-trained on financial text, so
  it already knows the domain vocabulary. I also tried DeBERTa (see section
  11).
- *Why AdamW + warmup?* The standard BERT fine-tuning recipe. Warmup avoids
  large early updates that destroy pre-trained weights, and the decoupled
  weight decay regularises properly.
- *Why only 4 epochs?* BERT fine-tuning converges fast and overfits after
  2–4 epochs; best-checkpoint restore guards against that.

---

## 6. Evaluation methodology

**Metric: macro-F1.** It's the unweighted mean of per-class F1, so the 303
negatives count as much as the 1,391 neutrals. Accuracy would reward ignoring
minority classes: always saying "neutral" scores 0.61 accuracy but only
**0.25 macro-F1** on PhraseBank's test split, and 0.66 accuracy but 0.26
macro-F1 on the news set.

**Layers of evidence, weakest to strongest:**
1. A single test split (n=226). Noisy.
2. 5-fold stratified CV on train + validation (n=2,033): mean ± std.
3. **Bootstrap 95% intervals** (`scripts/bootstrap_ci.py`, 2,000 resamples).
   For the real-news set I resample from the confusion matrix, which is
   equivalent to resampling the evaluation rows.
4. **Out-of-domain test** on real news. This is the one that changed the
   project.

**What the intervals showed**
- SVM test macro-F1 0.902 **[0.850, 0.945]** overlaps Gradient Boosting,
  Logistic Regression and the ensemble. They're statistically tied on 226
  sentences, which is why model choice is made on CV.
- Real news: SVM 0.46 [0.44, 0.49] and FinBERT 0.84 [0.82, 0.86]. These don't
  overlap, so the gap is real.

**Error analysis** (Error Analysis page): the misclassification flow
(Sankey), the most common confusion pairs, confidence on correct predictions
versus errors (0.69 vs 0.48 mean for the SVM), and the misclassified sentences
themselves.

**Likely questions**
- *Is 0.84 out-of-distribution?* No, and say so. FinBERT trained on the
  news training split, so the validation split is in-domain held-out. The
  claim is "domain-matched data closes the gap", not "FinBERT generalizes
  zero-shot".
- *Why bootstrap rather than a t-test?* It makes no distributional assumption
  about F1, works for any metric, and is easy to explain.
- *Is the model calibrated?* Not shown. Lower mean confidence on errors is
  consistent with calibration but doesn't prove it. A reliability diagram or
  expected calibration error (ECE) would, and temperature scaling would fix
  miscalibration.

---

## 7. Explainability

| Method | Where | What it tells you |
|---|---|---|
| **Term attribution** | Single Analysis, Explainability | For TF-IDF models: each term's TF-IDF weight × the class coefficient, so you see which words pushed the verdict. Exact for linear models. |
| **SHAP** | Explainability, Generalization | Global feature importance (mean \|SHAP\|) and signed per-class drivers. TreeExplainer for tree models; coefficient-based for linear models. |
| **Word Insights** | Word Insights | The top-weighted vocabulary per class; flags stopwords. |
| **Lexicon + entities** | Deep Analysis | Loughran-McDonald positive, negative, uncertainty and litigious word lists; regex extraction of percentages, amounts and company tickers. |
| **Rule-based reasoning** | Deep Analysis | Six deterministic steps (comprehension → entities → sentiment → context → confidence → synthesis). **Not an LLM**; say so if asked. |
| **Template rationale** | Single Analysis | A plain-language sentence built from the word weights. No external API. |

**Likely questions**
- *What is SHAP?* Shapley values from cooperative game theory: each feature's
  average marginal contribution across feature coalitions. Additive (they sum
  to prediction minus baseline) and consistent.
- *SHAP's limits?* It's expensive for non-tree models, correlated features
  split credit, and it explains the model, not the world.
- *How do you explain FinBERT?* Not built here. Options: attention rollout
  (weak evidence), Integrated Gradients via Captum (better), or SHAP's
  partition explainer on tokens.
- *What did explainability actually find?* Stopwords among the top SVM
  features, which explains the domain-shift failure. That's explainability
  doing real diagnostic work, not decoration.

---

## 8. Serving, MLOps and deployment

### API (`api/main.py`)

| Endpoint | Purpose | Guard |
|---|---|---|
| `GET /health` | Liveness; shows whether auth is on and the rate-limit backend | open |
| `GET /models` | Available model keys | open |
| `POST /predict` | One text → label, confidence, probabilities | API key, 60/min |
| `POST /predict/batch` | Up to 256 texts | API key, 10/min |
| `GET /metrics` | Counters and latency percentiles | API key |

- **Validation:** Pydantic models, with text ≤ 5,000 characters and batches
  of 1–256; malformed input gets a 422.
- **Auth:** `X-API-Key` header, checked only when `FINSIGHT_API_KEY` is set.
  The service warns loudly at startup if it isn't.
- **Rate limiting:** slowapi, with Redis storage so limits hold across
  replicas. In-memory if no Redis (per worker, resets on restart).
  `FINSIGHT_TRUST_PROXY` makes it read the real client IP behind nginx.

### Operations

- **Monitoring** (`src/monitoring.py`): structured logs, an optional JSON-lines
  audit trail (`logs/predictions.jsonl`), counters, and a rolling latency
  window with percentiles.
- **Registry** (`src/registry.py`): every artifact with its SHA-256, size,
  modification time and metrics in `models/registry.json`; optional MLflow
  logging.
- **Retraining** (`scripts/retrain.py`, `.github/workflows/retrain.yml`):
  runs monthly. Steps: train baselines, CV, evaluate, refresh the registry,
  run tests, then upload the artifacts for **human review**. Promotion is
  manual (`docs/RETRAINING.md`).
- **CI** (`.github/workflows/ci.yml`): ruff lint and format, mypy, and
  pytest, using pinned dependencies (`requirements-ci.txt`).

### Deployment

- **Docker:** `Dockerfile` (UI) and `Dockerfile.api`, both Python 3.10-slim
  with CPU-only torch and health checks. `docker-compose.yml` runs redis +
  api (scaled) + nginx + ui.
- **Streamlit Community Cloud** for the public demo, with CPU-only torch from
  `requirements.txt`.

**Likely questions**
- *Why FastAPI?* Async, Pydantic validation and OpenAPI docs come free, and
  it's the standard for Python model serving.
- *Why Redis for rate limits?* With N replicas, in-memory limits allow N times
  the intended rate. A shared store enforces one global limit.
- *How do you roll back a bad model?* Artifacts are versioned in git with
  hashes in the registry. Revert the commit and redeploy;
  `models/registry.json` records the hash of each artifact, so you can confirm
  which model shipped.
- *How do you detect drift?* Watch the predicted-class mix and mean confidence
  over time against a training-time baseline, and sample low-confidence
  predictions for labelling.

---

## 9. Dashboard and UX decisions

- **Router plus persistent sidebar** (`app/app.py`, `st.navigation`): the
  model selector is rendered once, so the choice survives page changes. It
  used to silently reset.
- **Design system in one module** (`app/ui.py`, documented in
  `docs/DESIGN.md`):
  - Styled as a central bank's report on its own models.
  - Numbered exhibits with a source line.
  - **Every score drawn with its uncertainty band.**
  - Sentiment always shown as colour + glyph + word (▲ ▶ ▼), so it's never
    colour alone.
  - Self-hosted Public Sans font.
  - All text colours meet WCAG AA contrast.
- **Honesty in the UI:**
  - The synthetic stock demo is labelled as a demo, with p-values.
  - The FinBERT row says "held-out in-domain, not out-of-domain".
  - Unsupported claims (for example "well-calibrated") were removed.
- **Why Streamlit?** Fastest way to put a Python model behind an interactive
  UI. The trade-off is limited layout control, handled with a theme plus
  injected CSS.

---

## 10. Testing and code quality

139 tests (`tests/`). Number of tests per area:

| Area | Count |
|---|---|
| NLP and entity extraction | 24 |
| Utilities | 19 |
| Rule-based reasoning | 13 |
| API | 10 |
| LLM benchmark (stubbed API) | 10 |
| Preprocessing | 10 |
| RAG news Q&A (fake embeddings, stubbed LLM) | 8 |
| News integration | 7 |
| Prediction | 6 |
| SHAP | 6 |
| Stock extension | 6 |
| API client | 5 |
| Real-news generalization | 1 |
| Model artifact label map | 1 |
| Page import smoke test | 1 test, run once per page file (13) |

**Notable tests**
- **API tests** use FastAPI's `TestClient`: validation errors, auth, batch
  limits.
- **`test_page_imports.py`** checks every symbol each UI page imports
  actually exists. It caught real bugs during the redesign.
- **The entity bug** was fixed test-first: failing tests, then the fix.

Tooling: ruff (lint + format), mypy on core modules, and pinned CI
dependencies.

*How do you test ML code?* Test the deterministic parts thoroughly
(preprocessing, splitting, entity extraction, API contracts). For models, test
shape and range invariants (probabilities sum to 1, confidence in [0, 1]), and
guard quality with evaluation thresholds, not unit tests.

---

## 11. Challenges and how I solved them

Tell these as stories: situation, what you tried, what you found, the result.

1. **The model that looked great but wasn't.** 0.90 macro-F1 in-domain, 0.46
   on real news. I didn't tune harder on PhraseBank. I used SHAP to find out
   *why*: the model relied on stopwords and house style. Then I fixed the data
   problem with domain data and a domain model. Result: 0.84.
2. **DeBERTa crashed under mixed precision.** Its logits came back in fp16
   while the loss expected fp32. I cast logits to float32 before the loss and
   disabled the GradScaler for that model family. Lesson: AMP isn't free for
   every architecture.
3. **Deployed app couldn't load FinBERT.** The host had checked out a ~130-byte
   Git LFS *pointer* instead of the 438 MB weights. I detect pointers by file
   size (`_has_usable_local_weights`) and fall back to the Hugging Face Hub.
   Later I moved the small result files out of LFS entirely.
4. **Retraining failed silently for three months.** The scheduled workflow
   never installed `matplotlib`, which `evaluate.py` imports. Separately, CI
   installed unpinned pandas/scikit-learn, which broke unpickling the models.
   Fix: one pinned `requirements-ci.txt` for both workflows. Lesson:
   scheduled jobs need the same rigour as CI, plus alerting.
5. **Entity extraction chopped words.** Every regex was compiled with
   `IGNORECASE`, so the ticker rule `[A-Z]{2,5}` matched any 2–5 letters
   ("The company" became "The co" / "mpany"). Test-first fix: compile that
   pattern case-sensitively with word boundaries, and filter currency codes
   like EUR. It surfaced only when the UI showed the output.
6. **Wrong label names, caught before publishing.** Preparing the model for
   the Hugging Face Hub, I noticed `config.json` still named the classes in
   ProsusAI's order (0 = positive) while training used mine (0 = negative).
   The app was fine because it decodes with its own map, but any `pipeline()`
   user would have got inverted labels. I fixed the config, added a test, and
   made the upload script refuse a mismatched config.
7. **Small test set, overconfident conclusions.** Bootstrap intervals showed
   the "best" baseline was statistically tied with three others, so I made
   model choice a CV decision and drew intervals everywhere in the UI.

---

## 12. Limitations: raise these yourself

Interviewers trust candidates who volunteer weaknesses. For each one, know the
fix.

| # | Limitation | Fix |
|---|---|---|
| 1 | FinBERT's 0.84 is **in-domain held-out**, not out-of-domain | Evaluate on a third, unseen source (e.g. SEC filings) |
| 2 | **Selection bias:** best epoch chosen on 500 rows of the same validation split that's reported | Carve a separate dev split out of `train` |
| 3 | 226-row test set is small | Already mitigated with CV + bootstrap; the AllAgree-only data also flatters in-domain scores |
| 4 | TF-IDF keeps stopwords; SHAP shows the model leaning on them | Stopword handling plus re-tuning; test the effect out of domain |
| 5 | No calibration analysis | Reliability diagram + ECE; temperature scaling |
| 6 | Zero-shot FinBERT loads revision `main`, so it isn't truly pinned | Pin a commit hash in `MODEL_REVISION` |
| 7 | Stock LSTM is chance-level on synthetic data; its `StandardScaler` is fitted before the train/test split (small leak) | Fit the scaler on the training slice only; use real news; walk-forward validation |
| 8 | Deep Analysis "reasoning" is rule-based templates, not an LLM | Already labelled in the UI |
| 9 | No latency benchmarks | Measure p50/p95 per model with `/metrics`; quantize or ONNX-export FinBERT |
| 10 | API auth is opt-in; rate limits are in-memory without Redis | Set `FINSIGHT_API_KEY` and `FINSIGHT_REDIS_URL` in any public deployment |
| 11 | Sentence-level only: "Apple up, Samsung down" gets one label | Aspect- or entity-level sentiment |
| 12 | English only; data is 2014 Finnish company reports + one tweet dataset | More sources, time-based splits |

A good line: *"The most valuable result in this project is a negative one: the
0.46."*

---

## 13. Trade-offs and alternatives considered

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Baseline features | TF-IDF n-grams | Word2Vec / GloVe averages | TF-IDF is strong on short text and directly explainable |
| Default model | SVM | Gradient Boosting (tied) | Same CV score, much faster, linear = exact attributions |
| Transformer | FinBERT | BERT, RoBERTa, DeBERTa | Domain pre-training; DeBERTa needed fp32 workarounds |
| Fine-tuning hardware | Free T4 GPU | CPU | 4 epochs on 9.5k rows is minutes on GPU, hours on CPU |
| Metric | Macro-F1 | Accuracy, weighted F1 | Imbalance; minority classes matter most |
| Uncertainty | Bootstrap | Analytic CI | No distribution assumptions; works for F1 |
| Serving | FastAPI + optional split | UI-only | Optional split shows production shape without forcing it |
| UI | Streamlit | React + API | Speed of iteration for a Python ML project |
| LLM for explanations | None (templates + rules) | GPT-style API | Deterministic, free, no data leaves the machine; honest about it |

---

## 14. Scaling and system-design follow-ups

*"How would you serve 1,000 requests per second?"*
- **Stateless API replicas** behind a load balancer (already built:
  `docker compose up --scale api=N`), with shared Redis limits.
- **Pick the model per budget.** The SVM costs microseconds per text. FinBERT
  needs batching, ONNX Runtime or int8 quantization, or a GPU node.
- **Dynamic batching** for the transformer: queue requests briefly, run one
  forward pass.
- **Cache repeat texts.** Headlines repeat across feeds; key on a hash.
- **Async batch jobs** through a queue (Celery/RQ/SQS) for large files, with
  results stored and polled.
- **Autoscale on queue depth or p95 latency**, not CPU alone.

*"Real-time news feed?"* Stream ingestion (Kafka), de-duplication, a scoring
service, results to a time-series store, and alerts on sentiment shifts per
ticker.

*"Multi-tenant?"* Per-key rate limits and quotas, audit logs (already JSONL),
and per-tenant model versions via the registry.

---

## 15. Security, ethics and responsible use

- **Input safety:**
  - Length and batch caps against abuse.
  - User text is HTML-escaped in the dashboard's custom HTML views.
  - No secrets in the code; configuration comes from environment variables.
- **Supply chain:**
  - Pinned CI dependencies.
  - Model hashes in the registry.
  - The FinBERT revision *should* be pinned to a commit (limitation 6).
- **Not financial advice:** the stock demo is labelled chance-level. Sentiment
  is one weak signal, never a trading system on its own.
- **Bias and representativeness:** PhraseBank is Finnish companies and the
  news set is mostly US tickers, so performance on other markets, languages
  or eras is unknown. Small caps and non-English sources are likely worse.
- **Human in the loop:** low-confidence predictions should route to a person.
  The Error Analysis page shows why.
- **Licensing:** PhraseBank is licensed CC BY-NC-SA 3.0 (non-commercial); the
  news dataset is MIT; the Public Sans font is OFL. Check every model and
  dataset licence again before any commercial use.

---

## 16. Future work

In priority order, with the reason for each:

1. **Clean FinBERT re-run** with a separate dev split. Removes the selection
   bias (limitation 2).
2. **Third-source evaluation** (e.g. 10-K or earnings-call sentences). Makes a
   real out-of-domain claim for FinBERT.
3. **Calibration:** reliability diagram, ECE, temperature scaling. Makes
   confidence trustworthy for routing.
4. **Entity/aspect-level sentiment.** One headline can be good for one company
   and bad for another.
5. **Distil or quantize FinBERT** for CPU serving, with measured latency.
6. **Drift monitoring dashboard** from the prediction logs.
7. **Active learning:** label the low-confidence predictions and feed the
   retraining job.

---

## 17. Live demo script

About 3–4 minutes; rehearse it.

| Step | Where | What to say |
|---|---|---|
| 1 | **Home** | "Every number here is read from the results files, so the UI can't drift from the evaluation." Point at Exhibit 1: each score is a dot with its 95% interval, and the 0.90 → 0.46 → 0.84 story is visible at a glance. |
| 2 | **Single Analysis** → *Plant closure, 300 jobs cut* | SVM says **neutral (81%)**. "That's wrong, and it shows the TF-IDF model's limit." Show the probability band and key drivers. |
| 3 | Sidebar → **FinBERT (fine-tuned)**, same example | Compare the verdicts. If fine-tuned FinBERT isn't listed on the deployed app, its weights aren't on the host (README → Deployment); use **FinBERT (zero-shot)** instead. |
| 4 | **Explainability**, then **Word Insights** | The tinted sentence, then SHAP: `is`, `to`, `the` among the strongest features. "This is how I diagnosed the generalization failure." |
| 5 | **Model Registry** | Exhibit 1: "The top baselines' intervals overlap: statistically tied. That's why I pick models on cross-validation." |
| 6 | **Error Analysis** | The error flow, then confidence. "Mean confidence is lower on errors, so a threshold could route uncertain sentences to a human. That alone doesn't prove calibration." |
| 7 | **Generalization** | The real-news exhibit and FinBERT's confusion matrix (most remaining errors: 92 neutral headlines read as negative). Then say plainly that the synthetic section is a pipeline demo, not market evidence. |
| 8 | Terminal: `curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text":"Profit fell 20%"}'` | "Same models behind an API, with validation, auth and rate limits." |

---

## 18. Question bank

Short answers; expand from the sections above.

### ML and NLP fundamentals

- **TF-IDF?** Term frequency × log inverse document frequency. It up-weights
  words that are frequent in a document but rare across the corpus.
- **Why bigrams?** They capture "not good" and "fell sharply", which
  unigrams miss.
- **Linear SVM?** Finds the maximum-margin hyperplane. Works well in sparse,
  high-dimensional spaces like TF-IDF.
- **Naive Bayes assumption?** Features are conditionally independent given
  the class. Wrong for text, yet it works as a fast baseline.
- **Bagging vs boosting?** Random Forest averages decorrelated deep trees
  (reduces variance). Gradient Boosting adds shallow trees that fit the
  residuals (reduces bias).
- **Soft vs hard voting?** Soft averages probabilities and keeps confidence
  information; hard counts votes.
- **Precision, recall, F1?** Of the predicted positives, how many are right;
  of the actual positives, how many were found; F1 is their harmonic mean.
- **Macro vs weighted F1?** Macro averages classes equally; weighted averages
  by support.
- **Stratified k-fold?** Each fold keeps the class proportions. Essential with
  imbalance.
- **Overfitting signs?** Train ≫ validation score, high variance across
  folds, a big out-of-domain drop.
- **Class imbalance options?** Class weights (used), resampling, threshold
  tuning, focal loss.

### Transformers

- **Self-attention?** Each token builds its representation as a weighted sum
  of all tokens, with weights from query·key similarity. That gives context:
  "positive outcome" vs "tested positive".
- **What is BERT?** An encoder-only transformer pre-trained with masked
  language modelling. Fine-tuning adds a classification head on the `[CLS]`
  token.
- **Fine-tuning vs feature extraction?** Fine-tuning updates all weights.
  It's better with enough data, but needs a small learning rate and warmup.
- **Why max length 128?** Headlines and sentences are short; 128 tokens covers
  them and saves memory and time.
- **Mixed precision?** fp16 computation with an fp32 master copy and loss
  scaling. About 2× faster on a T4; some models need fp32 for stability.
- **Catastrophic forgetting?** Large learning rates or too many epochs erase
  pre-trained knowledge. Low lr, warmup, few epochs and best-checkpoint
  restore prevent it.

### Evaluation and statistics

- **Bootstrap?** Resample the evaluation set with replacement, recompute the
  metric each time, and take the 2.5th and 97.5th percentiles.
- **Why do overlapping intervals mean "tied"?** The data can't tell the
  models apart at this sample size. A paired bootstrap of the difference is
  the stricter test.
- **What's domain shift?** Train and test distributions differ (covariate or
  label shift). Detect it by evaluating on target-domain data, as I did.
- **Calibration?** Predicted probabilities match observed frequencies.
  Measured by reliability diagrams and ECE; fixed with temperature scaling.

### MLOps and engineering

- **Model versioning?** Git + SHA-256 registry + metrics per artifact;
  optional MLflow.
- **Reproducibility?** Fixed seeds, pinned dependencies, a data manifest
  (`data_manifest.toml`), and scripts for every number.
- **CI for ML?** Lint, types, tests, and evaluation gates. Retraining
  artifacts get human review before promotion.
- **Why Docker?** Same environment everywhere; health checks; easy horizontal
  scaling.

---

## 19. Behavioural questions

Use the STAR format: situation, task, action, result.

- **"Tell me about a failure."** The 0.46. Situation: a model that looked
  done. Action: tested out of domain, used SHAP to diagnose, fine-tuned with
  domain data. Result: 0.84, and a habit of always testing on real target
  data.
- **"A time you found your own bug."** The entity-extraction regex, or the
  silently failing retraining job. Stress how you found it and the test you
  added so it can't come back.
- **"How do you handle ambiguity?"** Choosing metrics and evaluation data was
  ambiguous. You reasoned from the use case (minority classes matter, real
  news is the target) and made the reasoning visible in the README.
- **"What would you do differently?"** Set up the out-of-domain test and a
  separate dev split on day one, not after the results looked good.
- **"What did you learn?"** A high score on one dataset says little about
  real-world performance. Uncertainty matters as much as the point estimate.

---

## 20. Numbers cheat sheet

| Item | Value |
|---|---|
| PhraseBank (AllAgree) | 2,264 sentences · 80/10/10 · ~61% neutral |
| SVM, test (n=226) | 92.5% acc · **0.90** macro-F1 · 95% CI 0.850–0.945 |
| SVM, 5-fold CV | **0.838 ± 0.016** macro-F1 |
| Gradient Boosting, CV | 0.835 ± 0.008 (tied with SVM) |
| Voting ensemble, test | 92.0% acc · 0.891 macro-F1 |
| SVM on real news (n=2,388) | 0.67 acc (always-neutral 0.66) · **0.46** macro-F1 · CI 0.436–0.485 |
| DistilBERT, 1 epoch, CPU | 0.81 acc · 0.73 macro-F1 |
| FinBERT fine-tuned, 4 epochs, T4 | **0.87** acc · **0.84** macro-F1 · CI 0.823–0.858 (9,543 train / 2,388 val) |
| FinBERT's main error | 92 neutral headlines read as negative |
| Always-"neutral" macro-F1 | 0.25 (PhraseBank test) · 0.26 (news) |
| API limits | 60/min single · 10/min batch · 5,000 chars · 256 texts |
| Tests | 139 · ruff + mypy in CI |
| LLM benchmark | your numbers from `results/llm_benchmark.json` (fill in after running) |

---

## 21. Code walkthrough map

If they say "show me the code", open these in order:

1. `src/predict.py`: `SentimentPredictor`, the one interface for every model,
   plus the LFS-pointer guard.
2. `src/train.py`: the baselines, `cross_validate_baselines`, `tune_hyperparameters`.
3. `src/model.py`: the fine-tuning loop (AdamW, warmup, class weights, AMP,
   best checkpoint).
4. `scripts/bootstrap_ci.py`: short and easy to explain end to end.
5. `api/main.py`: validation, auth dependency, rate limits, endpoints.
6. `app/app.py` and `app/ui.py`: the router and the design system.
7. `tests/test_api.py` and `tests/test_nlp_advanced.py`: how it's tested.

---

## 22. Delivery tips

- **Lead with the story, not the stack.** Name tools only when they explain a
  decision.
- **Use "because".** "I chose SVM *because* CV showed it tied with Gradient
  Boosting at a fraction of the cost."
- **Quantify everything** with the numbers in section 20.
- **Say "I don't know, but here's how I'd find out"** rather than guessing.
- **Volunteer limitations early** (section 12). It signals maturity.
- **Close with what you learned:** "A high test score on one dataset told me
  almost nothing about real-world performance. Out-of-domain evaluation is now
  the first thing I do."

---

## 23. LLM layer: benchmark and RAG

This is where the AI-engineering course work shows: prompting, structured
output, LangChain, embeddings and retrieval, applied to a question the
project already asks. Both features need `OPENAI_API_KEY` and
`FINSIGHT_LLM_MODEL`; nothing about them is hardcoded or committed.

### 23.1 LLM vs FinBERT (`scripts/llm_benchmark.py`, `src/llm_sentiment.py`)

**The question.** Does a general-purpose LLM, prompted zero-shot or
few-shot, beat a 110M-parameter model fine-tuned on in-domain news? And
what does each point of macro-F1 cost?

**How it's fair.**
- Same 2,388 headlines FinBERT was evaluated on.
- Same metric and the same bootstrap code (`src/stats.py`, shared by every
  interval in the project), so the intervals are directly comparable.
- Few-shot examples come from the PhraseBank *training* split, 2 per class,
  interleaved so their order doesn't hint at the answer. A test asserts none
  of them appears in the evaluation set.
- Temperature 0. The prompt has a version (`PROMPT_VERSION`) that is part of
  the cache key, so changing the prompt never reuses stale answers.

**How it's robust.**
- **Structured output** (`with_structured_output` on a Pydantic model whose
  field is `Literal["negative", "neutral", "positive"]`). The model can't
  answer "Bullish!". If parsing fails the answer is *invalid* and counted as
  wrong, not silently dropped, which would flatter the score.
- **Retries and timeouts** via the client; a failed call is recorded with
  its error and the run continues.
- **Disk cache** of answers, keyed by model, prompt version, mode and text.
  Re-runs are free; invalid answers are not cached so they get retried.
- **Concurrency** with a thread pool (I/O bound), order preserved.

**Cost and latency.** Token counts come from the API's usage metadata.
Dollars are computed from prices passed on the command line, because prices
change and a hardcoded table goes stale. Latency is p50/p95 per call over
fresh (uncached) calls only.

**How to run it.**

```bash
python scripts/llm_benchmark.py --dry-run          # prompts + token estimate, free
python scripts/llm_benchmark.py --limit 200        # stratified trial, cents
python scripts/llm_benchmark.py --price-in X --price-out Y   # full set
```

The dry run estimates roughly 340k input tokens for a zero-shot pass over
all 2,388 headlines (few-shot is about 2.5x that, because the examples ride
along with every call).

**What to say about the result.** Whichever way it comes out, the story
works:
- *If FinBERT wins:* "A small model fine-tuned on 9.5k in-domain labels
  beats a general LLM, runs locally in milliseconds, and costs nothing per
  call. Fine-tuning still pays when you have labels."
- *If the LLM wins or ties:* "Without any labels it matches a fine-tuned
  model, so for a cold start I'd use the LLM to label data, then distil into
  a small model for cost and latency."
- Either way, point at the interval overlap before claiming a winner.

**Likely questions.**
- *Why not just use the LLM everywhere?* Cost per call, latency, rate
  limits, a network dependency, data leaving your boundary, and the model
  can change under you. FinBERT is pinned by a hash.
- *How do you know the LLM hasn't seen these headlines?* I don't. The
  dataset is public and older than the model's training cutoff, so treat
  its score as an upper bound. That is a known limitation of benchmarking
  closed models.
- *Why few-shot from PhraseBank, not the news train split?* To keep the
  setting comparable to "no in-domain labels". Using news examples is an
  easy next experiment.
- *Why not chain-of-thought?* More output tokens (cost and latency) for a
  three-way label. Worth trying as another mode; the prompt version makes it
  a clean comparison.

### 23.2 Ask the News (`src/rag_news.py`, `app/views/10_Ask_the_News.py`)

**Pipeline.**
1. Embed all 2,388 headlines once (`text-embedding-3-small` by default)
   into LangChain's `InMemoryVectorStore`; save it under `data/cache`, keyed
   by embedding model and a hash of the corpus, so it's rebuilt only when
   either changes.
2. Embed the question, retrieve the top `k` by cosine similarity.
3. Score each retrieved headline with the active FinSight model and show
   the sentiment mix next to the answer.
4. Ask the chat model to answer only from the numbered headlines, with
   structured output `{answer, citations}`.
5. **Validate citations**: any number that isn't one of the retrieved
   headlines is removed and flagged in the UI. An answer with no citations
   is labelled unsupported.

**Safety.** Headlines are untrusted input. The system prompt says they are
data and to ignore instructions inside them; the test corpus includes an
"Ignore previous instructions" headline. The API key is read from the
environment or Streamlit secrets and never logged or committed. Errors from
the API are shown as a banner, not a stack trace.

**Why these choices.**
- *In-memory store, not a vector database:* 2,388 short texts is about
  15 MB of vectors; brute-force cosine is instant. At 1M+ documents I'd move
  to pgvector or a managed store with an ANN index (HNSW).
- *No chunking:* every document is a single headline.
- *`langchain-core` + `langchain-openai` only:* small dependency surface;
  `langchain-community` is being wound down.

**Limitations to raise yourself.**
- The headlines have no timestamps, so it can't answer "what happened this
  week"; it describes the dataset.
- Retrieval is pure semantic similarity. Ticker queries like `$TSLA` would
  benefit from hybrid search (BM25 + embeddings) and metadata filters.
- No retrieval evaluation yet. Next step: a small labelled question set and
  recall@k, plus answer faithfulness scored by an LLM judge.

**Likely questions.**
- *How do you stop hallucination?* Grounding instructions, structured
  citations, citation validation against what was actually retrieved, and an
  explicit "not in sources" answer. It reduces, not eliminates, the risk; the
  sources table lets a user check every claim.
- *How would you make it an agent?* Give the model tools (search headlines,
  score sentiment, fetch prices with yfinance) and let it decide the order.
  For this question set a fixed pipeline is cheaper and more predictable,
  so I kept it a chain.
- *How is it tested without an API key?* `DeterministicFakeEmbedding` and a
  stub chat model: exact-match retrieval, index save/reload, the prompt's
  injection guard, invented citations being dropped, and no LLM call when
  nothing is retrieved.


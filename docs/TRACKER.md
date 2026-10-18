# Tracker

## Epic 1 — Core sentiment system
- [x] Build baseline training pipeline
- [x] Save trained baseline artifacts
- [x] Add single-text prediction
- [x] Add batch prediction

## Epic 2 — Explainability
- [x] Add TF-IDF / coefficient-based word importance
- [x] Add SHAP summaries for baseline models
- [x] Add readable text highlighting

## Epic 3 — Dashboard
- [x] Home page
- [x] Single analysis page
- [x] Batch processing page
- [x] Explainability page
- [x] Word insights page
- [x] Deep analysis page
- [x] Model info page
- [x] Sentiment trends page
- [x] Error analysis page
- [x] Stock prediction extension page

## Epic 4 — Quality and reproducibility
- [x] Add tests
- [x] Add CI workflow
- [x] Add Dockerfile
- [x] Pin dependencies
- [x] Add README and supporting docs

## Epic 5 — Deployment hardening
- [ ] Deploy Streamlit app publicly
- [ ] Add live URL to README
- [ ] Add a short demo GIF
- [x] Add monitoring/logging if hosted publicly
- [x] Add rate limiting if exposed beyond a private demo

## Backlog
- [x] Fine-tune FinBERT instead of only using the pre-trained checkpoint (0.87 acc / 0.84 macro-F1 on held-out news)
- [x] Add a small API layer for programmatic inference
- [x] Add model registry / versioning
- [ ] Add persistent storage for user sessions and audit logs
- [x] Add scheduled retraining workflow

## Epic 6 — Service & MLOps
- [x] FastAPI inference service (`api/main.py`) with `/predict`, `/predict/batch`, `/metrics`
- [x] Input validation + IP rate limiting for public exposure
- [x] Structured prediction/error/latency logging (`src/monitoring.py`)
- [x] Model registry with hashes + metrics (`src/registry.py`), optional MLflow
- [x] Retraining orchestrator (`scripts/retrain.py`) + checklist (`docs/RETRAINING.md`)
- [x] Scheduled retraining CI (`.github/workflows/retrain.yml`)
- [x] Repo cleanup: removed Code Runner scratch artifact, hardened `.gitignore`

## Epic 7 — UX and reliability
- [x] `st.navigation` router: grouped sidebar, model selection persists across pages
- [x] Home page built around the evaluation story, numbers read from `results/`
- [x] Pinned CI dependencies; fixed scheduled retraining (missing matplotlib)
- [ ] Re-run FinBERT fine-tuning with a separate dev split for early stopping
- [ ] Add stopword handling / re-tune TF-IDF (SHAP shows stopwords among top SVM features)

> Note: public deployment URL and demo GIF are owner-action items — the app is
> deploy-ready (Dockerfile + `requirements-deploy.txt`), but the live URL and
> recorded GIF must be added by the repository owner.

## Epic 8 — LLM layer (LangChain + Claude)
- [x] LLM vs FinBERT benchmark on the 2,388 real headlines: zero-/few-shot, structured output, same bootstrap, cost and latency (`scripts/llm_benchmark.py`)
- [x] Shared bootstrap in `src/stats.py` (baselines, FinBERT and LLM intervals computed identically)
- [x] Generalization page shows the LLM comparison, or how to produce it
- [x] Ask the News: RAG over the real headlines with validated citations and per-source sentiment (`src/rag_news.py`)
- [x] Hugging Face Hub model card + label-map guard for the fine-tuned FinBERT
- [x] Owner: ran the benchmark (Claude Haiku zero-shot: macro-F1 0.792 vs FinBERT 0.841)
- [ ] Optional: top up credit and finish the few-shot run
- [ ] Owner: upload FinBERT to the Hub and set `FINSIGHT_FINBERT_REPO`
- [x] Claude as the default provider (OpenAI optional); local TF-IDF/LSA embeddings, so one key is enough
- [x] OpenRouter as a third provider (OpenAI-compatible client, tool-calling structured output)
- [ ] Retrieval evaluation (labelled questions, recall@k) and hybrid BM25 + embedding search

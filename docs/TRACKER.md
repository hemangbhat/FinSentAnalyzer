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
- [ ] Fine-tune FinBERT instead of only using the pre-trained checkpoint
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

> Note: public deployment URL and demo GIF are owner-action items — the app is
> deploy-ready (Dockerfile + `requirements-deploy.txt`), but the live URL and
> recorded GIF must be added by the repository owner.

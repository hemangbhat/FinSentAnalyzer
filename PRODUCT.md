# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Two audiences, weighted equally (confirmed):
- **Finance analysts** who score financial text (news headlines, filings, earnings-release sentences) and need a verdict they can trust and audit.
- **Technical evaluators** (interviewers, recruiters, reviewers) assessing the author's ML engineering skill, typically in a 5–10 minute screen-share or by opening the deployed link from a résumé.

## Product Purpose
FinSight classifies financial text as positive, neutral or negative, explains each decision at token level, and shows how its models perform, including where they fail. Success: an analyst can score and inspect text without friction, and an evaluator leaves convinced the system is engineered like production software.

## Positioning
FinSight evaluates itself honestly. It ships the failure next to the fix: a TF-IDF + SVM baseline scoring 0.90 macro-F1 in-domain collapses to 0.46 on real news, and FinBERT fine-tuned on in-domain news recovers to 0.84. The interface exposes the engineering underneath: model registry with hashes, cross-validation, error analysis, an inference API, and CI.

## Operating Context
- Streamlit multipage app (Python), `st.navigation` router in `app/app.py`, pages in `app/views/`, design system in `app/ui.py`.
- The same models are served by a FastAPI service (`api/main.py`) behind nginx, with Redis rate limiting; the UI can run in split mode against it.
- Typical evaluator session: open Home, run one prediction, switch models, open Explainability, Error Analysis and Generalization.

## Capabilities and Constraints
- 9 selectable models: 7 TF-IDF baselines (Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, MLP, Voting Ensemble), zero-shot FinBERT, fine-tuned FinBERT.
- Pages: Home, Single Analysis, Batch Processing, Explainability, Word Insights, Deep Analysis, Model Registry, Error Analysis, Generalization, Stock Prediction (experimental).
- Constraint: Streamlit renders its own widgets; styling is via theme config plus injected CSS. No custom frontend build.
- The Stock Prediction LSTM is chance-level on synthetic data and must stay labelled experimental. The Deep Analysis reasoning is rule-based, not an LLM.

## Brand Commitments
- Product name **FinSight** is fixed (confirmed). Everything else about the visual identity may be replaced.

## Evidence on Hand
- Real metrics in `results/*.json` (evaluation, 5-fold CV, OOD comparison, FinBERT confusion matrix, SHAP).
- Datasets: Financial PhraseBank (2,264 sentences), `zeroshot/twitter-financial-news-sentiment` (MIT, n=2,388 validation).
- No users, testimonials, customers, deployment URL or usage statistics exist. Do not fabricate any.

## Product Principles
1. Show the evidence, including the failures.
2. Every number on screen traces to a results artifact.
3. Depth on demand: the verdict first, the machinery one step away.
4. Credible over impressive: nothing decorative should compete with the data.

## Accessibility & Inclusion
Body text and data must stay readable (WCAG AA contrast); never encode sentiment by colour alone; usable at phone width.

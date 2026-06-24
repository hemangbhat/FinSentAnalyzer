# Product Requirements Document (PRD)

## 1. Product summary
Financial Sentiment Analyzer is a Streamlit-based analytics app that helps users classify financial text and inspect market tone from news or earnings-style text.

## 2. Problem statement
Financial text is dense, time-sensitive, and difficult to analyze manually at scale. The product converts unstructured text into a sentiment signal and a readable explanation.

## 3. Target users
- ML / data science interviewers
- students building portfolio projects
- analysts who want a quick sentiment read
- hackathon judges evaluating end-to-end product thinking

## 4. Goals
- classify sentiment as positive / neutral / negative
- support single-text and batch analysis
- compare multiple models
- surface explainability and error analysis
- provide a clean dashboard for demos and interviews

## 5. Non-goals
- Do not claim direct stock-price prediction from sentiment alone
- Do not claim LLM-powered reasoning unless an external LLM API is actually wired in
- Do not position the project as a trading system or financial advisor

## 6. Functional requirements
- Upload or paste financial text
- Choose a model
- View sentiment label, confidence, and class probabilities
- Analyze a CSV/TXT batch
- View charts and summary metrics
- Inspect model information and error analysis
- Run the stock extension as a separate workflow

## 7. Success metrics
- high F1 macro on the primary dataset
- stable cross-validation scores
- consistent predictions on batch input
- clean demo on local and deployed environments

## 8. Acceptance criteria
- App runs from a clean clone
- Required models load successfully
- Pages render without broken imports
- Batch processing and explainability work end-to-end
- README and docs match the codebase

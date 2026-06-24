# App Flow

## 1. Entry point
User opens `app/app.py`, which initializes the theme, sidebar, and model selector.

## 2. Sidebar selection
The user chooses a model from the sidebar. The selected model is loaded once and reused across pages.

## 3. Primary journeys
### A. Single analysis
1. Paste financial text
2. Select model
3. Click analyze
4. View prediction, confidence, probabilities, and explanation

### B. Batch processing
1. Upload CSV or TXT
2. Parse and validate the input
3. Score each row
4. Show aggregate sentiment distribution and downloadable output

### C. Explainability
1. Enter text or select a scored example
2. Generate token-level / word-level explanation
3. Display local and global feature importance

### D. Trends and error analysis
1. Inspect model-level evaluation outputs
2. View confusion patterns and misclassifications
3. Review trend summaries from batch data

### E. Stock extension
1. Load external news and stock data
2. Aggregate sentiment by day
3. Train or load the LSTM workflow
4. Review next-day direction output

## 4. Page order in the app
- Home
- Single Analysis
- Batch Processing
- Explainability
- Word Insights
- Deep Analysis
- Model Info
- Sentiment Trends
- Error Analysis
- Stock Prediction Extension

## 5. Shared runtime behavior
- shared CSS is injected once
- model loading is cached
- file paths resolve from the project root
- optional modules should fail gracefully if unavailable

## 6. Success path
The app is successful when a user can:
- pick a model
- analyze text
- understand the explanation
- inspect metrics
- export results
- deploy and demo the app without broken imports

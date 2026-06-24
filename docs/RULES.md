# Rules

## Product rules
1. Do not overclaim the model as a trading oracle.
2. Do not call rule-based explanations “LLM-powered” unless an actual LLM API is integrated.
3. Clearly separate sentiment classification from stock prediction.

## Data rules
1. Preserve the `sentence` and `label` schema consistently.
2. Do not leak test data into training or tuning.
3. Remove duplicates before training when required.
4. Keep split generation deterministic.

## Model rules
1. Use cross-validation for trustworthy comparison.
2. Report F1 macro, not accuracy alone.
3. Treat FinBERT as a separate model path.
4. Cache expensive loads in Streamlit.

## UI rules
1. One page, one task.
2. Show confidence and probabilities for every prediction.
3. Prefer compact charts over dense tables.
4. Make empty states helpful.

## Engineering rules
1. Run formatting and linting before commits.
2. Keep imports stable and minimal.
3. Use project-root-aware paths.
4. Update docs whenever code behavior changes.

## Deployment rules
1. The app must work from a clean clone.
2. Every runtime dependency must be declared.
3. Health checks should be valid inside the chosen container image.
4. Do not expose private keys in repo or docs.

## Resume rules
1. Only include features that are actually implemented.
2. Mention metrics only if they are reproducible.
3. Keep the project description truthful and specific.
4. Prefer “pre-trained FinBERT” unless fine-tuning was truly done.

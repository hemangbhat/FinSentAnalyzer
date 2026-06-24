# Design Document

## Design objective
Create a premium-looking Streamlit experience that feels like a fintech analytics product rather than a classroom demo.

## Visual direction
- Dark fintech theme
- High contrast text
- Soft gradient backgrounds
- Rounded metric cards
- Clear spacing and hierarchy
- Consistent accent color for positive / negative / neutral signals

## Layout principles
1. One job per page
2. Keep the sidebar for model selection and navigation
3. Use KPI strips at the top of each page
4. Prefer charts with limited, purposeful colors
5. Avoid overcrowding with too many controls

## Recommended page structure
### Home
- Hero section
- 3–6 feature cards
- quick stats
- call-to-action buttons for key pages

### Single Analysis
- text area
- model selector
- analyze button
- label + confidence + probability chart
- explanation card

### Batch Processing
- file uploader
- preview table
- summary metrics
- distribution chart
- downloadable results

### Explainability
- local word-attribution view
- SHAP summary
- highlighted input text

### Error Analysis
- confusion matrix
- misclassified examples
- common failure modes

### Sentiment Trends
- sentiment over time
- cross-dataset summary
- correlation chart

## UI components to standardize
- metric cards
- section headers
- status pills
- info banners
- empty states
- loading placeholders

## Streamlit-specific recommendations
- use `st.cache_resource` for model loading
- use `st.session_state` for selected model and last prediction
- centralize CSS in one shared module
- keep reusable cards in a component helper file
- prefer Plotly for interactive charts

## Accessibility
- ensure readable contrast
- avoid using only color to encode meaning
- keep mobile width in mind
- maintain keyboard-friendly flows

## Design anti-patterns to avoid
- large paragraphs on the home page
- too many animations
- inconsistent card styles
- multiple font families
- overly bright accent colors on a dark theme

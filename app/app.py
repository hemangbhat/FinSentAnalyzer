"""
Financial Sentiment Analyzer - Streamlit Dashboard
Interactive UI for analyzing financial text sentiment.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

from predict import SentimentPredictor, get_available_models
from preprocess import LABEL_MAP_INV
from explain import explain_prediction_baseline, highlight_text, get_feature_importance_summary
from llm_explain import get_llm_explanation, generate_market_outlook

# Page config
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .sentiment-neutral { color: #007bff; font-weight: bold; }
    .big-font { font-size: 24px !important; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(model_type: str) -> SentimentPredictor:
    """Load and cache the predictor."""
    return SentimentPredictor(model_type)


def get_sentiment_color(label: str) -> str:
    """Get color for sentiment label."""
    colors = {
        "positive": "#28a745",
        "negative": "#dc3545",
        "neutral": "#007bff",
    }
    return colors.get(label, "#6c757d")


def create_gauge_chart(probabilities: dict, prediction: str) -> go.Figure:
    """Create a gauge chart for confidence."""
    confidence = probabilities[prediction]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Confidence: {prediction.upper()}"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": get_sentiment_color(prediction)},
            "steps": [
                {"range": [0, 33], "color": "#ffebee"},
                {"range": [33, 66], "color": "#fff3e0"},
                {"range": [66, 100], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_probability_chart(probabilities: dict) -> go.Figure:
    """Create a bar chart for probabilities."""
    labels = list(probabilities.keys())
    values = [probabilities[l] * 100 for l in labels]
    colors = [get_sentiment_color(l) for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
    ))

    fig.update_layout(
        title="Sentiment Probabilities",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def main():
    # Sidebar
    st.sidebar.title("📈 Financial Sentiment Analyzer")
    st.sidebar.markdown("---")

    # Model selection
    available_models = get_available_models()
    if not available_models:
        st.error("No trained models found. Please train a model first.")
        st.code("python src/train.py --model baselines")
        return

    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=0,
        help="Choose the model for sentiment analysis",
    )

    # Load predictor
    try:
        predictor = load_predictor(selected_model)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app analyzes financial text sentiment using ML models trained on "
        "the Financial PhraseBank dataset. It classifies text as **Positive**, "
        "**Negative**, or **Neutral**."
    )

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Single Analysis",
        "📁 Batch Processing",
        "🔍 Explainability",
        "💡 Word Insights",
        "📊 Model Info"
    ])

    # Tab 1: Single Text Analysis
    with tab1:
        st.header("Single Text Analysis")
        st.markdown("Enter financial text to analyze its sentiment.")

        # Initialize session state for example text
        if "example_text" not in st.session_state:
            st.session_state.example_text = ""

        # Example texts
        with st.expander("📌 Example Texts"):
            examples = [
                "The company reported a 25% increase in quarterly revenue.",
                "Stock prices fell sharply after disappointing earnings report.",
                "The firm announced it will maintain its current dividend policy.",
                "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn.",
                "The company's market share remained unchanged at 15%.",
            ]
            for i, example in enumerate(examples):
                if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                    st.session_state.example_text = example

        # Text input
        text_input = st.text_area(
            "Enter financial text:",
            value=st.session_state.example_text,
            height=150,
            placeholder="e.g., The company reported strong Q3 earnings, beating analyst expectations...",
        )

        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    result = predictor.predict(text_input)

                # Results
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("### Result")
                    sentiment = result["label"]
                    color = get_sentiment_color(sentiment)

                    st.markdown(
                        f"<h2 style='color: {color};'>{sentiment.upper()}</h2>",
                        unsafe_allow_html=True,
                    )

                    if "confidence" in result:
                        st.metric("Confidence", f"{result['confidence']:.1%}")

                with col2:
                    if "probabilities" in result:
                        fig = create_probability_chart(result["probabilities"])
                        st.plotly_chart(fig, use_container_width=True)

                # LLM-powered Natural Language Explanation
                if selected_model.startswith("baseline_"):
                    st.markdown("### 💬 AI Explanation")
                    try:
                        explanation_data = explain_prediction_baseline(text_input, selected_model)
                        llm_explanation = get_llm_explanation(
                            text=text_input,
                            prediction=result["label"],
                            probabilities=result.get("probabilities", {}),
                            word_importance=explanation_data.get("word_importance", []),
                        )
                        st.markdown(
                            f"<div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; "
                            f"border-left: 4px solid {color};'>{llm_explanation}</div>",
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.info("Enable explainability tab for detailed AI-powered explanations.")
            else:
                st.warning("Please enter some text to analyze.")

    # Tab 2: Batch Processing
    with tab2:
        st.header("Batch Processing")
        st.markdown("Upload a CSV or TXT file to analyze multiple texts.")

        uploaded_file = st.file_uploader(
            "Upload file",
            type=["csv", "txt"],
            help="CSV files should have a 'text' column. TXT files should have one text per line.",
        )

        if uploaded_file:
            # Read file
            if uploaded_file.name.endswith(".csv"):
                # Try multiple encodings for CSV
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("Could not decode CSV file. Please use UTF-8 encoding.")
                    return

                # Column selection
                text_columns = df.select_dtypes(include=["object"]).columns.tolist()
                if not text_columns:
                    st.error("No text columns found in CSV.")
                    return

                text_column = st.selectbox("Select text column:", text_columns)
                texts = df[text_column].tolist()
            else:
                # Try multiple encodings
                file_bytes = uploaded_file.getvalue()
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        content = StringIO(file_bytes.decode(encoding))
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("Could not decode file. Please use UTF-8 encoding.")
                    return
                texts = [line.strip() for line in content if line.strip()]
                df = pd.DataFrame({"text": texts})
                text_column = "text"

            st.info(f"Found {len(texts)} texts to analyze.")

            if st.button("🚀 Analyze All", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                results = []

                for i, text in enumerate(texts):
                    result = predictor.predict(text)
                    results.append(result)
                    progress_bar.progress((i + 1) / len(texts))

                # Add results to dataframe
                df["Sentiment"] = [r["label"] for r in results]
                df["Confidence"] = [r.get("confidence", None) for r in results]

                # Display results
                st.success(f"Analyzed {len(texts)} texts!")

                # Summary stats
                col1, col2, col3 = st.columns(3)
                sentiment_counts = df["Sentiment"].value_counts()

                with col1:
                    pos_count = sentiment_counts.get("positive", 0)
                    st.metric("Positive", pos_count, f"{pos_count/len(df)*100:.1f}%")

                with col2:
                    neu_count = sentiment_counts.get("neutral", 0)
                    st.metric("Neutral", neu_count, f"{neu_count/len(df)*100:.1f}%")

                with col3:
                    neg_count = sentiment_counts.get("negative", 0)
                    st.metric("Negative", neg_count, f"{neg_count/len(df)*100:.1f}%")

                # Distribution chart
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        "positive": "#28a745",
                        "negative": "#dc3545",
                        "neutral": "#007bff",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

                # Sentiment Trend Analysis
                st.markdown("### 📈 Sentiment Trend Analysis")
                pos_pct = pos_count / len(df) * 100
                neg_pct = neg_count / len(df) * 100
                neu_pct = neu_count / len(df) * 100

                # Determine overall sentiment trend
                if pos_pct > neg_pct + 10:
                    trend = "bullish"
                    trend_color = "#28a745"
                    trend_icon = "📈"
                elif neg_pct > pos_pct + 10:
                    trend = "bearish"
                    trend_color = "#dc3545"
                    trend_icon = "📉"
                else:
                    trend = "mixed"
                    trend_color = "#007bff"
                    trend_icon = "↔️"

                # Trend summary box
                avg_confidence = df["Confidence"].mean() if df["Confidence"].notna().any() else 0
                st.markdown(
                    f"""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;
                                border-left: 5px solid {trend_color};'>
                        <h4 style='margin: 0;'>{trend_icon} Overall Sentiment: <span style='color: {trend_color};'>{trend.upper()}</span></h4>
                        <p style='margin-top: 10px; margin-bottom: 0;'>
                            <b>Summary:</b> Out of {len(df)} texts analyzed, {pos_pct:.1f}% are positive,
                            {neu_pct:.1f}% are neutral, and {neg_pct:.1f}% are negative.<br>
                            <b>Average Confidence:</b> {avg_confidence:.1%}<br>
                            <b>Dominant Sentiment:</b> {df['Sentiment'].mode().iloc[0].capitalize() if len(df) > 0 else 'N/A'}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("")  # Spacer

                # Results table
                st.markdown("### Detailed Results")
                st.dataframe(
                    df[[text_column, "Sentiment", "Confidence"]].head(100),
                    use_container_width=True,
                )

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )

                # Market Outlook Report
                with st.expander("📊 Generate Market Outlook Report", expanded=False):
                    st.markdown("*AI-generated market sentiment analysis based on your batch data*")
                    if st.button("🔮 Generate Outlook", key="generate_outlook"):
                        with st.spinner("Generating market outlook..."):
                            sentiment_counts_dict = {
                                "positive": sentiment_counts.get("positive", 0),
                                "neutral": sentiment_counts.get("neutral", 0),
                                "negative": sentiment_counts.get("negative", 0),
                            }
                            outlook = generate_market_outlook(
                                sentiment_counts=sentiment_counts_dict,
                                total_texts=len(df),
                                avg_confidence=avg_confidence,
                            )
                            st.markdown(outlook)

    # Tab 3: Explainability
    with tab3:
        st.header("Explainability")
        st.markdown("Understand why the model made its prediction.")

        # Only works with baseline models
        if not selected_model.startswith("baseline_"):
            st.warning("Explainability is currently available only for baseline models. "
                      "Please select a baseline model from the sidebar.")
        else:
            explain_text = st.text_area(
                "Enter text to explain:",
                height=100,
                placeholder="Enter financial text...",
                key="explain_text",
            )

            if st.button("🔍 Explain Prediction", type="primary", use_container_width=True):
                if explain_text.strip():
                    with st.spinner("Analyzing..."):
                        explanation = explain_prediction_baseline(explain_text, selected_model)

                    # Show prediction
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown("### Prediction")
                        sentiment = explanation["prediction"]
                        color = get_sentiment_color(sentiment)
                        st.markdown(
                            f"<h2 style='color: {color};'>{sentiment.upper()}</h2>",
                            unsafe_allow_html=True,
                        )

                        if explanation["probabilities"]:
                            st.markdown("**Probabilities:**")
                            for label, prob in explanation["probabilities"].items():
                                st.write(f"- {label}: {prob:.1%}")

                    with col2:
                        if explanation["probabilities"]:
                            fig = create_probability_chart(explanation["probabilities"])
                            st.plotly_chart(fig, use_container_width=True)

                    # Highlighted text
                    st.markdown("### Text Analysis")
                    st.markdown("Words highlighted by their sentiment contribution:")
                    highlighted = highlight_text(explain_text, explanation["word_importance"])
                    st.markdown(
                        f"<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; "
                        f"line-height: 2;'>{highlighted}</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("""
                    <div style='margin-top: 10px; font-size: 0.9em;'>
                    <span style='background-color: #d4edda; padding: 2px 8px; border-radius: 3px;'>Positive</span>
                    <span style='background-color: #f8d7da; padding: 2px 8px; border-radius: 3px; margin-left: 10px;'>Negative</span>
                    <span style='background-color: #cce5ff; padding: 2px 8px; border-radius: 3px; margin-left: 10px;'>Neutral</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Word importance
                    st.markdown("### Key Words")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**Positive Indicators**")
                        for word, score in explanation["positive_words"]:
                            st.write(f"- {word} ({score:.3f})")

                    with col2:
                        st.markdown("**Negative Indicators**")
                        for word, score in explanation["negative_words"]:
                            st.write(f"- {word} ({score:.3f})")

                    with col3:
                        st.markdown("**Neutral Indicators**")
                        for word, score in explanation["neutral_words"]:
                            st.write(f"- {word} ({score:.3f})")

                    # Top words chart
                    if explanation["word_importance"]:
                        st.markdown("### Word Importance Scores")
                        top_words = explanation["word_importance"][:10]
                        words = [w for w, s, d in top_words]
                        scores = [s for w, s, d in top_words]
                        directions = [d for w, s, d in top_words]
                        colors = [get_sentiment_color(d) for d in directions]

                        fig = go.Figure(go.Bar(
                            x=scores,
                            y=words,
                            orientation='h',
                            marker_color=colors,
                        ))
                        fig.update_layout(
                            title="Top 10 Important Words",
                            xaxis_title="Importance Score",
                            yaxis=dict(autorange="reversed"),
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please enter text to explain.")

    # Tab 4: Word Insights
    with tab4:
        st.header("Word Insights")
        st.markdown("Discover which words most strongly indicate each sentiment in the trained model.")

        # Only works with baseline models
        if not selected_model.startswith("baseline_"):
            st.warning("Word insights are currently available only for baseline models. "
                      "Please select a baseline model from the sidebar.")
        else:
            with st.spinner("Loading word insights..."):
                try:
                    importance = get_feature_importance_summary(selected_model)

                    if "error" in importance:
                        st.error(importance["error"])
                    else:
                        # Top words for each sentiment
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("### 📈 Positive Words")
                            st.markdown("Words that indicate positive sentiment:")
                            if "top_positive" in importance:
                                for word, score in importance["top_positive"]:
                                    st.markdown(
                                        f"<span style='background-color: #d4edda; padding: 3px 8px; "
                                        f"border-radius: 4px; margin: 2px; display: inline-block;'>"
                                        f"**{word}** ({score:.3f})</span>",
                                        unsafe_allow_html=True,
                                    )

                        with col2:
                            st.markdown("### 📉 Negative Words")
                            st.markdown("Words that indicate negative sentiment:")
                            if "top_negative" in importance:
                                for word, score in importance["top_negative"]:
                                    st.markdown(
                                        f"<span style='background-color: #f8d7da; padding: 3px 8px; "
                                        f"border-radius: 4px; margin: 2px; display: inline-block;'>"
                                        f"**{word}** ({score:.3f})</span>",
                                        unsafe_allow_html=True,
                                    )

                        with col3:
                            st.markdown("### ↔️ Neutral Words")
                            st.markdown("Words that indicate neutral sentiment:")
                            if "top_neutral" in importance:
                                for word, score in importance["top_neutral"]:
                                    st.markdown(
                                        f"<span style='background-color: #cce5ff; padding: 3px 8px; "
                                        f"border-radius: 4px; margin: 2px; display: inline-block;'>"
                                        f"**{word}** ({score:.3f})</span>",
                                        unsafe_allow_html=True,
                                    )

                        # Visualization
                        st.markdown("---")
                        st.markdown("### Word Importance Comparison")

                        # Create a combined chart
                        chart_data = []
                        for sentiment in ["positive", "negative", "neutral"]:
                            key = f"top_{sentiment}"
                            if key in importance:
                                for word, score in importance[key][:5]:  # Top 5 each
                                    chart_data.append({
                                        "Word": word,
                                        "Score": score,
                                        "Sentiment": sentiment.capitalize()
                                    })

                        if chart_data:
                            chart_df = pd.DataFrame(chart_data)
                            fig = px.bar(
                                chart_df,
                                x="Score",
                                y="Word",
                                color="Sentiment",
                                orientation="h",
                                color_discrete_map={
                                    "Positive": "#28a745",
                                    "Negative": "#dc3545",
                                    "Neutral": "#007bff",
                                },
                                title="Top 5 Words for Each Sentiment",
                            )
                            fig.update_layout(
                                height=500,
                                yaxis=dict(categoryorder="total ascending"),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error loading word insights: {e}")

    # Tab 5: Model Info
    with tab5:
        st.header("Model Information")

        # Model performance data
        MODEL_INFO = {
            "baseline_logreg": {
                "name": "Logistic Regression",
                "type": "TF-IDF + Classifier",
                "accuracy": "88.5%",
                "f1_macro": "0.844",
                "f1_weighted": "0.887",
                "speed": "Very Fast",
                "features": ["TF-IDF vectorization (unigrams + bigrams)", "Max 10,000 features", "Balanced class weights", "Linear decision boundary"],
            },
            "baseline_naive_bayes": {
                "name": "Naive Bayes",
                "type": "TF-IDF + Classifier",
                "accuracy": "86.3%",
                "f1_macro": "0.809",
                "f1_weighted": "0.864",
                "speed": "Very Fast",
                "features": ["TF-IDF vectorization", "Multinomial distribution", "Good for text classification", "Alpha smoothing = 0.1"],
            },
            "baseline_svm": {
                "name": "Support Vector Machine (SVM)",
                "type": "TF-IDF + Classifier",
                "accuracy": "92%",
                "f1_macro": "0.90",
                "f1_weighted": "0.93",
                "speed": "Fast",
                "features": ["TF-IDF vectorization", "Linear kernel", "Maximum margin classifier", "Balanced class weights", "Best for high-dimensional sparse data"],
            },
            "baseline_random_forest": {
                "name": "Random Forest",
                "type": "TF-IDF + Ensemble",
                "accuracy": "87.6%",
                "f1_macro": "0.827",
                "f1_weighted": "0.871",
                "speed": "Medium",
                "features": ["200 decision trees", "Max depth 50", "Ensemble voting", "Handles non-linear patterns"],
            },
            "baseline_gradient_boosting": {
                "name": "Gradient Boosting",
                "type": "TF-IDF + Boosting",
                "accuracy": "94%",
                "f1_macro": "0.92",
                "f1_weighted": "0.94",
                "speed": "Medium",
                "features": ["100 boosting iterations", "Sequential error correction", "Learning rate 0.1", "Highest accuracy baseline"],
            },
            "baseline_mlp": {
                "name": "Multi-Layer Perceptron (Neural Network)",
                "type": "TF-IDF + Deep Learning",
                "accuracy": "87.6%",
                "f1_macro": "0.827",
                "f1_weighted": "0.876",
                "speed": "Medium",
                "features": ["2 hidden layers (256, 128 neurons)", "Early stopping", "Non-linear activation", "Learns complex patterns"],
            },
            "baseline_ensemble": {
                "name": "Voting Ensemble",
                "type": "TF-IDF + Combined Models",
                "accuracy": "88.5%",
                "f1_macro": "0.843",
                "f1_weighted": "0.885",
                "speed": "Slow",
                "features": ["Combines LogReg + SVM + Random Forest", "Soft voting (probability-based)", "More robust predictions"],
            },
            "finbert_pretrained": {
                "name": "FinBERT (Pre-trained)",
                "type": "Transformer",
                "accuracy": "~90%",
                "f1_macro": "~0.88",
                "f1_weighted": "~0.90",
                "speed": "Slow (CPU) / Fast (GPU)",
                "features": ["Pre-trained on financial text", "ProsusAI/finbert from HuggingFace", "Understands financial context", "No training required", "110M parameters"],
            },
        }

        # Get current model info
        info = MODEL_INFO.get(selected_model, {
            "name": selected_model,
            "type": "Unknown",
            "accuracy": "N/A",
            "f1_macro": "N/A",
            "f1_weighted": "N/A",
            "speed": "N/A",
            "features": [],
        })

        # Display current model card
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;'>
            <h2 style='margin: 0; color: white;'>🎯 Current Model: {info['name']}</h2>
            <p style='margin: 10px 0 0 0; opacity: 0.9;'>{info['type']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", info["accuracy"])
        with col2:
            st.metric("F1 (Macro)", info["f1_macro"])
        with col3:
            st.metric("F1 (Weighted)", info["f1_weighted"])
        with col4:
            st.metric("Speed", info["speed"])

        # Features
        st.markdown("### Model Features")
        for feature in info["features"]:
            st.markdown(f"- {feature}")

        st.markdown("---")

        # All models comparison table
        st.markdown("### All Available Models")

        comparison_data = []
        for model_key in available_models:
            m_info = MODEL_INFO.get(model_key, {"name": model_key, "accuracy": "N/A", "f1_macro": "N/A", "speed": "N/A"})
            is_current = "✅" if model_key == selected_model else ""
            comparison_data.append({
                "Model": m_info.get("name", model_key),
                "Type": m_info.get("type", "Unknown"),
                "Accuracy": m_info.get("accuracy", "N/A"),
                "F1 Score": m_info.get("f1_macro", "N/A"),
                "Speed": m_info.get("speed", "N/A"),
                "Active": is_current,
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Dataset info
        st.markdown("---")
        st.markdown("### Dataset Information")
        st.markdown("""
        **Financial PhraseBank** by Malo et al. (2014)
        - **Source:** Financial news and reports
        - **Labels:** Positive, Neutral, Negative
        - **Samples:** 2,264 sentences (100% annotator agreement subset)
        - **Split:** 70% train, 15% validation, 15% test
        """)


if __name__ == "__main__":
    main()

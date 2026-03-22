"""Tests for src/llm_enhanced.py"""

from llm_enhanced import (
    ChainOfThoughtReasoner,
    EliteExplanationGenerator,
    ChainOfThought,
    ThoughtStep,
    ReasoningStep,
    generate_elite_market_outlook,
)


class TestChainOfThoughtReasoner:
    def setup_method(self):
        self.reasoner = ChainOfThoughtReasoner()

    def test_analyze_returns_chain(self):
        cot = self.reasoner.analyze("Revenue increased 25%.")
        assert isinstance(cot, ChainOfThought)
        assert len(cot.steps) == 6  # All 6 reasoning steps

    def test_final_sentiment_valid(self):
        cot = self.reasoner.analyze("Strong earnings growth beating expectations.")
        assert cot.final_sentiment in ["positive", "negative", "neutral"]

    def test_confidence_in_range(self):
        cot = self.reasoner.analyze("Revenue declined sharply.")
        assert 0.0 <= cot.final_confidence <= 1.0

    def test_explanation_not_empty(self):
        cot = self.reasoner.analyze("Profits soared to record levels.")
        assert len(cot.explanation) > 0

    def test_reasoning_trace_not_empty(self):
        cot = self.reasoner.analyze("Revenue was flat.")
        trace = cot.get_reasoning_trace()
        assert len(trace) > 0
        assert "Step 1" in trace

    def test_with_model_prediction(self):
        cot = self.reasoner.analyze(
            "Strong growth in Q4.",
            model_prediction="positive",
            model_confidence=0.9,
        )
        assert cot.final_confidence > 0

    def test_key_factors_list(self):
        cot = self.reasoner.analyze("Company faces risk of litigation and declining revenue.")
        assert isinstance(cot.key_factors, list)


class TestEliteExplanationGenerator:
    def setup_method(self):
        self.generator = EliteExplanationGenerator()

    def test_comprehensive_explanation(self):
        result = self.generator.generate_comprehensive_explanation(
            text="Revenue increased 25% beating expectations.",
            model_prediction="positive",
            model_confidence=0.85,
            probabilities={"positive": 0.85, "neutral": 0.10, "negative": 0.05},
        )
        assert isinstance(result, dict)
        assert "summary" in result
        assert "reasoning_trace" in result
        assert "model_vs_reasoning" in result

    def test_quick_explanation(self):
        result = self.generator.generate_quick_explanation(
            text="Losses mounted significantly.",
            prediction="negative",
            confidence=0.8,
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestMarketOutlook:
    def test_generate_outlook(self):
        texts = ["Revenue up.", "Losses mount.", "Flat results."]
        predictions = ["positive", "negative", "neutral"]
        confidences = [0.8, 0.7, 0.6]

        report = generate_elite_market_outlook(texts, predictions, confidences)
        assert "trend" in report
        assert "statistics" in report
        assert "narrative" in report
        assert report["statistics"]["total_analyzed"] == 3

    def test_bullish_trend(self):
        predictions = ["positive"] * 5 + ["negative"]
        confidences = [0.9] * 6
        texts = ["text"] * 6
        report = generate_elite_market_outlook(texts, predictions, confidences)
        assert "bullish" in report["trend"]

    def test_bearish_trend(self):
        predictions = ["negative"] * 5 + ["positive"]
        confidences = [0.9] * 6
        texts = ["text"] * 6
        report = generate_elite_market_outlook(texts, predictions, confidences)
        assert "bearish" in report["trend"]


class TestThoughtStep:
    def test_to_dict(self):
        step = ThoughtStep(
            step=ReasoningStep.COMPREHENSION,
            observation="Test observation",
            reasoning="Test reasoning",
            conclusion="Test conclusion",
            confidence=0.8,
        )
        d = step.to_dict()
        assert d["step"] == "comprehension"
        assert d["confidence"] == 0.8

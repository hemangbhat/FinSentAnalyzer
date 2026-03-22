"""Tests for src/nlp_advanced.py"""

from nlp_advanced import (
    AdvancedTextProcessor,
    FinancialTextAnalyzer,
    TextFeatures,
    analyze_financial_text,
    extract_financial_features,
    get_lexicon_sentiment,
)
from lexicons import FINANCIAL_POSITIVE, FINANCIAL_NEGATIVE, FINANCIAL_UNCERTAINTY


class TestAdvancedTextProcessor:
    def setup_method(self):
        self.processor = AdvancedTextProcessor()

    def test_process_returns_processed_text(self):
        result = self.processor.process("Revenue increased 25%.")
        assert result.original == "Revenue increased 25%."
        assert len(result.tokens) > 0

    def test_clean_text_removes_urls(self):
        cleaned = self.processor._clean_text("Visit https://example.com for details.")
        assert "https" not in cleaned

    def test_normalize_replaces_percentages(self):
        normalized = self.processor._normalize_text("up 25%")
        assert "_percent_" in normalized.lower()

    def test_feature_extraction_counts_words(self):
        result = self.processor.process("one two three four five")
        assert result.features.word_count == 5

    def test_feature_extraction_detects_negation(self):
        result = self.processor.process("The company did not report any growth.")
        assert result.features.has_negation is True
        assert result.features.negation_count > 0

    def test_feature_extraction_detects_positive_words(self):
        result = self.processor.process("The company achieved strong growth and profitable results.")
        assert result.features.positive_word_count > 0

    def test_feature_extraction_detects_negative_words(self):
        result = self.processor.process("Revenue declined sharply causing significant losses.")
        assert result.features.negative_word_count > 0


class TestFinancialTextAnalyzer:
    def setup_method(self):
        self.analyzer = FinancialTextAnalyzer()

    def test_analyze_returns_dict(self):
        result = self.analyzer.analyze("Revenue increased.")
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "features" in result

    def test_positive_text_detected(self):
        result = self.analyzer.analyze(
            "The company reported strong earnings growth, beating expectations."
        )
        # Should lean positive or at least not negative
        assert result["sentiment"]["score"] >= 0

    def test_negative_text_detected(self):
        result = self.analyzer.analyze(
            "Revenue declined sharply due to weak demand and rising losses."
        )
        assert result["sentiment"]["score"] <= 0

    def test_batch_analyze(self):
        texts = ["Good growth.", "Bad losses.", "Flat results."]
        results = self.analyzer.batch_analyze(texts)
        assert len(results) == 3

    def test_aggregate_sentiment(self):
        texts = ["Strong growth.", "Severe decline.", "Flat results."]
        agg = self.analyzer.get_aggregate_sentiment(texts)
        assert "total_texts" in agg
        assert agg["total_texts"] == 3
        assert "overall_trend" in agg


class TestConvenienceFunctions:
    def test_analyze_financial_text(self):
        result = analyze_financial_text("Revenue increased 10%.")
        assert isinstance(result, dict)

    def test_extract_financial_features(self):
        features = extract_financial_features("Revenue increased 10%.")
        assert isinstance(features, TextFeatures)

    def test_get_lexicon_sentiment(self):
        result = get_lexicon_sentiment("Strong profit growth and excellent earnings.")
        assert result["positive_count"] > 0


class TestLexicons:
    def test_positive_lexicon_populated(self):
        assert len(FINANCIAL_POSITIVE) > 100

    def test_negative_lexicon_populated(self):
        assert len(FINANCIAL_NEGATIVE) > 100

    def test_uncertainty_lexicon_populated(self):
        assert len(FINANCIAL_UNCERTAINTY) > 50

    def test_growth_is_positive(self):
        assert "growth" in FINANCIAL_POSITIVE

    def test_loss_is_negative(self):
        assert "loss" in FINANCIAL_NEGATIVE

    def test_may_is_uncertain(self):
        assert "may" in FINANCIAL_UNCERTAINTY

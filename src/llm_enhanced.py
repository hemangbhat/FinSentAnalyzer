"""
Elite LLM-powered Analysis for Financial Sentiment.
Chain-of-thought reasoning, multi-step analysis, and structured outputs.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json

# Import advanced NLP module
try:
    from nlp_advanced import (
        FinancialTextAnalyzer,
        TextFeatures,
    )
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Import lexicons directly
try:
    from lexicons import (
        FINANCIAL_POSITIVE,
        FINANCIAL_NEGATIVE,
        FINANCIAL_UNCERTAINTY,
    )
    LEXICON_AVAILABLE = True
except ImportError:
    LEXICON_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ReasoningStep(Enum):
    """Steps in chain-of-thought reasoning."""
    COMPREHENSION = "comprehension"
    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_DETECTION = "sentiment_detection"
    CONTEXT_ANALYSIS = "context_analysis"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    FINAL_SYNTHESIS = "final_synthesis"


@dataclass
class ThoughtStep:
    """A single step in the reasoning chain."""
    step: ReasoningStep
    observation: str
    reasoning: str
    conclusion: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step.value,
            "observation": self.observation,
            "reasoning": self.reasoning,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
        }


@dataclass
class ChainOfThought:
    """Complete chain-of-thought analysis."""
    text: str
    steps: List[ThoughtStep] = field(default_factory=list)
    final_sentiment: str = "neutral"
    final_confidence: float = 0.5
    explanation: str = ""
    key_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "steps": [s.to_dict() for s in self.steps],
            "final_sentiment": self.final_sentiment,
            "final_confidence": self.final_confidence,
            "explanation": self.explanation,
            "key_factors": self.key_factors,
        }

    def get_reasoning_trace(self) -> str:
        """Get human-readable reasoning trace."""
        trace = []
        for i, step in enumerate(self.steps, 1):
            trace.append(f"**Step {i}: {step.step.value.replace('_', ' ').title()}**")
            trace.append(f"  • Observation: {step.observation}")
            trace.append(f"  • Reasoning: {step.reasoning}")
            trace.append(f"  • Conclusion: {step.conclusion}")
            trace.append("")
        return "\n".join(trace)


# =============================================================================
# CHAIN-OF-THOUGHT REASONING ENGINE
# =============================================================================

class ChainOfThoughtReasoner:
    """
    Elite chain-of-thought reasoning for financial sentiment analysis.

    Implements a multi-step reasoning process:
    1. Comprehension - Understand the text
    2. Entity Extraction - Identify key financial entities
    3. Sentiment Detection - Detect sentiment signals
    4. Context Analysis - Analyze context and modifiers
    5. Confidence Calibration - Calibrate confidence
    6. Final Synthesis - Combine all factors
    """

    def __init__(self):
        """Initialize the reasoner."""
        if NLP_AVAILABLE:
            self.nlp_analyzer = FinancialTextAnalyzer()
        else:
            self.nlp_analyzer = None

    def analyze(self, text: str, model_prediction: Optional[str] = None,
                model_confidence: Optional[float] = None) -> ChainOfThought:
        """
        Perform chain-of-thought analysis on financial text.

        Args:
            text: Financial text to analyze
            model_prediction: Optional ML model prediction to incorporate
            model_confidence: Optional ML model confidence

        Returns:
            ChainOfThought with complete reasoning trace
        """
        cot = ChainOfThought(text=text)

        # Step 1: Comprehension
        cot.steps.append(self._step_comprehension(text))

        # Step 2: Entity Extraction
        cot.steps.append(self._step_entity_extraction(text))

        # Step 3: Sentiment Detection
        cot.steps.append(self._step_sentiment_detection(text))

        # Step 4: Context Analysis
        cot.steps.append(self._step_context_analysis(text))

        # Step 5: Confidence Calibration
        cot.steps.append(self._step_confidence_calibration(text, cot.steps, model_confidence))

        # Step 6: Final Synthesis
        synthesis = self._step_final_synthesis(cot.steps, model_prediction)
        cot.steps.append(synthesis)

        # Set final results
        cot.final_sentiment = synthesis.conclusion.split(":")[0].lower().strip()
        cot.final_confidence = synthesis.confidence
        cot.explanation = self._generate_explanation(cot)
        cot.key_factors = self._extract_key_factors(cot)

        return cot

    def _step_comprehension(self, text: str) -> ThoughtStep:
        """Step 1: Understand what the text is about."""
        # Identify text type
        text_lower = text.lower()

        if any(word in text_lower for word in ["earnings", "revenue", "profit", "loss", "quarter", "fiscal"]):
            text_type = "earnings/financial results"
        elif any(word in text_lower for word in ["acquire", "merger", "acquisition", "deal", "buyout"]):
            text_type = "M&A activity"
        elif any(word in text_lower for word in ["stock", "shares", "price", "trading", "market"]):
            text_type = "market/stock commentary"
        elif any(word in text_lower for word in ["forecast", "outlook", "guidance", "expect"]):
            text_type = "forward-looking guidance"
        elif any(word in text_lower for word in ["lawsuit", "litigation", "court", "regulatory"]):
            text_type = "legal/regulatory"
        else:
            text_type = "general financial news"

        # Count sentences
        sentence_count = len(re.findall(r'[.!?]+', text)) or 1
        word_count = len(text.split())

        observation = f"Text is {word_count} words, {sentence_count} sentence(s), discussing {text_type}"
        reasoning = f"Identified this as {text_type} based on domain-specific keywords. Length suggests {'detailed' if word_count > 30 else 'brief'} coverage."
        conclusion = f"Text type: {text_type}; Complexity: {'high' if word_count > 50 else 'medium' if word_count > 20 else 'low'}"

        return ThoughtStep(
            step=ReasoningStep.COMPREHENSION,
            observation=observation,
            reasoning=reasoning,
            conclusion=conclusion,
            confidence=0.9,
        )

    def _step_entity_extraction(self, text: str) -> ThoughtStep:
        """Step 2: Extract key financial entities."""
        entities = {
            "percentages": re.findall(r'[-+]?\d+(?:\.\d+)?%', text),
            "currency": re.findall(r'\$\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|[MBT]))?\b', text, re.I),
            "companies": re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|Co)\.?\b', text),
        }

        total_entities = sum(len(v) for v in entities.values())

        observation = f"Found {total_entities} financial entities: {len(entities['percentages'])} percentages, {len(entities['currency'])} currency amounts"

        if entities['percentages']:
            # Analyze percentage direction
            positive_pcts = [p for p in entities['percentages'] if not p.startswith('-')]
            negative_pcts = [p for p in entities['percentages'] if p.startswith('-')]
            reasoning = f"Percentages: {entities['percentages']}. "
            if negative_pcts:
                reasoning += f"Negative percentages ({negative_pcts}) suggest decline. "
            if positive_pcts:
                reasoning += f"Positive percentages ({positive_pcts}) suggest growth."
        else:
            reasoning = "No specific numerical data found; sentiment will rely on qualitative language."

        conclusion = f"Entity density: {'high' if total_entities > 3 else 'medium' if total_entities > 0 else 'low'}"

        return ThoughtStep(
            step=ReasoningStep.ENTITY_EXTRACTION,
            observation=observation,
            reasoning=reasoning,
            conclusion=conclusion,
            confidence=0.85,
        )

    def _step_sentiment_detection(self, text: str) -> ThoughtStep:
        """Step 3: Detect sentiment signals."""
        # Strip punctuation from words before matching against lexicon
        words = [re.sub(r'[^\w]', '', w.lower()) for w in text.split()]
        words = [w for w in words if w]  # Remove empty strings

        # Count sentiment words
        positive_found = [w for w in words if w in FINANCIAL_POSITIVE] if LEXICON_AVAILABLE else []
        negative_found = [w for w in words if w in FINANCIAL_NEGATIVE] if LEXICON_AVAILABLE else []
        uncertainty_found = [w for w in words if w in FINANCIAL_UNCERTAINTY] if LEXICON_AVAILABLE else []

        # Fallback if NLP not available
        if not LEXICON_AVAILABLE:
            positive_words = {"growth", "strong", "profit", "gain", "increase", "beat", "success", "improve", "positive", "rise", "record", "exceed"}
            negative_words = {"loss", "decline", "fall", "weak", "miss", "fail", "drop", "negative", "concern", "down", "risk", "cut"}
            uncertainty_words = {"may", "might", "could", "expect", "forecast", "uncertain", "potential", "possible"}
            positive_found = [w for w in words if w in positive_words]
            negative_found = [w for w in words if w in negative_words]
            uncertainty_found = [w for w in words if w in uncertainty_words]

        pos_count = len(positive_found)
        neg_count = len(negative_found)
        unc_count = len(uncertainty_found)

        observation = f"Sentiment words: +{pos_count} positive, -{neg_count} negative, ~{unc_count} uncertain"

        # Determine dominant sentiment
        if pos_count > neg_count * 1.5:
            dominant = "positive"
            reasoning = f"Positive words ({positive_found[:3]}) significantly outweigh negatives"
        elif neg_count > pos_count * 1.5:
            dominant = "negative"
            reasoning = f"Negative words ({negative_found[:3]}) significantly outweigh positives"
        elif pos_count > neg_count:
            dominant = "slightly positive"
            reasoning = f"Slight positive lean with words like: {positive_found[:2]}"
        elif neg_count > pos_count:
            dominant = "slightly negative"
            reasoning = f"Slight negative lean with words like: {negative_found[:2]}"
        else:
            dominant = "neutral/balanced"
            reasoning = "No strong sentiment direction; balanced or factual language"

        if unc_count > 2:
            reasoning += f". High uncertainty ({uncertainty_found[:2]}) suggests forward-looking or speculative content."

        conclusion = f"Dominant signal: {dominant}"

        return ThoughtStep(
            step=ReasoningStep.SENTIMENT_DETECTION,
            observation=observation,
            reasoning=reasoning,
            conclusion=conclusion,
            confidence=0.8 if abs(pos_count - neg_count) > 2 else 0.6,
        )

    def _step_context_analysis(self, text: str) -> ThoughtStep:
        """Step 4: Analyze context modifiers."""
        text_lower = text.lower()

        # Check for negations
        negations = ["not", "no", "never", "neither", "cannot", "can't", "won't", "don't", "didn't", "without", "fail"]
        negation_found = [n for n in negations if n in text_lower]

        # Check for comparisons
        comparisons = ["better than", "worse than", "compared to", "versus", "vs", "higher than", "lower than"]
        comparison_found = [c for c in comparisons if c in text_lower]

        # Check for temporal indicators
        temporal = ["will", "expect", "anticipate", "forecast", "plan", "future", "upcoming", "next"]
        future_focused = any(t in text_lower for t in temporal)

        observation = f"Context modifiers: {len(negation_found)} negations, {len(comparison_found)} comparisons, {'future-focused' if future_focused else 'current/past focused'}"

        reasoning_parts = []
        if negation_found:
            reasoning_parts.append(f"Negations ({negation_found[:2]}) may invert sentiment of adjacent words")
        if comparison_found:
            reasoning_parts.append(f"Comparative language suggests relative performance assessment")
        if future_focused:
            reasoning_parts.append("Forward-looking language indicates predictions/expectations")

        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Standard declarative context without significant modifiers"

        impact = "high" if negation_found else "medium" if comparison_found or future_focused else "low"
        conclusion = f"Context impact: {impact}"

        return ThoughtStep(
            step=ReasoningStep.CONTEXT_ANALYSIS,
            observation=observation,
            reasoning=reasoning,
            conclusion=conclusion,
            confidence=0.75,
        )

    def _step_confidence_calibration(self, text: str, steps: List[ThoughtStep],
                                     model_confidence: Optional[float] = None) -> ThoughtStep:
        """Step 5: Calibrate confidence based on all factors."""
        # Calculate base confidence from step confidences
        avg_step_confidence = sum(s.confidence for s in steps) / len(steps) if steps else 0.5

        # Text length factor (longer = more signal = higher confidence)
        word_count = len(text.split())
        length_factor = min(1.0, word_count / 30)

        # Sentiment clarity (from step 3)
        sentiment_step = next((s for s in steps if s.step == ReasoningStep.SENTIMENT_DETECTION), None)
        clarity_factor = sentiment_step.confidence if sentiment_step else 0.5

        # Context complexity (from step 4)
        context_step = next((s for s in steps if s.step == ReasoningStep.CONTEXT_ANALYSIS), None)
        context_impact = "high" in (context_step.conclusion if context_step else "")
        complexity_penalty = 0.1 if context_impact else 0

        # Calculate final confidence
        base_confidence = (avg_step_confidence + length_factor + clarity_factor) / 3 - complexity_penalty

        # Incorporate model confidence if available
        if model_confidence is not None:
            final_confidence = (base_confidence + model_confidence) / 2
            observation = f"Reasoning confidence: {base_confidence:.2f}, Model confidence: {model_confidence:.2f}"
        else:
            final_confidence = base_confidence
            observation = f"Reasoning confidence: {base_confidence:.2f} (no model confidence provided)"

        final_confidence = max(0.1, min(0.99, final_confidence))

        reasoning = f"Confidence derived from: text length ({length_factor:.2f}), sentiment clarity ({clarity_factor:.2f}), context complexity penalty ({complexity_penalty:.2f})"

        if final_confidence > 0.8:
            conclusion = f"High confidence ({final_confidence:.2f}) - clear sentiment signals"
        elif final_confidence > 0.6:
            conclusion = f"Moderate confidence ({final_confidence:.2f}) - reasonably clear signals"
        else:
            conclusion = f"Low confidence ({final_confidence:.2f}) - ambiguous or mixed signals"

        return ThoughtStep(
            step=ReasoningStep.CONFIDENCE_CALIBRATION,
            observation=observation,
            reasoning=reasoning,
            conclusion=conclusion,
            confidence=final_confidence,
        )

    def _step_final_synthesis(self, steps: List[ThoughtStep],
                             model_prediction: Optional[str] = None) -> ThoughtStep:
        """Step 6: Synthesize all analysis into final verdict."""
        # Get key conclusions from each step
        sentiment_step = next((s for s in steps if s.step == ReasoningStep.SENTIMENT_DETECTION), None)
        confidence_step = next((s for s in steps if s.step == ReasoningStep.CONFIDENCE_CALIBRATION), None)
        context_step = next((s for s in steps if s.step == ReasoningStep.CONTEXT_ANALYSIS), None)

        # Extract dominant sentiment from detection step
        dominant = ""
        if sentiment_step:
            conclusion = sentiment_step.conclusion.lower()
            if "positive" in conclusion:
                dominant = "positive"
            elif "negative" in conclusion:
                dominant = "negative"
            else:
                dominant = "neutral"

        # Get confidence
        final_confidence = confidence_step.confidence if confidence_step else 0.5

        # Check for context that might flip sentiment
        if context_step and "negation" in context_step.observation.lower():
            if "high" in context_step.conclusion.lower():
                # Significant negations present - might need to flip
                if dominant == "positive":
                    dominant = "negative"
                elif dominant == "negative":
                    dominant = "positive"
                final_confidence *= 0.8  # Reduce confidence due to complexity

        # Incorporate model prediction if available
        if model_prediction:
            if model_prediction.lower() == dominant:
                final_confidence = min(0.99, final_confidence * 1.1)  # Agreement boosts confidence
            else:
                # Disagreement - use model but reduce confidence
                dominant = model_prediction.lower()
                final_confidence *= 0.85

        observation = f"Synthesized from {len(steps)} reasoning steps"

        reasoning_parts = [f"Sentiment detection: {dominant}"]
        if context_step:
            reasoning_parts.append(f"Context analysis: {context_step.conclusion}")
        if model_prediction:
            reasoning_parts.append(f"Model prediction: {model_prediction} ({'agrees' if model_prediction.lower() == dominant else 'disagrees'})")
        reasoning = "; ".join(reasoning_parts)

        conclusion = f"{dominant.upper()}: Final sentiment with {final_confidence:.1%} confidence"

        return ThoughtStep(
            step=ReasoningStep.FINAL_SYNTHESIS,
            observation=observation,
            reasoning=reasoning,
            conclusion=conclusion,
            confidence=final_confidence,
        )

    def _generate_explanation(self, cot: ChainOfThought) -> str:
        """Generate human-readable explanation from chain of thought."""
        # Get key information
        final_step = cot.steps[-1] if cot.steps else None
        sentiment_step = next((s for s in cot.steps if s.step == ReasoningStep.SENTIMENT_DETECTION), None)

        sentiment = cot.final_sentiment.upper()
        confidence = cot.final_confidence

        # Build explanation
        parts = [f"**{sentiment}** sentiment detected with {confidence:.1%} confidence."]

        if sentiment_step:
            parts.append(f"\n\n**Key Finding:** {sentiment_step.reasoning}")

        # Add context if relevant
        context_step = next((s for s in cot.steps if s.step == ReasoningStep.CONTEXT_ANALYSIS), None)
        if context_step and "high" in context_step.conclusion.lower():
            parts.append(f"\n\n**Context Note:** {context_step.reasoning}")

        return "".join(parts)

    def _extract_key_factors(self, cot: ChainOfThought) -> List[str]:
        """Extract key factors that influenced the decision."""
        factors = []

        for step in cot.steps:
            if step.step == ReasoningStep.SENTIMENT_DETECTION:
                # Extract sentiment words mentioned
                if "positive words" in step.reasoning.lower():
                    match = re.search(r'\[([^\]]+)\]', step.reasoning)
                    if match:
                        factors.append(f"Positive indicators: {match.group(1)}")
                if "negative words" in step.reasoning.lower():
                    match = re.search(r'\[([^\]]+)\]', step.reasoning)
                    if match:
                        factors.append(f"Negative indicators: {match.group(1)}")

            elif step.step == ReasoningStep.ENTITY_EXTRACTION:
                if "percentages" in step.observation.lower():
                    factors.append("Contains quantitative data")

            elif step.step == ReasoningStep.CONTEXT_ANALYSIS:
                if "negation" in step.observation.lower():
                    factors.append("Negation patterns detected")
                if "future" in step.observation.lower():
                    factors.append("Forward-looking language")

        # Add general factor based on confidence
        if cot.final_confidence > 0.85:
            factors.append("High signal clarity")
        elif cot.final_confidence < 0.5:
            factors.append("Ambiguous/mixed signals")

        return factors[:5]  # Top 5 factors


# =============================================================================
# ENHANCED EXPLANATION GENERATOR
# =============================================================================

class EliteExplanationGenerator:
    """
    Elite-level explanation generator combining ML predictions with
    chain-of-thought reasoning for transparent, trustworthy outputs.
    """

    def __init__(self):
        """Initialize the generator."""
        self.reasoner = ChainOfThoughtReasoner()
        if NLP_AVAILABLE:
            self.nlp_analyzer = FinancialTextAnalyzer()
        else:
            self.nlp_analyzer = None

    def generate_comprehensive_explanation(
        self,
        text: str,
        model_prediction: str,
        model_confidence: float,
        probabilities: Dict[str, float],
        word_importance: Optional[List[Tuple[str, float, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation combining all analysis methods.

        Args:
            text: Original financial text
            model_prediction: ML model's prediction
            model_confidence: ML model's confidence
            probabilities: Class probabilities from model
            word_importance: Word importance from explainability module

        Returns:
            Comprehensive explanation dictionary
        """
        # Step 1: Chain-of-thought analysis
        cot = self.reasoner.analyze(
            text=text,
            model_prediction=model_prediction,
            model_confidence=model_confidence,
        )

        # Step 2: NLP feature analysis (if available)
        nlp_analysis = None
        if self.nlp_analyzer:
            nlp_analysis = self.nlp_analyzer.analyze(text)

        # Step 3: Build comprehensive explanation
        explanation = {
            "summary": self._generate_summary(model_prediction, model_confidence, cot),
            "confidence_explanation": self._explain_confidence(model_confidence, cot),
            "reasoning_trace": cot.get_reasoning_trace(),
            "key_factors": cot.key_factors,
            "model_vs_reasoning": {
                "model_prediction": model_prediction,
                "model_confidence": model_confidence,
                "reasoning_prediction": cot.final_sentiment,
                "reasoning_confidence": cot.final_confidence,
                "agreement": model_prediction.lower() == cot.final_sentiment.lower(),
            },
        }

        # Add word importance if available
        if word_importance:
            explanation["influential_words"] = {
                "positive": [(w, s) for w, s, d in word_importance if d == "positive"][:5],
                "negative": [(w, s) for w, s, d in word_importance if d == "negative"][:5],
            }

        # Add NLP features if available
        if nlp_analysis:
            explanation["linguistic_analysis"] = {
                "lexicon_sentiment_score": nlp_analysis["features"]["sentiment_score"],
                "uncertainty_level": nlp_analysis["features"]["uncertainty_score"],
                "positive_words_found": nlp_analysis["features"]["positive_words"],
                "negative_words_found": nlp_analysis["features"]["negative_words"],
                "entities_found": nlp_analysis["entities"],
            }

        return explanation

    def _generate_summary(self, prediction: str, confidence: float, cot: ChainOfThought) -> str:
        """Generate executive summary."""
        agreement = prediction.lower() == cot.final_sentiment.lower()

        if agreement:
            return (
                f"The text is classified as **{prediction.upper()}** with **{confidence:.1%}** confidence. "
                f"This prediction is supported by chain-of-thought analysis, which identified {len(cot.key_factors)} "
                f"key factors confirming this sentiment. "
                f"Primary reasoning: {cot.steps[-1].reasoning if cot.steps else 'N/A'}"
            )
        else:
            return (
                f"The ML model predicts **{prediction.upper()}** ({confidence:.1%}), but detailed analysis suggests "
                f"**{cot.final_sentiment.upper()}** ({cot.final_confidence:.1%}). This disagreement may indicate "
                f"nuanced language or context that affects interpretation. Key factors: {', '.join(cot.key_factors[:3])}"
            )

    def _explain_confidence(self, model_confidence: float, cot: ChainOfThought) -> str:
        """Explain the confidence level."""
        if model_confidence > 0.85:
            return (
                f"High confidence ({model_confidence:.1%}) indicates clear sentiment signals in the text. "
                "The language strongly and consistently points to this sentiment direction."
            )
        elif model_confidence > 0.65:
            return (
                f"Moderate confidence ({model_confidence:.1%}) suggests reasonably clear signals, "
                "though some ambiguity or mixed indicators may be present."
            )
        else:
            return (
                f"Lower confidence ({model_confidence:.1%}) indicates the text contains ambiguous, "
                "mixed, or subtle sentiment signals. Interpretation should be made cautiously."
            )

    def generate_quick_explanation(
        self,
        text: str,
        prediction: str,
        confidence: float,
    ) -> str:
        """Generate a quick, single-paragraph explanation."""
        # Quick chain of thought
        cot = self.reasoner.analyze(text, prediction, confidence)
        return cot.explanation


# =============================================================================
# MARKET OUTLOOK GENERATOR (ENHANCED)
# =============================================================================

def generate_elite_market_outlook(
    texts: List[str],
    predictions: List[str],
    confidences: List[float],
    include_reasoning: bool = True,
) -> Dict[str, Any]:
    """
    Generate elite market outlook with chain-of-thought backing.

    Args:
        texts: List of analyzed texts
        predictions: List of predictions
        confidences: List of confidence scores
        include_reasoning: Whether to include detailed reasoning

    Returns:
        Comprehensive market outlook dictionary
    """
    # Aggregate statistics
    total = len(predictions)
    pos_count = sum(1 for p in predictions if p.lower() == "positive")
    neg_count = sum(1 for p in predictions if p.lower() == "negative")
    neu_count = sum(1 for p in predictions if p.lower() == "neutral")

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Determine trend
    if pos_count > neg_count * 2:
        trend = "strongly_bullish"
        trend_desc = "Strongly Bullish"
        emoji = "🚀"
    elif pos_count > neg_count * 1.5:
        trend = "bullish"
        trend_desc = "Bullish"
        emoji = "📈"
    elif neg_count > pos_count * 2:
        trend = "strongly_bearish"
        trend_desc = "Strongly Bearish"
        emoji = "📉"
    elif neg_count > pos_count * 1.5:
        trend = "bearish"
        trend_desc = "Bearish"
        emoji = "⚠️"
    elif pos_count > neg_count:
        trend = "cautiously_optimistic"
        trend_desc = "Cautiously Optimistic"
        emoji = "📊"
    elif neg_count > pos_count:
        trend = "cautiously_pessimistic"
        trend_desc = "Cautiously Pessimistic"
        emoji = "📉"
    else:
        trend = "neutral"
        trend_desc = "Neutral/Mixed"
        emoji = "↔️"

    # Build report
    report = {
        "trend": trend,
        "trend_description": trend_desc,
        "emoji": emoji,
        "statistics": {
            "total_analyzed": total,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "positive_percentage": pos_count / total * 100 if total else 0,
            "negative_percentage": neg_count / total * 100 if total else 0,
            "neutral_percentage": neu_count / total * 100 if total else 0,
            "average_confidence": avg_confidence,
        },
        "narrative": _generate_outlook_narrative(
            trend_desc, pos_count, neg_count, neu_count, total, avg_confidence
        ),
    }

    # Add sample reasoning if requested
    if include_reasoning and texts:
        reasoner = ChainOfThoughtReasoner()
        # Analyze a sample
        sample_size = min(3, len(texts))
        sample_analyses = []
        for i in range(sample_size):
            cot = reasoner.analyze(texts[i], predictions[i], confidences[i])
            sample_analyses.append({
                "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "prediction": predictions[i],
                "key_factors": cot.key_factors[:3],
            })
        report["sample_analyses"] = sample_analyses

    return report


def _generate_outlook_narrative(
    trend: str,
    pos: int,
    neg: int,
    neu: int,
    total: int,
    confidence: float,
) -> str:
    """Generate narrative for market outlook."""
    pos_pct = pos / total * 100 if total else 0
    neg_pct = neg / total * 100 if total else 0

    narrative = f"## {trend} Market Outlook\n\n"
    narrative += f"Based on comprehensive analysis of **{total} financial texts**, "
    narrative += f"the overall market sentiment appears **{trend.lower()}**.\n\n"

    narrative += "### Key Observations\n\n"

    if pos_pct > 60:
        narrative += f"- **Strong positive sentiment** ({pos_pct:.1f}%) dominates the analyzed content\n"
    elif neg_pct > 60:
        narrative += f"- **Strong negative sentiment** ({neg_pct:.1f}%) dominates the analyzed content\n"
    elif pos_pct > neg_pct + 10:
        narrative += f"- Positive sentiment ({pos_pct:.1f}%) outweighs negative ({neg_pct:.1f}%)\n"
    elif neg_pct > pos_pct + 10:
        narrative += f"- Negative sentiment ({neg_pct:.1f}%) outweighs positive ({pos_pct:.1f}%)\n"
    else:
        narrative += f"- Sentiment is relatively balanced between positive ({pos_pct:.1f}%) and negative ({neg_pct:.1f}%)\n"

    if confidence > 0.8:
        narrative += f"- **High confidence** ({confidence:.1%}) in classifications suggests clear sentiment signals\n"
    elif confidence < 0.6:
        narrative += f"- **Lower confidence** ({confidence:.1%}) indicates some ambiguity in the analyzed texts\n"

    if pos > 0 and neg > 0:
        ratio = pos / neg
        narrative += f"- Positive-to-negative ratio: **{ratio:.2f}:1**\n"

    narrative += "\n### Investment Implications\n\n"

    if "bullish" in trend.lower():
        narrative += "The predominantly positive sentiment suggests favorable market perception. "
        narrative += "This could indicate growth expectations, strong fundamentals, or positive catalysts. "
        narrative += "However, investors should conduct independent due diligence before making decisions.\n"
    elif "bearish" in trend.lower():
        narrative += "The predominantly negative sentiment signals market concerns or challenges. "
        narrative += "This may reflect headwinds, disappointing results, or risk factors. "
        narrative += "Careful risk assessment is recommended before any investment actions.\n"
    else:
        narrative += "Mixed or neutral sentiment suggests a transitional or uncertain period. "
        narrative += "Monitor for emerging trends that could shift sentiment decisively. "
        narrative += "This may be a period of consolidation before a clearer direction emerges.\n"

    narrative += "\n*Note: This analysis is AI-generated and should be used as one input among many in investment decision-making.*"

    return narrative


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

def generate_explanation_template(
    text: str,
    prediction: str,
    confidence: float,
    positive_words: List[tuple],
    negative_words: List[tuple],
    neutral_words: List[tuple],
) -> str:
    """
    Backward-compatible explanation generator.
    Enhanced with chain-of-thought underneath.
    """
    generator = EliteExplanationGenerator()
    return generator.generate_quick_explanation(text, prediction, confidence)


def generate_market_outlook(
    sentiment_counts: Dict[str, int],
    total_texts: int,
    avg_confidence: float,
    sample_texts: Optional[List[Dict]] = None,
) -> str:
    """Backward-compatible market outlook generator."""
    # Convert to new format
    predictions = []
    for sentiment, count in sentiment_counts.items():
        predictions.extend([sentiment] * count)

    confidences = [avg_confidence] * total_texts
    texts = ["Sample text"] * total_texts

    report = generate_elite_market_outlook(texts, predictions, confidences, include_reasoning=False)
    return report["narrative"]


def get_llm_explanation(
    text: str,
    prediction: str,
    probabilities: Dict[str, float],
    word_importance: List[tuple],
    use_api: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """Backward-compatible LLM explanation."""
    confidence = probabilities.get(prediction, 0.5) if probabilities else 0.5
    generator = EliteExplanationGenerator()
    return generator.generate_quick_explanation(text, prediction, confidence)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ELITE CHAIN-OF-THOUGHT FINANCIAL SENTIMENT ANALYSIS")
    print("=" * 80)

    test_texts = [
        "The company reported record-breaking earnings of $2.5 billion, exceeding analyst expectations by 15%.",
        "Revenue declined 20% year-over-year as the company struggles with supply chain disruptions.",
        "The quarterly results were in line with expectations, with no significant surprises.",
    ]

    reasoner = ChainOfThoughtReasoner()

    for text in test_texts:
        print(f"\n{'='*80}")
        print(f"TEXT: {text[:70]}...")
        print("=" * 80)

        cot = reasoner.analyze(text)

        print(f"\n📊 FINAL VERDICT: {cot.final_sentiment.upper()} ({cot.final_confidence:.1%})")
        print(f"\n📝 EXPLANATION:\n{cot.explanation}")
        print(f"\n🔑 KEY FACTORS: {', '.join(cot.key_factors)}")
        print(f"\n🔍 REASONING TRACE:\n{cot.get_reasoning_trace()}")

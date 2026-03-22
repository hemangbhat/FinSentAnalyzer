"""
Advanced NLP Processing Module for Financial Sentiment Analysis.
Elite-level text preprocessing, feature extraction, and linguistic analysis.
"""

import re
import string
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import numpy as np


# =============================================================================
# FINANCIAL SENTIMENT LEXICONS (Loughran-McDonald Dictionary)
# =============================================================================

# Core financial sentiment words from Loughran-McDonald (2011) dictionary
# Reference: https://sraf.nd.edu/loughranmcdonald-master-dictionary/

FINANCIAL_POSITIVE = {
    # Strong positive
    "achieve", "achieved", "achieves", "achieving", "accomplishment", "accomplishments",
    "advantage", "advantages", "breakthrough", "breakthroughs", "benefit", "benefits",
    "beneficial", "boom", "booming", "bullish", "confident", "confidence",
    "creative", "creativity", "deliver", "delivered", "delivers", "delivering",
    "earnings", "efficient", "efficiency", "enable", "enabled", "enables",
    "enhance", "enhanced", "enhances", "enhancing", "excellent", "exceptional",
    "excite", "excited", "exciting", "exclusive", "expansion", "favorable",
    "gain", "gained", "gaining", "gains", "good", "great", "greater", "greatest",
    "grow", "growing", "grown", "grows", "growth", "highest", "improve",
    "improved", "improvement", "improvements", "improves", "improving", "increase",
    "increased", "increases", "increasing", "innovation", "innovations", "innovative",
    "leading", "leader", "leadership", "leverage", "leveraged", "leveraging",
    "momentum", "opportunities", "opportunity", "optimal", "optimism", "optimistic",
    "outperform", "outperformed", "outperforming", "outperforms", "outstanding",
    "positive", "positively", "premium", "profitable", "profitability", "profit",
    "profits", "progress", "progressed", "progresses", "progressing", "promising",
    "prosper", "prospered", "prospering", "prosperity", "prosperous", "rally",
    "rallied", "rallies", "rallying", "recover", "recovered", "recovering",
    "recovers", "recovery", "record", "rebound", "rebounded", "rebounding",
    "reward", "rewarded", "rewarding", "rewards", "rise", "risen", "rises",
    "rising", "robust", "solid", "soar", "soared", "soaring", "soars",
    "stabilize", "stabilized", "stabilizes", "stabilizing", "stable", "stellar",
    "strength", "strengthen", "strengthened", "strengthening", "strengthens",
    "strong", "stronger", "strongest", "succeed", "succeeded", "succeeding",
    "succeeds", "success", "successes", "successful", "successfully", "superior",
    "surge", "surged", "surges", "surging", "surpass", "surpassed", "surpasses",
    "surpassing", "sustainable", "thriving", "top", "transform", "transformed",
    "transforming", "transforms", "turnaround", "upbeat", "upgrade", "upgraded",
    "upgrades", "upgrading", "upturn", "upward", "upwards", "win", "winner",
    "winning", "wins", "won",
}

FINANCIAL_NEGATIVE = {
    # Strong negative
    "abandon", "abandoned", "abandoning", "abandons", "abnormal", "abuse",
    "accident", "accidents", "accuse", "accused", "accuses", "accusing",
    "adverse", "adversely", "against", "allegation", "allegations", "allege",
    "alleged", "allegedly", "alleges", "alleging", "annul", "annulled",
    "annulling", "annuls", "anomalous", "anomaly", "anxious", "anxiety",
    "attack", "attacked", "attacking", "attacks", "bad", "bail", "bailout",
    "bankrupt", "bankruptcies", "bankruptcy", "bear", "bearish", "blame",
    "blamed", "blames", "blaming", "breach", "breached", "breaches", "breaching",
    "burden", "burdened", "burdening", "burdens", "catastrophe", "catastrophic",
    "caution", "cautionary", "cautious", "cautiously", "challenge", "challenged",
    "challenges", "challenging", "chaos", "chaotic", "claim", "claims",
    "closure", "closures", "collapse", "collapsed", "collapses", "collapsing",
    "compete", "competes", "competing", "competition", "competitive", "complaint",
    "complaints", "concern", "concerned", "concerning", "concerns", "conflict",
    "conflicts", "constrain", "constrained", "constraining", "constrains",
    "constraint", "constraints", "contraction", "crash", "crashed", "crashes",
    "crashing", "crime", "criminal", "crisis", "critical", "criticism",
    "criticisms", "criticize", "criticized", "criticizes", "criticizing",
    "crucial", "cut", "cutback", "cutbacks", "cuts", "cutting", "damage",
    "damaged", "damages", "damaging", "danger", "dangerous", "dangerously",
    "dangers", "deadlock", "deadlocked", "deadlocking", "deadlocks", "death",
    "deaths", "debt", "debts", "deceive", "deceived", "deceives", "deceiving",
    "decline", "declined", "declines", "declining", "decrease", "decreased",
    "decreases", "decreasing", "default", "defaulted", "defaulting", "defaults",
    "defect", "defective", "defects", "deficit", "deficits", "delay", "delayed",
    "delaying", "delays", "demise", "demolish", "demolished", "demolishes",
    "demolishing", "deny", "denied", "denies", "denying", "deplete", "depleted",
    "depletes", "depleting", "depletion", "depreciate", "depreciated",
    "depreciates", "depreciating", "depreciation", "depressed", "depression",
    "destabilize", "destabilized", "destabilizes", "destabilizing", "destroy",
    "destroyed", "destroying", "destroys", "destruction", "destructive",
    "deteriorate", "deteriorated", "deteriorates", "deteriorating", "deterioration",
    "devastating", "devastation", "difficult", "difficulties", "difficulty",
    "dilute", "diluted", "dilutes", "diluting", "dilution", "diminish",
    "diminished", "diminishes", "diminishing", "dire", "disappoint", "disappointed",
    "disappointing", "disappointingly", "disappointment", "disappointments",
    "disappoints", "disaster", "disasters", "disastrous", "disclaim", "disclaimer",
    "disclaimers", "disclaims", "disclose", "disclosed", "discloses", "disclosing",
    "disclosure", "disclosures", "discontinue", "discontinued", "discontinues",
    "discontinuing", "discourage", "discouraged", "discourages", "discouraging",
    "dispute", "disputed", "disputes", "disputing", "disrupt", "disrupted",
    "disrupting", "disruption", "disruptions", "disruptive", "disrupts",
    "distress", "distressed", "distresses", "distressing", "divest", "divested",
    "divesting", "divestiture", "divestitures", "divests", "doubt", "doubted",
    "doubtful", "doubting", "doubts", "down", "downgrade", "downgraded",
    "downgrades", "downgrading", "downturn", "downturns", "downward", "downwards",
    "drag", "dragged", "dragging", "drags", "drain", "drained", "draining",
    "drains", "drop", "dropped", "dropping", "drops", "drought", "droughts",
    "dull", "dump", "dumped", "dumping", "dumps", "erode", "eroded", "erodes",
    "eroding", "erosion", "error", "errors", "evade", "evaded", "evades",
    "evading", "evasion", "exacerbate", "exacerbated", "exacerbates", "exacerbating",
    "excessive", "excessively", "exit", "exited", "exiting", "exits", "expire",
    "expired", "expires", "expiring", "expose", "exposed", "exposes", "exposing",
    "exposure", "exposures", "fail", "failed", "failing", "fails", "failure",
    "failures", "fall", "fallen", "falling", "falls", "false", "falsely",
    "falsification", "falsifications", "falsified", "falsifies", "falsify",
    "falsifying", "fatal", "fatalities", "fatality", "fault", "faulted",
    "faulting", "faults", "faulty", "fear", "feared", "fearful", "fearing",
    "fears", "felonies", "felony", "fine", "fined", "fines", "fining", "fire",
    "fired", "fires", "firing", "flaw", "flawed", "flaws", "flee", "fleeing",
    "flees", "fled", "fluctuate", "fluctuated", "fluctuates", "fluctuating",
    "fluctuation", "fluctuations", "fraud", "frauds", "fraudulent", "fraudulently",
    "grim", "halt", "halted", "halting", "halts", "hamper", "hampered",
    "hampering", "hampers", "harm", "harmed", "harmful", "harmfully", "harming",
    "harms", "harsh", "harshly", "hazard", "hazardous", "hazards", "hinder",
    "hindered", "hindering", "hinders", "hurt", "hurting", "hurts", "idle",
    "idled", "idling", "ignore", "ignored", "ignores", "ignoring", "illegal",
    "illegally", "impair", "impaired", "impairing", "impairment", "impairments",
    "impairs", "impediment", "impediments", "implode", "imploded", "implodes",
    "imploding", "implosion", "inability", "inadequate", "inadequately",
    "insolvent", "instability", "insufficient", "insufficiently", "interrupt",
    "interrupted", "interrupting", "interruption", "interruptions", "interrupts",
    "investigation", "investigations", "issue", "issues", "jeopardize",
    "jeopardized", "jeopardizes", "jeopardizing", "jeopardy", "lack", "lacked",
    "lacking", "lacks", "lag", "lagged", "lagging", "lags", "late", "later",
    "latest", "lawsuit", "lawsuits", "layoff", "layoffs", "liability",
    "liabilities", "liquidate", "liquidated", "liquidates", "liquidating",
    "liquidation", "liquidations", "litigate", "litigated", "litigates",
    "litigating", "litigation", "litigations", "lose", "losing", "loss",
    "losses", "lost", "low", "lower", "lowered", "lowering", "lowers", "lowest",
    "manipulation", "manipulations", "meltdown", "miss", "missed", "misses",
    "missing", "mistake", "mistaken", "mistakes", "mitigate", "mitigated",
    "mitigates", "mitigating", "mitigation", "negative", "negatively", "neglect",
    "neglected", "neglecting", "neglects", "obstacle", "obstacles", "obsolete",
    "ominous", "overdue", "penalty", "penalties", "peril", "perils", "perilous",
    "pessimism", "pessimistic", "plummet", "plummeted", "plummeting", "plummets",
    "plunge", "plunged", "plunges", "plunging", "poor", "poorer", "poorest",
    "poorly", "pressure", "pressured", "pressures", "pressuring", "problem",
    "problematic", "problems", "pullback", "questionable", "recall", "recalled",
    "recalling", "recalls", "recession", "recessionary", "recessions", "reduce",
    "reduced", "reduces", "reducing", "reduction", "reductions", "reject",
    "rejected", "rejecting", "rejection", "rejections", "rejects", "restructure",
    "restructured", "restructures", "restructuring", "restructurings", "retrench",
    "retrenched", "retrenches", "retrenching", "retrenchment", "retrenchments",
    "risk", "risked", "riskier", "riskiest", "risking", "risks", "risky",
    "sabotage", "sabotaged", "sabotages", "sabotaging", "sacrifice", "sacrificed",
    "sacrifices", "sacrificing", "scandal", "scandals", "scrutinize", "scrutinized",
    "scrutinizes", "scrutinizing", "scrutiny", "sell", "selloff", "selloffs",
    "selling", "sells", "serious", "seriously", "setback", "setbacks", "severe",
    "severely", "severity", "shock", "shocked", "shocking", "shocks", "short",
    "shortage", "shortages", "shortfall", "shortfalls", "shrink", "shrinkage",
    "shrinking", "shrinks", "shrunk", "shut", "shutdown", "shutdowns", "shuts",
    "shutting", "sink", "sinking", "sinks", "slid", "slide", "slides", "sliding",
    "slip", "slipped", "slipping", "slips", "slow", "slowdown", "slowdowns",
    "slowed", "slower", "slowest", "slowing", "slowly", "slows", "sluggish",
    "slump", "slumped", "slumping", "slumps", "stall", "stalled", "stalling",
    "stalls", "stagnant", "stagnate", "stagnated", "stagnates", "stagnating",
    "stagnation", "strain", "strained", "straining", "strains", "stress",
    "stressed", "stresses", "stressful", "stressing", "strike", "strikes",
    "striking", "struck", "struggle", "struggled", "struggles", "struggling",
    "subpoena", "subpoenaed", "subpoenaing", "subpoenas", "sue", "sued", "sues",
    "suffer", "suffered", "suffering", "suffers", "suing", "susceptible",
    "suspend", "suspended", "suspending", "suspends", "suspension", "suspensions",
    "target", "targeted", "targeting", "targets", "terminate", "terminated",
    "terminates", "terminating", "termination", "terminations", "terrible",
    "terribly", "threat", "threaten", "threatened", "threatening", "threatens",
    "threats", "tough", "tougher", "toughest", "trouble", "troubled", "troubles",
    "troublesome", "troubling", "tumble", "tumbled", "tumbles", "tumbling",
    "turmoil", "unable", "uncertain", "uncertainly", "uncertainties", "uncertainty",
    "undermine", "undermined", "undermines", "undermining", "underperform",
    "underperformed", "underperforming", "underperforms", "unfavorable",
    "unfavorably", "unfortunate", "unfortunately", "unlawful", "unlawfully",
    "unprofitable", "unreliable", "unsuccessful", "unsuccessfully", "unstable",
    "untimely", "urgent", "urgently", "violate", "violated", "violates",
    "violating", "violation", "violations", "volatile", "volatility", "vulnerable",
    "warn", "warned", "warning", "warnings", "warns", "weak", "weaken",
    "weakened", "weakening", "weakens", "weaker", "weakest", "weakness",
    "weaknesses", "woes", "worrisome", "worry", "worrying", "worse", "worsen",
    "worsened", "worsening", "worsens", "worst", "worthless", "writedown",
    "writedowns", "writeoff", "writeoffs",
}

FINANCIAL_UNCERTAINTY = {
    "anticipate", "anticipated", "anticipates", "anticipating", "apparent",
    "apparently", "appear", "appeared", "appearing", "appears", "approximate",
    "approximately", "assume", "assumed", "assumes", "assuming", "assumption",
    "assumptions", "believe", "believed", "believes", "believing", "conceivable",
    "conceivably", "conditional", "conditionally", "contingencies", "contingency",
    "contingent", "contingently", "could", "depend", "depended", "depending",
    "depends", "doubt", "doubted", "doubtful", "doubting", "doubts", "estimate",
    "estimated", "estimates", "estimating", "estimation", "estimations",
    "eventual", "eventually", "expect", "expectation", "expectations", "expected",
    "expecting", "expects", "expose", "exposed", "exposes", "exposing",
    "exposure", "exposures", "fluctuate", "fluctuated", "fluctuates",
    "fluctuating", "fluctuation", "fluctuations", "forecast", "forecasted",
    "forecasting", "forecasts", "foreseeable", "foreseen", "hypotheses",
    "hypothetical", "hypothetically", "if", "imprecise", "imprecision",
    "indefinite", "indefinitely", "indeterminable", "indeterminate", "indicate",
    "indicated", "indicates", "indicating", "indication", "indications",
    "indicator", "indicators", "inexact", "intend", "intended", "intending",
    "intends", "intention", "intentions", "likelihood", "likely", "may",
    "maybe", "might", "nearly", "occasionally", "opinion", "opinions", "outlook",
    "pending", "perhaps", "plan", "planned", "planning", "plans", "plausible",
    "plausibly", "possible", "possibly", "potential", "potentially", "predict",
    "predictability", "predicted", "predicting", "prediction", "predictions",
    "predictive", "predicts", "preliminary", "presumably", "presume", "presumed",
    "presumes", "presuming", "presumption", "presumptions", "probabilistic",
    "probabilities", "probability", "probable", "probably", "project",
    "projected", "projecting", "projection", "projections", "projects", "random",
    "randomly", "reestimate", "reestimated", "reestimates", "reestimating",
    "roughly", "seem", "seemed", "seeming", "seemingly", "seems", "should",
    "sometimes", "somewhat", "somewhere", "speculate", "speculated", "speculates",
    "speculating", "speculation", "speculations", "speculative", "speculatively",
    "suggest", "suggested", "suggesting", "suggestion", "suggestions", "suggests",
    "suppose", "supposed", "supposedly", "supposes", "supposing", "tend",
    "tended", "tendency", "tendencies", "tending", "tends", "tentative",
    "tentatively", "uncertain", "uncertainly", "uncertainties", "uncertainty",
    "unclear", "unconfirmed", "undecided", "undefined", "undetermined",
    "unexpected", "unexpectedly", "unforeseen", "unknown", "unknowns",
    "unpredictable", "unpredictably", "unproven", "unquantifiable", "unquantified",
    "unreliable", "unresolved", "unsettled", "unspecified", "unusual",
    "unusually", "variable", "variably", "variance", "variances", "variant",
    "variants", "variation", "variations", "varied", "varies", "vary", "varying",
    "volatile", "volatility",
}

FINANCIAL_LITIGIOUS = {
    "adjudicate", "adjudicated", "adjudicates", "adjudicating", "adjudication",
    "allege", "alleged", "allegedly", "alleges", "alleging", "allegation",
    "allegations", "appeal", "appealed", "appealing", "appeals", "arbitrate",
    "arbitrated", "arbitrates", "arbitrating", "arbitration", "arbitrations",
    "arbitrator", "arbitrators", "claim", "claimant", "claimants", "claimed",
    "claiming", "claims", "complaint", "complaints", "confiscate", "confiscated",
    "confiscates", "confiscating", "confiscation", "convict", "convicted",
    "convicting", "conviction", "convictions", "convicts", "counsel", "countersue",
    "countersued", "countersues", "countersuing", "court", "courts", "crime",
    "crimes", "criminal", "criminally", "criminals", "damages", "decree",
    "decrees", "defend", "defendant", "defendants", "defended", "defending",
    "defends", "depose", "deposed", "deposes", "deposing", "deposition",
    "depositions", "discovery", "enjoin", "enjoined", "enjoining", "enjoins",
    "felonies", "felony", "fraud", "frauds", "fraudulent", "fraudulently",
    "grievance", "grievances", "guilt", "guilty", "hearing", "hearings",
    "illegal", "illegally", "implead", "impleaded", "impleader", "impleading",
    "impleads", "indict", "indicted", "indicting", "indictment", "indictments",
    "indicts", "infringement", "infringements", "injunction", "injunctions",
    "injunctive", "inquiry", "interlocutory", "interrogation", "interrogations",
    "judge", "judges", "judicial", "judiciary", "jurisdiction", "jurisdictional",
    "jurisdictions", "juror", "jurors", "jury", "law", "lawful", "lawfully",
    "laws", "lawsuit", "lawsuits", "lawyer", "lawyers", "legal", "legally",
    "liabilities", "liability", "libel", "libelous", "libels", "litigant",
    "litigants", "litigate", "litigated", "litigates", "litigating", "litigation",
    "litigations", "mistrial", "mistrials", "motion", "motions", "offence",
    "offences", "offend", "offended", "offender", "offenders", "offending",
    "offends", "offense", "offenses", "order", "ordered", "ordering", "orders",
    "perjure", "perjured", "perjures", "perjuries", "perjuring", "perjury",
    "plaintiff", "plaintiffs", "plea", "plead", "pleaded", "pleading", "pleadings",
    "pleads", "pleas", "pled", "prejudice", "prejudiced", "prejudices",
    "prejudicing", "prejudicial", "prison", "prisoned", "prisoner", "prisoners",
    "prisoning", "prisons", "probation", "probationary", "probations", "prosecute",
    "prosecuted", "prosecutes", "prosecuting", "prosecution", "prosecutions",
    "prosecutor", "prosecutors", "punish", "punished", "punishes", "punishing",
    "punishment", "punishments", "punitive", "remand", "remanded", "remanding",
    "remands", "remedy", "remedies", "restrain", "restrained", "restraining",
    "restrains", "restraint", "restraints", "rule", "ruled", "rules", "ruling",
    "rulings", "sanction", "sanctioned", "sanctioning", "sanctions", "sentence",
    "sentenced", "sentences", "sentencing", "settle", "settled", "settlement",
    "settlements", "settles", "settling", "severance", "slander", "slanderous",
    "slanders", "statute", "statutes", "statutory", "subpoena", "subpoenaed",
    "subpoenaing", "subpoenas", "sue", "sued", "sues", "suing", "suit", "suits",
    "summon", "summoned", "summoning", "summons", "testify", "testified",
    "testifies", "testifying", "testimonial", "testimonies", "testimony",
    "tort", "tortious", "torts", "trial", "trials", "tribunal", "tribunals",
    "verdict", "verdicts", "violate", "violated", "violates", "violating",
    "violation", "violations", "violator", "violators", "witness", "witnessed",
    "witnesses", "witnessing",
}


# =============================================================================
# DATA CLASSES FOR STRUCTURED OUTPUT
# =============================================================================

@dataclass
class TextFeatures:
    """Structured features extracted from financial text."""
    # Basic stats
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0

    # Lexicon-based sentiment
    positive_word_count: int = 0
    negative_word_count: int = 0
    uncertainty_word_count: int = 0
    litigious_word_count: int = 0
    positive_words: List[str] = field(default_factory=list)
    negative_words: List[str] = field(default_factory=list)
    uncertainty_words: List[str] = field(default_factory=list)

    # Financial entities
    percentages: List[str] = field(default_factory=list)
    currencies: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)

    # Derived scores
    sentiment_score: float = 0.0  # -1 to 1
    uncertainty_score: float = 0.0  # 0 to 1
    subjectivity_score: float = 0.0  # 0 to 1

    # Linguistic features
    has_negation: bool = False
    negation_count: int = 0
    exclamation_count: int = 0
    question_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "word_count": self.word_count,
            "char_count": self.char_count,
            "sentence_count": self.sentence_count,
            "avg_word_length": round(self.avg_word_length, 2),
            "positive_word_count": self.positive_word_count,
            "negative_word_count": self.negative_word_count,
            "uncertainty_word_count": self.uncertainty_word_count,
            "litigious_word_count": self.litigious_word_count,
            "positive_words": self.positive_words[:10],  # Top 10
            "negative_words": self.negative_words[:10],
            "uncertainty_words": self.uncertainty_words[:10],
            "percentages": self.percentages,
            "currencies": self.currencies,
            "numbers": self.numbers[:10],
            "companies": self.companies[:10],
            "sentiment_score": round(self.sentiment_score, 3),
            "uncertainty_score": round(self.uncertainty_score, 3),
            "subjectivity_score": round(self.subjectivity_score, 3),
            "has_negation": self.has_negation,
            "negation_count": self.negation_count,
        }


@dataclass
class ProcessedText:
    """Result of advanced text processing."""
    original: str
    cleaned: str
    normalized: str
    tokens: List[str]
    features: TextFeatures

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original,
            "cleaned": self.cleaned,
            "normalized": self.normalized,
            "token_count": len(self.tokens),
            "features": self.features.to_dict(),
        }


# =============================================================================
# ADVANCED TEXT PREPROCESSING
# =============================================================================

class AdvancedTextProcessor:
    """
    Elite-level text preprocessing for financial sentiment analysis.

    Features:
    - Financial-specific normalization (currencies, percentages, numbers)
    - Sentiment lexicon scoring (Loughran-McDonald)
    - Entity extraction (companies, amounts)
    - Negation handling
    - Linguistic feature extraction
    """

    # Negation words that flip sentiment
    NEGATION_WORDS = {
        "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
        "none", "nor", "cannot", "can't", "won't", "wouldn't", "shouldn't",
        "couldn't", "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't",
        "weren't", "hasn't", "haven't", "hadn't", "without", "lack", "lacking",
        "fail", "failed", "fails", "unable", "unlikely",
    }

    # Intensifiers that amplify sentiment
    INTENSIFIERS = {
        "very": 1.5, "extremely": 2.0, "highly": 1.5, "significantly": 1.5,
        "substantially": 1.5, "considerably": 1.5, "remarkably": 1.5,
        "exceptionally": 1.8, "tremendously": 1.8, "drastically": 1.8,
        "sharply": 1.5, "strongly": 1.5, "deeply": 1.5, "greatly": 1.5,
        "massively": 1.8, "hugely": 1.8, "overwhelmingly": 2.0,
        "slightly": 0.5, "somewhat": 0.7, "marginally": 0.5, "modestly": 0.7,
        "mildly": 0.5, "fairly": 0.8, "relatively": 0.8,
    }

    # Financial term patterns
    PATTERNS = {
        "percentage": r"[-+]?\d+(?:\.\d+)?%",
        "currency": r"(?:USD|EUR|GBP|JPY|CNY|INR|\$|€|£|¥|₹)\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|[MBTmbt]))?",
        "number": r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|[MBTmbt]))?",
        "company": r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|Group|Holdings|Co|Company|Industries|Technologies|Systems|Solutions|Services)\.?)|(?:[A-Z]{2,5})",
        "fiscal_term": r"(?:Q[1-4]|FY\d{2,4}|H[12]|fiscal\s+(?:year|quarter))",
        "date": r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?|\d{1,2}/\d{1,2}/\d{2,4}",
    }

    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 normalize_numbers: bool = True,
                 handle_negations: bool = True):
        """
        Initialize the processor.

        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            normalize_numbers: Normalize numeric expressions
            handle_negations: Apply negation handling
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_numbers = normalize_numbers
        self.handle_negations = handle_negations

        # Compile regex patterns
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }

    def process(self, text: str) -> ProcessedText:
        """
        Process text with full feature extraction.

        Args:
            text: Raw input text

        Returns:
            ProcessedText with all features
        """
        # Step 1: Clean text
        cleaned = self._clean_text(text)

        # Step 2: Normalize text
        normalized = self._normalize_text(cleaned)

        # Step 3: Tokenize
        tokens = self._tokenize(normalized)

        # Step 4: Extract features
        features = self._extract_features(text, cleaned, tokens)

        return ProcessedText(
            original=text,
            cleaned=cleaned,
            normalized=normalized,
            tokens=tokens,
            features=features,
        )

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€"', "-")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)

        return text.strip()

    def _normalize_text(self, text: str) -> str:
        """Normalize text with financial-specific rules."""
        normalized = text

        # Normalize percentages
        if self.normalize_numbers:
            # Replace percentage with placeholder
            normalized = re.sub(r'[-+]?\d+(?:\.\d+)?%', ' _PERCENT_ ', normalized)

            # Normalize large numbers
            normalized = re.sub(
                r'\$?\d+(?:\.\d+)?\s*(?:billion|bn)\b',
                ' _BILLION_DOLLARS_ ',
                normalized, flags=re.IGNORECASE
            )
            normalized = re.sub(
                r'\$?\d+(?:\.\d+)?\s*(?:million|mn|m)\b',
                ' _MILLION_DOLLARS_ ',
                normalized, flags=re.IGNORECASE
            )
            normalized = re.sub(
                r'\$?\d+(?:\.\d+)?\s*(?:trillion|tn|tr)\b',
                ' _TRILLION_DOLLARS_ ',
                normalized, flags=re.IGNORECASE
            )

            # Normalize currency amounts
            normalized = re.sub(r'\$\d+(?:,\d{3})*(?:\.\d+)?', ' _CURRENCY_ ', normalized)

        # Apply lowercasing if enabled
        if self.lowercase:
            normalized = normalized.lower()

        # Remove punctuation if enabled
        if self.remove_punctuation:
            normalized = normalized.translate(str.maketrans('', '', string.punctuation))

        # Clean up whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple whitespace tokenization (can be enhanced with spaCy)
        tokens = text.split()

        # Remove empty tokens
        tokens = [t for t in tokens if t.strip()]

        return tokens

    def _extract_features(self, original: str, cleaned: str, tokens: List[str]) -> TextFeatures:
        """Extract comprehensive features from text."""
        features = TextFeatures()

        # Basic stats
        features.word_count = len(tokens)
        features.char_count = len(cleaned)
        features.sentence_count = max(1, len(re.findall(r'[.!?]+', cleaned)))

        if features.word_count > 0:
            features.avg_word_length = sum(len(t) for t in tokens) / features.word_count

        # Extract entities
        features.percentages = self.compiled_patterns["percentage"].findall(original)
        features.currencies = self.compiled_patterns["currency"].findall(original)
        features.numbers = self.compiled_patterns["number"].findall(original)
        features.companies = self.compiled_patterns["company"].findall(original)

        # Lexicon-based analysis
        words_lower = [t.lower() for t in tokens]

        for word in words_lower:
            if word in FINANCIAL_POSITIVE:
                features.positive_word_count += 1
                features.positive_words.append(word)
            if word in FINANCIAL_NEGATIVE:
                features.negative_word_count += 1
                features.negative_words.append(word)
            if word in FINANCIAL_UNCERTAINTY:
                features.uncertainty_word_count += 1
                features.uncertainty_words.append(word)
            if word in FINANCIAL_LITIGIOUS:
                features.litigious_word_count += 1

        # Calculate sentiment score (-1 to 1)
        total_sentiment_words = features.positive_word_count + features.negative_word_count
        if total_sentiment_words > 0:
            features.sentiment_score = (
                (features.positive_word_count - features.negative_word_count) /
                total_sentiment_words
            )

        # Calculate uncertainty score (0 to 1)
        if features.word_count > 0:
            features.uncertainty_score = min(1.0, features.uncertainty_word_count / features.word_count * 5)

        # Calculate subjectivity score
        opinion_words = features.positive_word_count + features.negative_word_count + features.uncertainty_word_count
        if features.word_count > 0:
            features.subjectivity_score = min(1.0, opinion_words / features.word_count * 3)

        # Negation handling
        features.negation_count = sum(1 for w in words_lower if w in self.NEGATION_WORDS)
        features.has_negation = features.negation_count > 0

        # Punctuation counts
        features.exclamation_count = original.count('!')
        features.question_count = original.count('?')

        # Adjust sentiment for negations
        if self.handle_negations and features.has_negation:
            # Flip sentiment direction if negation present
            features.sentiment_score *= -0.5  # Dampen but flip

        return features

    def get_sentiment_explanation(self, features: TextFeatures) -> str:
        """Generate human-readable explanation of sentiment features."""
        explanations = []

        # Sentiment direction
        if features.sentiment_score > 0.3:
            explanations.append(f"Strong positive sentiment (score: {features.sentiment_score:.2f})")
        elif features.sentiment_score > 0:
            explanations.append(f"Mildly positive sentiment (score: {features.sentiment_score:.2f})")
        elif features.sentiment_score < -0.3:
            explanations.append(f"Strong negative sentiment (score: {features.sentiment_score:.2f})")
        elif features.sentiment_score < 0:
            explanations.append(f"Mildly negative sentiment (score: {features.sentiment_score:.2f})")
        else:
            explanations.append("Neutral sentiment")

        # Key words
        if features.positive_words:
            top_pos = features.positive_words[:3]
            explanations.append(f"Positive indicators: {', '.join(top_pos)}")

        if features.negative_words:
            top_neg = features.negative_words[:3]
            explanations.append(f"Negative indicators: {', '.join(top_neg)}")

        # Uncertainty
        if features.uncertainty_score > 0.3:
            explanations.append(f"High uncertainty language detected")
            if features.uncertainty_words:
                explanations.append(f"Uncertainty words: {', '.join(features.uncertainty_words[:3])}")

        # Negation
        if features.has_negation:
            explanations.append(f"Contains {features.negation_count} negation(s) - may flip sentiment")

        # Financial entities
        if features.percentages:
            explanations.append(f"Contains percentages: {', '.join(features.percentages[:3])}")

        if features.currencies:
            explanations.append(f"Contains currency amounts: {', '.join(features.currencies[:3])}")

        return " | ".join(explanations)


# =============================================================================
# FINANCIAL TEXT ANALYZER (HIGH-LEVEL API)
# =============================================================================

class FinancialTextAnalyzer:
    """
    High-level API for comprehensive financial text analysis.

    Combines:
    - Advanced preprocessing
    - Lexicon-based sentiment
    - Entity extraction
    - Linguistic analysis
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.processor = AdvancedTextProcessor(
            lowercase=True,
            remove_punctuation=False,
            normalize_numbers=True,
            handle_negations=True,
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on financial text.

        Args:
            text: Input financial text

        Returns:
            Dictionary with all analysis results
        """
        # Process text
        processed = self.processor.process(text)

        # Get sentiment explanation
        explanation = self.processor.get_sentiment_explanation(processed.features)

        # Determine overall sentiment category
        score = processed.features.sentiment_score
        if score > 0.2:
            sentiment_category = "positive"
        elif score < -0.2:
            sentiment_category = "negative"
        else:
            sentiment_category = "neutral"

        # Calculate confidence based on feature strength
        confidence = self._calculate_confidence(processed.features)

        return {
            "text": text,
            "cleaned_text": processed.cleaned,
            "sentiment": {
                "category": sentiment_category,
                "score": processed.features.sentiment_score,
                "confidence": confidence,
                "explanation": explanation,
            },
            "features": processed.features.to_dict(),
            "entities": {
                "percentages": processed.features.percentages,
                "currencies": processed.features.currencies,
                "companies": processed.features.companies,
            },
            "linguistic": {
                "word_count": processed.features.word_count,
                "sentence_count": processed.features.sentence_count,
                "avg_word_length": processed.features.avg_word_length,
                "has_negation": processed.features.has_negation,
                "uncertainty_score": processed.features.uncertainty_score,
                "subjectivity_score": processed.features.subjectivity_score,
            },
        }

    def _calculate_confidence(self, features: TextFeatures) -> float:
        """Calculate confidence score based on features."""
        # More sentiment words = higher confidence
        sentiment_strength = abs(features.sentiment_score)

        # Less uncertainty = higher confidence
        uncertainty_penalty = features.uncertainty_score * 0.3

        # Negations reduce confidence
        negation_penalty = min(0.2, features.negation_count * 0.1)

        # Base confidence
        confidence = 0.5 + (sentiment_strength * 0.3) - uncertainty_penalty - negation_penalty

        return max(0.1, min(1.0, confidence))

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]

    def get_aggregate_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Get aggregate sentiment for multiple texts."""
        analyses = self.batch_analyze(texts)

        scores = [a["sentiment"]["score"] for a in analyses]
        categories = [a["sentiment"]["category"] for a in analyses]

        # Count categories
        category_counts = Counter(categories)

        # Calculate aggregate
        avg_score = np.mean(scores)
        std_score = np.std(scores)

        # Determine overall trend
        if category_counts["positive"] > category_counts["negative"] * 1.5:
            trend = "bullish"
        elif category_counts["negative"] > category_counts["positive"] * 1.5:
            trend = "bearish"
        else:
            trend = "mixed"

        return {
            "total_texts": len(texts),
            "average_sentiment_score": round(avg_score, 3),
            "sentiment_std": round(std_score, 3),
            "category_distribution": dict(category_counts),
            "overall_trend": trend,
            "positive_ratio": category_counts["positive"] / len(texts) if texts else 0,
            "negative_ratio": category_counts["negative"] / len(texts) if texts else 0,
            "neutral_ratio": category_counts["neutral"] / len(texts) if texts else 0,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_financial_text(text: str) -> Dict[str, Any]:
    """
    Quick analysis of financial text.

    Args:
        text: Financial text to analyze

    Returns:
        Comprehensive analysis dictionary
    """
    analyzer = FinancialTextAnalyzer()
    return analyzer.analyze(text)


def extract_financial_features(text: str) -> TextFeatures:
    """
    Extract features from financial text.

    Args:
        text: Financial text

    Returns:
        TextFeatures dataclass
    """
    processor = AdvancedTextProcessor()
    processed = processor.process(text)
    return processed.features


def get_lexicon_sentiment(text: str) -> Dict[str, Any]:
    """
    Get lexicon-based sentiment (Loughran-McDonald).

    Args:
        text: Financial text

    Returns:
        Sentiment dictionary
    """
    features = extract_financial_features(text)

    return {
        "positive_count": features.positive_word_count,
        "negative_count": features.negative_word_count,
        "uncertainty_count": features.uncertainty_word_count,
        "sentiment_score": features.sentiment_score,
        "positive_words": features.positive_words,
        "negative_words": features.negative_words,
        "uncertainty_words": features.uncertainty_words,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test the analyzer
    test_texts = [
        "The company reported strong earnings growth of 25%, beating analyst expectations significantly.",
        "Revenue declined sharply by 15% due to weak demand, raising concerns about future profitability.",
        "The quarterly results were in line with expectations, with revenue remaining flat year-over-year.",
        "Despite challenges, management expects a strong recovery in Q4 driven by new product launches.",
        "The company faces potential litigation risks that could negatively impact shareholder value.",
    ]

    print("=" * 80)
    print("ADVANCED FINANCIAL TEXT ANALYSIS")
    print("=" * 80)

    analyzer = FinancialTextAnalyzer()

    for text in test_texts:
        print(f"\nText: {text[:80]}...")
        result = analyzer.analyze(text)

        print(f"  Sentiment: {result['sentiment']['category'].upper()} "
              f"(score: {result['sentiment']['score']:.2f}, "
              f"confidence: {result['sentiment']['confidence']:.2f})")
        print(f"  Explanation: {result['sentiment']['explanation']}")

        if result['entities']['percentages']:
            print(f"  Percentages: {result['entities']['percentages']}")

        print("-" * 80)

    # Aggregate analysis
    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)

    aggregate = analyzer.get_aggregate_sentiment(test_texts)
    print(f"Total texts: {aggregate['total_texts']}")
    print(f"Average sentiment: {aggregate['average_sentiment_score']:.3f}")
    print(f"Overall trend: {aggregate['overall_trend'].upper()}")
    print(f"Distribution: {aggregate['category_distribution']}")

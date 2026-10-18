"""Tests for the LLM benchmark logic, using a stub in place of the OpenAI API."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_sentiment import (  # noqa: E402
    LABELS,
    SYSTEM_PROMPT,
    AnswerCache,
    build_messages,
    classify,
    few_shot_examples,
    summarize_usage,
)
from stats import bootstrap_macro_f1, macro_f1  # noqa: E402

ROOT = Path(__file__).parent.parent


class StubRunnable:
    """Mimics ``ChatOpenAI.with_structured_output(..., include_raw=True)``."""

    def __init__(self, answers: dict[str, str | None], fail_on: set[str] | None = None):
        self.answers = answers
        self.fail_on = fail_on or set()
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        text = messages[-1][1].removeprefix("Headline: ")
        if text in self.fail_on:
            raise RuntimeError("rate limited")
        label = self.answers.get(text)
        parsed = SimpleNamespace(label=label) if label else None
        raw = SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 5})
        return {"raw": raw, "parsed": parsed, "parsing_error": None if label else "bad output"}


def test_zero_shot_prompt_is_system_plus_headline():
    msgs = build_messages("Shares fell 8%", "zero_shot")
    assert msgs[0] == ("system", SYSTEM_PROMPT)
    assert msgs[-1] == ("human", "Headline: Shares fell 8%")
    assert len(msgs) == 2


def test_few_shot_examples_are_balanced_and_never_from_the_evaluation_set():
    examples = few_shot_examples(per_class=2)
    assert sorted(label for _, label in examples) == sorted(list(LABELS) * 2)
    eval_headlines = set(pd.read_csv(ROOT / "data" / "external" / "real_financial_news.csv")["headline"])
    assert not {text for text, _ in examples} & eval_headlines
    msgs = build_messages("x", "few_shot", examples)
    assert len(msgs) == 1 + 2 * len(examples) + 1


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        build_messages("x", "chain_of_thought")


def test_classify_records_labels_invalid_answers_and_failures_in_order():
    stub = StubRunnable({"good": "positive", "bad": None}, fail_on={"boom"})
    out = classify(["good", "bad", "boom"], stub, "m", "zero_shot", max_concurrency=2)
    assert [r["label"] for r in out] == ["positive", None, None]
    assert out[0]["input_tokens"] == 100 and out[0]["error"] is None
    assert out[1]["error"] == "bad output"
    assert "rate limited" in out[2]["error"]


def test_cache_prevents_paying_twice(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    stub = StubRunnable({"a": "negative", "b": "neutral"})
    classify(["a", "b"], stub, "m", "zero_shot", cache=AnswerCache(cache_path))
    assert stub.calls == 2
    again = classify(["a", "b"], stub, "m", "zero_shot", cache=AnswerCache(cache_path))
    assert stub.calls == 2  # no new API calls
    assert all(r["cached"] for r in again)
    # A different model or mode is a different cache entry.
    classify(["a"], stub, "other-model", "zero_shot", cache=AnswerCache(cache_path))
    assert stub.calls == 3


def test_invalid_answers_are_not_cached(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    stub = StubRunnable({"bad": None})
    classify(["bad"], stub, "m", "zero_shot", cache=AnswerCache(cache_path))
    classify(["bad"], stub, "m", "zero_shot", cache=AnswerCache(cache_path))
    assert stub.calls == 2


def test_usage_summary_cost_only_with_prices():
    records = [
        {"label": "positive", "input_tokens": 1000, "output_tokens": 10, "latency_ms": 100.0},
        {"label": None, "input_tokens": 1000, "output_tokens": 10, "latency_ms": 300.0, "error": "x"},
    ]
    no_price = summarize_usage(records)
    assert no_price["cost_usd"] is None and no_price["invalid"] == 1 and no_price["errors"] == 1
    priced = summarize_usage(records, price_in=1.0, price_out=2.0)
    assert priced["cost_usd"] == pytest.approx(2000 / 1e6 * 1.0 + 20 / 1e6 * 2.0)
    assert priced["cost_per_1k_usd"] == pytest.approx(priced["cost_usd"] / 2 * 1000)
    assert priced["latency_ms_p50"] == pytest.approx(200.0)
    cached = summarize_usage([{**r, "cached": True} for r in records])
    assert cached["latency_ms_p50"] == pytest.approx(200.0) and cached["cached"] == 2


def test_invalid_predictions_count_as_wrong_in_macro_f1():
    y_true = [0, 1, 2, 1]
    assert macro_f1(y_true, [0, 1, 2, 1]) == pytest.approx(1.0)
    assert macro_f1(y_true, [0, 1, 2, -1]) < 1.0


def test_bootstrap_interval_brackets_the_point_estimate():
    y_true = [0, 1, 2] * 50
    y_pred = [0, 1, 2] * 40 + [1, 1, 1] * 10
    r = bootstrap_macro_f1(y_true, y_pred, n_boot=300)
    assert r["ci_low"] <= r["f1_macro"] <= r["ci_high"]
    assert r["n"] == 150

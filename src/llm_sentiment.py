"""
LLM sentiment classification (Claude, OpenRouter or OpenAI, via LangChain) for benchmarking against FinBERT.

The question this answers: does a general-purpose LLM, prompted zero-shot or
few-shot, beat a small fine-tuned model on real financial headlines, and at
what cost and latency?

Design:
- Structured output (a Pydantic schema with a ``Literal`` label), so the model
  can only answer negative / neutral / positive. Anything else is recorded as
  invalid and scored as wrong.
- Temperature 0 for repeatable answers.
- Few-shot examples come from the Financial PhraseBank *training* split, never
  from the evaluation headlines.
- Every answer is cached on disk, keyed by model, prompt version, mode and
  text, so re-running the benchmark never pays twice.

Requires ``ANTHROPIC_API_KEY`` (Claude, the default), ``OPENROUTER_API_KEY`` or ``OPENAI_API_KEY``;
see ``llm_provider`` for how the provider and model are chosen.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from llm_provider import get_model_name, make_chat_model, with_structured_output  # noqa: F401  (re-exported)
from utils import LABEL_MAP, get_project_root

PROMPT_VERSION = "v1"
LABELS = ("negative", "neutral", "positive")
MODES = ("zero_shot", "few_shot")

SYSTEM_PROMPT = (
    "You are a financial analyst. Classify the sentiment of a financial news headline "
    "from the point of view of an investor in the company or market it mentions.\n"
    "- positive: good news for the investor (growth, beats, upgrades, rising prices, new contracts).\n"
    "- negative: bad news for the investor (losses, misses, downgrades, cuts, falling prices, lawsuits).\n"
    "- neutral: factual or mixed, with no clear effect (announcements, schedules, routine updates).\n"
    "Answer with exactly one label."
)


class SentimentLabel(BaseModel):
    """The only answer the model is allowed to give."""

    label: Literal["negative", "neutral", "positive"] = Field(description="The sentiment of the headline")


def few_shot_examples(per_class: int = 2, seed: int = 42) -> list[tuple[str, str]]:
    """Balanced examples from the PhraseBank training split (never from the evaluation set)."""
    import pandas as pd

    df = pd.read_csv(get_project_root() / "data" / "processed" / "train.csv")
    inv = {v: k for k, v in LABEL_MAP.items()}
    examples = []
    for label_id in sorted(inv):
        rows = df[df["label"] == label_id].sample(n=per_class, random_state=seed)
        examples += [(str(t), inv[label_id]) for t in rows["sentence"]]
    # Interleave classes so the order doesn't hint at the answer.
    return [examples[i] for j in range(per_class) for i in range(j, len(examples), per_class)]


def build_messages(text: str, mode: str, examples: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    """Chat messages for one headline. ``few_shot`` adds worked examples as prior turns."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    messages: list[tuple[str, str]] = [("system", SYSTEM_PROMPT)]
    if mode == "few_shot":
        for ex_text, ex_label in examples or few_shot_examples():
            messages.append(("human", f"Headline: {ex_text}"))
            messages.append(("ai", ex_label))
    messages.append(("human", f"Headline: {text}"))
    return messages


def make_runnable(model: str, provider: str | None = None, timeout: float = 60.0, max_retries: int = 3):
    """Chat model with structured output; ``include_raw`` keeps token usage."""
    llm = make_chat_model(model, provider, timeout=timeout, max_retries=max_retries, max_tokens=64)
    return with_structured_output(llm, SentimentLabel, provider, include_raw=True)


def _cache_key(model: str, mode: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{PROMPT_VERSION}|{mode}|{text}".encode()).hexdigest()


class AnswerCache:
    """Append-only JSON-lines cache of LLM answers."""

    def __init__(self, path: Path | None):
        self.path = path
        self._data: dict[str, dict] = {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, record: dict) -> None:
        record = {"key": key, **record}
        self._data[key] = record
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")


def _call_one(runnable, messages) -> dict[str, Any]:
    """One LLM call → label (or None), token usage, latency, error."""
    start = time.perf_counter()
    try:
        out = runnable.invoke(messages)
    except Exception as exc:  # noqa: BLE001  (record the failure, keep the benchmark running)
        return {"label": None, "input_tokens": 0, "output_tokens": 0, "latency_ms": None, "error": str(exc)[:300]}
    latency_ms = (time.perf_counter() - start) * 1000
    parsed = out.get("parsed") if isinstance(out, dict) else out
    raw = out.get("raw") if isinstance(out, dict) else None
    usage = getattr(raw, "usage_metadata", None) or {}
    label = getattr(parsed, "label", None)
    return {
        "label": label if label in LABELS else None,
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "latency_ms": latency_ms,
        "error": None if label in LABELS else str((out or {}).get("parsing_error") or "no valid label")[:300],
    }


def classify(
    texts: list[str],
    runnable,
    model: str,
    mode: str,
    cache: AnswerCache | None = None,
    max_concurrency: int = 8,
    examples: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Classify ``texts`` with the LLM; returns one record per text, in order.

    Cached answers are reused (``cached: True``) and cost nothing.
    """
    examples = examples if mode == "zero_shot" else (examples or few_shot_examples())
    results: list[dict[str, Any] | None] = [None] * len(texts)
    todo = []
    for i, text in enumerate(texts):
        key = _cache_key(model, mode, text)
        hit = cache.get(key) if cache else None
        if hit and hit.get("label") in LABELS:
            results[i] = {**hit, "cached": True}
        else:
            todo.append((i, key, text))

    def work(item):
        i, key, text = item
        rec = _call_one(runnable, build_messages(text, mode, examples))
        if cache and rec["label"] in LABELS:
            cache.put(key, rec)
        return i, {**rec, "cached": False}

    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        for i, rec in pool.map(work, todo):
            results[i] = rec
    return results  # type: ignore[return-value]


def summarize_usage(records: list[dict], price_in: float | None = None, price_out: float | None = None) -> dict:
    """
    Token, cost and latency summary for one benchmark run.

    Prices are USD per 1M tokens, supplied by the caller from OpenAI's pricing
    page; without them ``cost_usd`` is None rather than a guess. Cached answers
    are excluded from latency and from the cost actually spent this run.
    """
    import numpy as np

    fresh = [r for r in records if not r.get("cached")]
    tokens_in = sum(r.get("input_tokens", 0) for r in records)
    tokens_out = sum(r.get("output_tokens", 0) for r in records)
    lat = [r["latency_ms"] for r in fresh if r.get("latency_ms") is not None]
    cost = None
    if price_in is not None and price_out is not None:
        cost = tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out
    n = max(len(records), 1)
    return {
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cost_usd": cost,
        "cost_per_1k_usd": None if cost is None else cost / n * 1000,
        "latency_ms_p50": float(np.percentile(lat, 50)) if lat else None,
        "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
        "invalid": sum(1 for r in records if r.get("label") not in LABELS),
        "errors": sum(1 for r in records if r.get("error")),
        "cached": len(records) - len(fresh),
    }

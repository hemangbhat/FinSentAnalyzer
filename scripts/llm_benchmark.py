"""
Benchmark an LLM (Claude by default, or OpenAI) against the fine-tuned FinBERT on real financial news.

Same 2,388 headlines (`zeroshot/twitter-financial-news-sentiment`, validation),
same metric (macro-F1 with a 2,000-resample bootstrap interval) as every other
number in the project, plus what an LLM costs that a local model doesn't:
tokens, dollars and latency.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...      # Claude (default model: Claude Haiku)
    # 1. See the exact prompts and a token estimate; no API calls, no cost:
    python scripts/llm_benchmark.py --dry-run
    # 2. Cheap trial on a stratified sample:
    python scripts/llm_benchmark.py --limit 200
    # 3. Full run, with prices in USD per 1M tokens from the provider's pricing page
    #    (https://www.anthropic.com/pricing or https://openai.com/api/pricing):
    python scripts/llm_benchmark.py --price-in <input> --price-out <output>

    # OpenRouter instead: export OPENROUTER_API_KEY=... and pass --provider openrouter --model <vendor/model>
    # OpenAI instead:     export OPENAI_API_KEY=...     and pass --provider openai --model <model>

Answers are cached in data/cache/llm_answers.jsonl, so re-runs are free.
Writes results/llm_benchmark.json, which the Generalization page displays.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_provider import KEY_ENV, PROVIDERS, get_provider, has_api_key  # noqa: E402
from llm_sentiment import (  # noqa: E402
    MODES,
    PROMPT_VERSION,
    AnswerCache,
    build_messages,
    classify,
    few_shot_examples,
    get_model_name,
    make_runnable,
    summarize_usage,
)
from stats import bootstrap_macro_f1  # noqa: E402
from utils import LABEL_MAP, get_results_dir  # noqa: E402

DATA = ROOT / "data" / "external" / "real_financial_news.csv"
CACHE = ROOT / "data" / "cache" / "llm_answers.jsonl"
DATASET = "zeroshot/twitter-financial-news-sentiment (validation, MIT)"


def load_eval(limit: int | None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(DATA)
    if limit and limit < len(df):
        # Stratified sample so a small trial keeps the real class mix.
        df = df.groupby("label", group_keys=False).sample(frac=limit / len(df), random_state=seed)
    return df.reset_index(drop=True)


def dry_run(df: pd.DataFrame, modes: list[str]) -> None:
    examples = few_shot_examples()
    for mode in modes:
        msgs = build_messages(df["headline"].iloc[0], mode, examples)
        chars = sum(len(c) for _, c in msgs) - len(df["headline"].iloc[0])
        est_tokens = (chars + df["headline"].str.len().mean()) / 4 * len(df)  # ~4 characters per token
        print(f"\n=== {mode}: {len(msgs)} messages, prompt version {PROMPT_VERSION} ===")
        for role, content in msgs:
            print(f"[{role}] {content}")
        print(
            f"\n≈ {est_tokens:,.0f} prompt tokens for {len(df):,} headlines. Providers add the output-schema "
            "definition to every call (a few hundred tokens with tool calling, e.g. on OpenRouter), so the "
            "billed total can be several times higher; the real count is in results/llm_benchmark.json."
        )
    print("\nDry run only: no API calls were made.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--provider", choices=PROVIDERS, help="anthropic (default), openrouter or openai; or FINSIGHT_LLM_PROVIDER"
    )
    ap.add_argument("--model", help="Chat model (or FINSIGHT_LLM_MODEL); Claude defaults to Haiku")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--limit", type=int, help="Stratified sample size for a cheap trial run")
    ap.add_argument("--price-in", type=float, help="USD per 1M input tokens (from the provider's pricing page)")
    ap.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="Print prompts and a token estimate; no API calls")
    args = ap.parse_args()

    df = load_eval(args.limit)
    if args.dry_run:
        dry_run(df, args.modes)
        return

    provider = get_provider(args.provider)
    model = get_model_name(args.model, provider)
    if not has_api_key(provider):
        sys.exit(
            f"No API key for {provider}: set {KEY_ENV[provider]} first "
            '(PowerShell: $env:NAME = "value"; bash: export NAME=value).'
        )
    runnable = make_runnable(model, provider)
    cache = AnswerCache(CACHE)
    y_true = df["label"].to_numpy()
    full_set = len(df) == len(pd.read_csv(DATA))

    runs = []
    for mode in args.modes:
        print(f"[{mode}] classifying {len(df):,} headlines with {model} …")
        records = classify(df["headline"].tolist(), runnable, model, mode, cache, args.max_concurrency)
        failed = [r for r in records if r.get("error") and not r.get("cached")]
        if len(failed) == len(records):
            sys.exit(
                f"Every call failed, so nothing was saved. First error:\n  {failed[0]['error']}\n"
                "Check the API key, the model id and your credit."
            )
        y_pred = np.array([LABEL_MAP.get(r["label"], -1) for r in records])
        ci = bootstrap_macro_f1(y_true, y_pred)
        usage = summarize_usage(records, args.price_in, args.price_out)
        runs.append(
            {
                "mode": mode,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                **ci,
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
                **usage,
            }
        )
        if failed:
            from collections import Counter

            top = Counter(r["error"][:160] for r in failed).most_common(3)
            print(f"  {len(failed)} calls failed (scored as wrong; not cached, so a re-run retries them):")
            for msg, n in top:
                print(f"    {n}× {msg}")
        cost = "n/a (no prices given)" if usage["cost_usd"] is None else f"${usage['cost_usd']:.4f}"
        print(
            f"  macro-F1 {ci['f1_macro']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] · "
            f"invalid {usage['invalid']} · cached {usage['cached']} · cost {cost} · "
            f"p50 {usage['latency_ms_p50'] or 0:.0f} ms"
        )

    out = {
        "provider": provider,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "dataset": DATASET,
        "n": len(df),
        "full_set": full_set,
        "prices_usd_per_1m": {"input": args.price_in, "output": args.price_out},
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": runs,
    }
    dest = get_results_dir() / "llm_benchmark.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {dest}")
    if not full_set:
        print("Note: this was a sample. Run without --limit to compare with FinBERT on identical rows.")


if __name__ == "__main__":
    main()

"""
Which LLM provider and chat model to use: Anthropic (Claude) or OpenAI.

Resolution order:
- Provider: ``FINSIGHT_LLM_PROVIDER`` (``anthropic`` or ``openai``); otherwise
  whichever API key is set (Anthropic first); otherwise ``anthropic``.
- Model: the explicit argument, then ``FINSIGHT_LLM_MODEL``, then the
  provider's default. Claude defaults to Haiku, the fastest and cheapest
  tier, which is what a classification benchmark wants. OpenAI has no
  default: pass one from OpenAI's model list.
- Temperature: 0 for repeatable answers. Set ``FINSIGHT_LLM_TEMPERATURE=none``
  for a model that doesn't accept the parameter.
"""

from __future__ import annotations

import os

PROVIDERS = ("anthropic", "openai")
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": None}


def get_provider(provider: str | None = None) -> str:
    name = (provider or os.getenv("FINSIGHT_LLM_PROVIDER") or "").strip().lower()
    if not name:
        name = next((p for p in PROVIDERS if os.getenv(KEY_ENV[p])), "anthropic")
    if name not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider {name!r}; use one of {PROVIDERS}")
    return name


def get_model_name(model: str | None = None, provider: str | None = None) -> str:
    provider = get_provider(provider)
    name = model or os.getenv("FINSIGHT_LLM_MODEL") or DEFAULT_MODELS[provider]
    if not name:
        raise ValueError(
            f"No {provider} model set. Pass --model or set FINSIGHT_LLM_MODEL "
            "(pick a current, inexpensive chat model from the provider's model list)."
        )
    return name


def get_temperature(default: float = 0.0) -> float | None:
    raw = os.getenv("FINSIGHT_LLM_TEMPERATURE")
    if raw is None or raw.strip() == "":
        return default
    return None if raw.strip().lower() == "none" else float(raw)


def has_api_key(provider: str | None = None) -> bool:
    return bool(os.getenv(KEY_ENV[get_provider(provider)]))


def make_chat_model(
    model: str | None = None,
    provider: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    max_tokens: int = 512,
):
    """A LangChain chat model for the resolved provider."""
    provider = get_provider(provider)
    model = get_model_name(model, provider)
    kwargs: dict = {"model": model, "timeout": timeout, "max_retries": max_retries}
    temperature = get_temperature()
    if temperature is not None:
        kwargs["temperature"] = temperature
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # pyre-ignore

        return ChatAnthropic(max_tokens=max_tokens, **kwargs)
    from langchain_openai import ChatOpenAI  # pyre-ignore

    return ChatOpenAI(max_tokens=max_tokens, **kwargs)

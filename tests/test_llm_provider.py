"""Provider and model resolution for the LLM features (no API calls)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_provider import get_model_name, get_provider, get_temperature, make_chat_model  # noqa: E402

ENV = (
    "FINSIGHT_LLM_PROVIDER",
    "FINSIGHT_LLM_MODEL",
    "FINSIGHT_LLM_TEMPERATURE",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


def test_claude_is_the_default_provider_with_a_default_model():
    assert get_provider() == "anthropic"
    assert get_model_name().startswith("claude-")


def test_provider_follows_the_key_that_is_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_provider() == "openai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert get_provider() == "anthropic"  # Claude wins when both are set
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")
    assert get_provider() == "openai"  # explicit setting wins


def test_openai_needs_an_explicit_model(monkeypatch):
    with pytest.raises(ValueError):
        get_model_name(provider="openai")
    monkeypatch.setenv("FINSIGHT_LLM_MODEL", "some-model")
    assert get_model_name(provider="openai") == "some-model"
    assert get_model_name("explicit", provider="openai") == "explicit"


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        get_provider("gemini")


def test_temperature_can_be_omitted(monkeypatch):
    assert get_temperature() == 0.0
    monkeypatch.setenv("FINSIGHT_LLM_TEMPERATURE", "none")
    assert get_temperature() is None
    monkeypatch.setenv("FINSIGHT_LLM_TEMPERATURE", "0.3")
    assert get_temperature() == 0.3


def test_claude_chat_model_builds_without_calling_the_api(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from llm_sentiment import SentimentLabel

    llm = make_chat_model(max_tokens=64)
    assert type(llm).__name__ == "ChatAnthropic"
    assert llm.model.startswith("claude-") and llm.temperature == 0.0
    assert llm.with_structured_output(SentimentLabel, include_raw=True) is not None


def test_openrouter_key_selects_openrouter_and_needs_a_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert get_provider() == "openrouter"
    with pytest.raises(ValueError):
        get_model_name()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert get_provider() == "anthropic"  # Claude still wins when both are set


def test_openrouter_client_points_at_openrouter(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    from llm_provider import OPENROUTER_BASE_URL, with_structured_output
    from llm_sentiment import SentimentLabel

    llm = make_chat_model("vendor/some-model")
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.openai_api_base == OPENROUTER_BASE_URL
    assert llm.openai_api_key.get_secret_value() == "sk-or-test"
    assert with_structured_output(llm, SentimentLabel, include_raw=True) is not None

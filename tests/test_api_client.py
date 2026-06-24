"""
Tests for the RemotePredictor client used in the UI/API split.

HTTP is monkeypatched, so these run without a live server.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import httpx  # noqa: E402
from api_client import RemotePredictor, RemotePredictorError, remote_available_models  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_remote_predict_single(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(
            200, {"label": "positive", "confidence": 0.9, "probabilities": {}, "model": "baseline_svm"}
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    pred = RemotePredictor("baseline_svm", base_url="http://api", api_key="k")
    out = pred.predict("Revenue rose sharply.")
    assert out["label"] == "positive"
    assert captured["url"].endswith("/predict")
    assert captured["json"]["model"] == "baseline_svm"


def test_remote_predict_batch(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        assert url.endswith("/predict/batch")
        return _FakeResponse(200, {"results": [{"label": "positive"}, {"label": "negative"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    pred = RemotePredictor("baseline_svm", base_url="http://api")
    out = pred.predict(["a", "b"])
    assert isinstance(out, list) and len(out) == 2


def test_remote_predict_raises_on_error(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(500, text="boom"))
    pred = RemotePredictor("baseline_svm", base_url="http://api")
    with pytest.raises(RemotePredictorError):
        pred.predict("text")


def test_remote_predict_raises_on_connection_failure(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("no server")

    monkeypatch.setattr(httpx, "post", boom)
    pred = RemotePredictor("baseline_svm", base_url="http://api")
    with pytest.raises(RemotePredictorError):
        pred.predict("text")


def test_remote_models_fallback_empty(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("no server")

    monkeypatch.setattr(httpx, "get", boom)
    assert remote_available_models("http://api") == []

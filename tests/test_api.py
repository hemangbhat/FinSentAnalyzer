"""
Tests for the FinSight inference API (api/main.py).

Skipped gracefully when FastAPI is not installed so the rest of the suite
still runs in minimal environments.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("slowapi")

from fastapi.testclient import TestClient  # noqa: E402

# Make api/ importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app  # noqa: E402

client = TestClient(app)


class TestInferenceAPI:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert isinstance(body["available_models"], list)

    def test_models(self):
        resp = client.get("/models")
        assert resp.status_code == 200
        assert "models" in resp.json()

    def test_predict_single(self):
        resp = client.post(
            "/predict",
            json={"text": "The company reported record quarterly profit.", "model": "baseline_svm"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] in {"positive", "neutral", "negative"}
        assert body["model"] == "baseline_svm"
        assert "X-Process-Time-ms" in resp.headers

    def test_predict_rejects_blank_text(self):
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 422  # validation error

    def test_predict_rejects_oversized_text(self):
        resp = client.post("/predict", json={"text": "a" * 6000})
        assert resp.status_code == 422

    def test_predict_batch(self):
        resp = client.post(
            "/predict/batch",
            json={"texts": ["Revenue rose sharply.", "Losses widened this quarter."], "model": "baseline_svm"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert len(body["results"]) == 2

    def test_batch_rejects_empty_list(self):
        resp = client.post("/predict/batch", json={"texts": []})
        assert resp.status_code == 422

    def test_metrics(self):
        # Exercise an endpoint first so counters are populated.
        client.post("/predict", json={"text": "Strong earnings beat expectations.", "model": "baseline_svm"})
        resp = client.get("/metrics")
        assert resp.status_code == 200
        snap = resp.json()
        assert "counters" in snap
        assert snap["counters"].get("predictions_total", 0) >= 1

    def test_auth_enforced_when_key_set(self, monkeypatch):
        # When FINSIGHT_API_KEY is set, requests without the header are rejected.
        monkeypatch.setenv("FINSIGHT_API_KEY", "secret-key")
        denied = client.post("/predict", json={"text": "Revenue rose.", "model": "baseline_svm"})
        assert denied.status_code == 401

        allowed = client.post(
            "/predict",
            json={"text": "Revenue rose.", "model": "baseline_svm"},
            headers={"X-API-Key": "secret-key"},
        )
        assert allowed.status_code == 200

    def test_health_reports_auth_flag(self):
        body = client.get("/health").json()
        assert "auth_enabled" in body
        assert "rate_limit_backend" in body

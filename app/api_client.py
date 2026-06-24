"""
Remote predictor client.

Lets the Streamlit UI consume predictions from the FastAPI inference service
instead of loading models in-process. This is what enables a clean separation
of concerns: the UI becomes a stateless presentation layer, and the model
service scales independently behind a load balancer.

Activated by setting ``FINSIGHT_API_URL`` (see ui.load_predictor). The
``predict`` interface mirrors ``predict.SentimentPredictor.predict`` so the
pages don't care whether inference is local or remote.
"""

from __future__ import annotations

from typing import List, Union

import httpx


class RemotePredictorError(RuntimeError):
    """Raised when the inference service cannot be reached or returns an error."""


class RemotePredictor:
    """Drop-in replacement for SentimentPredictor that calls the HTTP API."""

    def __init__(self, model_type: str, base_url: str, api_key: str | None = None, timeout: float = 30.0):
        self.model_type = model_type
        self.base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key} if api_key else {}
        self._timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = httpx.post(f"{self.base_url}{path}", json=payload, headers=self._headers, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise RemotePredictorError(f"Could not reach inference service at {self.base_url}: {exc}") from exc
        if resp.status_code != 200:
            raise RemotePredictorError(f"Inference service returned {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def predict(self, text: Union[str, List[str]]) -> Union[dict, List[dict]]:
        """Predict sentiment for a single string or a list of strings."""
        if isinstance(text, str):
            return self._post("/predict", {"text": text, "model": self.model_type})
        body = self._post("/predict/batch", {"texts": list(text), "model": self.model_type})
        return body.get("results", [])


def remote_available_models(base_url: str, api_key: str | None = None, timeout: float = 10.0) -> List[str]:
    """Fetch the model list from the API's /models endpoint (empty list on failure)."""
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except httpx.HTTPError:
        pass
    return []

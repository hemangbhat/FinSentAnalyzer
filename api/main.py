"""
FinSight inference API.

A small, reusable FastAPI service that wraps the same SentimentPredictor used
by the dashboard, so the project is a service as well as a UI.

Endpoints
---------
GET  /health        Liveness + which model is loaded.
GET  /models        List available trained models.
POST /predict       Classify a single text.
POST /predict/batch Classify a list of texts.
GET  /metrics       In-process counters + latency percentiles.

Features
--------
- Strict input validation (length limits, batch-size cap, non-empty) via Pydantic.
- Per-request latency + structured prediction logging (see src/monitoring.py).
- IP-based rate limiting via slowapi (configurable through env vars).

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Make src/ importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from monitoring import log_event, logger, metrics  # pyre-ignore
from predict import SentimentPredictor, get_available_models  # pyre-ignore

# ── Config (env-overridable) ─────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("FINSIGHT_DEFAULT_MODEL", "baseline_svm")
MAX_TEXT_CHARS = int(os.getenv("FINSIGHT_MAX_TEXT_CHARS", "5000"))
MAX_BATCH_SIZE = int(os.getenv("FINSIGHT_MAX_BATCH_SIZE", "256"))
RATE_LIMIT = os.getenv("FINSIGHT_RATE_LIMIT", "60/minute")
BATCH_RATE_LIMIT = os.getenv("FINSIGHT_BATCH_RATE_LIMIT", "10/minute")
# Storage for the rate limiter. Set FINSIGHT_REDIS_URL (e.g. redis://host:6379)
# to share limits across workers/replicas; falls back to in-process memory.
REDIS_URL = os.getenv("FINSIGHT_REDIS_URL")
# Only trust X-Forwarded-For when running behind a known, trusted proxy.
TRUST_PROXY = os.getenv("FINSIGHT_TRUST_PROXY", "false").lower() in {"1", "true", "yes"}


def _client_key(request: Request) -> str:
    """Rate-limit key. Uses the first X-Forwarded-For hop only when explicitly trusted."""
    if TRUST_PROXY:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return get_remote_address(request)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Enforce an API key when FINSIGHT_API_KEY is set.

    If the env var is unset (local/dev), auth is disabled. Read at call time so
    it can be toggled per-deployment (and tested) without re-importing the app.
    """
    expected = os.getenv("FINSIGHT_API_KEY")
    if expected and x_api_key != expected:
        metrics.incr("auth_failures")
        raise HTTPException(status_code=401, detail="Invalid or missing API key (set the X-API-Key header).")


# ── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=_client_key, default_limits=[], storage_uri=REDIS_URL or "memory://")

app = FastAPI(
    title="FinSight Sentiment API",
    description="Programmatic access to the Financial Sentiment Analyzer models.",
    version="1.0.0",
)
app.state.limiter = limiter

if not os.getenv("FINSIGHT_API_KEY"):
    logger.warning("FINSIGHT_API_KEY is not set — the API is UNAUTHENTICATED. Set it before public exposure.")
if not REDIS_URL:
    logger.warning("FINSIGHT_REDIS_URL not set — rate limiting is in-process only (resets on restart, per-worker).")

# Cache loaded predictors across requests.
_PREDICTORS: dict[str, SentimentPredictor] = {}


def _get_predictor(model: Optional[str]) -> SentimentPredictor:
    key = model or DEFAULT_MODEL
    if key not in _PREDICTORS:
        _PREDICTORS[key] = SentimentPredictor(key)
        logger.info("Loaded model into API cache: %s", key)
    return _PREDICTORS[key]


# ── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS, description="Financial text to classify.")
    model: Optional[str] = Field(None, description="Model key; defaults to the server default.")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    model: Optional[str] = None

    @field_validator("texts")
    @classmethod
    def _validate_texts(cls, v: List[str]) -> List[str]:
        cleaned = [t for t in v if isinstance(t, str) and t.strip()]
        if not cleaned:
            raise ValueError("texts must contain at least one non-empty string")
        for t in cleaned:
            if len(t) > MAX_TEXT_CHARS:
                raise ValueError(f"each text must be <= {MAX_TEXT_CHARS} characters")
        return cleaned


class PredictResponse(BaseModel):
    label: str
    confidence: Optional[float] = None
    probabilities: Optional[dict] = None
    model: str


# ── Error handlers ───────────────────────────────────────────────────────────
@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    metrics.incr("rate_limited")
    log_event("rate_limited", path=request.url.path, client=get_remote_address(request))
    return JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {exc.detail}"})


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    metrics.incr("requests_total")
    try:
        response = await call_next(request)
    except Exception as exc:  # pragma: no cover - defensive
        metrics.incr("errors_total")
        logger.exception("Unhandled error on %s", request.url.path)
        log_event("error", path=request.url.path, error=str(exc))
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    elapsed_ms = (time.perf_counter() - start) * 1000
    metrics.observe_latency(elapsed_ms)
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
    return response


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "available_models": get_available_models(),
        "auth_enabled": bool(os.getenv("FINSIGHT_API_KEY")),
        "rate_limit_backend": "redis" if REDIS_URL else "memory",
    }


@app.get("/models")
def models():
    return {"models": get_available_models(), "default": DEFAULT_MODEL}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT)
def predict(request: Request, payload: PredictRequest):
    start = time.perf_counter()
    try:
        predictor = _get_predictor(payload.model)
        result = predictor.predict(payload.text)
    except FileNotFoundError as exc:
        metrics.incr("errors_total")
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except Exception as exc:  # noqa: BLE001
        metrics.incr("errors_total")
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"detail": f"Prediction failed: {exc}"})

    metrics.incr("predictions_total")
    metrics.incr(f"label_{result['label']}")
    log_event(
        "prediction",
        model=payload.model or DEFAULT_MODEL,
        label=result["label"],
        confidence=result.get("confidence"),
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        text_len=len(payload.text),
    )
    return PredictResponse(
        label=result["label"],
        confidence=result.get("confidence"),
        probabilities=result.get("probabilities"),
        model=payload.model or DEFAULT_MODEL,
    )


@app.post("/predict/batch", dependencies=[Depends(require_api_key)])
@limiter.limit(BATCH_RATE_LIMIT)
def predict_batch(request: Request, payload: BatchPredictRequest):
    start = time.perf_counter()
    try:
        predictor = _get_predictor(payload.model)
        results = predictor.predict(payload.texts)
    except Exception as exc:  # noqa: BLE001
        metrics.incr("errors_total")
        logger.exception("Batch prediction failed")
        return JSONResponse(status_code=500, content={"detail": f"Batch prediction failed: {exc}"})

    metrics.incr("predictions_total", amount=len(results))
    log_event(
        "batch_prediction",
        model=payload.model or DEFAULT_MODEL,
        count=len(results),
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
    return {
        "model": payload.model or DEFAULT_MODEL,
        "count": len(results),
        "results": [
            {
                "label": r["label"],
                "confidence": r.get("confidence"),
                "probabilities": r.get("probabilities"),
            }
            for r in results
        ],
    }


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def get_metrics():
    return metrics.snapshot()

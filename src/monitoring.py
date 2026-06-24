"""
Lightweight monitoring for the inference service.

Provides:
- A structured logger that writes human-readable logs to stderr and an
  optional JSON-lines audit trail to ``logs/predictions.jsonl``.
- An in-process metrics registry tracking request counts, error counts, and
  latency percentiles — exposed via the API ``/metrics`` endpoint.

Intentionally dependency-free (stdlib only) so it works everywhere the app
runs, including constrained deploy targets.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional

# ── Paths ────────────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_PREDICTION_LOG = _LOG_DIR / "predictions.jsonl"

# ── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("finsight.api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: str, **fields: Any) -> None:
    """Append a structured event to the JSONL audit trail (best-effort)."""
    record = {"ts": _now_iso(), "event": event, **fields}
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_PREDICTION_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:  # pragma: no cover - logging must never crash a request
        pass


class MetricsRegistry:
    """Thread-safe in-process counters and latency tracking."""

    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._latencies_ms: Deque[float] = deque(maxlen=window)
        self._started = time.time()

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe_latency(self, ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lat = sorted(self._latencies_ms)
            counters = dict(self._counters)

        def pct(p: float) -> Optional[float]:
            if not lat:
                return None
            idx = min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))
            return round(lat[idx], 2)

        return {
            "uptime_seconds": round(time.time() - self._started, 1),
            "counters": counters,
            "latency_ms": {
                "count": len(lat),
                "p50": pct(50),
                "p95": pct(95),
                "p99": pct(99),
                "max": round(lat[-1], 2) if lat else None,
            },
        }


# Module-level singleton used by the API.
metrics = MetricsRegistry()

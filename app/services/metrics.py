"""Lightweight in-process metrics for ingestion and query paths."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

_lock = threading.Lock()
_counters: dict[str, float] = defaultdict(float)
_timers: dict[str, list[float]] = defaultdict(list)


def incr(name: str, value: float = 1.0) -> None:
    with _lock:
        _counters[name] += value


def observe(name: str, seconds: float) -> None:
    with _lock:
        _timers[name].append(seconds)
        # keep last 500
        if len(_timers[name]) > 500:
            _timers[name] = _timers[name][-500:]


class Timer:
    def __init__(self, name: str) -> None:
        self.name = name
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        observe(self.name, time.perf_counter() - self._start)


def snapshot() -> dict[str, Any]:
    with _lock:
        timers = {
            k: {
                "count": len(v),
                "avg_ms": round(1000 * sum(v) / len(v), 2) if v else 0,
                "p95_ms": round(1000 * sorted(v)[int(0.95 * (len(v) - 1))] , 2) if v else 0,
            }
            for k, v in _timers.items()
        }
        return {"counters": dict(_counters), "timers": timers}

"""Bounded in-process observability for agents and governed tools."""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

from config import settings


@dataclass(frozen=True)
class ExecutionEvent:
    """One completed agent or tool invocation."""

    component: str
    success: bool
    latency_ms: float
    degraded: bool = False
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerformanceMonitor:
    """Collect metrics, derive alerts, and provide bounded routing penalties.

    This first implementation is intentionally process-local. Prometheus will
    scrape the eventual API metrics endpoint, while this class remains the
    low-latency source of routing feedback inside one application process.
    """

    def __init__(
        self,
        *,
        window_size: int | None = None,
        min_samples: int | None = None,
    ) -> None:
        self._window_size = window_size or settings.monitor_window_size
        self._min_samples = min_samples or settings.monitor_min_samples
        self._events: dict[str, deque[ExecutionEvent]] = defaultdict(
            lambda: deque(maxlen=self._window_size)
        )
        self._lock = Lock()

    def record_agent(self, agent_type: str, *, success: bool, latency_ms: float) -> None:
        self._record("agent:" + agent_type, success=success, latency_ms=latency_ms)

    def record_tool(
        self,
        tool_name: str,
        *,
        success: bool,
        latency_ms: float,
        degraded: bool = False,
    ) -> None:
        self._record(
            "tool:" + tool_name,
            success=success,
            latency_ms=latency_ms,
            degraded=degraded,
        )

    def routing_penalty(self, agent_type: str) -> float:
        """Return a reliability penalty in [0, configured maximum].

        Penalties are withheld until a minimum sample count exists. Thereafter,
        failure rate dominates, degraded results add a smaller cost, and mean
        latency above the service-level threshold adds a bounded cost.
        """
        events = self._snapshot("agent:" + agent_type)
        if len(events) < self._min_samples:
            return 0.0
        total = len(events)
        failure_rate = sum(not event.success for event in events) / total
        degraded_rate = sum(event.degraded for event in events) / total
        avg_latency = statistics.fmean(event.latency_ms for event in events)
        latency_excess = max(0.0, avg_latency / settings.monitor_latency_ms_threshold - 1.0)
        raw_penalty = failure_rate * 0.45 + degraded_rate * 0.15 + min(latency_excess, 1.0) * 0.20
        return round(min(settings.monitor_max_routing_penalty, raw_penalty), 4)

    def summary(self) -> dict[str, Any]:
        """Return API-ready metrics and actionable alerts for every component."""
        with self._lock:
            names = sorted(self._events)
        components = {name: self._component_summary(name) for name in names}
        agent_penalties = {
            name.removeprefix("agent:"): self.routing_penalty(name.removeprefix("agent:"))
            for name in names
            if name.startswith("agent:")
        }
        alerts = [alert for item in components.values() for alert in item["alerts"]]
        return {
            "components": components,
            "agent_penalties": agent_penalties,
            "alerts": alerts,
        }

    def _record(
        self,
        component: str,
        *,
        success: bool,
        latency_ms: float,
        degraded: bool = False,
    ) -> None:
        event = ExecutionEvent(
            component=component,
            success=success,
            latency_ms=max(0.0, float(latency_ms)),
            degraded=degraded,
            timestamp=time.time(),
        )
        with self._lock:
            self._events[component].append(event)

    def _snapshot(self, component: str) -> list[ExecutionEvent]:
        with self._lock:
            return list(self._events.get(component, ()))

    def _component_summary(self, component: str) -> dict[str, Any]:
        events = self._snapshot(component)
        if not events:
            return {
                "samples": 0,
                "success_rate": None,
                "avg_latency_ms": None,
                "p95_latency_ms": None,
                "degraded_rate": None,
                "latency_zscore": None,
                "alerts": [],
            }
        latencies = [event.latency_ms for event in events]
        success_rate = sum(event.success for event in events) / len(events)
        degraded_rate = sum(event.degraded for event in events) / len(events)
        zscore = self._latest_latency_zscore(latencies)
        alerts: list[dict[str, Any]] = []
        if len(events) >= self._min_samples and success_rate < settings.monitor_success_rate_threshold:
            alerts.append({"component": component, "type": "low_success_rate", "value": round(success_rate, 4)})
        if statistics.fmean(latencies) > settings.monitor_latency_ms_threshold:
            alerts.append({"component": component, "type": "high_latency", "value": round(statistics.fmean(latencies), 2)})
        if zscore is not None and zscore >= settings.monitor_anomaly_zscore:
            alerts.append({"component": component, "type": "latency_anomaly", "value": round(zscore, 2)})
        return {
            "samples": len(events),
            "success_rate": round(success_rate, 4),
            "avg_latency_ms": round(statistics.fmean(latencies), 2),
            "p95_latency_ms": round(self._percentile(latencies, 0.95), 2),
            "degraded_rate": round(degraded_rate, 4),
            "latency_zscore": None if zscore is None else round(zscore, 4),
            "alerts": alerts,
        }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        index = (len(ordered) - 1) * percentile
        lower, upper = math.floor(index), math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    @staticmethod
    def _latest_latency_zscore(latencies: list[float]) -> float | None:
        if len(latencies) < 3:
            return None
        history, latest = latencies[:-1], latencies[-1]
        deviation = statistics.pstdev(history)
        if deviation == 0:
            return None
        return (latest - statistics.fmean(history)) / deviation

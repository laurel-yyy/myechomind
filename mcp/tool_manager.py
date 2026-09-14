"""Tool governance harness: cache, circuit breaker, timeout, fallback.

Any concrete tool subclasses `Tool` and implements `call(params) -> ToolResult`.
`ToolManager.execute(name, params)` is the single funnel that adds observability
and safety around every tool invocation:

    +-----------+     hit      +-----------+
    | cache     |------------->| return    |
    | (TTL)     |              +-----------+
    +-----------+
        | miss
        v
    +-----------+  open  +-----------+
    | breaker   |------->| fallback  |
    +-----------+        +-----------+
        | closed
        v
    +-------------------------+   fail/timeout   +-----------+
    | tool.call(params) with  |----------------->| fallback  |
    |   timeout guard         |                  +-----------+
    +-------------------------+
        | ok
        v
    +-----------+   +-----------+
    | cache put |-->| return    |
    +-----------+   +-----------+
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from config import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from monitor.performance_monitor import PerformanceMonitor


# =====================================================================
# Result shape
# =====================================================================
@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    degraded: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# Tool ABC
# =====================================================================
class Tool(ABC):
    name: str = "unnamed_tool"

    @abstractmethod
    def call(self, params: dict[str, Any]) -> ToolResult:
        """Do the real work. Raise on failure; ToolManager handles it."""

    def fallback(self, params: dict[str, Any], error: BaseException) -> ToolResult:
        """Degraded response when the tool cannot run. Override for domain-specific
        graceful degradation (e.g. return a canned answer)."""
        return ToolResult(
            ok=False,
            degraded=True,
            error=f"{type(error).__name__}: {error}",
            data=None,
        )


# =====================================================================
# TTL cache (in-process)
# =====================================================================
class TTLCache:
    def __init__(self, ttl: int) -> None:
        self._ttl = ttl
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() > expires_at:
                del self._data[key]
                return None
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.time() + self._ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# =====================================================================
# Circuit breaker (per-tool)
# =====================================================================
class CircuitBreaker:
    """Two-state breaker with time-based auto-recovery.

    State transitions:
        closed --(>= fail_threshold consecutive failures)--> open
        open   --(cooldown elapsed on next call check)-----> closed (probe)
    """

    def __init__(self, fail_threshold: int, cooldown_sec: int) -> None:
        self._fail_threshold = fail_threshold
        self._cooldown = cooldown_sec
        self._fail_count = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def should_reject(self) -> bool:
        """Return True if the caller should skip the tool right now."""
        with self._lock:
            if self._opened_at is None:
                return False
            if time.time() - self._opened_at >= self._cooldown:
                # Cooldown elapsed: allow one probe call, reset counters.
                self._opened_at = None
                self._fail_count = 0
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._fail_count = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._fail_count += 1
            if (
                self._fail_count >= self._fail_threshold
                and self._opened_at is None
            ):
                self._opened_at = time.time()
                logger.warning(
                    "Circuit opened after %s consecutive failures", self._fail_count
                )

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "open"


# =====================================================================
# Tool manager
# =====================================================================
class ToolManager:
    def __init__(
        self,
        tools: list[Tool] | None = None,
        *,
        monitor: PerformanceMonitor | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._cache = TTLCache(settings.tool_cache_ttl)
        # Simple counters for /monitor later.
        self._counters: dict[str, dict[str, int]] = {}
        self._monitor = monitor
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._breakers[tool.name] = CircuitBreaker(
            fail_threshold=settings.tool_circuit_fail_threshold,
            cooldown_sec=settings.tool_circuit_cooldown_sec,
        )
        self._counters[tool.name] = {
            "calls": 0,
            "hits": 0,
            "failures": 0,
            "rejected": 0,
            "cache_hits": 0,
        }

    def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(ok=False, error=f"unknown tool: {tool_name}")
        started = time.perf_counter()
        tool = self._tools[tool_name]
        breaker = self._breakers[tool_name]
        counters = self._counters[tool_name]
        counters["calls"] += 1

        # ---- Cache ----
        key = self._cache_key(tool_name, params)
        cached = self._cache.get(key)
        if cached is not None:
            counters["cache_hits"] += 1
            counters["hits"] += 1
            result = _clone_result(cached, extra_meta={"cache_hit": True})
            self._record_monitor(tool_name, result, started)
            return result

        # ---- Breaker ----
        if breaker.should_reject():
            counters["rejected"] += 1
            logger.warning("Tool %s rejected by open circuit", tool_name)
            result = tool.fallback(params, RuntimeError("circuit_open"))
            result.meta.setdefault("cache_hit", False)
            result.meta.setdefault("breaker", "open")
            self._record_monitor(tool_name, result, started)
            return result

        # ---- Execute with timeout ----
        start = time.time()
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(tool.call, params)
                result = future.result(timeout=settings.tool_timeout_sec)
        except FuturesTimeoutError as e:
            counters["failures"] += 1
            breaker.record_failure()
            logger.warning(
                "Tool %s timed out after %ss", tool_name, settings.tool_timeout_sec
            )
            fb = tool.fallback(params, e)
            fb.meta.update({"cache_hit": False, "reason": "timeout"})
            self._record_monitor(tool_name, fb, started)
            return fb
        except Exception as e:  # noqa: BLE001
            counters["failures"] += 1
            breaker.record_failure()
            logger.exception("Tool %s raised", tool_name)
            fb = tool.fallback(params, e)
            fb.meta.update({"cache_hit": False, "reason": "exception"})
            self._record_monitor(tool_name, fb, started)
            return fb

        elapsed_ms = int((time.time() - start) * 1000)
        result.meta.setdefault("cache_hit", False)
        result.meta.setdefault("elapsed_ms", elapsed_ms)

        if result.ok:
            counters["hits"] += 1
            breaker.record_success()
            self._cache.put(key, result)
        else:
            counters["failures"] += 1
            breaker.record_failure()

        self._record_monitor(tool_name, result, started)
        return result

    def _record_monitor(self, tool_name: str, result: ToolResult, started: float) -> None:
        if self._monitor is None:
            return
        self._monitor.record_tool(
            tool_name,
            success=result.ok,
            latency_ms=(time.perf_counter() - started) * 1000,
            degraded=result.degraded,
        )

    # ------------------------------------------------------------------
    # Introspection (used by /monitor and /health later)
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "tools": list(self._tools.keys()),
            "breakers": {n: b.state for n, b in self._breakers.items()},
            "counters": {n: dict(c) for n, c in self._counters.items()},
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(tool_name: str, params: dict[str, Any]) -> str:
        blob = json.dumps({"t": tool_name, "p": params}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _clone_result(result: ToolResult, extra_meta: dict[str, Any]) -> ToolResult:
    """Return a shallow copy so the cached value isn't mutated by callers."""
    return ToolResult(
        ok=result.ok,
        data=result.data,
        error=result.error,
        degraded=result.degraded,
        meta={**result.meta, **extra_meta},
    )

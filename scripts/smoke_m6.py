"""Smoke test for M6 monitoring and routing penalty feedback."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.agent_orchestrator import AgentOrchestrator
from mcp.tool_manager import Tool, ToolManager, ToolResult
from monitor.performance_monitor import PerformanceMonitor


class SuccessfulTool(Tool):
    name = "successful_tool"

    def call(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(ok=True, data=params)


def main() -> None:
    monitor = PerformanceMonitor(min_samples=5)
    for _ in range(5):
        monitor.record_agent("technical", success=False, latency_ms=4_500)
        monitor.record_agent("billing", success=True, latency_ms=100)
    tools = ToolManager([SuccessfulTool()], monitor=monitor)
    assert tools.execute("successful_tool", {"query": "hello"}).ok is True

    penalty = monitor.routing_penalty("technical")
    assert 0.45 <= penalty <= 0.60, penalty
    assert monitor.routing_penalty("billing") == 0.0

    summary = monitor.summary()
    technical = summary["components"]["agent:technical"]
    assert technical["success_rate"] == 0.0
    assert technical["avg_latency_ms"] == 4_500.0
    assert any(alert["type"] == "low_success_rate" for alert in technical["alerts"])
    assert any(alert["type"] == "high_latency" for alert in technical["alerts"])
    assert summary["components"]["tool:successful_tool"]["success_rate"] == 1.0

    orchestrator = AgentOrchestrator(monitor=monitor)
    result = orchestrator.handle("I get a 401 error when I log in and was charged twice")
    assert result.routing.primary_agent == "billing", result.routing.to_dict()
    assert result.routing.supporting_agents == ["technical"]
    assert result.routing.monitor_penalties["technical"] == penalty

    print("M6 SMOKE TEST OK")


if __name__ == "__main__":
    main()

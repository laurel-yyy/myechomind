"""Smoke test for M5 multi-agent routing and Skill-scoped responses."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.agent_orchestrator import AgentOrchestrator


def main() -> None:
    orchestrator = AgentOrchestrator()

    billing = orchestrator.handle("I was charged twice and need a refund for order AB12345678")
    assert billing.routing.primary_agent == "billing"
    assert billing.primary_response.agent_type == "billing"
    assert billing.primary_response.matched_skills == ["billing_support"]
    assert billing.intent.entities["order_id"] == ["AB12345678"]

    compound = orchestrator.handle("I get a 401 error when I log in and was charged twice")
    assert compound.routing.primary_agent == "technical"
    assert compound.routing.supporting_agents == ["billing"]
    assert compound.primary_response.matched_skills == ["technical_support"]
    assert compound.supporting_responses[0].matched_skills == ["billing_support"]

    escalation = orchestrator.handle("Please transfer me to a human agent")
    assert escalation.routing.escalation_requested is True
    assert escalation.routing.primary_agent == "general"
    assert "human-agent escalation" in escalation.answer

    print("M5 SMOKE TEST OK")


if __name__ == "__main__":
    main()

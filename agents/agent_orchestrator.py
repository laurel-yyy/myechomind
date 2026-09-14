"""Structured routing and multi-agent response orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from agents.base_agent import AgentResponse, BaseAgent
from agents.billing_agent import BillingAgent
from agents.general_agent import GeneralAgent
from agents.technical_agent import TechnicalAgent
from core.intent_recognizer import IntentRecognizer, IntentResult
from core.intents import INTENT_CATALOG
from core.skill_loader import SkillManager


@dataclass
class RoutingDecision:
    """Auditable decision emitted before any specialist is invoked."""

    primary_agent: str
    supporting_agents: list[str]
    intent: str
    intent_group: str
    routing_confidence: float
    routing_reason: str
    escalation_requested: bool = False
    domain_signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestrationResult:
    """Complete output of one multi-agent turn."""

    answer: str
    routing: RoutingDecision
    intent: IntentResult
    primary_response: AgentResponse
    supporting_responses: list[AgentResponse] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "routing": self.routing.to_dict(),
            "intent": self.intent.to_dict(),
            "primary_response": self.primary_response.to_dict(),
            "supporting_responses": [response.to_dict() for response in self.supporting_responses],
        }


class AgentOrchestrator:
    """Routes a request to a primary specialist and optional domain supporters."""

    _AGENT_BY_GROUP = {
        "general": "general",
        "technical": "technical",
        "billing": "billing",
        # Escalation is a workflow signal, not a separate LLM specialist.
        "escalation": "general",
    }

    def __init__(
        self,
        *,
        recognizer: IntentRecognizer | None = None,
        skill_manager: SkillManager | None = None,
        agents: dict[str, BaseAgent] | None = None,
    ) -> None:
        self._recognizer = recognizer or IntentRecognizer()
        skills = skill_manager or SkillManager()
        self._agents = agents or {
            "general": GeneralAgent(skill_manager=skills),
            "technical": TechnicalAgent(skill_manager=skills),
            "billing": BillingAgent(skill_manager=skills),
        }

    def route(self, user_message: str, context: str = "") -> tuple[IntentResult, RoutingDecision]:
        """Recognize intent and produce a deterministic structured route."""
        intent = self._recognizer.recognize(user_message, context)
        primary_agent = self._AGENT_BY_GROUP[intent.group]
        domain_signals = self._domain_signals(user_message)
        active_domains = [group for group, score in domain_signals.items() if score > 0.0]
        compound_override = False
        if len(active_domains) >= 2:
            strongest_group = max(active_domains, key=domain_signals.get)
            strongest_agent = self._AGENT_BY_GROUP[strongest_group]
            if strongest_agent != primary_agent:
                # A single intent classifier necessarily returns one label. For a
                # compound request, direct evidence across domains is a better
                # primary-work selector than that one-label constraint.
                primary_agent = strongest_agent
                compound_override = True
        supporters = self._supporting_agents(primary_agent, domain_signals)
        escalation = intent.group == "escalation"
        reason = (
            f"intent={intent.intent} group={intent.group} -> primary={primary_agent}; "
            f"secondary keyword domains={','.join(supporters) or 'none'}; "
            f"compound_override={compound_override}"
        )
        return intent, RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporters,
            intent=intent.intent,
            intent_group=intent.group,
            routing_confidence=intent.confidence,
            routing_reason=reason,
            escalation_requested=escalation,
            domain_signals=domain_signals,
        )

    def handle(
        self,
        user_message: str,
        *,
        memory_context: dict[str, Any] | None = None,
        knowledge_docs: list[dict[str, Any]] | None = None,
    ) -> OrchestrationResult:
        """Route, invoke selected agents, and merge their specialist responses."""
        intent, routing = self.route(user_message, self._context_text(memory_context or {}))
        selected = [routing.primary_agent, *routing.supporting_agents]
        kwargs = {
            "intent": intent.intent,
            "entities": intent.entities,
            "memory_context": memory_context or {},
            "knowledge_docs": knowledge_docs or [],
        }
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {
                agent_name: executor.submit(self._agents[agent_name].respond, user_message, **kwargs)
                for agent_name in selected
            }
            primary = futures[routing.primary_agent].result()
            supporting = [futures[name].result() for name in routing.supporting_agents]
        return OrchestrationResult(
            answer=self._merge(primary, supporting, routing.escalation_requested),
            routing=routing,
            intent=intent,
            primary_response=primary,
            supporting_responses=supporting,
        )

    @staticmethod
    def _domain_signals(user_message: str) -> dict[str, float]:
        """Measure explicit keywords by domain for compound-request detection."""
        message = user_message.lower()
        scores: dict[str, float] = {"general": 0.0, "technical": 0.0, "billing": 0.0}
        counts: dict[str, int] = {"general": 0, "technical": 0, "billing": 0}
        for spec in INTENT_CATALOG:
            if spec.group not in scores or not spec.keywords:
                continue
            hits = sum(1 for keyword in spec.keywords if keyword.lower() in message)
            scores[spec.group] += hits / len(spec.keywords)
            counts[spec.group] += 1
        return {
            group: round(score / max(1, counts[group]), 4)
            for group, score in scores.items()
        }

    def _supporting_agents(self, primary_agent: str, domain_signals: dict[str, float]) -> list[str]:
        candidates = [
            (self._AGENT_BY_GROUP[group], score)
            for group, score in domain_signals.items()
            if self._AGENT_BY_GROUP[group] != primary_agent and score > 0.0
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        # At most two supporters; every selected supporter has direct lexical evidence.
        return [agent for agent, _ in candidates[:2]]

    @staticmethod
    def _context_text(memory_context: dict[str, Any]) -> str:
        return str(memory_context)[:2_000] if memory_context else ""

    @staticmethod
    def _merge(
        primary: AgentResponse,
        supporting: list[AgentResponse],
        escalation_requested: bool,
    ) -> str:
        answer = primary.answer
        if supporting:
            supplements = "\n\n".join(
                f"Additional {response.agent_type} guidance: {response.answer}"
                for response in supporting
            )
            answer = f"{answer}\n\n{supplements}"
        if escalation_requested:
            answer += "\n\nYour request has been marked for human-agent escalation."
        return answer

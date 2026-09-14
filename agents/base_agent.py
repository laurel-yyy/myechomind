"""Common implementation shared by all domain-specialist agents."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from core.llm_client import LLMClient, get_llm_client
from core.skill_loader import SkillManager

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """A specialist response plus the diagnostic inputs used to create it."""

    agent_type: str
    answer: str
    matched_skills: list[str] = field(default_factory=list)
    system_prompt: str = ""
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseAgent:
    """Base class that injects scoped Skills and invokes the shared LLM client."""

    agent_type = "base"
    role_description = "general customer-support specialist"

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        skill_manager: SkillManager | None = None,
    ) -> None:
        self._llm = llm or get_llm_client()
        self._skills = skill_manager or SkillManager()

    def respond(
        self,
        user_message: str,
        *,
        intent: str,
        entities: dict[str, list[str]] | None = None,
        memory_context: dict[str, Any] | None = None,
        knowledge_docs: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        """Generate a response from domain rules and explicitly supplied context."""
        skill_prompt = self._skills.render_prompt(self.agent_type, user_message)
        matched_skills = [skill.name for skill in self._skills.match(self.agent_type, user_message)]
        system = self._build_system_prompt(
            intent=intent,
            entities=entities or {},
            memory_context=memory_context or {},
            knowledge_docs=knowledge_docs or [],
            skill_prompt=skill_prompt,
        )
        degraded = False
        try:
            answer = self._llm.chat(
                system=system,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=700,
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            # Preserve the /chat SLA: any LLM transport / auth / rate-limit error
            # turns into a degraded reply so downstream memory writes still run
            # and the monitor sees a documented failure.
            logger.warning("Agent %s: LLM chat failed, returning degraded reply: %s", self.agent_type, exc)
            degraded = True
            answer = (
                f"[{self.agent_type} agent] The upstream language model is temporarily "
                f"unavailable ({type(exc).__name__}). Please retry shortly; the rest of "
                "the pipeline (intent detection, knowledge retrieval, routing) is still working."
            )
        return AgentResponse(
            agent_type=self.agent_type,
            answer=answer,
            matched_skills=matched_skills,
            system_prompt=system,
            degraded=degraded,
        )

    def _build_system_prompt(
        self,
        *,
        intent: str,
        entities: dict[str, list[str]],
        memory_context: dict[str, Any],
        knowledge_docs: list[dict[str, Any]],
        skill_prompt: str,
    ) -> str:
        context_parts = [
            "You are the " + self.role_description + ".",
            "Answer accurately, concisely, and empathetically.",
            f"Detected intent: {intent}.",
            f"Extracted entities: {entities or 'none'}.",
        ]
        if memory_context:
            context_parts.append(f"Conversation context: {memory_context}.")
        if knowledge_docs:
            evidence = "\n".join(
                f"- {doc.get('text', '')[:700]}" for doc in knowledge_docs[:5]
            )
            context_parts.append("Knowledge context (do not invent beyond it):\n" + evidence)
        if skill_prompt:
            context_parts.append(skill_prompt)
        return "\n\n".join(context_parts)

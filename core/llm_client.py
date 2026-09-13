"""LLM abstraction layer.

Downstream modules (intent recognizer, RAG rerank, judge, agents) call these
semantic methods instead of raw `messages.create`. Two backends are provided:

    - MockLLMClient : deterministic rule-based responses (no network, no key).
    - RealLLMClient : Anthropic-compatible HTTP calls (also works for DeepSeek).

Switching between them is a `.env` change (LLM_MODE=mock|real). Business code
never touches the SDK directly, so no other module changes when we swap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


# =====================================================================
# Abstract interface
# =====================================================================
class LLMClient(ABC):
    """Semantic LLM interface shared by mock and real implementations."""

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Return the assistant's free-form reply."""

    @abstractmethod
    def classify_intent(
        self,
        user_message: str,
        candidates: list[dict[str, Any]],
        context: str = "",
    ) -> dict[str, Any]:
        """Classify user_message against candidates.

        Each candidate is expected to look like:
            {"intent": "refund", "group": "billing", "keywords": [...], "description": "..."}

        Returns:
            {
                "intent": str,
                "group": str,
                "confidence": float in [0,1],
                "reasoning": str,
                "urgency": "low"|"medium"|"high",
                "entities": {...}
            }
        """

    @abstractmethod
    def rewrite_query(self, query: str, n: int = 3) -> list[str]:
        """Return `n` alternative phrasings that preserve intent."""

    @abstractmethod
    def rerank(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        """Return a relevance score in [0,1] for each doc.

        Each doc: {"id": str, "text": str, ...}
        """

    @abstractmethod
    def judge(
        self,
        question: str,
        answer: str,
        reference: str | None = None,
    ) -> dict[str, Any]:
        """LLM-as-Judge scoring for the evaluation pipeline.

        Returns: {"score": float in [0,10], "pass": bool, "comments": str}
        """


# =====================================================================
# Mock implementation (deterministic, dependency-free)
# =====================================================================
_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def _tokenize(text: str) -> list[str]:
    """Very small tokenizer: split into ASCII alphanumerics and CJK characters."""
    tokens: list[str] = []
    for match in _TOKEN_RE.findall((text or "").lower()):
        if re.match(r"^[\u4e00-\u9fff]+$", match):
            tokens.extend(list(match))  # CJK: split into single characters
        else:
            tokens.append(match)
    return tokens


def _overlap_score(a: list[str], b: list[str]) -> float:
    """Symmetric token-overlap score in [0,1]."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


class MockLLMClient(LLMClient):
    """Rule-based stand-in that lets us build the whole pipeline without an API key."""

    URGENT_KEYWORDS = ("紧急", "崩溃", "投诉", "urgent", "asap", "immediately")
    HIGH_URGENCY_INTENTS = {"crash", "duplicate_charge", "human_handoff"}

    ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
        "order_id": re.compile(r"\b([A-Z]{2,}\d{6,}|\d{10,})\b"),
        "amount": re.compile(r"([¥$]\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?元)"),
        "error_code": re.compile(r"\b(?:HTTP\s?)?(4\d{2}|5\d{2}|E\d{3,})\b", re.I),
        "date": re.compile(r"\b(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)\b"),
    }

    # -------- chat --------
    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        role_hint = ""
        if "billing" in (system or "").lower() or "退款" in (system or ""):
            role_hint = "[Billing agent]"
        elif "technical" in (system or "").lower() or "技术" in (system or ""):
            role_hint = "[Technical agent]"
        elif "general" in (system or "").lower() or "客服" in (system or ""):
            role_hint = "[General agent]"
        return (
            f"{role_hint} (mock reply) I received your message: '{last_user[:120]}'. "
            "In real mode this would be a Claude response."
        ).strip()

    # -------- classify_intent --------
    def classify_intent(
        self,
        user_message: str,
        candidates: list[dict[str, Any]],
        context: str = "",
    ) -> dict[str, Any]:
        # Approximate what a real LLM does: strongly reward candidate keywords
        # that appear verbatim in the message, and use description-token overlap
        # only as a weak tiebreaker. This keeps single-word high-signal intents
        # (e.g. "refund") from losing to broader intents that happen to share a
        # generic token (e.g. "order").
        user_lower = (user_message or "").lower()
        user_tokens = _tokenize(user_message)
        best_intent = candidates[0] if candidates else {"intent": "unknown", "group": "general"}
        best_score = 0.0
        scores: list[tuple[float, dict[str, Any]]] = []
        for c in candidates:
            keywords = [kw for kw in c.get("keywords", []) if kw]
            kw_hits = sum(1 for kw in keywords if kw.lower() in user_lower)
            kw_score = kw_hits / len(keywords) if keywords else 0.0
            desc_score = _overlap_score(user_tokens, _tokenize(c.get("description", "")))
            score = kw_score * 0.7 + desc_score * 0.3
            scores.append((score, c))
            if score > best_score:
                best_score = score
                best_intent = c

        # Confidence: raw score + margin over runner-up.
        scores.sort(key=lambda x: x[0], reverse=True)
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        margin = max(0.0, best_score - runner_up)
        confidence = min(0.99, 0.35 + best_score * 0.5 + margin * 0.5)

        urgency = "medium"
        if any(kw in user_message.lower() for kw in self.URGENT_KEYWORDS):
            urgency = "high"
        elif best_intent.get("intent") in self.HIGH_URGENCY_INTENTS:
            urgency = "high"
        elif best_intent.get("intent") in {"greeting", "farewell", "feedback"}:
            urgency = "low"

        entities: dict[str, list[str]] = {}
        for name, pattern in self.ENTITY_PATTERNS.items():
            hits = pattern.findall(user_message or "")
            if hits:
                entities[name] = list({h for h in hits})

        return {
            "intent": best_intent.get("intent", "unknown"),
            "group": best_intent.get("group", "general"),
            "confidence": round(confidence, 3),
            "reasoning": f"mock: keyword overlap={best_score:.2f}, margin={margin:.2f}",
            "urgency": urgency,
            "entities": entities,
        }

    # -------- rewrite_query --------
    def rewrite_query(self, query: str, n: int = 3) -> list[str]:
        base = (query or "").strip()
        if not base:
            return []
        templates = [
            "{q}",
            "What does '{q}' mean and how to handle it?",
            "Please explain: {q}",
            "Details about {q}",
            "How to resolve {q}",
        ]
        seen: list[str] = []
        for tpl in templates:
            variant = tpl.format(q=base)
            if variant not in seen:
                seen.append(variant)
            if len(seen) >= n:
                break
        return seen[:n]

    # -------- rerank --------
    def rerank(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        q_tokens = _tokenize(query)
        scores: list[float] = []
        for d in docs:
            d_tokens = _tokenize(d.get("text", ""))
            scores.append(round(_overlap_score(q_tokens, d_tokens), 4))
        return scores

    # -------- judge --------
    def judge(
        self,
        question: str,
        answer: str,
        reference: str | None = None,
    ) -> dict[str, Any]:
        base_score = 6.5
        if reference:
            overlap = _overlap_score(_tokenize(reference), _tokenize(answer))
            base_score = 4.0 + overlap * 5.0
        length_bonus = 0.5 if 40 <= len(answer or "") <= 800 else 0.0
        score = round(min(10.0, base_score + length_bonus), 2)
        return {
            "score": score,
            "pass": score >= 6.0,
            "comments": (
                f"mock judge: base={base_score:.2f}, length_bonus={length_bonus:.1f}. "
                "Enable real LLM for meaningful qualitative review."
            ),
        }


# =====================================================================
# Real implementation (Anthropic-compatible)
# =====================================================================
class RealLLMClient(LLMClient):
    """Thin wrapper around anthropic.Anthropic that works with any base_url
    exposing the Anthropic protocol (Anthropic official, DeepSeek, etc.).
    """

    def __init__(self) -> None:
        # Import lazily so mock users never hit an SDK import error.
        from anthropic import Anthropic

        if not settings.anthropic_api_key:
            raise RuntimeError(
                "LLM_MODE=real requires ANTHROPIC_API_KEY to be set in .env"
            )
        self._client = Anthropic(
            api_key=settings.anthropic_api_key,
            base_url=settings.anthropic_base_url,
        )
        self._model = settings.anthropic_model

    # -------- chat --------
    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        # Anthropic returns a list of content blocks; join text blocks.
        parts: list[str] = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()

    # -------- shared JSON prompt helper --------
    def _ask_json(self, system: str, user: str, max_tokens: int = 512) -> dict[str, Any]:
        raw = self.chat(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        # Try direct parse first, then extract the first {...} block.
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            logger.warning("LLM returned non-JSON payload: %s", raw[:200])
            return {}

    # -------- classify_intent --------
    def classify_intent(
        self,
        user_message: str,
        candidates: list[dict[str, Any]],
        context: str = "",
    ) -> dict[str, Any]:
        system = (
            "You are an intent classifier for a customer-support platform. "
            "Return STRICT JSON only, no prose."
        )
        cand_desc = "\n".join(
            f"- {c['intent']} (group={c.get('group','general')}): {c.get('description','')}"
            for c in candidates
        )
        user_prompt = (
            f"Context: {context or 'none'}\n"
            f"User message: {user_message}\n\n"
            f"Candidate intents:\n{cand_desc}\n\n"
            "Return JSON with fields: intent, group, confidence(0-1), reasoning, "
            "urgency(low|medium|high), entities(object)."
        )
        result = self._ask_json(system, user_prompt, max_tokens=400)
        if not result:
            return MockLLMClient().classify_intent(user_message, candidates, context)
        result.setdefault("urgency", "medium")
        result.setdefault("entities", {})
        return result

    # -------- rewrite_query --------
    def rewrite_query(self, query: str, n: int = 3) -> list[str]:
        system = (
            "Rewrite a user query into diverse alternative phrasings that preserve intent. "
            "Return STRICT JSON: {\"rewrites\": [\"...\", \"...\"]}"
        )
        user = f"Query: {query}\nProduce {n} rewrites."
        result = self._ask_json(system, user, max_tokens=300)
        rewrites = result.get("rewrites", []) if isinstance(result, dict) else []
        return [str(r) for r in rewrites][:n] or [query]

    # -------- rerank --------
    def rerank(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        if not docs:
            return []
        system = (
            "Score each document for relevance to the query on a 0-1 scale. "
            "Return STRICT JSON: {\"scores\": [0.9, 0.4, ...]} in the same order."
        )
        payload = "\n".join(f"[{i}] {d.get('text','')[:400]}" for i, d in enumerate(docs))
        user = f"Query: {query}\n\nDocuments:\n{payload}"
        result = self._ask_json(system, user, max_tokens=200)
        scores = result.get("scores", []) if isinstance(result, dict) else []
        if len(scores) != len(docs):
            # Fallback to keyword overlap if the model returned a mismatched list.
            return MockLLMClient().rerank(query, docs)
        return [float(s) for s in scores]

    # -------- judge --------
    def judge(
        self,
        question: str,
        answer: str,
        reference: str | None = None,
    ) -> dict[str, Any]:
        system = (
            "You are an expert judge scoring a customer-support answer. "
            "Return STRICT JSON: {\"score\": 0-10, \"pass\": bool, \"comments\": \"...\"}"
        )
        ref_block = f"Reference answer: {reference}\n\n" if reference else ""
        user = (
            f"Question: {question}\n\nCandidate answer: {answer}\n\n{ref_block}"
            "Score correctness, helpfulness, and tone. Pass threshold = 6."
        )
        result = self._ask_json(system, user, max_tokens=300)
        if not result:
            return MockLLMClient().judge(question, answer, reference)
        result.setdefault("pass", float(result.get("score", 0)) >= 6.0)
        result.setdefault("comments", "")
        return result


# =====================================================================
# Factory
# =====================================================================
_INSTANCE: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return a cached singleton client according to settings.llm_mode."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    if settings.llm_mode == "real":
        logger.info("Instantiating RealLLMClient (model=%s)", settings.anthropic_model)
        _INSTANCE = RealLLMClient()
    else:
        logger.info("Instantiating MockLLMClient")
        _INSTANCE = MockLLMClient()
    return _INSTANCE


def reset_llm_client() -> None:
    """Testing helper: drop the cached client so a new one is built on next call."""
    global _INSTANCE
    _INSTANCE = None

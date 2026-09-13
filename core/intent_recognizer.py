"""Three-path fusion intent recognizer.

For each user message we run three parallel classifiers and combine them:

    LLM       (weight 0.7)  primary semantic judgement; also produces
                             entities and an urgency estimate.
    Embedding (weight 0.2)  cosine-max against pre-embedded examples per
                             intent, softmax-normalized across intents.
    Keyword   (weight 0.1)  substring match against curated keyword lists
                             per intent, normalized so scores sum to 1.

Fused score = 0.7 * LLM + 0.2 * Embedding + 0.1 * Keyword. Argmax is the
final intent; the winning fused score is reported as `confidence`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from core.embedding_client import cosine, get_embedding_client
from core.intents import INTENT_CATALOG, IntentSpec, get_spec
from core.llm_client import get_llm_client


@dataclass
class IntentResult:
    intent: str
    group: str
    confidence: float
    urgency: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    reasoning: str = ""
    fusion_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntentRecognizer:
    LLM_WEIGHT = 0.70
    EMB_WEIGHT = 0.20
    KW_WEIGHT = 0.10

    def __init__(self, specs: tuple[IntentSpec, ...] | None = None) -> None:
        self._llm = get_llm_client()
        self._emb = get_embedding_client()
        self._specs: tuple[IntentSpec, ...] = specs or INTENT_CATALOG
        self._example_vectors = self._precompute_example_vectors()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recognize(self, user_message: str, context: str = "") -> IntentResult:
        llm_scores, llm_full = self._llm_path(user_message, context)
        emb_scores = self._embedding_path(user_message)
        kw_scores = self._keyword_path(user_message)

        fused: dict[str, float] = {}
        for spec in self._specs:
            i = spec.intent
            fused[i] = (
                self.LLM_WEIGHT * llm_scores.get(i, 0.0)
                + self.EMB_WEIGHT * emb_scores.get(i, 0.0)
                + self.KW_WEIGHT * kw_scores.get(i, 0.0)
            )

        winner_intent = max(fused, key=fused.get)
        winner_spec = get_spec(winner_intent)

        return IntentResult(
            intent=winner_intent,
            group=winner_spec.group,
            confidence=round(fused[winner_intent], 4),
            urgency=str(llm_full.get("urgency") or winner_spec.urgency_bias),
            entities=dict(llm_full.get("entities") or {}),
            reasoning=(
                f"fusion -> llm={llm_scores.get(winner_intent, 0):.2f} "
                f"emb={emb_scores.get(winner_intent, 0):.2f} "
                f"kw={kw_scores.get(winner_intent, 0):.2f}"
            ),
            fusion_scores={
                "llm": llm_scores,
                "embedding": emb_scores,
                "keyword": kw_scores,
                "fused": fused,
            },
        )

    # ------------------------------------------------------------------
    # Path 1: LLM
    # ------------------------------------------------------------------
    def _llm_path(
        self, user_message: str, context: str
    ) -> tuple[dict[str, float], dict[str, Any]]:
        candidates = [s.to_candidate() for s in self._specs]
        result = self._llm.classify_intent(user_message, candidates, context)
        scores = {s.intent: 0.0 for s in self._specs}
        chosen = str(result.get("intent") or "unknown")
        if chosen not in scores:
            chosen = "unknown"
        scores[chosen] = float(result.get("confidence") or 0.5)
        return scores, result

    # ------------------------------------------------------------------
    # Path 2: Embedding (cosine-max vs examples, softmax across intents)
    # ------------------------------------------------------------------
    def _precompute_example_vectors(self) -> dict[str, list[list[float]]]:
        out: dict[str, list[list[float]]] = {}
        for spec in self._specs:
            if not spec.examples:
                continue
            out[spec.intent] = self._emb.embed(list(spec.examples))
        return out

    def _embedding_path(self, user_message: str) -> dict[str, float]:
        vec = self._emb.embed([user_message])[0]
        raw: dict[str, float] = {}
        for spec in self._specs:
            vectors = self._example_vectors.get(spec.intent, [])
            if not vectors:
                raw[spec.intent] = 0.0
                continue
            raw[spec.intent] = max(cosine(vec, ev) for ev in vectors)
        return _softmax(raw, temperature=0.1)

    # ------------------------------------------------------------------
    # Path 3: Keyword matching (normalized so scores sum to 1)
    # ------------------------------------------------------------------
    def _keyword_path(self, user_message: str) -> dict[str, float]:
        msg_lower = user_message.lower()
        raw: dict[str, float] = {}
        for spec in self._specs:
            if not spec.keywords:
                raw[spec.intent] = 0.0
                continue
            hits = sum(1 for kw in spec.keywords if kw.lower() in msg_lower)
            raw[spec.intent] = hits / len(spec.keywords)
        return _normalize_sum_to_one(raw)


# ----------------------------------------------------------------------
# Numeric helpers
# ----------------------------------------------------------------------
def _softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    m = max(values)
    exps = {k: math.exp((v - m) / max(temperature, 1e-6)) for k, v in scores.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


def _normalize_sum_to_one(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        # Uniform fallback so keyword path never contributes NaN.
        n = len(scores) or 1
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}

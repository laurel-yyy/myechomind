"""Layered conversation memory: Redis working memory + ChromaDB long-term stores.

Layout
------
Redis
    wm:{user_id}:{conv_id}       list    recent chat turns, capped at N, TTL 1d
    summary:{user_id}:{conv_id}  string  compressed summary of older turns

ChromaDB
    episodic     collection   {user_id, conv_id, ts, ...}   semantic recall across sessions
    user_profile collection   {user_id, category, ts, ...}  durable user facts / preferences

All embeddings go through the shared EmbeddingClient via the chroma bridge, so
mock/real mode switches propagate automatically.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from config import settings

from .chroma_client import get_or_create_collection
from .redis_client import get_redis_client

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Facade over the working-memory + long-term-memory stack."""

    def __init__(self) -> None:
        self._redis = get_redis_client()
        self._episodic = get_or_create_collection(settings.chroma_collection_episodic)
        self._profile = get_or_create_collection(settings.chroma_collection_profile)

    # ------------------------------------------------------------------
    # Redis key helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    # ------------------------------------------------------------------
    # Working memory
    # ------------------------------------------------------------------
    def append_message(
        self,
        user_id: str,
        conv_id: str,
        role: str,
        content: str,
    ) -> None:
        """Append one turn and enforce the size cap + TTL atomically."""
        key = self._wm_key(user_id, conv_id)
        payload = json.dumps(
            {"role": role, "content": content, "ts": time.time()},
            ensure_ascii=False,
        )
        pipe = self._redis.pipeline()
        pipe.rpush(key, payload)
        pipe.ltrim(key, -settings.working_memory_max, -1)
        pipe.expire(key, settings.working_memory_ttl)
        pipe.execute()

    def get_working_memory(
        self,
        user_id: str,
        conv_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most recent `limit` turns in chronological order."""
        key = self._wm_key(user_id, conv_id)
        n = limit or settings.working_memory_max
        raw = self._redis.lrange(key, -n, -1)
        return [json.loads(item) for item in raw]

    def clear_working_memory(self, user_id: str, conv_id: str) -> None:
        self._redis.delete(self._wm_key(user_id, conv_id))

    # ------------------------------------------------------------------
    # Conversation summary
    # ------------------------------------------------------------------
    def set_summary(self, user_id: str, conv_id: str, summary: str) -> None:
        self._redis.setex(
            self._summary_key(user_id, conv_id),
            settings.conversation_summary_ttl,
            summary,
        )

    def get_summary(self, user_id: str, conv_id: str) -> str | None:
        return self._redis.get(self._summary_key(user_id, conv_id))

    # ------------------------------------------------------------------
    # Episodic memory (cross-session semantic recall)
    # ------------------------------------------------------------------
    def add_episodic(
        self,
        user_id: str,
        conv_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        doc_id = f"ep_{uuid.uuid4().hex[:12]}"
        meta = self._flatten_metadata(
            {"user_id": user_id, "conv_id": conv_id, "ts": time.time()},
            metadata,
        )
        self._episodic.add(ids=[doc_id], documents=[text], metadatas=[meta])
        return doc_id

    def search_episodic(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        res = self._episodic.query(
            query_texts=[query],
            n_results=top_k,
            where={"user_id": user_id},
        )
        return self._flatten_query_result(res)

    # ------------------------------------------------------------------
    # User profile (durable preferences / facts)
    # ------------------------------------------------------------------
    def upsert_profile_fact(
        self,
        user_id: str,
        category: str,
        fact: str,
        fact_id: str | None = None,
    ) -> str:
        doc_id = fact_id or f"pf_{uuid.uuid4().hex[:12]}"
        meta = {"user_id": user_id, "category": category, "ts": time.time()}
        self._profile.upsert(ids=[doc_id], documents=[fact], metadatas=[meta])
        return doc_id

    def get_profile(
        self,
        user_id: str,
        query: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantic search when a query is given, otherwise return latest facts."""
        if query:
            res = self._profile.query(
                query_texts=[query],
                n_results=top_k,
                where={"user_id": user_id},
            )
            return self._flatten_query_result(res)
        res = self._profile.get(where={"user_id": user_id}, limit=top_k)
        return [
            {"id": i, "text": d, "metadata": m, "distance": None}
            for i, d, m in zip(
                res.get("ids", []),
                res.get("documents", []),
                res.get("metadatas", []),
            )
        ]

    # ------------------------------------------------------------------
    # Aggregate context for the /chat pipeline
    # ------------------------------------------------------------------
    def build_context(
        self,
        user_id: str,
        conv_id: str,
        current_message: str,
    ) -> dict[str, Any]:
        return {
            "working_memory": self.get_working_memory(user_id, conv_id),
            "summary": self.get_summary(user_id, conv_id),
            "episodic": self.search_episodic(user_id, current_message, top_k=3),
            "profile": self.get_profile(user_id, current_message, top_k=3),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_metadata(
        base: dict[str, Any],
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Keep only Chroma-supported scalar metadata values."""
        meta = dict(base)
        if not extra:
            return meta
        for k, v in extra.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            elif v is None:
                continue
            else:
                meta[k] = json.dumps(v, ensure_ascii=False)
        return meta

    @staticmethod
    def _flatten_query_result(res: dict[str, Any]) -> list[dict[str, Any]]:
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists_all = res.get("distances")
        dists = dists_all[0] if dists_all else [None] * len(ids)
        return [
            {"id": i, "text": d, "metadata": m, "distance": dist}
            for i, d, m, dist in zip(ids, docs, metas, dists)
        ]

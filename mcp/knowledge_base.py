"""RAG knowledge base backed by the ChromaDB `knowledge_base` collection.

This is the low-level store: raw add/search only. The production RAG path
(query rewrite + parallel recall + LLM rerank) lives in KnowledgeSearchTool
so this layer stays simple to test and reuse.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from config import settings
from memory.chroma_client import get_or_create_collection

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self, collection_name: str | None = None) -> None:
        self._collection = get_or_create_collection(
            collection_name or settings.chroma_collection_kb
        )

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        doc_id = doc_id or f"kb_{uuid.uuid4().hex[:12]}"
        meta = self._flatten_metadata(metadata)
        meta.setdefault("ts", time.time())
        self._collection.add(ids=[doc_id], documents=[text], metadatas=[meta])
        return doc_id

    def add_batch(self, docs: list[dict[str, Any]]) -> list[str]:
        """Batch insert. Each doc: {"text": str, "metadata": dict, "id": str?}"""
        if not docs:
            return []
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        for d in docs:
            doc_id = d.get("id") or f"kb_{uuid.uuid4().hex[:12]}"
            meta = self._flatten_metadata(d.get("metadata"))
            meta.setdefault("ts", time.time())
            ids.append(doc_id)
            texts.append(d["text"])
            metas.append(meta)
        self._collection.add(ids=ids, documents=texts, metadatas=metas)
        return ids

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        res = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists_all = res.get("distances")
        dists = dists_all[0] if dists_all else [None] * len(ids)
        return [
            {"id": i, "text": t, "metadata": m or {}, "distance": d}
            for i, t, m, d in zip(ids, docs, metas, dists)
        ]

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {"count": self._collection.count(), "name": self._collection.name}

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
        """Chroma only stores scalar metadata values; drop or coerce the rest."""
        if not meta:
            return {}
        import json as _json

        out: dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif v is None:
                continue
            else:
                out[k] = _json.dumps(v, ensure_ascii=False)
        return out

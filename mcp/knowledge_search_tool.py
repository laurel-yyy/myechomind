"""Production RAG tool: query rewrite -> parallel recall -> LLM rerank.

Wraps KnowledgeBase. Registered with ToolManager so it inherits cache /
breaker / timeout / fallback for free.
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings
from core import get_llm_client

from .knowledge_base import KnowledgeBase
from .tool_manager import Tool, ToolResult

logger = logging.getLogger(__name__)


class KnowledgeSearchTool(Tool):
    name = "knowledge_search"

    def __init__(self, kb: KnowledgeBase | None = None) -> None:
        self._kb = kb or KnowledgeBase()
        self._llm = get_llm_client()

    def call(self, params: dict[str, Any]) -> ToolResult:
        query = str(params.get("query") or "").strip()
        if not query:
            return ToolResult(ok=False, error="missing 'query' param")
        top_k = int(params.get("top_k") or settings.rag_topk)
        rewrite_n = int(params.get("rewrite_n") or settings.rag_rewrite_n)
        rerank = bool(params.get("rerank", settings.rag_rerank_enabled))
        where = params.get("where") or None

        # 1. Rewrite the query into multiple perspectives, always keeping the original.
        rewrites: list[str] = []
        if rewrite_n > 1:
            try:
                rewrites = self._llm.rewrite_query(query, n=rewrite_n)
            except Exception:  # noqa: BLE001
                logger.exception("query rewrite failed; falling back to original only")
                rewrites = []
        all_queries = _unique_preserve_order([query, *rewrites])

        # 2. Recall for each variant, dedupe hits by id.
        docs_by_id: dict[str, dict[str, Any]] = {}
        for q in all_queries:
            for hit in self._kb.search(q, top_k=top_k, where=where):
                if hit["id"] not in docs_by_id:
                    docs_by_id[hit["id"]] = hit
        docs = list(docs_by_id.values())

        # 3. Optional LLM rerank on the merged candidates, then take top_k.
        if rerank and docs:
            try:
                scores = self._llm.rerank(query, docs)
                for d, s in zip(docs, scores):
                    d["rerank_score"] = float(s)
                docs.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
            except Exception:  # noqa: BLE001
                logger.exception("rerank failed; falling back to recall order")

        return ToolResult(
            ok=True,
            data={
                "query": query,
                "rewrites": rewrites,
                "docs": docs[:top_k],
            },
            meta={
                "variants": len(all_queries),
                "candidates": len(docs),
                "rerank": rerank,
            },
        )

    def fallback(self, params: dict[str, Any], error: BaseException) -> ToolResult:
        return ToolResult(
            ok=False,
            degraded=True,
            error=f"{type(error).__name__}: {error}",
            data={
                "query": params.get("query", ""),
                "rewrites": [],
                "docs": [],
                "message": (
                    "Knowledge search is temporarily unavailable. "
                    "The agent will answer from general knowledge only."
                ),
            },
        )


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

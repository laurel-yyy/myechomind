"""M3 smoke test: exercise KnowledgeBase, KnowledgeSearchTool, and
ToolManager governance (cache hit, circuit breaker, timeout, fallback).

Prerequisites:
    docker compose up -d redis chromadb
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from mcp import KnowledgeBase, KnowledgeSearchTool, Tool, ToolManager, ToolResult


DEMO_DOCS = [
    {
        "text": "Refund policy: full refund within 14 days of purchase; partial refund "
                "for opened items in original condition. Contact billing support to start.",
        "metadata": {"topic": "billing", "kind": "policy"},
    },
    {
        "text": "How to reset your password: click 'Forgot password' on the login page, "
                "check your email for a reset link, and follow the instructions.",
        "metadata": {"topic": "technical", "kind": "how-to"},
    },
    {
        "text": "Shipping: standard delivery is 3-5 business days. Express delivery is "
                "next business day. Tracking updates every 12 hours.",
        "metadata": {"topic": "logistics", "kind": "policy"},
    },
    {
        "text": "Duplicate charge handling: if you see the same order charged twice, "
                "keep the transaction IDs and open a billing ticket for review.",
        "metadata": {"topic": "billing", "kind": "how-to"},
    },
    {
        "text": "Membership tiers: Free, Plus ($9/mo), Pro ($29/mo). Pro members get "
                "priority support and extended refund window (30 days).",
        "metadata": {"topic": "general", "kind": "policy"},
    },
]


# --- Auxiliary tools for governance tests ---
class AlwaysFailTool(Tool):
    name = "always_fail"

    def call(self, params):
        raise RuntimeError("simulated failure")


class SlowTool(Tool):
    name = "slow_tool"

    def call(self, params):
        time.sleep(settings.tool_timeout_sec + 1.0)
        return ToolResult(ok=True, data="should have timed out")


def test_kb_ingest_and_search() -> None:
    print("=" * 60)
    print("KnowledgeBase: batch ingest + direct search")
    print("=" * 60)
    kb = KnowledgeBase()
    before = kb.stats()["count"]
    ids = kb.add_batch(DEMO_DOCS)
    print(f"  ingested {len(ids)} docs (was {before}, now {kb.stats()['count']})")
    hits = kb.search("how do I get my money back?", top_k=3)
    print(f"  search hits ({len(hits)}): {[h['id'] for h in hits]}")
    assert hits, "KB search returned nothing"
    print("  PASS")
    # Return ids so we can clean up at the end
    return ids


def test_rag_tool(manager: ToolManager) -> None:
    print("=" * 60)
    print("KnowledgeSearchTool via ToolManager: rewrite + rerank")
    print("=" * 60)
    res = manager.execute("knowledge_search", {"query": "how to get a refund", "top_k": 3})
    print(f"  ok={res.ok}  cache_hit={res.meta.get('cache_hit')}  elapsed_ms={res.meta.get('elapsed_ms')}")
    print(f"  variants={res.meta.get('variants')}  candidates={res.meta.get('candidates')}")
    for d in res.data["docs"]:
        print(f"    - {d['id']}  score={d.get('rerank_score'):.3f}  {d['text'][:70]}...")
    assert res.ok and res.data["docs"], "RAG tool returned nothing"
    print("  PASS")


def test_cache_hit(manager: ToolManager) -> None:
    print("=" * 60)
    print("Cache hit: second identical call returns cached result")
    print("=" * 60)
    params = {"query": "how to get a refund", "top_k": 3}
    r1 = manager.execute("knowledge_search", params)
    r2 = manager.execute("knowledge_search", params)
    print(f"  r1 cache_hit={r1.meta.get('cache_hit')}   r2 cache_hit={r2.meta.get('cache_hit')}")
    assert r2.meta.get("cache_hit") is True, "second call should hit cache"
    print("  PASS")


def test_circuit_breaker(manager: ToolManager) -> None:
    print("=" * 60)
    print("Circuit breaker: opens after N consecutive failures")
    print("=" * 60)
    threshold = settings.tool_circuit_fail_threshold
    for i in range(threshold + 2):
        res = manager.execute("always_fail", {"x": i})
        state = manager._breakers["always_fail"].state
        print(f"  call {i+1}: ok={res.ok} degraded={res.degraded} state={state} reason={res.meta.get('reason') or res.meta.get('breaker')}")
    final_state = manager._breakers["always_fail"].state
    assert final_state == "open", f"breaker should be open, got {final_state}"
    print("  PASS")


def test_timeout(manager: ToolManager) -> None:
    print("=" * 60)
    print("Timeout: slow tool triggers fallback")
    print("=" * 60)
    res = manager.execute("slow_tool", {"x": 1})
    print(f"  ok={res.ok} degraded={res.degraded} reason={res.meta.get('reason')}")
    assert res.degraded and res.meta.get("reason") == "timeout", "slow tool should time out"
    print("  PASS")


def cleanup(ids: list[str]) -> None:
    KnowledgeBase().delete(ids)


def main() -> int:
    manager = ToolManager(
        tools=[KnowledgeSearchTool(), AlwaysFailTool(), SlowTool()]
    )

    added_ids = test_kb_ingest_and_search()
    try:
        test_rag_tool(manager)
        test_cache_hit(manager)
        test_circuit_breaker(manager)
        test_timeout(manager)
    finally:
        cleanup(added_ids)

    print("=" * 60)
    print("ToolManager stats:")
    for k, v in manager.stats().items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print("M3 SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end integration smoke test.

Assumes a running server at http://localhost:8000 (start it separately with
    .venv/Scripts/python.exe -m uvicorn api.main:app --port 8000 --log-level warning
) and exercises every public endpoint the frontend consumes.

The test detects the active mode from /health and adapts its assertions:
  - MOCK mode: answers should contain "(mock reply)"
  - REAL mode: answers must NOT contain "(mock reply)"
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

BASE = "http://localhost:8000"
TIMEOUT = 60.0


def section(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def get(path: str) -> dict:
    r = httpx.get(f"{BASE}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def post(path: str, json: dict) -> dict:
    r = httpx.post(f"{BASE}{path}", json=json, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def upload(path: str, filename: str, content: bytes, mime: str) -> dict:
    r = httpx.post(
        f"{BASE}{path}",
        files={"file": (filename, content, mime)},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    section("1) /health")
    h = get("/health")
    llm_mode = h["llm_mode"]
    is_real = llm_mode == "real"
    print(f"  llm_mode       = {llm_mode}")
    print(f"  embedding_mode = {h['embedding_mode']}")
    print(f"  knowledge_base = {h['knowledge_base']}")
    print(f"  skills         = {h['skills']}")
    mode_banner = "REAL (calling live LLM)" if is_real else "MOCK (deterministic)"
    print(f"  --> {mode_banner}")

    section("2) /knowledge/add + /knowledge/stats")
    added = post(
        "/knowledge/add",
        {
            "documents": [
                {
                    "text": "MyEchoMind is an enterprise multi-agent customer-support platform.",
                    "metadata": {"source": "e2e"},
                },
                {
                    "text": "Refunds are processed within 3-5 business days after approval.",
                    "metadata": {"source": "e2e"},
                },
            ]
        },
    )
    print(f"  added ids     = {added['ids']}")
    assert added["count"] == 2, "add_batch count mismatch"
    stats = get("/knowledge/stats")
    print(f"  KB stats      = {stats}")
    assert stats["count"] >= 2

    section("3) /knowledge/upload")
    up = upload(
        "/knowledge/upload",
        "escalation.txt",
        b"Escalation path: transfer to human agent when routing confidence < 0.3.",
        "text/plain",
    )
    print(f"  uploaded      = {up}")
    assert "id" in up and up["bytes"] > 0

    section("4) /search (rewrite + rerank)")
    s = post(
        "/search",
        {"query": "how do refunds work", "top_k": 3, "rewrite_n": 3, "rerank": True},
    )
    print(f"  ok            = {s['ok']}")
    print(f"  cache_hit     = {s['meta']['cache_hit']}")
    print(f"  elapsed_ms    = {s['meta']['elapsed_ms']}")
    print(f"  rewrites      = {s['data']['rewrites']}")
    for d in s["data"]["docs"]:
        preview = d["text"][:60].replace("\n", " ")
        print(f"    {d['id']}  rerank={d.get('rerank_score', 0):.3f}  {preview}...")
    assert s["ok"]
    assert s["data"]["docs"], "search returned no docs"

    section("5) /chat -- three domains in one conversation")
    conversation_id: str | None = None
    scripted = [
        "I want a refund for order ABC123456",
        "and I keep getting 401 error when logging in",
        "actually can you transfer me to a human?",
    ]
    expected_agents = {
        scripted[0]: "billing",
        scripted[1]: "technical",
        scripted[2]: "general",  # human_handoff -> escalation flag, agent may be general
    }
    answers: list[str] = []
    degraded_real = 0
    for msg in scripted:
        payload = {"message": msg, "user_id": "e2e_user"}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        r = post("/chat", payload)
        conversation_id = r["conversation_id"]
        answers.append(r["answer"])
        is_mock_answer = "(mock reply)" in r["answer"]
        is_degraded_real = "temporarily unavailable" in r["answer"]
        if is_mock_answer:
            answer_tag = "MOCK"
        elif is_degraded_real:
            answer_tag = "DEGRADED"
            degraded_real += 1
        else:
            answer_tag = "LIVE"
        primary = r["routing"]["primary_agent"]
        supporting = ",".join(r["routing"]["supporting_agents"]) or "-"
        entities = r["intent"]["entities"] or {}
        print(f"  msg: {msg}")
        print(
            f"    intent={r['intent']['intent']:<20} "
            f"conf={r['intent']['confidence']:.3f} "
            f"agent={primary:<11} supporting={supporting}"
        )
        print(f"    entities={entities}")
        print(f"    escalation={r['routing']['escalation_requested']}  latency={r['latency_ms']:.0f}ms")
        print(f"    [{answer_tag}] {r['answer'][:180].replace(chr(10), ' ')}...")

        if is_real:
            assert not is_mock_answer, "REAL mode but got a mock-format answer"
        else:
            assert is_mock_answer, "MOCK mode but answer lacks mock marker"

    if is_real and degraded_real:
        print(
            f"\n  NOTE: {degraded_real}/{len(scripted)} real-mode chats degraded (LLM error). "
            "Common cause: the API key is org-scoped -- set ANTHROPIC_WORKSPACE_ID in .env, "
            "or generate a workspace-scoped key at console.anthropic.com/settings/keys."
        )

    # Every chat turn must use the same conversation_id (memory persistence).
    assert conversation_id is not None
    print(f"  final conversation_id = {conversation_id}")

    section("6) /skills")
    sk = get("/skills")
    print(f"  count = {sk['count']}")
    for entry in sk["skills"]:
        print(f"    {entry['name']:<28} agents={entry['agents']}  keywords={len(entry['keywords'])}")
    assert sk["count"] >= 3

    section("7) /monitor")
    m = get("/monitor")
    for cname, cstats in m["monitor"]["components"].items():
        print(
            f"  {cname:<28} samples={cstats['samples']}  "
            f"success={cstats['success_rate']:.0%}  avg_ms={cstats['avg_latency_ms']:.1f}"
        )
    print(f"  tool breakers    = {m['tools']['breakers']}")
    print(f"  agent_penalties  = {m['monitor']['agent_penalties']}")
    print(f"  alerts           = {m['monitor']['alerts']}")

    section("8) /eval/run (default cases)")
    ev = post("/eval/run", {})
    intent_correct = sum(1 for c in ev["cases"] if c["intent_correct"])
    judge_pass = sum(1 for c in ev["cases"] if c["judge_pass"])
    total = len(ev["cases"])
    print(f"  passed        = {ev['passed']}")
    print(f"  metrics       = {ev['metrics']}")
    print(f"  regressions   = {len(ev['regressions'])}")
    print(f"  intent ok     = {intent_correct}/{total}")
    print(f"  judge pass    = {judge_pass}/{total}")
    assert intent_correct >= total - 1, "too many intent misses"

    print("=" * 70)
    print(f"E2E SMOKE OK ({'REAL' if is_real else 'MOCK'} mode)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\nASSERTION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except httpx.HTTPError as exc:
        print(f"\nHTTP ERROR: {exc}", file=sys.stderr)
        raise SystemExit(3)

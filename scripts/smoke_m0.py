"""M0 smoke test: verify settings load, mock LLM works, mock embedding works.

Run from project root:
    .venv/Scripts/python.exe scripts/smoke_m0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from core import get_embedding_client, get_llm_client
from core.embedding_client import cosine


def main() -> int:
    print("=" * 60)
    print("Settings loaded")
    print("=" * 60)
    print(f"  llm_mode        = {settings.llm_mode}")
    print(f"  embedding_mode  = {settings.embedding_mode}")
    print(f"  redis_url       = {settings.redis_url}")
    print(f"  chroma_mode     = {settings.chroma_mode}")
    print()

    llm = get_llm_client()
    emb = get_embedding_client()

    print("=" * 60)
    print("LLM.classify_intent")
    print("=" * 60)
    candidates = [
        {"intent": "refund", "group": "billing",
         "keywords": ["refund", "退款", "money back"],
         "description": "user wants a refund for a purchase"},
        {"intent": "login_failure", "group": "technical",
         "keywords": ["login", "401", "登录", "sign in"],
         "description": "user cannot log in, sees an error"},
        {"intent": "greeting", "group": "general",
         "keywords": ["hi", "hello", "你好"],
         "description": "user greeting"},
    ]
    result = llm.classify_intent("我要退款，帮我处理一下 订单 ABC123456", candidates)
    print(f"  -> {result}")
    assert result["intent"] == "refund", "expected 'refund' intent"
    assert "order_id" in result["entities"], "expected order_id entity"
    print("  PASS")
    print()

    print("=" * 60)
    print("LLM.rewrite_query / rerank / judge / chat")
    print("=" * 60)
    rewrites = llm.rewrite_query("how to reset password", n=3)
    print(f"  rewrites  = {rewrites}")
    assert len(rewrites) == 3

    docs = [
        {"id": "1", "text": "To reset your password, click 'Forgot password' on the login page."},
        {"id": "2", "text": "Our refund policy allows returns within 30 days."},
    ]
    scores = llm.rerank("reset password", docs)
    print(f"  scores    = {scores}")
    assert scores[0] > scores[1], "password doc should rank higher"

    judged = llm.judge("What is 2+2?", "2+2 equals 4.", reference="4")
    print(f"  judged    = {judged}")
    assert 0 <= judged["score"] <= 10

    chat_reply = llm.chat(
        system="You are a general customer support agent.",
        messages=[{"role": "user", "content": "hello"}],
    )
    print(f"  chat      = {chat_reply}")
    assert "mock" in chat_reply.lower()
    print("  PASS")
    print()

    print("=" * 60)
    print("Embedding: dim + determinism + similarity")
    print("=" * 60)
    v1, v2, v3 = emb.embed(["refund my money", "refund my money", "reset password"])
    print(f"  dim       = {emb.dim}")
    print(f"  v1[:4]    = {v1[:4]}")
    print(f"  cos(v1,v2)= {cosine(v1, v2):.4f}  (identical inputs -> should be 1.0)")
    print(f"  cos(v1,v3)= {cosine(v1, v3):.4f}  (different inputs -> should be <1.0)")
    assert emb.dim == settings.embedding_dim
    assert cosine(v1, v2) > 0.999
    assert cosine(v1, v3) < 0.999
    print("  PASS")
    print()

    print("=" * 60)
    print("M0 SMOKE TEST OK")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

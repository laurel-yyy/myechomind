"""M1 smoke test: verify Redis working memory + Chroma episodic/profile.

Prerequisites: Redis and ChromaDB must be running.
    docker compose up -d redis chromadb

Run from project root:
    .venv/Scripts/python.exe scripts/smoke_m1.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.conversation_memory import ConversationMemory


def main() -> int:
    mem = ConversationMemory()
    user_id = f"smoke_user_{uuid.uuid4().hex[:6]}"
    conv_id = f"conv_{uuid.uuid4().hex[:6]}"
    print(f"user_id={user_id}  conv_id={conv_id}")

    # ---- Working memory: append + cap ----
    print("=" * 60)
    print("Working memory (cap = 20)")
    print("=" * 60)
    for i in range(25):
        mem.append_message(user_id, conv_id, "user", f"msg {i}")
    wm = mem.get_working_memory(user_id, conv_id)
    print(f"  count = {len(wm)}   first='{wm[0]['content']}'   last='{wm[-1]['content']}'")
    assert len(wm) == 20, "must be capped at 20"
    assert wm[0]["content"] == "msg 5", "oldest surviving should be msg 5"
    assert wm[-1]["content"] == "msg 24", "newest should be msg 24"
    print("  PASS")

    # ---- Summary set/get ----
    print("=" * 60)
    print("Summary set / get")
    print("=" * 60)
    mem.set_summary(user_id, conv_id, "User asked about refund for order ABC123456.")
    got = mem.get_summary(user_id, conv_id)
    print(f"  summary = {got}")
    assert got and "refund" in got
    print("  PASS")

    # ---- Episodic add + search ----
    print("=" * 60)
    print("Episodic memory: add + semantic search")
    print("=" * 60)
    mem.add_episodic(user_id, conv_id, "User complained about repeated charge on 2024-11-30.", {"topic": "billing"})
    mem.add_episodic(user_id, conv_id, "User asked how to reset password.", {"topic": "technical"})
    time.sleep(0.3)  # small grace for indexing
    hits = mem.search_episodic(user_id, "duplicate charge", top_k=2)
    print(f"  hits = {[h['text'] for h in hits]}")
    assert hits, "episodic search returned nothing"
    print("  PASS")

    # ---- Profile upsert + search ----
    print("=" * 60)
    print("User profile: upsert + search")
    print("=" * 60)
    mem.upsert_profile_fact(user_id, category="preference", fact="prefers email over phone")
    mem.upsert_profile_fact(user_id, category="plan", fact="paid subscriber since 2023")
    time.sleep(0.3)
    facts = mem.get_profile(user_id, query="how does the user prefer to be contacted?", top_k=1)
    print(f"  facts = {[f['text'] for f in facts]}")
    assert facts, "profile search returned nothing"
    print("  PASS")

    # ---- Aggregate context ----
    print("=" * 60)
    print("build_context aggregate")
    print("=" * 60)
    ctx = mem.build_context(user_id, conv_id, "refund duplicate charge")
    print(f"  working_memory count = {len(ctx['working_memory'])}")
    print(f"  summary              = {ctx['summary']}")
    print(f"  episodic count       = {len(ctx['episodic'])}")
    print(f"  profile count        = {len(ctx['profile'])}")
    assert len(ctx["working_memory"]) == 20
    assert ctx["summary"] is not None
    print("  PASS")

    print("=" * 60)
    print("M1 SMOKE TEST OK")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

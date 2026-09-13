"""M2 smoke test: run the three-path fusion intent recognizer against a
battery of realistic user messages and verify each falls into the expected
bucket, plus entities are extracted when present.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import IntentRecognizer

CASES: list[tuple[str, str, str, set[str]]] = [
    # (message, expected_intent, expected_group, required_entity_keys)
    ("I want a refund for order ABC123456", "refund", "billing", {"order_id"}),
    ("I keep getting 401 error when logging in", "login_failure", "technical", {"error_code"}),
    ("the app crashes every time I open it", "crash", "technical", set()),
    ("hi there", "greeting", "general", set()),
    ("transfer me to a human agent please", "human_handoff", "escalation", set()),
    ("I was charged twice for the same purchase", "duplicate_charge", "billing", set()),
    ("please issue an invoice for my order", "invoice", "billing", set()),
    ("what is the status of order XY7788990011?", "order_status", "general", {"order_id"}),
    ("what's your refund policy?", "policy", "general", set()),
    ("I want to cancel my subscription", "subscription", "billing", set()),
]


def main() -> int:
    rec = IntentRecognizer()
    failures = 0

    for msg, want_intent, want_group, want_entity_keys in CASES:
        result = rec.recognize(msg)
        intent_ok = result.intent == want_intent and result.group == want_group
        entity_ok = want_entity_keys.issubset(result.entities.keys())
        status = "PASS" if (intent_ok and entity_ok) else "FAIL"
        if status == "FAIL":
            failures += 1
        print("-" * 70)
        print(f"[{status}] msg: {msg}")
        print(f"    want intent={want_intent}/{want_group}, entities>={want_entity_keys or '-'}")
        print(f"    got  intent={result.intent}/{result.group}  urgency={result.urgency}  conf={result.confidence}")
        print(f"    entities   = {result.entities}")
        print(f"    reasoning  = {result.reasoning}")

    print("=" * 70)
    if failures == 0:
        print("M2 SMOKE TEST OK")
        return 0
    print(f"M2 SMOKE TEST FAILED: {failures}/{len(CASES)} cases wrong")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Smoke test for M8 FastAPI endpoints using live Redis and ChromaDB."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.main import create_app


def main() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["status"] == "ok"

        added = client.post(
            "/knowledge/add",
            json={"documents": [{"text": "Refunds are available within 14 days of purchase."}]},
        )
        assert added.status_code == 200, added.text
        assert added.json()["count"] == 1

        upload = client.post(
            "/knowledge/upload",
            files={"file": ("shipping.md", b"Shipping normally takes three business days.", "text/markdown")},
        )
        assert upload.status_code == 200, upload.text

        search = client.post("/search", json={"query": "How do I get a refund?", "top_k": 3})
        assert search.status_code == 200, search.text
        assert search.json()["ok"] is True

        chat = client.post(
            "/chat",
            json={"user_id": "m8_smoke_user", "message": "I need a refund for order AB12345678"},
        )
        assert chat.status_code == 200, chat.text
        body = chat.json()
        assert body["intent"]["intent"] == "refund", body
        assert body["routing"]["primary_agent"] == "billing", body
        assert body["knowledge"]["enabled"] is True

        skills = client.get("/skills")
        assert skills.status_code == 200 and skills.json()["count"] == 3
        assert client.get("/monitor").status_code == 200
        assert client.get("/metrics").status_code == 200
        evaluation = client.post(
            "/eval/run",
            json={
                "cases": [
                    {
                        "case_id": "api_greeting",
                        "message": "Hello",
                        "expected_intent": "greeting",
                        "expected_group": "general",
                    }
                ]
            },
        )
        assert evaluation.status_code == 200, evaluation.text
        assert evaluation.json()["metrics"]["intent_accuracy"] == 1.0

    print("M8 SMOKE TEST OK")


if __name__ == "__main__":
    main()

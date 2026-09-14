"""Smoke test for M4 dynamic Skill discovery and injection."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow this smoke test to run directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.skill_loader import SkillManager


def main() -> None:
    manager = SkillManager()
    catalog = manager.list_skills()
    assert len(catalog) == 3, f"Expected 3 skills, got {len(catalog)}"
    assert {item["name"] for item in catalog} == {
        "general_customer_service",
        "technical_support",
        "billing_support",
    }

    technical = manager.match("technical", "I get a 401 error when I log in")
    assert [skill.name for skill in technical] == ["technical_support"]
    technical_prompt = manager.render_prompt("technical", "I get a 401 error when I log in")
    assert "Technical Support SOP" in technical_prompt
    assert "full payment details" in technical_prompt

    billing = manager.match("billing", "I was charged twice and need a refund")
    assert [skill.name for skill in billing] == ["billing_support"]
    billing_prompt = manager.render_prompt("billing", "I was charged twice and need a refund")
    assert "Billing Support SOP" in billing_prompt

    no_cross_domain = manager.match("general", "I get a 401 error when I log in")
    assert no_cross_domain == [], "Technical skills must not contaminate GeneralAgent"

    reloaded = manager.reload()
    assert len(reloaded) == 3
    print("M4 SMOKE TEST OK")


if __name__ == "__main__":
    main()

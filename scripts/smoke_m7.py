"""Smoke test for M7 end-to-end evaluation and baseline comparison."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluator import EvaluationCase, Evaluator


CASES = [
    EvaluationCase("greeting", "Hello there", "greeting", "general"),
    EvaluationCase("refund", "I need a refund for order AB12345678", "refund", "billing"),
    EvaluationCase("login", "I get 401 when I log in", "login_failure", "technical"),
    EvaluationCase("crash", "The mobile app crashes when I open it", "crash", "technical"),
    EvaluationCase("duplicate", "I was charged twice for the same order", "duplicate_charge", "billing"),
    EvaluationCase("invoice", "Please send my invoice", "invoice", "billing"),
    EvaluationCase("logistics", "When will my package arrive?", "logistics", "general"),
    EvaluationCase("handoff", "Transfer me to a human agent", "human_handoff", "escalation"),
]


def main() -> None:
    evaluator = Evaluator()
    report = evaluator.run(CASES)
    assert report.intent_accuracy == 1.0, report.to_dict()
    assert report.group_accuracy == 1.0, report.to_dict()
    assert report.macro_f1 == 1.0, report.to_dict()
    assert report.average_judge_score == 7.0, report.to_dict()
    assert report.judge_pass_rate == 1.0, report.to_dict()
    assert report.passed is True, report.to_dict()
    assert report.baseline is not None

    with tempfile.TemporaryDirectory() as directory:
        baseline_path = Path(directory) / "baseline.json"
        baseline_path.write_text(
            '{"tolerance": 0.0, "metrics": {"intent_accuracy": 1.0}}',
            encoding="utf-8",
        )
        regression_report = Evaluator(baseline_path=baseline_path).run(
            [EvaluationCase("wrong_expectation", "Hello", "refund", "billing")]
        )
        assert regression_report.passed is False
        assert regression_report.regressions[0]["metric"] == "intent_accuracy"
    print("M7 SMOKE TEST OK")


if __name__ == "__main__":
    main()

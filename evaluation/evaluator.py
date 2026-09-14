"""End-to-end evaluation: intent metrics, LLM-as-Judge, and regression checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agents.agent_orchestrator import AgentOrchestrator
from config import settings
from core.llm_client import LLMClient, get_llm_client


@dataclass(frozen=True)
class EvaluationCase:
    """One expectation for a full request-to-answer execution."""

    case_id: str
    message: str
    expected_intent: str
    expected_group: str
    reference_answer: str | None = None


@dataclass
class CaseResult:
    case_id: str
    expected_intent: str
    predicted_intent: str
    expected_group: str
    predicted_group: str
    intent_correct: bool
    group_correct: bool
    judge_score: float
    judge_pass: bool
    judge_comments: str
    routing: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Aggregate metrics, case-level traces, and baseline comparison."""

    cases: list[CaseResult]
    intent_accuracy: float
    group_accuracy: float
    macro_f1: float
    average_judge_score: float
    judge_pass_rate: float
    baseline: dict[str, Any] | None = None
    regressions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.regressions

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": {
                "intent_accuracy": self.intent_accuracy,
                "group_accuracy": self.group_accuracy,
                "macro_f1": self.macro_f1,
                "average_judge_score": self.average_judge_score,
                "judge_pass_rate": self.judge_pass_rate,
            },
            "baseline": self.baseline,
            "regressions": self.regressions,
            "cases": [case.to_dict() for case in self.cases],
        }


class Evaluator:
    """Runs the production orchestration path against labeled cases."""

    _METRICS = ("intent_accuracy", "group_accuracy", "macro_f1", "average_judge_score", "judge_pass_rate")

    def __init__(
        self,
        *,
        orchestrator: AgentOrchestrator | None = None,
        llm: LLMClient | None = None,
        baseline_path: str | Path | None = None,
    ) -> None:
        self._orchestrator = orchestrator or AgentOrchestrator()
        self._llm = llm or get_llm_client()
        configured_path = Path(baseline_path or settings.eval_baseline_path)
        self._baseline_path = (
            configured_path
            if configured_path.is_absolute()
            else settings.project_root / configured_path
        )

    def run(self, cases: list[EvaluationCase]) -> EvaluationReport:
        """Execute all cases through the real orchestrator and score outcomes."""
        if not cases:
            raise ValueError("At least one evaluation case is required")
        results: list[CaseResult] = []
        for case in cases:
            output = self._orchestrator.handle(case.message)
            judge = self._llm.judge(case.message, output.answer, case.reference_answer)
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    expected_intent=case.expected_intent,
                    predicted_intent=output.intent.intent,
                    expected_group=case.expected_group,
                    predicted_group=output.intent.group,
                    intent_correct=output.intent.intent == case.expected_intent,
                    group_correct=output.intent.group == case.expected_group,
                    judge_score=float(judge.get("score", 0.0)),
                    judge_pass=bool(judge.get("pass", False)),
                    judge_comments=str(judge.get("comments", "")),
                    routing=output.routing.to_dict(),
                )
            )

        report = EvaluationReport(
            cases=results,
            intent_accuracy=_ratio(sum(item.intent_correct for item in results), len(results)),
            group_accuracy=_ratio(sum(item.group_correct for item in results), len(results)),
            macro_f1=_macro_f1(results),
            average_judge_score=round(sum(item.judge_score for item in results) / len(results), 4),
            judge_pass_rate=_ratio(sum(item.judge_pass for item in results), len(results)),
        )
        report.baseline = self.load_baseline()
        report.regressions = self._compare_to_baseline(report, report.baseline)
        return report

    def load_baseline(self) -> dict[str, Any] | None:
        """Load an approved baseline, returning None when one is not present."""
        if not self._baseline_path.exists():
            return None
        with self._baseline_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("Evaluation baseline must be a JSON object")
        return payload

    def save_baseline(self, report: EvaluationReport, *, tolerance: float = 0.05) -> None:
        """Persist explicitly approved aggregate metrics as the regression baseline."""
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError("Baseline tolerance must be in [0, 1]")
        payload = {
            "schema_version": 1,
            "tolerance": tolerance,
            "metrics": {
                metric: getattr(report, metric)
                for metric in self._METRICS
            },
        }
        self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self._baseline_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _compare_to_baseline(
        self,
        report: EvaluationReport,
        baseline: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not baseline:
            return []
        metrics = baseline.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Evaluation baseline requires a 'metrics' object")
        tolerance = float(baseline.get("tolerance", 0.05))
        regressions: list[dict[str, Any]] = []
        for metric in self._METRICS:
            baseline_value = metrics.get(metric)
            if baseline_value is None:
                continue
            current_value = float(getattr(report, metric))
            baseline_number = float(baseline_value)
            if current_value < baseline_number - tolerance:
                regressions.append(
                    {
                        "metric": metric,
                        "baseline": baseline_number,
                        "current": current_value,
                        "delta": round(current_value - baseline_number, 4),
                        "tolerance": tolerance,
                    }
                )
        return regressions


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator if denominator else 0.0, 4)


def _macro_f1(results: list[CaseResult]) -> float:
    """Compute unweighted mean F1 over every expected or predicted intent label."""
    labels = sorted({item.expected_intent for item in results} | {item.predicted_intent for item in results})
    f1_scores: list[float] = []
    for label in labels:
        true_positive = sum(
            item.expected_intent == label and item.predicted_intent == label
            for item in results
        )
        false_positive = sum(
            item.expected_intent != label and item.predicted_intent == label
            for item in results
        )
        false_negative = sum(
            item.expected_intent == label and item.predicted_intent != label
            for item in results
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(f1_scores) / len(f1_scores) if f1_scores else 0.0, 4)

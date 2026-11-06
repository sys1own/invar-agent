"""Heuristic patch fitness evaluation: error deltas and regression scores."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RepairFitnessScore:
    improved: bool
    error_count_delta: int
    regression_risk: float
    details: list[str] = field(default_factory=list)


class PatchFitnessEvaluator:
    """Scores candidate repairs by their effect on failure counts."""

    def evaluate_repair_delta(
        self,
        baseline_failures: int,
        candidate_failures: int,
        new_errors: int = 0,
    ) -> RepairFitnessScore:
        delta = baseline_failures - candidate_failures
        improved = delta > 0
        regression_risk = 0.0
        details: list[str] = []
        if candidate_failures > baseline_failures:
            regression_risk = min(1.0, (candidate_failures - baseline_failures) / max(1, baseline_failures))
            details.append("candidate increased failure count")
        if new_errors > 0:
            regression_risk = min(1.0, regression_risk + 0.2 * new_errors)
            details.append(f"{new_errors} new error(s) introduced")
        if improved:
            details.append(f"failures reduced by {delta}")
        return RepairFitnessScore(
            improved=improved,
            error_count_delta=delta,
            regression_risk=regression_risk,
            details=details,
        )

    def compute_regression_score(self, baseline_failures: int, candidate_failures: int) -> float:
        """0.0 = strictly improving, 1.0 = fully regressed."""
        if candidate_failures <= baseline_failures:
            return 0.0 if candidate_failures < baseline_failures else 0.5
        return min(1.0, candidate_failures / max(1, baseline_failures))

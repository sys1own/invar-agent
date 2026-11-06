"""Formal error explainer: UNSAT cores and counterexamples -> plain language."""
from __future__ import annotations

from typing import Any, Optional


class FormalErrorExplainer:
    """Translates formal verification artifacts into actionable diagnostics."""

    def translate_unsat_core_to_nl(self, unsat_core: list[str], spec_clauses: dict[str, str]) -> str:
        if not unsat_core:
            return "Verification succeeded: no conflicting obligations."
        named = [spec_clauses.get(c, c) for c in unsat_core]
        return (
            "The Precision Shield proved the mutation unsafe. The conflicting "
            f"obligations (unsat core) are: {'; '.join(named)}. "
            "Adjust the implementation or relax the contract before retrying."
        )

    def explain_concolic_counterexample(
        self,
        witness: Optional[dict[str, Any]],
        failing_branch: str,
        exception: Optional[str] = None,
    ) -> str:
        parts = []
        if witness:
            assignments = ", ".join(f"{k}={v}" for k, v in witness.items())
            parts.append(f"concolic witness: {assignments}")
        if failing_branch:
            parts.append(f"failing branch: {failing_branch}")
        if exception:
            parts.append(f"raised: {exception}")
        base = "Concolic execution uncovered a concrete failing input."
        return base + (" " + "; ".join(parts) if parts else "")

    def explain_chiasmus_violation(self, violation: str) -> str:
        return (
            f"Chiasmus symmetry violation: {violation}. Every acquisition must be "
            "released in exact reverse order on every exit path (including "
            "exception edges); restructure with try/finally."
        )

    def summarize(self, category: str, diagnostic: str) -> str:
        return f"[{category}] {diagnostic}"

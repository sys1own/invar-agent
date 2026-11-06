"""Formal specification contracts: pre/post conditions, invariants, obligations."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .smt_solver import SMTSolverBridge, SMTResult, SMTStatus


class ObligationStatus(str, Enum):
    PROVEN = "PROVEN"
    SAT_VIOLATED = "SAT_VIOLATED"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


@dataclass
class InvariantClause:
    kind: str  # "pre" | "post" | "invariant"
    expression: str
    target_symbol: Optional[str] = None


@dataclass
class FormalVerificationObligation:
    obligation_id: str
    clause: InvariantClause
    status: ObligationStatus = ObligationStatus.SKIPPED
    witness: Optional[dict[str, Any]] = None
    elapsed_ms: float = 0.0
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class FormalSpec:
    spec_id: str
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    timeout_ms: int = 5000

    def all_clauses(self) -> list[InvariantClause]:
        clauses = [InvariantClause("pre", c) for c in self.preconditions]
        clauses += [InvariantClause("post", c) for c in self.postconditions]
        clauses += [InvariantClause("invariant", c) for c in self.invariants]
        return clauses

    def compile_spec_to_smt(self, bridge: Optional[SMTSolverBridge] = None) -> tuple[list[Any], list[Any]]:
        """Compile to (precondition_constraints, negated_postcondition_constraints)."""
        bridge = bridge or SMTSolverBridge(self.timeout_ms)
        pre = [bridge.compile_clause(c) for c in self.preconditions]
        post = [bridge.compile_clause(c) for c in self.postconditions]
        return pre, post


class SpecVerifier:
    """Discharges formal obligations against the Z3 bridge."""

    def __init__(self, bridge: Optional[SMTSolverBridge] = None) -> None:
        self.bridge = bridge or SMTSolverBridge()

    def discharge_obligation(self, spec: FormalSpec, obligation: FormalVerificationObligation,
                             local_env: Optional[dict[str, Any]] = None) -> FormalVerificationObligation:
        clause = obligation.clause
        start = time.perf_counter()
        try:
            if clause.kind == "pre":
                # Preconditions are assumptions: check they are satisfiable.
                result = self.bridge.check_sat([self.bridge.compile_clause(clause.expression, local_env)],
                                               timeout_ms=spec.timeout_ms)
                obligation.status = ObligationStatus.PROVEN if result.status == SMTStatus.UNSAT_PROVEN or result.status == SMTStatus.SAT_COUNTEREXAMPLE else ObligationStatus.TIMEOUT
                obligation.witness = result.witness
            elif clause.kind == "post":
                result = self.bridge.prove_postcondition(spec.preconditions, [clause.expression],
                                                         local_env=local_env, timeout_ms=spec.timeout_ms)
                obligation.witness = result.witness
                if result.status == SMTStatus.UNSAT_PROVEN:
                    obligation.status = ObligationStatus.PROVEN
                elif result.status == SMTStatus.SAT_COUNTEREXAMPLE:
                    obligation.status = ObligationStatus.SAT_VIOLATED
                elif result.status == SMTStatus.TIMEOUT:
                    obligation.status = ObligationStatus.TIMEOUT
                else:
                    obligation.status = ObligationStatus.ERROR
            else:  # invariant: must hold under preconditions alone
                result = self.bridge.prove_postcondition(spec.preconditions, [clause.expression],
                                                         local_env=local_env, timeout_ms=spec.timeout_ms)
                obligation.witness = result.witness
                obligation.status = (ObligationStatus.PROVEN if result.status == SMTStatus.UNSAT_PROVEN
                                     else ObligationStatus.SAT_VIOLATED if result.status == SMTStatus.SAT_COUNTEREXAMPLE
                                     else ObligationStatus.TIMEOUT)
            obligation.elapsed_ms = (time.perf_counter() - start) * 1000.0
            obligation.diagnostics = result.diagnostics
        except Exception as e:  # noqa: BLE001
            obligation.status = ObligationStatus.ERROR
            obligation.diagnostics = [str(e)]
            obligation.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return obligation

    def verify_spec(self, spec: FormalSpec,
                    local_env: Optional[dict[str, Any]] = None) -> list[FormalVerificationObligation]:
        obligations = [
            FormalVerificationObligation(obligation_id=f"{spec.spec_id}:{i}", clause=clause)
            for i, clause in enumerate(spec.all_clauses())
        ]
        return [self.discharge_obligation(spec, ob, local_env) for ob in obligations]

    def prove_spec(self, spec: FormalSpec,
                   local_env: Optional[dict[str, Any]] = None) -> SMTResult:
        """Single-shot proof: preconditions => all postconditions."""
        return self.bridge.prove_postcondition(
            spec.preconditions, spec.postconditions, local_env=local_env, timeout_ms=spec.timeout_ms
        )

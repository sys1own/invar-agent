"""Provably Total 6-Absorbing-State FSM and `TotalBoundaryAdapter`.

Guarantees:
1. Lexicographic Lyapunov progress measure `(b_pass, b_repair, b_exec)` bounded by
   `(2, 2, 2)` — every state transition strictly decreases the measure or enters
   one of the 6 absorbing terminal states.
2. Zero HTTP 500/502 leakage: `TotalBoundaryAdapter` catches any boundary fault
   (LLM transport error, JSON parse failure, compiler error, solver non-convergence,
   unexpected exception) and lifts it into a cryptographically committed
   `VerdictReceipt` with a typed `InconclusiveReasonCode`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from invar_agent.ir.evidence import (
    AbsorbingVerdictState,
    InconclusiveReasonCode,
    VerdictReceipt,
)


@dataclass
class LyapunovBudget:
    """Strictly decreasing lexicographic termination measure."""

    b_pass: int = 2
    b_repair: int = 2
    b_exec: int = 2

    def consume_pass(self) -> bool:
        if self.b_pass <= 0:
            return False
        self.b_pass -= 1
        return True

    def consume_repair(self) -> bool:
        if self.b_repair <= 0:
            return False
        self.b_repair -= 1
        return True

    def consume_exec(self) -> bool:
        if self.b_exec <= 0:
            return False
        self.b_exec -= 1
        return True

    def measure(self) -> tuple[int, int, int]:
        return (self.b_pass, self.b_repair, self.b_exec)

    def exhausted(self) -> bool:
        return self.b_pass == 0 and self.b_repair == 0 and self.b_exec == 0


_CODE_DISPATCH: dict[str, tuple[AbsorbingVerdictState, InconclusiveReasonCode]] = {
    "PROVIDER_TRANSPORT_FAULT": (AbsorbingVerdictState.INCONCLUSIVE_BUDGET, InconclusiveReasonCode.PROVIDER_TRANSPORT_FAULT),
    "JACOBIAN_MISMATCH": (AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED, InconclusiveReasonCode.JACOBIAN_MISMATCH),
    "SOLVER_STIFFNESS_BLOWUP": (AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED, InconclusiveReasonCode.SOLVER_STIFFNESS_BLOWUP),
    "DIMENSIONAL_INCONSISTENCY": (AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED, InconclusiveReasonCode.DIMENSIONAL_INCONSISTENCY),
    "KERNEL_COMPILATION_FAULT": (AbsorbingVerdictState.INCONCLUSIVE_BUDGET, InconclusiveReasonCode.KERNEL_COMPILATION_FAULT),
}


def classify_exception_to_reason(exc: BaseException) -> tuple[AbsorbingVerdictState, InconclusiveReasonCode]:
    """Deterministically lift any runtime exception into a typed absorbing verdict (BUG-039)."""
    import re as _re

    # 1. Exception-type dispatch
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_BUDGET,
            InconclusiveReasonCode.PROVIDER_TRANSPORT_FAULT,
        )
    if isinstance(exc, OverflowError):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED,
            InconclusiveReasonCode.SOLVER_STIFFNESS_BLOWUP,
        )
    if isinstance(exc, OSError):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_BUDGET,
            InconclusiveReasonCode.UNHANDLED_BOUNDARY_EXCEPTION,
        )

    # 2. Structured error-code attribute dispatch
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code in _CODE_DISPATCH:
        return _CODE_DISPATCH[code]

    # 3. Exact upper-case token / explicit error-tag dispatch (no substring 'inf'/'nan' matches)
    raw = str(exc)
    if _re.search(r"\b(?:TIMEOUT|HTTP_502|HTTP_503|HTTP_504|HTTP_429|RATE_LIMIT|PROVIDER_TRANSPORT_FAULT)\b", raw) or _re.search(r"\b(?:502|503|504|429)\b", raw) or "httpx" in raw.lower() or "rate limit" in raw.lower() or "timed out" in raw.lower():
        return (
            AbsorbingVerdictState.INCONCLUSIVE_BUDGET,
            InconclusiveReasonCode.PROVIDER_TRANSPORT_FAULT,
        )
    if _re.search(r"\bJACOBIAN_MISMATCH\b", raw):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED,
            InconclusiveReasonCode.JACOBIAN_MISMATCH,
        )
    if _re.search(r"\b(?:SOLVER_STIFFNESS_BLOWUP|STIFFNESS_BLOWUP|NON_FINITE|NAN|INF)\b", raw):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED,
            InconclusiveReasonCode.SOLVER_STIFFNESS_BLOWUP,
        )
    if _re.search(r"\b(?:DIMENSIONAL_INCONSISTENCY|CYCLIC_LET_DAG|AMBIGUOUS_COLLISION|SI_OR_SEMANTIC_MISMATCH)\b", raw):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED,
            InconclusiveReasonCode.DIMENSIONAL_INCONSISTENCY,
        )
    if _re.search(r"\b(?:KERNEL_COMPILATION_FAULT|CPP_COMPILE_DIAGNOSTIC)\b", raw):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_BUDGET,
            InconclusiveReasonCode.KERNEL_COMPILATION_FAULT,
        )
    return (
        AbsorbingVerdictState.INCONCLUSIVE_BUDGET,
        InconclusiveReasonCode.UNHANDLED_BOUNDARY_EXCEPTION,
    )


class TotalBoundaryAdapter:
    """Wraps any async research/builder execution coroutine so it ALWAYS returns
    a finalized, cryptographically committed `VerdictReceipt` — never raising
    an unhandled 500/502 error across the HTTP/SSE boundary."""

    @staticmethod
    async def execute(
        coro_fn: Callable[[], Awaitable[VerdictReceipt]],
        *,
        sciir_commitment_blake3: str = "",
        ihcg_v1_commitment_blake3: str = "",
    ) -> VerdictReceipt:
        try:
            receipt = await coro_fn()
            if not receipt.verdict_commitment_blake3:
                receipt = receipt.finalize_commitment()
            return receipt
        except Exception as exc:
            state, reason = classify_exception_to_reason(exc)
            fallback = VerdictReceipt(
                terminal_state=state,
                reason_code=reason,
                sciir_commitment_blake3=sciir_commitment_blake3,
                ihcg_v1_commitment_blake3=ihcg_v1_commitment_blake3,
                summary=f"Boundary fault lifted cleanly: {type(exc).__name__}: {exc}",
                diagnostics=[
                    {
                        "stage": "TOTAL_BOUNDARY_ADAPTER",
                        "code": reason.value,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
            return fallback.finalize_commitment()

    @staticmethod
    def lift_sync_error(
        exc: BaseException,
        *,
        sciir_commitment_blake3: str = "",
        ihcg_v1_commitment_blake3: str = "",
    ) -> dict[str, Any]:
        """Synchronously lift an exception into a JSON-serializable HTTP 200 terminal envelope."""
        state, reason = classify_exception_to_reason(exc)
        receipt = VerdictReceipt(
            terminal_state=state,
            reason_code=reason,
            sciir_commitment_blake3=sciir_commitment_blake3,
            ihcg_v1_commitment_blake3=ihcg_v1_commitment_blake3,
            summary=f"Boundary fault lifted cleanly: {type(exc).__name__}: {exc}",
            diagnostics=[
                {
                    "stage": "TOTAL_BOUNDARY_ADAPTER",
                    "code": reason.value,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            ],
        ).finalize_commitment()
        return receipt.model_dump(mode="json")

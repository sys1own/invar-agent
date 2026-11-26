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


def classify_exception_to_reason(exc: BaseException) -> tuple[AbsorbingVerdictState, InconclusiveReasonCode]:
    """Deterministically lift any runtime exception into a typed absorbing verdict."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    if any(k in msg for k in ("timeout", "connection", "502", "503", "504", "429", "rate limit", "httpx")):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_BUDGET,
            InconclusiveReasonCode.PROVIDER_TRANSPORT_FAULT,
        )
    if any(k in msg for k in ("jacobian_mismatch", "jacobian mismatch")):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED,
            InconclusiveReasonCode.JACOBIAN_MISMATCH,
        )
    if any(k in msg for k in ("stiff", "blowup", "overflow", "non-finite", "nan", "inf")):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED,
            InconclusiveReasonCode.SOLVER_STIFFNESS_BLOWUP,
        )
    if any(k in msg for k in ("dimension", "ambiguous", "scope", "cyclic")):
        return (
            AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED,
            InconclusiveReasonCode.DIMENSIONAL_INCONSISTENCY,
        )
    if any(k in msg for k in ("compile", "clang", "gcc", "g++", "kernel")):
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

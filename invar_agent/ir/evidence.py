"""Typed EvidenceRecord, VerdictReceipt, and InconclusiveReasonCode envelopes.

Every terminal run produces a cryptographically committed `EvidenceRecord` (`0x04`)
and `VerdictReceipt` (`0x05`) linked back to `0x01 || JCS(SciIR)` and `0x03 || kernel`.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from invar_agent.ir.ihcg_bridge import (
    DOMAIN_EVIDENCE_RECORD,
    DOMAIN_VERDICT_RECEIPT,
    domain_commit_json,
)


class AbsorbingVerdictState(str, Enum):
    """The 6 provably total absorbing terminal states of Invar-Agent vNext."""

    VERIFIED = "VERIFIED"
    FALSIFIED = "FALSIFIED"
    REFUTED_FEASIBILITY = "REFUTED_FEASIBILITY"
    INCONCLUSIVE_BUDGET = "INCONCLUSIVE_BUDGET"
    INCONCLUSIVE_ILL_POSED = "INCONCLUSIVE_ILL_POSED"
    FRONTIER_RESCUED_SHBT = "FRONTIER_RESCUED_SHBT"


class InconclusiveReasonCode(str, Enum):
    """Typed reason codes emitted by `TotalBoundaryAdapter` (0% HTTP 500/502)."""

    NONE = "NONE"
    PASS1_SYNTHESIS_EXHAUSTED = "PASS1_SYNTHESIS_EXHAUSTED"
    PASS2_SYNTHESIS_EXHAUSTED = "PASS2_SYNTHESIS_EXHAUSTED"
    MICRO_REPAIR_BUDGET_EXHAUSTED = "MICRO_REPAIR_BUDGET_EXHAUSTED"
    SCOPE_CHECK_AMBIGUOUS_SYMBOL = "SCOPE_CHECK_AMBIGUOUS_SYMBOL"
    DIMENSIONAL_INCONSISTENCY = "DIMENSIONAL_INCONSISTENCY"
    SOLVER_NON_CONVERGENT = "SOLVER_NON_CONVERGENT"
    SOLVER_STIFFNESS_BLOWUP = "SOLVER_STIFFNESS_BLOWUP"
    JACOBIAN_MISMATCH = "JACOBIAN_MISMATCH"
    MONTE_CARLO_UNCONVERGED = "MONTE_CARLO_UNCONVERGED"
    PROVIDER_TRANSPORT_FAULT = "PROVIDER_TRANSPORT_FAULT"
    KERNEL_COMPILATION_FAULT = "KERNEL_COMPILATION_FAULT"
    EXECUTION_BUDGET_EXHAUSTED = "EXECUTION_BUDGET_EXHAUSTED"
    UNHANDLED_BOUNDARY_EXCEPTION = "UNHANDLED_BOUNDARY_EXCEPTION"


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "invar.evidence.v2"
    sciir_commitment_blake3: str
    kernel_commitment_blake3: str = ""
    solver_used: str = ""
    archetype: str = ""
    max_conservation_drift: float = 0.0
    max_step_error: float = 0.0
    obligation_slacks: dict[str, float] = Field(default_factory=dict)
    counterexample_witness: Optional[dict[str, Any]] = None
    mc_diagnostics: Optional[dict[str, Any]] = None
    stability_certificate: Optional[dict[str, Any]] = None
    shbt_rescue_summary: Optional[dict[str, Any]] = None
    heal_count: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)

    def commitment_blake3(self) -> str:
        return domain_commit_json(DOMAIN_EVIDENCE_RECORD, self.model_dump_json())


class VerdictReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "invar.verdict.v2"
    terminal_state: AbsorbingVerdictState
    reason_code: InconclusiveReasonCode = InconclusiveReasonCode.NONE
    sciir_commitment_blake3: str = ""
    ihcg_v1_commitment_blake3: str = ""
    kernel_commitment_blake3: str = ""
    evidence_commitment_blake3: str = ""
    summary: str = ""
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    verdict_commitment_blake3: str = ""

    def finalize_commitment(self) -> "VerdictReceipt":
        payload = self.model_dump(exclude={"verdict_commitment_blake3"})
        digest = domain_commit_json(DOMAIN_VERDICT_RECEIPT, payload)
        return self.model_copy(update={"verdict_commitment_blake3": digest})

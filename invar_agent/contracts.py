"""Central Pydantic v2 data contracts for the invar-agent kernel."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class AgentStateStage(str, Enum):
    INGEST = "Ingest"
    GOI_PLAN = "GoI_Plan"
    WAVEFRONT_SCHEDULE = "Wavefront_Schedule"
    OVERLAY_MUTATE = "Overlay_Mutate"
    PREWRITE_GUARD = "PreWrite_Guard"
    SMT_CONCOLIC_VERIFY = "SMT_Concolic_Verify"
    SANDBOX_EXECUTE = "Sandbox_Execute"
    REFLEX_HEAL = "Reflex_Heal"
    ADJOINT_COMMIT_OR_ROLLBACK = "Adjoint_Commit_or_Rollback"


class PortSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    port_name: str
    source_node_id: Optional[str] = None
    symbol_contract: str


class VerificationContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    foge_guard_pre: list[str] = Field(default_factory=list)
    smt_preconditions: list[str] = Field(default_factory=list)
    smt_postconditions: list[str] = Field(default_factory=list)
    chiasmus_pairs: list[tuple[str, str]] = Field(default_factory=list)
    concolic_targets: list[str] = Field(default_factory=list)
    sandbox_verification_cmd: Optional[str] = None
    smt_timeout_ms: int = 5000


class TaskWavefrontNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    wavefront_generation: int = 0
    assigned_tier: Literal["TIER_0_REFLEX", "TIER_1_FLASH", "TIER_2_PRO"] = "TIER_1_FLASH"
    description: str
    input_ports: list[PortSpec] = Field(default_factory=list)
    output_ports: list[PortSpec] = Field(default_factory=list)
    read_set_paths: list[str] = Field(default_factory=list)
    write_set_paths: list[str] = Field(default_factory=list)
    target_ast_symbols: list[str] = Field(default_factory=list)
    modifies_holographic_boundary: bool = False
    expected_boundary_delta: list[str] = Field(default_factory=list)
    verification_contract: VerificationContract = Field(default_factory=VerificationContract)


class ASTSurgicalPatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_path: str
    operation: Literal["REPLACE_SYMBOL", "INSERT_AFTER_SYMBOL", "REPLACE_EXACT_SPAN", "CREATE_FILE"]
    target_symbol: Optional[str] = None
    search_block: str = ""
    replacement_code: str
    pre_hash: Optional[str] = None
    post_hash: Optional[str] = None
    adjoint_inverse_code: Optional[str] = None


class OverlayTransaction(BaseModel):
    model_config = ConfigDict(frozen=False)

    tx_id: str
    node_id: str
    base_snapshot_digest: str
    pre_holographic_boundary: str
    post_holographic_boundary: Optional[str] = None
    staged_patches: list[ASTSurgicalPatch] = Field(default_factory=list)
    prewrite_guard_passed: bool = False
    shield_verified: bool = False
    sandbox_passed: bool = False
    committed: bool = False
    rolled_back: bool = False


class ShieldVerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    foge_passed: bool = True
    smt_status: Literal["UNSAT_PROVEN", "SAT_COUNTEREXAMPLE", "SKIPPED", "TIMEOUT"] = "UNSAT_PROVEN"
    smt_counterexample_witness: Optional[dict[str, Any]] = None
    concolic_branches_explored: int = 0
    concolic_failures: list[str] = Field(default_factory=list)
    chiasmus_symmetric: bool = True
    chiasmus_violations: list[str] = Field(default_factory=list)
    holographic_parity_valid: bool = True
    rust_shield_passed: bool = True
    diagnostics: list[str] = Field(default_factory=list)


class HealingVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    healed: bool
    resolved_by_tier: Literal["TIER_0_REFLEX", "TIER_1_FLASH", "TIER_2_PRO", "UNRESOLVED"]
    attempts_used: int = 0
    tokens_consumed: int = 0
    latency_ms: float = 0.0
    trigger_adjoint_rollback: bool = False
    applied_rule_or_rationale: str = ""
    error_category: Optional[str] = None
    repaired_patches: list[ASTSurgicalPatch] = Field(default_factory=list)

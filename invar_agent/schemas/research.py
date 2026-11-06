"""Pydantic v2 schemas for the Research Agent verification contracts."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class InvariantConstraint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variable_name: str
    assertion: str
    tolerance: float = Field(default=1e-6, ge=0.0)


class VerificationContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str
    target_backend: Literal["rust_hin", "cpp", "cuda", "multi_variant"]
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    invariants: List[InvariantConstraint] = Field(min_length=1)
    max_memory_mb: int = Field(default=1024, le=8192)
    # Simulation parameters consumed by Sandbox_Execute.
    initial_state: Optional[List[float]] = Field(default=None, max_length=4096)
    dt: float = Field(default=0.01, gt=0.0, le=10.0)
    steps: int = Field(default=1000, ge=1, le=100000)


class ModuleLayer(BaseModel):
    """A pinned, verified checkpoint layer of the unified research module."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_id: str
    stage: int = Field(ge=1)
    code: str
    bundle_hash: str
    contracts: List[VerificationContract] = Field(default_factory=list)
    baseline_invariants: List[InvariantConstraint] = Field(default_factory=list)
    telemetry: Dict[str, float] = Field(default_factory=dict)


class StageTelemetry(BaseModel):
    """Per-stage outcome recorded in the blueprint evolution history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: int = Field(ge=1)
    status: Literal["VERIFIED", "FALSIFIED", "DIVERGED", "ERROR", "REFUSED"]
    bundle_hash: str = ""
    telemetry: Dict[str, float] = Field(default_factory=dict)
    note: str = ""
    timestamp: float = 0.0


FeasibilityVerdict = Literal[
    "OPTIMAL_DIRECT",
    "REFORMULATED_SURROGATE",
    "REFUSED_IMPOSSIBLE",
    "REFUSED_UNSUITABLE",
]


class FeasibilityAssessment(BaseModel):
    """Epistemic self-awareness verdict produced before any simulation runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: FeasibilityVerdict
    rationale: str
    violated_boundary: str = ""
    reformulation: str = ""
    assessed_by: Literal["heuristic", "llm"] = "heuristic"


class ResearchBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blueprint_id: str
    title: str = Field(min_length=5, max_length=120)
    falsifiable_hypothesis: str
    search_queries: List[str]
    contracts: List[VerificationContract] = Field(min_length=1)
    code_template: str
    max_feedback_rounds: int = Field(default=3, ge=1, le=10)
    # -- cumulative accretion state (populated by the state machine) --------
    verified_layers: List[ModuleLayer] = Field(default_factory=list)
    target_research_stages: int = Field(default=1, ge=1, le=10)
    stage_objectives: List[str] = Field(default_factory=list, max_length=10)
    current_stage: int = Field(default=1, ge=1, le=10)
    evolution_history: List[StageTelemetry] = Field(default_factory=list)
    unified_module_code: Optional[str] = None
    feasibility: Optional[FeasibilityAssessment] = None
    # Named observables: human-readable metric (with units, e.g.
    # "magnetic_field_B_tesla") -> index into the final buf.data state vector.
    # Sandbox_Execute publishes each mapping as telemetry["obs_<name>"].
    variable_map: Dict[str, int] = Field(default_factory=dict, max_length=64)


class SimulationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    blueprint_id: str
    contract_id: str
    status: Literal["VERIFIED", "FALSIFIED", "DIVERGED", "ERROR"]
    execution_time_ms: float
    telemetry: Dict[str, float]
    bundle_hash: str
    failure_counterexample: Optional[str] = None
    stage: Optional[int] = None

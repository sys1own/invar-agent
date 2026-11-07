"""Pydantic v2 schemas for the Research Agent verification contracts."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _clean_text(value: Any, fallback: str = "") -> str:
    """Collapse whitespace and trim; never raise on odd LLM output."""
    if value is None:
        return fallback
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.strip().split())


def _slugify(value: Any, fallback: str = "unnamed", max_length: int = 64) -> str:
    """Slugify an identifier: lowercase, safe chars only, length-capped."""
    if not value or not isinstance(value, str):
        return fallback
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return (slug[:max_length] if slug else fallback)


class InvariantConstraint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variable_name: str
    assertion: str
    tolerance: float = Field(default=1e-6, ge=0.0)


class VerificationContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str

    @field_validator("contract_id", mode="before")
    @classmethod
    def _sanitize_contract_id(cls, value: Any) -> str:
        return _slugify(value, fallback="contract_unnamed")
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


class FeasibilityVerdict(str, Enum):
    OPTIMAL_DIRECT = "OPTIMAL_DIRECT"
    REFORMULATED_SURROGATE = "REFORMULATED_SURROGATE"
    REFUSED_IMPOSSIBLE = "REFUSED_IMPOSSIBLE"
    REFUSED_UNSUITABLE = "REFUSED_UNSUITABLE"

    @classmethod
    def _missing_(cls, value):
        # Case-insensitive coercion: LLM JSON often lowercases verdicts.
        for member in cls:
            if member.value.lower() == str(value).lower():
                return member
        return None


class FeasibilityAssessment(BaseModel):
    """Epistemic self-awareness verdict produced before any simulation runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: FeasibilityVerdict
    rationale: str
    violated_boundary: Optional[str] = None
    reformulation: Optional[str] = None
    assessed_by: Literal["heuristic", "llm"] = "heuristic"

    @field_validator("verdict", mode="before")
    @classmethod
    def _coerce_verdict_case(cls, value):
        # LLM JSON often lowercases/mixed-cases verdicts; Enum._missing_ alone
        # is not consulted by pydantic's str-enum path, so normalise here.
        if isinstance(value, str):
            upper = value.upper()
            for member in FeasibilityVerdict:
                if member.value == upper:
                    return member
        return value

    @model_validator(mode="before")
    @classmethod
    def _feasible_verdicts_carry_no_refusal_fields(cls, data):
        # A passing verdict must never look like a refusal downstream:
        # OPTIMAL_DIRECT has no violated boundary and no reformulation.
        # REFORMULATED_SURROGATE keeps its reformulation (that IS the explicit
        # surrogate reduction) but has no violated boundary.
        if not isinstance(data, dict):
            return data
        verdict = data.get("verdict")
        if isinstance(verdict, FeasibilityVerdict):
            verdict = verdict.value
        verdict = str(verdict or "").upper()
        if verdict == FeasibilityVerdict.OPTIMAL_DIRECT.value:
            data = dict(data)
            data["violated_boundary"] = None
            data["reformulation"] = None
        elif verdict == FeasibilityVerdict.REFORMULATED_SURROGATE.value \
                and not str(data.get("reformulation") or "").strip():
            data = dict(data)
            data["reformulation"] = None
        return data


class ResearchBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blueprint_id: str

    @field_validator("blueprint_id", mode="before")
    @classmethod
    def _sanitize_blueprint_id(cls, value: Any) -> str:
        return _slugify(value, fallback="blueprint_unnamed")

    title: str = Field(min_length=5, max_length=256)

    @field_validator("title", mode="before")
    @classmethod
    def _normalize_title(cls, value: Any) -> str:
        if isinstance(value, str):
            cleaned = _clean_text(value)
            if len(cleaned) > 120:
                return cleaned[:117] + "..."
            return cleaned
        return _clean_text(value) if value is not None else "Untitled Research Blueprint"
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

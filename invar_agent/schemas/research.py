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


class ResearchBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blueprint_id: str
    title: str = Field(min_length=5, max_length=120)
    falsifiable_hypothesis: str
    search_queries: List[str]
    contracts: List[VerificationContract] = Field(min_length=1)
    code_template: str
    max_feedback_rounds: int = Field(default=3, ge=1, le=10)


class SimulationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    blueprint_id: str
    contract_id: str
    status: Literal["VERIFIED", "FALSIFIED", "DIVERGED", "ERROR"]
    execution_time_ms: float
    telemetry: Dict[str, float]
    bundle_hash: str
    failure_counterexample: Optional[str] = None

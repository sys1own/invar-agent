"""Strict receipt schema and canonical domain-separated Merkle root."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_serializer

from ..his_cache import digest
from ..ingestion.schemas import StrictModel
from .base import ModuleEmission


class ArtifactReceipt(StrictModel):
    relpath: str
    module_id: str
    his_digest: str
    content_blake3: str
    byte_size: int
    emitted_by: Literal['deterministic_emitter','llm_extension_slot','deterministic_fallback']


class SlotFallbackRecord(StrictModel):
    module_id: str
    fallback_emitter_id: str
    frozen_his_digest: str
    trip_reason: str
    fault_fingerprint_digest: str
    tier2_attempts_used: int
    fallback_verified: bool


class VerificationReceipt(StrictModel):
    goi_spectral_radius: float
    hin_homogeneous_equations: int
    hin_total_equations: int
    z3_contracts_proved: int
    z3_contracts_failed: int
    pytest_exit_code: int
    pytest_passed: int
    pytest_failed: int
    cabi_compiled: bool
    python_fallback_active: bool
    evidence: dict[str, Any] | None = None
    research_parity: str | None = None

    @model_serializer(mode="wrap")
    def _omit_empty(self, handler):
        data = handler(self)
        if self.evidence is None:
            data.pop("evidence", None)
        if self.research_parity is None:
            data.pop("research_parity", None)
        return data


class BuildReceipt(StrictModel):
    schema_version: Literal['build_receipt.v1']='build_receipt.v1'
    build_id: str
    project_id: str
    status: Literal['COMPLETED','COMPLETED_WITH_FALLBACKS','ABORTED_CRITICAL']
    canonical_ir_digest: str
    uast_digest: str
    equation_provenance_map: dict[str,str]=Field(default_factory=dict)
    artifacts: list[ArtifactReceipt]=Field(default_factory=list)
    fallbacks: list[SlotFallbackRecord]=Field(default_factory=list)
    verification: VerificationReceipt
    total_llm_tokens_used: int=0
    merkle_root_blake3: str=Field(min_length=64,max_length=64)


def seal_receipt(payload):
    payload={key:value for key,value in payload.items() if key!='merkle_root_blake3'}
    return BuildReceipt(**payload,merkle_root_blake3=digest('invar.receipt.merkle.v1',payload))


def emit(node,context):
    tests=next(item.target_relpaths for item in context.uast.nodes if item.node_id=='mod_tests')
    return ModuleEmission({'provenance/build_receipt.json':'{}\n'},pytest_paths=tests)

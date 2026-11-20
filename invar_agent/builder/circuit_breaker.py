"""Ordered, monotone, build-wide repair budgets."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from .error_taxonomy import CircuitAction, ErrorTier
from .ingestion.schemas import StrictModel
from .uast_models import builder_backend


class ProgressMetric(StrictModel):
    has_critical: int = Field(ge=0, le=1)
    hard_error_count: int = Field(ge=0)
    hin_violation_count: int = Field(ge=0)
    z3_violation_count: int = Field(ge=0)
    failed_test_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)

    def as_tuple(self):
        return (self.has_critical, self.hard_error_count, self.hin_violation_count, self.z3_violation_count, self.failed_test_count, self.warning_count)


class FaultFingerprint(StrictModel):
    module_id: str
    stage: Literal["Ingest", "Wavefront_Schedule", "Overlay_Mutate", "PreWrite_Guard", "SMT_Concolic_Verify", "Sandbox_Execute"]
    error_code: str
    exception_class: str
    canonical_location: str
    normalized_message: str
    blake3_digest: str = Field(min_length=64, max_length=64)


class HealingDecision(StrictModel):
    module_id: str
    round_index: int
    tier: ErrorTier
    action: CircuitAction
    reason_code: str
    fault_fingerprint: FaultFingerprint
    progress_metric: ProgressMetric
    hin_causal_slice_nodes: list[str] = Field(default_factory=list)
    estimated_llm_tokens: int = 0
    fallback_emitter_id: str | None = None


_VOLATILE_PATTERNS = [
    (re.compile(r"round_\d+"), "round_<N>"),
    (re.compile(r"0x[0-9a-fA-F]{6,16}"), "0x<ADDR>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"), "<TS>"),
    (re.compile(r'''(/tmp/|/var/folders/|C:\\Users\\[^\\]+\\AppData\\Local\\Temp\\)[^\s:"']+'''), "<TMP_PATH>"),
    (re.compile(r"\s+"), " "),
]


def normalize_diagnostic(raw_message: str) -> str:
    text = raw_message.strip()
    for pattern, replacement in _VOLATILE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text[:512]


def compute_fault_fingerprint(module_id, stage, error_code, exception_class, canonical_location, raw_message):
    message = normalize_diagnostic(raw_message)
    payload = [module_id, stage, error_code, exception_class, canonical_location, message]
    return FaultFingerprint(module_id=module_id, stage=stage, error_code=error_code, exception_class=exception_class,
        canonical_location=canonical_location, normalized_message=message,
        blake3_digest=builder_backend().compute_fault_fingerprint(json.dumps(payload, separators=(",", ":"))))


@dataclass
class SlotBreakerState:
    module_id: str
    tier0_attempts: int = 0
    tier1_attempts: int = 0
    tier2_attempts: int = 0
    seen_fault_digests: set[str] = field(default_factory=set)
    seen_ast_digests: set[str] = field(default_factory=set)
    best_metric: ProgressMetric | None = None
    tripped_reason: str | None = None


@dataclass
class BuildBreakerBudget:
    max_tier0_per_slot: int = 2
    max_tier1_per_slot: int = 2
    max_tier2_per_slot: int = 2
    max_global_tier2_repairs: int = 6
    max_global_repair_tokens: int = 24000
    max_wall_clock_sec: float = 300.0
    used_global_tier2_repairs: int = 0
    used_repair_tokens: int = 0
    elapsed_sec: float = 0.0


class CircuitBreakerController:
    def decide(self, *, module_id, node_kind, round_index, tier, fingerprint, metric, candidate_ast_digest,
               slot_state, budget, fallback_emitter_id=None, missing_compiler_fallback=False, hin_slice_nodes=None, max_tier2_per_slot=None):
        def emit(action, reason, fb_id=None):
            return HealingDecision(module_id=module_id, round_index=round_index, tier=tier, action=action, reason_code=reason,
                fault_fingerprint=fingerprint, progress_metric=metric, hin_causal_slice_nodes=hin_slice_nodes or [], fallback_emitter_id=fb_id)

        def trip(reason):
            slot_state.tripped_reason = reason
            return emit(CircuitAction.SUBSTITUTE_DETERMINISTIC_FALLBACK if node_kind == "llm_extension_slot" else CircuitAction.ABORT_BUILD_CRITICAL,
                        reason, fallback_emitter_id if node_kind == "llm_extension_slot" else None)

        if tier == ErrorTier.CRITICAL_NON_RETRYABLE:
            if missing_compiler_fallback:
                return emit(CircuitAction.SWITCH_TO_PYTHON_FALLBACK, f"CRITICAL_TOOLCHAIN_FALLBACK:{fingerprint.error_code}")
            if node_kind == "llm_extension_slot" and fingerprint.error_code in {"LLM_AUTH_ERROR", "LLM_QUOTA_EXHAUSTED", "LLM_RATE_LIMIT_EXHAUSTED", "SANDBOX_SECURITY_VIOLATION"}:
                return trip(f"CRITICAL_SLOT_TRIP:{fingerprint.error_code}")
            reason = "TIER2_FORBIDDEN_FOR_DETERMINISTIC_MODULE" if fingerprint.error_code == "DETERMINISTIC_EMITTER_BUG" else f"CRITICAL_ABORT:{fingerprint.error_code}"
            return emit(CircuitAction.ABORT_BUILD_CRITICAL, reason)
        if budget.elapsed_sec >= budget.max_wall_clock_sec:
            return trip("WALL_CLOCK_BUDGET_EXHAUSTED")
        if fingerprint.blake3_digest in slot_state.seen_fault_digests:
            return trip(f"DUPLICATE_FAULT_FINGERPRINT:{fingerprint.blake3_digest[:12]}")
        if candidate_ast_digest in slot_state.seen_ast_digests:
            return trip(f"OSCILLATING_PATCH_DIGEST:{candidate_ast_digest[:12]}")
        if slot_state.best_metric is not None and not metric.as_tuple() < slot_state.best_metric.as_tuple():
            return trip(f"NON_MONOTONIC_PROGRESS:{metric.as_tuple()}>={slot_state.best_metric.as_tuple()}")
        slot_state.seen_fault_digests.add(fingerprint.blake3_digest)
        slot_state.seen_ast_digests.add(candidate_ast_digest)
        slot_state.best_metric = metric
        if tier == ErrorTier.TIER0_REFLEX:
            if slot_state.tier0_attempts < budget.max_tier0_per_slot:
                slot_state.tier0_attempts += 1
                return emit(CircuitAction.APPLY_TIER0, f"TIER0_ATTEMPT_{slot_state.tier0_attempts}")
            if node_kind != "llm_extension_slot":
                return emit(CircuitAction.ABORT_BUILD_CRITICAL, "DETERMINISTIC_NODE_TIER0_EXHAUSTED")
        if tier == ErrorTier.TIER1_STRUCTURAL:
            if slot_state.tier1_attempts < budget.max_tier1_per_slot:
                slot_state.tier1_attempts += 1
                return emit(CircuitAction.APPLY_TIER1, f"TIER1_ATTEMPT_{slot_state.tier1_attempts}")
            if node_kind != "llm_extension_slot":
                return emit(CircuitAction.ABORT_BUILD_CRITICAL, "DETERMINISTIC_NODE_TIER1_EXHAUSTED")
        if node_kind != "llm_extension_slot":
            return emit(CircuitAction.ABORT_BUILD_CRITICAL, f"TIER2_FORBIDDEN_FOR_{node_kind.upper()}")
        if slot_state.tier2_attempts >= min(budget.max_tier2_per_slot, max_tier2_per_slot if max_tier2_per_slot is not None else budget.max_tier2_per_slot):
            return trip("SLOT_TIER2_CAP_EXHAUSTED")
        if budget.used_global_tier2_repairs >= budget.max_global_tier2_repairs:
            return trip("GLOBAL_TIER2_CAP_EXHAUSTED")
        if budget.used_repair_tokens >= budget.max_global_repair_tokens:
            return trip("REPAIR_TOKEN_BUDGET_EXHAUSTED")
        slot_state.tier2_attempts += 1
        budget.used_global_tier2_repairs += 1
        return emit(CircuitAction.INVOKE_TIER2_LLM, f"TIER2_ATTEMPT_{slot_state.tier2_attempts}")

"""Deterministic Three-Tier Cognitive Router gating Tier 0 Reflex, Tier 1 Flash, and Tier 2 Pro."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CognitiveTier(str, Enum):
    TIER_0_REFLEX = "TIER_0_REFLEX"
    TIER_1_FLASH = "TIER_1_FLASH"
    TIER_2_PRO = "TIER_2_PRO"


class TaskEventType(str, Enum):
    GOI_DAG_PLANNING = "GOI_DAG_PLANNING"
    CONTRACT_SYNTHESIS = "CONTRACT_SYNTHESIS"
    WAVEFRONT_NODE_EXEC = "WAVEFRONT_NODE_EXEC"
    TYPED_HOLE_FILL = "TYPED_HOLE_FILL"
    CONTEXT_COMPRESSION = "CONTEXT_COMPRESSION"
    STRUCTURAL_MERGE_CONFLICT = "STRUCTURAL_MERGE_CONFLICT"
    SELF_HEAL_DIAGNOSTIC = "SELF_HEAL_DIAGNOSTIC"
    CROSS_MODULE_REFACTOR = "CROSS_MODULE_REFACTOR"


@dataclass(frozen=True)
class ModelRouteDecision:
    tier: CognitiveTier
    model_identifier: str
    event_type: TaskEventType
    max_input_tokens: int
    max_output_tokens: int
    temperature: float
    confidence_score: float
    blast_radius_score: float
    routing_rationale: str
    reflex_handler: Optional[str] = None


@dataclass
class TokenBudgetGuardrail:
    max_pro_tokens: int = 250_000
    max_flash_tokens: int = 1_000_000
    max_tier_1_retries: int = 2
    max_tier_2_retries: int = 2
    used_pro_tokens: int = 0
    used_flash_tokens: int = 0

    def record_usage(self, tier: CognitiveTier, input_tokens: int, output_tokens: int) -> None:
        total = input_tokens + output_tokens
        if tier == CognitiveTier.TIER_2_PRO:
            self.used_pro_tokens += total
            if self.used_pro_tokens > self.max_pro_tokens:
                raise RuntimeError("Pro-Tier token budget exceeded; halting autonomous escalation.")
        elif tier == CognitiveTier.TIER_1_FLASH:
            self.used_flash_tokens += total
            if self.used_flash_tokens > self.max_flash_tokens:
                raise RuntimeError("Flash-Tier token budget exceeded; halting autonomous execution.")


class CognitiveModelRouter:
    """Deterministic Three-Tier Cognitive Router gating Tier 0 Reflex, Tier 1 Flash, and Tier 2 Pro."""

    def __init__(
        self,
        pro_model: str = "gemini-2.5-pro",
        flash_model: str = "gemini-2.5-flash",
        cache_confidence_threshold: float = 0.92,
        blast_radius_threshold: float = 0.65,
        budget_guardrail: Optional[TokenBudgetGuardrail] = None,
    ) -> None:
        self.pro_model = pro_model
        self.flash_model = flash_model
        self.cache_threshold = cache_confidence_threshold
        self.blast_threshold = blast_radius_threshold
        self.budget = budget_guardrail or TokenBudgetGuardrail()

    @staticmethod
    def compute_blast_radius(
        affected_hin_nodes: int,
        cyclomatic_complexity: int,
        modifies_holographic_boundary: bool,
    ) -> float:
        boundary_weight = 0.50 if modifies_holographic_boundary else 0.0
        node_weight = 0.25 * min(1.0, affected_hin_nodes / 10.0)
        cc_weight = 0.25 * min(1.0, cyclomatic_complexity / 25.0)
        return round(min(1.0, boundary_weight + node_weight + cc_weight), 4)

    def route(
        self,
        event_type: TaskEventType,
        *,
        rust_healer_matched: bool = False,
        fix_cache_confidence: float = 0.0,
        affected_hin_nodes: int = 1,
        cyclomatic_complexity: int = 5,
        modifies_holographic_boundary: bool = False,
        tier_1_failures: int = 0,
        tier_2_failures: int = 0,
    ) -> ModelRouteDecision:
        blast_score = self.compute_blast_radius(
            affected_hin_nodes, cyclomatic_complexity, modifies_holographic_boundary
        )

        # 1. Tier 0 Zero-Token Reflex Path always takes precedence if deterministic match exists
        if rust_healer_matched:
            return ModelRouteDecision(
                tier=CognitiveTier.TIER_0_REFLEX,
                model_identifier="native::PyDeterministicHealer",
                event_type=event_type,
                max_input_tokens=0,
                max_output_tokens=0,
                temperature=0.0,
                confidence_score=1.0,
                blast_radius_score=blast_score,
                routing_rationale="Matched deterministic Rust AST/token transducer rule.",
                reflex_handler="deterministic_healer.rs",
            )

        if fix_cache_confidence >= self.cache_threshold:
            return ModelRouteDecision(
                tier=CognitiveTier.TIER_0_REFLEX,
                model_identifier="cache::ContentAddressableFixCache",
                event_type=event_type,
                max_input_tokens=0,
                max_output_tokens=0,
                temperature=0.0,
                confidence_score=fix_cache_confidence,
                blast_radius_score=blast_score,
                routing_rationale=f"FixCache hit ({fix_cache_confidence:.3f}) >= {self.cache_threshold}.",
                reflex_handler="fix_cache.py",
            )

        # 2. Check if Tier 2 retries are exhausted -> force Adjoint Rollback upstream
        if tier_2_failures >= self.budget.max_tier_2_retries:
            raise RuntimeError(
                f"Tier 2 Pro exhausted ({tier_2_failures} failures); trigger Adjoint Rollback."
            )

        # 3. Tier 2 Pro Escalation Check
        pro_mandatory_events = {
            TaskEventType.GOI_DAG_PLANNING,
            TaskEventType.CONTRACT_SYNTHESIS,
            TaskEventType.CROSS_MODULE_REFACTOR,
        }
        if (
            event_type in pro_mandatory_events
            or modifies_holographic_boundary
            or blast_score >= self.blast_threshold
            or tier_1_failures >= self.budget.max_tier_1_retries
        ):
            return ModelRouteDecision(
                tier=CognitiveTier.TIER_2_PRO,
                model_identifier=self.pro_model,
                event_type=event_type,
                max_input_tokens=32_768,
                max_output_tokens=8_192,
                temperature=0.1,
                confidence_score=max(0.5, 1.0 - 0.15 * tier_2_failures),
                blast_radius_score=blast_score,
                routing_rationale=(
                    f"Architectural event={event_type.value}, "
                    f"blast_radius={blast_score:.2f}, tier_1_failures={tier_1_failures}"
                ),
            )

        # 4. Default to Tier 1 Flash High-Throughput Worker
        return ModelRouteDecision(
            tier=CognitiveTier.TIER_1_FLASH,
            model_identifier=self.flash_model,
            event_type=event_type,
            max_input_tokens=12_288,
            max_output_tokens=4_096,
            temperature=0.0,
            confidence_score=max(0.5, 0.90 - 0.20 * tier_1_failures),
            blast_radius_score=blast_score,
            routing_rationale="Localized wavefront task within Flash complexity envelope.",
        )

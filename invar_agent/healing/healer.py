"""Autonomic healer orchestrator: Tier-0 reflex -> Tier-1 flash -> Tier-2 pro."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from invar_agent.contracts import ASTSurgicalPatch, HealingVerdict, TaskWavefrontNode

from .context_builder import HealingContextBuilder
from .error_classifier import ClassifiedErrorPayload, DeterministicErrorClassifier, FailureCategory
from .error_explainer import FormalErrorExplainer
from .evaluator import PatchFitnessEvaluator
from .hin_graph import CausalFaultInteractionGraph
from .llm_healer import LLMRepairResult, SurgicalLLMHealer
from .router import HealingTierRouter

try:
    from invar_agent._native import PyDeterministicHealer as _NativeHealer
except Exception:  # pragma: no cover
    _NativeHealer = None


@dataclass
class HealingAttempt:
    tier: str
    success: bool
    rule_id: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    rationale: str = ""


class AutonomicHealerOrchestrator:
    """Runs the Reflex -> Flash -> Pro healing escalation ladder."""

    def __init__(
        self,
        native_healer=None,
        fix_cache=None,
        flash_healer: Optional[SurgicalLLMHealer] = None,
        pro_healer: Optional[SurgicalLLMHealer] = None,
        classifier: Optional[DeterministicErrorClassifier] = None,
        explainer: Optional[FormalErrorExplainer] = None,
        evaluator: Optional[PatchFitnessEvaluator] = None,
        context_builder: Optional[HealingContextBuilder] = None,
        tier_router: Optional[HealingTierRouter] = None,
        hin_graph: Optional[CausalFaultInteractionGraph] = None,
    ) -> None:
        self.native_healer = native_healer if native_healer is not None else (_NativeHealer() if _NativeHealer else None)
        self.fix_cache = fix_cache
        self.flash_healer = flash_healer or SurgicalLLMHealer()
        self.pro_healer = pro_healer or SurgicalLLMHealer()
        self.classifier = classifier or DeterministicErrorClassifier()
        self.explainer = explainer or FormalErrorExplainer()
        self.evaluator = evaluator or PatchFitnessEvaluator()
        self.context_builder = context_builder or HealingContextBuilder()
        self.tier_router = tier_router or HealingTierRouter()
        self.hin_graph = hin_graph

    # -- main entry -----------------------------------------------------------------

    def run_reflex_then_escalate(
        self,
        classified: ClassifiedErrorPayload,
        source_by_file: dict[str, str],
        node: Optional[TaskWavefrontNode] = None,
        failing_source: Optional[str] = None,
        failing_line: int = 0,
        max_tier1_attempts: int = 2,
        max_tier2_attempts: int = 1,
    ) -> HealingVerdict:
        attempts: list[HealingAttempt] = []
        total_tokens = 0
        start_all = time.perf_counter()

        # ---- Tier 0: deterministic reflex -------------------------------------
        reflex = self._try_tier0(classified, source_by_file, failing_source, failing_line)
        if reflex is not None:
            healed_source, rule_id, latency = reflex
            attempts.append(HealingAttempt("TIER_0_REFLEX", True, rule_id, latency))
            return HealingVerdict(
                healed=True,
                resolved_by_tier="TIER_0_REFLEX",
                attempts_used=1,
                tokens_consumed=0,
                latency_ms=(time.perf_counter() - start_all) * 1000.0,
                applied_rule_or_rationale=rule_id,
                error_category=classified.category.value,
                repaired_patches=self._patches_from_source(classified, healed_source, source_by_file),
            )

        tier1_failures = 0
        tier2_failures = 0
        last_result: Optional[LLMRepairResult] = None

        # ---- Tier 1: flash surgical --------------------------------------------
        for _ in range(max_tier1_attempts):
            result = self.flash_healer.generate_ast_repair_patch(
                classified, source_by_file, prompt_payload=self._payload(classified, source_by_file, node)
            )
            total_tokens += result.tokens_used
            tier1_failures += 1
            attempts.append(HealingAttempt("TIER_1_FLASH", result.success, latency_ms=0.0,
                                           tokens_used=result.tokens_used, rationale=result.rationale))
            if result.success:
                return HealingVerdict(
                    healed=True,
                    resolved_by_tier="TIER_1_FLASH",
                    attempts_used=len(attempts),
                    tokens_consumed=total_tokens,
                    latency_ms=(time.perf_counter() - start_all) * 1000.0,
                    applied_rule_or_rationale=result.rationale,
                    error_category=classified.category.value,
                    repaired_patches=result.patches,
                )
            last_result = result
            if result.unrecoverable:
                break

        # ---- Tier 2: pro architectural ------------------------------------------
        for _ in range(max_tier2_attempts):
            result = self.pro_healer.generate_ast_repair_patch(
                classified, source_by_file, prompt_payload=self._payload(classified, source_by_file, node)
            )
            total_tokens += result.tokens_used
            tier2_failures += 1
            attempts.append(HealingAttempt("TIER_2_PRO", result.success, latency_ms=0.0,
                                           tokens_used=result.tokens_used, rationale=result.rationale))
            if result.success:
                # Record into fix cache for future Tier-0 reflex resolution.
                self._record_fix(classified, result)
                return HealingVerdict(
                    healed=True,
                    resolved_by_tier="TIER_2_PRO",
                    attempts_used=len(attempts),
                    tokens_consumed=total_tokens,
                    latency_ms=(time.perf_counter() - start_all) * 1000.0,
                    applied_rule_or_rationale=result.rationale,
                    error_category=classified.category.value,
                    repaired_patches=result.patches,
                )
            last_result = result
            if result.unrecoverable:
                break

        return HealingVerdict(
            healed=False,
            resolved_by_tier="UNRESOLVED",
            attempts_used=len(attempts),
            tokens_consumed=total_tokens,
            latency_ms=(time.perf_counter() - start_all) * 1000.0,
            trigger_adjoint_rollback=True,
            applied_rule_or_rationale=(last_result.rationale if last_result else "all tiers exhausted"),
            error_category=classified.category.value,
        )

    def heal_from_stderr(
        self,
        stderr: str,
        exit_code: int,
        source_by_file: dict[str, str],
        node: Optional[TaskWavefrontNode] = None,
        **kwargs: Any,
    ) -> HealingVerdict:
        classified = self.classifier.classify_stderr_trace(stderr, exit_code)
        failing_source = None
        failing_line = 0
        if classified.failing_file and classified.failing_file in source_by_file:
            failing_source = source_by_file[classified.failing_file]
        if classified.failing_line:
            failing_line = classified.failing_line
        return self.run_reflex_then_escalate(
            classified, source_by_file, node=node,
            failing_source=failing_source, failing_line=failing_line, **kwargs
        )

    # -- Tier 0 ---------------------------------------------------------------------

    def _try_tier0(self, classified: ClassifiedErrorPayload, source_by_file: dict[str, str],
                   failing_source: Optional[str], source_line: int) -> Optional[tuple[str, str, float]]:
        start = time.perf_counter()
        # 1. Native deterministic healer.
        if self.native_healer is not None and failing_source:
            try:
                healed = self.native_healer.match_and_heal(
                    classified.error_code, classified.diagnostic_msg, failing_source, max(0, source_line)
                )
            except Exception:  # noqa: BLE001
                healed = None
            if healed:
                new_source, rule_id = healed
                return new_source, rule_id, (time.perf_counter() - start) * 1000.0
        # 2. Fix cache lookup.
        if self.fix_cache is not None and failing_source:
            record = self.fix_cache.lookup_reflex_patch(
                classified.category.value, classified.diagnostic_msg, failing_source[:400]
            )
            if record is not None:
                payload = record.patch_payload
                replacements = payload.get("replacements") or {}
                target = classified.failing_file or next(iter(source_by_file), "")
                if target in replacements:
                    latency = (time.perf_counter() - start) * 1000.0
                    return replacements[target], f"fix_cache:{record.signature.key()[:12]}", latency
        return None

    # -- helpers --------------------------------------------------------------------

    def heal_staged_sources_tier0_only(
        self,
        staged: dict[str, str],
        diagnostic: str,
        node: Optional[TaskWavefrontNode] = None,
    ) -> Optional[dict[str, str]]:
        """Tier-0-only heal of staged buffers; returns healed sources or None."""
        classified = self.classifier.classify_stderr_trace(diagnostic, 1)
        # Python syntax errors embedded in tool output (libcst etc.) can be
        # misclassified as Rust syntax; force the Python category.
        if "SyntaxError" in diagnostic or "Syntax Error" in diagnostic:
            classified = ClassifiedErrorPayload(
                category=FailureCategory.SYNTAX_ERROR,
                error_code="SyntaxError",
                diagnostic_msg=diagnostic.strip().splitlines()[-1].strip(),
                failing_file=classified.failing_file,
                failing_line=classified.failing_line or 0,
                stderr_tail=diagnostic[-2000:],
            )
        # Native Tier-0 reflex: try each staged file as the failing source.
        native_diag = classified.diagnostic_msg
        if classified.category == FailureCategory.SYNTAX_ERROR and "expected ':'" not in native_diag:
            native_diag = "SyntaxError: expected ':'"
        if self.native_healer is not None:
            for path, content in staged.items():
                try:
                    healed_pair = self.native_healer.match_and_heal(
                        classified.error_code, native_diag, content, 0
                    )
                except Exception:  # noqa: BLE001
                    healed_pair = None
                if healed_pair:
                    new_source, _rule = healed_pair
                    healed = dict(staged)
                    healed[path] = new_source
                    return healed
        verdict = self.run_reflex_then_escalate(
            classified, dict(staged), node=node,
            max_tier1_attempts=0, max_tier2_attempts=0,
        )
        if not verdict.healed or verdict.resolved_by_tier != "TIER_0_REFLEX":
            return None
        healed = dict(staged)
        for patch in verdict.repaired_patches:
            if patch.operation == "REPLACE_EXACT_SPAN":
                base = healed.get(patch.file_path, "")
                if patch.search_block and patch.search_block in base:
                    healed[patch.file_path] = base.replace(patch.search_block, patch.replacement_code, 1)
            elif patch.replacement_code.strip():
                healed[patch.file_path] = patch.replacement_code
        # The Tier-0 reflex may have healed a file not covered by patches when the
        # native healer returned whole-file content; detect via patch count.
        if not verdict.repaired_patches:
            return None
        return healed

    def _payload(self, classified: ClassifiedErrorPayload, sources: dict[str, str],
                 node: Optional[TaskWavefrontNode]) -> dict[str, Any]:
        fault_slice: list[str] = []
        if self.hin_graph is not None and classified.failing_file:
            symbol = Path(classified.failing_file).stem
            fault_slice = self.context_builder.extract_minimal_fault_slice(self.hin_graph, symbol)
        return self.context_builder.build_diagnostic_prompt_payload(classified, sources, fault_slice)

    def _patches_from_source(self, classified: ClassifiedErrorPayload, healed_source: str,
                             source_by_file: dict[str, str]) -> list[ASTSurgicalPatch]:
        target = classified.failing_file or next(iter(source_by_file), "")
        original = source_by_file.get(target, "")
        if not original or healed_source == original:
            return []
        from ..overlay.patch import compute_ast_delta

        delta = compute_ast_delta(original, healed_source)
        if not (delta["added"] or delta["removed"] or delta["modified"]):
            return []
        return [ASTSurgicalPatch(
            file_path=target,
            operation="REPLACE_EXACT_SPAN",
            search_block=original,
            replacement_code=healed_source,
        )]

    def _record_fix(self, classified: ClassifiedErrorPayload, result: LLMRepairResult) -> None:
        if self.fix_cache is None:
            return
        try:
            self.fix_cache.record_verified_fix(
                classified.category.value,
                classified.diagnostic_msg,
                {"replacements": result.patched_sources},
                confidence=0.9,
            )
        except Exception:  # noqa: BLE001
            pass

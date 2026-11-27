"""3-Pass Staged Scientific IR Synthesizer (`Pass 1 -> Pass 2 -> Pass 3`).

Routes strictly through the repository's configured LLM provider interface
(never hardcoding external model names). Enforces:
  - Pass 1: Prompt + literature/registry -> `FrozenSymbolTable` (immutable once frozen)
  - Pass 2: `FrozenSymbolTable` -> `ModelBodyDraft` -> Rust `scope_check_and_lower`
  - Pass 3: Bounded Single-Symbol Micro-Repair (`max_tokens=256`, `max_micro_repairs=2`)
            under strict Epistemic Lock (NEVER mutates `obligations` or `FrozenSymbolTable`)
  - Sealing: Single-authority Rust RFC 8785 JCS + 1-byte domain BLAKE3 (`0x01`, `0x02`)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from invar_agent.ir.evidence import InconclusiveReasonCode
from invar_agent.ir.ihcg_bridge import SealedSciIR, seal_sciir_ihcg
from invar_agent.ir.json_recovery import (
    EmptyWhitespaceResponseError,
    recover_json_object,
)
from invar_agent.ir.prompt_templates import (
    build_pass1_messages,
    build_pass2_messages,
    build_pass3_micro_repair_messages,
)
from invar_agent.ir.schema import (
    BuilderStudioSpec,
    FrozenSymbolTable,
    ModelBodyDraft,
    SingleSymbolRepairPayload,
)
from invar_agent.ir.typechecker import scope_check_and_lower

# Async callback signature: (messages, max_tokens, tier) -> raw_json_text
LlmCallFn = Callable[[list[dict[str, str]], int, str], Awaitable[str]]


@dataclass
class StagedSynthesisResult:
    ok: bool
    sealed: SealedSciIR | None = None
    frozen_table: FrozenSymbolTable | None = None
    reason_code: InconclusiveReasonCode = InconclusiveReasonCode.NONE
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    resolved_aliases: dict[str, str] = field(default_factory=dict)
    pass1_attempts: int = 0
    pass2_attempts: int = 0
    micro_repairs_used: int = 0
    whitespace_retries: int = 0


class StagedSciIRSynthesizer:
    """Correct-by-construction 3-pass staged synthesizer with lexicographic budgets."""

    def __init__(
        self,
        llm_call: LlmCallFn,
        *,
        max_pass_retries: int = 2,
        max_micro_repairs: int = 2,
        micro_repair_max_tokens: int = 256,
        check_dimensions: bool = False,
    ) -> None:
        self._llm_call = llm_call
        self._max_pass_retries = max(1, min(max_pass_retries, 2))
        self._max_micro_repairs = max(0, min(max_micro_repairs, 2))
        self._micro_repair_max_tokens = min(micro_repair_max_tokens, 256)
        self._check_dimensions = check_dimensions

    async def _call_json_with_whitespace_retry(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        tier: str,
        stats: StagedSynthesisResult,
    ) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(2):
            raw = await self._llm_call(messages, max_tokens, tier)
            try:
                recovered = recover_json_object(raw)
                return recovered.data
            except EmptyWhitespaceResponseError as exc:
                stats.whitespace_retries += 1
                last_err = exc
                if attempt == 0:
                    continue
                raise
            except Exception as exc:
                last_err = exc
                raise
        raise last_err or ValueError("LLM JSON recovery failed")

    async def synthesize(
        self,
        user_prompt: str,
        grounded_context: str = "",
        builder_studio: BuilderStudioSpec | None = None,
    ) -> StagedSynthesisResult:
        result = StagedSynthesisResult(ok=False)

        # ---- PASS 1: FrozenSymbolTable ----
        frozen_table: FrozenSymbolTable | None = None
        p1_messages = build_pass1_messages(user_prompt, grounded_context)
        for _ in range(self._max_pass_retries):
            result.pass1_attempts += 1
            try:
                data = await self._call_json_with_whitespace_retry(
                    p1_messages, max_tokens=4096, tier="fast", stats=result
                )
                frozen_table = FrozenSymbolTable.model_validate(data).assign_canonical_slot_indices()
                break
            except Exception as exc:
                result.diagnostics.append(
                    {"stage": "PASS1", "code": "PASS1_PARSE_ERROR", "message": str(exc)}
                )

        if frozen_table is None:
            result.reason_code = InconclusiveReasonCode.PASS1_SYNTHESIS_EXHAUSTED
            return result
        result.frozen_table = frozen_table

        # ---- PASS 2: ModelBodyDraft against FrozenSymbolTable ----
        body: ModelBodyDraft | None = None
        p2_messages = build_pass2_messages(user_prompt, frozen_table, grounded_context)
        for _ in range(self._max_pass_retries):
            result.pass2_attempts += 1
            try:
                data = await self._call_json_with_whitespace_retry(
                    p2_messages, max_tokens=12288, tier="reasoning", stats=result
                )
                data = self._normalize_pass2_structural_fields(data, frozen_table)
                body = ModelBodyDraft.model_validate(data)
                break
            except Exception as exc:
                result.diagnostics.append(
                    {"stage": "PASS2", "code": "PASS2_PARSE_ERROR", "message": str(exc)}
                )

        if body is None:
            result.reason_code = InconclusiveReasonCode.PASS2_SYNTHESIS_EXHAUSTED
            return result

        # ARCH-009: Allow ONE versioned, ledgered symbol-table revision when Pass 2
        # legitimately introduces a derived observable absent from the Pass-1 table.
        existing_names = {s.name for s in frozen_table.symbols}
        missing_obs_targets = [
            obs.target for obs in (body.observables or [])
            if obs.target and obs.target not in existing_names and obs.target.isidentifier()
        ]
        if missing_obs_targets:
            from invar_agent.ir.schema import AssumptionEntry, SiDimension, SlotRole, SymbolDecl
            new_syms = list(frozen_table.symbols)
            new_ledger = list(frozen_table.assumptions_ledger)
            for tgt in dict.fromkeys(missing_obs_targets):
                new_syms.append(SymbolDecl(
                    name=tgt,
                    display_symbol=tgt,
                    role=SlotRole.DERIVED_OBSERVABLE,
                    slot_index=len(new_syms),
                    dimension=SiDimension(),
                    initial_or_default=0.0,
                    description=f"Pass-2 derived observable {tgt} added via versioned table revision",
                ))
                new_ledger.append(AssumptionEntry(
                    id=f"REV_OBS_{tgt}",
                    statement=f"Added derived observable '{tgt}' introduced by Pass-2 body.",
                    source="PASS3_TABLE_REVISION",
                    falsifiable_impact="Observable slot only; dynamics unchanged.",
                ))
            frozen_table = frozen_table.model_copy(
                update={"symbols": new_syms, "assumptions_ledger": new_ledger}
            ).assign_canonical_slot_indices()
            result.frozen_table = frozen_table

        # ---- PASS 3: Deterministic Rust Scope Check + Bounded Single-Symbol Micro-Repair ----
        outcome = scope_check_and_lower(
            frozen_table,
            body,
            user_prompt=user_prompt,
            check_dimensions=self._check_dimensions,
            builder_studio=builder_studio,
        )
        result.resolved_aliases.update(outcome.resolved_aliases)

        while not outcome.ok and result.micro_repairs_used < self._max_micro_repairs:
            target_diag = self._select_repairable_diagnostic(outcome.diagnostics)
            if target_diag is None:
                break
            field_path = str(target_diag.get("field", ""))
            current_expr, target_kind, target_name, allowed_lets = self._locate_target_expr(
                body, field_path
            )
            if current_expr is None:
                break

            result.micro_repairs_used += 1
            repair_msgs = build_pass3_micro_repair_messages(
                frozen_table=frozen_table,
                allowed_let_names=allowed_lets,
                diagnostic=target_diag,
                current_expr_text=current_expr,
            )
            try:
                repair_data = await self._call_json_with_whitespace_retry(
                    repair_msgs,
                    max_tokens=self._micro_repair_max_tokens,
                    tier="fast",
                    stats=result,
                )
                patch = SingleSymbolRepairPayload.model_validate(repair_data)
                # Epistemic lock: apply ONLY to the offending let-binding / equation
                body = self._apply_single_symbol_patch(body, target_kind, target_name, patch.expr_text)
            except Exception as exc:
                result.diagnostics.append(
                    {"stage": "PASS3", "code": "MICRO_REPAIR_ERROR", "message": str(exc)}
                )
                break

            outcome = scope_check_and_lower(
                frozen_table,
                body,
                user_prompt=user_prompt,
                check_dimensions=self._check_dimensions,
                builder_studio=builder_studio,
            )
            result.resolved_aliases.update(outcome.resolved_aliases)

        if not outcome.ok or outcome.sci_ir is None:
            result.diagnostics.extend(outcome.diagnostics)
            codes = {d.get("code") for d in outcome.diagnostics}
            if "AMBIGUOUS_COLLISION" in codes:
                result.reason_code = InconclusiveReasonCode.SCOPE_CHECK_AMBIGUOUS_SYMBOL
            elif "DIMENSION_MISMATCH" in codes:
                result.reason_code = InconclusiveReasonCode.DIMENSIONAL_INCONSISTENCY
            else:
                result.reason_code = InconclusiveReasonCode.MICRO_REPAIR_BUDGET_EXHAUSTED
            return result

        # ---- SEALING: Single-Authority Rust RFC 8785 JCS + BLAKE3 ----
        sealed = seal_sciir_ihcg(outcome.sci_ir)
        result.ok = True
        result.sealed = sealed
        result.reason_code = InconclusiveReasonCode.NONE
        return result

    @staticmethod
    def _normalize_pass2_structural_fields(
        data: dict[str, Any], frozen_table: FrozenSymbolTable
    ) -> dict[str, Any]:
        """Pure structural normalization of Pass-2 `config` keys and zero-tolerance
        obligations (clamped to 1e-9). Never alters mathematical expressions or bounds."""
        out = dict(data)
        arch_to_cfg = {
            "ODE_PDE": (
                "ode_pde",
                {"kind", "t_span", "sample_points", "rtol", "atol", "preferred_solver", "enable_buckingham_scaling"},
            ),
            "MONTE_CARLO": (
                "monte_carlo",
                {"kind", "base_seed", "replicates", "confidence_alpha", "use_crn", "observable_range_bounds"},
            ),
            "FIXED_POINT": (
                "fixed_point",
                {"kind", "max_newton_iters", "residual_tol", "require_krawczyk_proof", "require_stability_cert"},
            ),
            "PARAM_SWEEP": (
                "param_sweep",
                {"kind", "grid_strategy", "points_per_axis", "bisection_tol"},
            ),
            "ALGEBRAIC_CLOSURE": (
                "algebraic_closure",
                {"kind", "shbt_federation_enabled", "open_boundary_vars"},
            ),
        }
        kind_val, allowed_keys = arch_to_cfg[frozen_table.archetype.value]
        raw_cfg = out.get("config")
        if isinstance(raw_cfg, dict):
            cleaned_cfg = {k: v for k, v in raw_cfg.items() if k in allowed_keys}
            cleaned_cfg["kind"] = kind_val
            out["config"] = cleaned_cfg
        else:
            out["config"] = {"kind": kind_val}

        raw_obs = out.get("obligations")
        if isinstance(raw_obs, list):
            norm_obs = []
            for ob in raw_obs:
                if isinstance(ob, dict):
                    ob_copy = dict(ob)
                    tol = ob_copy.get("tolerance", 1e-6)
                    if not isinstance(tol, (int, float)) or tol <= 0.0:
                        ob_copy["tolerance"] = 1e-9
                    norm_obs.append(ob_copy)
                else:
                    norm_obs.append(ob)
            out["obligations"] = norm_obs
        return out

    @staticmethod
    def _select_repairable_diagnostic(
        diagnostics: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for d in diagnostics:
            field_path = str(d.get("field", ""))
            if field_path.startswith(
                ("let_bindings.", "dynamics_or_residuals.", "observables.")
            ):
                return d
        return None

    @staticmethod
    def _locate_target_expr(
        body: ModelBodyDraft, field_path: str
    ) -> tuple[str | None, str, str, list[str]]:
        if field_path.startswith("let_bindings."):
            name = field_path.split(".", 1)[1]
            allowed: list[str] = []
            for lb in body.let_bindings:
                if lb.name == name:
                    return lb.expr_text, "let_binding", name, allowed
                allowed.append(lb.name)
        elif field_path.startswith("dynamics_or_residuals."):
            name = field_path.split(".", 1)[1]
            allowed = [lb.name for lb in body.let_bindings]
            for eq in body.dynamics_or_residuals:
                if eq.target == name:
                    return eq.expr_text, "dynamics_or_residuals", name, allowed
        elif field_path.startswith("observables."):
            name = field_path.split(".", 1)[1]
            allowed = [lb.name for lb in body.let_bindings]
            for eq in body.observables:
                if eq.target == name:
                    return eq.expr_text, "observables", name, allowed
        return None, "", "", []

    @staticmethod
    def _apply_single_symbol_patch(
        body: ModelBodyDraft,
        target_kind: str,
        target_name: str,
        new_expr_text: str,
    ) -> ModelBodyDraft:
        if target_kind == "let_binding":
            new_lets = [
                lb.model_copy(update={"expr_text": new_expr_text}) if lb.name == target_name else lb
                for lb in body.let_bindings
            ]
            return body.model_copy(update={"let_bindings": new_lets})
        if target_kind == "dynamics_or_residuals":
            new_dyn = [
                eq.model_copy(update={"expr_text": new_expr_text}) if eq.target == target_name else eq
                for eq in body.dynamics_or_residuals
            ]
            return body.model_copy(update={"dynamics_or_residuals": new_dyn})
        if target_kind == "observables":
            new_obs = [
                eq.model_copy(update={"expr_text": new_expr_text}) if eq.target == target_name else eq
                for eq in body.observables
            ]
            return body.model_copy(update={"observables": new_obs})
        return body

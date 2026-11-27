"""Deterministic PyO3 bridge to Rust `expr_parser.rs` and `scope_check.rs`."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from invar_agent.ir.schema import (
    BuilderStudioSpec,
    EquationSpec,
    FrozenSymbolTable,
    LetBinding,
    ModelBodyDraft,
    ObligationSpec,
    SciIR,
)


@dataclass(frozen=True)
class ScopeCheckOutcome:
    ok: bool
    sci_ir: SciIR | None
    diagnostics: list[dict[str, Any]]
    resolved_aliases: dict[str, str]


def _get_native():
    from invar_agent import _native  # type: ignore[attr-defined]
    return _native


def parse_sci_expr(expr: str) -> dict[str, Any]:
    """Parse a mathematical expression string into a canonical AST dict via Rust."""
    native = _get_native()
    raw = native.py_parse_sci_expr(expr)
    return json.loads(raw)


def scope_check_and_lower(
    frozen_table: FrozenSymbolTable,
    body: ModelBodyDraft,
    user_prompt: str = "",
    check_dimensions: bool = False,
    builder_studio: BuilderStudioSpec | None = None,
) -> ScopeCheckOutcome:
    """Run Rust 4-scope lexical graph, Kahn/Tarjan let-DAG, 4-tier injective
    alias resolution, `Chk-ODE-RHS`, and bidirectional `Q^7` SI checking,
    returning a lowered `SciIR` when `ok=True`."""
    canonical_table = frozen_table.assign_canonical_slot_indices()
    req_payload = {
        "archetype": canonical_table.archetype.value,
        "symbols": [s.model_dump(mode="json") for s in canonical_table.symbols],
        "let_bindings": [lb.model_dump(mode="json") for lb in body.let_bindings],
        "dynamics_or_residuals": [
            eq.model_dump(mode="json") for eq in body.dynamics_or_residuals
        ],
        "observables": [obs.model_dump(mode="json") for obs in body.observables],
        "obligations": [ob.model_dump(mode="json") for ob in body.obligations],
        "check_dimensions": bool(check_dimensions),
    }

    native = _get_native()
    resp = json.loads(native.py_scope_check_sci_ir(json.dumps(req_payload)))
    ok = bool(resp.get("ok", False))
    diagnostics = list(resp.get("diagnostics", []))
    resolved_aliases = dict(resp.get("resolved_aliases", {}))

    if not ok:
        return ScopeCheckOutcome(
            ok=False,
            sci_ir=None,
            diagnostics=diagnostics,
            resolved_aliases=resolved_aliases,
        )

    req_reductions = {o.target: getattr(o, "reduction", None) for o in body.observables}
    resp_reductions = {o["target"]: o.get("reduction") for o in resp.get("observables", [])}
    if req_reductions != resp_reductions:
        return ScopeCheckOutcome(
            ok=False,
            sci_ir=None,
            diagnostics=diagnostics
            + [
                {
                    "code": "COMPILER_INVARIANT",
                    "field": "observables",
                    "symbol": None,
                    "message": (
                        f"Trajectory reduction mismatch between request {req_reductions!r} "
                        f"and scope-checked response {resp_reductions!r}"
                    ),
                }
            ],
            resolved_aliases=resolved_aliases,
        )

    prompt_digest = hashlib.blake2b(
        user_prompt.encode("utf-8"), digest_size=32
    ).hexdigest()

    sci_ir = SciIR(
        schema_version="invar.sciir.v2",
        ir_id=canonical_table.ir_id,
        archetype=canonical_table.archetype,
        domain=canonical_table.domain,
        user_prompt_digest=prompt_digest,
        assumptions_ledger=list(canonical_table.assumptions_ledger),
        symbols=list(canonical_table.symbols),
        conservation_pools=list(canonical_table.conservation_pools),
        let_bindings=[LetBinding.model_validate(x) for x in resp["let_bindings"]],
        dynamics_or_residuals=[
            EquationSpec.model_validate(x) for x in resp["dynamics_or_residuals"]
        ],
        observables=[EquationSpec.model_validate(x) for x in resp["observables"]],
        obligations=[ObligationSpec.model_validate(x) for x in resp["obligations"]],
        config=body.config,
        builder_studio=builder_studio or BuilderStudioSpec(),
    )
    return ScopeCheckOutcome(
        ok=True,
        sci_ir=sci_ir,
        diagnostics=diagnostics,
        resolved_aliases=resolved_aliases,
    )

"""Deterministic lowering from sealed `SciIR` (`CanonicalResearchIR`) to Builder Studio UAST."""
from __future__ import annotations

from typing import Any

from invar_agent.ir.ihcg_bridge import SealedSciIR, seal_sciir_ihcg
from invar_agent.ir.schema import SciIR, SlotRole


def lower_sciir_to_builder_uast(
    sealed_or_ir: SealedSciIR | SciIR,
    *,
    target_languages: list[str] | None = None,
) -> dict[str, Any]:
    """Deterministically lower a sealed `SciIR` into a Builder Studio UAST descriptor
    preserving the exact `0x01 || JCS(SciIR)` BLAKE3 commitment."""
    sealed = (
        sealed_or_ir
        if isinstance(sealed_or_ir, SealedSciIR)
        else seal_sciir_ihcg(sealed_or_ir)
    )
    ir = sealed.ir
    langs = list(target_languages or ir.builder_studio.target_languages or ["python"])

    state_symbols = [s for s in ir.symbols if s.role == SlotRole.STATE]
    param_symbols = [s for s in ir.symbols if s.role == SlotRole.CONTROL_PARAM]
    const_symbols = [s for s in ir.symbols if s.role == SlotRole.CONSTANT]
    obs_symbols = [s for s in ir.symbols if s.role == SlotRole.DERIVED_OBSERVABLE]

    return {
        "schema_version": "invar.builder.uast.v2",
        "ir_id": ir.ir_id,
        "sciir_commitment_blake3": sealed.sciir_commitment_blake3,
        "ihcg_v1_commitment_blake3": sealed.ihcg_v1_commitment_blake3,
        "archetype": ir.archetype.value,
        "domain": ir.domain.value,
        "macro_domain": ir.to_macro_domain(),
        "target_languages": langs,
        "export_workbench_ui": ir.builder_studio.export_workbench_ui,
        "buffer_layout": {
            "n_states": len(state_symbols),
            "n_params": len(param_symbols),
            "n_constants": len(const_symbols),
            "n_observables": len(obs_symbols),
            "states": [
                {"name": s.name, "slot": s.slot_index, "initial": s.initial_or_default, "unit": s.dimension.unit_label}
                for s in state_symbols
            ],
            "params": [
                {
                    "name": s.name,
                    "slot": s.slot_index,
                    "default": s.initial_or_default,
                    "domain": s.domain.model_dump() if s.domain else None,
                    "unit": s.dimension.unit_label,
                }
                for s in param_symbols
            ],
            "constants": [
                {"name": s.name, "slot": s.slot_index, "value": s.initial_or_default, "unit": s.dimension.unit_label}
                for s in const_symbols
            ],
            "observables": [
                {"name": s.name, "slot": s.slot_index, "unit": s.dimension.unit_label}
                for s in obs_symbols
            ],
        },
        "equations": {
            "let_bindings": [lb.model_dump(mode="json") for lb in ir.let_bindings],
            "dynamics_or_residuals": [eq.model_dump(mode="json") for eq in ir.dynamics_or_residuals],
            "observables": [obs.model_dump(mode="json") for obs in ir.observables],
        },
        "obligations": [ob.model_dump(mode="json") for ob in ir.obligations],
        "conservation_pools": [p.model_dump(mode="json") for p in ir.conservation_pools],
    }


def _render_expr_to_python(expr: Any, sym_names: set[str], *, for_ihcg: bool = False) -> str:
    """Deterministically lower a SciIR ExprNode AST to pure scalar Python (or IHCG formula)."""
    if isinstance(expr, str):
        return expr
    if hasattr(expr, "model_dump"):
        expr = expr.model_dump(mode="json")
    if not isinstance(expr, dict):
        return repr(float(expr)) if isinstance(expr, (int, float)) else str(expr)
    kind = expr.get("kind") or expr.get("op")
    if kind in ("const", "literal", "number", "Literal") or ("value" in expr and kind is None):
        return repr(float(expr["value"]))
    if kind in ("sym", "symbol", "var", "Symbol"):
        return str(expr["name"])
    if kind in ("unary", "Unary"):
        op = str(expr.get("op") or expr.get("operator") or "")
        inner = _render_expr_to_python(expr.get("arg") if "arg" in expr else expr.get("operand"), sym_names, for_ihcg=for_ihcg)
        if op in ("neg", "-"):
            return f"(-({inner}))"
        if op == "abs":
            return f"abs({inner})"
        if op == "step":
            return f"(1.0 if ({inner}) >= 0.0 else 0.0)" if not for_ihcg else f"max(0.0, min(1.0, ({inner})))"
        math_unary = {"exp", "log", "sqrt", "sin", "cos", "tanh", "tan", "sinh", "cosh"}
        if op in math_unary:
            return f"{op}({inner})" if for_ihcg else f"math.{op}({inner})"
        return f"({inner})"
    if kind in ("binary", "Binary"):
        raw_op = str(expr.get("op") or expr.get("operator") or "+")
        l = _render_expr_to_python(expr.get("lhs") if "lhs" in expr else expr["left"], sym_names, for_ihcg=for_ihcg)
        r = _render_expr_to_python(expr.get("rhs") if "rhs" in expr else expr["right"], sym_names, for_ihcg=for_ihcg)
        if raw_op in ("min", "max"):
            return f"{raw_op}({l}, {r})"
        op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}
        op = op_map.get(raw_op, raw_op)
        return f"({l} {op} {r})"
    if kind in ("call", "Call"):
        fn = str(expr.get("func") or expr.get("name"))
        args = [_render_expr_to_python(a, sym_names, for_ihcg=for_ihcg) for a in (expr.get("args") or [])]
        math_fns = {"exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "sinh", "cosh", "fabs", "floor", "ceil"}
        if fn in math_fns:
            return f"{fn}({', '.join(args)})" if for_ihcg else f"math.{fn}({', '.join(args)})"
        return f"{fn}({', '.join(args)})"
    if kind in ("if_then_else", "IfThenElse"):
        cond_ops = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "=="}
        if "cond_op" in expr:
            cop = cond_ops.get(str(expr["cond_op"]), "<=")
            lhs = _render_expr_to_python(expr["lhs"], sym_names, for_ihcg=for_ihcg)
            rhs = _render_expr_to_python(expr["rhs"], sym_names, for_ihcg=for_ihcg)
            c = f"({lhs} {cop} {rhs})"
            t = _render_expr_to_python(expr["then_branch"], sym_names, for_ihcg=for_ihcg)
            e = _render_expr_to_python(expr["else_branch"], sym_names, for_ihcg=for_ihcg)
        else:
            c = _render_expr_to_python(expr["cond"], sym_names, for_ihcg=for_ihcg)
            t = _render_expr_to_python(expr["then_expr"], sym_names, for_ihcg=for_ihcg)
            e = _render_expr_to_python(expr["else_expr"], sym_names, for_ihcg=for_ihcg)
        return f"({t} if ({c}) else {e})"
    if kind == "rand":
        return "0.5"
    if "expr_text" in expr:
        return str(expr["expr_text"])
    return "0.0"


def emit_python_kernel_from_sciir(ir: SciIR) -> str:
    """ARCH-004: Deterministically emit a pure v9_transpiler_friendly Python kernel from SciIR.

    Control parameters are strictly read-only (never written back to buf.data),
    making T_CONST_PARAM_WRITE (BUG-027) impossible by construction.
    """
    lines: list[str] = ["import math", "", "def simulation_step(buf, dt):"]
    symbols_sorted = sorted(ir.symbols, key=lambda x: (x.slot_index, x.name))
    slot_idx_map = {s.name: idx for idx, s in enumerate(symbols_sorted)}
    sym_names = set(slot_idx_map.keys())
    # 1. Read all symbols from their unique contiguous canonical slots (CONTROL_PARAM read-only)
    for idx, s in enumerate(symbols_sorted):
        if s.role == SlotRole.CONSTANT:
            lines.append(f"    {s.name} = {float(s.initial_or_default)!r}")
        else:
            lines.append(f"    {s.name} = buf.data[{idx}]")

    # 2. Emit let-bindings in topological order
    for lb in ir.let_bindings:
        rhs = getattr(lb, "expr_text", None) or _render_expr_to_python(lb.expr, sym_names)
        lines.append(f"    {lb.name} = {rhs}")

    # 3. Emit dynamics (ODE) or algebraic updates
    state_syms = [s for s in symbols_sorted if s.role == SlotRole.STATE]
    state_eqs = {eq.target: eq for eq in ir.dynamics_or_residuals}
    if state_syms and ir.archetype.value == "ODE_PDE":
        for s in state_syms:
            eq = state_eqs.get(s.name)
            rhs = _render_expr_to_python(eq.expr, sym_names) if eq else "0.0"
            lines.append(f"    _d_{s.name} = {rhs}")
        for s in state_syms:
            lines.append(f"    {s.name} = {s.name} + dt * _d_{s.name}")
            lines.append(f"    buf.data[{slot_idx_map[s.name]}] = {s.name}")
    else:
        for eq in ir.dynamics_or_residuals:
            rhs = _render_expr_to_python(eq.expr, sym_names)
            tgt = next((s for s in symbols_sorted if s.name == eq.target), None)
            if tgt is not None and tgt.role != SlotRole.CONTROL_PARAM:
                lines.append(f"    {tgt.name} = {rhs}")
                lines.append(f"    buf.data[{slot_idx_map[tgt.name]}] = {tgt.name}")

    # 4. Emit derived observables (and trajectory reductions)
    for obs in ir.observables:
        rhs = _render_expr_to_python(obs.expr, sym_names)
        tgt = next((s for s in symbols_sorted if s.name == obs.target), None)
        if tgt is not None and tgt.role != SlotRole.CONTROL_PARAM:
            lines.append(f"    {tgt.name} = {rhs}")
            lines.append(f"    buf.data[{slot_idx_map[tgt.name]}] = {tgt.name}")

    if len(lines) == 3:
        lines.append("    pass")
    return "\n".join(lines) + "\n"


def sciir_to_research_blueprint(
    sealed: SealedSciIR,
    *,
    user_prompt: str,
    title: str = "",
) -> dict[str, Any]:
    """ARCH-001: Deterministically derive a ResearchBlueprint dict from a SealedSciIR.

    Contracts, slot indices, initial_state, variable_map, ihcg_evidence, and kernel code
    are ALL compiler-derived — the LLM never authors raw `state[i]` expressions.
    """
    import re as _re
    from invar_agent.research.claim_ledger import extract_claim_ledger, reconcile_si_horizon

    ir = sealed.ir
    symbols_sorted = sorted(ir.symbols, key=lambda x: (x.slot_index, x.name))
    sym_names = {s.name for s in symbols_sorted}
    slot_map = {s.name: idx for idx, s in enumerate(symbols_sorted)}

    # Build IHCG-compatible slots, domain, equations, and obligations directly from SciIR
    role_map = {
        SlotRole.STATE: "DYNAMIC_STATE",
        SlotRole.CONTROL_PARAM: "CONTROL_PARAM",
        SlotRole.CONSTANT: "CONTROL_PARAM",
        SlotRole.DERIVED_OBSERVABLE: "DERIVED_OBSERVABLE",
        SlotRole.INDEPENDENT_VAR: "CONTROL_PARAM",
    }
    ihcg_slots: list[dict[str, Any]] = []
    domain: dict[str, list[float]] = {}
    initial_state: list[float] = []
    mechanism_params: list[str] = []

    ledger = extract_claim_ledger(user_prompt)
    pinned_syms = {lit.symbol for lit in ledger.pinned_literals} | {lit.symbol.lower() for lit in ledger.pinned_literals}

    for idx, s in enumerate(symbols_sorted):
        si_dim = [
            (int(n // d) if (d != 0 and n % d == 0) else (n / d if d != 0 else 0))
            for n, d in s.dimension.exponents
        ]
        ihcg_role = role_map.get(s.role, "CONTROL_PARAM")
        if s.domain is not None:
            lo, hi = float(s.domain.lo), float(s.domain.hi)
            if lo > hi:
                lo, hi = hi, lo
            domain[s.name] = [lo, hi]
            init_v = float(s.initial_or_default if s.initial_or_default is not None else 0.5 * (lo + hi))
            if init_v < lo or init_v > hi:
                init_v = 0.5 * (lo + hi)
        elif s.role == SlotRole.DERIVED_OBSERVABLE:
            # Leave DERIVED_OBSERVABLE domain omitted so blueprint_compiler derives the exact
            # interval-arithmetic bounds from its governing equation (never a tiny [0, 1] box!).
            init_v = float(s.initial_or_default or 0.0)
        else:
            v0 = float(s.initial_or_default or 0.0)
            if s.role == SlotRole.CONSTANT or s.name in pinned_syms or s.name.lower() in pinned_syms:
                lo, hi = v0, v0
            else:
                span = max(1.0, abs(v0) * 2.0)
                lo = 0.0 if s.strictly_positive else (v0 - span)
                hi = v0 + span
            if lo > hi:
                lo, hi = hi, lo
            domain[s.name] = [lo, hi]
            init_v = v0 if lo <= v0 <= hi else 0.5 * (lo + hi)
        initial_state.append(init_v)
        if (
            s.role == SlotRole.CONTROL_PARAM
            and s.name in domain
            and domain[s.name][1] > domain[s.name][0]
            and s.name not in pinned_syms
            and s.name.lower() not in pinned_syms
        ):
            mechanism_params.append(s.name)
        ihcg_slots.append({
            "slot_idx": idx,
            "symbol": s.name,
            "si_dim": si_dim,
            "role": ihcg_role,
            "semantic_tag": 0,
        })

    # Inline let-bindings into IHCG formula expressions so IHCG sees only declared slot symbols
    let_map: dict[str, str] = {}
    for lb in ir.let_bindings:
        rhs_txt = _render_expr_to_python(lb.expr, sym_names, for_ihcg=True)
        for prev_k, prev_v in let_map.items():
            rhs_txt = _re.sub(rf"\b{_re.escape(prev_k)}\b", f"({prev_v})", rhs_txt)
        let_map[lb.name] = rhs_txt

    def _inline_lets(expr_str: str) -> str:
        out = expr_str
        for k, v in reversed(list(let_map.items())):
            out = _re.sub(rf"\b{_re.escape(k)}\b", f"({v})", out)
        return out

    sym_by_name = {s.name: s for s in symbols_sorted}
    ihcg_equations: dict[str, dict[str, Any]] = {}
    for eq in ir.dynamics_or_residuals:
        s_obj = sym_by_name.get(eq.target)
        s_dim = [
            (int(n // d) if (d != 0 and n % d == 0) else (n / d if d != 0 else 0))
            for n, d in s_obj.dimension.exponents
        ] if s_obj else [0] * 7
        rhs_ihcg = _inline_lets(_render_expr_to_python(eq.expr, sym_names, for_ihcg=True))
        if ir.archetype.value == "ODE_PDE" and s_obj and s_obj.role == SlotRole.STATE:
            eq_dim = list(s_dim)
            eq_dim[2] = eq_dim[2] - 1  # d/dt has T^-1
            ihcg_equations[f"d_{eq.target}_dt"] = {
                "expression": rhs_ihcg,
                "si_dim": eq_dim,
                "kind": "ODE_RHS",
                "state_symbol": eq.target,
            }
        elif ir.archetype.value == "FIXED_POINT" and s_obj and s_obj.role == SlotRole.STATE:
            ihcg_equations[f"res_{eq.target}"] = {
                "expression": rhs_ihcg,
                "si_dim": s_dim,
                "kind": "FIXED_POINT_RESIDUAL",
                "state_symbol": eq.target,
            }
        else:
            ihcg_equations[eq.target] = {
                "expression": rhs_ihcg,
                "si_dim": s_dim,
                "kind": "ALGEBRAIC",
            }

    # Detect boolean-indicator observables (`1.0 if (lhs op rhs) else 0.0`) so we can
    # unwrap obligations directly onto `lhs op rhs` instead of emitting Python ternaries into IHCG.
    cond_ops_map = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "=="}
    indicator_predicates: dict[str, str] = {}
    for obs in ir.observables:
        if getattr(obs.expr, "kind", None) == "if_then_else":
            e_dict = obs.expr.model_dump(mode="json") if hasattr(obs.expr, "model_dump") else obs.expr
            if isinstance(e_dict, dict) and "cond_op" in e_dict:
                cop = cond_ops_map.get(str(e_dict["cond_op"]), "<=")
                l_s = _inline_lets(_render_expr_to_python(e_dict["lhs"], sym_names, for_ihcg=True))
                r_s = _inline_lets(_render_expr_to_python(e_dict["rhs"], sym_names, for_ihcg=True))
                indicator_predicates[obs.target] = f"{l_s} {cop} {r_s}"
            continue
        s_obj = sym_by_name.get(obs.target)
        s_dim = [
            (int(n // d) if (d != 0 and n % d == 0) else (n / d if d != 0 else 0))
            for n, d in s_obj.dimension.exponents
        ] if s_obj else [0] * 7
        rhs_ihcg = _inline_lets(_render_expr_to_python(obs.expr, sym_names, for_ihcg=True))
        ihcg_equations[obs.target] = {
            "expression": rhs_ihcg,
            "si_dim": s_dim,
            "kind": "ALGEBRAIC",
        }

    def _to_state_expr(expr_text: str) -> str:
        out = expr_text
        for name in sorted(slot_map, key=len, reverse=True):
            out = _re.sub(rf"\b{_re.escape(name)}\b", f"state[{slot_map[name]}]", out)
        return out

    ihcg_obligations: list[dict[str, Any]] = []
    invariants: list[dict[str, Any]] = []
    q_mode = "FORALL"
    for ob in ir.obligations:
        lhs_txt = _inline_lets(_render_expr_to_python(ob.lhs_expr, sym_names, for_ihcg=True))
        rhs_txt = _inline_lets(_render_expr_to_python(ob.rhs_expr, sym_names, for_ihcg=True))
        if lhs_txt.strip() in indicator_predicates:
            expr_t = indicator_predicates[lhs_txt.strip()]
        else:
            expr_t = f"{lhs_txt} {ob.comparator} {rhs_txt}"
        if ob.quantifier.upper() in {"EXISTS", "BOUNDARY", "FORALL"}:
            q_mode = ob.quantifier.upper()
        ihcg_obligations.append({
            "id": ob.id,
            "expression": expr_t,
            "obligation_class": "HYPOTHESIS_TARGET",
            "prompt_quote": expr_t,
        })
        invariants.append({
            "variable_name": ob.id,
            "assertion": _to_state_expr(expr_t),
            "tolerance": float(getattr(ob, "tolerance", 1e-6) or 1e-6),
        })
    if not invariants:
        invariants.append({"variable_name": "finite", "assertion": "telemetry['finite'] == 1.0", "tolerance": 0.0})

    dt = 0.05
    steps = 200
    if ir.archetype.value == "ODE_PDE" and hasattr(ir.config, "t_span"):
        t0, t1 = ir.config.t_span
        span = max(1e-6, float(t1) - float(t0))
        steps = max(20, min(4000, int(getattr(ir.config, "sample_points", 200) or 200)))
        dt = span / steps

    # Reconcile SI horizon from user prompt ledger (BUG-006 / ARCH-002)
    ledger = extract_claim_ledger(user_prompt)
    if ledger.duration_si_s is not None and ir.archetype.value == "ODE_PDE":
        dt, steps = reconcile_si_horizon(ledger, dt, steps)

    arch_map = {
        "ODE_PDE": "ODE_PDE",
        "MONTE_CARLO": "MONTE_CARLO",
        "FIXED_POINT": "FIXED_POINT",
        "PARAM_SWEEP": "PARAM_SWEEP",
        "ALGEBRAIC_CLOSURE": "PARAM_SWEEP",
    }
    ihcg_archetype = arch_map.get(ir.archetype.value, "ODE_PDE")
    ihcg_evidence = {
        "archetype": ihcg_archetype,
        "domain": domain,
        "evidence": {
            "slots": ihcg_slots,
            "equations": ihcg_equations,
            "environmental_assumptions": [],
            "quantifier_mode": q_mode,
            "mechanism_parameters": mechanism_params[:2],
            "obligations": ihcg_obligations,
        },
    }

    kernel_code = emit_python_kernel_from_sciir(ir)
    return {
        "blueprint_id": ir.ir_id[:32],
        "title": title or getattr(ir, "title", None) or ir.ir_id,
        "research_prompt": user_prompt,
        "falsifiable_hypothesis": " and ".join(ob["expression"] for ob in ihcg_obligations) or user_prompt[:200],
        "search_queries": [getattr(ir, "title", None) or "scientific model"],
        "variable_map": slot_map,
        "code_template": kernel_code,
        "contracts": [
            {
                "contract_id": "c1_sciir_derived",
                "target_backend": "cpp",
                "initial_state": initial_state,
                "dt": dt,
                "steps": steps,
                "invariants": invariants,
            }
        ],
        "ihcg_evidence": ihcg_evidence,
        "sciir_commitment_blake3": sealed.sciir_commitment_blake3,
    }

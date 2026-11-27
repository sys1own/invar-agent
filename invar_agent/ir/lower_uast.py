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


def _render_expr_to_python(expr: Any, sym_names: set[str]) -> str:
    """Deterministically lower a SciIR Expr AST (or canonical expression string) to pure scalar Python."""
    if isinstance(expr, str):
        return expr
    if hasattr(expr, "model_dump"):
        expr = expr.model_dump(mode="json")
    if not isinstance(expr, dict):
        return repr(float(expr)) if isinstance(expr, (int, float)) else str(expr)
    kind = expr.get("kind") or expr.get("op")
    if kind == "literal" or "value" in expr and kind in (None, "const", "number", "Literal"):
        return repr(float(expr["value"]))
    if kind in ("symbol", "var", "sym", "Symbol"):
        return str(expr["name"])
    if kind in ("unary", "Unary"):
        op = expr.get("op") or expr.get("operator")
        inner = _render_expr_to_python(expr.get("operand") or expr.get("arg"), sym_names)
        return f"(-({inner}))" if op in ("neg", "-") else f"({inner})"
    if kind in ("binary", "Binary"):
        op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}
        op = op_map.get(str(expr.get("op") or expr.get("operator")), str(expr.get("op") or "+"))
        l = _render_expr_to_python(expr.get("lhs") if "lhs" in expr else expr["left"], sym_names)
        r = _render_expr_to_python(expr.get("rhs") if "rhs" in expr else expr["right"], sym_names)
        return f"({l} {op} {r})"
    if kind in ("call", "Call"):
        fn = str(expr.get("func") or expr.get("name"))
        args = [_render_expr_to_python(a, sym_names) for a in (expr.get("args") or [])]
        math_fns = {"exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "sinh", "cosh", "fabs", "floor", "ceil"}
        if fn in math_fns:
            return f"math.{fn}({', '.join(args)})"
        if fn in ("abs", "min", "max"):
            return f"{fn}({', '.join(args)})"
        return f"{fn}({', '.join(args)})"
    if kind in ("if_then_else", "IfThenElse"):
        c = _render_expr_to_python(expr["cond"], sym_names)
        t = _render_expr_to_python(expr["then_expr"], sym_names)
        e = _render_expr_to_python(expr["else_expr"], sym_names)
        return f"({t} if ({c}) else {e})"
    if "expr_text" in expr:
        return str(expr["expr_text"])
    return "0.0"


def emit_python_kernel_from_sciir(ir: SciIR) -> str:
    """ARCH-004: Deterministically emit a pure v9_transpiler_friendly Python kernel from SciIR.

    Control parameters are strictly read-only (never written back to buf.data),
    making T_CONST_PARAM_WRITE (BUG-027) impossible by construction.
    """
    lines: list[str] = ["import math", "", "def simulation_step(buf, dt):"]
    sym_names = {s.name for s in ir.symbols}
    # 1. Read all symbols from their canonical slots (CONTROL_PARAM read-only)
    for s in sorted(ir.symbols, key=lambda x: x.slot_index):
        if s.role == SlotRole.CONSTANT:
            lines.append(f"    {s.name} = {float(s.initial_or_default)!r}")
        else:
            lines.append(f"    {s.name} = buf.data[{s.slot_index}]")

    # 2. Emit let-bindings in topological order
    for lb in ir.let_bindings:
        rhs = getattr(lb, "expr_text", None) or _render_expr_to_python(lb.expr, sym_names)
        lines.append(f"    {lb.name} = {rhs}")

    # 3. Emit dynamics (ODE) or algebraic updates
    state_syms = [s for s in ir.symbols if s.role == SlotRole.STATE]
    state_eqs = {eq.target: eq for eq in ir.dynamics_or_residuals}
    if state_syms and ir.archetype.value == "ODE_PDE":
        for s in state_syms:
            eq = state_eqs.get(s.name)
            rhs = _render_expr_to_python(eq.expr, sym_names) if eq else "0.0"
            lines.append(f"    _d_{s.name} = {rhs}")
        for s in state_syms:
            lines.append(f"    {s.name} = {s.name} + dt * _d_{s.name}")
            lines.append(f"    buf.data[{s.slot_index}] = {s.name}")
    else:
        for eq in ir.dynamics_or_residuals:
            rhs = _render_expr_to_python(eq.expr, sym_names)
            tgt = next((s for s in ir.symbols if s.name == eq.target), None)
            if tgt is not None and tgt.role != SlotRole.CONTROL_PARAM:
                lines.append(f"    {tgt.name} = {rhs}")
                lines.append(f"    buf.data[{tgt.slot_index}] = {tgt.name}")

    # 4. Emit derived observables (and trajectory reductions)
    for obs in ir.observables:
        rhs = _render_expr_to_python(obs.expr, sym_names)
        tgt = next((s for s in ir.symbols if s.name == obs.target), None)
        if tgt is not None and tgt.role != SlotRole.CONTROL_PARAM:
            lines.append(f"    {tgt.name} = {rhs}")
            lines.append(f"    buf.data[{tgt.slot_index}] = {tgt.name}")

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

    Contracts, slot indices, initial_state, variable_map, and kernel code are ALL
    compiler-derived — the LLM never authors raw `state[i]` expressions.
    """
    ir = sealed.ir
    symbols_sorted = sorted(ir.symbols, key=lambda x: x.slot_index)
    slot_map = {s.name: s.slot_index for s in symbols_sorted}
    initial_state = [float(s.initial_or_default or 0.0) for s in symbols_sorted]

    # Compiler-derived contracts: translate symbol names in obligations to state[idx]
    import re as _re

    def _to_state_expr(expr_text: str) -> str:
        out = expr_text
        for name in sorted(slot_map, key=len, reverse=True):
            out = _re.sub(rf"\b{_re.escape(name)}\b", f"state[{slot_map[name]}]", out)
        return out

    invariants = []
    for ob in ir.obligations:
        expr_t = getattr(ob, "expr_text", None) or str(ob.lhs_expr)
        invariants.append({
            "variable_name": ob.obligation_id,
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

    kernel_code = emit_python_kernel_from_sciir(ir)
    return {
        "blueprint_id": ir.ir_id[:32],
        "title": title or ir.title or ir.ir_id,
        "research_prompt": user_prompt,
        "falsifiable_hypothesis": " and ".join(
            getattr(ob, "expr_text", None) or ob.obligation_id for ob in ir.obligations
        ) or user_prompt[:200],
        "search_queries": [ir.title or "scientific model"],
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
        "ihcg_evidence": sealed.ihcg_v1_payload,
        "sciir_commitment_blake3": sealed.sciir_commitment_blake3,
    }

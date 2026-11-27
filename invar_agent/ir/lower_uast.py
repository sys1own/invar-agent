"""Deterministic lowering from sealed `SciIR` (`CanonicalResearchIR`) to Builder Studio UAST."""
from __future__ import annotations

from typing import Any

from invar_agent.ir.function_registry import FUNCTION_REGISTRY, lower_nary_binary_tree
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


_SPLITMIX64_HELPERS = """_SPLITMIX_STATE = {}

def _splitmix64_next(stream_id):
    z = (_SPLITMIX_STATE.get(stream_id, (int(stream_id) + 1) * 0x9E3779B97F4A7C15) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    _SPLITMIX_STATE[stream_id] = z
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

def _splitmix64_uniform01(stream_id=0):
    return ((_splitmix64_next(stream_id) >> 11) + 0.5) * (1.0 / (1 << 53))

def _splitmix64_normal01(stream_id=0):
    u1 = max(1e-15, _splitmix64_uniform01(stream_id))
    u2 = _splitmix64_uniform01(stream_id + 104729)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

def _splitmix64_exponential1(stream_id=0):
    u = max(1e-15, 1.0 - _splitmix64_uniform01(stream_id))
    return -math.log(u)
"""


def _render_expr_to_python(expr: Any, sym_names: set[str], *, for_ihcg: bool = False) -> str:
    """ARCH-006: Single lowering authority for SciIR ExprNode -> executable & formal Python.

    - Exhaustive node matching: unknown node kinds raise ValueError (never 0.0).
    - Identical piecewise semantics for ``step(x)`` and ``if_then_else`` across
      executable and formal paths.
    - Seeded SplitMix64 streams for ``rand`` draws (never constant 0.5).
    - n-ary ``min``/``max`` lowered via balanced binary trees; ``ln``/``clamp``/``tanh``
      resolved through the single canonical function registry.
    """
    if isinstance(expr, str):
        return expr
    if hasattr(expr, "model_dump"):
        expr = expr.model_dump(mode="json")
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return repr(float(expr))
    if not isinstance(expr, dict):
        raise ValueError(f"Unsupported SciIR expression node: {expr!r}")
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
            # Identical Heaviside semantics in both executable and formal paths.
            return f"(1.0 if ({inner}) >= 0.0 else 0.0)"
        if op == "ln":
            return f"log({inner})" if for_ihcg else f"math.log({inner})"
        math_unary = {"exp", "log", "sqrt", "sin", "cos", "tanh", "tan", "sinh", "cosh"}
        if op in math_unary:
            return f"{op}({inner})" if for_ihcg else f"math.{op}({inner})"
        raise ValueError(f"Unsupported SciIR unary operator: {op!r}")
    if kind in ("binary", "Binary"):
        raw_op = str(expr.get("op") or expr.get("operator") or "")
        if not raw_op:
            raise ValueError("Missing binary operator in SciIR expression")
        l = _render_expr_to_python(expr.get("lhs") if "lhs" in expr else expr["left"], sym_names, for_ihcg=for_ihcg)
        raw_rhs = expr.get("rhs") if "rhs" in expr else expr["right"]
        if raw_op in ("pow", "**"):
            rhs_dict = raw_rhs.model_dump(mode="json") if hasattr(raw_rhs, "model_dump") else raw_rhs
            if isinstance(rhs_dict, dict) and (rhs_dict.get("kind") in ("const", "literal", "number", "Literal") or ("value" in rhs_dict and rhs_dict.get("kind") is None)):
                val = float(rhs_dict["value"])
                if val.is_integer() and abs(val) < (1 << 53):
                    r = str(int(val))
                else:
                    r = _render_expr_to_python(raw_rhs, sym_names, for_ihcg=for_ihcg)
            elif isinstance(rhs_dict, (int, float)) and not isinstance(rhs_dict, bool) and float(rhs_dict).is_integer() and abs(float(rhs_dict)) < (1 << 53):
                r = str(int(float(rhs_dict)))
            else:
                r = _render_expr_to_python(raw_rhs, sym_names, for_ihcg=for_ihcg)
        else:
            r = _render_expr_to_python(raw_rhs, sym_names, for_ihcg=for_ihcg)
        if raw_op in ("min", "max"):
            return f"{raw_op}({l}, {r})"
        op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}
        op = op_map.get(raw_op, raw_op)
        if op not in {"+", "-", "*", "/", "**"}:
            raise ValueError(f"Unsupported SciIR binary operator: {raw_op!r}")
        return f"({l} {op} {r})"
    if kind in ("call", "Call"):
        fn = str(expr.get("func") or expr.get("name") or "")
        args = [_render_expr_to_python(a, sym_names, for_ihcg=for_ihcg) for a in (expr.get("args") or [])]
        if fn == "ln" and len(args) == 1:
            return f"log({args[0]})" if for_ihcg else f"math.log({args[0]})"
        if fn == "clamp" and len(args) == 3:
            return f"min({args[2]}, max({args[1]}, {args[0]}))"
        if fn == "step" and len(args) == 1:
            return f"(1.0 if ({args[0]}) >= 0.0 else 0.0)"
        if fn in ("min", "max") and len(args) >= 2:
            return lower_nary_binary_tree(fn, args)
        math_fns = {"exp", "log", "log10", "sqrt", "sin", "cos", "tan", "tanh", "sinh", "cosh", "fabs", "floor", "ceil"}
        if fn in math_fns:
            return f"{fn}({', '.join(args)})" if for_ihcg else f"math.{fn}({', '.join(args)})"
        if fn in FUNCTION_REGISTRY:
            return f"{fn}({', '.join(args)})"
        raise ValueError(f"Unsupported SciIR function call: {fn!r}")
    if kind in ("if_then_else", "IfThenElse"):
        cond_ops = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}
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
        dist = str(expr.get("dist") or "uniform01").lower()
        stream_id = int(expr.get("draw_stream_id", 0) or 0)
        if dist in ("uniform01", "uniform"):
            return f"rand_uniform01({stream_id})" if for_ihcg else f"_splitmix64_uniform01({stream_id})"
        if dist in ("normal01", "normal", "gaussian"):
            return f"rand_normal01({stream_id})" if for_ihcg else f"_splitmix64_normal01({stream_id})"
        if dist in ("exponential1", "exponential", "exp"):
            return f"rand_exponential1({stream_id})" if for_ihcg else f"_splitmix64_exponential1({stream_id})"
        raise ValueError(f"Unsupported SciIR random distribution: {dist!r}")
    if "expr_text" in expr and kind is None:
        return str(expr["expr_text"])
    raise ValueError(f"Unsupported SciIR expression node kind: {kind!r}")


def _substitute_state_vars(expr_str: str, replacements: dict[str, str]) -> str:
    """UPG-006: Capture-free AST substitution replacing regex word substitution."""
    import ast as _ast

    if not replacements:
        return expr_str
    try:
        tree = _ast.parse(expr_str, mode="eval")
        parsed_repls = {k: _ast.parse(v, mode="eval").body for k, v in replacements.items()}
    except SyntaxError:
        import re as _re
        pattern = _re.compile(
            r"\b(" + "|".join(_re.escape(k) for k in sorted(replacements, key=len, reverse=True)) + r")\b"
        )
        return pattern.sub(lambda m: f"({replacements[m.group(1)]})", expr_str)

    import copy as _copy

    class _Sub(_ast.NodeTransformer):
        def visit_Name(self, node: _ast.Name) -> _ast.AST:
            if node.id in parsed_repls:
                return _copy.deepcopy(parsed_repls[node.id])
            return node

    out = _Sub().visit(tree)
    _ast.fix_missing_locations(out)
    return _ast.unparse(out.body)


def compile_time_forcing_schedule(
    expr_str: str,
    t_symbol: str,
    t_start: float,
    t_end: float,
    *,
    state_symbols: Optional[set[str]] = None,
    param_prefix: str = "u__",
) -> tuple[str, Optional["ForcingSchedule"], dict[str, tuple[float, float]]]:
    """UPG-006 / NF-04: Lower time-only `Cond`s in an ODE expression into a smooth
    `ForcingSchedule` over compiler-owned `CONTROL_PARAM` slots `u__0`, `u__1`, ...

    If any `Cond` predicate references a state symbol, raises
    `ValueError("PIECEWISE_STATE_DEPENDENT_UNSUPPORTED")`.
    """
    import ast as _ast
    from fractions import Fraction
    from invar_agent.research.meaningfulness import ForcingSchedule, ForcingSegment

    tree = _ast.parse(expr_str, mode="eval")
    conds: list[_ast.IfExp] = [n for n in _ast.walk(tree) if isinstance(n, _ast.IfExp)]
    if not conds:
        return expr_str, None, {}

    state_set = set(state_symbols or ())

    def _free_names(node: _ast.AST) -> set[str]:
        return {n.id for n in _ast.walk(node) if isinstance(n, _ast.Name)}

    def _const_frac(node: _ast.AST) -> Optional[Fraction]:
        if isinstance(node, _ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub) and isinstance(node.operand, _ast.Constant) and type(node.operand.value) in (int, float):
            return -Fraction(str(node.operand.value))
        return None

    def _extract_t_breakpoints(test: _ast.AST, bps: set[Fraction]) -> None:
        names = _free_names(test)
        if names & state_set:
            raise ValueError(
                f"PIECEWISE_STATE_DEPENDENT_UNSUPPORTED: conditional predicate references state symbol(s) {sorted(names & state_set)}"
            )
        if not names <= {t_symbol}:
            raise ValueError(
                f"PIECEWISE_STATE_DEPENDENT_UNSUPPORTED: conditional predicate references non-time symbol(s) {sorted(names - {t_symbol})}"
            )
        if isinstance(test, _ast.BoolOp):
            for v in test.values:
                _extract_t_breakpoints(v, bps)
            return
        if isinstance(test, _ast.UnaryOp) and isinstance(test.op, _ast.Not):
            _extract_t_breakpoints(test.operand, bps)
            return
        if isinstance(test, _ast.Compare):
            operands = [test.left, *test.comparators]
            for a, b in zip(operands, operands[1:]):
                if isinstance(a, _ast.Name) and a.id == t_symbol and _const_frac(b) is not None:
                    bps.add(_const_frac(b))  # type: ignore[arg-type]
                elif isinstance(b, _ast.Name) and b.id == t_symbol and _const_frac(a) is not None:
                    bps.add(_const_frac(a))  # type: ignore[arg-type]
                else:
                    raise ValueError(f"PIECEWISE_STATE_DEPENDENT_UNSUPPORTED: non-linear time predicate {_ast.unparse(test)}")
            return
        raise ValueError(f"PIECEWISE_STATE_DEPENDENT_UNSUPPORTED: unsupported predicate {_ast.unparse(test)}")

    def _eval_pred_at(test: _ast.AST, t_val: Fraction) -> bool:
        if isinstance(test, _ast.BoolOp):
            vals = [_eval_pred_at(v, t_val) for v in test.values]
            return all(vals) if isinstance(test.op, _ast.And) else any(vals)
        if isinstance(test, _ast.UnaryOp) and isinstance(test.op, _ast.Not):
            return not _eval_pred_at(test.operand, t_val)
        if isinstance(test, _ast.Compare):
            operands = [test.left, *test.comparators]
            for a, op, b in zip(operands, test.ops, operands[1:]):
                av = t_val if (isinstance(a, _ast.Name) and a.id == t_symbol) else _const_frac(a)
                bv = t_val if (isinstance(b, _ast.Name) and b.id == t_symbol) else _const_frac(b)
                if av is None or bv is None:
                    raise ValueError("PIECEWISE_STATE_DEPENDENT_UNSUPPORTED")
                ok = (
                    (av < bv) if isinstance(op, _ast.Lt)
                    else (av <= bv) if isinstance(op, _ast.LtE)
                    else (av > bv) if isinstance(op, _ast.Gt)
                    else (av >= bv) if isinstance(op, _ast.GtE)
                    else (av == bv) if isinstance(op, _ast.Eq)
                    else (av != bv)
                )
                if not ok:
                    return False
            return True
        raise ValueError("PIECEWISE_STATE_DEPENDENT_UNSUPPORTED")

    t0_f = Fraction(str(t_start))
    t1_f = Fraction(str(t_end))
    breakpoints: set[Fraction] = {t0_f, t1_f}
    slot_for_cond: dict[str, tuple[str, _ast.IfExp]] = {}
    counter = 0

    class _Replacer(_ast.NodeTransformer):
        def visit_IfExp(self, node: _ast.IfExp) -> _ast.AST:
            nonlocal counter
            _extract_t_breakpoints(node.test, breakpoints)
            b_val = _const_frac(node.body)
            e_val = _const_frac(node.orelse)
            if b_val is None or e_val is None:
                raise ValueError(
                    f"PIECEWISE_STATE_DEPENDENT_UNSUPPORTED: time forcing branches must be constants in {_ast.unparse(node)}"
                )
            key = _ast.unparse(node)
            if key not in slot_for_cond:
                sname = f"{param_prefix}{counter}"
                counter += 1
                slot_for_cond[key] = (sname, node)
            sname = slot_for_cond[key][0]
            return _ast.copy_location(_ast.Name(id=sname, ctx=_ast.Load()), node)

    new_tree = _Replacer().visit(tree)
    _ast.fix_missing_locations(new_tree)
    sorted_bps = sorted(bp for bp in breakpoints if t0_f <= bp <= t1_f)
    if sorted_bps[0] != t0_f:
        sorted_bps.insert(0, t0_f)
    if sorted_bps[-1] != t1_f:
        sorted_bps.append(t1_f)

    symbols_ordered = [slot_for_cond[k][0] for k in slot_for_cond]
    segments: list[ForcingSegment] = []
    domains: dict[str, tuple[float, float]] = {}
    for a, b in zip(sorted_bps, sorted_bps[1:]):
        if a >= b:
            continue
        mid = (a + b) / 2
        vals: dict[str, str] = {}
        for _key, (sname, cnode) in slot_for_cond.items():
            chosen = _const_frac(cnode.body) if _eval_pred_at(cnode.test, mid) else _const_frac(cnode.orelse)
            assert chosen is not None
            vals[sname] = f"{chosen.numerator}/{chosen.denominator}" if chosen.denominator != 1 else str(chosen.numerator)
            fv = float(chosen)
            if sname not in domains:
                domains[sname] = (fv, fv)
            else:
                domains[sname] = (min(domains[sname][0], fv), max(domains[sname][1], fv))
        segments.append(
            ForcingSegment(
                t_start=f"{a.numerator}/{a.denominator}" if a.denominator != 1 else str(a.numerator),
                t_end=f"{b.numerator}/{b.denominator}" if b.denominator != 1 else str(b.numerator),
                values=vals,
            )
        )
    sched = ForcingSchedule(symbols=tuple(symbols_ordered), segments=tuple(segments))
    return _ast.unparse(new_tree.body), sched, domains


def emit_python_kernel_from_sciir(ir: SciIR) -> str:
    """ARCH-006: Deterministically emit a pure v9-friendly Python kernel from SciIR.

    - ODE_PDE emits genuine 4-stage classical RK4 matching ``_prove_rk4_assignments``.
    - Rejects archetype/equation mismatches (derivative equations on STATE symbols
      require ``ODE_PDE``).
    - Supports first-class ``TRAJECTORY_MAX`` / ``TRAJECTORY_MIN`` / ``TRAJECTORY_FINAL``
      observable reductions.
    - Control parameters are strictly read-only.
    """
    symbols_sorted = sorted(ir.symbols, key=lambda x: (x.slot_index, x.name))
    slot_idx_map = {s.name: idx for idx, s in enumerate(symbols_sorted)}
    sym_names = set(slot_idx_map.keys())
    state_syms = [s for s in symbols_sorted if s.role == SlotRole.STATE]
    state_names = [s.name for s in state_syms]
    state_eqs = {eq.target: eq for eq in ir.dynamics_or_residuals}

    # Reject archetype/equation mismatches: PARAM_SWEEP / ALGEBRAIC_CLOSURE cannot
    # assign ODE derivatives directly onto evolving STATE variables.
    if state_syms and ir.archetype.value in ("PARAM_SWEEP", "ALGEBRAIC_CLOSURE"):
        has_state_dynamics = any(eq.target in set(state_names) for eq in ir.dynamics_or_residuals)
        if has_state_dynamics:
            raise ValueError(
                f"ARCHETYPE_EQUATION_MISMATCH: archetype={ir.archetype.value} cannot "
                f"evolve STATE variables {state_names}; derivative equations require ODE_PDE"
            )

    # UPG-006 / NF-03: ODE autonomization (`d t/dt = 1`) for non-autonomous ODE_PDE models.
    clock_syms: list[Any] = []
    if ir.archetype.value == "ODE_PDE":
        indep_syms = [s for s in symbols_sorted if s.role == SlotRole.INDEPENDENT_VAR]
        indep_names = {s.name for s in indep_syms}
        if indep_names:
            used_indep: set[str] = set()

            def _scan_expr_for_indep(node: Any) -> None:
                if node is None:
                    return
                if isinstance(node, dict):
                    if node.get("kind") == "sym" and node.get("name") in indep_names:
                        used_indep.add(str(node["name"]))
                    for v in node.values():
                        if isinstance(v, (dict, list)):
                            _scan_expr_for_indep(v)
                    return
                if isinstance(node, list):
                    for item in node:
                        _scan_expr_for_indep(item)
                    return
                kind = getattr(node, "kind", None)
                if kind == "sym" and getattr(node, "name", None) in indep_names:
                    used_indep.add(str(node.name))
                for attr in ("arg", "lhs", "rhs", "then_branch", "else_branch", "args"):
                    sub = getattr(node, attr, None)
                    if sub is not None:
                        _scan_expr_for_indep(sub)

            for lb in ir.let_bindings:
                _scan_expr_for_indep(getattr(lb, "expr", None))
            for eq in ir.dynamics_or_residuals:
                _scan_expr_for_indep(getattr(eq, "expr", None))
            for obs in ir.observables:
                _scan_expr_for_indep(getattr(obs, "expr", None))
            clock_syms = [s for s in indep_syms if s.name in used_indep]

    lines: list[str] = ["import math", "", "def simulation_step(buf, dt):"]
    # 1. Read all symbols from their unique contiguous canonical slots (CONTROL_PARAM / CONSTANT read-only)
    for idx, s in enumerate(symbols_sorted):
        lines.append(f"    {s.name} = buf.data[{idx}]")

    # Build inlined let-binding map so RK4 stages re-evaluate let-bindings at intermediate states
    let_exprs: list[tuple[str, str]] = []
    let_map: dict[str, str] = {}
    for lb in ir.let_bindings:
        rhs = _render_expr_to_python(lb.expr, sym_names) if getattr(lb, "expr", None) is not None else str(getattr(lb, "expr_text", "0.0"))
        inlined = _substitute_state_vars(rhs, let_map)
        let_map[lb.name] = inlined
        let_exprs.append((lb.name, rhs))

    # 2. Emit dynamics (genuine RK4 for ODE_PDE, including autonomized clock states d t/dt = 1) or algebraic updates
    rk4_syms = list(state_syms) + list(clock_syms)
    clock_name_set = {s.name for s in clock_syms}
    if rk4_syms and ir.archetype.value == "ODE_PDE":
        raw_rhs: dict[str, str] = {}
        for s in rk4_syms:
            if s.name in clock_name_set:
                raw_rhs[s.name] = "1.0"
            else:
                eq = state_eqs.get(s.name)
                base_rhs = _render_expr_to_python(eq.expr, sym_names) if eq else "0.0"
                raw_rhs[s.name] = _substitute_state_vars(base_rhs, let_map)

        # Stage 1: k1 = f(y)
        for s in rk4_syms:
            lines.append(f"    _k1_{s.name} = {raw_rhs[s.name]}")
        # Stage 2: k2 = f(y + (dt / 2) * k1)
        sub2 = {s.name: f"{s.name} + (dt / 2) * _k1_{s.name}" for s in rk4_syms}
        for s in rk4_syms:
            lines.append(f"    _k2_{s.name} = {_substitute_state_vars(raw_rhs[s.name], sub2)}")
        # Stage 3: k3 = f(y + (dt / 2) * k2)
        sub3 = {s.name: f"{s.name} + (dt / 2) * _k2_{s.name}" for s in rk4_syms}
        for s in rk4_syms:
            lines.append(f"    _k3_{s.name} = {_substitute_state_vars(raw_rhs[s.name], sub3)}")
        # Stage 4: k4 = f(y + dt * k3)
        sub4 = {s.name: f"{s.name} + dt * _k3_{s.name}" for s in rk4_syms}
        for s in rk4_syms:
            lines.append(f"    _k4_{s.name} = {_substitute_state_vars(raw_rhs[s.name], sub4)}")
        # Combine RK4 update
        for s in rk4_syms:
            lines.append(
                f"    {s.name} = {s.name} + dt * (_k1_{s.name} + 2 * _k2_{s.name} + 2 * _k3_{s.name} + _k4_{s.name}) / 6"
            )
            lines.append(f"    buf.data[{slot_idx_map[s.name]}] = {s.name}")
        # Re-emit let-bindings at the updated state for downstream observables
        for lname, lrhs in let_exprs:
            lines.append(f"    {lname} = {lrhs}")
    else:
        for lname, lrhs in let_exprs:
            lines.append(f"    {lname} = {lrhs}")
        for eq in ir.dynamics_or_residuals:
            rhs = _render_expr_to_python(eq.expr, sym_names)
            tgt = next((s for s in symbols_sorted if s.name == eq.target), None)
            if tgt is not None and tgt.role != SlotRole.CONTROL_PARAM:
                lines.append(f"    {tgt.name} = {rhs}")
                lines.append(f"    buf.data[{slot_idx_map[tgt.name]}] = {tgt.name}")

    # 3. Emit derived observables (with first-class TRAJECTORY_MAX / TRAJECTORY_MIN / TRAJECTORY_FINAL)
    for obs in ir.observables:
        rhs = _render_expr_to_python(obs.expr, sym_names)
        tgt = next((s for s in symbols_sorted if s.name == obs.target), None)
        if tgt is not None and tgt.role != SlotRole.CONTROL_PARAM:
            slot_i = slot_idx_map[tgt.name]
            reduction = str(getattr(obs, "reduction", "") or "").upper()
            if reduction == "TRAJECTORY_MAX":
                lines.append(f"    _obs_cur_{tgt.name} = {rhs}")
                lines.append(f"    {tgt.name} = max({tgt.name}, _obs_cur_{tgt.name})")
                lines.append(f"    buf.data[{slot_i}] = {tgt.name}")
            elif reduction == "TRAJECTORY_MIN":
                lines.append(f"    _obs_cur_{tgt.name} = {rhs}")
                lines.append(f"    {tgt.name} = min({tgt.name}, _obs_cur_{tgt.name})")
                lines.append(f"    buf.data[{slot_i}] = {tgt.name}")
            else:
                lines.append(f"    {tgt.name} = {rhs}")
                lines.append(f"    buf.data[{slot_i}] = {tgt.name}")

    if len(lines) == 3:
        lines.append("    pass")
    code = "\n".join(lines) + "\n"
    if "_splitmix64_" in code:
        code = "import math\n\n" + _SPLITMIX64_HELPERS + "\n" + "\n".join(lines[2:]) + "\n"
    return code


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
            is_pinnable = s.role in (SlotRole.CONSTANT, SlotRole.CONTROL_PARAM)
            if s.role == SlotRole.CONSTANT or (is_pinnable and (s.name in pinned_syms or s.name.lower() in pinned_syms)) or s.role == SlotRole.CONTROL_PARAM:
                lo, hi = v0, v0
            else:
                span = max(1.0, abs(v0) * 1.5)
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
    # UPG-008: Neutral time dimension when all declared symbols are strictly dimensionless.
    all_dimensionless = all(
        all(n == 0 for n, _d in s.dimension.exponents)
        for s in symbols_sorted
    )
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
            if not all_dimensionless:
                eq_dim[2] = eq_dim[2] - 1  # d/dt has T^-1 for dimensional systems
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
        reduction = str(getattr(obs, "reduction", "") or "").upper()
        if reduction in {"TRAJECTORY_MAX", "TRAJECTORY_MIN"}:
            obs_expr_kind = getattr(obs.expr, "kind", None)
            if obs_expr_kind != "sym":
                raise ValueError(
                    f"TRAJECTORY_EXPRESSION_UNSUPPORTED: observable {obs.target!r} with {reduction} "
                    f"requires a direct STATE symbol expression, got kind={obs_expr_kind!r}"
                )
            state_name = str(getattr(obs.expr, "name", ""))
            state_obj = sym_by_name.get(state_name)
            if state_obj is None or state_obj.role != SlotRole.STATE:
                raise ValueError(
                    f"TRAJECTORY_EXPRESSION_UNSUPPORTED: observable {obs.target!r} with {reduction} "
                    f"must reference a STATE symbol, got {state_name!r}"
                )
            state_dim = [
                (int(n // d) if (d != 0 and n % d == 0) else (n / d if d != 0 else 0))
                for n, d in state_obj.dimension.exponents
            ]
            ihcg_equations[obs.target] = {
                "expression": state_name,
                "si_dim": state_dim,
                "kind": reduction,
                "state_symbol": state_name,
            }
            obs_slot_idx = slot_map.get(obs.target)
            state_slot_idx = slot_map.get(state_name)
            if obs_slot_idx is not None:
                ihcg_slots[obs_slot_idx]["semantic_tag"] = 252 if reduction == "TRAJECTORY_MAX" else 251
                if state_slot_idx is not None:
                    initial_state[obs_slot_idx] = initial_state[state_slot_idx]
            if state_name in domain:
                domain[obs.target] = list(domain[state_name])
            continue

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

    # Final invariant (HOTFIX-A / NF-01): every MAX/MIN trajectory observable must appear in IHCG with same kind
    for obs in ir.observables:
        red = str(getattr(obs, "reduction", "") or "").upper()
        if red in {"TRAJECTORY_MAX", "TRAJECTORY_MIN"}:
            eq_entry = ihcg_equations.get(obs.target)
            if not isinstance(eq_entry, dict) or eq_entry.get("kind") != red:
                raise ValueError(
                    f"COMPILER_INVARIANT: observable {obs.target!r} reduction {red} was not preserved in IHCG"
                )

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
        # ARCH-006: If no explicit user duration, ensure horizon >= 10x slowest SI timescale
        ledger = extract_claim_ledger(user_prompt)
        if ledger.duration_si_s is None:
            rate_values: list[float] = []
            for s in symbols_sorted:
                exps = [
                    (int(n // d) if (d != 0 and n % d == 0) else (n / d if d != 0 else 0))
                    for n, d in s.dimension.exponents
                ]
                if exps == [0, 0, -1, 0, 0, 0, 0]:
                    val = float(s.initial_or_default or 0.0)
                    if 1e-7 < abs(val) < 1e6:
                        rate_values.append(abs(val))
            if rate_values:
                slowest_tau = 1.0 / min(rate_values)
                min_horizon = min(1e7, 10.0 * slowest_tau)
                if span < min_horizon:
                    span = min_horizon
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

    prompt_tokens = [
        tok for tok in _re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", user_prompt or "")
        if tok.lower() not in {"verify", "whether", "check", "compute", "model", "with", "that", "from", "this", "using", "given", "show", "prove"}
    ]
    derived_query = " ".join(prompt_tokens[:8]) or (title or getattr(ir, "title", None) or ir.ir_id)
    kernel_code = emit_python_kernel_from_sciir(ir)
    return {
        "blueprint_id": ir.ir_id[:32],
        "title": title or getattr(ir, "title", None) or ir.ir_id,
        "research_prompt": user_prompt,
        "falsifiable_hypothesis": " and ".join(ob["expression"] for ob in ihcg_obligations) or user_prompt[:200],
        "search_queries": [derived_query],
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

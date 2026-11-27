"""Tier-0 Deterministic Blueprint Compiler & Pre-Seal Epistemic Preflight Gate.

Implements the Four-Namespace Blueprint AST compiler:
  1. ``typed_slots``: mutable simulation buffer slots (DYNAMIC_STATE, CONTROL_PARAM,
     DERIVED_OBSERVABLE, CONSERVATION_AUX, TIME_STEP).
  2. ``constants``: immutable typed SI physical/model constants (never required to
     occupy mutable ``typed_slots``).
  3. ``derived_expressions``: pure algebraic definitions over ``typed_slots`` and
     ``constants`` (inlined cleanly for Z3/CSEL/HIN lowering unless explicitly promoted
     to a ``DERIVED_OBSERVABLE`` slot).
  4. Standard math/runtime built-ins.

The deterministic engine NEVER parses raw natural-language user prompts. All
formalization assumptions introduced during NL->AST translation or Tier-0 SI constant
resolution are recorded transparently in ``assumptions_ledger``.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import meaningfulness as m
from .expr_frontend import normalize_expression as _fe_normalize
from .formal_evidence import _anchor_evidence
from .ihcg_normalization import normalize_ihcg_evidence

# Unambiguous CODATA 2018 SI physical constants: symbol -> (value, si_unit, si_dim)
# SI vector order: [kg, m, s, A, K, mol, cd]
STANDARD_SI_CONSTANTS: Dict[str, Tuple[float, str, List[int]]] = {
    "R_gas": (8.314462618, "J/(mol*K)", [1, 2, -2, 0, -1, -1, 0]),
    "F_far": (96485.33212, "C/mol", [0, 0, 1, 1, 0, -1, 0]),
    "k_B": (1.380649e-23, "J/K", [1, 2, -2, 0, -1, 0, 0]),
    "c": (299792458.0, "m/s", [0, 1, -1, 0, 0, 0, 0]),
    "c_light": (299792458.0, "m/s", [0, 1, -1, 0, 0, 0, 0]),
    "N_A": (6.02214076e23, "1/mol", [0, 0, 0, 0, 0, -1, 0]),
    "g0": (9.80665, "m/s^2", [0, 1, -2, 0, 0, 0, 0]),
    "g_0": (9.80665, "m/s^2", [0, 1, -2, 0, 0, 0, 0]),
    "h": (6.62607015e-34, "J*s", [1, 2, -1, 0, 0, 0, 0]),
    "h_planck": (6.62607015e-34, "J*s", [1, 2, -1, 0, 0, 0, 0]),
    "hbar": (1.054571817e-34, "J*s", [1, 2, -1, 0, 0, 0, 0]),
    "sigma_SB": (5.670374419e-8, "W/(m^2*K^4)", [1, 0, -3, 0, -4, 0, 0]),
}

BUILTIN_MATH_AND_RUNTIME: Set[str] = {
    "math", "exp", "log", "log10", "sqrt", "abs", "min", "max",
    "sin", "cos", "tan", "tanh", "pow", "pi", "e", "inf", "isfinite",
    "t", "dt", "state", "initial", "telemetry", "variants",
    *m._ARITIES.keys(),
}

_UNIT_SUFFIX_RE = re.compile(
    r"_(?:mol_m3|mM|uM|nM|M|v|V|mV|s|ms|K|J|W|kW|MW|nW|Pa|m|km|cm|nm|kg|g|S_m2|F_m2|uF_cm2|rad|deg|ratio|norm|n)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BlueprintDiagnostic:
    code: str
    field: str
    message: str
    symbol: Optional[str] = None
    suggested_fix: str = ""
    original: Optional[str] = None
    column: Optional[int] = None
    offending_token: Optional[str] = None
    hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }
        if self.symbol:
            out["symbol"] = self.symbol
        if self.suggested_fix:
            out["suggested_fix"] = self.suggested_fix
        if self.original is not None:
            out["original"] = self.original
        if self.column is not None:
            out["column"] = self.column
        if self.offending_token is not None:
            out["offending_token"] = self.offending_token
        if self.hint:
            out["hint"] = self.hint
        return out

    def format_one(self) -> str:
        parts = [f"[{self.code}] {self.field}: {self.message}"]
        if self.original is not None:
            parts.append(f"expression={self.original!r}")
            if self.column is not None and self.column >= 1:
                col_idx = min(max(self.column - 1, 0), len(self.original))
                caret = " " * col_idx + "^"
                parts.append(f"col {self.column}:\n  {self.original}\n  {caret}")
        if self.offending_token:
            parts.append(f"token={self.offending_token!r}")
        if self.hint:
            parts.append(f"hint={self.hint}")
        return " | ".join(parts)


def _make_parse_diagnostic(
    field: str,
    expression: str,
    exc: Exception,
    *,
    code: str = "INVALID_EXPRESSION",
    symbol: Optional[str] = None,
    suggested_fix: str = "",
) -> BlueprintDiagnostic:
    col: Optional[int] = None
    tok: Optional[str] = None
    if isinstance(exc, SyntaxError):
        raw_offset = getattr(exc, "offset", None)
        if isinstance(raw_offset, int) and raw_offset >= 1:
            col = raw_offset
            tail = expression[max(0, col - 1):]
            m_tok = re.match(r"\s*([A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s])", tail)
            if m_tok:
                tok = m_tok.group(1)
    hint = (
        "Use formal Python-subset expressions over declared symbols "
        "(+, -, *, /, **, ==, !=, <, <=, >, >=, and, or, not, implies(a,b)); "
        "never English prose, quantifiers, or trailing units."
    )
    return BlueprintDiagnostic(
        code=code,
        field=field,
        message=str(exc),
        symbol=symbol,
        suggested_fix=suggested_fix or hint,
        original=expression,
        column=col,
        offending_token=tok,
        hint=hint,
    )


@dataclass
class BlueprintPreflightResult:
    ok: bool
    diagnostics: List[BlueprintDiagnostic] = field(default_factory=list)
    compiled_blueprint: Dict[str, Any] = field(default_factory=dict)
    compiled_ihcg_pack: Dict[str, Any] = field(default_factory=dict)
    assumptions_ledger: List[Dict[str, Any]] = field(default_factory=list)
    compiled_graph: Optional[m.IhcgGraph] = None

    def diagnostic_list_json(self) -> str:
        return json.dumps([d.to_dict() for d in self.diagnostics], indent=2)

    def error_summary(self) -> str:
        return "; ".join(d.format_one() for d in self.diagnostics)


def _extract_module_scalar_constants(code: str) -> Dict[str, float]:
    """Deterministically extract top-level scalar constant assignments from ``code_template``."""
    if not isinstance(code, str) or not code.strip():
        return {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    constants: Dict[str, float] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            val = _eval_scalar_ast(node.value, constants)
            if val is not None and math.isfinite(val):
                constants[name] = float(val)
    return constants


def _eval_scalar_ast(node: ast.AST, env: Mapping[str, float]) -> Optional[float]:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in env:
        return float(env[node.id])
    if isinstance(node, ast.UnaryOp):
        v = _eval_scalar_ast(node.operand, env)
        if v is None:
            return None
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
    if isinstance(node, ast.BinOp):
        a = _eval_scalar_ast(node.left, env)
        b = _eval_scalar_ast(node.right, env)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div) and b != 0:
            return a / b
        if isinstance(node.op, ast.Pow) and abs(b) <= 12:
            try:
                return a ** b
            except (OverflowError, ValueError):
                return None
    return None


def _normalize_constants_namespace(
    blueprint: Mapping[str, Any],
    pack: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Collect typed constants from blueprint.constants, pack.constants, extracted_constants, and fixed slots."""
    out: Dict[str, Dict[str, Any]] = {}

    def _ingest(raw_map: Any, default_source: str) -> None:
        if isinstance(raw_map, dict):
            for sym, val in raw_map.items():
                if not isinstance(sym, str) or not sym.isidentifier():
                    continue
                if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                    out[sym] = {
                        "symbol": sym,
                        "value": float(val),
                        "si_unit": "",
                        "si_dim": [0] * 7,
                        "source": default_source,
                    }
                elif isinstance(val, dict) and "value" in val:
                    v = val["value"]
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                        dims = val.get("si_dim")
                        if not (isinstance(dims, (list, tuple)) and len(dims) == 7 and all(type(x) is int for x in dims)):
                            dims = [0] * 7
                        out[sym] = {
                            "symbol": sym,
                            "value": float(v),
                            "si_unit": str(val.get("si_unit") or ""),
                            "si_dim": list(dims),
                            "source": str(val.get("source") or default_source),
                        }
        elif isinstance(raw_map, list):
            for item in raw_map:
                if isinstance(item, dict) and isinstance(item.get("symbol"), str) and "value" in item:
                    _ingest({item["symbol"]: item}, default_source)

    _ingest(blueprint.get("extracted_constants"), "LITERATURE")
    _ingest(blueprint.get("constants"), "LLM_DOMAIN_DEFAULT")
    formal_spec = blueprint.get("formal_spec")
    if isinstance(formal_spec, dict):
        _ingest(formal_spec.get("constants"), "LLM_DOMAIN_DEFAULT")
    _ingest(pack.get("constants"), "LLM_DOMAIN_DEFAULT")
    if isinstance(pack.get("evidence"), dict):
        _ingest(pack["evidence"].get("constants"), "LLM_DOMAIN_DEFAULT")
    return out


def _normalize_derived_expressions(
    blueprint: Mapping[str, Any],
    pack: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Collect pure derived algebraic expressions from blueprint and pack."""
    out: Dict[str, Dict[str, Any]] = {}

    def _ingest(raw_map: Any) -> None:
        if isinstance(raw_map, dict):
            for sym, val in raw_map.items():
                if not isinstance(sym, str) or not sym.isidentifier():
                    continue
                if isinstance(val, str) and val.strip():
                    out[sym] = {
                        "symbol": sym,
                        "expression": val.strip(),
                        "si_unit": "",
                        "si_dim": [0] * 7,
                        "promote_to_slot": False,
                    }
                elif isinstance(val, dict) and isinstance(val.get("expression"), str) and val["expression"].strip():
                    dims = val.get("si_dim")
                    if not (isinstance(dims, (list, tuple)) and len(dims) == 7 and all(type(x) is int for x in dims)):
                        dims = [0] * 7
                    out[sym] = {
                        "symbol": sym,
                        "expression": val["expression"].strip(),
                        "si_unit": str(val.get("si_unit") or ""),
                        "si_dim": list(dims),
                        "promote_to_slot": bool(val.get("promote_to_slot", False)),
                    }
        elif isinstance(raw_map, list):
            for item in raw_map:
                if isinstance(item, dict) and isinstance(item.get("symbol"), str):
                    _ingest({item["symbol"]: item})

    _ingest(blueprint.get("derived_expressions"))
    formal_spec = blueprint.get("formal_spec")
    if isinstance(formal_spec, dict):
        _ingest(formal_spec.get("derived_expressions"))
    _ingest(pack.get("derived_expressions"))
    if isinstance(pack.get("evidence"), dict):
        _ingest(pack["evidence"].get("derived_expressions"))
    return out


def _substitute_ast_symbols(expr: str, replacements: Mapping[str, ast.AST]) -> str:
    """Safely substitute symbol names in a formal expression using AST transformation."""
    if not replacements:
        return expr
    tree = m._parse(expr)

    class _Sub(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            # Never replace function call identifiers; only transform arguments.
            node.args = [self.visit(a) for a in node.args]
            return node

        def visit_Name(self, node: ast.Name):
            if node.id in replacements:
                return copy.deepcopy(replacements[node.id])
            return node

    replaced = _Sub().visit(tree)
    return ast.unparse(ast.fix_missing_locations(replaced))


def _resolve_derived_topological(
    derived: Mapping[str, Dict[str, Any]],
) -> Tuple[Dict[str, ast.AST], List[BlueprintDiagnostic]]:
    """Topologically expand pure derived expressions, detecting cycles cleanly."""
    diagnostics: List[BlueprintDiagnostic] = []
    parsed: Dict[str, ast.AST] = {}
    deps: Dict[str, Set[str]] = {}
    for sym, spec in derived.items():
        try:
            node = m._parse(spec["expression"])
        except (ValueError, SyntaxError) as exc:
            diagnostics.append(_make_parse_diagnostic(
                f"derived_expressions.{sym}",
                spec["expression"],
                exc,
                code="INVALID_DERIVED_EXPRESSION",
                symbol=sym,
                suggested_fix=f"Rewrite derived_expressions.{sym} using supported scalar math syntax.",
            ))
            continue
        parsed[sym] = node
        deps[sym] = m._variables(node) & set(derived.keys())

    expanded: Dict[str, ast.AST] = {}
    remaining = set(parsed.keys())
    while remaining:
        ready = sorted(s for s in remaining if deps[s] <= set(expanded.keys()))
        if not ready:
            for s in sorted(remaining):
                diagnostics.append(BlueprintDiagnostic(
                    code="CYCLIC_DERIVED_EXPRESSION",
                    field=f"derived_expressions.{s}",
                    symbol=s,
                    message=f"Cyclic dependency among derived_expressions: {sorted(remaining)}",
                    suggested_fix="Break the algebraic cycle or promote the state to typed_slots.",
                ))
            break
        for sym in ready:
            expr_str = ast.unparse(parsed[sym])
            if deps[sym]:
                expr_str = _substitute_ast_symbols(expr_str, {k: expanded[k] for k in deps[sym]})
            expanded[sym] = m._parse(expr_str)
            remaining.remove(sym)
    return expanded, diagnostics


def _find_bijective_aliases(
    unresolved: Set[str],
    known_symbols: Set[str],
) -> Dict[str, str]:
    """Resolve unambiguous case or unit-suffix aliases bijectively (never collapsing 2->1)."""
    if not unresolved or not known_symbols:
        return {}
    lower_map: Dict[str, List[str]] = {}
    stem_map: Dict[str, List[str]] = {}
    for sym in sorted(known_symbols):
        lower_map.setdefault(sym.lower(), []).append(sym)
        stem = _UNIT_SUFFIX_RE.sub("", sym).lower()
        stem_map.setdefault(stem, []).append(sym)

    candidates: Dict[str, str] = {}
    for sym in sorted(unresolved):
        low = sym.lower()
        if low in lower_map and len(lower_map[low]) == 1:
            candidates[sym] = lower_map[low][0]
            continue
        stem = _UNIT_SUFFIX_RE.sub("", sym).lower()
        if stem in stem_map and len(stem_map[stem]) == 1:
            candidates[sym] = stem_map[stem][0]
            continue
        if stem in lower_map and len(lower_map[stem]) == 1:
            candidates[sym] = lower_map[stem][0]

    # Enforce strict bijectivity: drop any target claimed by >1 unresolved symbol.
    target_counts: Dict[str, int] = {}
    for tgt in candidates.values():
        target_counts[tgt] = target_counts.get(tgt, 0) + 1
    return {src: tgt for src, tgt in candidates.items() if target_counts[tgt] == 1}


def _collect_assumptions_ledger(
    blueprint: Mapping[str, Any],
    pack: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    from ..schemas.research import coerce_assumption_entry
    ledger: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()

    def _add(item: Any) -> None:
        coerced = coerce_assumption_entry(item)
        if not coerced:
            return
        key = (coerced["kind"], coerced["symbol_or_field"], coerced["value_or_expr"])
        if key in seen:
            return
        seen.add(key)
        ledger.append(coerced)

    for container in (
        blueprint.get("assumptions_ledger"),
        (blueprint.get("formal_spec") or {}).get("assumptions_ledger") if isinstance(blueprint.get("formal_spec"), dict) else None,
        pack.get("assumptions_ledger"),
        (pack.get("evidence") or {}).get("assumptions_ledger") if isinstance(pack.get("evidence"), dict) else None,
    ):
        if isinstance(container, (list, tuple)):
            for entry in container:
                if isinstance(entry, dict):
                    _add(entry)
                elif hasattr(entry, "model_dump"):
                    _add(entry.model_dump())
    return ledger


def _check_initial_pool_conservation(
    blueprint: Mapping[str, Any],
    graph: m.IhcgGraph,
    constants: Mapping[str, Dict[str, Any]],
) -> List[BlueprintDiagnostic]:
    """Verify initial_state satisfies declared equality pool conservation laws within tolerance."""
    diagnostics: List[BlueprintDiagnostic] = []
    contracts = blueprint.get("contracts") or []
    if not contracts:
        return diagnostics
    slot_by_sym = {s.symbol: s.slot_idx for s in graph.slots}
    const_vals = {k: float(v["value"]) for k, v in constants.items()}
    for k, v in graph.control_values.items():
        const_vals.setdefault(k, float(v))
    for k, (lo, hi) in graph.omega.items():
        if lo == hi:
            const_vals.setdefault(k, float(lo))

    # Inspect equality conservation guards and algebraic conservation equations.
    eq_exprs: List[Tuple[str, str]] = []
    for ob in graph.obligations:
        if ob.obligation_class == "CONSERVATION_GUARD":
            eq_exprs.append((f"obligations[{ob.id!r}]", ob.expression))
    for name, eq in graph.equations.items():
        try:
            node = m._parse(eq.expression)
        except (ValueError, SyntaxError):
            continue
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            eq_exprs.append((f"equations[{name!r}]", eq.expression))

    for field_path, expr in eq_exprs:
        try:
            node = m._parse(expr)
        except (ValueError, SyntaxError):
            continue
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
            continue
        lhs_vars = m._variables(node.left)
        rhs_vars = m._variables(node.comparators[0])
        if not (lhs_vars | rhs_vars):
            continue
        # Only check pure additive pool equations whose symbols are all dynamic states or known constants.
        dynamic_syms = {s.symbol for s in graph.slots if s.role == "DYNAMIC_STATE"}
        if not ((lhs_vars | rhs_vars) & dynamic_syms):
            continue
        for c_idx, contract in enumerate(contracts):
            init = contract.get("initial_state") if isinstance(contract, dict) else getattr(contract, "initial_state", None)
            if not isinstance(init, (list, tuple)):
                continue
            env = dict(const_vals)
            complete = True
            for sym in (lhs_vars | rhs_vars):
                if sym in env:
                    continue
                idx = slot_by_sym.get(sym)
                if idx is not None and 0 <= idx < len(init):
                    env[sym] = float(init[idx])
                else:
                    complete = False
                    break
            if not complete:
                continue
            lhs_val = _eval_scalar_ast(node.left, env)
            rhs_val = _eval_scalar_ast(node.comparators[0], env)
            if lhs_val is None or rhs_val is None:
                continue
            scale = max(1.0, abs(lhs_val), abs(rhs_val))
            if abs(lhs_val - rhs_val) > 1e-2 * scale:
                cid = contract.get("contract_id", f"contract[{c_idx}]") if isinstance(contract, dict) else getattr(contract, "contract_id", f"contract[{c_idx}]")
                diagnostics.append(BlueprintDiagnostic(
                    code="INITIAL_POOL_MISMATCH",
                    field=f"contracts.{cid}.initial_state",
                    message=(
                        f"Initial state violates declared conservation law {field_path} ({expr}): "
                        f"lhs={lhs_val:.6g} vs rhs={rhs_val:.6g}"
                    ),
                    suggested_fix="Adjust initial_state values so conserved pool components sum to the declared pool constant.",
                ))
    return diagnostics


_ALLOWED_REL_OPS: Set[str] = {"<", "<=", ">", ">=", "==", "!="}


def _eval_numeric_bound(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        return f if math.isfinite(f) else None
    if isinstance(val, str) and val.strip():
        res = _fe_normalize(val)
        if res.ok and res.canonical is not None:
            try:
                node = m._parse(res.canonical)
                if not m._variables(node):
                    iv = m._iv(node, {})
                    if math.isfinite(iv.lo) and math.isfinite(iv.hi) and abs(iv.hi - iv.lo) <= 1e-9 * max(1.0, abs(iv.lo)):
                        return 0.5 * (float(iv.lo) + float(iv.hi))
            except Exception:
                return None
    return None


def _format_bound_endpoint(
    val: Any,
    field: str,
    declared_symbols: Set[str],
    domain_preview: Mapping[str, Tuple[float, float]],
    si_dims: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[BlueprintDiagnostic]]:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if not math.isfinite(float(val)):
            return None, BlueprintDiagnostic(
                code="INVALID_BOUND",
                field=field,
                message=f"Non-finite bound value {val!r}.",
            )
        return repr(val), None
    if isinstance(val, str) and val.strip():
        res = _fe_normalize(val, declared_symbols=declared_symbols, domain=domain_preview, si_dims=si_dims)
        if res.ok and res.canonical is not None:
            node = m._parse(res.canonical)
            if isinstance(node, (ast.Compare, ast.BoolOp)):
                return None, BlueprintDiagnostic(
                    code="INVALID_BOUND",
                    field=field,
                    message="Bound endpoint must be an arithmetic expression, not a comparison/boolean.",
                    original=val,
                    column=1,
                )
            return res.canonical, None
        ed = res.diagnostics[0] if res.diagnostics else None
        return None, BlueprintDiagnostic(
            code=ed.code if ed else "INVALID_BOUND",
            field=field,
            message=ed.message if ed else f"Invalid bound expression {val!r}.",
            original=val,
            column=ed.column if ed else 1,
            offending_token=ed.offending_token if ed else None,
            hint=ed.hint if ed else "",
        )
    return None, BlueprintDiagnostic(
        code="INVALID_BOUND",
        field=field,
        message=f"Bound endpoint must be a finite number or arithmetic string, got {type(val).__name__}.",
    )


def _lower_predicate_tree(
    pred: Any,
    field: str,
    *,
    declared_symbols: Set[str],
    domain_preview: Mapping[str, Tuple[float, float]],
    si_dims: Mapping[str, Any],
    depth: int = 0,
) -> Tuple[Optional[str], List[BlueprintDiagnostic]]:
    if depth > 32:
        return None, [BlueprintDiagnostic(code="PREDICATE_TOO_DEEP", field=field, message="Predicate nesting exceeds cap (32).")]
    if isinstance(pred, str):
        res = _fe_normalize(pred, declared_symbols=declared_symbols, domain=domain_preview, si_dims=si_dims)
        if res.ok and res.canonical is not None:
            return res.canonical, []
        return None, [
            BlueprintDiagnostic(
                code=ed.code,
                field=field,
                message=ed.message,
                original=ed.original,
                column=ed.column,
                offending_token=ed.offending_token,
                hint=ed.hint,
                suggested_fix=ed.hint,
            )
            for ed in res.diagnostics
        ]
    if isinstance(pred, dict) and {"lhs", "op", "rhs"} <= set(pred.keys()) and len(pred) == 3:
        pred = {"rel": pred}
    if not isinstance(pred, dict) or len(pred) != 1:
        return None, [BlueprintDiagnostic(
            code="INVALID_PREDICATE",
            field=field,
            message="Predicate must be a single-key object: rel | all_of | any_of | not | implies.",
        )]
    key, val = next(iter(pred.items()))
    if key == "rel":
        if not isinstance(val, dict) or not {"lhs", "op", "rhs"} <= set(val.keys()):
            return None, [BlueprintDiagnostic(
                code="INVALID_PREDICATE",
                field=f"{field}.rel",
                message="rel predicate requires lhs, op, rhs.",
            )]
        op = val["op"]
        if op not in _ALLOWED_REL_OPS:
            return None, [BlueprintDiagnostic(
                code="INVALID_PREDICATE",
                field=f"{field}.rel.op",
                message=f"Unsupported relational operator {op!r}; allowed: {sorted(_ALLOWED_REL_OPS)}.",
            )]
        lhs_raw, rhs_raw = val["lhs"], val["rhs"]
        if not isinstance(lhs_raw, str) or not isinstance(rhs_raw, str):
            return None, [BlueprintDiagnostic(
                code="INVALID_PREDICATE",
                field=f"{field}.rel",
                message="rel.lhs and rel.rhs must be arithmetic expression strings.",
            )]
        diags: List[BlueprintDiagnostic] = []
        lhs_res = _fe_normalize(lhs_raw, declared_symbols=declared_symbols, domain=domain_preview, si_dims=si_dims)
        rhs_res = _fe_normalize(rhs_raw, declared_symbols=declared_symbols, domain=domain_preview, si_dims=si_dims)
        for side_name, side_raw, side_res in (("lhs", lhs_raw, lhs_res), ("rhs", rhs_raw, rhs_res)):
            if not side_res.ok or side_res.canonical is None:
                for ed in side_res.diagnostics:
                    diags.append(BlueprintDiagnostic(
                        code=ed.code,
                        field=f"{field}.rel.{side_name}",
                        message=ed.message,
                        original=ed.original,
                        column=ed.column,
                        offending_token=ed.offending_token,
                        hint=ed.hint,
                    ))
            else:
                side_node = m._parse(side_res.canonical)
                if isinstance(side_node, (ast.Compare, ast.BoolOp)):
                    diags.append(BlueprintDiagnostic(
                        code="NON_ARITHMETIC_OPERAND",
                        field=f"{field}.rel.{side_name}",
                        message=f"rel.{side_name} must be an arithmetic expression, not a comparison/boolean.",
                        original=side_raw,
                        column=1,
                    ))
        if diags:
            return None, diags
        op_map = {
            "<": ast.Lt(), "<=": ast.LtE(), ">": ast.Gt(),
            ">=": ast.GtE(), "==": ast.Eq(), "!=": ast.NotEq(),
        }
        cmp_node = ast.Compare(
            left=m._parse(lhs_res.canonical),
            ops=[op_map[op]],
            comparators=[m._parse(rhs_res.canonical)],
        )
        return ast.unparse(ast.fix_missing_locations(cmp_node)), []

    if key in ("all_of", "any_of"):
        if not isinstance(val, list) or not val:
            return None, [BlueprintDiagnostic(code="INVALID_PREDICATE", field=f"{field}.{key}", message=f"{key} must be a non-empty list.")]
        sub_nodes: List[ast.AST] = []
        diags = []
        for idx, sub in enumerate(val):
            s_expr, s_diags = _lower_predicate_tree(
                sub, f"{field}.{key}[{idx}]",
                declared_symbols=declared_symbols, domain_preview=domain_preview, si_dims=si_dims, depth=depth + 1,
            )
            diags.extend(s_diags)
            if s_expr is not None:
                sub_nodes.append(m._parse(s_expr))
        if diags:
            return None, diags
        if len(sub_nodes) == 1:
            return ast.unparse(sub_nodes[0]), []
        bool_node = ast.BoolOp(op=ast.And() if key == "all_of" else ast.Or(), values=sub_nodes)
        return ast.unparse(ast.fix_missing_locations(bool_node)), []

    if key == "not":
        s_expr, s_diags = _lower_predicate_tree(
            val, f"{field}.not",
            declared_symbols=declared_symbols, domain_preview=domain_preview, si_dims=si_dims, depth=depth + 1,
        )
        if s_diags or s_expr is None:
            return None, s_diags
        not_node = ast.UnaryOp(op=ast.Not(), operand=m._parse(s_expr))
        return ast.unparse(ast.fix_missing_locations(not_node)), []

    if key == "implies":
        if not isinstance(val, list) or len(val) != 2:
            return None, [BlueprintDiagnostic(code="INVALID_PREDICATE", field=f"{field}.implies", message="implies requires a 2-element list [antecedent, consequent].")]
        p_expr, p_diags = _lower_predicate_tree(
            val[0], f"{field}.implies[0]",
            declared_symbols=declared_symbols, domain_preview=domain_preview, si_dims=si_dims, depth=depth + 1,
        )
        q_expr, q_diags = _lower_predicate_tree(
            val[1], f"{field}.implies[1]",
            declared_symbols=declared_symbols, domain_preview=domain_preview, si_dims=si_dims, depth=depth + 1,
        )
        if p_diags or q_diags or p_expr is None or q_expr is None:
            return None, p_diags + q_diags
        return f"implies({p_expr}, {q_expr})", []

    return None, [BlueprintDiagnostic(
        code="INVALID_PREDICATE",
        field=field,
        message=f"Unknown predicate operator {key!r}; allowed: rel, all_of, any_of, not, implies.",
    )]


def _lower_bound_object(
    bound: Any,
    field: str,
    *,
    declared_symbols: Set[str],
    domain_preview: Mapping[str, Tuple[float, float]],
    si_dims: Mapping[str, Any],
) -> Tuple[Optional[str], List[BlueprintDiagnostic]]:
    if not isinstance(bound, dict) or not {"symbol", "lo", "hi"} <= set(bound.keys()):
        return None, [BlueprintDiagnostic(
            code="INVALID_BOUND",
            field=field,
            message="Bound object requires symbol, lo, hi (optional lo_open, hi_open).",
        )]
    sym = bound["symbol"]
    if not isinstance(sym, str) or not sym.isidentifier():
        return None, [BlueprintDiagnostic(code="INVALID_BOUND", field=f"{field}.symbol", message=f"Invalid symbol {sym!r}.")]
    lo_open = bool(bound.get("lo_open", False))
    hi_open = bool(bound.get("hi_open", False))
    lo_s, lo_d = _format_bound_endpoint(bound["lo"], f"{field}.lo", declared_symbols, domain_preview, si_dims)
    hi_s, hi_d = _format_bound_endpoint(bound["hi"], f"{field}.hi", declared_symbols, domain_preview, si_dims)
    diags = [d for d in (lo_d, hi_d) if d is not None]
    if diags or lo_s is None or hi_s is None:
        return None, diags

    lo_val = _eval_numeric_bound(bound["lo"])
    hi_val = _eval_numeric_bound(bound["hi"])
    if lo_val is not None and hi_val is not None and lo_val > hi_val:
        return None, [BlueprintDiagnostic(
            code="INVALID_BOUND",
            field=field,
            symbol=sym,
            message=f"Bound lower {lo_val} exceeds upper {hi_val} for {sym!r}.",
        )]
    if sym in domain_preview and lo_val is not None and hi_val is not None:
        dom_lo, dom_hi = domain_preview[sym]
        tol = 1e-12 * max(1.0, abs(dom_lo), abs(dom_hi))
        if lo_val < dom_lo - tol or hi_val > dom_hi + tol:
            return None, [BlueprintDiagnostic(
                code="BINDER_OUTSIDE_DOMAIN",
                field=field,
                symbol=sym,
                message=f"Binder/bound [{lo_val}, {hi_val}] for {sym!r} is not a subset of declared domain[{sym!r}] = [{dom_lo}, {dom_hi}].",
                suggested_fix=f"Ensure binder/bound for {sym!r} lies within domain[{sym!r}].",
            )]
    lo_op = "<" if lo_open else "<="
    hi_op = "<" if hi_open else "<="
    return f"{lo_s} {lo_op} {sym} {hi_op} {hi_s}", []


def _lower_typed_channel(
    pack: Dict[str, Any],
    *,
    declared_symbols: Set[str],
    domain_preview: Mapping[str, Tuple[float, float]],
    si_dims: Mapping[str, Any],
    assumptions_ledger: List[Dict[str, Any]],
) -> List[BlueprintDiagnostic]:
    """Deterministically lower LLM-facing typed predicates/bounds/fixed/binders/notes into canonical strings."""
    evidence = pack["evidence"]
    diags: List[BlueprintDiagnostic] = []

    def _remember(path: str, orig: Any) -> None:
        ex = pack.setdefault("execution", {})
        if isinstance(ex, dict):
            hist = ex.setdefault("normalization_source", {})
            if isinstance(hist, dict):
                hist.setdefault(path, copy.deepcopy(orig))

    # 1. Obligations: accept either `expression` OR typed `predicate` (never both, never neither)
    for idx, ob in enumerate(evidence.get("obligations") or []):
        if not isinstance(ob, dict):
            continue
        has_expr = ob.get("expression") is not None
        has_pred = ob.get("predicate") is not None
        if has_expr and has_pred:
            diags.append(BlueprintDiagnostic(
                code="INVALID_OBLIGATION",
                field=f"obligations[{idx}]",
                message="Obligation must specify exactly one of 'expression' or 'predicate', not both.",
            ))
            continue
        if has_pred:
            pred_obj = ob.pop("predicate")
            lowered, p_diags = _lower_predicate_tree(
                pred_obj,
                f"obligations[{idx}].predicate",
                declared_symbols=declared_symbols,
                domain_preview=domain_preview,
                si_dims=si_dims,
            )
            diags.extend(p_diags)
            if lowered is not None:
                _remember(f"evidence.obligations[{idx}].predicate", pred_obj)
                ob["expression"] = lowered

    # 2. Environmental assumptions: string OR {"bound":...} | {"fixed":...} | {"predicate":...}
    raw_env = evidence.get("environmental_assumptions")
    if isinstance(raw_env, list):
        new_env: List[str] = []
        for idx, item in enumerate(raw_env):
            if isinstance(item, str):
                new_env.append(item)
                continue
            if isinstance(item, dict) and len(item) == 1:
                kind, payload = next(iter(item.items()))
                if kind == "bound":
                    expr_s, b_diags = _lower_bound_object(
                        payload, f"environmental_assumptions[{idx}].bound",
                        declared_symbols=declared_symbols, domain_preview=domain_preview, si_dims=si_dims,
                    )
                    diags.extend(b_diags)
                    if expr_s is not None:
                        _remember(f"evidence.environmental_assumptions[{idx}]", item)
                        new_env.append(expr_s)
                    continue
                if kind == "fixed":
                    if not isinstance(payload, dict) or not {"symbol", "value"} <= set(payload.keys()):
                        diags.append(BlueprintDiagnostic(
                            code="INVALID_ASSUMPTION",
                            field=f"environmental_assumptions[{idx}].fixed",
                            message="fixed assumption requires symbol and value.",
                        ))
                        continue
                    sym = payload["symbol"]
                    if not isinstance(sym, str) or not sym.isidentifier():
                        diags.append(BlueprintDiagnostic(
                            code="INVALID_ASSUMPTION",
                            field=f"environmental_assumptions[{idx}].fixed.symbol",
                            message=f"Invalid symbol {sym!r}.",
                        ))
                        continue
                    val_s, val_d = _format_bound_endpoint(
                        payload["value"], f"environmental_assumptions[{idx}].fixed.value",
                        declared_symbols, domain_preview, si_dims,
                    )
                    if val_d is not None or val_s is None:
                        if val_d:
                            diags.append(val_d)
                        continue
                    _remember(f"evidence.environmental_assumptions[{idx}]", item)
                    new_env.append(f"{sym} == {val_s}")
                    continue
                if kind == "predicate":
                    expr_s, p_diags = _lower_predicate_tree(
                        payload, f"environmental_assumptions[{idx}].predicate",
                        declared_symbols=declared_symbols, domain_preview=domain_preview, si_dims=si_dims,
                    )
                    diags.extend(p_diags)
                    if expr_s is not None:
                        _remember(f"evidence.environmental_assumptions[{idx}]", item)
                        new_env.append(expr_s)
                    continue
            diags.append(BlueprintDiagnostic(
                code="INVALID_ASSUMPTION",
                field=f"environmental_assumptions[{idx}]",
                message="Environmental assumption must be a formal string or typed {bound|fixed|predicate} object.",
            ))
        evidence["environmental_assumptions"] = new_env

    # 3. Quantifier binders: [{symbol, lo, hi, lo_open?, hi_open?}] -> checked ⊆ domain & lowered to env assumptions
    binders = evidence.pop("quantifier_binders", None)
    if binders is None:
        binders = pack.pop("quantifier_binders", None)
    else:
        pack.pop("quantifier_binders", None)
    if binders:
        if not isinstance(binders, (list, tuple)):
            diags.append(BlueprintDiagnostic(
                code="INVALID_QUANTIFIER_BINDERS",
                field="quantifier_binders",
                message="quantifier_binders must be a list of {symbol, lo, hi, lo_open?, hi_open?} objects.",
            ))
        else:
            _remember("evidence.quantifier_binders", binders)
            env_list = list(evidence.get("environmental_assumptions") or [])
            for idx, b in enumerate(binders):
                expr_s, b_diags = _lower_bound_object(
                    b, f"quantifier_binders[{idx}]",
                    declared_symbols=declared_symbols, domain_preview=domain_preview, si_dims=si_dims,
                )
                diags.extend(b_diags)
                if expr_s is not None and expr_s not in env_list:
                    env_list.append(expr_s)
            evidence["environmental_assumptions"] = env_list

    # 4. Notes: free prose -> lowered into assumptions_ledger as kind="NOTE" (never executed, never in sealed IhcgGraph)
    raw_notes: List[Any] = []
    if isinstance(pack.get("notes"), (list, tuple)):
        raw_notes.extend(pack["notes"])
    ev_notes = evidence.pop("notes", None)
    if isinstance(ev_notes, (list, tuple)):
        raw_notes.extend(ev_notes)
    seen_notes = {(e.get("kind"), e.get("value_or_expr")) for e in assumptions_ledger}
    for note in raw_notes:
        if isinstance(note, str) and note.strip():
            text = note.strip()[:400]
            if ("NOTE", text) not in seen_notes:
                seen_notes.add(("NOTE", text))
                assumptions_ledger.append({
                    "kind": "NOTE",
                    "symbol_or_field": "notes",
                    "value_or_expr": text,
                    "source": "LLM_DOMAIN_DEFAULT",
                    "rationale": "Non-executable formalization note.",
                })

    return diags


def compile_and_preflight_blueprint(
    blueprint_dict: Mapping[str, Any],
    ihcg_pack: Mapping[str, Any],
    *,
    question: str = "",
    run_z3_preflight: bool = True,
) -> BlueprintPreflightResult:
    """Tier-0 deterministic Blueprint AST compiler and pre-seal epistemic preflight gate.

    Compiles a 4-namespace ``FormalResearchSpec`` / ``ihcg_evidence`` payload into a
    self-consistent, IHCG-sealable blueprint + pack without ever parsing raw user prose.
    """
    bp = copy.deepcopy(dict(blueprint_dict))
    pack = copy.deepcopy(dict(ihcg_pack))
    diagnostics: List[BlueprintDiagnostic] = []

    if not isinstance(pack, dict) or not isinstance(pack.get("evidence"), dict):
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=[BlueprintDiagnostic(
                code="INVALID_PACK_STRUCTURE",
                field="ihcg_evidence",
                message="ihcg_evidence must be a JSON object with 'evidence', 'domain', and 'archetype'.",
            )],
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
        )

    assumptions_ledger = _collect_assumptions_ledger(bp, pack)
    constants = _normalize_constants_namespace(bp, pack)
    derived_exprs = _normalize_derived_expressions(bp, pack)
    code_constants = _extract_module_scalar_constants(str(bp.get("code_template") or ""))

    # Strip non-IHCG top-level helper keys from pack before normalization.
    pack.pop("constants", None)
    pack.pop("derived_expressions", None)
    pack.pop("assumptions_ledger", None)
    pack_typed_slots = pack.pop("typed_slots", None)
    evidence = pack["evidence"]
    evidence.pop("constants", None)
    evidence.pop("derived_expressions", None)
    evidence.pop("assumptions_ledger", None)

    # 1. Assemble mutable typed_slots namespace.
    raw_slots = bp.get("typed_slots") or pack_typed_slots or evidence.get("slots") or []
    if not isinstance(raw_slots, (list, tuple)) or any(not isinstance(s, dict) for s in raw_slots):
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=[BlueprintDiagnostic(
                code="INVALID_TYPED_SLOTS",
                field="typed_slots",
                message="typed_slots must be an array of typed slot objects.",
            )],
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
        )

    variable_map: Dict[str, int] = dict(bp.get("variable_map") or {})
    if not raw_slots and variable_map:
        # When blueprint emitted variable_map without typed_slots (and formalization obeyed
        # "do not restate slots"), derive typed_slots deterministically from variable_map + equations.
        ev_eqs = evidence.get("equations") if isinstance(evidence.get("equations"), dict) else {}
        dynamic_states: Set[str] = set()
        eq_dims: Dict[str, List[int]] = {}
        for eq_k, eq_v in ev_eqs.items():
            if isinstance(eq_v, dict):
                kind_v = str(eq_v.get("kind") or "ALGEBRAIC")
                st_sym = eq_v.get("state_symbol")
                if kind_v in {"ODE_RHS", "FIXED_POINT_RESIDUAL", "FIXED_POINT_MAP"}:
                    if isinstance(st_sym, str) and st_sym.isidentifier():
                        dynamic_states.add(st_sym)
                    elif isinstance(eq_k, str) and eq_k in variable_map:
                        dynamic_states.add(eq_k)
                dims_v = eq_v.get("si_dim")
                if isinstance(dims_v, (list, tuple)) and len(dims_v) == 7 and all(type(x) is int for x in dims_v):
                    eq_dims[eq_k] = list(dims_v)
        synthesized: List[Dict[str, Any]] = []
        for sym, idx_v in sorted(variable_map.items(), key=lambda kv: int(kv[1])):
            if not isinstance(sym, str) or not sym.isidentifier():
                continue
            if sym in dynamic_states:
                role = "DYNAMIC_STATE"
                dims = [0] * 7
            elif sym in ev_eqs:
                role = "DERIVED_OBSERVABLE"
                dims = eq_dims.get(sym, [0] * 7)
            else:
                role = "CONTROL_PARAM"
                dims = [0] * 7
            synthesized.append({
                "symbol": sym,
                "slot_idx": int(idx_v),
                "si_dim": dims,
                "role": role,
                "semantic_tag": 0,
            })
        raw_slots = synthesized
        bp["typed_slots"] = copy.deepcopy(synthesized)
    slots: List[Dict[str, Any]] = []
    slot_bounds: Dict[str, List[float]] = {}
    slot_fixed_values: Dict[str, float] = {}
    used_indices: Set[int] = set(variable_map.values())
    next_idx = (max(used_indices) + 1) if used_indices else 0

    for idx_pos, raw_slot in enumerate(raw_slots):
        slot = copy.deepcopy(raw_slot)
        sym = slot.get("symbol")
        if not isinstance(sym, str) or not sym.isidentifier():
            diagnostics.append(BlueprintDiagnostic(
                code="INVALID_SLOT_SYMBOL",
                field=f"typed_slots[{idx_pos}].symbol",
                message=f"Invalid slot symbol {sym!r}.",
            ))
            continue
        # Normalize TIME_STEP role to DYNAMIC_STATE for SlotSpec compatibility.
        if slot.get("role") == "TIME_STEP":
            slot["role"] = "DYNAMIC_STATE"
        # Extract optional extended metadata before SlotSpec validation.
        val = slot.pop("value", None)
        slot.pop("prompt_symbol", None)
        slot.pop("si_unit", None)
        slot.pop("initial_value", None)
        bounds = slot.pop("bounds", None)
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            try:
                lo, hi = float(bounds[0]), float(bounds[1])
                if math.isfinite(lo) and math.isfinite(hi) and lo <= hi:
                    slot_bounds[sym] = [lo, hi]
            except (ValueError, TypeError):
                pass
        if val is not None and isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
            if slot.get("role") == "CONTROL_PARAM":
                slot_fixed_values[sym] = float(val)
            else:
                diagnostics.append(BlueprintDiagnostic(
                    code="INVALID_FIXED_VALUE_ROLE",
                    field=f"typed_slots[{idx_pos}].value",
                    symbol=sym,
                    message=f"Only CONTROL_PARAM slots may declare a fixed value; {sym!r} has role={slot.get('role')!r}.",
                ))
        if sym in variable_map:
            slot["slot_idx"] = variable_map[sym]
        elif isinstance(slot.get("slot_idx"), int) and slot["slot_idx"] >= 0:
            variable_map[sym] = slot["slot_idx"]
            used_indices.add(slot["slot_idx"])
            next_idx = max(next_idx, slot["slot_idx"] + 1)
        else:
            slot["slot_idx"] = next_idx
            variable_map[sym] = next_idx
            used_indices.add(next_idx)
            next_idx += 1
        slots.append(slot)

    if diagnostics:
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=diagnostics,
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    evidence["slots"] = slots
    declared_slots: Set[str] = {s["symbol"] for s in slots}
    if len(declared_slots) != len(slots):
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=[BlueprintDiagnostic(
                code="DUPLICATE_SLOT_SYMBOL",
                field="typed_slots",
                message="Duplicate symbol declarations in typed_slots.",
            )],
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    # Pre-pass: run the deterministic token-level expression front end on obligations,
    # equations, derived_expressions, and environmental_assumptions BEFORE the first _parse.
    raw_domain = pack.get("domain", {})
    if not isinstance(raw_domain, dict):
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=[BlueprintDiagnostic(
                code="INVALID_DOMAIN",
                field="ihcg_evidence.domain",
                message="domain must be a JSON object mapping symbol -> [lower, upper].",
            )],
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    slot_dims_map = {s["symbol"]: s.get("si_dim") for s in slots if isinstance(s, dict) and "symbol" in s}
    all_declared = declared_slots | set(constants.keys()) | set(derived_exprs.keys())
    q_mode = evidence.get("quantifier_mode")
    # Build numeric domain preview for binder subset checks
    domain_preview: Dict[str, Tuple[float, float]] = {}
    for k, v in raw_domain.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                domain_preview[k] = (float(v[0]), float(v[1]))
            except (TypeError, ValueError):
                pass
        elif isinstance(v, dict):
            lo_k = "lower" if "lower" in v else ("min" if "min" in v else None)
            hi_k = "upper" if "upper" in v else ("max" if "max" in v else None)
            if lo_k and hi_k:
                try:
                    domain_preview[k] = (float(v[lo_k]), float(v[hi_k]))
                except (TypeError, ValueError):
                    pass
    for sym, bnd in slot_bounds.items():
        domain_preview.setdefault(sym, (float(bnd[0]), float(bnd[1])))

    def _fe_record_transform(path: str, original_expr: str, transforms: Tuple[str, ...]) -> None:
        if not transforms:
            return
        ex = pack.setdefault("execution", {})
        if isinstance(ex, dict):
            hist = ex.setdefault("normalization_source", {})
            if isinstance(hist, dict):
                hist.setdefault(path, original_expr)
                tmap = hist.setdefault("expr_frontend_transforms", {})
                if isinstance(tmap, dict):
                    tmap[path] = list(transforms)

    fe_diags: List[BlueprintDiagnostic] = []
    fe_binder_assumptions: List[str] = []

    # Step 3 typed-channel lowering pre-pass (obligations.predicate, typed environmental_assumptions,
    # quantifier_binders, notes -> assumptions_ledger[kind=NOTE]).
    typed_diags = _lower_typed_channel(
        pack,
        declared_symbols=all_declared,
        domain_preview=domain_preview,
        si_dims=slot_dims_map,
        assumptions_ledger=assumptions_ledger,
    )
    fe_diags.extend(typed_diags)

    # ARCH-001 (BUG-004): Normalize evidence["equations"] through the exact same
    # expr_frontend as derived_expressions and obligations so variadic min/max,
    # math.fabs, conditionals, and unicode operators have 100% parity everywhere.
    eqs_map = evidence.get("equations")
    if isinstance(eqs_map, dict):
        for eq_sym, eq_spec in eqs_map.items():
            if isinstance(eq_spec, dict) and isinstance(eq_spec.get("expression"), str):
                res = _fe_normalize(
                    eq_spec["expression"],
                    declared_symbols=all_declared,
                    domain=domain_preview,
                    si_dims=slot_dims_map,
                    quantifier_mode=q_mode,
                )
                if res.ok and res.canonical is not None:
                    if res.canonical != eq_spec["expression"] or res.transforms:
                        _fe_record_transform(f"equations.{eq_sym}.expression", eq_spec["expression"], res.transforms)
                        eq_spec["expression"] = res.canonical
                    fe_binder_assumptions.extend(res.binder_assumptions)
                else:
                    for ed in res.diagnostics:
                        fe_diags.append(BlueprintDiagnostic(
                            code=ed.code,
                            field=f"equations.{eq_sym}.expression",
                            symbol=eq_sym,
                            message=ed.message,
                            original=ed.original,
                            column=ed.column,
                            offending_token=ed.offending_token,
                            hint=ed.hint,
                            suggested_fix=ed.hint,
                        ))

    # Normalize derived_expressions via front end (non-blocking here so unreferenced
    # blueprint helpers never block compilation; referenced ones are checked in _resolve_derived_topological).
    derived_fe_errors: Dict[str, List[BlueprintDiagnostic]] = {}
    for d_sym, d_spec in derived_exprs.items():
        if isinstance(d_spec, dict) and isinstance(d_spec.get("expression"), str):
            res = _fe_normalize(
                d_spec["expression"],
                declared_symbols=all_declared,
                domain=domain_preview,
                si_dims=slot_dims_map,
                quantifier_mode=q_mode,
            )
            if res.ok and res.canonical is not None:
                if res.canonical != d_spec["expression"] or res.transforms:
                    _fe_record_transform(f"derived_expressions.{d_sym}.expression", d_spec["expression"], res.transforms)
                    d_spec["expression"] = res.canonical
            else:
                derived_fe_errors[d_sym] = [
                    BlueprintDiagnostic(
                        code=ed.code,
                        field=f"derived_expressions.{d_sym}.expression",
                        symbol=d_sym,
                        message=ed.message,
                        original=ed.original,
                        column=ed.column,
                        offending_token=ed.offending_token,
                        hint=ed.hint,
                        suggested_fix=ed.hint,
                    )
                    for ed in res.diagnostics
                ]

    # Normalize obligations via front end (surfaces rich ExprDiagnostics on failure)
    for ob_idx, ob in enumerate(evidence.get("obligations") or []):
        if isinstance(ob, dict) and isinstance(ob.get("expression"), str):
            res = _fe_normalize(
                ob["expression"],
                declared_symbols=all_declared,
                domain=domain_preview,
                si_dims=slot_dims_map,
                quantifier_mode=q_mode,
            )
            if res.ok and res.canonical is not None:
                if res.canonical != ob["expression"] or res.transforms:
                    _fe_record_transform(f"evidence.obligations[{ob_idx}].expression", ob["expression"], res.transforms)
                    ob["expression"] = res.canonical
                fe_binder_assumptions.extend(res.binder_assumptions)
            else:
                for ed in res.diagnostics:
                    fe_diags.append(BlueprintDiagnostic(
                        code=ed.code,
                        field=f"obligations[{ob_idx}].expression",
                        message=ed.message,
                        original=ed.original,
                        column=ed.column,
                        offending_token=ed.offending_token,
                        hint=ed.hint,
                        suggested_fix=ed.hint,
                    ))

    if fe_binder_assumptions:
        env_list = list(evidence.get("environmental_assumptions") or [])
        for ba in fe_binder_assumptions:
            if ba not in env_list:
                env_list.append(ba)
        evidence["environmental_assumptions"] = env_list

    if fe_diags:
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=fe_diags,
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    # Normalize pack syntax (equations assignment splitting, domain objects, etc.).
    pack["domain"] = {}
    try:
        pack = normalize_ihcg_evidence(pack)
    except ValueError as exc:
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=[BlueprintDiagnostic(
                code="NORMALIZATION_ERROR",
                field="ihcg_evidence",
                message=str(exc),
            )],
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )
    evidence = pack["evidence"]

    # Move non-executable prose items in environmental_assumptions to notes non-blockingly.
    notes = list(pack.get("notes") or [])
    clean_env: List[str] = []
    for expr in evidence.get("environmental_assumptions") or []:
        if not isinstance(expr, str):
            continue
        try:
            m._parse(expr)
        except (ValueError, SyntaxError):
            notes.append(expr)
        else:
            clean_env.append(expr)
    evidence["environmental_assumptions"] = clean_env
    if notes:
        pack["notes"] = list(dict.fromkeys(notes))

    # Any non-slot ALGEBRAIC equation in evidence.equations is also treated as a pure derived_expression!
    equations = evidence.get("equations") or {}
    if not isinstance(equations, dict):
        equations = {}
        evidence["equations"] = equations

    promoted_derived_to_inline: Dict[str, Dict[str, Any]] = {}
    for sym, spec in derived_exprs.items():
        if sym not in declared_slots and not spec.get("promote_to_slot"):
            promoted_derived_to_inline[sym] = spec
    for eq_name, eq_obj in list(equations.items()):
        if eq_name not in declared_slots and isinstance(eq_obj, dict):
            kind = eq_obj.get("kind", "ALGEBRAIC")
            expr = eq_obj.get("expression", "")
            if kind == "ALGEBRAIC" and isinstance(expr, str):
                try:
                    node = m._parse(expr)
                except (ValueError, SyntaxError):
                    node = None
                if node is not None and not isinstance(node, (ast.Compare, ast.BoolOp)):
                    promoted_derived_to_inline.setdefault(eq_name, {
                        "symbol": eq_name,
                        "expression": expr,
                        "si_unit": "",
                        "si_dim": list(eq_obj.get("si_dim") or [0] * 7),
                        "promote_to_slot": False,
                    })
                    equations.pop(eq_name, None)

    # Only expand pure derived_expressions that are actually referenced by equations, obligations, or assumptions
    # (never failing on unreferenced helper notes left on blueprint.derived_expressions).
    referenced_in_pack: Set[str] = set()
    for eq_obj in equations.values():
        if isinstance(eq_obj, dict) and isinstance(eq_obj.get("expression"), str):
            try:
                referenced_in_pack.update(m._variables(m._parse(eq_obj["expression"])))
            except (ValueError, SyntaxError):
                pass
    for ob in (evidence.get("obligations") or []):
        if isinstance(ob, dict) and isinstance(ob.get("expression"), str):
            try:
                referenced_in_pack.update(m._variables(m._parse(ob["expression"])))
            except (ValueError, SyntaxError):
                pass
    for expr in (evidence.get("environmental_assumptions") or []):
        if isinstance(expr, str):
            try:
                referenced_in_pack.update(m._variables(m._parse(expr)))
            except (ValueError, SyntaxError):
                pass

    # Transitive closure of referenced derived_expressions.
    needed_derived: Dict[str, Dict[str, Any]] = {}
    frontier = set(referenced_in_pack) & set(promoted_derived_to_inline.keys())
    while frontier:
        sym = frontier.pop()
        if sym in needed_derived:
            continue
        spec = promoted_derived_to_inline[sym]
        needed_derived[sym] = spec
        try:
            deps = m._variables(m._parse(spec["expression"]))
            frontier.update((deps & set(promoted_derived_to_inline.keys())) - set(needed_derived.keys()))
        except (ValueError, SyntaxError):
            pass

    # Topologically expand referenced pure derived_expressions and inline them into equations, obligations, and assumptions.
    for sym in sorted(needed_derived):
        if sym in derived_fe_errors:
            diagnostics.extend(derived_fe_errors[sym])
    expanded_derived, derived_diags = _resolve_derived_topological(
        {k: v for k, v in needed_derived.items() if k not in derived_fe_errors}
    )
    diagnostics.extend(derived_diags)
    if diagnostics:
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=diagnostics,
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    if expanded_derived:
        for eq_name, eq_obj in equations.items():
            if isinstance(eq_obj, dict) and isinstance(eq_obj.get("expression"), str):
                try:
                    eq_obj["expression"] = _substitute_ast_symbols(eq_obj["expression"], expanded_derived)
                except (ValueError, SyntaxError) as exc:
                    diagnostics.append(_make_parse_diagnostic(
                        f"equations.{eq_name}.expression",
                        eq_obj["expression"],
                        exc,
                    ))
        for ob_idx, ob in enumerate(evidence.get("obligations") or []):
            if isinstance(ob, dict) and isinstance(ob.get("expression"), str):
                try:
                    ob["expression"] = _substitute_ast_symbols(ob["expression"], expanded_derived)
                except (ValueError, SyntaxError) as exc:
                    diagnostics.append(_make_parse_diagnostic(
                        f"obligations[{ob_idx}].expression",
                        ob["expression"],
                        exc,
                    ))
        new_env = []
        for e_idx, expr in enumerate(evidence.get("environmental_assumptions") or []):
            try:
                new_env.append(_substitute_ast_symbols(expr, expanded_derived))
            except (ValueError, SyntaxError) as exc:
                diagnostics.append(_make_parse_diagnostic(
                    f"environmental_assumptions[{e_idx}]",
                    str(expr),
                    exc,
                ))
        evidence["environmental_assumptions"] = new_env

    # Collect all referenced symbols across equations, obligations, assumptions, mechanisms.
    def _gather_references() -> Tuple[Set[str], List[BlueprintDiagnostic]]:
        refs: Set[str] = set()
        errs: List[BlueprintDiagnostic] = []
        for mech in (evidence.get("mechanism_parameters") or []):
            if isinstance(mech, str):
                refs.add(mech)
        for s in slots:
            if s.get("stochastic_role"):
                refs.add(s["symbol"])
        for ob_idx, ob in enumerate(evidence.get("obligations") or []):
            if not isinstance(ob, dict) or not isinstance(ob.get("expression"), str):
                errs.append(BlueprintDiagnostic(
                    code="INVALID_OBLIGATION",
                    field=f"obligations[{ob_idx}]",
                    message="Each obligation must be an object with a formal string 'expression'.",
                ))
                continue
            try:
                refs.update(m._variables(m._parse(ob["expression"])))
            except (ValueError, SyntaxError) as exc:
                errs.append(_make_parse_diagnostic(
                    f"obligations[{ob_idx}].expression",
                    ob["expression"],
                    exc,
                ))
        for e_idx, expr in enumerate(evidence.get("environmental_assumptions") or []):
            try:
                refs.update(m._variables(m._parse(expr)))
            except (ValueError, SyntaxError) as exc:
                errs.append(_make_parse_diagnostic(
                    f"environmental_assumptions[{e_idx}]",
                    str(expr),
                    exc,
                ))
        for eq_name, eq_obj in equations.items():
            if not isinstance(eq_obj, dict) or not isinstance(eq_obj.get("expression"), str):
                errs.append(BlueprintDiagnostic(
                    code="INVALID_EQUATION",
                    field=f"equations.{eq_name}",
                    message="Equation must be an object with a formal string 'expression'.",
                ))
                continue
            try:
                refs.update(m._variables(m._parse(eq_obj["expression"])))
            except (ValueError, SyntaxError) as exc:
                errs.append(_make_parse_diagnostic(
                    f"equations.{eq_name}.expression",
                    eq_obj["expression"],
                    exc,
                ))
            if eq_name in declared_slots:
                refs.add(eq_name)
            if eq_obj.get("state_symbol"):
                refs.add(eq_obj["state_symbol"])
        return refs, errs

    references, parse_errs = _gather_references()
    diagnostics.extend(parse_errs)
    if diagnostics:
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=diagnostics,
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    # Preserve slots referenced by compiled contracts (state[i]).
    slot_by_idx = {s["slot_idx"]: s["symbol"] for s in slots if isinstance(s.get("slot_idx"), int)}
    for contract in (bp.get("contracts") or []):
        for inv in (contract.get("invariants") or []):
            assertion = inv.get("assertion") if isinstance(inv, dict) else getattr(inv, "assertion", "")
            for idx_str in re.findall(r"(?:state|buf\.data|data)\[(\d+)\]", str(assertion or "")):
                sym = slot_by_idx.get(int(idx_str))
                if sym in declared_slots:
                    references.add(sym)

    # 2. Bijective alias resolution for any unresolved symbols matching declared slots/constants.
    unresolved = references - declared_slots - set(constants.keys())
    known_for_alias = declared_slots | set(constants.keys())
    aliases = _find_bijective_aliases(unresolved, known_for_alias)
    if aliases:
        alias_ast = {src: ast.Name(id=tgt, ctx=ast.Load()) for src, tgt in aliases.items()}
        for eq_obj in equations.values():
            eq_obj["expression"] = _substitute_ast_symbols(eq_obj["expression"], alias_ast)
            if eq_obj.get("state_symbol") in aliases:
                eq_obj["state_symbol"] = aliases[eq_obj["state_symbol"]]
        for ob in (evidence.get("obligations") or []):
            ob["expression"] = _substitute_ast_symbols(ob["expression"], alias_ast)
        evidence["environmental_assumptions"] = [
            _substitute_ast_symbols(expr, alias_ast) for expr in (evidence.get("environmental_assumptions") or [])
        ]
        if evidence.get("mechanism_parameters"):
            evidence["mechanism_parameters"] = [aliases.get(p, p) for p in evidence["mechanism_parameters"]]
        references, _ = _gather_references()
        for contract in (bp.get("contracts") or []):
            for inv in (contract.get("invariants") or []):
                assertion = inv.get("assertion") if isinstance(inv, dict) else getattr(inv, "assertion", "")
                for idx_str in re.findall(r"(?:state|buf\.data|data)\[(\d+)\]", str(assertion or "")):
                    sym = slot_by_idx.get(int(idx_str))
                    if sym in declared_slots:
                        references.add(sym)

    # 3. Auto-populate unambiguous CODATA SI constants or module-level code_template constants.
    still_unresolved = references - declared_slots - set(constants.keys())
    for sym in sorted(still_unresolved):
        if sym in STANDARD_SI_CONSTANTS:
            val, unit, dims = STANDARD_SI_CONSTANTS[sym]
            constants[sym] = {
                "symbol": sym,
                "value": val,
                "si_unit": unit,
                "si_dim": list(dims),
                "source": "REFERENCE_REGISTRY",
            }
            assumptions_ledger.append({
                "kind": "SI_CONSTANT",
                "symbol_or_field": sym,
                "value_or_expr": f"{val} {unit}",
                "source": "REFERENCE_REGISTRY",
                "rationale": f"Auto-populated standard CODATA 2018 SI physical constant {sym}.",
            })
        elif sym in code_constants:
            val = code_constants[sym]
            constants[sym] = {
                "symbol": sym,
                "value": val,
                "si_unit": "",
                "si_dim": [0] * 7,
                "source": "LLM_DOMAIN_DEFAULT",
            }
            assumptions_ledger.append({
                "kind": "SI_CONSTANT",
                "symbol_or_field": sym,
                "value_or_expr": str(val),
                "source": "LLM_DOMAIN_DEFAULT",
                "rationale": f"Bound module-level constant {sym} from simulation code_template.",
            })
        elif sym == "pi":
            constants["pi"] = {
                "symbol": "pi",
                "value": math.pi,
                "si_unit": "",
                "si_dim": [0] * 7,
                "source": "REFERENCE_REGISTRY",
            }

    # 4. Bind referenced constants cleanly:
    #    - Dimensional constants -> promoted to locked CONTROL_PARAM slots (preserving exact si_dim)
    #      and appended to variable_map + contract initial_state so default_verification stays consistent.
    #    - Dimensionless constants not in typed_slots -> inlined cleanly as exact float literals
    #      so they never consume buffer slots or hit MAX_VARIABLES!
    inline_const_ast: Dict[str, ast.AST] = {}
    referenced_constants = sorted((references & set(constants.keys())) - declared_slots)
    for sym in referenced_constants:
        c_info = constants[sym]
        c_val = float(c_info["value"])
        c_dim = list(c_info.get("si_dim") or [0] * 7)
        is_dimensional = any(d != 0 for d in c_dim)
        if is_dimensional and len(slots) < m.MAX_VARIABLES:
            new_slot_idx = next_idx
            next_idx += 1
            new_slot = {
                "symbol": sym,
                "slot_idx": new_slot_idx,
                "si_dim": c_dim,
                "role": "CONTROL_PARAM",
                "semantic_tag": 0,
            }
            slots.append(new_slot)
            declared_slots.add(sym)
            variable_map[sym] = new_slot_idx
            if isinstance(bp.get("typed_slots"), list):
                bp["typed_slots"].append(dict(new_slot, value=c_val))
            slot_fixed_values[sym] = c_val
            evidence.setdefault("control_values", {})[sym] = c_val
            # Pad contract initial_state so buffer length covers the newly promoted constant slot.
            for contract in (bp.get("contracts") or []):
                if isinstance(contract, dict):
                    init = contract.get("initial_state")
                    if isinstance(init, list):
                        while len(init) < new_slot_idx:
                            init.append(0.0)
                        if len(init) == new_slot_idx:
                            init.append(c_val)
                    init_vals = contract.get("initial_values")
                    if isinstance(init_vals, dict):
                        init_vals.setdefault(sym, c_val)
        else:
            inline_const_ast[sym] = ast.Constant(value=c_val)

    if inline_const_ast:
        for eq_obj in equations.values():
            eq_obj["expression"] = _substitute_ast_symbols(eq_obj["expression"], inline_const_ast)
        for ob in (evidence.get("obligations") or []):
            ob["expression"] = _substitute_ast_symbols(ob["expression"], inline_const_ast)
        evidence["environmental_assumptions"] = [
            _substitute_ast_symbols(expr, inline_const_ast) for expr in (evidence.get("environmental_assumptions") or [])
        ]
        # Re-gather references now that dimensionless constants were inlined.
        references, _ = _gather_references()
        for contract in (bp.get("contracts") or []):
            for inv in (contract.get("invariants") or []):
                assertion = inv.get("assertion") if isinstance(inv, dict) else getattr(inv, "assertion", "")
                for idx_str in re.findall(r"(?:state|buf\.data|data)\[(\d+)\]", str(assertion or "")):
                    sym = slot_by_idx.get(int(idx_str))
                    if sym in declared_slots:
                        references.add(sym)

    # 5. Check for any remaining genuinely unresolved symbols.
    missing_symbols = sorted(references - declared_slots)
    if missing_symbols:
        for sym in missing_symbols:
            diagnostics.append(BlueprintDiagnostic(
                code="UNRESOLVED_SYMBOL",
                field="typed_slots|constants|derived_expressions",
                symbol=sym,
                message=f"Symbol {sym!r} is referenced in equations/obligations but is not declared in typed_slots, constants, or derived_expressions.",
                suggested_fix=f"Declare {sym!r} in 'constants' (if immutable), 'derived_expressions' (if algebraic helper), or 'typed_slots' (if mutable buffer slot).",
            ))
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=diagnostics,
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    # Prune unreferenced slots before MAX_VARIABLES check.
    selected_slots = [s for s in slots if s["symbol"] in references]
    if len(selected_slots) > m.MAX_VARIABLES:
        diagnostics.append(BlueprintDiagnostic(
            code="TOO_MANY_SLOTS",
            field="typed_slots",
            message=f"Model references {len(selected_slots)} slots, exceeding MAX_VARIABLES={m.MAX_VARIABLES}.",
            suggested_fix="Move immutable constants to 'constants' and intermediate algebraic terms to 'derived_expressions'.",
        ))
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=diagnostics,
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    # Normalize dummy sequential non-reserved semantic_tag IDs (1..250) to 0 BEFORE trajectory tag derivation
    # when all equations use generic semantic_tag=0. Reserved tags (251..254) are NEVER altered.
    eq_uses_nonzero_nonreserved = any(
        isinstance(eq, dict) and 1 <= int(eq.get("semantic_tag", 0) or 0) <= 250
        for eq in equations.values()
    )
    if not eq_uses_nonzero_nonreserved:
        for slot_obj in selected_slots:
            tag = int(slot_obj.get("semantic_tag", 0) or 0)
            if 1 <= tag <= 250:
                slot_obj["semantic_tag"] = 0
                if isinstance(bp.get("typed_slots"), list):
                    for ts in bp["typed_slots"]:
                        if isinstance(ts, dict) and ts.get("symbol") == slot_obj.get("symbol"):
                            ts["semantic_tag"] = 0

    # Derive trajectory reduction state_symbol and semantic_tag when unspecified.
    dynamic_names = {s["symbol"] for s in selected_slots if s.get("role") == "DYNAMIC_STATE"}
    for eq_name, eq_obj in equations.items():
        kind = eq_obj.get("kind")
        if kind in {"ODE_RHS", "FIXED_POINT_RESIDUAL", "FIXED_POINT_MAP"} and not eq_obj.get("state_symbol") and eq_name in dynamic_names:
            eq_obj["state_symbol"] = eq_name
        if kind in {"TRAJECTORY_MAX", "TRAJECTORY_MIN"}:
            if not eq_obj.get("state_symbol") and str(eq_obj.get("expression", "")).strip() in dynamic_names:
                eq_obj["state_symbol"] = str(eq_obj.get("expression", "")).strip()
            if eq_obj.get("state_symbol") in dynamic_names:
                target_slot = next((s for s in selected_slots if s["symbol"] == eq_name and s.get("role") == "DERIVED_OBSERVABLE"), None)
                if target_slot is not None and target_slot.get("semantic_tag", 0) == 0:
                    target_slot["semantic_tag"] = m.TRAJECTORY_MAXIMUM if kind == "TRAJECTORY_MAX" else m.TRAJECTORY_MINIMUM

    evidence["slots"] = selected_slots
    for slot_idx_pos, slot_obj in enumerate(selected_slots):
        try:
            m.SlotSpec.model_validate_json(json.dumps(slot_obj, allow_nan=False))
        except ValueError as exc:
            diagnostics.append(BlueprintDiagnostic(
                code="INVALID_SLOT_SPEC",
                field=f"typed_slots[{slot_idx_pos}]",
                symbol=str(slot_obj.get("symbol") or ""),
                message=str(exc),
                suggested_fix="Fix slot role (CONTROL_PARAM|DYNAMIC_STATE|DERIVED_OBSERVABLE|CONSERVATION_AUX) and 7-int si_dim.",
            ))
    if diagnostics:
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=diagnostics,
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )
    if "control_values" in evidence:
        evidence["control_values"] = {k: v for k, v in evidence["control_values"].items() if k in references}

    # 6. Assemble finite domains strictly from declared domain/slot bounds/constants (NEVER parsing user prose!).
    domain: Dict[str, Any] = {k: v for k, v in raw_domain.items() if k in references}
    for sym, bnd in slot_bounds.items():
        if sym in references and sym not in domain:
            domain[sym] = bnd
    pack["domain"] = domain
    try:
        pack = normalize_ihcg_evidence(pack)
    except ValueError as exc:
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=[BlueprintDiagnostic(code="INVALID_DOMAIN", field="ihcg_evidence.domain", message=str(exc))],
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )
    domain = pack["domain"]

    for sym in sorted(references):
        if sym in domain:
            continue
        slot = next(s for s in selected_slots if s["symbol"] == sym)
        if slot.get("role") == "CONTROL_PARAM" and sym not in domain:
            if sym in slot_fixed_values:
                v = slot_fixed_values[sym]
                domain[sym] = [v, v]
            elif sym in constants:
                v = float(constants[sym]["value"])
                domain[sym] = [v, v]

    # Derive domains for algebraic derived observables via interval arithmetic when omitted.
    definitions: Dict[str, ast.AST] = {}
    for eq_name, eq_obj in equations.items():
        if eq_name in references and eq_obj.get("kind", "ALGEBRAIC") == "ALGEBRAIC":
            node = m._parse(eq_obj["expression"])
            if not isinstance(node, (ast.Compare, ast.BoolOp)):
                definitions[eq_name] = node

    free_missing = sorted(references - set(definitions.keys()) - set(domain.keys()))
    if free_missing:
        for sym in free_missing:
            diagnostics.append(BlueprintDiagnostic(
                code="MISSING_DOMAIN",
                field=f"domain.{sym}",
                symbol=sym,
                message=f"Missing finite [lower, upper] operating domain for symbol {sym!r}.",
                suggested_fix=f"Add domain[{sym!r}] = [lo, hi] with finite ordered bounds.",
            ))
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=diagnostics,
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    pending = {name: node for name, node in definitions.items() if name not in domain}
    resolved = set(domain.keys())
    while pending:
        progressed = False
        for name, node in list(pending.items()):
            if not m._variables(node) <= resolved:
                continue
            try:
                enclosure = m._iv(node, {k: (float(v[0]), float(v[1])) for k, v in domain.items()})
                if not math.isfinite(enclosure.lo) or not math.isfinite(enclosure.hi):
                    raise ValueError("unbounded interval")
                lo_d, hi_d = float(enclosure.lo), float(enclosure.hi)
                scale = max(1e-30, abs(lo_d), abs(hi_d))
                if hi_d - lo_d <= 1e-6 * scale:
                    pad = 1e-4 * scale
                    lo_d, hi_d = lo_d - pad, hi_d + pad
                domain[name] = [lo_d, hi_d]
                pending.pop(name)
                resolved.add(name)
                progressed = True
            except (ValueError, KeyError, ZeroDivisionError, OverflowError) as exc:
                diagnostics.append(BlueprintDiagnostic(
                    code="DERIVED_DOMAIN_UNBOUNDED",
                    field=f"domain.{name}",
                    symbol=name,
                    message=f"Cannot derive finite domain for {name!r}: {exc}",
                    suggested_fix=f"Provide an explicit finite domain[{name!r}] = [lo, hi] and ensure denominators are bounded away from 0.",
                ))
                pending.pop(name)
        if not progressed:
            for name in sorted(set(pending) - set(domain)):
                diagnostics.append(BlueprintDiagnostic(
                    code="MISSING_DOMAIN",
                    field=f"domain.{name}",
                    symbol=name,
                    message=f"Missing finite domain for cyclic or unresolved symbol {name!r}.",
                ))
            break

    if diagnostics:
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=diagnostics,
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    # Attach non-blocking provenance spans and assumptions ledger.
    # NOTE entries live in bp["assumptions_ledger"] / preflight.assumptions_ledger, never in sealed IhcgGraph hashes.
    sealed_ledger = [e for e in assumptions_ledger if e.get("kind") != "NOTE"]
    if sealed_ledger:
        pack["evidence"]["assumptions_ledger"] = sealed_ledger
    pack = _anchor_evidence(question or bp.get("research_prompt") or "", pack)

    # 7. Compile candidate IhcgGraph (strict structural validation without prose blocking).
    compile_kwargs = {k: v for k, v in pack.items() if k not in {"extensions", "execution", "notes"}}
    # 7. Compile and validate IHCG structure.
    # ARCH-002 (BUG-026): Pin user-fixed prompt literals (e.g. T_hot=600 K, gamma=0.1/day)
    # to exact point domains so the LLM can never silently widen them.
    from .claim_ledger import extract_claim_ledger, pin_user_literals_in_domain

    prompt_for_ledger = question or bp.get("research_prompt") or ""
    if prompt_for_ledger and isinstance(compile_kwargs.get("domain"), dict):
        ledger = extract_claim_ledger(prompt_for_ledger)
        if ledger.pinned_literals:
            # Never clobber an explicit non-degenerate sweep/boundary domain when quantifier_mode == 'BOUNDARY'
            # or when the symbol is a swept mechanism parameter.
            if str(evidence.get("quantifier_mode") or "").upper() != "BOUNDARY":
                pinned_dom = pin_user_literals_in_domain(ledger, compile_kwargs["domain"])
                compile_kwargs["domain"] = pinned_dom
                pack["domain"] = pinned_dom

    try:
        graph = m.IhcgCompiler().compile_from_spec(
            compile_kwargs["evidence"],
            compile_kwargs["domain"],
            compile_kwargs["archetype"],
            prompt=question or bp.get("research_prompt") or "",
        )
    except ValueError as exc:
        diagnostics.append(BlueprintDiagnostic(
            code="IHCG_STRUCTURAL_ERROR",
            field="ihcg_evidence",
            message=str(exc),
        ))
        return BlueprintPreflightResult(
            ok=False,
            diagnostics=diagnostics,
            compiled_blueprint=bp,
            compiled_ihcg_pack=pack,
            assumptions_ledger=assumptions_ledger,
        )

    # 8. Initial-state consistency check with declared conservation laws.
    pool_diags = _check_initial_pool_conservation(bp, graph, constants)
    diagnostics.extend(pool_diags)

    # 8b. Non-degenerate mechanism parameter domain check (R2-BUG-003 / BUG-024):
    # Declared mechanism_parameters must have a non-degenerate sweep interval [lo, hi] with hi > lo;
    # fixed non-sweep physical constants (not in mechanism_parameters) remain valid with [v, v].
    for mp in graph.mechanism_parameters:
        lo_m, hi_m = graph.omega[mp]
        if not (hi_m > lo_m):
            diagnostics.append(BlueprintDiagnostic(
                code="DEGENERATE_MECHANISM_DOMAIN",
                field=f"domain.{mp}",
                symbol=mp,
                message=f"Mechanism parameter {mp!r} has a zero-width point domain [{lo_m}, {hi_m}].",
                suggested_fix=(
                    f"Give mechanism_parameter {mp!r} a non-degenerate finite interval [lo, hi] with lo < hi "
                    f"(e.g. [0.0, {max(1.0, abs(hi_m) * 1.5):.6g}]) if it is a swept mechanism parameter, "
                    f"or remove {mp!r} from evidence.mechanism_parameters if it is a fixed physical constant."
                ),
            ))

    # 9. Fast Z3 non-vacuity / non-tautology sanity check.
    #    IMPORTANT: Genuine INFEASIBLE / INFEASIBLE_UNDER_MODEL claims (e.g. Carnot eta >= 0.65)
    #    MUST be allowed to seal so they route to VERIFIED_FALSIFICATION!
    if run_z3_preflight and not diagnostics:
        vacuity = m.classify_contract_vacuity(graph)
        if vacuity.classification in {"VACUOUS", "TAUTOLOGICAL"}:
            taut_tag = "TAUTOLOGICAL_TARGET (TRIVIAL_DEDUCIBLE_WITHOUT_MODEL): " if vacuity.classification == "TAUTOLOGICAL" else ""
            diagnostics.append(BlueprintDiagnostic(
                code=f"PREFLIGHT_{vacuity.classification}",
                field="ihcg_evidence.evidence.obligations",
                message=(
                    f"{taut_tag}Candidate specification is {vacuity.classification} before sealing "
                    f"(query_statuses={dict((k, v.status) for k, v in vacuity.queries.items())})."
                ),
                suggested_fix="Widen DERIVED_OBSERVABLE domain bounds and ensure environmental_assumptions do not already imply or contradict the domain.",
            ))
        elif vacuity.redundant_conjuncts:
            # BUG-025: Deterministically prune strictly redundant conjuncts when at least
            # one non-redundant HYPOTHESIS_TARGET remains, rather than aborting planning.
            redundant_ids = set(str(r) for r in vacuity.redundant_conjuncts)
            raw_obs = list(evidence.get("obligations") or [])
            pruned_obs = [
                ob for idx_o, ob in enumerate(raw_obs)
                if not (
                    isinstance(ob, dict)
                    and ob.get("obligation_class") == "HYPOTHESIS_TARGET"
                    and (str(ob.get("id", "")) in redundant_ids or str(idx_o) in redundant_ids)
                )
            ]
            has_hyp = any(
                isinstance(ob, dict) and ob.get("obligation_class") == "HYPOTHESIS_TARGET"
                for ob in pruned_obs
            )
            if has_hyp and len(pruned_obs) < len(raw_obs):
                evidence["obligations"] = pruned_obs
                compile_kwargs["evidence"] = evidence
                graph = m.IhcgCompiler().compile_from_spec(
                    compile_kwargs["evidence"],
                    compile_kwargs["domain"],
                    compile_kwargs["archetype"],
                    prompt=question or bp.get("research_prompt") or "",
                )
            else:
                diagnostics.append(BlueprintDiagnostic(
                    code="PREFLIGHT_REDUNDANT_CONJUNCTS",
                    field="ihcg_evidence.evidence.obligations",
                    message=f"Redundant conjuncts detected in obligations: {vacuity.redundant_conjuncts}.",
                    suggested_fix="Remove redundant HYPOTHESIS_TARGET clauses already implied by domain/environment or other obligations.",
                ))

    bp["variable_map"] = variable_map
    bp["assumptions_ledger"] = assumptions_ledger
    bp["ihcg_evidence"] = pack

    return BlueprintPreflightResult(
        ok=not diagnostics,
        diagnostics=diagnostics,
        compiled_blueprint=bp,
        compiled_ihcg_pack=pack,
        assumptions_ledger=assumptions_ledger,
        compiled_graph=graph if not diagnostics else None,
    )

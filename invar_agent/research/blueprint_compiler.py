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
        return out


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
        return "; ".join(f"[{d.code}] {d.field}: {d.message}" for d in self.diagnostics)


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
            diagnostics.append(BlueprintDiagnostic(
                code="INVALID_DERIVED_EXPRESSION",
                field=f"derived_expressions.{sym}",
                symbol=sym,
                message=str(exc),
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
    evidence = pack["evidence"]
    evidence.pop("constants", None)
    evidence.pop("derived_expressions", None)
    evidence.pop("assumptions_ledger", None)

    # 1. Assemble mutable typed_slots namespace.
    raw_slots = bp.get("typed_slots") or evidence.get("slots") or []
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

    # Normalize pack syntax (math.exp -> exp, ^ -> **, lower/upper -> [lo, hi], etc.).
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
    expanded_derived, derived_diags = _resolve_derived_topological(needed_derived)
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
                    diagnostics.append(BlueprintDiagnostic(
                        code="INVALID_EXPRESSION",
                        field=f"equations.{eq_name}",
                        message=str(exc),
                    ))
        for ob_idx, ob in enumerate(evidence.get("obligations") or []):
            if isinstance(ob, dict) and isinstance(ob.get("expression"), str):
                try:
                    ob["expression"] = _substitute_ast_symbols(ob["expression"], expanded_derived)
                except (ValueError, SyntaxError) as exc:
                    diagnostics.append(BlueprintDiagnostic(
                        code="INVALID_EXPRESSION",
                        field=f"obligations[{ob_idx}].expression",
                        message=str(exc),
                    ))
        new_env = []
        for e_idx, expr in enumerate(evidence.get("environmental_assumptions") or []):
            try:
                new_env.append(_substitute_ast_symbols(expr, expanded_derived))
            except (ValueError, SyntaxError) as exc:
                diagnostics.append(BlueprintDiagnostic(
                    code="INVALID_EXPRESSION",
                    field=f"environmental_assumptions[{e_idx}]",
                    message=str(exc),
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
                errs.append(BlueprintDiagnostic(
                    code="INVALID_EXPRESSION",
                    field=f"obligations[{ob_idx}].expression",
                    message=str(exc),
                ))
        for e_idx, expr in enumerate(evidence.get("environmental_assumptions") or []):
            try:
                refs.update(m._variables(m._parse(expr)))
            except (ValueError, SyntaxError) as exc:
                errs.append(BlueprintDiagnostic(
                    code="INVALID_EXPRESSION",
                    field=f"environmental_assumptions[{e_idx}]",
                    message=str(exc),
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
                errs.append(BlueprintDiagnostic(
                    code="INVALID_EXPRESSION",
                    field=f"equations.{eq_name}.expression",
                    message=str(exc),
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
        if slot.get("role") == "CONTROL_PARAM":
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
    if assumptions_ledger:
        pack["evidence"]["assumptions_ledger"] = assumptions_ledger
    pack = _anchor_evidence(question or bp.get("research_prompt") or "", pack)

    # 7. Compile candidate IhcgGraph (strict structural validation without prose blocking).
    compile_kwargs = {k: v for k, v in pack.items() if k not in {"extensions", "execution", "notes"}}
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

    # 8b. Non-degenerate mechanism parameter domain check (catch point-domain mechanisms before sealing).
    for mp in graph.mechanism_parameters:
        lo_m, hi_m = graph.omega[mp]
        if not (hi_m > lo_m):
            diagnostics.append(BlueprintDiagnostic(
                code="DEGENERATE_MECHANISM_DOMAIN",
                field=f"domain.{mp}",
                symbol=mp,
                message=f"Mechanism parameter {mp!r} has a zero-width point domain [{lo_m}, {hi_m}].",
                suggested_fix=f"Give mechanism_parameter {mp!r} a non-degenerate finite interval [lo, hi] with lo < hi (e.g. [0.0, {max(1.0, abs(hi_m) * 1.5):.6g}]) so sensitivity and null-baseline separation are well-defined.",
            ))

    # 9. Fast Z3 non-vacuity / non-tautology sanity check.
    #    IMPORTANT: Genuine INFEASIBLE / INFEASIBLE_UNDER_MODEL claims (e.g. Carnot eta >= 0.65)
    #    MUST be allowed to seal so they route to VERIFIED_FALSIFICATION!
    if run_z3_preflight and not diagnostics:
        vacuity = m.classify_contract_vacuity(graph)
        if vacuity.classification in {"VACUOUS", "TAUTOLOGICAL"}:
            diagnostics.append(BlueprintDiagnostic(
                code=f"PREFLIGHT_{vacuity.classification}",
                field="ihcg_evidence.evidence.obligations",
                message=(
                    f"Candidate specification is {vacuity.classification} before sealing "
                    f"(query_statuses={dict((k, v.status) for k, v in vacuity.queries.items())})."
                ),
                suggested_fix="Widen DERIVED_OBSERVABLE domain bounds and ensure environmental_assumptions do not already imply or contradict the domain.",
            ))
        elif vacuity.redundant_conjuncts:
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

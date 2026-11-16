"""Tier-0 Deterministic UAST/HIN Healer (Reflex_Heal 2.0) & Localized Single-Hole Re-Prompter."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .csel import (
    CselSyntaxError,
    collect_csel_symbols,
    csel_ast_to_str,
    parse_csel_to_ast,
)
from .schemas import (
    CounterexampleDiagnosticCard,
    PhysicalQuantity,
    ResearchComputeUAST,
)

try:
    from invar_agent._native import (
        csel_egraph_optimize,
        desugar_math_to_csel,
        resolve_symbol_fuzzy,
    )
except Exception:  # pragma: no cover
    csel_egraph_optimize = None
    desugar_math_to_csel = None
    resolve_symbol_fuzzy = None

# CODATA 2022 / Standard Physical Constants (SI units)
CODATA_CONSTANTS: Dict[str, Tuple[float, List[int], str]] = {
    "g": (9.80665, [0, 1, -2, 0, 0, 0, 0], "m/s^2"),
    "g_E": (9.80665, [0, 1, -2, 0, 0, 0, 0], "m/s^2"),
    "g_0": (9.80665, [0, 1, -2, 0, 0, 0, 0], "m/s^2"),
    "c": (299792458.0, [0, 1, -1, 0, 0, 0, 0], "m/s"),
    "h": (6.62607015e-34, [1, 2, -1, 0, 0, 0, 0], "J*s"),
    "hbar": (1.054571817e-34, [1, 2, -1, 0, 0, 0, 0], "J*s"),
    "k_B": (1.380649e-23, [1, 2, -2, 0, -1, 0, 0], "J/K"),
    "kB": (1.380649e-23, [1, 2, -2, 0, -1, 0, 0], "J/K"),
    "R": (8.314462618, [1, 2, -2, 0, -1, -1, 0], "J/(mol*K)"),
    "N_A": (6.02214076e23, [0, 0, 0, 0, 0, -1, 0], "1/mol"),
    "G": (6.67430e-11, [-1, 3, -2, 0, 0, 0, 0], "m^3/(kg*s^2)"),
    "epsilon_0": (8.8541878128e-12, [-1, -3, 4, 2, 0, 0, 0], "F/m"),
    "mu_0": (1.25663706212e-6, [1, 1, -2, -2, 0, 0, 0], "N/A^2"),
    "sigma_sb": (5.670374419e-8, [1, 0, -3, 0, -4, 0, 0], "W/(m^2*K^4)"),
    "m_e": (9.1093837015e-31, [1, 0, 0, 0, 0, 0, 0], "kg"),
    "m_p": (1.67262192369e-27, [1, 0, 0, 0, 0, 0, 0], "kg"),
    "e_charge": (1.602176634e-19, [0, 0, 1, 1, 0, 0, 0], "C"),
    "atm": (101325.0, [1, -1, -2, 0, 0, 0, 0], "Pa"),
    "T_0": (273.15, [0, 0, 0, 0, 1, 0, 0], "K"),
}

_BUILTIN_FREE_SYMBOLS = {"t", "pi", "e"}


def _py_desugar_fallback(raw: str) -> str:
    s = raw.strip().strip("$").strip()
    s = s.replace("\\left(", "(").replace("\\right)", ")")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("**", "^")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"safe_div(\1, \2)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"safe_sqrt(\1)", s)
    s = re.sub(r"\b(?:np|numpy|math|jnp|torch)\.", "", s)
    return s


def _strip_lhs_assignment(expr: str) -> str:
    """If an LLM wrote `dx/dt = -k*x` or `dx_dt = -k*x`, keep only the RHS."""
    if "=" in expr and "==" not in expr and "<=" not in expr and ">=" not in expr:
        lhs, rhs = expr.split("=", 1)
        if len(lhs.strip()) <= 32:
            return rhs.strip()
    return expr


def _replace_symbol_in_ast(ast_node: Dict[str, Any], old_sym: str, new_sym: str) -> None:
    kind = ast_node.get("kind")
    if kind == "symbol" and ast_node.get("name") == old_sym:
        ast_node["name"] = new_sym
    elif kind == "unary":
        _replace_symbol_in_ast(ast_node["arg"], old_sym, new_sym)
    elif kind == "binary":
        _replace_symbol_in_ast(ast_node["left"], old_sym, new_sym)
        _replace_symbol_in_ast(ast_node["right"], old_sym, new_sym)
    elif kind == "call":
        for arg in ast_node.get("args", []):
            _replace_symbol_in_ast(arg, old_sym, new_sym)


def _desugar_and_parse(raw_expr: str) -> Tuple[Dict[str, Any], List[str]]:
    logs: List[str] = []
    cleaned = _strip_lhs_assignment(raw_expr)
    if cleaned != raw_expr:
        logs.append("stripped_lhs_assignment")
    if desugar_math_to_csel is not None:
        desugared = desugar_math_to_csel(cleaned)
    else:
        desugared = _py_desugar_fallback(cleaned)
    if desugared != cleaned:
        logs.append("desugared_math_to_csel")
    # Handle any nested \frac or residual whitespace-separated implicit multiplication
    # after LaTeX expansion (e.g. `safe_div(-k x, m)` -> `safe_div(-k * x, m)`)
    desugared = re.sub(r"([A-Za-z0-9_\)])\s+([A-Za-z_\(])", r"\1 * \2", desugared)
    ast_node = parse_csel_to_ast(desugared)
    return ast_node, logs


def tier0_heal_uast(
    uast: ResearchComputeUAST,
    *,
    extracted_constants: Optional[Dict[str, float]] = None,
    auto_inject_unknown: bool = False,
) -> Tuple[ResearchComputeUAST, List[str]]:
    """Tier-0 Deterministic UAST/HIN Healer (0 LLM tokens).

    1. Math/LaTeX desugaring -> valid CSEL AST
    2. Symbol & Dimensional Auto-Resolution (fuzzy Levenshtein, snake_case,
       extracted_constants, CODATA physical constants)
    3. Rust `egg` E-Graph equality saturation (log1p, expm1, rad-diff, IMEX split)
    """
    healed = copy.deepcopy(uast)
    actions: List[str] = []
    ext_consts = extracted_constants or {}

    def _heal_expr_map(expr_map: Dict[str, str], label: str) -> None:
        for key in list(expr_map.keys()):
            raw = expr_map[key]
            ast_node, logs = _desugar_and_parse(raw)
            for tag in logs:
                actions.append(f"{label}.{key}:{tag}")

            # Symbol resolution
            used_syms = collect_csel_symbols(ast_node)
            ontology_keys = list(healed.ontology.quantities.keys())
            for sym in sorted(used_syms):
                if sym in healed.ontology.quantities or sym in _BUILTIN_FREE_SYMBOLS:
                    continue
                # 1. Fuzzy match against declared ontology
                matched = (
                    resolve_symbol_fuzzy(sym, ontology_keys)
                    if resolve_symbol_fuzzy is not None
                    else None
                )
                if matched is not None:
                    _replace_symbol_in_ast(ast_node, sym, matched)
                    actions.append(f"{label}.{key}:resolved_symbol({sym}->{matched})")
                    continue
                # 2. Literature extracted_constants
                if sym in ext_consts:
                    healed.ontology.quantities[sym] = PhysicalQuantity(
                        symbol=sym,
                        default_value=float(ext_consts[sym]),
                        initial_value=float(ext_consts[sym]),
                        dimension_unspecified=True,
                        description="Auto-injected from literature QuantitativeInsightExtractor",
                    )
                    ontology_keys.append(sym)
                    actions.append(f"{label}.{key}:injected_literature_const({sym}={ext_consts[sym]})")
                    continue
                # 3. CODATA constants
                if sym in CODATA_CONSTANTS:
                    val, dim, unit = CODATA_CONSTANTS[sym]
                    healed.ontology.quantities[sym] = PhysicalQuantity(
                        symbol=sym,
                        si_dimension=dim,
                        default_value=val,
                        initial_value=val,
                        unit=unit,
                        description="Auto-injected CODATA physical constant",
                    )
                    ontology_keys.append(sym)
                    actions.append(f"{label}.{key}:injected_codata({sym}={val})")
                    continue
                # 4. Optional fallback parameter injection
                if auto_inject_unknown:
                    healed.ontology.quantities[sym] = PhysicalQuantity(
                        symbol=sym,
                        default_value=1.0,
                        initial_value=1.0,
                        dimension_unspecified=True,
                        description="Auto-injected default parameter",
                    )
                    ontology_keys.append(sym)
                    actions.append(f"{label}.{key}:injected_default_param({sym}=1.0)")

            # 3. E-Graph equality saturation
            if csel_egraph_optimize is not None:
                opt_json = csel_egraph_optimize(json.dumps(ast_node))
                opt_data = json.loads(opt_json)
                ast_node = opt_data["expr"]
                if opt_data.get("has_imex_split"):
                    actions.append(f"{label}.{key}:egraph_imex_split")

            expr_map[key] = csel_ast_to_str(ast_node)

    _heal_expr_map(healed.governance.equations, "eq")
    _heal_expr_map(healed.governance.observables, "obs")
    return healed, actions


def _strip_json_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    elif start != -1:
        text = text[start:]
    # Strip // comments outside strings (simple line-end comments)
    text = re.sub(r'(?m)^\s*//.*$', "", text)
    # Quote bare -inf, +inf, inf, Infinity, -Infinity, NaN values
    text = re.sub(r'(?<=[:\[,\s])(-?inf(?:inity)?|\+inf(?:inity)?|NaN)\b', r'"\1"', text, flags=re.IGNORECASE)
    # Insert missing commas between `}` and `{` across newlines (never inside single-line LaTeX strings)
    text = re.sub(r'\}\s*\n\s*\{', '},\n{', text)
    text = re.sub(r'\]\s*\n\s*"', '],\n"', text)
    text = re.sub(r'\}\s*\n\s*"', '},\n"', text)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _loads_repaired_json(raw_text: str) -> Any:
    cleaned = _strip_json_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt truncated-JSON bracket balancing
        stack: List[str] = []
        in_str = False
        esc = False
        for ch in cleaned:
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]" and stack:
                    stack.pop()
        if in_str:
            cleaned += '"'
        cleaned = re.sub(r",\s*$", "", cleaned.rstrip())
        cleaned += "".join(reversed(stack))
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return json.loads(cleaned)


def coerce_llm_json_to_uast(
    raw_text: str,
    *,
    default_title: str = "Verified UAST Research Model",
    default_hypothesis: str = "Falsifiable dynamical system model.",
) -> ResearchComputeUAST:
    """Deterministically coerce flexible LLM JSON output into a validated ResearchComputeUAST."""
    data = _loads_repaired_json(raw_text)
    if not isinstance(data, dict):
        raise ValueError("LLM UAST output must be a JSON object")

    # Normalize top-level variations
    if "uast" in data and isinstance(data["uast"], dict):
        data = data["uast"]

    if "ontology" not in data:
        quantities: Dict[str, Any] = {}
        for k in ("state_variables", "parameters", "constants", "quantities"):
            if isinstance(data.get(k), dict):
                for sym, val in data[k].items():
                    if isinstance(val, dict):
                        quantities[sym] = {
                            **val,
                            "symbol": sym,
                            "is_state_variable": (k == "state_variables") or bool(val.get("is_state_variable", False)),
                        }
                    elif isinstance(val, (int, float)):
                        quantities[sym] = {
                            "symbol": sym,
                            "default_value": float(val),
                            "initial_value": float(val),
                            "is_state_variable": (k == "state_variables"),
                            "dimension_unspecified": True,
                        }
        data["ontology"] = {"quantities": quantities}

    if "governance" not in data:
        eqs = data.get("equations") or data.get("rhs") or {}
        obs = data.get("observables") or {}
        data["governance"] = {"equations": eqs, "observables": obs}

    if "invariants" not in data or not isinstance(data["invariants"], dict):
        data["invariants"] = {}

    data.setdefault("title", default_title)
    data.setdefault("hypothesis", default_hypothesis)
    return ResearchComputeUAST.model_validate(data)


def build_counterexample_diagnostic_card(
    uast: ResearchComputeUAST,
    target_equation_name: str,
    *,
    hin_report: Optional[Dict[str, Any]] = None,
    adjoint_report: Optional[Dict[str, Any]] = None,
    smt_assertion_failed: str = "",
    counterexample_vector: Optional[Dict[str, float]] = None,
) -> CounterexampleDiagnosticCard:
    """Synthesize a structured CounterexampleDiagnosticCard isolating the single failing node f_k."""
    failed_expr = uast.governance.equations.get(target_equation_name, "")
    dim_residual = ""
    if hin_report:
        for eq_rep in hin_report.get("equations", []):
            if eq_rep.get("equation_name") == target_equation_name:
                if eq_rep.get("error_message"):
                    dim_residual = eq_rep["error_message"]
                elif eq_rep.get("dimensional_warnings"):
                    dim_residual = "; ".join(eq_rep["dimensional_warnings"])

    target_si = [0, 0, 0, 0, 0, 0, 0]
    if target_equation_name in uast.ontology.quantities:
        q_dim = uast.ontology.quantities[target_equation_name].si_dimension
        # df_k / dt has dimension [x_k] / [T]
        target_si = [q_dim[0], q_dim[1], q_dim[2] - 1, q_dim[3], q_dim[4], q_dim[5], q_dim[6]]

    return CounterexampleDiagnosticCard(
        target_equation_name=target_equation_name,
        failed_expression=failed_expr,
        counterexample_vector=counterexample_vector or {
            k: q.initial_value for k, q in uast.ontology.quantities.items() if q.is_state_variable
        },
        sub_expression_trace=[f"{target_equation_name}: {failed_expr}"],
        dimension_residual=dim_residual or "None",
        smt_assertion_failed=smt_assertion_failed or "None",
        jacobian_spectrum={
            "spectral_radius": (adjoint_report or {}).get("spectral_radius", 0.0),
            "stiffness_ratio": (adjoint_report or {}).get("stiffness_ratio", 1.0),
        },
        available_ontology_keys=sorted(uast.ontology.quantities.keys()),
        target_si_vector=target_si,
    )


def format_single_hole_repair_prompt(card: CounterexampleDiagnosticCard) -> str:
    """Render the single-hole constrained repair prompt from invar_agent/prompts/uast_repair.jinja."""
    tpl_path = Path(__file__).resolve().parent.parent / "prompts" / "uast_repair.jinja"
    if tpl_path.exists():
        text = tpl_path.read_text(encoding="utf-8")
        replacements = {
            "{{ target_equation_name }}": card.target_equation_name,
            "{{ failed_expression }}": card.failed_expression,
            "{{ counterexample_vector }}": json.dumps(card.counterexample_vector),
            "{{ sub_expression_trace }}": json.dumps(card.sub_expression_trace),
            "{{ dimension_residual }}": card.dimension_residual,
            "{{ smt_assertion_failed }}": card.smt_assertion_failed,
            "{{ jacobian_spectrum }}": json.dumps(card.jacobian_spectrum),
            "{{ available_ontology_keys }}": json.dumps(card.available_ontology_keys),
            "{{ target_si_vector }}": json.dumps(card.target_si_vector),
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text
    return (
        f"[SYSTEM: TIER-1 REPAIR] Repair node {card.target_equation_name}: "
        f"{card.failed_expression}. Witness: {card.counterexample_vector}. "
        f"Allowed keys: {card.available_ontology_keys}. "
        f'OUTPUT FORMAT: {{"target_node_id": "{card.target_equation_name}", "new_csel_expression": "..."}}'
    )

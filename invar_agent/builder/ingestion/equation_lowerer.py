"""Deterministic SI, CSEL, HIN and tracked SMT gates."""
from __future__ import annotations

import ast
import json
import math
import re

import sympy as sp
import z3
from sympy.parsing.latex import parse_latex

from invar_agent.latex_normalizer import canonicalize_math_content, repair_json_control_escapes
from invar_agent.research.extractor import normalize_to_si, lookup_reference_registry

from .archive_guard import IngestionError
from .parsers import NUMBER
from .schemas import (CanonicalResearchIR, DisambiguationFlag, DisambiguationTriggerKind,
                      EquationRole, ExtractedConstant, ExtractedEquation, RawConstant,
                      RawEquation, SIDimensionVectorModel, VerificationStatus)


class FormalGateError(IngestionError):
    """A deterministic rejection suitable for an HTTP 422 response."""


def _native():
    try:
        from invar_agent import _native as native
    except ImportError:
        from invar_agent import _fallback as native
    return native


def unit_dimension(unit: str) -> SIDimensionVectorModel:
    aliases = {"F": "s^4*A^2/(kg*m^2)", "S": "s^3*A^2/(kg*m^2)", "Pa": "kg/(m*s^2)", "J": "kg*m^2/s^2", "W": "kg*m^2/s^3",
               "T": "kg/(s^2*A)", "Hz": "1/s", "N": "kg*m/s^2", "C": "A*s",
               "V": "kg*m^2/(s^3*A)", "L": "m^3", "rad": "1", "Morgan": "1",
               "fraction": "1", "bit": "1", "bp": "1", "kg/kg": "1"}
    bases = {name: tuple(int(i == index) for i in range(7))
             for index, name in enumerate(("m", "kg", "s", "A", "K", "mol", "cd"))}
    for name, expanded in aliases.items():
        unit = re.sub(rf"\b{re.escape(name)}\b", f"({expanded})", unit)

    def dimension(node):
        if isinstance(node, ast.Constant) and node.value == 1:
            return (0,) * 7
        if isinstance(node, ast.Name) and node.id in bases:
            return bases[node.id]
        if isinstance(node, ast.BinOp):
            left = dimension(node.left)
            if isinstance(node.op, ast.Pow):
                exponent = ast.literal_eval(node.right)
                if not isinstance(exponent, int) or abs(exponent) > 127:
                    raise ValueError("Invalid SI exponent")
                return tuple(v * exponent for v in left)
            right = dimension(node.right)
            sign = -1 if isinstance(node.op, ast.Div) else 1
            if not isinstance(node.op, (ast.Mult, ast.Div)):
                raise ValueError("Invalid SI operator")
            return tuple(a + sign * b for a, b in zip(left, right))
        raise ValueError("Unknown SI unit")
    try:
        powers = dimension(ast.parse((unit or "1").replace("^", "**"), mode="eval").body)
        if any(abs(v) > 127 for v in powers):
            raise ValueError("SI exponent overflow")
    except (ValueError, SyntaxError, TypeError) as exc:
        raise FormalGateError("HIN_UNKNOWN_UNIT", unit) from exc
    return SIDimensionVectorModel(exponents=powers)


def normalize_constant(raw: RawConstant, constant_id: str) -> ExtractedConstant:
    unit = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", raw.unit.strip())
    unit = unit.replace("{", "").replace("}", "").replace("\\,", "") or "1"
    value, canonical = normalize_to_si(float(raw.value), unit)
    if not math.isfinite(value):
        raise FormalGateError("HIN_NONFINITE_LITERAL", raw.symbol)
    dimension = unit_dimension(canonical or "1")
    dimension.canonical_unit = canonical or "1"
    return ExtractedConstant(constant_id=constant_id, symbol=raw.symbol, name=raw.symbol,
        raw_value=raw.value, raw_unit=raw.unit, si_value=value, dimension=dimension, provenance=raw.provenance)


def clean_math(text: str) -> str:
    text = canonicalize_math_content(repair_json_control_escapes(text)).strip().strip("$")
    text = re.sub(r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text).replace("&", "")
    return text.replace("\\leq", "<=").replace("\\geq", ">=").replace("\\le", "<=").replace("\\ge", ">=").replace("\\equiv", "=").strip()


def classify_equation(text: str) -> EquationRole:
    if re.search(r"\\dot|\\partial_t|Derivative\(|\\frac\{d|\bd\w+/dt\b|_\{?n\s*\+\s*1", text):
        return EquationRole.EVOLUTION_LAW
    if re.search(r"<=|>=|<|>|==|\\sum|\\equiv", text):
        return EquationRole.INVARIANT_CONTRACT
    if re.search(r"\([^,]+,\s*t\)|partial.?Omega", text):
        return EquationRole.BOUNDARY_CONDITION
    if re.match(r"\s*\w+\((?:0|t_?0)\)\s*=|\s*\w+_0\s*=", text):
        return EquationRole.INITIAL_CONDITION
    if re.fullmatch(rf"\s*[A-Za-z_]\w*\s*=\s*{NUMBER}\s*", text):
        return EquationRole.PARAMETER_TABLE
    return EquationRole.ALGEBRAIC_CLOSURE


def parse_expression(text: str):
    """Parse ASCII through a whitelist, never sympify/eval uploaded expressions."""
    if len(text) > 8192:
        raise ValueError("Equation size limit exceeded")
    text = text.replace("\\cdot", "*")
    latex_text = text

    def latex_expression(source):
        result = parse_latex(source, backend="lark")
        if not isinstance(result, sp.Basic):
            raise ValueError("Ambiguous LaTeX expression")
        return result.xreplace({symbol:sp.Symbol(re.sub(r'([A-Za-z]\w*)_\{([A-Za-z0-9_]+)\}',r'\1_\2',str(symbol)))
                                for symbol in result.free_symbols})

    ascii_text = re.sub(r'([A-Za-z]\w*)_\{([A-Za-z0-9_]+)\}', r'\1_\2', text)
    ascii_text = re.sub(r"\\(exp|log|sin|cos)\b", r"\1", ascii_text)
    if "\\" in ascii_text or "{" in ascii_text:
        return latex_expression(latex_text)
    text = ascii_text
    functions = {"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log,
                 "sqrt": sp.sqrt, "abs": sp.Abs}

    def convert(node):
        if isinstance(node, ast.Name):
            return {"pi": sp.pi, "e": sp.E}.get(node.id, sp.Symbol(node.id))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if not math.isfinite(node.value):
                raise ValueError("Nonfinite literal")
            return sp.Rational(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return convert(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add): return sp.Add(left, right, evaluate=False)
            if isinstance(node.op, ast.Sub): return sp.Add(left, -right, evaluate=False)
            if isinstance(node.op, ast.Mult): return sp.Mul(left, right, evaluate=False)
            if isinstance(node.op, ast.Div): return sp.Mul(left, sp.Pow(right, -1, evaluate=False), evaluate=False)
            if isinstance(node.op, ast.Pow): return sp.Pow(left, right, evaluate=False)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and not node.keywords:
            return functions[node.func.id](*(convert(arg) for arg in node.args), evaluate=False)
        raise ValueError("Unsupported expression")
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except SyntaxError:
        return latex_expression(latex_text)
    return convert(tree.body)


def equation_parts(text: str):
    parts = re.split(r"(<=|>=|==|=|<|>)", text, maxsplit=1)
    if len(parts) != 3:
        raise ValueError("Equation requires a relation")
    left, relation, right = parts
    left = re.sub(r"\\partial_t\s*([A-Za-z_]\w*)", r"\1", left)
    left = re.sub(r"\bd([A-Za-z_]\w*)/dt\b", r"\1", left)
    left = re.sub(r"\\dot\{([A-Za-z_]\w*)\}", r"\1", left)
    left = re.sub(r"\\frac\{d\s*([A-Za-z_]\w*)\}\{d\s*t\}", r"\1", left)
    left = re.sub(r"([A-Za-z_]\w*)\((?:0|t_?0)\)", r"\1", left)
    left = re.sub(r"([A-Za-z_]\w*)_0\b", r"\1", left)
    left = re.sub(r"([A-Za-z_]\w*)_\{?n\s*\+\s*1\}?", r"\1", left)
    return parse_expression(left.strip()), relation, parse_expression(right.strip())


def to_csel(expr) -> dict:
    if expr.is_Number or expr in (sp.pi, sp.E):
        return {"kind": "literal", "value": float(expr)}
    if isinstance(expr, sp.Symbol):
        return {"kind": "symbol", "name": str(expr)}
    if isinstance(expr, (sp.Add, sp.Mul)):
        values = [to_csel(arg) for arg in expr.args]
        result = values[0]
        for value in values[1:]:
            result = {"kind": "binary", "op": "+" if isinstance(expr, sp.Add) else "*", "left": result, "right": value}
        return result
    if isinstance(expr, sp.Pow):
        if expr.exp == sp.Rational(1, 2):
            return {"kind": "call", "func": "sqrt", "args": [to_csel(expr.base)]}
        return {"kind": "binary", "op": "^", "left": to_csel(expr.base), "right": to_csel(expr.exp)}
    names = {sp.log: "log", sp.exp: "exp", sp.sin: "sin", sp.cos: "cos", sp.Abs: "abs"}
    if expr.func in names:
        return {"kind": "call", "func": names[expr.func], "args": [to_csel(arg) for arg in expr.args]}
    raise ValueError(f"Unsupported CSEL expression: {expr.func}")


def _dimension(csel, dimensions):
    report = json.loads(_native().verify_equation_dimensions_hin(json.dumps(csel), json.dumps(dimensions)))
    return SIDimensionVectorModel(exponents=report["output_dimensions"], semantic_tag=report["output_semantic_tag"])


def _hin_compatible_csel(expr):
    if isinstance(expr, list):
        return [_hin_compatible_csel(item) for item in expr]
    if not isinstance(expr, dict):
        return expr
    expr = {key: _hin_compatible_csel(value) for key, value in expr.items()}
    if expr.get("kind") == "call" and expr.get("func") == "safe_pow":
        return {"kind": "binary", "op": "^", "left": expr["args"][0], "right": expr["args"][1]}
    return expr


def _z3(expr):
    if isinstance(expr, sp.Symbol): return z3.Real(str(expr))
    if expr.is_Number: return z3.RealVal(str(expr))
    if isinstance(expr, sp.Add): return sum(_z3(arg) for arg in expr.args)
    if isinstance(expr, sp.Mul):
        result = z3.RealVal(1)
        for arg in expr.args: result *= _z3(arg)
        return result
    if isinstance(expr, sp.Pow) and expr.exp.is_Integer and abs(int(expr.exp)) <= 12:
        return _z3(expr.base) ** int(expr.exp)
    raise FormalGateError("Z3_UNSUPPORTED_EXPRESSION", str(expr))


def check_z3(ir: CanonicalResearchIR) -> None:
    solver = z3.Solver()
    solver.set(timeout=2000)
    for constant in ir.constants:
        solver.assert_and_track(z3.Real(constant.symbol) == z3.RealVal(str(constant.si_value)), f"const:{constant.constant_id}")
    for item in [*ir.constants, *ir.state_variables]:
        symbol = z3.Real(item.symbol)
        bounds = item.bounds
        if bounds:
            if bounds.lower is not None:
                solver.assert_and_track(symbol >= bounds.lower if bounds.lower_inclusive else symbol > bounds.lower, f"bound:{item.symbol}:lower")
            if bounds.upper is not None:
                solver.assert_and_track(symbol <= bounds.upper if bounds.upper_inclusive else symbol < bounds.upper, f"bound:{item.symbol}:upper")
        if hasattr(item, "initial_value") and item.initial_value is not None:
            solver.assert_and_track(symbol == item.initial_value, f"initial:{item.symbol}")
    for equation in ir.equations:
        if equation.role not in (EquationRole.INVARIANT_CONTRACT, EquationRole.INITIAL_CONDITION):
            continue
        left, relation, right = equation_parts(equation.canonical_latex)
        a, b = _z3(left), _z3(right)
        constraint = {"<": lambda: a < b, "<=": lambda: a <= b, ">": lambda: a > b,
                      ">=": lambda: a >= b, "=": lambda: a == b, "==": lambda: a == b}[relation]()
        solver.assert_and_track(constraint, f"eq:{equation.equation_id}")
    result = solver.check()
    if result == z3.unsat:
        raise FormalGateError("Z3_SPEC_UNSAT", "Joint specification is contradictory", details=[str(label) for label in solver.unsat_core()])
    if result == z3.unknown:
        raise FormalGateError("Z3_SPEC_UNKNOWN", solver.reason_unknown())
    for equation in ir.equations:
        equation.z3_satisfiable = True


def lower_and_verify_equations(raw_eqs: list[RawEquation], raw_constants: list[RawConstant], vault_metadata=None):
    metadata = vault_metadata or {}
    constants = [normalize_constant(raw, f"constant:{index}") for index, raw in enumerate(raw_constants)]
    equations, flags = [], []
    for raw in raw_eqs:
        canonical = clean_math(raw.text)
        role = classify_equation(canonical)
        numeric_text = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", raw.text.strip().strip("$"))
        numeric = re.fullmatch(rf"([A-Za-z_]\w*)\s*=\s*({NUMBER})(?:\s+(.+))?", numeric_text)
        if numeric and role != EquationRole.INITIAL_CONDITION and (numeric[3] is None or not re.search(r"[+*=<>]", numeric[3])):
            try:
                constants.append(normalize_constant(RawConstant(symbol=numeric[1], value=numeric[2], unit=numeric[3] or "1", provenance=raw.provenance), f"constant:{len(constants)}"))
                continue
            except FormalGateError:
                if numeric[3] is not None:
                    raise
        equations.append(ExtractedEquation(equation_id=f"equation:{len(equations)}", raw_latex=raw.text,
            canonical_latex=canonical, role=role, provenance=raw.provenance))
    defined = {c.symbol for c in constants} | set(metadata.get("state_symbols", []))
    for equation in equations:
        try:
            left, _, right = equation_parts(equation.canonical_latex)
            equation.lhs_symbol = str(left) if isinstance(left, sp.Symbol) else None
            equation.referenced_symbols = sorted(str(symbol) for symbol in left.free_symbols | right.free_symbols)
            if equation.lhs_symbol and equation.role in (EquationRole.EVOLUTION_LAW, EquationRole.ALGEBRAIC_CLOSURE, EquationRole.INITIAL_CONDITION):
                defined.add(equation.lhs_symbol)
        except Exception as exc:
            equation.diagnostics.append(f"CSEL_PARSE_ERROR: {exc}")
    context = metadata.get("source_text") or "\n".join(raw.provenance.quote for raw in raw_eqs)
    referenced = {name for eq in equations for name in eq.referenced_symbols}
    for parameter in lookup_reference_registry(context):
        if parameter.symbol in referenced - defined:
            constants.append(normalize_constant(RawConstant(symbol=parameter.symbol, value=str(parameter.raw_value), unit=parameter.raw_unit,
                provenance=raw_eqs[0].provenance), f"constant:{len(constants)}"))
            defined.add(parameter.symbol)
    dimensions = {c.symbol: c.dimension.model_dump() for c in constants}
    dimensions.update(metadata.get("symbol_dimensions", {}))
    for equation in equations:
        if equation.role == EquationRole.EVOLUTION_LAW and equation.lhs_symbol:
            dimensions.setdefault(equation.lhs_symbol, SIDimensionVectorModel().model_dump())
    for equation in equations:
        missing = set(equation.referenced_symbols) - defined - {"t", "pi", "e"}
        if equation.diagnostics or missing:
            for symbol in sorted(missing) or [equation.equation_id]:
                flags.append(DisambiguationFlag(flag_id=f"flag:{len(flags)}", trigger=DisambiguationTriggerKind.UNRESOLVED_SYMBOL,
                    target_symbol_or_eq=symbol, context_excerpt=equation.provenance.quote,
                    source_span=equation.provenance))
            continue
        try:
            left, _, right = equation_parts(equation.canonical_latex)
            csel = to_csel(right)
            rhs_dim = _dimension(csel, dimensions)
            optimizer = getattr(_native(), "csel_egraph_optimize", None)
            if optimizer is not None:
                csel = _hin_compatible_csel(json.loads(optimizer(json.dumps(csel)))["expr"])
                _dimension(csel, dimensions)
            if equation.lhs_symbol not in dimensions and equation.lhs_symbol:
                dimensions[equation.lhs_symbol] = rhs_dim.model_dump()
            lhs = to_csel(left)
            if equation.role == EquationRole.EVOLUTION_LAW:
                lhs = {"kind": "binary", "op": "/", "left": lhs, "right": {"kind": "symbol", "name": "t"}}
            lhs_dim = _dimension(lhs, dimensions)
            if equation.role == EquationRole.INVARIANT_CONTRACT and right.is_number:
                rhs_dim = lhs_dim.model_copy(deep=True)
            if lhs_dim.exponents != rhs_dim.exponents and right != 0:
                raise ValueError("HIN_SI_DIM_MISMATCH")
            if lhs_dim.semantic_tag and rhs_dim.semantic_tag and lhs_dim.semantic_tag != rhs_dim.semantic_tag:
                raise ValueError("SEMANTIC_TAG_MISMATCH")
            equation.csel_ast = csel
            equation.sympy_expr = str(right)
            equation.dimension_lhs, equation.dimension_rhs = lhs_dim, rhs_dim
            equation.hin_homogeneous = True
            equation.verification_status = VerificationStatus.VERIFIED
        except (ValueError, TypeError, KeyError) as exc:
            raise FormalGateError("HIN_SI_DIM_MISMATCH", f"{equation.equation_id}: {exc}") from exc
    evolution = [eq for eq in equations if eq.role == EquationRole.EVOLUTION_LAW]
    if len(evolution) > 1 and not metadata.get("state_symbols"):
        flags.append(DisambiguationFlag(flag_id=f"flag:{len(flags)}", trigger=DisambiguationTriggerKind.AMBIGUOUS_STATE_ORDERING,
            target_symbol_or_eq="state_order", context_excerpt=evolution[0].provenance.quote, source_span=evolution[0].provenance))
    for equation in evolution:
        if "\\partial" in equation.canonical_latex and equation.lhs_symbol not in (metadata.get("boundary_conditions") or {}):
            flags.append(DisambiguationFlag(flag_id=f"flag:{len(flags)}", trigger=DisambiguationTriggerKind.IMPLICIT_BOUNDARY_CONDITION,
                target_symbol_or_eq=equation.equation_id, context_excerpt=equation.provenance.quote, source_span=equation.provenance))
    temporary = CanonicalResearchIR(ir_id="verify", source_mode="upload", title="Verification", equations=equations, constants=constants)
    check_z3(temporary)
    return equations, constants, flags

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
               "fraction": "1", "dimensionless": "1", "unitless": "1", "units": "1", "dB": "1", "bit": "1", "bp": "1", "kg/kg": "1"}
    unit = (unit or "1").strip()
    # ARCH-009 (BUG-018): Normalize Unicode superscripts and middle-dots in compound units
    # such as `W/(m²·K)`, `m·s⁻¹`, `J/(kg·K)`, `F/m²`.
    unit = (
        unit.replace("·", "*")
        .replace("⋅", "*")
        .replace("×", "*")
        .replace("−", "-")
        .replace("²", "**2")
        .replace("³", "**3")
        .replace("⁻¹", "**-1")
        .replace("⁻²", "**-2")
        .replace("⁻³", "**-3")
    )
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
                if isinstance(exponent, float) and abs(exponent * 2 - round(exponent * 2)) < 1e-9 and abs(exponent) <= 127:
                    return tuple(int(round(v * exponent)) for v in left)
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


_RESERVED_KW_MANGLE = {
    "lambda": "lambda_var",
    "from": "from_var",
    "class": "class_var",
    "global": "global_var",
}


def mangle_reserved_symbol(name: str) -> str:
    return _RESERVED_KW_MANGLE.get(name, name)


def normalize_constant(raw: RawConstant, constant_id: str) -> ExtractedConstant:
    unit = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", raw.unit.strip())
    unit = unit.replace("{", "").replace("}", "").replace("\\,", "").rstrip(".,;") or "1"
    value, canonical = normalize_to_si(float(raw.value), unit)
    if not math.isfinite(value):
        raise FormalGateError("HIN_NONFINITE_LITERAL", raw.symbol)
    dimension = unit_dimension(canonical or "1")
    dimension.canonical_unit = canonical or "1"
    return ExtractedConstant(constant_id=constant_id, symbol=raw.symbol, name=raw.symbol,
        raw_value=raw.value, raw_unit=raw.unit, si_value=value, dimension=dimension, provenance=raw.provenance)


def clean_math(text: str) -> str:
    raw = str(text or "").strip()
    if "Piecewise(" in raw or re.match(r"^\s*d[A-Za-z_]\w*/dt\s*=", raw):
        return raw.strip("$")
    text = canonicalize_math_content(repair_json_control_escapes(raw)).strip().strip("$")
    # Strip trailing LaTeX unit annotations attached via \, or \; (e.g. \,\mathrm{m}/\sqrt{\mathrm{Hz}})
    text = re.sub(
        r"\s*\\[,;:]\s*(?:\\(?:mathrm|text)\{[^{}]+\}|\\sqrt\{\\(?:mathrm|text)\{[^{}]+\}\}|[/^*0-9()+-])+\s*$",
        "",
        text,
    )
    # Desugar LaTeX norm notation \|\Theta\|_\infty or \|x\|_{p} -> Theta_norm_inf / x_norm_p
    text = re.sub(
        r"\\\|\s*\\?([A-Za-z_]\w*)\s*\\\|(?:_\{?\\?([A-Za-z0-9_]+)\}?)?",
        lambda m: f"{m.group(1)}_norm_{(m.group(2) or '2').replace('infty', 'inf')}",
        text,
    )
    text = re.sub(r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}", r"\1", text)
    # Desugar compound \Delta s^2_{2PN} or \Delta x_{sub} -> Delta_s2_2PN / Delta_x_sub
    text = re.sub(
        r"\\Delta\s+([A-Za-z_]\w*)\^\{?(\d+)\}?_\{?([A-Za-z0-9_]+)\}?",
        r"Delta_\1\2_\3",
        text,
    )
    text = re.sub(
        r"\\Delta\s+([A-Za-z_]\w*)(?:_\{?([A-Za-z0-9_]+)\}?)?",
        lambda m: f"Delta_{m.group(1)}_{m.group(2)}" if m.group(2) else f"Delta_{m.group(1)}",
        text,
    )
    # Desugar ASD / half-power observable symbol S_r^{1/2} -> S_r_asd on LHS of relations
    text = re.sub(r"\b([A-Za-z_]\w*)\^\{1/2\}", r"\1_asd", text)
    # Desugar scientific notation a \times 10^{b} or 10^{b}
    text = re.sub(r"\\times\s*10\^\{([+-]?\d+)\}", r"* 1e\1", text)
    text = re.sub(r"(?<![A-Za-z0-9_.])10\^\{([+-]?\d+)\}", r"1e\1", text)
    text = text.replace("\\times", "*")
    text = re.sub(r"\\label\{[^}]*\}", "", text).replace("&", "")
    return text.replace("\\leq", "<=").replace("\\geq", ">=").replace("\\le", "<=").replace("\\ge", ">=").replace("\\equiv", "=").strip()


def classify_equation(text: str) -> EquationRole:
    if re.search(r"\\dot|\\partial_t|Derivative\(|\\frac\{d|\bd\w+/dt\b|_\{?n\s*\+\s*1", text):
        return EquationRole.EVOLUTION_LAW
    if re.search(r"<=|>=|<|>|==|\\sum|\\equiv", text):
        return EquationRole.INVARIANT_CONTRACT
    if re.search(r"\([^,]+,\s*t\)|partial.?Omega", text):
        return EquationRole.BOUNDARY_CONDITION
    # BUG-036: Only treat `\w+_0 = <number>` as INITIAL_CONDITION; algebraic formulas
    # like `P_0 = 1 / (sum + ...)` are ALGEBRAIC_CLOSURE, not initial conditions!
    if re.match(r"\s*\w+\((?:0|t_?0)\)\s*=", text) or re.fullmatch(rf"\s*\w+_0\s*=\s*{NUMBER}\s*", text):
        return EquationRole.INITIAL_CONDITION
    if re.fullmatch(rf"\s*[A-Za-z_]\w*\s*=\s*{NUMBER}\s*", text):
        return EquationRole.PARAMETER_TABLE
    return EquationRole.ALGEBRAIC_CLOSURE


def parse_expression(text: str, *, declared_symbols: set[str] | None = None):
    """Parse ASCII through a whitelist, never sympify/eval uploaded expressions.

    ARCH-009 (BUG-016): Strict Symbol Identity — any identifier in `declared_symbols`
    (including `e`, `E`, `pi`, `I`, `S`, `N`, `lambda`, `gamma`) ALWAYS binds to a user
    `sp.Symbol`, NEVER silently substituted with `sp.E`/`sp.pi`/`sp.I`.
    """
    if len(text) > 8192:
        raise ValueError("Equation size limit exceeded")
    declared = set(declared_symbols or ())
    text = text.replace("\\cdot", "*")
    latex_text = text

    def _restore_declared(expr):
        if not declared:
            return expr
        subs = {}
        if "e" in declared and expr.has(sp.E):
            subs[sp.E] = sp.Symbol("e")
        if "E" in declared and expr.has(sp.E):
            subs[sp.E] = sp.Symbol("E")
        if "pi" in declared and expr.has(sp.pi):
            subs[sp.pi] = sp.Symbol("pi")
        if "I" in declared and expr.has(sp.I):
            subs[sp.I] = sp.Symbol("I")
        return expr.xreplace(subs) if subs else expr

    def latex_expression(source):
        # ARCH-014 (BUG-022):
        # 1. Prefix bare Greek names (e.g. `sigma`) with `\` so parse_latex does not
        #    split `sigma` into `s*i*g*m*a`.
        # 2. Insert explicit `\cdot` for implicit products after closing parens like
        #    `(...) P_0` and `(...) \exp(...)`.
        src = re.sub(
            r"(?<![\\A-Za-z])(alpha|beta|gamma|delta|epsilon|eta|theta|kappa|lambda|mu|nu|xi|rho|sigma|tau|phi|psi|omega)(?![A-Za-z])",
            r"\\mathit{\1}",
            source,
        )
        src = re.sub(r"([)}])\s+(?=[A-Za-z\\])", r"\1 \\cdot ", src)
        try:
            result = parse_latex(src, backend="lark")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Unparseable LaTeX expression: {exc}") from exc
        if isinstance(result, (list, tuple, set)):
            basics = [item for item in result if isinstance(item, sp.Basic)]
            if basics:
                result = basics[0]
        elif hasattr(result, "children"):
            basics = [item for item in result.children if isinstance(item, sp.Basic)]
            if basics:
                result = basics[0]
        if not isinstance(result, sp.Basic):
            raise ValueError("Ambiguous LaTeX expression")
        result = result.xreplace({
            symbol: sp.Symbol(
                re.sub(r'([A-Za-z]\w*)_\{([A-Za-z0-9_]+)\}', r'\1_\2', str(symbol))
            )
            for symbol in result.free_symbols
        })
        return _restore_declared(result)

    ascii_text = re.sub(
        r'\\?([A-Za-z]\w*)_\{([A-Za-z0-9_,]+)\}',
        lambda m: f"{m.group(1)}_{m.group(2).replace(',', '_')}",
        text,
    )
    ascii_text = re.sub(
        r"\\(exp|log|sin|cos|alpha|beta|gamma|Gamma|delta|Delta|epsilon|eta|theta|Theta|kappa|lambda|Lambda|mu|nu|xi|pi|Pi|rho|sigma|Sigma|tau|phi|Phi|psi|Psi|omega|Omega)(?![A-Za-z])",
        r"\1",
        ascii_text,
    )
    ascii_text = ascii_text.replace("\\times", "*").replace("\\le", "<=").replace("\\ge", ">=").replace("\\leq", "<=").replace("\\geq", ">=")
    ascii_text = re.sub(r"10\^\{([+-]?\d+)\}", r"10**(\1)", ascii_text)
    if "\\" in ascii_text or "{" in ascii_text:
        return latex_expression(latex_text)
    # ARCH-014 (BUG-022): Temporarily escape Python keyword `lambda` for ast.parse,
    # then restore exact mathematical symbol `sp.Symbol("lambda")` in convert().
    # Also insert explicit `*` for implicit products after `)` like `(a + b) P_0`.
    ascii_text = re.sub(r"\)\s+(?=[A-Za-z_])", ") * ", ascii_text)
    text = re.sub(r"\blambda\b", "__invar_kw_lambda__", ascii_text)
    functions = {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "sqrt": sp.sqrt, "abs": sp.Abs, "Abs": sp.Abs,
        "erf": sp.erf, "erfc": sp.erfc,
        "Phi": lambda x, **_kw: sp.Rational(1, 2) * (1 + sp.erf(x / sp.sqrt(2))),
        "min": sp.Min, "Min": sp.Min, "max": sp.Max, "Max": sp.Max,
    }

    def convert(node):
        if isinstance(node, ast.Name):
            name = "lambda" if node.id == "__invar_kw_lambda__" else node.id
            if name in declared:
                return sp.Symbol(name)
            return {"pi": sp.pi, "e": sp.E, "True": sp.true, "False": sp.false}.get(name, sp.Symbol(name))
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return sp.true if node.value else sp.false
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if not math.isfinite(node.value):
                raise ValueError("Nonfinite literal")
            return sp.Rational(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return convert(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return sp.Not(convert(node.operand))
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add): return sp.Add(left, right, evaluate=False)
            if isinstance(node.op, ast.Sub): return sp.Add(left, -right, evaluate=False)
            if isinstance(node.op, ast.Mult): return sp.Mul(left, right, evaluate=False)
            if isinstance(node.op, ast.Div): return sp.Mul(left, sp.Pow(right, -1, evaluate=False), evaluate=False)
            if isinstance(node.op, ast.Pow): return sp.Pow(left, right, evaluate=False)
            if isinstance(node.op, ast.BitAnd): return sp.And(left, right)
            if isinstance(node.op, ast.BitOr): return sp.Or(left, right)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            rel_map = {ast.Lt: sp.Lt, ast.LtE: sp.Le, ast.Gt: sp.Gt, ast.GtE: sp.Ge, ast.Eq: sp.Eq, ast.NotEq: sp.Ne}
            rel = rel_map.get(type(node.ops[0]))
            if rel is not None:
                return rel(convert(node.left), convert(node.comparators[0]), evaluate=False)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id in functions:
                return functions[node.func.id](*(convert(arg) for arg in node.args), evaluate=False)
            if node.func.id == "Piecewise":
                pairs = []
                for arg in node.args:
                    if isinstance(arg, ast.Tuple) and len(arg.elts) == 2:
                        pairs.append((convert(arg.elts[0]), convert(arg.elts[1])))
                if pairs:
                    return sp.Piecewise(*pairs)
            if node.func.id == "ITE" and len(node.args) == 3:
                c, t_v, f_v = (convert(a) for a in node.args)
                return sp.Or(sp.And(c, t_v), sp.And(sp.Not(c), f_v))
        raise ValueError("Unsupported expression")
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except SyntaxError:
        return latex_expression(latex_text)
    return convert(tree.body)


def equation_parts(text: str, *, declared_symbols: set[str] | None = None):
    parts = re.split(r"(<=|>=|==|=|<|>)", text, maxsplit=1)
    if len(parts) != 3:
        raise ValueError("Equation requires a relation")
    left, relation, right = parts
    left = re.sub(r"\\partial_t\s*([A-Za-z_]\w*)", r"\1", left)
    left = re.sub(r"\bd([A-Za-z_]\w*)/dt\b", r"\1", left)
    left = re.sub(r"\\dot\{([A-Za-z_]\w*)\}", r"\1", left)
    left = re.sub(r"\\frac\{d\s*([A-Za-z_]\w*)\}\{d\s*t\}", r"\1", left)
    left = re.sub(r"([A-Za-z_]\w*)\((?:0|t_?0)\)", r"\1", left)
    if re.fullmatch(rf"\s*{NUMBER}\s*", right.strip()):
        left = re.sub(r"([A-Za-z_]\w*)_0\b", r"\1", left)
    left = re.sub(r"([A-Za-z_]\w*)_\{?n\s*\+\s*1\}?", r"\1", left)
    return (
        parse_expression(left.strip(), declared_symbols=declared_symbols),
        relation,
        parse_expression(right.strip(), declared_symbols=declared_symbols),
    )


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
    if isinstance(expr, sp.Piecewise):
        # For CSEL/HIN lowering, pick the primary non-zero interior branch (last default branch if non-zero,
        # otherwise the first non-zero branch).
        for branch_expr, cond in reversed(expr.args):
            if cond == sp.true and branch_expr != 0:
                return to_csel(branch_expr)
        for branch_expr, _cond in expr.args:
            if branch_expr != 0:
                return to_csel(branch_expr)
        return {"kind": "literal", "value": 0.0}
    if expr.func in (sp.Min, sp.Max) and expr.args:
        for arg in expr.args:
            if arg != 0:
                return to_csel(arg)
        return {"kind": "literal", "value": 0.0}
    names = {
        sp.log: "log", sp.exp: "exp", sp.sin: "sin", sp.cos: "cos", sp.Abs: "abs",
        sp.tan: "tan", sp.asin: "asin", sp.acos: "acos", sp.atan: "atan",
        sp.sinh: "sinh", sp.cosh: "cosh", sp.tanh: "tanh",
        sp.erf: "erf", sp.erfc: "erfc",
    }
    if expr.func in names:
        return {"kind": "call", "func": names[expr.func], "args": [to_csel(arg) for arg in expr.args]}
    raise ValueError(f"Unsupported CSEL expression: {expr.func}")


def _dimension(csel, dimensions):
    report = json.loads(_native().verify_equation_dimensions_hin(json.dumps(_hin_compatible_csel(csel)), json.dumps(dimensions)))
    return SIDimensionVectorModel(exponents=report["output_dimensions"], semantic_tag=report["output_semantic_tag"])


def _hin_compatible_csel(expr):
    if isinstance(expr, list):
        return [_hin_compatible_csel(item) for item in expr]
    if not isinstance(expr, dict):
        return expr
    expr = {key: _hin_compatible_csel(value) for key, value in expr.items()}
    if expr.get("kind") == "call" and expr.get("func") == "safe_pow":
        return {"kind": "binary", "op": "^", "left": expr["args"][0], "right": expr["args"][1]}
    if expr.get("kind") == "call" and expr.get("func") in {"erf", "erfc", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh"}:
        # Dimensionless transcendental special functions: identical dimensional signature to sin(x) (1 -> 1).
        return {"kind": "call", "func": "sin", "args": expr["args"]}
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
    from invar_agent.runtime.isolation import run_on_z3_thread
    run_on_z3_thread(_check_z3_locked, ir)


def _check_z3_locked(ir: CanonicalResearchIR) -> None:
    solver = z3.Solver()
    solver.set(timeout=2000)
    const_vals = {}
    for constant in ir.constants:
        if constant.symbol in const_vals:
            prev_v = const_vals[constant.symbol]
            if abs(float(constant.si_value) - prev_v) <= 1e-6 * max(1.0, abs(prev_v), abs(float(constant.si_value))):
                continue
        const_vals[constant.symbol] = float(constant.si_value)
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
            prev_c = const_vals.get(item.symbol)
            if prev_c is not None and abs(float(item.initial_value) - prev_c) <= 1e-6 * max(1.0, abs(prev_c), abs(float(item.initial_value))):
                continue
            solver.assert_and_track(symbol == item.initial_value, f"initial:{item.symbol}")
    for equation in ir.equations:
        if equation.diagnostics or equation.role not in (EquationRole.INVARIANT_CONTRACT, EquationRole.INITIAL_CONDITION):
            continue
        left, relation, right = equation_parts(equation.canonical_latex)
        try:
            a, b = _z3(left), _z3(right)
        except FormalGateError:
            continue
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
            left, _, right = equation_parts(equation.canonical_latex, declared_symbols=defined)
            equation.lhs_symbol = str(left) if isinstance(left, sp.Symbol) else None
            equation.referenced_symbols = sorted(str(symbol) for symbol in left.free_symbols | right.free_symbols)
            if equation.lhs_symbol and equation.role in (EquationRole.EVOLUTION_LAW, EquationRole.ALGEBRAIC_CLOSURE, EquationRole.INITIAL_CONDITION):
                defined.add(equation.lhs_symbol)
            elif equation.lhs_symbol and equation.role == EquationRole.INVARIANT_CONTRACT and right.is_number and "_" in equation.lhs_symbol:
                # Compound/subscripted audit observables in uploaded specs (e.g. `S_r_asd <= 1e-14`, `Theta_norm_inf <= 1e-8`)
                # declare their own observable symbol; bare single-token symbols like `k >= 0` stay unresolved for disambiguation.
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
    # ARCH-009 (BUG-018): Jointly infer SI dimensions for ODE states AND algebraic closures
    # from declared parameter units via linear dimensional system rather than defaulting to [0]*7.
    unknown_syms = [
        eq.lhs_symbol for eq in equations
        if eq.role in (EquationRole.EVOLUTION_LAW, EquationRole.ALGEBRAIC_CLOSURE)
        and eq.lhs_symbol and eq.lhs_symbol not in dimensions
    ]
    # Deduplicate preserving order
    unknown_syms = list(dict.fromkeys(unknown_syms))
    if unknown_syms:
        dim_vars = {name: [sp.Symbol(f"_sdim_{idx}_{ax}") for ax in range(7)] for idx, name in enumerate(unknown_syms)}
        known_dims = {name: list(val.get("exponents", (0,) * 7)) for name, val in dimensions.items()}
        known_dims["t"] = [0, 0, 1, 0, 0, 0, 0]
        dim_constraints = []

        def _expr_dim(ex):
            if ex.is_Number or ex in (sp.pi, sp.E):
                return [0] * 7
            if isinstance(ex, sp.Symbol):
                s = str(ex)
                return known_dims.get(s, dim_vars.get(s, [0] * 7))
            if isinstance(ex, sp.Add):
                parts = [_expr_dim(a) for a in ex.args]
                for part in parts[1:]:
                    dim_constraints.extend(a - b for a, b in zip(parts[0], part))
                return parts[0]
            if isinstance(ex, sp.Mul):
                parts = [_expr_dim(a) for a in ex.args]
                return [sum(p[ax] for p in parts) for ax in range(7)]
            if isinstance(ex, sp.Pow) and ex.exp.is_Rational:
                base_d = _expr_dim(ex.base)
                return [v * ex.exp for v in base_d]
            if ex.func in (sp.sin, sp.cos, sp.exp, sp.log):
                inner = _expr_dim(ex.args[0])
                dim_constraints.extend(inner)
                return [0] * 7
            if ex.func == sp.Abs:
                return _expr_dim(ex.args[0])
            return [0] * 7

        for eq in equations:
            if eq.role in (EquationRole.EVOLUTION_LAW, EquationRole.ALGEBRAIC_CLOSURE) and eq.lhs_symbol in dim_vars:
                try:
                    _l, _, _r = equation_parts(eq.canonical_latex, declared_symbols=defined)
                    rhs_d = _expr_dim(_r)
                    if eq.role == EquationRole.EVOLUTION_LAW:
                        dim_constraints.extend(
                            a - b - c for a, b, c in zip(dim_vars[eq.lhs_symbol], known_dims["t"], rhs_d)
                        )
                    else:
                        dim_constraints.extend(
                            a - c for a, c in zip(dim_vars[eq.lhs_symbol], rhs_d)
                        )
                except Exception:
                    pass
        all_v = [v for vec in dim_vars.values() for v in vec]
        if dim_constraints and all_v:
            sol_set = sp.linsolve(dim_constraints, all_v)
            if sol_set and sol_set is not sp.EmptySet:
                vals = list(next(iter(sol_set)))
                free_map = {sym: 0 for v in vals for sym in v.free_symbols}
                vals = [v.subs(free_map) for v in vals]
                for idx, name in enumerate(unknown_syms):
                    exps = vals[idx * 7 : (idx + 1) * 7]
                    if all(v.is_Integer for v in exps):
                        p = tuple(int(v) for v in exps)
                        unit_lbl = "*".join(
                            b + (f"^{pw}" if pw != 1 else "")
                            for b, pw in zip(("m", "kg", "s", "A", "K", "mol", "cd"), p)
                            if pw
                        ) or "1"
                        dimensions[name] = SIDimensionVectorModel(exponents=p, canonical_unit=unit_lbl).model_dump()
    for sym in metadata.get("state_symbols", []):
        dimensions.setdefault(sym, SIDimensionVectorModel().model_dump())
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
        is_vault_law = (
            equation.role == EquationRole.EVOLUTION_LAW
            and str(getattr(equation.provenance, "location", "") or "").startswith("vault:")
        )
        try:
            left, _, right = equation_parts(equation.canonical_latex, declared_symbols=defined)
            csel = to_csel(right)
            lhs = to_csel(left)
            if equation.role == EquationRole.EVOLUTION_LAW:
                lhs = {"kind": "binary", "op": "/", "left": lhs, "right": {"kind": "symbol", "name": "t"}}
            try:
                rhs_dim = _dimension(csel, dimensions)
                optimizer = getattr(_native(), "csel_egraph_optimize", None)
                if optimizer is not None:
                    csel = _hin_compatible_csel(json.loads(optimizer(json.dumps(csel)))["expr"])
                    _dimension(csel, dimensions)
            except (ValueError, TypeError, KeyError):
                if not is_vault_law:
                    raise
                rhs_dim = _dimension(lhs, dimensions)
            if equation.lhs_symbol not in dimensions and equation.lhs_symbol:
                dimensions[equation.lhs_symbol] = rhs_dim.model_dump()
            lhs_dim = _dimension(lhs, dimensions)
            if equation.role in (EquationRole.INVARIANT_CONTRACT, EquationRole.INITIAL_CONDITION) and right.is_number:
                rhs_dim = lhs_dim.model_copy(deep=True)
            if lhs_dim.exponents != rhs_dim.exponents and right != 0:
                if is_vault_law:
                    rhs_dim = lhs_dim.model_copy(deep=True)
                else:
                    raise ValueError("HIN_SI_DIM_MISMATCH")
            if lhs_dim.semantic_tag and rhs_dim.semantic_tag and lhs_dim.semantic_tag != rhs_dim.semantic_tag:
                if not is_vault_law:
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

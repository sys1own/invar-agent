"""Zero-LLM-token quantitative insight extraction + SI-unit normalization.

Deterministically turns raw literature snippets (Wikipedia wikitext, ar5iv
HTML, arXiv/Semantic Scholar abstracts, Wikidata/OEIS/PubChem payloads) into a
strongly typed :class:`StructuredResearchEvidence`:

* SI-normalized :class:`ExtractedParameter` constants (``normalize_to_si``),
* LaTeX equations with ``v9_transpiler_friendly`` Python hints
  (``latex_to_python``),
* benchmark anchors and suggested invariants from the curated
  ``DOMAIN_REFERENCE_REGISTRY`` (7 quantitative domains),
* a recommended backend/stepper (``recommend_execution_strategy``).

Nothing in this module calls an LLM or the network.
"""

from __future__ import annotations

import ast
import html
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schemas.research import (
    BenchmarkAnchor,
    ExtractedEquation,
    ExtractedParameter,
    StructuredResearchEvidence,
)

# ---------------------------------------------------------------------------
# SI-unit normalization
# ---------------------------------------------------------------------------

_EV = 1.602176634e-19

# unit -> (factor to SI, SI unit). Case-sensitive (MW vs mW, T vs t, G vs g).
_UNIT_TABLE: Dict[str, Tuple[float, str]] = {
    # length
    "m": (1.0, "m"), "km": (1e3, "m"), "cm": (1e-2, "m"), "mm": (1e-3, "m"),
    "um": (1e-6, "m"), "nm": (1e-9, "m"), "AU": (1.495978707e11, "m"),
    "au": (1.495978707e11, "m"),
    # mass
    "kg": (1.0, "kg"), "g": (1e-3, "kg"), "mg": (1e-6, "kg"),
    "t": (1e3, "kg"), "ton": (1e3, "kg"), "tons": (1e3, "kg"),
    "tonne": (1e3, "kg"), "tonnes": (1e3, "kg"),
    # time
    "s": (1.0, "s"), "sec": (1.0, "s"), "ms": (1e-3, "s"), "us": (1e-6, "s"),
    "ns": (1e-9, "s"), "min": (60.0, "s"), "h": (3600.0, "s"),
    "hr": (3600.0, "s"), "hour": (3600.0, "s"), "hours": (3600.0, "s"),
    "d": (86400.0, "s"), "day": (86400.0, "s"), "days": (86400.0, "s"),
    "yr": (3.15576e7, "s"), "year": (3.15576e7, "s"), "years": (3.15576e7, "s"),
    # velocity
    "m/s": (1.0, "m/s"), "km/s": (1e3, "m/s"), "km/h": (1.0 / 3.6, "m/s"),
    "cm/s": (1e-2, "m/s"), "c": (299792458.0, "m/s"),
    # acceleration
    "m/s^2": (1.0, "m/s^2"), "g_E": (9.80665, "m/s^2"), "gE": (9.80665, "m/s^2"),
    "g_0": (9.80665, "m/s^2"),
    # density
    "kg/m^3": (1.0, "kg/m^3"), "g/cm^3": (1e3, "kg/m^3"), "g/cc": (1e3, "kg/m^3"),
    "kg/L": (1e3, "kg/m^3"), "g/L": (1.0, "kg/m^3"),
    # number density
    "m^-3": (1.0, "m^-3"), "cm^-3": (1e6, "m^-3"), "/cm^3": (1e6, "m^-3"),
    # pressure / stress
    "Pa": (1.0, "Pa"), "kPa": (1e3, "Pa"), "MPa": (1e6, "Pa"), "GPa": (1e9, "Pa"),
    "bar": (1e5, "Pa"), "atm": (101325.0, "Pa"),
    # power
    "W": (1.0, "W"), "kW": (1e3, "W"), "MW": (1e6, "W"), "GW": (1e9, "W"),
    # heat flux
    "W/m^2": (1.0, "W/m^2"), "W/cm^2": (1e4, "W/m^2"), "kW/m^2": (1e3, "W/m^2"),
    "MW/m^2": (1e6, "W/m^2"),
    # energy
    "J": (1.0, "J"), "kJ": (1e3, "J"), "MJ": (1e6, "J"), "GJ": (1e9, "J"),
    "eV": (_EV, "J"), "keV": (_EV * 1e3, "J"), "MeV": (_EV * 1e6, "J"),
    "GeV": (_EV * 1e9, "J"),
    # heat load
    "J/m^2": (1.0, "J/m^2"), "kJ/m^2": (1e3, "J/m^2"), "MJ/m^2": (1e6, "J/m^2"),
    "J/cm^2": (1e4, "J/m^2"),
    # magnetic field
    "T": (1.0, "T"), "mT": (1e-3, "T"), "uT": (1e-6, "T"), "nT": (1e-9, "T"),
    "G": (1e-4, "T"), "Gauss": (1e-4, "T"), "gauss": (1e-4, "T"),
    # angle
    "rad": (1.0, "rad"), "deg": (math.pi / 180.0, "rad"),
    "degree": (math.pi / 180.0, "rad"), "degrees": (math.pi / 180.0, "rad"),
    "°": (math.pi / 180.0, "rad"),
    # temperature / molar
    "K": (1.0, "K"), "g/mol": (1e-3, "kg/mol"), "kg/mol": (1.0, "kg/mol"),
    "mol": (1.0, "mol"),
    # thermal conductivity / specific heat / rates
    "W/(m*K)": (1.0, "W/(m*K)"), "W/(m K)": (1.0, "W/(m*K)"),
    "W/m/K": (1.0, "W/(m*K)"), "mW/(m*K)": (1e-3, "W/(m*K)"),
    "J/(kg*K)": (1.0, "J/(kg*K)"), "J/(kg K)": (1.0, "J/(kg*K)"),
    "J/kg/K": (1.0, "J/(kg*K)"), "kJ/(kg*K)": (1e3, "J/(kg*K)"),
    "1/s": (1.0, "1/s"), "s^-1": (1.0, "1/s"), "Hz": (1.0, "1/s"),
}

# Units that are safe to match case-insensitively (no SI-prefix ambiguity).
_CASE_INSENSITIVE_UNITS = {u.lower(): u for u in (
    "km/s", "km/h", "kg/m^3", "g/cm^3", "g/cc", "kg/L", "g/L", "atm", "bar",
    "kPa", "GPa", "W/m^2", "W/cm^2", "kW/m^2", "J/m^2", "J/cm^2", "eV", "keV",
    "deg", "degree", "degrees", "rad", "Gauss", "AU", "min", "hr", "yr", "day",
    "days", "tonne", "tonnes", "ton", "tons", "g/mol", "kg/mol")}


def _canonical_unit(unit_str: str) -> str:
    u = (unit_str or "").strip()
    u = (u.replace("²", "^2").replace("³", "^3").replace("⁻", "^-")
          .replace("¹", "").replace("μ", "u").replace("µ", "u")
          .replace("−", "-").replace("·", " ").replace("⋅", " "))
    u = re.sub(r"\s*/\s*", "/", u).strip()
    # "W/(m·K)" / "W/(m . K)" -> "W/(m*K)"
    u = re.sub(r"\(\s*(\w+)\s*[*.\s]\s*(\w+)\s*\)", r"(\1*\2)", u)
    u = re.sub(r"\^\{(-?\d)\}", r"^\1", u)
    # m2 / m3 / cm3 -> m^2 / m^3 / cm^3 (unit tokens only, never CO2)
    u = re.sub(r"(?<![A-Za-z])((?:k|c|m|M)?m)([23])\b", r"\1^\2", u)
    u = re.sub(r"(?<=/)((?:k|c|m)?m)([23])\b", r"\1^\2", u)
    # "kg m^-3" -> "kg/m^3", "m s^-1" -> "m/s", "W m^-2" -> "W/m^2"
    m = re.fullmatch(r"(\w+)\s+(\w+)\^-(\d)", u)
    if m:
        exp = "" if m.group(3) == "1" else f"^{m.group(3)}"
        u = f"{m.group(1)}/{m.group(2)}{exp}"
    if u in ("degC", "°C"):
        return u
    return u


def normalize_to_si(value: float, unit_str: str) -> Tuple[float, str]:
    """Convert ``value`` in ``unit_str`` to SI. Unknown units pass through."""
    value = float(value)
    canonical = _canonical_unit(unit_str)
    if not canonical:
        return value, ""
    if canonical in ("degC", "°C"):
        return value + 273.15, "K"
    entry = _UNIT_TABLE.get(canonical)
    if entry is None:
        mapped = _CASE_INSENSITIVE_UNITS.get(canonical.lower())
        entry = _UNIT_TABLE.get(mapped) if mapped else None
    if entry is None:
        return value, canonical
    factor, si_unit = entry
    return value * factor, si_unit


def is_known_unit(unit_str: str) -> bool:
    canonical = _canonical_unit(unit_str)
    return (canonical in _UNIT_TABLE or canonical.lower() in _CASE_INSENSITIVE_UNITS
            or canonical in ("degC", "°C"))


# ---------------------------------------------------------------------------
# LaTeX -> v9_transpiler_friendly Python hints
# ---------------------------------------------------------------------------

_LATEX_FUNCS = {"sin": "math.sin", "cos": "math.cos", "tan": "math.tan",
                "exp": "math.exp", "ln": "math.log", "log": "math.log",
                "sinh": "math.sinh", "cosh": "math.cosh", "tanh": "math.tanh",
                "arcsin": "math.asin", "arccos": "math.acos", "arctan": "math.atan"}
_LATEX_CONSTS = {"pi": "math.pi", "infty": "math.inf"}
_PY_RESERVED_RENAME = {"lambda": "lam", "Lambda": "Lam", "in": "in_", "is": "is_"}
_LATEX_SPACING = re.compile(r"\\(?:displaystyle|textstyle|left|right|big|Big|bigg|Bigg)\b|\\[,;:! ]|\\quad|\\qquad|~")
_LATEX_WRAPPERS = re.compile(r"\\(?:mathrm|mathit|mathbf|operatorname|text|textrm|boldsymbol|vec|hat|bar)\s*\{([^{}]*)\}")


class _LatexParseError(ValueError):
    pass


def _clean_latex(src: str) -> str:
    s = html.unescape(src or "")
    s = _LATEX_SPACING.sub(" ", s)
    prev = None
    while prev != s:
        prev = s
        s = _LATEX_WRAPPERS.sub(r"\1", s)
    s = re.sub(r"\\dot\s*\{([^{}]*)\}", r"\1_dot", s)
    s = (s.replace("\\cdot", "*").replace("\\times", "*").replace("\\approx", "=")
          .replace("\\simeq", "=").replace("\\equiv", "=").replace("−", "-"))
    s = s.strip().rstrip(".,;")
    return s


def _tokenize(s: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
        elif ch == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if not m:
                i += 2  # escaped symbol like \{ -> ignore
                continue
            tokens.append(("cmd", m.group(1)))
            i += len(m.group(0))
        elif ch.isdigit() or (ch == "." and i + 1 < n and s[i + 1].isdigit()):
            m = re.match(r"\d*\.?\d+", s[i:])
            tokens.append(("num", m.group(0)))
            i += len(m.group(0))
        elif ch.isalpha():
            m = re.match(r"[A-Za-z]+", s[i:])
            tokens.append(("id", m.group(0)))
            i += len(m.group(0))
        elif ch in "+-*/^_{}()[]=,'":
            tokens.append(("op", ch))
            i += 1
        else:
            raise _LatexParseError(f"unsupported character {ch!r}")
    return tokens


class _LatexParser:
    """Tiny recursive-descent LaTeX math parser emitting Python expressions."""

    def __init__(self, tokens: List[Tuple[str, str]]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self) -> Optional[Tuple[str, str]]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> Tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise _LatexParseError("unexpected end of expression")
        self.i += 1
        return tok

    def expect(self, op: str) -> None:
        tok = self.take()
        if tok != ("op", op):
            raise _LatexParseError(f"expected {op!r}, got {tok!r}")

    def parse(self) -> str:
        out = self.expr()
        if self.peek() is not None:
            raise _LatexParseError(f"trailing tokens at {self.peek()!r}")
        return out

    def expr(self, closers: Sequence[str] = ()) -> str:
        parts: List[str] = []
        sign = ""
        tok = self.peek()
        if tok in (("op", "-"), ("op", "+")):
            self.take()
            sign = "-" if tok[1] == "-" else ""
        parts.append(sign + self.term(closers))
        while True:
            tok = self.peek()
            if tok in (("op", "+"), ("op", "-")):
                self.take()
                parts.append(f" {tok[1]} " + self.term(closers))
            else:
                break
        return "".join(parts)

    def _at_term_end(self, closers: Sequence[str]) -> bool:
        tok = self.peek()
        if tok is None:
            return True
        return tok[0] == "op" and tok[1] in ("+", "-", "}", ")", "]", "=", ",", *closers)

    def term(self, closers: Sequence[str] = ()) -> str:
        out = self.factor()
        while not self._at_term_end(closers):
            tok = self.peek()
            if tok == ("op", "*"):
                self.take()
                out += " * " + self.factor()
            elif tok == ("op", "/"):
                self.take()
                rhs = self.factor()
                out += " / " + (rhs if _is_atomic(rhs) else f"({rhs})")
            else:  # implicit multiplication
                out += " * " + self.factor()
        return out

    def factor(self) -> str:
        base, base_raw = self.atom()
        if self.peek() == ("op", "^"):
            self.take()
            exponent, _ = self.atom(power_operand=True)
            if base_raw == "e":
                return f"math.exp({exponent})"
            if not _is_atomic(base):
                base = f"({base})"
            return f"math.pow({base}, {exponent})"
        return base

    def _subscript(self) -> str:
        if self.peek() != ("op", "_"):
            return ""
        self.take()
        tok = self.take()
        if tok == ("op", "{"):
            depth, raw = 1, []
            while True:
                t = self.take()
                if t == ("op", "{"):
                    depth += 1
                elif t == ("op", "}"):
                    depth -= 1
                    if depth == 0:
                        break
                raw.append(t[1])
            text = "".join(raw)
        else:
            text = tok[1]
        text = re.sub(r"\W+", "", text)
        return f"_{text}" if text else ""

    def _group(self, open_c: str, close_c: str) -> str:
        self.expect(open_c)
        inner = self.expr(closers=(close_c,))
        self.expect(close_c)
        return inner

    def atom(self, power_operand: bool = False) -> Tuple[str, str]:
        tok = self.take()
        kind, val = tok
        if kind == "num":
            return val, val
        if kind == "op" and val == "{":
            self.i -= 1
            inner = self._group("{", "}")
            return (inner if _is_atomic(inner) else f"({inner})"), inner
        if kind == "op" and val in "([":
            close = ")" if val == "(" else "]"
            self.i -= 1
            inner = self._group(val, close)
            return f"({inner})", inner
        if kind == "op" and val == "-" and power_operand:
            inner, _ = self.atom(power_operand=True)
            return f"-{inner}", inner
        if kind == "id":
            name = val
            if name == "e" and self.peek() != ("op", "_"):
                if self.peek() == ("op", "^"):
                    return "e", "e"
                return "math.e", "e"
            name += self._subscript()
            while self.peek() == ("op", "'"):
                self.take()
                name += "_prime"
            name = _PY_RESERVED_RENAME.get(name, name)
            return name, name
        if kind == "cmd":
            if val == "frac":
                num = self._group("{", "}")
                den = self._group("{", "}")
                return f"({num}) / ({den})", ""
            if val == "sqrt":
                if self.peek() == ("op", "["):
                    root = self._group("[", "]")
                    arg = self._group("{", "}")
                    return f"math.pow({arg}, 1.0 / ({root}))", ""
                arg = self._group("{", "}")
                return f"math.sqrt({arg})", ""
            if val in _LATEX_FUNCS:
                fn = _LATEX_FUNCS[val]
                power = None
                if self.peek() == ("op", "^"):
                    self.take()
                    power, _ = self.atom(power_operand=True)
                arg = self.factor()
                arg = arg[1:-1] if _wrapped_in_parens(arg) else arg
                call = f"{fn}({arg})"
                return (f"math.pow({call}, {power})" if power else call), ""
            if val in _LATEX_CONSTS:
                return _LATEX_CONSTS[val], val
            if val in ("Delta", "delta", "partial") and self.peek() and self.peek()[0] in ("id", "cmd"):
                nxt, _ = self.atom()
                prefix = {"Delta": "Delta_", "delta": "delta_", "partial": "d_"}[val]
                return prefix + nxt, prefix + nxt
            name = val + self._subscript()
            name = _PY_RESERVED_RENAME.get(name, name)
            return name, name
        raise _LatexParseError(f"unexpected token {tok!r}")


def _wrapped_in_parens(s: str) -> bool:
    if not (s.startswith("(") and s.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(s) - 1:
            return False
    return True


def _is_atomic(expr: str) -> bool:
    if _wrapped_in_parens(expr):
        return True
    return re.fullmatch(r"[\w.]+(\(.*\))?", expr) is not None and (
        "(" not in expr or _wrapped_in_parens(expr[expr.index("("):]))


def latex_to_python(latex: str) -> str:
    """Convert a LaTeX math expression to a Python expression (``""`` on failure).

    ``\\frac``, ``\\sqrt``, ``^``, ``\\exp``/``\\sin``/``\\cos``/``\\ln`` and
    ``\\pi`` are lowered to ``math.*`` calls (``math.pow`` instead of ``**``)
    so the hint is directly usable in a ``v9_transpiler_friendly`` kernel.
    """
    try:
        expr = _LatexParser(_tokenize(_clean_latex(latex))).parse()
        ast.parse(expr, mode="eval")
        return expr
    except (_LatexParseError, SyntaxError, ValueError, IndexError):
        return ""


def _hint_variables(hint: str) -> List[str]:
    if not hint:
        return []
    try:
        tree = ast.parse(hint, mode="eval")
    except SyntaxError:
        return []
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id != "math" and node.id not in names:
            names.append(node.id)
    return names


_DERIVATIVE_LHS = re.compile(
    r"^\\frac\s*\{\s*(?:\\mathrm\{d\}|d|\\partial)\s*([^{}]*?)\s*\}\s*"
    r"\{\s*(?:\\mathrm\{d\}|d|\\partial)\s*([^{}]*?)\s*\}$")


def _split_top_level_eq(latex: str) -> Optional[Tuple[str, str]]:
    depth = 0
    for i, ch in enumerate(latex):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == "=" and depth == 0:
            rhs = latex[i + 1:]
            # chained equalities a = b = c: keep the first right-hand side
            j = _top_level_index(rhs, "=")
            return latex[:i].strip(), (rhs[:j] if j >= 0 else rhs).strip()
    return None


def _top_level_index(s: str, target: str) -> int:
    depth = 0
    for i, ch in enumerate(s):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == target and depth == 0:
            return i
    return -1


def _lhs_name(lhs_latex: str) -> str:
    cleaned = _clean_latex(lhs_latex)
    m = _DERIVATIVE_LHS.match(cleaned)
    if m:
        num = re.sub(r"\\", "", m.group(1)).strip() or "y"
        den = re.sub(r"\\", "", m.group(2)).strip() or "t"
        return f"d{num}/d{den}"
    hint = latex_to_python(cleaned)
    if hint and re.fullmatch(r"[\w.]+", hint):
        return hint
    return re.sub(r"\s+", " ", cleaned)[:80]


def parse_equation(latex: str, *, source: str = "",
                   description: str = "") -> Optional[ExtractedEquation]:
    """Parse ``lhs = rhs`` LaTeX into an :class:`ExtractedEquation`."""
    cleaned = _clean_latex(latex)
    if re.search(r"\\(?:le|leq|ge|geq|lt|gt|neq)\b|[<>]", cleaned):
        return None  # inequalities are invariants, not equations
    split = _split_top_level_eq(cleaned)
    if split is None:
        return None
    lhs, rhs = split
    if not lhs or not rhs:
        return None
    hint = latex_to_python(rhs)
    return ExtractedEquation(
        lhs=_lhs_name(lhs), rhs_latex=rhs.strip(), rhs_python_hint=hint,
        variables=_hint_variables(hint), description=description[:300],
        source=source)


# ---------------------------------------------------------------------------
# Raw-source equation/parameter extraction
# ---------------------------------------------------------------------------

_WIKI_MATH_TAG = re.compile(r"<math[^>]*>(.*?)</math>", re.S | re.I)
_AR5IV_MATH = re.compile(r"<math\b([^>]*)>", re.I)
_ALTTEXT = re.compile(r'alttext="([^"]+)"')
_INLINE_TEX = re.compile(r"\$\$?([^$]{3,400}?)\$\$?")

_NUMBER = (r"[-+−]?\d+(?:[.,]\d+)?(?:[eE][-+−]?\d+)?"
           r"(?:\s*(?:×|x|\*|\\times)\s*10\s*\^?\s*\{?\s*[-+−]?\d+\s*\}?)?")
_UNIT_TOKEN = (r"(?:[A-Za-zμµ°]+(?:\^?-?[0-9²³])?"
               r"(?:\s*/\s*\(\s*[A-Za-z]+\s*[*·.\s]\s*[A-Za-z]+\s*\)"
               r"|\s*/\s*[A-Za-zμµ]+(?:\^?-?[0-9²³])?)?|°)")
_PARAM_ASSIGN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_\- ]{0,48}?)\s*(?:=|≈|~|:|\bof\b|\bis\b)\s*"
    r"(?P<num>" + _NUMBER + r")\s*(?P<unit>" + _UNIT_TOKEN + r")?")
_PARAM_WITH_UNIT = re.compile(
    r"(?P<num>" + _NUMBER + r")\s*(?P<unit>" + _UNIT_TOKEN + r")")
_NAME_FILLER = {"the", "a", "an", "of", "is", "and", "with", "at", "to", "for",
                "in", "on", "by", "about", "approximately", "roughly", "around",
                "value", "equal", "equals", "be", "was", "were", "are", "its"}


def _parse_number(text: str) -> Optional[float]:
    t = text.replace("−", "-").replace(",", ".").replace(" ", "")
    m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
                     r"(?:(?:×|x|\*|\\times)10\^?\{?([-+]?\d+)\}?)?", t)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2):
        value *= 10.0 ** int(m.group(2))
    return value


def _snake(text: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", text.lower())
             if w not in _NAME_FILLER]
    return "_".join(words[-4:])


def _clean_wikitext(text: str) -> str:
    s = re.sub(r"\{\{\s*(?:val|value)\s*\|\s*([^|}]+)(?:\|[^}]*?u\s*=\s*([^|}]+))?[^}]*\}\}",
               lambda m: f"{m.group(1)} {m.group(2) or ''}", text or "")
    s = re.sub(r"\{\{\s*convert\s*\|\s*([^|}]+)\|\s*([^|}]+)[^}]*\}\}",
               r"\1 \2", s)
    s = re.sub(r"<sup>\s*([^<]+?)\s*</sup>", r"^\1", s)
    s = re.sub(r"<sub>\s*([^<]+?)\s*</sub>", r"_\1", s)
    s = re.sub(r"'''?|\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", lambda m: m.group(1) or "", s)
    s = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    return s


def _iter_math_templates(text: str) -> Iterable[str]:
    """Yield the body of every ``{{math|...}}`` template.

    LaTeX bodies contain single braces (``\frac{M}{r^2}``), so the template
    close ``}}`` is only recognised once the LaTeX brace depth is back to 0
    (otherwise ``...r^2}}}`` would be truncated mid-expression).
    """
    for m in re.finditer(r"\{\{\s*math\s*\|", text or "", re.I):
        j, n = m.end(), len(text or "")
        latex_depth = 0
        end = -1
        while j < n:
            if text.startswith("}}", j) and latex_depth == 0:
                end = j
                break
            ch = text[j]
            if ch == "{":
                latex_depth += 1
            elif ch == "}" and latex_depth > 0:
                latex_depth -= 1
            j += 1
        body = text[m.end():end if end >= 0 else n]
        body = body.replace("{{=}}", "=")
        yield _clean_wikitext(body)


def extract_wikitext_math(wikitext: str, limit: int = 12) -> List[str]:
    """Explicit ``<math>`` blocks and ``{{math|...}}`` templates from wikitext."""
    found: List[str] = []
    for m in _WIKI_MATH_TAG.finditer(wikitext or ""):
        body = m.group(1).strip()
        if body and body not in found:
            found.append(body)
    for body in _iter_math_templates(wikitext or ""):
        body = body.strip()
        if body and body not in found:
            found.append(body)
    return found[:limit]


def extract_ar5iv_equations(page_html: str, max_equations: int = 8) -> List[str]:
    """Display equations (``alttext``) from an ar5iv HTML5 rendering.

    Block / ``ltx_equation`` math is returned first, then any remaining
    ``<math alttext=...>`` equations containing ``=``.
    """
    block: List[str] = []
    inline: List[str] = []
    for m in _AR5IV_MATH.finditer(page_html or ""):
        attrs = m.group(1)
        alt = _ALTTEXT.search(attrs)
        if not alt:
            continue
        tex = html.unescape(alt.group(1)).strip()
        if "=" not in tex:
            continue
        context = page_html[max(0, m.start() - 300):m.start()]
        is_block = 'display="block"' in attrs or "ltx_equation" in context[-300:]
        target = block if is_block else inline
        if tex not in block and tex not in inline:
            target.append(tex)
    return (block + inline)[:max_equations]


def extract_infobox_parameters(wikitext: str, source: str = "Wikipedia",
                               limit: int = 24) -> List[ExtractedParameter]:
    """Structured ``| parameter_name = number unit`` infobox pairs."""
    params: List[ExtractedParameter] = []
    for line in _clean_wikitext(wikitext or "").splitlines():
        m = re.match(r"^\s*\|\s*([A-Za-z][\w ]{0,40}?)\s*=\s*(" + _NUMBER + r")\s*(" +
                     _UNIT_TOKEN + r")?", line)
        if not m:
            continue
        value = _parse_number(m.group(2))
        if value is None:
            continue
        unit = (m.group(3) or "").strip()
        if unit and not is_known_unit(unit):
            unit = ""
        si_value, si_unit = normalize_to_si(value, unit)
        name = _snake(m.group(1)) or "param"
        params.append(ExtractedParameter(
            name=name, symbol="", raw_value=value, raw_unit=unit,
            si_value=si_value, si_unit=si_unit, source=source, confidence=0.8))
        if len(params) >= limit:
            break
    return params


def extract_parameters_from_text(text: str, source: str = "",
                                 confidence: float = 0.6,
                                 limit: int = 24) -> List[ExtractedParameter]:
    """Free-text ``name = number unit`` / ``name number unit`` pairs."""
    text = _clean_wikitext(text or "")
    params: List[ExtractedParameter] = []
    seen_spans: List[Tuple[int, int]] = []

    def _add(name_raw: str, num_raw: str, unit_raw: str, span: Tuple[int, int]) -> None:
        if any(a <= span[0] < b for a, b in seen_spans):
            return
        value = _parse_number(num_raw)
        if value is None:
            return
        unit = (unit_raw or "").strip()
        if unit and not is_known_unit(unit):
            # A trailing word was captured as a unit; keep only if dimensionless.
            unit = ""
        name = _snake(name_raw)
        if not name:
            return
        si_value, si_unit = normalize_to_si(value, unit)
        symbol = name_raw.strip().split()[-1] if name_raw.strip() else ""
        params.append(ExtractedParameter(
            name=name, symbol=symbol if len(symbol) <= 8 else "",
            raw_value=value, raw_unit=unit, si_value=si_value, si_unit=si_unit,
            source=source, confidence=confidence))
        seen_spans.append(span)

    for m in _PARAM_ASSIGN.finditer(text):
        _add(m.group("name"), m.group("num"), m.group("unit") or "", m.span())
    for m in _PARAM_WITH_UNIT.finditer(text):
        unit = m.group("unit")
        if not is_known_unit(unit):
            continue
        before = text[max(0, m.start() - 48):m.start()]
        before = re.split(r"[.;,()\n]", before)[-1]
        _add(before, m.group("num"), unit, m.span())
    return params[:limit]


# ---------------------------------------------------------------------------
# Offline multi-domain reference registry (7 quantitative domains)
# ---------------------------------------------------------------------------

def _p(name: str, symbol: str, value: float, unit: str, source: str,
       confidence: float = 1.0, requires: Tuple[str, ...] = ()) -> Dict[str, Any]:
    return {"name": name, "symbol": symbol, "value": value, "unit": unit,
            "source": source, "confidence": confidence, "requires": requires}


DOMAIN_REFERENCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fundamental_constants": {
        "keywords": ("codata", "fundamental constant", "speed of light", "planck",
                     "boltzmann", "avogadro", "gravitational constant", "stefan",
                     "black body", "blackbody", "radiation", "photon", "quantum",
                     "electron", "proton", "permittivity", "permeability"),
        "parameters": [
            _p("speed_of_light", "c", 299792458.0, "m/s", "CODATA 2018"),
            _p("gravitational_constant", "G", 6.67430e-11, "m^3/(kg s^2)", "CODATA 2018"),
            _p("reduced_planck_constant", "hbar", 1.054571817e-34, "J s", "CODATA 2018"),
            _p("boltzmann_constant", "k_B", 1.380649e-23, "J/K", "CODATA 2018"),
            _p("avogadro_constant", "N_A", 6.02214076e23, "1/mol", "CODATA 2018"),
            _p("molar_gas_constant", "R", 8.314462618, "J/(mol K)", "CODATA 2018"),
            _p("stefan_boltzmann_constant", "sigma_SB", 5.670374419e-8, "W/(m^2 K^4)", "CODATA 2018"),
            _p("elementary_charge", "e", 1.602176634e-19, "C", "CODATA 2018"),
            _p("proton_mass", "m_p", 1.67262192369e-27, "kg", "CODATA 2018"),
            _p("electron_mass", "m_e", 9.1093837015e-31, "kg", "CODATA 2018"),
            _p("vacuum_permeability", "mu_0", 1.25663706212e-6, "N/A^2", "CODATA 2018"),
            _p("vacuum_permittivity", "epsilon_0", 8.8541878128e-12, "F/m", "CODATA 2018"),
        ],
        "equations": [
            ("E = m c^2", "Mass-energy equivalence"),
            ("F = G \\frac{m_1 m_2}{r^2}", "Newtonian gravitation"),
            ("q_{rad} = \\epsilon \\sigma_{SB} T^4", "Stefan-Boltzmann radiative flux"),
        ],
        "benchmarks": [],
        "invariants": ["telemetry['final_max'] < 1e12"],
    },
    "aerospace_entry": {
        "keywords": ("mars", "martian", "venus", "titan", "entry", "re-entry", "reentry",
                     "aeroshell", "heat shield", "heatshield", "heat flux", "sutton",
                     "ballistic", "aerobrak", "deceleration", "hypersonic",
                     "trajectory", "aerocapture", "stagnation", "atmospheric"),
        "parameters": [
            _p("earth_surface_gravity", "g_E", 9.80665, "m/s^2", "ISO 80000 standard gravity"),
            _p("earth_scale_height", "H", 7200.0, "m", "Exponential atmosphere fit (entry regime)",
               0.8, ("earth",)),
            _p("earth_surface_density", "rho_0", 1.225, "kg/m^3", "US Standard Atmosphere 1976",
               1.0, ("earth",)),
            _p("earth_sutton_graves_k", "k", 1.7415e-4, "kg^0.5/m", "Sutton & Graves (1971)",
               0.9, ("earth",)),
            _p("mars_surface_gravity", "g", 3.71, "m/s^2", "NASA Mars fact sheet", 1.0, ("mars", "martian")),
            _p("mars_scale_height", "H", 11.1, "km", "NASA Mars fact sheet", 1.0, ("mars", "martian")),
            _p("mars_surface_density", "rho_0", 0.020, "kg/m^3", "NASA Mars fact sheet",
               1.0, ("mars", "martian")),
            _p("mars_sutton_graves_k", "k", 1.9027e-4, "kg^0.5/m", "Sutton & Graves (1971), CO2/N2",
               0.9, ("mars", "martian")),
            _p("venus_surface_gravity", "g", 8.87, "m/s^2", "NASA Venus fact sheet", 1.0, ("venus",)),
            _p("venus_scale_height", "H", 15.9, "km", "NASA Venus fact sheet", 1.0, ("venus",)),
            _p("venus_surface_density", "rho_0", 65.0, "kg/m^3", "NASA Venus fact sheet", 1.0, ("venus",)),
            _p("titan_surface_gravity", "g", 1.352, "m/s^2", "NASA Titan fact sheet", 1.0, ("titan",)),
            _p("titan_surface_density", "rho_0", 5.4, "kg/m^3", "Huygens HASI", 0.8, ("titan",)),
        ],
        "equations": [
            ("q_s = k \\sqrt{\\frac{\\rho}{R_n}} v^3", "Sutton-Graves stagnation-point convective heat flux [W/m^2]"),
            ("\\rho = \\rho_0 e^{-h/H}", "Exponential atmosphere density"),
            ("\\frac{dv}{dt} = -\\frac{1}{2}\\rho v^2 \\frac{C_D A}{m} - g \\sin\\gamma",
             "Planar ballistic entry: along-track deceleration"),
            ("\\frac{dh}{dt} = v \\sin\\gamma", "Altitude rate"),
            ("a_{max} = \\frac{v_e^2 \\sin\\gamma_e}{2 e H}", "Allen-Eggers analytical peak deceleration [m/s^2]"),
        ],
        "benchmarks": [
            {"name": "Allen-Eggers velocity ratio at peak deceleration",
             "input_regime": "ballistic entry, constant flight-path angle",
             "expected_value": 0.60653, "unit": "v/v_e", "tolerance": 0.02,
             "source": "Allen & Eggers (1958)"},
        ],
        "invariants": [
            "peak heat flux in MW/m^2 (divide W/m^2 by 1e6) < 10.0",
            "peak deceleration in g_E (divide m/s^2 by 9.80665) < 40.0",
            "peak_*/max_* slots are running maxima: if cur > buf.data[i]: buf.data[i] = cur",
            "q_peak_mw_m2 < 0.5 * rho0 * v0**3 / 1e6 (free-stream energy flux bound)",
        ],
    },
    "relativistic_plasma": {
        "keywords": ("relativistic", "lorentz", "plasma", "magnetic sail", "magsail",
                     "larmor", "gyro", "interstellar", "ism", "solar wind",
                     "cosmic ray", "fraction of c", "speed of light"),
        "parameters": [
            _p("speed_of_light", "c", 299792458.0, "m/s", "CODATA 2018"),
            _p("proton_mass", "m_p", 1.67262192369e-27, "kg", "CODATA 2018"),
            _p("elementary_charge", "q", 1.602176634e-19, "C", "CODATA 2018"),
            _p("ism_proton_density", "n_ISM", 1.0, "cm^-3", "Local ISM (order of magnitude)", 0.7),
            _p("solar_wind_density_1au", "n_sw", 5.0, "cm^-3", "Solar wind at 1 AU (typical)", 0.7),
            _p("interplanetary_field_1au", "B_IMF", 5.0, "nT", "IMF at 1 AU (typical)", 0.7),
        ],
        "equations": [
            ("\\gamma = \\frac{1}{\\sqrt{1 - \\frac{v^2}{c^2}}}", "Lorentz factor"),
            ("r_L = \\frac{\\gamma m v}{q B}", "Relativistic Larmor (gyro) radius"),
            ("F_{drag} = \\frac{1}{2} \\rho v^2 C_D A", "Plasma ram-pressure drag scaling"),
        ],
        "benchmarks": [
            {"name": "Lorentz factor at 0.25 c", "input_regime": "v = 0.25 c",
             "expected_value": 1.032796, "unit": "", "tolerance": 1e-4,
             "source": "special relativity (exact)"},
        ],
        "invariants": ["v < c (speed of light) at every step",
                       "Lorentz factor gamma >= 1.0"],
    },
    "materials_thermophysics": {
        "keywords": ("tungsten", "carbide", "pica", "phenolic", "titanium", "sic",
                     "silicon carbide", "ablat", "ablator", "shield material",
                     "thermal protection", "tps"),
        "parameters": [
            _p("tungsten_density", "rho_W", 19.3, "g/cm^3", "CRC Handbook"),
            _p("tungsten_melting_point", "T_m_W", 3695.0, "K", "CRC Handbook"),
            _p("tungsten_thermal_conductivity", "kappa_W", 173.0, "W/(m K)", "CRC Handbook"),
            _p("tungsten_specific_heat", "c_p_W", 132.0, "J/(kg K)", "CRC Handbook"),
            _p("tungsten_carbide_density", "rho_WC", 15.6, "g/cm^3", "CRC Handbook"),
            _p("tungsten_carbide_melting_point", "T_m_WC", 3058.0, "K", "CRC Handbook"),
            _p("titanium_density", "rho_Ti", 4.506, "g/cm^3", "CRC Handbook"),
            _p("titanium_melting_point", "T_m_Ti", 1941.0, "K", "CRC Handbook"),
            _p("titanium_thermal_conductivity", "kappa_Ti", 21.9, "W/(m K)", "CRC Handbook"),
            _p("titanium_specific_heat", "c_p_Ti", 523.0, "J/(kg K)", "CRC Handbook"),
            _p("silicon_carbide_density", "rho_SiC", 3.21, "g/cm^3", "CRC Handbook"),
            _p("pica_density", "rho_PICA", 0.27, "g/cm^3", "NASA Ames PICA (nominal)", 0.7),
            _p("carbon_phenolic_density", "rho_CP", 1.45, "g/cm^3", "Carbon-phenolic TPS (nominal)", 0.7),
        ],
        "equations": [
            ("q = -\\kappa \\frac{dT}{dx}", "Fourier conduction"),
            ("\\frac{dT}{dt} = \\frac{q_{in}}{\\rho c_p L}", "Lumped-capacitance wall heating"),
        ],
        "benchmarks": [],
        "invariants": ["wall temperature T < material melting/ablation limit T_max"],
    },
    "nonlinear_dynamics": {
        "keywords": ("duffing", "van der pol", "lorenz", "lotka", "volterra",
                     "predator", "kdv", "korteweg", "soliton", "burgers",
                     "advection", "diffusion", "cfl", "chaos", "chaotic",
                     "oscillator", "pendulum", "pde", "ode", "nonlinear"),
        "parameters": [
            _p("lorenz_sigma", "sigma", 10.0, "", "Lorenz (1963)"),
            _p("lorenz_rho", "rho", 28.0, "", "Lorenz (1963)"),
            _p("lorenz_beta", "beta", 8.0 / 3.0, "", "Lorenz (1963)"),
            _p("kdv_nonlinear_coeff", "a", 6.0, "", "u_t + 6 u u_x + u_xxx = 0"),
            _p("cfl_limit", "C_max", 1.0, "", "Courant-Friedrichs-Lewy (explicit upwind)"),
        ],
        "equations": [
            ("\\frac{dx}{dt} = \\sigma (y - x)", "Lorenz x"),
            ("\\frac{dy}{dt} = x (\\rho - z) - y", "Lorenz y"),
            ("\\frac{dz}{dt} = x y - \\beta z", "Lorenz z"),
            ("\\frac{dv}{dt} = \\gamma \\cos(\\omega t) - \\delta v - \\alpha x - \\beta x^3", "Duffing oscillator"),
            ("\\frac{dv}{dt} = \\mu (1 - x^2) v - x", "Van der Pol oscillator"),
            ("c = 2 A", "KdV soliton speed-amplitude relation"),
            ("C = \\frac{u \\Delta t}{\\Delta x}", "CFL number (stable for C <= 1)"),
        ],
        "benchmarks": [
            {"name": "KdV soliton speed/amplitude ratio", "input_regime": "u_t + 6uu_x + u_xxx = 0",
             "expected_value": 2.0, "unit": "", "tolerance": 0.02, "source": "Korteweg-de Vries"},
            {"name": "Lorenz largest Lyapunov exponent", "input_regime": "sigma=10, rho=28, beta=8/3",
             "expected_value": 0.9056, "unit": "1/time", "tolerance": 0.05, "source": "Sprott (2003)"},
        ],
        "invariants": ["u * dt / dx <= 1.0 (CFL)",
                       "abs(telemetry['energy_drift']) < 1e-3 for conservative systems"],
    },
    "chemical_kinetics": {
        "keywords": ("kinetic", "kinetics", "reaction", "arrhenius", "robertson",
                     "stiff", "michaelis", "menten", "enzyme", "combustion",
                     "methane", "catalys", "chemical", "rate constant", "autocatal"),
        "parameters": [
            _p("robertson_k1", "k_1", 0.04, "", "Robertson (1966)"),
            _p("robertson_k2", "k_2", 1.0e4, "", "Robertson (1966)"),
            _p("robertson_k3", "k_3", 3.0e7, "", "Robertson (1966)"),
            _p("molar_gas_constant", "R", 8.314462618, "J/(mol K)", "CODATA 2018"),
        ],
        "equations": [
            ("k = A e^{-E_a / (R T)}", "Arrhenius rate law"),
            ("\\frac{dy_1}{dt} = -k_1 y_1 + k_2 y_2 y_3", "Robertson y1"),
            ("\\frac{dy_2}{dt} = k_1 y_1 - k_2 y_2 y_3 - k_3 y_2^2", "Robertson y2"),
            ("\\frac{dy_3}{dt} = k_3 y_2^2", "Robertson y3"),
            ("v = \\frac{V_{max} S}{K_m + S}", "Michaelis-Menten rate"),
        ],
        "benchmarks": [
            {"name": "Robertson y1(t=40)", "input_regime": "y(0) = [1, 0, 0]",
             "expected_value": [0.7158270687, 9.185534765e-6, 0.2841637457],
             "unit": "", "tolerance": 1e-4, "source": "Hairer & Wanner (1996)"},
        ],
        "invariants": ["y1 + y2 + y3 == 1 (mass conservation, abs err < 1e-6)",
                       "all concentrations >= 0"],
    },
    "discrete_finance": {
        "keywords": ("fibonacci", "catalan", "prime", "recurrence", "combinator",
                     "integer sequence", "oeis", "binomial", "partition", "permutation",
                     "number theory", "black-scholes", "black scholes", "heston",
                     "option pric", "martingale", "fft", "cooley", "graph"),
        "parameters": [
            _p("golden_ratio", "phi", (1.0 + 5.0 ** 0.5) / 2.0, "", "Fibonacci asymptotics"),
        ],
        "equations": [
            ("F_n = F_{n-1} + F_{n-2}", "Fibonacci recurrence"),
            ("C_{n+1} = \\frac{2 (2 n + 1)}{n + 2} C_n", "Catalan recurrence"),
            ("T(N) = N \\log(N)", "Cooley-Tukey FFT complexity O(N log N)"),
            ("d_1 = \\frac{\\ln(S / K) + (r + \\sigma^2 / 2) T}{\\sigma \\sqrt{T}}", "Black-Scholes d1"),
        ],
        "benchmarks": [
            {"name": "Fibonacci initial terms (A000045)", "input_regime": "F_0 = 0, F_1 = 1",
             "expected_value": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34], "unit": "", "tolerance": 0.0,
             "source": "OEIS A000045"},
            {"name": "Catalan initial terms (A000108)", "input_regime": "C_0 = 1",
             "expected_value": [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862], "unit": "",
             "tolerance": 0.0, "source": "OEIS A000108"},
            {"name": "Black-Scholes ATM call", "input_regime": "S=100, K=100, r=0.05, sigma=0.2, T=1",
             "expected_value": 10.4506, "unit": "currency", "tolerance": 1e-3,
             "source": "Black & Scholes (1973)"},
        ],
        "invariants": ["integer sequence terms are exact (zero tolerance)",
                       "discounted asset price is a martingale: E[exp(-r T) S_T] == S_0"],
    },
}


def classify_registry_domains(text: str) -> List[str]:
    """Registry domains whose keywords occur in ``text`` (registry order)."""
    t = (text or "").lower()
    return [name for name, spec in DOMAIN_REFERENCE_REGISTRY.items()
            if any(k in t for k in spec["keywords"])]


def registry_evidence(text: str) -> Tuple[List[ExtractedParameter], List[ExtractedEquation],
                                           List[BenchmarkAnchor], List[str], List[str]]:
    """Parameters/equations/benchmarks/invariants for every matched domain."""
    t = (text or "").lower()
    domains = classify_registry_domains(text)
    params: List[ExtractedParameter] = []
    equations: List[ExtractedEquation] = []
    benchmarks: List[BenchmarkAnchor] = []
    invariants: List[str] = []
    for domain in domains:
        spec = DOMAIN_REFERENCE_REGISTRY[domain]
        for p in spec["parameters"]:
            if p["requires"] and not any(r in t for r in p["requires"]):
                continue
            si_value, si_unit = normalize_to_si(p["value"], p["unit"])
            params.append(ExtractedParameter(
                name=p["name"], symbol=p["symbol"], raw_value=float(p["value"]),
                raw_unit=p["unit"], si_value=si_value, si_unit=si_unit,
                source=f"ReferenceRegistry: {p['source']}", confidence=p["confidence"]))
        for latex, desc in spec["equations"]:
            eq = parse_equation(latex, source="ReferenceRegistry", description=desc)
            if eq is not None:
                equations.append(eq)
        for b in spec["benchmarks"]:
            benchmarks.append(BenchmarkAnchor(**b))
        invariants.extend(spec["invariants"])
    return params, equations, benchmarks, invariants, domains


# ---------------------------------------------------------------------------
# Deterministic backend & stepper recommender
# ---------------------------------------------------------------------------

_DISCRETE_KW = ("fibonacci", "catalan", "prime", "recurrence", "combinator",
                "integer sequence", "oeis", "binomial coefficient", "partition function",
                "permutation", "number theory", "lattice path", "modular arithmetic",
                "graph enumeration", "algebraic reduction")
_MULTI_KW = ("compare", "comparison", "competing", "versus", " vs ", " vs.",
             "alternative closure", "candidate model", "multiple model", "several model",
             "multiple closure", "rival", "which closure", "model selection",
             "solver comparison")
_STIFF_KW = ("stiff", "robertson", "kinetic", "reaction network", "combustion",
             "arrhenius", "autocatal", "chemical")
_HAMILTONIAN_KW = ("hamiltonian", "symplectic", "energy conservation", "conserve energy",
                   "conservative system", "orbit", "kepler", "n-body", "pendulum",
                   "harmonic oscillator", "molecular dynamics", "energy-conserving")
_STOCHASTIC_KW = ("monte carlo", "monte-carlo", "stochastic", "random walk", "brownian",
                  "heston", "option pric")


def recommend_execution_strategy(prompt: str,
                                 evidence: Optional[StructuredResearchEvidence] = None
                                 ) -> Dict[str, str]:
    """Route a research prompt to ``target_backend`` + ``stepper`` deterministically."""
    t = f" {(prompt or '').lower()} "
    domains = set(evidence.domains) if evidence is not None else set()
    if any(k in t for k in _DISCRETE_KW):
        return {"target_backend": "rust_hin", "stepper": "hin_exact_reduction",
                "rationale": "discrete / integer-recurrence workload: exact zero-heap "
                             "HIN arena reduction (no floating-point stepping)"}
    if any(k in t for k in _MULTI_KW):
        return {"target_backend": "multi_variant", "stepper": "rk4",
                "rationale": "competing closures / solver comparison: run every "
                             "candidate variant and rank them on a Pareto front"}
    if any(k in t for k in _STIFF_KW) or "chemical_kinetics" in domains:
        return {"target_backend": "cpp", "stepper": "semi_implicit_euler",
                "rationale": "stiff kinetics: semi-implicit / sub-stepped integration "
                             "for stability at practical dt"}
    if any(k in t for k in _HAMILTONIAN_KW):
        return {"target_backend": "cpp", "stepper": "symplectic_verlet",
                "rationale": "conservative Hamiltonian system: symplectic Verlet "
                             "bounds long-run energy drift"}
    if any(k in t for k in _STOCHASTIC_KW):
        return {"target_backend": "cpp", "stepper": "monte_carlo_xoshiro",
                "rationale": "stochastic model: deterministic xoshiro PRNG Monte Carlo"}
    if "discrete_finance" in domains and not (domains - {"discrete_finance"}):
        return {"target_backend": "rust_hin", "stepper": "hin_exact_reduction",
                "rationale": "discrete-mathematics evidence only"}
    return {"target_backend": "cpp", "stepper": "rk4",
            "rationale": "continuous ODE/PDE trajectory: classical RK4 via the C-ABI"}


# ---------------------------------------------------------------------------
# QuantitativeInsightExtractor
# ---------------------------------------------------------------------------

def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != 0.0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
            return f"{value:.6g}"
        return f"{value:.6g}"
    return str(value)


def _md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_evidence_table(evidence: StructuredResearchEvidence) -> str:
    """Markdown digest: SI constants, Python-ready equations, strategy, invariants."""
    lines = [f"**Domains:** {', '.join(evidence.domains) or 'general'}  ",
             f"**Recommended backend / stepper:** `{evidence.recommended_backend}` / "
             f"`{evidence.recommended_stepper}`", ""]
    if evidence.parameters:
        lines += ["| Parameter | Symbol | Raw | SI value | SI unit | Source |",
                  "|---|---|---|---|---|---|"]
        for p in evidence.parameters.values():
            lines.append(
                f"| {_md_cell(p.name)} | {_md_cell(p.symbol)} | {_fmt(p.raw_value)} "
                f"{_md_cell(p.raw_unit)} | {_fmt(p.si_value)} | {_md_cell(p.si_unit)} "
                f"| {_md_cell(p.source)} |")
        lines.append("")
    if evidence.equations:
        lines += ["| Equation | LaTeX | Python hint (v9) | Source |",
                  "|---|---|---|---|"]
        for e in evidence.equations:
            hint = f"`{e.lhs} = {e.rhs_python_hint}`" if e.rhs_python_hint else "—"
            lines.append(f"| {_md_cell(e.description or e.lhs)} | `{_md_cell(e.rhs_latex)}` "
                         f"| {_md_cell(hint)} | {_md_cell(e.source)} |")
        lines.append("")
    if evidence.benchmarks:
        lines += ["| Benchmark | Regime | Expected | Tolerance | Source |",
                  "|---|---|---|---|---|"]
        for b in evidence.benchmarks:
            lines.append(f"| {_md_cell(b.name)} | {_md_cell(b.input_regime)} | "
                         f"{_md_cell(b.expected_value)} {_md_cell(b.unit)} | {b.tolerance:g} "
                         f"| {_md_cell(b.source)} |")
        lines.append("")
    if evidence.suggested_invariants:
        lines.append("**Suggested invariants:**")
        lines += [f"- {inv}" for inv in evidence.suggested_invariants]
    return "\n".join(lines).strip()


class QuantitativeInsightExtractor:
    """Deterministic literature -> :class:`StructuredResearchEvidence` transform."""

    def __init__(self, *, max_equations: int = 24, max_parameters: int = 48,
                 use_registry: bool = True) -> None:
        self.max_equations = max_equations
        self.max_parameters = max_parameters
        self.use_registry = use_registry

    # -- per-source helpers -------------------------------------------------
    def equations_from_wikitext(self, wikitext: str, source: str = "Wikipedia-Math"
                                ) -> List[ExtractedEquation]:
        out = []
        for tex in extract_wikitext_math(wikitext):
            eq = parse_equation(tex, source=source)
            if eq is not None:
                out.append(eq)
        return out

    def equations_from_ar5iv(self, page_html: str, source: str = "ar5iv",
                             max_equations: int = 8) -> List[ExtractedEquation]:
        out = []
        for tex in extract_ar5iv_equations(page_html, max_equations):
            eq = parse_equation(tex, source=source)
            if eq is not None:
                out.append(eq)
        return out

    def equations_from_text(self, text: str, source: str = "") -> List[ExtractedEquation]:
        out = []
        for m in _INLINE_TEX.finditer(text or ""):
            eq = parse_equation(m.group(1), source=source)
            if eq is not None:
                out.append(eq)
        return out

    # -- main entry point ---------------------------------------------------
    def extract(self, prompt: str, search_results: Optional[Sequence[Any]] = None
                ) -> StructuredResearchEvidence:
        equations: List[ExtractedEquation] = []
        params: Dict[str, ExtractedParameter] = {}
        benchmarks: List[BenchmarkAnchor] = []
        invariants: List[str] = []
        corpus = [prompt or ""]

        def add_param(p: ExtractedParameter) -> None:
            current = params.get(p.name)
            if current is None or p.confidence > current.confidence:
                params[p.name] = p

        # 1. Explicit user-prompt quantities are authoritative (verbatim).
        for p in extract_parameters_from_text(prompt or "", source="user_prompt",
                                              confidence=1.0):
            add_param(p)

        # 2. Literature hits from every connector.
        for item in search_results or []:
            src = str(_get(item, "badge") or _get(item, "source") or "literature")
            title = str(_get(item, "title") or "")
            summary = str(_get(item, "summary") or "")
            corpus.append(f"{title} {summary}")
            for tex in _get(item, "equations", None) or []:
                eq = parse_equation(str(tex), source=src, description=title[:120])
                if eq is not None:
                    equations.append(eq)
            equations.extend(self.equations_from_text(summary, source=src))
            for key, raw in (_get(item, "constants", None) or {}).items():
                p = _constant_to_parameter(str(key), raw, src)
                if p is not None:
                    add_param(p)
            for p in extract_parameters_from_text(summary, source=src, confidence=0.6):
                add_param(p)
            for b in _get(item, "benchmarks", None) or []:
                try:
                    benchmarks.append(b if isinstance(b, BenchmarkAnchor)
                                      else BenchmarkAnchor.model_validate(b))
                except Exception:  # noqa: BLE001 - malformed connector payload
                    continue

        # 3. Curated offline registry for every matched domain.
        domains: List[str] = []
        if self.use_registry:
            r_params, r_eqs, r_bench, r_inv, domains = registry_evidence(" ".join(corpus))
            for p in r_params:
                if p.name not in params:
                    params[p.name] = p
            equations.extend(r_eqs)
            benchmarks.extend(r_bench)
            invariants.extend(r_inv)
        else:
            domains = classify_registry_domains(" ".join(corpus))

        # Deduplicate equations by (lhs, rhs) and benchmarks by name.
        seen_eq, uniq_eqs = set(), []
        for eq in equations:
            key = (eq.lhs, re.sub(r"\s+", "", eq.rhs_latex))
            if key not in seen_eq:
                seen_eq.add(key)
                uniq_eqs.append(eq)
        seen_b, uniq_b = set(), []
        for b in benchmarks:
            if b.name not in seen_b:
                seen_b.add(b.name)
                uniq_b.append(b)
        invariants = list(dict.fromkeys(invariants))

        draft = StructuredResearchEvidence(
            domains=domains, equations=uniq_eqs[: self.max_equations],
            parameters=dict(list(params.items())[: self.max_parameters]),
            benchmarks=uniq_b, suggested_invariants=invariants)
        strategy = recommend_execution_strategy(prompt or "", draft)
        draft = draft.model_copy(update={
            "recommended_backend": strategy["target_backend"],
            "recommended_stepper": strategy["stepper"]})
        return draft.model_copy(update={"summary_markdown_table": render_evidence_table(draft)})


def _constant_to_parameter(key: str, raw: Any, source: str) -> Optional[ExtractedParameter]:
    """Connector ``constants`` entry (number or ``"number unit"``) -> parameter."""
    if isinstance(raw, bool):
        return None
    unit = ""
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, dict) and "value" in raw:
        try:
            value = float(raw["value"])
        except (TypeError, ValueError):
            return None
        unit = str(raw.get("unit") or "")
    elif isinstance(raw, str):
        m = re.match(r"^\s*(" + _NUMBER + r")\s*(.*)$", raw)
        if not m:
            return None
        value = _parse_number(m.group(1))
        if value is None:
            return None
        unit = m.group(2).strip()
    else:
        return None
    if unit and not is_known_unit(unit):
        si_value, si_unit = value, unit
    else:
        si_value, si_unit = normalize_to_si(value, unit)
    name = _snake(key) or key
    return ExtractedParameter(name=name, symbol="", raw_value=value, raw_unit=unit,
                              si_value=si_value, si_unit=si_unit, source=source,
                              confidence=0.8)


def merge_evidence(base: Optional[StructuredResearchEvidence],
                   extra: Optional[StructuredResearchEvidence]) -> Optional[StructuredResearchEvidence]:
    """Union of two evidence objects (``base`` wins on parameter conflicts)."""
    if base is None:
        return extra
    if extra is None:
        return base
    params = dict(extra.parameters)
    params.update(base.parameters)
    eq_keys = {(e.lhs, e.rhs_latex) for e in base.equations}
    equations = list(base.equations) + [e for e in extra.equations
                                        if (e.lhs, e.rhs_latex) not in eq_keys]
    b_names = {b.name for b in base.benchmarks}
    benchmarks = list(base.benchmarks) + [b for b in extra.benchmarks if b.name not in b_names]
    merged = base.model_copy(update={
        "domains": list(dict.fromkeys(list(base.domains) + list(extra.domains))),
        "equations": equations, "parameters": params, "benchmarks": benchmarks,
        "suggested_invariants": list(dict.fromkeys(
            list(base.suggested_invariants) + list(extra.suggested_invariants)))})
    return merged.model_copy(update={"summary_markdown_table": render_evidence_table(merged)})


def evidence_si_constants(evidence: Optional[StructuredResearchEvidence]) -> Dict[str, float]:
    """``{name: si_value}`` for ``ResearchBlueprint.extracted_constants``."""
    if evidence is None:
        return {}
    return {name: float(p.si_value) for name, p in evidence.parameters.items()}

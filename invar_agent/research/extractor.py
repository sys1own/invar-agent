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


def _electrical_unit(unit):
    bases = {"F", "S", "V", "A", "C", "Ohm"}
    prefixes = {"": 1.0, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6}
    match = re.fullmatch(r"([pnumkM]?)(F|S|V|A|C|Ohm)(?:/(m|cm|mm)(?:\^(\d+))?)?", unit)
    if not match or match[2] not in bases:
        return None
    prefix, base, length, power = match.groups()
    factor = prefixes[prefix]
    if length:
        exponent = int(power or 1)
        factor /= {"m": 1.0, "cm": 1e-2, "mm": 1e-3}[length] ** exponent
        base += "/m" + (f"^{exponent}" if exponent != 1 else "")
    return factor, base

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
    "m/s": (1.0, "m/s"), "m/min": (1.0 / 60.0, "m/s"), "m/h": (1.0 / 3600.0, "m/s"),
    "km/s": (1e3, "m/s"), "km/h": (1.0 / 3.6, "m/s"),
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
    "W/(m^2*K)": (1.0, "W/(m^2*K)"), "W/(m^2 K)": (1.0, "W/(m^2*K)"),
    "J/(kg*K)": (1.0, "J/(kg*K)"), "J/(kg K)": (1.0, "J/(kg*K)"),
    "J/kg/K": (1.0, "J/(kg*K)"), "kJ/(kg*K)": (1e3, "J/(kg*K)"),
    "1/s": (1.0, "1/s"), "s^-1": (1.0, "1/s"), "Hz": (1.0, "1/s"),
    # chemistry & pharmacology: molarity (1 M = 1 mol/L = 1e3 mol/m^3)
    "M": (1e3, "mol/m^3"), "mol/L": (1e3, "mol/m^3"),
    "mM": (1.0, "mol/m^3"), "mmol/L": (1.0, "mol/m^3"),
    "uM": (1e-3, "mol/m^3"), "umol/L": (1e-3, "mol/m^3"),
    "nM": (1e-6, "mol/m^3"), "nmol/L": (1e-6, "mol/m^3"),
    "pM": (1e-9, "mol/m^3"), "pmol/L": (1e-9, "mol/m^3"),
    "mol/m^3": (1.0, "mol/m^3"),
    # concentration rates (1 M/s = 1000 mol/(m^3*s), 1 M/min = 1000/60 mol/(m^3*s))
    "M/s": (1e3, "mol/(m^3*s)"), "M/min": (1000.0 / 60.0, "mol/(m^3*s)"),
    "M/h": (1000.0 / 3600.0, "mol/(m^3*s)"), "M/hr": (1000.0 / 3600.0, "mol/(m^3*s)"),
    "mM/s": (1.0, "mol/(m^3*s)"), "mM/min": (1.0 / 60.0, "mol/(m^3*s)"),
    "mM/h": (1.0 / 3600.0, "mol/(m^3*s)"), "mM/hr": (1.0 / 3600.0, "mol/(m^3*s)"),
    "uM/s": (1e-3, "mol/(m^3*s)"), "uM/min": (1e-3 / 60.0, "mol/(m^3*s)"),
    "uM/h": (1e-3 / 3600.0, "mol/(m^3*s)"),
    "nM/s": (1e-6, "mol/(m^3*s)"), "nM/min": (1e-6 / 60.0, "mol/(m^3*s)"),
    "mol/(m^3*s)": (1.0, "mol/(m^3*s)"),
    "mol/(L*s)": (1e3, "mol/(m^3*s)"), "mol/(L*min)": (1000.0 / 60.0, "mol/(m^3*s)"),
    "mol/L/s": (1e3, "mol/(m^3*s)"), "mol/L/min": (1000.0 / 60.0, "mol/(m^3*s)"),
    "1/min": (1.0 / 60.0, "1/s"), "min^-1": (1.0 / 60.0, "1/s"),
    # clinical gravimetrics, doses, bio-mass, thermochemistry
    "mg/dL": (1e-2, "kg/m^3"), "ug/mL": (1e-3, "kg/m^3"),
    "ng/mL": (1e-6, "kg/m^3"), "mg/L": (1e-3, "kg/m^3"),
    "mg/kg": (1e-6, "kg/kg"), "ug/kg": (1e-9, "kg/kg"),
    "IU/L": (1e3, "IU/m^3"), "U/L": (1e3, "U/m^3"),
    "Da": (1.66053906660e-27, "kg"), "kDa": (1.66053906660e-24, "kg"),
    "MDa": (1.66053906660e-21, "kg"),
    "kcal/mol": (4184.0, "J/mol"), "cal/mol": (4.184, "J/mol"),
    "kJ/mol": (1e3, "J/mol"), "J/mol": (1.0, "J/mol"),
    # biology, genetics & epidemiology
    "bp": (1.0, "bp"), "kb": (1e3, "bp"), "Mb": (1e6, "bp"), "Gb": (1e9, "bp"),
    "cM": (1e-2, "Morgan"), "Morgan": (1.0, "Morgan"),
    "CFU/mL": (1e6, "m^-3"), "PFU/mL": (1e6, "m^-3"), "copies/mL": (1e6, "m^-3"),
    "1/day": (1.0 / 86400.0, "1/s"), "day^-1": (1.0 / 86400.0, "1/s"),
    "d^-1": (1.0 / 86400.0, "1/s"),
    "1/year": (1.0 / 3.15576e7, "1/s"), "year^-1": (1.0 / 3.15576e7, "1/s"),
    "yr^-1": (1.0 / 3.15576e7, "1/s"), "1/yr": (1.0 / 3.15576e7, "1/s"),
    "1/h": (1.0 / 3600.0, "1/s"), "1/hr": (1.0 / 3600.0, "1/s"), "h^-1": (1.0 / 3600.0, "1/s"),
    # earth, climate & astronomy
    "ppm": (1e-6, "fraction"), "ppb": (1e-9, "fraction"), "ppt": (1e-12, "fraction"),
    "GtC": (1e12, "kg"), "PgC": (1e12, "kg"), "GtCO2": (1e12, "kg"),
    "GtC/yr": (1e12 / 3.15576e7, "kg/s"), "GtC/year": (1e12 / 3.15576e7, "kg/s"),
    "PgC/yr": (1e12 / 3.15576e7, "kg/s"),
    "Sv": (1e6, "m^3/s"),
    "pc": (3.085677581491367e16, "m"), "kpc": (3.085677581491367e19, "m"),
    "Mpc": (3.085677581491367e22, "m"), "ly": (9.4607304725808e15, "m"),
    # information, statistics & economics
    "bit": (1.0, "bit"), "bits": (1.0, "bit"),
    "byte": (8.0, "bit"), "bytes": (8.0, "bit"),
    "nat": (1.0 / math.log(2.0), "bit"), "nats": (1.0 / math.log(2.0), "bit"),
    "dB": (1.0, "dB"),
    "%": (0.01, "fraction"), "percent": (0.01, "fraction"),
    "bps": (1e-4, "fraction"),
}

# Units that are safe to match case-insensitively (no SI-prefix ambiguity).
_CASE_INSENSITIVE_UNITS = {u.lower(): u for u in (
    "km/s", "km/h", "kg/m^3", "g/cm^3", "g/cc", "kg/L", "g/L", "atm", "bar",
    "kPa", "GPa", "W/m^2", "W/cm^2", "kW/m^2", "J/m^2", "J/cm^2", "eV", "keV",
    "deg", "degree", "degrees", "rad", "Gauss", "AU", "min", "hr", "yr", "day",
    "days", "tonne", "tonnes", "ton", "tons", "g/mol", "kg/mol",
    "mg/dL", "ug/mL", "ng/mL", "mg/kg", "IU/L", "kcal/mol", "kJ/mol", "J/mol",
    "CFU/mL", "PFU/mL", "1/day", "day^-1", "1/year", "year^-1", "yr^-1", "1/yr",
    "ppm", "ppb", "GtC", "PgC", "GtC/yr", "PgC/yr", "pc", "kpc", "Mpc",
    "bit", "bits", "nat", "nats", "dB", "bps", "percent", "Da", "kDa", "bp", "kb", "Sv")}


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
    entry = _UNIT_TABLE.get(canonical) or _electrical_unit(canonical)
    if entry is None:
        mapped = _CASE_INSENSITIVE_UNITS.get(canonical.lower())
        entry = _UNIT_TABLE.get(mapped) if mapped else None
    if entry is None:
        return value, canonical
    factor, si_unit = entry
    return value * factor, si_unit


def is_known_unit(unit_str: str) -> bool:
    canonical = _canonical_unit(unit_str)
    return (canonical in _UNIT_TABLE or _electrical_unit(canonical) is not None or canonical.lower() in _CASE_INSENSITIVE_UNITS
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
        elif (ch.isascii() and ch.isdigit()) or (ch == "." and i + 1 < n and s[i + 1].isascii() and s[i + 1].isdigit()):
            m = re.match(r"\d*\.?\d+", s[i:])
            if not m:
                raise _LatexParseError(f"unsupported numeric token {ch!r}")
            tokens.append(("num", m.group(0)))
            i += len(m.group(0))
        elif ch.isascii() and ch.isalpha():
            m = re.match(r"[A-Za-z]+", s[i:])
            if not m:
                raise _LatexParseError(f"unsupported identifier {ch!r}")
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

_NUMBER = (r"[-+−]?\d+(?:[.,]\d+)?(?:[eE][-+−]?\d+|\s*/\s*\d+(?:[.,]\d+)?)?"
           r"(?:\s*(?:×|x|\*|\\times)\s*10\s*\^?\s*\{?\s*[-+−]?\d+\s*\}?)?")
_UNIT_TOKEN = (r"(?:[A-Za-zμµ°]+(?:\^?-?[0-9²³])?"
               r"(?:\s*/\s*\(\s*[A-Za-z]+(?:\^?-?[0-9²³])?\s*[*·.\s]\s*[A-Za-z]+(?:\^?-?[0-9²³])?\s*\)"
               r"|\s*/\s*[A-Za-zμµ]+(?:\^?-?[0-9²³])?(?:\s*/\s*[A-Za-zμµ]+)?)?|°)")
_PARAM_ASSIGN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_\- ]{0,48}?)[ \t]*(?P<op>=|≈|~|:|\bof\b|\bis\b)[ \t]*"
    r"(?P<num>" + _NUMBER + r")(?!\s*\.[ \t]+[A-Za-z])(?:[ \t]*(?P<unit>" + _UNIT_TOKEN + r"))?")
_PARAM_WITH_UNIT = re.compile(
    r"(?P<num>" + _NUMBER + r")[ \t]*(?P<unit>" + _UNIT_TOKEN + r")")
# Single-letter subscript state tokens (``z_k``, ``q_0``, ``x_n``, ``i_2``,
# ``j_m``, ...) are code identifiers, never physical constants.
_SUBSCRIPT_STATE_RE = re.compile(r"^[a-zA-Z]_[knij012m]$")
_NAME_FILLER = {
    "the", "a", "an", "of", "is", "and", "or", "with", "without", "at", "to", "for",
    "in", "on", "by", "about", "approximately", "roughly", "around", "across",
    "between", "through", "during", "after", "before", "since", "until", "upon",
    "above", "below", "against", "toward", "towards", "among", "amongst",
    "throughout", "regarding", "within", "via", "per", "vs", "versus",
    "value", "equal", "equals", "be", "was", "were", "are", "its",
    "case", "literature", "violate", "violates", "diverge", "diverges",
    "positivity", "when", "where", "if", "then", "else", "such", "that",
    "which", "from", "into", "over", "under", "rich", "strict", "must",
    "may", "can", "should", "limit", "closed", "open", "given", "using",
    "let", "set", "assume", "assuming", "all",
}


_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_RESERVED_MATH_NAMES = {
    "ln2", "ln_2", "log2", "log_2", "ln10", "ln_10", "log10", "log_10",
    "pi", "sin", "cos", "tan", "exp", "sqrt", "log", "ln", "abs",
    "inf", "nan", "pow",
}


def _parse_number(text: str) -> Optional[float]:
    t = _THOUSANDS_COMMA_RE.sub("", text).replace("−", "-").replace(",", ".").replace(" ", "")
    m_frac = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", t)
    if m_frac:
        den = float(m_frac.group(2))
        return None if den == 0.0 else float(m_frac.group(1)) / den
    m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
                     r"(?:(?:×|x|\*|\\times)10\^?\{?([-+]?\d+)\}?)?", t)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2):
        value *= 10.0 ** int(m.group(2))
    return value


def _snake(text: str) -> str:
    raw_words = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    # Drop everything up to and including the last prose stopword/filler so
    # clauses like `positivity or diverge kappa` reduce cleanly to `kappa`.
    last_filler = -1
    for i, w in enumerate(raw_words):
        if w in _NAME_FILLER:
            last_filler = i
    if last_filler >= 0 and last_filler + 1 < len(raw_words):
        raw_words = raw_words[last_filler + 1 :]
    words = [w for w in raw_words if w not in _NAME_FILLER]
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


_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+")
_LATEX_GROUP = re.compile(r"\\[a-zA-Z]*\{[^{}]*\}")


def _strip_latex_commands(text: str) -> str:
    """Remove LaTeX control sequences so ``\\rangle = 0`` never parses as a
    physical constant named ``rangle`` (or ``m``/``n`` from boundary math)."""
    s = _LATEX_GROUP.sub(" ", text or "")
    s = _LATEX_COMMAND.sub(" ", s)
    return s


_PARAM_STOPWORDS = {
    "simulate", "sweep", "verify", "compute", "benchmark", "protocol", "dual_k", "corridor",
    "at", "in", "to", "of", "by", "for", "with", "without", "from", "least", "most",
    "continuous", "raw", "total", "net", "system_physics", "baseline",
    "closure", "model", "stage", "part", "step", "steps", "and", "or", "the", "a",
    "an", "is", "are", "was", "were", "as", "on", "per", "below", "above",
    "under", "over", "down", "up", "into", "onto", "than", "then", "max", "min",
    "across", "between", "within", "about", "around", "through", "during",
    "after", "before", "since", "until", "upon", "against", "toward", "towards",
    "among", "amongst", "throughout", "regarding", "vs", "versus",
    "service", "servicing", "covering", "drop", "cell", "accuracy",
    "calculating", "computing", "modeling", "verifying", "coupling",
    "overcoming", "connecting", "incorporating", "shunting", "via",
}

_BARE_PROSE_NOUNS = {
    "resistance", "mortality", "fixation", "treatment", "coverage",
    "concentration", "infusion", "fraction", "threshold", "duration",
    "horizon", "layer", "stage", "section", "objective", "regime", "regimes",
    "sample", "samples", "sensitivity", "collapse", "transition", "boundary",
    "crossover", "advantage", "equilibrium", "stability", "value", "values",
    "rate", "factor", "coefficient", "parameter", "variable", "result", "error",
}


def extract_parameters_from_text(text: str, source: str = "",
                                 confidence: float = 0.6,
                                 limit: int = 24) -> List[ExtractedParameter]:
    """Free-text ``name = number unit`` / ``name number unit`` pairs."""
    text = _THOUSANDS_COMMA_RE.sub("", _strip_latex_commands(_clean_wikitext(text or "")))
    params: List[ExtractedParameter] = []
    seen_spans: List[Tuple[int, int]] = []

    def _add(name_raw: str, num_raw: str, unit_raw: str, span: Tuple[int, int], *, assign_op: str = "=") -> None:
        if any(a <= span[0] < b for a, b in seen_spans):
            return
        # Ignore ratio expressions like `10:23` matched as `name 10 : 23`.
        if name_raw and re.search(r"\d$", name_raw.strip()) and ":" in text[span[0]:span[1]]:
            return
        value = _parse_number(num_raw)
        if value is None:
            return
        unit = (unit_raw or "").strip()
        if unit and not is_known_unit(unit):
            # A trailing word was captured as a unit; keep only if dimensionless.
            unit = ""
        cleaned_raw = re.sub(r"(?:>=|<=|==|!=|~=|[<>=\u2264\u2265])", " ", name_raw or "").strip()
        name = _snake(cleaned_raw)
        if not name or re.match(r"^\d", name):
            return
        if len(name) <= 1 and not unit:
            return
        if name in _PARAM_STOPWORDS or name in _RESERVED_MATH_NAMES:
            return
        if not unit and name in _BARE_PROSE_NOUNS:
            return
        if assign_op in (":", "of", "is") and not unit and "_" not in name and len(name) > 3 and name.islower():
            return
        if source == "user_prompt" and len(name) <= 1:
            return
        # Check if raw identifier is a reserved math function/constant such as `ln(2)`
        raw_tail = re.split(r"[\s,;]+", cleaned_raw.lower())[-1].strip()
        if raw_tail in _RESERVED_MATH_NAMES or re.sub(r"[()]", "", raw_tail) in _RESERVED_MATH_NAMES:
            return
        # Subscripted state tokens (``z_k``, ``q_0``, ``x_n``, ...) extracted
        # from code-like prompt fragments are not physical constants.
        if source == "user_prompt" and not unit and _SUBSCRIPT_STATE_RE.fullmatch(name):
            return
        si_value, si_unit = normalize_to_si(value, unit)
        toks = [t for t in cleaned_raw.split() if re.search(r"[A-Za-z]", t)]
        last_tok = toks[-1] if toks else ""
        symbol = last_tok if (last_tok and len(last_tok) <= 8 and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", last_tok)) else ""
        params.append(ExtractedParameter(
            name=name, symbol=symbol,
            raw_value=value, raw_unit=unit, si_value=si_value, si_unit=si_unit,
            source=source, confidence=confidence))
        seen_spans.append(span)

    for m in _PARAM_ASSIGN.finditer(text):
        _add(m.group("name"), m.group("num"), m.group("unit") or "", m.span(), assign_op=m.group("op"))
    for m in _PARAM_WITH_UNIT.finditer(text):
        unit = m.group("unit")
        if not is_known_unit(unit):
            continue
        before = text[max(0, m.start() - 48):m.start()]
        before = re.split(r"[.;,()\n=\[\]]|>=|<=", before)[-1].strip()
        # Strip trailing stopwords/prepositions/operators; if empty, inspect trailing noun phrase.
        words = [
            w for w in re.split(r"\s+", before)
            if w and not re.fullmatch(r"(?:>=|<=|==|!=|~=|[<>=\u2264\u2265]+)", w)
        ]
        while words and (words[-1].lower() in _PARAM_STOPWORDS or re.match(r"^\d", words[-1])):
            words.pop()
        if source == "user_prompt":
            # Filter out long un-assigned prose phrases (e.g. "visible sensory reset dissipation")
            # while keeping genuine variable identifiers or compact 1-2 word parameter names.
            has_ident = any("_" in w or any(ch.isupper() for ch in w[1:]) for w in words[-3:])
            if len(words) >= 3 and not has_ident:
                words = words[-2:]
            while words and words[-1].lower() in _PARAM_STOPWORDS:
                words.pop()
            if len(words) >= 2 and all(w.islower() and "_" not in w for w in words):
                # Multi-word lowercase prose captured before a unit without '='
                continue
        candidate_name = " ".join(words[-4:]) if words else ""
        if not candidate_name or candidate_name.lower() in _PARAM_STOPWORDS or re.match(r"^\d", candidate_name):
            if source == "user_prompt":
                continue
            after = text[m.end():min(len(text), m.end() + 48)]
            after = re.split(r"[.;,()\n]", after)[0].strip()
            awords = [w for w in re.split(r"\s+", after) if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", w)]
            while awords and awords[0].lower() in _PARAM_STOPWORDS:
                awords.pop(0)
            candidate_name = " ".join(awords[:4]) if awords else ""
        if candidate_name:
            _add(candidate_name, m.group("num"), unit, m.span())
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
        "keywords": ("robertson", "arrhenius", "oregonator", "brusselator",
                     "combustion", "chemical kinetic", "chemical kinetics",
                     "reaction network", "inorganic reaction", "stiff chemical",
                     "autocatal"),
        "parameters": [
            _p("robertson_k1", "k_1", 0.04, "", "Robertson (1966)", 1.0, ("robertson",)),
            _p("robertson_k2", "k_2", 1.0e4, "", "Robertson (1966)", 1.0, ("robertson",)),
            _p("robertson_k3", "k_3", 3.0e7, "", "Robertson (1966)", 1.0, ("robertson",)),
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
        "keywords": ("catalan", "prime", "recurrence", "combinator",
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
    "epidemiology_population": {
        "keywords": ("kermack", "mckendrick", "seir", "seir-v", "seirhd", "sir model",
                     "reproduction number", "herd immunity", "compartmental disease",
                     "epidemic", "infectious disease", "predator-prey"),
        "parameters": [
            _p("baseline_transmission_rate", "beta_0", 0.60, "1/day", "Kermack-McKendrick SEIR benchmark"),
            _p("incubation_rate", "sigma", 0.20, "1/day", "Kermack-McKendrick SEIR benchmark"),
            _p("recovery_rate", "gamma", 0.10, "1/day", "Kermack-McKendrick SEIR benchmark"),
            _p("basic_reproduction_number", "R_0", 6.0, "", "R_0 = beta_0 / gamma"),
            _p("herd_immunity_threshold", "p_c", 1.0 - 1.0 / 6.0, "", "p_c = 1 - 1 / R_0"),
        ],
        "equations": [
            ("R_0 = \\frac{\\beta}{\\gamma}", "Basic reproduction number"),
            ("p_c = 1 - \\frac{1}{R_0}", "Herd immunity critical threshold"),
            ("\\frac{dS}{dt} = -\\frac{\\beta S I}{N}", "Susceptible compartment decay"),
            ("\\frac{dE}{dt} = \\frac{\\beta S I}{N} - \\sigma E", "Exposed compartment dynamics"),
            ("\\frac{dI}{dt} = \\sigma E - \\gamma I", "Infectious compartment dynamics"),
            ("R_{eff} = \\frac{\\beta_{eff}}{\\gamma} \\frac{S}{N}", "Effective reproduction number"),
        ],
        "benchmarks": [
            {"name": "Epidemic suppression threshold", "input_regime": "SEIR / SEIR-V compartmental model",
             "expected_value": 1.0, "unit": "R_eff", "tolerance": 1e-6,
             "source": "Kermack & McKendrick (1927)"},
        ],
        "invariants": [
            "sum_{i=1..K} C_i == N_pop (exact population conservation across compartments)",
            "min(S, E, I, R, V) >= 0.0 (non-negativity of all compartments)",
            "0.0 < R_eff < 1.0 in suppressed epidemic regime",
        ],
    },
    "pharmacology_biochemistry": {
        "keywords": ("pharmacokinetic", "pharmacodynamic", "pk/pd", "pk-pd", "hill-langmuir",
                     "hill equation", "hill coefficient", "ec50", "ic50", "receptor occupancy",
                     "hodgkin-huxley", "volume of distribution", "clearance"),
        "parameters": [
            _p("half_maximal_effective_concentration", "EC50", 0.50, "uM", "Hill-Langmuir PD benchmark"),
            _p("hill_coefficient", "n_H", 2.0, "", "Hill-Langmuir cooperativity benchmark"),
            _p("max_antiviral_efficacy", "E_max", 0.85, "", "Antiviral Hill inhibition benchmark"),
        ],
        "equations": [
            ("\\theta_{Hill} = \\frac{C_p^n}{EC_{50}^n + C_p^n}", "Hill-Langmuir receptor occupancy"),
            ("V_d = \\frac{Dose}{C_p}", "Volume of distribution"),
            ("CL = k_{el} V_d", "Pharmacokinetic systemic clearance"),
            ("v = \\frac{V_{max} S}{K_m (1 + I / K_i) + S}", "Michaelis-Menten competitive inhibition"),
        ],
        "benchmarks": [
            {"name": "Hill occupancy at C_p = EC50", "input_regime": "C_p = EC50, any n_H > 0",
             "expected_value": 0.5, "unit": "fraction", "tolerance": 1e-6,
             "source": "Hill (1910)"},
        ],
        "invariants": [
            "0.0 <= theta_Hill <= 1.0 (Hill receptor occupancy bounded to unit interval)",
            "C_p >= 0.0 (non-negative plasma drug concentration)",
        ],
    },
    "genetics_evolution": {
        "keywords": ("hardy-weinberg", "wright-fisher", "price equation", "allele frequency",
                     "genotype frequency", "population genetics", "directional selection",
                     "linkage disequilibrium", "coalescent"),
        "parameters": [
            _p("selection_coefficient", "s_sel", 0.05, "", "Wright-Fisher directional selection"),
            _p("mutation_rate_per_base", "mu_gen", 1.0e-8, "1/bp", "Population genetics typical rate"),
        ],
        "equations": [
            ("p + q = 1", "Biallelic frequency conservation"),
            ("f_{AA} = p^2", "Hardy-Weinberg homozygous dominant genotype frequency"),
            ("f_{Aa} = 2 p q", "Hardy-Weinberg heterozygous genotype frequency"),
            ("\\frac{dp}{dt} = \\frac{s p (1 - p)}{\\bar{w}}", "Wright-Fisher directional selection ODE"),
        ],
        "benchmarks": [
            {"name": "Maximum heterozygosity 2pq", "input_regime": "Hardy-Weinberg equilibrium p = q = 0.5",
             "expected_value": 0.5, "unit": "fraction", "tolerance": 1e-6,
             "source": "Hardy (1908) & Weinberg (1908)"},
        ],
        "invariants": [
            "abs(p + q - 1.0) <= 1e-12 (allele frequency simplex conservation)",
            "abs(p**2 + 2*p*q + q**2 - 1.0) <= 1e-12 (Hardy-Weinberg genotype simplex)",
            "0.0 <= p <= 1.0 and 0.0 <= q <= 1.0",
        ],
    },
    "climate_earth_systems": {
        "keywords": ("carbon cycle", "carbon-cycle", "revelle", "radiative forcing",
                     "planck feedback", "terrestrial biosphere", "deep ocean", "surface ocean",
                     "pre-industrial", "gtc", "pgc", "thermohaline"),
        "parameters": [
            _p("preindustrial_atmospheric_carbon", "M_atm_0", 590.0, "GtC", "IPCC AR6 Carbon Cycle"),
            _p("total_carbon_inventory", "M_total", 41000.0, "GtC", "4-reservoir global carbon inventory"),
            _p("co2_radiative_forcing_coeff", "alpha_CO2", 5.35, "W/m^2", "Myhre et al. (1998) Arrhenius forcing"),
            _p("planck_feedback_parameter", "lambda_Planck", 1.25, "W/(m^2*K)", "CMIP6 radiative feedback"),
            _p("revelle_buffer_factor_nominal", "gamma_Revelle", 10.0, "", "Revelle & Suess (1957)"),
        ],
        "equations": [
            ("\\Delta F = 5.35 \\ln\\left(\\frac{M_{atm}}{M_{atm,0}}\\right)", "Arrhenius logarithmic CO2 radiative forcing [W/m^2]"),
            ("C_{eff} \\frac{dT_{surf}}{dt} = \\Delta F - \\lambda_{Planck} T_{surf}", "Global surface temperature anomaly energy balance"),
            ("M_{total} = M_{atm} + M_{bio} + M_{surf} + M_{deep}", "4-reservoir global carbon mass conservation"),
        ],
        "benchmarks": [
            {"name": "Doubled-CO2 radiative forcing (2x M_atm_0)", "input_regime": "M_atm / M_atm_0 = 2.0",
             "expected_value": 3.7083, "unit": "W/m^2", "tolerance": 0.05,
             "source": "Myhre et al. (1998)"},
        ],
        "invariants": [
            "abs((M_atm + M_bio + M_surf + M_deep) - M_total) <= 1e-6 (total carbon mass conservation)",
            "min(M_atm, M_bio, M_surf, M_deep) > 0.0 (strictly positive reservoir inventories)",
            "8.0 <= gamma_Revelle <= 15.0 (physical Revelle carbonate buffer bounds)",
            "dV_climate <= 0.0 (monotonic free-energy Lyapunov relaxation in unforced regime)",
        ],
    },
    "statistics_causal_inference": {
        "keywords": ("structural causal model", "instrumental variable", "2sls", "two-stage least squares",
                     "wald estimator", "cramer-rao", "cramér-rao", "fisher information",
                     "unmeasured confounder", "confounding bias", "do-calculus", "average treatment effect",
                     "welford"),
        "parameters": [
            _p("true_causal_effect_benchmark", "beta_true", 2.0, "", "SCM Instrumental Variable benchmark"),
            _p("first_stage_instrument_strength", "pi_Z", 0.8, "", "SCM first-stage Z -> X coefficient"),
        ],
        "equations": [
            ("\\beta_{OLS} = \\frac{Cov(X, Y)}{Var(X)}", "Ordinary Least Squares (OLS) slope estimator"),
            ("\\beta_{IV} = \\frac{Cov(Z, Y)}{Cov(Z, X)}", "Instrumental Variable (IV) Wald / 2SLS estimator"),
            ("Var(\\hat{\\theta}) = \\frac{1}{I_F(\\theta)}", "Cramer-Rao Fisher Information variance lower bound"),
            ("r_{XZ} = \\frac{Cov(X, Z)}{\\sqrt{Var(X) Var(Z)}}", "Pearson sample correlation coefficient"),
        ],
        "benchmarks": [
            {"name": "Asymptotic IV consistency under unmeasured confounding", "input_regime": "Z _||_ U, Cov(Z, X) != 0",
             "expected_value": 2.0, "unit": "beta_IV", "tolerance": 0.05,
             "source": "Wald (1940) & Pearl (2000)"},
        ],
        "invariants": [
            "abs(Cov_XZ) <= sqrt(Var_X * Var_Z) (Cauchy-Schwarz covariance inequality)",
            "-1.0 <= r_XZ <= 1.0 (correlation bound)",
            "Var_beta_IV >= CRLB_floor and CRLB_floor > 0.0 (Cramer-Rao variance lower bound)",
            "abs(beta_IV - beta_true) <= 0.05 (deconfounded IV consistency)",
        ],
    },
    "economics_game_theory": {
        "keywords": ("replicator", "replicator-mutator", "nash equilibrium", "exploitability",
                     "nikaido-isoda", "rock-paper-scissors", "evolutionary game", "payoff matrix",
                     "bellman", "walrasian", "vcg mechanism", "regret"),
        "parameters": [
            _p("interior_nash_3simplex", "p_star", 1.0 / 3.0, "", "Symmetric 3-strategy Nash equilibrium"),
            _p("mutation_damping_rate", "mu_mut", 0.05, "", "Replicator-mutator interior convergence rate"),
        ],
        "equations": [
            ("\\frac{dp_i}{dt} = p_i (f_i - \\bar{f}) + \\mu (p_i^* - p_i)", "Damped replicator-mutator dynamics on K-simplex"),
            ("D_{KL}(p^* || p) = p_1^* \\ln\\left(\\frac{p_1^*}{p_1}\\right)", "Kullback-Leibler relative-entropy Lyapunov divergence"),
            ("\\Psi_{regret} = \\max(0, f_i - \\bar{f})", "Nikaido-Isoda Nash exploitability regret"),
        ],
        "benchmarks": [
            {"name": "Symmetric 3-strategy Nash fixed point", "input_regime": "Cyclic RPS with positive mutation/damping",
             "expected_value": [0.333333333333, 0.333333333333, 0.333333333333], "unit": "probability",
             "tolerance": 1e-3, "source": " Maynard Smith (1982) & Hofbauer-Sigmund (1998)"},
        ],
        "invariants": [
            "abs((p_1 + p_2 + p_3) - 1.0) <= 1e-12 (unit 3-simplex probability conservation)",
            "min(p_1, p_2, p_3) >= 0.0 (non-negative strategy frequencies)",
            "D_KL >= 0.0 and d_D_KL <= 0.0 (monotonic KL Lyapunov contraction)",
            "0.0 <= Psi_regret <= 1e-4 (Nash exploitability convergence)",
        ],
    },
    "math_info_theory": {
        "keywords": ("shannon-hartley", "channel capacity", "shannon entropy", "source coding",
                     "kullback-leibler", "mutual information", "rate-distortion", "hamming bound",
                     "ergodic markov", "spectral gap"),
        "parameters": [
            _p("binary_entropy_max", "H_2_max", 1.0, "bit", "Shannon (1948) binary source entropy limit"),
            _p("nat_to_bit_factor", "log2_e", 1.0 / math.log(2.0), "bit", "1 / ln(2)"),
        ],
        "equations": [
            ("C = B \\log_2\\left(1 + \\frac{S}{N}\\right)", "Shannon-Hartley AWGN channel capacity [bit/s]"),
            ("H = -p \\ln(p) - (1 - p) \\ln(1 - p)", "Binary Shannon entropy [nat]"),
        ],
        "benchmarks": [
            {"name": "Shannon capacity at SNR = 1 (0 dB)", "input_regime": "S/N = 1.0, B = 1.0 Hz",
             "expected_value": 1.0, "unit": "bit/s", "tolerance": 1e-6,
             "source": "Shannon (1948)"},
        ],
        "invariants": [
            "H_shannon >= 0.0 and D_KL >= 0.0 (non-negativity of Shannon entropy and KL divergence)",
            "R_transmission <= C_shannon (Shannon channel coding capacity bound)",
        ],
    },
}


# Domains only unlock on EXPLICIT benchmark-system mentions; generic words
# like "discrete", "thermal", "stiff" or "nonlinear" must never pull in
# tungsten densities, Black-Scholes or Mars heat-shield constants.
_REGISTRY_ANCHORS: Dict[str, Tuple[str, ...]] = {
    "fundamental_constants": ("codata", "fundamental constant", "speed of light",
               "planck constant", "boltzmann", "avogadro", "fine-structure",
               "electron mass", "gravitational constant"),
    "aerospace_entry": ("mars entry", "martian entry", "venus entry",
                        "sutton-graves", "allen-eggers", "re-entry",
                        "reentry", "heat shield", "ablation", "stagnation-point"),
    "relativistic_plasma": ("relativistic plasma", "magnetic sail", "magsail",
                            "plasma sail", "solar wind", "aurora",
                            "lorentz plasma"),
    "materials_thermophysics": ("tungsten", "pica", "silicon carbide", "titanium",
                                "carbon phenolic", "ablative", "ablation",
                                "re-entry", "reentry", "hypersonic",
                                "heat shield", "tps", "thermal protection",
                                "refractory", "ceramic matrix"),
    "nonlinear_dynamics": ("duffing", "van der pol", "lorenz", "lotka",
                           "volterra", "henon", "circle map", "standard map"),
    "chemical_kinetics": ("robertson", "arrhenius", "oregonator", "brusselator",
                          "combustion", "chemical kinetic", "chemical kinetics",
                          "chemical reaction network", "inorganic reaction",
                          "stiff chemical", "autocatal"),
    "discrete_finance": ("black-scholes", "black scholes",
                         "catalan", "option pricing", "binomial model",
                         "lattice model", "risk-neutral"),
    "epidemiology_population": ("kermack", "mckendrick", "seir", "seir-v", "seirhd",
                                "sir model", "reproduction number", "herd immunity",
                                "infectious disease", "epidemic"),
    "pharmacology_biochemistry": ("pharmacokinetic", "pharmacodynamic", "pk/pd", "pk-pd",
                                  "hill-langmuir", "hill coefficient", "ec50", "ic50",
                                  "receptor occupancy", "hodgkin-huxley", "antiviral"),
    "genetics_evolution": ("hardy-weinberg", "wright-fisher", "price equation",
                           "allele frequency", "genotype frequency", "population genetics"),
    "climate_earth_systems": ("carbon cycle", "carbon-cycle", "revelle", "radiative forcing",
                              "planck feedback", "terrestrial biosphere", "deep ocean",
                              "surface ocean", "gtc", "pgc"),
    "statistics_causal_inference": ("structural causal model", "instrumental variable",
                                    "2sls", "two-stage least squares", "wald estimator",
                                    "cramer-rao", "cramér-rao", "fisher information",
                                    "unmeasured confounder", "confounding bias", "do-calculus",
                                    "welford"),
    "economics_game_theory": ("replicator", "replicator-mutator", "nash equilibrium",
                              "exploitability", "nikaido-isoda", "rock-paper-scissors",
                              "evolutionary game", "bellman", "walrasian", "vcg mechanism"),
    "math_info_theory": ("shannon-hartley", "channel capacity", "shannon entropy",
                         "source coding", "kullback-leibler", "mutual information",
                         "rate-distortion"),
}


ReferenceRegistry = DOMAIN_REFERENCE_REGISTRY

MACRO_SCIENTIFIC_DOMAINS: Tuple[str, ...] = (
    "PHYSICAL_SCIENCES_AND_ENG",
    "CHEMICAL_AND_MATERIALS",
    "LIFE_SCIENCES_AND_BIOMEDICINE",
    "EPIDEMIOLOGY_AND_ECOLOGY",
    "EARTH_CLIMATE_AND_ENVIRONMENT",
    "STATISTICS_AND_CAUSAL_INFERENCE",
    "ECONOMICS_AND_GAME_THEORY",
    "MATHEMATICS_CS_AND_INFO_THEORY",
)

_MACRO_DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "PHYSICAL_SCIENCES_AND_ENG": (
        "mechanics", "acoustics", "optics", "elasticity", "heat transfer",
        "quantum", "relativit", "lorentz", "plasma", "fluid dynamics", "navier-stokes",
        "condensed matter", "astrodynamics", "orbital", "thermodynamic", "electromagnet",
        "maxwell", "hamiltonian", "lindblad", "bloch", "soliton", "korteweg", "kdv",
        "aeroshell", "sutton-graves", "hypersonic", "schrodinger", "magnetism",
    ),
    "CHEMICAL_AND_MATERIALS": (
        "biochem", "membrane transport", "proton-motive", "chemiosmo",
        "reaction kinetic", "chemical kinetic", "catalys", "electrochem", "battery",
        "degradation", "sei growth", "polymer", "thermochem", "arrhenius", "robertson",
        "oregonator", "stoichiometr", "tungsten", "silicon carbide", "material science",
        "molarity", "redox", "butler-volmer",
    ),
    "LIFE_SCIENCES_AND_BIOMEDICINE": (
        "bioenerget", "metaboli", "biochem", "physiolog", "cardiolog", "patholog", "membrane transport",
        "systems biology", "michaelis-menten", "enzyme", "genomic", "gene regulatory",
        "population genetics", "hardy-weinberg", "wright-fisher", "pharmacolog",
        "pharmacokinetic", "pharmacodynamic", "pk/pd", "pk-pd", "hill-langmuir",
        "hill coefficient", "ec50", "ic50", "receptor occupancy", "oncolog", "tumor",
        "neuroscience", "hodgkin-huxley", "membrane potential", "biophysic", "antiviral",
        "antimicrobial", "bactericidal", "antibiotic", "pathogen", "dosing",
        "infusion", "hospital", "mortality", "haldane", "recombination", "epista",
        "linkage", "clearance", "volume of distribution", "allele", "clinical",
    ),
    "EPIDEMIOLOGY_AND_ECOLOGY": (
        "epidemiolog", "infectious disease", "seir", "seir-v", "seirhd", "sir model",
        "compartmental", "kermack", "mckendrick", "reproduction number", "herd immunity",
        "vector-borne", "metapopulation", "lotka-volterra", "predator-prey", "food web",
        "biodiversity", "island biogeography", "outbreak", "transmission rate",
        "vaccination", "epidemic", "ecolog", "droplet transmission",
    ),
    "EARTH_CLIMATE_AND_ENVIRONMENT": (
        "climat", "oceanograph", "atmospher", "seismolog", "meteorolog",
        "radiative-convective", "radiative forcing", "carbon cycle", "carbon-cycle",
        "reservoir", "revelle", "ocean carbonate", "thermohaline", "hydrolog",
        "geophysic", "terrestrial biosphere", "deep ocean", "surface ocean",
        "planck feedback", "climate feedback", "gtc", "pgc", "sverdrup", "albedo",
        "greenhouse", "temperature anomaly",
    ),
    "STATISTICS_AND_CAUSAL_INFERENCE": (
        "regression", "statistic", "likelihood", "randomized trial",
        "bayesian inference", "neyman-pearson", "hypothesis test", "cramer-rao",
        "cramér-rao", "fisher information", "structural causal model", "do-calculus",
        "potential outcome", "average treatment effect", "instrumental variable",
        "2sls", "two-stage least squares", "wald estimator", "difference-in-differences",
        "unmeasured confound", "confounding bias", "ordinary least squares", "welford",
        "biostatistic", "covariance", "p-value", "estimator variance",
    ),
    "ECONOMICS_AND_GAME_THEORY": (
        "economic", "monetary", "fiscal", "market equilibrium",
        "game theory", "evolutionary game", "nash equilibrium", "replicator",
        "replicator-mutator", "exploitability", "nikaido-isoda", "rock-paper-scissors",
        "prisoner's dilemma", "mechanism design", "vcg", "bellman", "dynamic programming",
        "walrasian", "dsge", "macroeconomic", "quantitative finance", "black-scholes",
        "martingale", "utility", "payoff matrix", "regret",
    ),
    "MATHEMATICS_CS_AND_INFO_THEORY": (
        "algebra", "topology", "computer science", "numerical analysis",
        "information theory", "channel coding", "shannon-hartley", "shannon entropy",
        "kullback-leibler", "source coding", "spectral graph", "markov chain",
        "algorithmic complexity", "combinator", "number theory", "catalan", "fibonacci",
        "rate-distortion", "mutual information", "ergodic",
    ),
}


def classify_macro_scientific_domains(text: str) -> List[str]:
    """Zero-shot NLP classification mapping a prompt/query into the 8 Macro-Scientific Domains.

    Returns matched domains sorted by match score (descending); defaults to
    ``["PHYSICAL_SCIENCES_AND_ENG"]`` when no domain keyword matches.
    """
    t = (text or "").lower()
    scores: List[Tuple[int, int, str]] = []
    for idx, domain in enumerate(MACRO_SCIENTIFIC_DOMAINS):
        kws = _MACRO_DOMAIN_KEYWORDS.get(domain, ())
        hits = sum(1 for kw in kws if kw in t)
        if hits > 0:
            scores.append((hits, -idx, domain))
    if not scores:
        return ["PHYSICAL_SCIENCES_AND_ENG"]
    scores.sort(reverse=True)
    return [d for _, _, d in scores]


def classify_primary_macro_domain(text: str) -> str:
    """Return the highest-scoring macro-scientific domain for ``text``."""
    return classify_macro_scientific_domains(text)[0]


def classify_registry_domains(text: str) -> List[str]:
    """Registry domains whose EXPLICIT anchors occur in ``text``.

    Strict gating: a domain matches only when one of its benchmark-system
    anchors (e.g. "Sutton-Graves", "Black-Scholes", "Robertson", "Lorenz")
    appears in the text — generic scientific vocabulary never matches.
    """
    t = (text or "").lower()
    out = []
    for name, spec in DOMAIN_REFERENCE_REGISTRY.items():
        anchors = _REGISTRY_ANCHORS.get(name)
        if anchors is None:
            anchors = tuple(spec.get("keywords") or ())
        if any(re.search(rf"\b{re.escape(a)}\b", t) for a in anchors):
            out.append(name)
    return out


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
            if "robertson" in desc.lower() and "robertson" not in t:
                continue
            eq = parse_equation(latex, source="ReferenceRegistry", description=desc)
            if eq is not None:
                equations.append(eq)
        for b in spec["benchmarks"]:
            if "robertson" in str(b.get("name", "")).lower() and "robertson" not in t:
                continue
            benchmarks.append(BenchmarkAnchor(**b))
        invariants.extend(spec["invariants"])
    return params, equations, benchmarks, invariants, domains


# ---------------------------------------------------------------------------
# Deterministic backend & stepper recommender
# ---------------------------------------------------------------------------

_DISCRETE_KW = ("catalan", "prime", "recurrence", "combinator",
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
        from ..latex_normalizer import _balance_math_expr, canonicalize_math_content
        lines += ["| Equation | LaTeX | Python hint (v9) | Source |",
                  "|---|---|---|---|"]
        for e in evidence.equations:
            hint = f"`{e.lhs} = {e.rhs_python_hint}`" if e.rhs_python_hint else "—"
            raw_tex = (e.rhs_latex or "").strip().strip("$").strip("`").strip()
            tex_cell = f"${_balance_math_expr(canonicalize_math_content(raw_tex))}$" if raw_tex else "—"
            lines.append(f"| {_md_cell(e.description or e.lhs)} | {_md_cell(tex_cell)} "
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
        candidates = [m.group(1) for m in _INLINE_TEX.finditer(text or "")]
        for match in re.finditer(
                r"(?<![\w])(?:[A-Za-z][A-Za-z0-9_]*(?:\([^\n=;]{1,60}\))?|d[A-Za-z]+/dt)\s*=\s*[^\n;!?$]{1,240}", text or ""):
            candidate = re.split(r'\.(?:\s|$)', match.group(0), maxsplit=1)[0].strip()
            rhs = candidate.split('=', 1)[-1].strip()
            if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', rhs) and not re.search(r'\b[A-Za-z_]\w*\s+[A-Za-z_]\w*\b', candidate):
                candidates.append(candidate)
        for candidate in dict.fromkeys(candidates):
            if '=' not in candidate:
                continue
            eq = parse_equation(candidate, source=source)
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
            # 3b. Non-intrusive SHBT Frontier Federation (activates ONLY on unsolvable/frontier SHBT domains).
            from .shbt_federation import filter_shbt_spec_for_prompt, resolve_shbt_solvability
            shbt_res = resolve_shbt_solvability(prompt or "")
            if shbt_res.should_activate:
                for spec in shbt_res.matched_repos:
                    domains.append(spec.repo)
                    rel_consts, rel_eqs, rel_invs = filter_shbt_spec_for_prompt(spec, prompt or "")
                    for sc in rel_consts:
                        add_param(ExtractedParameter(
                            name=sc.name, symbol=sc.symbol, raw_value=float(sc.si_value),
                            raw_unit=sc.si_unit, si_value=float(sc.si_value), si_unit=sc.si_unit,
                            source=f"SHBT: {spec.repo}", confidence=0.99,
                        ))
                    for seq in rel_eqs:
                        eq_obj = parse_equation(seq.latex, source=f"SHBT: {spec.repo}", description=seq.description)
                        if eq_obj is not None:
                            equations.append(eq_obj)
                    invariants.extend(rel_invs)
            equations.extend(r_eqs)
            benchmarks.extend(r_bench)
            invariants.extend(r_inv)
        else:
            domains = classify_registry_domains(" ".join(corpus))
        macro_domains = classify_macro_scientific_domains(prompt or " ".join(corpus))
        if macro_domains:
            domains = list(dict.fromkeys([d for d in domains if d != 'general'] + macro_domains))

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

        max_p = (max(self.max_parameters, 64)
                 if any(d.startswith("sys1own/shbt-") for d in domains)
                 else self.max_parameters)
        draft = StructuredResearchEvidence(
            domains=domains, equations=uniq_eqs[: self.max_equations],
            parameters=dict(list(params.items())[:max_p]),
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


normalize_unit = normalize_to_si


def lookup_reference_registry(text: str) -> List[ExtractedParameter]:
    """Convenience wrapper returning matched reference parameters for ``text``."""
    params, _eqs, _bench, _inv, _doms = registry_evidence(text)
    return params

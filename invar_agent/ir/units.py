"""Single authoritative SI unit parser & dimension-order converter (UPG-008, UPG-003).

Orders:
- SCI_ORDER     = ("M", "L", "T", "I", "Theta", "N", "J")   # SciIR, IHCG, HIN, certificates
- BUILDER_ORDER = ("m", "kg", "s", "A", "K", "mol", "cd")   # SIDimensionVectorModel only
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
import re
from typing import Literal, Sequence

SCI_ORDER = ("M", "L", "T", "I", "Theta", "N", "J")
BUILDER_ORDER = ("m", "kg", "s", "A", "K", "mol", "cd")

Dim = tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]
DIM_ZERO: Dim = (
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
)


def sci_to_builder(d: Sequence) -> tuple:
    """Convert [M, L, T, I, Theta, N, J] -> [m, kg, s, A, K, mol, cd]."""
    return (d[1], d[0], d[2], d[3], d[4], d[5], d[6])


def builder_to_sci(d: Sequence) -> tuple:
    """Convert [m, kg, s, A, K, mol, cd] -> [M, L, T, I, Theta, N, J]."""
    return (d[1], d[0], d[2], d[3], d[4], d[5], d[6])


class UnitError(ValueError):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


@dataclass(frozen=True)
class UnitParse:
    dim: Dim
    scale: Fraction | float
    offset: Fraction = Fraction(0)
    kind: Literal["linear", "affine"] = "linear"
    tags: frozenset[str] = field(default_factory=frozenset)
    advisories: tuple[str, ...] = ()

    @property
    def is_dimensionless(self) -> bool:
        return all(e == 0 for e in self.dim)

    def to_si_rational(self, value: Fraction | int | float | str) -> Fraction:
        v = value if isinstance(value, Fraction) else Fraction(str(value))
        sc = self.scale if isinstance(self.scale, Fraction) else Fraction(str(self.scale))
        if self.kind == "affine":
            return v * sc + self.offset
        return v * sc

    def canonical_si_unit(self) -> str:
        if self.is_dimensionless:
            return "1"
        base_names = ("kg", "m", "s", "A", "K", "mol", "cd")
        num_parts: list[str] = []
        den_parts: list[str] = []
        for name, exp in zip(base_names, self.dim):
            if exp == 0:
                continue
            abs_e = abs(exp)
            exp_s = "" if abs_e == 1 else (f"^{abs_e.numerator}" if abs_e.denominator == 1 else f"^({abs_e.numerator}/{abs_e.denominator})")
            if exp > 0:
                num_parts.append(f"{name}{exp_s}")
            else:
                den_parts.append(f"{name}{exp_s}")
        num_s = "*".join(num_parts) if num_parts else "1"
        if not den_parts:
            return num_s
        den_s = "*".join(den_parts)
        return f"{num_s}/({den_s})" if len(den_parts) > 1 else f"{num_s}/{den_s}"


def _d(M=0, L=0, T=0, I=0, Th=0, N=0, J=0) -> Dim:
    return (Fraction(M), Fraction(L), Fraction(T), Fraction(I), Fraction(Th), Fraction(N), Fraction(J))


def _dim_add(a: Dim, b: Dim) -> Dim:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def _dim_sub(a: Dim, b: Dim) -> Dim:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def _dim_scale(a: Dim, k: Fraction) -> Dim:
    return tuple(x * k for x in a)  # type: ignore[return-value]


_PREFIXES: dict[str, Fraction] = {
    "y": Fraction(1, 10**24),
    "z": Fraction(1, 10**21),
    "a": Fraction(1, 10**18),
    "f": Fraction(1, 10**15),
    "p": Fraction(1, 10**12),
    "n": Fraction(1, 10**9),
    "u": Fraction(1, 10**6),
    "µ": Fraction(1, 10**6),
    "μ": Fraction(1, 10**6),
    "m": Fraction(1, 10**3),
    "c": Fraction(1, 10**2),
    "d": Fraction(1, 10),
    "da": Fraction(10, 1),
    "h": Fraction(100, 1),
    "k": Fraction(10**3, 1),
    "M": Fraction(10**6, 1),
    "G": Fraction(10**9, 1),
    "T": Fraction(10**12, 1),
    "P": Fraction(10**15, 1),
    "E": Fraction(10**18, 1),
    "Z": Fraction(10**21, 1),
    "Y": Fraction(10**24, 1),
}

# (dim, scale, prefixable, offset, kind, tags)
_UnitDef = tuple[Dim, Fraction | float, bool, Fraction, Literal["linear", "affine"], frozenset[str]]

def _u(dim: Dim, scale: Fraction | float = Fraction(1), prefixable: bool = True,
       offset: Fraction = Fraction(0), kind: Literal["linear", "affine"] = "linear",
       tags: frozenset[str] = frozenset()) -> _UnitDef:
    return (dim, scale, prefixable, offset, kind, tags)


_UNITS: dict[str, _UnitDef] = {
    # Dimensionless / special tags
    "1": _u(DIM_ZERO, Fraction(1), False),
    "dimensionless": _u(DIM_ZERO, Fraction(1), False),
    "unitless": _u(DIM_ZERO, Fraction(1), False),
    "none": _u(DIM_ZERO, Fraction(1), False),
    "rad": _u(DIM_ZERO, Fraction(1), True),
    "sr": _u(DIM_ZERO, Fraction(1), True),
    "deg": _u(DIM_ZERO, math.pi / 180.0, False),
    "bit": _u(DIM_ZERO, Fraction(1), True),
    "bits": _u(DIM_ZERO, Fraction(1), False),
    "nat": _u(DIM_ZERO, 1.0 / math.log(2.0), False),
    "nats": _u(DIM_ZERO, 1.0 / math.log(2.0), False),
    "bp": _u(DIM_ZERO, Fraction(1), True),
    "%": _u(DIM_ZERO, Fraction(1, 100), False),
    "percent": _u(DIM_ZERO, Fraction(1, 100), False),
    "ppm": _u(DIM_ZERO, Fraction(1, 10**6), False),
    "ppb": _u(DIM_ZERO, Fraction(1, 10**9), False),
    "$": _u(DIM_ZERO, Fraction(1), False, tags=frozenset({"currency"})),
    "USD": _u(DIM_ZERO, Fraction(1), False, tags=frozenset({"currency"})),
    "EUR": _u(DIM_ZERO, Fraction(1), False, tags=frozenset({"currency"})),
    # Base SI
    "g": _u(_d(M=1), Fraction(1, 1000), True),
    "kg": _u(_d(M=1), Fraction(1), False),
    "m": _u(_d(L=1), Fraction(1), True),
    "s": _u(_d(T=1), Fraction(1), True),
    "sec": _u(_d(T=1), Fraction(1), False),
    "secs": _u(_d(T=1), Fraction(1), False),
    "second": _u(_d(T=1), Fraction(1), False),
    "seconds": _u(_d(T=1), Fraction(1), False),
    "A": _u(_d(I=1), Fraction(1), True),
    "K": _u(_d(Th=1), Fraction(1), True),
    "mol": _u(_d(N=1), Fraction(1), True),
    "cd": _u(_d(J=1), Fraction(1), True),
    # Derived SI
    "Hz": _u(_d(T=-1), Fraction(1), True),
    "N": _u(_d(M=1, L=1, T=-2), Fraction(1), True),
    "Pa": _u(_d(M=1, L=-1, T=-2), Fraction(1), True),
    "J": _u(_d(M=1, L=2, T=-2), Fraction(1), True),
    "W": _u(_d(M=1, L=2, T=-3), Fraction(1), True),
    "C": _u(_d(T=1, I=1), Fraction(1), True),
    "V": _u(_d(M=1, L=2, T=-3, I=-1), Fraction(1), True),
    "F": _u(_d(M=-1, L=-2, T=4, I=2), Fraction(1), True),
    "Ohm": _u(_d(M=1, L=2, T=-3, I=-2), Fraction(1), True),
    "ohm": _u(_d(M=1, L=2, T=-3, I=-2), Fraction(1), True),
    "Ω": _u(_d(M=1, L=2, T=-3, I=-2), Fraction(1), True),
    "S": _u(_d(M=-1, L=-2, T=3, I=2), Fraction(1), True),
    "Wb": _u(_d(M=1, L=2, T=-2, I=-1), Fraction(1), True),
    "T": _u(_d(M=1, T=-2, I=-1), Fraction(1), True),
    "H": _u(_d(M=1, L=2, T=-2, I=-2), Fraction(1), True),
    "Bq": _u(_d(T=-1), Fraction(1), True),
    "Gy": _u(_d(L=2, T=-2), Fraction(1), True),
    "Sv": _u(_d(L=2, T=-2), Fraction(1), True),
    "kat": _u(_d(T=-1, N=1), Fraction(1), True),
    "lm": _u(_d(J=1), Fraction(1), True),
    "lx": _u(_d(L=-2, J=1), Fraction(1), True),
    # Common non-SI
    "L": _u(_d(L=3), Fraction(1, 1000), True),
    "l": _u(_d(L=3), Fraction(1, 1000), False),
    "M": _u(_d(L=-3, N=1), Fraction(1000, 1), True),  # mol/L = 1000 mol/m^3
    "min": _u(_d(T=1), Fraction(60), False),
    "mins": _u(_d(T=1), Fraction(60), False),
    "minute": _u(_d(T=1), Fraction(60), False),
    "minutes": _u(_d(T=1), Fraction(60), False),
    "h": _u(_d(T=1), Fraction(3600), False),
    "hr": _u(_d(T=1), Fraction(3600), False),
    "hrs": _u(_d(T=1), Fraction(3600), False),
    "hour": _u(_d(T=1), Fraction(3600), False),
    "hours": _u(_d(T=1), Fraction(3600), False),
    "d": _u(_d(T=1), Fraction(86400), False),
    "day": _u(_d(T=1), Fraction(86400), False),
    "days": _u(_d(T=1), Fraction(86400), False),
    "wk": _u(_d(T=1), Fraction(604800), False),
    "week": _u(_d(T=1), Fraction(604800), False),
    "weeks": _u(_d(T=1), Fraction(604800), False),
    "mo": _u(_d(T=1), Fraction(2629746), False),
    "month": _u(_d(T=1), Fraction(2629746), False),
    "months": _u(_d(T=1), Fraction(2629746), False),
    "quarter": _u(_d(T=1), Fraction(7889238), False),
    "quarters": _u(_d(T=1), Fraction(7889238), False),
    "yr": _u(_d(T=1), Fraction(31557600), False),
    "yrs": _u(_d(T=1), Fraction(31557600), False),
    "year": _u(_d(T=1), Fraction(31557600), False),
    "years": _u(_d(T=1), Fraction(31557600), False),
    "eV": _u(_d(M=1, L=2, T=-2), Fraction(1602176634, 10**28), True),
    "bar": _u(_d(M=1, L=-1, T=-2), Fraction(10**5), True),
    "atm": _u(_d(M=1, L=-1, T=-2), Fraction(101325), False),
    "Wh": _u(_d(M=1, L=2, T=-2), Fraction(3600), True),
    "kWh": _u(_d(M=1, L=2, T=-2), Fraction(3600000), False),
    "Msun": _u(_d(M=1), Fraction(198847, 10**5) * Fraction(10**30), False),
    "M_sun": _u(_d(M=1), Fraction(198847, 10**5) * Fraction(10**30), False),
    "degC": _u(_d(Th=1), Fraction(1), False, offset=Fraction(27315, 100), kind="affine"),
    "°C": _u(_d(Th=1), Fraction(1), False, offset=Fraction(27315, 100), kind="affine"),
    "degF": _u(_d(Th=1), Fraction(5, 9), False, offset=Fraction(45967, 100) * Fraction(5, 9), kind="affine"),
    "°F": _u(_d(Th=1), Fraction(5, 9), False, offset=Fraction(45967, 100) * Fraction(5, 9), kind="affine"),
    "Morgan": _u(DIM_ZERO, Fraction(1), True),
    "M_centi": _u(DIM_ZERO, Fraction(1, 100), False),
    "cM": _u(DIM_ZERO, Fraction(1, 100), False),
    "GtC": _u(_d(M=1), Fraction(10**12), False),
    "AU": _u(_d(L=1), Fraction(149597870700), False),
    "pc": _u(_d(L=1), 3.085677581491367e16, True),
    "ly": _u(_d(L=1), 9.4607304725808e15, False),
    "K_B": _u(_d(M=1, L=2, T=-2, Th=-1), Fraction(1), False),
}

_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _lookup_atom(sym: str) -> UnitParse:
    if sym in _UNITS:
        dim, sc, _, off, kind, tags = _UNITS[sym]
        return UnitParse(dim=dim, scale=sc, offset=off, kind=kind, tags=tags)
    # Try longest prefix first ("da" before single-char prefixes)
    for pref in sorted(_PREFIXES.keys(), key=len, reverse=True):
        if sym.startswith(pref) and len(sym) > len(pref):
            base = sym[len(pref):]
            if base in _UNITS:
                dim, sc, prefixable, off, kind, tags = _UNITS[base]
                if prefixable and kind == "linear":
                    pf = _PREFIXES[pref]
                    new_scale = sc * pf if isinstance(sc, Fraction) else float(sc) * float(pf)
                    return UnitParse(dim=dim, scale=new_scale, offset=Fraction(0), kind="linear", tags=tags)
    raise UnitError("UNIT_UNKNOWN", f"Unknown unit symbol {sym!r}")


def _parse_exponent(raw: str) -> Fraction:
    s = raw.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) != 2:
            raise UnitError("UNIT_UNKNOWN", f"Invalid rational exponent {raw!r}")
        num, den = int(parts[0].strip()), int(parts[1].strip())
        if den <= 0:
            raise UnitError("UNIT_UNKNOWN", f"Invalid exponent denominator {raw!r}")
        return Fraction(num, den)
    if "." in s:
        f = Fraction(s)
        if f.denominator not in (1, 2, 4):
            raise UnitError("UNIT_UNKNOWN", f"Unsupported decimal exponent {raw!r}")
        return f
    return Fraction(int(s), 1)


def _preprocess_unit(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return "1"
    # Strip LaTeX wrappers like \mathrm{...}, \text{...}, \unit{...}
    s = re.sub(r"\\(?:mathrm|text|operatorname|unit)\s*\{([^}]*)\}", r"\1", s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\Omega", "Ω").replace(r"\mu", "µ")
    s = s.replace("·", "*").replace("⋅", "*").replace("×", "*").replace("−", "-")
    s = re.sub(r"\bper\b", "/", s, flags=re.I)
    # Replace unicode superscripts attached to tokens with ^exp
    def _sup_repl(m: re.Match) -> str:
        return "^" + m.group(0).translate(_SUPERSCRIPT_MAP)
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", _sup_repl, s)
    # Handle trailing/inline 1/X or ^-1
    return s.strip()


_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(sqrt\()"
    r"|(\()"
    r"|(\))"
    r"|(\^)"
    r"|(\*)"
    r"|(/)"
    r"|(\{[^}]+\})"
    r"|([-+]?\d+(?:\.\d+)?)"
    r"|([$%°ΩµμA-Za-z_][A-Za-z0-9_°Ωµμ]*)"
    r")"
)


def parse_unit(label: str) -> UnitParse:
    """Parse a physical unit string into exact SI rational dimension, scale, offset, and tags."""
    raw = _preprocess_unit(label)
    if not raw or raw == "1":
        return UnitParse(dim=DIM_ZERO, scale=Fraction(1))

    tokens: list[tuple[str, str]] = []
    pos = 0
    n = len(raw)
    while pos < n:
        if raw[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(raw, pos)
        if not m or m.end() == pos:
            raise UnitError("UNIT_UNKNOWN", f"Cannot tokenize unit {label!r} at {raw[pos:]!r}")
        tok = m.group(0).strip()
        if m.group(1):
            tokens.append(("SQRT", tok))
        elif m.group(2):
            tokens.append(("LPAREN", tok))
        elif m.group(3):
            tokens.append(("RPAREN", tok))
        elif m.group(4):
            tokens.append(("CARET", tok))
        elif m.group(5):
            tokens.append(("MUL", tok))
        elif m.group(6):
            tokens.append(("DIV", tok))
        elif m.group(7):
            tokens.append(("BRACE", tok))
        elif m.group(8):
            tokens.append(("NUM", tok))
        else:
            tokens.append(("IDENT", tok))
        pos = m.end()

    idx = 0
    advisories: list[str] = []
    compound_ops = 0

    def peek() -> tuple[str, str] | None:
        return tokens[idx] if idx < len(tokens) else None

    def consume() -> tuple[str, str]:
        nonlocal idx
        t = tokens[idx]
        idx += 1
        return t

    def parse_expr(in_parens: bool = False) -> UnitParse:
        nonlocal compound_ops
        acc = parse_term()
        saw_div = False
        while True:
            t = peek()
            if t is None or t[0] == "RPAREN":
                break
            op = "MUL"
            if t[0] in ("MUL", "DIV"):
                op = consume()[0]
                if op == "DIV":
                    saw_div = True
                elif saw_div and not in_parens:
                    advisories.append("UNIT_AMBIGUOUS")
            elif t[0] in ("IDENT", "LPAREN", "SQRT", "NUM"):
                op = "MUL"
                if saw_div and not in_parens:
                    advisories.append("UNIT_AMBIGUOUS")
            else:
                break
            rhs = parse_term()
            compound_ops += 1
            if acc.kind == "affine" or rhs.kind == "affine":
                raise UnitError("UNIT_AFFINE_IN_COMPOUND", f"Affine unit cannot appear in compound expression {label!r}")
            if op == "MUL":
                new_dim = _dim_add(acc.dim, rhs.dim)
                new_sc = (acc.scale * rhs.scale) if isinstance(acc.scale, Fraction) and isinstance(rhs.scale, Fraction) else float(acc.scale) * float(rhs.scale)
            else:
                new_dim = _dim_sub(acc.dim, rhs.dim)
                new_sc = (acc.scale / rhs.scale) if isinstance(acc.scale, Fraction) and isinstance(rhs.scale, Fraction) else float(acc.scale) / float(rhs.scale)
            acc = UnitParse(dim=new_dim, scale=new_sc, tags=acc.tags | rhs.tags)
        return acc

    def parse_term() -> UnitParse:
        nonlocal compound_ops
        atom = parse_atom()
        t = peek()
        if t is not None and t[0] == "CARET":
            consume()
            nxt = peek()
            if nxt is None:
                raise UnitError("UNIT_UNKNOWN", f"Missing exponent in {label!r}")
            if nxt[0] == "LPAREN":
                consume()
                parts = []
                while peek() is not None and peek()[0] != "RPAREN":  # type: ignore[index]
                    parts.append(consume()[1])
                if peek() is None or peek()[0] != "RPAREN":  # type: ignore[index]
                    raise UnitError("UNIT_UNKNOWN", f"Unclosed exponent parenthesis in {label!r}")
                consume()
                exp = _parse_exponent("".join(parts))
            elif nxt[0] in ("NUM", "BRACE"):
                exp = _parse_exponent(consume()[1])
            else:
                raise UnitError("UNIT_UNKNOWN", f"Invalid exponent token {nxt[1]!r} in {label!r}")
            if atom.kind == "affine" and exp != 1:
                raise UnitError("UNIT_AFFINE_IN_COMPOUND", f"Affine unit cannot be exponentiated in {label!r}")
            compound_ops += 1
            new_dim = _dim_scale(atom.dim, exp)
            if isinstance(atom.scale, Fraction) and exp.denominator == 1:
                new_sc: Fraction | float = atom.scale ** exp.numerator
            else:
                new_sc = float(atom.scale) ** float(exp)
            return UnitParse(dim=new_dim, scale=new_sc, offset=atom.offset if exp == 1 else Fraction(0), kind=atom.kind, tags=atom.tags)
        return atom

    def parse_atom() -> UnitParse:
        t = peek()
        if t is None:
            raise UnitError("UNIT_UNKNOWN", f"Unexpected end of unit {label!r}")
        if t[0] == "SQRT":
            consume()
            inner = parse_expr(in_parens=True)
            if peek() is None or peek()[0] != "RPAREN":  # type: ignore[index]
                raise UnitError("UNIT_UNKNOWN", f"Unclosed sqrt( in {label!r}")
            consume()
            if inner.kind == "affine":
                raise UnitError("UNIT_AFFINE_IN_COMPOUND", f"Affine unit inside sqrt in {label!r}")
            new_dim = _dim_scale(inner.dim, Fraction(1, 2))
            new_sc = math.sqrt(float(inner.scale))
            return UnitParse(dim=new_dim, scale=new_sc, tags=inner.tags)
        if t[0] == "LPAREN":
            consume()
            inner = parse_expr(in_parens=True)
            if peek() is None or peek()[0] != "RPAREN":  # type: ignore[index]
                raise UnitError("UNIT_UNKNOWN", f"Unclosed ( in {label!r}")
            consume()
            return inner
        if t[0] == "NUM":
            val_s = consume()[1]
            if val_s == "1":
                return UnitParse(dim=DIM_ZERO, scale=Fraction(1))
            raise UnitError("UNIT_UNKNOWN", f"Unexpected numeric literal {val_s!r} in unit {label!r}")
        if t[0] == "IDENT":
            return _lookup_atom(consume()[1])
        raise UnitError("UNIT_UNKNOWN", f"Unexpected token {t[1]!r} in unit {label!r}")

    res = parse_expr(in_parens=False)
    if idx != len(tokens):
        raise UnitError("UNIT_UNKNOWN", f"Trailing tokens in unit {label!r}")
    return UnitParse(
        dim=res.dim,
        scale=res.scale,
        offset=res.offset,
        kind=res.kind,
        tags=res.tags,
        advisories=tuple(dict.fromkeys(advisories)),
    )

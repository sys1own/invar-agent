"""Deterministic, token-level expression front end for the LLM->deterministic boundary.

Pure Python, closed lexer + precedence-climbing parser.
- Fast path: expressions that already pass ``meaningfulness._parse`` (and are free of
  unsafe constant power-towers) are returned BYTE-IDENTICAL with zero transforms.
- Meaning-preserving, logged normalizations only; never guesses on English prose,
  approximate operators (``≈``, ``~``), or unknown adjacent identifier runs.
- Never raises; enforces input size, token count, nesting depth, and wall-time caps.
- Final strict gate: every emitted canonical string is verified with ``meaningfulness._parse``.
"""

from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import meaningfulness as m


# Closed set of SI / scientific unit tokens that may trail a numeric literal or interval
# ONLY when the token is NOT a declared symbol in the model.
_CLOSED_SI_UNITS: frozenset[str] = frozenset({
    "K", "J", "W", "kW", "MW", "GW", "mW", "uW", "nW",
    "Pa", "kPa", "MPa", "GPa",
    "s", "ms", "us", "ns", "ps", "hr", "day", "days",
    "Hz", "kHz", "MHz", "GHz",
    "m", "km", "cm", "mm", "um", "nm",
    "kg", "g", "mg", "ug",
    "mol", "mmol", "umol", "M", "mM", "uM", "nM",
    "A", "mA", "uA", "V", "mV", "kV", "C", "N", "kN",
    "rad", "deg", "eV", "keV", "MeV", "GeV",
    "cd", "lm", "lx", "T", "mT", "Wb", "H", "F", "Ohm", "S",
    "bit", "bits",
})

_MATH_PREFIXES: frozenset[str] = frozenset({"math", "np", "numpy"})
_PREFIX_FN_MAP: dict[str, str] = {
    "exp": "exp",
    "log": "log",
    "ln": "log",
    "sqrt": "sqrt",
    "sin": "sin",
    "cos": "cos",
    "abs": "abs",
    "fabs": "abs",
    "pow": "pow",
    "min": "min",
    "max": "max",
}

_APPROX_CHARS: frozenset[str] = frozenset({"≈", "~", "≃", "≅", "∼", "±"})
_APPROX_WORDS: frozenset[str] = frozenset({"approximately", "approx"})

_UNICODE_OP_MAP: dict[str, tuple[str, str]] = {
    "≤": ("<=", "unicode_le"),
    "≥": (">=", "unicode_ge"),
    "≠": ("!=", "unicode_ne"),
    "−": ("-", "unicode_minus"),
    "–": ("-", "unicode_en_dash"),
    "—": ("-", "unicode_em_dash"),
    "×": ("*", "unicode_times"),
    "·": ("*", "unicode_cdot"),
    "÷": ("/", "unicode_div"),
    "∧": ("and", "unicode_and"),
    "∨": ("or", "unicode_or"),
    "¬": ("not", "unicode_not"),
    "→": ("->", "unicode_arrow"),
    "⇒": ("->", "unicode_double_arrow"),
    "∈": ("in", "unicode_in"),
    "∀": ("FORALL", "unicode_forall"),
    "∃": ("EXISTS", "unicode_exists"),
    "√": ("√", "unicode_sqrt"),
}


@dataclass(frozen=True)
class ExprDiagnostic:
    code: str
    message: str
    original: str
    column: Optional[int] = None
    offending_token: Optional[str] = None
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "original": self.original,
        }
        if self.column is not None:
            out["column"] = self.column
        if self.offending_token is not None:
            out["offending_token"] = self.offending_token
        if self.hint:
            out["hint"] = self.hint
        return out


@dataclass(frozen=True)
class QuantifierBinder:
    symbol: str
    lo_expr: str
    hi_expr: str
    lo_open: bool
    hi_open: bool
    assumption_expr: str


@dataclass(frozen=True)
class NormalizedExpr:
    canonical: Optional[str]
    transforms: tuple[str, ...] = ()
    diagnostics: tuple[ExprDiagnostic, ...] = ()
    quantifier: Optional[str] = None
    binders: tuple[QuantifierBinder, ...] = ()
    binder_assumptions: tuple[str, ...] = ()
    stripped_units: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        return self.canonical is not None and not self.diagnostics


@dataclass(frozen=True)
class _Tok:
    kind: str  # NUM, IDENT, OP, EOF
    text: str
    col: int  # 1-based column in original text
    end_col: int  # 1-based exclusive end column
    preceded_by_ws: bool = True


def _has_unsafe_power_tower(node: ast.AST) -> bool:
    """Reject constant exponentiation towers like 9**9**9 that would overflow or DoS."""
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            # Nested constant power tower a ** (b ** c)
            if isinstance(n.right, ast.BinOp) and isinstance(n.right.op, ast.Pow):
                if (
                    isinstance(n.left, ast.Constant)
                    and isinstance(n.right.left, ast.Constant)
                    and isinstance(n.right.right, ast.Constant)
                ):
                    try:
                        b = float(n.right.left.value)
                        c = float(n.right.right.value)
                        if abs(c) > 20 or (abs(b) > 1 and abs(c) > 4):
                            return True
                    except (TypeError, ValueError, OverflowError):
                        return True
            if isinstance(n.right, ast.Constant) and isinstance(n.right.value, (int, float)):
                if abs(float(n.right.value)) > 1000:
                    return True
    return False


def _strip_surface_wrappers(text: str) -> tuple[str, int, list[str]]:
    """Strip outer whitespace, markdown fences, backticks, $..$, comments, trailing punctuation.

    Returns (cleaned, base_col_offset, transforms).
    """
    transforms: list[str] = []
    s = text
    offset = 0

    # Leading/trailing whitespace
    stripped = s.lstrip()
    if len(stripped) != len(s):
        offset += len(s) - len(stripped)
        s = stripped
    s = s.rstrip()

    # Markdown fenced block ```...```
    if s.startswith("```") and s.endswith("```") and len(s) >= 6:
        inner = s[3:-3]
        nl = inner.find("\n")
        if nl != -1 and inner[:nl].strip().isalpha():
            offset += 3 + nl + 1
            s = inner[nl + 1:].strip()
        else:
            offset += 3
            s = inner.strip()
        transforms.append("strip_markdown_fence")

    # Backticks `...`
    if len(s) >= 2 and s[0] == "`" and s[-1] == "`":
        offset += 1
        s = s[1:-1].strip()
        transforms.append("strip_backticks")

    # LaTeX math delimiters $$...$$ or $...$
    if len(s) >= 4 and s.startswith("$$") and s.endswith("$$"):
        offset += 2
        s = s[2:-2].strip()
        transforms.append("strip_latex_math")
    elif len(s) >= 2 and s.startswith("$") and s.endswith("$") and "$" not in s[1:-1]:
        offset += 1
        s = s[1:-1].strip()
        transforms.append("strip_latex_math")

    # Strip inline # or // comments (not inside anything since strings aren't supported)
    for marker in ("#", "//"):
        idx = s.find(marker)
        if idx != -1:
            s = s[:idx].rstrip()
            transforms.append("strip_comment")

    # Strip trailing sentence punctuation: ; or , or trailing . (not decimal like `1.`)
    while s:
        if s[-1] in (";", ","):
            s = s[:-1].rstrip()
            transforms.append("strip_trailing_punct")
        elif s[-1] == ".":
            # Only strip if it's not a standalone decimal like "1." (check if preceded by non-digit
            # or if the numeric token already contains a dot like "0.85.")
            if len(s) == 1:
                break
            prev = s[-2]
            if not prev.isdigit():
                s = s[:-1].rstrip()
                transforms.append("strip_trailing_punct")
            else:
                # Scan backwards through digits to see if a '.' already precedes them
                j = len(s) - 2
                while j >= 0 and s[j].isdigit():
                    j -= 1
                if j >= 0 and s[j] == ".":
                    s = s[:-1].rstrip()
                    transforms.append("strip_trailing_punct")
                else:
                    break
        else:
            break

    return s, offset, transforms


def _lex(
    s: str,
    original: str,
    base_offset: int,
    *,
    max_tokens: int,
    deadline: float,
) -> tuple[list[_Tok], list[str], Optional[ExprDiagnostic]]:
    tokens: list[_Tok] = []
    transforms: list[str] = []
    n = len(s)
    i = 0
    prev_was_ws = True

    while i < n:
        if time.monotonic() > deadline:
            return [], transforms, ExprDiagnostic(
                code="EXPRESSION_TIMEOUT",
                message="Expression lexing exceeded time budget.",
                original=original,
                column=base_offset + i + 1,
                hint="Simplify the expression.",
            )
        if len(tokens) >= max_tokens:
            return [], transforms, ExprDiagnostic(
                code="EXPRESSION_TOO_LARGE",
                message=f"Expression exceeds maximum token limit ({max_tokens}).",
                original=original,
                column=base_offset + i + 1,
                hint="Simplify the expression.",
            )

        ch = s[i]
        if ch.isspace():
            prev_was_ws = True
            i += 1
            continue

        col = base_offset + i + 1

        # Reject approximate/fuzzy operators immediately
        if ch in _APPROX_CHARS:
            return [], transforms, ExprDiagnostic(
                code="APPROXIMATE_OPERATOR",
                message=f"Approximate operator {ch!r} is not allowed in formal expressions.",
                original=original,
                column=col,
                offending_token=ch,
                hint="Use an exact inequality (e.g. abs(a - b) <= tol) instead of an approximate symbol.",
            )

        # Unicode superscripts ² ³ -> ** 2 / ** 3
        if ch in ("²", "³"):
            exp_val = "2" if ch == "²" else "3"
            tokens.append(_Tok("OP", "**", col, col + 1, False))
            tokens.append(_Tok("NUM", exp_val, col, col + 1, False))
            transforms.append(f"unicode_sup_{exp_val}")
            prev_was_ws = False
            i += 1
            continue

        # Unicode mapped operators
        if ch in _UNICODE_OP_MAP:
            mapped, tag = _UNICODE_OP_MAP[ch]
            kind = "IDENT" if mapped in ("and", "or", "not", "in", "FORALL", "EXISTS") else "OP"
            tokens.append(_Tok(kind, mapped, col, col + 1, prev_was_ws))
            transforms.append(tag)
            prev_was_ws = False
            i += 1
            continue

        # Numbers: digits or .digits
        if ch.isdigit() or (ch == "." and i + 1 < n and s[i + 1].isdigit()):
            start = i
            has_dot = False
            if s[i] == ".":
                has_dot = True
                i += 1
            while i < n and s[i].isdigit():
                i += 1
            if i < n and s[i] == "." and not has_dot:
                has_dot = True
                i += 1
                while i < n and s[i].isdigit():
                    i += 1
            # Optional scientific exponent
            if i < n and s[i] in ("e", "E"):
                exp_start = i
                i += 1
                if i < n and s[i] in ("+", "-"):
                    i += 1
                if i < n and s[i].isdigit():
                    while i < n and s[i].isdigit():
                        i += 1
                else:
                    # Not a valid exponent; backtrack so 'e' can be lexed as IDENT (e.g. 2e)
                    i = exp_start
            num_text = s[start:i]
            tokens.append(_Tok("NUM", num_text, col, base_offset + i + 1, prev_was_ws))
            prev_was_ws = False
            continue

        # ASCII Identifiers
        if ch.isascii() and (ch.isalpha() or ch == "_"):
            start = i
            i += 1
            while i < n and s[i].isascii() and (s[i].isalnum() or s[i] == "_"):
                i += 1
            ident = s[start:i]
            if ident.lower() in _APPROX_WORDS:
                return [], transforms, ExprDiagnostic(
                    code="APPROXIMATE_OPERATOR",
                    message=f"Approximate qualifier {ident!r} is not allowed in formal expressions.",
                    original=original,
                    column=col,
                    offending_token=ident,
                    hint="Replace approximate prose with an exact inequality or move it to notes[].",
                )
            tokens.append(_Tok("IDENT", ident, col, base_offset + i + 1, prev_was_ws))
            prev_was_ws = False
            continue

        # Multi-character ASCII operators
        two = s[i:i + 2]
        if two in ("**", "<=", ">=", "==", "!="):
            tokens.append(_Tok("OP", two, col, col + 2, prev_was_ws))
            prev_was_ws = False
            i += 2
            continue
        if two == "<>":
            tokens.append(_Tok("OP", "!=", col, col + 2, prev_was_ws))
            transforms.append("alias_ne_diamond")
            prev_was_ws = False
            i += 2
            continue
        if two in ("->", "=>"):
            tokens.append(_Tok("OP", "->", col, col + 2, prev_was_ws))
            transforms.append("arrow_implies")
            prev_was_ws = False
            i += 2
            continue
        if two == "&&":
            tokens.append(_Tok("IDENT", "and", col, col + 2, prev_was_ws))
            transforms.append("alias_and_amp")
            prev_was_ws = False
            i += 2
            continue
        if two == "||":
            tokens.append(_Tok("IDENT", "or", col, col + 2, prev_was_ws))
            transforms.append("alias_or_pipe")
            prev_was_ws = False
            i += 2
            continue

        # Single-character ASCII operators
        if ch == "^":
            tokens.append(_Tok("OP", "**", col, col + 1, prev_was_ws))
            transforms.append("alias_caret_pow")
            prev_was_ws = False
            i += 1
            continue
        if ch == "!":
            tokens.append(_Tok("IDENT", "not", col, col + 1, prev_was_ws))
            transforms.append("alias_bang_not")
            prev_was_ws = False
            i += 1
            continue
        if ch in "+-*/<>()=,:.[ ]":
            if ch != " ":
                tokens.append(_Tok("OP", ch, col, col + 1, prev_was_ws))
                prev_was_ws = False
            i += 1
            continue

        return [], transforms, ExprDiagnostic(
            code="UNSUPPORTED_TOKEN",
            message=f"Unsupported character {ch!r} in formal expression.",
            original=original,
            column=col,
            offending_token=ch,
            hint="Use only supported Python-subset arithmetic/boolean syntax over declared symbols.",
        )

    tokens.append(_Tok("EOF", "", base_offset + n + 1, base_offset + n + 1, True))
    return tokens, transforms, None


def _rewrite_tokens(
    tokens: list[_Tok],
    declared: set[str],
    transforms: list[str],
    stripped_units: list[tuple[str, str]],
) -> list[_Tok]:
    """Apply local, exact token-level rewrites (math.* prefix, ln->log, AND/OR/NOT, trailing SI units)."""
    # Pass 1: math./np./numpy. prefix collapse, ln( -> log(, AND/OR/NOT aliases
    out: list[_Tok] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        # math.exp / np.sqrt / numpy.log etc.
        if (
            tok.kind == "IDENT"
            and tok.text in _MATH_PREFIXES
            and tok.text not in declared
            and i + 2 < n
            and tokens[i + 1].kind == "OP"
            and tokens[i + 1].text == "."
            and tokens[i + 2].kind == "IDENT"
            and tokens[i + 2].text in _PREFIX_FN_MAP
        ):
            fn = _PREFIX_FN_MAP[tokens[i + 2].text]
            out.append(_Tok("IDENT", fn, tok.col, tokens[i + 2].end_col, tok.preceded_by_ws))
            transforms.append(f"strip_prefix_{tok.text}")
            i += 3
            continue

        # ln( -> log(
        if (
            tok.kind == "IDENT"
            and tok.text == "ln"
            and "ln" not in declared
            and i + 1 < n
            and tokens[i + 1].kind == "OP"
            and tokens[i + 1].text == "("
        ):
            out.append(_Tok("IDENT", "log", tok.col, tok.end_col, tok.preceded_by_ws))
            transforms.append("alias_ln")
            i += 1
            continue

        # ASCII uppercase/mixed keyword aliases: AND, OR, NOT (only if not declared symbol)
        if tok.kind == "IDENT" and tok.text not in declared:
            upper = tok.text.upper()
            if upper == "AND" and tok.text != "and":
                out.append(_Tok("IDENT", "and", tok.col, tok.end_col, tok.preceded_by_ws))
                transforms.append("alias_AND")
                i += 1
                continue
            if upper == "OR" and tok.text != "or":
                out.append(_Tok("IDENT", "or", tok.col, tok.end_col, tok.preceded_by_ws))
                transforms.append("alias_OR")
                i += 1
                continue
            if upper == "NOT" and tok.text != "not":
                out.append(_Tok("IDENT", "not", tok.col, tok.end_col, tok.preceded_by_ws))
                transforms.append("alias_NOT")
                i += 1
                continue
            # Infix 'implies' (not followed by '(') -> '->'
            if tok.text.lower() == "implies":
                if not (i + 1 < n and tokens[i + 1].kind == "OP" and tokens[i + 1].text == "("):
                    out.append(_Tok("OP", "->", tok.col, tok.end_col, tok.preceded_by_ws))
                    transforms.append("infix_implies")
                    i += 1
                    continue

        out.append(tok)
        i += 1

    # Pass 2: strip closed SI unit tokens immediately after NUM (e.g. `300 K`, `200 J`, `20 W/K`)
    # ONLY if the unit token is NOT in declared_symbols.
    out2: list[_Tok] = []
    i = 0
    m_len = len(out)
    while i < m_len:
        tok = out[i]
        out2.append(tok)
        if tok.kind == "NUM" and i + 1 < m_len:
            nxt = out[i + 1]
            if (
                nxt.kind == "IDENT"
                and nxt.text in _CLOSED_SI_UNITS
                and nxt.text not in declared
                and not (i + 2 < m_len and out[i + 2].kind == "OP" and out[i + 2].text == "(")
            ):
                # Check for compound unit like W/K or m/s
                if (
                    i + 3 < m_len
                    and out[i + 2].kind == "OP"
                    and out[i + 2].text == "/"
                    and out[i + 3].kind == "IDENT"
                    and out[i + 3].text in _CLOSED_SI_UNITS
                    and out[i + 3].text not in declared
                ):
                    unit_str = f"{nxt.text}/{out[i + 3].text}"
                    stripped_units.append((tok.text, unit_str))
                    transforms.append(f"strip_unit:{unit_str}")
                    i += 4
                    continue
                stripped_units.append((tok.text, nxt.text))
                transforms.append(f"strip_unit:{nxt.text}")
                i += 2
                continue
        i += 1

    return out2


class _ParseError(Exception):
    def __init__(self, code: str, message: str, tok: _Tok, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.tok = tok
        self.hint = hint


class _Parser:
    def __init__(
        self,
        tokens: list[_Tok],
        original: str,
        declared: set[str],
        domain: Optional[Mapping[str, Sequence[float]]],
        transforms: list[str],
        stripped_units: list[tuple[str, str]],
        *,
        max_depth: int,
        deadline: float,
    ) -> None:
        self.tokens = tokens
        self.pos = 0
        self.original = original
        self.declared = declared
        self.domain = domain
        self.transforms = transforms
        self.stripped_units = stripped_units
        self.max_depth = max_depth
        self.deadline = deadline
        self.depth = 0
        self.call_depth = 0
        self.quantifier: Optional[str] = None
        self.binders: list[QuantifierBinder] = []
        self.binder_assumptions: list[str] = []
        self.binder_diagnostics: list[ExprDiagnostic] = []

    def _check_budget(self) -> None:
        if time.monotonic() > self.deadline:
            raise _ParseError("EXPRESSION_TIMEOUT", "Expression parsing exceeded time budget.", self._cur())
        if self.depth > self.max_depth:
            raise _ParseError("EXPRESSION_TOO_DEEP", f"Expression exceeds nesting depth cap ({self.max_depth}).", self._cur())

    def _cur(self) -> _Tok:
        return self.tokens[self.pos]

    def _peek(self, offset: int = 1) -> _Tok:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def _eat(self) -> _Tok:
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def _match_op(self, op: str) -> bool:
        t = self._cur()
        if t.kind == "OP" and t.text == op:
            self.pos += 1
            return True
        return False

    def _match_kw(self, kw: str) -> bool:
        t = self._cur()
        if t.kind == "IDENT" and t.text == kw:
            self.pos += 1
            return True
        return False

    def _try_quantifier_prefix(self) -> None:
        t0 = self._cur()
        q_kind: Optional[str] = None
        advance = 0
        if t0.kind == "IDENT":
            low = t0.text.lower()
            if low in ("forall",):
                q_kind, advance = "FORALL", 1
            elif low == "for" and self._peek().kind == "IDENT" and self._peek().text.lower() in ("all", "every"):
                q_kind, advance = "FORALL", 2
            elif low in ("exists",):
                q_kind, advance = "EXISTS", 1
            elif low == "there" and self._peek().kind == "IDENT" and self._peek().text.lower() == "exists":
                q_kind, advance = "EXISTS", 2
        if q_kind is None:
            return

        # Speculatively parse quantifier prefix; must have at least one `<sym> in <interval>`
        saved_pos = self.pos
        self.pos += advance
        binders: list[QuantifierBinder] = []
        while True:
            sym_tok = self._cur()
            if sym_tok.kind != "IDENT":
                self.pos = saved_pos
                return
            in_tok = self._peek()
            if not (in_tok.kind == "IDENT" and in_tok.text.lower() == "in"):
                self.pos = saved_pos
                return
            self.pos += 2  # consume sym, 'in'
            binder = self._parse_interval_binder(sym_tok)
            if binder is None:
                self.pos = saved_pos
                return
            binders.append(binder)
            # Optional separator between binders: 'and' or ',' (followed by another `<sym> in`)
            if self._cur().kind == "IDENT" and self._cur().text == "and":
                if (
                    self._peek(1).kind == "IDENT"
                    and self._peek(2).kind == "IDENT"
                    and self._peek(2).text.lower() == "in"
                ):
                    self.pos += 1
                    continue
            if self._cur().kind == "OP" and self._cur().text == ",":
                if (
                    self._peek(1).kind == "IDENT"
                    and self._peek(2).kind == "IDENT"
                    and self._peek(2).text.lower() == "in"
                ):
                    self.pos += 1
                    continue
            break

        # Consume separator ':' or ',' before body
        if self._cur().kind == "OP" and self._cur().text in (":", ","):
            self.pos += 1
        elif self._cur().kind == "EOF":
            self.pos = saved_pos
            return

        self.quantifier = q_kind
        self.binders = binders
        self.transforms.append(f"strip_quantifier_{q_kind.lower()}")
        for b in binders:
            self.binder_assumptions.append(b.assumption_expr)
            self._check_binder_against_domain(b, sym_tok)

    def _parse_interval_binder(self, sym_tok: _Tok) -> Optional[QuantifierBinder]:
        open_tok = self._cur()
        if open_tok.kind != "OP" or open_tok.text not in ("[", "("):
            return None
        lo_open = open_tok.text == "("
        self.pos += 1
        try:
            lo_node = self._parse_add()
        except _ParseError:
            return None
        if not self._match_op(","):
            return None
        try:
            hi_node = self._parse_add()
        except _ParseError:
            return None
        close_tok = self._cur()
        if close_tok.kind != "OP" or close_tok.text not in ("]", ")"):
            return None
        hi_open = close_tok.text == ")"
        self.pos += 1
        # Optional trailing SI unit after interval (e.g. [0, 200] J)
        if (
            self._cur().kind == "IDENT"
            and self._cur().text in _CLOSED_SI_UNITS
            and self._cur().text not in self.declared
        ):
            unit_tok = self._eat()
            self.stripped_units.append((f"[{ast.unparse(lo_node)},{ast.unparse(hi_node)}]", unit_tok.text))
            self.transforms.append(f"strip_unit:{unit_tok.text}")

        lo_str = ast.unparse(lo_node)
        hi_str = ast.unparse(hi_node)
        lo_op = "<" if lo_open else "<="
        hi_op = "<" if hi_open else "<="
        assumption = f"{lo_str} {lo_op} {sym_tok.text} {hi_op} {hi_str}"
        return QuantifierBinder(
            symbol=sym_tok.text,
            lo_expr=lo_str,
            hi_expr=hi_str,
            lo_open=lo_open,
            hi_open=hi_open,
            assumption_expr=assumption,
        )

    def _check_binder_against_domain(self, binder: QuantifierBinder, tok: _Tok) -> None:
        if not self.domain or binder.symbol not in self.domain:
            return
        pair = self.domain[binder.symbol]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return
        try:
            dom_lo, dom_hi = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            return
        lo_val = self._eval_bound_expr(binder.lo_expr)
        hi_val = self._eval_bound_expr(binder.hi_expr)
        if lo_val is None or hi_val is None:
            return
        tol = 1e-12 * max(1.0, abs(dom_lo), abs(dom_hi))
        if lo_val < dom_lo - tol or hi_val > dom_hi + tol or lo_val > hi_val + tol:
            self.binder_diagnostics.append(ExprDiagnostic(
                code="BINDER_OUTSIDE_DOMAIN",
                message=(
                    f"Quantifier binder {binder.symbol} in [{lo_val}, {hi_val}] "
                    f"is not a subset of declared domain[{binder.symbol!r}] = [{dom_lo}, {dom_hi}]."
                ),
                original=self.original,
                column=tok.col,
                offending_token=binder.symbol,
                hint=f"Ensure binder bounds for {binder.symbol} lie within domain[{binder.symbol!r}].",
            ))

    def _eval_bound_expr(self, expr_str: str) -> Optional[float]:
        try:
            node = m._parse(expr_str)
        except Exception:
            return None
        if _has_unsafe_power_tower(node):
            return None
        vars_in = m._variables(node)
        if not vars_in:
            try:
                iv = m._iv(node, {})
                if math.isfinite(iv.lo) and math.isfinite(iv.hi) and abs(iv.hi - iv.lo) <= 1e-9 * max(1.0, abs(iv.lo)):
                    return 0.5 * (float(iv.lo) + float(iv.hi))
            except Exception:
                return None
        return None

    def parse(self) -> ast.AST:
        self._try_quantifier_prefix()
        node = self._parse_ternary()
        if self._cur().kind != "EOF":
            tok = self._cur()
            raise _ParseError(
                "UNEXPECTED_TOKEN",
                f"Unexpected token {tok.text!r} at column {tok.col}; prose or malformed expression.",
                tok,
                hint="Write a single formal expression over declared symbols; move prose explanations to notes[].",
            )
        return node

    # Precedence 0 (lowest): Ternary conditional `a if cond else b` (UPG-006)
    def _parse_ternary(self) -> ast.AST:
        self._check_budget()
        body = self._parse_implies()
        if self._match_kw("if"):
            self.depth += 1
            try:
                test = self._parse_implies()
                if not self._match_kw("else"):
                    raise _ParseError(
                        "EXPECTED_ELSE",
                        "Expected 'else' in conditional expression.",
                        self._cur(),
                    )
                orelse = self._parse_ternary()
            finally:
                self.depth -= 1
            return ast.IfExp(test=test, body=body, orelse=orelse)
        return body

    # Precedence 1 (lowest): Implication -> (right-associative)
    def _parse_implies(self) -> ast.AST:
        self._check_budget()
        left = self._parse_or()
        if self._match_op("->"):
            self.depth += 1
            try:
                right = self._parse_implies()  # right-associative
            finally:
                self.depth -= 1
            return ast.Call(func=ast.Name(id="implies", ctx=ast.Load()), args=[left, right], keywords=[])
        return left

    # Precedence 2: Logical OR
    def _parse_or(self) -> ast.AST:
        self._check_budget()
        values = [self._parse_and()]
        while self._match_kw("or"):
            values.append(self._parse_and())
        return values[0] if len(values) == 1 else ast.BoolOp(op=ast.Or(), values=values)

    # Precedence 3: Logical AND
    def _parse_and(self) -> ast.AST:
        self._check_budget()
        values = [self._parse_not()]
        while self._match_kw("and"):
            values.append(self._parse_not())
        return values[0] if len(values) == 1 else ast.BoolOp(op=ast.And(), values=values)

    # Precedence 4: Logical NOT
    def _parse_not(self) -> ast.AST:
        self._check_budget()
        if self._match_kw("not"):
            self.depth += 1
            try:
                operand = self._parse_not()
            finally:
                self.depth -= 1
            return ast.UnaryOp(op=ast.Not(), operand=operand)
        return self._parse_comparison()

    # Precedence 5: Comparisons and interval membership `x in [a, b]`
    def _parse_comparison(self) -> ast.AST:
        self._check_budget()
        left = self._parse_add()

        # Check interval membership: `<expr> in [a, b]` / `<expr> in (a, b]`
        if self._cur().kind == "IDENT" and self._cur().text == "in":
            in_tok = self._eat()
            open_tok = self._cur()
            if open_tok.kind != "OP" or open_tok.text not in ("[", "("):
                raise _ParseError(
                    "INVALID_MEMBERSHIP",
                    f"Expected interval '[' or '(' after 'in' at column {open_tok.col}.",
                    open_tok,
                    hint="Use interval notation like `x in [0, 1]` or explicit comparisons `0 <= x <= 1`.",
                )
            lo_open = open_tok.text == "("
            self.pos += 1
            lo_node = self._parse_add()
            if not self._match_op(","):
                raise _ParseError(
                    "INVALID_MEMBERSHIP",
                    f"Expected ',' in membership interval at column {self._cur().col}.",
                    self._cur(),
                    hint="Intervals must have two bounds: `[lo, hi]`.",
                )
            hi_node = self._parse_add()
            close_tok = self._cur()
            if close_tok.kind != "OP" or close_tok.text not in ("]", ")"):
                raise _ParseError(
                    "INVALID_MEMBERSHIP",
                    f"Expected ']' or ')' to close membership interval at column {close_tok.col}.",
                    close_tok,
                )
            hi_open = close_tok.text == ")"
            self.pos += 1
            # Optional trailing closed SI unit (e.g. `Qin in [1, 200] J`)
            if (
                self._cur().kind == "IDENT"
                and self._cur().text in _CLOSED_SI_UNITS
                and self._cur().text not in self.declared
            ):
                unit_tok = self._eat()
                self.stripped_units.append((f"[{ast.unparse(lo_node)},{ast.unparse(hi_node)}]", unit_tok.text))
                self.transforms.append(f"strip_unit:{unit_tok.text}")
            self.transforms.append("membership_to_chained_cmp")
            lo_op: ast.cmpop = ast.Lt() if lo_open else ast.LtE()
            hi_op: ast.cmpop = ast.Lt() if hi_open else ast.LtE()
            return ast.Compare(left=lo_node, ops=[lo_op, hi_op], comparators=[left, hi_node])

        ops: list[ast.cmpop] = []
        comparators: list[ast.AST] = []
        while True:
            t = self._cur()
            cmp_op: Optional[ast.cmpop] = None
            if t.kind == "OP":
                if t.text == "<":
                    cmp_op = ast.Lt()
                elif t.text == "<=":
                    cmp_op = ast.LtE()
                elif t.text == ">":
                    cmp_op = ast.Gt()
                elif t.text == ">=":
                    cmp_op = ast.GtE()
                elif t.text == "==":
                    cmp_op = ast.Eq()
                elif t.text == "!=":
                    cmp_op = ast.NotEq()
                elif t.text == "=":
                    if self.call_depth > 0:
                        raise _ParseError(
                            "ASSIGNMENT_IN_CALL",
                            f"Single '=' inside function call at column {t.col} is not allowed.",
                            t,
                            hint="Use positional arguments and '==' for equality comparisons.",
                        )
                    cmp_op = ast.Eq()
                    self.transforms.append("single_eq_to_eq")
            if cmp_op is None:
                break
            self.pos += 1
            ops.append(cmp_op)
            comparators.append(self._parse_add())

        if not ops:
            return left
        return ast.Compare(left=left, ops=ops, comparators=comparators)

    # Precedence 6: Additive +, -
    def _parse_add(self) -> ast.AST:
        self._check_budget()
        node = self._parse_mul()
        while self._cur().kind == "OP" and self._cur().text in ("+", "-"):
            op_tok = self._eat()
            rhs = self._parse_mul()
            op: ast.operator = ast.Add() if op_tok.text == "+" else ast.Sub()
            node = ast.BinOp(left=node, op=op, right=rhs)
        return node

    # Precedence 7: Multiplicative *, / and implicit multiplication (`2R0`, `2(x+1)`, `)(`)
    def _parse_mul(self) -> ast.AST:
        self._check_budget()
        node = self._parse_unary()
        while True:
            t = self._cur()
            if t.kind == "OP" and t.text in ("*", "/"):
                self.pos += 1
                rhs = self._parse_unary()
                op: ast.operator = ast.Mult() if t.text == "*" else ast.Div()
                node = ast.BinOp(left=node, op=op, right=rhs)
                continue
            # Implicit multiplication:
            # 1) NUM followed by adjacent IDENT (no whitespace, e.g. `2R0`, `3x`) or `(` (e.g. `2(x+1)`)
            # 2) `)` followed by `(` (e.g. `(a+b)(c+d)`)
            prev_tok = self.tokens[self.pos - 1] if self.pos > 0 else None
            if prev_tok is not None:
                if prev_tok.kind == "NUM" and (
                    (t.kind == "IDENT" and not t.preceded_by_ws and t.text.lower() not in ("and", "or", "not", "in", "implies"))
                    or (t.kind == "OP" and t.text == "(")
                ):
                    self.transforms.append("implicit_mul")
                    rhs = self._parse_power()
                    node = ast.BinOp(left=node, op=ast.Mult(), right=rhs)
                    continue
                if prev_tok.kind == "OP" and prev_tok.text == ")" and t.kind == "OP" and t.text == "(":
                    self.transforms.append("implicit_mul")
                    rhs = self._parse_power()
                    node = ast.BinOp(left=node, op=ast.Mult(), right=rhs)
                    continue
            break
        return node

    # Precedence 8: Unary +, -
    def _parse_unary(self) -> ast.AST:
        self._check_budget()
        t = self._cur()
        if t.kind == "OP" and t.text in ("+", "-"):
            self.pos += 1
            self.depth += 1
            try:
                operand = self._parse_unary()
            finally:
                self.depth -= 1
            return ast.UnaryOp(op=ast.UAdd() if t.text == "+" else ast.USub(), operand=operand)
        return self._parse_power()

    # Precedence 9: Power ** (right-associative; RHS may be unary e.g. 2**-3)
    def _parse_power(self) -> ast.AST:
        self._check_budget()
        base = self._parse_primary()
        if self._match_op("**"):
            self.depth += 1
            try:
                exp = self._parse_unary()
            finally:
                self.depth -= 1
            return ast.BinOp(left=base, op=ast.Pow(), right=exp)
        return base

    # Precedence 10: Primary
    def _parse_primary(self) -> ast.AST:
        self._check_budget()
        t = self._cur()

        # Unicode √x or √(expr)
        if t.kind == "OP" and t.text == "√":
            self.pos += 1
            self.depth += 1
            try:
                arg = self._parse_primary()
            finally:
                self.depth -= 1
            return ast.Call(func=ast.Name(id="sqrt", ctx=ast.Load()), args=[arg], keywords=[])

        if t.kind == "NUM":
            self.pos += 1
            if "." in t.text or "e" in t.text or "E" in t.text:
                val: int | float = float(t.text)
                if not math.isfinite(val):
                    raise _ParseError("NON_FINITE_CONSTANT", f"Non-finite numeric constant {t.text!r}.", t)
            else:
                val = int(t.text)
            return ast.Constant(value=val)

        if t.kind == "IDENT":
            if t.text in ("True", "False"):
                self.pos += 1
                return ast.Constant(value=(t.text == "True"))
            if t.text in ("and", "or", "not", "in"):
                raise _ParseError(
                    "UNEXPECTED_KEYWORD",
                    f"Unexpected keyword {t.text!r} at column {t.col}.",
                    t,
                    hint="Check operator placement and parentheses.",
                )
            self.pos += 1
            if self._match_op("("):
                self.depth += 1
                self.call_depth += 1
                try:
                    args: list[ast.AST] = []
                    if not (self._cur().kind == "OP" and self._cur().text == ")"):
                        args.append(self._parse_implies())
                        while self._match_op(","):
                            args.append(self._parse_implies())
                    if not self._match_op(")"):
                        raise _ParseError(
                            "UNCLOSED_CALL",
                            f"Expected ')' to close call {t.text}(...) at column {self._cur().col}.",
                            self._cur(),
                        )
                finally:
                    self.call_depth -= 1
                    self.depth -= 1
                # Fold variadic min/max (>=3 args) into nested binary calls
                if t.text in ("min", "max") and len(args) >= 3:
                    self.transforms.append(f"fold_variadic_{t.text}")
                    acc = args[-1]
                    for arg in reversed(args[:-1]):
                        acc = ast.Call(func=ast.Name(id=t.text, ctx=ast.Load()), args=[arg, acc], keywords=[])
                    return acc
                return ast.Call(func=ast.Name(id=t.text, ctx=ast.Load()), args=args, keywords=[])
            return ast.Name(id=t.text, ctx=ast.Load())

        if self._match_op("("):
            self.depth += 1
            try:
                inner = self._parse_implies()
                if not self._match_op(")"):
                    raise _ParseError(
                        "UNCLOSED_PAREN",
                        f"Expected ')' at column {self._cur().col}.",
                        self._cur(),
                    )
            finally:
                self.depth -= 1
            return inner

        raise _ParseError(
            "UNEXPECTED_TOKEN",
            f"Unexpected token {t.text!r} at column {t.col}.",
            t,
            hint="Use formal arithmetic/boolean expressions over declared symbols.",
        )


def normalize_expression(
    text: str,
    *,
    declared_symbols: Iterable[str] = (),
    domain: Optional[Mapping[str, Sequence[float]]] = None,
    si_dims: Optional[Mapping[str, Sequence[int]]] = None,
    quantifier_mode: Optional[str] = None,
    max_chars: int = 16384,
    max_tokens: int = 2048,
    max_depth: int = 64,
    time_budget_s: float = 0.25,
) -> NormalizedExpr:
    """Deterministically normalize a formal expression or return structured diagnostics.

    Never raises. Already-valid expressions pass through BYTE-IDENTICAL.
    """
    _ = si_dims  # Reserved for downstream dimensional consistency hooks; SI check runs in IhcgCompiler.
    if not isinstance(text, str):
        return NormalizedExpr(
            canonical=None,
            diagnostics=(ExprDiagnostic(
                code="INVALID_TYPE",
                message=f"Expected string expression, got {type(text).__name__}.",
                original=str(text),
                column=1,
            ),),
        )
    if len(text) > max_chars:
        return NormalizedExpr(
            canonical=None,
            diagnostics=(ExprDiagnostic(
                code="EXPRESSION_TOO_LARGE",
                message=f"Expression length {len(text)} exceeds maximum cap ({max_chars} chars).",
                original=text[:120] + "...",
                column=1,
                hint="Keep formal expressions concise.",
            ),),
        )
    if not text.strip():
        return NormalizedExpr(
            canonical=None,
            diagnostics=(ExprDiagnostic(
                code="EMPTY_EXPRESSION",
                message="Expression is empty.",
                original=text,
                column=1,
                hint="Provide a non-empty formal expression.",
            ),),
        )

    # Check raw bracket/paren nesting depth before fast-path (ast.parse strips redundant parens).
    cur_nest = 0
    for idx_ch, ch in enumerate(text):
        if ch in "([":
            cur_nest += 1
            if cur_nest > max_depth:
                return NormalizedExpr(
                    canonical=None,
                    diagnostics=(ExprDiagnostic(
                        code="EXPRESSION_TOO_DEEP",
                        message=f"Expression nesting depth exceeds cap ({max_depth}).",
                        original=text[:120],
                        column=idx_ch + 1,
                    ),),
                )
        elif ch in ")]":
            cur_nest = max(0, cur_nest - 1)

    # Fast path: if meaningfulness._parse(text) already accepts it (and no unsafe power tower,
    # no ternary IfExp that should canonicalize to min/max/abs, and no variadic min/max > 2 args),
    # return BYTE-IDENTICAL with zero transforms.
    _needs_canonical_lowering = (" if " in text and " else " in text)
    if not _needs_canonical_lowering and ("min(" in text or "max(" in text):
        try:
            _raw_tree = ast.parse(text, mode="eval")
            for _sub in ast.walk(_raw_tree):
                if (
                    isinstance(_sub, ast.Call)
                    and isinstance(_sub.func, ast.Name)
                    and _sub.func.id in ("min", "max")
                    and len(_sub.args) > 2
                ):
                    _needs_canonical_lowering = True
                    break
        except SyntaxError:
            pass
    try:
        fast_node = None if _needs_canonical_lowering else m._parse(text)
    except (ValueError, SyntaxError):
        fast_node = None

    if fast_node is not None:
        if _has_unsafe_power_tower(fast_node):
            return NormalizedExpr(
                canonical=None,
                diagnostics=(ExprDiagnostic(
                    code="UNSAFE_POWER_TOWER",
                    message="Constant exponentiation exceeds safe finite numeric bounds.",
                    original=text,
                    column=1,
                    hint="Avoid nested constant power towers or huge exponents.",
                ),),
            )
        return NormalizedExpr(canonical=text, transforms=(), diagnostics=())

    deadline = time.monotonic() + max(0.01, float(time_budget_s))
    declared = {str(s) for s in declared_symbols if isinstance(s, str)}

    cleaned, base_offset, transforms = _strip_surface_wrappers(text)
    if " if " in cleaned and " else " in cleaned:
        try:
            _py_tree = ast.parse(cleaned, mode="eval")
            _ifexp_changed = False

            class _IfExpLowerer(ast.NodeTransformer):
                def visit_IfExp(self, node: ast.IfExp):
                    nonlocal _ifexp_changed
                    self.generic_visit(node)
                    if isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 and len(node.test.comparators) == 1:
                        lhs_s = ast.unparse(node.test.left)
                        rhs_s = ast.unparse(node.test.comparators[0])
                        body_s = ast.unparse(node.body)
                        else_s = ast.unparse(node.orelse)
                        op = node.test.ops[0]
                        # a if a > b else b  OR  b if b > a else a -> max(a, b)
                        if (
                            (isinstance(op, (ast.Gt, ast.GtE)) and lhs_s == body_s and rhs_s == else_s)
                            or (isinstance(op, (ast.Lt, ast.LtE)) and rhs_s == body_s and lhs_s == else_s)
                        ):
                            _ifexp_changed = True
                            return ast.Call(func=ast.Name(id="max", ctx=ast.Load()), args=[node.body, node.orelse], keywords=[])
                        # a if a < b else b  OR  b if b < a else a -> min(a, b)
                        if (
                            (isinstance(op, (ast.Lt, ast.LtE)) and lhs_s == body_s and rhs_s == else_s)
                            or (isinstance(op, (ast.Gt, ast.GtE)) and rhs_s == body_s and lhs_s == else_s)
                        ):
                            _ifexp_changed = True
                            return ast.Call(func=ast.Name(id="min", ctx=ast.Load()), args=[node.body, node.orelse], keywords=[])
                        # x if x >= 0 else -x -> abs(x)
                        if (
                            isinstance(op, (ast.Gt, ast.GtE))
                            and lhs_s == body_s
                            and rhs_s in {"0", "0.0"}
                            and isinstance(node.orelse, ast.UnaryOp)
                            and isinstance(node.orelse.op, ast.USub)
                            and ast.unparse(node.orelse.operand) == lhs_s
                        ):
                            _ifexp_changed = True
                            return ast.Call(func=ast.Name(id="abs", ctx=ast.Load()), args=[node.body], keywords=[])
                    return node

            _lowered_tree = _IfExpLowerer().visit(_py_tree)
            if _ifexp_changed:
                cleaned = ast.unparse(ast.fix_missing_locations(_lowered_tree))
                transforms.append("ifexp_lower")
        except SyntaxError:
            pass
    if not cleaned:
        return NormalizedExpr(
            canonical=None,
            transforms=tuple(transforms),
            diagnostics=(ExprDiagnostic(
                code="EMPTY_EXPRESSION",
                message="Expression is empty after stripping wrappers/comments.",
                original=text,
                column=1,
            ),),
        )

    # Re-check fast path after wrapper stripping (preserves inner expression cleanly)
    try:
        stripped_node = None if _needs_canonical_lowering else m._parse(cleaned)
    except (ValueError, SyntaxError):
        stripped_node = None
    if stripped_node is not None:
        if _has_unsafe_power_tower(stripped_node):
            return NormalizedExpr(
                canonical=None,
                transforms=tuple(transforms),
                diagnostics=(ExprDiagnostic(
                    code="UNSAFE_POWER_TOWER",
                    message="Constant exponentiation exceeds safe finite numeric bounds.",
                    original=text,
                    column=1,
                ),),
            )
        return NormalizedExpr(canonical=cleaned, transforms=tuple(transforms), diagnostics=())

    tokens, lex_transforms, lex_err = _lex(
        cleaned, text, base_offset, max_tokens=max_tokens, deadline=deadline
    )
    transforms.extend(lex_transforms)
    if lex_err is not None:
        return NormalizedExpr(
            canonical=None,
            transforms=tuple(transforms),
            diagnostics=(lex_err,),
        )

    stripped_units: list[tuple[str, str]] = []
    tokens = _rewrite_tokens(tokens, declared, transforms, stripped_units)

    parser = _Parser(
        tokens,
        text,
        declared,
        domain,
        transforms,
        stripped_units,
        max_depth=max_depth,
        deadline=deadline,
    )
    try:
        tree = parser.parse()
    except _ParseError as exc:
        return NormalizedExpr(
            canonical=None,
            transforms=tuple(dict.fromkeys(transforms)),
            diagnostics=(ExprDiagnostic(
                code=exc.code,
                message=str(exc),
                original=text,
                column=exc.tok.col,
                offending_token=exc.tok.text or None,
                hint=exc.hint or (
                    "Use formal Python-subset expressions over declared symbols; "
                    "move prose to notes[] and quantifiers/ranges to typed fields."
                ),
            ),),
            quantifier=parser.quantifier,
            binders=tuple(parser.binders),
            binder_assumptions=tuple(parser.binder_assumptions),
            stripped_units=tuple(stripped_units),
        )
    except Exception as exc:  # pragma: no cover - defensive never-raise guard
        return NormalizedExpr(
            canonical=None,
            transforms=tuple(dict.fromkeys(transforms)),
            diagnostics=(ExprDiagnostic(
                code="INTERNAL_PARSE_ERROR",
                message=str(exc),
                original=text,
                column=1,
            ),),
        )

    # Record quantifier_mode disagreement (never override quantifier_mode)
    if parser.quantifier and quantifier_mode:
        qm = str(quantifier_mode).upper()
        if (parser.quantifier == "FORALL" and qm == "EXISTS") or (parser.quantifier == "EXISTS" and qm == "FORALL"):
            transforms.append(f"quantifier_mode_mismatch:{parser.quantifier}_vs_{qm}")

    if parser.binder_diagnostics:
        return NormalizedExpr(
            canonical=None,
            transforms=tuple(dict.fromkeys(transforms)),
            diagnostics=tuple(parser.binder_diagnostics),
            quantifier=parser.quantifier,
            binders=tuple(parser.binders),
            binder_assumptions=tuple(parser.binder_assumptions),
            stripped_units=tuple(stripped_units),
        )

    if _has_unsafe_power_tower(tree):
        return NormalizedExpr(
            canonical=None,
            transforms=tuple(dict.fromkeys(transforms)),
            diagnostics=(ExprDiagnostic(
                code="UNSAFE_POWER_TOWER",
                message="Constant exponentiation exceeds safe finite numeric bounds.",
                original=text,
                column=1,
            ),),
        )

    canonical = ast.unparse(ast.fix_missing_locations(tree))

    # Final strict gate: meaningfulness._parse
    try:
        m._parse(canonical)
    except (ValueError, SyntaxError) as exc:
        return NormalizedExpr(
            canonical=None,
            transforms=tuple(dict.fromkeys(transforms)),
            diagnostics=(ExprDiagnostic(
                code="STRICT_GATE_REJECTED",
                message=str(exc),
                original=text,
                column=1,
                hint="Only supported scalar functions and finite arithmetic/boolean operators are allowed.",
            ),),
            quantifier=parser.quantifier,
            binders=tuple(parser.binders),
            binder_assumptions=tuple(parser.binder_assumptions),
            stripped_units=tuple(stripped_units),
        )

    return NormalizedExpr(
        canonical=canonical,
        transforms=tuple(dict.fromkeys(transforms)),
        diagnostics=(),
        quantifier=parser.quantifier,
        binders=tuple(parser.binders),
        binder_assumptions=tuple(parser.binder_assumptions),
        stripped_units=tuple(stripped_units),
    )

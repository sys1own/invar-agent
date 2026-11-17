"""Deterministic JSON control-escape repair and LaTeX/KaTeX canonicalizer.

Heals LLM control-character-corrupted LaTeX (e.g. ``\\x0crac`` -> ``\\frac``),
protects fenced/inline code spans, auto-wraps bare formulas in prose and
LaTeX/Equation table columns, canonicalizes ASCII-math hybrids inside ``$...$``,
and balances braces, ``\\left``/``\\right``, and math delimiters.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1. JSON control-character escape recovery
# ---------------------------------------------------------------------------

_CONTROL_ESCAPE_RULES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    # \f -> \x0c
    (re.compile(r"\x0crac\b"), r"\\frac"),
    (re.compile(r"\x0corall\b"), r"\\forall"),
    # \b -> \x08
    (re.compile(r"\x08eta\b"), r"\\beta"),
    (re.compile(r"\x08egin\b"), r"\\begin"),
    (re.compile(r"\x08ar\b"), r"\\bar"),
    (re.compile(r"\x08ig(?:g?[lr]?)\b"), lambda m: "\\" + "b" + m.group(0)[1:]),  # type: ignore[arg-type]
    (re.compile(r"\x08inom\b"), r"\\binom"),
    (re.compile(r"\x08ullet\b"), r"\\bullet"),
    # \r -> \x0d (only when followed by LaTeX command suffixes, not \r\n)
    (re.compile(r"\x0dho\b"), r"\\rho"),
    (re.compile(r"\x0dight\b"), r"\\right"),
    (re.compile(r"\x0dathrm\b"), r"\\mathrm"),
    (re.compile(r"\x0dathbb\b"), r"\\mathbb"),
    (re.compile(r"\x0dathcal\b"), r"\\mathcal"),
    (re.compile(r"\x0de\b"), r"\\mathrm{Re}"),
    # \t -> \x09
    (re.compile(r"\x09heta\b"), r"\\theta"),
    (re.compile(r"\x09au\b"), r"\\tau"),
    (re.compile(r"\x09ext\b"), r"\\text"),
    (re.compile(r"\x09imes\b"), r"\\times"),
    (re.compile(r"\x09ilde\b"), r"\\tilde"),
    (re.compile(r"\x09o\b"), r"\\to"),
    (re.compile(r"\x09op\b"), r"\\top"),
    (re.compile(r"\x09anh\b"), r"\\tanh"),
    (re.compile(r"\x09race\b"), r"\\mathrm{Tr}"),
    # \n -> \x0a (only when immediately followed by LaTeX suffixes)
    (re.compile(r"\x0aabla\b"), r"\\nabla"),
    (re.compile(r"\x0aeq\b"), r"\\neq"),
    (re.compile(r"\x0au(?![a-z])"), r"\\nu"),
)


def repair_json_control_escapes(text: str) -> str:
    """Restore LaTeX commands corrupted into ASCII control chars by ``json.loads``."""
    if not text:
        return text
    out = text
    for pat, repl in _CONTROL_ESCAPE_RULES:
        out = pat.sub(repl, out)
    return out


def repair_json_strings(obj: Any) -> Any:
    """Recursively apply ``repair_json_control_escapes`` to all strings in JSON data."""
    if isinstance(obj, str):
        return repair_json_control_escapes(obj)
    if isinstance(obj, list):
        return [repair_json_strings(item) for item in obj]
    if isinstance(obj, dict):
        return {k: repair_json_strings(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# 2. Code-span protection
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _protect_code_spans(text: str) -> Tuple[str, Dict[str, str]]:
    tokens: Dict[str, str] = {}
    idx = 0

    def _stash(match: re.Match[str]) -> str:
        nonlocal idx
        key = f"@@INVAR_CODE_{idx}@@"
        idx += 1
        tokens[key] = match.group(0)
        return key

    masked = _FENCE_RE.sub(_stash, text)
    masked = _INLINE_CODE_RE.sub(_stash, masked)
    return masked, tokens


def _restore_tokens(text: str, tokens: Dict[str, str]) -> str:
    for key, val in tokens.items():
        text = text.replace(key, val)
    return text


# ---------------------------------------------------------------------------
# 3 & 4. Math canonicalization & bare formula wrapping
# ---------------------------------------------------------------------------

_SCI_NOTATION_RE = re.compile(r"(?<![A-Za-z0-9_\\])(\d+(?:\.\d+)?)[eE]([+-]?\d+)\b")

_GREEK_IDENT_RULES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<!\\)\brho_ss\b"), r"\\rho_{\\mathrm{ss}}"),
    (re.compile(r"(?<!\\)\blambda_min\b"), r"\\lambda_{\\min}"),
    (re.compile(r"(?<!\\)\bomega_0\b"), r"\\omega_0"),
    (re.compile(r"(?<!\\)\bomega0\b"), r"\\omega_0"),
    (re.compile(r"(?<!\\)\bDelta\s+t\b"), r"\\Delta t"),
    (re.compile(r"(?<!\\)\brho\b"), r"\\rho"),
    (re.compile(r"(?<!\\)\bkappa\b"), r"\\kappa"),
    (re.compile(r"(?<!\\)\bgamma\b"), r"\\gamma"),
    (re.compile(r"(?<!\\)\bOmega\b"), r"\\Omega"),
    (re.compile(r"(?<!\\)\balpha\b"), r"\\alpha"),
    (re.compile(r"(?<!\\)\bbeta\b"), r"\\beta"),
    (re.compile(r"(?<!\\)\bPsi\b"), r"\\Psi"),
    (re.compile(r"(?<!\\)\bPhi\b"), r"\\Phi"),
    (re.compile(r"(?<!\\)\bzeta\b"), r"\\zeta"),
)


def _format_sci(match: re.Match[str]) -> str:
    mant, exp = match.group(1), match.group(2)
    exp_i = int(exp)
    if mant in ("1", "1.0"):
        return f"10^{{{exp_i}}}"
    return f"{mant} \\times 10^{{{exp_i}}}"


def canonicalize_math_content(expr: str) -> str:
    """Normalize ASCII-math operators, scientific notation, and bare Greek names."""
    s = expr
    s = s.replace("||", r"\|")
    s = re.sub(r"<=", r" \\le ", s)
    s = re.sub(r">=", r" \\ge ", s)
    s = re.sub(r"!=", r" \\ne ", s)
    s = re.sub(r"->", r" \\to ", s)
    s = re.sub(r"<<", r" \\ll ", s)
    s = re.sub(r">>", r" \\gg ", s)
    s = _SCI_NOTATION_RE.sub(_format_sci, s)
    for pat, repl in _GREEK_IDENT_RULES:
        s = pat.sub(repl, s)
    # Collapse excess spaces introduced around operators
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _balance_math_expr(expr: str) -> str:
    """Balance ``{``/``}`` and ``\\left``/``\\right`` inside a single math segment."""
    s = expr.strip()
    if not s:
        return s
    # Balance braces (ignoring escaped \{ and \})
    depth = 0
    out_chars: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s) and s[i + 1] in ("{", "}"):
            out_chars.append(s[i:i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
            out_chars.append(ch)
        elif ch == "}":
            if depth > 0:
                depth -= 1
                out_chars.append(ch)
            else:
                out_chars.append(r"\}")
        else:
            out_chars.append(ch)
        i += 1
    if depth > 0:
        out_chars.append("}" * depth)
    s = "".join(out_chars)

    # Balance \left and \right
    n_left = len(re.findall(r"\\left\b", s))
    n_right = len(re.findall(r"\\right\b", s))
    if n_left > n_right:
        s += r" \right." * (n_left - n_right)
    elif n_right > n_left:
        s = (r"\left. " * (n_right - n_left)) + s
    return s


# Bare equation / formula detector for prose (outside $...$ and code tokens)
_BARE_FORMULA_RE = re.compile(
    r"(?<!\$)"
    r"(?:"
    r"(?:\\(?:frac|sum|int|sqrt|alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|phi|chi|psi|omega|Delta|Gamma|Lambda|Omega|Phi|Psi|Sigma|nabla|partial|left)\b"
    r"|[A-Za-z_][A-Za-z0-9_]*_\{[^}\n]+\}"
    r"|\|\|[^|\n]+\|\|)"
    r"(?:[^\n$]*?(?:<=|>=|!=|==|[=<>])[^\n$,;.]*)?"
    r"|"
    r"[A-Za-z][A-Za-z0-9_]*\([^)\n]+\)\s*(?:<=|>=|!=|=|<|>)\s*[^\n$,;.]+"
    r")"
)


_CODE_EXPRESSION_RE = re.compile(
    r"\b(?:state|initial|trace_min|trace_max|telemetry|buf\.data)\[|\{['\"]"
)


def _wrap_table_equation_cells(text: str) -> str:
    """Wrap bare math cells under LaTeX/Equation columns in markdown tables."""
    lines = text.splitlines()
    out: List[str] = []
    eq_cols: List[int] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not in_table:
                eq_cols = [
                    idx for idx, h in enumerate(cells)
                    if any(tok in h.lower() for tok in ("latex", "equation", "formula"))
                ]
                in_table = True
                out.append(line)
                continue
            if all(set(c) <= {"-", ":"} for c in cells if c):
                out.append(line)
                continue
            if eq_cols:
                for idx in eq_cols:
                    if idx < len(cells):
                        c = cells[idx]
                        if (c and not c.startswith("$") and not c.startswith("@@INVAR_CODE_")
                                and c not in ("-", "n/a", "—")
                                and not _CODE_EXPRESSION_RE.search(c)
                                and re.search(r"[=\\^_<>]", c)):
                            cells[idx] = f"${_balance_math_expr(canonicalize_math_content(c))}$"
                line = "| " + " | ".join(cells) + " |"
        else:
            in_table = False
            eq_cols = []
        out.append(line)
    return "\n".join(out)


def _wrap_bare_math_in_prose(text: str) -> str:
    """Wrap bare math expressions outside existing ``$...$`` / ``$$...$$``."""
    # Split by existing display/inline math so we never double-wrap
    math_split_re = re.compile(r"(\$\$[\s\S]*?\$\$|\$(?:\\.|[^$\n])+?\$)")
    parts = math_split_re.split(text)
    for i in range(0, len(parts), 2):
        seg = parts[i]
        if not seg:
            continue

        def _wrap_match(m: re.Match[str]) -> str:
            raw = m.group(0).strip()
            if "@@INVAR_CODE_" in raw or not raw or _CODE_EXPRESSION_RE.search(raw):
                return m.group(0)
            return f"${_balance_math_expr(canonicalize_math_content(raw))}$"

        parts[i] = _BARE_FORMULA_RE.sub(_wrap_match, seg)
    return "".join(parts)


def balance_math_delimiters_and_braces(text: str) -> str:
    """Canonicalize + balance every ``$$...$$`` and ``$...$`` block, and pair stray ``$``."""
    # Canonicalize display math $$...$$
    def _fix_display(m: re.Match[str]) -> str:
        inner = _balance_math_expr(canonicalize_math_content(m.group(1)))
        return f"$${inner}$$"

    text = re.sub(r"\$\$([\s\S]*?)\$\$", _fix_display, text)

    # Canonicalize inline math $...$ (avoid matching $$); keep code assertions in backticks
    def _fix_inline(m: re.Match[str]) -> str:
        raw_inner = m.group(1)
        if _CODE_EXPRESSION_RE.search(raw_inner):
            return f"`{raw_inner.strip()}`"
        inner = _balance_math_expr(canonicalize_math_content(raw_inner))
        return f"${inner}$"

    text = re.sub(r"(?<!\$)\$(?!\$)((?:\\.|[^$\n])+?)(?<!\$)\$(?!\$)", _fix_inline, text)

    # Escape any remaining unpaired single `$` per line so KaTeX never breaks
    fixed_lines: List[str] = []
    for line in text.splitlines():
        # Count unescaped single $ outside $$...$$
        tmp = re.sub(r"\$\$[\s\S]*?\$\$", "", line)
        singles = [m.start() for m in re.finditer(r"(?<!\\)(?<!\$)\$(?!\$)", tmp)]
        if len(singles) % 2 == 1:
            # Escape the last stray $
            pos = line.rfind("$")
            if pos >= 0 and (pos == 0 or line[pos - 1] != "\\"):
                line = line[:pos] + r"\$" + line[pos + 1:]
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def normalize_report_latex(text: str) -> str:
    """Run all 5 deterministic LaTeX repair & canonicalization passes on ``text``."""
    if not text:
        return text
    out = repair_json_control_escapes(text)
    out, tokens = _protect_code_spans(out)
    out = _wrap_table_equation_cells(out)
    out = _wrap_bare_math_in_prose(out)
    out = balance_math_delimiters_and_braces(out)
    out = _restore_tokens(out, tokens)
    return out


def format_contract_constraint_math(assertion: str, var_map: Dict[str, int] | None = None) -> str:
    """Format a contract assertion (e.g. ``state[6] <= 1e-14``) as clean LaTeX."""
    inv_map = {idx: name for name, idx in (var_map or {}).items() if isinstance(idx, int)}
    expr = str(assertion or "").strip()

    def _sub_slot(m: re.Match[str]) -> str:
        prefix = m.group(1)
        idx = int(m.group(2))
        sym = inv_map.get(idx)
        if sym:
            safe_sym = sym.replace("_", r"\_")
            return f"\\mathrm{{{safe_sym}}}" if prefix == "state" else f"\\mathrm{{{safe_sym}}}_0"
        return f"s_{{{idx}}}" if prefix == "state" else f"s_{{{idx},0}}"

    expr = re.sub(r"\b(state|initial|buf\.data|data)\[(\d+)\]", _sub_slot, expr)
    expr = re.sub(r"\bmath\.isfinite\(([^)]+)\)", r"\\mathrm{finite}(\1)", expr)
    expr = re.sub(r"\band\b", r"\\land", expr)
    expr = _balance_math_expr(canonicalize_math_content(expr))
    return f"${expr}$"

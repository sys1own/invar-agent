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
_URL_RE = re.compile(r"https?://[^\s)>\]]+")


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
    masked = _URL_RE.sub(_stash, masked)
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
    (re.compile(r"(?<!\\)\bTheta\b"), r"\\Theta"),
    (re.compile(r"(?<!\\)\bGamma\b"), r"\\Gamma"),
    (re.compile(r"(?<!\\)\bSigma\b"), r"\\Sigma"),
    (re.compile(r"(?<!\\)\bLambda\b"), r"\\Lambda"),
    (re.compile(r"(?<!\\)\bzeta\b"), r"\\zeta"),
)


_GREEK_BASES = frozenset({
    "delta", "kappa", "alpha", "beta", "gamma", "eta", "theta", "sigma",
    "phi", "psi", "rho", "omega", "epsilon", "mu", "nu", "tau", "lambda",
    "zeta", "chi", "xi",
    "Delta", "Gamma", "Omega", "Phi", "Psi", "Sigma", "Lambda", "Theta",
})
_ACRONYM_BASES = frozenset({
    "SNR", "TOF", "BEP", "MHD", "LHHW", "HER", "SEI", "CCZ4", "PEGS", "TMSV", "SQL", "SHBT",
})
_MULTI_SUB_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_\\])(\\[A-Za-z]+|[A-Za-z][A-Za-z0-9]*)(?:_([A-Za-z0-9]+)){2,}\b"
)


def _format_sub_segment(seg: str) -> str:
    s = seg.strip()
    if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        s = s[1:-1].strip()
    if s.startswith(r"\mathrm{") and s.endswith("}"):
        return s
    if len(s) >= 2 and not s.isdigit():
        return f"\\mathrm{{{s}}}"
    return s


def _read_script_arg(s: str, idx: int) -> Tuple[Optional[str], int]:
    """Read `{...}` (balanced) or single alnum char starting at ``idx``."""
    n = len(s)
    if idx >= n:
        return None, idx
    if s[idx] == "{":
        depth = 0
        j = idx
        while j < n:
            ch = s[j]
            if ch == "\\" and j + 1 < n and s[j + 1] in ("{", "}"):
                j += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[idx + 1:j], j + 1
            j += 1
        return s[idx + 1:], n
    if s[idx].isalnum():
        return s[idx], idx + 1
    return None, idx


def _merge_consecutive_scripts(expr: str, marker: str = "_") -> str:
    """Merge consecutive subscripts (`_a_b`, `_{a}_{b}`) or superscripts into one."""
    out = expr
    for _ in range(4):
        buf: List[str] = []
        i = 0
        n = len(out)
        merged_any = False
        while i < n:
            ch = out[i]
            if ch == marker and (i == 0 or out[i - 1] != "\\"):
                arg1, next_i = _read_script_arg(out, i + 1)
                if arg1 is not None:
                    args = [arg1]
                    j = next_i
                    while True:
                        k = j
                        while k < n and out[k] == " ":
                            k += 1
                        if k < n and out[k] == marker:
                            arg_next, next_k = _read_script_arg(out, k + 1)
                            if arg_next is not None:
                                args.append(arg_next)
                                j = next_k
                                merged_any = True
                                continue
                        break
                    if len(args) >= 2:
                        if marker == "_":
                            fmt_parts: List[str] = []
                            for a in args:
                                fmt_parts.append(_format_sub_segment(a))
                            buf.append(f"_{{{','.join(fmt_parts)}}}")
                        else:
                            buf.append(f"^{{{' '.join(a.strip() for a in args)}}}")
                        i = j
                        continue
            buf.append(ch)
            i += 1
        out = "".join(buf)
        if not merged_any:
            break
    return out


def _replace_multi_sub_token(match: re.Match[str]) -> str:
    full = match.group(0)
    parts = full.split("_")
    base = parts[0]
    segs = parts[1:]
    if base.startswith("\\"):
        base_tex = base
    elif base in _GREEK_BASES:
        base_tex = f"\\{base}"
    elif base in _ACRONYM_BASES:
        base_tex = f"\\mathrm{{{base}}}"
    else:
        base_tex = base
    fmt = [_format_sub_segment(seg) for seg in segs]
    return f"{base_tex}_{{{','.join(fmt)}}}"


def _format_sci(match: re.Match[str]) -> str:
    mant, exp = match.group(1), match.group(2)
    exp_i = int(exp)
    if mant in ("1", "1.0"):
        return f"10^{{{exp_i}}}"
    return f"{mant} \\times 10^{{{exp_i}}}"


def canonicalize_math_content(expr: str) -> str:
    """Normalize ASCII-math operators, scientific notation, subscripts, and Greek names."""
    s = expr
    s = s.replace("||", r"\|")
    s = re.sub(r"<=", r" \\le ", s)
    s = re.sub(r">=", r" \\ge ", s)
    s = re.sub(r"!=", r" \\ne ", s)
    s = re.sub(r"->", r" \\to ", s)
    s = re.sub(r"<<", r" \\ll ", s)
    s = re.sub(r">>", r" \\gg ", s)
    s = _SCI_NOTATION_RE.sub(_format_sci, s)
    s = re.sub(r"(?<!\\)(?<!\*)\*(?!\*)", r" \\cdot ", s)
    s = re.sub(r"(?<!\\)\b(exp|ln|log)\s*\(", r"\\\1(", s)
    for pat, repl in _GREEK_IDENT_RULES:
        s = pat.sub(repl, s)
    # Handle full multi-underscore tokens first (delta_g_PEGS -> \delta_{g,\mathrm{PEGS}})
    s = _MULTI_SUB_TOKEN_RE.sub(_replace_multi_sub_token, s)
    # Handle single-underscore Greek prefix (delta_SEI -> \delta_SEI)
    s = re.sub(
        r"(?<!\\)\b(delta|kappa|alpha|beta|gamma|eta|theta|sigma|phi|psi|rho|omega|epsilon|mu|nu|tau|lambda|zeta|chi|xi|Delta|Gamma|Omega|Phi|Psi|Sigma|Lambda|Theta)_",
        r"\\\1_",
        s,
    )
    # Acronym single-underscore bases (SNR_SQL -> \mathrm{SNR}_SQL)
    s = re.sub(
        r"(?<![A-Za-z0-9_\\])(SNR|TOF|BEP|MHD|LHHW|HER|SEI|PEGS|TMSV|SQL|SHBT)_",
        r"\\mathrm{\1}_",
        s,
    )
    # Brace multi-letter unbraced subscripts even when another `_...` follows (M_dot_{\mathrm{moment}} -> M_{\mathrm{dot}}_{\mathrm{moment}})
    s = re.sub(r"(?<!\\)_([A-Za-z][A-Za-z0-9]{1,})(?=_|[^A-Za-z0-9]|$)", r"_{\\mathrm{\1}}", s)
    # Convert infinity subscripts `_inf` / `_{\mathrm{inf}}` -> `_{\infty}`
    s = re.sub(r"_(?:\{\s*\\mathrm\{inf\}\s*\}|inf\b)", r"_{\\infty}", s)
    # Convert bare `Delta ` / `Theta ` etc. before math symbols
    s = re.sub(r"(?<!\\)\b(Delta|Gamma|Omega|Phi|Psi|Sigma|Lambda|Theta)\s+(?=[A-Za-z\\])", r"\\\1 ", s)
    # Guarantee zero double subscripts or double superscripts remain (e.g. \delta_g_{\mathrm{PEGS}} -> \delta_{g,\mathrm{PEGS}})
    s = _merge_consecutive_scripts(s, "_")
    s = _merge_consecutive_scripts(s, "^")
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
_CODE_EXPRESSION_RE = re.compile(
    r"\b(?:state|initial|trace_min|trace_max|telemetry|buf\.data)\[|\{['\"]"
)
_TELEMETRY_PREFIX_RE = re.compile(r"^(?:final_|obs_|execution_time)")
_MATH_FUNCS = frozenset(
    {"exp", "ln", "log", "log10", "log2", "sqrt", "abs", "fabs", "min", "max",
     "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
     "asinh", "acosh", "atanh", "erf", "cbrt", "pow"}
)
_MATH_WORD_SYMS = _GREEK_BASES | _ACRONYM_BASES | _MATH_FUNCS | frozenset({"pi", "inf", "nan", "eff", "kin"})
_MATH_IDENT_START_RE = re.compile(
    r"(?:"
    r"\\(?:frac|sum|int|sqrt|mathrm|mathbf|mathbb|mathit|text|left|right|exp|ln|log|"
    r"delta|kappa|alpha|beta|gamma|eta|theta|sigma|phi|psi|rho|omega|epsilon|mu|nu|tau|lambda|zeta|chi|xi|"
    r"Delta|Gamma|Omega|Phi|Psi|Sigma|Lambda|Theta|nabla|partial|le|ge|ne|approx|times|cdot|to|ll|gg)(?=_|[^A-Za-z]|$)"
    r"|(?<![A-Za-z0-9_\\])(?:[A-Za-z][A-Za-z0-9_]*(?:_\{|\^\{|\^[A-Za-z0-9]))"
    r"|(?<![A-Za-z0-9_\\])(?:delta|kappa|alpha|beta|gamma|eta|theta|sigma|phi|psi|rho|omega|epsilon|mu|nu|tau|lambda|zeta|chi|xi|"
    r"Delta|Gamma|Omega|Phi|Psi|Sigma|Lambda|Theta|SNR|TOF|BEP|MHD|LHHW|HER|SEI|PEGS|TMSV|SQL|SHBT|[A-Za-z]{1,2})_[A-Za-z0-9_]+"
    r"|\|\|[^|\n]+\|\|(?:_[A-Za-z0-9_{}\\]+)?(?:\^[A-Za-z0-9_{}\\]+)?"
    r"|(?<![A-Za-z0-9_\\|])\|(?![\s|])[A-Za-z0-9_\\^{}()/*+\-.\s]+(?<!\s)\|(?:_[A-Za-z0-9_{}\\]+)?(?:\^[A-Za-z0-9_{}\\]+)?(?=\s*(?:<=|>=|!=|==|[=<>]|\\le|\\ge|\\ne|\\approx))"
    r"|(?<![A-Za-z0-9_\\])[A-Za-z][A-Za-z0-9_]*\([^)\n]+\)\s*(?:<=|>=|!=|==|[=<>]|\\le|\\ge|\\ne|\\approx)"
    r"|(?<![A-Za-z0-9_\\])[A-Za-z][A-Za-z0-9_]*\s*\*\s*\d+(?:\.\d+)?[eE][+-]?\d+"
    r")"
)
_BARE_LATEX_LEFTOVER_RE = re.compile(
    r"(?:[^\s$]*(?:\\(?:mathrm|mathbf|mathbb|mathit|text|frac|sqrt|delta|sigma|eta|alpha|beta|gamma|omega|theta|kappa|mu|nu|tau|lambda|zeta|Phi|Psi|Delta|Gamma|Omega|Sigma|ge|le|ne|approx|times|cdot|to|exp|ln)(?=_|[^A-Za-z]|$)|_\{|\^\{)[^\s$]*)"
    r"(?:\s*(?:<=|>=|!=|==|[=<>+\-*/]|\\le|\\ge|\\ne|\\approx|\\times|\\cdot)\s*"
    r"(?:[0-9.eE+-]+|\\[A-Za-z]+(?:\{[^{}\n]*\})*|[A-Za-z0-9_^{}()\\]+))*"
)


def _wrap_table_equation_cells(text: str, tokens: Dict[str, str] | None = None) -> str:
    """Wrap bare math cells under LaTeX/Equation columns in markdown tables."""
    lines = text.splitlines()
    out: List[str] = []
    eq_cols: List[int] = []
    in_table = False
    tok_map = tokens or {}
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
                        if c.startswith("@@INVAR_CODE_") and c in tok_map:
                            orig = tok_map[c].strip()
                            if orig.startswith("`") and orig.endswith("`") and not orig.startswith("```"):
                                inner_c = orig[1:-1].strip()
                                if (inner_c and inner_c not in ("-", "n/a", "—")
                                        and not _CODE_EXPRESSION_RE.search(inner_c)
                                        and re.search(r"[=\\^_<>]", inner_c)):
                                    cells[idx] = f"${_balance_math_expr(canonicalize_math_content(inner_c))}$"
                                    continue
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


def _scan_math_run_end(seg: str, start: int, *, in_table_row: bool = False) -> int:
    """Scan greedily from ``start`` across a bare mathematical formula in prose."""
    n = len(seg)
    i = start
    p_depth = 0
    b_depth = 0
    in_pipe = False
    expect_rhs = False
    while i < n:
        ch = seg[i]
        if ch in "\r\n;`":
            break
        if ch == "|":
            is_norm_pipe = seg.startswith("||", i) or (i > 0 and seg[i - 1] == "|")
            if in_table_row and not is_norm_pipe:
                break
            if not is_norm_pipe:
                in_pipe = not in_pipe
            i += 1
            expect_rhs = False
            continue
        if seg.startswith("@@INVAR_", i) or seg.startswith("**", i):
            break
        if ch in "{(":
            if ch == "(":
                p_depth += 1
            else:
                b_depth += 1
            expect_rhs = False
            i += 1
            continue
        if ch in "})":
            if ch == ")" and p_depth == 0:
                break
            if ch == "}" and b_depth == 0:
                break
            if ch == ")":
                p_depth -= 1
            else:
                b_depth -= 1
            expect_rhs = False
            i += 1
            continue
        if ch == ",":
            if p_depth == 0 and b_depth == 0:
                break
            i += 1
            continue
        if ch == "." and p_depth == 0 and b_depth == 0:
            # Decimal inside number (2.50 or .5) vs sentence period
            prev_ok = i > 0 and seg[i - 1].isdigit()
            next_ok = i + 1 < n and seg[i + 1].isdigit()
            if not (prev_ok and next_ok):
                break
            i += 1
            continue
        if ch in ";:":
            if p_depth == 0 and b_depth == 0:
                break
            i += 1
            continue
        if ch in " \t":
            # Look ahead to decide whether formula continues
            j = i
            while j < n and seg[j] in " \t":
                j += 1
            if j >= n or seg[j] in "\r\n|;:`$," or seg.startswith("@@INVAR_", j) or seg.startswith("**", j):
                break
            nxt = seg[j:]
            # Sentence period?
            if nxt[0] == "." and not (len(nxt) > 1 and nxt[1].isdigit()):
                break
            if nxt[0] == ")" and p_depth == 0:
                break
            # Operator continuation always continues
            if re.match(r"^(?:<=|>=|!=|==|[=<>+\-*/^_]|\\(?:le|ge|ne|approx|times|cdot|to|ll|gg)\b)", nxt):
                i = j
                continue
            # If preceding token was an operator (`=`, `<`, `*`, `\le`, etc.), accept numbers, LaTeX cmds, or math idents
            if expect_rhs:
                if re.match(r"^(?:[0-9.\-+(]|\\[A-Za-z]+|[A-Za-z0-9]+_[A-Za-z0-9_{}]+|[A-Za-z](?![A-Za-z]))", nxt):
                    i = j
                    continue
                m_w = re.match(r"^([A-Za-z]{2,})\b", nxt)
                if m_w and (m_w.group(1) in _MATH_WORD_SYMS or (j + len(m_w.group(1)) < n and seg[j + len(m_w.group(1))] in "_^(")):
                    i = j
                    continue
                break
            # After a closing brace/paren/number, only continue on operator or math symbol (`f = ...` or `\cdot ...`)
            if p_depth > 0 or b_depth > 0 or in_pipe:
                i = j
                continue
            if re.match(r"^\\(?:le|ge|ne|approx|times|cdot|to|ll|gg)\b", nxt):
                i = j
                continue
            break

        if ch == "\\":
            m_cmd = re.match(r"^\\[A-Za-z]+", seg[i:])
            if m_cmd:
                cmd = m_cmd.group(0)
                expect_rhs = cmd in (r"\le", r"\ge", r"\ne", r"\approx", r"\times", r"\cdot", r"\to", r"\ll", r"\gg")
                i += len(cmd)
                continue
            i += 1
            continue

        m_op2 = re.match(r"^(?:<=|>=|!=|==)", seg[i:])
        if m_op2:
            expect_rhs = True
            i += 2
            continue
        if ch in "=<>+-*/^_":
            expect_rhs = ch in "=<>+-*/"
            i += 1
            continue
        m_num = re.match(r"^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", seg[i:])
        if m_num:
            expect_rhs = False
            i += len(m_num.group(0))
            continue
        m_id = re.match(r"^[A-Za-z][A-Za-z0-9_]*(?:_[A-Za-z0-9]+)*", seg[i:])
        if m_id:
            tok = m_id.group(0)
            after_pos = i + len(tok)
            has_math_follow = after_pos < n and seg[after_pos] in "(^_{*"
            if (
                i > start
                and "_" not in tok
                and len(tok) >= 2
                and not expect_rhs
                and p_depth == 0
                and b_depth == 0
                and not has_math_follow
                and tok not in _MATH_WORD_SYMS
            ):
                break
            expect_rhs = False
            i = after_pos
            continue
        break

    # Trim trailing whitespace and trailing unbalanced/dangling binary operators
    raw_end = i
    while raw_end > start and seg[raw_end - 1] in " \t+-*/=<>,:":
        raw_end -= 1
    return raw_end


def _wrap_bare_math_in_prose(text: str) -> str:
    """Wrap bare math expressions outside existing ``$...$`` / ``$$...$$``."""
    math_split_re = re.compile(r"(\$\$[\s\S]*?\$\$|\$(?:\\.|[^$\n])+?\$)")
    parts = math_split_re.split(text)
    for i in range(0, len(parts), 2):
        seg = parts[i]
        if not seg:
            continue
        out_chunks: List[str] = []
        pos = 0
        n = len(seg)
        while pos < n:
            m = _MATH_IDENT_START_RE.search(seg, pos)
            if not m:
                out_chunks.append(seg[pos:])
                break
            st = m.start()
            # Do not wrap markdown bold tokens or inside @@INVAR_ tokens
            before_ctx = seg[max(0, st - 15):st]
            if "@@INVAR_" in before_ctx and "@@" not in before_ctx.split("@@INVAR_")[-1]:
                out_chunks.append(seg[pos:m.end()])
                pos = m.end()
                continue
            line_start = seg.rfind("\n", 0, st) + 1
            in_table_row = seg[line_start:st].lstrip().startswith("|")
            end = _scan_math_run_end(seg, st, in_table_row=in_table_row)
            if end <= st:
                end = m.end()
            raw = seg[st:end]
            raw_s = raw.strip()
            req_pat = r"[=<>\\^*]" if in_table_row else r"[=<>\\^_*]"
            if (
                not raw_s
                or "@@INVAR_CODE_" in raw_s
                or _CODE_EXPRESSION_RE.search(raw_s)
                or _TELEMETRY_PREFIX_RE.match(raw_s)
                or not re.search(req_pat, raw_s)
            ):
                out_chunks.append(seg[pos:end])
                pos = end
                continue
            lead_ws = raw[: len(raw) - len(raw.lstrip())]
            trail_ws = raw[len(raw.rstrip()) :]
            out_chunks.append(seg[pos:st] + lead_ws + f"${_balance_math_expr(canonicalize_math_content(raw_s))}$" + trail_ws)
            pos = end
        parts[i] = "".join(out_chunks)
    return "".join(parts)


def _ensure_no_bare_latex_outside_math(text: str) -> str:
    """Guarantee zero bare LaTeX commands or ``_{...}``/``^{...}`` remain outside ``$...$``."""
    math_split_re = re.compile(r"(\$\$[\s\S]*?\$\$|\$(?:\\.|[^$\n])+?\$)")
    parts = math_split_re.split(text)
    for i in range(0, len(parts), 2):
        seg = parts[i]
        if not seg or ("\\" not in seg and "_{" not in seg and "^{" not in seg):
            continue

        def _wrap_leftover(m: re.Match[str]) -> str:
            raw = m.group(0).strip()
            if not raw or "@@INVAR_CODE_" in raw or _CODE_EXPRESSION_RE.search(raw):
                return m.group(0)
            return f"${_balance_math_expr(canonicalize_math_content(raw))}$"

        parts[i] = _BARE_LATEX_LEFTOVER_RE.sub(_wrap_leftover, seg)
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
    """Run all deterministic LaTeX repair & canonicalization passes on ``text``."""
    if not text:
        return text
    out = repair_json_control_escapes(text)
    out, tokens = _protect_code_spans(out)
    out = _wrap_table_equation_cells(out, tokens)
    out = _wrap_bare_math_in_prose(out)
    out = _ensure_no_bare_latex_outside_math(out)
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

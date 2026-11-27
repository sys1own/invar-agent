"""UPG-015 (BUG-014, 029, NF-14): Grounded equation proposals, ROOT_FIND safeguarded Newton
(Python/C++20/Rust), and DISCRETE_TIME_SYSTEM Direct Form I DSP (Python/Rust).
"""
from __future__ import annotations

import cmath
import math
import re
import unicodedata
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from invar_agent.builder.ingestion.schemas import (
    EquationProposal,
    RootCertificate,
)
from invar_agent.ir.expr_ast import expr_free_symbols, parse_expr

_NUM_RE = re.compile(r"(?<![A-Za-z_])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_STRUCTURAL_NUMS = {"0", "1", "2", "0.0", "1.0", "2.0"}


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def validate_equation_proposal(
    proposal: EquationProposal,
    source_texts: Sequence[str],
    declared_symbols: Mapping[str, str] | None = None,
) -> Tuple[bool, Optional[str]]:
    """UPG-015 / NF-14: 5-point groundedness check for equation proposals and `equation_patch`.

    Returns `(True, None)` if grounded, else `(False, "PROPOSAL_UNGROUNDED: ...")`.
    """
    q_norm = _norm_ws(proposal.quote_span)
    if not q_norm:
        return False, "PROPOSAL_UNGROUNDED: empty quote_span"

    # (1) quote_span occurs verbatim (after NFC + whitespace norm) in a source file
    if not any(q_norm in _norm_ws(src) for src in source_texts):
        return False, "PROPOSAL_UNGROUNDED: quote_span not found verbatim in source"

    # (2) every numeric literal in latex appears in the quote (0, 1, 2 allowed structurally)
    quote_nums = set(_NUM_RE.findall(q_norm))
    for num in _NUM_RE.findall(proposal.latex):
        if num in _STRUCTURAL_NUMS:
            continue
        if num not in quote_nums:
            # Also accept float-equivalent match (e.g. 1e5 vs 100000)
            try:
                nv = float(num)
                if not any(abs(float(qn) - nv) <= 1e-12 * max(1.0, abs(nv)) for qn in quote_nums):
                    return False, f"PROPOSAL_UNGROUNDED: numeric literal {num!r} absent from quote_span"
            except ValueError:
                return False, f"PROPOSAL_UNGROUNDED: numeric literal {num!r} absent from quote_span"

    # (4) parses with unified parser (handle lhs = rhs or residual)
    text = proposal.latex.strip()
    if "=" in text and "==" not in text:
        lhs_s, rhs_s = text.split("=", 1)
        expr_to_parse = f"({lhs_s}) - ({rhs_s})"
    else:
        expr_to_parse = text
    try:
        parsed = parse_expr(expr_to_parse, "formal")
    except Exception as exc:
        return False, f"PROPOSAL_UNGROUNDED: unified parser rejected latex ({exc})"

    # (3) every symbol appears in quote_span or is declared with a unit
    sym_units = dict(declared_symbols or {})
    for ps in proposal.symbols:
        sym_units[ps.name] = ps.unit
    free_syms = expr_free_symbols(parsed)
    for sym in free_syms:
        if sym in sym_units:
            continue
        if re.search(rf"\b{re.escape(sym)}\b", q_norm) is None:
            return False, f"PROPOSAL_UNGROUNDED: symbol {sym!r} not in quote_span and undeclared"

    return True, None


# ---------------------------------------------------------------------------
# B. ProgramKind.ROOT_FIND — Safeguarded Newton (Python / C++20 / Rust)
# ---------------------------------------------------------------------------

def solve_safeguarded_newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    symbol: str = "x",
    equation_id: str = "eq_root",
    max_iter: int = 100,
) -> RootCertificate:
    """UPG-015B: Bracket-preserving safeguarded Newton-Bisection solver.

    Maintains `f(a)*f(b) <= 0`; takes Newton step iff strictly inside `(a, b)` and halves
    width, else bisects. Stops when `hi - lo <= 4 * ulp(u)` or `f(u) == 0`.
    """
    a, b = float(lo), float(hi)
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise ValueError(f"ROOT_BRACKET_REQUIRED: invalid bracket [{a}, {b}]")
    fa, fb = float(f(a)), float(f(b))
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise ValueError(f"ROOT_BRACKET_REQUIRED: non-finite endpoint residual f({a})={fa}, f({b})={fb}")
    if fa == 0.0:
        return RootCertificate(symbol=symbol, equation_id=equation_id, root=a, residual=0.0, bracket=(a, a), iterations=0)
    if fb == 0.0:
        return RootCertificate(symbol=symbol, equation_id=equation_id, root=b, residual=0.0, bracket=(b, b), iterations=0)
    if fa * fb > 0.0:
        raise ValueError(f"ROOT_BRACKET_REQUIRED: no sign change f({a})={fa}, f({b})={fb}")

    u = 0.5 * (a + b)
    for it in range(1, max_iter + 1):
        fu = float(f(u))
        if not math.isfinite(fu):
            u = 0.5 * (a + b)
            fu = float(f(u))
        if fu == 0.0 or (b - a) <= 4.0 * math.ulp(max(abs(u), 1e-300)):
            return RootCertificate(
                symbol=symbol,
                equation_id=equation_id,
                root=u,
                residual=fu,
                bracket=(a, b),
                iterations=it,
            )
        if fa * fu < 0.0:
            b, fb = u, fu
        else:
            a, fa = u, fu
        width = b - a
        dfu = float(df(u))
        cand = u - fu / dfu if (math.isfinite(dfu) and abs(dfu) > 1e-300) else math.nan
        if math.isfinite(cand) and a < cand < b and abs(cand - u) <= 0.5 * width:
            u = cand
        else:
            u = 0.5 * (a + b)

    raise ValueError(f"ROOT_NOT_CONVERGED: exceeded {max_iter} iterations on [{a}, {b}]")


def emit_root_find_python(fn_name: str, residual_expr: str, deriv_expr: str, sym: str, params: Sequence[str]) -> str:
    p_sig = ", ".join([*params, "lo: float", "hi: float", "max_iter: int = 100"])
    return (
        "import math\n\n"
        f"def {fn_name}({p_sig}) -> float:\n"
        f"    def _f({sym}: float) -> float:\n"
        f"        return {residual_expr}\n"
        f"    def _df({sym}: float) -> float:\n"
        f"        return {deriv_expr}\n"
        "    a, b = float(lo), float(hi)\n"
        "    fa, fb = _f(a), _f(b)\n"
        "    if fa * fb > 0.0:\n"
        "        raise ValueError('ROOT_BRACKET_REQUIRED')\n"
        "    u = 0.5 * (a + b)\n"
        "    for _ in range(max_iter):\n"
        "        fu = _f(u)\n"
        "        if fu == 0.0 or (b - a) <= 4.0 * math.ulp(max(abs(u), 1e-300)):\n"
        "            return u\n"
        "        if fa * fu < 0.0:\n"
        "            b, fb = u, fu\n"
        "        else:\n"
        "            a, fa = u, fu\n"
        "        dfu = _df(u)\n"
        "        cand = u - fu / dfu if (math.isfinite(dfu) and abs(dfu) > 1e-300) else math.nan\n"
        "        u = cand if (math.isfinite(cand) and a < cand < b and abs(cand - u) <= 0.5 * (b - a)) else 0.5 * (a + b)\n"
        "    raise ValueError('ROOT_NOT_CONVERGED')\n"
    )


def emit_root_find_cpp(fn_name: str, residual_expr: str, deriv_expr: str, sym: str, params: Sequence[str]) -> str:
    p_sig = ", ".join([*(f"double {p}" for p in params), "double lo", "double hi", "int max_iter = 100"])
    return (
        "#include <cmath>\n#include <limits>\n\n"
        f"double {fn_name}({p_sig}) {{\n"
        f"    auto f = [&](double {sym}) -> double {{ return {residual_expr}; }};\n"
        f"    auto df = [&](double {sym}) -> double {{ return {deriv_expr}; }};\n"
        "    double a = lo, b = hi, fa = f(a), fb = f(b);\n"
        "    if (fa * fb > 0.0) return std::numeric_limits<double>::quiet_NaN();\n"
        "    double u = 0.5 * (a + b);\n"
        "    for (int it = 0; it < max_iter; ++it) {\n"
        "        double fu = f(u);\n"
        "        double tol = 4.0 * std::abs(std::nextafter(u, std::numeric_limits<double>::infinity()) - u);\n"
        "        if (fu == 0.0 || (b - a) <= tol) return u;\n"
        "        if (fa * fu < 0.0) { b = u; fb = fu; } else { a = u; fa = fu; }\n"
        "        double dfu = df(u);\n"
        "        double cand = (std::isfinite(dfu) && std::abs(dfu) > 1e-300) ? (u - fu / dfu) : std::numeric_limits<double>::quiet_NaN();\n"
        "        u = (std::isfinite(cand) && cand > a && cand < b && std::abs(cand - u) <= 0.5 * (b - a)) ? cand : 0.5 * (a + b);\n"
        "    }\n"
        "    return std::numeric_limits<double>::quiet_NaN();\n"
        "}\n"
    )


def emit_root_find_rust(fn_name: str, residual_expr: str, deriv_expr: str, sym: str, params: Sequence[str]) -> str:
    p_sig = ", ".join([*(f"{p}: f64" for p in params), "lo: f64", "hi: f64", "max_iter: usize"])
    return (
        f"pub fn {fn_name}({p_sig}) -> f64 {{\n"
        f"    let f = |{sym}: f64| -> f64 {{ {residual_expr} }};\n"
        f"    let df = |{sym}: f64| -> f64 {{ {deriv_expr} }};\n"
        "    let (mut a, mut b) = (lo, hi);\n"
        "    let (mut fa, fb) = (f(a), f(b));\n"
        "    if fa * fb > 0.0 { return f64::NAN; }\n"
        "    let mut u = 0.5 * (a + b);\n"
        "    for _ in 0..max_iter {\n"
        "        let fu = f(u);\n"
        "        let tol = 4.0 * (u.abs().max(1e-300) * f64::EPSILON);\n"
        "        if fu == 0.0 || (b - a) <= tol { return u; }\n"
        "        if fa * fu < 0.0 { b = u; } else { a = u; fa = fu; }\n"
        "        let dfu = df(u);\n"
        "        let cand = if dfu.is_finite() && dfu.abs() > 1e-300 { u - fu / dfu } else { f64::NAN };\n"
        "        u = if cand.is_finite() && cand > a && cand < b && (cand - u).abs() <= 0.5 * (b - a) { cand } else { 0.5 * (a + b) };\n"
        "    }\n"
        "    f64::NAN\n"
        "}\n"
    )


# ---------------------------------------------------------------------------
# C. ProgramKind.DISCRETE_TIME_SYSTEM — Direct Form I DSP (Python / Rust)
# ---------------------------------------------------------------------------

_INDEXED_DELAY_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?:\[\s*([a-z])\s*(?:-\s*(\d+))?\s*\]|_\{?\s*([a-z])\s*(?:-\s*(\d+))?\s*\}?)"
)
_AMBIGUOUS_SUB_RE = re.compile(r"\b([A-Za-z])_(\d+)\b")


def parse_discrete_difference_equation(text: str) -> Dict[str, Any]:
    """Recognize discrete-time lag terms (`y[n]`, `y[n-1]`, `x_{n-2}`, `y_n`).

    Plain `y_1`/`x_1` without an index variable raises `ValueError("DISCRETE_LAG_AMBIGUOUS")`.
    """
    matches = list(_INDEXED_DELAY_RE.finditer(text))
    if not matches:
        if _AMBIGUOUS_SUB_RE.search(text):
            raise ValueError("DISCRETE_LAG_AMBIGUOUS: plain subscript without index variable (use y[n-1] or SymbolResolution(role='DELAY'))")
        return {"is_discrete": False, "terms": []}
    terms = []
    for m_obj in matches:
        sym = m_obj.group(1)
        idx_var = m_obj.group(2) or m_obj.group(4)
        lag_s = m_obj.group(3) or m_obj.group(5) or "0"
        lag = int(lag_s)
        terms.append({"symbol": sym, "index_var": idx_var, "lag": lag})
    return {"is_discrete": True, "terms": terms}


def _poly_roots_companion(a: Sequence[float]) -> List[complex]:
    """Compute polynomial roots of `a[0]*z^p + a[1]*z^{p-1} + ... + a[p] = 0` via numpy companion eigenvalues."""
    import numpy as np

    coeffs = [float(c) for c in a]
    if len(coeffs) <= 1:
        return []
    return [complex(r) for r in np.roots(coeffs)]


def analyze_discrete_iir(
    b: Sequence[float],
    a: Sequence[float],
    *,
    step_samples: int = 128,
    omegas: Sequence[float] = (0.0, math.pi / 2.0, math.pi),
) -> Dict[str, Any]:
    """Direct Form I IIR analysis: pole radius, stability, |H(e^{jw})|, and unit-step response."""
    b_c = [float(x) for x in b]
    a_c = [float(x) for x in a]
    if not a_c or a_c[0] == 0.0:
        raise ValueError("IIR denominator a[0] must be non-zero")
    a0 = a_c[0]
    b_n = [x / a0 for x in b_c]
    a_n = [x / a0 for x in a_c]

    poles = _poly_roots_companion(a_n)
    pole_radius = max((abs(p) for p in poles), default=0.0)
    stable = pole_radius < 1.0

    # Frequency response |H(e^{j*omega})|
    mag_resp: Dict[float, float] = {}
    for w in omegas:
        ejw = cmath.exp(-1j * w)
        num = sum(b_n[k] * (ejw ** k) for k in range(len(b_n)))
        den = sum(a_n[k] * (ejw ** k) for k in range(len(a_n)))
        mag_resp[float(w)] = abs(num / den) if abs(den) > 1e-300 else math.inf

    # Direct Form I unit-step response
    x_hist = [0.0] * len(b_n)
    y_hist = [0.0] * len(a_n)
    step_resp: List[float] = []
    for _n in range(step_samples):
        x_hist = [1.0] + x_hist[:-1]
        y_val = sum(b_n[k] * x_hist[k] for k in range(len(b_n))) - sum(
            a_n[k] * y_hist[k - 1] for k in range(1, len(a_n))
        )
        y_hist = [y_val] + y_hist[:-1]
        step_resp.append(y_val)

    dc_gain = mag_resp.get(0.0, sum(b_n) / sum(a_n))
    peak_val = max(step_resp) if step_resp else 0.0
    peak_idx = step_resp.index(peak_val) if step_resp else 0
    overshoot_pct = ((peak_val - dc_gain) / abs(dc_gain) * 100.0) if abs(dc_gain) > 1e-12 else 0.0

    return {
        "pole_radius": pole_radius,
        "status": "STABLE" if stable else "UNSTABLE",
        "frequency_response": mag_resp,
        "step_response": step_resp,
        "peak_value": peak_val,
        "peak_index": peak_idx,
        "overshoot_pct": overshoot_pct,
    }


def emit_discrete_iir_python(fn_name: str, b: Sequence[float], a: Sequence[float]) -> str:
    return (
        f"def {fn_name}(x_seq):\n"
        f"    b = {list(b)!r}\n"
        f"    a = {list(a)!r}\n"
        "    a0 = a[0]\n"
        "    bn = [v / a0 for v in b]\n"
        "    an = [v / a0 for v in a]\n"
        "    xh = [0.0] * len(bn)\n"
        "    yh = [0.0] * len(an)\n"
        "    out = []\n"
        "    for xv in x_seq:\n"
        "        xh = [float(xv)] + xh[:-1]\n"
        "        yv = sum(bn[k] * xh[k] for k in range(len(bn))) - sum(an[k] * yh[k - 1] for k in range(1, len(an)))\n"
        "        yh = [yv] + yh[:-1]\n"
        "        out.append(yv)\n"
        "    return out\n"
    )


def emit_discrete_iir_rust(fn_name: str, b: Sequence[float], a: Sequence[float]) -> str:
    b_lit = ", ".join(f"{float(v):.17e}" for v in b)
    a_lit = ", ".join(f"{float(v):.17e}" for v in a)
    return (
        f"pub fn {fn_name}(x_seq: &[f64]) -> Vec<f64> {{\n"
        f"    let b = [{b_lit}];\n"
        f"    let a = [{a_lit}];\n"
        "    let a0 = a[0];\n"
        "    let mut xh = vec![0.0_f64; b.len()];\n"
        "    let mut yh = vec![0.0_f64; a.len()];\n"
        "    let mut out = Vec::with_capacity(x_seq.len());\n"
        "    for &xv in x_seq {\n"
        "        for k in (1..xh.len()).rev() { xh[k] = xh[k - 1]; }\n"
        "        xh[0] = xv;\n"
        "        let mut yv = 0.0_f64;\n"
        "        for k in 0..b.len() { yv += (b[k] / a0) * xh[k]; }\n"
        "        for k in 1..a.len() { yv -= (a[k] / a0) * yh[k - 1]; }\n"
        "        for k in (1..yh.len()).rev() { yh[k] = yh[k - 1]; }\n"
        "        yh[0] = yv;\n"
        "        out.push(yv);\n"
        "    }\n"
        "    out\n"
        "}\n"
    )

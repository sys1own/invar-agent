"""UPG-009 (BUG-022): Formal Model Fidelity Checker & Compiler-Owned Analytics."""
from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from invar_agent import _native
from invar_agent.ir.expr_ast import canonicalize, parse_expr, substitute_expr, Sym
from invar_agent.research import meaningfulness as m
from invar_agent.research.claims import SpanRef


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StatedEquation(_Strict):
    id: str
    kind: Literal["DERIVATIVE", "ALGEBRAIC", "FORCING", "INITIAL", "CONSERVATION"]
    lhs: str
    rhs_text: str
    condition: Optional[str] = None
    span: SpanRef
    symbol_map: Dict[str, str] = Field(default_factory=dict)


class FidelityFinding(_Strict):
    code: Literal[
        "FIDELITY_MISMATCH",
        "FIDELITY_MISSING_FORCING",
        "FIDELITY_UNMAPPED_SYMBOL",
        "FIDELITY_UNDECIDED",
        "FIDELITY_SPAN_UNCLAIMED",
    ]
    equation_id: str
    stated: str
    compiled: str
    witness: Optional[Dict[str, float]] = None


class AnalysisRequest(_Strict):
    kind: Literal["EQUILIBRIUM", "JACOBIAN", "EIGENVALUES", "STABILITY", "SENSITIVITY"]
    states: List[str] = Field(default_factory=list)
    parameter: Optional[str] = None
    output: Optional[str] = None


_DELIM_MATH_RE = re.compile(
    r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+\$|```math[\s\S]+?```)"
)
_INLINE_EQ_RE = re.compile(
    r"\b(?:d[A-Za-z_][A-Za-z0-9_]*\s*/\s*d[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*(?:\([A-Za-z_][A-Za-z0-9_]*\))?)\s*=\s*([^\n,;]+?)(?=\s+(?:and|where|with|for|subject)\b|\.(?:\s|$)|[,;\n]|$)"
)


def scan_math_spans(prompt: str) -> List[SpanRef]:
    """UPG-009 step 2: Scan prompt for delimited math or inline `d?ID(/dt)? = EXPR` with >=1 operator."""
    if not prompt:
        return []
    spans: List[SpanRef] = []
    occupied: List[Tuple[int, int]] = []

    def _overlaps(s: int, e: int) -> bool:
        return any(not (e <= os or s >= oe) for os, oe in occupied)

    for match in _DELIM_MATH_RE.finditer(prompt):
        s, e = match.start(), match.end()
        if "=" in match.group(0):
            spans.append(SpanRef(start=s, end=e, quote=match.group(0)[:600]))
            occupied.append((s, e))

    for match in _INLINE_EQ_RE.finditer(prompt):
        s, e = match.start(), match.end()
        if _overlaps(s, e):
            continue
        rhs = match.group(1).strip().rstrip(".")
        if not any(op in rhs for op in ("+", "-", "*", "/", "^", "(")):
            continue
        try:
            parse_expr(rhs, "formal")
        except Exception:
            continue
        spans.append(SpanRef(start=s, end=e, quote=match.group(0)[:600]))
        occupied.append((s, e))

    spans.sort(key=lambda sp: sp.start)
    return spans


def _halton(index: int, base: int) -> float:
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


def check_expression_equivalence(
    stated_rhs: str,
    compiled_rhs: str,
    claim_box: Mapping[str, Tuple[float, float]],
) -> Tuple[Literal["EQUIVALENT", "MISMATCH", "UNDECIDED"], Optional[Dict[str, float]]]:
    """UPG-009 step 3 equivalence ladder:
    (a) canonical AST equality ->
    (b) exact rational normal form (`sympy.cancel(a - b) == 0`) ->
    (c) `py_check_algebraic_equivalence` ->
    (d) 64 Halton point interval refutation of `a - b`.
    """
    import sympy as sp

    # (a) Canonical AST equality
    try:
        ea = canonicalize(parse_expr(stated_rhs, "formal"))
        eb = canonicalize(parse_expr(compiled_rhs, "formal"))
        if ea == eb:
            return "EQUIVALENT", None
    except Exception:
        pass

    # (b) Exact rational normal form via sympy.cancel / trigsimp
    try:
        sa = sp.sympify(stated_rhs.replace("^", "**"), rational=True)
        sb = sp.sympify(compiled_rhs.replace("^", "**"), rational=True)
        diff = sp.cancel(sa - sb)
        if diff == 0 or sp.trigsimp(diff) == 0:
            return "EQUIVALENT", None
    except Exception:
        pass

    # (c) Native CSEL e-graph equivalence
    try:
        ca = json.dumps(m._csel(m._parse(stated_rhs.replace("^", "**"))))
        cb = json.dumps(m._csel(m._parse(compiled_rhs.replace("^", "**"))))
        eq_res = json.loads(_native.py_check_algebraic_equivalence(ca, cb))
        if eq_res.get("equivalent"):
            return "EQUIVALENT", None
    except Exception:
        pass

    # (d) 64 Halton point interval refutation in claim_box
    try:
        diff_ast = m._parse(f"({stated_rhs.replace('^', '**')}) - ({compiled_rhs.replace('^', '**')})")
        free_syms = sorted(m._variables(diff_ast))
        box_syms = [s for s in free_syms if s in claim_box]
        if len(box_syms) == len(free_syms):
            for k in range(1, 65):
                pt: Dict[str, float] = {}
                pt_box: Dict[str, Tuple[float, float]] = {}
                for dim_i, sym in enumerate(box_syms):
                    lo, hi = claim_box[sym]
                    u = _halton(k, _PRIMES[dim_i % len(_PRIMES)])
                    val = lo + u * (hi - lo)
                    pt[sym] = val
                    pt_box[sym] = (val, val)
                try:
                    iv = m._iv(diff_ast, pt_box)
                    if iv.lo > 1e-12 or iv.hi < -1e-12:
                        return "MISMATCH", pt
                except Exception:
                    continue
    except Exception:
        pass
    return "UNDECIDED", None


def check_model_fidelity(
    prompt: str,
    stated_equations: Sequence[StatedEquation],
    compiled_equations: Mapping[str, str],
    claim_box: Mapping[str, Tuple[float, float]],
    *,
    forcing_schedule: Optional[m.ForcingSchedule] = None,
    stated_forcing_segments: Sequence[Tuple[str, str, str, str]] = (),
) -> List[FidelityFinding]:
    """Run UPG-009 fidelity check across all stated equations and forcing intervals."""
    findings: List[FidelityFinding] = []

    # 1. Check unclaimed math spans
    math_spans = scan_math_spans(prompt)
    for idx, ms in enumerate(math_spans):
        claimed = any(
            not (se.span.end <= ms.start or se.span.start >= ms.end)
            for se in stated_equations
        )
        if not claimed:
            findings.append(
                FidelityFinding(
                    code="FIDELITY_SPAN_UNCLAIMED",
                    equation_id=f"span_{idx}",
                    stated=ms.quote,
                    compiled="",
                )
            )

    # 2. Check each stated equation against compiled equations
    for se in stated_equations:
        if se.kind == "FORCING":
            continue
        target = se.symbol_map.get(se.lhs, se.lhs)
        if target not in compiled_equations:
            findings.append(
                FidelityFinding(
                    code="FIDELITY_UNMAPPED_SYMBOL",
                    equation_id=se.id,
                    stated=f"{se.lhs} = {se.rhs_text}",
                    compiled="",
                )
            )
            continue
        # Map stated symbols to compiled symbols via AST substitution
        try:
            ast_rhs = parse_expr(se.rhs_text, "formal")
            if se.symbol_map:
                ast_rhs = substitute_expr(ast_rhs, {k: Sym(name=v) for k, v in se.symbol_map.items()})
            from invar_agent.ir.expr_ast import render_expr
            mapped_rhs = render_expr(ast_rhs, "formal")
        except Exception:
            mapped_rhs = se.rhs_text

        status, witness = check_expression_equivalence(mapped_rhs, compiled_equations[target], claim_box)
        if status == "MISMATCH":
            findings.append(
                FidelityFinding(
                    code="FIDELITY_MISMATCH",
                    equation_id=se.id,
                    stated=mapped_rhs,
                    compiled=compiled_equations[target],
                    witness=witness,
                )
            )
        elif status == "UNDECIDED":
            findings.append(
                FidelityFinding(
                    code="FIDELITY_UNDECIDED",
                    equation_id=se.id,
                    stated=mapped_rhs,
                    compiled=compiled_equations[target],
                )
            )

    # 3. Forcing schedule verification
    if stated_forcing_segments:
        if forcing_schedule is None:
            for idx, (sym, t0, t1, val) in enumerate(stated_forcing_segments):
                findings.append(
                    FidelityFinding(
                        code="FIDELITY_MISSING_FORCING",
                        equation_id=f"forcing_{idx}",
                        stated=f"{sym}={val} on [{t0}, {t1}]",
                        compiled="",
                    )
                )
        else:
            for idx, (sym, t0, t1, val) in enumerate(stated_forcing_segments):
                t0_f, t1_f, val_f = Fraction(t0), Fraction(t1), Fraction(val)
                matched = any(
                    Fraction(seg.t_start) == t0_f
                    and Fraction(seg.t_end) == t1_f
                    and any(Fraction(v) == val_f for v in seg.values.values())
                    for seg in forcing_schedule.segments
                )
                if not matched:
                    findings.append(
                        FidelityFinding(
                            code="FIDELITY_MISSING_FORCING",
                            equation_id=f"forcing_{idx}",
                            stated=f"{sym}={val} on [{t0}, {t1}]",
                            compiled=str([(s.t_start, s.t_end, s.values) for s in forcing_schedule.segments]),
                        )
                    )

    return findings


# ---------------------------------------------------------------------------
# Compiler-Owned Analytics: Equilibrium, Jacobian, 2x2 Interval Eigenvalues, IFT Sensitivities
# ---------------------------------------------------------------------------

def _iv_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (math.nextafter(a[0] + b[0], -math.inf), math.nextafter(a[1] + b[1], math.inf))


def _iv_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (math.nextafter(a[0] - b[1], -math.inf), math.nextafter(a[1] - b[0], math.inf))


def _iv_mul(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    ps = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (math.nextafter(min(ps), -math.inf), math.nextafter(max(ps), math.inf))


def compute_2x2_interval_eigenvalues(
    j_iv: Sequence[Sequence[Tuple[float, float]]],
) -> Dict[str, Any]:
    """Certified 2x2 interval complex eigenvalues `tr/2 +/- sqrt(tr^2 - 4*det)/2`.

    Raises `ValueError("EIGENVALUE_MULTIPLICITY_UNRESOLVED")` if the discriminant
    interval straddles 0.
    """
    if len(j_iv) != 2 or len(j_iv[0]) != 2 or len(j_iv[1]) != 2:
        raise ValueError("compute_2x2_interval_eigenvalues requires a 2x2 interval matrix")
    a, b = j_iv[0][0], j_iv[0][1]
    c, d = j_iv[1][0], j_iv[1][1]
    tr = _iv_add(a, d)
    det = _iv_sub(_iv_mul(a, d), _iv_mul(b, c))
    tr2 = _iv_mul(tr, tr)
    four_det = (4.0 * det[0], 4.0 * det[1])
    disc = _iv_sub(tr2, four_det)
    half_tr = (0.5 * tr[0], 0.5 * tr[1])

    if disc[0] <= 0.0 <= disc[1]:
        raise ValueError(f"EIGENVALUE_MULTIPLICITY_UNRESOLVED: discriminant interval {disc} straddles 0")

    if disc[1] < 0.0:
        # Complex conjugate pair: re = tr/2, im = +/- sqrt(-disc)/2
        neg_lo, neg_hi = -disc[1], -disc[0]
        im_lo = math.nextafter(0.5 * math.sqrt(max(0.0, neg_lo)), -math.inf)
        im_hi = math.nextafter(0.5 * math.sqrt(neg_hi), math.inf)
        return {
            "complex": True,
            "eig0__re": half_tr,
            "eig0__im": (im_lo, im_hi),
            "eig1__re": half_tr,
            "eig1__im": (-im_hi, -im_lo),
            "stable": half_tr[1] < 0.0,
        }

    # Distinct real eigenvalues: re = tr/2 +/- sqrt(disc)/2, im = 0
    s_lo = math.nextafter(0.5 * math.sqrt(max(0.0, disc[0])), -math.inf)
    s_hi = math.nextafter(0.5 * math.sqrt(disc[1]), math.inf)
    e0 = _iv_add(half_tr, (s_lo, s_hi))
    e1 = _iv_sub(half_tr, (s_lo, s_hi))
    return {
        "complex": False,
        "eig0__re": e0,
        "eig0__im": (0.0, 0.0),
        "eig1__re": e1,
        "eig1__im": (0.0, 0.0),
        "stable": e0[1] < 0.0 and e1[1] < 0.0,
    }


def compute_equilibrium_analytics(
    state_names: Sequence[str],
    rhs_exprs: Mapping[str, str],
    params: Mapping[str, float],
    initial_guess: Mapping[str, float],
    *,
    sensitivity_params: Sequence[str] = (),
) -> Dict[str, Any]:
    """Compute compiler-owned equilibrium `eq__*`, Jacobian `jac__*`, eigenvalues `eig*__re/im`,
    and IFT sensitivities `sens__*__*` (UPG-009).
    """
    import sympy as sp

    n = len(state_names)
    syms = [sp.Symbol(s) for s in state_names]
    p_syms = {p: sp.Symbol(p) for p in params}
    all_locals = {s: sp.Symbol(s) for s in state_names}
    all_locals.update(p_syms)
    f_exprs = [sp.sympify(rhs_exprs[s].replace("^", "**"), locals=all_locals, rational=True) for s in state_names]
    j_sym = sp.Matrix(f_exprs).jacobian(syms)

    # Damped Newton solve for equilibrium F(x*, p) = 0
    sub_p = {p_syms[k]: sp.Float(v) for k, v in params.items()}
    f_num = sp.lambdify(syms, [e.subs(sub_p) for e in f_exprs], "math")
    j_num = sp.lambdify(syms, j_sym.subs(sub_p).tolist(), "math")

    x = [float(initial_guess[s]) for s in state_names]
    for _ in range(50):
        fv = [float(v) for v in f_num(*x)]
        if max(abs(v) for v in fv) < 1e-13:
            break
        jm = sp.Matrix([[float(c) for c in row] for row in j_num(*x)])
        dx = list(jm.LUsolve(sp.Matrix([-v for v in fv])))
        x = [x[i] + float(dx[i]) for i in range(n)]

    eq_map = {f"eq__{s}": x[i] for i, s in enumerate(state_names)}
    j_at = [[float(c) for c in row] for row in j_num(*x)]
    jac_map = {
        f"jac__{i}_{j}": j_at[i][j]
        for i in range(n)
        for j in range(n)
    }

    # Tiny interval box around x* for certified 2x2 eigenvalues
    eps_box = 1e-10
    j_iv = [[(j_at[i][j] - eps_box, j_at[i][j] + eps_box) for j in range(n)] for i in range(n)]
    eigs: Dict[str, Any] = {}
    if n == 2:
        eigs = compute_2x2_interval_eigenvalues(j_iv)

    # IFT equilibrium sensitivities: dx*/dp = -J^{-1} (dF/dp)
    sens_map: Dict[str, float] = {}
    if sensitivity_params:
        j_mat = sp.Matrix(j_at)
        for p in sensitivity_params:
            if p not in p_syms:
                continue
            df_dp = sp.Matrix([
                float(sp.diff(e, p_syms[p]).subs({**sub_p, **{syms[i]: x[i] for i in range(n)}}))
                for e in f_exprs
            ])
            dx_dp = -j_mat.LUsolve(df_dp)
            for i, s in enumerate(state_names):
                sens_map[f"sens__{s}__{p}"] = float(dx_dp[i])

    return {
        "equilibrium": eq_map,
        "jacobian": jac_map,
        "eigenvalues": eigs,
        "sensitivities": sens_map,
        "ledger_tag": "ANALYTIC_COMPILER_OWNED",
    }

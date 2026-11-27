"""ARCH-002 Claim Ledger & Fidelity Gate (BUG-005, BUG-006, BUG-024, BUG-026).

Deterministically extracts span-anchored numeric literals, units, intervals,
comparators, and time durations from the user's natural-language prompt so:
1. User-fixed values (e.g. `T_hot = 600 K`, `gamma = 0.1/day`) are pinned to
   immutable point domains `[v, v]` and never widened by the LLM (BUG-026).
2. Tautological obligations entailed by guards/environment alone (e.g. replacing
   a 4 K tipping claim with mass conservation) are rejected as `TAUTOLOGICAL_TARGET`
   (BUG-005).
3. Missing or weakened thresholds are flagged as `CLAIM_UNCOVERED` / `CLAIM_WEAKENED`.
4. SI time horizons match explicit user durations or span >= 10x the slowest
   characteristic timescale (`1 / min_rate`) so unit-rescaled models (e.g. SIR in
   seconds) always cover the true peak (BUG-006).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


TIME_UNIT_TO_SECONDS: Dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "wk": 7.0 * 86400.0,
    "week": 7.0 * 86400.0,
    "weeks": 7.0 * 86400.0,
    "mo": 2629746.0,
    "month": 2629746.0,
    "months": 2629746.0,
    "quarter": 7889238.0,
    "quarters": 7889238.0,
    "yr": 365.25 * 86400.0,
    "yrs": 365.25 * 86400.0,
    "year": 365.25 * 86400.0,
    "years": 365.25 * 86400.0,
}


@dataclass(frozen=True)
class PinnedLiteral:
    symbol: str
    value: float
    raw_unit: str
    si_value: float
    span: Tuple[int, int]


@dataclass(frozen=True)
class ClaimClause:
    raw_text: str
    comparator: str  # "<=", ">=", "<", ">", "=="
    threshold: float
    symbol_hint: Optional[str]
    span: Tuple[int, int]


@dataclass(frozen=True)
class ClaimFidelityViolation:
    code: str  # "TAUTOLOGICAL_TARGET" | "CLAIM_UNCOVERED" | "CLAIM_WEAKENED" | "HORIZON_INSUFFICIENT"
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimLedger:
    prompt: str
    pinned_literals: List[PinnedLiteral] = field(default_factory=list)
    intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    claim_clauses: List[ClaimClause] = field(default_factory=list)
    duration_si_s: Optional[float] = None
    duration_raw: Optional[str] = None


_NUM_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

_INTERVAL_RE = re.compile(
    rf"\b([A-Za-z][A-Za-z0-9_]{{0,24}})\s*(?:in|∈)\s*\[\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*\]"
)

_PINNED_ASSIGN_RE = re.compile(
    rf"\b([A-Za-z][A-Za-z0-9_]{{0,24}})\s*=\s*({_NUM_RE})\s*(?:/\s*(day|days|hr|hour|hours|min|s|sec|yr|year|years)\b|(K|kg|m|s|W|J|Pa|Hz|V|mV|mol|F/m\^?2|W/\(m\^?2\*?K\)|GtC|%))?"
)

_CLAIM_CMP_RE = re.compile(
    rf"\b([A-Za-z][A-Za-z0-9_]{{0,24}})\s*(<=|>=|<|>)\s*({_NUM_RE})\s*(%|K|W|nW|mV|V)?"
)

_PCT_CLAIM_RE = re.compile(
    rf"\b(?:at\s+least|exceed\w*|>=?|above|greater\s+than|reach\w*|achieve\w*)\s+({_NUM_RE})\s*%",
    re.I,
)

_DURATION_RE = re.compile(
    rf"\b(?:over|for|during|up\s+to|horizon\s*(?:of|=)?|t\s*(?:in|from\s+0\s+to|<=|=)\s*(?:\[0\s*,\s*)?)?\s*({_NUM_RE})\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|years?|yrs?|s|ms)\b",
    re.I,
)


def _convert_rate_or_unit(value: float, rate_unit: Optional[str], unit: Optional[str]) -> Tuple[float, str]:
    if rate_unit:
        scale = TIME_UNIT_TO_SECONDS.get(rate_unit.lower(), 1.0)
        return (value / scale, f"1/{rate_unit}")
    if unit == "%":
        return (value / 100.0, "%")
    if unit == "mV":
        return (value * 1e-3, "V")
    return (value, unit or "")


def extract_claim_ledger(prompt: str) -> ClaimLedger:
    """Deterministically extract span-anchored literals, intervals, comparators, and SI durations."""
    text = str(prompt or "")
    ledger = ClaimLedger(prompt=text)

    for m_iv in _INTERVAL_RE.finditer(text):
        sym = m_iv.group(1)
        try:
            lo, hi = float(m_iv.group(2)), float(m_iv.group(3))
            if lo < hi:
                ledger.intervals[sym] = (lo, hi)
        except ValueError:
            pass

    for m_as in _PINNED_ASSIGN_RE.finditer(text):
        sym = m_as.group(1)
        if sym in ledger.intervals or sym.lower() in {"if", "and", "or", "is", "for", "over", "at"}:
            continue
        # NF-10: Reject equation fragments preceded/followed by arithmetic operators or inside parenthesised equations
        pre = text[:m_as.start()].rstrip()
        post = text[m_as.end():].lstrip()
        if pre and pre[-1] in "+-*/^":
            continue
        if post and post[0] in "+-*/^":
            continue
        # Check if inside an unclosed parenthesis (e.g., `(u_c = 1.0 - ...)` or `f(x = 1.0)`)
        if pre.count("(") > pre.count(")") and ")" in post[:40]:
            continue
        try:
            raw_val = float(m_as.group(2))
        except ValueError:
            continue
        si_val, unit_str = _convert_rate_or_unit(raw_val, m_as.group(3), m_as.group(4))
        ledger.pinned_literals.append(
            PinnedLiteral(
                symbol=sym,
                value=raw_val,
                raw_unit=unit_str,
                si_value=si_val,
                span=(m_as.start(), m_as.end()),
            )
        )

    for m_cmp in _CLAIM_CMP_RE.finditer(text):
        sym, op, num_s, unit = m_cmp.group(1), m_cmp.group(2), m_cmp.group(3), m_cmp.group(4)
        if sym.lower() in {"if", "and", "or", "for", "over", "t"}:
            continue
        try:
            val = float(num_s)
        except ValueError:
            continue
        if unit == "%":
            val = val / 100.0
        ledger.claim_clauses.append(
            ClaimClause(
                raw_text=m_cmp.group(0),
                comparator=op,
                threshold=val,
                symbol_hint=sym,
                span=(m_cmp.start(), m_cmp.end()),
            )
        )

    for m_pct in _PCT_CLAIM_RE.finditer(text):
        try:
            val = float(m_pct.group(1)) / 100.0
        except ValueError:
            continue
        if not any(abs(c.threshold - val) < 1e-9 for c in ledger.claim_clauses):
            ledger.claim_clauses.append(
                ClaimClause(
                    raw_text=m_pct.group(0),
                    comparator=">=",
                    threshold=val,
                    symbol_hint=None,
                    span=(m_pct.start(), m_pct.end()),
                )
            )

    # Extract explicit time horizon (prefer matches preceded by "over/for/during/horizon/t")
    best_dur: Optional[Tuple[int, float, str]] = None
    for m_dur in _DURATION_RE.finditer(text):
        prefix = m_dur.group(0).lower()
        unit = m_dur.group(2).lower()
        if unit not in TIME_UNIT_TO_SECONDS:
            continue
        # Avoid matching rate denominators like "/day" or "per day"
        if m_dur.start() > 0 and text[m_dur.start() - 1] == "/":
            continue
        try:
            num = float(m_dur.group(1))
        except ValueError:
            continue
        if num <= 0:
            continue
        si_s = num * TIME_UNIT_TO_SECONDS[unit]
        priority = 2 if any(k in prefix for k in ("over", "for", "during", "horizon", "t ")) else 1
        if best_dur is None or priority > best_dur[0]:
            best_dur = (priority, si_s, m_dur.group(0).strip())
    if best_dur is not None:
        ledger.duration_si_s = best_dur[1]
        ledger.duration_raw = best_dur[2]

    return ledger


def pin_user_literals_in_domain(
    ledger: ClaimLedger,
    domain: Mapping[str, Sequence[float]],
) -> Dict[str, List[float]]:
    """Pin user-fixed prompt values to immutable point domains `[v, v]` (BUG-026).

    Never widens a user-fixed value (like `T_hot = 600 K` or `gamma = 0.1/day`)
    into an arbitrary interval.
    """
    out: Dict[str, List[float]] = {k: [float(v[0]), float(v[1])] for k, v in domain.items()}
    pinned_by_sym: Dict[str, PinnedLiteral] = {}
    for lit in ledger.pinned_literals:
        pinned_by_sym.setdefault(lit.symbol, lit)
        pinned_by_sym.setdefault(lit.symbol.lower(), lit)

    for sym, bounds in list(out.items()):
        if sym in ledger.intervals:
            continue
        lit = pinned_by_sym.get(sym) or pinned_by_sym.get(sym.lower())
        if lit is None:
            # Also match unit-suffixed names like T_hot_K -> T_hot
            base = re.sub(r"_(?:K|s|m|kg|W|J|Pa|V|Hz|mol)$", "", sym)
            lit = pinned_by_sym.get(base) or pinned_by_sym.get(base.lower())
        if lit is None:
            continue
        lo, hi = bounds[0], bounds[1]
        # Pin if the LLM widened around either the raw value or the SI-converted value
        for candidate in (lit.si_value, lit.value):
            if lo <= candidate <= hi and hi > lo:
                out[sym] = [candidate, candidate]
                break
    return out


def _inline_static_algebraic_observables(
    expr: str,
    equations: Optional[Mapping[str, Any]],
) -> str:
    """ARCH-008: Inline algebraic observable definitions that do not depend on dynamics."""
    if not equations:
        return expr
    alg_defs: Dict[str, str] = {}
    for name, eq in equations.items():
        if isinstance(eq, Mapping):
            kind = str(eq.get("kind") or "")
            rhs = str(eq.get("expression") or "").strip()
        else:
            kind = str(getattr(eq, "kind", "") or "")
            rhs = str(getattr(eq, "expression", "") or "").strip()
        if kind == "ALGEBRAIC" and rhs:
            alg_defs[str(name)] = rhs
    if not alg_defs:
        return expr
    out = expr
    for _ in range(len(alg_defs) + 1):
        changed = False
        for name, rhs in sorted(alg_defs.items(), key=lambda kv: len(kv[0]), reverse=True):
            replaced = re.sub(rf"\b{re.escape(name)}\b", f"({rhs})", out)
            if replaced != out:
                out = replaced
                changed = True
        if not changed:
            break
    return out


def _is_symbolic_identity_tautology(expr: str) -> bool:
    """Detect algebraic/trigonometric identities (e.g. sin(x)**2 + cos(x)**2 <= 1.05) via SymPy trigsimp."""
    import ast as _ast
    import sympy as sp
    try:
        node = _ast.parse(expr, mode="eval").body
    except SyntaxError:
        return False
    if not isinstance(node, _ast.Compare) or len(node.ops) != 1:
        return False
    op = node.ops[0]
    try:
        lhs = sp.sympify(_ast.unparse(node.left))
        rhs = sp.sympify(_ast.unparse(node.comparators[0]))
        diff = sp.trigsimp(sp.simplify(lhs - rhs))
    except Exception:
        return False
    if not diff.is_number or not diff.is_real:
        return False
    val = float(diff.evalf())
    if not math.isfinite(val):
        return False
    if isinstance(op, _ast.LtE):
        return val <= 0.0
    if isinstance(op, _ast.Lt):
        return val < 0.0
    if isinstance(op, _ast.GtE):
        return val >= 0.0
    if isinstance(op, _ast.Gt):
        return val > 0.0
    if isinstance(op, _ast.Eq):
        return abs(val) <= 1e-12
    return False


def verify_claim_fidelity(
    ledger: ClaimLedger,
    obligations: Sequence[Mapping[str, Any]],
    environmental_assumptions: Sequence[str],
    domain: Mapping[str, Sequence[float]],
    *,
    equations: Optional[Mapping[str, Any]] = None,
) -> List[ClaimFidelityViolation]:
    """Verify obligations against the prompt's ClaimLedger (ARCH-008 / BUG-012).

    1. Inlines algebraic observable definitions (without dynamics) and checks
       symbolic/trig identities before Z3, rejecting tautologies as `TAUTOLOGICAL_TARGET`.
    2. Converts solver exceptions into `NONTRIVIALITY_UNDECIDED` (never silent pass).
    3. Checks that explicit numeric claim thresholds from the prompt are covered
       by at least one HYPOTHESIS_TARGET obligation.
    """
    from .meaningfulness import _and, solve_query

    violations: List[ClaimFidelityViolation] = []
    hyp_obs = [
        ob for ob in obligations
        if isinstance(ob, Mapping)
        and str(ob.get("obligation_class") or "HYPOTHESIS_TARGET") == "HYPOTHESIS_TARGET"
        and str(ob.get("expression") or "").strip()
    ]
    if not hyp_obs:
        violations.append(ClaimFidelityViolation(
            code="CLAIM_UNCOVERED",
            message="No HYPOTHESIS_TARGET obligation was emitted.",
        ))
        return violations

    box = {k: (float(v[0]), float(v[1])) for k, v in domain.items()}
    env_list = [str(e) for e in environmental_assumptions if str(e).strip()]
    for ob in hyp_obs:
        expr = str(ob["expression"]).strip()
        inlined_expr = _inline_static_algebraic_observables(expr, equations)
        if _is_symbolic_identity_tautology(inlined_expr):
            violations.append(ClaimFidelityViolation(
                code="TAUTOLOGICAL_TARGET",
                message=f"HYPOTHESIS_TARGET {expr!r} simplifies to a mathematical identity ({inlined_expr!r}).",
                detail={"obligation_id": ob.get("id"), "expression": expr, "inlined": inlined_expr},
            ))
            continue
        try:
            q_neg = solve_query(_and([*env_list, f"not ({expr})"]), box, witness=True, timeout_ms=800)
            if q_neg.status == "UNSAT":
                violations.append(ClaimFidelityViolation(
                    code="TAUTOLOGICAL_TARGET",
                    message=f"HYPOTHESIS_TARGET {expr!r} is already entailed by domain/environment alone (tautology).",
                    detail={"obligation_id": ob.get("id"), "expression": expr},
                ))
        except Exception as exc:
            violations.append(ClaimFidelityViolation(
                code="NONTRIVIALITY_UNDECIDED",
                message=f"Nontriviality check raised {type(exc).__name__}: {exc}",
                detail={"obligation_id": ob.get("id"), "expression": expr},
            ))

    # Check threshold coverage for explicit prompt claim clauses
    all_hyp_text = " ".join(str(ob.get("expression") or "") for ob in hyp_obs)
    hyp_nums = [float(x) for x in re.findall(_NUM_RE, all_hyp_text)]
    for clause in ledger.claim_clauses:
        matched = any(
            math.isclose(n, clause.threshold, rel_tol=1e-3, abs_tol=1e-6)
            for n in hyp_nums
        )
        if not matched:
            violations.append(ClaimFidelityViolation(
                code="CLAIM_UNCOVERED",
                message=f"Prompt claim threshold {clause.raw_text!r} ({clause.threshold}) is not represented in HYPOTHESIS_TARGET obligations.",
                detail={"raw_text": clause.raw_text, "threshold": clause.threshold},
            ))
    return violations


def reconcile_si_horizon(
    ledger: ClaimLedger,
    dt: float,
    steps: int,
    *,
    rate_constants_si: Sequence[float] = (),
    min_timescales: float = 10.0,
    max_steps: int = 20000,
) -> Tuple[float, int]:
    """Ensure SI simulation horizon covers explicit user duration or >= 10x slowest rate timescale (BUG-006)."""
    cur_dt = max(1e-12, float(dt))
    cur_steps = max(1, int(steps))
    cur_horizon = cur_dt * cur_steps

    target_horizon = cur_horizon
    if ledger.duration_si_s is not None and ledger.duration_si_s > 0:
        target_horizon = ledger.duration_si_s
    else:
        positive_rates = [abs(float(r)) for r in rate_constants_si if abs(float(r)) > 0]
        if positive_rates:
            slowest_tau = 1.0 / min(positive_rates)
            min_required = min_timescales * slowest_tau
            if target_horizon < min_required:
                target_horizon = min_required

    if math.isclose(target_horizon, cur_horizon, rel_tol=1e-3):
        return (cur_dt, cur_steps)

    # Preserve step count within safe bounds and adjust dt to hit target_horizon exactly
    new_steps = max(200, min(max_steps, cur_steps))
    new_dt = target_horizon / float(new_steps)
    return (new_dt, new_steps)

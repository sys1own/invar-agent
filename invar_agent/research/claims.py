"""Typed claim records, span verification, and sound SMT contradiction detection (UPG-002, UPG-003)."""
from __future__ import annotations

from enum import Enum
from fractions import Fraction
import re
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from invar_agent.ir.units import DIM_ZERO, UnitError, parse_unit


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ClaimRole(str, Enum):
    HYPOTHESIS = "HYPOTHESIS"
    OPERATING_DOMAIN = "OPERATING_DOMAIN"
    SIMULATION_SETTING = "SIMULATION_SETTING"
    REQUESTED_QUANTITY = "REQUESTED_QUANTITY"


class Relation(str, Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="
    IN = "in"
    NONE = "none"


class SpanRef(_Strict):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=600)


class ClaimRecord(_Strict):
    id: str = Field(pattern=r"^c[0-9]{1,3}$")
    role: ClaimRole
    modality: Literal["REQUIREMENT", "CLAIM"] = "CLAIM"
    subject: str = Field(min_length=1, max_length=120)
    symbol: str | None = None
    relation: Relation = Relation.NONE
    value: str | None = None
    value_hi: str | None = None
    unit: str | None = None
    quantifier: Literal["FORALL", "EXISTS", "BOUNDARY", "POINT"] = "POINT"
    time_scope: Literal["NONE", "ALWAYS", "EVENTUALLY", "AT_END", "WINDOW"] = "NONE"
    window: tuple[str, str] | None = None
    entity: str = "system"
    scope_id: str = "s0"
    condition: str | None = None
    span: SpanRef


class ClaimSet(_Strict):
    question_kind: Literal["POLAR", "EXISTENTIAL", "UNIVERSAL", "COMPUTE", "DESIGN", "MIXED"] = "MIXED"
    records: list[ClaimRecord] = Field(default_factory=list, max_length=64)


class ObligationClass(str, Enum):
    NONTRIVIAL = "NONTRIVIAL"
    DEFINITIONAL = "DEFINITIONAL"
    DOMAIN_ENTAILED = "DOMAIN_ENTAILED"
    UNDECIDED = "UNDECIDED"


class ContradictionItem(_Strict):
    claim_id: str
    quote: str
    start: int
    end: int
    quantity: str
    relation: str
    value: str
    unit: str | None = None
    si_value: str
    si_unit: str


def verify_claim_spans(prompt: str, claim_set: ClaimSet) -> tuple[ClaimSet, list[str]]:
    """Verify each record's SpanRef against `prompt`; relocate unique matches, drop absent/ambiguous."""
    verified: list[ClaimRecord] = []
    ledger_notes: list[str] = []
    text = str(prompt or "")
    for rec in claim_set.records:
        q = rec.span.quote
        if 0 <= rec.span.start < rec.span.end <= len(text) and text[rec.span.start:rec.span.end] == q:
            verified.append(rec)
            continue
        first = text.find(q)
        if first != -1 and text.find(q, first + 1) == -1:
            fixed_span = SpanRef(start=first, end=first + len(q), quote=q)
            verified.append(rec.model_copy(update={"span": fixed_span}))
            continue
        ledger_notes.append(f"CLAIM_SPAN_REJECTED:{rec.id}")
    return ClaimSet(question_kind=claim_set.question_kind, records=verified), ledger_notes


def _parse_rational(raw: str | None) -> Fraction | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        try:
            return Fraction(float(s))
        except (ValueError, OverflowError):
            return None


def _norm_subject(rec: ClaimRecord) -> str:
    raw = (rec.symbol or rec.subject or "").strip()
    return re.sub(r"\s+", " ", raw).lower()


def _format_fraction(f: Fraction) -> str:
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def detect_contradiction(
    claims: ClaimSet,
    prompt: str | None = None,
) -> tuple[list[ContradictionItem] | None, list[str]]:
    """Sound, unit-aware, scope-grouped SMT contradiction detection (UPG-003).

    Returns (contradiction_items_or_None, advisories).
    """
    advisories: list[str] = []
    working = claims
    if prompt is not None:
        working, span_notes = verify_claim_spans(prompt, claims)
        advisories.extend(span_notes)

    eligible_relations = {Relation.LT, Relation.LE, Relation.GT, Relation.GE, Relation.EQ, Relation.IN}
    groups: Dict[Tuple, List[ClaimRecord]] = {}
    for rec in working.records:
        if rec.modality != "REQUIREMENT":
            continue
        if rec.relation not in eligible_relations:
            continue
        if _parse_rational(rec.value) is None:
            continue
        if rec.relation == Relation.IN and _parse_rational(rec.value_hi) is None:
            continue
        key = (
            (rec.entity or "system").strip().lower(),
            _norm_subject(rec),
            (rec.scope_id or "s0").strip().lower(),
            (rec.condition or "").strip().lower(),
            rec.quantifier,
            rec.time_scope,
            rec.window,
        )
        groups.setdefault(key, []).append(rec)

    candidates = {k: recs for k, recs in groups.items() if len(recs) >= 2}
    if not candidates:
        return None, advisories

    from invar_agent.runtime.isolation import run_on_z3_thread
    import z3

    for key, recs in candidates.items():
        # Require pairwise distinct spans (dedupe duplicate spans)
        deduped: list[ClaimRecord] = []
        seen_spans: set[tuple[int, int]] = set()
        for r in recs:
            sp = (r.span.start, r.span.end)
            if sp in seen_spans:
                continue
            seen_spans.add(sp)
            deduped.append(r)
        if len(deduped) < 2:
            continue

        # Parse units and verify shared dimension
        parsed_entries: list[tuple[ClaimRecord, Fraction, Fraction | None, tuple, str]] = []
        dims_seen: set[tuple] = set()
        unit_error = False
        for r in deduped:
            val_lo = _parse_rational(r.value)
            val_hi = _parse_rational(r.value_hi) if r.relation == Relation.IN else None
            assert val_lo is not None
            if r.unit and r.unit.strip():
                try:
                    up = parse_unit(r.unit)
                except UnitError as exc:
                    advisories.append(f"CONTRADICTION_UNIT_UNPARSED:{r.id}:{exc.code}")
                    unit_error = True
                    break
                si_lo = up.to_si_rational(val_lo)
                si_hi = up.to_si_rational(val_hi) if val_hi is not None else None
                dims_seen.add(up.dim)
                si_u = up.canonical_si_unit()
            else:
                si_lo = val_lo
                si_hi = val_hi
                dims_seen.add(DIM_ZERO)
                si_u = "1"
            parsed_entries.append((r, si_lo, si_hi, up.dim if (r.unit and r.unit.strip()) else DIM_ZERO, si_u))

        if unit_error:
            continue
        if len(dims_seen) > 1:
            advisories.append(f"CONTRADICTION_DIMENSION_MISMATCH:{key[1]}")
            continue

        def _check_group(entries=parsed_entries):
            solver = z3.Solver()
            solver.set("smt.core.minimize", True)
            solver.set(timeout=300)
            var = z3.Real("x")
            label_map: dict[str, tuple[ClaimRecord, Fraction, Fraction | None, str]] = {}
            for idx, (rec, si_lo, si_hi, _dim, si_u) in enumerate(entries):
                lo_rv = z3.Q(si_lo.numerator, si_lo.denominator)
                if rec.relation == Relation.LT:
                    expr = var < lo_rv
                elif rec.relation == Relation.LE:
                    expr = var <= lo_rv
                elif rec.relation == Relation.GT:
                    expr = var > lo_rv
                elif rec.relation == Relation.GE:
                    expr = var >= lo_rv
                elif rec.relation == Relation.EQ:
                    expr = var == lo_rv
                elif rec.relation == Relation.IN and si_hi is not None:
                    hi_rv = z3.Q(si_hi.numerator, si_hi.denominator)
                    expr = z3.And(var >= lo_rv, var <= hi_rv)
                else:
                    continue
                lbl = z3.Bool(f"$c_{idx}")
                label_map[str(lbl)] = (rec, si_lo, si_hi, si_u)
                solver.assert_and_track(expr, lbl)
            status = solver.check()
            if status == z3.unsat:
                core_names = [str(b) for b in solver.unsat_core() if str(b) in label_map]
                if len(core_names) >= 2:
                    items: list[ContradictionItem] = []
                    for cname in sorted(core_names):
                        rec, si_lo, si_hi, si_u = label_map[cname]
                        si_val_str = (
                            f"[{_format_fraction(si_lo)}, {_format_fraction(si_hi)}]"
                            if rec.relation == Relation.IN and si_hi is not None
                            else _format_fraction(si_lo)
                        )
                        items.append(
                            ContradictionItem(
                                claim_id=rec.id,
                                quote=rec.span.quote,
                                start=rec.span.start,
                                end=rec.span.end,
                                quantity=rec.symbol or rec.subject,
                                relation=rec.relation.value,
                                value=str(rec.value),
                                unit=rec.unit,
                                si_value=si_val_str,
                                si_unit=si_u,
                            )
                        )
                    return ("unsat", items)
            if status == z3.unknown:
                return ("unknown", None)
            return ("sat", None)

        verdict_status, items = run_on_z3_thread(_check_group)
        if verdict_status == "unsat" and items:
            return items, advisories
        if verdict_status == "unknown":
            advisories.append(f"CONTRADICTION_SOLVER_UNKNOWN:{key[1]}")

    return None, advisories


# ---------------------------------------------------------------------------
# UPG-002 Full (Sprint 2): Quantitative clause scanner, role admissibility,
# coverage & strength verification, obligation classification, COMPUTED_NO_CLAIM
# ---------------------------------------------------------------------------

import ast as _ast
import copy as _copy
import hmac as _hmac
import json as _json
import re as _re
from typing import Any as _Any, Mapping as _Mapping, Sequence as _Sequence

_QUANT_PHRASE_RE = _re.compile(
    r"(?:<=|>=|<|>|==|≤|≥|"
    r"\b(?:never\s+exceeds?|exceeds?|at\s+least|at\s+most|no\s+more\s+than|"
    r"less\s+than|greater\s+than|below|above|under|over)\b)",
    _re.IGNORECASE,
)
_NUM_TOKEN_RE = _re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


class QuantitativeClause(_Strict):
    start: int
    end: int
    comparator: str
    number_text: str
    quote: str


def scan_quantitative_clauses(prompt: str) -> list[QuantitativeClause]:
    """UPG-002 step 3: Scan prompt for comparator tokens/phrases within 40 chars of a number.

    The left side may be any arbitrary expression (e.g. `V_c^2 / (V_c^2 + 1.0) <= 1.05`
    or `never exceeds 1.25 V`). Used strictly for coverage and repair, never to block prose.
    """
    if not prompt:
        return []
    clauses: list[QuantitativeClause] = []
    for m_cmp in _QUANT_PHRASE_RE.finditer(prompt):
        win_start = max(0, m_cmp.start() - 40)
        win_end = min(len(prompt), m_cmp.end() + 40)
        after = prompt[m_cmp.end():win_end]
        m_num = _NUM_TOKEN_RE.search(after)
        if m_num is not None:
            num_end = m_cmp.end() + m_num.end()
            quote_start = max(0, m_cmp.start() - 24)
            quote_end = min(len(prompt), num_end + 12)
            clauses.append(
                QuantitativeClause(
                    start=m_cmp.start(),
                    end=num_end,
                    comparator=m_cmp.group(0),
                    number_text=m_num.group(0),
                    quote=prompt[quote_start:quote_end].strip(),
                )
            )
    return clauses


def enforce_role_admissibility(
    claims: ClaimSet,
    symbol_roles: _Mapping[str, str],
) -> tuple[ClaimSet, list[dict[str, _Any]]]:
    """UPG-002 / §2.2 anti-gaming: Out/In/Cfg role admissibility.

    Any ClaimRecord with a comparison (`relation != NONE`) whose bound symbol is an
    Output (`STATE`, `DYNAMIC_STATE`, `DERIVED_OBSERVABLE`, `OBSERVABLE`) cannot be
    downgraded to `OPERATING_DOMAIN` or `SIMULATION_SETTING`; escalate it to `HYPOTHESIS`
    and record `ROLE_ESCALATED`.
    """
    out_roles = {"STATE", "DYNAMIC_STATE", "DERIVED_OBSERVABLE", "OBSERVABLE"}
    updated: list[ClaimRecord] = []
    ledger: list[dict[str, _Any]] = []
    for rec in claims.records:
        sym = rec.symbol or rec.subject
        srole = symbol_roles.get(sym, "")
        if (
            rec.relation != Relation.NONE
            and srole in out_roles
            and rec.role in (ClaimRole.OPERATING_DOMAIN, ClaimRole.SIMULATION_SETTING)
        ):
            updated.append(rec.model_copy(update={"role": ClaimRole.HYPOTHESIS}))
            ledger.append({
                "code": "ROLE_ESCALATED",
                "claim_id": rec.id,
                "symbol": sym,
                "from_role": rec.role.value,
                "to_role": ClaimRole.HYPOTHESIS.value,
            })
        else:
            updated.append(rec)
    return claims.model_copy(update={"records": updated}), ledger


def _parse_obligation_bound(expr: str) -> Optional[tuple[str, str, Fraction]]:
    """Extract simple `(symbol, op_str, exact_rational)` from a comparison obligation."""
    try:
        tree = _ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(tree, _ast.Compare) and len(tree.ops) == 1 and len(tree.comparators) == 1):
        return None
    op_map = {
        _ast.Lt: "<", _ast.LtE: "<=", _ast.Gt: ">", _ast.GtE: ">=", _ast.Eq: "==", _ast.NotEq: "!=",
    }
    flip_map = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
    op_s = op_map.get(type(tree.ops[0]))
    if not op_s:
        return None

    def _const(n: _ast.AST) -> Optional[Fraction]:
        if isinstance(n, _ast.Constant) and type(n.value) in (int, float):
            return Fraction(str(n.value))
        if isinstance(n, _ast.UnaryOp) and isinstance(n.op, _ast.USub) and isinstance(n.operand, _ast.Constant) and type(n.operand.value) in (int, float):
            return -Fraction(str(n.operand.value))
        return None

    if isinstance(tree.left, _ast.Name) and _const(tree.comparators[0]) is not None:
        return (tree.left.id, op_s, _const(tree.comparators[0]))  # type: ignore[return-value]
    if isinstance(tree.comparators[0], _ast.Name) and _const(tree.left) is not None:
        return (tree.comparators[0].id, flip_map[op_s], _const(tree.left))  # type: ignore[return-value]
    return None


def check_claim_coverage_and_strength(
    prompt: str,
    claims: ClaimSet,
    obligations: _Sequence[_Mapping[str, _Any]],
    *,
    temporal_reductions: _Mapping[str, str] | None = None,
) -> tuple[list[dict[str, _Any]], list[dict[str, _Any]]]:
    """UPG-002 anti-gaming: check clause coverage, exact rational `CLAIM_WEAKENED`,
    and `CLAIM_TEMPORAL_SCOPE_UNBOUND`. Returns `(diagnostics, synthesized_obligations)`.
    """
    diags: list[dict[str, _Any]] = []
    synthesized: list[dict[str, _Any]] = []
    t_reds = temporal_reductions or {}

    # 1. Clause span coverage
    clauses = scan_quantitative_clauses(prompt)
    for cl in clauses:
        covered = any(
            not (r.span.end <= cl.start or r.span.start >= cl.end)
            for r in claims.records
        )
        if not covered:
            diags.append({
                "code": "CLAIM_UNCOVERED",
                "message": f"Quantitative clause {cl.quote!r} is not covered by any ClaimRecord span.",
                "span": {"start": cl.start, "end": cl.end},
            })

    # 2. Per-HYPOTHESIS obligation matching + exact rational CLAIM_WEAKENED + temporal scope check
    for rec in claims.records:
        if rec.role != ClaimRole.HYPOTHESIS:
            continue
        matched_obs = [
            ob for ob in obligations
            if rec.id in (ob.get("claim_ids") or ())
        ]
        if not matched_obs:
            # Attempt deterministic synthesis when fully typed
            if rec.symbol and rec.relation not in (Relation.NONE, Relation.IN) and rec.value is not None:
                val_si = Fraction(rec.value)
                if rec.unit and rec.unit.strip():
                    try:
                        val_si = parse_unit(rec.unit).to_si_rational(val_si)
                    except UnitError:
                        pass
                expr_s = f"{rec.symbol} {rec.relation.value} {float(val_si)!r}"
                synthesized.append({
                    "id": f"synth_{rec.id}",
                    "obligation_class": "HYPOTHESIS_TARGET",
                    "expression": expr_s,
                    "claim_ids": [rec.id],
                })
            else:
                diags.append({
                    "code": "CLAIM_UNFORMALIZED",
                    "claim_id": rec.id,
                    "message": f"HYPOTHESIS claim {rec.id!r} ({rec.span.quote!r}) has no obligation and is not fully typed.",
                })
            continue

        # Temporal scope binding check
        if rec.time_scope in ("ALWAYS", "EVENTUALLY"):
            bound_sym = rec.symbol or ""
            if bound_sym and bound_sym not in t_reds:
                # Check if any matched obligation references a symbol with a trajectory reduction
                has_red = any(
                    any(ts in str(ob.get("expression", "")) for ts in t_reds)
                    for ob in matched_obs
                )
                if not has_red:
                    diags.append({
                        "code": "CLAIM_TEMPORAL_SCOPE_UNBOUND",
                        "claim_id": rec.id,
                        "message": f"Claim {rec.id!r} has time_scope={rec.time_scope} without a trajectory reduction observable.",
                    })

        # Exact rational CLAIM_WEAKENED check
        if rec.value is not None and rec.relation in (Relation.LT, Relation.LE, Relation.GT, Relation.GE):
            claimed_si = Fraction(rec.value)
            if rec.unit and rec.unit.strip():
                try:
                    claimed_si = parse_unit(rec.unit).to_si_rational(claimed_si)
                except UnitError:
                    pass
            for ob in matched_obs:
                parsed = _parse_obligation_bound(str(ob.get("expression", "")))
                if parsed is None:
                    continue
                _sym, ob_op, ob_val = parsed
                weakened = False
                if rec.relation in (Relation.LE, Relation.LT) and ob_op in ("<=", "<"):
                    if ob_val > claimed_si or (rec.relation == Relation.LT and ob_op == "<=" and ob_val == claimed_si):
                        weakened = True
                elif rec.relation in (Relation.GE, Relation.GT) and ob_op in (">=", ">"):
                    if ob_val < claimed_si or (rec.relation == Relation.GT and ob_op == ">=" and ob_val == claimed_si):
                        weakened = True
                elif (rec.relation in (Relation.LE, Relation.LT) and ob_op in (">=", ">")) or (
                    rec.relation in (Relation.GE, Relation.GT) and ob_op in ("<=", "<")
                ):
                    weakened = True
                if weakened:
                    diags.append({
                        "code": "CLAIM_WEAKENED",
                        "claim_id": rec.id,
                        "claimed_si": f"{claimed_si.numerator}/{claimed_si.denominator}",
                        "obligation_si": f"{ob_val.numerator}/{ob_val.denominator}",
                        "message": f"Obligation {ob.get('expression')!r} weakens claimed bound {rec.relation.value} {claimed_si}.",
                    })
    return diags, synthesized


def _inline_ast_fixed_point(
    expr_str: str,
    algebraic_defs: _Mapping[str, str],
    max_iters: int = 16,
) -> Optional[str]:
    """Inline algebraic observables & lets to a fixed point on the AST; return None on cycle."""
    try:
        tree = _ast.parse(expr_str, mode="eval").body
        def_asts = {k: _ast.parse(v, mode="eval").body for k, v in algebraic_defs.items()}
    except SyntaxError:
        return expr_str

    for _ in range(max_iters):
        changed = False

        class _Inline(_ast.NodeTransformer):
            def visit_Name(self, node: _ast.Name) -> _ast.AST:
                nonlocal changed
                if node.id in def_asts:
                    changed = True
                    return _copy.deepcopy(def_asts[node.id])
                return node

        tree = _Inline().visit(tree)
        _ast.fix_missing_locations(tree)
        if not changed:
            return _ast.unparse(tree)
    return None  # Cycle detected


def classify_obligations(
    obligations: _Sequence[_Mapping[str, _Any]],
    algebraic_defs: _Mapping[str, str],
    env_assumptions: _Sequence[str],
    claim_box: _Mapping[str, tuple[float, float]],
    *,
    conservation_pools: set[str] | None = None,
) -> dict[str, _Any]:
    """UPG-002 obligation classification & auxiliary pruning before preflight/ingest.

    Returns:
      {
        "hypothesis_obligations": [...],
        "conservation_guards": [...],
        "diagnostic_assertions": [...],
        "pruned_auxiliary": [...],
        "classifications": {ob_id: ObligationClass.value},
        "abort_tautological": bool,
      }
    """
    import sympy as sp
    from invar_agent.research import meaningfulness as m

    pools = conservation_pools or set()
    hyp_obs: list[dict[str, _Any]] = []
    guards: list[dict[str, _Any]] = []
    diag_asserts: list[dict[str, _Any]] = []
    pruned: list[dict[str, _Any]] = []
    classes: dict[str, str] = {}

    env_expr = m._and(env_assumptions) if env_assumptions else "True"

    for idx, raw_ob in enumerate(obligations):
        ob = dict(raw_ob)
        ob_id = str(ob.get("id") or f"ob_{idx}")
        expr_str = str(ob.get("expression") or "").strip()
        claim_ids = list(ob.get("claim_ids") or [])

        inlined = _inline_ast_fixed_point(expr_str, algebraic_defs)
        if inlined is None:
            cls = ObligationClass.UNDECIDED
        else:
            # (a) Exact symbolic identity check (cancel / trigsimp)
            is_identity = False
            try:
                node = m._parse(inlined)
                if isinstance(node, _ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], _ast.Eq):
                    lhs_s = sp.sympify(_ast.unparse(node.left), rational=True)
                    rhs_s = sp.sympify(_ast.unparse(node.comparators[0]), rational=True)
                    diff = sp.cancel(lhs_s - rhs_s)
                    if diff == 0 or sp.trigsimp(diff) == 0:
                        is_identity = True
            except Exception:
                is_identity = False

            if is_identity:
                cls = ObligationClass.DEFINITIONAL
            else:
                # (b)/(c)/(d) Z3 check on inlined expression over claim_box
                try:
                    free_vars = m._variables(m._parse(inlined))
                    if env_assumptions:
                        free_vars |= m._variables(m._parse(env_expr))
                    box = dict(claim_box)
                    for fv in free_vars:
                        if fv not in box:
                            box[fv] = (-1e6, 1e6)
                    q_expr = m._and([env_expr, f"not ({inlined})"]) if env_assumptions else f"not ({inlined})"
                    res = m.solve_query(q_expr, box, witness=False, timeout_ms=500)
                    if res.status == "UNSAT":
                        cls = ObligationClass.DOMAIN_ENTAILED
                    elif res.status == "SAT":
                        cls = ObligationClass.NONTRIVIAL
                    else:
                        cls = ObligationClass.UNDECIDED
                except Exception:
                    cls = ObligationClass.UNDECIDED

        classes[ob_id] = cls.value
        if not claim_ids:
            # Auxiliary obligation
            if cls in (ObligationClass.DEFINITIONAL, ObligationClass.DOMAIN_ENTAILED):
                pruned.append({
                    "code": "PRUNED_AUXILIARY_OBLIGATION",
                    "id": ob_id,
                    "expression": expr_str,
                    "classification": cls.value,
                })
            elif ob.get("pool_id") in pools or ob.get("obligation_class") == "CONSERVATION_GUARD":
                guards.append(ob)
            else:
                diag_asserts.append(ob)
        else:
            hyp_obs.append(ob)

    # Abort TAUTOLOGICAL_TARGET iff EVERY hypothesis obligation is DEFINITIONAL or DOMAIN_ENTAILED
    abort_taut = bool(hyp_obs) and all(
        classes[str(ob.get("id") or f"ob_{i}")] in (ObligationClass.DEFINITIONAL.value, ObligationClass.DOMAIN_ENTAILED.value)
        for i, ob in enumerate(hyp_obs)
        if str(ob.get("id") or f"ob_{i}") in classes
    )
    return {
        "hypothesis_obligations": hyp_obs,
        "conservation_guards": guards,
        "diagnostic_assertions": diag_asserts,
        "pruned_auxiliary": pruned,
        "classifications": classes,
        "abort_tautological": abort_taut,
    }


# ---------------------------------------------------------------------------
# COMPUTED_NO_CLAIM (deliberately NOT a member of EpistemicVerdict)
# ---------------------------------------------------------------------------

class ComputedVerdict(str, Enum):
    COMPUTED_NO_CLAIM = "COMPUTED_NO_CLAIM"


class ComputedQuantity(_Strict):
    name: str
    expression: str
    unit: str
    value: float
    lo: float
    hi: float
    epsilon_total: float = Field(ge=0.0)
    method: Literal[
        "ALGEBRAIC_INTERVAL",
        "ODE_GCI",
        "TRAJECTORY_FUNCTIONAL",
        "FIXED_POINT_KRAWCZYK",
        "MC_EMPIRICAL_BERNSTEIN",
    ]
    claim_ids: tuple[str, ...] = ()


class ComputationCertificate(_Strict):
    schema_version: Literal["computation_no_claim.v1"] = "computation_no_claim.v1"
    run_id: str
    epistemic_verdict: ComputedVerdict = ComputedVerdict.COMPUTED_NO_CLAIM
    ihcg_digest: str
    uast_digest: str
    purity_audit: _Any
    fidelity_audits: dict[str, _Any] = Field(default_factory=dict)
    quantities: tuple[ComputedQuantity, ...]
    claim_set_digest: str
    assumptions_ledger: tuple[_Any, ...] = ()
    blake3_receipt: str = Field(pattern=r"^[0-9a-f]{64}$")


def seal_computation_certificate(cert: ComputationCertificate) -> ComputationCertificate:
    from invar_agent import _native

    unsigned = cert.model_copy(update={"blake3_receipt": "0" * 64})
    receipt = _native.py_compute_certificate_receipt(
        _json.dumps(unsigned.model_dump(mode="json"), allow_nan=False)
    )
    return unsigned.model_copy(update={"blake3_receipt": receipt})


def verify_computation_certificate(cert: ComputationCertificate) -> bool:
    from invar_agent import _native

    unsigned = cert.model_copy(update={"blake3_receipt": "0" * 64})
    expected = _native.py_compute_certificate_receipt(
        _json.dumps(unsigned.model_dump(mode="json"), allow_nan=False)
    )
    return _hmac.compare_digest(cert.blake3_receipt, expected)

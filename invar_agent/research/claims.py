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

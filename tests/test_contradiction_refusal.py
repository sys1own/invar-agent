"""UPG-003 + UPG-002 Part 1 (BUG-003, NF-10) tests: sound contradiction refusal & no state pinning."""
from __future__ import annotations

from invar_agent.ir.ihcg_bridge import seal_sciir_ihcg
from invar_agent.ir.lower_uast import sciir_to_research_blueprint
from invar_agent.ir.schema import (
    Archetype,
    FrozenSymbolTable,
    ModelBodyDraft,
    OdePdeConfig,
    RawEquationDraft,
    ScientificDomain,
    SiDimension,
    SlotRole,
    SymbolDecl,
)
from invar_agent.ir.typechecker import scope_check_and_lower
from invar_agent.research.claim_ledger import extract_claim_ledger
from invar_agent.research.claims import (
    ClaimRecord,
    ClaimRole,
    ClaimSet,
    Relation,
    SpanRef,
    detect_contradiction,
)
from invar_agent.research.feasibility import _heuristic_verdict


def _rec(prompt: str, cid: str, quote: str, subject: str, relation: Relation, value: str, *,
         unit: str | None = None, entity: str = "system", scope_id: str = "s0",
         condition: str | None = None, modality: str = "REQUIREMENT") -> ClaimRecord:
    idx = prompt.index(quote)
    return ClaimRecord(
        id=cid,
        role=ClaimRole.OPERATING_DOMAIN,
        modality=modality,
        subject=subject,
        symbol=subject,
        relation=relation,
        value=value,
        unit=unit,
        entity=entity,
        scope_id=scope_id,
        condition=condition,
        span=SpanRef(start=idx, end=idx + len(quote), quote=quote),
    )


def test_contradiction_corpus_all_7_audit_cases():
    # 1. genuine contradiction -> REFUSE
    p1 = "Design a vessel with P >= 25.0 MPa and P <= 12.0 MPa simultaneously."
    cs1 = ClaimSet(records=[
        _rec(p1, "c1", "P >= 25.0 MPa", "P", Relation.GE, "25.0", unit="MPa"),
        _rec(p1, "c2", "P <= 12.0 MPa", "P", Relation.LE, "12.0", unit="MPa"),
    ])
    items1, _ = detect_contradiction(cs1, prompt=p1)
    assert items1 is not None and len(items1) == 2

    # 2. units_infeasible (25000 kPa = 25 MPa > 12 MPa) -> REFUSE
    p2 = "Require P >= 25000 kPa and P <= 12 MPa."
    cs2 = ClaimSet(records=[
        _rec(p2, "c1", "P >= 25000 kPa", "P", Relation.GE, "25000", unit="kPa"),
        _rec(p2, "c2", "P <= 12 MPa", "P", Relation.LE, "12", unit="MPa"),
    ])
    items2, _ = detect_contradiction(cs2, prompt=p2)
    assert items2 is not None and len(items2) == 2

    # 3. units feasible (2000 kPa = 2 MPa <= 30 MPa) -> ACCEPT
    p3 = "Operate with P >= 2000 kPa and P <= 30 MPa."
    cs3 = ClaimSet(records=[
        _rec(p3, "c1", "P >= 2000 kPa", "P", Relation.GE, "2000", unit="kPa"),
        _rec(p3, "c2", "P <= 30 MPa", "P", Relation.LE, "30", unit="MPa"),
    ])
    items3, _ = detect_contradiction(cs3, prompt=p3)
    assert items3 is None

    # 4. piecewise forcing -> ACCEPT (even with shared scope_id!)
    p4 = "Forcing u = 1 for t < 10 and u = 0.5 for t > 40."
    cs4 = ClaimSet(records=[
        _rec(p4, "c1", "u = 1", "u", Relation.EQ, "1", scope_id="s0", condition="t < 10"),
        _rec(p4, "c2", "u = 0.5", "u", Relation.EQ, "0.5", scope_id="s0", condition="t > 40"),
    ])
    items4, _ = detect_contradiction(cs4, prompt=p4)
    assert items4 is None

    # 5. regimes (underdamped vs overdamped) -> ACCEPT (even with shared scope_id!)
    p5 = "Compare underdamped zeta < 1.0 vs overdamped zeta > 1.0."
    cs5 = ClaimSet(records=[
        _rec(p5, "c1", "zeta < 1.0", "zeta", Relation.LT, "1.0", scope_id="s0", condition="underdamped"),
        _rec(p5, "c2", "zeta > 1.0", "zeta", Relation.GT, "1.0", scope_id="s0", condition="overdamped"),
    ])
    items5, _ = detect_contradiction(cs5, prompt=p5)
    assert items5 is None

    # 6. distinct entities (reactor A vs reactor B) -> ACCEPT (even with shared scope_id!)
    p6 = "Reactor A has T > 350 K while Reactor B has T < 300 K."
    cs6 = ClaimSet(records=[
        _rec(p6, "c1", "T > 350 K", "T", Relation.GT, "350", unit="K", entity="reactor_A", scope_id="s0"),
        _rec(p6, "c2", "T < 300 K", "T", Relation.LT, "300", unit="K", entity="reactor_B", scope_id="s0"),
    ])
    items6, _ = detect_contradiction(cs6, prompt=p6)
    assert items6 is None

    # 7. control vs treatment -> ACCEPT
    p7 = "Active case has k > 0.05 while control has k < 0.01."
    cs7 = ClaimSet(records=[
        _rec(p7, "c1", "k > 0.05", "k", Relation.GT, "0.05", entity="active", scope_id="s0"),
        _rec(p7, "c2", "k < 0.01", "k", Relation.LT, "0.01", entity="control", scope_id="s0"),
    ])
    items7, _ = detect_contradiction(cs7, prompt=p7)
    assert items7 is None


def test_heuristic_no_longer_regex_refuses_legitimate_prompts():
    for prompt in [
        "Set u = 1 for t < 10 and u = 0.5 for t > 40.",
        "Compare underdamped zeta < 1.0 vs overdamped zeta > 1.0.",
        "Operate at P >= 2000 kPa and P <= 30 MPa.",
        "Reactor A has T > 350 K and Reactor B has T < 300 K.",
        "Treatment has k > 0.05 and control has k < 0.01.",
    ]:
        h = _heuristic_verdict(prompt)
        assert h.verdict.value != "REFUSED_IMPOSSIBLE", f"False refusal on: {prompt}"


def test_nf10_no_state_pinning_and_operator_guarded_literals():
    # Operator-adjacent equation fragments must not pin literals
    ledger = extract_claim_ledger("With C = 3/(32*pi^2) and (u_c = 1.0 - x), solve ODE.")
    pinned_names = {lit.symbol for lit in ledger.pinned_literals}
    assert "C" not in pinned_names
    assert "u_c" not in pinned_names

    # Even if a state name appears in pinned_literals, STATE must NOT get a point domain [v0, v0]
    dimless = SiDimension.dimensionless()
    table = FrozenSymbolTable(
        ir_id="no_state_pin",
        archetype=Archetype.ODE_PDE,
        domain=ScientificDomain.PHYSICS_THERMO,
        symbols=[
            SymbolDecl(name="u_c", display_symbol="u_c", role=SlotRole.STATE, slot_index=0, dimension=dimless, initial_or_default=1.0),
        ],
    )
    body = ModelBodyDraft(
        dynamics_or_residuals=[RawEquationDraft(target="u_c", expr_text="-u_c")],
        config=OdePdeConfig(t_span=(0.0, 5.0), sample_points=50),
    )
    outcome = scope_check_and_lower(table, body)
    assert outcome.ok and outcome.sci_ir is not None
    bp = sciir_to_research_blueprint(seal_sciir_ihcg(outcome.sci_ir), user_prompt="Start at u_c = 1.0 and evolve.")
    lo, hi = bp["ihcg_evidence"]["domain"]["u_c"]
    assert hi > lo, f"STATE u_c was pinned to point domain [{lo}, {hi}]"

"""NF-09: IhcgGraph.claim_bounds splitting Ω_claim vs Ω_cert with zero-drift serialization."""
from __future__ import annotations

import json

from invar_agent.research import meaningfulness as m


def _base_ihcg(claim_bounds=None):
    return m.IhcgGraph(
        slots=(
            m.SlotSpec(symbol="x", slot_idx=0, role="DYNAMIC_STATE", si_dim=(0, 0, 0, 0, 0, 0, 0)),
        ),
        equations={
            "x": m.EquationSpec(
                kind="ODE_RHS",
                state_symbol="x",
                expression="-0.5 * x",
                si_dim=(0, 0, 0, 0, 0, 0, 0),
            ),
        },
        environmental_assumptions=(),
        obligations=(
            m.Obligation(
                id="ob1",
                obligation_class="HYPOTHESIS_TARGET",
                expression="x <= 10.0",
            ),
        ),
        omega={"x": (0.0, 2.0)},  # Tight certified box Ω_cert where x <= 10 is trivially true
        claim_bounds=claim_bounds,
        quantifier_mode="FORALL",
        archetype="ODE_PDE",
    )


def test_nf09_claim_bounds_omitted_when_none_zero_drift():
    g = _base_ihcg(claim_bounds=None)
    raw = json.loads(g.model_dump_json())
    assert "claim_bounds" not in raw
    assert "forcing_schedule" not in raw
    assert "claimless" not in raw


def test_nf09_claim_box_prevents_false_tautology_when_omega_cert_is_tight():
    # Without claim_bounds, Ω_cert=[0, 2] makes `x <= 10.0` TAUTOLOGICAL
    g_no_cb = _base_ihcg(claim_bounds=None)
    audit_taut = m._classify(g_no_cb, ("x <= 10.0",), None)
    assert audit_taut.classification == "TAUTOLOGICAL"

    # With claim_bounds Ω_claim=[0, 20], `x <= 10.0` is CONSTRAINING!
    g_with_cb = _base_ihcg(claim_bounds={"x": (0.0, 20.0)})
    audit_con = m._classify(g_with_cb, ("x <= 10.0",), None)
    assert audit_con.classification == "CONSTRAINING"
    assert g_with_cb.omega == {"x": (0.0, 2.0)}  # Ω_cert untouched

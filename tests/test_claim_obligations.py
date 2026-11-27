"""UPG-002 Full (BUG-002, 012, 013): Typed claims, anti-gaming, obligation classification, COMPUTED_NO_CLAIM."""
from __future__ import annotations

import pytest

from invar_agent.research import meaningfulness as m
from invar_agent.research.claims import (
    ClaimRecord,
    ClaimRole,
    ClaimSet,
    ComputationCertificate,
    ComputedQuantity,
    ComputedVerdict,
    ObligationClass,
    Relation,
    SpanRef,
    check_claim_coverage_and_strength,
    classify_obligations,
    enforce_role_admissibility,
    scan_quantitative_clauses,
    seal_computation_certificate,
    verify_computation_certificate,
)
from invar_agent.research.epistemic import parse_run_certificate


def test_scan_quantitative_clauses_finds_arbitrary_lhs_and_phrases():
    prompt = "Verify V_c^2 / (V_c^2 + 1.0) <= 1.05 and ensure peak voltage never exceeds 1.25 V."
    clauses = scan_quantitative_clauses(prompt)
    assert len(clauses) == 2
    assert any(c.number_text == "1.05" for c in clauses)
    assert any(c.number_text == "1.25" for c in clauses)


def test_role_admissibility_escalates_output_comparisons_to_hypothesis():
    prompt = "Check that peak B_max <= 0.6 while rate k1 in [0.1, 0.5]."
    cs = ClaimSet(
        question_kind="UNIVERSAL",
        records=[
            ClaimRecord(
                id="c1",
                role=ClaimRole.OPERATING_DOMAIN,  # Attempted downgrade of output symbol!
                subject="B_max",
                symbol="B_max",
                relation=Relation.LE,
                value="0.6",
                span=SpanRef(start=11, end=27, quote="peak B_max <= 0.6"),
            ),
            ClaimRecord(
                id="c2",
                role=ClaimRole.OPERATING_DOMAIN,
                subject="k1",
                symbol="k1",
                relation=Relation.IN,
                value="0.1",
                value_hi="0.5",
                span=SpanRef(start=39, end=54, quote="k1 in [0.1, 0.5]"),
            ),
        ],
    )
    enforced, ledger = enforce_role_admissibility(cs, {"B_max": "DERIVED_OBSERVABLE", "k1": "CONTROL_PARAM"})
    assert enforced.records[0].role == ClaimRole.HYPOTHESIS
    assert enforced.records[1].role == ClaimRole.OPERATING_DOMAIN
    assert len(ledger) == 1 and ledger[0]["code"] == "ROLE_ESCALATED"


def test_anti_gaming_claim_weakened_and_temporal_scope_unbound():
    prompt = "Verify B_max <= 0.6 always."
    cs = ClaimSet(
        question_kind="UNIVERSAL",
        records=[
            ClaimRecord(
                id="c1",
                role=ClaimRole.HYPOTHESIS,
                subject="B_max",
                symbol="B_max",
                relation=Relation.LE,
                value="0.6",
                time_scope="ALWAYS",
                span=SpanRef(start=7, end=19, quote="B_max <= 0.6"),
            )
        ],
    )
    # Weakened bound (0.8 > 0.6) and missing trajectory reduction
    diags, _synth = check_claim_coverage_and_strength(
        prompt,
        cs,
        [{"id": "ob1", "expression": "B_max <= 0.8", "claim_ids": ["c1"]}],
        temporal_reductions={},
    )
    codes = {d["code"] for d in diags}
    assert "CLAIM_WEAKENED" in codes
    assert "CLAIM_TEMPORAL_SCOPE_UNBOUND" in codes


def test_classify_obligations_prunes_auxiliary_identities_and_keeps_nontrivial_hypothesis():
    obs = [
        # Auxiliary definitional identity (R8a style): T_K - T_C == 273.15
        {"id": "aux_id", "expression": "T_K - T_C == 273.15", "claim_ids": []},
        # Auxiliary domain-entailed bound
        {"id": "aux_dom", "expression": "a >= 0.0", "claim_ids": []},
        # Genuine hypothesis obligation (R9a style): C_p >= 0.65
        {"id": "hyp_1", "expression": "C_p >= 0.65", "claim_ids": ["c1"]},
    ]
    res = classify_obligations(
        obs,
        algebraic_defs={"T_K": "T_C + 273.15", "C_p": "4 * a * (1 - a)**2"},
        env_assumptions=(),
        claim_box={"T_C": (0.0, 100.0), "a": (0.0, 1.0), "C_p": (0.0, 1.0)},
    )
    assert res["classifications"]["aux_id"] == ObligationClass.DEFINITIONAL.value
    assert res["classifications"]["aux_dom"] == ObligationClass.DOMAIN_ENTAILED.value
    assert res["classifications"]["hyp_1"] == ObligationClass.NONTRIVIAL.value
    assert len(res["pruned_auxiliary"]) == 2
    assert len(res["hypothesis_obligations"]) == 1
    assert res["abort_tautological"] is False


def test_computed_no_claim_certificate_seal_parse_and_not_epistemic_verdict():
    assert "COMPUTED_NO_CLAIM" not in {v.value for v in m.EpistemicVerdict}
    cert = ComputationCertificate(
        run_id="run_compute_01",
        ihcg_digest="a" * 64,
        uast_digest="b" * 64,
        purity_audit={"passed": True, "projected_steps": 0, "stabilized_steps": 0},
        fidelity_audits={},
        quantities=(
            ComputedQuantity(
                name="V_final",
                expression="V_c",
                unit="V",
                value=2.555852,
                lo=2.5558,
                hi=2.5559,
                epsilon_total=1e-4,
                method="ODE_GCI",
                claim_ids=("c1",),
            ),
        ),
        claim_set_digest="c" * 64,
        blake3_receipt="0" * 64,
    )
    sealed = seal_computation_certificate(cert)
    assert verify_computation_certificate(sealed) is True
    parsed = parse_run_certificate(sealed.model_dump(mode="json"))
    assert isinstance(parsed, ComputationCertificate)
    assert parsed.epistemic_verdict == ComputedVerdict.COMPUTED_NO_CLAIM

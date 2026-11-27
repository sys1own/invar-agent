"""HOTFIX-A (NF-01) regression tests: trajectory reductions preserved end-to-end."""
from __future__ import annotations

import pytest

from invar_agent.ir.lower_uast import sciir_to_research_blueprint
from invar_agent.ir.schema import (
    Archetype,
    FrozenSymbolTable,
    IntervalBound,
    ModelBodyDraft,
    OdePdeConfig,
    RawEquationDraft,
    RawObligationDraft,
    ScientificDomain,
    SiDimension,
    SlotRole,
    SymbolDecl,
)
from invar_agent.ir.ihcg_bridge import seal_sciir_ihcg
from invar_agent.ir.typechecker import scope_check_and_lower
from invar_agent.research.blueprint_compiler import compile_and_preflight_blueprint
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.schemas.research import ResearchBlueprint


def _build_abc_chain_blueprint(quantifier: str, comparator: str, rhs: str) -> dict:
    """Build A -> B -> C chain (k1=1, k2 in [0.5, 1.5], t in [0, 10]) with B_max = TRAJECTORY_MAX(B)."""
    dimless = SiDimension.dimensionless()
    rate_dim = SiDimension(exponents=[(0, 1), (0, 1), (-1, 1), (0, 1), (0, 1), (0, 1), (0, 1)], unit_label="1/s")
    table = FrozenSymbolTable(
        ir_id="abc_chain_nf01",
        archetype=Archetype.ODE_PDE,
        domain=ScientificDomain.BIOCHEM_SYSTEMS,
        symbols=[
            SymbolDecl(
                name="A",
                display_symbol="A",
                role=SlotRole.STATE,
                slot_index=0,
                dimension=dimless,
                initial_or_default=1.0,
                domain=IntervalBound(lo=-0.05, hi=1.1),
            ),
            SymbolDecl(
                name="B",
                display_symbol="B",
                role=SlotRole.STATE,
                slot_index=1,
                dimension=dimless,
                initial_or_default=0.0,
                domain=IntervalBound(lo=-0.05, hi=1.1),
            ),
            SymbolDecl(
                name="C",
                display_symbol="C",
                role=SlotRole.STATE,
                slot_index=2,
                dimension=dimless,
                initial_or_default=0.0,
                domain=IntervalBound(lo=-0.05, hi=1.1),
            ),
            SymbolDecl(
                name="k1",
                display_symbol="k_1",
                role=SlotRole.CONSTANT,
                slot_index=3,
                dimension=rate_dim,
                initial_or_default=1.0,
                domain=IntervalBound(lo=1.0, hi=1.0),
            ),
            SymbolDecl(
                name="k2",
                display_symbol="k_2",
                role=SlotRole.CONTROL_PARAM,
                slot_index=4,
                dimension=rate_dim,
                initial_or_default=1.0,
                domain=IntervalBound(lo=0.5, hi=1.5),
            ),
            SymbolDecl(
                name="B_max",
                display_symbol="B_{max}",
                role=SlotRole.DERIVED_OBSERVABLE,
                slot_index=5,
                dimension=dimless,
                initial_or_default=0.0,
            ),
        ],
    )
    body = ModelBodyDraft(
        dynamics_or_residuals=[
            RawEquationDraft(target="A", expr_text="-k1 * A"),
            RawEquationDraft(target="B", expr_text="k1 * A - k2 * B"),
            RawEquationDraft(target="C", expr_text="k2 * B"),
        ],
        observables=[
            RawEquationDraft(target="B_max", expr_text="B", reduction="TRAJECTORY_MAX"),
        ],
        obligations=[
            RawObligationDraft(
                id="ob1",
                quantifier=quantifier,
                lhs_text="B_max",
                comparator=comparator,
                rhs_text=rhs,
            )
        ],
        config=OdePdeConfig(t_span=(0.0, 10.0), sample_points=200),
    )
    outcome = scope_check_and_lower(table, body, user_prompt="A -> B -> C chain peak B_max over 10 s")
    assert outcome.ok, f"scope_check failed: {outcome.diagnostics}"
    assert outcome.sci_ir is not None
    assert outcome.sci_ir.observables[0].reduction == "TRAJECTORY_MAX"
    sealed = seal_sciir_ihcg(outcome.sci_ir)
    bp = sciir_to_research_blueprint(sealed, user_prompt="A -> B -> C chain peak B_max over 10 s")
    assert bp["ihcg_evidence"]["evidence"]["equations"]["B_max"]["kind"] == "TRAJECTORY_MAX"
    bmax_slot = next(s for s in bp["ihcg_evidence"]["evidence"]["slots"] if s["symbol"] == "B_max")
    assert bmax_slot["semantic_tag"] == 252
    return bp


def test_reduction_not_allowed_in_dynamics():
    dimless = SiDimension.dimensionless()
    table = FrozenSymbolTable(
        ir_id="dyn_red_err",
        archetype=Archetype.ODE_PDE,
        domain=ScientificDomain.PHYSICS_THERMO,
        symbols=[
            SymbolDecl(name="x", display_symbol="x", role=SlotRole.STATE, slot_index=0, dimension=dimless, initial_or_default=1.0),
        ],
    )
    body = ModelBodyDraft(
        dynamics_or_residuals=[
            RawEquationDraft(target="x", expr_text="-x", reduction="TRAJECTORY_MAX"),
        ],
        config=OdePdeConfig(t_span=(0.0, 5.0), sample_points=50),
    )
    outcome = scope_check_and_lower(table, body)
    assert not outcome.ok
    assert any(d.get("code") == "REDUCTION_NOT_ALLOWED_IN_DYNAMICS" for d in outcome.diagnostics)


def test_non_state_trajectory_reduction_rejected():
    dimless = SiDimension.dimensionless()
    table = FrozenSymbolTable(
        ir_id="nonstate_red_err",
        archetype=Archetype.ODE_PDE,
        domain=ScientificDomain.PHYSICS_THERMO,
        symbols=[
            SymbolDecl(name="x", display_symbol="x", role=SlotRole.STATE, slot_index=0, dimension=dimless, initial_or_default=1.0),
            SymbolDecl(name="y", display_symbol="y", role=SlotRole.DERIVED_OBSERVABLE, slot_index=1, dimension=dimless, initial_or_default=0.0),
        ],
    )
    body = ModelBodyDraft(
        dynamics_or_residuals=[RawEquationDraft(target="x", expr_text="-x")],
        observables=[RawEquationDraft(target="y", expr_text="x + 1", reduction="TRAJECTORY_MAX")],
        config=OdePdeConfig(t_span=(0.0, 5.0), sample_points=50),
    )
    outcome = scope_check_and_lower(table, body)
    assert outcome.ok and outcome.sci_ir is not None
    sealed = seal_sciir_ihcg(outcome.sci_ir)
    with pytest.raises(ValueError, match="TRAJECTORY_EXPRESSION_UNSUPPORTED"):
        sciir_to_research_blueprint(sealed, user_prompt="test")


async def test_abc_chain_forall_bmax_ge_02_does_not_falsify(tmp_path):
    bp_dict = _build_abc_chain_blueprint("FORALL", ">=", "0.2")
    preflight = compile_and_preflight_blueprint(bp_dict, bp_dict["ihcg_evidence"])
    assert preflight.ok, f"preflight failed: {preflight}"
    sm = ResearchStateMachine(tmp_path)
    res = await sm.run(ResearchBlueprint.model_validate({k: v for k, v in bp_dict.items() if k != "sciir_commitment_blake3"}))
    verdict = getattr(res.epistemic_verdict, "value", res.epistemic_verdict)
    assert verdict != "VERIFIED_FALSIFICATION", (
        f"True claim FORALL B_max >= 0.2 was falsely falsified: {res}"
    )


async def test_abc_chain_exists_bmax_le_01_does_not_verify(tmp_path):
    bp_dict = _build_abc_chain_blueprint("EXISTS", "<=", "0.1")
    preflight = compile_and_preflight_blueprint(bp_dict, bp_dict["ihcg_evidence"])
    assert preflight.ok, f"preflight failed: {preflight}"
    sm = ResearchStateMachine(tmp_path)
    res = await sm.run(ResearchBlueprint.model_validate({k: v for k, v in bp_dict.items() if k != "sciir_commitment_blake3"}))
    verdict = getattr(res.epistemic_verdict, "value", res.epistemic_verdict)
    assert verdict != "VERIFIED_DISCOVERY", (
        f"False claim EXISTS B_max <= 0.1 was falsely verified: {res}"
    )


async def test_abc_chain_forall_bmax_le_02_certifies_falsification(tmp_path):
    bp_dict = _build_abc_chain_blueprint("FORALL", "<=", "0.2")
    preflight = compile_and_preflight_blueprint(bp_dict, bp_dict["ihcg_evidence"])
    assert preflight.ok, f"preflight failed: {preflight}"
    sm = ResearchStateMachine(tmp_path)
    res = await sm.run(ResearchBlueprint.model_validate({k: v for k, v in bp_dict.items() if k != "sciir_commitment_blake3"}))
    verdict = getattr(res.epistemic_verdict, "value", res.epistemic_verdict)
    assert verdict == "VERIFIED_FALSIFICATION", (
        f"False claim FORALL B_max <= 0.2 should certify Mode 2A falsification, got: {verdict} / {res.failure_reason}"
    )

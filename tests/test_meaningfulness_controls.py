"""Offline positive controls execute actual kernels through the default runner."""
import pytest

from trap_control_cases import blueprint,run_case
from test_stochastic_equilibrium_default import sirs_model
from invar_agent.research import meaningfulness as m
from invar_agent.research.shbt_federation import SHBT_REPOSITORIES


def iv_ols_model():
    """Designed, confounded linear structural sample with an exogenous instrument.

    z, v and w are orthogonal centered contrasts. x=z+v and y=βx+c v+εw,
    so z is a valid instrument although OLS is confounded. The random
    coefficient c is bounded and rational, not a polynomial sampling law.
    Neither estimator is replaced by its population formula in the kernel.
    """
    code = """def simulation_step(buf, dt):
    u = buf.data[1]
    beta = buf.data[2]
    d = 1+(u-0.5)*(u-0.5)
    buf.data[6] = d
    c = 1.2+0.1*(u-0.5)/d
    noise = 0.3*(u-0.5)
    y0 = -2*beta-c+noise
    y1 = c-noise
    y2 = -c-noise
    y3 = 2*beta+c+noise
    iv = (-y0-y1+y2+y3)/4
    ols = (-2*y0+2*y3)/8
    buf.data[3] = iv
    buf.data[4] = ols
    selected = buf.data[0]*iv+(1-buf.data[0])*ols
    buf.data[5] = ols-selected
"""
    bp = blueprint("control_p2_iv_ols",dict(d="1+(u-0.5)**2",iv="truth",ols="truth+0.6+0.05*(u-0.5)/d",draw="theta*(ols-iv)"),
        dict(theta=[0,1],u=[0,1],truth=[1,1],iv=[-2,4],ols=[-2,4],draw=[-2,2],d=[1,1.25]),[1,.5,1,0,0,0,1],"draw > 0.45",
        controls=("theta","u","truth"),mechanism=("theta",),archetype="MONTE_CARLO",code=code,steps=2000)
    evidence = bp.ihcg_evidence["evidence"]
    evidence["slots"][1]["stochastic_role"] = "UNIFORM"
    evidence["equations"]["draw"].update(kind="MC_SAMPLE",state_symbol="draw")
    return bp


@pytest.mark.asyncio
async def test_p2_iv_2sls_vs_ols(tmp_path):
    _,certificate,_ = await run_case(tmp_path,iv_ols_model(),"VERIFIED_DISCOVERY")
    audit = certificate.significance_audit
    assert audit.method == "MC_EMPIRICAL_BERNSTEIN"
    assert audit.certified_lower_margin > .45
    assert audit.replicate_count == 2000
    assert audit.confidence_level == .99
    assert audit.win_rate_method == "MC_CLOPPER_PEARSON"
    assert audit.win_rate_interval[0] > .99
    assert certificate.fidelity_audit.statistical_audit.passed


@pytest.mark.asyncio
async def test_p3_sirs_endemic_equilibrium(tmp_path):
    _,certificate,_ = await run_case(tmp_path,sirs_model(),"VERIFIED_DISCOVERY")
    assert certificate.significance_audit.method == "FIXED_POINT_KRAWCZYK_HURWITZ"
    active = certificate.identifiability_audit
    assert active.equilibrium_restoring_force > 1e-5
    assert all(value > 0 for value in active.column_norms.values())
    assert not active.dead_parameters


@pytest.mark.asyncio
async def test_p1_seir_v_herd_immunity_boundary(tmp_path):
    bp = blueprint("control_p1_seir_v",dict(Reff="R0*(1-e*pv)",response="Reff*Reff+1"),
        dict(R0=[2.5,4],e=[.85,.85],pv=[0,1],Reff=[0,5],response=[0,30]),[3,.85,.9,0,1],
        "Reff < 1",controls=("R0","e","pv"),mechanism=("pv",),mode="BOUNDARY",archetype="PARAM_SWEEP")
    _,certificate,_ = await run_case(tmp_path,bp,"VERIFIED_BOUNDARY_DISCOVERY")
    boundary = certificate.boundary_certificate
    bracket = sorted([boundary.valid_witness_theta["pv"],boundary.falsified_witness_theta["pv"]])
    assert .780 <= bracket[0] < (1-1/3)/.85 < bracket[1] <= .788
    safe = certificate.certified_valid_domain
    assert safe["pv"][0] > (1-1/4)/.85
    assert certificate.domain_compression_bits > 0
    from invar_agent.builder.ingestion.vault_adapter import load_vault_bundle
    from invar_agent.builder.ingestion.pipeline import seal_ir
    from invar_agent.builder.research_bridge import validate_research_ir
    bundle = load_vault_bundle(tmp_path,bp.blueprint_id)
    seal_ir(bundle)
    validate_research_ir(bundle)
    assert bundle.certified_valid_domain == safe


@pytest.mark.asyncio
@pytest.mark.parametrize("hardcoded",[False,True])
async def test_p4_shbt_open_boundary_zero_dissipation(tmp_path,hardcoded):
    """Registry SHBT-CF open LANR power and MgB2 temperature frontier, in SI.

    Precision's delta_fr=0 and QC's Q_internal=0 are evaluated as paired
    forward/inverse ledger contributions, not assigned zero. The latter is
    boundary work per reversible QC operation (bits are dimensionless in SI).
    This tests the registry's assumed reversible closure, not its physical truth.
    The actual power/temperature mechanisms are orthogonal (gamma=1), unlike
    the former tuned dimensionless fixture (gamma=1.25).
    """
    cf = SHBT_REPOSITORIES["sys1own/shbt-cf"]
    precision = SHBT_REPOSITORIES["sys1own/shbt-precision"]
    qc = SHBT_REPOSITORIES["sys1own/shbt-qc"]
    constants = {c.symbol:c.si_value for pack in (precision,cf,qc) for c in pack.constants}
    equations = dict(e.python_expr.split(" = ",1) for e in cf.equations if e.python_expr.startswith(("P_gen =", "P_net =", "MgB2_margin =")))
    assert any(e.python_expr == "delta_fr = 0.0" for e in precision.equations)
    assert any(e.python_expr == "Q_internal = 0.0" for e in qc.equations)
    equations.update(delta_fr="P_net/P_ref-P_net/P_ref",Q_internal="P_net*t_prop-P_net*t_prop")
    bounds = dict(P_cell=[0,1000],T_op=[0,35],N_cells=[1800,1800],P_debt=[906000,906000],T_c_MgB2=[39,39],
        t_prop=[constants["t_prop"]]*2,P_ref=[constants["P_cell"]]*2,
        P_gen=[0,1800000],P_net=[-1000000,1000000],MgB2_margin=[0,40],delta_fr=[-1,1],Q_internal=[-1,1])
    bp = blueprint("control_p4_shbt_"+str(hardcoded).lower(),equations,bounds,
        [constants["P_cell"],27.21,constants["N_cells"],constants["P_debt"],constants["T_c_MgB2"]],
        ["P_net > 0","MgB2_margin > 0"],controls=tuple(bounds)[:7],mechanism=("P_cell","T_op"),
        units="SI: power quantities in W; T_op,T_c_MgB2,MgB2_margin in K; t_prop in s; Q_internal in J/bit; N_cells and delta_fr dimensionless.",
        model_notes="Open SHBT-CF boundary includes the 906000 W Landauer debt. Reversible forward/inverse framing and heat terms cancel; no claim of zero external heat or boundary cost.")
    bp.contracts[0] = bp.contracts[0].model_copy(update={"initial_state":[constants["P_cell"],27.21,constants["N_cells"],constants["P_debt"],constants["T_c_MgB2"],constants["t_prop"],constants["P_cell"],0,0,0,0,0]})
    dimensions = {n:([1,2,-3,0,0,0,0] if n.startswith("P_") else [0,0,0,0,1,0,0] if n in {"T_op","T_c_MgB2","MgB2_margin"} else [0,0,1,0,0,0,0] if n == "t_prop" else [1,2,-2,0,0,0,0] if n == "Q_internal" else [0]*7) for n in bounds}
    for slot in bp.ihcg_evidence["evidence"]["slots"]:
        name = slot["symbol"]
        slot["si_dim"] = dimensions[name]
        if name == "P_net": slot["semantic_tag"] = m.BOUNDARY_FLUX
        if name in {"delta_fr","Q_internal"}: slot.update(role="CONSERVATION_AUX",semantic_tag=m.ZERO_DISSIPATION_INVARIANT)
    for name,equation in bp.ihcg_evidence["evidence"]["equations"].items():
        equation["si_dim"] = dimensions[name]
        if name in {"delta_fr","Q_internal"}: equation["semantic_tag"] = m.BOUNDARY_FLUX
    if hardcoded:
        bp.code_template = bp.code_template[:bp.code_template.index("    buf.data[10]")]+"    buf.data[10] = 0.0\n    buf.data[11] = 0.0\n"
    _,certificate,_ = await run_case(tmp_path,bp,"INCONCLUSIVE_UNVERIFIED" if hardcoded else "VERIFIED_DISCOVERY")
    if hardcoded: assert "TRIVIAL_HARDCODED_INVARIANT" in certificate.failure_reason
    else:
        assert certificate.identifiability_audit.brun_collinearity_gamma_k == pytest.approx(1.0)
        assert set(certificate.identifiability_audit.excluded_certified_invariants) == {"delta_fr","Q_internal"}
        assert not certificate.identifiability_audit.dead_parameters
        assert certificate.uniqueness_audit.reference_nontrivial
        assert certificate.uniqueness_audit.shbt_open_boundary_closure

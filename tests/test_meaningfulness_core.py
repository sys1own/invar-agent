"""Hermetic tests for the bounded scientific proof substrate."""

import json
import math
import time

import pytest
from pydantic import ValidationError

from invar_agent import _native
from invar_agent.research import meaningfulness as m


def graph(claim="x > 0", *, env=(), archetype="ODE_PDE", mode="FORALL", equation="-k*x", mc_bounds=None):
    evidence = dict(
        slots=[dict(slot_idx=i, symbol=s, si_dim=[0]*7, role=role) for i, (s, role) in enumerate((("x", "DYNAMIC_STATE"), ("k", "CONTROL_PARAM")))],
        equations={"dx": dict(expression=equation, si_dim=[0]*7)}, environmental_assumptions=list(env),
        obligations=[dict(id="target", expression=claim, obligation_class="HYPOTHESIS_TARGET", prompt_quote=claim, prompt_span=[0, len(claim)])],
        quantifier_mode=mode, mechanism_parameters=["k"], control_values={"k": 1.0}, mc_difference_bounds=mc_bounds,
    )
    return m.IhcgCompiler().compile_from_prompt(claim, evidence, {"x": [-2, 2], "k": [0, 1]}, archetype)


def fidelity(error=1e-8):
    return m.FidelityAudit(passed=True, algebraic_equiv_passed=True, si_and_semantic_tag_passed=True,
                           convergence_passed=True, physical_checks_passed=True, epsilon_disc=error, epsilon_total=error)


def active():
    return m.IdentifiabilityAudit(coupling_passed=True)


def purity():
    return m.PurityAudit(static_purity_passed=True, domain_tag="stage1/test")


def contract():
    return m.ContractAudit(entailment_locked=True, classification="CONSTRAINING", z3_logic_tier="QF_LRA")


def test_ihcg_quote_binding_seal_and_mutation_detection():
    g = graph(); g.assert_sealed()
    assert graph().h_ihcg == g.h_ihcg
    with pytest.raises(ValidationError):
        g.h_ihcg = "x"
    g.omega["x"] = (-3.0, 3.0)
    with pytest.raises(ValueError, match="SEAL_MISMATCH"):
        m.verify_claim_nontriviality(g, ["x > 0"])
    source = graph().model_dump(mode="json", exclude={"omega", "archetype", "h_ihcg"})
    source["obligations"][0]["prompt_quote"] = "wrong"
    # Non-blocking provenance: mismatched prompt_quote drops prompt_span without blocking deterministic sealing.
    recompiled = m.IhcgCompiler().compile_from_prompt("x > 0", source, {"x": [-2, 2], "k": [0, 1]}, "ODE_PDE")
    assert recompiled.obligations[0].prompt_span is None
    recompiled.assert_sealed()
    with pytest.raises(ValueError): m.solve_query("__import__('os')", {})
    with pytest.raises(ValueError): m.solve_query("x > 0", {"x": [math.nan, 2]})


@pytest.mark.parametrize("claim,env,expected", [
    ("x > 0", (), "CONSTRAINING"), ("x > 0", ("x < -1", "x > 1"), "VACUOUS"),
    ("x <= 2", (), "TAUTOLOGICAL"), ("x > 3", (), "INFEASIBLE"),
    ("implies(x > 3, x > 0)", (), "VACUOUS"),
])
def test_q1_q5_classification(claim, env, expected):
    result = m.classify_contract_vacuity(graph(claim, env=env))
    assert result.classification == expected
    assert {"Q1", "Q2", "Q3", "Q4_0"} <= result.queries.keys()


def test_entailment_nontriviality_and_model_separation():
    g = graph()
    assert m.verify_entailment_lock(g, ["x > 0.5"]).entailment_locked
    weakened = m.verify_entailment_lock(g, ["x >= -1"])
    assert not weakened.entailment_locked and weakened.counterexample_if_weakened is not None
    assert m.classify_contract_vacuity(g, ["x < -1"]).classification == "INFEASIBLE_UNDER_MODEL"
    assert m.verify_claim_nontriviality(g, ["x > 0"]).passed
    trivial = m.verify_claim_nontriviality(g, ["x*x >= 0"])
    assert not trivial.passed and trivial.failure_reason == "TRIVIAL_DEDUCIBLE_WITHOUT_MODEL"
    assert trivial.contract_audit.queries["Q3"].status == "UNSAT"
    assert not m.verify_claim_nontriviality(graph(env=("x > 0",)), ["x > 0"]).passed
    with pytest.raises(ValueError): m.verify_claim_nontriviality(g, [])
    assert not m.verify_entailment_lock(g, ["False"]).entailment_locked
    assert m.classify_contract_vacuity(graph("x > 0 and x <= 2")).redundant_conjuncts == ("1",)


@pytest.mark.parametrize("expression,domain", [
    ("exp(x) < 0", {"x": [-1, 1]}), ("log(x) > 3", {"x": [1, 2]}),
    ("sqrt(x) < 0", {"x": [1, 2]}), ("sin(x) > 2", {"x": [0, 1]}),
    ("cos(x) < -2", {"x": [0, 1]}), ("logistic_sigmoid(x) < 0", {"x": [-1, 1]}),
    ("kl_term(x,x) < -2", {"x": [1, 2]}), ("hill_eq(x,1,2) > 2", {"x": [0.5, 1]}),
    ("michaelis_menten(x,1,1) > 2", {"x": [0.5, 1]}), ("x**(1/3) < 0", {"x": [1, 2]}),
])
def test_tier3_operator_outer_enclosures(expression, domain):
    result = m.solve_query(expression, domain)
    assert result.status == "UNSAT"
    assert result.z3_tier_used == "MCCORMICK_QF_LRA"


def test_scientific_operator_semantics_and_native_healing_domains():
    value = m._iv(m._parse("kl_term(0.2,0.5)"), {})
    expected = 0.2*math.log(0.2/0.5)-0.2+0.5
    assert value.lo <= expected <= value.hi
    assert m.solve_query("kl_term(0.2,0.5) < 0", {}).status == "UNSAT"
    assert m._iv(m._parse("kl_term(0,0.5)"), {}).lo == 0.5
    assert not m._native_equation_domain_valid(m._parse("exp(100*x)"), {"x": (-2.0, 2.0)})
    assert not m._native_equation_domain_valid(m._parse("1/x"), {"x": (0.0, 1.0)})


def test_solver_tiers_unknown_and_verified_halton_refinement():
    assert m.solve_query("x > 0", {"x": [-1, 1]}).z3_tier_used == "QF_LRA"
    assert m.solve_query("x*x == 2", {"x": [1, 2]}).z3_tier_used == "QF_NRA"
    result = m.solve_query("sin(x) == 0.1234567890123456", {"x": [0, 6]}, max_boxes=1, witness=True)
    assert result.status == "UNKNOWN" and result.z3_tier_used == "UNKNOWN"
    witness, probes = m._witness_probe(m._parse("sin(x) > 0.9"), {"x": (0.0, 6.0)}, time.monotonic()+1)
    assert witness is not None and 1 <= probes <= 512
    assert m._truth(m._parse("sin(x) > 0.9"), {"x": (witness["x"], witness["x"])}) is True
    assert m.solve_query("False", {}).status == "UNSAT"
    with pytest.raises(ValueError): m.solve_query("True", {}, max_boxes=257)


def test_bounded_coordinate_refinement_and_actual_solver_unknown(monkeypatch):
    node = m._parse("(x-0.1234)**2 < 1e-10")
    witness, probes = m._witness_probe(node, {"x": (0.0, 1.0)}, time.monotonic()+1)
    assert witness is not None and probes > 512
    assert m._truth(node, {"x": (witness["x"], witness["x"])}) is True
    class IndecisiveSolver:
        def set(self, **kwargs): pass
        def add(self, *args): pass
        def check(self): return m.z3.unknown
    monkeypatch.setattr(m.z3, "SolverFor", lambda logic: IndecisiveSolver())
    assert m.solve_query("x > 0", {"x": [-1, 1]}, max_boxes=1).status == "UNKNOWN"
    constructive = m.solve_query("x > 0", {"x": [-1, 1]}, witness=True, max_boxes=1)
    assert constructive.status == "SAT" and constructive.z3_tier_used == "WITNESS_PROBE"


def test_peak_interpolation_gci_and_dual_scheme_reliability_rejection():
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    values = [3-(t-0.83)**2 for t in times]
    peak, location, eps = m.quadratic_peak(times, values)
    assert peak == pytest.approx(3) and location == pytest.approx(0.83)
    assert eps == pytest.approx(2*0.5**2/8)
    with pytest.raises(ValueError): m.quadratic_peak([], [])
    a = m.roache_gci(1.0+0.01**4, 1.0+0.02**4, 1.0+0.04**4)
    assert a.passed and a.apparent_order_p == pytest.approx(4, rel=1e-6)
    b = m.roache_gci(1.1+0.01**2, 1.1+0.02**2, 1.1+0.04**2, nominal_order=2)
    failed = m.dual_scheme_fidelity(a, b, algebraic_equiv_passed=True, si_and_semantic_tag_passed=True, physical_checks_passed=True)
    assert not failed.passed and failed.convergence_reliability_ratio > 1
    assert failed.failure_reason == "CONVERGENCE_ESTIMATOR_UNRELIABLE"
    decision = m.route_post_execution(graph(), contract(), purity(), failed, active(), m.HypothesisEvalResult(signed_margin=-1.0, epsilon_total=failed.epsilon_total), 0, 1)
    assert decision.action == "REFINE_PRECISION_ONCE" and not decision.halt_healing
    decision = m.route_post_execution(graph(), contract(), purity(), failed, active(), m.HypothesisEvalResult(signed_margin=-1.0, epsilon_total=failed.epsilon_total, precision_refined=True), 0, 1)
    assert decision.action == "TERMINATE_INCONCLUSIVE" and not decision.halt_healing
    assert not m.roache_gci(1.0, 2.0, 1.5).passed


def test_fidelity_runner_uses_two_independent_three_grid_schemes():
    calls = []
    def runner(uast, *, scheme, step_size, timeout_ms):
        calls.append((scheme, step_size))
        times, trajectory = m.integrate_ode(lambda t, y: (-y[0],), [1], start=0.0, stop=1.0, steps=round(1/step_size), scheme=scheme)
        return m.GridRun(scheme=scheme, times=times, values=tuple(y[0] for y in trajectory), physical_checks={})
    audit = m.certify_implementation_fidelity(graph(), {"equations": {"dx": "-k*x"}, "parameters": {"k": 1.0}, "step_size": 0.05}, runner)
    assert audit.passed and audit.convergence_reliability_ratio <= 1
    assert calls == [(s, h) for s in ("RK4", "HEUN") for h in (0.025, 0.05, 0.1)]
    failure = m.certify_implementation_fidelity(graph(), {"equations": {"dx": "-k*x"}, "parameters": {"k": 0.9}}, runner)
    assert not failure.passed and failure.failure_reason == "CONTROL_PARAM_CHANGED"


def test_mc_replicates_reliability_and_statistical_routing():
    blocks, keys = m.run_mc_replicate_blocks(lambda u: u, seed=19)
    assert (blocks, keys) == m.run_mc_replicate_blocks(lambda u: u, seed=19)
    stable = tuple(tuple([0.0, 1.0]*16) for _ in range(8))
    assert m.statistical_reliability(stable, keys).passed
    broken = tuple(tuple([0.0]*16+[1.0, -1.0]*8) for _ in range(8))
    unreliable = m.statistical_reliability(broken, keys)
    assert not unreliable.passed and unreliable.failure_reason == "STATISTICAL_ESTIMATOR_UNRELIABLE"
    bad = m.FidelityAudit(passed=False, algebraic_equiv_passed=True, si_and_semantic_tag_passed=True, convergence_passed=False,
                          physical_checks_passed=True, epsilon_disc=0.0, epsilon_total=0.0, statistical_audit=unreliable, failure_reason=unreliable.failure_reason)
    route = m.route_post_execution(graph(archetype="MONTE_CARLO"), contract(), purity(), bad, active(), m.HypothesisEvalResult(signed_margin=-1.0, epsilon_total=0.0), 0, 2)
    assert route.action == "TERMINATE_INCONCLUSIVE" and not route.halt_healing
    with pytest.raises(ValueError): m.statistical_reliability([], [])
    with pytest.raises(ValueError): m.statistical_reliability(stable, [keys[0]]*8)
    shifted = tuple(tuple([0.0]*16+[1.0]*16) for _ in range(8))
    assert not m.statistical_reliability(shifted, keys).passed


def test_fd_jacobian_checks_executable_at_center_and_krawczyk_corners():
    expr = json.dumps({"kind": "binary", "op": "-", "left": {"kind": "binary", "op": "*", "left": {"kind": "symbol", "name": "x0"}, "right": {"kind": "symbol", "name": "x0"}}, "right": {"kind": "literal", "value": 2.0}})
    report = json.loads(_native.py_certify_krawczyk_fixed_point([expr], [math.sqrt(2)], 0.05))
    assert report["certified"]
    seen = []
    def executable(point): seen.append(tuple(point)); return [point[0]**2-2]
    audit = m.cross_check_krawczyk_jacobian(executable, [math.sqrt(2)], report, [[0.0]], executable_rounding_bound=1e-15)
    assert audit.passed and audit.samples_checked == 3 and len(seen) == 6
    mismatch = m.cross_check_krawczyk_jacobian(lambda p: [3*p[0]**2-2], [math.sqrt(2)], report, [[0.0]], executable_rounding_bound=1e-15)
    assert not mismatch.passed and mismatch.failure_reason == "JACOBIAN_ENCLOSURE_MISMATCH"
    with pytest.raises(ValueError): m.cross_check_krawczyk_jacobian(executable, [], report, [], executable_rounding_bound=0.0)


def test_purity_counts_and_null_baseline_are_not_model_mutations():
    audit = m.audit_purity({}, [], projection_drifts=[0, 1e-13, 1e-5, 0], stabilization_hits=[False, False, False, True], domain_tag="stage1/test")
    assert (audit.total_steps, audit.preserved_steps, audit.projected_steps, audit.stabilized_steps) == (4, 2, 1, 1)
    assert audit.enforcement_ratio == 0.5
    assert m.audit_purity({}, [], projection_drifts=[], stabilization_hits=[], domain_tag="empty").enforcement_ratio == 0
    uast = {"parameters": {"k": 1.0, "seed": 42}, "other": [1, 2]}
    baseline = m.synthesize_null_baseline(uast, graph(), 1)
    assert baseline.parameter_overrides == {"k": 0.0} and uast["parameters"]["k"] == 1
    baseline.uast["other"].append(3)
    assert uast["other"] == [1, 2]
    with pytest.raises(ValueError): m.synthesize_null_baseline(uast, graph(), 2)
    parent = graph()
    evidence = parent.model_dump(mode="json", exclude={"omega", "archetype", "h_ihcg"})
    child = m.IhcgCompiler().extend_stage(parent, "x > 0", evidence, parent.omega, "ODE_PDE")
    child.assert_sealed()
    assert child.parent_ihcg_digest == parent.h_ihcg and len(child.stage_chain_digests) == 1
    foundation = m.synthesize_null_baseline({"pinned_foundation": {"uast": uast, "ihcg_digest": parent.h_ihcg}}, child, 2)
    assert foundation.foundation_digest == parent.h_ihcg


def test_significance_for_all_four_archetypes_and_micro_effect_diagnostics():
    r = m.SignificanceResult(observables=(1000.0,), fidelity=fidelity())
    r0 = m.SignificanceResult(observables=(999.9995,), fidelity=fidelity())
    audit = m.certify_archetype_significance("ODE_PDE", r, r0, active(), graph())
    assert audit.passed and audit.micro_effect_warning and audit.relative_effect_size < 1e-3
    assert m.domain_compression_bits({"x": [0, 4]}, {"x": [1, 2]}) == 2
    with pytest.raises(ValueError): m.domain_compression_bits({"x": [0, 1]}, {"x": [0, 0]})
    keys = tuple((1, i, 0) for i in range(2048))
    reliability = m.statistical_reliability(tuple(tuple([0.0, 1.0]*16) for _ in range(8)), keys[:8])
    mc_fid = fidelity().model_copy(update={"statistical_audit": reliability})
    mc = m.SignificanceResult(observables=(0.5,), fidelity=mc_fid, paired_samples=(0.5,)*2048, stream_keys=keys)
    null = m.SignificanceResult(observables=(0.0,), fidelity=mc_fid, paired_samples=(0.0,)*2048, stream_keys=keys)
    assert m.certify_archetype_significance("MONTE_CARLO", mc, null, active(), graph(archetype="MONTE_CARLO", mc_bounds=[0, 1])).passed
    fixed = m.SignificanceResult(observables=(1.0,), fidelity=fidelity(), krawczyk={"certified": True}, stability_matrix=(((-1.0, -1.0),),))
    assert m.certify_archetype_significance("FIXED_POINT", fixed, null.model_copy(update={"observables": (0.0,)}), active(), graph(archetype="FIXED_POINT")).passed
    boundary = m.BoundaryCertificate(valid_witness_theta={"x": 0.505}, falsified_witness_theta={"x": 0.495}, valid_margin_lower=0.001, falsified_margin_upper=-0.001, bracket_width=0.01, target_tolerance=0.02)
    sweep = r.model_copy(update={"boundary": boundary})
    assert m.certify_archetype_significance("PARAM_SWEEP", sweep, r0, active(), graph(archetype="PARAM_SWEEP")).passed
    indicators = mc.model_copy(update={"bernoulli_indicators": True, "paired_samples": (1.0,)*2048})
    zeros = null.model_copy(update={"bernoulli_indicators": True})
    cp = m.certify_archetype_significance("MONTE_CARLO", indicators, zeros, active(), graph(archetype="MONTE_CARLO", mc_bounds=[0, 1]))
    assert cp.passed and cp.method == "MC_CLOPPER_PEARSON"
    assert m.clopper_pearson_interval(0, 100)[0] == 0
    assert m.clopper_pearson_interval(100, 100)[1] == 1
    with pytest.raises(ValueError): m.clopper_pearson_interval(1, 0)
    chaotic = r.model_copy(update={"lyapunov_t": 0.1, "lyapunov_2t": 0.101, "lyapunov_disc_error": 0.001})
    stable = r0.model_copy(update={"lyapunov_t": -0.1, "lyapunov_2t": -0.101, "lyapunov_disc_error": 0.001})
    assert m.certify_archetype_significance("PARAM_SWEEP", chaotic, stable, active(), graph(archetype="PARAM_SWEEP")).passed


def test_authoritative_margin_router_and_retry_caps():
    g = graph(); f = fidelity(0.01)
    for margin in (-0.01, 0.0, 0.01):
        assert m.route_post_execution(g, contract(), purity(), f, active(), m.HypothesisEvalResult(signed_margin=margin, epsilon_total=0.01), 0, 1).action == "REFINE_PRECISION_ONCE"
        decision = m.route_post_execution(g, contract(), purity(), f, active(), m.HypothesisEvalResult(signed_margin=margin, epsilon_total=0.01, precision_refined=True), 0, 1)
        assert decision.action == "TERMINATE_INCONCLUSIVE" and decision.reason == "MARGIN_WITHIN_UNCERTAINTY"
    for mode, action in (("FORALL", "RUN_MODE2A_COUNTEREXAMPLE_CERTIFIER"), ("EXISTS", "RUN_MODE2B_EXHAUSTIVE_REFUTATION"), ("BOUNDARY", "RUN_BOUNDARY_CERTIFIER")):
        decision = m.route_post_execution(graph(mode=mode), contract(), purity(), f, active(), m.HypothesisEvalResult(signed_margin=-1.0, epsilon_total=0.01), 0, 1)
        assert decision.action == action and decision.halt_healing
    with pytest.raises(ValueError): m.route_post_execution(g, contract(), purity(), f, active(), m.HypothesisEvalResult(signed_margin=1.0, epsilon_total=0.01), 0, 4)


def test_certificate_seal_eighth_gate_strict_schema_and_persistence(tmp_path):
    g = graph(); f = fidelity(); audit = m.verify_claim_nontriviality(g, ["x > 0"])
    uniqueness = m.UniquenessAudit(passed=True, cds_model_hash="a"*64, cds_claim_hash="b"*64, cds_full_hash="c"*64, vault_class="NOVEL_FIRST_IN_VAULT", reference_nontrivial=True)
    significance = m.certify_archetype_significance("ODE_PDE", m.SignificanceResult(observables=(1.0,), fidelity=f), m.SignificanceResult(observables=(0.0,), fidelity=f), active(), g)
    kwargs = dict(run_id="test", verdict=m.EpistemicVerdict.VERIFIED_DISCOVERY, ihcg=g, contract_audit=contract(), nontriviality_audit=audit,
                  purity_audit=purity(), fidelity_audit=f, identifiability_audit=active(), uniqueness_audit=uniqueness, significance_audit=significance, uast_digest="d"*64)
    certificate = m.build_meaningfulness_certificate(**kwargs)
    assert m.verify_certificate_receipt(certificate)
    assert len(m.meaningfulness_gates(certificate)) == 8 and all(m.meaningfulness_gates(certificate).values())
    path = tmp_path/"meaningfulness_certificate.json"
    m.write_meaningfulness_certificate(certificate, path)
    restored = m.MeaningfulnessCertificate.model_validate_json(path.read_text())
    assert restored.nontriviality_audit.passed and m.verify_certificate_receipt(restored)
    assert not m.verify_certificate_receipt(certificate.model_copy(update={"run_id": "changed"}))
    with pytest.raises(ValidationError):
        m.MeaningfulnessCertificate.model_validate_json(certificate.model_dump_json()[:-1]+',"bypass":true}')
    kwargs["nontriviality_audit"] = m.verify_claim_nontriviality(g, ["x*x >= 0"])
    with pytest.raises(ValidationError, match="eight"):
        m.build_meaningfulness_certificate(**kwargs)
    kwargs["verdict"] = m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED
    assert not m.build_meaningfulness_certificate(**kwargs).nontriviality_audit.passed

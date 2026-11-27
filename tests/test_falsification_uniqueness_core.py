import json

import pytest
from pydantic import ValidationError

from invar_agent import _native
from invar_agent.research import falsification as f
from invar_agent.research import meaningfulness as m
from invar_agent.research import uniqueness as u


def graph(claim="x >= 0", *, equations=None, mode="FORALL", domain=None, archetype="ODE_PDE", roles=None, dimensions=None, environment=()):
    domain = domain or {"x": [-1, 1]}
    evidence = dict(slots=[dict(slot_idx=i, symbol=s, si_dim=(dimensions or {}).get(s, [0]*7), role=(roles or {}).get(s, "DYNAMIC_STATE")) for i, s in enumerate(domain)],
        equations={k: dict(expression=v, si_dim=[0]*7) for k, v in (equations if equations is not None else {"constraint": "x == x"}).items()},
        environmental_assumptions=list(environment), obligations=[dict(id="h", expression=claim, obligation_class="HYPOTHESIS_TARGET", prompt_quote=claim, prompt_span=[0, len(claim)])], quantifier_mode=mode)
    return m.IhcgCompiler().compile_from_prompt(claim, evidence, domain, archetype)


def fidelity(error=1e-6):
    return m.FidelityAudit(passed=True, algebraic_equiv_passed=True, si_and_semantic_tag_passed=True,
        convergence_passed=True, physical_checks_passed=True, convergence_reliability_ratio=0.5,
        epsilon_disc=error, epsilon_total=error)


def test_mode1_minimal_core_and_consistency():
    g = graph("x >= 0.5", equations={"governing": "x <= 0", "irrelevant": "y == 0"}, domain={"x": [-1, 1], "y": [-1, 1]})
    certificate = f.run_mode1_deductive_falsification(g, fidelity=fidelity(), run_carcara=True)
    assert certificate.passed and certificate.core_minimal
    assert "h_user" in certificate.minimal_unsat_core and "model:irrelevant" not in certificate.minimal_unsat_core
    solver = f.z3.Solver(); solver.from_string(certificate.proof_smt2)
    assert solver.check() == f.z3.unsat
    assert certificate.external_checker_status in {"UNAVAILABLE", "VERIFIED", "FAILED"}
    inconsistent = graph(equations={"a": "x > 2"})
    bad = f.run_mode1_deductive_falsification(inconsistent, fidelity=fidelity())
    assert not bad.passed and bad.failure_reason == "MODEL_ENVIRONMENT_INCONSISTENT"
    with pytest.raises(ValueError): f.run_mode1_deductive_falsification(g, {"fake": "False"}, fidelity=fidelity())


def test_mode2a_interval_witness_and_error_budget():
    g = graph()
    result = f.run_mode2a_counterexample_falsification(g, {"x": -0.5}, f.WitnessEvaluation(signed_margin=-0.5, epsilon_total=1e-6), fidelity())
    assert result.passed and result.witness_margin_interval[1]+result.epsilon_total < 0
    uncertain = f.run_mode2a_counterexample_falsification(g, {"x": 0.0}, f.WitnessEvaluation(signed_margin=0.0, epsilon_total=1e-6), fidelity())
    assert not uncertain.passed
    with pytest.raises(ValueError): f.run_mode2a_counterexample_falsification(g, {"x": -2.0}, f.WitnessEvaluation(signed_margin=-2.0, epsilon_total=1e-6), fidelity())
    with pytest.raises(ValueError): f.run_mode2a_counterexample_falsification(g, {"x": -0.5}, f.WitnessEvaluation(signed_margin=-0.5, epsilon_total=0.0), fidelity())


def test_mode2b_native_dimension_budget_and_lipschitz_grid():
    g = graph("x >= 2", mode="EXISTS")
    result = f.run_mode2b_exhaustive_refutation(g, f.MarginSpec(), fidelity())
    assert result.passed and result.certified_refuted_volume_fraction == 1
    assert result.certified_subbox_count > 0 and result.max_unresolved_upper_bound is None
    assert f.dimension_scaled_box_budget(5) == 8192 and f.dimension_scaled_box_budget(12) == 32768
    unresolved = f.run_mode2b_exhaustive_refutation(graph(mode="EXISTS"), f.MarginSpec(), fidelity(), max_boxes=1)
    assert unresolved.budget_exhausted and not unresolved.passed
    assert unresolved.epistemic_verdict == m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED
    oracle = f.MarginSpec(evaluator=lambda point: (point["x"]-2, point["x"]-2), gradient_enclosure=lambda box: ((0.999, 1.001),), ihcg_digest=g.h_ihcg)
    grid = f.run_mode2b_exhaustive_refutation(g, oracle, fidelity())
    assert grid.passed and grid.enclosure_method == "LIPSCHITZ_GRID"


def test_invalid_model_witness_oracle_and_unknown_are_not_falsifications(monkeypatch):
    g = graph(equations={"law": "x == 0"})
    result = f.run_mode2a_counterexample_falsification(g, {"x": -0.5}, f.WitnessEvaluation(signed_margin=-0.5, epsilon_total=1e-6), fidelity())
    assert result.failure_reason == "WITNESS_MODEL_NOT_CERTIFIED"
    exists = graph("x >= 0.5", mode="EXISTS")
    forged = f.MarginSpec(evaluator=lambda p: (p["x"]-0.5, p["x"]-0.5), gradient_enclosure=lambda box: ((0.0, 0.0),), ihcg_digest=exists.h_ihcg)
    assert f.run_mode2b_exhaustive_refutation(exists, forged, fidelity()).failure_reason == "GRADIENT_ENCLOSURE_MISMATCH"
    monkeypatch.setattr(f.z3.Solver, "check", lambda *a, **k: f.z3.unknown)
    unknown = f.run_mode1_deductive_falsification(g, fidelity=fidelity())
    assert not unknown.passed and unknown.failure_reason == "SOLVER_UNKNOWN" and unknown.model_consistency_status == "UNKNOWN"


def test_reliability_blocks_all_falsification_modes_and_strict_schema():
    bad = fidelity().model_copy(update={"passed": False, "convergence_passed": False, "convergence_reliability_ratio": 2.0, "failure_reason": "CONVERGENCE_ESTIMATOR_UNRELIABLE"})
    result = f.run_mode1_deductive_falsification(graph(), fidelity=bad)
    assert not result.passed and result.failure_reason == "CONVERGENCE_ESTIMATOR_UNRELIABLE"
    with pytest.raises(ValidationError):
        m.FalsificationCertificate(mode="MODE1_DEDUCTIVE_MUS", epsilon_total=0.0, epistemic_verdict=m.EpistemicVerdict.VERIFIED_FALSIFICATION)


def test_sharp_boundary_with_uncertain_midpoint():
    g = graph("x >= 0.5", mode="BOUNDARY", domain={"x": [0, 1]}, roles={"x": "CONTROL_PARAM"})
    certificate = f.certify_phase_boundary(g, lambda p: f.WitnessEvaluation(signed_margin=p["x"]-0.5, epsilon_total=1e-6), 1e-3, fidelity=fidelity())
    assert certificate.passed and certificate.bracket_width <= 1e-3
    assert certificate.valid_margin_lower > 0 > certificate.falsified_margin_upper


def cds(g, verdict=m.EpistemicVerdict.VERIFIED_DISCOVERY):
    return u.CanonicalSignatureEngine().compute_cds(g, {"equation_form": "residual"}, verdict)


def test_native_cds_renaming_and_duplicate_domain_extension_precision_gain():
    original = graph(equations={"law": "x*x-x"})
    renamed = graph("z >= 0", equations={"renamed": "z*z-z"}, domain={"z": [-1, 1]})
    first, clone = cds(original), cds(renamed)
    assert first.model_hash == clone.model_hash and first.claim_hash == clone.claim_hash
    entry = u.VaultEntry(run_id="old", cds=first, verdict=m.EpistemicVerdict.VERIFIED_DISCOVERY, epsilon_disc=1e-3, epsilon_total=1e-3)
    duplicate = u.check_intra_vault_uniqueness(clone, renamed, m.EpistemicVerdict.VERIFIED_DISCOVERY, [entry], epsilon_disc=1e-3, epsilon_total=1e-3)
    assert not duplicate.passed and duplicate.vault_class == "EXACT_ALGEBRAIC_DUPLICATE"
    precise = u.check_intra_vault_uniqueness(clone, renamed, m.EpistemicVerdict.VERIFIED_DISCOVERY, [entry], epsilon_disc=1e-5, epsilon_total=1e-5)
    assert precise.passed and precise.vault_class == "VALID_BOUND_SHARPENING"
    extended = graph(equations={"law": "x*x-x"}, domain={"x": [-2, 2]})
    extension = u.check_intra_vault_uniqueness(cds(extended), extended, m.EpistemicVerdict.VERIFIED_DISCOVERY, [entry], epsilon_disc=1e-3, epsilon_total=1e-3)
    assert extension.passed and extension.vault_class == "VALID_DOMAIN_EXTENSION"


def test_reference_registry_and_shbt_has_no_name_based_bypass():
    g = graph(equations={"law": "x*x-x"})
    candidate = cds(g)
    tag = dict(role="DYNAMIC_STATE", si_dim=[0]*7, semantic_tag=0, unit_scale=[1, 1], positive=False)
    formal = dict(expr=json.dumps(m._csel(m._parse("x*x-x"))), symbols={"x": tag})
    registry = u.ReferenceRegistry({"textbook": {"canonical_equations": [formal]}})
    audit = u.check_reference_nontriviality(candidate, g, m.EpistemicVerdict.VERIFIED_DISCOVERY, registry)
    assert not audit.passed and audit.reason == "TRIVIAL_TEXTBOOK_PLAYBACK"
    assert u.check_reference_nontriviality(candidate, g, m.EpistemicVerdict.VERIFIED_FALSIFICATION, registry).passed
    legacy = u.ReferenceRegistry({"shbt-frontier": {"equations": [("x = y", "identity")]}})
    assert not u.check_reference_nontriviality(candidate, g, m.EpistemicVerdict.VERIFIED_DISCOVERY, legacy).passed


def test_refutation_coupled_extension_and_strict_strengthening():
    g = graph("x >= 0", equations={"law": "x*x-x"})
    old = u.VaultEntry(run_id="prior", cds=cds(g), verdict=m.EpistemicVerdict.VERIFIED_DISCOVERY, epsilon_disc=1e-6, epsilon_total=1e-6)
    refuted = u.check_intra_vault_uniqueness(cds(g, m.EpistemicVerdict.VERIFIED_FALSIFICATION), g, m.EpistemicVerdict.VERIFIED_FALSIFICATION,
        [old], epsilon_disc=1e-6, epsilon_total=1e-6)
    assert refuted.passed and refuted.vault_class == "VALID_REFUTATION_OF_PRIOR" and refuted.lineage_edge == "refutes"
    stronger = graph("x >= 0.5", equations={"law": "x*x-x"})
    sharpened = u.check_intra_vault_uniqueness(cds(stronger), stronger, m.EpistemicVerdict.VERIFIED_DISCOVERY, [old], epsilon_disc=1e-6, epsilon_total=1e-6)
    assert sharpened.passed and sharpened.vault_class == "VALID_BOUND_SHARPENING"
    changed = graph("x >= 0", equations={"law": "x*x*x-x"})
    active = m.IdentifiabilityAudit(coupling_passed=True)
    sig = m.SignificanceAudit(passed=True, null_baseline_description="prior", effect_size_estimate=1.0, certified_lower_margin=0.9,
        method="ODE_GCI_SEPARATION", relative_effect_size=1.0, micro_effect_warning=False)
    coupled = u.check_intra_vault_uniqueness(cds(changed), changed, m.EpistemicVerdict.VERIFIED_DISCOVERY, [old], epsilon_disc=1e-6, epsilon_total=1e-6,
        active_audit=active, significance_audit=sig, compared_run_id="prior")
    assert coupled.passed and coupled.vault_class == "VALID_COUPLED_EXTENSION"
    rejected = u.check_intra_vault_uniqueness(cds(changed), changed, m.EpistemicVerdict.VERIFIED_DISCOVERY, [old], epsilon_disc=1e-6, epsilon_total=1e-6)
    assert not rejected.passed


def test_cds_unit_rescaling_and_payload_tamper_rejected():
    first = graph("x >= 1000", equations={"law": "x-y"}, domain={"x": [100, 2000], "y": [100, 2000]})
    clone = graph("a >= 1", equations={"other": "a-b"}, domain={"a": [0.1, 2], "b": [0.1, 2]})
    engine = u.CanonicalSignatureEngine()
    a = cds(first)
    b = engine.compute_cds(clone, {"equation_form": "residual", "symbol_metadata": {s: {"unit_scale": [1000, 1]} for s in ("a", "b")}}, m.EpistemicVerdict.VERIFIED_DISCOVERY)
    assert (a.model_hash, a.claim_hash, a.full_hash) == (b.model_hash, b.claim_hash, b.full_hash)
    raw = a.model_dump(mode="json"); raw["equation_count"] = 12
    with pytest.raises(ValidationError, match="committed"):
        u.CdsTriplet.model_validate_json(json.dumps(raw))


def test_mc_reliability_is_required_and_default_registry_coverage_is_explicit():
    g = graph(archetype="MONTE_CARLO")
    cert = f.run_mode2a_counterexample_falsification(g, {"x": -0.5}, f.WitnessEvaluation(signed_margin=-0.5, epsilon_total=1e-6), fidelity())
    assert cert.failure_reason == "STATISTICAL_RELIABILITY_UNVERIFIED"
    registry = u.ReferenceRegistry()
    assert registry.unresolved_references and not registry.coverage_complete
    with pytest.raises(ValueError): f.dimension_scaled_box_budget(0)


def test_empty_vault_and_actual_shbt_open_boundary_structure():
    from invar_agent.research.shbt_federation import SHBT_REPOSITORIES
    assert SHBT_REPOSITORIES
    g = graph("x >= 0", equations={"open_flux": "x-y", "boundary_closure": "y-z"}, domain={s: [-1, 1] for s in "xyz"})
    signature = cds(g)
    first = u.check_intra_vault_uniqueness(signature, g, m.EpistemicVerdict.VERIFIED_DISCOVERY, [], epsilon_disc=1e-6, epsilon_total=1e-6)
    assert first.vault_class == "NOVEL_FIRST_IN_VAULT"
    reference = u.check_reference_nontriviality(signature, g, m.EpistemicVerdict.VERIFIED_DISCOVERY, u.ReferenceRegistry())
    assert reference.passed and not reference.shbt_open_boundary_closure


def test_partial_native_coverage_and_five_dimensional_budget():
    g = graph("x >= 0.1", mode="EXISTS")
    partial = f.run_mode2b_exhaustive_refutation(g, f.MarginSpec(), fidelity(), max_boxes=2)
    assert partial.budget_exhausted and 0 < partial.certified_refuted_volume_fraction < 1
    assert partial.certified_subbox_count > 0 and partial.max_unresolved_upper_bound is not None
    domain = {f"x{i}": [-1, 1] for i in range(5)}
    high = graph("x0+x1+x2+x3+x4 >= 9", equations={"identity": "x0 == x0"}, mode="EXISTS", domain=domain)
    certified = f.run_mode2b_exhaustive_refutation(high, f.MarginSpec(), fidelity())
    assert certified.passed and certified.effective_max_boxes == 8192
    assert f.run_mode1_deductive_falsification(graph(equations={}), fidelity=fidelity()).failure_reason == "EMPTY_MODEL"


def test_dimensional_pi_signature_and_actual_shbt_ledger_equations():
    length, time, velocity = [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0, 0]
    first = graph("x >= v*t", equations={"law": "x-v*t"}, domain={"x": [1, 2], "v": [1, 1], "t": [1, 1]}, dimensions={"x": length, "v": velocity, "t": time})
    clone = graph("a >= b*c", equations={"law2": "a-b*c"}, domain={"a": [0.001, 0.002], "b": [0.001, 0.001], "c": [1, 1]}, dimensions={"a": length, "b": velocity, "c": time})
    engine = u.CanonicalSignatureEngine(); a = cds(first)
    b = engine.compute_cds(clone, {"equation_form": "residual", "symbol_metadata": {s: {"unit_scale": [1000, 1]} for s in ("a", "b")}}, m.EpistemicVerdict.VERIFIED_DISCOVERY)
    assert (a.model_hash, a.claim_hash, a.full_hash) == (b.model_hash, b.claim_hash, b.full_hash)
    from invar_agent.research.shbt_federation import SHBT_REPOSITORIES
    equations = {}
    for equation in SHBT_REPOSITORIES["sys1own/shbt-cf"].equations[:2]:
        name, expression = equation.python_expr.split("=", 1); equations[name.strip()] = expression.strip()
    power = [1, 2, -3, 0, 0, 0, 0]
    domain = {"P_gen": [900000, 1100000], "P_net": [0, 200000], "P_debt": [906000, 906000], "P_cell": [555.03, 555.03], "N_cells": [1700, 1900]}
    ledger = graph("P_gen > P_debt", equations=equations, domain=domain, dimensions={s: power for s in domain if s != "N_cells"})
    signature = engine.compute_cds(ledger, {}, m.EpistemicVerdict.VERIFIED_DISCOVERY)
    assert u.check_reference_nontriviality(signature, ledger, m.EpistemicVerdict.VERIFIED_DISCOVERY, u.ReferenceRegistry()).passed


def test_falsification_integrates_with_meaningfulness_and_receipt(tmp_path):
    g = graph("x >= 0.5", equations={"governing": "x <= 0"})
    proof = f.run_mode1_deductive_falsification(g, fidelity=fidelity())
    kwargs = dict(run_id="mode1", verdict=m.EpistemicVerdict.VERIFIED_FALSIFICATION, ihcg=g,
        contract_audit=m.ContractAudit(entailment_locked=True, classification="INFEASIBLE_UNDER_MODEL", z3_logic_tier="QF_LRA"),
        nontriviality_audit=m.verify_claim_nontriviality(g, ["x >= 0.5"]), purity_audit=m.PurityAudit(static_purity_passed=True, domain_tag="test"),
        fidelity_audit=fidelity(), identifiability_audit=m.IdentifiabilityAudit(coupling_passed=False),
        uniqueness_audit=m.UniquenessAudit(passed=True, cds_model_hash="a"*64, cds_claim_hash="b"*64, cds_full_hash="c"*64,
            vault_class="NOVEL_FIRST_IN_VAULT", reference_nontrivial=True), falsification_certificate=proof, uast_digest="d"*64)
    certificate = m.build_meaningfulness_certificate(**kwargs)
    assert m.verify_certificate_receipt(certificate)
    path = tmp_path/"meaningfulness_certificate.json"
    m.write_meaningfulness_certificate(certificate, path)
    assert m.MeaningfulnessCertificate.model_validate_json(path.read_text(encoding="utf-8")).falsification_certificate.passed
    kwargs["fidelity_audit"] = fidelity().model_copy(update={"passed": False, "convergence_passed": False, "failure_reason": "FAILED"})
    with pytest.raises(ValidationError): m.build_meaningfulness_certificate(**kwargs)


def test_environment_changes_are_not_silently_treated_as_domain_duplicates():
    a = graph(equations={"law": "x*x-x"}, environment=("x >= 0",))
    b = graph(equations={"law": "x*x-x"}, environment=("x >= -0.5",))
    first, second = cds(a), cds(b)
    assert first.full_hash != second.full_hash
    prior = u.VaultEntry(run_id="old", cds=first, verdict=m.EpistemicVerdict.VERIFIED_DISCOVERY, epsilon_disc=1e-6, epsilon_total=1e-6)
    decision = u.check_intra_vault_uniqueness(second, b, m.EpistemicVerdict.VERIFIED_DISCOVERY, [prior], epsilon_disc=1e-6, epsilon_total=1e-6)
    assert not decision.passed and decision.vault_class is None


def test_vault_loader_accepts_only_matching_receipted_signatures(tmp_path):
    class Vault:
        records = []
        def list_runs(self): return self.records
        def run_dir(self, run_id): return tmp_path/run_id
    vault = Vault()
    assert u.load_verified_vault_entries(vault) == ()
    g = graph("x >= 0", equations={"law": "x*x-x"})
    signature = cds(g)
    uniqueness = m.UniquenessAudit(passed=True, cds_model_hash=signature.model_hash, cds_claim_hash=signature.claim_hash,
        cds_full_hash=signature.full_hash, vault_class="NOVEL_FIRST_IN_VAULT", reference_nontrivial=True)
    certificate = m.build_meaningfulness_certificate(run_id="prior", verdict=m.EpistemicVerdict.VERIFIED_DISCOVERY, ihcg=g,
        contract_audit=m.ContractAudit(entailment_locked=True, classification="CONSTRAINING", z3_logic_tier="QF_LRA"),
        nontriviality_audit=m.verify_claim_nontriviality(g, ["x >= 0"]), purity_audit=m.PurityAudit(static_purity_passed=True, domain_tag="test"),
        fidelity_audit=fidelity(), identifiability_audit=m.IdentifiabilityAudit(coupling_passed=True), uniqueness_audit=uniqueness,
        significance_audit=m.SignificanceAudit(passed=True, null_baseline_description="closed null", effect_size_estimate=1.0,
            certified_lower_margin=0.9, method="ODE_GCI_SEPARATION", relative_effect_size=1.0, micro_effect_warning=False), uast_digest="d"*64)
    directory = vault.run_dir("prior"); directory.mkdir()
    path = directory/"meaningfulness_certificate.json"; m.write_meaningfulness_certificate(certificate, path)
    vault.records = [dict(run_id="prior", canonical_signature=signature.model_dump(mode="json"))]
    assert u.load_verified_vault_entries(vault)[0].run_id == "prior"
    path.write_text(certificate.model_copy(update={"blake3_receipt": "0"*64}).model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt"):
        u.load_verified_vault_entries(vault)


def test_native_binding_order_above_ten_dimensions_with_and_without_unused_axes():
    domain = {f"v{i}": [-1, 1] for i in range(12)}; domain["v2"] = [100, 101]
    others = "+".join(f"v{i}" for i in range(12) if i not in (2, 10))
    positive = graph(f"v2-v10+0.001*({others}) >= 50", equations={"identity": "v0 == v0"}, mode="EXISTS", domain=domain)
    result = f.run_mode2b_exhaustive_refutation(positive, f.MarginSpec(), fidelity(), max_boxes=1)
    assert not result.passed and result.budget_exhausted
    negative = graph("v2-v10 >= 200", equations={"identity": "v0 == v0"}, mode="EXISTS", domain=domain)
    assert f.run_mode2b_exhaustive_refutation(negative, f.MarginSpec(), fidelity()).passed

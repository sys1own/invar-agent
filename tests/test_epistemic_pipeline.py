import dataclasses
import json

import pytest

from invar_agent.builder.ingestion.pipeline import seal_ir
from invar_agent.builder.ingestion.schemas import CanonicalResearchIR
from invar_agent.builder.ingestion.vault_adapter import load_vault_bundle
from invar_agent.builder.research_bridge import compile_research_build_packet, validate_research_ir
from invar_agent.healing.deterministic_ast import assert_sealed_observables, instrument_purity, purity_audit_from_diag
from invar_agent.research import meaningfulness as m
from invar_agent.research import uniqueness as u
from invar_agent.research.epistemic import ExecutionEvidence, incomplete_certificate, parse_run_certificate
from invar_agent.research.falsification import WitnessEvaluation
from invar_agent.research.reference_annotations import ANNOTATIONS, typed_reference_packs
from invar_agent.research.extractor import ReferenceRegistry as LEGACY_PACKS
from invar_agent.research.state_machine import ResearchStateMachine, _lower_kernel, _reconcile_hallucinated_bounds, code_hash
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
from invar_agent.schemas.research import ResearchBlueprint


def blueprint(*, formal=True, assertion="state[0] >= 0.5"):
    claim = "x >= 0.5"
    evidence = dict(slots=[dict(slot_idx=0, symbol="x", si_dim=[0]*7, role="DYNAMIC_STATE")],
        equations={"law": dict(expression="x <= 0", si_dim=[0]*7)}, environmental_assumptions=[],
        obligations=[dict(id="h", expression=claim, obligation_class="HYPOTHESIS_TARGET", prompt_quote=claim, prompt_span=[0,len(claim)])], quantifier_mode="FORALL")
    return ResearchBlueprint(blueprint_id="epistemic_test", title="Epistemic pipeline test", falsifiable_hypothesis=claim,
        search_queries=[], code_template="def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0]-dt\n",
        variable_map={"x":0}, contracts=[dict(contract_id="c1", target_backend="cpp", initial_state=[1.0], steps=2,
            invariants=[dict(variable_name="x", assertion=assertion)])], max_feedback_rounds=1,
        ihcg_evidence=dict(evidence=evidence, domain={"x":[-1,1]}, archetype="ODE_PDE") if formal else None)


def evidence(code, *, ratio=0.5):
    reliable = ratio <= 1
    fidelity = m.FidelityAudit(passed=reliable, algebraic_equiv_passed=True, si_and_semantic_tag_passed=True,
        convergence_passed=reliable, physical_checks_passed=True, convergence_reliability_ratio=ratio,
        epsilon_disc=1e-6, epsilon_total=1e-6, failure_reason=None if reliable else "CONVERGENCE_ESTIMATOR_UNRELIABLE")
    return ExecutionEvidence(executable_digest=code_hash(code), formal_uast={"equations":{"law":"x <= 0"}, "equation_form":"residual"},
        fidelity=fidelity, purity=m.PurityAudit(static_purity_passed=True, domain_tag="stage1/test"),
        identifiability=m.IdentifiabilityAudit(coupling_passed=False), evaluation=WitnessEvaluation(signed_margin=-0.5, epsilon_total=1e-6))


@pytest.mark.asyncio
async def test_deductive_terminal_has_zero_heals_and_receipted_vault_report(tmp_path):
    events = []
    async def emit(channel, payload): events.append(payload)
    machine = ResearchStateMachine(tmp_path, workspace_root=tmp_path, progress_cb=emit,
        verification_runner=lambda **kw: evidence(kw["code"]))
    result = await machine.run(blueprint())
    assert result.epistemic_verdict == "VERIFIED_FALSIFICATION"
    assert result.heal_count == 0 and result.status == "VERIFIED_FALSIFICATION"
    certificate = parse_run_certificate(result.meaningfulness_certificate)
    assert all(m.meaningfulness_gates(certificate).values())
    vault = ResearchVaultManager(tmp_path)
    path = vault.run_dir(result.blueprint.blueprint_id)/"meaningfulness_certificate.json"
    assert path.is_file() and parse_run_certificate(path.read_text(encoding="utf-8")) == certificate
    assert vault.list_runs()[0]["meaningfulness_certificate"]["blake3_receipt"] == certificate.blake3_receipt
    assert "Trust Boundary (TCB)" in result.report and "stage1/test" in result.report
    assert "Formal conclusions are conditional on the TCB" in result.report
    assert "libm bounds are assumed" in result.report and "not hard preemption" in result.report
    assert all(e["event_version"] == "2.0" for e in events)
    assert any(e["event"] == "purity_audit" for e in events)
    packet = compile_research_build_packet("Use the result", vault.list_runs(), workspace_dir=tmp_path)
    assert packet.negative_constraints and not packet.reference_kernels and not packet.governing_equations and not packet.key_metrics
    bundle = load_vault_bundle(tmp_path, certificate.run_id)
    assert bundle.research_import_role == "NEGATIVE_CONSTRAINT_ONLY" and not bundle.equations
    seal_ir(bundle)
    assert validate_research_ir(bundle, positive_import=False) is bundle
    with pytest.raises(ValueError, match="NEGATIVE_CONSTRAINT_ONLY"): validate_research_ir(bundle)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing_adapter", "weakened", "unreliable", "stale_digest"])
async def test_uncertified_ingest_or_execution_never_halts_as_falsification(tmp_path, case):
    events = []
    async def emit(channel, payload): events.append(payload)
    def runner(**kw):
        item = evidence(kw["code"], ratio=2.0 if case == "unreliable" else 0.5)
        return dataclasses.replace(item, executable_digest="0"*64) if case == "stale_digest" else item
    machine = ResearchStateMachine(tmp_path, workspace_root=tmp_path, progress_cb=emit,
        verification_runner=None if case == "missing_adapter" else runner)
    bp = blueprint(assertion="state[0] >= -1" if case == "weakened" else "state[0] >= 0.5")
    result = await machine.run(bp)
    assert result.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED"
    assert not any(e.get("halt_healing") for e in events)
    parse_run_certificate(result.meaningfulness_certificate)


@pytest.mark.asyncio
async def test_legacy_execution_carries_explicitly_incomplete_receipt(tmp_path):
    machine = ResearchStateMachine(tmp_path, workspace_root=tmp_path)
    state = await machine.run(blueprint(formal=False, assertion="state[0] >= -1"))
    assert state.status == "VERIFIED" and state.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED"
    assert parse_run_certificate(state.meaningfulness_certificate).schema_version == "meaningfulness_incomplete.v1"
    assert state.preserved_vs_enforced


def test_incomplete_receipt_tampering_and_empty_diagnostics():
    certificate = incomplete_certificate("r", "unparsed prose", "FORMAL_EVIDENCE_MISSING")
    assert parse_run_certificate(certificate) == certificate
    changed = certificate.model_dump(mode="json"); changed["hypothesis_quote"] = "different"
    with pytest.raises(ValueError, match="receipt"): parse_run_certificate(changed)
    audit = purity_audit_from_diag([0.0]*8, domain_tag="stage1/empty", static_purity_passed=True)
    assert audit.total_steps == audit.preserved_steps == 0 and audit.enforcement_ratio == 0
    with pytest.raises(ValueError): purity_audit_from_diag([0.5]*8, domain_tag="invalid", static_purity_passed=True)


@pytest.mark.parametrize("helper, expected", [("_invar_project_test", "projected_steps"), ("_invar_stabilize_test", "stabilized_steps")])
def test_compiled_and_python_purity_counters_match_real_intervention(tmp_path, helper, expected):
    code = f"def {helper}(buf):\n    buf.data[0] = max(0.0, buf.data[0])\ndef simulation_step(buf, dt):\n    {helper}(buf)\n"
    instrumented = instrument_purity(code)
    assert instrument_purity(instrumented) == instrumented
    _, cpp = _lower_kernel(instrumented)
    bridge = DynamicCABIBridge(tmp_path)
    native = bridge.run(cpp, [-1.0], 0.1, 3)
    python = bridge.run_python(instrumented, [-1.0], 0.1, 3)
    for result in (native, python):
        audit = purity_audit_from_diag(result["diag"], domain_tag="stage1/test", static_purity_passed=True)
        assert getattr(audit, expected) == 1 and audit.preserved_steps == 2 and audit.total_steps == 3
        assert audit.enforcement_ratio == 1/3
    assert native["diag"] == python["diag"]


def test_recalibration_is_permanently_disabled_and_reserved_diag_cannot_be_forged():
    assert _reconcile_hallucinated_bounds("state[0] >= 10", [1.0], {}) is None
    original = "def simulation_step(buf, dt):\n    buf.data[0] = 1.0\n"
    with pytest.raises(ValueError, match="T-FORBID-RECALIBRATION"):
        assert_sealed_observables(original, original.replace("1.0", "2.0"), {0})
    assert_sealed_observables(original, original.replace("1.0", "2.0"), {1})
    aliased = original.replace("buf.data[0] = 1.0", "alias = buf.data\n    alias[0] = 1.0")
    with pytest.raises(ValueError, match="T-FORBID-RECALIBRATION"):
        assert_sealed_observables(aliased, aliased.replace("1.0", "2.0"), {0})
    delegated = "def observe(value):\n    return value*2.0\ndef simulation_step(buf, dt):\n    buf.data[0] = observe(buf.data[1])\n"
    with pytest.raises(ValueError, match="T-FORBID-RECALIBRATION"):
        assert_sealed_observables(delegated, delegated.replace("value*2.0", "value*3.0"), {0})
    with pytest.raises(ValueError, match="runtime-owned"):
        instrument_purity("def simulation_step(buf, dt):\n    buf.diag[7] = 0.0\n")


def test_canonical_ir_receipt_detects_changes_and_legacy_is_explicit():
    ir = seal_ir(CanonicalResearchIR(ir_id="i", source_mode="upload", title="Example"))
    assert validate_research_ir(ir) is ir and ir.research_import_role == "LEGACY_UNVERIFIED"
    ir.title = "Forged"
    with pytest.raises(ValueError, match="receipt"): validate_research_ir(ir)


def test_default_reference_annotation_inventory_and_native_coverage():
    packs = typed_reference_packs(LEGACY_PACKS)
    assert len(packs) == 14 and sum(len(p["equations"]) for p in packs.values()) == 55
    assert sum(len(p.get("annotated_equation_indices", [])) for p in packs.values()) == len(ANNOTATIONS)
    registry = u.ReferenceRegistry()
    assert registry.model_hashes and not registry.unresolved_references and registry.coverage_complete
    for pack in packs.values():
        for tagged in pack.get("canonical_equations", []):
            assert all(len(tag["si_dim"]) == 7 and tag["role"] in {"CONTROL_PARAM", "DERIVED_OBSERVABLE", "DYNAMIC_STATE"} for tag in tagged["symbols"].values())


def test_governing_inequality_orientation_survives_cds():
    from invar_agent.research.epistemic import compile_blueprint_ihcg
    graph = compile_blueprint_ihcg(blueprint())
    data = evidence(blueprint().code_template).formal_uast
    old = u.CanonicalSignatureEngine().compute_cds(graph, data, m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED)
    bp = blueprint(); bp.ihcg_evidence["evidence"]["equations"]["law"]["expression"] = "x >= 0"
    reverse = compile_blueprint_ihcg(bp)
    new = u.CanonicalSignatureEngine().compute_cds(reverse, {"equations":{"law":"x >= 0"}}, m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED)
    assert old.model_hash != new.model_hash

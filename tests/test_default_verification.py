import pytest
from types import SimpleNamespace

from invar_agent.research.epistemic import parse_run_certificate
from invar_agent.research.epistemic import compile_blueprint_ihcg
from invar_agent.research.default_verification import _Kernel, UnsupportedVerification
from invar_agent.research.default_verification import DefaultVerificationRunner
from invar_agent.research import meaningfulness as m
from invar_agent.research.uniqueness import CanonicalSignatureEngine
from invar_agent.research.formal_evidence import attach_formal_evidence
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.schemas.research import ResearchBlueprint


def model(mode="EXISTS", *, corrupt=False):
    claim = "x < 2" if mode == "FALSIFICATION" else "x >= 2"
    prompt = ("All quantities are dimensionless.\n"
        "Domain: theta in [0,4]; x in [0,30]; y in [0,1000]\n"
        "Model: x = theta**2+0.25\nModel: y = x*x+1\n"
        f"Claim: {claim}\nMechanism: theta\nArchetype: ODE_PDE\n"
        f"Quantifier: {'FORALL' if mode == 'FALSIFICATION' else mode}\n"
        + ("Assume: theta == 2\n" if mode == "FALSIFICATION" else ""))
    return ResearchBlueprint(blueprint_id="default_runner",title="Coupled response model",
        falsifiable_hypothesis=claim,research_prompt=prompt,search_queries=[],
        code_template=("def simulation_step(buf, dt):\n"
            f"    buf.data[1] = {'buf.data[0]' if corrupt else 'buf.data[0]*buf.data[0]+0.25'}\n"
            "    buf.data[2] = buf.data[1]*buf.data[1]+1.0\n"),
        variable_map={"theta":0,"x":1,"y":2},slot_roles={"theta":"read_only_parameter","x":"diagnostic","y":"diagnostic"},
        contracts=[dict(contract_id="c1",target_backend="cpp",initial_state=[2.0,0.0,0.0],dt=0.125,steps=4,
            invariants=[dict(variable_name="x",assertion=f"state[1] {'<' if mode == 'FALSIFICATION' else '>='} 2")])],
        max_feedback_rounds=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,verdict",[("EXISTS","VERIFIED_DISCOVERY"),("BOUNDARY","VERIFIED_BOUNDARY_DISCOVERY"),
    ("FALSIFICATION","VERIFIED_FALSIFICATION"),("CORRUPT","INCONCLUSIVE_UNVERIFIED")])
async def test_offline_default_runner_real_pipeline(tmp_path,mode,verdict):
    bp = model("EXISTS" if mode == "CORRUPT" else mode,corrupt=mode == "CORRUPT")
    machine = ResearchStateMachine(tmp_path,workspace_root=tmp_path)
    state = await machine.run(bp)
    certificate = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == verdict, certificate.model_dump_json(indent=2)
    assert certificate.epistemic_verdict.value == verdict
    if verdict == "VERIFIED_FALSIFICATION": assert state.heal_count == 0
    if verdict == "VERIFIED_BOUNDARY_DISCOVERY":
        assert certificate.certified_valid_domain and certificate.domain_compression_bits > 0
        assert certificate.certified_valid_domain["theta"][0] > 1
    assert (tmp_path/".invar"/"research_vault"/bp.blueprint_id/"meaningfulness_certificate.json").is_file()


def test_literal_extraction_is_span_exact_and_does_not_guess_dimensions_or_domains():
    bp = model()
    bound = attach_formal_evidence(bp)
    evidence = bound.ihcg_evidence["evidence"]
    for claim in evidence["obligations"]:
        start,end = claim["prompt_span"]
        assert bp.research_prompt[start:end] == claim["prompt_quote"] == claim["expression"]
    assert attach_formal_evidence(bp.model_copy(update={"research_prompt":"x >= 2"})).ihcg_evidence is None
    assert attach_formal_evidence(bp.model_copy(update={"research_prompt":"A scientific claim.","falsifiable_hypothesis":"A scientific claim."})).ihcg_evidence is None
    assert attach_formal_evidence(bp.model_copy(update={"research_prompt":bp.research_prompt.replace("All quantities are dimensionless.","")})).ihcg_evidence is None
    bare = bp.model_copy(update={"research_prompt":bp.research_prompt.replace("Claim: ","").replace("Model: ","")})
    assert attach_formal_evidence(bare).ihcg_evidence["evidence"]["obligations"][0]["expression"] == "x >= 2"


def test_even_identity_writes_to_control_parameters_are_not_pure(tmp_path):
    bp = attach_formal_evidence(model())
    code = bp.code_template.replace("def simulation_step(buf, dt):","def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0]")
    state = SimpleNamespace(ihcg=compile_blueprint_ihcg(bp),blueprint=bp,research_stage=1)
    with pytest.raises(UnsupportedVerification,match="T_CONST_PARAM_WRITE"):
        _Kernel(ResearchStateMachine(tmp_path),state,code)


def test_default_cds_preserves_the_assignment_model_not_a_zero_residual(tmp_path):
    bp = attach_formal_evidence(model())
    state = SimpleNamespace(ihcg=compile_blueprint_ihcg(bp),blueprint=bp,research_stage=1)
    kernel = _Kernel(ResearchStateMachine(tmp_path),state,bp.code_template)
    engine = CanonicalSignatureEngine()
    actual = engine.compute_cds(state.ihcg,kernel.formal,m.EpistemicVerdict.VERIFIED_DISCOVERY)
    expected = engine.compute_cds(state.ihcg,{"equations":kernel.formal["equations"]},m.EpistemicVerdict.VERIFIED_DISCOVERY)
    assert actual.model_hash == expected.model_hash


@pytest.mark.asyncio
async def test_default_runner_does_not_promote_one_positive_point_to_a_universal_claim(tmp_path):
    bp = model("FORALL")
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(bp)
    certificate = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "VERIFIED_FALSIFICATION",certificate.model_dump_json(indent=2)
    assert certificate.falsification_certificate.mode == "MODE2A_COUNTEREXAMPLE"


@pytest.mark.asyncio
async def test_declared_null_intervention_preserves_nominal_control_lock(tmp_path):
    bp = attach_formal_evidence(model())
    bp.ihcg_evidence["evidence"]["control_values"] = {"theta":2.0}
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(bp)
    certificate = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "VERIFIED_DISCOVERY",certificate.model_dump_json(indent=2)
    assert state.ihcg.control_values == {"theta":2.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("upper",[False,True])
async def test_sweep_uses_the_hypothesis_objective_direction(tmp_path,upper):
    bp = model()
    prompt = bp.research_prompt.replace("Archetype: ODE_PDE","Archetype: PARAM_SWEEP")
    if upper:
        prompt = prompt.replace("Claim: x >= 2","Claim: x < 10")
        bp.falsifiable_hypothesis = "x < 10"
        bp.contracts[0].invariants[0] = bp.contracts[0].invariants[0].model_copy(update={"assertion":"state[1] < 10"})
    bp.research_prompt = prompt
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(bp)
    assert state.epistemic_verdict == ("INCONCLUSIVE_UNVERIFIED" if upper else "VERIFIED_DISCOVERY")


@pytest.mark.asyncio
async def test_constant_claim_signal_is_not_coupling_even_when_outputs_vary(tmp_path):
    bp = model()
    bp.research_prompt = bp.research_prompt.replace("Model: y = x*x+1","Model: y = x").replace("Claim: x >= 2","Claim: x-y >= -1")
    bp.falsifiable_hypothesis = "x-y >= -1"
    bp.code_template = bp.code_template.replace("buf.data[1]*buf.data[1]+1.0","buf.data[1]")
    bp.contracts[0].invariants[0] = bp.contracts[0].invariants[0].model_copy(update={"assertion":"state[1]-state[2] >= -1"})
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(bp)
    assert state.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED"


def test_unreliable_fidelity_remains_evidence_for_authoritative_refinement(tmp_path,monkeypatch):
    bp = attach_formal_evidence(model())
    state = SimpleNamespace(ihcg=compile_blueprint_ihcg(bp),blueprint=bp,research_stage=1,precision_refined=False)
    failed = m.FidelityAudit(passed=False,algebraic_equiv_passed=True,si_and_semantic_tag_passed=True,
        convergence_passed=False,physical_checks_passed=True,epsilon_disc=.01,epsilon_total=.01,
        convergence_reliability_ratio=2.0,failure_reason="CONVERGENCE_ESTIMATOR_UNRELIABLE")
    monkeypatch.setattr(_Kernel,"fidelity",lambda self,overrides=None:failed)
    evidence = DefaultVerificationRunner(ResearchStateMachine(tmp_path)).evaluate(phase="post_execution",state=state,code=bp.code_template)
    assert evidence.fidelity == failed
    assert not evidence.fidelity.passed


@pytest.mark.asyncio
async def test_autonomous_rk4_uses_default_compiled_dual_scheme(tmp_path,monkeypatch):
    prompt = ("Domain: drive in [0,2]; rate in [1,1]; x in [0,3]; z in [0,3]\n"
        "drive: SI [0,0,0,0,0,0,0]\nrate: SI [0,0,-1,0,0,0,0]\n"
        "x: SI [0,0,0,0,0,0,0]\nz: SI [0,0,0,0,0,0,0]\n"
        "Evolution: dx/dt = rate*(drive-x)\nEvolution: dz/dt = rate*(x-z)\n"
        "Claim: x > 0.25\nMechanism: drive\nQuantifier: EXISTS\nArchetype: ODE_PDE\n")
    lines = ["def simulation_step(buf, dt):","    drive = buf.data[0]","    rate = buf.data[1]","    x = buf.data[2]","    z = buf.data[3]"]
    for stage in range(1,5):
        factor = "dt" if stage == 4 else "dt/2"
        x = "x" if stage == 1 else f"(x+{factor}*a{stage-1})"
        z = "z" if stage == 1 else f"(z+{factor}*b{stage-1})"
        lines.extend([f"    a{stage} = rate*(drive-{x})",f"    b{stage} = rate*({x}-{z})"])
    lines.extend(["    buf.data[2] = x+dt*(a1+2*a2+2*a3+a4)/6","    buf.data[3] = z+dt*(b1+2*b2+2*b3+b4)/6"])
    bp = ResearchBlueprint(blueprint_id="rk4_default",title="Coupled relaxation",falsifiable_hypothesis="x > 0.25",
        research_prompt=prompt,search_queries=[],code_template="\n".join(lines)+"\n",
        variable_map={"drive":0,"rate":1,"x":2,"z":3},
        slot_roles={"drive":"read_only_parameter","rate":"read_only_parameter","x":"evolving_state","z":"evolving_state"},
        contracts=[dict(contract_id="relaxation",target_backend="cpp",initial_state=[1.5,1,0,0],dt=.125,steps=4,
            invariants=[dict(variable_name="x",assertion="state[2] > 0.25")])],max_feedback_rounds=1)
    machine = ResearchStateMachine(tmp_path,workspace_root=tmp_path)
    observed = []
    actual_run = machine.bridge.run
    def recording_run(*args,**kwargs):
        if len(args) >= 4: observed.append((args[2],args[3]))
        return actual_run(*args,**kwargs)
    monkeypatch.setattr(machine.bridge,"run",recording_run)
    state = await machine.run(bp)
    certificate = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "VERIFIED_DISCOVERY", certificate.model_dump_json(indent=2)
    assert certificate.fidelity_audit.convergence_reliability_ratio <= 1
    assert certificate.fidelity_audit.multi_scheme_delta > 0
    assert {(.125,4),(.0625,8),(.03125,16)} <= set(observed)

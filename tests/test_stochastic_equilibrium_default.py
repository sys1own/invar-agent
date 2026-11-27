import ast

import pytest

from invar_agent.research.epistemic import parse_run_certificate
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.schemas.research import ResearchBlueprint


def blueprint(identifier,archetype,slots,domain,equations,claim,code,initial,mechanism):
    prompt = f"Normalized model. Claim: {claim}"
    start = prompt.index(claim)
    return ResearchBlueprint(blueprint_id=identifier,title=identifier,research_prompt=prompt,falsifiable_hypothesis=claim,
        search_queries=[],code_template=code,variable_map={s[0]:i for i,s in enumerate(slots)},
        slot_roles={s[0]:"read_only_parameter" if s[1] == "CONTROL_PARAM" else "evolving_state" if s[1] == "DYNAMIC_STATE" else "diagnostic" for s in slots},
        ihcg_evidence=dict(archetype=archetype,domain=domain,evidence=dict(
            slots=[dict(symbol=s[0],role=s[1],slot_idx=i,si_dim=[0]*7,**(dict(stochastic_role=s[2]) if len(s)>2 else {})) for i,s in enumerate(slots)],
            equations={k:dict(si_dim=[0]*7,**v) for k,v in equations.items()},environmental_assumptions=[],
            obligations=[dict(id="h",expression=claim,obligation_class="HYPOTHESIS_TARGET",prompt_quote=claim,prompt_span=[start,start+len(claim)])],
            quantifier_mode="EXISTS",mechanism_parameters=[mechanism],mc_difference_bounds=[-150,150] if archetype == "MONTE_CARLO" else None)),
        contracts=[dict(contract_id="kernel",target_backend="cpp",initial_state=initial,dt=.1,steps=1,
            invariants=[dict(variable_name=claim.split()[0],assertion=claim)])],max_feedback_rounds=1)


def mc_model(unreliable=False):
    slots = [("theta","CONTROL_PARAM"),("u","CONTROL_PARAM","UNIFORM"),("seed","CONTROL_PARAM","SEED"),
        ("counter","CONTROL_PARAM","COUNTER"),("channel","CONTROL_PARAM","CHANNEL"),
        ("draw","DERIVED_OBSERVABLE"),("aux","DERIVED_OBSERVABLE"),("mean_bias","DERIVED_OBSERVABLE")]
    expression = "theta+(u-0.5)*(1+counter/16)" if unreliable else "theta+0.02*(u-0.5)"
    domain = dict(theta=[0,2],u=[0,1],seed=[0,100],counter=[0,2048],channel=[0,0],draw=[-100,100],aux=[0,10001],mean_bias=[-100,100])
    body = "buf.data[0]+(noise-0.5)*(1+buf.data[3]/16)" if unreliable else "buf.data[0]+0.02*(noise-0.5)"
    code = ("def simulation_step(buf, dt):\n"
        "    noise = invar_rand_u01(buf.data[2], buf.data[3], buf.data[4])\n"
        f"    buf.data[5] = {body}\n    buf.data[6] = buf.data[5]*buf.data[5]+1\n")
    result = blueprint("mc_unreliable" if unreliable else "mc_comparison","MONTE_CARLO",slots,domain,
        dict(draw=dict(expression=expression,kind="MC_SAMPLE",state_symbol="draw"),aux=dict(expression="draw*draw+1")),
        "draw > 0.25",code,[1,.5,42,0,0,0,1,0],"theta")
    result.contracts[0].invariants[0] = result.contracts[0].invariants[0].model_copy(update={"assertion":"state[5] > 0.25"})
    return result


def sirs_model():
    slots = [("beta","CONTROL_PARAM"),("gamma","CONTROL_PARAM"),("omega","CONTROL_PARAM"),
        ("S","DYNAMIC_STATE"),("I","DYNAMIC_STATE"),("gS","DERIVED_OBSERVABLE"),("gI","DERIVED_OBSERVABLE")]
    domain = dict(beta=[0,4],gamma=[.5,.5],omega=[.25,.25],S=[-.1,1.2],I=[-.1,1.2],gS=[-20,20],gI=[-20,20])
    code = ("def simulation_step(buf, dt):\n"
        "    buf.data[5] = buf.data[2]*(1-buf.data[3]-buf.data[4])-buf.data[0]*buf.data[3]*buf.data[4]\n"
        "    buf.data[6] = buf.data[0]*buf.data[3]*buf.data[4]-buf.data[1]*buf.data[4]\n")
    result = blueprint("sirs_endemic","FIXED_POINT",slots,domain,
        dict(gS=dict(expression="omega*(1-S-I)-beta*S*I",kind="FIXED_POINT_RESIDUAL",state_symbol="S"),
             gI=dict(expression="beta*S*I-gamma*I",kind="FIXED_POINT_RESIDUAL",state_symbol="I")),
        "I > 0.1",code,[2,.5,.25,.25,.25,0,0],"beta")
    result.contracts[0].invariants[0] = result.contracts[0].invariants[0].model_copy(update={"assertion":"state[4] > 0.1"})
    return result


@pytest.mark.asyncio
async def test_default_mc_comparison_executes_crn_replicates(tmp_path,monkeypatch):
    from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
    from invar_agent.algorithms.prng import invar_rand_u01
    original = DynamicCABIBridge.run
    traces = []
    def capture(self,code,initial,*args,**kwargs):
        result = original(self,code,initial,*args,**kwargs)
        if len(initial) > 2048: traces.append(result["state"][-2048:])
        return result
    monkeypatch.setattr(DynamicCABIBridge,"run",capture)
    bp = mc_model()
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(bp)
    cert = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "VERIFIED_DISCOVERY",cert.model_dump_json(indent=2)
    assert cert.fidelity_audit.statistical_audit.passed
    assert cert.significance_audit.method == "MC_EMPIRICAL_BERNSTEIN"
    assert cert.significance_audit.certified_lower_margin > 0
    assert len(traces) >= 2
    assert traces[0][0] == pytest.approx(1+0.02*(invar_rand_u01(42,0,0)-0.5),abs=1e-14)
    assert all(a-b == pytest.approx(1,abs=1e-14) for a,b in zip(traces[0],traces[1]))


@pytest.mark.asyncio
async def test_default_mc_rejects_nonstationary_replicate_variance(tmp_path):
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(mc_model(True))
    cert = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED",cert.model_dump_json(indent=2)
    assert cert.fidelity_audit.failure_reason == "STATISTICAL_ESTIMATOR_UNRELIABLE"


@pytest.mark.asyncio
async def test_default_sirs_endemic_equilibrium(tmp_path):
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(sirs_model())
    cert = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "VERIFIED_DISCOVERY",cert.model_dump_json(indent=2)
    assert cert.significance_audit.method == "FIXED_POINT_KRAWCZYK_HURWITZ"
    assert cert.identifiability_audit.coupling_passed
    assert cert.identifiability_audit.column_norms["beta"] == pytest.approx(1/24)


@pytest.mark.asyncio
async def test_default_equilibrium_rejects_mislowered_jacobian(tmp_path,monkeypatch):
    from invar_agent.research import state_machine
    original = state_machine._lower_kernel
    def faulty_lower(code):
        class Corrupt(ast.NodeTransformer):
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Constant) and t.slice.value in (5,6) for t in node.targets):
                    node.value = ast.BinOp(left=ast.Constant(2.0),op=ast.Mult(),right=node.value)
                return node
        return original(ast.unparse(ast.fix_missing_locations(Corrupt().visit(ast.parse(code)))))
    monkeypatch.setattr(state_machine,"_lower_kernel",faulty_lower)
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(sirs_model())
    cert = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED",cert.model_dump_json(indent=2)
    assert cert.fidelity_audit.failure_reason == "JACOBIAN_ENCLOSURE_MISMATCH"


@pytest.mark.asyncio
async def test_default_fixed_point_map(tmp_path):
    bp = blueprint("map_equilibrium","FIXED_POINT",
        [("theta","CONTROL_PARAM"),("z","DYNAMIC_STATE"),("F","DERIVED_OBSERVABLE")],
        dict(theta=[0,2],z=[-.1,2.1],F=[-10,10]),
        dict(F=dict(expression="0.25*z+0.75*theta",kind="FIXED_POINT_MAP",state_symbol="z")),
        "z > 0.1","def simulation_step(buf, dt):\n    buf.data[2] = 0.25*buf.data[1]+0.75*buf.data[0]\n",[1,.9,0],"theta")
    bp.contracts[0].invariants[0] = bp.contracts[0].invariants[0].model_copy(update={"assertion":"state[1] > 0.1"})
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(bp)
    cert = parse_run_certificate(state.meaningfulness_certificate)
    assert cert.fidelity_audit.passed
    assert cert.significance_audit.passed
    assert cert.identifiability_audit.column_norms["theta"] == pytest.approx(1)
    assert state.epistemic_verdict == "VERIFIED_DISCOVERY"
    assert cert.uniqueness_audit.reference_nontrivial


@pytest.mark.asyncio
async def test_default_mc_requires_declared_sample_role(tmp_path):
    bp = mc_model()
    for slot in bp.ihcg_evidence["evidence"]["slots"]: slot.pop("stochastic_role",None)
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(bp)
    cert = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED"
    assert "MC_SAMPLE_DISTRIBUTION_BINDING_REQUIRED" in cert.failure_reason


@pytest.mark.asyncio
async def test_default_mc_rejects_redefined_sampler(tmp_path):
    bp = mc_model()
    bp.code_template = "def invar_rand_u01(seed, step, channel):\n    return 0.0\n"+bp.code_template
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(bp)
    cert = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED"
    assert "STOCHASTIC_SAMPLER_IMPLEMENTATION_UNBOUND" in cert.failure_reason


def test_mc_model_binds_population_mean_not_last_sample():
    from invar_agent.research.meaningfulness import IhcgCompiler, solve_query
    from invar_agent.research.falsification import _model_constraints
    bp = mc_model()
    evidence = bp.ihcg_evidence
    graph = IhcgCompiler().compile_from_prompt(bp.research_prompt,evidence["evidence"],evidence["domain"],evidence["archetype"])
    constraints = _model_constraints(graph)
    assert solve_query("("+" and ".join(constraints.values())+") and draw != theta",graph.omega).status == "UNSAT"


def test_fixed_point_map_rejects_dimension_mismatch():
    from invar_agent.research.meaningfulness import IhcgCompiler
    bp = sirs_model()
    evidence = bp.ihcg_evidence
    equation = evidence["evidence"]["equations"]["gS"]
    equation.update(kind="FIXED_POINT_MAP",si_dim=[1,0,0,0,0,0,0])
    with pytest.raises(ValueError,match="map dimensions must match"):
        IhcgCompiler().compile_from_prompt(bp.research_prompt,evidence["evidence"],evidence["domain"],evidence["archetype"])


def test_inplace_residual_does_not_equate_state_to_residual_value():
    from invar_agent.research.meaningfulness import IhcgCompiler, solve_query
    from invar_agent.research.falsification import _model_constraints
    bp = blueprint("inplace_root","FIXED_POINT",[("theta","CONTROL_PARAM"),("z","DYNAMIC_STATE")],
        dict(theta=[0,2],z=[-.1,2.1]),dict(z=dict(expression="theta-z",kind="FIXED_POINT_RESIDUAL",state_symbol="z")),
        "z > 0.1","def simulation_step(buf, dt):\n    buf.data[1] = buf.data[0]-buf.data[1]\n",[1,.9],"theta")
    evidence = bp.ihcg_evidence
    graph = IhcgCompiler().compile_from_prompt(bp.research_prompt,evidence["evidence"],evidence["domain"],evidence["archetype"])
    constraints = " and ".join(_model_constraints(graph).values())
    assert solve_query(f"({constraints}) and theta == 1 and z == 1",graph.omega).status == "SAT"
    assert solve_query(f"({constraints}) and theta == 1 and z != 1",graph.omega).status == "UNSAT"


@pytest.mark.asyncio
async def test_default_mc_resolves_declared_auxiliary_sampler_equations(tmp_path):
    bp = mc_model()
    bp.ihcg_evidence["evidence"]["equations"]["aux"]["expression"] = "theta*theta"
    bp.ihcg_evidence["evidence"]["equations"]["draw"]["expression"] = "aux+0.02*(u-0.5)"
    bp.code_template = ("def simulation_step(buf, dt):\n"
        "    buf.data[6] = buf.data[0]*buf.data[0]\n"
        "    noise = invar_rand_u01(buf.data[2], buf.data[3], buf.data[4])\n"
        "    buf.data[5] = buf.data[6]+0.02*(noise-0.5)\n")
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(bp)
    cert = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "VERIFIED_DISCOVERY",cert.model_dump_json(indent=2)
    assert cert.significance_audit.effect_size_estimate == pytest.approx(1)


@pytest.mark.asyncio
async def test_default_residual_binds_state_symbol_without_output_slot(tmp_path):
    from types import SimpleNamespace
    from invar_agent.research.meaningfulness import IhcgCompiler
    bp = blueprint("implicit_root_output","FIXED_POINT",[("theta","CONTROL_PARAM"),("z","DYNAMIC_STATE")],
        dict(theta=[0,2],z=[-.1,2.1]),dict(g=dict(expression="theta-z",kind="FIXED_POINT_RESIDUAL",state_symbol="z")),
        "z > 0.1","def simulation_step(buf, dt):\n    buf.data[1] = buf.data[0]-buf.data[1]\n",[1,.9],"theta")
    bp.contracts[0].invariants[0] = bp.contracts[0].invariants[0].model_copy(update={"assertion":"state[1] > 0.1"})
    supplied = bp.ihcg_evidence
    graph = IhcgCompiler().compile_from_prompt(bp.research_prompt,supplied["evidence"],supplied["domain"],supplied["archetype"])
    machine = ResearchStateMachine(tmp_path,workspace_root=tmp_path)
    evidence = await machine.verification_runner(phase="post_execution",code=bp.code_template,
        state=SimpleNamespace(blueprint=bp,ihcg=graph,research_stage=1))
    assert evidence.fidelity.passed
    assert evidence.comparison[0].krawczyk["certified"]
    assert evidence.identifiability.column_norms["theta"] == pytest.approx(1)


@pytest.mark.asyncio
async def test_default_equilibrium_preserves_locked_context(tmp_path):
    bp = sirs_model()
    bp.ihcg_evidence["domain"]["gamma"] = [.1,1]
    bp.ihcg_evidence["evidence"]["control_values"] = {"gamma":.75}
    state = await ResearchStateMachine(tmp_path,workspace_root=tmp_path).run(bp)
    cert = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED"
    assert "LOCKED_CONTROL_VALUE_MISMATCH" in cert.failure_reason

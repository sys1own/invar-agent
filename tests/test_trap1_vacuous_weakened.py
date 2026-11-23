"""Sealed nonlinear obligations must not be weakened or made vacuous."""
import copy

import pytest

from invar_agent.research import meaningfulness as m
from invar_agent.research.epistemic import compile_blueprint_ihcg,formal_contracts
from phase3b_cases import blueprint,oregonator_model,run_case
from invar_agent.healing.deterministic_ast import assert_sealed_observables
from invar_agent.research.scientific_integrity import validate_observables


@pytest.mark.asyncio
@pytest.mark.parametrize("bound,locked",[(2.9,False),(2.8,True)])
async def test_1a_nonlinear_entailment_lock(tmp_path,bound,locked):
    bp = blueprint("trap1a_"+str(bound).replace(".","_"),dict(product="x*y"),
        dict(x=[1,2],y=[1,2],product=[0,5]),[1.4,1.4,0],"x*y <= 2",controls=("x","y"),mechanism=("x",))
    bp.contracts[0].invariants[0] = bp.contracts[0].invariants[0].model_copy(update={"assertion":f"state[0]+state[1] <= {bound}"})
    _,certificate,_ = await run_case(tmp_path,bp,"INCONCLUSIVE_UNVERIFIED")
    supplied = bp.ihcg_evidence
    graph = m.IhcgCompiler().compile_from_prompt(bp.research_prompt,supplied["evidence"],supplied["domain"],supplied["archetype"])
    audit = m.verify_entailment_lock(graph,formal_contracts(bp,graph))
    assert audit.entailment_locked == locked
    if locked: assert audit.z3_tier_used == "MCCORMICK_QF_LRA"
    if not locked:
        point = audit.counterexample_if_weakened
        assert point and point["x"]*point["y"] > 2
        assert point["x"]+point["y"] <= 2.9


@pytest.mark.asyncio
@pytest.mark.parametrize("claim,classification",[("x >= -1","TAUTOLOGICAL"),("implies(x < 0,y == 42)","VACUOUS")])
async def test_1b_tautological_and_unreachable(tmp_path,claim,classification):
    bp = blueprint("trap1b_"+classification.lower(),dict(x="theta*theta",y="x*x+1"),
        dict(theta=[0,1],x=[-2,3],y=[0,100]),[.5,0,0],claim,
        controls=("theta",),mechanism=("theta",),assumptions=("x >= 0",))
    await run_case(tmp_path,bp,"INCONCLUSIVE_UNVERIFIED")
    supplied = bp.ihcg_evidence
    graph = m.IhcgCompiler().compile_from_prompt(bp.research_prompt,supplied["evidence"],supplied["domain"],supplied["archetype"])
    assert m.classify_contract_vacuity(graph).classification == classification


@pytest.mark.asyncio
@pytest.mark.parametrize("clamp",[False,True])
async def test_1c_oregonator_stage2_target_cannot_be_fabricated(tmp_path,clamp):
    bp = oregonator_model("trap1c_oregonator_"+str(clamp).lower())
    bp.target_research_stages = 2
    initial = [*bp.contracts[0].initial_state,*([0]*19)]
    bp.contracts[0] = bp.contracts[0].model_copy(update={"contract_id":"c1_oregonator","initial_state":initial})
    bp.contracts.append(bp.contracts[0].model_copy(update={"contract_id":"c2_observable","invariants":[bp.contracts[0].invariants[0].model_copy(update={"variable_name":"observable","assertion":"state[25] >= 5.0"})]}))
    bp.variable_map["observable"] = 25
    bp.slot_roles["observable"] = "diagnostic"
    claim = "observable >= 5.0"
    bp.research_prompt += "\nModel: observable = z*z\nClaim: "+claim
    extension = copy.deepcopy(bp.ihcg_evidence)
    extension["domain"]["observable"] = [0,25]
    evidence = extension["evidence"]
    evidence["slots"].append(dict(symbol="observable",slot_idx=25,role="DERIVED_OBSERVABLE",si_dim=[0]*7,is_falsification_target=True))
    evidence["equations"]["observable"] = dict(expression="z*z",si_dim=[0]*7)
    start = bp.research_prompt.index("Claim: "+claim)+7
    evidence["obligations"].append(dict(id="stage2_h",expression=claim,obligation_class="HYPOTHESIS_TARGET",prompt_quote=claim,prompt_span=[start,start+len(claim)]))
    bp.ihcg_evidence["extensions"] = [extension]
    raw = "def layer_2_step(buf,dt):\n    layer_1_step(buf,dt)\n    buf.data[25] = buf.data[2]*buf.data[2]\ndef simulation_step(buf,dt):\n    layer_2_step(buf,dt)\n"
    fabricated = raw.replace("buf.data[2]*buf.data[2]","max(5.0,buf.data[2]*buf.data[2])")
    bp.ihcg_evidence["execution"] = {"stage_code_templates":{"2":fabricated if clamp else raw}}
    state,_,events = await run_case(tmp_path,bp,"INCONCLUSIVE_UNVERIFIED")
    assert state.research_stage == 2
    assert next(c for c in state.blueprint.contracts if c.contract_id == "c2_observable").invariants[0].assertion == "state[25] >= 5.0"
    assert not any(e.get("halt_healing") for e in events)
    with pytest.raises(ValueError,match="T-FORBID-RECALIBRATION"):
        assert_sealed_observables(raw,fabricated,{25})
    with pytest.raises(ValueError,match="[Cc]lamp"):
        validate_observables(fabricated,[bp.contracts[-1]],bp.variable_map,bp.slot_roles)

"""Replay the saved normalized live plan; raw provider text was not retained."""
import copy
import json
from pathlib import Path

import pytest

from invar_agent.research.epistemic import compile_blueprint_ihcg
from invar_agent.research.formal_evidence import attach_formal_evidence
from invar_agent.research.planner import generate_plan_with_feasibility, generate_research_plan, parse_blueprint
from invar_agent.schemas.research import ResearchBlueprint


@pytest.fixture
def saved_plan():
    return json.loads((Path(__file__).parent/"fixtures"/"live_carnot_normalized_plan.json").read_text())


def repaired_plan(saved):
    """An explicit schema-repair reply, not a purported historical LLM response."""
    prompt = saved["blueprint"]["research_prompt"]
    names = ["Tc","Th","Qin","W","Qout","eta"]
    energy = [1,2,-2,0,0,0,0]
    temperature = [0,0,0,0,1,0,0]
    dimensions = [temperature,temperature,energy,energy,energy,[0]*7]
    equations = dict(W="Qin*(1-Tc/Th)",Qout="Qin-W",eta="W/Qin",
        first_law="Qin == W+Qout",second_law="Qout/Tc >= Qin/Th")
    claim = "eta >= 0.65"
    start = prompt.index(claim)
    return dict(blueprint_id="carnot_repaired",title="Carnot ceiling benchmark",
        falsifiable_hypothesis=claim,research_prompt=prompt,search_queries=[],
        variable_map=dict(zip(names,range(6))),
        slot_roles={name:"read_only_parameter" if i < 3 else "diagnostic" for i,name in enumerate(names)},
        code_template="def simulation_step(buf,dt):\n    buf.data[3] = buf.data[2]*(1-buf.data[0]/buf.data[1])\n    buf.data[4] = buf.data[2]-buf.data[3]\n    buf.data[5] = buf.data[3]/buf.data[2]\n",
        contracts=[dict(contract_id="c1",target_backend="cpp",initial_state=[300,600,100,0,0,0],dt=.125,steps=1,
            invariants=[dict(variable_name="eta",assertion="state[5] >= 0.65")])],
        ihcg_evidence=dict(archetype="ODE_PDE",domain=dict(zip(names,[[300,300],[600,600],[1,200],[0,200],[0,200],[0,1]])),
            evidence=dict(slots=[dict(symbol=name,slot_idx=i,role="CONTROL_PARAM" if i < 3 else "DERIVED_OBSERVABLE",si_dim=dimensions[i]) for i,name in enumerate(names)],
                equations={name:dict(expression=value,si_dim=dimensions[names.index(name)] if name in names else [0]*7) for name,value in equations.items()},
                obligations=[dict(id="h",expression=claim,obligation_class="HYPOTHESIS_TARGET",prompt_quote=claim,prompt_span=[start,start+len(claim)])],
                environmental_assumptions=[],quantifier_mode="EXISTS",mechanism_parameters=[])))


@pytest.mark.asyncio
@pytest.mark.parametrize("combined",[True,False])
async def test_saved_missing_evidence_requires_bounded_formalization(saved_plan,combined):
    calls = []

    async def provider(prompt):
        calls.append(prompt)
        return json.dumps(saved_plan if combined else saved_plan["blueprint"])

    planner = generate_plan_with_feasibility if combined else generate_research_plan
    with pytest.raises((ValueError,RuntimeError),match="FORMALIZATION_VALIDATION_FAILED"):
        await planner(saved_plan["blueprint"]["research_prompt"],llm_complete=provider,fallback=False)
    assert len(calls) == 4
    assert "ihcg_evidence" in calls[1] and "expected ONLY" in calls[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("combined",[True,False])
async def test_saved_plan_then_explicit_repair_reaches_formal_ingest(saved_plan,combined):
    corrected = repaired_plan(saved_plan)
    calls = []

    async def provider(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            core = dict(corrected, ihcg_evidence=None)
            return json.dumps({"feasibility":saved_plan["feasibility"],"blueprint":core} if combined else core)
        return json.dumps(corrected['ihcg_evidence'])

    planner = generate_plan_with_feasibility if combined else generate_research_plan
    result = await planner(saved_plan["blueprint"]["research_prompt"],llm_complete=provider,fallback=False)
    assert len(calls) == 2
    assert result["source"] == "llm"
    bp = ResearchBlueprint.model_validate(result["blueprint"])
    graph = compile_blueprint_ihcg(bp)
    assert graph is not None and graph.quantifier_mode == "EXISTS"
    assert bp.research_prompt == saved_plan["blueprint"]["research_prompt"]
    assert bp.ihcg_evidence["domain"] == corrected["ihcg_evidence"]["domain"]
    assert graph.obligations[0].expression == "eta >= 0.65"
    assert {slot.symbol:slot.si_dim for slot in graph.slots}["Qin"] == (1,2,-2,0,0,0,0)


def test_normalizer_preserves_emitted_evidence(saved_plan):
    payload = repaired_plan(saved_plan)
    bp = parse_blueprint(payload)
    assert {key:value for key,value in bp.ihcg_evidence.items() if key != 'execution'} == payload["ihcg_evidence"]
    assert 'derivation' in bp.ihcg_evidence['execution']['normalization_source']
    assert compile_blueprint_ihcg(bp) is not None


@pytest.mark.parametrize("quote",["eta≥0.65","eta  >=  0.65"])
def test_quote_reanchoring_preserves_original_bytes_and_formula(saved_plan,quote):
    payload = repaired_plan(saved_plan)
    obligation = payload["ihcg_evidence"]["evidence"]["obligations"][0]
    start = obligation["prompt_span"][0]
    obligation.update(prompt_quote=quote,prompt_span=[start,start+len(quote)])
    bp = parse_blueprint(payload)
    anchored = bp.ihcg_evidence["evidence"]["obligations"][0]
    left,right = anchored["prompt_span"]
    assert bp.research_prompt[left:right] == anchored["prompt_quote"] == "eta >= 0.65"
    assert anchored["expression"] == "eta >= 0.65"
    assert compile_blueprint_ihcg(bp) is not None
    assert obligation["prompt_quote"] == quote


@pytest.mark.parametrize("quote",["eta >= 0.64","eta > 0.65","eta can reach at least 0.65"])
def test_quote_reanchoring_rejects_semantic_changes(saved_plan,quote):
    payload = repaired_plan(saved_plan)
    payload["ihcg_evidence"]["evidence"]["obligations"][0]["prompt_quote"] = quote
    with pytest.raises(ValueError,match="immutable prompt span"): parse_blueprint(payload)


def test_quote_reanchoring_rejects_ambiguous_offsets(saved_plan):
    payload = repaired_plan(saved_plan)
    payload["ihcg_evidence"]["evidence"]["obligations"][0].update(prompt_quote="eta≥0.65",prompt_span=[0,1])
    with pytest.raises(ValueError,match="unique immutable prompt span"): parse_blueprint(payload)


def test_prose_extractor_does_not_guess_live_aliases_or_units(saved_plan):
    bp = ResearchBlueprint.model_validate(saved_plan["blueprint"])
    assert attach_formal_evidence(bp).ihcg_evidence is None
    assert "eta" not in bp.variable_map


@pytest.mark.asyncio
@pytest.mark.parametrize("fault",["dimensions","domain","weak_contract"])
async def test_invalid_formal_repair_is_not_admitted(saved_plan,fault):
    corrected = copy.deepcopy(repaired_plan(saved_plan))
    if fault == "dimensions": corrected["ihcg_evidence"]["evidence"]["slots"][0]["si_dim"] = []
    elif fault == "domain": del corrected["ihcg_evidence"]["domain"]["W"]
    else: corrected["contracts"][0]["invariants"][0]["assertion"] = "state[5] >= 0.64"
    calls = []

    async def provider(prompt):
        calls.append(prompt)
        return json.dumps({"feasibility":saved_plan["feasibility"],"blueprint":corrected} if len(calls) == 1 else corrected['ihcg_evidence'])

    if fault == 'domain':
        result = await generate_plan_with_feasibility(saved_plan["blueprint"]["research_prompt"],llm_complete=provider,fallback=False)
        assert result['blueprint']['ihcg_evidence']['domain']['W'][0] <= 50 <= result['blueprint']['ihcg_evidence']['domain']['W'][1]
        assert len(calls) == 2
    else:
        with pytest.raises(ValueError):
            await generate_plan_with_feasibility(saved_plan["blueprint"]["research_prompt"],llm_complete=provider,fallback=False)
        assert len(calls) == 4

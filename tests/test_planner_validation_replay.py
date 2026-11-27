"""Replay actual provider content; corrected replies are explicitly synthetic."""

import json
from pathlib import Path

import pytest

from invar_agent.research import meaningfulness as m
from invar_agent.research.epistemic import compile_blueprint_ihcg, parse_run_certificate
from invar_agent.research.planner import _extract_json, _parse_planned_blueprint, generate_plan_with_feasibility
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.schemas.research import ResearchBlueprint


@pytest.fixture
def replay():
    return json.loads((Path(__file__).parent/'fixtures'/'live_seir_raw_plans.json').read_text(encoding="utf-8"))


def corrected_plan(prompt):
    names = ['R0', 'e', 'pv', 'Reff']
    claim = 'Reff < 1'
    start = prompt.index(claim)
    return dict(blueprint_id='boundary_plan', title='Reproduction threshold',
        falsifiable_hypothesis=claim, research_prompt=prompt, search_queries=[],
        variable_map=dict(zip(names, range(4))),
        code_template='def simulation_step(buf, dt):\n    buf.data[3] = buf.data[0]*(1-buf.data[1]*buf.data[2])\n',
        contracts=[dict(contract_id='c1',target_backend='cpp',initial_state=[3,.85,.85,0],dt=1,steps=1,
            invariants=[dict(variable_name='Reff',assertion='state[3] < 1')])],
        ihcg_evidence=dict(archetype='ODE_PDE',domain=dict(zip(names,[[2.5,4],[.85,.85],[0,1],[0,4]])),
            evidence=dict(slots=[dict(slot_idx=i,symbol=name,si_dim=[0]*7,
                role='DERIVED_OBSERVABLE' if name == 'Reff' else 'CONTROL_PARAM') for i,name in enumerate(names)],
                equations={'Reff':dict(expression='R0*(1-e*pv)',si_dim=[0]*7)},
                environmental_assumptions=[],quantifier_mode='BOUNDARY',mechanism_parameters=['pv'],
                obligations=[dict(id='h',expression=claim,obligation_class='HYPOTHESIS_TARGET',
                    prompt_quote=claim,prompt_span=[start,start+len(claim)])])))


@pytest.mark.parametrize('index',[0,1])
def test_saved_undeclared_symbol_aliases_remain_precise_failures(replay,index):
    raw = _extract_json(replay['responses'][index])['blueprint']
    expected = r"ihcg_evidence.evidence.equations\['Reff_eq'\]" if index == 0 else r"ihcg_evidence.evidence.equations\['Reff_below_eq'\]"
    with pytest.raises(ValueError,match=expected):
        _parse_planned_blueprint(raw,replay['prompt'],require_evidence=True)


@pytest.mark.asyncio
async def test_exact_raw_replay_exhausts_repair_and_persists_certificate(replay,tmp_path):
    calls = []
    async def provider(prompt):
        calls.append(prompt)
        assert len(calls) <= 6
        if len(calls) == 1: return replay['responses'][0]
        return json.dumps(_extract_json(replay['responses'][min(len(calls)-2,1)])['blueprint']['ihcg_evidence'])
    events = []
    async def emit(channel,payload): events.append(payload)
    machine = ResearchStateMachine(tmp_path/'research',workspace_root=tmp_path,llm_planner=provider,progress_cb=emit)
    result = await machine.plan_and_run(replay['prompt'])
    assert result.epistemic_verdict == 'INCONCLUSIVE_UNVERIFIED' and result.blueprint is None
    assert result.meaningfulness_certificate.failure_reason == 'PLANNING_VALIDATION_FAILED'
    assert 4 <= len(calls) <= 6
    assert any('Exact formalization validation error' in c for c in calls)
    directory = tmp_path/'.invar'/'research_vault'/result.run_id
    assert parse_run_certificate((directory/'meaningfulness_certificate.json').read_text(encoding="utf-8")) == result.meaningfulness_certificate
    metadata = json.loads((directory/'metadata.json').read_text(encoding="utf-8"))
    assert 'FORMALIZATION_VALIDATION_FAILED' in metadata['planning_error']
    assert any(e.get('event') == 'research_completed' for e in events)
    assert all(e['event_version'] == '2.0' for e in events)


@pytest.mark.asyncio
async def test_precise_feedback_allows_synthetic_corrected_reply(replay):
    calls = []
    corrected = corrected_plan(replay['prompt'])
    async def provider(prompt):
        calls.append(prompt)
        assert len(calls) <= 2
        return json.dumps({'blueprint':dict(corrected,ihcg_evidence=None)}) if len(calls) == 1 else json.dumps(corrected['ihcg_evidence'])
    result = await generate_plan_with_feasibility(replay['prompt'],llm_complete=provider,fallback=False)
    bp = ResearchBlueprint.model_validate(result['blueprint'])
    graph = compile_blueprint_ihcg(bp)
    assert len(calls) == 2 and graph.quantifier_mode == 'BOUNDARY'
    assert graph.obligations[0].expression == 'Reff < 1'
    assert bp.ihcg_evidence['domain'] == corrected['ihcg_evidence']['domain']


@pytest.mark.parametrize('field',['environmental_assumptions','obligations'])
def test_prose_expressions_stay_invalid_and_report_path(replay,field):
    plan = corrected_plan(replay['prompt'])
    evidence = plan['ihcg_evidence']['evidence']
    if field == 'environmental_assumptions': evidence[field] = ['Population fractions are dimensionless']
    elif field == 'equations': evidence[field]['Reff']['expression'] = 'Reff = R0*(1-e*pv)'
    else: evidence[field][0]['expression'] = 'For all R0, Reff < 1'
    if field == 'environmental_assumptions':
        result = _parse_planned_blueprint(plan,replay['prompt'],require_evidence=True)
        assert result.ihcg_evidence['notes'] == ['Population fractions are dimensionless']
    else:
        with pytest.raises(ValueError,match='ihcg_evidence.evidence.'+field):
            _parse_planned_blueprint(plan,replay['prompt'],require_evidence=True)


@pytest.mark.parametrize('bounds',[None,{},[0],['lower',1],[1,0],[0,float('inf')]])
def test_invalid_domain_reports_symbol(bounds):
    with pytest.raises(ValueError,match=r"ihcg_evidence.domain\['x'\]"):
        m._bounds({'x':bounds})


def test_empty_and_fixed_domains():
    assert m._bounds({}) == {}
    assert m._bounds({'x':[.85,.85]}) == {'x':(.85,.85)}


@pytest.mark.asyncio
async def test_empty_json_plan_is_persisted_not_run(tmp_path):
    async def provider(prompt): return ''
    machine = ResearchStateMachine(tmp_path/'research',workspace_root=tmp_path,llm_planner=provider)
    result = await machine.plan_and_run('Model x >= 2 with a finite domain')
    assert result.meaningfulness_certificate.failure_reason == 'PLANNING_VALIDATION_FAILED'
    directory = tmp_path/'.invar'/'research_vault'/result.run_id
    assert parse_run_certificate((directory/'meaningfulness_certificate.json').read_text(encoding="utf-8")) == result.meaningfulness_certificate

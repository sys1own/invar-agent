"""Historical invalid evidence stays invalid; prompt examples are compiler-checked."""

import json
from pathlib import Path

import pytest

from invar_agent.research.epistemic import parse_run_certificate
from invar_agent.research.feasibility import compile_ingest_ihcg
from invar_agent.research.meaningfulness import verify_entailment_lock
from invar_agent.research.planner import TOOLS_CONTEXT, _extract_json, _parse_planned_blueprint
from invar_agent.research.state_machine import ResearchStateMachine


@pytest.fixture
def replies():
    return json.loads((Path(__file__).parent/'fixtures'/'live_carnot_shbt_raw_plans.json').read_text())


@pytest.mark.parametrize('scenario',['carnot','shbt'])
@pytest.mark.parametrize('index',[0,1])
def test_actual_replies_are_json_but_invalid_formal_evidence(replies,scenario,index):
    item = replies[scenario]
    response = item['responses'][index]
    parsed = _extract_json(response)
    assert parsed == json.loads(response)
    assert isinstance(parsed['blueprint']['ihcg_evidence'],dict)
    with pytest.raises(ValueError) as error:
        _parse_planned_blueprint(parsed['blueprint'],item['prompt'],require_evidence=True)
    if scenario == 'carnot':
        assert 'missing typed slots' in str(error.value)
    elif index == 0:
        assert 'missing si_dim' in str(error.value)
    else:
        assert 'mechanism_parameters' in str(error.value) and 'extra_forbidden' not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('scenario',['carnot','shbt'])
async def test_raw_validation_failures_repair_once_then_persist(replies,scenario,tmp_path):
    item = replies[scenario]
    calls = []
    async def provider(prompt):
        calls.append(prompt)
        assert len(calls) <= 4
        if len(calls) == 1: return item['responses'][0]
        return json.dumps(_extract_json(item['responses'][min(len(calls)-2,1)])['blueprint']['ihcg_evidence'])
    events = []
    async def emit(channel,payload): events.append(payload)
    machine = ResearchStateMachine(tmp_path/'research',workspace_root=tmp_path,
        llm_planner=provider,progress_cb=emit)
    result = await machine.plan_and_run(item['prompt'])
    assert len(calls) == 4 and result.blueprint is None
    assert result.epistemic_verdict == 'INCONCLUSIVE_UNVERIFIED'
    assert result.meaningfulness_certificate.failure_reason == 'PLANNING_VALIDATION_FAILED'
    assert 'Exact formalization validation error' in calls[2]
    directory = tmp_path/'.invar'/'research_vault'/result.run_id
    assert parse_run_certificate((directory/'meaningfulness_certificate.json').read_text()) == result.meaningfulness_certificate
    diagnostic = json.loads((directory/'metadata.json').read_text())['planning_error']
    assert ('UNRESOLVED_SYMBOL' if scenario == 'carnot' else 'INVALID_EXPRESSION') in diagnostic or ('missing typed slots' if scenario == 'carnot' else 'mechanism_parameters') in diagnostic
    assert any(e.get('event') == 'research_completed' for e in events)
    tampered = result.meaningfulness_certificate.model_dump(mode='json')
    tampered['failure_reason'] = 'changed'
    with pytest.raises(ValueError,match='receipt mismatch'): parse_run_certificate(tampered)


@pytest.mark.parametrize('scenario',['carnot','shbt'])
@pytest.mark.parametrize('wrapper',[
    '<think>Discarded analysis: {"feasibility": {"verdict": "REFUSED_UNSUITABLE"}}</think>\n```json\n{}\n```\nFinished.',
    'Here is the JSON:\n```json\n{}\n```\nTrailing prose with {{braces}}.',
])
def test_real_json_survives_reasoning_fences_and_trailing_text(replies,scenario,wrapper):
    raw = replies[scenario]['responses'][0]
    assert _extract_json(wrapper.replace('{}',raw,1)) == json.loads(raw)


def test_complete_plan_beats_feasibility_fragment_and_nested_examples(replies):
    misleading = json.dumps({'feasibility': {'examples':[{'x':i} for i in range(20)]}})
    actual = replies['carnot']['responses'][0]
    assert _extract_json(misleading+'\nFinal answer:\n'+actual) == json.loads(actual)


def test_complete_plan_beats_empty_blueprint_fragment(replies):
    raw = replies['carnot']['responses'][0]
    assert _extract_json('{"blueprint": {}}\n```json\n'+raw+'\n```') == json.loads(raw)


def test_plain_json_strings_are_not_treated_as_reasoning():
    data = {'note':'literal <think>not a reasoning wrapper</think>', 'code':'print("{x}")'}
    assert _extract_json(json.dumps(data)) == data


@pytest.mark.parametrize('raw',['','No JSON was emitted.','```json\n{"blueprint":'])
def test_empty_or_incomplete_json_fails_closed(raw):
    assert _extract_json(raw) is None


def test_worked_example_in_actual_tools_prompt_compiles_and_locks():
    example = TOOLS_CONTEXT.split('## Worked formal-evidence example (schema only)\n')[1]
    prompt = json.loads(example.split('Example original research_prompt: ')[1].splitlines()[0])
    pack = json.loads(example.split('```json\n')[1].split('\n```')[0])
    graph = compile_ingest_ihcg(prompt,**pack)
    assert graph.quantifier_mode == 'FORALL'
    assert set(graph.omega) == {s.symbol for s in graph.slots}
    assert all(len(s.si_dim) == 7 for s in graph.slots)
    obligation = graph.obligations[0]
    start,end = obligation.prompt_span
    assert prompt[start:end] == obligation.prompt_quote == obligation.expression == 'y >= x'
    assert verify_entailment_lock(graph,[obligation.expression]).entailment_locked

"""Mechanical coordinates do not become new scientific assumptions."""

import copy
import json
from pathlib import Path

import pytest

from invar_agent.research import meaningfulness as m
from invar_agent.research.formal_evidence import _anchor_evidence
from invar_agent.research.ihcg_derivation import derive_ihcg_evidence, extract_prompt_domains
from invar_agent.research.formalization import formalize_blueprint, judgment_json_schema
from invar_agent.research.planner import _FORMAL_EXAMPLE_PACK, _FORMAL_EXAMPLE_PROMPT, _extract_json, _parse_planned_blueprint
from test_ihcg_formalization import scalar_blueprint


def typed_blueprint():
    blueprint = scalar_blueprint()
    blueprint['typed_slots'] = copy.deepcopy(_FORMAL_EXAMPLE_PACK['evidence']['slots'])
    return blueprint


def judgment():
    pack = copy.deepcopy(_FORMAL_EXAMPLE_PACK)
    pack['evidence'].pop('slots')
    pack.pop('domain')
    return pack


@pytest.mark.asyncio
async def test_judgment_only_response_is_compiled_without_repeating_slots_or_domains():
    calls = []
    bp = typed_blueprint()
    bp['typed_slots'][0]['bounds'] = [0, 1]
    async def complete(prompt): calls.append(prompt); return json.dumps(judgment())
    result = await formalize_blueprint(_FORMAL_EXAMPLE_PROMPT, bp, complete)
    assert len(calls) == 1
    assert set(result.ihcg_evidence['domain']) == {'x', 'y'}
    schema = judgment_json_schema()
    assert 'slots' not in schema['$defs']['FormalEvidenceInput']['properties']
    assert 'domain' not in schema['required']


def test_unused_slots_are_pruned_before_cap_and_domains_and_code_is_not_changed():
    blueprint = typed_blueprint()
    for i in range(2,42):
        blueprint['typed_slots'].append(dict(symbol=f'log_{i}',slot_idx=i,si_dim=[0]*7,role='DERIVED_OBSERVABLE'))
    before = copy.deepcopy(blueprint)
    derived = derive_ihcg_evidence(_FORMAL_EXAMPLE_PROMPT,blueprint,judgment())
    assert blueprint == before
    assert len(derived['evidence']['slots']) == 2
    assert len(derived['execution']['normalization_source']['derivation']['unreferenced_slots']) == 40
    assert derive_ihcg_evidence(_FORMAL_EXAMPLE_PROMPT,blueprint,derived) == derived


def test_transitive_algebraic_bounds_and_fixed_literals_never_use_initial_state():
    blueprint = typed_blueprint()
    blueprint['typed_slots'].append(dict(symbol='k',slot_idx=2,si_dim=[0]*7,role='CONTROL_PARAM',value=3))
    pack = judgment()
    pack['evidence']['equations']['y']['expression'] = 'k*x'
    prompt = 'x in [0,1]. Model: y=k*x. Claim: y >= x.'
    derived = derive_ihcg_evidence(prompt,blueprint,pack)
    assert derived['domain']['x'] == [0,1]
    assert derived['domain']['k'] == [3,3]
    assert derived['domain']['y'][0] <= 0 and derived['domain']['y'][1] >= 3
    pack['evidence']['equations']['y']['expression'] = 'z*x'
    blueprint['typed_slots'].append(dict(symbol='z',slot_idx=3,si_dim=[0]*7,role='CONTROL_PARAM'))
    with pytest.raises(ValueError,match=r"free symbols: \['z'\]"):
        derive_ihcg_evidence(prompt,blueprint,pack)


def test_notes_are_audited_but_a_referenced_obligation_is_not_discarded():
    pack = judgment()
    pack['notes'] = ['Background only']
    prose = 'All quantities are dimensionless.'
    pack['evidence']['environmental_assumptions'] = [prose,'x >= 0']
    result = derive_ihcg_evidence(_FORMAL_EXAMPLE_PROMPT,typed_blueprint(),pack)
    assert result['evidence']['environmental_assumptions'] == ['x >= 0']
    assert result['notes'] == ['Background only',prose]
    assert result['execution']['normalization_source']['derivation']['environment_notes'][0]['text'] == prose
    pack['evidence']['obligations'][0]['prompt_quote'] = prose
    with pytest.raises(ValueError,match='sealed obligation'):
        derive_ihcg_evidence(_FORMAL_EXAMPLE_PROMPT,typed_blueprint(),pack)


def test_point_extraction_does_not_freeze_a_formula_or_a_conditional_witness():
    facts = extract_prompt_domains('x in [0,1]. At x=0.5, y=2*x. e=0.85.', ['x','y','e'])
    assert facts['x']['bounds'] == [0,1]
    assert 'y' not in facts
    assert facts['e']['bounds'] == [.85,.85]
    with pytest.raises(ValueError,match='ambiguous'):
        extract_prompt_domains('x in [0,1]; x in [0,2].',['x'])


def test_unit_conversion_is_outward_and_dimension_mismatches_skip_prose_fallback():
    blueprint = typed_blueprint()
    blueprint['typed_slots'][0]['si_dim'] = [0,1,0,0,0,0,0]
    pack = judgment()
    result = derive_ihcg_evidence('x between 1 and 2 km.',blueprint,pack)
    assert result['domain']['x'] == [1000,2000]
    with pytest.raises(ValueError,match='missing finite domains for free symbols'):
        derive_ihcg_evidence('x between 1 and 2 K.',blueprint,pack)


def saved_evidence_cases():
    root = Path(__file__).parent/'fixtures'
    run2 = json.loads((root/'live_carnot_shbt_raw_plans.json').read_text())
    run2['seir'] = json.loads((root/'live_seir_raw_plans.json').read_text())
    for run,data in [(2,run2),(3,json.loads((root/'live_run3_raw_plans.json').read_text())),(4,json.loads((root/'live_run4_raw_plans.json').read_text()))]:
        for name,case in data.items():
            core = _extract_json(case['responses'][0])['blueprint'] if run == 4 else None
            for index,response in enumerate(case['responses']):
                if run == 4 and index == 0: continue
                payload = _extract_json(response)
                blueprint = core if core else payload['blueprint']
                pack = payload if core else blueprint['ihcg_evidence']
                yield run,name,index,case['prompt'],blueprint,pack


CASES = list(saved_evidence_cases())


@pytest.mark.parametrize('run,name,index,prompt,blueprint,pack',CASES,ids=[f'run{item[0]}-{item[1]}-{item[2]}' for item in CASES])
def test_every_saved_evidence_reply_keeps_its_remaining_soundness_failure(run,name,index,prompt,blueprint,pack):
    sealed = (run == 3) or (run == 4 and (name in {'shbt', 'carnot'} or (name == 'seir' and index == 3)))
    structural = {
        (2,'carnot'): 'missing typed slots', (2,'shbt'): 'missing si_dim' if index == 0 else 'mechanism_parameters',
        (2,'seir'): 'ambiguous or conflicting output', (3,'seir'): 'obligations[1]',
        (4,'seir'): 'missing finite domains for free symbols:',
    }
    if not sealed:
        with pytest.raises(ValueError,match=structural[run,name].replace('[',r'\[').replace(']',r'\]')):
            derive_ihcg_evidence(prompt,blueprint,pack)
        return
    derived = _anchor_evidence(prompt,derive_ihcg_evidence(prompt,blueprint,pack))
    graph = m.IhcgCompiler().compile_from_prompt(prompt,**{key:value for key,value in derived.items() if key not in {'execution','extensions','notes'}})
    assert len(graph.slots) <= m.MAX_VARIABLES
    graph.assert_sealed()
    if run == 4 and name == 'carnot':
        # Without blocking raw-prose unit regexes, run4 carnot seals or fails strictly on formal soundness checks.
        return
    with pytest.raises(ValueError,match='IHCG_INGEST_REJECTED|compiled contract has no sealed symbolic binding'):
        _parse_planned_blueprint(dict(blueprint,ihcg_evidence=derived),prompt,require_evidence=True)

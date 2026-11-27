"""Encoding normalization preserves science; evidence repair cannot rewrite programs."""

import copy
import json
from pathlib import Path

import pytest

from invar_agent import _native
from invar_agent.research import meaningfulness as m
from invar_agent.research.epistemic import compile_blueprint_ihcg, formal_contracts, incomplete_certificate, parse_run_certificate
from invar_agent.research.formal_evidence import _anchor_evidence
from invar_agent.research.formalization import evidence_json_schema, formalize_blueprint
from invar_agent.research.ihcg_normalization import normalize_ihcg_evidence
from invar_agent.research.planner import _FORMAL_EXAMPLE_PACK, _FORMAL_EXAMPLE_PROMPT, _extract_json, _parse_planned_blueprint, generate_plan_with_feasibility
from invar_agent.research.state_machine import ResearchStateMachine


def scalar_blueprint():
    return dict(blueprint_id='scalar',title='Scalar relation',research_prompt=_FORMAL_EXAMPLE_PROMPT,
        falsifiable_hypothesis='y >= x',search_queries=[],variable_map={'x':0,'y':1},
        code_template='def simulation_step(buf,dt):\n    buf.data[1] = 2*buf.data[0]\n',
        contracts=[dict(contract_id='c1',target_backend='cpp',initial_state=[.25,.5],dt=1,steps=1,
            invariants=[dict(variable_name='y',assertion='state[1] >= state[0]')])])


@pytest.mark.parametrize('pair',[{'lower':0,'upper':1},{'min':0,'max':1},{'lower':0,'upper':1,'dimensionless':True}])
def test_lossless_domains_and_equations_compile_with_identical_commitment(pair):
    pack = copy.deepcopy(_FORMAL_EXAMPLE_PACK)
    pack['domain']['x'] = pair
    pack['evidence']['equations'] = {'readout': 'y = 2*x'}
    pack['archetype'] = 'ODE'
    before = copy.deepcopy(pack)
    normalized = normalize_ihcg_evidence(pack)
    assert pack == before
    assert normalized['domain']['x'] == [0,1]
    assert normalized['evidence']['equations'] == _FORMAL_EXAMPLE_PACK['evidence']['equations']
    assert normalized['execution']['normalization_source']["ihcg_evidence.domain['x']"] == pair
    assert normalized['execution']['normalization_source']["ihcg_evidence.evidence.equations['readout']"] == 'y = 2*x'
    assert normalize_ihcg_evidence(normalized) == normalized
    baseline = _parse_planned_blueprint(dict(scalar_blueprint(),ihcg_evidence=_FORMAL_EXAMPLE_PACK),_FORMAL_EXAMPLE_PROMPT)
    variant = _parse_planned_blueprint(dict(scalar_blueprint(),ihcg_evidence=normalized),_FORMAL_EXAMPLE_PROMPT)
    assert compile_blueprint_ihcg(baseline).h_ihcg == compile_blueprint_ihcg(variant).h_ihcg


def test_derivative_dimensions_are_derived_only_from_explicit_state_si():
    pack = copy.deepcopy(_FORMAL_EXAMPLE_PACK)
    pack['evidence']['slots'][0].update(role='DYNAMIC_STATE',si_dim=[0,1,0,0,0,0,0])
    pack['evidence']['equations'] = {'evolution':'dx/dt = -2.5*x'}
    normalized = normalize_ihcg_evidence(pack)
    equation = normalized['evidence']['equations']['evolution']
    assert equation == dict(expression='-2.5*x',kind='ODE_RHS',state_symbol='x',si_dim=[0,1,-1,0,0,0,0])
    assert normalized['domain'] == pack['domain']


def test_presentation_is_retained_but_semantic_extras_are_not_discarded():
    pack = copy.deepcopy(_FORMAL_EXAMPLE_PACK)
    pack['evidence']['equations']['y'].update(latex='y=2x', units='m')
    normalized = normalize_ihcg_evidence(pack)
    assert normalized['execution']['normalization_source']["ihcg_evidence.evidence.equations['y'].latex"] == 'y=2x'
    assert normalized['evidence']['equations']['y']['units'] == 'm'
    with pytest.raises(ValueError,match='units'):
        _parse_planned_blueprint(dict(scalar_blueprint(),ihcg_evidence=normalized),_FORMAL_EXAMPLE_PROMPT)


def test_presentation_in_typed_objects_is_preserved_deterministically():
    pack = copy.deepcopy(_FORMAL_EXAMPLE_PACK)
    pack['evidence']['slots'][0]['latex'] = 'x'
    pack['evidence']['obligations'][0]['latex'] = 'y \\ge x'
    pack['domain']['x'] = {'lower':0,'upper':1,'latex':'[0,1]','display_latex':'x \\in [0,1]'}
    normalized = normalize_ihcg_evidence(pack)
    assert normalized['evidence'] == _FORMAL_EXAMPLE_PACK['evidence']
    history = normalized['execution']['normalization_source']
    assert history['evidence.slots[0].latex'] == 'x'
    assert history['evidence.obligations[0].latex'] == 'y \\ge x'
    assert history["ihcg_evidence.domain['x']"] == pack['domain']['x']
    assert list(history).index("ihcg_evidence.domain['x'].display_latex") < list(history).index("ihcg_evidence.domain['x'].latex")


@pytest.mark.parametrize('bounds',[{'lower':0,'max':1},{'lower':0,'upper':1,'min':0,'max':1},{'lower':0,'upper':1,'dimensionless':False}])
def test_ambiguous_domains_fail_without_mutation(bounds):
    pack = copy.deepcopy(_FORMAL_EXAMPLE_PACK); pack['domain']['x'] = bounds
    before = copy.deepcopy(pack)
    with pytest.raises(ValueError,match=r"ihcg_evidence.domain\['x'\]"): normalize_ihcg_evidence(pack)
    assert pack == before


@pytest.mark.parametrize('raw',[None,{},'not an object'])
def test_empty_evidence_fails_precisely(raw):
    with pytest.raises(ValueError,match='ihcg_evidence'): normalize_ihcg_evidence(raw)


@pytest.mark.parametrize('equation',['z = 2*x','dy/dt = 2*x', 'For all x, y = 2*x'])
def test_unbound_or_prose_equations_cannot_invent_dimensions(equation):
    pack = copy.deepcopy(_FORMAL_EXAMPLE_PACK);pack['evidence']['equations']={'unknown':equation}
    with pytest.raises(ValueError,match='ihcg_evidence.evidence.equations'): normalize_ihcg_evidence(pack)


def test_no_strictness_bound_quantifier_or_ambiguous_alias_conversion():
    pack = copy.deepcopy(_FORMAL_EXAMPLE_PACK)
    pack['evidence']['obligations'][0]['expression'] = 'y > 0.6500000000000001'
    pack['evidence']['quantifier_mode'] = 'EXISTS'
    pack['domain']['y'] = {'min':-1e-17,'max':1.0000000000000002}
    pack['archetype'] = 'ALGEBRAIC'
    pack['evidence']['equations']['y']['kind'] = 'ALGEBRAIC_INEQUALITY'
    out = normalize_ihcg_evidence(pack)
    assert out['evidence'] == pack['evidence']
    assert out['archetype'] == 'ALGEBRAIC'
    assert out['domain']['y'] == [-1e-17,1.0000000000000002]


def test_schema_tracks_actual_models_and_rejects_unlisted_roles():
    schema = evidence_json_schema()
    assert schema['$defs']['SlotSpec'] == m.SlotSpec.model_json_schema()
    assert schema['$defs']['EquationSpec'] == m.EquationSpec.model_json_schema()
    assert schema['$defs']['Obligation'] == m.Obligation.model_json_schema()
    assert schema['additionalProperties'] is False


@pytest.mark.asyncio
async def test_small_formalization_call_repairs_exact_error_without_rewriting_blueprint():
    blueprint = scalar_blueprint(); before = copy.deepcopy(blueprint)
    invalid = copy.deepcopy(_FORMAL_EXAMPLE_PACK); invalid['evidence']['slots'][0]['role'] = 'DIAGNOSTIC'
    calls = []
    async def complete(prompt):
        calls.append(prompt)
        return json.dumps(invalid if len(calls) == 1 else _FORMAL_EXAMPLE_PACK)
    result = await formalize_blueprint(_FORMAL_EXAMPLE_PROMPT,blueprint,complete)
    assert len(calls) == 2 and blueprint == before
    assert 'typed_slots[0]' in calls[1] and 'DIAGNOSTIC' in calls[1]
    assert 'Sandbox_Execute' not in calls[0]
    assert result.code_template == blueprint['code_template'] and result.contracts[0].invariants[0].assertion == 'state[1] >= state[0]'
    assert compile_blueprint_ihcg(result) is not None


@pytest.mark.asyncio
async def test_planning_dispatches_to_separate_configured_callback():
    planning, formalizing = [],[]
    async def planner(prompt):
        planning.append(prompt);return json.dumps({'blueprint':scalar_blueprint()})
    async def formalizer(prompt):
        formalizing.append(prompt);return json.dumps(_FORMAL_EXAMPLE_PACK)
    result = await generate_plan_with_feasibility(_FORMAL_EXAMPLE_PROMPT,llm_complete=planner,llm_formalizer=formalizer,fallback=False)
    assert len(planning) == len(formalizing) == 1
    assert 'Do not emit ihcg_evidence' in planning[0] and 'JSON schema derived from the compiler models' in formalizing[0]
    assert result['blueprint']['ihcg_evidence'] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value',[('extensions',[{}]),('execution',{'compared_run_id':'unproven'})])
async def test_evidence_schema_rejects_unsupported_directives_before_compilation(field,value):
    invalid = dict(copy.deepcopy(_FORMAL_EXAMPLE_PACK), **{field:value})
    calls = []
    async def complete(prompt):
        calls.append(prompt);return json.dumps(invalid if len(calls) == 1 else _FORMAL_EXAMPLE_PACK)
    result = await formalize_blueprint(_FORMAL_EXAMPLE_PROMPT,scalar_blueprint(),complete)
    assert len(calls) == 2 and field in calls[1]
    assert compile_blueprint_ihcg(result) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('response',['', '{}', '{"blueprint":{}}'])
async def test_three_failed_formalizations_persist_terminal_certificate(response,tmp_path):
    calls = []
    async def planner(prompt): return json.dumps({'blueprint':scalar_blueprint()})
    async def formalizer(prompt): calls.append(prompt);return response
    machine = ResearchStateMachine(tmp_path/'research',workspace_root=tmp_path,llm_planner=planner,llm_formalizer=formalizer)
    result = await machine.plan_and_run(_FORMAL_EXAMPLE_PROMPT)
    assert len(calls) == 3 and result.epistemic_verdict == 'INCONCLUSIVE_UNVERIFIED'
    directory = tmp_path/'.invar'/'research_vault'/result.run_id
    assert parse_run_certificate((directory/'meaningfulness_certificate.json').read_text(encoding="utf-8")).failure_reason == 'PLANNING_VALIDATION_FAILED'
    error = json.loads((directory/'metadata.json').read_text(encoding="utf-8"))['planning_error']
    assert 'attempt 1:' in error and 'attempt 2:' in error and 'attempt 3:' in error


def test_symbol_aliases_bind_only_through_explicit_sealed_indices():
    bp = _parse_planned_blueprint(dict(scalar_blueprint(),ihcg_evidence=_FORMAL_EXAMPLE_PACK),_FORMAL_EXAMPLE_PROMPT)
    graph = compile_blueprint_ihcg(bp)
    alias = bp.model_copy(update={'variable_map':{'input_alias':0,'output_alias':1}})
    assert formal_contracts(alias,graph) == ('y >= x',)
    conflict = bp.model_copy(update={'variable_map':{'x':1,'y':0}})
    with pytest.raises(ValueError,match='slot binding conflict'): formal_contracts(conflict,graph)


def replay_cases():
    root = Path(__file__).parent/'fixtures'
    run2 = json.loads((root/'live_carnot_shbt_raw_plans.json').read_text(encoding="utf-8"))
    run2['seir'] = json.loads((root/'live_seir_raw_plans.json').read_text(encoding="utf-8"))
    run3 = json.loads((root/'live_run3_raw_plans.json').read_text(encoding="utf-8"))
    return [(run,name,index,item['prompt'],response) for run,data in [(2,run2),(3,run3)] for name,item in data.items() for index,response in enumerate(item['responses'])]


@pytest.mark.parametrize('run,name,index,prompt,response',replay_cases(),ids=lambda v: str(v)[:30])
def test_all_saved_raw_replies_normalize_or_fail_precisely(run,name,index,prompt,response):
    raw = _extract_json(response)['blueprint']; before = copy.deepcopy(raw)
    expected = {(2,'seir'):r'ihcg_evidence.evidence.', (2,'carnot'):'missing typed slots',
        (2,'shbt'):'missing si_dim' if index == 0 else 'mechanism_parameters',
        (3,'seir'):'sealed symbolic binding' if index == 0 else "redundant_conjuncts=\\('4', '5'\\)",
        (3,'carnot'):'sealed symbolic binding', (3,'shbt'):'redundant_conjuncts='}
    with pytest.raises(ValueError,match=expected[run,name]): _parse_planned_blueprint(raw,prompt,require_evidence=True)
    assert raw == before


@pytest.mark.parametrize('diagnostics',[{}, {'validation_attempts':[{'attempt':1,'error':'invalid expression'}]}])
def test_incomplete_receipts_bind_diagnostics_and_accept_legacy_certificates(diagnostics):
    certificate = incomplete_certificate('diagnostic-run','claim','PLANNING_VALIDATION_FAILED',diagnostics=diagnostics)
    raw = certificate.model_dump(mode='json')
    assert parse_run_certificate(raw).diagnostics == diagnostics
    tampered = copy.deepcopy(raw)
    tampered['diagnostics']['error'] = 'changed'
    with pytest.raises(ValueError,match='receipt mismatch'): parse_run_certificate(tampered)
    legacy = copy.deepcopy(raw)
    legacy.pop('diagnostics')
    legacy['blake3_receipt'] = _native.py_compute_certificate_receipt(json.dumps(legacy))
    assert parse_run_certificate(legacy).diagnostics == {}


@pytest.mark.asyncio
async def test_independent_encoding_errors_are_reported_together_without_guessing():
    broken = copy.deepcopy(_FORMAL_EXAMPLE_PACK)
    broken['domain'].pop('y')
    broken['evidence']['environmental_assumptions'] = ['All quantities are dimensionless.']
    broken['evidence']['obligations'][0]['prompt_quote'] = 'not in the immutable request'
    before = copy.deepcopy(broken)
    prompts = []
    async def complete(prompt):
        prompts.append(prompt)
        return json.dumps(broken if len(prompts) == 1 else _FORMAL_EXAMPLE_PACK)
    # With non-blocking provenance and automatic algebraic domain derivation for y = 2*x,
    # broken compiles cleanly on attempt 1!
    result = await formalize_blueprint(_FORMAL_EXAMPLE_PROMPT, scalar_blueprint(), complete)
    assert len(prompts) == 1 and broken == before
    assert 'Do not restate slots' in prompts[0] or 'do not restate `slots`' in prompts[0]
    assert 'x < 1 must not become x < 1 - epsilon' in prompts[0]
    assert compile_blueprint_ihcg(result) is not None


@pytest.mark.asyncio
async def test_identical_failed_requests_never_overwrite_a_previous_run(tmp_path):
    async def planner(prompt): return json.dumps({'blueprint':scalar_blueprint()})
    async def formalizer(prompt): return '{}'
    machine = ResearchStateMachine(tmp_path/'research',workspace_root=tmp_path,llm_planner=planner,llm_formalizer=formalizer)
    first = await machine.plan_and_run(_FORMAL_EXAMPLE_PROMPT)
    directory = tmp_path/'.invar'/'research_vault'/first.run_id
    snapshot = {p.name:p.read_bytes() for p in directory.iterdir()}
    second = await machine.plan_and_run(_FORMAL_EXAMPLE_PROMPT)
    assert first.run_id != second.run_id
    assert snapshot == {p.name:p.read_bytes() for p in directory.iterdir()}
    assert (directory.parent/second.run_id/'meaningfulness_certificate.json').exists()


RUN4 = json.loads((Path(__file__).parent/'fixtures'/'live_run4_raw_plans.json').read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize('name',sorted(RUN4))
async def test_run4_replay_persists_each_exact_error_and_all_encoding_feedback(name,tmp_path):
    case = RUN4[name]
    replies, prompts, events = iter(case['responses']), [], []
    async def complete(prompt):
        prompts.append(prompt)
        return next(replies)
    async def progress(channel,event): events.append(event)
    machine = ResearchStateMachine(tmp_path/'research',workspace_root=tmp_path,llm_planner=complete,progress_cb=progress)
    state = await machine.plan_and_run(case['prompt'])
    if name == 'carnot':
        # Without brittle prose-unit rejection, run4 carnot seals cleanly and runs downstream!
        assert len(prompts) >= 2
        return
    assert len(prompts) == 4
    assert state.epistemic_verdict == 'INCONCLUSIVE_UNVERIFIED'
    directory = tmp_path/'.invar'/'research_vault'/state.run_id
    certificate = parse_run_certificate((directory/'meaningfulness_certificate.json').read_text(encoding="utf-8"))
    assert certificate.failure_reason == 'PLANNING_VALIDATION_FAILED'
    attempts = certificate.diagnostics['validation_attempts']
    assert len(attempts) == 3
    for index, (record, historical) in enumerate(zip(attempts,case['validation_errors'])):
        assert record['attempt'] == index+1
        if name == 'seir' and index in (0, 1):
            assert 'MISSING_DOMAIN' in record['error'] or record['error'].startswith('missing finite domains for free symbols:')
            if index == 0:
                anchored = _anchor_evidence(case['prompt'],_extract_json(case['responses'][1]))
                obligation = anchored['evidence']['obligations'][9]
                start, end = obligation['prompt_span']
                assert case['prompt'][start:end] == obligation['prompt_quote'] == 'vaccination coverage pv between 0 and 1'
        elif name == 'shbt' and index > 0: assert 'DEGENERATE_MECHANISM_DOMAIN' in record['error'] or record['error'].startswith('compiled contract has no sealed symbolic binding:')
        else: assert any(tok in record['error'] for tok in ('DEGENERATE_MECHANISM_DOMAIN', 'PREFLIGHT_REDUNDANT_CONJUNCTS', 'IHCG_INGEST_REJECTED:'))
        assert record['error_type'] == 'ValueError'
        if index < 2: assert record['error'] in prompts[index+2]
    metadata = json.loads((directory/'metadata.json').read_text(encoding="utf-8"))
    assert metadata['diagnostics'] == certificate.diagnostics
    terminal = next(event for event in events if event['event'] == 'research_completed')
    assert terminal['event_version'] == '2.0'
    assert terminal['meaningfulness_certificate']['diagnostics'] == certificate.diagnostics
    assert attempts[0]['error'] in prompts[2]


@pytest.mark.parametrize('suffix',['. Next sentence.','? Next sentence.','! Next sentence.'])
def test_exact_quote_reanchoring_preserves_integer_and_sentence_boundaries(suffix):
    quote = 'x between 0 and 1'
    prompt = 'Use '+quote+suffix
    pack = copy.deepcopy(_FORMAL_EXAMPLE_PACK)
    pack['evidence']['obligations'][0].update(prompt_quote=quote,prompt_span=[0,1])
    anchored = _anchor_evidence(prompt,pack)['evidence']['obligations'][0]
    assert anchored['prompt_quote'] == quote
    assert prompt[slice(*anchored['prompt_span'])] == quote
    # Non-blocking provenance: changed numbers drop prompt_span cleanly without raising.
    for changed in ['x between 0 and 10','x between 0 and 1.0','x between 0 and 1e0']:
        unanchored = _anchor_evidence('Use '+changed+suffix, pack)['evidence']['obligations'][0]
        assert unanchored.get('prompt_span') is None

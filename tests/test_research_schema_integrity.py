"""Hermetic regressions for buffer, contract, literature and report schemas."""
import ast
import json
import re
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

from invar_agent.healing.buffer_api import STEP_ALIASES, TIME_ALIASES
from invar_agent.healing.deterministic_ast import sanitize_intrinsics
from invar_agent.research.extractor import (
    QuantitativeInsightExtractor, classify_macro_scientific_domains,
    extract_parameters_from_text, normalize_to_si,
)
from invar_agent.research.planner import apply_concept_bucket_remap, canonicalize_contract_dicts
from invar_agent.research.report import build_research_report, extract_observables
from invar_agent.research.search import LiteratureSearchClient, _channels_for_prompt
from invar_agent.research.slot_schema import canonical_variable_map, validate_contract_slots, validate_slot_writes
from invar_agent.research.state_machine import (
    ResearchStateMachine, _BufView, _auto_populate_variable_map, v9_transpiler_violations,
)
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
from invar_agent.schemas.research import ResearchBlueprint, SimulationResult, VerificationContract
from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
from invar_agent.transpiler.emitters.rust_emitter import RustEmitter
from invar_agent.transpiler.frontend import python_source_to_uast


@pytest.mark.parametrize('alias', sorted(STEP_ALIASES | TIME_ALIASES))
def test_buffer_clock_aliases_monotonic_and_idempotent(alias):
    contracts = [{'initial_state': [0.0] * 5, 'invariants': [{'assertion': 'state[19] >= 0'}]}]
    source = f'def simulation_step(buf, dt):\n    buf.data[27] = 0.0\n    buf.data[0] = buf.{alias}\n'
    lowered = sanitize_intrinsics(source, contracts=contracts, preserve_dynamics=True).code
    assert not v9_transpiler_violations(lowered)
    slot = int(re.search(r'_invar_step_slot = (\d+)', lowered)[1])
    assert slot > max(19, 5, 27) + 1
    assert len(contracts[0]['initial_state']) > slot
    assert sanitize_intrinsics(lowered, contracts=contracts, preserve_dynamics=True).code == lowered
    ns = {}
    exec(lowered, ns)
    buf = _BufView([0.0] * (slot + 1))
    for step in range(4):
        ns['simulation_step'](buf, 0.25)
        assert buf.data[slot] == step + 1
        assert buf.data[0] == step * (0.25 if alias in TIME_ALIASES else 1)
    ns['simulation_step'](_BufView([]), 0.25)


def test_buffer_clock_python_cpp_and_rust_early_exit(tmp_path):
    source = '''def helper(buf, dt, scale):
    return buf.dt * scale

def simulation_step(buf, dt):
    buf.data[0] = buf.steps + buf.step + buf.t + helper(buf, dt, 1.0)
    if buf.data[0] > 0.0:
        return
    buf.data[1] = 1.0
'''
    lowered = sanitize_intrinsics(source, preserve_dynamics=True).code
    assert 'buf.dt' not in lowered and 'buf.steps' not in lowered
    assert not v9_transpiler_violations(lowered)
    slot = int(re.search(r'_invar_step_slot = (\d+)', lowered)[1])
    ns = {}
    exec(lowered, ns)
    buf = _BufView([0.0] * (slot + 1))
    for _ in range(3):
        ns['simulation_step'](buf, 0.5)
    assert buf.data[slot] == 3 and buf.data[0] == 5.5
    uast = python_source_to_uast(lowered)
    cpp = CppEmitter().emit_module(uast)
    assert 'extern "C" void execute_step' in cpp
    out = DynamicCABIBridge(tmp_path).run(cpp, [0.0] * (slot + 1), dt=0.5, steps=3)
    assert out['state'] == pytest.approx(buf.data)
    assert 'pub fn simulation_step' in RustEmitter().emit_module(uast)


@pytest.mark.parametrize('arity', ['buf', 'buf, dt', 'buf, dt, extra'])
def test_buffer_dt_normalized_independent_of_arity(arity):
    code = sanitize_intrinsics(f'def simulation_step({arity}):\n    buf.data[0] = buf.dt\n', preserve_dynamics=True).code
    assert 'buf.dt' not in code
    assert not v9_transpiler_violations(code)


def test_unknown_buffer_attribute_is_not_silently_allocated():
    code = sanitize_intrinsics('def simulation_step(buf, dt):\n    buf.data[0] = buf.nonexistent\n', preserve_dynamics=True).code
    errors = v9_transpiler_violations(code)
    assert any('buf.nonexistent' in error and 'line 2' in error and 'ONLY buf.data[i] and buf.size' in error for error in errors)


def test_running_extrema_remain_observations_not_projected_bounds():
    source = 'def simulation_step(buf, dt):\n    previous = buf.data[4]\n    buf.data[4] = max(previous, buf.data[0])\n'
    contracts = [{'invariants': [{'variable_name': 'peak_signal_bound', 'assertion': 'state[4] < 1.0'}]}]
    code = sanitize_intrinsics(source, contracts=contracts).code
    ns = {}
    exec(code, ns)
    buf = _BufView([5.0, 0.0, 0.0, 0.0, 0.0])
    ns['simulation_step'](buf, 0.1)
    assert buf.data[4] == 5.0


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ["PreWrite_Guard python_smoke: AttributeError: '_BufView' object has no attribute 'bad'", "PreWrite_Guard python_smoke: NameError: name 'missing' is not defined"])
async def test_structural_healing_bypasses_bound_conditioning(tmp_path, monkeypatch, failure):
    prompts = []
    async def llm(prompt):
        prompts.append(prompt)
        return json.dumps({'code': 'def simulation_step(buf, dt):\n    buf.data[0] = dt\n'})
    async def forbidden(*args, **kwargs):
        raise AssertionError('Numerical directive routing must not execute')
    monkeypatch.setattr('invar_agent.research.state_machine.request_healing_directives', forbidden)
    machine = ResearchStateMachine(tmp_path, llm_fast=llm)
    out = await machine._heal_and_refine(stage=1, round_idx=1, objective='repair', contracts=[], failure=failure,
                                         candidate_code='def simulation_step(buf, dt):\n    buf.data[0] = buf.bad + missing\n')
    assert out['applied_ops'] == ['STRUCTURAL_API_REPAIR']
    assert len(prompts) == 1
    assert failure in prompts[0] and '2: ' in prompts[0] and 'ONLY buf.data[i] and buf.size' in prompts[0]


@pytest.mark.asyncio
async def test_structural_repair_rejects_unresolved_symbol_without_retry(tmp_path):
    async def llm(prompt):
        return json.dumps({'code': 'def simulation_step(buf, dt):\n    buf.data[0] = missing + 1.0\n'})
    out = await ResearchStateMachine(tmp_path, llm_fast=llm)._heal_and_refine(
        stage=1, round_idx=1, objective='repair', contracts=[],
        failure="PreWrite_Guard: NameError: name 'missing' is not defined",
        candidate_code='def simulation_step(buf, dt):\n    buf.data[0] = missing\n')
    assert out['terminated'] and not out['applied_ops']
    assert 'Unresolved offending symbol missing' in out['failure']


def pool_contracts():
    return [{'contract_id': 'c1_baseline', 'target_backend': 'cpp', 'initial_state': [0.1, 0.5, 4, 4, 4, 1, 2, 0, 0], 'invariants': [
        *[{'variable_name': name + '_positivity', 'assertion': f'state[{slot}] >= 0'}
          for name, slot in [('a', 2), ('b', 3), ('c', 4), ('d', 5), ('e', 6)]],
        {'variable_name': 'first_pool_conservation', 'assertion': 'abs(state[2] + state[3] + state[4] - 12) < 1e-8'},
        {'variable_name': 'second_pool_conservation', 'assertion': 'abs(state[5] + state[6] - 3) < 1e-8'},
        {'variable_name': 'peak_signal_finite', 'assertion': 'state[8] < 1e10'},
    ]}]


def test_distinct_pools_and_canonical_variable_map():
    contracts, vm, _ = canonicalize_contract_dicts(pool_contracts())
    validate_contract_slots(contracts)
    mapping = canonical_variable_map(contracts, vm)
    assert len(mapping) == len(set(mapping.values())) == 9
    assert {mapping[name] for name in ['a', 'b', 'c', 'd', 'e']} == {2, 3, 4, 5, 6}
    bp = ResearchBlueprint(blueprint_id='pools', title='Two coupled pools', falsifiable_hypothesis='Conserved pools',
                           search_queries=[], code_template='def simulation_step(buf, dt):\n    pass\n',
                           contracts=[VerificationContract.model_validate(c) for c in contracts])
    _auto_populate_variable_map(bp)
    assert bp.variable_map and set(bp.slot_roles) == set(bp.variable_map)
    assert canonical_variable_map([], {'C_m': 0}) == {'C_m': 0}


@pytest.mark.parametrize('invariants', [
    [{'variable_name': 'pool_conservation', 'assertion': 'abs(state[4] + state[5] + state[4] - 3) < 1e-8'}],
    [{'variable_name': 'a_positivity', 'assertion': 'state[4] >= 0'}, {'variable_name': 'b_positivity', 'assertion': 'state[4] >= 0'}],
    [{'variable_name': 'peak_signal_finite', 'assertion': 'state[8] < 1e10'}, {'variable_name': 'peak_signal_bound', 'assertion': 'state[9] < 1'}],
])
def test_semantic_slot_errors_precede_smt(invariants):
    with pytest.raises(ValueError, match='SLOT_SCHEMA_ERROR'):
        validate_contract_slots([{'invariants': invariants}])


def test_remap_cannot_make_duplicate_conservation_operands():
    with pytest.raises(ValueError, match='merge distinct operands'):
        apply_concept_bucket_remap(pool_contracts(), {6: 5})


@pytest.mark.parametrize('prefix', ['peak_', 'min_', 'max_'])
def test_unrelated_pool_or_instantaneous_value_is_not_bridged(prefix):
    code = '''def simulation_step(buf, dt):
    pool = buf.data[0] + buf.data[1]
    signal = buf.data[2]
    buf.data[4] = pool
    buf.data[5] = signal
'''
    contracts = [{'invariants': [
        {'variable_name': 'other_pool_conservation', 'assertion': 'state[8] < 3'},
        {'variable_name': prefix + 'signal_bound', 'assertion': 'state[9] < 1'},
    ]}]
    lowered = sanitize_intrinsics(code, contracts=contracts, preserve_dynamics=True).code
    assert 'buf.data[8] =' not in lowered and 'buf.data[9] =' not in lowered
    with pytest.raises(ValueError, match='read-only parameter control'):
        validate_slot_writes('def simulation_step(buf, dt):\n    buf.data[4] = 1.0\n', {'control': 4}, {'control': 'read_only_parameter'})


@pytest.mark.parametrize('unit,value,target,expected', [
    ('uF/cm^2', 1, 'F/m^2', 0.01), ('mF/cm^2', 1, 'F/m^2', 10),
    ('F/m^2', 1, 'F/m^2', 1), ('mS/cm^2', 1, 'S/m^2', 10),
    ('uS/cm^2', 1, 'S/m^2', 0.01), ('S/m^2', 0.0025, 'S/m^2', 0.0025),
    ('mV', 165, 'V', 0.165), ('uV', 1, 'V', 1e-6), ('nm', 7.5, 'm', 7.5e-9),
    ('um', 1, 'm', 1e-6), ('mM', 12, 'mol/m^3', 12), ('uM', 2, 'mol/m^3', 0.002),
    ('nM', 1, 'mol/m^3', 1e-6), ('K', 310.15, 'K', 310.15),
])
def test_compound_si_units(unit, value, target, expected):
    actual, actual_unit = normalize_to_si(value, unit)
    assert actual == pytest.approx(expected) and actual_unit == target


def test_full_unit_extraction_and_procedural_stopwords():
    params = extract_parameters_from_text('C_m = 1.0 uF/cm^2; simulate = 600 s; dual_k = 3; steps = 400; protocol = 2 s;', source='user_prompt')
    assert len(params) == 1 and params[0].name == 'c_m'
    assert params[0].si_value == pytest.approx(0.01) and params[0].si_unit == 'F/m^2'
    equations = QuantitativeInsightExtractor().equations_from_text('The law is J = -0.125 * k * x; n = 18; protocol = compute the next step.')
    assert any('0.125' in eq.rhs_python_hint for eq in equations)
    assert all(eq.lhs not in ('n', 'protocol') for eq in equations)


def test_domain_connectors_preserve_categories_and_log_timeouts():
    context = 'Cellular bioenergetics metabolism biochemistry electrochemistry membrane transport physiology'
    domains = classify_macro_scientific_domains(context)
    assert {'LIFE_SCIENCES_AND_BIOMEDICINE', 'CHEMICAL_AND_MATERIALS'} <= set(domains)
    assert {'europe_pmc', 'pubmed', 'biorxiv', 'medrxiv', 'pubchem'} <= set(_channels_for_prompt(context))
    urls = []
    def fetch(url, timeout):
        urls.append(unquote(url))
        if 'europepmc' in url:
            raise TimeoutError('mock connector timeout')
        if 'arxiv' in url:
            return '<feed xmlns="http://www.w3.org/2005/Atom"/>'
        return '{}'
    client = LiteratureSearchClient(fetch=fetch, offline=False)
    client.search('membrane bioenergetics chemical transport', prompt_or_query=context)
    for source in ['europe_pmc', 'biorxiv', 'medrxiv']:
        assert 'TimeoutError' in client.last_query_meta['by_source'][source]['error']
    assert all('cat:q-bio.*' in url and 'cat:physics.chem-ph' in url for url in urls if 'search_query=' in url)
    assert any('PUBLISHER:"biorxiv"' in url for url in urls)
    assert any('PUBLISHER:"medrxiv"' in url for url in urls)
    assert not client.last_query_meta['offline']


@pytest.mark.parametrize('invariants', [
    [{'variable_name': 'pool_conservation', 'assertion': 'state[ 1 ] + data[1] == 3.0'}],
    [{'variable_name': 'signal_bound', 'assertion': 'state[1]<1'}, {'variable_name': 'peak_signal_finite', 'assertion': 'math.isfinite(state[1])'}],
    [{'variable_name': 'signal_bound', 'assertion': 'state[-1]<1'}],
])
def test_alias_indices_and_extrema_collisions_are_rejected(invariants):
    with pytest.raises(ValueError, match='SLOT_SCHEMA_ERROR'):
        validate_contract_slots([{'invariants': invariants}])


@pytest.mark.asyncio
async def test_search_source_diagnostics_and_original_context_survive_sse(tmp_path):
    class Search:
        offline = False
        last_query_meta = {'by_source': {'europe_pmc': {'count': 0, 'error': 'TimeoutError'}},
                           'arxiv_categories': ['q-bio.*']}

        def search(self, query, limit, sources, context):
            assert 'membrane physiology electrochemistry' in context
            return []

    events = []

    async def callback(channel, data):
        events.append(data)

    machine = ResearchStateMachine(tmp_path, progress_cb=callback, search_client=Search())
    machine.blueprint = ResearchBlueprint(blueprint_id='probe', title='Probe', falsifiable_hypothesis='H',
        search_queries=[],
        research_prompt='membrane physiology electrochemistry', code_template='def simulation_step(buf, dt):\n    pass\n',
        contracts=[VerificationContract(contract_id='c', target_backend='cpp',
            invariants=[{'variable_name': 'x', 'assertion': 'state[0] >= 0'}])])
    assert await machine._literature_search(['generic model'], {}) == []
    done = next(event for event in events if event.get('event') == 'query_done')
    assert done['by_source']['europe_pmc']['error'] == 'TimeoutError'
    assert done['arxiv_categories'] == ['q-bio.*']


def test_report_only_verified_finite_mapped_observables():
    bp = ResearchBlueprint(blueprint_id='report', title='Observable filtering', falsifiable_hypothesis='Mapped quantities',
                           search_queries=[], code_template='def simulation_step(buf, dt):\n    pass\n',
                           contracts=[VerificationContract(contract_id='c', target_backend='cpp', invariants=[{'variable_name': 'x', 'assertion': 'state[0] >= 0'}])],
                           variable_map={'x': 0, 'x_bound': 0, 'psi_new': 1, 'pool': 2})
    result = SimulationResult(blueprint_id='report', contract_id='c', status='VERIFIED', execution_time_ms=1,
                              telemetry={'obs_x': 2.5, 'obs_x_bound': 3.0}, bundle_hash='hash')
    metrics = extract_observables(bp.variable_map, [result])
    assert list(metrics) == ['x']
    section = build_research_report(SimpleNamespace(blueprint=bp, results=[result], literature=[], status='VERIFIED', checkpoint_path=None)).split('## 2.')[0]
    assert '**2.5**' in section
    assert all(name not in section for name in ['x_bound', 'psi_new', '`pool`'])
    assert extract_observables(bp.variable_map, [result.model_copy(update={'status': 'FALSIFIED'})]) == {}

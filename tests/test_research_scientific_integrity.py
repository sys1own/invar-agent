"""Scientific predicates must test observations rather than manufacture them."""
import ast
from types import SimpleNamespace

import pytest

from invar_agent.healing.deterministic_ast import sanitize_intrinsics
from invar_agent.research.planner import _reconcile_time_horizon
from invar_agent.research.scientific_integrity import (
    output_dataflow, physical_projection_contracts, validate_observables, validate_protocol, validate_variant_schemes,
)
from invar_agent.research.state_machine import _BufView, _eval_invariant, _guard_contract_initial_state
from invar_agent.schemas.research import VerificationContract
from invar_agent.runtime.research_reference import ReferenceError, _safe_namespace


def contract(assertion='state[0] >= 0', **kwargs):
    return dict(contract_id='c1_probe', target_backend='cpp', initial_state=[1., 0., 0.],
                dt=0.1, steps=10, invariants=[{'variable_name': 'measurement', 'assertion': assertion}], **kwargs)


def test_derived_bounds_remain_falsifiable_without_explicit_ode():
    source = 'def simulation_step(buf, dt):\n    x = buf.data[0]\n    x = x + dt\n    buf.data[0] = x\n    buf.data[1] = x / (x + 1.0)\n'
    c = contract('state[1] >= 0.78')
    lowered = sanitize_intrinsics(source, contracts=[c]).code
    namespace = _safe_namespace(lowered)
    buf = _BufView([1., 0., 0.])
    for _ in range(c['steps']):
        namespace['simulation_step'](buf, c['dt'])
    assert buf.data[1] == pytest.approx(2/3)
    assert not _eval_invariant(c['invariants'][0]['assertion'], buf.data, {})
    assert output_dataflow(lowered)['derived'] == {1}
    validate_observables(lowered, [c], {'x': 0, 'ratio': 1}, {})


def test_initial_conditions_are_not_seeded_or_projected_to_outcome_bounds():
    c = VerificationContract.model_validate({**contract('state[1] >= 0.78'), 'initial_state': [0.0, 0.0, 0.5]})
    initial, warnings = _guard_contract_initial_state('def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0] + dt\n', c)
    assert initial[:3] == c.initial_state
    assert warnings and 'NEVER assigned' in warnings[0]


@pytest.mark.parametrize('source', [
    'def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0] + dt\n',
    'def simulation_step(buf, dt):\n    buf.data[1] = 0.78\n',
    'def simulation_step(buf, dt):\n    buf.data[1] = buf.data[0] / 2\n    if buf.data[1] < 0.78:\n        buf.data[1] = 0.78\n',
    'def simulation_step(buf, dt):\n    buf.data[1] = max(0.78, buf.data[0] / 2)\n',
    'def simulation_step(buf, dt):\n    ratio = buf.data[0] / 2\n    ratio = max(0.78, ratio)\n    buf.data[1] = ratio\n',
    'def simulation_step(buf, dt):\n    ratio = buf.data[0] / 2\n    if ratio < 0.78:\n        ratio = 0.78\n    buf.data[1] = ratio\n',
])
def test_uncomputed_constant_or_projected_outcomes_rejected(source):
    with pytest.raises(ValueError, match='SCIENTIFIC_INTEGRITY'):
        validate_observables(source, [contract('state[1] >= 0.78')], {'corridor_value': 1}, {})


def test_computed_boolean_outcome_and_empty_contracts():
    source = 'def simulation_step(buf, dt):\n    buf.data[1] = 0.0\n    if buf.data[0] > 0.0:\n        buf.data[1] = 1.0\n'
    validate_observables(source, [contract('state[1] == 1')], {'corridor_found': 1}, {})
    validate_observables('def simulation_step(buf, dt):\n    pass\n', [], {}, {})


def test_seconds_horizon_overrides_hallucinated_duration_and_preserves_grid():
    bp = {'code_template': '', 'contracts': [dict(dt=0.5, steps=1800)]}
    result = _reconcile_time_horizon(bp, 'Simulate a 600 s time-phased protocol.')
    assert result['contracts'][0] == {'dt': 0.5, 'steps': 1200}
    assert bp['contracts'][0]['steps'] == 1800
    assert _reconcile_time_horizon(bp, 'No stated horizon.') is bp
    assert _reconcile_time_horizon(bp, 'Simulate a 600-second protocol.') == result


@pytest.mark.parametrize('switch,valid', [(0.5, True), (1.0, False), (2.0, False)])
def test_all_time_phases_must_execute_inside_horizon(switch, valid):
    code = f'def simulation_step(buf, dt):\n    t = buf.data[0]\n    rate = 1.0\n    if t < {switch}:\n        rate = 0.2\n    buf.data[1] = buf.data[1] + dt * rate\n    buf.data[0] = t + dt\n'
    c = contract()
    c['initial_state'] = [0., 0., 0.]
    if valid:
        coverage = validate_protocol(code, c, 'Simulate a phased protocol.')
        assert coverage and all(min(counts.values()) > 0 for counts in coverage.values())
    else:
        with pytest.raises(ValueError, match='unreachable protocol phases'):
            validate_protocol(code, c, 'Simulate a phased protocol.')
    assert validate_protocol(code, c, 'A stationary experiment.') == {}


def test_relabelled_euler_does_not_pass_as_rk4():
    code = 'def simulation_step_rk4(buf, dt):\n    buf.data[0] = buf.data[0] + dt * buf.data[0]\n\ndef simulation_step(buf, dt):\n    simulation_step_rk4(buf, dt)\n'
    with pytest.raises(ValueError, match='executed scheme is euler'):
        validate_variant_schemes(code, SimpleNamespace(variable_map={'x': 0}), [contract()])


def test_actual_rk4_and_static_constant_arithmetic_are_supported():
    code = '''RATE = 0.1 / 2.0
def rhs(x):
    return RATE * x
def simulation_step_rk4(buf, dt):
    x = buf.data[0]
    k1 = rhs(x)
    k2 = rhs(x + dt * k1 / 2)
    k3 = rhs(x + dt * k2 / 2)
    k4 = rhs(x + dt * k3)
    buf.data[0] = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
def simulation_step(buf, dt):
    simulation_step_rk4(buf, dt)
'''
    assert _safe_namespace(code)['RATE'] == 0.05
    assert validate_variant_schemes(code, SimpleNamespace(variable_map={'x': 0}), [contract()]) == {'simulation_step_rk4': 'rk4'}


@pytest.mark.parametrize('expression', ['__import__("os")', 'open("anything")', '2 ** 1000000000'])
def test_module_constant_folding_does_not_enable_capabilities(expression):
    with pytest.raises((ReferenceError, ValueError)):
        _safe_namespace(f'X = {expression}\ndef simulation_step(buf, dt):\n    pass\n')


def test_compound_contract_keeps_physical_projection_without_conditioning_ratio():
    code = 'def simulation_step(buf, dt):\n    buf.data[0] *= 0.9\n    buf.data[1] = buf.data[0] / (1.0 + buf.data[0])\n'
    c = contract('state[0] >= 0.0 and state[1] >= 0.78')
    filtered = physical_projection_contracts(ast.parse(code), [c])
    assert filtered[0]['invariants'][0]['assertion'] == 'state[0] >= 0.0'
    assert c['invariants'][0]['assertion'] == 'state[0] >= 0.0 and state[1] >= 0.78'
    ns = _safe_namespace(sanitize_intrinsics(code, contracts=[c]).code)
    buf = SimpleNamespace(data=[1.0, 0.0] + [0.0] * 62)
    ns['simulation_step'](buf, 0.1)
    assert buf.data[1] == pytest.approx(0.9 / 1.9)
    assert buf.data[1] < 0.78


def test_conditioner_does_not_invent_missing_outcome_writes():
    source = 'def simulation_step(buf, dt):\n    buf.data[0] *= 0.9\n'
    c = contract('state[1] >= 0.78')
    lowered = sanitize_intrinsics(source, contracts=[c]).code
    assert 1 not in output_dataflow(lowered)['writes']
    with pytest.raises(ValueError, match='uncomputed contract observables'):
        validate_observables(lowered, [c], {'measurement': 1}, {})


@pytest.mark.parametrize('source', ['', 'def simulation_step(buf, dt):\n    pass\n'])
def test_empty_projection_contract_filter(source):
    assert physical_projection_contracts(ast.parse(source), []) == []


def test_ratio_predicates_do_not_condition_their_physical_inputs():
    source = 'def simulation_step(buf, dt):\n    buf.data[0] = 1.0\n    buf.data[1] = 4.0\n'
    c = contract('state[0] / state[1] >= 0.78')
    assert not physical_projection_contracts(ast.parse(source), [c])[0]['invariants']
    ns = _safe_namespace(sanitize_intrinsics(source, contracts=[c]).code)
    buf = SimpleNamespace(data=[0.0] * 64)
    ns['simulation_step'](buf, 0.1)
    assert buf.data[:2] == [1.0, 4.0]

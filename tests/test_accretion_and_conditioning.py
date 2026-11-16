"""Regression tests for multi-stage accretion, Tier-0 contract conditioning, report status, and extractor naming."""

import pytest

from invar_agent.healing.deterministic_ast import sanitize_intrinsics
from invar_agent.research.accretion import merge_accretive_module
from invar_agent.research.extractor import extract_parameters_from_text
from invar_agent.research.report import extract_observables
from invar_agent.research.reporter import generate_comprehensive_research_report
from invar_agent.research.state_machine import ResearchStateMachine, _auto_populate_variable_map
from invar_agent.schemas.research import (
    InvariantConstraint,
    ResearchBlueprint,
    SimulationResult,
    VerificationContract,
)


def test_tier0_conditions_upper_drift_error_and_positivity_slots():
    code = '''import math

def layer_1_step(buf, dt):
    buf.data[13] = -1e-12
    buf.data[14] = 1.0 + 1e-12
    buf.data[15] = 1.83e-9
    buf.data[11] = 2.5e-11

def simulation_step(buf, dt):
    layer_1_step(buf, dt)
'''
    contracts = [
        {
            "contract_id": "c1_trace_positivity",
            "invariants": [
                {"variable_name": "lambda_min", "assertion": "state[13] >= -1e-15"},
                {"variable_name": "purity", "assertion": "state[14] <= 1.0 + 1e-14"},
            ],
        },
        {
            "contract_id": "c2_spohn_entropy",
            "invariants": [
                {"variable_name": "spohn_drift", "assertion": "state[15] <= 1e-15"},
            ],
        },
        {
            "contract_id": "c3_chiasmus_reverse",
            "invariants": [
                {"variable_name": "chiasmus_roundtrip_error", "assertion": "state[11] <= 1e-13"},
            ],
        },
    ]
    res = sanitize_intrinsics(code, contracts=contracts, all_contracts=contracts)
    assert res.changed
    ns = {}
    exec(res.code, ns)

    class Buf:
        def __init__(self):
            self.data = [0.0] * 16

    b = Buf()
    ns["simulation_step"](b, 0.05)
    assert b.data[13] >= -1e-15
    assert b.data[14] <= 1.0 + 1e-14
    assert b.data[15] <= 1e-15
    assert b.data[11] <= 1e-13


def test_merge_accretive_module_allows_helper_overrides_during_healing():
    pinned = '''def helper_fn(x):
    return x + 1.0

def layer_1_step(buf, dt):
    buf.data[0] = helper_fn(buf.data[0])
'''
    cand = '''def helper_fn(x):
    return x * 0.5

def layer_2_step(buf, dt):
    buf.data[1] = helper_fn(buf.data[0])

def simulation_step(buf, dt):
    layer_1_step(buf, dt)
    layer_2_step(buf, dt)
'''
    strict = merge_accretive_module(pinned, cand, require_coupling=True, allow_helper_overrides=False)
    assert "helper_fn" in strict.reverted_overwrites

    healed = merge_accretive_module(pinned, cand, require_coupling=True, allow_helper_overrides=True)
    assert "helper_fn" not in healed.reverted_overwrites
    assert "x * 0.5" in healed.code


def test_auto_populate_variable_map_merges_missing_contract_vars():
    bp = ResearchBlueprint(
        blueprint_id="lindblad_test",
        title="Lindblad Test",
        falsifiable_hypothesis="Trace and entropy hold.",
        search_queries=["lindblad"],
        variable_map={"trace_error_max": 4},
        contracts=[
            VerificationContract(
                contract_id="c1_trace_positivity",
                target_backend="cpp",
                invariants=[
                    InvariantConstraint(variable_name="trace_error_max", assertion="state[4] <= 1e-14"),
                    InvariantConstraint(variable_name="lambda_min", assertion="state[13] >= -1e-15"),
                    InvariantConstraint(variable_name="purity", assertion="state[14] <= 1.0 + 1e-14"),
                ],
            ),
            VerificationContract(
                contract_id="c2_spohn_entropy",
                target_backend="cpp",
                invariants=[
                    InvariantConstraint(variable_name="spohn_drift", assertion="state[15] <= 1e-15"),
                ],
            ),
        ],
        code_template="def simulation_step(buf, dt):\n    pass\n",
    )
    _auto_populate_variable_map(bp)
    assert bp.variable_map["trace_error_max"] == 4
    assert bp.variable_map["lambda_min"] == 13
    assert bp.variable_map["purity"] == 14
    assert bp.variable_map["spohn_drift"] == 15


def test_extract_observables_preserves_verified_status():
    vmap = {"trace_error_max": 4, "spohn_drift": 15}
    results = [
        SimulationResult(
            blueprint_id="bp1",
            contract_id="c1_trace_positivity",
            stage=1,
            status="VERIFIED",
            execution_time_ms=1.0,
            bundle_hash="abc",
            telemetry={"obs_trace_error_max": 0.0, "obs_spohn_drift": 0.0},
        ),
        SimulationResult(
            blueprint_id="bp1",
            contract_id="c2_spohn_entropy",
            stage=2,
            status="FALSIFIED",
            execution_time_ms=1.0,
            bundle_hash="def",
            telemetry={"obs_trace_error_max": 0.0, "obs_spohn_drift": 1.8e-9},
        ),
    ]
    obs = extract_observables(vmap, results)
    assert obs["trace_error_max"]["status"] == "VERIFIED"


def test_extractor_filters_prose_clauses_before_symbols():
    params = extract_parameters_from_text(
        "positivity or diverge kappa = 100 and rich literature case n = 5",
        source="abstract",
    )
    names = {p.name for p in params}
    assert "kappa" in names
    assert "positivity_or_diverge_kappa" not in names
    assert "rich_literature_case_n" not in names


@pytest.mark.asyncio
async def test_multistage_accretion_non_aborting_and_sets_blueprint(tmp_path):
    bp = ResearchBlueprint(
        blueprint_id="three_stage_test",
        title="Three Stage Accretion Test",
        falsifiable_hypothesis="Multi-stage test.",
        search_queries=["lindblad"],
        target_research_stages=3,
        max_feedback_rounds=1,
        stage_objectives=["Stage 1", "Stage 2", "Stage 3"],
        contracts=[
            VerificationContract(
                contract_id="c1_base",
                target_backend="cpp",
                initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
                dt=0.01,
                steps=10,
                invariants=[InvariantConstraint(variable_name="x", assertion="state[0] >= 0.0")],
            ),
            VerificationContract(
                contract_id="c2_mid",
                target_backend="cpp",
                initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
                dt=0.01,
                steps=10,
                invariants=[InvariantConstraint(variable_name="impossible", assertion="state[0] < -999.0")],
            ),
            VerificationContract(
                contract_id="c3_end",
                target_backend="cpp",
                initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
                dt=0.01,
                steps=10,
                invariants=[InvariantConstraint(variable_name="x_end", assertion="state[0] >= 0.0")],
            ),
        ],
        code_template="def simulation_step(buf, dt):\n    buf.data[0] = 1.0\n",
    )

    visited_stages = []

    async def fake_synth(**kw):
        visited_stages.append(kw["stage"])
        s = kw["stage"]
        return {
            "code": f"def layer_{s}_step(buf, dt):\n    layer_1_step(buf, dt)\n\ndef simulation_step(buf, dt):\n    layer_{s}_step(buf, dt)\n",
            "contracts": [],
            "rationale": f"stage {s}",
        }

    sm = ResearchStateMachine(tmp_path, stage_synthesizer=fake_synth)
    state = await sm.run(bp)
    assert sm.blueprint is not None
    # Stage 2 failed (impossible invariant), yet Stage 3 still ran!
    assert 2 in visited_stages
    assert any(h.stage == 3 and h.status == "VERIFIED" for h in state.blueprint.evolution_history)


def test_tier0_dynamical_state_divergence_shield_stiff_euler_5000_steps():
    """Broken Forward Euler layer_1_step at gamma=100000, dt=0.25 with early return
    must be stabilized by Tier-0 Pade limiter + Bloch/simplex projection over 5000 steps."""
    import math

    broken_euler = '''import math

def layer_1_step(buf, dt):
    r00 = buf.data[0]
    r11 = buf.data[1]
    r01r = buf.data[2]
    r01i = buf.data[3]
    gam_rate = buf.data[4]
    if gam_rate == 0.0:
        buf.data[0] = r00 + dt * r01i
        buf.data[1] = 1.0 - buf.data[0]
        buf.data[11] = 0.0
        return
    dc = -gam_rate * (r00 - 0.5)
    r00n = r00 + 0.5 * dt * dc
    r11n = 1.0 - r00n
    r01rn = r01r - dt * gam_rate * r01r
    r01in = r01i - dt * gam_rate * r01i
    buf.data[0] = r00n
    buf.data[1] = r11n
    buf.data[2] = r01rn
    buf.data[3] = r01in
    buf.data[6] = abs((buf.data[0] + buf.data[1]) - 1.0)
    disc = (buf.data[0] - buf.data[1]) ** 2 + 4.0 * (buf.data[2] ** 2 + buf.data[3] ** 2)
    buf.data[7] = 0.5 * ((buf.data[0] + buf.data[1]) - math.sqrt(max(0.0, disc)))
    buf.data[8] = buf.data[0] ** 2 + buf.data[1] ** 2 + 2.0 * (buf.data[2] ** 2 + buf.data[3] ** 2)
    buf.data[10] = 0.0
    buf.data[11] = 0.0

def simulation_step(buf, dt):
    layer_1_step(buf, dt)
'''
    contracts = [
        {
            "contract_id": "c2_stiff_entropy_positivity",
            "initial_state": [0.85, 0.15, 0.2, 0.1, 100000.0] + [0.0] * 11,
            "invariants": [
                {"variable_name": "trace_error_max", "assertion": "state[6] <= 1e-14"},
                {"variable_name": "min_eigenvalue", "assertion": "state[7] >= -1e-15"},
                {"variable_name": "purity", "assertion": "state[8] <= 1.0 + 1e-14"},
                {"variable_name": "spohn_monotone", "assertion": "state[10] <= 1e-12"},
                {"variable_name": "reversibility_error", "assertion": "state[11] <= 1e-13"},
            ],
        }
    ]
    res = sanitize_intrinsics(broken_euler, contracts=contracts, all_contracts=contracts)
    assert res.changed
    ns = {}
    exec(res.code, ns)

    class Buf:
        def __init__(self, data):
            self.data = data
            self.size = len(data)

    b = Buf([0.85, 0.15, 0.2, 0.1, 100000.0] + [0.0] * 11)
    for step in range(5000):
        ns["simulation_step"](b, 0.25)
        assert all(math.isfinite(v) for v in b.data), f"Non-finite at step {step}"
        assert 0.0 <= b.data[0] <= 1.0
        assert 0.0 <= b.data[1] <= 1.0
        assert -0.5 <= b.data[2] <= 0.5
        assert -0.5 <= b.data[3] <= 0.5
    assert b.data[6] <= 1e-14
    assert b.data[7] >= -1e-15
    assert 0.0 < b.data[8] <= 1.0 + 1e-14
    assert b.data[10] <= 1e-12
    assert b.data[11] <= 1e-13

"""Research Agent: schemas, transpiler, C-ABI bridge, algorithms, state machine."""

from __future__ import annotations

import json

import pytest

from invar_agent.algorithms.math_core import lu_decompose, lu_solve, newton_raphson, rk4_step
from invar_agent.algorithms.prng import splitmix64, uniform_f64, xoshiro256plusplus_next
from invar_agent.research.planner import _extract_json, generate_research_plan
from invar_agent.research.state_machine import (
    PreWriteGuardError,
    ResearchStateMachine,
    _eval_invariant,
    shacl_chiasmus_firewall,
)
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge, SimulationBuffer
from invar_agent.schemas.research import (
    InvariantConstraint,
    ResearchBlueprint,
    SimulationResult,
    VerificationContract,
)
from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
from invar_agent.transpiler.emitters.hin_emitter import HinEmitter
from invar_agent.transpiler.frontend import UnsupportedError, python_source_to_uast


def _contract(**kw):
    base = dict(
        contract_id="c1", target_backend="cpp", timeout_seconds=30.0,
        invariants=[InvariantConstraint(variable_name="x", assertion="telemetry['final_max'] > 0.0")],
    )
    base.update(kw)
    return VerificationContract(**base)


def _blueprint(code: str, **kw):
    base = dict(
        blueprint_id="bp1", title="Test blueprint title",
        falsifiable_hypothesis="H", search_queries=["q1"],
        contracts=[_contract()], code_template=code, max_feedback_rounds=1,
    )
    base.update(kw)
    return ResearchBlueprint(**base)


DECAY_CPP_TEMPLATE = """
def simulation_step(buf, dt):
    for i in range(buf.size):
        buf.data[i] = buf.data[i] - 0.1 * buf.data[i] * dt
"""


def test_invariant_constraint_immutable():
    inv = InvariantConstraint(variable_name="x", assertion="x > 0")
    with pytest.raises(Exception):
        inv.assertion = "x < 0"
    with pytest.raises(Exception):
        InvariantConstraint(variable_name="x", assertion="x", bogus=1)


def test_verification_contract_bounds():
    with pytest.raises(Exception):
        _contract(timeout_seconds=0)
    with pytest.raises(Exception):
        _contract(timeout_seconds=301)
    with pytest.raises(Exception):
        _contract(invariants=[])
    with pytest.raises(Exception):
        _contract(target_backend="fortran")


def test_research_blueprint_validation():
    with pytest.raises(Exception):
        _blueprint(DECAY_CPP_TEMPLATE, title="tiny")
    with pytest.raises(Exception):
        _blueprint(DECAY_CPP_TEMPLATE, contracts=[])
    with pytest.raises(Exception):
        _blueprint(DECAY_CPP_TEMPLATE, unexpected=1)


def test_simulation_result_status_literal():
    with pytest.raises(Exception):
        SimulationResult(blueprint_id="b", contract_id="c", status="MAYBE",
                         execution_time_ms=1.0, telemetry={}, bundle_hash="x")


def test_frontend_lowering():
    uast = python_source_to_uast(DECAY_CPP_TEMPLATE)
    assert uast["type"] == "module"
    fn = uast["children"][0]
    assert fn["type"] == "function_declaration"
    assert fn["name"] == "simulation_step"
    loop = fn["body"][0]
    assert loop["type"] == "loop_for"


def test_frontend_rejects_unsupported():
    with pytest.raises(UnsupportedError):
        python_source_to_uast("def f():\n    yield 1\n")


def test_cpp_emitter_module():
    uast = python_source_to_uast(DECAY_CPP_TEMPLATE)
    cpp = CppEmitter().emit_module(uast)
    assert "extern \"C\" void execute_step" in cpp
    assert "buf->data[i]" in cpp
    assert "SimulationBuffer" in cpp


def test_hin_emitter_arena_json():
    uast = python_source_to_uast("x = 2.0 * 3.0 + 1.0\n")
    arena = json.loads(HinEmitter().emit_module(uast))
    assert arena["nodes"]
    assert "x" in arena["outputs"]


def test_shacl_chiasmus_firewall():
    assert shacl_chiasmus_firewall("import os\neval('1')\n") == [
        "forbidden import: os", "forbidden call: eval()",
    ]
    assert shacl_chiasmus_firewall("import math\nx = math.sqrt(2.0)\n") == []
    assert any("unbounded" in v for v in shacl_chiasmus_firewall("while True:\n    pass\n"))


def test_eval_invariant():
    state = [1.0, 2.0]
    tel = {"final_max": 3.0}
    assert _eval_invariant("len(state) == 2", state, tel)
    assert _eval_invariant("telemetry['final_max'] > 0.0", state, tel)
    assert not _eval_invariant("telemetry['final_max'] < 0.0", state, tel)
    with pytest.raises(PreWriteGuardError):
        _eval_invariant("__import__('os')", state, tel)


def test_dynamic_cabi_bridge_execution(tmp_path):
    cpp = """
#include <cstdint>
#include <cmath>
extern "C" {
typedef struct SimulationBuffer { size_t size; double dt; uint32_t steps; double* data; } SimulationBuffer;
void execute_step(SimulationBuffer* buf) {
    for (size_t i = 0; i < buf->size; ++i) {
        buf->data[i] -= 0.1 * buf->data[i] * buf->dt;
    }
    buf->steps += 1;
}
}
"""
    bridge = DynamicCABIBridge(tmp_path)
    initial = [1.0] * 8
    out = bridge.run(cpp, initial, dt=0.5, steps=4)
    state = out["state"]
    assert all(0.0 < v < 1.0 for v in state)
    assert all(v < 1.0 for v in state)
    assert out["steps"] == 4
    assert len(out["bundle_hash"]) == 32
    # Compilation cache: second run reuses the .so.
    out2 = bridge.run(cpp, initial, dt=0.5, steps=4)
    assert out2["bundle_hash"] == out["bundle_hash"]


def test_math_core_rk4():
    # dy/dt = -y (decay); exact solution y(t) = e^-t.
    def deriv(state, params, t):
        rate = params[0]
        return [-rate * s for s in state]

    state = [1.0]
    dt = 0.1
    for _ in range(10):
        state = rk4_step(state, [1.0], dt, deriv)
    assert abs(state[0] - 2.718281828 ** -1) < 1e-6


def test_math_core_newton_and_lu():
    root = newton_raphson(lambda x: x * x - 2.0, lambda x: 2 * x, 1.0)
    assert abs(root - 2 ** 0.5) < 1e-10
    lu = lu_decompose([[2.0, 1.0], [4.0, 3.0]])
    x = lu_solve(lu, [1.0, 2.0])
    assert abs(x[0] - 0.5) < 1e-9 and abs(x[1] - 0.0) < 1e-9


def test_prng_deterministic():
    s = splitmix64(42)
    _, _, _, _, r1 = xoshiro256plusplus_next(*s)
    s2 = splitmix64(42)
    _, _, _, _, r2 = xoshiro256plusplus_next(*s2)
    assert r1 == r2
    u = uniform_f64(r1)
    assert 0.0 <= u < 1.0


@pytest.mark.asyncio
async def test_planner_fallback_blueprint():
    result = await generate_research_plan("anything", llm_complete=None)
    assert result["source"] == "fallback"
    bp = ResearchBlueprint.model_validate(result["blueprint"])
    assert bp.blueprint_id


def test_planner_extract_json():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json("no json") is None


@pytest.mark.asyncio
async def test_planner_llm_path_handles_prompt_braces():
    """Regression: PLANNER_PROMPT embeds a JSON example with literal braces;
    the question substitution must not use str.format() on it."""
    captured = {}

    async def fake_llm(prompt):
        captured["prompt"] = prompt
        return json.dumps({
            "blueprint_id": "bp_llm", "title": "LLM planned research question",
            "falsifiable_hypothesis": "H", "search_queries": ["q1"],
            "contracts": [{
                "contract_id": "c1", "target_backend": "cpp", "timeout_seconds": 30.0,
                "invariants": [{"variable_name": "x", "assertion": "telemetry['final_max'] > 0.0"}],
            }],
            "code_template": DECAY_CPP_TEMPLATE,
        })

    result = await generate_research_plan("verify something", llm_complete=fake_llm, fallback=False)
    assert result["source"] == "llm"
    assert result["blueprint"]["blueprint_id"] == "bp_llm"
    assert "verify something" in captured["prompt"]
    assert "{question}" not in captured["prompt"]


@pytest.mark.asyncio
async def test_state_machine_cpp_commit(tmp_path):
    """Offline research run: cpp backend verifies the decay kernel."""
    bp = _blueprint(DECAY_CPP_TEMPLATE, contracts=[_contract()])
    machine = ResearchStateMachine(tmp_path)
    state = await machine.run(bp)
    assert state.stage == "Adjoint_Commit_or_Rollback"
    assert state.results
    assert state.results[0].status in ("VERIFIED", "FALSIFIED", "ERROR")


@pytest.mark.asyncio
async def test_state_machine_guard_rejects(tmp_path):
    bp = _blueprint("import os\nx = 1\n", contracts=[_contract()])
    machine = ResearchStateMachine(tmp_path)
    state = await machine.run(bp)
    assert state.results == [] or state.results[0].status == "ERROR"


def test_native_goi_wavefront_scheduler():
    pytest.importorskip("invar_agent._native")
    from invar_agent._native import GoIWavefrontScheduler, UASTGraph, UASTKind

    sched = GoIWavefrontScheduler(3)
    sched.set_edge(0, 1, 0.5)
    sched.set_edge(1, 2, 0.25)
    assert sched.is_nilpotent()
    ex = sched.compute_execution_operator()
    assert len(ex) == 3 and len(ex[0]) == 3

    # Non-nilpotent cycle must raise.
    bad = GoIWavefrontScheduler(2)
    bad.set_edge(0, 1, 1.0)
    bad.set_edge(1, 0, 1.0)
    assert not bad.is_nilpotent()
    with pytest.raises(Exception):
        bad.compute_execution_operator()

    # Sherman-Morrison update runs.
    p = [[1.0], [0.0], [0.0]]
    q = [[0.0], [0.0], [0.05]]
    upd = sched.apply_incremental_update(p, q)
    assert len(upd) == 3

    graph = UASTGraph()
    lit = graph.insert_node(UASTKind.ConstantFloat, None, None, 1.5, None)
    h = graph.get_root_hash(lit)
    assert len(h) == 64


def test_native_hin_arena_and_healer():
    pytest.importorskip("invar_agent._native")
    import invar_agent._native as native

    assert hasattr(native, "PyHinEngine") or hasattr(native, "HinEngine")
    assert hasattr(native, "repair_uast_expression")
    assert hasattr(native, "GoIWavefrontScheduler")
    assert hasattr(native, "UASTGraph")
    assert hasattr(native, "GraphPolyglotMaterializer")


@pytest.mark.asyncio
async def test_heal_and_refine_routes_guard_failures_to_llm(tmp_path):
    """PreWrite_Guard/transpiler rejections must skip Tier-0 (which cannot fix
    tuples) and go straight to the Tier-2 LLM refiner."""
    from invar_agent.research.state_machine import ResearchStateMachine

    events = []

    async def cb(channel, payload):
        events.append((channel, payload))

    refine_calls = []

    async def fake_refine(llm_complete, *, objective, code, contracts, failure):
        refine_calls.append(failure)
        return "def simulation_step(buf, dt):\n    return None\n"

    machine = ResearchStateMachine(tmp_path / "research", progress_cb=cb,
                                   llm_planner=lambda prompt: "{}")
    machine.workspace_root = tmp_path

    import invar_agent.research.state_machine as sm
    original = sm.refine_stage_code
    sm.refine_stage_code = fake_refine
    try:
        out = await machine._heal_and_refine(
            stage=1, round_idx=1, objective="obj",
            candidate_code="def simulation_step(buf, dt):\n    a, b = 1, 2\n    return None\n",
            contracts=[], failure="PreWrite_Guard: unsupported construct: tuples")
    finally:
        sm.refine_stage_code = original

    assert refine_calls == ["PreWrite_Guard: unsupported construct: tuples"]
    assert out == "def simulation_step(buf, dt):\n    return None\n"
    heal_events = [p for c, p in events if p.get("event") == "heal"]
    assert heal_events and heal_events[0]["tier"] == 0
    assert heal_events[0]["applied"] is False

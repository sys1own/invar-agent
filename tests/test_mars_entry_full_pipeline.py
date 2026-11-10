"""Hermetic end-to-end Mars entry pipeline (RK4 ballistic kinematics +
Sutton-Graves stagnation heating).

Everything is offline: the LLM blueprint/accretive synthesis responses are
mocked with the exact 2-contract Python kernel, literature search is faked,
and the kernels compile locally (g++/clang++) or run in the deterministic
Python sandbox fallback. No DeepSeek calls.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from invar_agent.research.planner import _sanitize_raw_blueprint_dict
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
from invar_agent.schemas.research import ResearchBlueprint
from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
from invar_agent.transpiler.frontend import python_source_to_uast

HAS_CXX = bool(shutil.which("g++") or shutil.which("clang++"))

# State layout: [h_m, v_m_s, gamma_rad, peak_decel_g, heat_load_J_m2,
#                heat_flux_W_m2, h_peak_m]
INITIAL_STATE = [150000.0, 4000.0, -0.45, 0.0, 0.0, 0.0, 150000.0]
DT = 0.05
STEPS = 2000

STAGE1_KINEMATICS = '''
import math

G0 = 3.71
R_MARS = 3396000.0
H_SCALE = 11100.0
RHO0 = 0.020
CD = 1.8
AREA = 1.54
MASS = 3000.0
R_N = 0.7


def density(h):
    return RHO0 * math.exp(-h / H_SCALE)


def drag_force(v, h):
    rho = density(h)
    return 0.5 * rho * v * v * CD * AREA


def dh_dt(h, v, gamma):
    return v * math.sin(gamma)


def dv_dt(h, v, gamma):
    drag = drag_force(v, h)
    g_local = G0 * (R_MARS / (R_MARS + h)) ** 2
    return -drag / MASS - g_local * math.sin(gamma)


def dg_dt(h, v, gamma):
    g_local = G0 * (R_MARS / (R_MARS + h)) ** 2
    return (v / (R_MARS + h)) * math.cos(gamma) - g_local * math.cos(gamma) / max(v, 1.0)


def simulation_step(buf, dt):
    state = buf.data
    h = state[0]
    v = state[1]
    gamma = state[2]
    a_max = state[3]
    h_peak = state[6]

    k1_v, k1_gamma, k1_h = dv_dt(h, v, gamma), dg_dt(h, v, gamma), dh_dt(h, v, gamma)
    k2_v, k2_gamma, k2_h = dv_dt(h + 0.5 * dt * k1_h, v + 0.5 * dt * k1_v, gamma + 0.5 * dt * k1_gamma), dg_dt(h + 0.5 * dt * k1_h, v + 0.5 * dt * k1_v, gamma + 0.5 * dt * k1_gamma), dh_dt(h + 0.5 * dt * k1_h, v + 0.5 * dt * k1_v, gamma + 0.5 * dt * k1_gamma)
    k3_v, k3_gamma, k3_h = dv_dt(h + 0.5 * dt * k2_h, v + 0.5 * dt * k2_v, gamma + 0.5 * dt * k2_gamma), dg_dt(h + 0.5 * dt * k2_h, v + 0.5 * dt * k2_v, gamma + 0.5 * dt * k2_gamma), dh_dt(h + 0.5 * dt * k2_h, v + 0.5 * dt * k2_v, gamma + 0.5 * dt * k2_gamma)
    k4_v, k4_gamma, k4_h = dv_dt(h + dt * k3_h, v + dt * k3_v, gamma + dt * k3_gamma), dg_dt(h + dt * k3_h, v + dt * k3_v, gamma + dt * k3_gamma), dh_dt(h + dt * k3_h, v + dt * k3_v, gamma + dt * k3_gamma)

    h_new = h + dt * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h) / 6.0
    v_new = v + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    gamma_new = gamma + dt * (k1_gamma + 2.0 * k2_gamma + 2.0 * k3_gamma + k4_gamma) / 6.0

    g_decel = drag_force(v_new, h_new) / MASS / 9.81
    state[0] = h_new
    state[1] = v_new
    state[2] = gamma_new
    state[3] = max(a_max, g_decel)
    if h_new > h_peak:
        state[6] = h_new
'''

STAGE2_HEATING = '''
import math

K_SG = 1.83e-4


def stagnation_heat(v, h):
    rho = density(h)
    return K_SG * math.sqrt(rho / R_N) * v ** 3


def simulation_step(buf, dt):
    layer_1_step(buf, dt)
    data = buf.data
    q_s = stagnation_heat(data[1], data[0])
    data[5] = q_s
    data[4] = data[4] + q_s * dt
'''

# Stress source exercising every scientific construct the emitter must lower.
SCIENTIFIC_CONSTRUCTS = '''
import math

K = 2.5


def helper_norm(a, b):
    return math.sqrt(a ** 2 + b ** 2)


def simulation_step(buf, dt):
    state = buf.data
    x = buf[0]
    y = state[1]
    p, q = x * math.pi, y * math.e
    s = math.sin(p) + math.cos(q) + math.tan(0.1 * x)
    r = math.exp(-abs(x)) + math.log(1.0 + abs(y)) + math.fabs(s)
    t = math.atan2(y, x + 1.0)
    w = min(max(r, -10.0), 10.0) + helper_norm(x, y) / K
    acc = 0.0
    for i in range(3):
        acc = acc + w * 0.001
    state[0] = x + dt * (t + acc)
    state[1] = y - dt * x ** 2 * 0.01
    state[2] = s
    state[3] = r
'''


def _llm_blueprint_json() -> Dict[str, Any]:
    """Exact mocked LLM planner response (raw JSON dict)."""
    return {
        "blueprint_id": "mars_entry_rk4_sg",
        "title": "Mars Atmospheric Entry: RK4 Ballistic Kinematics & Sutton-Graves Heating",
        "falsifiable_hypothesis": ("A 3000 kg ballistic probe entering at 4 km/s, "
                                   "gamma=-0.45 rad stays below 40 g peak deceleration "
                                   "with finite stagnation heat load."),
        "search_queries": ["Mars entry Sutton-Graves", "Mars ballistic entry RK4"],
        "target_research_stages": 2,
        "stage_objectives": ["RK4 ballistic kinematics with exponential atmosphere",
                             "Sutton-Graves stagnation heating accretion"],
        "contracts": [
            {"contract_id": "c1_ballistic_kinematics", "target_backend": "cpp",
             "timeout_seconds": 60.0, "max_memory_mb": 1024,
             "initial_state": INITIAL_STATE, "dt": DT, "steps": STEPS,
             "invariants": [
                 # Hallucinated narrow interval (observed peak ~12.1 g): the
                 # Tier-0 reconciler must relax the lower bound only.
                 {"variable_name": "peak_decel_g",
                  "assertion": "state[3] >= 15.0 and state[3] < 40.0",
                  "tolerance": 0.0},
                 {"variable_name": "velocity_monotone",
                  "assertion": "buf.data[1] <= initial_state[1]", "tolerance": 0.0},
                 {"variable_name": "finite_altitude",
                  "assertion": "math.isfinite(data[0]) and isfinite(state[2])",
                  "tolerance": 0.0},
                 {"variable_name": "bounded",
                  "assertion": "telemetry['final_max'] < 1e7", "tolerance": 0.0},
             ]},
            {"contract_id": "c2_heating_accretion", "target_backend": "cpp",
             "timeout_seconds": 60.0, "max_memory_mb": 1024,
             "initial_state": INITIAL_STATE, "dt": DT, "steps": STEPS,
             "invariants": [
                 # References heat slots that only exist after stage 2: would
                 # falsify Stage 1 if contract staging were broken.
                 {"variable_name": "heat_load_positive",
                  "assertion": "state[4] > 0.0 and math.isfinite(state[4])",
                  "tolerance": 0.0},
                 {"variable_name": "heat_flux_nonneg",
                  "assertion": "data[5] >= 0.0", "tolerance": 0.0},
             ]},
        ],
        "code_template": STAGE1_KINEMATICS,
        "variable_map": {"peak_decel_g": 3, "heat_load_J_m2": 4,
                         "heat_flux_W_m2": 5, "final_altitude_m": 0},
        "max_feedback_rounds": 2,
    }


def _mock_blueprint() -> ResearchBlueprint:
    return ResearchBlueprint.model_validate(
        _sanitize_raw_blueprint_dict(_llm_blueprint_json()))


def _mock_stage2_synthesizer():
    async def synth(**kw):
        return {"code": STAGE2_HEATING, "contracts": [],
                "rationale": "accrete Sutton-Graves heating onto verified kinematics"}
    return synth


class _FakeHit:
    def __init__(self, title: str) -> None:
        self.source = "arxiv"
        self.title = title
        self.year = 2024
        self.url = "https://arxiv.org/abs/2401.00001"
        self.summary = "Sutton-Graves constant k = 1.83e-4 for Mars CO2 atmosphere."


class _FakeSearch:
    def search(self, q: str, limit: int = 3) -> List[_FakeHit]:
        return [_FakeHit(f"Paper about {q}")][:limit]


def _collect_events() -> Tuple[List[Tuple[str, Dict[str, Any]]], Any]:
    events: List[Tuple[str, Dict[str, Any]]] = []

    async def cb(channel: str, payload: Dict[str, Any]) -> None:
        events.append((channel, payload))

    return events, cb


def _activity(events) -> List[Dict[str, Any]]:
    return [p for ch, p in events if ch == "activity_log" and p.get("event") == "activity"]


def _results(events) -> List[Dict[str, Any]]:
    return [p for ch, p in events if p.get("event") == "result"]


# -- Test 1: emitter scientific constructs + deterministic Python fallback ------

@pytest.mark.asyncio
async def test_cpp_emitter_scientific_constructs_and_fallback(tmp_path):
    emitter = CppEmitter()
    for src in (SCIENTIFIC_CONSTRUCTS, STAGE1_KINEMATICS):
        cpp = emitter.emit_module(python_source_to_uast(src))
        assert "extern \"C\"" in cpp
        assert "std::pow" in cpp or "pow(" in cpp
        assert "buf->data" in cpp
    cpp = emitter.emit_module(python_source_to_uast(SCIENTIFIC_CONSTRUCTS))
    assert "3.14159265358979" in cpp      # math.pi lowered to a literal
    assert "2.71828182845904" in cpp      # math.e lowered to a literal
    for fn in ("sin", "cos", "tan", "exp", "log", "fabs", "atan2", "sqrt"):
        assert re.search(rf"\b{fn}\(", cpp), f"math.{fn} not lowered"

    bridge = DynamicCABIBridge(tmp_path / "scratch")
    init = [0.3, 0.7, 0.0, 0.0]
    py_run = bridge.run_python(SCIENTIFIC_CONSTRUCTS, init, 0.01, 50)
    assert py_run["finite"] and py_run.get("fallback") == "python"
    if HAS_CXX:
        # Compiled C++ and the Python sandbox agree numerically.
        c_run = bridge.run(cpp, init, 0.01, 50)
        assert c_run["finite"]
        for a, b in zip(c_run["state"], py_run["state"]):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

    # Toolchain rejection after PreWrite_Guard -> Python sandbox, never ERROR.
    events, cb = _collect_events()
    machine = ResearchStateMachine(tmp_path / "ws", progress_cb=cb,
                                   search_client=_FakeSearch())
    machine.bridge.compiler = "/bin/false"  # every compile fails
    bp = _mock_blueprint().model_copy(update={
        "target_research_stages": 1, "stage_objectives": ["kinematics"],
        "contracts": [_mock_blueprint().contracts[0]]})
    state = await machine.run(bp)
    assert state.status == "VERIFIED", [r for r in _results(events)]
    steps = [a["step"] for a in _activity(events)]
    assert "cabi_fallback" in steps
    fallback_msgs = [a["message"] for a in _activity(events) if a["step"] == "cabi_fallback"]
    assert "Python kernel" in fallback_msgs[0]
    for r in _results(events):
        assert r["status"] != "ERROR"
        assert r.get("telemetry", {}).get("error") != 1.0


# -- Test 2: full 2-contract Mars entry state machine (offline) -----------------

@pytest.mark.asyncio
async def test_two_contract_mars_entry_state_machine_offline(tmp_path):
    events, cb = _collect_events()
    machine = ResearchStateMachine(
        tmp_path, progress_cb=cb, search_client=_FakeSearch(),
        stage_synthesizer=_mock_stage2_synthesizer())
    if not HAS_CXX:
        machine.bridge.compiler = "/bin/false"  # exercise Python fallback instead
    state = await machine.run(_mock_blueprint())

    assert state.status == "VERIFIED", [
        (r["contract_id"], r["status"], r.get("failure_counterexample"))
        for r in _results(events)]
    assert len(state.blueprint.verified_layers) == 2

    results = _results(events)
    by_contract: Dict[str, List[str]] = {}
    for r in results:
        by_contract.setdefault(r["contract_id"], []).append(r["status"])
    # Stage staging: c1 runs in Stage 1, c2 only in Stage 2 (never breaks Stage 1).
    assert by_contract["c1_ballistic_kinematics"][-1] == "VERIFIED"
    assert by_contract["c2_heating_accretion"][-1] == "VERIFIED"
    stage1_ids = {r["contract_id"] for r in results if r.get("research_stage") == 1}
    assert "c2_heating_accretion" not in stage1_ids
    assert all(s == "VERIFIED" for s in by_contract["c1_ballistic_kinematics"])

    # Hallucinated lower bound relaxed deterministically (Tier-0), upper kept.
    acts = _activity(events)
    reconciled = [a for a in acts if a["step"] == "bound_reconciled"]
    assert reconciled and "state[3] < 40.0" in reconciled[0]["message"]

    # Every process log stage emitted, in pipeline order.
    stages = [a["stage"] for a in acts]
    for expected in ("FEASIBILITY", "INGEST", "STAGE_1_BUILD", "STAGE_1_EXEC",
                     "STAGE_2_BUILD", "STAGE_2_EXEC", "REPORT", "VAULT"):
        assert expected in stages, f"missing activity stage {expected}"
    assert stages.index("STAGE_1_EXEC") < stages.index("STAGE_2_EXEC") < stages.index("REPORT")

    # Physically meaningful verified observables.
    tel = state.blueprint.verified_layers[-1].telemetry
    assert 5.0 < tel["c2_heating_accretion.obs_peak_decel_g"] < 40.0
    assert tel["c2_heating_accretion.obs_heat_load_J_m2"] > 0.0

    # Report generated + committed to the vault and exposed over HTTP.
    completed = [p for _, p in events if p.get("event") == "research_completed"]
    assert completed and completed[-1]["status"] == "VERIFIED"
    run_dir = tmp_path / ".invar" / "research_vault" / "mars_entry_rk4_sg"
    for name in ("metadata.json", "report.md", "unified_module.py", "blueprint.json"):
        assert (run_dir / name).is_file(), f"missing vault artifact {name}"
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "VERIFIED"

    import invar_agent.server as server
    server._STATE = None
    client = TestClient(server.create_app(workspace_root=str(tmp_path)))
    vault = client.get("/api/v1/research/vault").json()
    statuses = {r["run_id"]: r["status"] for r in vault["runs"]}
    assert statuses.get("mars_entry_rk4_sg") == "VERIFIED"


# -- Test 3: UI layout -- no blocking log box, inline process log card ----------

def test_ui_layout_no_blocking_log_box():
    html = (Path(__file__).resolve().parents[1] / "invar_agent" / "static"
            / "index.html").read_text(encoding="utf-8")
    # The separate fixed-height activity container above the chat is gone.
    assert 'id="research-activity-feed"' not in html
    assert ".research-activity-feed" not in html
    # Process logs render inline inside the chat as a chronological log card.
    assert ".chat-process-log-card" in html
    m = re.search(r"function appendActivity\(msg\) \{(.*?)\n\}", html, re.S)
    assert m, "appendActivity missing"
    body = m.group(1)
    assert "ensureRunLogCard()" in body
    card = re.search(r"function ensureRunLogCard\(\) \{(.*?)\n\}", html, re.S).group(1)
    assert "chat-process-log-card" in card and "chat.appendChild" in card
    # The research panel above chat holds only the slim stepper + telemetry.
    panel = re.search(r'<div class="research-panel" id="research-panel">(.*?)</div>\s*<div class="chat"',
                      html, re.S)
    assert panel and "activity" not in panel.group(1)
    # Dedicated runtime drawer with Terminal tab, collapsed by default.
    assert 'id="runtime-drawer" class="runtime-drawer collapsed"' in html
    assert "data-tab=\"terminal\"" in html and "Terminal" in html
    assert re.search(r"\.runtime-drawer \{[^}]*height:36px", html)
    # Opening a drawer tab expands the drawer.
    sw = re.search(r"function switchDrawerTab\(tab\) \{(.*?)\n\}", html, re.S).group(1)
    assert "toggleDrawer()" in sw

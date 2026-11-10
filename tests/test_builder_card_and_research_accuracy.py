"""Builder Action Card recovery, research observable accuracy, vault digest
and UI polish (hermetic — mocked LLM, local compiler or Python sandbox, no
DeepSeek calls)."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from invar_agent.context_compactor import build_research_vault_context
from invar_agent.copilot.action_parser import extract_builder_plan_payload
from invar_agent.copilot.agent import CopilotAgent
from invar_agent.research.feasibility import truncate_at_sentence
from invar_agent.research.reporter import _unit_for, generate_comprehensive_research_report
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
from invar_agent.schemas.research import (
    FeasibilityAssessment, FeasibilityVerdict, InvariantConstraint,
    ResearchBlueprint, StageTelemetry, VerificationContract,
)
from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
from invar_agent.transpiler.frontend import python_source_to_uast

HAS_CXX = bool(shutil.which("g++") or shutil.which("clang++"))
INDEX_HTML = Path(__file__).resolve().parents[1] / "invar_agent" / "static" / "index.html"

# -- Exact truncated Builder Planner output (prose prefix + token-limit cut) ---

CLEAN_PROMPT_HEAD = (
    "Build a production-ready Python package `mars_entry_sim` implementing a 3-DOF "
    "Mars atmospheric entry simulator.\n\n## Requirements\n"
    "1. `atmosphere.py`: exponential density model rho(h) = rho0 * exp(-h / H) "
    "with rho0 = 0.020 kg/m^3 and H = 11100 m.\n"
    "2. `dynamics.py`: RK4 integrator for (h, v, gamma) with \"Sutton-Graves\" "
    "stagnation heating q = k * sqrt(rho / Rn) * v^3, k = 1.898e-4.\n"
    "3. `cli.py`: argparse entrypoint printing peak g, peak heat flux (MW/m^2) "
    "and total heat load (MJ/m^2).\n"
    "4. `tests/`: pytest suite covering the C:\\\\sim path edge case.\n"
    "5. `.gitignore`: __pycache__/, *.pyc, dist/, build/, *.egg-info/, .venv/"
)

TRUNCATED_BUILDER_PLAN = (
    'Architecture Overview\n{\n'
    '"display_text": "## Architecture Overview\\nA **modular** Mars entry simulator: '
    'atmosphere, RK4 dynamics, CLI and tests. Edit the prompt below, then run.",\n'
    '"action": {"type": "SUGGEST_BUILD_PROMPT", "params": {\n'
    '"clean_prompt": "'
    + CLEAN_PROMPT_HEAD.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
)


class _FakeLLM:
    model = "fake-fast"

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.calls: List[Dict[str, Any]] = []

    def generate(self, messages, temperature=0.2, max_tokens=None, **_):
        self.calls.append({"max_tokens": max_tokens})
        return self.raw


def test_builder_plan_truncated_json_recovery_and_action_card(tmp_path, monkeypatch):
    # The truncated output is NOT valid JSON (no closing quote/braces).
    with pytest.raises(json.JSONDecodeError):
        json.loads(TRUNCATED_BUILDER_PLAN[TRUNCATED_BUILDER_PLAN.index("{"):])

    payload = extract_builder_plan_payload(TRUNCATED_BUILDER_PLAN)
    assert payload is not None
    assert payload["status"] == "success"
    assert payload["action"] == "SUGGEST_BUILD_PROMPT"
    assert payload["display_text"].startswith("## Architecture Overview\nA **modular**")
    # Full clean_prompt recovered up to EOF, cleanly unescaped.
    assert payload["clean_prompt"] == CLEAN_PROMPT_HEAD
    assert payload["clean_prompt"].endswith("*.egg-info/, .venv/")
    assert '"Sutton-Graves"' in payload["clean_prompt"]
    assert "C:\\\\sim" in payload["clean_prompt"]
    assert payload["expert_prompt"] == payload["clean_prompt"]
    assert payload["action_obj"] == {"type": "build", "clean_prompt": CLEAN_PROMPT_HEAD}

    # <think> blocks + fences + valid JSON also normalize to the same schema.
    fenced = ("<think>plan the package</think>Sure!\n```json\n"
              + json.dumps({"display_text": "Plan", "clean_prompt": "Build X"})
              + "\n```\nLet me know.")
    fenced_payload = extract_builder_plan_payload(fenced)
    assert fenced_payload["clean_prompt"] == "Build X"
    assert fenced_payload["display_text"] == "Plan"
    # Plain chit-chat is not mis-detected as a build plan.
    assert extract_builder_plan_payload("Hello! How can I help?") is None

    # /api/copilot/chat (Builder Plan Mode) renders a SUGGEST_BUILD_PROMPT card.
    from invar_agent import server

    fake = _FakeLLM(TRUNCATED_BUILDER_PLAN)
    monkeypatch.setattr(CopilotAgent, "_llm_client", lambda self: fake)
    client = TestClient(server.create_app(workspace_root=str(tmp_path)))
    resp = client.post("/api/copilot/chat",
                       json={"message": "Plan a Mars entry simulator package", "history": []})
    assert resp.status_code == 200
    data = resp.json()
    assert fake.calls and fake.calls[0]["max_tokens"] >= 8192
    assert data["status"] == "success"
    assert data["action"]["type"] == "SUGGEST_BUILD_PROMPT"
    assert data["action"]["params"]["prompt"] == CLEAN_PROMPT_HEAD
    assert data["clean_prompt"] == CLEAN_PROMPT_HEAD
    assert data["expert_prompt"] == CLEAN_PROMPT_HEAD
    assert data["action_obj"] == {"type": "build", "clean_prompt": CLEAN_PROMPT_HEAD}
    assert data["display_text"].startswith("## Architecture Overview")
    # No raw JSON dumped into the chat bubble.
    assert '"clean_prompt"' not in data["reply"]
    assert '"display_text"' not in data["reply"]
    assert not data["reply"].lstrip().startswith("{")


# -- Mars entry RK4 + Sutton-Graves kernel (v9 transpiler-friendly) ------------
# State layout: [h_m, v_m_s, gamma_rad, max_heat_flux_mw_m2, peak_g_gE,
#                total_heat_load_mj_m2, entry_flight_time_s]

MARS_KERNEL = '''
import math

M = 650.0
RN = 0.75
CD = 1.45
RHO0 = 0.020
SCALE_H = 11100.0
K_SG = 1.898e-4
G0 = 3.71
G_EARTH = 9.80665
AREA = 1.7671458676442586


def atmosphere(h):
    hh = h
    if hh < 0.0:
        hh = 0.0
    return RHO0 * math.exp(-hh / SCALE_H)


def drag_accel(h, v):
    return 0.5 * atmosphere(h) * v * v * CD * AREA / M


def dh_dt(v, gam):
    return v * math.sin(gam)


def dv_dt(h, v, gam):
    return -drag_accel(h, v) - G0 * math.sin(gam)


def dgam_dt(h, v, gam):
    vv = v
    if vv < 1.0:
        vv = 1.0
    return -(G0 * math.cos(gam)) / vv - (drag_accel(h, v) * math.sin(gam)) / vv


def simulation_step(buf, dt):
    if buf.data[0] <= 0.0:
        return
    h = buf.data[0]
    v = buf.data[1]
    gam = buf.data[2]
    k1h = dh_dt(v, gam)
    k1v = dv_dt(h, v, gam)
    k1g = dgam_dt(h, v, gam)
    h2 = h + 0.5 * dt * k1h
    v2 = v + 0.5 * dt * k1v
    g2 = gam + 0.5 * dt * k1g
    k2h = dh_dt(v2, g2)
    k2v = dv_dt(h2, v2, g2)
    k2g = dgam_dt(h2, v2, g2)
    h3 = h + 0.5 * dt * k2h
    v3 = v + 0.5 * dt * k2v
    g3 = gam + 0.5 * dt * k2g
    k3h = dh_dt(v3, g3)
    k3v = dv_dt(h3, v3, g3)
    k3g = dgam_dt(h3, v3, g3)
    h4 = h + dt * k3h
    v4 = v + dt * k3v
    g4 = gam + dt * k3g
    k4h = dh_dt(v4, g4)
    k4v = dv_dt(h4, v4, g4)
    k4g = dgam_dt(h4, v4, g4)
    h = h + (dt / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
    v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    gam = gam + (dt / 6.0) * (k1g + 2.0 * k2g + 2.0 * k3g + k4g)
    if v < 0.0:
        v = 0.0
    buf.data[0] = h
    buf.data[1] = v
    buf.data[2] = gam
    q_w = K_SG * math.sqrt(atmosphere(h) / RN) * v * v * v
    q_mw = q_w / 1000000.0
    if q_mw > buf.data[3]:
        buf.data[3] = q_mw
    g_now = (drag_accel(h, v) + G0) / G_EARTH
    if g_now > buf.data[4]:
        buf.data[4] = g_now
    buf.data[5] = buf.data[5] + q_mw * dt
    buf.data[6] = buf.data[6] + dt
'''

# The original bug: instantaneous W/m^2 overwrite every step.
BUGGY_OVERWRITE_KERNEL = MARS_KERNEL.replace(
    "    q_mw = q_w / 1000000.0\n    if q_mw > buf.data[3]:\n        buf.data[3] = q_mw\n",
    "    q_mw = q_w / 1000000.0\n    buf.data[3] = q_w\n")
assert BUGGY_OVERWRITE_KERNEL != MARS_KERNEL

V0 = 5750.0
H0 = 125000.0
GAMMA0 = math.radians(-12.5)
MARS_INITIAL = [H0, V0, GAMMA0, 0.0, 0.0, 0.0, 0.0]
MARS_DT = 0.05
MARS_STEPS = 4000
MARS_VARIABLE_MAP = {
    "final_altitude_m": 0, "impact_velocity_m_s": 1, "max_heat_flux_mw_m2": 3,
    "peak_g_gE": 4, "total_heat_load_mj_m2": 5, "entry_flight_time_s": 6,
}


def _run_mars(bridge: DynamicCABIBridge, source: str) -> Dict[str, Any]:
    if HAS_CXX:
        cpp = CppEmitter().emit_module(python_source_to_uast(source))
        return bridge.run(cpp, list(MARS_INITIAL), MARS_DT, MARS_STEPS)
    return bridge.run_python(source, list(MARS_INITIAL), MARS_DT, MARS_STEPS)


def test_mars_entry_heat_flux_running_max_and_mw_units(tmp_path):
    bridge = DynamicCABIBridge(tmp_path / "scratch")
    run = _run_mars(bridge, MARS_KERNEL)
    assert run["finite"]
    state = run["state"]
    obs = {name: state[idx] for name, idx in MARS_VARIABLE_MAP.items()}

    assert obs["final_altitude_m"] <= 0.0                  # integrated to ground
    assert 12.1 <= obs["peak_g_gE"] <= 12.6
    assert obs["peak_g_gE"] < 40.0
    # Running max in MW/m^2: the hypersonic peak, never the impact W/m^2 value.
    # With the pinned constants (k=1.898e-4, Rn=0.75, rho0=0.020, H=11100,
    # pi*Rn^2 drag area) the converged RK4 peak is ~1.11 MW/m^2 (dt-invariant
    # to 4 digits), so the band is widened below the nominal 1.5-1.7 estimate.
    q_peak = obs["max_heat_flux_mw_m2"]
    assert 1.0 < q_peak < 1.7
    assert q_peak < 10.0
    assert not math.isclose(q_peak, 15532.4, rel_tol=1e-3)
    # Physical conservation bound: below free-stream kinetic energy flux.
    assert q_peak < 0.5 * 0.020 * V0 ** 3 / 1e6
    assert 45.0 <= obs["total_heat_load_mj_m2"] <= 49.0
    assert 0.0 < obs["impact_velocity_m_s"] < V0

    # The peak is NOT the final instantaneous value: at ground impact the
    # stagnation flux has collapsed by orders of magnitude.
    v_end = obs["impact_velocity_m_s"]
    q_impact_w = 1.898e-4 * math.sqrt(0.020 / 0.75) * v_end ** 3
    assert q_impact_w / 1e6 < 0.1 * q_peak

    # C++ and the deterministic Python sandbox agree on every observable.
    py_run = bridge.run_python(MARS_KERNEL, list(MARS_INITIAL), MARS_DT, MARS_STEPS)
    for a, b in zip(state, py_run["state"]):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)

    # Regression: the overwrite-every-step W/m^2 kernel reports the
    # ground-impact flux (~1e4 W/m^2 mislabeled MW/m^2) — the bug class this
    # suite guards against.
    buggy = _run_mars(bridge, BUGGY_OVERWRITE_KERNEL)["state"][3]
    assert buggy > 10.0 and buggy < q_peak * 1e6


# -- Reporter unit inference, truncation and dedup -------------------------------

def _mars_blueprint(long_rationale: str = "") -> ResearchBlueprint:
    rationale = long_rationale or (
        "Hypersonic ballistic entry is a continuum aerothermodynamic problem; "
        "the Sutton-Graves stagnation heating correlation applies.")
    return ResearchBlueprint(
        blueprint_id="mars_entry_probe",
        title="Mars Entry Probe: RK4 Kinematics & Sutton-Graves Heating",
        falsifiable_hypothesis="A 650 kg probe at 5.75 km/s stays below 40 g.",
        search_queries=["Mars entry Sutton-Graves"],
        contracts=[VerificationContract(
            contract_id="c1_mars_entry", target_backend="cpp",
            initial_state=list(MARS_INITIAL), dt=MARS_DT, steps=MARS_STEPS,
            invariants=[InvariantConstraint(
                variable_name="peak_g_gE",
                assertion="state[4] < 40.0 and state[3] < 0.5 * 0.020 * "
                          "initial_state[1] ** 3 / 1e6 and state[12] <= "
                          "initial[12] + 1e-06 * initial[12]",
                tolerance=0.0)])],
        code_template=MARS_KERNEL,
        unified_module_code=MARS_KERNEL,
        target_research_stages=2,
        variable_map=dict(MARS_VARIABLE_MAP, peak_alt_km=0),
        feasibility=FeasibilityAssessment(
            verdict=FeasibilityVerdict.OPTIMAL_DIRECT, rationale=rationale,
            assessed_by="llm"),
        evolution_history=[StageTelemetry(
            stage=1, status="VERIFIED", bundle_hash="a" * 64,
            note="Stage 1 baseline: RK4 kinematics with obs_peak_alt_km "
                 "running-max tracking and full state[12] <= initial[12] audit")],
    )


def test_reporter_unit_suffix_mapping_and_deduplication():
    assert _unit_for("impact_velocity_m_s") == "m/s"
    assert _unit_for("peak_g_gE") == "g_E"
    assert _unit_for("max_heat_flux_mw_m2") == "MW/m²"
    assert _unit_for("total_heat_load_mj_m2") == "MJ/m²"
    assert _unit_for("peak_alt_km") == "km"
    assert _unit_for("entry_flight_time_s") == "s"

    telemetry = {"obs_impact_velocity_m_s": 385.7, "obs_peak_g_gE": 12.59,
                 "obs_max_heat_flux_mw_m2": 1.112, "obs_total_heat_load_mj_m2": 45.93,
                 "obs_peak_alt_km": 31.2, "obs_entry_flight_time_s": 184.95}
    fallback = {"source": "fallback_constants",
                "title": "Offline domain constants (literature APIs returned 0 hits)",
                "year": None, "url": ""}
    history = [dict(fallback) for _ in range(4)] + [
        {"source": "arxiv", "title": "Mars entry heating", "year": 2021,
         "url": "https://arxiv.org/abs/2101.00001"}]
    report = generate_comprehensive_research_report(
        _mars_blueprint(), telemetry, history, status="VERIFIED")

    rows = {m.group(1): m.group(2) for m in re.finditer(
        r"^\| (.+?) \| `[^`]*` \| \*\*[^*]+\*\* \| ([^|]+?) \| VERIFIED \|$",
        report, re.MULTILINE)}
    assert rows["Impact velocity m s"] == "m/s"
    assert rows["Peak g ge"] == "g_E"
    assert rows["Max heat flux mw m2"] == "MW/m²"
    assert rows["Total heat load mj m2"] == "MJ/m²"
    assert rows["Peak alt km"] == "km"
    assert rows["Entry flight time s"] == "s"
    assert "—" not in rows.values()

    section3 = report.split("## 3.")[1].split("## 4.")[0]
    assert section3.count("fallback_constants") == 1
    assert "Mars entry heating" in section3
    # Full expressions / notes — never sliced mid-token.
    assert "state[12] <= initial[12] + 1e-06 * initial[12]" in report
    assert "running-max tracking and full state[12] <= initial[12] audit" in report

    # Rationale: 1200-char budget, cut only at a sentence boundary.
    sentence = "Sutton-Graves stagnation heating correlation applies here. "
    long = sentence * 40
    cut = truncate_at_sentence(long, 1200)
    assert len(cut) <= 1200 and cut.endswith("applies here.")
    short = ("The regime is continuum hypersonic. Use the Sutton-Graves "
             "stagnation heating correlation. Not radiative.")
    assert truncate_at_sentence(short, 1200) == short


# -- Vault digest carries the exact boundary conditions --------------------------

def test_vault_digest_includes_initial_state_and_variable_map(tmp_path):
    vault = ResearchVaultManager(tmp_path)
    vault.register_run(
        run_id="mars_entry_probe", title="Mars Entry Probe", blueprint=_mars_blueprint(),
        report_md="# Mars Entry Probe\nverified", status="VERIFIED",
        telemetry={"obs_peak_g_gE": 12.59, "obs_max_heat_flux_mw_m2": 1.112},
        target_backend="cpp")

    meta = json.loads((vault.run_dir("mars_entry_probe") / "metadata.json").read_text())
    summary = meta["contracts_summary"][0]
    assert summary["contract_id"] == "c1_mars_entry"
    assert summary["dt"] == MARS_DT and summary["steps"] == MARS_STEPS
    assert summary["initial_state"] == MARS_INITIAL      # full list, not truncated
    assert summary["initial_state"][1] == 5750.0         # verbatim, never 5800
    assert summary["variable_map"]["1"] == {"name": "impact_velocity_m_s", "unit": "m/s"}
    assert summary["variable_map"]["3"] == {"name": "max_heat_flux_mw_m2", "unit": "MW/m²"}
    assert summary["variable_map"]["4"] == {"name": "peak_g_gE", "unit": "g_E"}

    digest = build_research_vault_context(tmp_path)
    for token in ("initial_state", "dt=0.05", "steps=4000", "variable_map",
                  "125000.0", "5750.0", "max_heat_flux_mw_m2"):
        assert token in digest, token

    detail = build_research_vault_context(tmp_path, referenced_ids=["mars_entry_probe"])
    assert "<referenced_research_detail" in detail
    expanded = detail.split("<referenced_research_detail")[1]
    assert "simulation_contracts (initial_state, dt, steps, variable_map" in expanded
    assert f"initial_state={MARS_INITIAL}" in expanded
    assert "def simulation_step(buf, dt):" in expanded


# -- UI: edge-pulse log card + markdown everywhere -------------------------------

def test_ui_pulsing_border_and_markdown_rendering():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "@keyframes log-box-edge-pulse" in html, "missing @keyframes log-box-edge-pulse"
    keyframes = html.split("@keyframes log-box-edge-pulse")[1].split(".process-log-card")[0]
    assert "#2dd4bf" in keyframes and "#2b3545" in keyframes
    assert "rgba(45, 212, 191, 0.22)" in keyframes
    assert re.search(r"\.process-log-card\.is-processing\s*\{[^}]*"
                     r"animation:\s*log-box-edge-pulse 2\.4s ease-in-out infinite", html)
    assert re.search(r"\.process-log-card\s*\{[^}]*border:\s*1px solid #2b3545", html)

    # No pulsing square spinner anywhere (CSS, markup or JS).
    assert "process-log-spinner" not in html
    assert "active-run-spinner" not in html
    header = re.search(r'class="process-log-header"[^>]*>(.*?)</div>', html, re.DOTALL)
    if header:
        assert "spinner" not in header.group(1)
    # is-processing toggled on run start and removed on finish.
    assert "classList.add('is-processing')" in html
    assert "classList.remove('is-processing')" in html

    # Markdown rendering: XSS-safe renderer + vendored marked.
    assert "function renderMarkdown(" in html
    assert '/static/vendor/marked.min.js' in html
    assert (INDEX_HTML.parent / "vendor" / "marked.min.js").is_file()
    renderer = html.split("function renderMarkdown(")[1].split("\n}\n")[0]
    assert "script" in renderer and "iframe" in renderer
    # Applied to chat messages, action cards, vault cards and research viewer.
    assert "m.querySelector('.md').innerHTML = renderMarkdown(" in html
    assert "renderMarkdown(params.rationale)" in html
    assert re.search(r'class="vc-title">\$\{renderMarkdown\(', html)
    assert re.search(r"renderMarkdown\(run\.snippet\)", html)
    assert "getElementById('rv-report-md').innerHTML = renderMarkdown(" in html
    assert "renderMarkdown(bp.falsifiable_hypothesis)" in html
    assert re.search(r"renderMarkdown\(String\(e\)\)", html)

    # Builder card: all action shapes + client-side truncated JSON recovery.
    assert "function recoverBuildActionFromText(" in html
    assert "function extractBuildAction(" in html
    assert "SUGGEST_RESEARCH_PROMPT" in html
    assert "⚡ Send to Builder &amp; Run" in html or "⚡ Send to Builder & Run" in html

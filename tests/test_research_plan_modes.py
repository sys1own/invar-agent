"""Plan/Direct modes, fallback eradication, schema-drift coercion, observables."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from invar_agent.research.planner import (
    _extract_json,
    generate_expert_research_prompt,
    generate_plan_with_feasibility,
    normalize_blueprint_payload,
    parse_blueprint,
)
from invar_agent.research.report import build_research_report, extract_observables
from invar_agent.schemas.research import ResearchBlueprint
from invar_agent.server import create_app


# -- JSON extraction / think blocks -------------------------------------------

def test_extract_json_strips_think_blocks_and_fences():
    text = "<think>reasoning about the problem...</think>\n```json\n{\"a\": 1}\n```"
    assert _extract_json(text) == {"a": 1}


def test_extract_json_plain_object():
    assert _extract_json('prefix {"b": [1, 2]} suffix') == {"b": [1, 2]}


# -- schema-drift coercion ------------------------------------------------------

def test_normalize_blueprint_payload_coerces_drift():
    raw = {
        "blueprint_id": "bp1",
        "title": "Thermal shield surrogate",
        "falsifiable_hypothesis": "H",
        "code_template": "def simulation_step(buf, ctx):\n    return None\n",
        "unknown_key": "drop me",
        "search_queries": "single query",
        "contracts": [{
            "contract_id": "c1",
            "target_backend": "bogus-backend",
            "initial_state": [1, 2, 3],
            "dt": "0.01",
            "steps": "100",
            "invariants": [{"variable_name": "x", "assertion": "x > 0", "tolerance": "1e-6"}],
            "extra": "drop",
        }],
        "variable_map": {"magnetic_field_B_tesla": "4", "bad": "nan", "neg": -1},
    }
    out = normalize_blueprint_payload(raw)
    assert "unknown_key" not in out
    assert out["search_queries"] == ["single query"]
    contract = out["contracts"][0]
    assert contract["target_backend"] == "cpp"  # unknown backend coerced
    assert contract["initial_state"] == [1.0, 2.0, 3.0]
    assert contract["dt"] == 0.01 and contract["steps"] == 100
    assert "extra" not in contract
    # "nan" coerced away, negative index dropped, string index coerced
    assert out["variable_map"] == {"magnetic_field_B_tesla": 4}
    bp = parse_blueprint(out)
    assert isinstance(bp, ResearchBlueprint)


# -- combined feasibility + blueprint (single call) ----------------------------

_COMBINED_JSON = (
    "<think>deep reasoning</think>"
    '{"feasibility": {"verdict": "REFORMULATED_SURROGATE",'
    ' "rationale": "0D energy balance surrogate",'
    ' "violated_boundary": "",'
    ' "reformulation": "0D radiative equilibrium: sigma T^4 = absorbed flux"},'
    ' "blueprint": {"blueprint_id": "bp-combined", "title": "Thermal",'
    ' "falsifiable_hypothesis": "T rises monotonically",'
    ' "search_queries": ["q1"],'
    ' "contracts": [{"contract_id": "c1", "target_backend": "cpp",'
    '   "initial_state": [300.0], "dt": 0.01, "steps": 10,'
    '   "invariants": [{"variable_name": "T", "assertion": "T > 0", "tolerance": 1e-6}]}],'
    ' "code_template": "def simulation_step(buf, ctx):\\n    return None\\n",'
    ' "variable_map": {"temperature_K": 0}}}'
)


@pytest.mark.asyncio
async def test_generate_plan_with_feasibility_single_call():
    calls = []

    async def llm(prompt: str):
        calls.append(prompt)
        return _COMBINED_JSON

    out = await generate_plan_with_feasibility("how hot does the shield get?", llm_complete=llm)
    assert len(calls) == 1  # feasibility + blueprint in ONE Tier-2 call
    assert out["source"] == "llm"
    assert out["feasibility"].verdict == "REFORMULATED_SURROGATE"
    assert out["blueprint"]["blueprint_id"] == "bp-combined"
    assert out["blueprint"]["variable_map"] == {"temperature_K": 0}


@pytest.mark.asyncio
async def test_generate_plan_with_feasibility_refusal_fast_path():
    async def llm(prompt: str):  # pragma: no cover - must not be called
        raise AssertionError("LLM must not be called for clear-cut refusals")

    out = await generate_plan_with_feasibility(
        "design a perpetual motion machine with COP = 1.25", llm_complete=llm)
    assert out["blueprint"] is None
    assert out["feasibility"].verdict.startswith("REFUSED")


@pytest.mark.asyncio
async def test_generate_plan_with_feasibility_no_fallback_raises():
    async def llm(prompt: str):
        return "no json here"

    with pytest.raises(RuntimeError):
        await generate_plan_with_feasibility("quantify drag on a magnetic sail",
                                             llm_complete=llm, fallback=False)


@pytest.mark.asyncio
async def test_generate_plan_with_feasibility_offline_fallback_stub():
    # Explicit offline (no-key) mode is the ONLY place the canned stub may appear.
    out = await generate_plan_with_feasibility("quantify drag on a magnetic sail",
                                               llm_complete=None, fallback=True)
    assert out["source"] == "fallback"
    assert out["blueprint"]["blueprint_id"] == "kdv_soliton_scaling"


# -- expert prompt (Plan Mode) ---------------------------------------------------

_EXPERT_JSON = (
    '{"feasibility": {"verdict": "REFORMULATED_SURROGATE",'
    ' "rationale": "surrogate needed", "violated_boundary": "",'
    ' "reformulation": "0D ISM drag balance"},'
    ' "expert_prompt": "# Expert Deep Research Prompt\\n\\n'
    '## 1. Objective & Quantitative Targets\\nCompute B in Tesla."}'
)


@pytest.mark.asyncio
async def test_generate_expert_research_prompt():
    async def llm(prompt: str):
        assert "ResearchBlueprint JSON" in prompt or "expert_prompt" in prompt
        return _EXPERT_JSON

    out = await generate_expert_research_prompt("0.25c spacecraft shield", llm_complete=llm)
    assert out["feasibility"].verdict == "REFORMULATED_SURROGATE"
    assert "Objective & Quantitative Targets" in out["expert_prompt"]
    assert out["source"] == "llm"


@pytest.mark.asyncio
async def test_generate_expert_research_prompt_refused():
    async def llm(prompt: str):  # pragma: no cover - must not be called
        raise AssertionError("LLM must not be called for clear-cut refusals")

    out = await generate_expert_research_prompt(
        "COP = 1.25 heat pump", llm_complete=llm)
    assert out["expert_prompt"] == ""
    assert out["feasibility"].verdict.startswith("REFUSED")


# -- observables + quantitative report -------------------------------------------

def _fake_result(contract_id, stage, telemetry, status="VERIFIED"):
    from invar_agent.schemas.research import SimulationResult
    return SimulationResult(
        blueprint_id="bp-report", contract_id=contract_id, status=status,
        execution_time_ms=12.5, telemetry=telemetry, bundle_hash="ab" * 32,
        stage=stage)


def test_extract_observables_from_variable_map():
    results = [
        _fake_result("c1", 1, {"obs_drag_force_newtons": 1.5e-3, "final_l2": 0.1}),
        _fake_result("c2", 2, {"obs_drag_force_newtons": 2.5e-3, "obs_shield_mass_tons": 850.0}),
    ]
    obs = extract_observables({"drag_force_newtons": 3, "shield_mass_tons": 4,
                               "missing_metric": 9}, results)
    assert obs["drag_force_newtons"]["value"] == 2.5e-3  # later stage wins
    assert obs["drag_force_newtons"]["stage"] == 2
    assert obs["shield_mass_tons"]["value"] == 850.0
    assert "missing_metric" not in obs


def test_build_research_report_contains_quantities():
    from invar_agent.research.state_machine import ResearchRunState
    from invar_agent.schemas.research import (
        FeasibilityAssessment, StageTelemetry,
    )
    bp = ResearchBlueprint(
        blueprint_id="bp-report", title="0.25c shield", falsifiable_hypothesis="H",
        search_queries=["q"],
        contracts=[{
            "contract_id": "c1", "target_backend": "cpp", "initial_state": [1.0],
            "dt": 0.01, "steps": 10,
            "invariants": [{"variable_name": "x", "assertion": "x > 0", "tolerance": 1e-6}],
        }],
        code_template="def simulation_step(buf, ctx):\n    return None\n",
        variable_map={"magnetic_field_B_tesla": 0},
        feasibility=FeasibilityAssessment(
            verdict="REFORMULATED_SURROGATE", rationale="0D surrogate",
            violated_boundary="", reformulation="gamma = 1/sqrt(1-beta^2)",
            assessed_by="llm"),
    )
    state = ResearchRunState(blueprint=bp)
    state.status = "VERIFIED"
    state.results = [_fake_result("c1", 1, {
        "obs_magnetic_field_B_tesla": 0.042, "final_l2": 0.01})]
    bp.evolution_history = [StageTelemetry(
        stage=1, status="VERIFIED", bundle_hash="ab" * 32, note="ok")]
    report = build_research_report(state)
    assert "Verified Quantitative Research Report" in report
    assert "`magnetic_field_B_tesla`" in report
    assert "0.042" in report
    assert "REFORMULATED_SURROGATE" in report
    assert "0.25c shield" in report


# -- API endpoints ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def _hermetic_llm_env(monkeypatch):
    from invar_agent.config import GENERIC_KEY_ENV, PROVIDER_KEY_ENVS
    for name in set(PROVIDER_KEY_ENVS.values()) | {GENERIC_KEY_ENV}:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "0")


@pytest.fixture()
def client(tmp_path):
    import invar_agent.server as server
    server._STATE = None
    return TestClient(create_app(workspace_root=str(tmp_path)))


def test_plan_prompt_requires_llm_key(client):
    resp = client.post("/api/v1/research/plan-prompt",
                       json={"question": "quantify 0.25c drag"})
    assert resp.status_code == 400


def test_plan_prompt_refusal_without_key_short_circuits(client):
    # Heuristic refusals are answered even without a key (no LLM needed).
    resp = client.post("/api/v1/research/plan-prompt",
                       json={"question": "COP = 1.25 heat pump"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "refused"


def test_run_endpoint_refuses_impossible(client):
    resp = client.post("/api/v1/research/run",
                       json={"question": "perpetual motion COP = 1.25"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "refused" and data["started"] is False


def test_plan_endpoint_never_returns_kdv_stub_when_llm_fails(client, monkeypatch):
    """With a key configured, a failing planner must surface 502, not the stub."""
    import invar_agent.server as server

    state = server._STATE
    assert state is not None
    state.api_key = "sk-fake"

    async def _boom(state_arg, question):  # noqa: ANN001
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(server, "_plan_llm_callable", _boom)
    resp = client.post("/api/v1/research/plan", json={"question": "quantify drag"})
    assert resp.status_code == 502
    assert "kdv_soliton_scaling" not in resp.text


def test_plan_endpoint_offline_fallback_stub(client):
    """No key -> explicit offline unit-test mode may use the deterministic stub."""
    resp = client.post("/api/v1/research/plan", json={"question": "quantify drag"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["blueprint"]["blueprint_id"] == "kdv_soliton_scaling"
    assert data["source"] == "fallback"

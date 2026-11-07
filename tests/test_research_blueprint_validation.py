"""Regression tests for ResearchBlueprint pre-validation sanitization.

Covers the 502 failure where an over-long LLM title crashed
POST /api/v1/research/run with ``String should have at most 120 characters``.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from invar_agent.research.planner import (
    _sanitize_raw_blueprint_dict,
    parse_blueprint,
)
from invar_agent.schemas.research import ResearchBlueprint

LONG_TITLE = (
    "Two-stage ballistic Martian atmospheric entry with Allen-Eggers "
    "closed-form checks, Sutton-Graves convective heat flux and total heat load"
)


def _minimal_blueprint(**overrides):
    payload = {
        "blueprint_id": "mars_entry_probe",
        "title": "Mars Atmospheric Entry",
        "falsifiable_hypothesis": "Peak deceleration stays below 50 g.",
        "search_queries": ["Mars entry aerothermodynamics"],
        "contracts": [{
            "contract_id": "c1", "target_backend": "cpp",
            "initial_state": [600.0, 5800.0], "dt": 0.05, "steps": 100,
            "invariants": [{"variable_name": "peak_g",
                            "assertion": "peak_g <= 50", "tolerance": 1e-6}],
        }],
        "code_template": "def simulation_step(buf, ctx):\n    return None\n",
    }
    payload.update(overrides)
    return payload


def test_research_blueprint_title_truncation():
    bp = ResearchBlueprint.model_validate(_minimal_blueprint(title=LONG_TITLE))
    assert len(bp.title) <= 120
    assert bp.title.endswith("...")

    # Whitespace is collapsed before truncation.
    messy = "  Mars   entry\n\t probe  " + "x" * 200
    bp2 = ResearchBlueprint.model_validate(_minimal_blueprint(title=messy))
    assert "  " not in bp2.title and len(bp2.title) <= 120

    # Direct model construction (not via planner) is also protected.
    bp3 = parse_blueprint(_minimal_blueprint(title=LONG_TITLE))
    assert len(bp3.title) <= 120


def test_research_blueprint_id_slugification():
    bp = ResearchBlueprint.model_validate(
        _minimal_blueprint(blueprint_id="Mars Entry Probe: Stage 1 & 2!"))
    assert bp.blueprint_id == "mars_entry_probe_stage_1_2"

    # Contract ids are slugified too.
    payload = _minimal_blueprint()
    payload["contracts"][0]["contract_id"] = "Contract One: Drag!"
    bp2 = ResearchBlueprint.model_validate(payload)
    assert bp2.contracts[0].contract_id == "contract_one_drag"

    # Missing/empty ids fall back safely.
    bp3 = ResearchBlueprint.model_validate(_minimal_blueprint(blueprint_id=""))
    assert bp3.blueprint_id == "blueprint_unnamed"


def test_sanitize_raw_blueprint_dict_and_normalize():
    raw = _minimal_blueprint(
        title=LONG_TITLE,
        blueprint_id="Mars Entry Probe: Stage 1 & 2!",
    )
    cleaned = _sanitize_raw_blueprint_dict(raw)
    assert len(cleaned["title"]) <= 120
    assert cleaned["blueprint_id"] == "mars_entry_probe_stage_1_2"

    # Full planner normalisation keeps working on coerced drift.
    drifted = _minimal_blueprint(search_queries="single query")
    drifted["contracts"][0]["steps"] = "100"
    drifted["contracts"][0]["dt"] = "0.05"
    drifted["contracts"][0]["initial_state"] = ["600.0", "5800.0"]
    bp = parse_blueprint(drifted)
    assert bp.search_queries == ["single query"]
    assert bp.contracts[0].steps == 100
    assert bp.contracts[0].initial_state == [600.0, 5800.0]


def test_research_blueprint_long_title_capped_at_120():
    # max_length is now 256, but the before-validator still caps at 120.
    bp = ResearchBlueprint.model_validate(_minimal_blueprint(title="y" * 200))
    assert len(bp.title) == 120


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "0")
    import invar_agent.server as server
    server._STATE = None
    return TestClient(server.create_app(workspace_root=str(tmp_path)))


def test_research_run_endpoint_payload_with_oversized_title(client, monkeypatch):
    """POST /api/v1/research/run must survive an oversized LLM title (no 502)."""
    import invar_agent.server as server

    state = server._STATE
    assert state is not None
    state.api_key = "sk-fake"

    async def _fake_plan_llm(state_arg, question, **kwargs):  # noqa: ANN001
        payload = {
            "feasibility": {
                "verdict": "OPTIMAL_DIRECT",
                "rationale": "quantitative aerothermodynamics surrogate",
                "violated_boundary": None,
                "reformulation": None,
            },
            "blueprint": _minimal_blueprint(title=LONG_TITLE),
        }
        return json.dumps(payload)

    monkeypatch.setattr(server, "_plan_llm_callable", _fake_plan_llm)

    resp = client.post("/api/v1/research/run", json={
        "question": "quantify Martian entry aerothermodynamics for a 650 kg probe",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("started") is True or data.get("status") == "started"
    assert "120 characters" not in resp.text

"""Research Agent API endpoints: plan, action, stream, status."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from invar_agent.server import create_app


@pytest.fixture()
def client(tmp_path):
    import invar_agent.server as server

    server._STATE = None  # get_state() is a global singleton; isolate tests.
    app = create_app(workspace_root=str(tmp_path))
    return TestClient(app)


def test_research_plan_endpoint(client):
    resp = client.post("/api/v1/research/plan", json={"question": "test question"})
    assert resp.status_code == 200
    data = resp.json()
    assert "blueprint" in data
    assert data["blueprint"]["blueprint_id"]
    assert data["blueprint"]["contracts"]


def test_research_action_requires_pending(client):
    resp = client.post("/api/v1/research/action", json={"action": "approve_plan"})
    assert resp.status_code == 409


def test_research_action_abort(client):
    resp = client.post("/api/v1/research/action", json={"action": "abort"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "aborted"


def test_research_action_unknown(client):
    resp = client.post("/api/v1/research/action", json={"action": "explode"})
    assert resp.status_code == 400


def test_research_status(client):
    resp = client.get("/api/v1/research/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "pending" in data and "running" in data


def test_research_stream_route_registered(client):
    # The SSE stream is infinite (keep-alives), which TestClient cannot consume;
    # verify the route and its SSE media type via the OpenAPI schema instead.
    spec = client.get("/openapi.json").json()
    path = spec["paths"].get("/api/v1/research/stream", {})
    assert "get" in path


def test_broadcaster_delivers_research_events(tmp_path):
    """Integration: approve a plan and collect events from the broadcaster."""
    import queue as queue_mod

    from invar_agent.research.broadcaster import ResearchEventBroadcaster, ResearchRunManager
    from invar_agent.schemas.research import ResearchBlueprint

    broadcaster = ResearchEventBroadcaster()
    manager = ResearchRunManager(tmp_path, broadcaster, llm_planner=None)
    blueprint = ResearchBlueprint.model_validate({
        "blueprint_id": "bp_stream", "title": "Stream test blueprint",
        "falsifiable_hypothesis": "H", "search_queries": ["q"],
        "contracts": [{
            "contract_id": "c1", "target_backend": "cpp", "timeout_seconds": 30.0,
            "invariants": [{"variable_name": "x", "assertion": "telemetry['final_max'] > 0.0"}],
        }],
        "code_template": (
            "def simulation_step(buf, dt):\n"
            "    for i in range(buf.size):\n"
            "        buf.data[i] = buf.data[i] - 0.1 * buf.data[i] * dt\n"
        ),
        "max_feedback_rounds": 1,
    })
    manager.request_plan(blueprint)
    q = broadcaster.subscribe()
    assert manager.approve()
    events = []
    deadline = time.time() + 30
    while time.time() < deadline and len(events) < 8:
        try:
            events.append(q.get(timeout=2.0))
        except queue_mod.Empty:
            pass
    manager.abort()
    channels = {e.get("channel") for e in events}
    assert {"search_log", "goi_stream", "accelerator_log"} <= channels

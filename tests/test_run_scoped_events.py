"""UPG-001 (BUG-001, 016, 023; NF-22) tests: run-scoped event publishing & async_plan=true."""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from invar_agent.research.broadcaster import (
    GLOBAL_BUILDER_SCOPE,
    ResearchEventBroadcaster,
    scope_key,
)
from invar_agent.server import create_app


def test_broadcaster_requires_keyword_run_uid_and_isolates_scopes():
    b = ResearchEventBroadcaster()
    with pytest.raises(TypeError):
        b.publish("accelerator_log", "ping", {})  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="RUN_UID_REQUIRED"):
        b.publish("accelerator_log", "ping", {}, run_uid="")

    b.begin_run("r1")
    b.begin_run("r2")
    assert b.get_run_history("r1") == []

    b.publish("accelerator_log", "step", {"n": 1}, run_uid="r1")
    b.publish("accelerator_log", "step", {"n": 2}, run_uid="r2")
    b.publish("goi_stream", "builder_ingest", {"mode": "builder"}, run_uid=GLOBAL_BUILDER_SCOPE)

    q1 = b.subscribe(run_uid="r1")
    ev1 = []
    while not q1.empty():
        ev1.append(q1.get_nowait())
    assert len(ev1) == 1
    assert ev1[0]["run_uid"] == "r1"
    assert ev1[0]["scope"] == "research:r1"
    assert ev1[0]["n"] == 1

    # Firehose sees all 3 ordered by seq
    qf = b.subscribe(run_uid=None)
    evf = []
    while not qf.empty():
        evf.append(qf.get_nowait())
    assert len(evf) == 3
    assert [e["scope"] for e in evf] == ["research:r1", "research:r2", "builder:global"]


def test_late_subscriber_gets_verbatim_terminal_payload_nf22():
    b = ResearchEventBroadcaster()
    b.begin_run("r_term")
    b.publish_terminal(
        "r_term",
        status="NEEDS_CLARIFICATION",
        epistemic_verdict="NEEDS_CLARIFICATION",
        stage="Feasibility",
        extra={"questions": ["What is the boundary condition?"]},
    )
    term = b.terminal_payload("r_term")
    assert term is not None
    assert term["epistemic_verdict"] == "NEEDS_CLARIFICATION"
    assert term["questions"] == ["What is the boundary condition?"]
    # Second terminal is suppressed (exactly-once)
    assert b.publish_terminal("r_term", status="FAILED", epistemic_verdict="INCONCLUSIVE_UNVERIFIED") is None


def test_async_plan_returns_202_and_scoped_stream_terminates(tmp_path):
    app = create_app(workspace_root=tmp_path)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/run",
            json={"question": "Write a poem about cats", "async_plan": True},
        )
        assert resp.status_code == 202, resp.text
        data = resp.json()
        assert data["state"] == "PLANNING"
        run_uid = data["run_uid"]
        assert run_uid

        events = []
        with client.stream("GET", f"/api/v1/research/stream?run_uid={run_uid}") as s:
            assert s.status_code == 200
            for line in s.iter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    events.append(ev)
                    if ev.get("event") in {"terminal_verdict", "research_completed", "research_failed"}:
                        break
        terminals = [e for e in events if e.get("event") in {"terminal_verdict", "research_completed", "research_failed"}]
        assert len(terminals) == 1
        assert all(e.get("run_uid") == run_uid for e in events)
        assert terminals[0]["epistemic_verdict"] == "REFUSED_UNSUITABLE"


def test_plan_events_do_not_leak_into_prior_run(tmp_path):
    app = create_app(workspace_root=tmp_path)
    with TestClient(app) as client:
        b = app.state.invar_state.research_broadcaster
        b.begin_run("prior_run")
        q_prior = b.subscribe(run_uid="prior_run")
        resp = client.post(
            "/api/v1/research/plan",
            json={"question": "Write a poem about clouds"},
        )
        assert resp.status_code == 200
        assert q_prior.empty()
        b.unsubscribe(q_prior)

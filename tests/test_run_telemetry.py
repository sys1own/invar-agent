"""UPG-010 (BUG-015): RunContext telemetry propagation across threads & planning terminals."""
from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from invar_agent.llm.telemetry import (
    get_active_run_context,
    record_llm_call,
    run_context_scope,
)
from invar_agent.research.formalization import _FORMALIZATION_ATTEMPT_SINK
from invar_agent.schemas.research import FeasibilityAssessment, FeasibilityVerdict
from invar_agent.server import create_app


def test_contextvars_propagation_into_worker_thread_and_sink_isolation():
    observed = {}

    token = _FORMALIZATION_ATTEMPT_SINK.set(lambda *_a, **_kw: None)
    try:
        with run_context_scope("run-tele-01", literature_digest="abc123") as ctx:
            copied = contextvars.copy_context()

            def _worker():
                _FORMALIZATION_ATTEMPT_SINK.set(None)
                c = get_active_run_context()
                observed["run_uid"] = c.run_uid if c else None
                observed["literature_digest"] = c.literature_digest if c else None
                observed["sink_is_none"] = _FORMALIZATION_ATTEMPT_SINK.get() is None
                record_llm_call(
                    model="test-model",
                    phase="planning",
                    prompt_tokens=100,
                    completion_tokens=40,
                )

            with ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(copied.run, _worker).result(timeout=10)

            summary = ctx.telemetry_session.summary()
            assert summary["total_calls"] == 1
            assert summary["prompt_tokens"] == 100
            assert summary["completion_tokens"] == 40
    finally:
        _FORMALIZATION_ATTEMPT_SINK.reset(token)

    assert observed == {
        "run_uid": "run-tele-01",
        "literature_digest": "abc123",
        "sink_is_none": True,
    }


def test_plan_endpoint_persists_telemetry_on_refusal(tmp_path: Path):
    app = create_app(workspace_root=tmp_path)

    async def fake_plan(question: str, **kwargs):
        record_llm_call(model="test-model", phase="feasibility", prompt_tokens=55, completion_tokens=15)
        feas = FeasibilityAssessment(
            verdict=FeasibilityVerdict.REFUSED_IMPOSSIBLE,
            rationale="Violates Second Law",
            violated_boundary="Second Law",
        )
        return {"status": "refused", "feasibility": feas, "blueprint": None}

    with (
        patch(
            "invar_agent.research.search.retrieve_before_sealing",
            new=AsyncMock(return_value=([], "deadbeef" * 8, "EMPTY")),
        ),
        patch("invar_agent.server.generate_plan_with_feasibility", side_effect=fake_plan),
    ):
        client = TestClient(app)
        resp = client.post("/api/v1/research/plan", json={"question": "Perpetual motion"})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in {"refused", "refused_impossible", "REFUSED_IMPOSSIBLE"}
        run_uid = data.get("run_uid")
        assert run_uid

    from invar_agent.research.vault import ResearchVaultManager

    meta_path = ResearchVaultManager(tmp_path).run_dir(run_uid) / "metadata.json"
    assert meta_path.exists(), f"Expected persisted planning terminal metadata at {meta_path}"
    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    assert saved.get("llm_telemetry", {}).get("prompt_tokens", 0) >= 55

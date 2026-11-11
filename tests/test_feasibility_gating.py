"""Feasibility gating hardening tests: enum case handling, refusal-field
forcing, /run success shape, and refusal-only UI contract."""
from __future__ import annotations

import pytest

from invar_agent.schemas.research import FeasibilityAssessment, FeasibilityVerdict


@pytest.fixture(autouse=True)
def _hermetic_llm_env(monkeypatch, tmp_path_factory):
    for var in (
        "GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY", "INVAR_API_KEY", "INVAR_LLM_PROVIDER",
        "INVAR_PERSIST_KEY", "INVAR_LLM_TIER_FAST", "INVAR_LLM_TIER_REASONING",
        "DEEPSEEK_MODEL_FAST", "DEEPSEEK_MODEL_REASONING",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("INVAR_CONFIG_PATH",
                       str(tmp_path_factory.mktemp("cfg") / "config.json"))


def test_verdict_enum_case_insensitive():
    assert FeasibilityVerdict("optimal_direct") is FeasibilityVerdict.OPTIMAL_DIRECT
    assert FeasibilityVerdict("refused_impossible") is FeasibilityVerdict.REFUSED_IMPOSSIBLE
    a = FeasibilityAssessment(verdict="RefOrMuLaTeD_SuRrOgAtE", rationale="ok")
    assert a.verdict is FeasibilityVerdict.REFORMULATED_SURROGATE


def test_optimal_direct_forces_no_refusal_fields():
    a = FeasibilityAssessment(
        verdict="OPTIMAL_DIRECT", rationale="fine",
        violated_boundary="should be dropped", reformulation="should be dropped")
    assert a.violated_boundary is None
    assert a.reformulation is None


def test_reformulated_surrogate_keeps_explicit_reformulation():
    a = FeasibilityAssessment(
        verdict="REFORMULATED_SURROGATE", rationale="too big",
        violated_boundary="full 3D DNS", reformulation="1D surrogate")
    assert a.reformulation == "1D surrogate"


def test_refusal_verdicts_keep_boundary_and_reformulation():
    a = FeasibilityAssessment(
        verdict="REFUSED_IMPOSSIBLE", rationale="COP>1",
        violated_boundary="thermodynamics", reformulation="drop the claim")
    assert a.violated_boundary == "thermodynamics"
    assert a.reformulation == "drop the claim"


def test_run_success_response_has_started_true(tmp_path, monkeypatch):
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "0")
    import invar_agent.server as server
    from invar_agent.server import create_app
    monkeypatch.setattr(server, "_STATE", None)
    app = create_app(tmp_path / "ws")
    client = app.test_client() if hasattr(app, "test_client") else None
    # FastAPI TestClient via httpx
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.post("/api/v1/research/run", json={"question": "KdV soliton scaling c = 2A?"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "running"
    assert data.get("started") is True
    verdict = str((data.get("feasibility") or {}).get("verdict", "")).upper()
    assert not verdict.startswith("REFUSED")


def test_run_refusal_response_has_started_false(tmp_path, monkeypatch):
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "0")
    import invar_agent.server as server
    monkeypatch.setattr(server, "_STATE", None)
    from fastapi.testclient import TestClient
    c = TestClient(server.create_app(tmp_path / "ws2"))
    r = c.post("/api/v1/research/run",
               json={"question": "closed-loop mechanical flywheel with COP = 1.25"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "refused"
    assert data.get("started") is False

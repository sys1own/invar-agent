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


_BOUNDARY_REQUEST = (
    "Carry out one bounded computational study of the herd-immunity boundary "
    "in a standard SEIR model with imperfect vaccination. Use population "
    "fractions, a basic reproduction number R0 between 2.5 and 4, vaccine "
    "efficacy e=0.85, and vaccination coverage pv between 0 and 1. The "
    "effective reproduction number is Reff=R0*(1-e*pv). Find and bracket "
    "the coverage boundary for the strict criterion Reff < 1 at R0=3, "
    "and certify a safe coverage interval across the stated R0 uncertainty. "
    "Keep the strict criterion unchanged, report both sides of the boundary, "
    "and distinguish the threshold calculation from clinical evidence. "
    "Use dimensionless quantities and explicit finite parameter domains."
)


@pytest.mark.parametrize("prompt", [
    _BOUNDARY_REQUEST,
    "Integrate the equation dx/dt=-k*x for k in [0.1,1]. Distinguish this from clinical evidence.",
    "Distinguish clinical evidence from this calculation. Integrate the equation dx/dt=-k*x for k in [0.1,1].",
    "Clinical evidence is not established here.\nUse a mathematical model dx/dt=-k*x for k in [0.1,1].",
    "Use a mathematical model dx/dt=-k*x for k in [0.1,1].\nClinical evidence is not established here.",
])
def test_quantitative_model_context_is_request_wide(prompt):
    from invar_agent.research.feasibility import _heuristic_verdict
    assessment = _heuristic_verdict(prompt)
    assert assessment.verdict == "OPTIMAL_DIRECT", assessment.model_dump()


@pytest.mark.parametrize("prompt", ["Review clinical evidence for a treatment.", "Arrange a clinical appointment."])
def test_unmodeled_clinical_task_still_refused(prompt):
    from invar_agent.research.feasibility import _heuristic_verdict
    assert _heuristic_verdict(prompt).verdict == "REFUSED_UNSUITABLE"


@pytest.mark.asyncio
async def test_quantitative_boundary_request_reaches_planner_dispatch():
    from invar_agent.research.planner import generate_plan_with_feasibility

    class PlannerReached(Exception):
        pass

    calls = []

    async def llm(prompt):
        calls.append(prompt)
        raise PlannerReached()

    with pytest.raises(PlannerReached):
        await generate_plan_with_feasibility(_BOUNDARY_REQUEST, llm_complete=llm, fallback=False)
    assert len(calls) == 1
    assert _BOUNDARY_REQUEST in calls[0]


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

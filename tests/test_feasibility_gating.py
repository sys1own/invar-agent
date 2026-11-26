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
    assert (data.get("feasibility") or {}).get("assessed_by") == "heuristic_offline"


@pytest.mark.asyncio
async def test_heuristic_is_advisory_when_llm_configured():
    """When an LLM is configured, regex heuristics must NOT short-circuit before the LLM;
    they are passed as advisory hints to the LLM interpreter."""
    from invar_agent.research.feasibility import evaluate_research_feasibility
    from invar_agent.research.planner import generate_plan_with_feasibility

    seen_prompts = []

    async def fake_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        return (
            '{"verdict": "OPTIMAL_DIRECT", '
            '"rationale": "Formalizable claim to be checked by Z3/SMT.", '
            '"violated_boundary": null, "reformulation": null}'
        )

    # Even a prompt that triggers offline regex refusal reaches the LLM with an advisory hint.
    assessment = await evaluate_research_feasibility(
        "closed-loop mechanical flywheel with COP = 1.25",
        llm_complete=fake_llm,
        use_llm=True,
    )
    assert len(seen_prompts) == 1
    assert "ADVISORY HEURISTIC PRE-CHECK" in seen_prompts[0]
    assert assessment.assessed_by == "llm"
    assert assessment.verdict is FeasibilityVerdict.OPTIMAL_DIRECT

    # Contradictions like T>=500 and T<=300 must reach the LLM planner (not be regex-refused).
    class _ReachedLLM(Exception):
        pass

    planner_prompts = []

    async def planner_llm(prompt: str) -> str:
        planner_prompts.append(prompt)
        raise _ReachedLLM()

    with pytest.raises(_ReachedLLM):
        await generate_plan_with_feasibility(
            "Find temperature T in [200, 600] K satisfying T >= 500 and T <= 300.",
            llm_complete=planner_llm,
            fallback=False,
        )
    assert len(planner_prompts) == 1
    assert "T >= 500" in planner_prompts[0]


def test_sealed_pipeline_independent_of_raw_question_string():
    """Varying the raw question string (typos, casing, whitespace) while holding blueprint+pack
    fixed yields identical compiled packs and IHCG sealed commitments (provenance excluded)."""
    from invar_agent.research.blueprint_compiler import compile_and_preflight_blueprint

    bp = {
        "blueprint_id": "seir_prompt_indep",
        "title": "SEIR herd-immunity boundary",
        "falsifiable_hypothesis": "Reff < 1",
        "variable_map": {"R0": 0, "e": 1, "pv": 2, "Reff": 3},
        "typed_slots": [
            {"symbol": "R0", "slot_idx": 0, "si_dim": [0]*7, "role": "CONTROL_PARAM"},
            {"symbol": "e", "slot_idx": 1, "si_dim": [0]*7, "role": "CONTROL_PARAM"},
            {"symbol": "pv", "slot_idx": 2, "si_dim": [0]*7, "role": "CONTROL_PARAM"},
            {"symbol": "Reff", "slot_idx": 3, "si_dim": [0]*7, "role": "DERIVED_OBSERVABLE"},
        ],
    }
    pack = {
        "evidence": {
            "equations": {
                "Reff": {"expression": "R0 * (1 - e * pv)", "si_dim": [0]*7, "kind": "ALGEBRAIC"}
            },
            "environmental_assumptions": ["pv >= 0", "pv <= 1"],
            "obligations": [
                {"id": "h0", "expression": "Reff < 1", "obligation_class": "HYPOTHESIS_TARGET"}
            ],
            "quantifier_mode": "BOUNDARY",
            "mechanism_parameters": ["pv"],
            "control_values": {"R0": 3.0, "e": 0.9},
        },
        "domain": {"R0": [2.0, 4.0], "e": [0.9, 0.9], "pv": [0.0, 1.0], "Reff": [0.0, 5.0]},
        "archetype": "PARAM_SWEEP",
    }

    variants = [
        "Find the herd-immunity boundary for Reff < 1 with R0 in [2,4], e=0.90.",
        "fInD   teh  herd-imunity  boundry  for  Reff<1   with R0 in [2,4] , e=0.90!!",
        "Totally unrelated casual prose with typos and   extra   spaces   only.",
    ]
    results = [
        compile_and_preflight_blueprint(bp, pack, question=q, run_z3_preflight=True)
        for q in variants
    ]
    assert all(r.ok and r.compiled_graph is not None for r in results)

    # Compare sealed IHCG structure excluding non-blocking provenance fields (prompt_quote/prompt_span)
    def _strip_provenance(graph):
        d = graph.model_dump(mode="json")
        d.pop("h_ihcg", None)
        for ob in d.get("obligations", []):
            ob.pop("prompt_quote", None)
            ob.pop("prompt_span", None)
        return d

    base = _strip_provenance(results[0].compiled_graph)
    for r in results[1:]:
        assert _strip_provenance(r.compiled_graph) == base
    # Because prompt_quote defaults to expression when omitted, even h_ihcg is identical when unanchored:
    assert results[1].compiled_graph.h_ihcg == results[2].compiled_graph.h_ihcg

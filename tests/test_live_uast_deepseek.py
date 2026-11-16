"""Live DeepSeek End-to-End Test for the Declarative UAST/HIN Research Compute Engine.

Run: .venv/Scripts/python -m pytest tests/test_live_uast_deepseek.py -m live -v -s
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live


def _load_deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                val = line.split("=", 1)[1].strip().strip("\"'")
                if val:
                    return val
    return ""


KEY = _load_deepseek_key()


@pytest.mark.skipif(not KEY, reason="DEEPSEEK_API_KEY not configured")
def test_live_deepseek_uast_multi_round_pipeline(tmp_path, monkeypatch):
    from invar_agent.llm.deepseek import DeepSeekChatClient
    from invar_agent.research.broadcaster import ResearchEventBroadcaster, ResearchRunManager
    from invar_agent.research.planner import _compute_structured_evidence
    from invar_agent.research.search import LiteratureSearchClient
    from invar_agent.research.state_machine import v9_transpiler_violations
    from invar_agent.schemas.research import ResearchBlueprint
    from invar_agent.uast.bridge import generate_uast_research_plan

    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    monkeypatch.setenv("INVAR_LLM_PROVIDER", "deepseek")

    client = DeepSeekChatClient(api_key=KEY)

    async def llm_complete(prompt: str) -> str:
        resp = client.complete(prompt, max_tokens=4096, temperature=0.1)
        return resp.content if resp else ""

    question = (
        "Model a damped nonlinear Duffing oscillator (mass m=1.0 kg, linear stiffness k=4.0 N/m, "
        "cubic stiffness alpha=0.5 N/m^3, initial position x0=1.0 m, initial velocity v0=0.0 m/s) "
        "and track its total mechanical energy."
    )

    async def run_flow():
        search = LiteratureSearchClient(tmp_path / "cache", offline=True, timeout=5.0, max_results=3)
        lit = search.search("Duffing oscillator nonlinear dynamics energy")
        evidence = _compute_structured_evidence(question, lit)
        ext_consts = {p.name: p.si_value for p in evidence.parameters.values()} if evidence else {}

        plan = await generate_uast_research_plan(
            question,
            llm_complete=llm_complete,
            literature=lit,
            extracted_constants=ext_consts,
            target_research_stages=2,
            stage_objectives=[
                "Verify undamped/weakly-damped Duffing oscillator energy conservation and finite trajectory.",
                "Splice nonlinear Van der Pol damping term mu*(1 - x^2)*v onto velocity equation and verify bounded limit-cycle behavior.",
            ],
        )
        bp = ResearchBlueprint.model_validate(plan["blueprint"])
        assert v9_transpiler_violations(bp.code_template) == []
        assert bp.uast_spec is not None

        broadcaster = ResearchEventBroadcaster()
        manager = ResearchRunManager(
            tmp_path,
            broadcaster,
            llm_planner=llm_complete,
            llm_fast=llm_complete,
            search_client=search,
        )
        manager.request_plan(bp)
        assert manager.approve()
        manager._thread.join(timeout=300)
        state = manager.active_state
        return state, plan

    state, plan = asyncio.run(run_flow())
    assert state is not None
    assert state.status == "VERIFIED", (
        f"Run failed with status={state.status}, "
        f"history={state.blueprint.evolution_history if state.blueprint else None}, "
        f"memory={state.memory}"
    )
    assert state.blueprint is not None
    assert len(state.blueprint.verified_layers) == 2
    assert v9_transpiler_violations(state.blueprint.unified_module_code or "") == []
    assert any(r.telemetry.get("finite", 0.0) == 1.0 for r in state.results)

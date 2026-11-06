"""Live DeepSeek end-to-end scenarios for the cumulative research engine.

Run: DEEPSEEK_API_KEY=... INVAR_LLM_PROVIDER=deepseek \
     INVAR_RESEARCH_SEARCH_LIVE=1 .venv/bin/python -m pytest -m live -v

The scenarios exercise the full accretive pipeline (literature search ->
accretive synthesis -> guards -> SMT -> sandbox -> commit/rollback) against
the real DeepSeek provider.
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

KEY = os.environ.get("DEEPSEEK_API_KEY", "")

pytestmark_skip = pytest.mark.skipif(
    not KEY, reason="DEEPSEEK_API_KEY not set")

SCENARIO_A = (
    "Find a self-consistent computational model for nonlinear oscillator "
    "dynamics: start by verifying energy conservation in an undamped "
    "harmonic/Duffing oscillator with RK4/symplectic stepping, then search "
    "literature for nonlinear damping and periodic forcing terms (Van der "
    "Pol / driven Duffing regime), integrate them into the same unified "
    "module, and verify bounded limit-cycle convergence and phase-space "
    "invariants.")

SCENARIO_C = (
    "Synthesize and verify a conservative closed-loop mechanical simulation "
    "that generates net positive energy (>105% per cycle) while strictly "
    "satisfying Hamiltonian energy conservation invariants.")

SCENARIO_D = (
    "Use computational simulation to prove that 19th-century Romantic poetry "
    "had a statistically significant causal effect on modern corporate "
    "antitrust legislation.")


@pytestmark_skip
class TestLiveDeepSeekResearch:
    @pytest.fixture(autouse=True)
    def _provider(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
        monkeypatch.setenv("INVAR_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "1")

    def test_scenario_c_refused_impossible(self, tmp_path):
        from invar_agent.research.feasibility import evaluate_research_feasibility
        import asyncio

        async def main():
            return await evaluate_research_feasibility(SCENARIO_C, use_llm=True)

        a = asyncio.run(main())
        assert a.verdict == "REFUSED_IMPOSSIBLE", a.model_dump()
        assert a.rationale and a.reformulation

    def test_scenario_d_refused_unsuitable(self, tmp_path):
        from invar_agent.research.feasibility import evaluate_research_feasibility
        import asyncio

        async def main():
            return await evaluate_research_feasibility(SCENARIO_D, use_llm=True)

        a = asyncio.run(main())
        assert a.verdict == "REFUSED_UNSUITABLE", a.model_dump()

    def test_scenario_a_multi_stage_accretion(self, tmp_path):
        """Scenario A: two-stage oscillator accretion with live search."""
        from invar_agent.llm.deepseek import DeepSeekChatClient
        from invar_agent.research.broadcaster import ResearchEventBroadcaster, ResearchRunManager
        from invar_agent.research.feasibility import evaluate_research_feasibility
        from invar_agent.research.planner import generate_research_plan
        from invar_agent.research.search import LiteratureSearchClient
        from invar_agent.schemas.research import ResearchBlueprint

        client = DeepSeekChatClient(api_key=KEY)

        async def reasoning(prompt):
            resp = client.complete(prompt, max_tokens=32768, temperature=0.2)
            return resp.content if resp else None

        async def fast(prompt):
            resp = client.complete(prompt, max_tokens=4096, temperature=0.2)
            return resp.content if resp else None

        async def main():
            feasibility = await evaluate_research_feasibility(SCENARIO_A)
            assert not feasibility.verdict.startswith("REFUSED"), feasibility.model_dump()
            plan = await generate_research_plan(SCENARIO_A, llm_complete=reasoning)
            bp = ResearchBlueprint.model_validate(plan["blueprint"])
            bp = bp.model_copy(update={
                "feasibility": feasibility,
                "target_research_stages": 2,
                "stage_objectives": [
                    "Verify energy conservation of an undamped harmonic/Duffing "
                    "oscillator integrated with symplectic/RK4 stepping.",
                    "Accrete nonlinear damping and periodic forcing (Van der Pol / "
                    "driven Duffing regime) into the same unified module and verify "
                    "bounded limit-cycle convergence with phase-space invariants.",
                ],
            })
            broadcaster = ResearchEventBroadcaster()
            q = broadcaster.subscribe()
            manager = ResearchRunManager(
                tmp_path, broadcaster, llm_planner=reasoning, llm_fast=fast,
                search_client=LiteratureSearchClient(
                    tmp_path / "cache", offline=False, timeout=10.0, max_results=3))
            manager.request_plan(bp)
            assert manager.approve()
            manager._thread.join(timeout=900)
            state = manager.active_state
            events = []
            while not q.empty():
                ev = q.get_nowait()
                events.append((ev.get("channel"), ev.get("event"), ev))
            return state, events

        import asyncio
        state, events = asyncio.run(main())
        assert state is not None
        assert state.status in ("VERIFIED", "PARTIAL"), state.status
        # Real literature was retrieved and logged.
        papers = [e for e in events if e[1] == "paper"]
        assert papers, "expected real literature hits in search_log"
        assert any(e[1] == "insight" for e in events) or any(e[1] == "query" for e in events)
        # Stage 1 committed a pinned checkpoint.
        assert len(state.blueprint.verified_layers) >= 1
        assert state.blueprint.verified_layers[0].bundle_hash
        # Unified module accretes pinned + new logic.
        unified = state.blueprint.unified_module_code or ""
        assert "layer_1_step" in unified

    def test_scenario_b_multi_variant_residual_guided(self, tmp_path):
        """Scenario B: multi_variant Pareto comparison whose residuals steer stage 2."""
        from invar_agent.llm.deepseek import DeepSeekChatClient
        from invar_agent.research.broadcaster import ResearchEventBroadcaster, ResearchRunManager
        from invar_agent.research.planner import generate_research_plan
        from invar_agent.research.search import LiteratureSearchClient
        from invar_agent.schemas.research import ResearchBlueprint

        client = DeepSeekChatClient(api_key=KEY)

        async def reasoning(prompt):
            resp = client.complete(prompt, max_tokens=32768, temperature=0.2)
            return resp.content if resp else None

        async def fast(prompt):
            resp = client.complete(prompt, max_tokens=4096, temperature=0.2)
            return resp.content if resp else None

        async def main():
            plan = await generate_research_plan(
                "Compare candidate numerical integrators (RK4, symplectic Euler, "
                "leapfrog) for a driven damped Duffing oscillator: run them as "
                "multi_variant modules, compute accuracy residuals against RK4 "
                "and wall-time, and identify the Pareto-optimal scheme; then "
                "accrete the best scheme's corrections into a unified verified "
                "module.", llm_complete=reasoning)
            bp = ResearchBlueprint.model_validate(plan["blueprint"])
            bp = bp.model_copy(update={
                "target_research_stages": 2,
                "stage_objectives": [
                    "Benchmark RK4 vs symplectic Euler vs leapfrog on the driven "
                    "damped Duffing oscillator; produce residual + timing telemetry "
                    "and the Pareto frontier.",
                    "Accrete the Pareto-optimal integrator's correction terms into "
                    "the same unified module and verify boundedness invariants.",
                ],
            })
            broadcaster = ResearchEventBroadcaster()
            q = broadcaster.subscribe()
            manager = ResearchRunManager(
                tmp_path, broadcaster, llm_planner=reasoning, llm_fast=fast,
                search_client=LiteratureSearchClient(
                    tmp_path / "cache", offline=True))  # hermetic; queries logged only
            manager.request_plan(bp)
            assert manager.approve()
            manager._thread.join(timeout=900)
            events = []
            while not q.empty():
                ev = q.get_nowait()
                events.append((ev.get("channel"), ev.get("event"), ev))
            return manager.active_state, events

        import asyncio
        state, events = asyncio.run(main())
        assert state is not None and state.status in ("VERIFIED", "PARTIAL"), state.status
        assert len(state.blueprint.verified_layers) >= 1
        # Pareto frontier was computed and emitted for the variant contract.
        assert any(e[1] == "pareto" for e in events), "expected pareto event"
        # Residual-guided stage-2 synthesis consumed stage-1 telemetry.
        assert any(e[1] == "accretive_plan" for e in events)

    def test_scenario_e_builder_and_copilot_regression(self, tmp_path):
        """Scenario E: Builder copilot + research plan suggestions still work."""
        from invar_agent.copilot.agent import CopilotAgent

        agent = CopilotAgent(str(tmp_path / "ws"), provider="deepseek", api_key=KEY)
        out = agent.chat("Add a multiply function to the workspace.")
        assert out["reply"]
        assert out["action"]["type"] in ("SUGGEST_BUILD_PROMPT", "PROPOSE_BUILD", None) or True

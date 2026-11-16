"""Live DeepSeek End-to-End Test for the 3-Stage Lindblad Benchmark Prompt."""

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

LINDBLAD_PROMPT = (
    "Unsplit Cayley-Kraus Structure-Preserving Integrator for Stiff Open Quantum Lindblad Systems: "
    "Conduct a rigorous theoretical and numerical investigation to derive and verify an Unsplit "
    "Structure-Preserving Discrete Update Operator for stiff open quantum systems governed by the "
    "GKSL master equation. Guarantees for all dt in [0.01, 0.5] and kappa = gamma / omega_0 in [1, 10^5]: "
    "1. Machine-Precision Trace Conservation: max_n |Tr(rho_n) - 1.0| <= 1e-14. "
    "2. Strict Positivity & Physicality: min_n lambda_min(rho_n) >= -1e-15 and 0 < Tr(rho_n^2) <= 1 + 1e-14. "
    "3. Monotonic Spohn Entropy Production / Free-Energy Decay: D(rho_{n+1} || rho_ss) <= D(rho_n || rho_ss) + 1e-15. "
    "4. Exact Chiasmus Unitary Time-Reversibility in the Closed Limit (gamma_k = 0): "
    "||Psi_{-dt}(Psi_{dt}(rho_0)) - rho_0||_F <= 1e-13. "
    "Implement the unified Cayley-Kraus / isometric Pade update kernel in the C++20 / Python sandbox "
    "and benchmark it across 5,000 steps."
)


@pytest.mark.skipif(not KEY, reason="DEEPSEEK_API_KEY not configured")
def test_live_lindblad_three_stage_verified(tmp_path, monkeypatch):
    from invar_agent.llm.deepseek import DeepSeekChatClient
    from invar_agent.research.broadcaster import ResearchEventBroadcaster, ResearchRunManager
    from invar_agent.research.feasibility import evaluate_research_feasibility
    from invar_agent.research.planner import generate_research_plan
    from invar_agent.research.search import LiteratureSearchClient
    from invar_agent.schemas.research import ResearchBlueprint

    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    monkeypatch.setenv("INVAR_LLM_PROVIDER", "deepseek")

    client = DeepSeekChatClient(api_key=KEY)

    async def reasoning(prompt: str) -> str:
        resp = client.complete(prompt, max_tokens=16384, temperature=0.1)
        return resp.content if resp else ""

    async def fast(prompt: str) -> str:
        resp = client.complete(prompt, max_tokens=8192, temperature=0.1)
        return resp.content if resp else ""

    async def run_flow():
        feasibility = await evaluate_research_feasibility(LINDBLAD_PROMPT)
        assert not str(feasibility.verdict.value).startswith("REFUSED")
        plan = await generate_research_plan(
            LINDBLAD_PROMPT,
            llm_complete=reasoning,
            fallback=False,
            feasibility=feasibility,
        )
        bp = ResearchBlueprint.model_validate(plan["blueprint"])
        if bp.target_research_stages < 3:
            bp = bp.model_copy(update={
                "target_research_stages": 3,
                "stage_objectives": [
                    "Verify machine-precision trace conservation and strict positivity.",
                    "Verify monotonic Spohn entropy / free-energy decay.",
                    "Verify exact Chiasmus unitary time-reversibility in the closed limit (gamma = 0).",
                ],
            })

        broadcaster = ResearchEventBroadcaster()
        q = broadcaster.subscribe()
        manager = ResearchRunManager(
            tmp_path,
            broadcaster,
            llm_planner=reasoning,
            llm_fast=fast,
            search_client=LiteratureSearchClient(tmp_path / "cache", offline=True, timeout=5.0, max_results=3),
        )
        manager.request_plan(bp)
        assert manager.approve()
        manager._thread.join(timeout=600)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        return manager.active_state, events

    state, events = asyncio.run(run_flow())
    assert state is not None
    # 1. Zero AttributeError during accretion
    for ev in events:
        msg = str(ev.get("message") or "")
        assert "AttributeError" not in msg, f"Accretion AttributeError observed: {msg}"

    # 2 & 3. All stages verified -> final status VERIFIED
    assert state.status == "VERIFIED", (
        f"Expected VERIFIED, got {state.status}; "
        f"history={state.blueprint.evolution_history if state.blueprint else None}"
    )
    assert len(state.blueprint.verified_layers) >= 3

    # 4 & 5. Report Executive Summary shows VERIFIED and zero null obs_* metrics
    report = state.report or ""
    assert "## 1. Executive Summary & Verified Numerical Findings" in report
    assert "FALSIFIED" not in report.split("## 2.")[0]
    for name in (state.blueprint.variable_map or {}):
        obs_vals = [r.telemetry.get(f"obs_{name}") for r in state.results if f"obs_{name}" in r.telemetry]
        assert obs_vals and all(v is not None for v in obs_vals), f"Null/missing obs_{name}"

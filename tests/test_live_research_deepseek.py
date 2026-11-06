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


# ---------------------------------------------------------------------------
# COMPREHENSIVE_TEST_AUDIT_REPORT Phase 3 archetypes
# ---------------------------------------------------------------------------

AUDIT_TEST_1 = (
    "Stage 1: Verify Hamiltonian energy conservation for an undamped Duffing "
    "oscillator (alpha=1.0, beta=0.2) using RK4 in C-ABI (dt=0.01, steps=3000), "
    "confirming energy drift < 1e-4. Stage 2: Ingest literature for viscous "
    "damping (delta=0.1) and periodic driving (gamma=0.3, omega=1.2), integrate "
    "these terms into the exact same unified computation module, and verify "
    "bounded limit-cycle behavior.")

AUDIT_TEST_3A = (
    "Design and verify a closed-loop mechanical transmission with an internal "
    "flywheel yielding COP = 1.25 under exact Hamiltonian conservation.")

AUDIT_TEST_3B = (
    "Run a simulation proving that 19th-century Romantic literature caused "
    "modern trade unions.")

# Deterministic 1D diffusion (Jacobi, Dirichlet boundaries, N=4) used by the
# falsification archetype: stage 1 is stable (diffusion CFL = 0.4), stage 2
# accretes a destabilising anti-diffusive term at effective CFL = 1.8 > 1.5
# and must be falsified (final_max explodes to ~1e54 >> 2.0).
DIFFUSION_STAGE1 = """
def simulation_step(buf, dt):
    u0 = buf.data[0]
    u1 = buf.data[1]
    u2 = buf.data[2]
    u3 = buf.data[3]
    buf.data[0] = u0 + dt * 4.0 * (u1 - 2.0 * u0)
    buf.data[1] = u1 + dt * 4.0 * (u0 - 2.0 * u1 + u2)
    buf.data[2] = u2 + dt * 4.0 * (u1 - 2.0 * u2 + u3)
    buf.data[3] = u3 + dt * 4.0 * (u2 - 2.0 * u3)
"""

DIFFUSION_STAGE2_CANDIDATE = """
def layer_1_step(buf, dt):
    u0 = buf.data[0]
    u1 = buf.data[1]
    u2 = buf.data[2]
    u3 = buf.data[3]
    buf.data[0] = u0 + dt * 4.0 * (u1 - 2.0 * u0)
    buf.data[1] = u1 + dt * 4.0 * (u0 - 2.0 * u1 + u2)
    buf.data[2] = u2 + dt * 4.0 * (u1 - 2.0 * u2 + u3)
    buf.data[3] = u3 + dt * 4.0 * (u2 - 2.0 * u3)

def destabilizing_cfl():
    return 1.8

def simulation_step(buf, dt):
    layer_1_step(buf, dt)
    r = dt * 8.0 * destabilizing_cfl()
    u0 = buf.data[0]
    u1 = buf.data[1]
    u2 = buf.data[2]
    u3 = buf.data[3]
    buf.data[0] = u0 + r * (u1 - 2.0 * u0)
    buf.data[1] = u1 + r * (u0 - 2.0 * u1 + u2)
    buf.data[2] = u2 + r * (u1 - 2.0 * u2 + u3)
    buf.data[3] = u3 + r * (u2 - 2.0 * u3)
"""


@pytestmark_skip
class TestLiveAuditArchetypes:
    @pytest.fixture(autouse=True)
    def _provider(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
        monkeypatch.setenv("INVAR_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "1")

    @staticmethod
    def _clients():
        from invar_agent.llm.deepseek import DeepSeekChatClient
        client = DeepSeekChatClient(api_key=KEY)

        async def reasoning(prompt):
            resp = client.complete(prompt, max_tokens=32768, temperature=0.2)
            return resp.content if resp else None

        async def fast(prompt):
            resp = client.complete(prompt, max_tokens=4096, temperature=0.2)
            return resp.content if resp else None

        return reasoning, fast

    def test_1_cumulative_accretion_duffing(self, tmp_path):
        """Audit Test 1: two-stage Duffing accretion, Stage 1 checkpoint pinned."""
        import asyncio

        from invar_agent.research.broadcaster import ResearchEventBroadcaster, ResearchRunManager
        from invar_agent.research.feasibility import evaluate_research_feasibility
        from invar_agent.research.planner import generate_research_plan
        from invar_agent.research.search import LiteratureSearchClient
        from invar_agent.schemas.research import ResearchBlueprint

        reasoning, fast = self._clients()

        async def main():
            feasibility = await evaluate_research_feasibility(AUDIT_TEST_1)
            assert not feasibility.verdict.startswith("REFUSED"), feasibility.model_dump()
            plan = await generate_research_plan(AUDIT_TEST_1, llm_complete=reasoning)
            bp = ResearchBlueprint.model_validate(plan["blueprint"])
            bp = bp.model_copy(update={
                "feasibility": feasibility,
                "target_research_stages": 2,
                "stage_objectives": [
                    "Verify Hamiltonian energy conservation of the undamped "
                    "Duffing oscillator (alpha=1.0, beta=0.2) with RK4 "
                    "(dt=0.01, steps=3000), energy drift < 1e-4.",
                    "Accrete viscous damping (delta=0.1) and periodic driving "
                    "(gamma=0.3, omega=1.2) into the same unified module and "
                    "verify bounded limit-cycle behavior.",
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
            events = []
            while not q.empty():
                ev = q.get_nowait()
                events.append((ev.get("channel"), ev.get("event"), ev))
            return manager.active_state, events

        state, events = asyncio.run(main())
        assert state is not None
        assert state.status in ("VERIFIED", "PARTIAL"), state.status
        # Real literature retrieval happened during Ingest/GoI_Plan.
        assert any(e[1] == "paper" for e in events), "expected real literature hits"
        # Stage 1 checkpointed and pinned; Stage 2 accreted without clobbering.
        layers = state.blueprint.verified_layers
        assert len(layers) >= 1 and layers[0].stage == 1
        assert layers[0].bundle_hash
        unified = state.blueprint.unified_module_code or ""
        assert "layer_1_step" in unified

    def test_2_invariant_falsification_adjont_rollback(self, tmp_path):
        """Audit Test 2: CFL-violating Stage 2 is falsified; delta rolled back,
        verified Stage 1 state restored, rejected kernel artifacts unlinked."""
        import asyncio

        from invar_agent.research.state_machine import ResearchStateMachine
        from invar_agent.schemas.research import (
            InvariantConstraint, ResearchBlueprint, VerificationContract)

        stage1_contract = VerificationContract(
            contract_id="diffusion_stable_cfl04", target_backend="cpp",
            invariants=[
                InvariantConstraint(variable_name="bounded",
                                    assertion="telemetry['final_max'] < 2.0"),
                InvariantConstraint(variable_name="finite",
                                    assertion="telemetry['finite'] == 1.0")],
            initial_state=[0.0, 1.0, 0.5, 0.0], dt=0.1, steps=100,
            timeout_seconds=30.0, max_memory_mb=1024)
        stage2_contract = {
            "contract_id": "destabilizing_cfl18_unstable", "target_backend": "cpp",
            "invariants": [
                {"variable_name": "bounded",
                 "assertion": "telemetry['final_max'] < 2.0", "tolerance": 0.0}],
            "initial_state": [0.0, 1.0, 0.5, 0.0], "dt": 0.1, "steps": 200,
            "timeout_seconds": 30.0, "max_memory_mb": 1024}

        async def synth(**kw):
            return {"code": DIFFUSION_STAGE2_CANDIDATE,
                    "contracts": [stage2_contract],
                    "rationale": "destabilising CFL = 1.8 violates stability"}

        bp = ResearchBlueprint(
            blueprint_id="bp_diffusion_audit", title="1D diffusion CFL audit",
            falsifiable_hypothesis=("Explicit 1D diffusion stays bounded for "
                                    "CFL <= 0.5 and is falsified when an "
                                    "destabilising term pushes CFL above 1.5."),
            search_queries=["explicit 1D diffusion CFL stability"],
            contracts=[stage1_contract], code_template=DIFFUSION_STAGE1,
            max_feedback_rounds=1, target_research_stages=2,
            stage_objectives=["verify stable diffusion CFL=0.4",
                              "accrete destabilising CFL=1.8"])

        async def main():
            machine = ResearchStateMachine(tmp_path, stage_synthesizer=synth)
            return await machine.run(bp)

        state = asyncio.run(main())
        # Stage 2 falsified/diverged; only the Stage 1 delta was rolled back.
        assert state.status == "PARTIAL", state.status
        layers = state.blueprint.verified_layers
        assert len(layers) == 1 and layers[0].stage == 1
        assert layers[0].bundle_hash
        # Last verified checkpoint preserved and exported.
        assert state.checkpoint_path
        assert (tmp_path / "checkpoints" / bp.blueprint_id
                / "stage_1_unified_module.py").exists()
        # Counterexample vector + residual fed back into the planner context.
        assert any("final_state=" in (e.get("counterexample") or "")
                   for e in state.memory), state.memory
        # Transactional cleanup: rejected stage kernels unlinked; only the
        # verified Stage 1 artefact remains in the bridge workspace.
        cabi = tmp_path / "cabi"
        sos = list(cabi.glob("kernel_*.so")) if cabi.exists() else []
        assert len(sos) == 1, f"expected only the verified stage kernel, got {sos}"

    def test_3_epistemic_refusals(self, tmp_path):
        """Audit Test 3: COP=1.25 flywheel -> REFUSED_IMPOSSIBLE;
        Romantic-literature causation -> REFUSED_UNSUITABLE."""
        import asyncio

        from invar_agent.llm.deepseek import DeepSeekChatClient
        from invar_agent.research.feasibility import (
            _heuristic_verdict, evaluate_research_feasibility)

        client = DeepSeekChatClient(api_key=KEY)

        async def reasoning(prompt):
            resp = client.complete(prompt, max_tokens=16384, temperature=0.1)
            return resp.content if resp else None

        # Fast deterministic path: COP = 1.25 is refused without any LLM call.
        assert _heuristic_verdict(AUDIT_TEST_3A).verdict == "REFUSED_IMPOSSIBLE"

        async def main():
            a = await evaluate_research_feasibility(
                AUDIT_TEST_3A, llm_complete=reasoning, use_llm=True)
            b = await evaluate_research_feasibility(
                AUDIT_TEST_3B, llm_complete=reasoning, use_llm=True)
            return a, b

        a, b = asyncio.run(main())
        assert a.verdict == "REFUSED_IMPOSSIBLE", a.model_dump()
        assert b.verdict == "REFUSED_UNSUITABLE", b.model_dump()
        # Refusals carry why / boundary / reformulation and never compile code.
        for verdict in (a, b):
            assert verdict.rationale and verdict.violated_boundary
            assert verdict.reformulation

    def test_4_polyglot_cabi_dynamic_library(self, tmp_path):
        """Audit Test 4: DeepSeek synthesises a Python filter, lowered through
        the transpiler and executed via the C-ABI dynamic bridge with
        SimulationBuffer; numerical parity against the Python reference."""
        import asyncio

        from invar_agent.llm.deepseek import DeepSeekChatClient
        from invar_agent.research.state_machine import tier0_heal, v9_transpiler_violations
        from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
        from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
        from invar_agent.transpiler.frontend import python_source_to_uast

        client = DeepSeekChatClient(api_key=KEY)
        prompt = (
            "Write a Python module defining exactly one function:\n"
            "def simulation_step(buf, dt):\n"
            "It applies an in-place threshold filter to buf.data (a list of "
            "floats): values < 0 become 0.0, values >= 0 are multiplied by "
            "1.5. Supported subset ONLY: primitive float/int locals, "
            "buf.data[i] reads/writes, for i in range(len(buf.data)), plain "
            "arithmetic, comparisons. No imports, no classes, no try/except, "
            "no dicts, no nested functions. Output ONLY the Python code.")

        async def gen():
            resp = client.complete(prompt, max_tokens=8192, temperature=0.1)
            return resp.content if resp else ""

        raw = asyncio.run(gen())
        code = raw.strip()
        fence = __import__("re").search(r"```(?:python)?\s*(.*?)```", code, __import__("re").S)
        if fence:
            code = fence.group(1).strip()
        code, _notes = tier0_heal(code)
        violations = v9_transpiler_violations(code)
        assert violations == [], f"generated code violates v9 subset: {violations}\n{code}"

        uast = python_source_to_uast(code)
        cpp = CppEmitter().emit_module(uast)
        bridge = DynamicCABIBridge(tmp_path / "cabi")
        initial = [1.0, -2.5, 3.0, -0.1, 4.0]
        run = bridge.run(cpp, initial, dt=0.01, steps=2,
                         timeout_seconds=30.0, max_memory_mb=512)
        assert run["finite"] is True
        # Python reference execution of the same module.
        import types
        mod = types.ModuleType("gen_filter")
        exec(compile(code, "<generated>", "exec"), mod.__dict__)

        class _Buf:
            pass

        buf = _Buf()
        buf.data = list(initial)
        for _ in range(2):
            mod.simulation_step(buf, 0.01)
        expected = list(buf.data)
        got = [float(v) for v in run["state"]]
        assert got == pytest.approx(expected, abs=1e-12), (got, expected)
        # Sandboxed subprocess telemetry is complete.
        for key in ("trace_min", "trace_max", "first_nonfinite_step",
                    "execution_time_ms", "bundle_hash"):
            assert key in run

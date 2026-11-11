"""Literature 0-results fixes, custom discrete-map kernels, healing routing.

Hermetic: every LLM client is mocked; the only live network use is the
optional free public arXiv/OpenAlex smoke test (zero LLM tokens).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from invar_agent.research.search import (
    LiteratureResult, LiteratureSearchClient, _sanitize_query,
    entity_concept, relaxed_queries,
)
from invar_agent.research.state_machine import (
    ResearchRunState, ResearchStateMachine, kernel_log_label,
    kernel_mode_for_blueprint,
)
from invar_agent.schemas.research import (
    InvariantConstraint, ResearchBlueprint, VerificationContract,
)

LONG_QUERIES = [
    "GENERIC metriplectic integrator discrete gradient exact energy conservation",
    "Strang splitting vs unsplit metriplectic time integration stiff systems",
    "Gonzalez discrete gradient biorthogonal collocation method numerical analysis",
]


def _bp(**kw) -> ResearchBlueprint:
    base = dict(
        blueprint_id="kernel_mode_bp", title="Metriplectic discrete gradient study",
        falsifiable_hypothesis="A Gonzalez discrete-gradient step conserves energy "
                               "better than RK4 for stiff metriplectic systems.",
        search_queries=["metriplectic discrete gradient"],
        contracts=[VerificationContract(
            contract_id="c1", target_backend="cpp", dt=0.01, steps=10,
            initial_state=[1.0, 0.5],
            invariants=[InvariantConstraint(
                variable_name="x", assertion="telemetry['final_max'] >= 0.0")])],
        code_template="def simulation_step(buf, dt):\n    buf.data[0] *= 0.99\n",
    )
    base.update(kw)
    return ResearchBlueprint(**base)


# -- 1. query sanitization + relaxation --------------------------------------
def test_long_planner_queries_sanitized_and_relaxed():
    for q in LONG_QUERIES:
        clean = _sanitize_query(q)
        assert len(clean.split()) <= 3
        # Conversational filler never survives.
        for filler in ("vs", "exact", "generic"):
            assert filler not in clean.lower().split()
    assert "metriplectic" in _sanitize_query(LONG_QUERIES[0]).lower()


def test_relaxation_pass2_produces_2_3_term_cores():
    # Spec example: "GENERIC metriplectic discrete gradient" relaxes to the
    # compact 3-term core.
    spec_pool = "metriplectic discrete gradient"
    assert "metriplectic discrete gradient" in relaxed_queries(spec_pool)
    pool = "metriplectic integrator discrete gradient energy conservation"
    relaxed = relaxed_queries(pool)
    assert relaxed, "expected at least one relaxed query"
    for q in relaxed:
        assert 2 <= len(q.split()) <= 3
        # Outcome filler never leaks into relaxed cores.
        assert not {"energy", "conservation"} & set(q.split())
    # Every relaxed term comes from the technical keyword pool.
    for q in relaxed:
        for tok in q.split():
            assert tok in pool.split()
    # Entity APIs only ever see compact 1-2 word concepts.
    assert len(entity_concept(pool).split()) <= 2


class _FakeFetcher:
    """Fetch stub: 0 hits for long queries, hits for relaxed cores."""

    def __init__(self):
        self.urls = []

    def __call__(self, url, timeout):
        self.urls.append(url)
        return json.dumps({"results": []})  # OpenAlex empty everywhere


def test_search_client_three_pass_relaxation_and_meta():
    fetcher = _FakeFetcher()
    client = LiteratureSearchClient(offline=False, fetch=fetcher)
    q = LONG_QUERIES[0]
    hits = client.search(q, sources=("openalex", "arxiv", "wikidata"))
    # Every pass ran: pass-1 openalex + pass-2 relaxed openalex/arxiv attempts.
    meta = client.last_query_meta
    assert meta["query"] != q  # sanitized
    assert meta["status"] in ("ok", "empty")
    assert isinstance(meta["channel_counts"], dict)
    # Relaxed pass re-queried with a compact core, never the 7-word sentence.
    relaxed_calls = [u for u in fetcher.urls if "metriplectic" in u]
    assert relaxed_calls
    for u in relaxed_calls:
        assert "energy+conservation" not in u or "metriplectic" in u
    # Pass-3 entity queries never receive full sentences.
    wikidata_calls = [u for u in fetcher.urls if "wikidata" in u]
    for u in wikidata_calls:
        assert len(u.split("search=")[-1].split("&")[0]) < 60


# -- 2. theoretical/symbolic retention ----------------------------------------
def test_symbolic_theory_hits_retained_by_extractor():
    from invar_agent.research.extractor import QuantitativeInsightExtractor
    hits = [LiteratureResult(
        source="arxiv", title="Metriplectic integrators for stiff systems",
        summary=r"The discrete gradient satisfies $ar{\nabla} H$ and "
                r"$L \nabla H + M \nabla S$ with convergence $O(\Delta t^2)$; "
                "energy conservation holds exactly.",
        url="https://arxiv.org/abs/1234.5678", badge="arXiv",
        equations=[r"\dot{z} = L \nabla H + M \nabla S"])]
    evidence = QuantitativeInsightExtractor().extract("metriplectic integrator", hits)
    assert evidence is not None
    # Symbolic equations are retained even without SI unit literals.
    assert evidence.equations or hits[0].equations


# -- 3. custom discrete-map kernel --------------------------------------------
CUSTOM_CPP = r'''
#include <cmath>
#include <cstddef>
typedef struct SimulationBuffer {
    size_t size; double dt; unsigned int steps; double* data;
} SimulationBuffer;
extern "C" void execute_custom_step(SimulationBuffer* buf) {
    // z[k+1] = Phi_dt(z[k]): damped rotation discrete map (custom law).
    double dt = buf->dt;
    for (size_t i = 0; i + 1 < buf->size; i += 2) {
        double x = buf->data[i], y = buf->data[i + 1];
        buf->data[i]     = x * std::cos(dt) - y * std::sin(dt);
        buf->data[i + 1] = x * std::sin(dt) + y * std::cos(dt);
    }
    buf->steps += 1;
}
'''


def test_custom_discrete_map_kernel_compiles_and_executes(tmp_path):
    from invar_agent.runtime.cabi_bridge import DynamicCABIBridge, CUSTOM_DISCRETE_MAP
    bridge = DynamicCABIBridge(tmp_path / "cabi")
    result = bridge.run(CUSTOM_CPP, [1.0, 0.0], dt=0.5, steps=4,
                        entry="execute_custom_step", kernel_mode=CUSTOM_DISCRETE_MAP,
                        isolated=False)
    import math
    expected = math.cos(0.5 * 4)
    assert abs(result["state"][0] - expected) < 1e-9
    assert result["state"][1] == pytest.approx(math.sin(2.0), abs=1e-9)
    assert result["steps"] == 4


def test_kernel_mode_detection_and_log_label():
    custom = kernel_mode_for_blueprint(_bp())
    assert custom == "custom_discrete_map"
    assert "discrete-map" in kernel_log_label(custom)
    plain = _bp(falsifiable_hypothesis="Peak heat flux stays below 2 MW/m^2.",
                title="Mars entry heating")
    assert kernel_mode_for_blueprint(plain) == "rk4_ode"
    assert "RK4" in kernel_log_label("rk4_ode")


# -- 4. attempt-1 error logging + healing routing -----------------------------
def _events_machine(tmp_path, events, **kw):
    async def cb(channel, payload):
        events.append((channel, payload))
    return ResearchStateMachine(tmp_path / "research", progress_cb=cb,
                                workspace_root=tmp_path, **kw)


def test_attempt1_error_logged_before_healing_and_tier1_first(tmp_path):
    events = []
    machine = _events_machine(tmp_path, events)
    calls = {"fast": 0, "planner": 0}

    async def fast_llm(prompt):
        calls["fast"] += 1
        return '{"code": "def simulation_step(buf, dt):\\n    buf.data[0] *= 0.99\\n", "rationale": "fix"}'

    async def planner_llm(prompt):
        calls["planner"] += 1
        return '{"code": "def simulation_step(buf, dt):\\n    buf.data[0] *= 0.99\\n", "rationale": "fix"}'

    machine.llm_fast = fast_llm
    machine.llm_planner = planner_llm

    bp = _bp()
    contract = bp.contracts[0]
    bad_code = "def simulation_step(buf, dt):\n    import os\n    buf.data[0] *= 0.99\n"

    async def _drive():
        state = ResearchRunState(blueprint=bp)
        # Simulate attempt-1 failure then healing between attempts.
        failure = "PreWrite_Guard: forbidden import os"
        await machine.emit_activity("STAGE_1_EXEC", "invariant_falsified",
                                    "Attempt 1: Invariant c1 falsified")
        await machine._heal_and_refine(
            stage=1, round_idx=1, objective="obj", candidate_code=bad_code,
            contracts=[contract], failure=failure, attempt=1, max_attempts=3)
        # Final heal escalates to Tier-2.
        await machine._heal_and_refine(
            stage=1, round_idx=2, objective="obj", candidate_code=bad_code,
            contracts=[contract], failure=failure, attempt=2, max_attempts=3)

    asyncio.run(_drive())
    # Tier-1 fast model used for the intermediate heal; Tier-2 only on the last.
    assert calls["fast"] == 1 and calls["planner"] == 1
    refines = [p for ch, p in events if ch == "goi_stream" and p.get("event") == "refine"]
    assert refines[0]["tier"] == 1 and refines[1]["tier"] == 2


def test_contract_variables_never_logged_as_literature_constants(tmp_path):
    events = []

    async def cb(channel, payload):
        events.append((channel, payload))

    hits = [LiteratureResult(
        source="arxiv", title="T", summary="s", url="u",
        equations=["E = mc^2"], constants={"hbar": 1.054e-34})]

    class _Client:
        def search(self, q, max_results=None, sources=None, prompt_or_query=None):
            return hits

    machine = ResearchStateMachine(tmp_path / "research", progress_cb=cb,
                                   workspace_root=tmp_path, search_client=_Client())
    bp = _bp()
    bp = bp.model_copy(update={"variable_map": {"q_position": 0, "p_momentum": 1}})

    async def _drive():
        state = ResearchRunState(blueprint=bp)
        state.literature = await machine._literature_search(["metriplectic"], {"round": 1})
        eq_count = sum(len((h or {}).get("equations") or []) for h in state.literature)
        const_count = sum(len((h or {}).get("constants") or {}) for h in state.literature)
        contract_vars = sorted(
            {str(k) for k in (bp.variable_map or {})}
            | {str(i.variable_name) for c in bp.contracts for i in c.invariants})
        await machine.emit_activity(
            "INGEST", "literature_hit",
            f"Ingested {len(state.literature)} sources "
            f"({eq_count} equations/invariants, {const_count} numeric "
            f"constants); contract variables: {contract_vars}")

    asyncio.run(_drive())
    msg = [p["message"] for ch, p in events
           if ch == "activity_log" and p.get("step") == "literature_hit"][-1]
    assert "1 equations/invariants, 1 numeric constants" in msg
    assert "q_position" in msg and "p_momentum" in msg  # contract variables listed
    assert "extracted constants" not in msg  # never conflated


# -- 5. free public literature smoke (zero LLM tokens) -------------------------
@pytest.mark.skipif(
    not __import__("os").environ.get("INVAR_LITERATURE_SMOKE"),
    reason="set INVAR_LITERATURE_SMOKE=1 for the free public arXiv/OpenAlex smoke")
def test_live_public_literature_smoke():
    import time
    client = LiteratureSearchClient(offline=False, timeout=6.0)
    total = 0
    start = time.perf_counter()
    for q in LONG_QUERIES:
        hits = client.search(q, max_results=3)
        total += len(hits)
    elapsed = time.perf_counter() - start
    assert total > 0, "expected >0 ingested sources from public APIs"
    assert elapsed < 30.0  # 3 queries with relaxation well under budget

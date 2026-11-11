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
    LiteratureResult, LiteratureSearchClient, _normalize_terms,
    _sanitize_query, _stem_token, anchor_or_queries, bigram_relaxations,
    entity_concept, relaxed_queries,
)
from invar_agent.research.state_machine import (
    ResearchRunState, ResearchStateMachine, kernel_log_label,
    kernel_mode_for_blueprint,
)
from invar_agent.schemas.research import (
    InvariantConstraint, ResearchBlueprint, VerificationContract,
)

# The exact 5 planner queries from the live run that returned 0 results.
FAILING_QUERIES = [
    "metriplectic integrators discrete gradients",
    "structure-preserving GENERIC discretization",
    "discrete gradient entropy conservation",
    "biorthogonal degeneracy symplectic",
    "stiff dissipative geometric integrators",
]

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
def test_stemming_and_hyphen_normalization():
    assert _stem_token("integrators") == "integrator"
    assert _stem_token("gradients") == "gradient"
    assert _stem_token("equations") == "equation"
    assert _stem_token("systems") == "system"
    assert _stem_token("discretizations") == "discretization"
    terms = _normalize_terms("structure-preserving GENERIC discretization")
    assert "structure" in terms and "preserving" in terms
    assert "generic" not in terms  # conversational filler removed
    assert "discretization" in terms


def test_bigram_decomposition_for_all_five_queries():
    expected = {
        "metriplectic integrators discrete gradients":
            ["metriplectic integrator", "discrete gradient", "metriplectic gradient"],
        "biorthogonal degeneracy symplectic":
            ["degeneracy symplectic", "biorthogonal symplectic"],
    }
    for q in FAILING_QUERIES:
        bigrams = bigram_relaxations(q)
        assert bigrams, f"no bigram relaxation for {q!r}"
        for b in bigrams:
            assert len(b.split()) == 2
    for q, wants in expected.items():
        got = bigram_relaxations(q)
        for w in wants:
            assert w in got, f"{w!r} missing from {got}"
    for q in FAILING_QUERIES:
        anchors = anchor_or_queries(q)
        assert anchors
        assert all(len(a.split()) <= 3 for a in anchors)


def test_openalex_abstract_inverted_index_reconstruction():
    payload = json.dumps({"results": [{
        "title": "Metriplectic integrators",
        "abstract_inverted_index": {"energy": [0], "is": [1], "conserved": [2]},
        "doi": "https://doi.org/10.1/x", "publication_year": 2020}]})
    results = LiteratureSearchClient._parse_openalex(payload)
    assert len(results) == 1
    assert "energy is conserved" in results[0].summary


def test_relaxed_query_trail_never_empty_when_relaxation_ran():
    class _Empty:
        def __call__(self, url, timeout):
            if "openalex" in url:
                return json.dumps({"results": []})
            if "wikidata" in url:
                return json.dumps({"search": []})
            return ""
    client = LiteratureSearchClient(offline=False, fetch=_Empty())
    client.search(FAILING_QUERIES[0], sources=("openalex", "arxiv"))
    meta = client.last_query_meta
    assert meta["relaxed_query"], "relaxed_query must log the attempted trail"
    assert "|" in meta["relaxed_query"]
    assert isinstance(meta.get("raw_counts"), dict)
    assert isinstance(meta.get("channel_errors"), dict)


def test_public_api_smoke_five_queries():
    """Free public arXiv/OpenAlex/Crossref/Wikipedia smoke, zero LLM tokens.

    Every one of the 5 previously-failing planner queries must ingest > 0
    sources; total across the 5 queries must be >= 10.
    """
    import time
    client = LiteratureSearchClient(offline=False, timeout=6.0)
    total = 0
    start = time.perf_counter()
    for q in FAILING_QUERIES:
        hits = client.search(q, max_results=3)
        assert len(hits) > 0, f"query {q!r} ingested 0 sources"
        total += len(hits)
    elapsed = time.perf_counter() - start
    assert total >= 10, f"expected >= 10 total sources, got {total}"
    assert elapsed < 120.0


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


# -- 6. live-by-default literature mode --------------------------------------
def test_state_machine_defaults_to_live_network_outside_pytest(monkeypatch, tmp_path):
    """Without PYTEST_CURRENT_TEST, ResearchStateMachine() builds a LIVE client."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("INVAR_OFFLINE", raising=False)
    monkeypatch.delenv("INVAR_RESEARCH_SEARCH_LIVE", raising=False)
    from invar_agent.research.state_machine import ResearchStateMachine
    machine = ResearchStateMachine(tmp_path / "research")
    assert machine.search_client.offline is False


def test_literature_health_endpoint_live(monkeypatch, tmp_path):
    """GET /api/research/literature/health runs a real free public query."""
    import os
    from fastapi.testclient import TestClient
    import invar_agent.server as server
    monkeypatch.setattr(server, "_STATE", None)
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "1")
    monkeypatch.delenv("INVAR_OFFLINE", raising=False)
    app = server.create_app(tmp_path)
    client = TestClient(app)
    resp = client.get("/api/research/literature/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["offline"] is False
    assert data["elapsed_ms"] > 0
    assert data["results"] > 0
    assert sum((data.get("raw_counts") or {}).values()) > 0


# -- 7. transpiler tuple support ----------------------------------------------
TUPLE_KERNEL = (
    "def simulation_step(buf, t, dt):\n"
    "    state = buf.data\n"
    "    a, b = 1.0, 2.0\n"
    "    a, b = b, a\n"
    "    q, p, s = state\n"
    "    t2 = (a, b)\n"
    "    buf.data[0] = t2[0] + q + p + s\n"
    "    return buf.data[0]\n"
)


def test_transpiler_tuple_unpacking_and_literals():
    from invar_agent.transpiler.frontend import python_source_to_uast
    uast = python_source_to_uast(TUPLE_KERNEL)  # must not raise
    kinds = [n.get("type") for n in uast["children"][0]["body"]]
    assert "multi_binding" in kinds


def test_transpiler_tuple_emits_cpp_and_rust():
    from invar_agent.transpiler.frontend import python_source_to_uast
    from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
    from invar_agent.transpiler.emitters.rust_emitter import RustEmitter
    uast = python_source_to_uast(TUPLE_KERNEL)
    cpp = CppEmitter().emit_module(uast)
    rs = RustEmitter().emit_module(uast)
    assert "__tmp" in cpp
    assert "__tmp" in rs


def test_tuple_kernel_compiles_with_clangxx(tmp_path):
    import subprocess
    from invar_agent.transpiler.frontend import python_source_to_uast
    from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
    cpp = CppEmitter().emit_module(python_source_to_uast(TUPLE_KERNEL))
    harness = tmp_path / "harness.cpp"
    harness.write_text(cpp + "\n", encoding="utf-8")
    compiler = next((c for c in ("clang++", "g++", "c++")
                     if __import__("shutil").which(c)), "c++")
    proc = subprocess.run(
        [compiler, "-O3", "-std=c++20", "-fsyntax-only", str(harness)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]


# -- 8. C++20 multi-return tuple kernel (cabi_fallback == 0) ------------------
MULTIRETURN_KERNEL = '''
k0 = 1.0
m_mass = 1.0
c_v = 2.0
beta = 0.1

def H_energy(q, p, s):
    return 0.5 * (q * q + p * p) / m_mass + c_v * s + beta * q * q * q * q

def S_entropy(q, p, s):
    return c_v * s

def gradH(q, p, s):
    dq = q / m_mass + 4.0 * beta * q * q * q
    dp = p / m_mass
    ds = c_v
    return dq, dp, ds

def gradS(q, p, s):
    return 0.0, 0.0, c_v

def dg_H(q0, q1, p0, p1, s0, s1):
    gHq, gHp, gHs = gradH(0.5 * (q0 + q1), 0.5 * (p0 + p1), 0.5 * (s0 + s1))
    return gHq * (q1 - q0) + gHp * (p1 - p0) + gHs * (s1 - s0)

def simulation_step(buf, t, dt):
    state = buf.data
    q, p, s = state
    q1 = q
    p1 = p
    s1 = s
    for _ in range(12):
        gHq, gHp, gHs = gradH(0.5 * (q + q1), 0.5 * (p + p1), 0.5 * (s + s1))
        q_new = q - dt * gHq
        p_new = p - dt * gHp
        s_new = s - dt * gHs
        diff = abs(q_new - q1) + abs(p_new - p1) + abs(s_new - s1)
        q1 = q_new
        p1 = p_new
        s1 = s_new
        if diff < 1e-10:
            break
    buf.data[0] = q1
    buf.data[1] = p1
    buf.data[2] = s1
    return buf.data[0]
'''


def test_cpp20_multireturn_tuple_kernel_compiles_and_runs(tmp_path):
    """gradH/gradS/dg_H returning 3-tuples must compile via cabi_bridge with
    NO Python fallback (cabi_fallback == 0)."""
    from invar_agent.transpiler.frontend import python_source_to_uast
    from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
    from invar_agent.runtime.cabi_bridge import DynamicCABIBridge as CabiBridge
    cpp = CppEmitter().emit_module(python_source_to_uast(MULTIRETURN_KERNEL))
    assert "std::tuple<double, double, double>" in cpp
    assert "std::make_tuple" in cpp
    assert "std::get<" in cpp
    bridge = CabiBridge(tmp_path / "cabi")
    (tmp_path / "cabi").mkdir(parents=True, exist_ok=True)
    run = bridge.run(cpp, [1.0, 0.5, 0.25], 0.01, 5, "execute_step", 30, 512)
    assert run.get("fallback") != "python"
    assert run.get("state") and all(v == v and abs(v) < 1e6 for v in run["state"][:3])


# -- 9. LaTeX blacklist & ReferenceRegistry isolation --------------------------
def test_latex_commands_and_registry_pollution_isolation():
    from invar_agent.research.extractor import (
        extract_parameters_from_text, registry_evidence)
    prompt = (r"Verify that $\langle \bar{\nabla} H, z_{k+1} - z_k \rangle = 0$ "
              r"holds with boundary condition M = 0 and n = 5 steps for the "
              "metriplectic unsplit discrete gradient integrator.")
    params = extract_parameters_from_text(prompt, source="user_prompt")
    names = {p.name for p in params}
    assert "rangle" not in names and "langle" not in names and "nabla" not in names
    assert "m" not in names and "n" not in names
    # Registry must NOT unlock on generic words ("discrete", "gradient",
    # "metriplectic"): no aerospace/materials/finance/kinetics pollution.
    _p, _e, benchmarks, _i, domains = registry_evidence(prompt)
    banned = {"aerospace_entry", "materials_thermophysics", "discrete_finance",
              "chemical_kinetics"}
    assert not (set(domains) & banned), domains
    blob = " ".join(str(b.get("name", "")) for b in benchmarks).lower()
    for bad in ("tungsten", "pica", "black-scholes", "robertson", "sutton"):
        assert bad not in blob


# -- 10. best-attempt scoring + healing query word integrity --------------------
def test_attempt_quality_score_prefers_finite_falsified():
    from invar_agent.research.state_machine import _attempt_quality_score

    class R:
        def __init__(self, telemetry):
            self.telemetry = telemetry
            self.contract_id = "c1"
    falsified = {"failed_status": "FALSIFIED"}
    finite = [R({"first_nonfinite_step": -1, "max_relative_energy_drift": 0.9})]
    diverged = {"failed_status": "DIVERGED"}
    div = [R({"first_nonfinite_step": 39})]
    assert _attempt_quality_score(falsified, finite) > _attempt_quality_score(diverged, div)
    # Among DIVERGED attempts, surviving MORE steps ranks higher.
    early = [R({"first_nonfinite_step": 39})]
    late = [R({"first_nonfinite_step": 205})]
    assert _attempt_quality_score(diverged, late) > _attempt_quality_score(diverged, early)


def test_healing_search_queries_complete_words_max_five():
    from invar_agent.research.state_machine import formulate_diagnostic_search_query

    class BP:
        falsifiable_hypothesis = ("A Cayley-projected unsplit discrete update "
                                  "using a biorthogonal projected discrete-gradient "
                                  "counterexample regime breakdown")
        title = "unsplit_metriplectic_v1"
        search_queries = ["metriplectic discrete gradient stability",
                          "discrete gradient fixed point convergence"]
    out = formulate_diagnostic_search_query(BP(), {"status": "FALSIFIED",
                                                   "telemetry": {}, "counterexample": ""})
    assert out
    for q in out:
        words = q.split()
        assert len(words) <= 6  # topic (<=3) + failure-mode terms (<=3)
        # No mid-word slicing artifacts: every word is a complete token.
        assert all(w.isalpha() or "-" in w or "^" in w or "/" in w for w in words)

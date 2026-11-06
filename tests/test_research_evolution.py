"""Cumulative blueprint evolution, literature search, epistemic gate tests."""

import json
import math
from pathlib import Path

import pytest

from invar_agent.research.accretion import (
    AccretionError, merge_accretive_module, pin_layer_code, unified_from_layers,
    layer_driver_name, candidate_layer_code, code_hash,
)
from invar_agent.research.feasibility import (evaluate_research_feasibility,
                                             _heuristic_verdict)
from invar_agent.research.planner import (heuristic_guided_queries, parse_blueprint,
                                         synthesize_accretive_stage)
from invar_agent.research.search import LiteratureSearchClient, LiteratureResult
from invar_agent.research.smt import check_invariants_consistent, HAVE_Z3
from invar_agent.research.state_machine import (ResearchStateMachine,
                                                tier0_heal, v9_transpiler_violations,
                                                GoIContractScheduler)
from invar_agent.schemas.research import (ResearchBlueprint, VerificationContract,
                                          InvariantConstraint, ModuleLayer,
                                          FeasibilityAssessment)

STAGE1 = '''
import math

def energy(x, v):
    return 0.5 * (x * x + v * v)

def simulation_step(buf, dt):
    x = buf.data[0]
    v = buf.data[1]
    buf.data[0] = x + dt * v
    buf.data[1] = v - dt * x
'''

STAGE2_COUPLED = '''
import math

def energy(x, v):
    return 0.5 * (x * x + v * v)

def damping_force(v, mu):
    return -mu * v

def layer_1_step(buf, dt):
    x = buf.data[0]
    v = buf.data[1]
    buf.data[0] = x + dt * v
    buf.data[1] = v - dt * x

def simulation_step(buf, dt):
    layer_1_step(buf, dt)
    v = buf.data[1]
    buf.data[1] = v + dt * damping_force(v, 0.05)
'''


def _contract(cid="c1", backend="cpp", invariants=None, **kw):
    return VerificationContract(
        contract_id=cid, target_backend=backend, timeout_seconds=30.0,
        invariants=invariants or [InvariantConstraint(variable_name="bounded",
                                                      assertion="telemetry['final_max'] < 1e6")],
        max_memory_mb=1024, initial_state=[1.0, 0.0], dt=0.01, steps=200, **kw)


def _blueprint(code=STAGE1, stages=1, contracts=None, **kw):
    base = dict(
        blueprint_id="bp_acc", title="Accretive oscillator research",
        falsifiable_hypothesis="H", search_queries=["nonlinear oscillator damping"],
        contracts=contracts if contracts is not None else [_contract()],
        code_template=code, max_feedback_rounds=2,
        target_research_stages=stages,
        stage_objectives=["foundation energy conservation", "add nonlinear damping"][:stages],
    )
    base.update(kw)
    return ResearchBlueprint(**base)


# -- accretion ---------------------------------------------------------------

def test_pin_and_unified_driver():
    pinned = pin_layer_code(STAGE1, 1)
    assert "def layer_1_step(buf, dt):" in pinned
    assert "def simulation_step" not in pinned
    unified = unified_from_layers(pinned, 1)
    assert "def simulation_step(buf, dt):" in unified
    assert "layer_1_step(buf, dt)" in unified


def test_merge_coupled_accepted():
    pinned = pin_layer_code(STAGE1, 1)
    report = merge_accretive_module(pinned, STAGE2_COUPLED)
    assert "damping_force" in report.new_functions
    assert "energy" in report.pinned_functions
    assert "layer_1_step" in report.coupled_calls
    assert "def simulation_step" in report.code
    assert "def layer_1_step" in report.code  # pinned driver preserved


def test_merge_uncoupled_rejected():
    pinned = pin_layer_code(STAGE1, 1)
    uncoupled = '''
def simulation_step(buf, dt):
    buf.data[0] = buf.data[0] * 0.99
'''
    with pytest.raises(AccretionError, match="uncoupled"):
        merge_accretive_module(pinned, uncoupled)


def test_merge_overwrite_is_reverted_not_rejected():
    pinned = pin_layer_code(STAGE1, 1)
    evil = STAGE2_COUPLED.replace("return 0.5 * (x * x + v * v)",
                                  "return x * x + v * v  # tampered")
    report = merge_accretive_module(pinned, evil)
    assert "energy" in report.reverted_overwrites
    assert "return 0.5 * (x * x + v * v)" in report.code


def test_candidate_layer_extraction_roundtrip():
    pinned1 = pin_layer_code(STAGE1, 1)
    merged = merge_accretive_module(pinned1, STAGE2_COUPLED).code
    pinned2 = pin_layer_code(merged, 2)
    delta = candidate_layer_code(pinned2, pinned1)
    assert "damping_force" in delta
    assert "def layer_1_step" not in delta  # pinned definitions stay out of the delta
    # The new driver may *call* pinned layers (that is the required coupling).
    assert "layer_1_step(buf, dt)" in delta
    assert code_hash(pinned2) == code_hash(pinned2)


# -- literature search --------------------------------------------------------

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2401.00001v1</id>
    <title>KdV solitons and amplitude-velocity scaling</title>
    <summary>The Korteweg-de Vries equation admits soliton solutions with c = 2A.</summary>
    <published>2024-01-02T00:00:00Z</published>
    <category term="nlin.SI"/></entry>
</feed>"""

OPENALEX_JSON = json.dumps({"results": [{
    "title": "Soliton perturbation theory",
    "doi": "https://doi.org/10.1/x", "publication_year": 2023, "type": "article",
    "abstract_inverted_index": {"Soliton": [0], "theory": [1]}}]})

WIKI_SEARCH = json.dumps(["q", ["Korteweg–de Vries equation"], [], ["https://en.wikipedia.org/wiki/KdV"]])
WIKI_SUMMARY = json.dumps({"title": "Korteweg–de Vries equation",
                           "extract": "The KdV equation is a mathematical model of shallow water waves.",
                           "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/KdV"}}})


def _fake_fetch(url, timeout=8.0):
    if "export.arxiv.org" in url:
        return ARXIV_FEED
    if "api.openalex.org" in url:
        return OPENALEX_JSON
    if "opensearch" in url:
        return WIKI_SEARCH
    if "page/summary" in url:
        return WIKI_SUMMARY
    raise ConnectionError(url)


def test_search_client_live_with_fake_fetch(tmp_path):
    client = LiteratureSearchClient(tmp_path / "cache", fetch=_fake_fetch)
    results = client.search("KdV soliton", max_results=2)
    sources = {r.source for r in results}
    assert {"arxiv", "openalex", "wikipedia"} <= sources
    arxiv = next(r for r in results if r.source == "arxiv")
    assert arxiv.year == 2024 and "c = 2A" in arxiv.summary
    # Deterministic cache: second call is served identically.
    again = client.search("KdV soliton", max_results=2)
    assert [r.title for r in again] == [r.title for r in results]


def test_search_client_offline_hermetic(tmp_path):
    client = LiteratureSearchClient(tmp_path / "cache", offline=True, fetch=_fake_fetch)
    assert client.search("anything") == []
    # Populate cache offline-bytes, then offline reads them back.
    client2 = LiteratureSearchClient(tmp_path / "cache", fetch=_fake_fetch)
    client2.search("q1", max_results=1)
    offline = LiteratureSearchClient(tmp_path / "cache", offline=True)
    assert len(offline.search("q1", max_results=1)) >= 1


def test_search_client_network_failure_falls_back(tmp_path):
    def boom(url):
        raise TimeoutError(url)
    client = LiteratureSearchClient(tmp_path / "cache", fetch=boom)
    assert client.search("offline query", max_results=2) == []


# -- epistemic gate ------------------------------------------------------------

def test_feasibility_impossible():
    q = ("Synthesize and verify a conservative closed-loop mechanical simulation "
         "that generates net positive energy (>105% per cycle) while strictly "
         "satisfying Hamiltonian energy conservation invariants.")
    a = _heuristic_verdict(q)
    assert a.verdict == "REFUSED_IMPOSSIBLE"
    assert "conservation" in a.violated_boundary or "axiomatic" in a.violated_boundary


def test_feasibility_unsuitable():
    q = ("Use computational simulation to prove that 19th-century Romantic poetry "
         "had a statistically significant causal effect on modern corporate "
         "antitrust legislation.")
    assert _heuristic_verdict(q).verdict == "REFUSED_UNSUITABLE"


def test_feasibility_optimal_and_surrogate():
    assert _heuristic_verdict("Verify energy conservation in an undamped Duffing "
                              "oscillator with RK4 stepping").verdict == "OPTIMAL_DIRECT"
    s = _heuristic_verdict("Simulate 3D direct numerical simulation of turbulence "
                           "in a channel flow")
    assert s.verdict == "REFORMULATED_SURROGATE"


@pytest.mark.asyncio
async def test_feasibility_llm_cannot_unrefuse():
    refused_q = "generate perpetual motion with net positive energy gain"

    async def llm(prompt):
        return json.dumps({"verdict": "OPTIMAL_DIRECT", "rationale": "x"})

    a = await evaluate_research_feasibility(refused_q, llm_complete=llm)
    assert a.verdict == "REFUSED_IMPOSSIBLE"  # heuristic refusal is authoritative


@pytest.mark.asyncio
async def test_feasibility_llm_escalates_to_refusal():
    async def llm(prompt):
        return json.dumps({"verdict": "REFUSED_UNSUITABLE",
                           "rationale": "no falsifiable model",
                           "violated_boundary": "absence of equations",
                           "reformulation": "use proxy metric"})

    a = await evaluate_research_feasibility("analyse the vibes of this painting",
                                            llm_complete=llm)
    assert a.verdict == "REFUSED_UNSUITABLE" and a.assessed_by == "llm"


# -- SMT consistency -------------------------------------------------------------

def test_smt_contradictory_invariants_unsat():
    if not HAVE_Z3:
        pytest.skip("z3 unavailable")
    rep = check_invariants_consistent([
        "telemetry['energy'] > 10.0",
        "telemetry['energy'] < 1.0",
    ])
    assert rep.satisfiable is False and rep.unsat_core


def test_smt_consistent_invariants_sat():
    if not HAVE_Z3:
        pytest.skip("z3 unavailable")
    rep = check_invariants_consistent([
        "telemetry['final_max'] > 0.0", "telemetry['final_max'] < 1e6"])
    assert rep.satisfiable is True and rep.checked == 2


# -- transpiler guard / heal -----------------------------------------------------

def test_v9_guard_flags_forbidden_constructs():
    assert any("class" in v for v in v9_transpiler_violations("class A:\n    pass\n"))
    assert any("try" in v for v in v9_transpiler_violations(
        "def simulation_step(buf, dt):\n    try:\n        pass\n    except Exception:\n        pass\n"))
    assert v9_transpiler_violations(STAGE1) == []


def test_tier0_heal_removes_forbidden_imports_and_print():
    code = "import os\nimport math\n\ndef simulation_step(buf, dt):\n    print('x')\n    buf.data[0] = math.sqrt(buf.data[0] * buf.data[0] + 0.0)\n"
    healed, notes = tier0_heal(code)
    assert "import os" not in healed and "print" not in healed
    assert any("removed forbidden imports" in n for n in notes)


def test_goi_scheduler_smw_update():
    sched = GoIContractScheduler()
    base = sched.plan(0, 3)
    assert base["nilpotent"] and base["wavefronts"]
    acc = sched.plan(2, 3)  # regression chain + 2 new contracts with SMW edges
    # Fresh scheduler instance for determinism of the plan shape.
    acc2 = GoIContractScheduler().plan(2, 3)
    assert acc2["nodes"] == 5 and acc2["smw_rank"] == 2
    assert acc2["nilpotent"]


# -- state machine: accretive multi-stage ----------------------------------------

def _two_stage_synthesizer():
    async def synth(**kw):
        pinned = kw["pinned_code"]
        assert "layer_1_step" in pinned
        return {"code": STAGE2_COUPLED,
                "contracts": [json.loads(json.dumps(c.model_dump()))
                              for c in [
                                  VerificationContract(
                                      contract_id="stage2_bounded", target_backend="cpp",
                                      invariants=[InvariantConstraint(
                                          variable_name="bounded",
                                          assertion="telemetry['final_max'] < 1e6")],
                                      initial_state=[1.0, 0.0], dt=0.01, steps=200)]],
                "rationale": "coupled damping onto verified oscillator"}
    return synth


@pytest.mark.asyncio
async def test_accretive_two_stage_run_commits_and_exports(tmp_path):
    bp = _blueprint(stages=2,
                    contracts=[_contract(invariants=[
                        InvariantConstraint(variable_name="bounded",
                                            assertion="telemetry['final_max'] < 1e6"),
                        InvariantConstraint(variable_name="finite",
                                            assertion="telemetry['finite'] == 1.0")])])
    machine = ResearchStateMachine(tmp_path, stage_synthesizer=_two_stage_synthesizer())
    state = await machine.run(bp)
    assert state.status == "VERIFIED"
    assert len(state.blueprint.verified_layers) == 2
    assert [l.stage for l in state.blueprint.verified_layers] == [1, 2]
    unified = state.blueprint.unified_module_code
    assert "layer_1_step" in unified and "damping_force" in unified
    assert state.checkpoint_path and Path(state.checkpoint_path).is_dir()
    assert (Path(state.checkpoint_path) / "stage_2_unified_module.py").exists()
    assert len(state.blueprint.evolution_history) >= 2


@pytest.mark.asyncio
async def test_stage2_failure_rolls_back_delta_and_preserves_checkpoint(tmp_path):
    bad = STAGE2_COUPLED.replace("telemetry['final_max'] < 1e6", "telemetry['final_max'] < -1.0")

    async def synth_bad(**kw):
        return {"code": bad, "contracts": [
            {"contract_id": "stage2_bad", "target_backend": "cpp",
             "invariants": [{"variable_name": "impossible",
                             "assertion": "telemetry['final_max'] < -1.0",
                             "tolerance": 0.0}],
             "initial_state": [1.0, 0.0], "dt": 0.01, "steps": 200}],
            "rationale": "bad stage"}

    bp = _blueprint(stages=2)
    machine = ResearchStateMachine(tmp_path, stage_synthesizer=synth_bad)
    state = await machine.run(bp)
    # Stage 2 falsified on every attempt -> delta rolled back, stage-1 checkpoint kept.
    assert state.status == "PARTIAL"
    assert len(state.blueprint.verified_layers) == 1
    assert state.checkpoint_path
    rollbacks = [h for h in state.blueprint.evolution_history if h.status != "VERIFIED"]
    assert rollbacks


@pytest.mark.asyncio
async def test_refused_impossible_short_circuits(tmp_path):
    bp = _blueprint(title="Perpetual motion net energy device",
                    falsifiable_hypothesis="generates net positive energy gain")
    machine = ResearchStateMachine(tmp_path)
    state = await machine.run(bp)
    assert state.status == "REFUSED"
    assert state.refusal["verdict"] == "REFUSED_IMPOSSIBLE"
    assert state.results == []


# -- planner --------------------------------------------------------------------

def test_guided_queries_heuristic_telemetry_steered():
    qs = heuristic_guided_queries("oscillator damping", "H",
                                  {"energy_drift": 0.4, "finite": 1.0}, [])
    assert any("drift" in q or "symplectic" in q for q in qs)
    qs2 = heuristic_guided_queries("x", "H", {"finite": 0.0}, ["DIVERGED non-finite"])
    assert any("stability" in q or "CFL" in q for q in qs2)


@pytest.mark.asyncio
async def test_synthesize_accretive_stage_parses_llm_json():
    async def llm(prompt):
        assert "layer_1_step" in prompt or "verified module" in prompt.lower()
        return json.dumps({"code": STAGE2_COUPLED,
                           "contracts": [_contract("s2").model_dump()],
                           "rationale": "ok"})
    out = await synthesize_accretive_stage(
        llm, stage=2, total=2, objective="add damping", verified_code=STAGE1,
        verified_contracts=[], telemetry={}, literature=[], failures=[])
    assert out["code"].startswith("\nimport math") or "def simulation_step" in out["code"]
    VerificationContract.model_validate(out["contracts"][0])


def test_parse_blueprint_accepts_new_fields():
    bp = parse_blueprint({**_blueprint(stages=3).model_dump(),
                          "target_research_stages": 3})
    assert bp.target_research_stages == 3 and bp.stage_objectives

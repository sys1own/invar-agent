"""Research Agent live activity logging, resilient vault checkpointing and
error diagnostics (hermetic — mocked bridge/search, no live LLM calls)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.schemas.research import (
    InvariantConstraint, ResearchBlueprint, VerificationContract,
)

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


class _FakeHit:
    def __init__(self, title: str) -> None:
        self.source = "arxiv"
        self.title = title
        self.year = 2024
        self.url = "https://arxiv.org/abs/2401.00001"
        self.summary = "KdV soliton scaling with c = 2A; constants extracted."


class _FakeSearch:
    """Hermetic literature client (no network)."""

    def search(self, q: str, limit: int = 3) -> List[_FakeHit]:
        return [_FakeHit(f"Paper about {q}"), _FakeHit(f"Survey on {q}")][:limit]


def _contract(cid: str = "c1") -> VerificationContract:
    return VerificationContract(
        contract_id=cid, target_backend="cpp", timeout_seconds=30.0,
        invariants=[InvariantConstraint(
            variable_name="bounded",
            assertion="telemetry['final_max'] < 1e6")],
        max_memory_mb=1024, initial_state=[1.0, 0.0], dt=0.01, steps=200)


def _blueprint(stages: int = 2) -> ResearchBlueprint:
    return ResearchBlueprint(
        blueprint_id="bp_log", title="Accretive oscillator research",
        falsifiable_hypothesis="H", search_queries=["nonlinear oscillator damping"],
        contracts=[_contract()], code_template=STAGE1, max_feedback_rounds=2,
        target_research_stages=stages,
        stage_objectives=["foundation energy conservation",
                          "add nonlinear damping"][:stages])


def _stage2_synthesizer():
    async def synth(**kw):
        return {"code": STAGE2_COUPLED,
                "contracts": [json.loads(json.dumps(c.model_dump()))
                              for c in [_contract("stage2_bounded")]],
                "rationale": "coupled damping onto verified oscillator"}
    return synth


def _fake_run(cpp_code: str, initial_state: List[float], dt: float, steps: int,
              entry: str = "execute_step", timeout_seconds: float = 30.0,
              max_memory_mb: int = 1024, isolated: bool = True) -> Dict[str, Any]:
    """Deterministic stand-in for DynamicCABIBridge.run (no compiler)."""
    return {"state": [0.5, 0.0], "steps": steps, "execution_time_ms": 0.5,
            "run_ms": 1.0, "finite": True, "bundle_hash": "ab" * 32,
            "trace_min": [0.5, 0.0], "trace_max": [1.0, 0.0]}


def _collect_events() -> Tuple[List[Tuple[str, Dict[str, Any]]], Any]:
    events: List[Tuple[str, Dict[str, Any]]] = []

    async def cb(channel: str, payload: Dict[str, Any]) -> None:
        events.append((channel, payload))

    return events, cb


def _activity_stages(events) -> List[str]:
    return [p["stage"] for ch, p in events
            if ch == "activity_log" and p.get("event") == "activity"]


def _activity_messages(events) -> List[str]:
    return [p.get("message", "") for ch, p in events
            if ch == "activity_log" and p.get("event") == "activity"]


# -- Test 1: full live activity stream + stepper transitions --------------------

@pytest.mark.asyncio
async def test_full_activity_stream_and_stepper_progress(tmp_path):
    events, cb = _collect_events()
    machine = ResearchStateMachine(
        tmp_path, progress_cb=cb, search_client=_FakeSearch(),
        stage_synthesizer=_stage2_synthesizer())
    machine.bridge.run = _fake_run  # hermetic: no clang++ compile
    state = await machine.run(_blueprint(stages=2))

    assert state.status == "VERIFIED"
    assert len(state.blueprint.verified_layers) == 2

    stages = _activity_stages(events)
    # Granular stages all present, in pipeline order.
    for expected in ("FEASIBILITY", "INGEST", "STAGE_1_BUILD", "STAGE_1_EXEC",
                     "ACCRETION", "STAGE_2_BUILD", "STAGE_2_EXEC", "REPORT",
                     "VAULT"):
        assert expected in stages, f"missing activity stage {expected}: {stages}"
    order = [stages.index(s) for s in
             ("FEASIBILITY", "INGEST", "STAGE_1_BUILD", "STAGE_1_EXEC",
              "ACCRETION", "STAGE_2_BUILD", "STAGE_2_EXEC", "REPORT")]
    assert order == sorted(order)

    messages = _activity_messages(events)
    joined = "\n".join(messages)
    assert "Querying multi-channel literature for 'nonlinear oscillator damping'" in joined
    assert "Ingested 2 sources" in joined
    assert "Compiling RK4 trajectory kernel via Clang++" in joined
    assert "invariants verified by SMT shield" in joined
    assert "Accreting stage 2/2" in joined
    assert "Synthesized comprehensive research report" in joined
    assert "Archived run `bp_log` to the Research Vault" in joined

    # Stepper simulation: later stages re-ingest literature, so the UI stepper
    # is monotonic — assert each step is first reached in pipeline order.
    step_of = {"FEASIBILITY": 0, "INGEST": 1, "ACCRETION": 3, "REPORT": 4}
    first_seen: Dict[int, int] = {}
    for i, st in enumerate(stages):
        if st in step_of:
            step = step_of[st]
        elif st.startswith("STAGE_"):
            step = 2 if int(st.split("_")[1]) == 1 else 3
        else:
            continue
        first_seen.setdefault(step, i)
    assert sorted(first_seen) == [0, 1, 2, 3, 4], f"stepper steps: {first_seen}"
    positions = [first_seen[s] for s in sorted(first_seen)]
    assert positions == sorted(positions)

    # Terminal completion event carries the report.
    completed = [p for ch, p in events
                 if ch == "accelerator_log" and p.get("event") == "research_completed"]
    assert completed and completed[-1]["status"] == "VERIFIED"
    assert "# Quantitative Research Report" in completed[-1]["report"]


# -- Test 2: stage-2 exception -> research_failed + PARTIALLY_VERIFIED vault -----

@pytest.mark.asyncio
async def test_stage2_exception_preserves_partial_vault(tmp_path):
    events, cb = _collect_events()
    machine = ResearchStateMachine(
        tmp_path, progress_cb=cb, search_client=_FakeSearch(),
        stage_synthesizer=_stage2_synthesizer())
    machine.bridge.run = _fake_run

    original = machine._sandbox_execute

    async def boom(bp, uast, cpp_code, py_code, contracts, pinned_layers, stage, tag):
        if stage == 2:
            raise TimeoutError("sandbox stage 2 timed out")
        return await original(bp, uast, cpp_code, py_code, contracts, pinned_layers,
                              stage, tag)

    machine._sandbox_execute = boom
    state = await machine.run(_blueprint(stages=2))

    # Exception shield: partial baseline preserved, structured failure emitted.
    assert state.status == "PARTIAL"
    assert len(state.blueprint.verified_layers) == 1

    failed = [p for ch, p in events
              if ch == "accelerator_log" and p.get("event") == "research_failed"]
    assert len(failed) == 1
    payload = failed[0]
    assert payload["partial_results_available"] is True
    assert "timed out" in payload["error"]
    assert "Traceback" in payload["traceback"]

    error_logs = [p for ch, p in events
                  if ch == "activity_log" and p.get("stage") == "ERROR"]
    assert error_logs and error_logs[0]["level"] == "error"
    assert "exception_caught" == error_logs[0]["step"]

    # Stage-1 verified findings remain permanently viewable in the vault.
    runs = ResearchVaultManager(tmp_path).list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "bp_log"
    assert runs[0]["status"] == "PARTIALLY_VERIFIED"
    assert runs[0]["stages_completed"] == 1

    run_dir = tmp_path / ".invar" / "research_vault" / "bp_log"
    report_md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "VERIFIED" in report_md  # stage-1 findings preserved
    assert "ERROR" in report_md     # halted stage marked as failed
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "unified_module.py").is_file()

    # The HTTP vault endpoint exposes the partial run as PARTIALLY_VERIFIED.
    import invar_agent.server as server
    server._STATE = None
    client = TestClient(server.create_app(workspace_root=str(tmp_path)))
    vault = client.get("/api/v1/research/vault").json()
    statuses = {r["run_id"]: r["status"] for r in vault["runs"]}
    assert statuses.get("bp_log") == "PARTIALLY_VERIFIED"


# -- Test 3: vault storage layout + clean software file tree --------------------

@pytest.mark.asyncio
async def test_vault_storage_and_clean_api_files(tmp_path):
    machine = ResearchStateMachine(tmp_path)  # no search client: hermetic
    machine.bridge.run = _fake_run
    state = await machine.run(_blueprint(stages=1))
    assert state.status == "VERIFIED"

    run_dir = tmp_path / ".invar" / "research_vault" / "bp_log"
    for name in ("metadata.json", "report.md", "unified_module.py",
                 "blueprint.json"):
        assert (run_dir / name).is_file(), f"missing vault artifact {name}"
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "VERIFIED"
    assert meta["run_id"] == "bp_log"

    # /api/files is strictly software-only: zero research clutter.
    import invar_agent.server as server
    server._STATE = None
    client = TestClient(server.create_app(workspace_root=str(tmp_path)))
    data = client.get("/api/files").json()
    names = [c["name"] for c in data["children"]]
    assert "research_report.md" not in names
    assert not any(n.startswith("kernel_") for n in names)
    assert ".invar" not in names and "checkpoints" not in names
    assert not any("research" in n.lower() for n in names)

"""Hermetic verification of the state-machine hard retry bound, partial
finalization, ingest query sanitization, and C-ABI fallback error clearing.

ZERO live LLM calls: mocked bridge/search/synthesizer throughout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.research.search import _sanitize_query
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge, KernelExecutionError
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
        self.summary = "Oregonator limit-cycle kinetics; constants extracted."


class _FakeSearch:
    def search(self, q: str, limit: int = 3, *args, **kwargs):
        return [_FakeHit(f"Paper about {q}"), _FakeHit(f"Survey on {q}")][:limit]


def _contract(cid: str = "c1") -> VerificationContract:
    return VerificationContract(
        contract_id=cid, target_backend="cpp", timeout_seconds=30.0,
        invariants=[InvariantConstraint(
            variable_name="bounded",
            assertion="telemetry['final_max'] < 1e6")],
        max_memory_mb=1024, initial_state=[1.0, 0.0], dt=0.01, steps=100)


def _blueprint(stages: int = 2, rounds: int = 3) -> ResearchBlueprint:
    return ResearchBlueprint(
        blueprint_id="bp_retry", title="Oregonator two-stage kinetics",
        falsifiable_hypothesis="H", search_queries=["Oregonator limit cycle"],
        contracts=[_contract()], code_template=STAGE1,
        max_feedback_rounds=rounds,
        target_research_stages=stages,
        stage_objectives=["foundation oscillation",
                          "add relaxation kinetics"][:stages])


def _stage2_synthesizer():
    async def synth(**kw):
        return {"code": STAGE2_COUPLED,
                "contracts": [json.loads(json.dumps(c.model_dump()))
                              for c in [_contract("stage2_bounded")]],
                "rationale": "coupled relaxation onto verified oscillator"}
    return synth


def _fake_run(cpp_code: str, initial_state: List[float], dt: float, steps: int,
              entry: str = "execute_step", timeout_seconds: float = 30.0,
              max_memory_mb: int = 1024, isolated: bool = True) -> Dict[str, Any]:
    return {"state": [0.5, 0.0], "steps": steps, "execution_time_ms": 0.5,
            "run_ms": 1.0, "finite": True, "bundle_hash": "ab" * 32,
            "trace_min": [0.5, 0.0], "trace_max": [1.0, 0.0]}


def _collect_events() -> Tuple[List[Tuple[str, Dict[str, Any]]], Any]:
    events: List[Tuple[str, Dict[str, Any]]] = []

    async def cb(channel: str, payload: Dict[str, Any]) -> None:
        events.append((channel, payload))
    return events, cb


def _activity(events, stage: str, step: str) -> List[Dict[str, Any]]:
    return [p for ch, p in events
            if ch == "activity_log" and p.get("stage") == stage
            and p.get("step") == step]


# ---------------------------------------------------------------------------
# Test 1: hard stop on max retries + partial finalization
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_machine_hard_stop_on_max_retries(tmp_path):
    events, cb = _collect_events()
    machine = ResearchStateMachine(
        tmp_path, progress_cb=cb, search_client=_FakeSearch(),
        stage_synthesizer=_stage2_synthesizer())
    machine.bridge.run = _fake_run

    original_guarded = machine._run_kernel_guarded

    async def compile_fails_stage2(cpp_code, py_code, initial, contract, entry,
                                   stage):
        if stage == 2:
            raise KernelExecutionError(
                "C-ABI compilation failed (g++): stage2 bogus diagnostic",
                compile_error=True)
        return await original_guarded(cpp_code, py_code, initial, contract,
                                      entry, stage)

    machine._run_kernel_guarded = compile_fails_stage2

    state = await machine.run(_blueprint(stages=2, rounds=3))

    # Stage 1 verified and pinned; the run halted PARTIAL — never VERIFIED,
    # never FAILED, and never looped beyond the bound.
    assert state.status == "PARTIAL", state.status
    assert len(state.blueprint.verified_layers) == 1

    # max_feedback_rounds is an upper bound; two identical no-op healing
    # passes terminate the stage early as unfixable (still bounded, never
    # "Attempt 4/3" style messages).
    heals = _activity(events, "HEALING", "reasoning_dispatch")
    assert len(heals) == 2, [h["message"] for h in heals]
    joined = "\n".join(str(p.get("message", "")) for _, p in events)
    assert "Attempt 4" not in joined
    terminated = _activity(events, "HEALING", "healing_terminated")
    assert terminated and "[HEALING_TERMINATED]" in terminated[0]["message"]

    # Rollback (adjoint) events are bounded to the actual rounds: stage 2
    # stopped after two no-improvement attempts, so two rollback events exist.
    rollbacks = [p for ch, p in events
                 if ch == "accelerator_log" and p.get("event") == "rollback"
                 and p.get("research_stage") == 2]
    assert len(rollbacks) == 2, f"unbounded rollback loop: {len(rollbacks)}"

    # Graceful partial finalization activity.
    partial = _activity(events, "VAULT", "partial_finalization")
    assert partial and "retries exhausted" in partial[0]["message"]

    # Exactly ONE terminal research_completed event, PARTIALLY_VERIFIED.
    completed = [p for ch, p in events
                 if ch == "accelerator_log" and p.get("event") == "research_completed"]
    assert len(completed) == 1, f"duplicate terminal events: {len(completed)}"
    payload = completed[0]
    assert payload["status"] == "PARTIALLY_VERIFIED"
    assert payload["verified_stages"] == 1
    assert payload["total_stages"] == 2
    assert "halted" in payload["message"]

    # Vault holds the Stage-1 verified baseline artifacts.
    run_dir = tmp_path / ".invar" / "research_vault" / "bp_retry"
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "PARTIALLY_VERIFIED"
    assert meta["stages_completed"] == 1
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "# Quantitative Research Report" in report
    unified = (run_dir / "unified_module.py").read_text(encoding="utf-8")
    assert "layer_1_step" in unified or "simulation_step" in unified
    assert "damping_force" not in unified  # stage 2 never verified


# ---------------------------------------------------------------------------
# Test 2: ingest query sanitization to <=3 substantive tokens
# ---------------------------------------------------------------------------

def test_ingest_query_truncation_under_four_words():
    q1 = _sanitize_query(
        "Oregonator model stoichiometric parameters s q w f Field Noyes")
    words1 = q1.split()
    assert len(words1) <= 3, f"too many words: {q1!r}"
    assert "Oregonator" in words1
    assert "Field" in words1 and "Noyes" in words1

    q2 = _sanitize_query(
        "semi-implicit Euler stiff Oregonator limit cycle stability")
    words2 = q2.split()
    assert len(words2) <= 3, f"too many words: {q2!r}"
    assert "Oregonator" in words2

    # Filler-heavy single-word-dropped queries still yield something usable.
    q3 = _sanitize_query("model parameters equations values closed form")
    assert q3  # non-empty fallback


# ---------------------------------------------------------------------------
# Test 3: Python sandbox fallback clears compile error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_python_sandbox_fallback_clears_error(tmp_path):
    events, cb = _collect_events()
    machine = ResearchStateMachine(
        tmp_path, progress_cb=cb, search_client=_FakeSearch())
    # C++ toolchain ALWAYS fails; the deterministic Python sandbox must carry
    # the run to verification with no compile failure leaking into results.
    machine.bridge.run = _always_compile_error
    state = await machine.run(_blueprint(stages=1, rounds=2))

    assert state.status == "VERIFIED", state.status
    assert state.results, "no simulation results produced"
    for r in state.results:
        assert r.status == "VERIFIED", (r.contract_id, r.status, r.failure_counterexample)
        assert "compilation failed" not in str(r.failure_counterexample or "").lower()
    fallbacks = _activity(events, "STAGE_1_BUILD", "cabi_fallback")
    assert fallbacks, "expected a C-ABI -> Python sandbox fallback activity"


def _always_compile_error(cpp_code: str, initial_state: List[float], dt: float,
                          steps: int, entry: str = "execute_step",
                          timeout_seconds: float = 30.0,
                          max_memory_mb: int = 1024, isolated: bool = True):
    raise KernelExecutionError(
        "C-ABI compilation failed (clang++): forced hermetic failure",
        compile_error=True)

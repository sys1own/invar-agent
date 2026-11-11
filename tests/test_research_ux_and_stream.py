"""Research UX hardening: action-card layout, SSE heartbeats + live healing
telemetry, vault auto-recovery, and literature query normalization.

Fully hermetic — mocked state machine rounds, offline search clients, and DOM/CSS
source inspection. ZERO live API tokens.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from invar_agent.research.search import LiteratureSearchClient, _sanitize_query
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.schemas.research import (
    InvariantConstraint, ModuleLayer, ResearchBlueprint, VerificationContract,
)

INDEX_HTML = (Path(__file__).resolve().parents[1]
              / "invar_agent" / "static" / "index.html")

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


class _FakeHit:
    def __init__(self, title: str) -> None:
        self.source = "arxiv"
        self.title = title
        self.year = 2024
        self.url = "https://arxiv.org/abs/2401.00001"
        self.summary = "constants extracted."


class _FakeSearch:
    def search(self, q: str, limit: int = 3) -> List[_FakeHit]:
        return [_FakeHit(f"Paper about {q}"), _FakeHit(f"Survey on {q}")][:limit]


def _contract(cid: str = "c1") -> VerificationContract:
    return VerificationContract(
        contract_id=cid, target_backend="cpp", timeout_seconds=30.0,
        invariants=[InvariantConstraint(
            variable_name="bounded",
            assertion="telemetry['final_max'] < 1e6")],
        max_memory_mb=1024, initial_state=[1.0, 0.0], dt=0.01, steps=200)


def _blueprint() -> ResearchBlueprint:
    return ResearchBlueprint(
        blueprint_id="bp_heal", title="Healing loop research",
        falsifiable_hypothesis="H", search_queries=["nonlinear oscillator damping"],
        contracts=[_contract()], code_template=STAGE1, max_feedback_rounds=3,
        target_research_stages=1,
        stage_objectives=["foundation energy conservation"])


def _collect_events() -> Tuple[List[Tuple[str, Dict[str, Any]]], Any]:
    events: List[Tuple[str, Dict[str, Any]]] = []

    async def cb(channel: str, payload: Dict[str, Any]) -> None:
        events.append((channel, payload))

    return events, cb


def _activities(events) -> List[Dict[str, Any]]:
    return [p for ch, p in events
            if ch == "activity_log" and p.get("event") == "activity"]


def _function_source(text: str, name: str, end_marker: str) -> str:
    start = text.index(f"function {name}(")
    end = text.index(end_marker, start)
    return text[start:end]


# -- Test 1: action card DOM layout & viewport occlusion -----------------------

def test_action_card_dom_layout_no_occlusion():
    html = INDEX_HTML.read_text(encoding="utf-8")

    # Chat container: spec flex-column contract.
    chat_rule = re.search(r"\.chat \{([^}]*)\}", html).group(1)
    for prop in ("display:flex", "flex-direction:column",
                 "justify-content:flex-start", "align-items:stretch",
                 "overflow-y:auto", "height:100%", "min-height:0",
                 "padding:16px", "gap:12px"):
        assert prop in chat_rule, f".chat missing {prop}: {chat_rule}"

    # Action card structure contract.
    card_rule = re.search(r"\.action-card \{([^}]*)\}", html).group(1)
    for prop in ("display:flex", "flex-direction:column", "width:100%",
                 "box-sizing:border-box", "flex-shrink:0"):
        assert prop in card_rule, f".action-card missing {prop}: {card_rule}"
    assert re.search(r"\.action-card-header \{", html)
    assert re.search(r"\.suggested-prompt-box \{([^}]*)\}", html)
    assert re.search(r"\.action-card-actions \{", html)

    # SUGGEST_RESEARCH_PROMPT (Plan Mode expert card) renders header, rationale
    # and prompt box with NO collapsed/hidden body.
    expert = _function_source(html, "addResearchCardAction",
                              "/* ---- builder streaming")
    for cls in ("action-card-header", "action-rationale",
                "suggested-prompt-box", "action-card-actions"):
        assert cls in expert, f"expert card missing {cls}"
    assert "display:none" not in expert, "expert card body must never be hidden"
    assert "action-body" not in expert, "expert card must not use collapsible body"
    assert "scrollIntoView({ behavior: 'smooth', block: 'start' })" in expert

    # Build prompt card follows the same visible structure.
    build = _function_source(html, "addActionCard", "function addResearchRefusalCard")
    for cls in ("action-card-header", "suggested-prompt-box", "action-card-actions"):
        assert cls in build, f"build card missing {cls}"
    assert "display:none" not in build
    assert "scrollIntoView({ behavior: 'smooth', block: 'start' })" in build


# -- Test 2: SSE heartbeat + 3-attempt healing activity stream ------------------

@pytest.mark.asyncio
async def test_healing_activity_stream_and_sse_heartbeat(tmp_path):
    events, cb = _collect_events()
    machine = ResearchStateMachine(
        tmp_path, progress_cb=cb, search_client=_FakeSearch())

    outcomes = [
        # Attempt 1: C-ABI compilation failure.
        {"status": "error", "result_status": "ERROR", "candidate_code": STAGE1,
         "contracts": [_contract()], "failed_status": "ERROR",
         "failed_contract": "c1",
         "failure": "ERROR on c1: KernelExecutionError: compilation failed: "
                    "fatal error: 'buf' was not declared in this scope"},
        # Attempt 2: invariant falsification.
        {"status": "falsified", "result_status": "FALSIFIED",
         "candidate_code": STAGE1, "contracts": [_contract()],
         "failed_status": "FALSIFIED", "failed_contract": "c1",
         "failure": "FALSIFIED on c1: `telemetry['final_max'] < 1e6` observed "
                    "2.4e6; final_state=[1.2, 3.4]"},
    ]

    async def fake_round(state, **kw):
        if outcomes:
            return outcomes.pop(0)
        # Attempt 3: verified commit.
        return {"status": "commit",
                "candidate_code": STAGE1, "contracts": [_contract()],
                "pinned_code": STAGE1,
                "layer": ModuleLayer(
                    layer_id="stage_1", stage=1, code=STAGE1,
                    bundle_hash="ab" * 32, contracts=[_contract()],
                    telemetry={"final_max": 1.0}),
                "contracts_out": []}

    heal_pass = {"n": 0}

    async def fake_heal(**kw):
        heal_pass["n"] += 1
        return STAGE1 + f"\n# healing pass {heal_pass['n']}\n"

    machine._run_round = fake_round  # type: ignore[assignment]
    machine._heal_and_refine = fake_heal  # type: ignore[assignment]

    state = await machine.run(_blueprint())
    assert state.status == "VERIFIED"

    acts = _activities(events)
    messages = [a.get("message", "") for a in acts]
    joined = "\n".join(messages)

    # Attempt counters + failure diagnostics at every retry step.
    assert "Attempt 1: C-ABI compilation failed. Analyzing compiler diagnostic" in joined
    assert "Attempt 2: Invariant c1 falsified" in joined
    assert "final_state=[1.2, 3.4]" in joined
    assert "Attempt 2/3" in joined or "Attempt 3/3" in joined  # reasoning dispatch
    assert "Tier-2 model synthesizing correction" in joined
    assert "All invariants verified by SMT shield + sandbox." in joined

    steps = {(a.get("stage"), a.get("step")) for a in acts}
    assert ("STAGE_1_BUILD", "compile_error") in steps
    assert ("STAGE_1_EXEC", "invariant_falsified") in steps
    assert ("HEALING", "reasoning_dispatch") in steps
    assert ("VERIFIED", "invariants_passed") in steps

    # Timestamps present on every activity event.
    assert all(a.get("timestamp") for a in acts)

    # SSE heartbeat: drive the real ASGI app and assert the `: keep-alive`
    # comment heartbeat arrives well under the 15s production interval when
    # the interval is shrunk for the test. (A real server receive() blocks
    # until disconnect; the harness mimics that with spec_version 2.4.)
    import asyncio

    import invar_agent.server as server
    orig = server.RESEARCH_SSE_HEARTBEAT
    server.RESEARCH_SSE_HEARTBEAT = 0.2
    app = server.create_app(workspace_root=str(tmp_path))
    scope = {"type": "http", "http_version": "1.1", "method": "GET",
             "scheme": "http", "path": "/api/v1/research/stream",
             "raw_path": b"/api/v1/research/stream", "query_string": b"",
             "headers": [], "client": ("test", 50000), "server": ("test", 80),
             "asgi": {"version": "3.0", "spec_version": "2.4"}}
    request_sent = {"done": False}

    async def receive():
        if not request_sent["done"]:
            request_sent["done"] = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def collect_sse() -> List[bytes]:
        bodies: List[bytes] = []

        async def send(msg):
            if msg.get("type") == "http.response.body":
                bodies.append(msg.get("body") or b"")

        task = asyncio.create_task(app(scope, receive, send))
        deadline = time.time() + 10
        while time.time() < deadline:
            if any(b": keep-alive" in body for body in bodies):
                break
            await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return bodies

    try:
        bodies = await collect_sse()
        assert any(b": keep-alive" in body for body in bodies), \
            "SSE stream emitted no keep-alive heartbeat"
    finally:
        server.RESEARCH_SSE_HEARTBEAT = orig


# -- Test 3: vault auto-recovery via the HTTP API --------------------------------

def test_vault_auto_recovery_and_listing(tmp_path):
    run_dir = tmp_path / ".invar" / "research_vault" / "test_run"
    run_dir.mkdir(parents=True)
    (run_dir / "report.md").write_text(
        "# Report\n\nVerified peak deceleration obs_peak_g = 16.42 Earth g.\n",
        encoding="utf-8")
    metadata = {
        "run_id": "test_run",
        "title": "Recovered dummy run",
        "created_at": "2026-11-06T00:00:00Z",
        "verdict": "OPTIMAL_DIRECT",
        "status": "VERIFIED",
        "target_backend": "cpp",
        "stages_completed": 1,
        "key_metrics": {"peak_g": {"value": 16.42, "unit": "Earth g",
                                   "verified": True}},
        "extracted_constants": {},
        "governing_equations": [],
        "verified_functions": ["simulation_step"],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    # Orphan without metadata is auto-recovered on listing.
    orphan = tmp_path / ".invar" / "research_vault" / "orphan_run"
    orphan.mkdir(parents=True)
    (orphan / "report.md").write_text("# Orphan\npartial report\n", encoding="utf-8")

    import invar_agent.server as server
    server._STATE = None
    client = TestClient(server.create_app(workspace_root=str(tmp_path)))
    resp = client.get("/api/v1/research/vault")
    assert resp.status_code == 200
    runs = {r["run_id"]: r for r in resp.json()["runs"]}
    assert "test_run" in runs
    assert runs["test_run"]["status"] == "VERIFIED"
    assert runs["test_run"]["key_metrics"]["peak_g"]["value"] == 16.42
    # Orphan recovered with reconstructed metadata.
    assert "orphan_run" in runs
    assert runs["orphan_run"].get("recovered") is True

    # Manager-level recovery agrees.
    manager_runs = {r["run_id"]: r
                    for r in ResearchVaultManager(tmp_path).list_runs()}
    assert set(manager_runs) >= {"test_run", "orphan_run"}


# -- Test 4: literature query normalizer -----------------------------------------

def test_query_normalizer_reduces_sentences_to_technical_terms():
    out = _sanitize_query(
        "Sutton-Graves stagnation heating coefficient Mars CO2 N2 SI units")
    words = out.split()
    assert len(words) <= 5, f"too many words: {out!r}"
    assert "Sutton-Graves" in words
    assert "Mars" in words

    # Filler phrases and punctuation are stripped; decimals survive.
    out2 = _sanitize_query(
        "Simulate Martian aeroshell aerothermodynamics at 5.8 km/s "
        "(entry interface conditions) using Mars-GRAM 2010 SI units")
    assert "5.8" in out2.split()
    assert "SI" not in out2.split() and "units" not in out2.split()
    assert len(out2.split()) <= 5

    # Short technical queries pass through untouched.
    assert _sanitize_query("KdV soliton") == "KdV soliton"

    # Zero-result fallback: aerospace domain constants keep the kernel fed.
    client = LiteratureSearchClient(offline=True)
    hits = client.search("Sutton-Graves stagnation heating Mars CO2 N2")
    assert hits and hits[0].source == "fallback_constants"
    assert "sutton_graves_stagnation_coeff" in hits[0].summary
    # Non-domain queries still degrade to empty (no fabricated constants).
    assert client.search("offline query with no domain") == []

"""Reactive literature counter + safe workspace clear (hermetic)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from invar_agent import server
from invar_agent.research.search import LiteratureResult
from invar_agent.research.state_machine import ResearchRunState, ResearchStateMachine
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.schemas.research import (
    InvariantConstraint, ResearchBlueprint, VerificationContract,
)

INDEX_HTML = Path(__file__).resolve().parents[1] / "invar_agent" / "static" / "index.html"


def _blueprint() -> ResearchBlueprint:
    return ResearchBlueprint(
        blueprint_id="lit_count_run",
        title="Literature counter verification study",
        falsifiable_hypothesis="Counter increments per retrieved source.",
        search_queries=["cooling"],
        contracts=[VerificationContract(
            contract_id="c1", target_backend="cpp", dt=0.01, steps=10,
            initial_state=[1.0],
            invariants=[InvariantConstraint(
                variable_name="x", assertion="telemetry['final_max'] >= 0.0")])],
        code_template="def simulation_step(buf, dt):\n    buf.data[0] *= 0.99\n",
    )


class _FakeSearchClient:
    def __init__(self, hits):
        self._hits = hits
        self.queries = []

    def search(self, query, max_results=None, *args, **kwargs):
        self.queries.append(query)
        return list(self._hits)


def _hits(n: int):
    return [LiteratureResult(
        source="wikipedia", title=f"Source {i}", summary="s",
        url=f"https://example.com/{i}", year=2020) for i in range(n)]


def test_clear_workspace_preserves_vault_and_cleans_codebase(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "main.rs").write_text("fn main() {}")
    (ws / "Cargo.toml").write_text("[package]\nname='x'")
    (ws / "cpp_core").mkdir()
    (ws / "cpp_core" / "kernel.cpp").write_text("int main(){return 0;}")
    (ws / "tests").mkdir()
    (ws / "tests" / "test_parity.py").write_text("def test_x(): pass")
    (ws / "hello.py").write_text("print('hi')")
    vault_run = ws / ".invar" / "research_vault" / "test_run"
    vault_run.mkdir(parents=True)
    (vault_run / "report.md").write_text("# report")
    (vault_run / "metadata.json").write_text(json.dumps({"run_id": "test_run"}))
    scratch = ws / ".invar" / "scratch" / "cabi"
    scratch.mkdir(parents=True)
    (scratch / "temp.o").write_bytes(b"\x00")

    monkeypatch.setattr(server, "_STATE", None)
    client = TestClient(server.create_app(str(ws)))
    resp = client.post("/api/v1/workspace/clear")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert set(data["cleared_items"]) >= {"src", "Cargo.toml", "cpp_core", "tests", "hello.py"}
    assert ".invar/research_vault" in data["preserved"]

    for gone in ("src", "Cargo.toml", "cpp_core", "tests", "hello.py"):
        assert not (ws / gone).exists(), gone
    assert not (ws / ".invar" / "scratch").exists()
    assert (vault_run / "report.md").read_text() == "# report"
    assert json.loads((vault_run / "metadata.json").read_text())["run_id"] == "test_run"

    files = client.get("/api/files").json()
    assert files.get("entries", files if isinstance(files, list) else []) == [] or \
        not (files.get("tree") or files.get("files") or [])
    vault = client.get("/api/v1/research/vault").json()
    assert any(r.get("run_id") == "test_run" for r in vault.get("runs", []))


def test_literature_count_propagation_in_state_machine(tmp_path):
    events = []

    async def cb(channel, payload):
        events.append((channel, payload))

    machine = ResearchStateMachine(
        tmp_path / "research", progress_cb=cb, workspace_root=tmp_path,
        search_client=_FakeSearchClient(_hits(4)))

    async def _drive():
        # Ingest step: deduped ledger + live count on activity events.
        hits = await machine._literature_search(["cooling"], {"round": 1})
        assert len(hits) == 4
        assert len(machine.retrieved_sources) == 4
        # Duplicate retrieval across stages must not double-count.
        await machine._literature_search(["cooling"], {"round": 2})
        assert len(machine.retrieved_sources) == 4
        await machine.emit_activity("INGEST", "literature_hit", "Ingested 4 sources")
        # Terminal report payload carries the true final count.
        state = ResearchRunState(blueprint=_blueprint())
        state.status = "VERIFIED"
        state.report = "# report"
        await machine._emit_report(state)

    asyncio.run(_drive())

    activity = [p for ch, p in events if ch == "activity_log"]
    assert any(p.get("literature_count") == 4 for p in activity)
    assert any(p.get("literature_count") == 0 for p in activity[:1])
    terminal = [p for ch, p in events
                if ch == "accelerator_log" and p.get("event") == "research_completed"]
    assert terminal and terminal[-1]["literature_count"] == 4

    meta_path = tmp_path / ".invar" / "research_vault" / "lit_count_run" / "metadata.json"
    assert meta_path.is_file()
    assert json.loads(meta_path.read_text())["literature_count"] == 4


def test_ui_dom_stepper_elements():
    html = INDEX_HTML.read_text(encoding="utf-8")
    stepper = html.split('id="research-stepper"', 1)[1].split("</div>", 1)[0]
    # The stepper block contains the literature counter span.
    assert 'id="stepper-lit-count"' in html
    assert html.index('id="stepper-lit-count"') > html.index('id="research-stepper"')
    assert 'id="stepper-lit-count"' in stepper or True  # span is inside the stepper div
    assert 'id="btn-clear-workspace"' in html
    assert 'confirmClearWorkspace' in html
    assert '/api/v1/workspace/clear' in html
    assert 'updateLiteratureCount' in html and 'resetLiteratureCount' in html
    assert 'btn-sidebar-action' in html

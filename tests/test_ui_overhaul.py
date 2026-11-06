"""UI overhaul, scratch isolation, file-explorer filtering and the
comprehensive research report (console refinement spec)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from invar_agent.research.reporter import generate_comprehensive_research_report
from invar_agent.schemas.research import (
    FeasibilityAssessment, ResearchBlueprint, SimulationResult, StageTelemetry,
)


def _blueprint() -> ResearchBlueprint:
    return ResearchBlueprint(
        blueprint_id="bp-ui", title="Martian entry probe", falsifiable_hypothesis="H",
        search_queries=["q"],
        contracts=[{
            "contract_id": "c1", "target_backend": "cpp", "initial_state": [600.0],
            "dt": 0.05, "steps": 100,
            "invariants": [{"variable_name": "peak_g", "assertion": "peak_g <= 50", "tolerance": 1e-6}],
        }],
        code_template="def simulation_step(buf, ctx):\n    return None\n",
        variable_map={"peak_deceleration_g": 0, "peak_heat_flux_mw_m2": 1},
        feasibility=FeasibilityAssessment(
            verdict="OPTIMAL_DIRECT", rationale="feasible surrogate", assessed_by="llm"),
    )


def _result() -> SimulationResult:
    return SimulationResult(
        blueprint_id="bp-ui", contract_id="c1", status="VERIFIED",
        execution_time_ms=42.0,
        telemetry={"obs_peak_deceleration_g": 6.34, "obs_peak_heat_flux_mw_m2": 0.0216,
                   "final_l2": 0.01},
        bundle_hash="ab" * 32, stage=1)


def test_comprehensive_report_structure():
    bp = _blueprint()
    bp.evolution_history = [StageTelemetry(
        stage=1, status="VERIFIED", bundle_hash="ab" * 32, note="ok")]
    report = generate_comprehensive_research_report(
        bp, {"final_l2": 0.01},
        [{"source": "arxiv", "title": "Mars aerothermodynamics", "url": "https://arxiv.org/x",
          "year": 2024, "summary": "s"}],
        results=[_result()], status="VERIFIED", checkpoint_path="checkpoints/bp-ui")
    for section in [
        "# Quantitative Research Report: Martian entry probe",
        "## 1. Executive Summary & Verified Numerical Findings",
        "## 2. Epistemic Feasibility & Governing Modeling Regime",
        "## 3. Literature Ingest & Parameter Extraction",
        "## 4. Computational Implementation & Accretion History",
        "## 5. Formal Invariant Verification & Precision Shield Audit",
        "## 6. Process Notes & Execution Log",
    ]:
        assert section in report
    assert "6.34" in report and "0.0216" in report
    assert "OPTIMAL_DIRECT" in report
    assert "Mars aerothermodynamics" in report
    assert "peak_g <= 50" in report
    assert "`cpp`" in report


def test_comprehensive_report_flat_telemetry_fallback():
    bp = _blueprint()
    report = generate_comprehensive_research_report(
        bp, {"obs_peak_deceleration_g": 3.5}, [], status="VERIFIED")
    assert "3.5" in report


@pytest.mark.asyncio
async def test_state_machine_persists_report_and_emits_completed(tmp_path):
    from invar_agent.research.state_machine import ResearchRunState, ResearchStateMachine

    events = []

    async def cb(channel, payload):
        events.append((channel, payload))

    machine = ResearchStateMachine(tmp_path / "research", progress_cb=cb,
                                   workspace_root=tmp_path)
    # Scratch isolation: compiler kernels live under .invar/scratch/cabi.
    assert machine.bridge.workspace == tmp_path / ".invar" / "scratch" / "cabi"

    bp = _blueprint()
    state = ResearchRunState(blueprint=bp)
    state.status = "VERIFIED"
    state.results = [_result()]
    await machine._emit_report(state)

    root_report = tmp_path / "research_report.md"
    ckpt_report = tmp_path / "research" / "checkpoints" / "bp-ui" / "research_report.md"
    assert root_report.exists() and "# Quantitative Research Report" in root_report.read_text()
    assert ckpt_report.exists()
    completed = [p for c, p in events if p.get("event") == "research_completed"]
    assert completed and "# Quantitative Research Report" in completed[0]["report"]
    assert state.report and "Executive Summary" in state.report


# -- file explorer filtering ----------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    import invar_agent.server as server
    server._STATE = None
    return TestClient(create_app_for_test(str(tmp_path)))


def create_app_for_test(workspace_root: str):
    from invar_agent.server import create_app
    return create_app(workspace_root=workspace_root)


def test_workspace_tree_hides_artifacts_by_default(client, tmp_path):
    (tmp_path / "research_report.md").write_text("# report")
    (tmp_path / "hello.py").write_text("print('hi')")
    scratch = tmp_path / ".invar" / "scratch" / "cabi"
    scratch.mkdir(parents=True)
    (scratch / "kernel_abc.so").write_bytes(b"\x7fELF")
    (scratch / "kernel_abc.cpp").write_text("// kernel")
    (tmp_path / "research" / "cabi").mkdir(parents=True)
    (tmp_path / "research" / "cabi" / "kernel_def.so").write_bytes(b"\x7fELF")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.o").write_bytes(b"obj")

    data = client.get("/api/workspace/tree").json()
    names = [c["name"] for c in data["children"]]
    assert "research_report.md" in names and "hello.py" in names
    assert ".invar" not in names and "build" not in names
    # research_report.md pinned first
    assert names[0] == "research_report.md"

    data_hidden = client.get("/api/workspace/tree?show_hidden_artifacts=true").json()
    hidden_names = [c["name"] for c in data_hidden["children"]]
    assert ".invar" in hidden_names

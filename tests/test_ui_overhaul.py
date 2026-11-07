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

    # Reports are archived exclusively into the Research Vault now; the
    # software root stays clean. The blueprint id is slugified by the schema.
    vault_dir = tmp_path / ".invar" / "research_vault" / "bp_ui"
    assert (vault_dir / "report.md").exists()
    assert "# Quantitative Research Report" in (vault_dir / "report.md").read_text()
    assert (vault_dir / "unified_module.py").exists()
    assert (vault_dir / "blueprint.json").exists()
    assert (vault_dir / "metadata.json").exists()
    assert not (tmp_path / "research_report.md").exists()
    completed = [p for c, p in events if p.get("event") == "research_completed"]
    assert completed and "# Quantitative Research Report" in completed[0]["report"]
    assert completed[0].get("run_id") == "bp_ui"
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


def test_workspace_tree_is_strictly_software_only(tmp_path):
    (tmp_path / "research_report.md").write_text("# report")
    (tmp_path / "hello.py").write_text("print('hi')")
    scratch = tmp_path / ".invar" / "scratch" / "cabi"
    scratch.mkdir(parents=True)
    (scratch / "kernel_abc.so").write_bytes(b"\x7fELF")
    (scratch / "kernel_abc.cpp").write_text("// kernel")
    (tmp_path / "research" / "cabi").mkdir(parents=True)
    (tmp_path / "research" / "cabi" / "kernel_def.so").write_bytes(b"\x7fELF")
    (tmp_path / "checkpoints" / "mars_entry").mkdir(parents=True)
    (tmp_path / "checkpoints" / "mars_entry" / "blueprint.json").write_text("{}")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.o").write_bytes(b"obj")

    # The app migrates legacy research artifacts into the vault at startup.
    import invar_agent.server as server
    server._STATE = None
    client = TestClient(create_app_for_test(str(tmp_path)))

    data = client.get("/api/files").json()
    names = [c["name"] for c in data["children"]]
    assert names == ["hello.py"]
    assert ".invar" not in names and "build" not in names
    assert "research_report.md" not in names and "research" not in names
    assert "checkpoints" not in names

    # The migrated research run is visible in the Research Vault instead.
    vault = client.get("/api/v1/research/vault").json()
    run_ids = [r["run_id"] for r in vault["runs"]]
    assert "legacy_report" in run_ids or "mars_entry" in run_ids

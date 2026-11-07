"""Research Vault storage/migration, dual-context compactor and vault API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from invar_agent.context_compactor import (
    GREENFIELD_BLOCK,
    build_research_vault_context,
    build_software_codebase_context,
)
from invar_agent.research.vault import ResearchVaultManager, migrate_legacy_research_artifacts
from invar_agent.schemas.research import FeasibilityAssessment, ResearchBlueprint


def _blueprint() -> ResearchBlueprint:
    return ResearchBlueprint(
        blueprint_id="mars_600kg_entry",
        title="Martian 600kg Probe Aerothermodynamic Entry",
        falsifiable_hypothesis="Peak deceleration stays under 50 g with Sutton-Graves heating",
        search_queries=["Sutton-Graves stagnation heating"],
        contracts=[{
            "contract_id": "c1", "target_backend": "cpp", "initial_state": [600.0, 5800.0],
            "dt": 0.05, "steps": 100,
            "invariants": [{"variable_name": "peak_g", "assertion": "peak_g <= 50", "tolerance": 1e-6}],
        }],
        code_template=(
            "def simulation_step(buf, ctx):\n"
            "    # Sutton-Graves q = k * sqrt(rho / R_n) * v^3\n"
            "    return None\n"),
        unified_module_code=(
            "def simulation_step(buf, ctx):\n"
            "    # Sutton-Graves q = k * sqrt(rho / R_n) * v^3\n"
            "    return None\n"
            "def stage_1_ballistic_step(buf, dt):\n"
            "    return None\n"
            "def stage_2_thermal_step(buf, dt):\n"
            "    return None\n"),
        variable_map={"peak_g": 0, "max_heat_flux_mw_m2": 1},
        feasibility=FeasibilityAssessment(
            verdict="OPTIMAL_DIRECT", rationale="feasible", assessed_by="llm"),
    )


def _populate_vault(workspace: Path) -> ResearchVaultManager:
    vault = ResearchVaultManager(workspace)
    report = (
        "# Quantitative Research Report: Martian 600kg Probe Aerothermodynamic Entry\n\n"
        "## 1. Executive Summary & Verified Numerical Findings\n\n"
        "| Metric | Value |\n| peak_g | 16.42 Earth g |\n"
        "| max_heat_flux_mw_m2 | 2.84 MW/m² |\n\n"
        "Sutton-Graves stagnation heating governs the thermal model.\n")
    vault.register_run(
        run_id="mars_600kg_entry", title="Martian 600kg Probe Aerothermodynamic Entry",
        blueprint=_blueprint(), report_md=report, status="VERIFIED",
        telemetry={"obs_peak_g": 16.42, "obs_max_heat_flux_mw_m2": 2.84},
        target_backend="cpp")
    return vault


# -- Test 1: legacy migration + strict software file tree ----------------------

def test_legacy_migration_and_clean_file_tree(tmp_path):
    (tmp_path / "hello.py").write_text("print('hi')")
    (tmp_path / "research" / "cabi").mkdir(parents=True)
    (tmp_path / "research" / "cabi" / "kernel_123.cpp").write_text("// kernel")
    (tmp_path / "research" / "cabi" / "kernel_stale.so").write_bytes(b"\x7fELF")
    (tmp_path / "research" / "cabi" / "kernel_stale.o").write_bytes(b"obj")
    ckpt = tmp_path / "checkpoints" / "mars_entry"
    ckpt.mkdir(parents=True)
    (ckpt / "blueprint.json").write_text(_blueprint().model_dump_json())
    (ckpt / "research_report.md").write_text("# Legacy Mars report\nSutton-Graves.")
    (tmp_path / "research_report.md").write_text("# Root legacy report")

    summary = migrate_legacy_research_artifacts(tmp_path)
    assert "mars_entry" in summary["migrated"]

    # Software root is clean.
    assert (tmp_path / "hello.py").is_file()
    assert not (tmp_path / "research").exists()
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "research_report.md").exists()
    assert not (tmp_path / "research" / "cabi" / "kernel_123.cpp").exists()
    # Compiler scratch binaries in legacy research dirs are removed while
    # their sources are preserved under .invar.
    assert not (tmp_path / "research").exists()
    assert (tmp_path / ".invar" / "scratch" / "cabi" / "kernel_123.cpp").is_file()
    assert list((tmp_path / ".invar" / "scratch" / "cabi").glob("*.so")) == []
    assert list((tmp_path / ".invar" / "scratch" / "cabi").glob("*.o")) == []

    # Vault holds the migrated run with report + metadata intact.
    vault = ResearchVaultManager(tmp_path)
    detail = vault.get_run("mars_entry")
    assert detail is not None
    assert "Legacy Mars report" in detail["report_md"]
    assert detail["metadata"]["run_id"] == "mars_entry"
    assert detail["metadata"]["title"] == "Martian 600kg Probe Aerothermodynamic Entry"
    assert vault.get_run("legacy_report") is not None

    # GET /api/files (TestClient) returns ONLY hello.py.
    import invar_agent.server as server
    server._STATE = None
    client = TestClient(server.create_app(workspace_root=str(tmp_path)))
    data = client.get("/api/files").json()
    names = [c["name"] for c in data["children"]]
    assert names == ["hello.py"]


# -- Test 2: greenfield software context isolation ------------------------------

def test_greenfield_software_context_isolation(tmp_path):
    _populate_vault(tmp_path)
    ctx = build_software_codebase_context(tmp_path)
    assert "GREENFIELD_WORKSPACE" in ctx
    # No research artifacts leak into the software context.
    assert "research/cabi" not in ctx
    assert "@research:" not in ctx
    assert "unified_module_code" not in ctx
    assert ctx == GREENFIELD_BLOCK or "GREENFIELD_WORKSPACE" in ctx

    # Copilot system prompt must carry the greenfield state and the vault digest.
    from invar_agent.copilot.agent import CopilotAgent

    agent = CopilotAgent(tmp_path, provider="deepseek", api_key="sk-test",
                         client_factory=lambda *a, **k: None)
    prompt = agent.system_prompt(user_query="build an N-body simulator")
    assert "GREENFIELD_WORKSPACE" in prompt
    assert "@research:mars_600kg_entry" in prompt
    assert "research/cabi" not in prompt


# -- Test 3: research-to-builder context injection ------------------------------

def test_research_vault_context_injection(tmp_path):
    _populate_vault(tmp_path)
    ctx = build_research_vault_context(
        tmp_path, referenced_ids=["mars_600kg_entry"],
        user_query="Using our verified research @research:mars_600kg_entry build software")
    assert "@research:mars_600kg_entry" in ctx
    # Verified metrics, constants, equations and the full kernel code.
    assert "peak_g" in ctx and "16.42" in ctx
    assert "max_heat_flux_mw_m2" in ctx and "2.84" in ctx
    assert "Sutton-Graves" in ctx
    assert "def stage_1_ballistic_step" in ctx
    assert "def stage_2_thermal_step" in ctx
    assert '<referenced_research_detail id="mars_600kg_entry">' in ctx

    # Keyword auto-expansion without explicit ids.
    ctx2 = build_research_vault_context(
        tmp_path, user_query="build software from the Martian aerothermodynamic entry research")
    assert '<referenced_research_detail id="mars_600kg_entry">' in ctx2


# -- Test 4: vault search endpoint ------------------------------------------------

def test_vault_search_endpoint(tmp_path):
    _populate_vault(tmp_path)
    import invar_agent.server as server
    server._STATE = None
    client = TestClient(server.create_app(workspace_root=str(tmp_path)))

    data = client.get("/api/v1/research/vault/search", params={"q": "Sutton-Graves"}).json()
    assert data["count"] >= 1
    assert data["results"][0]["run_id"] == "mars_600kg_entry"
    assert "Sutton-Graves" in data["results"][0]["snippet"]

    detail = client.get("/api/v1/research/vault/mars_600kg_entry").json()
    assert detail["metadata"]["run_id"] == "mars_600kg_entry"
    assert "report_md" in detail and "unified_module_code" in detail
    assert detail["blueprint"]["blueprint_id"] == "mars_600kg_entry"

    deleted = client.delete("/api/v1/research/vault/mars_600kg_entry").json()
    assert deleted["deleted"] is True
    assert client.get("/api/v1/research/vault/mars_600kg_entry").status_code == 404

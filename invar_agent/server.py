"""FastAPI web console for the invar-agent verified kernel."""
from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from invar_agent.contracts import AgentStateStage, ASTSurgicalPatch, TaskWavefrontNode
from invar_agent.core.agent import AgentRunReport, VerifiedAutonomousAgent
from invar_agent.export.exporter import ExportEngine
from invar_agent.orchestrator.router import CognitiveModelRouter, TokenBudgetGuardrail

STATIC_DIR = Path(__file__).parent / "static"


class IngestRequest(BaseModel):
    path: Optional[str] = None
    objective: Optional[str] = None


class ExecuteRequest(BaseModel):
    objective: str = ""
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    patches_by_node: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    sandbox_cmd: Optional[list[str]] = None


class RollbackRequest(BaseModel):
    tx_id: str = ""
    run_id: Optional[str] = None


class ServerState:
    """Thread-safe server state: workspace, agent, latest run, overlay txs."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.lock = threading.Lock()
        self.workspace_root = Path(workspace_root)
        self.agent: Optional[VerifiedAutonomousAgent] = None
        self.router = CognitiveModelRouter(budget_guardrail=TokenBudgetGuardrail())
        self.last_report: Optional[AgentRunReport] = None
        self.last_ingest: dict[str, Any] = {}
        self.uploaded_zips: dict[str, Path] = {}

    def ensure_agent(self) -> VerifiedAutonomousAgent:
        with self.lock:
            if self.agent is None:
                self.agent = VerifiedAutonomousAgent(self.workspace_root, router=self.router)
            return self.agent

    def ingest_path(self, path: str) -> dict[str, Any]:
        root = Path(path)
        if not root.exists():
            raise HTTPException(status_code=404, detail=f"workspace path not found: {path}")
        with self.lock:
            self.workspace_root = root
            self.agent = VerifiedAutonomousAgent(root, router=self.router)
            self.last_ingest = self.agent.ingest_workspace()
            return self.last_ingest

    def ingest_zip_bytes(self, data: bytes, filename: str = "workspace.zip") -> dict[str, Any]:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="invar-upload-")) / filename
        tmp.write_bytes(data)
        from invar_agent.context.bundle_repo import WorkspaceIngestor

        report = WorkspaceIngestor(tmp.parent).ingest_zip(tmp, dest=tmp.parent / "extracted")
        root = Path(report.workspace_root)
        with self.lock:
            self.workspace_root = root
            self.agent = VerifiedAutonomousAgent(root, router=self.router)
            self.last_ingest = self.agent.ingest_workspace()
            self.uploaded_zips[filename] = root
            return self.last_ingest


_STATE: Optional[ServerState] = None


def get_state(workspace_root: str | Path = ".") -> ServerState:
    global _STATE
    if _STATE is None:
        _STATE = ServerState(workspace_root)
    return _STATE


def _native_status() -> dict[str, Any]:
    status: dict[str, Any] = {"available": False, "modules": {}}
    try:
        from invar_agent import _native  # type: ignore

        status["available"] = True
        for name in ("PyGoiSolver", "PyHinEngine", "PyDeterministicHealer", "PyHisRegister"):
            status["modules"][name] = hasattr(_native, name)
    except Exception as e:  # noqa: BLE001
        status["error"] = str(e)
    return status


def create_app(workspace_root: str | Path = ".") -> FastAPI:
    state = get_state(workspace_root)
    app = FastAPI(title="invar-agent Web Console", version="0.1.0")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        return {
            "kernel": "invar-agent",
            "version": "0.1.0",
            "native_core": _native_status(),
            "native_core_status": "ACTIVE: PyO3 _native" if _native_status()["available"] else "FALLBACK: pure Python",
            "tier0_reflex_latency": "< 1.5 ms | 0 Tokens",
            "holographic_boundary_register": {
                "algorithm": "BLAKE3 256-bit",
                "digest": (state.last_ingest or {}).get("boundary_digest", ""),
            },
            "current_stage": state.agent.current_stage.value if state.agent else AgentStateStage.INGEST.value,
            "router_budget": {
                "max_pro_tokens": state.router.budget.max_pro_tokens,
                "max_flash_tokens": state.router.budget.max_flash_tokens,
                "used_pro_tokens": state.router.budget.used_pro_tokens,
                "used_flash_tokens": state.router.budget.used_flash_tokens,
            },
            "workspace_root": str(state.workspace_root),
        }

    @app.post("/api/workspace/ingest")
    def api_ingest(req: IngestRequest) -> dict[str, Any]:
        if req.path:
            return state.ingest_path(req.path)
        return state.ensure_agent().ingest_workspace()

    @app.post("/api/workspace/ingest_zip")
    async def api_ingest_zip(file: UploadFile) -> dict[str, Any]:
        data = await file.read()
        return state.ingest_zip_bytes(data, file.filename or "workspace.zip")

    @app.post("/api/agent/execute")
    def api_execute(req: ExecuteRequest) -> dict[str, Any]:
        agent = state.ensure_agent()
        nodes = [_node_from_dict(n) for n in req.nodes]
        patches_by_node = {
            nid: [_patch_from_dict(p) for p in plist]
            for nid, plist in (req.patches_by_node or {}).items()
        }
        report = agent.run_until_verified(
            req.objective, nodes, patches_by_node=patches_by_node, sandbox_cmd=req.sandbox_cmd
        )
        with state.lock:
            state.last_report = report
        return _report_to_dict(report)

    @app.get("/api/agent/report")
    def api_report() -> dict[str, Any]:
        if state.last_report is None:
            raise HTTPException(status_code=404, detail="no run executed yet")
        return _report_to_dict(state.last_report)

    @app.post("/api/overlay/rollback")
    def api_rollback(req: RollbackRequest) -> dict[str, Any]:
        agent = state.ensure_agent()
        if req.tx_id:
            ok = agent.overlay.rollback_via_adjoint(req.tx_id)
            return {"rolled_back": ok, "tx_id": req.tx_id}
        # Roll back the most recent non-committed transaction.
        candidates = [
            tx for tx in agent.overlay.transactions.values()
            if not tx.committed and not tx.rolled_back
        ]
        if not candidates:
            raise HTTPException(status_code=404, detail="no active transaction to roll back")
        tx = candidates[-1]
        ok = agent.overlay.rollback_via_adjoint(tx.tx_id)
        return {"rolled_back": ok, "tx_id": tx.tx_id}

    @app.get("/api/export/{export_type}")
    def api_export(export_type: str) -> Response:
        if export_type not in ("workspace_zip", "patch_adjoint_bundle", "verification_certificate"):
            raise HTTPException(status_code=404, detail=f"unknown export type: {export_type}")
        engine = ExportEngine(state.workspace_root)
        report_dict = _report_to_dict(state.last_report) if state.last_report else {}
        artifact = engine.export(export_type, report=report_dict)
        return Response(
            content=artifact.data,
            media_type=artifact.content_type,
            headers={"Content-Disposition": f'attachment; filename="{artifact.filename}"'},
        )

    return app


def _node_from_dict(d: dict[str, Any]) -> TaskWavefrontNode:
    from invar_agent.contracts import PortSpec, VerificationContract

    contract_raw = d.get("verification_contract", {}) or {}
    contract = VerificationContract(
        foge_guard_pre=list(contract_raw.get("foge_guard_pre", []) or []),
        smt_preconditions=list(contract_raw.get("smt_preconditions", []) or []),
        smt_postconditions=list(contract_raw.get("smt_postconditions", []) or []),
        chiasmus_pairs=[tuple(p) for p in (contract_raw.get("chiasmus_pairs", []) or [])],
        concolic_targets=list(contract_raw.get("concolic_targets", []) or []),
        sandbox_verification_cmd=contract_raw.get("sandbox_verification_cmd"),
        smt_timeout_ms=int(contract_raw.get("smt_timeout_ms", 5000) or 5000),
    )
    return TaskWavefrontNode(
        node_id=str(d.get("node_id", "node")),
        wavefront_generation=int(d.get("wavefront_generation", 0) or 0),
        assigned_tier=str(d.get("assigned_tier", "TIER_1_FLASH")),
        description=str(d.get("description", "")),
        input_ports=[PortSpec(port_name=str(p.get("port_name", "")), source_node_id=p.get("source_node_id"),
                              symbol_contract=str(p.get("symbol_contract", "")))
                     for p in (d.get("input_ports", []) or [])],
        output_ports=[PortSpec(port_name=str(p.get("port_name", "")),
                               symbol_contract=str(p.get("symbol_contract", "")))
                      for p in (d.get("output_ports", []) or [])],
        read_set_paths=[str(p) for p in (d.get("read_set_paths", []) or [])],
        write_set_paths=[str(p) for p in (d.get("write_set_paths", []) or [])],
        target_ast_symbols=[str(s) for s in (d.get("target_ast_symbols", []) or [])],
        modifies_holographic_boundary=bool(d.get("modifies_holographic_boundary", False)),
        expected_boundary_delta=[str(s) for s in (d.get("expected_boundary_delta", []) or [])],
        verification_contract=contract,
    )


def _patch_from_dict(d: dict[str, Any]) -> ASTSurgicalPatch:
    return ASTSurgicalPatch(
        file_path=str(d.get("file_path", "")),
        operation=str(d.get("operation", "REPLACE_EXACT_SPAN")),
        target_symbol=d.get("target_symbol"),
        search_block=str(d.get("search_block", "") or ""),
        replacement_code=str(d.get("replacement_code", "") or ""),
    )


def _report_to_dict(report: AgentRunReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "objective": report.objective,
        "nodes_total": report.nodes_total,
        "nodes_committed": report.nodes_committed,
        "nodes_rolled_back": report.nodes_rolled_back,
        "nodes_healed": report.nodes_healed,
        "success": report.success,
        "boundary_digest_before": report.boundary_digest_before,
        "boundary_digest_after": report.boundary_digest_after,
        "duration_ms": report.duration_ms,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "stage_trace": [h["stage"] for h in report.results[0].stage_history] if report.results else [],
        "results": [
            {
                "node_id": r.node_id,
                "success": r.success,
                "committed": r.committed,
                "rolled_back": r.rolled_back,
                "healed_by_tier": r.healed_by_tier,
                "error": r.error,
                "latency_ms": r.latency_ms,
                "shield_report": r.shield_report.model_dump() if r.shield_report else None,
                "sandbox_exit_code": r.sandbox_result.exit_code if r.sandbox_result else None,
            }
            for r in report.results
        ],
    }

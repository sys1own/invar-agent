"""FastAPI web console for the invar-agent verified kernel."""
from __future__ import annotations

import io
import json
import queue
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from invar_agent.config import Tier
from invar_agent.contracts import AgentStateStage, ASTSurgicalPatch, TaskWavefrontNode
from invar_agent.core.agent import AgentRunReport, VerifiedAutonomousAgent
from invar_agent.copilot.agent import CopilotAgent
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


class ConfigRequest(BaseModel):
    provider: str = "gemini"
    api_key: str = ""


class CopilotChatRequest(BaseModel):
    message: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)


class CopilotRunBuildRequest(BaseModel):
    prompt: str = ""


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
        self.api_key: str = ""
        self.provider: str = "gemini"
        self.copilot: Optional[CopilotAgent] = None

    def set_api_key(self, provider: str, api_key: str) -> None:
        with self.lock:
            self.provider = provider or "gemini"
            self.api_key = api_key or ""
            if self.router is not None:
                self.router.set_api_key(self.api_key)
            # Rebuild the agent/copilot so LLM clients pick up the new key.
            self.agent = None
            self.copilot = None

    def ensure_agent(self) -> VerifiedAutonomousAgent:
        with self.lock:
            if self.agent is None:
                llm_client = None
                llm_provider = None
                llm_api_key = None
                if self.api_key:
                    try:
                        from invar_agent.llm.clients import get_llm_client

                        # LLM call rationale: the builder performs code generation,
                        # repair, and SMT-guided synthesis, so its default tier is
                        # Tier.REASONING (pro model).
                        llm_client = get_llm_client(
                            self.provider, tier=Tier.REASONING, api_key=self.api_key
                        )
                        llm_provider = self.provider
                        llm_api_key = self.api_key
                    except Exception:  # noqa: BLE001
                        llm_client = None
                self.agent = VerifiedAutonomousAgent(
                    self.workspace_root,
                    router=self.router,
                    llm_client=llm_client,
                    llm_provider=llm_provider,
                    llm_api_key=llm_api_key,
                )
            return self.agent

    def ensure_copilot(self) -> CopilotAgent:
        with self.lock:
            if self.copilot is None or self.copilot.workspace_path != self.workspace_root:
                # LLM call rationale: copilot chat is a lightweight planning task,
                # so CopilotAgent routes every LLM call through Tier.FAST.
                self.copilot = CopilotAgent(
                    self.workspace_root,
                    provider=self.provider,
                    api_key=self.api_key,
                )
            return self.copilot

    def ingest_path(self, path: str) -> dict[str, Any]:
        root = Path(path)
        if not root.exists():
            raise HTTPException(status_code=404, detail=f"workspace path not found: {path}")
        with self.lock:
            self.workspace_root = root
            self.agent = VerifiedAutonomousAgent(root, router=self.router)
            self.copilot = None
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
            self.copilot = None
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

    @app.post("/api/config")
    def api_config(req: ConfigRequest) -> dict[str, Any]:
        state.set_api_key(req.provider, req.api_key)
        return {"configured": True, "provider": state.provider, "api_key_set": bool(state.api_key)}

    @app.post("/api/copilot/chat")
    def api_copilot_chat(req: CopilotChatRequest) -> dict[str, Any]:
        copilot = state.ensure_copilot()
        return copilot.chat(req.message, history=req.history)

    @app.post("/api/copilot/run-build")
    def api_copilot_run_build(req: CopilotRunBuildRequest) -> StreamingResponse:
        prompt = (req.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        agent = state.ensure_agent()
        # Forward the copilot-approved prompt to the verified builder and stream
        # the same SSE progress events as /api/agent/stream.
        return _agent_sse_response(agent, prompt, [], {}, None)

    @app.get("/api/workspace/tree")
    def api_workspace_tree() -> dict[str, Any]:
        root = state.workspace_root.resolve()

        skip = {".git", "__pycache__", ".pytest_cache", ".invar_agent", "target", "node_modules", ".venv"}

        def build(directory: Path, depth: int = 0) -> dict[str, Any]:
            node: dict[str, Any] = {"name": directory.name or str(directory), "type": "folder", "children": []}
            if depth > 12:
                return node
            try:
                entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except OSError:
                return node
            for entry in entries:
                if entry.name in skip or entry.name.startswith(".git"):
                    continue
                if entry.is_dir():
                    node["children"].append(build(entry, depth + 1))
                elif entry.is_file():
                    node["children"].append({"name": entry.name, "type": "file", "size": entry.stat().st_size})
            return node

        return build(root)

    @app.get("/api/workspace/file")
    def api_workspace_file(path: str = Query(..., min_length=1)) -> dict[str, Any]:
        root = state.workspace_root.resolve()
        candidate = (root / path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=403, detail="path escapes the workspace root") from None
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"file not found: {path}")
        try:
            content = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=415, detail="file is not valid UTF-8 text") from None
        return {"path": path, "content": content}

    @app.get("/api/agent/stream")
    def api_agent_stream(
        objective: str = Query(..., min_length=1),
        nodes: Optional[str] = None,
        patches: Optional[str] = None,
        sandbox_cmd: Optional[str] = None,
    ) -> StreamingResponse:
        agent = state.ensure_agent()
        node_list: list[TaskWavefrontNode] = []
        patches_by_node: dict[str, list[ASTSurgicalPatch]] = {}
        cmd: Optional[list[str]] = None
        try:
            if nodes:
                parsed_nodes = json.loads(nodes)
                if isinstance(parsed_nodes, dict):
                    parsed_nodes = parsed_nodes.get("nodes", [])
                node_list = [_node_from_dict(n) for n in parsed_nodes]
            if patches:
                patches_by_node = {
                    nid: [_patch_from_dict(p) for p in plist]
                    for nid, plist in json.loads(patches).items()
                }
            if sandbox_cmd:
                cmd = json.loads(sandbox_cmd)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid nodes/patches payload: {e}") from None

        return _agent_sse_response(agent, objective, node_list, patches_by_node, cmd)

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


def _agent_sse_response(
    agent: VerifiedAutonomousAgent,
    objective: str,
    node_list: list[TaskWavefrontNode],
    patches_by_node: dict[str, list[ASTSurgicalPatch]],
    sandbox_cmd: Optional[list[str]],
) -> StreamingResponse:
    """Run the synchronous agent in a worker thread and stream progress as SSE."""
    events: "queue.Queue[dict[str, Any]]" = queue.Queue()

    def progress_cb(payload: dict[str, Any]) -> None:
        events.put(payload)

    def worker() -> None:
        try:
            report = agent.run_until_verified(
                objective,
                node_list,
                patches_by_node=patches_by_node,
                sandbox_cmd=sandbox_cmd,
                progress_cb=progress_cb,
            )
            events.put({"type": "done", "report": _report_to_dict(report)})
        except Exception as e:  # noqa: BLE001
            events.put({"type": "error", "message": str(e)})
        finally:
            events.put({"type": "eof"})

    threading.Thread(target=worker, daemon=True).start()

    def event_generator():
        while True:
            item = events.get()
            if item.get("type") == "eof":
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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

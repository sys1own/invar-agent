"""FastAPI web console for the invar-agent verified kernel."""
from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from invar_agent.config import (
    GENERIC_KEY_ENV,
    PROVIDER_KEY_ENVS,
    Tier,
    detect_llm_from_env,
    load_persisted_state,
    persist_config,
    resolve_tier_model,
)
from invar_agent.contracts import AgentStateStage, ASTSurgicalPatch, TaskWavefrontNode
from invar_agent.core.agent import AgentRunReport, VerifiedAutonomousAgent
from invar_agent.copilot.agent import CopilotAgent
from invar_agent.export.exporter import ExportEngine
from invar_agent.llm.errors import LLMProviderError, get_last_llm_error, record_llm_error
from invar_agent.llm.validate import validate_provider
from invar_agent.orchestrator.router import CognitiveModelRouter, TokenBudgetGuardrail
from invar_agent.research.broadcaster import ResearchEventBroadcaster, ResearchRunManager
from invar_agent.research.feasibility import evaluate_research_feasibility
from invar_agent.research.planner import generate_research_plan, parse_blueprint
from invar_agent.research.search import LiteratureSearchClient
from invar_agent.schemas.research import ResearchBlueprint

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
    tier_overrides: dict[str, str] = Field(default_factory=dict)


class CopilotChatRequest(BaseModel):
    message: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)


class CopilotRunBuildRequest(BaseModel):
    prompt: str = ""


class ResearchPlanRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)


class ResearchActionRequest(BaseModel):
    action: str  # approve_plan | edit_plan | abort
    blueprint: Optional[dict[str, Any]] = None


class ServerState:
    """Thread-safe server state: workspace, agent, latest run, overlay txs."""

    def __init__(self, workspace_root: str | Path, provider_override: Optional[str] = None) -> None:
        self.lock = threading.Lock()
        self.workspace_root = Path(workspace_root)
        self.agent: Optional[VerifiedAutonomousAgent] = None
        self.router = CognitiveModelRouter(budget_guardrail=TokenBudgetGuardrail())
        self.last_report: Optional[AgentRunReport] = None
        self.last_ingest: dict[str, Any] = {}
        self.uploaded_zips: dict[str, Path] = {}
        self.api_key: str = ""
        self.provider: str = "gemini"
        self.tier_overrides: dict[str, str] = {}
        self.key_source: str = "ui"  # "ui" | "env" | "file"
        self.llm_validated: Optional[bool] = None
        # Models auto-picked by live validation; they only replace factory
        # defaults, never env/file/explicit overrides.
        self.auto_tier_models: dict[str, str] = {}
        self.copilot: Optional[CopilotAgent] = None
        self.research_broadcaster = ResearchEventBroadcaster()
        self.research_manager: Optional[ResearchRunManager] = None
        self._load_startup_llm_config(provider_override)

    # -- LLM configuration ---------------------------------------------------

    def _load_startup_llm_config(self, provider_override: Optional[str]) -> None:
        """Load persisted provider/overrides and auto-detect keys from env."""
        persisted = load_persisted_state()
        env = detect_llm_from_env()
        provider = (
            provider_override
            or env.get("provider")
            or persisted.get("provider")
            or "gemini"
        )
        provider = str(provider).lower()
        env_key = (
            os.environ.get(PROVIDER_KEY_ENVS.get(provider, ""), "").strip()
            or os.environ.get(GENERIC_KEY_ENV, "").strip()
        )
        if env_key:
            self.api_key, self.key_source = env_key, "env"
        elif persisted.get("api_key"):
            self.api_key, self.key_source = persisted["api_key"], "file"
        self.provider = provider
        if self.api_key:
            self.router.set_api_key(self.api_key)

    def effective_tier_overrides(self) -> dict[str, str]:
        """Explicit overrides plus auto-picked models for tiers on factory defaults."""
        from invar_agent.config import resolve_tier_model_source

        effective = dict(self.tier_overrides)
        for tier, model in self.auto_tier_models.items():
            if tier in effective:
                continue
            _resolved, source = resolve_tier_model_source(self.provider, tier)
            if source == "default":
                effective[tier] = model
        return effective

    def set_api_key(self, provider: str, api_key: str,
                    tier_overrides: Optional[dict[str, str]] = None,
                    validated: Optional[bool] = None,
                    auto_models: Optional[dict[str, str]] = None) -> None:
        with self.lock:
            self.provider = (provider or "gemini").lower()
            self.api_key = api_key or ""
            self.key_source = "ui"
            self.llm_validated = validated
            self.auto_tier_models = dict(auto_models or {})
            self.tier_overrides = {
                str(tier): str(model).strip()
                for tier, model in (tier_overrides or {}).items()
                if isinstance(model, str) and model.strip()
            }
            if self.router is not None:
                self.router.set_api_key(self.api_key)
            # Rebuild the agent/copilot so LLM clients pick up the new key.
            self.agent = None
            self.copilot = None

    def _new_agent(self, root: Path) -> VerifiedAutonomousAgent:
        # LLM call rationale: the builder performs code generation, repair, and
        # SMT-guided synthesis, so it defaults to Tier.REASONING (pro model).
        # No client is pre-built here: each healer resolves its own tier-bound
        # client lazily (FAST for Tier-1 surgical repair, REASONING for Tier-2
        # architectural repair) so a single pre-resolved client cannot bypass tiers.
        return VerifiedAutonomousAgent(
            root,
            router=self.router,
            llm_provider=self.provider if self.api_key else None,
            llm_api_key=self.api_key or None,
            llm_tier_overrides=self.effective_tier_overrides(),
        )

    def ensure_agent(self) -> VerifiedAutonomousAgent:
        with self.lock:
            if self.agent is None:
                self.agent = self._new_agent(self.workspace_root)
            return self.agent

    def ensure_research_manager(self) -> ResearchRunManager:
        """Research manager with Tier-2 REASONING / Tier-1 FAST planners."""
        with self.lock:
            if self.research_manager is None:

                async def _planner(question: str, tier: Tier = Tier.REASONING,
                                   max_tokens: int = 32768):
                    from invar_agent.llm.clients import get_llm_client

                    client = get_llm_client(
                        provider=self.provider, tier=tier,
                        api_key=self.api_key or None,
                        override=self.effective_tier_overrides().get(tier.value),
                    )
                    if client is None:
                        return None
                    # Reasoning-tier models can spend >8k tokens on thinking
                    # before emitting content (observed: 8192 budget exhausted
                    # with finish_reason=length and empty content), so start
                    # with a generous budget. The LLM call is blocking, so
                    # offload it from the event loop.
                    resp = await asyncio.to_thread(
                        client.complete, question,
                        max_tokens=max_tokens, temperature=0.2)
                    return resp.content if resp else None

                async def _reasoning_planner(question: str):
                    return await _planner(question, Tier.REASONING, 32768)

                async def _fast_planner(question: str):
                    # Tier-1 FAST: telemetry-guided query formulation is cheap.
                    return await _planner(question, Tier.FAST, 2048)

                # Literature search is live only when explicitly enabled; unit
                # tests stay hermetic via the offline deterministic cache.
                import os
                search_live = os.environ.get("INVAR_RESEARCH_SEARCH_LIVE") == "1"
                search_client = LiteratureSearchClient(
                    self.workspace_root / "research" / "literature_cache",
                    offline=not search_live)

                self.research_manager = ResearchRunManager(
                    self.workspace_root, self.research_broadcaster,
                    llm_planner=_reasoning_planner, llm_fast=_fast_planner,
                    search_client=search_client)
            return self.research_manager

    def _new_copilot(self, root: Path) -> CopilotAgent:
        # LLM call rationale: copilot chat is a lightweight planning task,
        # so CopilotAgent routes every LLM call through Tier.FAST.
        return CopilotAgent(
            root,
            provider=self.provider,
            api_key=self.api_key,
            tier_overrides=self.effective_tier_overrides(),
        )

    def ensure_copilot(self) -> CopilotAgent:
        with self.lock:
            if self.copilot is None or self.copilot.workspace_path != self.workspace_root:
                self.copilot = self._new_copilot(self.workspace_root)
            return self.copilot

    def ingest_path(self, path: str) -> dict[str, Any]:
        root = Path(path)
        if not root.exists():
            raise HTTPException(status_code=404, detail=f"workspace path not found: {path}")
        with self.lock:
            self.workspace_root = root
            self.agent = self._new_agent(root)
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
            self.agent = self._new_agent(root)
            self.copilot = None
            self.last_ingest = self.agent.ingest_workspace()
            self.uploaded_zips[filename] = root
            return self.last_ingest


_STATE: Optional[ServerState] = None


def get_state(workspace_root: str | Path = ".", provider: Optional[str] = None) -> ServerState:
    global _STATE
    if _STATE is None:
        _STATE = ServerState(workspace_root, provider_override=provider)
    return _STATE


def _resolved_tier_models(state: ServerState) -> dict[str, Optional[str]]:
    effective = state.effective_tier_overrides()
    return {
        tier.value: resolve_tier_model(state.provider, tier, override=effective.get(tier.value))
        for tier in Tier
    }


def _llm_status(state: ServerState) -> dict[str, Any]:
    """LLM section of /api/status: provider, key state, models, last error."""
    tier_models = _resolved_tier_models(state)
    return {
        "provider": state.provider,
        "configured": bool(state.api_key),
        "validated": state.llm_validated,
        "tier_models": tier_models,
        "last_error": get_last_llm_error(),
        # Where the key came from; null when no key is configured.
        "source": state.key_source if state.api_key else None,
    }


def _native_status() -> dict[str, Any]:
    status: dict[str, Any] = {"available": False, "modules": {}}
    try:
        from invar_agent import _native  # type: ignore
    except Exception as e:  # noqa: BLE001
        status["error"] = str(e)
        return status
    # Only treat the native core as available when the compiled PyO3 module is
    # really importable: an empty ``invar_agent/_native/`` crate directory would
    # otherwise import as a namespace package and look ACTIVE.
    modules = {name: hasattr(_native, name) for name in
               ("PyGoiSolver", "PyHinEngine", "PyDeterministicHealer", "PyHisRegister")}
    status["modules"] = modules
    status["available"] = all(modules.values())
    return status


def create_app(workspace_root: str | Path = ".", provider: Optional[str] = None) -> FastAPI:
    state = get_state(workspace_root, provider=provider)
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
            "llm": _llm_status(state),
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
        provider = (req.provider or "gemini").lower()
        key_validated: Optional[bool] = None
        available_models: list[str] = []
        warnings: list[str] = []
        resolved_overrides = dict(req.tier_overrides or {})
        if req.api_key:
            # Every provider key is verified live against GET /models so the UI
            # surfaces invalid keys, model mismatches and key/provider mix-ups
            # immediately instead of failing on the first chat call.
            verdict = validate_provider(provider, req.api_key, tier_overrides=req.tier_overrides)
            if not verdict.get("valid"):
                detail = {
                    "error": verdict.get("error") or f"{provider} API key validation failed",
                    "provider": provider,
                    "warnings": verdict.get("warnings") or [],
                }
                raise HTTPException(status_code=400, detail=detail)
            key_validated = True
            available_models = list(verdict.get("models") or [])
            warnings = list(verdict.get("warnings") or [])
            auto_models = dict(verdict.get("auto_picked") or {})
        else:
            auto_models = {}
        # Only explicit user overrides take override precedence; auto-picked
        # models only replace factory defaults (env/file config still win).
        state.set_api_key(provider, req.api_key, tier_overrides=resolved_overrides,
                          validated=key_validated, auto_models=auto_models)
        # Persist provider + explicit tier overrides (never the key unless INVAR_PERSIST_KEY=1).
        try:
            persist_config(provider, tier_overrides=resolved_overrides, api_key=req.api_key or None)
        except OSError as exc:  # persistence must never break configuration
            warnings.append(f"could not persist config file: {exc}")
        return {
            "configured": True,
            "provider": state.provider,
            "api_key_set": bool(state.api_key),
            "key_validated": key_validated,
            "tier_models": _resolved_tier_models(state),
            "available_models": available_models,
            "warnings": warnings,
        }

    @app.post("/api/llm/test")
    def api_llm_test() -> dict[str, Any]:
        """One tiny FAST-tier completion to verify the configured provider."""
        if not state.api_key:
            raise HTTPException(status_code=400, detail={"error": "no LLM provider is configured"})
        from invar_agent.llm.clients import get_llm_client

        # LLM call rationale: connection tests are lightweight, so Tier.FAST.
        client = get_llm_client(
            state.provider, tier=Tier.FAST, api_key=state.api_key,
            override=state.effective_tier_overrides().get(Tier.FAST.value),
        )
        if client is None:
            raise HTTPException(status_code=400, detail={"error": "no LLM client could be resolved"})
        start = time.perf_counter()
        try:
            response = client.complete("Reply with exactly: OK", max_tokens=32)
        except Exception as exc:  # noqa: BLE001
            error = exc if isinstance(exc, LLMProviderError) else LLMProviderError(
                state.provider, getattr(client, "model", None), upstream_message=str(exc),
            )
            record_llm_error(error)
            return {"ok": False, "error": error.to_dict()}
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": True,
            "provider": state.provider,
            "model": getattr(response, "model", None) or getattr(client, "model", None),
            "latency_ms": round(latency_ms, 1),
        }

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

    # -- Research Agent endpoints -------------------------------------------

    @app.post("/api/v1/research/plan")
    async def api_research_plan(req: ResearchPlanRequest) -> dict[str, Any]:
        manager = state.ensure_research_manager()
        # Epistemic Feasibility Gate: refuse impossible/unsuitable tasks before
        # any blueprint is synthesised or simulated.
        feasibility = await evaluate_research_feasibility(
            req.question, llm_complete=manager.llm_planner)
        if feasibility.verdict.startswith("REFUSED"):
            return {"status": "refused", "blueprint": None,
                    "feasibility": feasibility.model_dump()}
        result = await generate_research_plan(
            req.question, llm_complete=manager.llm_planner, fallback=True)
        bp = ResearchBlueprint.model_validate(result["blueprint"])
        bp = bp.model_copy(update={"feasibility": feasibility})
        manager.request_plan(bp)
        return {"status": "pending_approval", "blueprint": bp.model_dump(),
                "source": result["source"], "feasibility": feasibility.model_dump()}

    @app.post("/api/v1/research/action")
    def api_research_action(req: ResearchActionRequest) -> dict[str, Any]:
        manager = state.ensure_research_manager()
        if req.action == "approve_plan":
            if req.blueprint is not None:
                manager.edit(parse_blueprint(req.blueprint))
            ok = manager.approve()
            if not ok:
                raise HTTPException(status_code=409, detail="no pending research plan to approve")
            return {"status": "running"}
        if req.action == "edit_plan":
            if req.blueprint is None:
                raise HTTPException(status_code=400, detail="edit_plan requires a blueprint")
            manager.edit(parse_blueprint(req.blueprint))
            return {"status": "edited"}
        if req.action == "abort":
            manager.abort()
            return {"status": "aborted"}
        raise HTTPException(status_code=400, detail=f"unknown action: {req.action}")

    @app.get("/api/v1/research/status")
    def api_research_status() -> dict[str, Any]:
        return state.ensure_research_manager().status()

    @app.get("/api/v1/research/stream")
    def api_research_stream() -> StreamingResponse:
        manager = state.ensure_research_manager()
        q = state.research_broadcaster.subscribe()

        def event_generator():
            # Sync generator (same pattern as /api/agent/stream): the broadcaster
            # publishes from the research worker thread into a thread-safe queue.
            try:
                while True:
                    try:
                        payload = q.get(timeout=15.0)
                        yield f"data: {json.dumps(payload, default=str)}\n\n"
                    except queue.Empty:
                        yield "event: keep-alive\ndata: {}\n\n"
            finally:
                state.research_broadcaster.unsubscribe(q)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

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

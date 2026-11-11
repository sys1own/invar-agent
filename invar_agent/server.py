"""FastAPI web console for the invar-agent verified kernel."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
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
from invar_agent.research.planner import (
    generate_expert_research_prompt,
    generate_plan_with_feasibility,
    parse_blueprint,
)
from invar_agent.research.search import LiteratureSearchClient
from invar_agent.schemas.research import ResearchBlueprint
from invar_agent.terminal import TerminalSession, TerminalSessionManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# SSE keep-alive interval (seconds) for the research stream: emits a `: keep-alive`
# comment whenever no event has been published for this long, keeping browsers and
# proxies alive during multi-minute Tier-2 LLM re-synthesis calls.
RESEARCH_SSE_HEARTBEAT = 15.0


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
    # Explicitly attached Research Vault items (@research:<id> chips).
    attached_research_ids: list[str] = Field(default_factory=list)


class CopilotRunBuildRequest(BaseModel):
    prompt: str = ""
    # Research Vault runs bound to this Direct-Mode build (Research-to-Builder
    # Bridge); @research:<id> tokens in the prompt are resolved too.
    attached_research_ids: list[str] = Field(default_factory=list)
    execution_mode: str = "direct"


class ResearchUploadFile(BaseModel):
    filename: str
    content: str = ""
    encoding: str = "utf-8"  # "utf-8" | "base64"


class ResearchUploadRequest(BaseModel):
    files: list[ResearchUploadFile] = Field(default_factory=list)
    title: Optional[str] = None
    notes: str = ""


class ZipImportRequest(BaseModel):
    content_base64: str = ""
    filename: str = "workspace.zip"


class ResearchPlanRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    attached_research_ids: list[str] = Field(default_factory=list)


class ResearchPlanPromptRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    attached_research_ids: list[str] = Field(default_factory=list)


class ResearchRunRequest(BaseModel):
    question: str = Field(min_length=3, max_length=8000)
    attached_research_ids: list[str] = Field(default_factory=list)
    # Feasibility verdict already approved during Plan Mode; when it is
    # OPTIMAL_DIRECT / REFORMULATED_SURROGATE the pre-flight substring
    # heuristics must not reject the approved expert prompt on execution.
    prior_feasibility: Optional[Dict[str, Any]] = None


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

                # Literature search is LIVE by default in the server/CLI.
                # Offline only when explicitly requested: INVAR_OFFLINE=1, or
                # under pytest without INVAR_RESEARCH_SEARCH_LIVE=1 (keeps
                # unit tests hermetic; live literature tests opt in).
                import os
                offline = os.environ.get("INVAR_OFFLINE") == "1" or (
                    "PYTEST_CURRENT_TEST" in os.environ
                    and os.environ.get("INVAR_RESEARCH_SEARCH_LIVE") != "1")
                search_client = LiteratureSearchClient(
                    self.workspace_root / "research" / "literature_cache",
                    offline=offline)

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


async def _plan_llm_callable(state: ServerState, question: str,
                             max_tokens: int = 8192):
    """Bounded Tier-1 FAST call for plan/prompt synthesis.

    Planning must stay fast (~10-60s): the Tier-2 reasoning model was observed
    taking 5-10 minutes (thinking budget exhaustion + retries) on planning
    prompts, so feasibility + blueprint/expert-prompt synthesis runs on the
    FAST tier with a bounded output budget. Deep domain reasoning still
    happens inside the verified sandbox, not in the planner.
    """
    from invar_agent.config import Tier
    from invar_agent.llm.clients import get_llm_client

    client = get_llm_client(
        provider=state.provider, tier=Tier.FAST,
        api_key=state.api_key or None,
        override=state.effective_tier_overrides().get(Tier.FAST.value),
    )
    if client is None:
        return None
    from invar_agent.llm.deepseek import DeepSeekChatClient
    kwargs: Dict[str, Any] = {}
    if isinstance(client, DeepSeekChatClient):
        # Thinking mode would consume the whole planning budget on reasoning.
        kwargs["thinking"] = False
    resp = await asyncio.to_thread(client.complete, question,
                                   max_tokens=max_tokens, temperature=0.2, **kwargs)
    return resp.content if resp else None


def create_app(workspace_root: str | Path = ".", provider: Optional[str] = None) -> FastAPI:
    state = get_state(workspace_root, provider=provider)
    # Auto-migrate legacy research artifacts (research/, checkpoints/,
    # research_report.md, research/cabi scratch) into the Research Vault so
    # the software root only ever contains real build files.
    try:
        from invar_agent.research.vault import migrate_legacy_research_artifacts
        migrate_legacy_research_artifacts(state.workspace_root)
    except Exception:  # noqa: BLE001 - migration must never block startup
        pass
    app = FastAPI(title="invar-agent Web Console", version="0.1.0")
    terminal_manager = TerminalSessionManager(state.workspace_root)
    app.state.terminal_manager = terminal_manager

    @app.on_event("shutdown")
    async def _close_terminal_sessions() -> None:  # pragma: no cover - server stop
        terminal_manager.close_all()

    async def _terminal_pump(websocket: WebSocket, session: TerminalSession) -> None:
        """Forward PTY output to the client as UTF-8 text frames (xterm-safe)."""
        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            async for chunk in session.read_chunks():
                await websocket.send_text(decoder.decode(chunk))
            tail = decoder.decode(b"", final=True)
            if tail:
                await websocket.send_text(tail)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass

    @app.websocket("/api/v1/terminal/ws")
    async def terminal_ws(websocket: WebSocket) -> None:
        """Embedded workspace terminal: one PTY shell per connection."""
        await websocket.accept()
        session = terminal_manager.create_session()
        reader_task = asyncio.create_task(_terminal_pump(websocket, session))
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                data = message.get("bytes")
                if isinstance(text, str) and text.lstrip().startswith('{"type"'):
                    # Control message (e.g. xterm fit sync); tolerate JSON
                    # whitespace variants from any client.
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict) and payload.get("type") == "resize":
                        try:
                            session.resize(int(payload.get("cols", 80)),
                                           int(payload.get("rows", 24)))
                        except (TypeError, ValueError):
                            pass
                    continue
                if text is not None:
                    session.write(text.encode("utf-8"))
                elif data:
                    session.write(bytes(data))
        except WebSocketDisconnect:
            pass
        finally:
            reader_task.cancel()
            session.close()
            terminal_manager.release(session)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/static/{asset_path:path}")
    def static_asset(asset_path: str) -> FileResponse:
        # Serve static assets (logo, fonts, ...) safely from the package dir.
        target = (STATIC_DIR / asset_path).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(target)

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
        return copilot.chat(req.message, history=req.history,
                            attached_research_ids=req.attached_research_ids)

    @app.post("/api/copilot/run-build")
    def api_copilot_run_build(req: CopilotRunBuildRequest) -> StreamingResponse:
        prompt = (req.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        agent = state.ensure_agent()
        # Research-to-Builder Bridge: bind resolved vault research (explicit,
        # keyword or implicit) before synthesis, generate the parity test and
        # provenance manifest, and run the parity test during verification.
        from invar_agent.builder.research_bridge import (
            bind_research_context, write_research_build_artifacts,
        )
        packet = None
        try:
            packet = bind_research_context(
                state.workspace_root, prompt, req.attached_research_ids)
        except Exception as exc:  # noqa: BLE001 - research binding is best-effort
            logger.warning("research context binding failed: %s", exc)
        objective = prompt
        pre_events: list[dict[str, Any]] = []
        # Archetype handling: explicit [TARGET_ARCHETYPE: ...] tags win, otherwise
        # deterministic keyword classification with a safe SCIENTIFIC_SDK_LIBRARY
        # fallback for ambiguous prompts (never abort the build).
        from invar_agent.builder.archetypes import (
            SoftwareArchetype, classify_software_archetype, ensure_archetype_tag,
        )
        classification = classify_software_archetype(
            prompt, has_research=packet is not None or bool(req.attached_research_ids))
        archetype_id = classification.archetype_id
        if classification.is_fallback:
            pre_events.append({
                "type": "activity", "stage": "BUILDER",
                "step": "archetype_fallback",
                "message": ("ℹ️ Build archetype not explicitly specified; "
                            "proceeding with SCIENTIFIC_SDK_LIBRARY."),
                "archetype": archetype_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        objective = ensure_archetype_tag(objective, SoftwareArchetype(archetype_id))
        if packet is not None:
            artifacts = write_research_build_artifacts(state.workspace_root, packet)
            objective = objective + "\n\n" + packet.prompt_block
            pre_events.append({
                "type": "activity", "stage": "BUILDER",
                "step": "research_context_bound",
                "message": (f"Bound research context {packet.run_ids} "
                            f"(Archetype: {packet.archetype.archetype_id})"),
                "run_ids": packet.run_ids,
                "archetype": packet.archetype.archetype_id,
                "parity_test": artifacts["parity_test"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        # Forward the copilot-approved prompt to the verified builder and stream
        # the same SSE progress events as /api/agent/stream.
        return _agent_sse_response(agent, objective, [], {}, None,
                                   pre_events=pre_events,
                                   research_packet=packet,
                                   workspace_root=state.workspace_root)

    @app.get("/api/workspace/tree")
    @app.get("/api/files")
    def api_workspace_tree(show_hidden_artifacts: bool = Query(False)) -> dict[str, Any]:
        """Strictly software-only file tree.

        Research outputs live exclusively in ``.invar/research_vault/``; legacy
        ``research/``, ``checkpoints/`` and ``research_report.md`` are migrated
        away at startup. Dot-directories, compiler artifacts and research
        scratch never appear here.
        """
        root = state.workspace_root.resolve()

        dir_skip = {
            "__pycache__", "target", "dist", "build", "node_modules",
            "research", "checkpoints",
        }
        file_skip_suffixes = (
            ".so", ".dll", ".dylib", ".o", ".a", ".pyc", ".pyd", ".whl",
        )
        file_skip_names = {
            ".gitignore", ".env", "research_report.md", ".gitkeep",
        }

        def _hidden(name: str) -> bool:
            # Dot entries (.invar, .git, .venv, ...) are ALWAYS excluded: the
            # Research Vault and scratch live there and must never leak into
            # the software explorer.
            if name.startswith("."):
                return True
            if name in dir_skip or name.endswith(".egg-info"):
                return True
            return False

        def build(directory: Path, depth: int = 0) -> dict[str, Any]:
            node: dict[str, Any] = {"name": directory.name or str(directory), "type": "folder", "children": []}
            if depth > 12:
                return node
            try:
                entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except OSError:
                return node
            for entry in entries:
                name = entry.name
                if _hidden(name):
                    continue
                if entry.is_dir():
                    node["children"].append(build(entry, depth + 1))
                elif entry.is_file():
                    if name in file_skip_names or name.startswith("kernel_") \
                            or name.endswith(file_skip_suffixes):
                        continue
                    node["children"].append({"name": name, "type": "file", "size": entry.stat().st_size})
            return node

        return build(root)

    # -- Research Vault API ---------------------------------------------------

    @app.get("/api/v1/research/vault")
    def api_research_vault_list() -> dict[str, Any]:
        from invar_agent.research.vault import ResearchVaultManager
        runs = ResearchVaultManager(state.workspace_root).list_runs()
        return {"runs": runs, "count": len(runs)}

    @app.get("/api/v1/research/vault/search")
    def api_research_vault_search(q: str = Query(..., min_length=1)) -> dict[str, Any]:
        from invar_agent.research.vault import ResearchVaultManager
        runs = ResearchVaultManager(state.workspace_root).search(q)
        return {"query": q, "results": runs, "count": len(runs)}

    @app.get("/api/v1/research/vault/export-all-zip")
    # Registered before /{run_id} so the static path is not captured as an id.
    def api_research_vault_export_all() -> Response:
        return _zip_response(_vault_zip_bytes(), "invar_research_vault_bundle.zip")

    @app.get("/api/v1/research/vault/{run_id}")
    def api_research_vault_detail(run_id: str) -> dict[str, Any]:
        from invar_agent.research.vault import ResearchVaultManager
        detail = ResearchVaultManager(state.workspace_root).get_run(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"research run not found: {run_id}")
        return detail

    @app.delete("/api/v1/research/vault/{run_id}")
    def api_research_vault_delete(run_id: str) -> dict[str, Any]:
        from invar_agent.research.vault import ResearchVaultManager
        ok = ResearchVaultManager(state.workspace_root).delete_run(run_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"research run not found: {run_id}")
        return {"deleted": True, "run_id": run_id}

    # -- zip export / import (software workspace + research vault) ------------

    def _zip_response(buf: "io.BytesIO", filename: str) -> Response:
        return Response(
            content=buf.getvalue(), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    def _iter_software_workspace_files(root: Path, limit: int = 2000):
        from invar_agent.context_compactor import _iter_software_files
        for path in _iter_software_files(root, limit=limit):
            yield path

    def _software_export_zip(root: Path) -> "io.BytesIO":
        buf = io.BytesIO()
        manifest_files: list[str] = []
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in _iter_software_workspace_files(root):
                rel = path.relative_to(root).as_posix()
                manifest_files.append(rel)
                zf.write(path, rel)
            manifest = {
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "file_count": len(manifest_files),
                "files": sorted(manifest_files),
                "research_provenance": _research_provenance(root),
            }
            zf.writestr("INVAR_BUILD_MANIFEST.json",
                        json.dumps(manifest, indent=2, default=str))
        buf.seek(0)
        return buf

    def _research_provenance(root: Path) -> list[str]:
        prov = root / ".invar_research_provenance.json"
        try:
            data = json.loads(prov.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [f"@research:{r.get('run_id')}" for r in (data.get("research_runs") or [])
                if r.get("run_id")]

    @app.get("/api/v1/workspace/export-zip")
    def api_workspace_export_zip() -> Response:
        buf = _software_export_zip(state.workspace_root)
        return _zip_response(buf, "invar_software_workspace.zip")

    @app.post("/api/v1/workspace/clear")
    async def api_workspace_clear() -> dict[str, Any]:
        """Safely wipe generated software files, preserving the Research Vault.

        Removes every workspace entry except ``.invar`` (research vault),
        ``.git`` and ``.venv``, then clears compiler scratch under
        ``.invar/scratch`` without touching ``.invar/research_vault``.
        """
        workspace = state.workspace_root
        protected = {".invar", ".git", ".venv"}
        cleared: list[str] = []
        for entry in sorted(workspace.iterdir()):
            if entry.name in protected:
                continue
            try:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                cleared.append(entry.name)
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                logger.warning("Could not remove %s: %s", entry.name, exc)
        scratch_dir = workspace / ".invar" / "scratch"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
        return {
            "status": "success",
            "message": "Workspace codebase cleared successfully.",
            "cleared_items": cleared,
            "preserved": [".invar/research_vault"],
        }

    @app.post("/api/v1/workspace/import-zip")
    async def api_workspace_import_zip(file: Optional[UploadFile] = None,
                                       payload: Optional[ZipImportRequest] = None) -> dict[str, Any]:
        data = b""
        filename = "workspace.zip"
        if file is not None:
            data = await file.read()
            filename = file.filename or filename
        elif payload is not None and payload.content_base64:
            import base64
            try:
                data = base64.b64decode(payload.content_base64, validate=True)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail="invalid base64 zip payload") from exc
            filename = payload.filename or filename
        if not data:
            raise HTTPException(status_code=400, detail="no zip file provided")
        if len(data) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="zip exceeds 100 MB")
        root = state.workspace_root
        imported: list[str] = []
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="uploaded file is not a valid zip") from exc
        with zf:
            for info in zf.infolist()[:5000]:
                name = info.filename.replace("\\", "/")
                if not name or info.is_dir():
                    continue
                parts = Path(name).parts
                if (name.startswith("/") or ".." in parts
                        or re.match(r"^[A-Za-z]:", name)):
                    raise HTTPException(
                        status_code=400,
                        detail=f"unsafe zip member rejected (zip-slip): {info.filename}")
                top = parts[0] if parts else ""
                if top in (".git", ".invar") or top.startswith("."):
                    continue
                if top == "INVAR_BUILD_MANIFEST.json":
                    continue
                if info.file_size > 20 * 1024 * 1024:
                    continue
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info))
                imported.append(name)
        state.last_ingest = state.agent.ingest_workspace() if state.agent else {}
        return {"status": "imported", "filename": filename,
                "files_imported": imported, "file_count": len(imported)}

    def _vault_zip_bytes(run_ids: Optional[list[str]] = None) -> "io.BytesIO":
        from invar_agent.research.vault import ResearchVaultManager
        vault = ResearchVaultManager(state.workspace_root)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            dirs = []
            if run_ids:
                for rid in run_ids:
                    d = vault.run_dir(rid)
                    if d.is_dir():
                        dirs.append(d)
            else:
                dirs = sorted((vault.root / d.name for d in vault.root.iterdir()
                               if d.is_dir()), key=lambda p: p.name) \
                    if vault.root.is_dir() else []
            if not dirs:
                raise HTTPException(status_code=404, detail="no research runs to export")
            for directory in dirs:
                for path in sorted(directory.rglob("*")):
                    if path.is_dir():
                        continue
                    rel = path.relative_to(directory).as_posix()
                    zf.write(path, f"{directory.name}/{rel}")
        buf.seek(0)
        return buf

    @app.get("/api/v1/research/vault/{run_id}/export-zip")
    def api_research_vault_export_zip(run_id: str) -> Response:
        from invar_agent.research.vault import ResearchVaultManager, _safe_run_id
        safe = _safe_run_id(run_id)
        if not ResearchVaultManager(state.workspace_root).run_dir(safe).is_dir():
            raise HTTPException(status_code=404, detail=f"research run not found: {run_id}")
        return _zip_response(_vault_zip_bytes([safe]), f"{safe}.invar-research.zip")

    @app.post("/api/v1/research/vault/upload")
    @app.post("/api/v1/research/vault/import")
    async def api_research_vault_upload(request: Request,
                                        file: Optional[list[UploadFile]] = None,
                                        title: Optional[str] = Query(None),
                                        notes: str = Query("")) -> dict[str, Any]:
        from invar_agent.research.vault import ResearchVaultManager

        payload_files: list[dict[str, Any]] = []
        title_value = title
        notes_value = notes
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            payload_files = list(body.get("files") or [])
            title_value = title_value or body.get("title")
            notes_value = notes_value or str(body.get("notes") or "")
        elif file:
            for upload in file:
                raw = await upload.read()
                import base64
                payload_files.append({"filename": upload.filename or "upload.bin",
                                      "content": base64.b64encode(raw).decode("ascii"),
                                      "encoding": "base64"})
        else:
            # multipart/form-data without the FastAPI `file` binding
            form = await request.form()
            payload_files = [{"filename": k, "content": base64.b64encode(
                await v.read() if hasattr(v, "read") else v.encode()).decode("ascii"),
                "encoding": "base64"} for k, v in form.multi_items() if k != "title"]
            title_value = title_value or str(form.get("title") or "") or None
            notes_value = notes_value or str(form.get("notes") or "")
        if not payload_files:
            raise HTTPException(status_code=400, detail="no research files uploaded")
        try:
            meta = ResearchVaultManager(state.workspace_root).ingest_uploaded_research(
                payload_files, title=title_value, custom_notes=notes_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "success", "run_id": meta.get("run_id"),
                "restored": bool(meta.get("restored")), "metadata": meta}

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

    def _research_vault_context(question: str,
                               attached_ids: Optional[list[str]] = None) -> str:
        """Compacted dual-context (software codebase + research vault) for planners."""
        try:
            from invar_agent.context_compactor import build_research_vault_context
            return build_research_vault_context(
                state.workspace_root, referenced_ids=attached_ids or [],
                user_query=question, agent_mode="research")
        except Exception:  # noqa: BLE001 - context is best-effort
            return ""

    def _seed_blueprint_from_uploads(bp: ResearchBlueprint,
                                     attached_ids: Optional[list[str]]
                                     ) -> tuple[ResearchBlueprint, list[str]]:
        """Pre-seed a blueprint from attached user-uploaded research runs:
        SI constants + structured evidence are merged so Stage 1 verifies the
        uploaded baseline and accretes extensions in Stage 2+."""
        seeded: list[str] = []
        try:
            from invar_agent.research.vault import ResearchVaultManager
            from invar_agent.schemas.research import (
                BenchmarkAnchor, ExtractedEquation, ExtractedParameter,
                StructuredResearchEvidence,
            )
            vault = ResearchVaultManager(state.workspace_root)
            for rid in attached_ids or []:
                detail = vault.get_run(rid)
                if not detail:
                    continue
                meta = detail.get("metadata") or {}
                if meta.get("origin") != "user_upload":
                    continue
                seeded.append(rid)
                constants = dict(getattr(bp, "extracted_constants", None) or {})
                for k, v in (meta.get("extracted_constants") or {}).items():
                    if isinstance(v, (int, float)):
                        constants.setdefault(str(k), float(v))
                updates: Dict[str, Any] = {"extracted_constants": constants}
                se_raw = meta.get("structured_evidence") or {}
                if getattr(bp, "structured_evidence", None) is None and se_raw:
                    try:
                        params = {}
                        for n, p in (se_raw.get("parameters") or {}).items():
                            if p.get("si_value") is None:
                                continue
                            params[n] = ExtractedParameter(
                                name=n, si_value=float(p["si_value"]),
                                si_unit=str(p.get("si_unit") or ""),
                                raw_value=float(p.get("raw_value", p["si_value"])),
                                raw_unit=str(p.get("raw_unit") or ""),
                                source=str(p.get("source") or "user_upload"))
                        eqs = [ExtractedEquation(
                            lhs=str(e.get("lhs") or ""),
                            rhs_latex=str(e.get("rhs_latex") or ""),
                            rhs_python_hint=str(e.get("rhs_python_hint") or ""),
                            variables=list(e.get("variables") or []),
                            source=str(e.get("source") or "user_upload"))
                            for e in (se_raw.get("equations") or [])]
                        benches = [BenchmarkAnchor.model_validate(b)
                                   for b in (se_raw.get("benchmarks") or [])]
                        updates["structured_evidence"] = StructuredResearchEvidence(
                            domains=list(se_raw.get("domains") or []),
                            parameters=params, equations=eqs, benchmarks=benches,
                            recommended_backend=se_raw.get("recommended_backend") or "cpp",
                            recommended_stepper=se_raw.get("recommended_stepper") or "rk4",
                            suggested_invariants=list(se_raw.get("suggested_invariants") or []))
                    except Exception:  # noqa: BLE001 - evidence is best-effort
                        pass
                bp = bp.model_copy(update=updates)
        except Exception:  # noqa: BLE001 - seeding is best-effort
            seeded = []
        return bp, seeded


    @app.post("/api/v1/research/plan")
    async def api_research_plan(req: ResearchPlanRequest) -> dict[str, Any]:
        """Feasibility gate + blueprint synthesis in a SINGLE Tier-2 call.

        Never returns the canned kdv_soliton_scaling stub when an LLM provider
        is configured: with a key, ``fallback=False`` and synthesis errors
        surface as HTTP 502. The stub is restricted to explicit offline
        (no-key) mode.
        """
        manager = state.ensure_research_manager()
        has_llm = bool(state.api_key)
        try:
            outcome = await generate_plan_with_feasibility(
                req.question,
                llm_complete=(
                    (lambda q: _plan_llm_callable(state, q)) if has_llm else None),
                fallback=not has_llm,
                extra_context=_research_vault_context(req.question, req.attached_research_ids),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"research plan synthesis failed: {str(exc)[:300]}")
        feasibility = outcome["feasibility"]
        if feasibility is not None and str(feasibility.verdict.value).upper().startswith("REFUSED"):
            return {"status": "refused", "blueprint": None,
                    "feasibility": feasibility.model_dump()}
        blueprint_dict = outcome["blueprint"]
        if not blueprint_dict:
            raise HTTPException(status_code=502,
                                detail="research plan synthesis returned no blueprint")
        bp = ResearchBlueprint.model_validate(blueprint_dict)
        if feasibility is not None:
            bp = bp.model_copy(update={"feasibility": feasibility})
        bp, seeded = _seed_blueprint_from_uploads(bp, req.attached_research_ids)
        manager.seeded_upload_ids = seeded
        manager.request_plan(bp)
        return {"status": "pending_approval", "blueprint": bp.model_dump(),
                "source": outcome.get("source"),
                "feasibility": feasibility.model_dump() if feasibility else None}

    @app.post("/api/v1/research/plan-prompt")
    async def api_research_plan_prompt(req: ResearchPlanPromptRequest) -> dict[str, Any]:
        """Plan Mode: feasibility + editable Expert Deep Research Prompt.

        One Tier-2 call; refusals come back as a refusal payload, otherwise an
        expert prompt with objective/targets, surrogate physics formulation,
        literature directives and the verification protocol. Never raw
        blueprint JSON. Requires an LLM provider (no offline stub path).
        """
        state.ensure_research_manager()
        # Clear-cut heuristic refusals are answered without an LLM key.
        from invar_agent.research.feasibility import _heuristic_verdict
        heuristic = _heuristic_verdict(req.question)
        if heuristic.verdict.startswith("REFUSED"):
            return {"status": "refused", "feasibility": heuristic.model_dump()}
        if not state.api_key:
            raise HTTPException(
                status_code=400,
                detail="Plan Mode requires an LLM provider: configure an API key first")
        try:
            result = await generate_expert_research_prompt(
                req.question,
                llm_complete=lambda q: _plan_llm_callable(state, q, max_tokens=8192),
                extra_context=_research_vault_context(req.question, req.attached_research_ids))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"expert prompt synthesis failed: {str(exc)[:300]}")
        feasibility = result["feasibility"]
        if str(feasibility.verdict.value).upper().startswith("REFUSED"):
            return {"status": "refused", "feasibility": feasibility.model_dump()}
        return {"status": "ok", "feasibility": feasibility.model_dump(),
                "expert_prompt": result["expert_prompt"],
                "source": result.get("source")}

    @app.post("/api/v1/research/run")
    async def api_research_run(req: ResearchRunRequest) -> dict[str, Any]:
        """Direct Mode: immediately generate the plan and start the full
        verified loop (single combined Tier-2 call), streaming via
        /api/v1/research/stream.
        """
        manager = state.ensure_research_manager()
        has_llm = bool(state.api_key)
        prior_verdict = str((req.prior_feasibility or {}).get("verdict") or "")
        assume_feasible = prior_verdict in ("OPTIMAL_DIRECT", "REFORMULATED_SURROGATE")
        try:
            outcome = await generate_plan_with_feasibility(
                req.question,
                llm_complete=(
                    (lambda q: _plan_llm_callable(state, q)) if has_llm else None),
                fallback=not has_llm,
                assume_feasible=assume_feasible,
                extra_context=_research_vault_context(req.question, req.attached_research_ids),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"research plan synthesis failed: {str(exc)[:300]}")
        feasibility = outcome["feasibility"]
        if feasibility is not None and str(feasibility.verdict.value).upper().startswith("REFUSED"):
            return {"status": "refused", "started": False, "blueprint": None,
                    "feasibility": feasibility.model_dump()}
        blueprint_dict = outcome["blueprint"]
        if not blueprint_dict:
            raise HTTPException(status_code=502,
                                detail="research plan synthesis returned no blueprint")
        bp = ResearchBlueprint.model_validate(blueprint_dict)
        if feasibility is not None:
            bp = bp.model_copy(update={"feasibility": feasibility})
        bp, seeded = _seed_blueprint_from_uploads(bp, req.attached_research_ids)
        manager.seeded_upload_ids = seeded
        manager.request_plan(bp)
        if not manager.approve():
            raise HTTPException(status_code=409, detail="could not start research run")
        return {"status": "running", "started": True, "blueprint_id": bp.blueprint_id,
                "source": outcome.get("source"),
                "feasibility": feasibility.model_dump() if feasibility else None}

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

    @app.get("/api/research/literature/health")
    def api_literature_health() -> dict[str, Any]:
        """Zero-token live literature health check (free public APIs only)."""
        import time as _time
        from invar_agent.research.search import LiteratureSearchClient
        import os
        offline = os.environ.get("INVAR_OFFLINE") == "1" or (
            "PYTEST_CURRENT_TEST" in os.environ
            and os.environ.get("INVAR_RESEARCH_SEARCH_LIVE") != "1")
        client = LiteratureSearchClient(offline=offline)
        started = _time.perf_counter()
        try:
            sources = client.search("discrete gradient", 5)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "offline": client.offline,
                    "results": 0, "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_ms": round((_time.perf_counter() - started) * 1000.0, 2)}
        meta = getattr(client, "last_query_meta", {}) or {}
        return {
            "ok": len(sources) > 0,
            "offline": client.offline,
            "results": len(sources),
            "raw_counts": meta.get("raw_counts") or {},
            "channel_errors": meta.get("channel_errors") or {},
            "elapsed_ms": round((_time.perf_counter() - started) * 1000.0, 2),
        }

    @app.get("/api/v1/research/stream")
    def api_research_stream() -> StreamingResponse:
        manager = state.ensure_research_manager()
        q = state.research_broadcaster.subscribe()
        # Sequence watermark: replayed history (seq <= watermark) must never
        # close the stream; only fresh terminal events do.
        initial_seq = state.research_broadcaster.seq
        terminal_events = ("research_completed", "research_failed")

        def event_generator():
            # Sync generator (same pattern as /api/agent/stream): the broadcaster
            # publishes from the research worker thread into a thread-safe queue.
            try:
                while True:
                    try:
                        payload = q.get(timeout=RESEARCH_SSE_HEARTBEAT)
                    except queue.Empty:
                        # SSE comment heartbeat: keeps browsers/proxies alive
                        # during long Tier-2 LLM re-synthesis calls.
                        yield ": keep-alive\n\n"
                        continue
                    terminal = (payload.get("event") in terminal_events
                                and payload.get("seq", 0) > initial_seq)
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
                    if terminal:
                        # Graceful close after the final event; EventSource
                        # reconnects and replays history without re-closing.
                        break
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
    *,
    pre_events: Optional[list[dict[str, Any]]] = None,
    research_packet: Any = None,
    workspace_root: Optional[Path] = None,
) -> StreamingResponse:
    """Run the synchronous agent in a worker thread and stream progress as SSE."""
    events: "queue.Queue[dict[str, Any]]" = queue.Queue()
    for item in pre_events or []:
        events.put(item)

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
            report_dict = _report_to_dict(report)
            if research_packet is not None and workspace_root is not None:
                from invar_agent.builder.research_bridge import run_research_parity_tests
                events.put({"type": "activity", "stage": "SANDBOX",
                            "step": "research_parity_test",
                            "message": "Executing tests/test_research_parity.py "
                                       "in Sandbox_Execute..."})
                parity = run_research_parity_tests(workspace_root)
                report_dict["research_parity"] = {
                    "passed": parity["passed"], "returncode": parity["returncode"],
                    "run_ids": research_packet.run_ids,
                    "output_tail": parity["output"][-1500:]}
                events.put({"type": "activity", "stage": "SANDBOX",
                            "step": "research_parity_result",
                            "level": "info" if parity["passed"] else "error",
                            "message": ("Research-parity tests PASSED" if parity["passed"]
                                        else "Research-parity tests FAILED"),
                            "passed": parity["passed"]})
            events.put({"type": "done", "report": report_dict})
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

"""Nine-stage builder with isolated module gates and atomic wavefront publication."""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from uuid import uuid4

import blake3
import z3
from pydantic import Field, ValidationError

from invar_agent.llm.errors import redact_secrets
from invar_agent.runtime.build_utils import collect_transients, rollback_overlay

from .circuit_breaker import BuildBreakerBudget, CircuitBreakerController, ProgressMetric, SlotBreakerState, compute_fault_fingerprint
from .error_taxonomy import BuildFault, CircuitAction, ErrorTier, classify_error
from .emitters import default_emitters
from .emitters.base import EmitterContext, EmitterRegistry, ModuleEmission
from .emitters.receipts import seal_receipt
from .fallback_registry import FallbackEmitterRegistry
from .healing_controller import HealingController
from .his_cache import HISCache, HISCacheState, artifact_path, digest, interface_digest, port_signature, target_relpaths
from .ingestion.equation_lowerer import _z3, equation_parts
from .ingestion.pipeline import CanonicalResearchIngestionPipeline, reverify_ir, seal_ir
from .ingestion.schemas import CanonicalResearchIR, IngestionRequest, StrictModel
from .leaf_policy import validate_source_capabilities
from .planner import BuilderPlanner
from .uast_models import BuilderArchitectureSpec, BuilderProfile, BuilderUAST, InterfacePortNodeModel, LLMExtensionSlotNodeModel, LLMExtensionSlotRequest, canonical_json
from .wavefront_scheduler import WavefrontScheduler


from ..runtime.isolation import Z3_GLOBAL_LOCK as _FORMAL_LOCK


def _verify_ir(ir):
    with _FORMAL_LOCK:
        reverify_ir(ir.model_copy(deep=True))


def _port_constraint(condition):
    left, relation, right = equation_parts(condition)
    a, b = _z3(left), _z3(right)
    return {"=": lambda: a == b, "==": lambda: a == b, "<": lambda: a < b, "<=": lambda: a <= b, ">": lambda: a > b, ">=": lambda: a >= b}[relation]()


def _field_bounds(fields):
    result = []
    for item in fields:
        if item.bounds and item.scalar_type in {"f64", "i64"}:
            symbol = z3.Int(item.name) if item.scalar_type == "i64" else z3.Real(item.name)
            if item.bounds.lower is not None:
                result.append(symbol >= item.bounds.lower if item.bounds.lower_inclusive else symbol > item.bounds.lower)
            if item.bounds.upper is not None:
                result.append(symbol <= item.bounds.upper if item.bounds.upper_inclusive else symbol < item.bounds.upper)
    return result


class BuilderStage(str, Enum):
    INGEST = "Ingest"
    GOI_PLAN = "GoI_Plan"
    WAVEFRONT_SCHEDULE = "Wavefront_Schedule"
    OVERLAY_MUTATE = "Overlay_Mutate"
    PREWRITE_GUARD = "PreWrite_Guard"
    SMT_CONCOLIC_VERIFY = "SMT_Concolic_Verify"
    SANDBOX_EXECUTE = "Sandbox_Execute"
    REFLEX_HEAL = "Reflex_Heal"
    ADJOINT_COMMIT_OR_ROLLBACK = "Adjoint_Commit_or_Rollback"


class BuildRequest(StrictModel):
    build_id: str = Field(default_factory=lambda: uuid4().hex)
    ir: CanonicalResearchIR | None = None
    ir_id: str | None = None
    vault_ids: list[str] = Field(default_factory=list)
    uploaded_files: list[str] = Field(default_factory=list)
    build_goal: str = ""
    explicit_profiles: list[BuilderProfile] | None = None
    leaf_slots: list[LLMExtensionSlotRequest] = Field(default_factory=list)
    architecture_spec: BuilderArchitectureSpec | None = None
    uast: BuilderUAST | None = None
    auto_approve_clean_ir: bool = False


class BuildRunState(StrictModel):
    build_id: str
    workspace: str
    status: str = "RUNNING"
    stage: str = "Ingest"
    error_code: str | None = None
    error: str | None = None
    committed_waves: list[list[str]] = Field(default_factory=list)
    python_fallback_active: bool = False
    cabi_compiled: bool = False
    unloaded_handles: int = 0
    pytest_exit_code: int = 5
    pytest_passed: int = 0
    pytest_failed: int = 0
    fallbacks: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    receipt: dict = Field(default_factory=dict)
    total_llm_tokens_used: int = 0


def _link_directory(link_path: Path, target_dir: Path) -> None:
    """Create a directory symlink, falling back to an NTFS junction on Windows without SeCreateSymbolicLinkPrivilege."""
    try:
        link_path.symlink_to(target_dir, target_is_directory=True)
    except OSError:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(Path(target_dir).resolve()), str(link_path))
        else:
            raise


def _unlink_directory_link(link_path: Path) -> None:
    """Remove a directory symlink or Windows NTFS junction without touching its target."""
    try:
        link_path.unlink(missing_ok=True)
    except (PermissionError, IsADirectoryError, OSError):
        if os.name == "nt":
            try:
                os.rmdir(link_path)
            except OSError:
                pass


def _replace_directory_link(src_link: Path, dst_link: Path) -> None:
    """Atomically replace directory symlink/junction `dst_link` with `src_link`."""
    try:
        os.replace(src_link, dst_link)
    except (PermissionError, OSError):
        if os.name == "nt":
            _unlink_directory_link(dst_link)
            os.replace(src_link, dst_link)
        else:
            raise


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            output.write(canonical_json(payload))
            output.flush()
            os.fsync(output.fileno())
        for _attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if path.exists():
                    try:
                        os.chmod(path, 0o666)
                    except OSError:
                        pass
                if _attempt == 5:
                    raise
                time.sleep(0.02 * (2 ** _attempt))
    finally:
        temporary.unlink(missing_ok=True)


class BuilderStateMachine:
    def __init__(self, workspace_root: Path, *, emitters: EmitterRegistry | None = None, llm_client=None,
                 progress_cb=None, ir_loader=None, budget: BuildBreakerBudget | None = None):
        self.workspace_root = Path(workspace_root).resolve()
        self.emitters = emitters or default_emitters()
        self.healer = HealingController(llm_client)
        self.planner = BuilderPlanner(llm_client)
        self.fallback_registry = FallbackEmitterRegistry()
        self.breaker = CircuitBreakerController()
        self.budget = budget or BuildBreakerBudget()
        self.progress_cb, self.ir_loader = progress_cb, ir_loader
        self._llm_lock = asyncio.Lock()
        self._telemetry_lock = threading.Lock()
        self._running = False

    async def _event(self, stage, *, module_id=None, channel="log", event="builder_stage", **data):
        self.state.stage = stage.value
        payload = dict(event_version="2.0", mode="builder", build_id=self.state.build_id,
            event=event, stage=stage.value, module_id=module_id, seq=len(self.state.events) + 1, **data)
        self.state.events.append(dict(channel=channel, **payload))
        if self.progress_cb:
            pending = self.progress_cb(channel, payload)
            if inspect.isawaitable(pending):
                await pending

    async def run(self, request: BuildRequest) -> BuildRunState:
        if request.ir is not None:
            from .research_bridge import validate_research_ir
            validate_research_ir(request.ir)
        if self._running:
            raise RuntimeError("A BuilderStateMachine executes one build at a time")
        self._running = True
        started = time.monotonic()
        cache = HISCache(self.workspace_root, request.build_id)
        self.root = cache.path.parent
        self.root.mkdir(parents=True, exist_ok=True)
        self.healer.diagnostics_dir = self.root / "diagnostics" / "leaf_responses"
        self.workspace = self.root / "workspace"
        self.state = BuildRunState(build_id=request.build_id, workspace=str(self.workspace))
        self.ir = self.uast = None
        self._slot_states = {}
        self._started = started
        self._plan = None
        self._goi_rho = 0.0
        try:
            await self._event(BuilderStage.INGEST)
            ir = request.ir.model_copy(deep=True) if request.ir is not None else self.ir_loader(request.ir_id) if request.ir_id and self.ir_loader else await asyncio.to_thread(
                CanonicalResearchIngestionPipeline(self.workspace_root).ingest,
                IngestionRequest(upload_paths=request.uploaded_files, vault_run_ids=request.vault_ids))
            if seal_ir(ir.model_copy(deep=True)).ir_blake3_digest != ir.ir_blake3_digest:
                raise BuildFault("IR_DIGEST_MISMATCH", "Research IR changed after sealing")
            _verify_ir(ir)
            if request.auto_approve_clean_ir and not ir.disambiguation_flags:
                ir.user_approved = True
                seal_ir(ir)
            if not ir.user_approved or any(not flag.resolved for flag in ir.disambiguation_flags):
                raise BuildFault("IR_UNVERIFIED_FLAGGED", "Researcher approval is required")
            self.ir = ir
            ir_file = self.root / "canonical_research_ir.json"
            ir_json = ir.model_dump(mode="json")
            if ir_file.exists() and json.loads(ir_file.read_text(encoding="utf-8")) != ir_json:
                raise BuildFault("IR_IMMUTABLE_CONFLICT", "Use a new build ID for a different research IR")
            if not ir_file.exists():
                _atomic_json(ir_file, ir_json)
            await self._event(BuilderStage.GOI_PLAN, channel="goi_stream")
            if request.uast is not None:
                self.uast = BuilderUAST.model_validate(request.uast.model_dump(mode="json"))
                if self.uast.canonical_ir_digest != ir.ir_blake3_digest:
                    raise BuildFault("IR_DIGEST_MISMATCH", "UAST belongs to another research IR")
            else:
                spec = request.architecture_spec
                if spec is None:
                    for attempt in range(2):
                        try:
                            spec = await asyncio.to_thread(self.planner.translate_intent, ir, request.build_goal, profiles=request.explicit_profiles)
                            break
                        except ValidationError:
                            if attempt == 0:
                                continue
                        except Exception:
                            break
                    if spec is None:
                        spec = BuilderArchitectureSpec(project_name="scientific_instrument", profiles=[BuilderProfile.HYBRID_COMPUTE_LIBRARY, BuilderProfile.EXECUTABLE_PAPER_AUDIT])
                if request.leaf_slots:
                    spec = spec.model_copy(deep=True, update={"llm_extension_slots": request.leaf_slots})
                self.uast = self.planner.compile(ir, spec)
            for node in self.uast.nodes:
                if node.kind == "interface_port" and interface_digest(port_signature(node)) != node.his_digest:
                    raise BuildFault("HIS_CONTRACT_DRIFT", "Unsealed public port: " + node.node_id)
            await self._event(BuilderStage.WAVEFRONT_SCHEDULE, channel="goi_stream")
            previous = HISCache(self.workspace_root, request.build_id, self.workspace).load() if self.workspace.exists() else None
            plan = WavefrontScheduler(self.workspace if self.workspace.exists() else None).plan(self.uast, previous)
            self._plan = plan
            prior_receipt = self.root / 'build_receipt.json'
            if previous and prior_receipt.is_file():
                prior = json.loads(prior_receipt.read_text(encoding="utf-8"))
                if seal_receipt(prior).merkle_root_blake3 == prior.get('merkle_root_blake3') and prior['canonical_ir_digest'] == ir.ir_blake3_digest:
                    self.state.fallbacks = [record.copy() for record in prior['fallbacks'] if record['module_id'] in plan.cached_nodes]
                    if 'mod_sdk' in plan.cached_nodes:
                        self.state.python_fallback_active = prior['verification']['python_fallback_active']
                        self.state.cabi_compiled = prior['verification']['cabi_compiled']
                    if len(plan.cached_nodes) == len(self.uast.nodes):
                        self.state.pytest_exit_code = prior['verification']['pytest_exit_code']
                        self.state.pytest_passed = prior['verification']['pytest_passed']
                        self.state.pytest_failed = prior['verification']['pytest_failed']
            if not self.workspace.exists():
                initial = self.root / "generations" / ("i_" + uuid4().hex[:8])
                initial.mkdir(parents=True)
                _link_directory(self.workspace, initial)
            cache = HISCache(self.workspace_root, request.build_id, self.workspace)
            checkpoint = cache.new_state(self.uast, plan)
            if previous:
                checkpoint.entries = {name: previous.entries[name] for name in plan.cached_nodes}
            self.state.python_fallback_active = not bool(shutil.which("g++") or shutil.which("clang++"))
            nodes = {node.node_id: node for node in self.uast.nodes}
            for wave in plan.wavefronts:
                active = [nodes[name] for name in wave if name in plan.rebuild_nodes]
                for name in wave:
                    if name in plan.cached_nodes:
                        await self._event(BuilderStage.WAVEFRONT_SCHEDULE, module_id=name, channel="goi_stream", event="CACHE_HIT_SKIPPED")
                outcomes = await asyncio.gather(*(self._module(node) for node in active), return_exceptions=True)
                failure = next((outcome for outcome in outcomes if isinstance(outcome, BaseException)), None)
                if failure:
                    for node in active:
                        self._rollback_module(node.node_id)
                    raise failure
                await self._event(BuilderStage.ADJOINT_COMMIT_OR_ROLLBACK, channel="goi_stream", wavefront=wave)
                if active:
                    checkpoint = await asyncio.to_thread(self._commit, active, outcomes, checkpoint)
                self.state.committed_waves.append(wave)
            self.state.status = "COMPLETED_WITH_FALLBACKS" if self.state.fallbacks or self.state.python_fallback_active else "COMPLETED"
        except Exception as exc:
            self.state.status = "ABORTED_CRITICAL"
            self.state.error_code = classify_error(exc)[0]
            self.state.error = redact_secrets(str(exc))[:6000]
            if isinstance(getattr(exc, "details", None), dict):
                self._goi_rho = float(exc.details.get("spectral_radius", 0.0))
            await self._event(BuilderStage.ADJOINT_COMMIT_OR_ROLLBACK, event="builder_aborted", error_code=self.state.error_code, error=self.state.error,
                              details=getattr(exc, "details", {}))
        finally:
            self.state.total_llm_tokens_used = self.budget.used_repair_tokens + self.planner.llm_tokens_used
            self._running = False
            self._receipt()
        await self._event(BuilderStage.ADJOINT_COMMIT_OR_ROLLBACK, event="builder_completed" if self.state.status != "ABORTED_CRITICAL" else "builder_failed", status=self.state.status)
        return self.state

    def _port(self, node):
        identifier = node.frozen_port_id if node.kind == "llm_extension_slot" else node.output_port_id if node.kind == "kernel" else node.provided_port_ids[0] if node.kind == "deterministic_module" and node.provided_port_ids else None
        return next((port for port in self.uast.nodes if port.node_id == identifier), None)

    def _rollback_module(self, module_id):
        for parent in (self.root / "overlays").glob("round_*"):
            rollback_overlay(parent / module_id)
        rollback_overlay(self.root / "overlays" / "fallback" / module_id)

    async def _module(self, node):
        slot = node.kind == "llm_extension_slot"
        port = self._port(node)
        breaker_state = self._slot_states.setdefault(node.node_id, SlotBreakerState(node.node_id))
        round_index, fallback, patched, emission, reuse_emission = 1, False, None, None, False
        while True:
            overlay = self.root / "overlays" / ("fallback" if fallback else f"round_{round_index}") / node.node_id
            rollback_overlay(overlay)
            overlay.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.workspace.resolve(), overlay)
            stage = BuilderStage.OVERLAY_MUTATE
            try:
                await self._event(stage, module_id=node.node_id, round_index=round_index)
                if fallback:
                    emission = ModuleEmission({node.target_relpath: self.fallback_registry.emit(node.fallback_emitter_id, port, self.ir)})
                elif patched is not None:
                    emission.files[primary] = patched
                elif reuse_emission:
                    pass
                elif slot and "slot" not in self.emitters.emitters:
                    async with self._llm_lock:
                        result = await asyncio.to_thread(self.healer.tier2, "", node, port, self.uast, node.intent_summary, self.budget)
                    emission = ModuleEmission({node.target_relpath: result.source})
                else:
                    emission = await asyncio.to_thread(self.emitters.emit, node, EmitterContext(self.ir.model_copy(deep=True), self.uast.model_copy(deep=True), overlay, self.state.python_fallback_active))
                if emission.python_fallback_active:
                    self.state.python_fallback_active = True
                for relpath, content in emission.files.items():
                    path = artifact_path(overlay, relpath)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content.encode() if isinstance(content, str) else content)
                stage = BuilderStage.PREWRITE_GUARD
                await self._event(stage, module_id=node.node_id)
                await asyncio.to_thread(self._guard, node, overlay, emission)
                stage = BuilderStage.SMT_CONCOLIC_VERIFY
                await self._event(stage, module_id=node.node_id)
                await asyncio.to_thread(self._formal, node)
                if port and (node.target_relpath.endswith(".py") if slot else emission.port_file is not None):
                    await asyncio.to_thread(self._sandbox, node, overlay, emission, True)
                stage = BuilderStage.SANDBOX_EXECUTE
                await self._event(stage, module_id=node.node_id, channel="accelerator_log")
                await asyncio.to_thread(self._sandbox, node, overlay, emission, False)
                if fallback:
                    next(record for record in self.state.fallbacks if record["module_id"] == node.node_id)["fallback_verified"] = True
                return overlay, emission, fallback
            except Exception as exc:
                if fallback:
                    rollback_overlay(overlay)
                    raise BuildFault("DETERMINISTIC_EMITTER_BUG", "Registered fallback failed verification: " + str(exc)) from exc
                code, tier = classify_error(exc, node.kind)
                if code == "CPP_COMPILE_DIAGNOSTIC" and breaker_state.tier1_attempts:
                    tier = ErrorTier.CRITICAL_NON_RETRYABLE
                if not slot and code in {"SLOT_RUNTIME_LOGIC_ERROR", "SLOT_PYTEST_ASSERTION"}:
                    code, tier = "DETERMINISTIC_EMITTER_BUG", ErrorTier.CRITICAL_NON_RETRYABLE
                fingerprint = compute_fault_fingerprint(node.node_id, stage.value, code, type(exc).__name__, node.target_relpath if slot else node.node_id, str(exc))
                metric = ProgressMetric(has_critical=int(tier == ErrorTier.CRITICAL_NON_RETRYABLE), hard_error_count=int(code != "SLOT_PYTEST_ASSERTION"),
                    hin_violation_count=int(code.startswith("HIN_")), z3_violation_count=int(code.startswith("Z3_")),
                    failed_test_count=getattr(exc, "details", {}).get("failed_test_count", 0), warning_count=0)
                primary = node.target_relpath if slot else next((path for path in emission.files if path.endswith(".py") and not Path(path).name.startswith("test_")), None) if emission else None
                source = (overlay / primary).read_text(encoding="utf-8") if primary and (overlay / primary).exists() else ""
                try:
                    syntax_digest = ast.dump(ast.parse(source), include_attributes=False)
                except SyntaxError:
                    syntax_digest = source
                self.budget.elapsed_sec = time.monotonic() - self._started
                await self._event(BuilderStage.REFLEX_HEAL, module_id=node.node_id, error_code=code, details=getattr(exc, "details", {}))
                decision = self.breaker.decide(module_id=node.node_id, node_kind=node.kind, round_index=round_index, tier=tier,
                    fingerprint=fingerprint, metric=metric, candidate_ast_digest=digest("invar.his.body.v2", syntax_digest),
                    slot_state=breaker_state, budget=self.budget, fallback_emitter_id=node.fallback_emitter_id if slot else None,
                    max_tier2_per_slot=node.max_tier2_retries if slot else None,
                    missing_compiler_fallback=code in {"TOOLCHAIN_MISSING_CXX", "TOOLCHAIN_MISSING_CARGO"} or code == "CPP_COMPILE_DIAGNOSTIC" and bool(breaker_state.tier1_attempts))
                await self._event(BuilderStage.REFLEX_HEAL, module_id=node.node_id, event="healing_decision", **decision.model_dump(mode="json", exclude={"module_id"}))
                if decision.action == CircuitAction.ABORT_BUILD_CRITICAL:
                    raise BuildFault(code, decision.reason_code, original_error=redact_secrets(str(exc))) from exc
                if decision.action == CircuitAction.SWITCH_TO_PYTHON_FALLBACK:
                    self.state.python_fallback_active = True
                    round_index += 1
                    continue
                if decision.action == CircuitAction.SUBSTITUTE_DETERMINISTIC_FALLBACK:
                    await self._event(BuilderStage.ADJOINT_COMMIT_OR_ROLLBACK, module_id=node.node_id, event="DEGRADED_DETERMINISTIC_FALLBACK", reason=decision.reason_code)
                    self._rollback_module(node.node_id)
                    self.state.fallbacks.append(dict(module_id=node.node_id, fallback_emitter_id=node.fallback_emitter_id,
                        frozen_his_digest=node.frozen_his_digest, trip_reason=decision.reason_code,
                        fault_fingerprint_digest=fingerprint.blake3_digest, tier2_attempts_used=breaker_state.tier2_attempts, fallback_verified=False))
                    fallback = True
                    continue
                try:
                    if decision.action == CircuitAction.APPLY_TIER0:
                        patched = self.healer.tier0(source, node.kind).source
                    elif decision.action == CircuitAction.APPLY_TIER1 and slot:
                        self.healer.causal_slice(self.uast, node.node_id)
                        patched = self.healer.tier1(source, port, error_code=code, details=getattr(exc, "details", {})).source
                    elif decision.action == CircuitAction.APPLY_TIER1 and code == "CPP_COMPILE_DIAGNOSTIC":
                        for path in emission.files:
                            if path.endswith(".cpp"):
                                text = emission.files[path]
                                headers = [header for header, token in (("vector", "std::vector"), ("cmath", "std::"), ("cstdint", "int64_t")) if token in text and f"#include <{header}>" not in text]
                                emission.files[path] = "".join(f"#include <{header}>\n" for header in headers) + text
                        reuse_emission = True
                    elif slot:
                        async with self._llm_lock:
                            patched = (await asyncio.to_thread(self.healer.tier2, source, node, port, self.uast, fingerprint.normalized_message, self.budget)).source
                    else:
                        raise BuildFault("DETERMINISTIC_EMITTER_BUG", "No deterministic structural repair applies")
                except Exception as repair_error:
                    repair_code, repair_tier = classify_error(repair_error, node.kind)
                    if slot and repair_tier == ErrorTier.CRITICAL_NON_RETRYABLE and repair_code in {"LLM_AUTH_ERROR", "LLM_QUOTA_EXHAUSTED", "LLM_RATE_LIMIT_EXHAUSTED"}:
                        self.state.fallbacks.append(dict(module_id=node.node_id, fallback_emitter_id=node.fallback_emitter_id, frozen_his_digest=node.frozen_his_digest,
                            trip_reason="CRITICAL_SLOT_TRIP:" + repair_code, fault_fingerprint_digest=fingerprint.blake3_digest, tier2_attempts_used=breaker_state.tier2_attempts, fallback_verified=False))
                        self._rollback_module(node.node_id)
                        fallback = True
                    elif slot:
                        patched = source
                    else:
                        raise
                round_index += 1

    def _guard(self, node, overlay, emission):
        if not set(target_relpaths(node)) <= emission.files.keys():
            raise BuildFault("DETERMINISTIC_EMITTER_BUG", "Emitter omitted declared artifacts")
        if node.kind == "llm_extension_slot" and set(emission.files) != {node.target_relpath}:
            raise BuildFault("SANDBOX_SECURITY_VIOLATION", "A leaf can only write its declared target")
        if node.kind == "deterministic_module" and set(emission.files) != set(node.target_relpaths):
            raise BuildFault("DETERMINISTIC_EMITTER_BUG", "Emitter wrote undeclared artifacts")
        for relpath in emission.pytest_paths:
            if not artifact_path(overlay, relpath).is_file():
                raise BuildFault("PYTEST_INTERNAL_OR_USAGE", "Missing declared pytest file")
        for relpath in emission.files:
            if not relpath.endswith(".py"):
                continue
            path = artifact_path(overlay, relpath)
            source = self.healer.tier0(path.read_text(encoding="utf-8"), node.kind if node.kind != "interface_port" else "module").source
            tree = ast.parse(source)
            validate_source_capabilities(tree, leaf=node.kind == "llm_extension_slot", relpath=relpath)
            path.write_text(source, encoding="utf-8")
        if node.kind == "llm_extension_slot" and node.target_relpath.endswith(".py"):
            tree = ast.parse((overlay / node.target_relpath).read_text(encoding="utf-8"))
            port = self._port(node)
            function = next((item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == port.port_name), None)
            if function is None or [arg.arg for arg in function.args.args] != [field.name for field in port.inputs] or function.args.vararg or function.args.kwarg or function.args.kwonlyargs or function.args.posonlyargs:
                raise BuildFault("HIS_CONTRACT_DRIFT", "Frozen function signature drift")

    def _formal(self, node):
        from ..runtime.isolation import run_on_z3_thread
        run_on_z3_thread(self._formal_locked, node)

    def _formal_locked(self, node):
        _verify_ir(self.ir)
        identifiers = {node.node_id} if node.kind == "interface_port" else {node.frozen_port_id} if node.kind == "llm_extension_slot" else {node.output_port_id} if node.kind == "kernel" else set(node.required_port_ids + node.provided_port_ids)
        for port in (port for port in self.uast.nodes if isinstance(port, InterfacePortNodeModel) and port.node_id in identifiers):
            solver = z3.Solver()
            solver.set(timeout=2000)
            solver.add(*_field_bounds(port.inputs + port.outputs))
            for index, condition in enumerate(port.z3_preconditions + port.z3_postconditions):
                solver.assert_and_track(_port_constraint(condition), f"port:{port.node_id}:{index}")
            if solver.check() == z3.unsat:
                raise BuildFault("Z3_SPEC_UNSAT", "Unsatisfiable port contract", unsat_core=[str(label) for label in solver.unsat_core()])
            if solver.check() == z3.unknown:
                raise BuildFault("Z3_SPEC_UNKNOWN", solver.reason_unknown())
        nodes = {item.node_id: item for item in self.uast.nodes}
        for edge in self.uast.edges:
            if edge.to_node != node.node_id:
                continue
            source = nodes[edge.from_node]
            provided = [source.node_id] if source.kind == "interface_port" else [source.output_port_id] if source.kind == "kernel" else [source.frozen_port_id] if source.kind == "llm_extension_slot" else source.provided_port_ids or source.required_port_ids
            for source_id in provided:
                upstream = nodes[source_id]
                for target_id in identifiers:
                    downstream = nodes[target_id]
                    by_name = {field.name: field for field in downstream.inputs}
                    for field in upstream.outputs:
                        required = by_name.get(field.name)
                        if required is None:
                            continue
                        if field.dimension.exponents != required.dimension.exponents:
                            raise BuildFault("HIN_SI_DIM_MISMATCH", "Cross-port dimension mismatch: " + field.name)
                        a, b = field.dimension.semantic_tag, required.dimension.semantic_tag
                        if a and b and a != b:
                            raise BuildFault("HIN_SEMANTIC_TAG_MISMATCH", "Cross-port semantic mismatch: " + field.name)
                    if port_signature(upstream) != port_signature(downstream):
                        proof = z3.Solver()
                        proof.set(timeout=2000)
                        proof.add(*_field_bounds(upstream.inputs + upstream.outputs))
                        proof.add(*[_port_constraint(value) for value in upstream.z3_preconditions + upstream.z3_postconditions])
                        requirements = _field_bounds(downstream.inputs) + [_port_constraint(value) for value in downstream.z3_preconditions]
                        proof.add(z3.Not(z3.And(*requirements)))
                        outcome = proof.check()
                        if outcome == z3.sat:
                            raise BuildFault("Z3_SPEC_UNSAT", "Cross-port precondition is not implied", counterexample=str(proof.model()), edge=edge.model_dump(mode="json"))
                        if outcome == z3.unknown:
                            raise BuildFault("Z3_SPEC_UNKNOWN", proof.reason_unknown())

    def _sandbox(self, node, overlay, emission, concolic):
        scratch = overlay / ".invar_native"
        scratch.mkdir(exist_ok=True)
        try:
            cpp = [str(artifact_path(overlay, path)) for path in emission.files if path.endswith(".cpp")]
            libraries = []
            if cpp and not concolic and not self.state.python_fallback_active:
                compiler = shutil.which("g++") or shutil.which("clang++")
                if compiler is None:
                    raise BuildFault("TOOLCHAIN_MISSING_CXX", "No C++20 compiler available")
                library = scratch / "module.so"
                compiled = subprocess.run([compiler, "-O3", "-std=c++20", "-ffp-contract=off", "-shared", "-fPIC", *cpp, "-o", str(library)],
                    cwd=overlay, capture_output=True, text=True, timeout=60)
                if compiled.returncode:
                    raise BuildFault("CPP_COMPILE_DIAGNOSTIC", compiled.stderr[-4000:])
                libraries.append(str(library))
                self.state.cabi_compiled = True
            port = self._port(node)
            cases = []
            if port:
                defaults = {item.symbol: getattr(item, "si_value", getattr(item, "initial_value", None)) for item in [*self.ir.constants, *self.ir.state_variables]}
                case = {}
                for item in port.inputs:
                    value = defaults.get(item.name, 0.0)
                    case[item.name] = value if value is not None else 0.0
                cases.append(case)
                for item in port.inputs:
                    if item.bounds and item.scalar_type in {"f64", "i64"}:
                        for value, inclusive, direction in ((item.bounds.lower, item.bounds.lower_inclusive, math.inf), (item.bounds.upper, item.bounds.upper_inclusive, -math.inf)):
                            if value is not None:
                                cases.append({**case, item.name: value if inclusive else math.nextafter(value, direction)})
            (overlay / ".sandbox_tmp").mkdir(exist_ok=True)
            environment = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR") if key in os.environ}
            sandbox_tmp_str = str(overlay / ".sandbox_tmp")
            environment.update(HOME=str(overlay), TMPDIR=sandbox_tmp_str, TMP=sandbox_tmp_str, TEMP=sandbox_tmp_str,
                PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                PYTHONPATH=os.pathsep.join([str(overlay), str(overlay / "src")]), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            from ..runtime.isolation import attach_os_limits

            environment["PYTHONUTF8"] = "1"
            config = dict(root=str(overlay), primary=node.target_relpath if node.kind == "llm_extension_slot" else emission.port_file, port=port.model_dump(mode="json") if port else None,
                cases=cases, imports=[path for path in emission.files if path.endswith(".py") and not Path(path).name.startswith("test_")],
                pytest_paths=[] if concolic else emission.pytest_paths, libraries=libraries)
            proc = subprocess.Popen(
                [sys.executable, "-I", "-B", str(Path(__file__).with_name("_sandbox.py"))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=overlay,
                env=environment,
            )
            with attach_os_limits(proc, memory_limit_mb=1024, cpu_time_limit_s=60, active_process_limit=4):
                try:
                    stdout_text, stderr_text = proc.communicate(canonical_json(config), timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    raise BuildFault("OS_OOM_OR_DISK", "Sandbox execution timed out")
            markers = [line[len("INVAR_SANDBOX="):] for line in (stdout_text or "").splitlines() if line.startswith("INVAR_SANDBOX=")]
            if proc.returncode or not markers:
                raise BuildFault("OS_OOM_OR_DISK" if proc.returncode == -9 else "SLOT_RUNTIME_LOGIC_ERROR", (stderr_text or "")[-4000:] or "Sandbox produced no result")
            result = json.loads(markers[-1])
            with self._telemetry_lock:
                self.state.unloaded_handles += result["unloaded_handles"]
                if result["pytest_ran"]:
                    self.state.pytest_exit_code = result["exit_code"]
                    self.state.pytest_passed += result["passed_test_count"]
                    self.state.pytest_failed += result["failed_test_count"]
            if result["exit_code"]:
                code = result.get("error_code", "SLOT_RUNTIME_LOGIC_ERROR")
                if node.kind != "llm_extension_slot" and code in {"SLOT_PYTEST_ASSERTION", "SLOT_RUNTIME_LOGIC_ERROR"}:
                    code = "DETERMINISTIC_EMITTER_BUG"
                raise BuildFault(code, result.get("message", "Sandbox failed"), failed_test_count=result.get("failed_test_count", 0), traceback=result.get("traceback", ""))
        finally:
            collect_transients(scratch)
            collect_transients(overlay / ".sandbox_tmp")

    def _commit(self, nodes, outcomes, checkpoint):
        candidate = self.root / "generations" / ("w_" + uuid4().hex[:8])
        shutil.copytree(self.workspace.resolve(), candidate)
        pending = self.root / (".p_" + uuid4().hex[:8])
        rollback = self.root / (".r_" + uuid4().hex[:8])
        old = self.workspace.resolve()
        promoted = False
        try:
            ownership = {}
            committed_owners = {path: identifier for identifier, entry in checkpoint.entries.items() for path in entry.artifact_digests}
            for node, (overlay, emission, fallback) in zip(nodes, outcomes):
                for relpath in emission.files:
                    if relpath in ownership:
                        raise BuildFault("DETERMINISTIC_EMITTER_BUG", "Duplicate wavefront target: " + relpath)
                    if relpath in committed_owners and committed_owners[relpath] != node.node_id:
                        raise BuildFault("DETERMINISTIC_EMITTER_BUG", "Attempted overwrite of a pinned peer artifact: " + relpath)
                    ownership[relpath] = node.node_id
                    target = artifact_path(candidate, relpath)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".tmp")
                    shutil.copyfile(artifact_path(overlay, relpath), temporary)
                    os.replace(temporary, target)
            updated = checkpoint.model_copy(deep=True, update={"artifact_root": str(candidate)})
            cache = HISCache(self.workspace_root, self.state.build_id, candidate)
            for node, (_, emission, fallback) in zip(nodes, outcomes):
                cache.record_verified(updated, self.uast, node.node_id, list(emission.files), fallback=fallback)
            _link_directory(pending, candidate)
            _link_directory(rollback, old)
            _replace_directory_link(pending, self.workspace)
            promoted = True
            cache.save(updated)
            for node in nodes:
                self._rollback_module(node.node_id)
            return updated
        except BaseException:
            if promoted:
                _replace_directory_link(rollback, self.workspace)
            rollback_overlay(candidate)
            for node in nodes:
                self._rollback_module(node.node_id)
            raise
        finally:
            _unlink_directory_link(pending)
            _unlink_directory_link(rollback)

    def _receipt(self):
        artifacts = []
        entries = {}
        if self.workspace.exists() and self.workspace.resolve().is_relative_to(self.root.resolve()):
            try:
                cache = HISCache(self.workspace_root, self.state.build_id, self.workspace).load()
                entries = cache.entries if cache else {}
            except ValueError:
                pass
            owners = {path: entry for entry in entries.values() for path in entry.artifact_digests}
            nodes = {node.node_id: node for node in self.uast.nodes} if self.uast else {}
            for path in sorted(self.workspace.rglob("*")):
                if path.is_file() and path.relative_to(self.workspace).as_posix() != 'provenance/build_receipt.json':
                    relpath = path.relative_to(self.workspace).as_posix()
                    entry = owners.get(relpath)
                    node = nodes.get(entry.module_id) if entry else None
                    emitted_by = "deterministic_fallback" if entry and entry.verified_status == "FALLBACK_VERIFIED" else "llm_extension_slot" if node and node.kind == "llm_extension_slot" else "deterministic_emitter"
                    artifacts.append(dict(relpath=relpath, module_id=entry.module_id if entry else "unattributed", his_digest=entry.his_digest if entry else "0" * 64,
                        emitted_by=emitted_by, content_blake3=blake3.blake3(path.read_bytes()).hexdigest(), byte_size=path.stat().st_size))
        receipt = dict(schema_version="build_receipt.v1", build_id=self.state.build_id, status=self.state.status,
            project_id=self.uast.project_id if self.uast else "unplanned", canonical_ir_digest=self.ir.ir_blake3_digest if self.ir else "0" * 64,
            uast_digest=digest("invar.builder.uast.v1", self.uast.model_dump(mode="json")) if self.uast else "0" * 64,
            equation_provenance_map={equation.equation_id: digest("invar.source_span.v1", equation.provenance.model_dump(mode="json")) for equation in self.ir.equations} if self.ir else {},
            artifacts=artifacts, fallbacks=self.state.fallbacks,
            verification=dict(python_fallback_active=self.state.python_fallback_active, cabi_compiled=self.state.cabi_compiled,
                goi_spectral_radius=self._plan.spectral_radius if self._plan else self._goi_rho,
                hin_homogeneous_equations=sum(e.hin_homogeneous for e in self.ir.equations) if self.ir else 0,
                hin_total_equations=len(self.ir.equations) if self.ir else 0,
                z3_contracts_proved=sum(e.z3_satisfiable is True for e in self.ir.equations) if self.ir else 0,
                z3_contracts_failed=int(self.state.error_code == "Z3_SPEC_UNSAT"),
                pytest_exit_code=self.state.pytest_exit_code, pytest_passed=self.state.pytest_passed, pytest_failed=self.state.pytest_failed),
            total_llm_tokens_used=self.state.total_llm_tokens_used)
        receipt = seal_receipt(receipt).model_dump(mode='json')
        self.state.receipt = receipt
        _atomic_json(self.root / ("partial_receipt.json" if self.state.status == "ABORTED_CRITICAL" else "build_receipt.json"), receipt)
        if self.state.status != 'ABORTED_CRITICAL' and (self.workspace / 'provenance/build_receipt.json').is_file():
            _atomic_json(self.workspace / 'provenance/build_receipt.json', receipt)
            cache_store = HISCache(self.workspace_root, self.state.build_id, self.workspace)
            checkpoint = cache_store.load()
            if checkpoint and 'mod_receipt' in checkpoint.entries:
                checkpoint.entries['mod_receipt'].artifact_digests['provenance/build_receipt.json'] = blake3.blake3((self.workspace / 'provenance/build_receipt.json').read_bytes()).hexdigest()
                cache_store.save(checkpoint)
        if self.state.status == "ABORTED_CRITICAL":
            _atomic_json(self.root / "diagnostic_report.json", dict(error_code=self.state.error_code, error=self.state.error,
                committed_waves=self.state.committed_waves, events=self.state.events))

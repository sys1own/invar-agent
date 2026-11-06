"""VerifiedAutonomousAgent: the 9-stage mathematically verified agent kernel."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from invar_agent.config import Tier
from invar_agent.contracts import (
    AgentStateStage,
    ASTSurgicalPatch,
    HealingVerdict,
    ShieldVerificationReport,
    TaskWavefrontNode,
)
from invar_agent.overlay.manager import OverlayTransactionManager
from invar_agent.sandbox.manager import HermeticSandboxManager, SandboxExecutionResult
from invar_agent.verification.shield import PrecisionShieldOrchestrator

try:
    from invar_agent._native import PyHisRegister
except Exception:  # pragma: no cover
    PyHisRegister = None  # type: ignore


@dataclass
class NodeExecutionResult:
    node_id: str
    success: bool
    committed: bool = False
    rolled_back: bool = False
    healed_by_tier: str = ""
    shield_report: Optional[ShieldVerificationReport] = None
    sandbox_result: Optional[SandboxExecutionResult] = None
    healing_verdict: Optional[HealingVerdict] = None
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class AgentRunReport:
    run_id: str
    objective: str
    nodes_total: int = 0
    nodes_committed: int = 0
    nodes_rolled_back: int = 0
    nodes_healed: int = 0
    results: list[NodeExecutionResult] = field(default_factory=list)
    boundary_digest_before: str = ""
    boundary_digest_after: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def success(self) -> bool:
        return self.nodes_committed == self.nodes_total and self.nodes_total > 0

    @property
    def duration_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000.0 if self.finished_at else 0.0


class VerifiedAutonomousAgent:
    """Drives the 9-stage verified state machine over a workspace.

    Stages: Ingest -> GoI_Plan -> Wavefront_Schedule -> Overlay_Mutate ->
    PreWrite_Guard -> SMT_Concolic_Verify -> Sandbox_Execute -> Reflex_Heal ->
    Adjoint_Commit_or_Rollback.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        router=None,
        overlay_manager: Optional[OverlayTransactionManager] = None,
        sandbox: Optional[HermeticSandboxManager] = None,
        shield: Optional[PrecisionShieldOrchestrator] = None,
        healer=None,
        llm_client=None,
        native_solver=None,
        tier: Tier = Tier.REASONING,
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_client_factory=None,
        llm_tier_overrides: Optional[dict] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.router = router
        self.overlay = overlay_manager or OverlayTransactionManager(self.workspace_root)
        self.sandbox = sandbox or HermeticSandboxManager(self.workspace_root)
        self.shield = shield or PrecisionShieldOrchestrator(str(self.workspace_root))
        self._healer = healer
        self._llm_client = llm_client
        self._native_solver = native_solver
        # Builder tier: the verified kernel performs code generation, repair, and
        # SMT-guided synthesis, so it defaults to the reasoning (pro) tier.
        self.tier = tier
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self._llm_client_factory = llm_client_factory
        self._llm_tier_overrides = dict(llm_tier_overrides or {})
        self.current_stage = AgentStateStage.INGEST
        self.stage_history: list[dict[str, Any]] = []
        self._sources_cache: dict[str, str] = {}
        self._progress_cb = None

    # -- stage plumbing ------------------------------------------------------------

    def _enter_stage(self, stage: AgentStateStage, detail: str = "") -> None:
        self.current_stage = stage
        entry = {"stage": stage.value, "at": time.time(), "detail": detail}
        self.stage_history.append(entry)
        if self._progress_cb is not None:
            try:
                self._progress_cb({"type": "stage", "stage": stage.value, "message": detail})
            except Exception:  # noqa: BLE001 - progress reporting must never break the kernel
                pass

    def _emit_thought(self, text: str) -> None:
        if self._progress_cb is not None:
            try:
                self._progress_cb({"type": "thought", "text": text})
            except Exception:  # noqa: BLE001
                pass

    # -- public API ------------------------------------------------------------------

    def ingest_workspace(self) -> dict[str, Any]:
        """Stage 1: Ingest — scan workspace, build symbol index and HIS state."""
        self._enter_stage(AgentStateStage.INGEST, "scanning workspace")
        from invar_agent.context.bundle_repo import WorkspaceIngestor
        from invar_agent.context.inspector import WorkspaceASTInspector

        ingestor = WorkspaceIngestor(self.workspace_root)
        report = ingestor.ingest_repository_graph()
        sources = ingestor.read_sources()
        self._sources_cache = sources
        inspector = WorkspaceASTInspector()
        symbol_table = inspector.extract_symbol_table(sources)
        digest = self._compute_boundary_digest(sources)
        return {
            "report": report.to_dict(),
            "symbol_index": {p: t.public_symbols() for p, t in symbol_table.items()},
            "boundary_digest": digest,
            "file_count": report.file_count,
        }

    def plan_wavefronts(self, nodes: list[TaskWavefrontNode]) -> list[list[str]]:
        """Stages 2-3: GoI_Plan + Wavefront_Schedule via the Rust GoI solver."""
        self._enter_stage(AgentStateStage.GOI_PLAN, f"{len(nodes)} candidate nodes")
        solver = self._get_solver()
        solver.build_from_nodes(nodes)
        if not solver.check_nilpotency():
            raise ValueError("GoI nilpotency check failed: cyclic port dependencies detected")
        self._enter_stage(AgentStateStage.WAVEFRONT_SCHEDULE, "solving disjoint wavefronts")
        return solver.solve_wavefronts()

    def step_wavefront(self, node: TaskWavefrontNode, patches: list[ASTSurgicalPatch],
                      sandbox_cmd: Optional[list[str]] = None) -> NodeExecutionResult:
        """Stages 4-9 for a single wavefront node (Overlay_Mutate .. Adjoint)."""
        result = NodeExecutionResult(node_id=node.node_id, success=False)
        node_start = time.perf_counter()
        max_heal_rounds = 3

        self._enter_stage(AgentStateStage.OVERLAY_MUTATE, f"node={node.node_id}")
        try:
            tx = self.overlay.begin_transaction(node)
            for patch in patches:
                self.overlay.stage_mutation(tx.tx_id, patch)
        except Exception as e:  # noqa: BLE001
            try:
                self.overlay.rollback_via_adjoint(tx.tx_id)  # type: ignore[possibly-undefined]
            except Exception:  # noqa: BLE001
                pass
            result.rolled_back = True
            result.error = f"overlay mutation failed: {e}"
            result.latency_ms = (time.perf_counter() - node_start) * 1000.0
            return result

        heal_rounds = 0
        while True:
            staged = self.overlay.staged_files(tx.tx_id)

            # ---- Stage 5: PreWrite_Guard --------------------------------------
            self._enter_stage(AgentStateStage.PREWRITE_GUARD, f"node={node.node_id}")
            from invar_agent.guards.pre_write_validator import PreWriteValidator
            from invar_agent.guards.syntax_guard import MultiLangSyntaxGuard

            validator = PreWriteValidator()
            guard = MultiLangSyntaxGuard()
            violations = []
            syntax_bad: list[str] = []
            for path, content in staged.items():
                violations.extend(validator.validate_staged_buffer(path, content))
                syn = guard.verify_by_extension(path, content)
                if not syn.valid:
                    syntax_bad.extend(syn.errors)
            if violations:
                self.overlay.rollback_via_adjoint(tx.tx_id)
                result.rolled_back = True
                result.error = "prewrite violations: " + "; ".join(v.rule for v in violations[:5])
                break
            if syntax_bad:
                healed_sources = self._healer_for(node).heal_staged_sources_tier0_only(
                    staged, "\n".join(syntax_bad), node=node
                )
                if healed_sources is not None and heal_rounds < max_heal_rounds:
                    result.healed_by_tier = result.healed_by_tier or "TIER_0_REFLEX"
                    self._apply_healed_sources(tx.tx_id, healed_sources, staged)
                    heal_rounds += 1
                    continue
                self.overlay.rollback_via_adjoint(tx.tx_id)
                result.rolled_back = True
                result.error = "syntax guard failed: " + "; ".join(syntax_bad[:3])
                break

            # ---- Stage 6: SMT_Concolic_Verify ---------------------------------
            self._enter_stage(AgentStateStage.SMT_CONCOLIC_VERIFY, f"node={node.node_id}")
            shield_report = self.shield.run_verification_gate(
                node,
                staged_files=staged,
                workspace_root=str(self.workspace_root),
                pre_source=self._read_workspace_file(node, patches),
                post_source=next(iter(staged.values()), ""),
            )
            result.shield_report = shield_report
            if not shield_report.passed:
                healed_sources = self._shield_heal_sources(node, staged, shield_report)
                if healed_sources is not None and heal_rounds < max_heal_rounds:
                    self._apply_healed_sources(tx.tx_id, healed_sources, staged)
                    heal_rounds += 1
                    continue
                self.overlay.rollback_via_adjoint(tx.tx_id)
                result.rolled_back = True
                result.error = "precision shield failed: " + "; ".join(shield_report.diagnostics[:3])
                break
            self.overlay.mark_guards(tx.tx_id, prewrite=True, shield=True)

            # ---- Stage 7: Sandbox_Execute --------------------------------------
            self._enter_stage(AgentStateStage.SANDBOX_EXECUTE, f"node={node.node_id}")
            sandbox_result = self._run_sandbox(tx, staged, sandbox_cmd)
            result.sandbox_result = sandbox_result
            if sandbox_result.exit_code != 0:
                verdict = self._sandbox_heal(node, staged, sandbox_result)
                if verdict is not None and verdict.healed and heal_rounds < max_heal_rounds:
                    result.healing_verdict = verdict
                    result.healed_by_tier = verdict.resolved_by_tier
                    healed_sources = self._verdict_sources(verdict, staged)
                    if healed_sources:
                        self._apply_healed_sources(tx.tx_id, healed_sources, staged)
                        heal_rounds += 1
                        continue
                self.overlay.rollback_via_adjoint(tx.tx_id)
                result.rolled_back = True
                result.error = f"sandbox failed (exit {sandbox_result.exit_code}): {sandbox_result.stderr[-300:]}"
                break
            self.overlay.mark_guards(tx.tx_id, sandbox=True)

            # ---- Stage 8: Reflex_Heal (already integrated per-stage) ------------
            self._enter_stage(AgentStateStage.REFLEX_HEAL, "no faults outstanding")

            # ---- Stage 9: Adjoint_Commit_or_Rollback -> COMMIT ------------------
            self._enter_stage(AgentStateStage.ADJOINT_COMMIT_OR_ROLLBACK, f"node={node.node_id} COMMIT")
            from invar_agent.overlay.apply import AtomicOverlayApplier

            try:
                AtomicOverlayApplier(self.workspace_root).apply_verified_transaction(
                    self.overlay.get_transaction(tx.tx_id), staged
                )
                result.committed = True
                result.success = True
            except Exception as e:  # noqa: BLE001
                self.overlay.rollback_via_adjoint(tx.tx_id)
                result.rolled_back = True
                result.error = f"commit failed: {e}"
            break

        result.latency_ms = (time.perf_counter() - node_start) * 1000.0
        result.stage_history = list(self.stage_history)
        return result

    def run_until_verified(
        self,
        objective: str,
        nodes: list[TaskWavefrontNode],
        patches_by_node: Optional[dict[str, list[ASTSurgicalPatch]]] = None,
        sandbox_cmd: Optional[list[str]] = None,
        progress_cb=None,
    ) -> AgentRunReport:
        """Full run: ingest -> plan -> schedule -> execute all wavefronts."""
        report = AgentRunReport(run_id=f"run-{uuid.uuid4().hex[:10]}", objective=objective)
        report.started_at = time.time()
        self.stage_history = []
        self._progress_cb = progress_cb
        try:
            return self._run_until_verified_inner(objective, nodes, patches_by_node, sandbox_cmd, report)
        finally:
            self._progress_cb = None

    def _run_until_verified_inner(
        self,
        objective: str,
        nodes: list[TaskWavefrontNode],
        patches_by_node: Optional[dict[str, list[ASTSurgicalPatch]]],
        sandbox_cmd: Optional[list[str]],
        report: AgentRunReport,
    ) -> AgentRunReport:
        ingest = self.ingest_workspace()
        report.boundary_digest_before = ingest["boundary_digest"]

        patches_by_node = patches_by_node or {}
        self._emit_thought(f"Objective accepted: {objective!r}. Ingesting workspace graph…")
        try:
            wavefronts = self.plan_wavefronts(nodes)
        except Exception as e:  # noqa: BLE001
            self._emit_thought(f"GoI planning rejected the candidate graph: {e}")
            report.finished_at = time.time()
            report.nodes_total = len(nodes)
            for node in nodes:
                report.results.append(NodeExecutionResult(node_id=node.node_id, success=False, error=str(e)))
                report.nodes_rolled_back += 1
            return report
        self._emit_thought(
            f"GoI nilpotency verified; {len(nodes)} nodes scheduled into "
            f"{len(wavefronts)} Bernstein-disjoint wavefront(s)."
        )

        report.nodes_total = len(nodes)
        node_by_id = {n.node_id: n for n in nodes}
        for generation in wavefronts:
            for node_id in generation:
                node = node_by_id.get(node_id)
                if node is None:
                    continue
                res = self.step_wavefront(node, patches_by_node.get(node_id, []), sandbox_cmd=sandbox_cmd)
                report.results.append(res)
                if res.committed:
                    report.nodes_committed += 1
                if res.rolled_back:
                    report.nodes_rolled_back += 1
                if res.healed_by_tier:
                    report.nodes_healed += 1
        # Refresh boundary digest from disk post-commit.
        from invar_agent.context.bundle_repo import WorkspaceIngestor

        sources = WorkspaceIngestor(self.workspace_root).read_sources()
        report.boundary_digest_after = self._compute_boundary_digest(sources)
        report.finished_at = time.time()
        return report

    # -- healing -----------------------------------------------------------------------

    def _healer_for(self, node=None):
        if self._healer is None:
            from invar_agent.healing.healer import AutonomicHealerOrchestrator
            from invar_agent.healing.llm_healer import SurgicalLLMHealer

            has_llm = (
                self._llm_client is not None
                or self.llm_api_key
                or self._llm_client_factory is not None
            )
            if has_llm:
                # LLM call rationale: Tier-1 flash surgical repair is a lightweight
                # sub-task, so it uses the FAST tier (light model).
                flash_healer = SurgicalLLMHealer(
                    client=self._llm_client,
                    tier=Tier.FAST,
                    provider=self.llm_provider,
                    api_key=self.llm_api_key,
                    client_factory=self._llm_client_factory,
                    tier_overrides=self._llm_tier_overrides,
                )
                # LLM call rationale: Tier-2 pro architectural repair and code
                # generation require deep reasoning, so it uses Tier.REASONING.
                pro_healer = SurgicalLLMHealer(
                    client=self._llm_client,
                    tier=Tier.REASONING,
                    provider=self.llm_provider,
                    api_key=self.llm_api_key,
                    client_factory=self._llm_client_factory,
                    tier_overrides=self._llm_tier_overrides,
                )
                self._healer = AutonomicHealerOrchestrator(
                    flash_healer=flash_healer, pro_healer=pro_healer
                )
            else:
                self._healer = AutonomicHealerOrchestrator()
        return self._healer

    def _apply_healed_sources(self, tx_id: str, healed_sources: dict[str, str],
                              previous_staged: dict[str, str]) -> None:
        """Overwrites staged buffers with healed contents (in-place CoW update)."""
        for path, content in healed_sources.items():
            if path in previous_staged or path in self.overlay.staged_files(tx_id):
                self.overlay.overwrite_staged_file(tx_id, path, content)

    def heal_staged_sources_tier0_only(self, staged, diagnostic, node=None):
        return self._healer_for(node).heal_staged_sources_tier0_only(staged, diagnostic, node=node)

    def _shield_heal_sources(self, node, staged, shield_report):
        """SMT counterexample healing: only reflex-eligible categories.

        LLM call rationale (when SMT-guided synthesis is wired): generating
        repairs from SMT counterexamples is code generation, so it must use
        Tier.REASONING (pro model), never the FAST tier.
        """
        return None

    def _sandbox_heal(self, node, staged, sandbox_result) -> Optional[HealingVerdict]:
        healer = self._healer_for(node)
        return healer.heal_from_stderr(
            sandbox_result.stderr + "\n" + sandbox_result.stdout,
            sandbox_result.exit_code,
            dict(staged),
            node=node,
            max_tier1_attempts=1,
            max_tier2_attempts=0,
        )

    def _verdict_sources(self, verdict: HealingVerdict, staged) -> dict[str, str]:
        """HealingVerdict carries repaired_patches; convert to full sources when possible."""
        # Tier-0 reflex path returns repaired patches with full-file replacements.
        out: dict[str, str] = {}
        for patch in verdict.repaired_patches:
            if patch.operation == "REPLACE_EXACT_SPAN" and patch.search_block in staged.get(patch.file_path, ""):
                out[patch.file_path] = staged[patch.file_path].replace(patch.search_block, patch.replacement_code, 1)
            elif patch.operation == "REPLACE_SYMBOL" or patch.operation == "CREATE_FILE":
                # Full-file replacement heuristic: replacement_code is the new content.
                if patch.replacement_code.strip():
                    out[patch.file_path] = patch.replacement_code
        return out

    def _run_sandbox(self, tx, staged, sandbox_cmd) -> SandboxExecutionResult:
        if sandbox_cmd:
            return self.sandbox.run_isolated_command(sandbox_cmd, tx_id=tx.tx_id, staged_files=staged)
        return self.sandbox.run_pytest(tx_id=tx.tx_id, staged_files=staged)

    # -- helpers ------------------------------------------------------------------------

    def _get_solver(self):
        if self._native_solver is not None:
            return self._native_solver
        from invar_agent.scheduler.goi_solver import GoiTaskScheduler

        self._native_solver = GoiTaskScheduler()
        return self._native_solver

    def _read_workspace_file(self, node, patches) -> str:
        if patches:
            return patches[0].replacement_code
        symbols = getattr(node, "target_ast_symbols", []) or []
        if symbols:
            name = symbols[0].split(".")[-1]
            for path, content in self._sources_cache.items():
                if f"def {name}" in content or f"class {name}" in content:
                    return content
        return ""

    def _compute_boundary_digest(self, sources: dict[str, str]) -> str:
        from invar_agent.verification.holographic import HolographicVerifier

        return HolographicVerifier().encode_bulk_ast_to_boundary(sources)

    def stage_outputs(self) -> list[dict[str, Any]]:
        return list(self.stage_history)

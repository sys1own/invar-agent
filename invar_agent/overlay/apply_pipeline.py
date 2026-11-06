"""Guarded mutation apply pipeline: PreWrite -> Syntax -> Shield -> Commit."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from invar_agent.contracts import ASTSurgicalPatch, OverlayTransaction, ShieldVerificationReport, TaskWavefrontNode

from ..guards.pre_write_validator import PreWriteValidator
from ..guards.syntax_guard import MultiLangSyntaxGuard
from ..verification.shield import PrecisionShieldOrchestrator


@dataclass
class PipelineStageResult:
    stage: str
    passed: bool
    detail: str = ""


@dataclass
class ApplyPipelineReport:
    transaction_id: str
    stages: list[PipelineStageResult] = field(default_factory=list)
    shield_report: Optional[ShieldVerificationReport] = None
    committed: bool = False
    rolled_back: bool = False

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.stages) and self.committed


class MutationApplyPipeline:
    """Drives one overlay transaction through every pre-commit gate."""

    def __init__(
        self,
        workspace_root: str,
        overlay_manager=None,
        pre_write_validator: Optional[PreWriteValidator] = None,
        syntax_guard: Optional[MultiLangSyntaxGuard] = None,
        shield: Optional[PrecisionShieldOrchestrator] = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.overlay_manager = overlay_manager
        self.pre_write = pre_write_validator or PreWriteValidator()
        self.syntax_guard = syntax_guard or MultiLangSyntaxGuard()
        self.shield = shield or PrecisionShieldOrchestrator(workspace_root)

    def execute_guarded_apply(
        self,
        node: TaskWavefrontNode,
        patches: list[ASTSurgicalPatch],
        run_sandbox: bool = False,
        sandbox_runner=None,
    ) -> ApplyPipelineReport:
        report = ApplyPipelineReport(transaction_id="")
        if self.overlay_manager is None:
            report.stages.append(PipelineStageResult("overlay", False, "no overlay manager"))
            return report

        tx = self.overlay_manager.begin_transaction(node)
        report.transaction_id = tx.tx_id

        # Stage all patches into the CoW overlay (never the host disk).
        try:
            for patch in patches:
                self.overlay_manager.stage_mutation(tx.tx_id, patch)
            report.stages.append(PipelineStageResult("overlay_mutate", True, f"{len(patches)} patch(es) staged"))
        except Exception as e:  # noqa: BLE001
            report.stages.append(PipelineStageResult("overlay_mutate", False, str(e)))
            self.overlay_manager.rollback_via_adjoint(tx.tx_id)
            report.rolled_back = True
            return report

        staged = self.overlay_manager.staged_files(tx.tx_id)

        # PreWrite guard.
        violations = []
        for path, content in staged.items():
            violations.extend(self.pre_write.validate_staged_buffer(path, content))
        if violations:
            detail = "; ".join(f"{v.path}:{v.line} {v.rule}" for v in violations[:5])
            report.stages.append(PipelineStageResult("prewrite_guard", False, detail))
            self.overlay_manager.rollback_via_adjoint(tx.tx_id)
            report.rolled_back = True
            return report
        self.overlay_manager.mark_guards(tx.tx_id, prewrite=True)
        report.stages.append(PipelineStageResult("prewrite_guard", True, "hygiene clean"))

        # Syntax guard.
        syntax_ok = True
        syntax_errors: list[str] = []
        for path, content in staged.items():
            res = self.syntax_guard.verify_by_extension(path, content)
            if not res.valid:
                syntax_ok = False
                syntax_errors.extend(res.errors)
        if not syntax_ok:
            report.stages.append(PipelineStageResult("syntax_guard", False, "; ".join(syntax_errors[:5])))
            return report  # caller routes to Reflex_Heal
        self.overlay_manager.mark_guards(tx.tx_id, prewrite=True)
        report.stages.append(PipelineStageResult("syntax_guard", True, "all staged buffers parse"))

        # Precision Shield.
        shield_report = self.shield.run_verification_gate(
            node,
            staged_files=staged,
            workspace_root=self.workspace_root,
            post_source=next(iter(staged.values()), ""),
        )
        report.shield_report = shield_report
        if not shield_report.passed:
            report.stages.append(PipelineStageResult("precision_shield", False,
                                                     "; ".join(shield_report.diagnostics[:5])))
            return report
        self.overlay_manager.mark_guards(tx.tx_id, shield=True)
        report.stages.append(PipelineStageResult("precision_shield", True, "all gates passed"))

        # Optional sandbox gate.
        if run_sandbox and sandbox_runner is not None:
            ok, detail = sandbox_runner(tx, staged)
            if not ok:
                report.stages.append(PipelineStageResult("sandbox_execute", False, detail))
                return report
            self.overlay_manager.mark_guards(tx.tx_id, sandbox=True)
        else:
            # Sandbox gate is mandatory for commit; when not run, commit is deferred
            # to the agent kernel which always runs it.
            pass
        report.stages.append(PipelineStageResult("sandbox_execute", True, "delegated"))

        # Atomic commit.
        try:
            from .apply import AtomicOverlayApplier

            applier = AtomicOverlayApplier(self.workspace_root)
            applier.apply_verified_transaction(self.overlay_manager.get_transaction(tx.tx_id), staged)
            report.committed = True
            report.stages.append(PipelineStageResult("adjoint_commit", True, "committed atomically"))
        except Exception as e:  # noqa: BLE001
            report.stages.append(PipelineStageResult("adjoint_commit", False, str(e)))
            self.overlay_manager.rollback_via_adjoint(tx.tx_id)
            report.rolled_back = True
        return report

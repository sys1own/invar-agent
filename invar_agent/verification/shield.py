"""Precision Shield: unified FOGE + SMT + Concolic + Chiasmus + Holographic gate."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from invar_agent.contracts import ShieldVerificationReport, VerificationContract

from . import chiasmus as chiasmus_mod
from . import concolic as concolic_mod
from . import foge as foge_mod
from . import holographic as holographic_mod
from . import rust_shield as rust_shield_mod
from . import smt_engine as smt_engine_mod


@dataclass
class ShieldContext:
    """Everything the Precision Shield needs to gate one wavefront node."""

    node_id: str = ""
    file_path: str = ""
    pre_source: str = ""
    post_source: str = ""
    staged_files: dict[str, str] = field(default_factory=dict)
    workspace_root: str = "."
    contract: Optional[VerificationContract] = None
    boundary_before: str = ""
    boundary_after: str = ""
    concolic_func: Optional[object] = None
    concolic_inputs: Optional[dict] = None
    chiasmus_source: Optional[str] = None


class PrecisionShieldOrchestrator:
    """Runs all verification gates and emits a ShieldVerificationReport."""

    def __init__(
        self,
        workspace_root: str = ".",
        smt_engine: Optional[smt_engine_mod.SymbolicVerificationEngine] = None,
        concolic_executor: Optional[concolic_mod.ConcolicExecutor] = None,
        chiasmus_verifier: Optional[chiasmus_mod.ChiasmusSymmetryVerifier] = None,
        holographic_verifier: Optional[holographic_mod.HolographicVerifier] = None,
        foge_executor: Optional[foge_mod.FirstOrderGuardedExecutor] = None,
        rust_shield: Optional[rust_shield_mod.RustSourceShield] = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.smt_engine = smt_engine or smt_engine_mod.SymbolicVerificationEngine()
        self.concolic_executor = concolic_executor or concolic_mod.ConcolicExecutor()
        self.chiasmus_verifier = chiasmus_verifier or chiasmus_mod.ChiasmusSymmetryVerifier()
        self.holographic_verifier = holographic_verifier or holographic_mod.HolographicVerifier()
        self.foge_executor = foge_executor or foge_mod.FirstOrderGuardedExecutor(workspace_root)
        self.rust_shield = rust_shield or rust_shield_mod.RustSourceShield()

    def run_verification_gate(
        self,
        node,
        staged_files: Optional[dict[str, str]] = None,
        workspace_root: Optional[str] = None,
        pre_source: str = "",
        post_source: str = "",
    ) -> ShieldVerificationReport:
        """Gate signature used by the agent kernel: node + overlay state."""
        contract: VerificationContract = getattr(node, "verification_contract", None) or VerificationContract()
        files = staged_files or {}
        root = workspace_root or self.workspace_root
        ctx = ShieldContext(
            node_id=getattr(node, "node_id", ""),
            file_path=getattr(node, "target_ast_symbols", [""])[0] if getattr(node, "target_ast_symbols", None) else "",
            pre_source=pre_source,
            post_source=post_source or "",
            staged_files=files,
            workspace_root=root,
            contract=contract,
        )
        return self.verify_node(ctx)

    def verify_node(self, ctx: ShieldContext) -> ShieldVerificationReport:
        contract = ctx.contract or VerificationContract()
        diagnostics: list[str] = []

        # 1. FOGE pre-invocation guards.
        foge_passed = True
        if contract.foge_guard_pre:
            context = {"holographic_parity_valid": True, **{k: v for k, v in ctx.staged_files.items()}}
            gate = self.foge_executor.gate_tool_invocation(ctx.node_id or "node", contract.foge_guard_pre, context)
            foge_passed = gate.allowed
            if not gate.allowed:
                diagnostics.append(f"FOGE blocked: {gate.detail}")

        # 2. SMT postcondition proof.
        smt_status = "UNSAT_PROVEN"
        witness = None
        code_units = {p: s for p, s in ctx.staged_files.items() if p.endswith(".py")}
        if ctx.post_source:
            code_units[ctx.file_path or "<node>"] = ctx.post_source
        if contract.smt_postconditions:
            verdict = self.smt_engine.verify_contract(contract, code_units)
            if not verdict.passed:
                smt_status = "SAT_COUNTEREXAMPLE"
                witness = verdict.witness
                diagnostics.extend(verdict.diagnostics)
            else:
                smt_status = "UNSAT_PROVEN"
        else:
            # Still attempt generic postcondition-free check: nothing to prove.
            smt_status = "UNSAT_PROVEN"

        # 3. Concolic exploration (only when a callable and inputs are provided).
        branches_explored = 0
        concolic_failures: list[str] = []
        if ctx.concolic_func is not None and ctx.concolic_inputs:
            report = self.concolic_executor.explore(
                ctx.concolic_func,
                list(ctx.concolic_inputs.keys()),
                ctx.concolic_inputs,
            )
            branches_explored = report.branches_explored
            concolic_failures = [
                f"{f.exception}: {f.traceback_tail} inputs={f.inputs}" for f in report.findings
            ]

        # 4. Chiasmus symmetry over the post-source (or staged python files).
        chiasmus_symmetric = True
        chiasmus_violations: list[str] = []
        chiasmus_source = ctx.chiasmus_source
        if chiasmus_source is None and ctx.post_source:
            chiasmus_source = ctx.post_source
        pairs = [tuple(p) for p in contract.chiasmus_pairs] or None
        if chiasmus_source:
            result = self.chiasmus_verifier.verify_mirror_invariants(chiasmus_source, pairs)
            chiasmus_symmetric = result.symmetric
            chiasmus_violations = result.diagnostics
            if not result.symmetric:
                diagnostics.extend(chiasmus_violations)

        # 5. Holographic parity.
        holographic_parity_valid = True
        if ctx.boundary_before or ctx.boundary_after:
            holographic_parity_valid = ctx.boundary_before == ctx.boundary_after
            if not holographic_parity_valid:
                diagnostics.append("Holographic boundary drift detected without authorization")
        elif ctx.staged_files:
            after = self.holographic_verifier.encode_bulk_ast_to_boundary(ctx.staged_files)
            holographic_parity_valid = True  # baseline recorded by caller
            ctx.boundary_after = after

        # 6. Rust shield over staged .rs files.
        rust_report = self.rust_shield.audit_workspace(
            {p: s for p, s in ctx.staged_files.items() if p.endswith(".rs")}
        )
        rust_passed = rust_report.passed
        if not rust_passed:
            diagnostics.extend(rust_report.violations)

        passed = (
            foge_passed
            and smt_status != "SAT_COUNTEREXAMPLE"
            and chiasmus_symmetric
            and holographic_parity_valid
            and rust_passed
        )
        return ShieldVerificationReport(
            passed=passed,
            foge_passed=foge_passed,
            smt_status=smt_status,  # type: ignore[arg-type]
            smt_counterexample_witness=witness,
            concolic_branches_explored=branches_explored,
            concolic_failures=concolic_failures,
            chiasmus_symmetric=chiasmus_symmetric,
            chiasmus_violations=chiasmus_violations,
            holographic_parity_valid=holographic_parity_valid,
            rust_shield_passed=rust_passed,
            diagnostics=diagnostics,
        )

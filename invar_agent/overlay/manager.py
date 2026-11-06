"""Transactional overlay manager: staged CoW mutations with Adjoint rollback."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

from invar_agent.contracts import ASTSurgicalPatch, OverlayTransaction, TaskWavefrontNode

from .store import ContentAddressableOverlayStore, SnapshotManifest

try:
    from invar_agent._native import PyHisRegister as _NativeHis
except Exception:  # pragma: no cover
    _NativeHis = None


def _make_his():
    if _NativeHis is not None:
        return _NativeHis()
    from invar_agent.verification.holographic import HolographicVerifier

    return HolographicVerifier().register


class OverlayTransactionManager:
    """Copy-on-Write overlay: LLM mutations never touch the host disk directly."""

    def __init__(
        self,
        workspace_root: str | os.PathLike,
        store: Optional[ContentAddressableOverlayStore] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.store = store or ContentAddressableOverlayStore()
        self.transactions: dict[str, OverlayTransaction] = {}
        # tx_id -> {path: staged_content}
        self._staged: dict[str, dict[str, str]] = {}
        # tx_id -> snapshot of pre-transaction staged buffers
        self._snapshots: dict[str, dict[str, str]] = {}
        self._write_sets: dict[str, set[str]] = {}

    # -- lifecycle ---------------------------------------------------------------

    def begin_transaction(self, node: TaskWavefrontNode) -> OverlayTransaction:
        tx_id = f"tx-{uuid.uuid4().hex[:12]}"
        write_set = set(node.write_set_paths)
        self._write_sets[tx_id] = write_set
        # Snapshot current staged view of the write set (empty for fresh txs).
        snapshot: dict[str, str] = {}
        for path in write_set:
            existing = self.read_file_any_tx(path)
            if existing is not None:
                snapshot[path] = existing
        self._snapshots[tx_id] = snapshot
        boundary = self._compute_boundary_digest({})
        tx = OverlayTransaction(
            tx_id=tx_id,
            node_id=node.node_id,
            base_snapshot_digest=self.store.hash_content(repr(sorted(snapshot.items()))),
            pre_holographic_boundary=boundary,
            staged_patches=[],
        )
        self.transactions[tx_id] = tx
        self._staged[tx_id] = {}
        return tx

    def begin_transaction_raw(self, node_id: str, write_set_paths: list[str]) -> OverlayTransaction:
        class _Node:
            pass

        node = _Node()
        node.node_id = node_id
        node.write_set_paths = write_set_paths
        return self.begin_transaction(node)  # type: ignore[arg-type]

    def get_transaction(self, tx_id: str) -> OverlayTransaction:
        return self.transactions[tx_id]

    # -- mutation ---------------------------------------------------------------

    def stage_mutation(self, tx_id: str, patch: ASTSurgicalPatch) -> OverlayTransaction:
        tx = self.transactions[tx_id]
        write_set = self._write_sets.get(tx_id, set())
        if write_set and patch.file_path not in write_set:
            raise PermissionError(
                f"Write-set violation: {patch.file_path} not in {sorted(write_set)}"
            )
        staged = self._staged[tx_id]
        if patch.file_path not in staged:
            base = self._read_disk_or_staged(tx_id, patch.file_path)
            staged[patch.file_path] = base
        from .patch import apply_single_patch

        result = apply_single_patch(staged[patch.file_path], patch)
        if not result.applied:
            raise ValueError(f"Patch application failed for {patch.file_path}: {result.diagnostics}")
        staged[patch.file_path] = result.source
        self.store.put_ast_blob(result.source, patch.file_path)
        tx.staged_patches.append(patch)
        return tx

    def read_file(self, tx_id: str, path: str) -> str:
        staged = self._staged.get(tx_id, {})
        if path in staged:
            return staged[path]
        return self._read_disk(path)

    def staged_files(self, tx_id: str) -> dict[str, str]:
        return dict(self._staged.get(tx_id, {}))

    def overwrite_staged_file(self, tx_id: str, path: str, content: str) -> None:
        """Replaces a staged buffer in place (used by Reflex_Heal rounds)."""
        staged = self._staged.setdefault(tx_id, {})
        staged[path] = content
        self.store.put_ast_blob(content, path)

    def mark_guards(self, tx_id: str, prewrite: bool = False, shield: bool = False,
                    sandbox: bool = False) -> None:
        tx = self.transactions[tx_id]
        tx.prewrite_guard_passed = tx.prewrite_guard_passed or prewrite
        tx.shield_verified = tx.shield_verified or shield
        tx.sandbox_passed = tx.sandbox_passed or sandbox

    # -- adjoint rollback ---------------------------------------------------------

    def rollback_via_adjoint(self, tx_id: str) -> bool:
        """Restore staged buffers to the pre-transaction snapshot (T_adjoint)."""
        tx = self.transactions.get(tx_id)
        if tx is None:
            return False
        snapshot = self._snapshots.get(tx_id, {})
        staged = self._staged.setdefault(tx_id, {})
        # Remove files that were created by this transaction.
        created = {p.file_path for p in tx.staged_patches if p.operation == "CREATE_FILE"}
        for path in list(staged.keys()):
            if path in created:
                del staged[path]
        # Restore snapshot contents bit-for-bit.
        for path, content in snapshot.items():
            staged[path] = content
        tx.rolled_back = True
        tx.staged_patches = []
        return True

    def compute_post_boundary(self, tx_id: str) -> str:
        tx = self.transactions[tx_id]
        digest = self._compute_boundary_digest(self._staged.get(tx_id, {}))
        tx.post_holographic_boundary = digest
        return digest

    # -- helpers -------------------------------------------------------------------

    def _read_disk(self, path: str) -> str:
        full = self.workspace_root / path
        if full.exists():
            return full.read_text(encoding="utf-8")
        return ""

    def _read_disk_or_staged(self, tx_id: str, path: str) -> str:
        staged = self._staged.get(tx_id, {})
        if path in staged:
            return staged[path]
        # Check other active transactions for a newer staged version.
        for other_tx, other_staged in self._staged.items():
            if other_tx != tx_id and path in other_staged:
                return other_staged[path]
        return self._read_disk(path)

    def read_file_any_tx(self, path: str) -> Optional[str]:
        best: Optional[str] = None
        for staged in self._staged.values():
            if path in staged:
                best = staged[path]
        if best is not None:
            return best
        full = self.workspace_root / path
        return full.read_text(encoding="utf-8") if full.exists() else None

    def _compute_boundary_digest(self, staged: dict[str, str]) -> str:
        from invar_agent.verification.holographic import extract_boundary_signature

        his = _make_his()
        paths = sorted(
            {p for p in staged.keys() if p.endswith((".py", ".rs"))}
        )
        for path in paths:
            his.update_module_boundary(path, extract_boundary_signature(staged[path]))
        return his.project_boundary_digest()

    def close(self) -> None:
        self.transactions.clear()
        self._staged.clear()
        self._snapshots.clear()
        self._write_sets.clear()

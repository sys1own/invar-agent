"""Atomic overlay committer: guarded, crash-safe writes to the host disk."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from invar_agent.contracts import OverlayTransaction


class AtomicOverlayApplier:
    """Applies verified overlay transactions to disk atomically (temp + rename)."""

    def __init__(self, workspace_root: str | os.PathLike) -> None:
        self.workspace_root = Path(workspace_root)

    def apply_verified_transaction(self, tx: OverlayTransaction, staged_files: dict[str, str]) -> bool:
        """Refuses to write unless all three gates passed."""
        if not tx.prewrite_guard_passed:
            raise PermissionError("PreWrite guard has not passed; commit refused.")
        if not tx.shield_verified:
            raise PermissionError("Precision Shield has not verified; commit refused.")
        if not tx.sandbox_passed:
            raise PermissionError("Sandbox execution has not passed; commit refused.")
        if tx.rolled_back:
            raise PermissionError("Transaction was rolled back; commit refused.")
        self.apply_staged_files(staged_files)
        tx.committed = True
        return True

    def apply_staged_files(self, staged_files: dict[str, str]) -> list[str]:
        written: list[str] = []
        for rel_path, content in staged_files.items():
            target = self.workspace_root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: temp file in same directory + os.replace.
            fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".invar-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(content)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, target)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            written.append(rel_path)
        return written

    def revert_transaction(self, tx: OverlayTransaction, backup_dir: Optional[str] = None) -> None:
        """Best-effort revert using pre-commit backups (adjoint at disk level)."""
        # The Adjoint path is overlay-level snapshot restore; disk-level revert
        # is only used by the sandbox materializer, which copies to temp dirs.
        tx.rolled_back = True

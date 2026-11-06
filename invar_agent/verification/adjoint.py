"""Adjoint state inversion: exact rollback operators for overlay transactions."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from invar_agent.contracts import ASTSurgicalPatch, OverlayTransaction


@dataclass
class AdjointComputationResult:
    adjoint_patches: list[ASTSurgicalPatch] = field(default_factory=list)
    verification_proof: str = ""
    holographic_parity_match: bool = False
    rollback_digest: str = ""


class AdjointStateOperator:
    """Computes \(T^\\dagger\): the exact inverse of a forward mutation batch."""

    def compute_adjoint_inverse(
        self,
        forward_patches: list[ASTSurgicalPatch],
        original_sources: Optional[dict[str, str]] = None,
    ) -> list[ASTSurgicalPatch]:
        """Reverse-order inverse patches restoring pre-mutation content."""
        originals = original_sources or {}
        inverses: list[ASTSurgicalPatch] = []
        for patch in reversed(forward_patches):
            original_block = originals.get(patch.file_path, "")
            if patch.adjoint_inverse_code:
                inverse_code = patch.adjoint_inverse_code
            elif patch.operation == "CREATE_FILE":
                inverse_code = ""
            else:
                inverse_code = original_block
            inverses.append(
                ASTSurgicalPatch(
                    file_path=patch.file_path,
                    operation="REPLACE_EXACT_SPAN" if patch.operation != "CREATE_FILE" else "REPLACE_SYMBOL",
                    target_symbol=patch.target_symbol,
                    search_block=patch.replacement_code,
                    replacement_code=inverse_code,
                    pre_hash=patch.post_hash,
                    post_hash=patch.pre_hash,
                    adjoint_inverse_code=patch.replacement_code,
                )
            )
        return inverses

    def verify_inner_product_duality(
        self, forward_patches: list[ASTSurgicalPatch], source: str
    ) -> tuple[bool, str]:
        """Checks apply(apply(src, T), T^dagger) == src for a single-file batch."""
        from invar_agent.overlay.patch import apply_patches_to_source

        try:
            after_forward, _ = apply_patches_to_source(source, forward_patches)
            inverses = self.compute_adjoint_inverse(forward_patches, {forward_patches[0].file_path: source} if forward_patches else {})
            # Inverse patches target the post-forward content.
            restored, _ = apply_patches_to_source(after_forward, inverses)
        except Exception as e:  # noqa: BLE001
            return False, f"adjoint application failed: {e}"
        if restored == source:
            return True, "inner product preserved: T_adjoint inverts T exactly"
        return False, "adjoint rollback diverged from original source"

    def compute_adjoint(self, transaction: OverlayTransaction,
                        original_sources: Optional[dict[str, str]] = None) -> AdjointComputationResult:
        adjoint = self.compute_adjoint_inverse(list(transaction.staged_patches), original_sources)
        digest = hashlib.sha256()
        for patch in transaction.staged_patches:
            digest.update(patch.file_path.encode())
            digest.update((patch.pre_hash or "").encode())
            digest.update((patch.post_hash or "").encode())
        return AdjointComputationResult(
            adjoint_patches=adjoint,
            verification_proof="snapshot_restore+inverse_patches",
            holographic_parity_match=transaction.pre_holographic_boundary == (transaction.post_holographic_boundary or ""),
            rollback_digest=digest.hexdigest(),
        )

    def composite_adjoint(self, patch_batches: list[list[ASTSurgicalPatch]]) -> list[ASTSurgicalPatch]:
        """Adjoint of a composite reverses batch order and inverts each patch."""
        composite: list[ASTSurgicalPatch] = []
        for batch in reversed(patch_batches):
            composite.extend(self.compute_adjoint_inverse(batch))
        return composite

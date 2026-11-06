"""Transactional Copy-on-Write overlay with Adjoint rollback."""
from .active_merge import ActiveWavefrontMerger, ConcurrentWriteCoordinator, WavefrontDelta, reconcile_wavefront_deltas
from .apply import AtomicOverlayApplier
from .apply_pipeline import ApplyPipelineReport, MutationApplyPipeline, PipelineStageResult
from .apply_pipeline_healer import ApplyPipelineHealer
from .manager import OverlayTransactionManager
from .patch import PatchApplicationResult, apply_patches_to_source, apply_single_patch, compute_ast_delta, invert_patch, unified_diff
from .rust_merge import RustSynMerger
from .store import BlobEntry, ContentAddressableOverlayStore, SnapshotManifest
from .structural_merger import MergeConflictNode, MergeResult, ThreeWayASTMerger

__all__ = [
    "ActiveWavefrontMerger", "ApplyPipelineHealer", "ApplyPipelineReport", "AtomicOverlayApplier",
    "BlobEntry", "ConcurrentWriteCoordinator", "ContentAddressableOverlayStore", "MergeConflictNode",
    "MergeResult", "MutationApplyPipeline", "OverlayTransactionManager", "PatchApplicationResult",
    "PipelineStageResult", "RustSynMerger", "SnapshotManifest", "ThreeWayASTMerger", "WavefrontDelta",
    "apply_patches_to_source", "apply_single_patch", "compute_ast_delta", "invert_patch",
    "reconcile_wavefront_deltas", "unified_diff",
]

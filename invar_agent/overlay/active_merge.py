"""Active wavefront merger: reconciles concurrent deltas into one overlay state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .structural_merger import MergeConflictNode, MergeResult, ThreeWayASTMerger


@dataclass
class WavefrontDelta:
    node_id: str
    file_path: str
    source: str


class ActiveWavefrontMerger:
    """Merges per-wavefront file states against the committed base."""

    def __init__(self, merger: Optional[ThreeWayASTMerger] = None) -> None:
        self.merger = merger or ThreeWayASTMerger()

    def merge_wavefront_deltas(
        self,
        base_files: dict[str, str],
        deltas: list[WavefrontDelta],
    ) -> tuple[dict[str, str], list[MergeConflictNode]]:
        """Sequentially merge each wavefront delta into the accumulated state."""
        merged_files = dict(base_files)
        all_conflicts: list[MergeConflictNode] = []
        # Group deltas by file, preserving node order.
        by_file: dict[str, list[WavefrontDelta]] = {}
        for delta in deltas:
            by_file.setdefault(delta.file_path, []).append(delta)
        for path, file_deltas in by_file.items():
            base = base_files.get(path, "")
            current = merged_files.get(path, base)
            for delta in file_deltas:
                result = self.merger.merge_ast_branches(base, current, delta.source, file_path=path)
                current = result.merged_source
                all_conflicts.extend(result.conflicts)
            merged_files[path] = current
        return merged_files, all_conflicts


class ConcurrentWriteCoordinator:
    """Arbitrates write-set ownership between concurrent wavefronts."""

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}

    def claim(self, node_id: str, write_set: list[str]) -> bool:
        for path in write_set:
            owner = self._owners.get(path)
            if owner is not None and owner != node_id:
                return False
        for path in write_set:
            self._owners[path] = node_id
        return True

    def release(self, node_id: str, write_set: list[str]) -> None:
        for path in write_set:
            if self._owners.get(path) == node_id:
                del self._owners[path]

    def conflicts_with(self, node_id: str, write_set: list[str]) -> list[str]:
        return [p for p in write_set if self._owners.get(p) not in (None, node_id)]


def reconcile_wavefront_deltas(
    base_files: dict[str, str],
    deltas: list[WavefrontDelta],
) -> MergeResult:
    """Convenience wrapper returning a single MergeResult over all files."""
    merger = ActiveWavefrontMerger()
    merged, conflicts = merger.merge_wavefront_deltas(base_files, deltas)
    # Flatten merged files into a pseudo-source for the result.
    combined = "\n\n".join(f"# --- {p} ---\n{c}" for p, c in sorted(merged.items()))
    return MergeResult(
        merged_source=combined,
        conflicts=conflicts,
        merged_symbols=sorted(merged.keys()),
    )

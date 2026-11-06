"""Workspace ingestion: repo scanning, zip import, HIS boundary state."""
from __future__ import annotations

import io
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ignore import HermeticIgnoreFilter


@dataclass
class WorkspaceIngestionReport:
    workspace_root: str
    file_count: int = 0
    python_files: int = 0
    rust_files: int = 0
    total_lines: int = 0
    files: list[str] = field(default_factory=list)
    boundary_digest: str = ""

    def to_dict(self) -> dict:
        return {
            "workspace_root": self.workspace_root,
            "file_count": self.file_count,
            "python_files": self.python_files,
            "rust_files": self.rust_files,
            "total_lines": self.total_lines,
            "files": self.files,
            "boundary_digest": self.boundary_digest,
        }


class WorkspaceIngestor:
    """Scans a workspace into a hermetic file inventory + HIS boundary digest."""

    def __init__(self, workspace_root: str | os.PathLike) -> None:
        self.workspace_root = Path(workspace_root)
        self.ignore_filter = HermeticIgnoreFilter(self.workspace_root)

    def ingest_repository_graph(self) -> WorkspaceIngestionReport:
        report = WorkspaceIngestionReport(workspace_root=str(self.workspace_root))
        files = self.ignore_filter.iter_workspace_files()
        sources: dict[str, str] = {}
        for rel in files:
            full = self.workspace_root / rel
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            sources[rel] = text
            report.files.append(rel)
            if rel.endswith(".py"):
                report.python_files += 1
            if rel.endswith(".rs"):
                report.rust_files += 1
            report.total_lines += text.count("\n") + 1
        report.file_count = len(report.files)
        report.boundary_digest = self.build_initial_his_state(sources)
        return report

    def build_initial_his_state(self, files: dict[str, str]) -> str:
        from invar_agent.verification.holographic import HolographicVerifier

        return HolographicVerifier().encode_bulk_ast_to_boundary(files)

    def read_sources(self, files: Optional[list[str]] = None) -> dict[str, str]:
        out: dict[str, str] = {}
        for rel in files or self.ignore_filter.iter_workspace_files():
            full = self.workspace_root / rel
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "\x00" in text:
                continue  # binary artifact (e.g. compiled extensions)
            out[rel] = text
        return out

    def ingest_zip(self, zip_path: str | os.PathLike, dest: Optional[str | os.PathLike] = None) -> WorkspaceIngestionReport:
        """Extracts an uploaded workspace zip into a temp dir and ingests it."""
        if dest is None:
            dest = Path(tempfile.mkdtemp(prefix="invar-ingest-"))
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_path)
        # If the zip contains a single top-level directory, descend into it.
        entries = [p for p in dest_path.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir():
            dest_path = entries[0]
        ingestor = WorkspaceIngestor(dest_path)
        report = ingestor.ingest_repository_graph()
        report.workspace_root = str(dest_path)
        return report

"""Export engine producing curated artifacts from agent runs."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

EXPORT_TYPES = ("workspace_zip", "patch_adjoint_bundle", "verification_certificate")


@dataclass
class ExportArtifact:
    export_type: str
    filename: str
    content_type: str
    data: bytes


class ExportEngine:
    """Generates the three curated export formats."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root)

    def workspace_zip(self, report: Optional[dict[str, Any]] = None) -> ExportArtifact:
        buffer = io.BytesIO()
        ignore_dirs = {".git", "__pycache__", ".invar_agent", "target", ".pytest_cache", "node_modules"}
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(self.workspace_root.rglob("*")):
                if path.is_dir():
                    continue
                rel = path.relative_to(self.workspace_root)
                if any(part in ignore_dirs for part in rel.parts):
                    continue
                zf.write(path, str(rel))
            if report is not None:
                zf.writestr("invar_agent/report.json", json.dumps(report, indent=2, default=str))
        return ExportArtifact(
            export_type="workspace_zip",
            filename="workspace.zip",
            content_type="application/zip",
            data=buffer.getvalue(),
        )

    def patch_adjoint_bundle(self, patches: list[dict[str, Any]],
                             adjoint_patches: Optional[list[dict[str, Any]]] = None,
                             report: Optional[dict[str, Any]] = None) -> ExportArtifact:
        bundle = {
            "schema": "invar-agent/patch-adjoint-bundle/v1",
            "patches": patches,
            "adjoint_patches": adjoint_patches or [],
            "report": report or {},
        }
        return ExportArtifact(
            export_type="patch_adjoint_bundle",
            filename="patch_adjoint_bundle.json",
            content_type="application/json",
            data=json.dumps(bundle, indent=2, default=str).encode("utf-8"),
        )

    def verification_certificate(self, report: dict[str, Any],
                                 shield_reports: Optional[list[dict[str, Any]]] = None) -> ExportArtifact:
        cert = {
            "schema": "invar-agent/verification-certificate/v1",
            "kernel": "invar-agent 9-stage verified autonomous agent kernel",
            "run_id": report.get("run_id", ""),
            "objective": report.get("objective", ""),
            "nodes_total": report.get("nodes_total", 0),
            "nodes_committed": report.get("nodes_committed", 0),
            "nodes_rolled_back": report.get("nodes_rolled_back", 0),
            "nodes_healed": report.get("nodes_healed", 0),
            "success": report.get("success", False),
            "boundary_digest_before": report.get("boundary_digest_before", ""),
            "boundary_digest_after": report.get("boundary_digest_after", ""),
            "duration_ms": report.get("duration_ms", 0.0),
            "shield_reports": shield_reports or [],
            "stage_trace": report.get("stage_trace", []),
        }
        return ExportArtifact(
            export_type="verification_certificate",
            filename="verification_certificate.json",
            content_type="application/json",
            data=json.dumps(cert, indent=2, default=str).encode("utf-8"),
        )

    def export(self, export_type: str, report: Optional[dict[str, Any]] = None,
               patches: Optional[list[dict[str, Any]]] = None,
               adjoint_patches: Optional[list[dict[str, Any]]] = None,
               shield_reports: Optional[list[dict[str, Any]]] = None) -> ExportArtifact:
        if export_type == "workspace_zip":
            return self.workspace_zip(report)
        if export_type == "patch_adjoint_bundle":
            return self.patch_adjoint_bundle(patches or [], adjoint_patches, report)
        if export_type == "verification_certificate":
            return self.verification_certificate(report or {}, shield_reports)
        raise ValueError(f"unknown export type: {export_type!r}; expected one of {EXPORT_TYPES}")

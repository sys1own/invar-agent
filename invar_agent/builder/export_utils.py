"""ARCH-012: Verified Builder export trust boundary (allow-list, host-path & secret scan)."""
from __future__ import annotations

import copy
import re
import zipfile
from pathlib import Path
from typing import Any


class ExportBlockedError(ValueError):
    """Raised when a Builder export violates trust-boundary or completion invariants."""

    def __init__(self, code: str, message: str, status_code: int = 409) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status_code = status_code


_HOST_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\(?:Users|Documents and Settings|Program Files|Windows)\\[^\s'\"`]+", re.I),
    re.compile(r"[A-Za-z]:/(?:Users|Documents and Settings|Program Files|Windows)/[^\s'\"`]+", re.I),
    re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home)/[A-Za-z0-9._-]+/[^\s'\"`]+"),
)

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bDEEPSEEK_API_KEY\s*=\s*[^\s'\"`]{8,}"),
    re.compile(r"\bOPENAI_API_KEY\s*=\s*[^\s'\"`]{8,}"),
    re.compile(r"\bANTHROPIC_API_KEY\s*=\s*[^\s'\"`]{8,}"),
)

_EXCLUDED_SEGMENTS = {"__pycache__", ".invar", ".pytest_cache", ".mypy_cache"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _relativize_path_str(raw: str) -> str:
    text = str(raw or "")
    if not text:
        return text
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith(("/Users/", "/home/", "/tmp/", "/var/", "/private/")):
        return Path(text.replace("\\", "/")).name
    return text


def sanitize_ir_for_export(ir_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip absolute host paths from an IR dict before embedding in exported sdk.py."""
    out = copy.deepcopy(ir_dict)
    for list_key in ("uploaded_files", "datasets_attached"):
        if isinstance(out.get(list_key), list):
            out[list_key] = [_relativize_path_str(p) for p in out[list_key]]
    for coll_key in ("constants", "equations", "state_variables", "disambiguation_flags"):
        items = out.get(coll_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            prov = item.get("provenance")
            if isinstance(prov, dict) and "file" in prov:
                prov["file"] = _relativize_path_str(prov["file"])
            if "source_file" in item:
                item["source_file"] = _relativize_path_str(item["source_file"])
    return out


def _is_excluded_relpath(rel_posix: str) -> bool:
    parts = rel_posix.split("/")
    if any(seg in _EXCLUDED_SEGMENTS for seg in parts):
        return True
    if "diagnostics/leaf_responses" in rel_posix:
        return True
    if Path(rel_posix).suffix.lower() in _EXCLUDED_SUFFIXES:
        return True
    return False


def scan_archive_for_leaks(archive_path: Path) -> list[str]:
    """Scan all text members in a zip archive for host paths, secrets, and forbidden bytecode."""
    leaks: list[str] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if _is_excluded_relpath(name):
                leaks.append(f"FORBIDDEN_MEMBER:{name}")
                continue
            if info.is_dir():
                continue
            raw = zf.read(name)
            if b"\x00" in raw[:4096]:
                continue  # skip compiled native binaries (.so / .dll)
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue
            for pat in _HOST_PATH_PATTERNS:
                m = pat.search(text)
                if m:
                    leaks.append(f"HOST_PATH_LEAK:{name}:{m.group(0)[:80]}")
                    break
            for pat in _SECRET_PATTERNS:
                if pat.search(text):
                    leaks.append(f"SECRET_LEAK:{name}")
                    break
    return leaks


def build_verified_export_zip(
    workspace: Path,
    archive_path: Path,
    *,
    build_status: str | None,
    cache_manifest: Any = None,
) -> Path:
    """Create an allow-listed, receipt-verified, leak-scanned export zip archive.

    Fails closed (deletes any partial zip and raises ExportBlockedError) if:
    - build_status is not COMPLETED / COMPLETED_WITH_FALLBACKS,
    - any receipt artifact is missing or tampered,
    - any symlink exists, or
    - any archive member contains a host path or secret pattern.
    """
    if build_status not in {"COMPLETED", "COMPLETED_WITH_FALLBACKS"}:
        raise ExportBlockedError(
            "EXPORT_BUILD_NOT_COMPLETED",
            f"Cannot export build with status {build_status!r}; a completed build is required",
            status_code=409,
        )
    ws = workspace.resolve()
    if not ws.is_dir():
        raise ExportBlockedError("EXPORT_WORKSPACE_MISSING", "Committed workspace not found", status_code=404)

    if cache_manifest is not None:
        from .his_cache import blake3_file
        for entry in (getattr(cache_manifest, "entries", None) or {}).values():
            if getattr(entry, "verified_status", None) not in {"VERIFIED", "FALLBACK_VERIFIED"}:
                continue
            for rel_artifact, expected_digest in (getattr(entry, "artifact_digests", None) or {}).items():
                art_path = (ws / rel_artifact).resolve()
                if not art_path.is_file() or not art_path.is_relative_to(ws):
                    raise ExportBlockedError("EXPORT_MISSING_ARTIFACT", f"Missing verified artifact {rel_artifact}")
                if blake3_file(art_path) != expected_digest:
                    raise ExportBlockedError("EXPORT_DIGEST_MISMATCH", f"Receipt digest mismatch for {rel_artifact}")
        for receipt in getattr(cache_manifest, "receipts", None) or []:
            for rel_artifact, expected_digest in (getattr(receipt, "output_hashes", None) or {}).items():
                art_path = (ws / rel_artifact).resolve()
                if not art_path.is_file() or not art_path.is_relative_to(ws):
                    raise ExportBlockedError("EXPORT_MISSING_ARTIFACT", f"Missing verified artifact {rel_artifact}")
                if blake3_file(art_path) != expected_digest:
                    raise ExportBlockedError("EXPORT_DIGEST_MISMATCH", f"Receipt digest mismatch for {rel_artifact}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(ws.rglob("*")):
                if path.is_symlink():
                    raise ExportBlockedError("EXPORT_UNSAFE_SYMLINK", f"Unsafe symlink: {path.name}", status_code=400)
                if not path.is_file():
                    continue
                rel_posix = path.relative_to(ws).as_posix()
                if _is_excluded_relpath(rel_posix):
                    continue
                zf.write(path, rel_posix)
        leaks = scan_archive_for_leaks(archive_path)
        if leaks:
            raise ExportBlockedError(
                "EXPORT_TRUST_BOUNDARY_LEAK",
                "Export blocked due to trust-boundary leaks: " + "; ".join(leaks[:5]),
                status_code=409,
            )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path

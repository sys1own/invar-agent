"""UPG-014 / ARCH-012: Fail-closed export trust boundary (allow-list, host-path, binary & secret scan)."""
from __future__ import annotations

import getpass
import os
from pathlib import Path
import re
from typing import Any, Mapping
import zipfile

from invar_agent.redact import sanitize_tree


class ExportBlockedError(ValueError):
    """Raised when an export violates trust-boundary or completion invariants."""

    def __init__(self, code: str, message: str, status_code: int = 409) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status_code = status_code


_HOST_PATH_NORM_RE = re.compile(
    r"(?i)(?<![A-Za-z])[A-Z]:[\\/](?![\\/])[^\s'\"`]+"
    r"|(?<![\w.])/(?:[a-z]/)?(?:Users|home)/[^\s'\"`]+"
    r"|\bfile://[^\s'\"`]+"
)
_UNC_PATH_RAW_RE = re.compile(r"(?<!\\)\\\\[A-Za-z0-9_.$-]+\\[A-Za-z0-9_.$-]+")

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bDEEPSEEK_API_KEY\s*=\s*[^\s'\"`]{8,}"),
    re.compile(r"\bOPENAI_API_KEY\s*=\s*[^\s'\"`]{8,}"),
    re.compile(r"\bANTHROPIC_API_KEY\s*=\s*[^\s'\"`]{8,}"),
)

_EXCLUDED_SEGMENTS = {"__pycache__", ".invar", ".pytest_cache", ".mypy_cache"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def normalise_for_scan(text: str) -> str:
    """Collapse repeated backslashes and URL-encoded path separators before scanning."""
    t = re.sub(r"\\{2,}", r"\\", text)
    t = (
        t.replace("%5C", "\\")
        .replace("%5c", "\\")
        .replace("%3A", ":")
        .replace("%3a", ":")
        .replace("%2F", "/")
        .replace("%2f", "/")
    )
    return t


def _is_excluded_relpath(rel_posix: str) -> bool:
    parts = rel_posix.split("/")
    if any(seg in _EXCLUDED_SEGMENTS for seg in parts):
        return True
    if "diagnostics/leaf_responses" in rel_posix:
        return True
    if Path(rel_posix).suffix.lower() in _EXCLUDED_SUFFIXES:
        return True
    return False


def _host_literals(workspace_root: Path | None = None) -> tuple[list[str], str | None]:
    lits: list[str] = []
    roots: list[Path] = []
    if workspace_root is not None:
        try:
            roots.append(Path(workspace_root).resolve())
        except Exception:
            pass
    try:
        roots.append(Path.home().resolve())
    except Exception:
        pass
    for r in roots:
        s = str(r)
        if len(s) >= 4:
            lits.append(s.lower())
            lits.append(s.replace("\\", "/").lower())
    user: str | None = None
    try:
        u = getpass.getuser()
        if u and len(u) >= 4:
            user = u.lower()
    except Exception:
        pass
    return list(dict.fromkeys(lits)), user


def _scan_text_member(name: str, text: str, literals: list[str], user: str | None) -> list[str]:
    leaks: list[str] = []
    # 1. UNC check on un-collapsed text (avoids matching LaTeX \\alpha which has no second \)
    if _UNC_PATH_RAW_RE.search(text):
        leaks.append(f"HOST_PATH_LEAK:{name}:unc")
    norm = normalise_for_scan(text)
    # 2. Drive / POSIX-home / MSYS / WSL / file:// check on normalized text
    if not leaks and _HOST_PATH_NORM_RE.search(norm):
        leaks.append(f"HOST_PATH_LEAK:{name}:path")
    # 3. Literal workspace root / home dir / username segment checks
    if not leaks:
        norm_lower = norm.lower()
        for lit in literals:
            if lit in norm_lower:
                leaks.append(f"HOST_PATH_LEAK:{name}:root")
                break
    if not leaks and user:
        if re.search(rf"(?i)(?<=[\\/]){re.escape(user)}(?=[\\/])", norm):
            leaks.append(f"HOST_PATH_LEAK:{name}:user")
    # 4. Secrets
    for pat in _SECRET_PATTERNS:
        if pat.search(text):
            leaks.append(f"SECRET_LEAK:{name}")
            break
    return leaks


def _scan_binary_member(name: str, raw: bytes, literals: list[str], user: str | None) -> list[str]:
    """Scan printable ASCII and UTF-16LE strings of a binary member for host root/home/user only."""
    leaks: list[str] = []
    ascii_chunks = [m.group(0).decode("ascii", errors="ignore") for m in re.finditer(rb"[\x20-\x7e]{4,}", raw)]
    utf16_chunks: list[str] = []
    for m in re.finditer(rb"(?:[\x20-\x7e]\x00){4,}", raw):
        try:
            utf16_chunks.append(m.group(0).decode("utf-16-le", errors="ignore"))
        except Exception:
            pass
    combined = "\n".join(ascii_chunks + utf16_chunks)
    if not combined:
        return leaks
    norm = normalise_for_scan(combined)
    norm_lower = norm.lower()
    for lit in literals:
        if lit in norm_lower:
            leaks.append(f"BINARY_HOST_PATH_LEAK:{name}:root")
            return leaks
    if user and re.search(rf"(?i)(?<=[\\/]){re.escape(user)}(?=[\\/])", norm):
        leaks.append(f"BINARY_HOST_PATH_LEAK:{name}:user")
    return leaks


def scan_archive_for_leaks(archive_path: Path, *, workspace_root: Path | None = None) -> list[str]:
    """Scan all text and binary members in a zip archive for host paths, secrets, and forbidden bytecode."""
    leaks: list[str] = []
    literals, user = _host_literals(workspace_root)
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
                leaks.extend(_scan_binary_member(name, raw, literals, user))
                continue
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue
            leaks.extend(_scan_text_member(name, text, literals, user))
    return leaks


def write_scanned_zip(
    members: Mapping[str, bytes | str | Path],
    archive_path: Path,
    *,
    workspace_root: Path | None = None,
) -> Path:
    """Single trust-boundary zip writer used by Builder, Workspace, Vault, and CLI exports."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel_name in sorted(members.keys()):
                rel_posix = rel_name.replace("\\", "/")
                if _is_excluded_relpath(rel_posix):
                    continue
                src = members[rel_name]
                if isinstance(src, Path):
                    if src.is_symlink():
                        raise ExportBlockedError("EXPORT_UNSAFE_SYMLINK", f"Unsafe symlink: {src.name}", status_code=400)
                    zf.write(src, rel_posix)
                elif isinstance(src, bytes):
                    zf.writestr(rel_posix, src)
                else:
                    zf.writestr(rel_posix, str(src).encode("utf-8"))
        leaks = scan_archive_for_leaks(archive_path, workspace_root=workspace_root)
        if leaks:
            offending = sorted({item.split(":")[1] for item in leaks if ":" in item})
            raise ExportBlockedError(
                "EXPORT_TRUST_BOUNDARY_LEAK",
                f"Export blocked due to trust-boundary leaks in members: {', '.join(offending[:10])}",
                status_code=409,
            )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path


def build_verified_export_zip(
    workspace: Path,
    archive_path: Path,
    *,
    build_status: str | None,
    cache_manifest: Any = None,
    workspace_root: Path | None = None,
) -> Path:
    """Create an allow-listed, receipt-verified, leak-scanned export zip archive."""
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

    members: dict[str, Path] = {}
    for path in sorted(ws.rglob("*")):
        if path.is_symlink():
            raise ExportBlockedError("EXPORT_UNSAFE_SYMLINK", f"Unsafe symlink: {path.name}", status_code=400)
        if not path.is_file():
            continue
        rel_posix = path.relative_to(ws).as_posix()
        if _is_excluded_relpath(rel_posix):
            continue
        members[rel_posix] = path
    return write_scanned_zip(members, archive_path, workspace_root=workspace_root or ws)

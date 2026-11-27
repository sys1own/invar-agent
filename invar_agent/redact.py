"""Portable URI normalization and recursive host-path sanitization (UPG-014, BUG-006, NF-15)."""
from __future__ import annotations

import getpass
from pathlib import Path
import re
from typing import Any


_DRIVE_PATH_RE = re.compile(r"(?i)(?<![A-Za-z])[A-Z]:(?:\\+|/)[^\s'\"`,;)\]}]+")
_POSIX_HOME_RE = re.compile(r"(?<![\w.])/(?:[a-z]/)?(?:Users|home)/[^\s'\"`,;)\]}]+")
_FILE_URI_RE = re.compile(r"(?i)\bfile://[^\s'\"`,;)\]}]+")
_UNC_PATH_RE = re.compile(r"\\\\[A-Za-z0-9_.$-]+\\[A-Za-z0-9_.$-]+(?:\\[^\s'\"`,;)\]}]*)*")


def _clean_single_path(raw: str, workspace_root: Path | None = None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    # Unescape doubled backslashes for matching
    un = re.sub(r"\\{2,}", r"\\", text)
    posix = un.replace("\\", "/")
    if posix.lower().startswith("file:///"):
        posix = posix[8:]
    elif posix.lower().startswith("file://"):
        posix = posix[7:]

    # Check vault path marker
    vm = re.search(r"\.invar/research_vault/([^/]+)/(.+)$", posix)
    if vm:
        return f"vault:{vm.group(1)}/{vm.group(2)}"

    # Check workspace root
    if workspace_root is not None:
        try:
            ws = Path(workspace_root).resolve()
            p = Path(un)
            if p.is_absolute():
                try:
                    rel = p.resolve().relative_to(ws)
                    return rel.as_posix()
                except Exception:
                    pass
            ws_posix = ws.as_posix().rstrip("/") + "/"
            if posix.lower().startswith(ws_posix.lower()):
                return posix[len(ws_posix):]
        except Exception:
            pass

    for marker in ("uploads/", "src/", "provenance/", "tests/", "diagnostics/"):
        idx = posix.find(marker)
        if idx >= 0:
            return posix[idx:]

    if re.match(r"^[A-Za-z]:/", posix) or posix.startswith("/") or text.startswith("\\\\"):
        name = Path(posix).name
        return name or "redacted_path"
    return posix


def to_portable_uri(
    path: str | Path,
    workspace_root: str | Path | None = None,
    *,
    vault_run_id: str | None = None,
) -> str:
    """Convert any local filesystem path into a portable workspace-relative, vault:, or basename URI."""
    raw = str(path or "").strip()
    if not raw:
        return ""
    if raw.startswith("vault:"):
        return raw
    if vault_run_id:
        rel = Path(raw.replace("\\", "/")).name
        if workspace_root is not None:
            try:
                ws = Path(workspace_root).resolve()
                vdir = (ws / ".invar" / "research_vault" / vault_run_id).resolve()
                p = Path(raw).resolve()
                if p.is_relative_to(vdir):
                    rel = p.relative_to(vdir).as_posix()
            except Exception:
                pass
        return f"vault:{vault_run_id}/{rel}"
    ws_path = Path(workspace_root) if workspace_root is not None else None
    return _clean_single_path(raw, ws_path)


def _sanitize_string(text: str, workspace_root: Path | None = None) -> str:
    if not text:
        return text
    out = text
    # Replace workspace root literally first (both slash styles & escaped)
    if workspace_root is not None:
        try:
            ws = str(Path(workspace_root).resolve())
            for variant in (ws, ws.replace("\\", "/"), ws.replace("\\", "\\\\")):
                if variant and variant in out:
                    out = out.replace(variant, ".")
        except Exception:
            pass
    # Replace home directory literally
    try:
        home = str(Path.home().resolve())
        for variant in (home, home.replace("\\", "/"), home.replace("\\", "\\\\")):
            if variant and variant in out:
                out = out.replace(variant, "<HOME>")
    except Exception:
        pass
    # Replace embedded file:// URIs, drive paths, POSIX/MSYS home paths, and UNC paths
    out = _FILE_URI_RE.sub(lambda m: _clean_single_path(m.group(0), workspace_root), out)
    out = _DRIVE_PATH_RE.sub(lambda m: _clean_single_path(m.group(0), workspace_root), out)
    out = _POSIX_HOME_RE.sub(lambda m: _clean_single_path(m.group(0), workspace_root), out)
    # UNC paths: only match when not preceded by backslash/word (avoid LaTeX \\alpha)
    out = _UNC_PATH_RE.sub(lambda m: Path(m.group(0).replace("\\", "/")).name or "redacted_unc", out)
    # Also redact bare username if it appears as a path segment
    try:
        user = getpass.getuser()
        if user and len(user) >= 4:
            out = re.sub(
                rf"(?i)(?<=[\\/]){re.escape(user)}(?=[\\/])",
                "<USER>",
                out,
            )
    except Exception:
        pass
    return out


def sanitize_tree(obj: Any, workspace_root: str | Path | None = None) -> Any:
    """Recursively sanitize every dict, list, tuple, and string in `obj` to remove host paths."""
    ws = Path(workspace_root) if workspace_root is not None else None

    def _walk(node: Any) -> Any:
        if isinstance(node, str):
            return _sanitize_string(node, ws)
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(_walk(item) for item in node)
        return node

    return _walk(obj)

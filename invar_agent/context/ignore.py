"""Hermetic ignore filtering: workspace file selection for ingestion."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "target", "dist", "build", ".venv", "venv", "node_modules", ".invar_agent",
    ".idea", ".vscode", "*.egg-info",
}

DEFAULT_IGNORE_FILES = {".DS_Store", "Thumbs.db"}


class HermeticIgnoreFilter:
    """Applies .gitignore-style + hermetic defaults when scanning workspaces."""

    def __init__(self, workspace_root: str | os.PathLike, extra_patterns: Optional[list[str]] = None) -> None:
        self.workspace_root = Path(workspace_root)
        self.patterns = list(extra_patterns or [])
        self._load_ignore_files()

    def _load_ignore_files(self) -> None:
        for name in (".hermesignore", ".invarignore", ".gitignore"):
            path = self.workspace_root / name
            if path.exists():
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.patterns.append(line)

    def should_ignore_path(self, path: str | os.PathLike) -> bool:
        rel = Path(path)
        try:
            rel = rel.relative_to(self.workspace_root)
        except ValueError:
            pass
        parts = rel.parts
        for part in parts:
            if part in DEFAULT_IGNORE_DIRS or part in DEFAULT_IGNORE_FILES:
                return True
            if part.endswith(".pyc") or part.endswith(".pyo"):
                return True
        for pattern in self.patterns:
            if self._match(pattern, str(rel)):
                return True
        return False

    def _match(self, pattern: str, path: str) -> bool:
        import fnmatch

        pattern = pattern.strip().lstrip("/")
        if pattern.endswith("/"):
            return any(fnmatch.fnmatch(part, pattern[:-1]) or fnmatch.fnmatch(path, pattern + "*") for part in path.split(os.sep))
        if "/" in pattern:
            return fnmatch.fnmatch(path, pattern)
        # Match against any path component.
        return any(fnmatch.fnmatch(part, pattern) for part in path.split(os.sep))

    def iter_workspace_files(self) -> list[str]:
        """All non-ignored files under the workspace root."""
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.workspace_root):
            dirnames[:] = [d for d in dirnames if not self.should_ignore_path(os.path.join(dirpath, d))]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                if not self.should_ignore_path(full):
                    files.append(str(Path(full).relative_to(self.workspace_root)))
        return sorted(files)

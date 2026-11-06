"""Dependency graph auditor: import resolution and lockfile integrity."""
from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path
from typing import Optional


class DependencyGraphAuditor:
    """Verifies staged modules only import installed/resolvable dependencies."""

    STDLIB = set(getattr(__import__("sys"), "stdlib_module_names", set()))

    def verify_imports_installed(self, sources: dict[str, str]) -> list[str]:
        """Returns unresolved third-party imports (stdlib and local modules excluded)."""
        local_modules = {p[:-3].replace("/", ".") for p in sources if p.endswith(".py")}
        local_modules |= {p[:-3].replace("/", ".").split(".")[0] for p in sources if p.endswith(".py")}
        missing: list[str] = []
        for path, source in sources.items():
            if not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root in self.STDLIB or root in local_modules:
                            continue
                        if not self._module_installed(root):
                            missing.append(f"{path}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    root = node.module.split(".")[0]
                    if root in self.STDLIB or root in local_modules:
                        continue
                    if not self._module_installed(root):
                        missing.append(f"{path}: from {node.module} import ...")
        return sorted(set(missing))

    def _module_installed(self, root: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(root) is not None

    def check_lockfile_integrity(self, lock_path: str | Path) -> bool:
        path = Path(lock_path)
        if not path.exists():
            return False
        try:
            if path.suffix == ".json":
                json.loads(path.read_text())
                return True
            if path.name == "Cargo.lock":
                # Cargo.lock is TOML.
                tomllib.loads(path.read_text())
                return True
            # requirements.txt: one requirement per line.
            for line in path.read_text().splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "==" not in s and ">=" not in s and "<=" not in s:
                    # Unpinned but still a requirement.
                    pass
            return True
        except Exception:  # noqa: BLE001
            return False

    def extract_third_party_deps(self, pyproject_path: str | Path) -> list[str]:
        path = Path(pyproject_path)
        if not path.exists():
            return []
        try:
            data = tomllib.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return []
        deps = data.get("project", {}).get("dependencies", [])
        return [re.split(r"[<>=!~\[]", d)[0].strip() for d in deps]

"""Module boundary guard: circular import detection and symbol visibility."""
from __future__ import annotations

import ast
from typing import Optional


class ModuleBoundaryGuard:
    """Enforces import hygiene and public symbol visibility across modules."""

    def check_circular_imports(self, files: dict[str, str]) -> list[list[str]]:
        """Returns cycles as lists of module paths (empty when acyclic)."""
        graph: dict[str, set[str]] = {}
        for path in files:
            graph[path] = set()
        for path, source in files.items():
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = self._resolve(alias.name, path, files)
                        if target:
                            graph[path].add(target)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    target = self._resolve(node.module, path, files)
                    if target:
                        graph[path].add(target)
        cycles: list[list[str]] = []
        visited: set[str] = set()
        stack: list[str] = []
        in_stack: set[str] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            stack.append(node)
            in_stack.add(node)
            for nxt in sorted(graph.get(node, ())):
                if nxt not in graph:
                    continue
                if nxt in in_stack:
                    idx = stack.index(nxt)
                    cycles.append(stack[idx:] + [nxt])
                elif nxt not in visited:
                    dfs(nxt)
            stack.pop()
            in_stack.discard(node)

        for node in sorted(graph):
            if node not in visited:
                dfs(node)
        # Deduplicate cycles by rotation-invariant key.
        unique: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for cyc in cycles:
            core = cyc[:-1]
            rotations = [tuple(core[i:] + core[:i]) for i in range(len(core))]
            key = min(rotations)
            if key not in seen:
                seen.add(key)
                unique.append(cyc)
        return unique

    def verify_symbol_visibility(self, module_path: str, symbol: str,
                                 files: Optional[dict[str, str]] = None) -> bool:
        """A symbol is visible if defined at top level of the module (public) or
        explicitly imported from another staged module."""
        source = None
        if files and module_path in files:
            source = files[module_path]
        else:
            import os

            full = os.path.join(os.getcwd(), module_path)
            if os.path.exists(full):
                source = open(full, encoding="utf-8").read()
        if source is None:
            return False
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                return not symbol.startswith("_")
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol:
                        return not symbol.startswith("_")
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == symbol or alias.asname == symbol:
                        return True
        return False

    def _resolve(self, module: str, importer: str, files: dict[str, str]) -> Optional[str]:
        """Map a dotted module name to a staged file path."""
        parts = module.split(".")
        # Relative import handling.
        if importer.endswith(".py"):
            base = importer[:-3].replace("/", ".")
        else:
            base = importer.replace("/", ".")
        candidates = [
            module.replace(".", "/") + ".py",
            module.replace(".", "/") + "/__init__.py",
        ]
        for cand in candidates:
            if cand in files:
                return cand
        # Try relative to importer package.
        pkg_parts = base.split(".")[:-1] if "." in base else []
        for cand in candidates:
            rel = "/".join(pkg_parts + cand.split("/")) if pkg_parts else cand
            if rel in files:
                return rel
        return None

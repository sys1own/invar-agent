"""Workspace AST inspector: symbol tables, imports, call hierarchies."""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SymbolInfo:
    name: str
    kind: str  # "function" | "class" | "assignment"
    lineno: int
    is_public: bool


@dataclass
class ModuleSymbolTable:
    module_path: str
    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    call_edges: list[tuple[str, str]] = field(default_factory=list)  # caller -> callee

    def symbol_names(self) -> list[str]:
        return [s.name for s in self.symbols]

    def public_symbols(self) -> list[str]:
        return [s.name for s in self.symbols if s.is_public]


class WorkspaceASTInspector:
    """Extracts symbol tables and call hierarchies from workspace sources."""

    def inspect_file(self, path: str, source: str) -> ModuleSymbolTable:
        table = ModuleSymbolTable(module_path=path)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return table
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                table.symbols.append(SymbolInfo(node.name, "function", node.lineno, not node.name.startswith("_")))
            elif isinstance(node, ast.ClassDef):
                table.symbols.append(SymbolInfo(node.name, "class", node.lineno, not node.name.startswith("_")))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        table.symbols.append(SymbolInfo(target.id, "assignment", node.lineno, not target.id.startswith("_")))
            elif isinstance(node, ast.Import):
                table.imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                table.imports.append(node.module)
        # Call hierarchy within the module.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller = node.name
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        callee = self._call_name(sub.func)
                        if callee:
                            table.call_edges.append((caller, callee))
        return table

    def extract_symbol_table(self, files: dict[str, str]) -> dict[str, ModuleSymbolTable]:
        return {path: self.inspect_file(path, source) for path, source in files.items()}

    def map_call_hierarchy(self, files: dict[str, str]) -> dict[str, list[str]]:
        """caller -> sorted unique callees across the workspace."""
        hierarchy: dict[str, set[str]] = {}
        for path, source in files.items():
            if not path.endswith(".py"):
                continue
            table = self.inspect_file(path, source)
            for caller, callee in table.call_edges:
                hierarchy.setdefault(caller, set()).add(callee)
        return {k: sorted(v) for k, v in sorted(hierarchy.items())}

    def _call_name(self, func: ast.expr) -> Optional[str]:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            cur = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
        return None

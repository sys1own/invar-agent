"""Deterministic import pruner: removes unused imports, canonicalizes ordering."""
from __future__ import annotations

import ast


class DeterministicImportPruner:
    """0-token import hygiene transducer (stdlib-only, no LLM)."""

    def prune_unused_imports(self, source: str) -> str:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        used_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # For `import a.b` usage `a.b.c`, the root Name is already recorded.
                pass
        lines = source.splitlines(keepends=True)
        remove_lines: set[int] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                keep_aliases = []
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    bound = alias.asname or root
                    if bound in used_names or alias.name in used_names:
                        keep_aliases.append(alias)
                if not keep_aliases:
                    for i in range(node.lineno - 1, (node.end_lineno or node.lineno)):
                        remove_lines.add(i)
                elif len(keep_aliases) < len(node.names):
                    # Rewrite the import line with only used aliases.
                    newline_idx = node.lineno - 1
                    indent = lines[newline_idx][: len(lines[newline_idx]) - len(lines[newline_idx].lstrip())]
                    names = ", ".join(a.asname and f"{a.name} as {a.asname}" or a.name for a in keep_aliases)
                    lines[newline_idx] = f"{indent}import {names}\n"
            elif isinstance(node, ast.ImportFrom):
                keep_names = []
                for alias in node.names:
                    bound = alias.asname or alias.name
                    if bound in used_names or alias.name == "*":
                        keep_names.append(alias)
                if not keep_names:
                    for i in range(node.lineno - 1, (node.end_lineno or node.lineno)):
                        remove_lines.add(i)
                elif len(keep_names) < len(node.names):
                    newline_idx = node.lineno - 1
                    indent = lines[newline_idx][: len(lines[newline_idx]) - len(lines[newline_idx].lstrip())]
                    names = ", ".join(a.asname and f"{a.name} as {a.asname}" or a.name for a in keep_names)
                    module = node.module or ""
                    lines[newline_idx] = f"{indent}from {module} import {names}\n"
        out = [l for i, l in enumerate(lines) if i not in remove_lines]
        return "".join(out)

    def sort_and_canonicalize_imports(self, source: str) -> str:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        import_nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        if not import_nodes:
            return source
        lines = source.splitlines(keepends=True)
        # Collect import line ranges.
        import_lines: list[str] = []
        remove: set[int] = set()
        for node in import_nodes:
            start = node.lineno - 1
            end = (node.end_lineno or node.lineno) - 1
            # Handle parenthesized multi-line from-imports: end_lineno covers them.
            segment = "".join(lines[start : end + 1])
            import_lines.append(segment.strip())
            for i in range(start, end + 1):
                remove.add(i)

        def sort_key(text: str):
            is_from = text.startswith("from ")
            body = text[5:] if is_from else text[7:]
            module = body.split(" import")[0].split(" ")[0] if is_from else body.split(",")[0].split(" ")[0]
            return (is_from, module, text)

        canonical = sorted(import_lines, key=sort_key)
        # Reinsert at the position of the first import.
        first_idx = min(n.lineno - 1 for n in import_nodes)
        out_lines = [l for i, l in enumerate(lines) if i not in remove]
        # Adjust insertion index after removals.
        removed_before = sum(1 for i in remove if i < first_idx)
        insert_at = first_idx - removed_before
        block = "\n".join(canonical) + "\n"
        out_lines.insert(insert_at, block)
        return "".join(out_lines)

    def canonicalize(self, source: str, prune: bool = True) -> str:
        if prune:
            source = self.prune_unused_imports(source)
        return self.sort_and_canonicalize_imports(source)

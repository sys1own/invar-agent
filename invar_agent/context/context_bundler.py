"""Holographic context bundler: AST skeleton compression and HIN relevance ranking."""
from __future__ import annotations

import ast
from typing import Optional


class HolographicContextBundler:
    """Compresses sources to dependency skeletons for prompt budgets."""

    def compress_ast_skeletons(self, sources: dict[str, str], max_chars_per_file: int = 2000) -> str:
        """Emits imports + signatures only (bodies stripped)."""
        sections: list[str] = []
        for path in sorted(sources):
            source = sources[path]
            skeleton = self._skeleton(source)
            if len(skeleton) > max_chars_per_file:
                skeleton = skeleton[:max_chars_per_file] + "\n# ... [skeleton truncated] ..."
            sections.append(f"### {path}\n{skeleton}")
        return "\n\n".join(sections)

    def _skeleton(self, source: str) -> str:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source[:500]
        lines = source.splitlines()
        out: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                out.append(ast.get_source_segment(source, node) or "")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                header = self._signature_line(lines, node)
                doc = ast.get_docstring(node)
                out.append(header)
                if doc:
                    out.append(f'    """{doc.splitlines()[0]}"""')
                out.append("    ...")
            elif isinstance(node, ast.ClassDef):
                header = self._signature_line(lines, node)
                out.append(header)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append("    " + self._signature_line(lines, item))
                        out.append("        ...")
            elif isinstance(node, ast.Assign):
                seg = ast.get_source_segment(source, node) or ""
                out.append(seg.splitlines()[0] if seg else "")
        return "\n".join(l for l in out if l is not None)

    def _signature_line(self, lines: list[str], node) -> str:
        lineno = node.lineno - 1
        end = node.end_lineno or node.lineno
        # Signature is up to the line ending with ':'
        for i in range(lineno, min(end, lineno + 10)):
            if lines[i].rstrip().endswith(":"):
                return "\n".join(lines[lineno : i + 1])
        return lines[lineno] if lineno < len(lines) else ""

    def rank_relevance_by_hin(
        self,
        query_terms: list[str],
        symbol_tables: dict[str, object],
    ) -> list[tuple[str, float]]:
        """Ranks modules by symbol-name overlap with query terms."""
        terms = {t.lower() for t in query_terms}
        scored: list[tuple[str, float]] = []
        for path, table in symbol_tables.items():
            names = getattr(table, "symbol_names", lambda: [])()
            score = 0.0
            for name in names:
                lowered = name.lower()
                for term in terms:
                    if term in lowered or lowered in term:
                        score += 1.0
            if score > 0:
                scored.append((path, score))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored

"""Three-way AST merger for concurrent wavefront branches."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

from .patch import unified_diff


@dataclass(frozen=True)
class MergeConflictNode:
    file_path: str
    symbol_name: str
    ours_text: str
    theirs_text: str
    reason: str = "concurrent modification"


@dataclass
class MergeResult:
    merged_source: str
    conflicts: list[MergeConflictNode] = field(default_factory=list)
    merged_symbols: list[str] = field(default_factory=list)


def _top_level_symbols(source: str) -> dict[str, tuple[int, int, str]]:
    """symbol name -> (start_line, end_line, kind) for top-level defs/classes."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines()
    symbols: dict[str, tuple[int, int, str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            # Include decorators.
            if getattr(node, "decorator_list", None):
                start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
            end = node.end_lineno or node.lineno
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            symbols[node.name] = (start, end, kind)
    return symbols


def _segment(lines: list[str], span: tuple[int, int] | tuple[int, int, str]) -> str:
    start, end = span[0], span[1]
    return "\n".join(lines[start:end]).rstrip() + "\n"


def _header_and_body(source: str) -> tuple[list[str], dict[str, tuple[int, int, str]]]:
    lines = source.splitlines()
    symbols = _top_level_symbols(source)
    if not symbols:
        return lines, {}
    first_start = min(s[0] for s in symbols.values())
    header = lines[:first_start]
    return header, symbols


def _imports_of(header_lines: list[str]) -> list[str]:
    imports = [l.rstrip() for l in header_lines if l.strip().startswith(("import ", "from "))]
    return sorted(set(imports))


def _other_header_lines(header_lines: list[str]) -> list[str]:
    return [l for l in header_lines if l.strip() and not l.strip().startswith(("import ", "from ", "#"))]


class ThreeWayASTMerger:
    """Merges two concurrent wavefront branches against a common base AST."""

    def merge_ast_branches(
        self,
        base_source: str,
        ours_source: str,
        theirs_source: str,
        file_path: str = "<memory>",
    ) -> MergeResult:
        base_lines = base_source.splitlines()
        ours_lines = ours_source.splitlines()
        theirs_lines = theirs_source.splitlines()

        base_header, base_syms = _header_and_body(base_source)
        ours_header, ours_syms = _header_and_body(ours_source)
        theirs_header, theirs_syms = _header_and_body(theirs_source)

        conflicts: list[MergeConflictNode] = []
        merged_symbols: list[str] = []

        # Union of import lines across all three headers, deduplicated & sorted.
        import_lines = sorted(set(_imports_of(base_header) + _imports_of(ours_header) + _imports_of(theirs_header)))

        # Order symbols: base order first, then ours-only, then theirs-only.
        ordered_names: list[str] = []
        for name in ours_syms:
            if name not in ordered_names:
                ordered_names.append(name)
        for name in theirs_syms:
            if name not in ordered_names:
                ordered_names.append(name)
        # Stable base ordering preference.
        base_order = [n for n in base_syms if n in ordered_names]
        ordered_names = base_order + [n for n in ordered_names if n not in base_order]

        symbol_blocks: list[str] = []
        for name in ordered_names:
            base_seg = _segment(base_lines, base_syms[name]) if name in base_syms else None
            ours_seg = _segment(ours_lines, ours_syms[name]) if name in ours_syms else None
            theirs_seg = _segment(theirs_lines, theirs_syms[name]) if name in theirs_syms else None

            ours_changed = ours_seg is not None and ours_seg != base_seg
            theirs_changed = theirs_seg is not None and theirs_seg != base_seg

            if ours_changed and theirs_changed:
                if ours_seg == theirs_seg:
                    chosen = ours_seg
                else:
                    conflicts.append(MergeConflictNode(
                        file_path=file_path, symbol_name=name,
                        ours_text=ours_seg or "", theirs_text=theirs_seg or "",
                    ))
                    chosen = ours_seg  # deterministic: ours wins, conflict recorded
            elif ours_changed:
                chosen = ours_seg
            elif theirs_changed:
                chosen = theirs_seg
            elif ours_seg is not None:
                chosen = ours_seg
            elif theirs_seg is not None:
                chosen = theirs_seg
            else:
                chosen = base_seg
            if chosen:
                symbol_blocks.append(chosen.rstrip("\n"))
                merged_symbols.append(name)

        # Non-import, non-def header content (module docstring, __future__, etc.)
        # taken from ours (arbitrary but deterministic).
        doc_lines = _other_header_lines(ours_header) or _other_header_lines(base_header)

        out_lines: list[str] = []
        out_lines.extend(doc_lines)
        out_lines.extend(import_lines)
        out_lines.append("")
        body = "\n\n".join(symbol_blocks)
        merged = "\n".join(out_lines).rstrip("\n") + "\n\n" + body + ("\n" if body else "")
        return MergeResult(merged_source=merged, conflicts=conflicts, merged_symbols=merged_symbols)

    def merge_file_triple(self, base_path: str, ours: str, theirs: str) -> MergeResult:
        return self.merge_ast_branches(base, ours, theirs, file_path=base_path)

    def diff_summary(self, before: str, after: str, path: str) -> str:
        return unified_diff(before, after, path)

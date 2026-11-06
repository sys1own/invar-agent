"""AST-level surgical patch representation and application."""
from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

from invar_agent.contracts import ASTSurgicalPatch


@dataclass
class PatchApplicationResult:
    source: str
    applied: bool
    diagnostics: list[str] = field(default_factory=list)


def _symbol_span(lines: list[str], symbol: str) -> Optional[tuple[int, int]]:
    """Return (start, end) 0-based line span of a top-level def/class block."""
    pattern = re.compile(rf"^(?:async\s+def|def|class)\s+{re.escape(symbol)}\b")
    start = None
    base_indent = 0
    for i, line in enumerate(lines):
        if pattern.match(line):
            start = i
            base_indent = len(line) - len(line.lstrip())
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent:
            end = j
            break
    # Trim trailing blank lines.
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return start, end


def apply_patches_to_source(source: str, patches: list[ASTSurgicalPatch]) -> tuple[str, list[str]]:
    """Apply a sequence of ASTSurgicalPatches (same-file) to source text."""
    diagnostics: list[str] = []
    current = source
    for patch in patches:
        result = apply_single_patch(current, patch)
        if not result.applied:
            diagnostics.append(f"patch failed on {patch.file_path}: {result.diagnostics}")
            continue
        current = result.source
    return current, diagnostics


def apply_single_patch(source: str, patch: ASTSurgicalPatch) -> PatchApplicationResult:
    if patch.operation == "CREATE_FILE":
        return PatchApplicationResult(source=patch.replacement_code, applied=True)
    if patch.operation == "REPLACE_EXACT_SPAN":
        if not patch.search_block:
            return PatchApplicationResult(source=source, applied=False, diagnostics=["empty search_block"])
        count = source.count(patch.search_block)
        if count == 0:
            return PatchApplicationResult(source=source, applied=False, diagnostics=["search_block not found"])
        if count > 1:
            return PatchApplicationResult(source=source, applied=False, diagnostics=["search_block ambiguous"])
        return PatchApplicationResult(source=source.replace(patch.search_block, patch.replacement_code, 1), applied=True)

    lines = source.splitlines(keepends=True)
    span = _symbol_span(lines, patch.target_symbol or "")
    if span is None:
        return PatchApplicationResult(source=source, applied=False, diagnostics=["symbol not found"])
    start, end = span
    if patch.operation == "REPLACE_SYMBOL":
        new_block = patch.replacement_code.rstrip("\n") + "\n"
        new_lines = lines[:start] + [new_block] + lines[end:]
        return PatchApplicationResult(source="".join(new_lines), applied=True)
    if patch.operation == "INSERT_AFTER_SYMBOL":
        insert_at = end
        block = patch.replacement_code.rstrip("\n") + "\n"
        new_lines = lines[:insert_at] + ["\n", block] if block else lines[:insert_at] + [block]
        return PatchApplicationResult(source="".join(new_lines), applied=True)
    return PatchApplicationResult(source=source, applied=False, diagnostics=[f"unknown operation {patch.operation}"])


def compute_ast_delta(before: str, after: str) -> dict:
    """Structural delta summary between two source versions."""
    def symbols(src: str) -> dict[str, str]:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return {}
        out = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out[node.name] = ast.dump(node)
        return out

    b, a = symbols(before), symbols(after)
    added = sorted(set(a) - set(b))
    removed = sorted(set(b) - set(a))
    modified = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return {"added": added, "removed": removed, "modified": modified}


def invert_patch(patch: ASTSurgicalPatch, original_block: str = "") -> ASTSurgicalPatch:
    """Build the adjoint inverse of a single patch."""
    inverse_code = patch.adjoint_inverse_code or original_block
    if patch.operation == "CREATE_FILE":
        return ASTSurgicalPatch(
            file_path=patch.file_path,
            operation="REPLACE_SYMBOL",
            target_symbol=None,
            search_block=patch.replacement_code,
            replacement_code="",
            pre_hash=patch.post_hash,
            post_hash=patch.pre_hash,
            adjoint_inverse_code=patch.replacement_code,
        )
    return ASTSurgicalPatch(
        file_path=patch.file_path,
        operation="REPLACE_EXACT_SPAN",
        target_symbol=patch.target_symbol,
        search_block=patch.replacement_code,
        replacement_code=inverse_code,
        pre_hash=patch.post_hash,
        post_hash=patch.pre_hash,
        adjoint_inverse_code=patch.replacement_code,
    )


def unified_diff(before: str, after: str, path: str = "") -> str:
    return "".join(
        difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                             fromfile=f"a/{path}", tofile=f"b/{path}")
    )

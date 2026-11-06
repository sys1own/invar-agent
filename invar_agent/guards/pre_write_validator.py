"""Pre-write validator: rejects stubs, TODOs, and placeholder bodies."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationViolation:
    path: str
    line: int
    rule: str
    snippet: str


class PreWriteValidator:
    """Static hygiene gate run on every staged buffer before commit."""

    _TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
    _STUB_RE = re.compile(r"^\s*(\.\.\.|pass)\s*$")

    def validate_staged_buffer(self, path: str, source: str) -> list[ValidationViolation]:
        violations: list[ValidationViolation] = []
        lines = source.splitlines()
        is_rust = path.endswith(".rs")
        is_python = path.endswith(".py")

        for lineno, line in enumerate(lines, start=1):
            if self._TODO_RE.search(line):
                violations.append(ValidationViolation(path, lineno, "todo_marker", line.strip()[:120]))
            if is_rust and re.search(r"\b(unimplemented!|todo!)\s*\(", line):
                violations.append(ValidationViolation(path, lineno, "rust_unimplemented", line.strip()[:120]))

        if is_python:
            violations.extend(self._python_stub_violations(path, source, lines))
        return violations

    def _python_stub_violations(self, path: str, source: str, lines: list[str]) -> list[ValidationViolation]:
        violations: list[ValidationViolation] = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return violations  # syntax guard reports this separately
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
                # Docstring-only or pass/ellipsis-only bodies are stubs.
                meaningful = [
                    s for s in body
                    if not (isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)))
                ]
                if node.body and not meaningful:
                    # Only docstring/pass/ellipsis: stub unless it's a trivial property.
                    has_doc = bool(body) and isinstance(body[0], ast.Expr)
                    rest = body[1:] if has_doc else body
                    if all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) for s in rest):
                        lineno = node.lineno
                        violations.append(ValidationViolation(path, lineno, "pass_only_body", lines[lineno - 1].strip()[:120]))
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                        fn = stmt.exc.func
                        name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
                        if name == "NotImplementedError":
                            violations.append(ValidationViolation(path, stmt.lineno, "not_implemented_error", lines[stmt.lineno - 1].strip()[:120]))
        return violations

    def is_valid(self, path: str, source: str) -> bool:
        return not self.validate_staged_buffer(path, source)

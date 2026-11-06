"""Chiasmus Symmetry Verifier: mirror-invariant lifecycle pairing over AST paths."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ChiasticFrame:
    """A mirror-symmetric operation pair, e.g. (lock.acquire, lock.release)."""

    operation_pairs: tuple[tuple[str, str], ...] = (("lock.acquire", "lock.release"),)


@dataclass(frozen=True)
class ChiasmusViolation:
    path_id: str
    operation_sequence: tuple[str, ...]
    reason: str


@dataclass
class ChiasmusResult:
    symmetric: bool
    frames_checked: int = 0
    paths_checked: int = 0
    violations: list[ChiasmusViolation] = field(default_factory=list)

    @property
    def diagnostics(self) -> list[str]:
        return [f"{v.path_id}: {v.reason} (ops={list(v.operation_sequence)})" for v in self.violations]


@dataclass
class _Event:
    kind: str  # "acquire" | "release"
    pair_index: int
    name: str


@dataclass
class _Path:
    events: list[_Event]
    exit_kind: str  # "normal" | "return" | "raise"


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _call_name(call: ast.Call) -> str:
    return _dotted_name(call.func)


def _matches(entry: str, dotted: str) -> bool:
    return dotted == entry or dotted.endswith("." + entry) or dotted.endswith(entry)


class ChiasmusSymmetryVerifier:
    """Verifies every exit path executes acquisitions/releases in mirror order."""

    def verify_mirror_invariants(
        self,
        source: str,
        operation_pairs: Optional[list[tuple[str, str]]] = None,
    ) -> ChiasmusResult:
        pairs = tuple(operation_pairs) if operation_pairs else ChiasticFrame().operation_pairs
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return ChiasmusResult(
                symmetric=False,
                violations=[ChiasmusViolation("<parse>", (), f"syntax error: {e}")],
            )
        result = ChiasmusResult(symmetric=True)
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not functions:
            # Module-level path check.
            paths = self._paths_from_stmts(tree.body, pairs)
            result.paths_checked += len(paths)
            result.frames_checked += 1
            self._check_paths(paths, "<module>", result)
            return result
        for fn in functions:
            result.frames_checked += 1
            paths = self._paths_from_stmts(fn.body, pairs)
            result.paths_checked += len(paths)
            self._check_paths(paths, fn.name, result)
        result.symmetric = not result.violations
        return result

    def check_resource_duality(self, events: list[str], operation_pairs: list[tuple[str, str]]) -> bool:
        """Pure sequence check: every release must mirror the most recent acquire."""
        stack: list[str] = []
        for ev in events:
            kind = None
            for acq, rel in operation_pairs:
                if _matches(acq, ev):
                    kind = "acquire"
                    break
                if _matches(rel, ev):
                    kind = "release"
                    break
            if kind == "acquire":
                stack.append(ev)
            elif kind == "release":
                # Find matching acquire anywhere in stack; must be on top for mirror order.
                if not stack:
                    return False
                stack.pop()
        return not stack

    # -- internals ---------------------------------------------------------------

    def _event_for_call(self, call: ast.Call, pairs) -> Optional[_Event]:
        name = _call_name(call)
        for idx, (acq, rel) in enumerate(pairs):
            if _matches(acq, name):
                return _Event("acquire", idx, name)
            if _matches(rel, name):
                return _Event("release", idx, name)
        return None

    def _events_in_stmt(self, stmt: ast.stmt, pairs) -> list[_Event]:
        events: list[_Event] = []
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                ev = self._event_for_call(node, pairs)
                if ev is not None:
                    events.append(ev)
        return events

    def _paths_from_stmts(self, stmts: list[ast.stmt], pairs) -> list[_Path]:
        paths: list[_Path] = [_Path([], "normal")]
        for stmt in stmts:
            new_paths: list[_Path] = []
            for path in paths:
                if path.exit_kind != "normal":
                    # Path already terminated (return/raise): subsequent
                    # statements are unreachable on this path.
                    new_paths.append(path)
                    continue
                new_paths.extend(self._paths_from_stmt(stmt, path, pairs))
            paths = new_paths
            if not paths:
                break
        return paths

    def _paths_from_stmt(self, stmt: ast.stmt, path: _Path, pairs) -> list[_Path]:
        # Terminal statements end the path.
        if isinstance(stmt, (ast.Return, ast.Raise)):
            evs = self._events_in_stmt(stmt, pairs)
            return [_Path(path.events + evs, "return" if isinstance(stmt, ast.Return) else "raise")]

        if isinstance(stmt, ast.If):
            body_paths = self._paths_from_stmts(stmt.body, pairs)
            else_paths = self._paths_from_stmts(stmt.orelse, pairs) if stmt.orelse else [_Path([], "normal")]
            out: list[_Path] = []
            for bp in body_paths:
                out.append(_Path(path.events + bp.events, bp.exit_kind))
            for ep in else_paths:
                out.append(_Path(path.events + ep.events, ep.exit_kind))
            return out

        if isinstance(stmt, ast.Try):
            body_paths = self._paths_from_stmts(stmt.body, pairs)
            handler_paths: list[_Path] = []
            for handler in stmt.handlers:
                handler_paths.extend(self._paths_from_stmts(handler.body, pairs))
            if not handler_paths:
                handler_paths = [_Path([], "normal")]
            finally_paths = self._paths_from_stmts(stmt.finalbody, pairs) if stmt.finalbody else [_Path([], "normal")]
            out: list[_Path] = []
            for base in body_paths + handler_paths:
                for fp in finally_paths:
                    out.append(_Path(path.events + base.events + fp.events, base.exit_kind))
            return out

        if isinstance(stmt, (ast.For, ast.While, ast.AsyncFor, ast.With, ast.AsyncWith)):
            inner = stmt.body
            orelse = getattr(stmt, "orelse", [])
            body_paths = self._paths_from_stmts(inner, pairs)
            else_paths = self._paths_from_stmts(orelse, pairs) if orelse else [_Path([], "normal")]
            out: list[_Path] = []
            for bp in body_paths:
                for ep in else_paths:
                    out.append(_Path(path.events + bp.events + ep.events, bp.exit_kind if bp.exit_kind != "normal" else ep.exit_kind))
            if not out:
                out = [_Path(path.events + self._events_in_stmt(stmt, pairs), path.exit_kind)]
            return out

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Nested definitions do not execute their bodies inline.
            return [_Path(path.events, path.exit_kind)]

        events = self._events_in_stmt(stmt, pairs)
        return [_Path(path.events + events, path.exit_kind)]

    def _check_paths(self, paths: list[_Path], frame_name: str, result: ChiasmusResult) -> None:
        for idx, path in enumerate(paths):
            path_id = f"{frame_name}#path{idx}"
            violation = self._check_single_path(path)
            if violation is not None:
                seq = tuple(e.name for e in path.events)
                result.violations.append(
                    ChiasmusViolation(path_id=path_id, operation_sequence=seq, reason=violation)
                )

    def _check_single_path(self, path: _Path) -> Optional[str]:
        stack: list[_Event] = []
        for ev in path.events:
            if ev.kind == "acquire":
                stack.append(ev)
            else:
                # Mirror order: release must match the most recent unmatched acquire of the same pair.
                if not stack:
                    return f"release {ev.name} without matching acquire"
                top = stack[-1]
                if top.pair_index != ev.pair_index:
                    return f"non-mirror release {ev.name} while {top.name} is pending"
                stack.pop()
        if stack:
            pending = ", ".join(e.name for e in stack)
            return f"acquisition(s) not released on exit path ({path.exit_kind}): {pending}"
        return None

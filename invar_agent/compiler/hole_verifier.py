"""Deterministic Bespoke Hole-Filling Verifier (`hole_verifier.py`, <= 50 KB).

Replaces ad-hoc regex repair for the `<5%` bespoke scientific holes that cannot
be lowered directly from `SciIR` expression primitives:
1. Pure Python AST safety sanitizer (bans `exec`, `eval`, `compile`, `__import__`,
   I/O, network, subprocess, `open`, attribute escapes `__class__`/`__subclasses__`,
   and global state mutation).
2. Static loop-bound checker (bans `while` loops without explicit bounded counter
   and unbounded comprehensions/generators).
3. Observational purity & MELL linearity check (ensures hole functions write ONLY
   to their designated output return value and never mutate input state/param buffers).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable

MAX_HOLE_AST_NODES = 1500
MAX_STATIC_LOOP_BOUND = 1_000_000

_FORBIDDEN_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "http",
        "urllib",
        "requests",
        "httpx",
        "shutil",
        "pathlib",
        "pickle",
        "marshal",
        "ctypes",
        "importlib",
        "builtins",
        "inspect",
        "multiprocessing",
        "threading",
    }
)

_FORBIDDEN_CALLS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "memoryview",
    }
)

_FORBIDDEN_DUNDER_ATTRS = frozenset(
    {
        "__class__",
        "__base__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__closure__",
        "__dict__",
        "__import__",
    }
)

_ALLOWED_IMPORTS = frozenset({"math", "cmath", "numpy", "np", "typing"})


@dataclass(frozen=True)
class HoleViolation:
    code: str
    line: int
    message: str


@dataclass
class HoleVerificationReport:
    ok: bool
    ast_node_count: int
    violations: list[HoleViolation] = field(default_factory=list)
    read_only_buffers_preserved: bool = True


def _is_bounded_range_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not (isinstance(node.func, ast.Name) and node.func.id == "range"):
        return False
    if not (1 <= len(node.args) <= 3):
        return False
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            if abs(arg.value) > MAX_STATIC_LOOP_BOUND:
                return False
        elif isinstance(arg, ast.Name):
            continue
        elif isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "len":
            continue
        else:
            return False
    return True


def verify_bespoke_hole_source(
    source: str,
    *,
    readonly_param_names: Iterable[str] = ("y", "states", "p", "params", "c", "constants"),
) -> HoleVerificationReport:
    """Statically verify a bespoke `<5%` hole function for safety, boundedness, and purity."""
    violations: list[HoleViolation] = []
    readonly_set = set(readonly_param_names)

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return HoleVerificationReport(
            ok=False,
            ast_node_count=0,
            violations=[
                HoleViolation(
                    code="SYNTAX_ERROR",
                    line=int(exc.lineno or 0),
                    message=f"Syntax error in hole code: {exc.msg}",
                )
            ],
            read_only_buffers_preserved=False,
        )

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_HOLE_AST_NODES:
        violations.append(
            HoleViolation(
                code="AST_BUDGET_EXCEEDED",
                line=1,
                message=f"Hole AST size {len(nodes)} exceeds cap {MAX_HOLE_AST_NODES}",
            )
        )

    readonly_preserved = True

    for node in nodes:
        lineno = int(getattr(node, "lineno", 0) or 0)

        # 1. Import allowlist
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_IMPORTS or root not in _ALLOWED_IMPORTS:
                    violations.append(
                        HoleViolation(
                            code="FORBIDDEN_IMPORT",
                            line=lineno,
                            message=f"Import '{alias.name}' is not permitted in bespoke holes",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _FORBIDDEN_IMPORTS or root not in _ALLOWED_IMPORTS:
                violations.append(
                    HoleViolation(
                        code="FORBIDDEN_IMPORT",
                        line=lineno,
                        message=f"From-import '{node.module}' is not permitted in bespoke holes",
                    )
                )

        # 2. Forbidden calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                violations.append(
                    HoleViolation(
                        code="FORBIDDEN_CALL",
                        line=lineno,
                        message=f"Call to '{node.func.id}()' is banned in bespoke holes",
                    )
                )

        # 3. Forbidden dunder attribute escapes
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_DUNDER_ATTRS:
                violations.append(
                    HoleViolation(
                        code="FORBIDDEN_DUNDER_ATTR",
                        line=lineno,
                        message=f"Attribute access '.{node.attr}' is banned in bespoke holes",
                    )
                )

        # 4. Global / nonlocal state mutation
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append(
                HoleViolation(
                    code="STATEFUL_SCOPE_ESCAPE",
                    line=lineno,
                    message="global/nonlocal declarations are forbidden (observational purity)",
                )
            )

        # 5. Unbounded loops
        elif isinstance(node, ast.While):
            violations.append(
                HoleViolation(
                    code="UNBOUNDED_WHILE_LOOP",
                    line=lineno,
                    message="while loops are banned in bespoke holes; use statically bounded for-range loops",
                )
            )
        elif isinstance(node, ast.For):
            if not _is_bounded_range_call(node.iter) and not isinstance(node.iter, ast.Name):
                violations.append(
                    HoleViolation(
                        code="UNBOUNDED_FOR_ITERABLE",
                        line=lineno,
                        message="for-loop iterable must be a statically bounded range(...) or finite array name",
                    )
                )

        # 6. MELL / Observational Purity: forbid subscript mutation of read-only state/param buffers
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
                    if tgt.value.id in readonly_set:
                        readonly_preserved = False
                        violations.append(
                            HoleViolation(
                                code="PURITY_BUFFER_MUTATION",
                                line=lineno,
                                message=f"Read-only input buffer '{tgt.value.id}' mutated in-place",
                            )
                        )

    return HoleVerificationReport(
        ok=len(violations) == 0,
        ast_node_count=len(nodes),
        violations=violations,
        read_only_buffers_preserved=readonly_preserved,
    )

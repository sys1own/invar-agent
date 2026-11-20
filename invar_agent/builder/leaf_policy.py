"""Shared source capability contract for isolated extension slots."""
from __future__ import annotations

import ast

from .error_taxonomy import BuildFault


FORBIDDEN_IMPORTS = frozenset({"os", "sys", "ctypes", "subprocess", "socket", "importlib", "inspect", "builtins"})
LEAF_SYSTEM_PROMPT = (
    "Return only a complete, compact Python leaf module, without Markdown fences or prose. "
    "Preserve the exact frozen function name, ordered parameters, and output dict keys/types. "
    "Do not modify scientific kernels or other modules. Keep the extension self-contained. "
    "Use pure computations on supplied inputs; pure stdlib imports such as math, json, csv, io, html, "
    "statistics, collections, itertools, functools and re are allowed. Do not depend on optional third-party packages. "
    "No file IO, network, processes, system access, dynamic execution, introspection or dunder attributes. "
    "Forbidden imports: " + ", ".join(sorted(FORBIDDEN_IMPORTS)) + ". "
    "Private module imports (except the safe __future__ language-feature module), eval, exec, compile, __import__, and shell=True are forbidden. "
    "Copy entry_point_signature exactly: no extra optional/required parameters or variadic arguments. "
    "The entry point returns only required_output_keys, never markup or metadata keys; put those in auxiliary functions. "
    "For in-memory CSV/JSON adapters use io.StringIO, csv and json, not files. "
    "Return visualization markup as a string from an auxiliary pure function; never launch a server/browser."
)


def validate_source_capabilities(tree, *, leaf: bool, relpath: str):
    for item in ast.walk(tree):
        capability = None
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id in {"eval", "exec", "compile", "__import__"}:
            capability = item.func.id
        elif isinstance(item, ast.keyword) and item.arg == "shell" and isinstance(item.value, ast.Constant) and item.value.value is True:
            capability = "shell=True"
        elif leaf:
            imported = [alias.name for alias in item.names] if isinstance(item, ast.Import) else [item.module] if isinstance(item, ast.ImportFrom) and item.module else []
            forbidden = next((name for name in imported if name.split(".")[0] in FORBIDDEN_IMPORTS or name.startswith("_") and name != "__future__"), None)
            if forbidden:
                capability = "import " + forbidden
            elif isinstance(item, ast.Attribute) and (item.attr.startswith("__") or item.attr in FORBIDDEN_IMPORTS):
                capability = "attribute " + item.attr
        if capability:
            line, column = item.lineno, item.col_offset
            raise BuildFault("SANDBOX_SECURITY_VIOLATION", f"Forbidden capability {capability!r} at {relpath}:{line}:{column}",
                capability=capability, line=line, column=column, relpath=relpath)

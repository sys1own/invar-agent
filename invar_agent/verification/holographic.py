"""Holographic Boundary Register bridge: bulk AST -> 256-bit BLAKE3 boundary digest."""
from __future__ import annotations

import ast
from typing import Optional

try:
    from invar_agent._native import PyHisRegister as _NativeHis
except Exception:  # pragma: no cover - fallback when native ext is absent
    _NativeHis = None


class _PurePythonHisFallback:
    """Deterministic pure-Python mirror of the Rust HIS boundary register."""

    def __init__(self) -> None:
        import blake3

        self._blake3 = blake3
        self._modules: dict[str, str] = {}

    def update_module_boundary(self, module_path: str, canonical_signature: str) -> None:
        self._modules[module_path] = canonical_signature

    def project_boundary_digest(self) -> str:
        acc = bytearray(32)
        for path in sorted(self._modules):
            digest = self._blake3.blake3(self._modules[path].encode("utf-8")).digest()
            for i in range(32):
                acc[i] ^= digest[i]
        return bytes(acc).hex()

    def verify_boundary_parity(self, expected_digest: str) -> bool:
        return self.project_boundary_digest() == expected_digest.strip().lower()


def _make_register():
    if _NativeHis is not None:
        return _NativeHis()
    return _PurePythonHisFallback()


def extract_boundary_signature(source: str) -> str:
    """Canonical public-interface signature of a module.

    Private implementation bodies are excluded: only top-level public symbol
    names, kinds, and parameter names contribute, so internal edits never move
    the holographic boundary digest.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ""
    entries: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            entries.append(f"def {node.name}({','.join(args)})")
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                    margs = [a.arg for a in item.args.args + item.args.kwonlyargs]
                    methods.append(f"{item.name}({','.join(margs)})")
            entries.append(f"class {node.name}[{';'.join(sorted(methods))}]")
    return "|".join(sorted(entries))


class HolographicVerifier:
    """Projects bulk module ASTs onto the 256-bit Holographic Boundary Register."""

    def __init__(self) -> None:
        self.register = _make_register()

    def reset(self) -> None:
        self.register = _make_register()

    def encode_bulk_ast_to_boundary(self, files: dict[str, str]) -> str:
        """files: mapping module_path -> source code. Returns boundary digest."""
        self.reset()
        for path in sorted(files):
            signature = extract_boundary_signature(files[path])
            self.register.update_module_boundary(path, signature)
        return self.register.project_boundary_digest()

    def verify_bulk_ast_parity(self, files: dict[str, str], expected_digest: str) -> bool:
        digest = self.encode_bulk_ast_to_boundary(files)
        return self.register.verify_boundary_parity(expected_digest) and digest == expected_digest.strip().lower()

    def detect_bulk_drift(self, before: dict[str, str], after: dict[str, str]) -> bool:
        """True when the public boundary moved between two workspace snapshots."""
        return self.encode_bulk_ast_to_boundary(before) != self.encode_bulk_ast_to_boundary(after)

    def boundary_delta(self, before: dict[str, str], after: dict[str, str]) -> list[str]:
        deltas: list[str] = []
        paths = sorted(set(before) | set(after))
        for path in paths:
            sig_before = extract_boundary_signature(before.get(path, ""))
            sig_after = extract_boundary_signature(after.get(path, ""))
            if sig_before != sig_after:
                deltas.append(path)
        return deltas

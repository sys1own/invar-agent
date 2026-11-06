"""Rust static shield: audits .rs sources for forbidden unsafe constructs."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RustShieldReport:
    passed: bool
    files_checked: int = 0
    violations: list[str] = field(default_factory=list)


class RustSourceShield:
    """Lightweight static audit of Rust sources staged in the overlay."""

    _UNSAFE_BLOCK = re.compile(r"\bunsafe\s*\{")
    _UNSAFE_FN = re.compile(r"\bunsafe\s+fn\b")
    _UNSAFE_IMPL = re.compile(r"\bunsafe\s+impl\b")
    _FORBID = re.compile(r"#!\[\s*forbid\(\s*unsafe_code\s*\)\s*\]")

    # Crates allowlisted to use `unsafe` (none by default in invar-agent).
    def __init__(self, allowlist: list[str] | None = None) -> None:
        self.allowlist = set(allowlist or [])

    def audit_rust_source(self, module_path: str, source: str) -> list[str]:
        violations: list[str] = []
        if module_path in self.allowlist:
            return violations
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            if self._UNSAFE_BLOCK.search(line):
                violations.append(f"{module_path}:{lineno}: forbidden `unsafe {{ }}` block")
            if self._UNSAFE_FN.search(line):
                violations.append(f"{module_path}:{lineno}: forbidden `unsafe fn`")
            if self._UNSAFE_IMPL.search(line):
                violations.append(f"{module_path}:{lineno}: forbidden `unsafe impl`")
        return violations

    def verify_pyOFI_boundary(self, rust_source: str) -> bool:
        """The PyO3 boundary must not expose raw pointers across the FFI."""
        return "*mut" not in rust_source and "*const" not in rust_source.replace("//", "")

    def audit_workspace(self, files: dict[str, str]) -> RustShieldReport:
        violations: list[str] = []
        checked = 0
        for path, source in files.items():
            if not path.endswith(".rs"):
                continue
            checked += 1
            violations.extend(self.audit_rust_source(path, source))
            if "#[pymodule]" in source and not self.verify_pyOFI_boundary(source):
                violations.append(f"{path}: PyO3 boundary exposes raw pointers")
        return RustShieldReport(passed=not violations, files_checked=checked, violations=violations)

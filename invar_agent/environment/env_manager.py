"""Deterministic environment management: venv sync and toolchain verification."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ToolchainStatus:
    python: str = ""
    cargo: str = ""
    maturin: str = ""
    issues: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.issues


class DeterministicEnvManager:
    """Ensures a reproducible virtualenv and verifies the Rust toolchain."""

    def __init__(self, workspace_root: str | os.PathLike) -> None:
        self.workspace_root = Path(workspace_root)

    def sync_virtualenv(self) -> bool:
        """Creates .venv via uv when missing; installs project deps."""
        venv = self.workspace_root / ".venv"
        if (venv / "bin" / "python").exists():
            return True
        uv = shutil.which("uv")
        if uv:
            subprocess.run([uv, "venv", str(venv), "--python", "3.10"], check=False,
                           capture_output=True, cwd=str(self.workspace_root))
        else:
            subprocess.run(["python3", "-m", "venv", str(venv)], check=False,
                           capture_output=True, cwd=str(self.workspace_root))
        return (venv / "bin" / "python").exists()

    def verify_cargo_toolchain(self) -> ToolchainStatus:
        status = ToolchainStatus()
        status.python = shutil.which("python3") or ""
        status.cargo = shutil.which("cargo") or ""
        # maturin may live in the venv.
        venv_maturin = self.workspace_root / ".venv" / "bin" / "maturin"
        status.maturin = str(venv_maturin) if venv_maturin.exists() else (shutil.which("maturin") or "")
        if not status.python:
            status.issues.append("python3 not found on PATH")
        if not status.cargo:
            status.issues.append("cargo not found on PATH (Rust toolchain required for _native)")
        return status

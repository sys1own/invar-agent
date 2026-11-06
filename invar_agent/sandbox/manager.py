"""Hermetic sandbox manager: overlay materialization + isolated execution."""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .command_inspector import ShellCommandInspector
from .firewall import SyscallAndNetworkFirewall
from .process_jail import ProcessJail, ResourceLimits, capture_streams_deterministic


@dataclass
class SandboxExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    command: str = ""
    materialized_path: str = ""


class HermeticSandboxManager:
    """Materializes the overlay into a temp dir and runs commands in isolation."""

    def __init__(
        self,
        workspace_root: str | os.PathLike,
        timeout_seconds: float = 120.0,
        jail: Optional[ProcessJail] = None,
        inspector: Optional[ShellCommandInspector] = None,
        firewall: Optional[SyscallAndNetworkFirewall] = None,
        limits: Optional[ResourceLimits] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.timeout_seconds = timeout_seconds
        self.jail = jail or ProcessJail(limits or ResourceLimits())
        self.inspector = inspector or ShellCommandInspector()
        self.firewall = firewall or SyscallAndNetworkFirewall()
        self._materialized: dict[str, Path] = {}

    # -- overlay materialization ----------------------------------------------------

    def materialize_overlay(self, tx_id: str, staged_files: dict[str, str]) -> Path:
        """Copies the workspace into a temp dir, then applies staged buffers."""
        tmp = Path(tempfile.mkdtemp(prefix=f"invar-sandbox-{tx_id[:12]}-"))
        self._copy_workspace(tmp)
        for rel_path, content in staged_files.items():
            target = tmp / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        self._materialized[tx_id] = tmp
        return tmp

    def materialize_plain(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="invar-sandbox-plain-"))
        self._copy_workspace(tmp)
        return tmp

    def _copy_workspace(self, dest: Path) -> None:
        ignore = shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", ".invar_agent", "target", ".venv", "node_modules",
            ".pytest_cache", "dist", "build", "*.egg-info",
        )
        for item in self.workspace_root.iterdir():
            if item.name.startswith(".") and item.name not in (".invarignore",):
                continue
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, ignore=ignore, symlinks=True)
            else:
                shutil.copy2(item, target)

    def cleanup(self, tx_id: Optional[str] = None) -> None:
        if tx_id is not None:
            path = self._materialized.pop(tx_id, None)
            if path and path.exists():
                shutil.rmtree(path, ignore_errors=True)
        else:
            for path in self._materialized.values():
                shutil.rmtree(path, ignore_errors=True)
            self._materialized.clear()

    # -- execution -------------------------------------------------------------------

    def run_isolated_command(
        self,
        argv: list[str],
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
        tx_id: Optional[str] = None,
        staged_files: Optional[dict[str, str]] = None,
    ) -> SandboxExecutionResult:
        verdict = self.inspector.validate_argv(argv)
        if not verdict.safe:
            return SandboxExecutionResult(
                exit_code=126,
                stdout="",
                stderr=f"Command rejected by ShellCommandInspector: {verdict.violations} (risk={verdict.risk_level})",
                duration_ms=0.0,
                command=" ".join(argv),
            )

        workdir: Optional[Path] = None
        if tx_id is not None and staged_files is not None:
            workdir = self.materialize_overlay(tx_id, staged_files)
        elif cwd is None:
            workdir = self.materialize_plain()

        effective_cwd = cwd or (str(workdir) if workdir else str(self.workspace_root))
        effective_env = self.jail.scrubbed_env(env)
        violations = self.firewall.check_command_env(effective_env)
        if violations:
            return SandboxExecutionResult(
                exit_code=126, stdout="", stderr="; ".join(violations), duration_ms=0.0,
                command=" ".join(argv),
            )

        result = capture_streams_deterministic(
            argv,
            cwd=effective_cwd,
            env=effective_env,
            timeout=timeout if timeout is not None else self.timeout_seconds,
            jail=self.jail,
        )
        return SandboxExecutionResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=result.duration_ms,
            timed_out=result.timed_out,
            command=" ".join(argv),
            materialized_path=str(workdir) if workdir else "",
        )

    def run_pytest(self, tx_id: Optional[str] = None, staged_files: Optional[dict[str, str]] = None,
                   pytest_args: list[str] | None = None, timeout: Optional[float] = None) -> SandboxExecutionResult:
        args = pytest_args or ["-q"]
        # Prefer the workspace venv python when available.
        venv_python = self.workspace_root / ".venv" / "bin" / "python"
        python = str(venv_python) if venv_python.exists() else "python3"
        return self.run_isolated_command(
            [python, "-m", "pytest", *args],
            tx_id=tx_id,
            staged_files=staged_files,
            timeout=timeout,
        )

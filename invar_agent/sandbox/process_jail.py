"""Process jail: rlimits, scrubbed environment, deterministic stream capture."""
from __future__ import annotations

import os
import resource
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ResourceLimits:
    cpu_seconds: int = 120
    memory_mb: int = 2048
    file_size_mb: int = 512
    max_processes: int = 256
    max_open_files: int = 1024


@dataclass
class CaptureResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False


class ProcessJail:
    """Applies OS-level resource limits to child processes."""

    def __init__(self, limits: Optional[ResourceLimits] = None) -> None:
        self.limits = limits or ResourceLimits()

    def enforce_rlimits(self) -> None:
        """Apply rlimits in the child process (preexec_fn)."""
        lim = self.limits
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (lim.cpu_seconds, lim.cpu_seconds + 5))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_AS, (lim.memory_mb * 1024 * 1024, lim.memory_mb * 1024 * 1024))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (lim.file_size_mb * 1024 * 1024, lim.file_size_mb * 1024 * 1024))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (lim.max_processes, lim.max_processes))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (lim.max_open_files, lim.max_open_files))
        except (ValueError, OSError):
            pass

    def scrubbed_env(self, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
        """Deterministic, minimal environment for hermetic execution."""
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TMPDIR": "/tmp",
        }
        if extra:
            env.update(extra)
        return env

    def preexec(self):
        self.enforce_rlimits()


def capture_streams_deterministic(
    cmd: list[str],
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
    jail: Optional[ProcessJail] = None,
) -> CaptureResult:
    """Runs a command capturing stdout/stderr deterministically."""
    start = time.perf_counter()
    jail = jail or ProcessJail()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env or jail.scrubbed_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            preexec_fn=jail.preexec,
        )
        duration = (time.perf_counter() - start) * 1000.0
        return CaptureResult(
            exit_code=proc.returncode,
            stdout=proc.stdout.decode("utf-8", errors="replace"),
            stderr=proc.stderr.decode("utf-8", errors="replace"),
            duration_ms=duration,
        )
    except subprocess.TimeoutExpired as e:
        duration = (time.perf_counter() - start) * 1000.0
        return CaptureResult(
            exit_code=124,
            stdout=(e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            stderr=(e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
            duration_ms=duration,
            timed_out=True,
        )

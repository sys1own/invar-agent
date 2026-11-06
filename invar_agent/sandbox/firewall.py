"""Syscall and network firewall policy for the hermetic sandbox."""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FirewallPolicy:
    allow_network: bool = False
    allowed_syscalls: set[str] = field(default_factory=lambda: {
        "read", "write", "open", "close", "stat", "fstat", "lseek", "mmap",
        "munmap", "brk", "rt_sigaction", "rt_sigprocmask", "ioctl", "access",
        "getcwd", "chdir", "futex", "getrandom", "clock_gettime", "exit",
        "exit_group", "execve", "wait4", "clone", "fork", "vfork", "pipe",
        "pipe2", "dup", "dup2", "dup3", "select", "poll", "epoll_create",
        "epoll_ctl", "epoll_wait", "getpid", "gettid", "getuid", "getgid",
    })
    denied_paths: set[str] = field(default_factory=lambda: {".git", ".ssh", ".aws", ".gnupg"})
    allowed_roots: list[str] = field(default_factory=list)


class SyscallAndNetworkFirewall:
    """Enforces offline, path-restricted execution policy."""

    def __init__(self, policy: Optional[FirewallPolicy] = None) -> None:
        self.policy = policy or FirewallPolicy()

    def verify_network_isolation(self, probe_host: str = "127.0.0.1", probe_port: int = 9) -> bool:
        """Attempts an outbound connection; isolation holds if it fails."""
        if self.policy.allow_network:
            return True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            sock.connect((probe_host, probe_port))
            return False  # connection succeeded -> not isolated
        except OSError:
            return True
        finally:
            sock.close()

    def audit_path_access(self, path: str) -> bool:
        """True when the path is permitted under the policy."""
        norm = os.path.normpath(path)
        for denied in self.policy.denied_paths:
            if denied in norm.split(os.sep):
                return False
        if self.policy.allowed_roots:
            resolved = os.path.abspath(norm)
            return any(resolved.startswith(os.path.abspath(root)) for root in self.policy.allowed_roots)
        return True

    def check_command_env(self, env: dict[str, str]) -> list[str]:
        """Flags environment variables that could break hermeticity."""
        violations: list[str] = []
        sensitive = ("AWS_", "GITHUB_", "OPENAI_", "ANTHROPIC_", "GEMINI_", "OPENROUTER_")
        for key in env:
            if any(key.startswith(prefix) for prefix in sensitive):
                violations.append(f"sensitive env var present: {key}")
        return violations

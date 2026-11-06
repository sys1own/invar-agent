"""Shell command inspector: static risk analysis before sandbox execution."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field


@dataclass
class CommandRiskVerdict:
    safe: bool
    risk_level: str  # "SAFE" | "LOW" | "HIGH" | "CRITICAL"
    violations: list[str] = field(default_factory=list)


class ShellCommandInspector:
    """Rejects destructive or non-hermetic shell commands."""

    _DENY_PATTERNS: list[tuple[str, re.Pattern]] = [
        ("recursive_force_delete", re.compile(r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+/")),
        ("git_destructive", re.compile(r"\bgit\s+(push|reset\s+--hard|clean\s+-[a-z]*f|checkout\s+--\s+\.|rebase)\b")),
        ("network_tool", re.compile(r"\b(curl|wget|nc|netcat|ssh|scp|rsync)\b")),
        ("privilege_escalation", re.compile(r"\b(sudo|su|doas)\b")),
        ("permission_escalation", re.compile(r"\bchmod\s+(-R\s+)?777\b|\bchown\b")),
        ("shell_background", re.compile(r"(&&|\|\||`|;|\|)")),
        ("process_control", re.compile(r"\b(nohup|disown|setsid|kill|pkill)\b")),
        ("destructive_fs", re.compile(r"\b(mkfs|dd\s+if=|shred|wipe)\b")),
    ]

    _ALLOW_PREFIXES = ("pytest", "python", "python3", "cargo", "rustc", "uv", "pip", "ls", "cat", "echo", "grep", "find", "maturin", "git", "mkdir", "cp", "mv", "touch", "rm")

    @staticmethod
    def _exe_name(token: str) -> str:
        """Normalise a path/executable token: backslashes -> slashes, strip .exe."""
        return re.sub(r"\.exe$", "", token.replace("\\", "/").split("/")[-1], flags=re.IGNORECASE)

    def parse_and_validate_shell_ast(self, command: str | list[str]) -> CommandRiskVerdict:
        if isinstance(command, list):
            tokens = list(command)
            raw = " ".join(shlex.quote(t) for t in tokens)
        else:
            raw = command
            try:
                tokens = shlex.split(command)
            except ValueError:
                tokens = command.split()

        violations: list[str] = []
        for reason, pattern in self._DENY_PATTERNS:
            if pattern.search(raw):
                violations.append(reason)

        if tokens:
            head = self._exe_name(tokens[0])
            if head not in self._ALLOW_PREFIXES and head not in ("env",):
                violations.append(f"non-allowlisted executable: {tokens[0]}")
        # `git` is allowlisted only for read/test operations.
        if tokens and self._exe_name(tokens[0]) == "git":
            if len(tokens) > 1 and tokens[1] not in ("status", "diff", "log", "show", "rev-parse", "ls-files"):
                violations.append("git mutation subcommand not permitted in sandbox")
        if not tokens:
            violations.append("empty command")

        if any(v in ("recursive_force_delete", "privilege_escalation", "destructive_fs", "git_destructive") for v in violations):
            risk = "CRITICAL"
        elif violations:
            risk = "HIGH"
        else:
            risk = "SAFE"
        return CommandRiskVerdict(safe=not violations, risk_level=risk, violations=violations)

    def validate_argv(self, argv: list[str]) -> CommandRiskVerdict:
        return self.parse_and_validate_shell_ast(argv)

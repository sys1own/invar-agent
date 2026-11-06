"""Deterministic error classification from stderr/exit codes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FailureCategory(str, Enum):
    SYNTAX_ERROR = "SYNTAX_ERROR"
    NAME_ERROR = "NAME_ERROR"
    TYPE_ERROR = "TYPE_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    ASSERTION_FAILURE = "ASSERTION_FAILURE"
    SMT_COUNTEREXAMPLE = "SMT_COUNTEREXAMPLE"
    CONCOLIC_BRANCH_PANIC = "CONCOLIC_BRANCH_PANIC"
    CHIASMUS_ASYMMETRY = "CHIASMUS_ASYMMETRY"
    SANDBOX_TEST_FAILURE = "SANDBOX_TEST_FAILURE"
    RUST_BORROW_ERROR = "RUST_BORROW_ERROR"
    RUST_SYNTAX_ERROR = "RUST_SYNTAX_ERROR"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class ClassifiedErrorPayload:
    category: FailureCategory
    error_code: str
    diagnostic_msg: str
    failing_file: Optional[str] = None
    failing_line: Optional[int] = None
    stderr_tail: str = ""
    stack_trace_frames: list[str] = field(default_factory=list)


class DeterministicErrorClassifier:
    """Classifies raw stderr into structured failure categories (0 tokens)."""

    _PATTERNS: list[tuple[FailureCategory, re.Pattern]] = [
        (FailureCategory.RUST_BORROW_ERROR, re.compile(r"error\[E0(38[0-9]|05[0-9])\]|borrow of moved|use of moved value")),
        (FailureCategory.RUST_SYNTAX_ERROR, re.compile(r"expected one of|mismatched closing delimiter|error: expected")),
        (FailureCategory.SYNTAX_ERROR, re.compile(r"SyntaxError|Syntax Error|IndentationError|TabError")),
        (FailureCategory.NAME_ERROR, re.compile(r"NameError|name '.*' is not defined")),
        (FailureCategory.IMPORT_ERROR, re.compile(r"ImportError|ModuleNotFoundError|cannot find module")),
        (FailureCategory.TYPE_ERROR, re.compile(r"TypeError")),
        (FailureCategory.CONCOLIC_BRANCH_PANIC, re.compile(r"ZeroDivisionError|concolic.*panic|branch.*panic", re.IGNORECASE)),
        (FailureCategory.SMT_COUNTEREXAMPLE, re.compile(r"SMT.*COUNTEREXAMPLE|SAT_COUNTEREXAMPLE", re.IGNORECASE)),
        (FailureCategory.CHIASMUS_ASYMMETRY, re.compile(r"chiasmus.*(violation|asymmetr)|mirror invariant", re.IGNORECASE)),
        (FailureCategory.ASSERTION_FAILURE, re.compile(r"AssertionError")),
        (FailureCategory.SANDBOX_TEST_FAILURE, re.compile(r"FAILED|ERROR at teardown|pytest.*failed", re.IGNORECASE)),
        (FailureCategory.TIMEOUT, re.compile(r"timed out|TimeoutError")),
    ]

    def classify_stderr_trace(self, stderr: str, exit_code: int = 1) -> ClassifiedErrorPayload:
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        category = FailureCategory.UNKNOWN
        error_code = "E_UNKNOWN"
        failing_file: Optional[str] = None
        failing_line: Optional[int] = None

        # Extract file/line from Python traceback.
        file_line = re.findall(r'File "([^"]+)", line (\d+)', stderr)
        if file_line:
            failing_file, failing_line = file_line[-1][0], int(file_line[-1][1])

        for cat, pattern in self._PATTERNS:
            if pattern.search(stderr):
                category = cat
                break
        if category == FailureCategory.UNKNOWN and exit_code == 124:
            category = FailureCategory.TIMEOUT
            error_code = "TIMEOUT"

        # Error code extraction.
        m = re.search(r"error\[(E\d+)\]", stderr)
        if m:
            error_code = m.group(1)
        elif category == FailureCategory.SYNTAX_ERROR:
            error_code = "SyntaxError"
        elif category == FailureCategory.NAME_ERROR:
            error_code = "NameError"
        elif category == FailureCategory.IMPORT_ERROR:
            error_code = "ImportError"
        elif category == FailureCategory.RUST_BORROW_ERROR:
            error_code = error_code if error_code != "E_UNKNOWN" else "E0382"
        elif category == FailureCategory.SANDBOX_TEST_FAILURE:
            error_code = "TEST_FAILED"
        elif category != FailureCategory.UNKNOWN:
            error_code = category.value

        frames = [f'File "{f[0]}", line {f[1]}' for f in file_line[-5:]]
        # Diagnostic message: last non-empty line.
        lines = [l for l in stderr.strip().splitlines() if l.strip()]
        diagnostic = lines[-1].strip() if lines else ""
        if category == FailureCategory.SYNTAX_ERROR:
            m2 = re.search(r"(SyntaxError.*)", stderr)
            if m2:
                diagnostic = m2.group(1).strip()
        return ClassifiedErrorPayload(
            category=category,
            error_code=error_code,
            diagnostic_msg=diagnostic[:500],
            failing_file=failing_file,
            failing_line=failing_line,
            stderr_tail=tail,
            stack_trace_frames=frames,
        )

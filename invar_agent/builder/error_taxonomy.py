"""Deterministic routing for builder diagnostics; provider text is never executable."""
from __future__ import annotations

import errno
from enum import Enum


class ErrorTier(str, Enum):
    CRITICAL_NON_RETRYABLE = "CRITICAL_NON_RETRYABLE"
    TIER0_REFLEX = "TIER0_REFLEX"
    TIER1_STRUCTURAL = "TIER1_STRUCTURAL"
    TIER2_BOUNDED_LLM = "TIER2_BOUNDED_LLM"


class CircuitAction(str, Enum):
    APPLY_TIER0 = "APPLY_TIER0"
    APPLY_TIER1 = "APPLY_TIER1"
    INVOKE_TIER2_LLM = "INVOKE_TIER2_LLM"
    SUBSTITUTE_DETERMINISTIC_FALLBACK = "SUBSTITUTE_DETERMINISTIC_FALLBACK"
    SWITCH_TO_PYTHON_FALLBACK = "SWITCH_TO_PYTHON_FALLBACK"
    ABORT_BUILD_CRITICAL = "ABORT_BUILD_CRITICAL"


TIER0_CODES = {"PY_SYNTAX_ERROR", "PY_MISSING_STDLIB_IMPORT", "PY_DOMAIN_ARITHMETIC_GUARD", "CSEL_ALGEBRAIC_REDUNDANCY"}
TIER1_CODES = {"HIS_CONTRACT_DRIFT", "HIN_SI_DIM_MISMATCH", "HIN_SEMANTIC_TAG_MISMATCH", "BOUND_CLAMP_VIOLATION", "CPP_COMPILE_DIAGNOSTIC"}
TIER2_CODES = {"SLOT_PYTEST_ASSERTION", "SLOT_RUNTIME_LOGIC_ERROR"}


class BuildFault(ValueError):
    def __init__(self, code: str, message: str, **details):
        self.code, self.details = code, details
        super().__init__(message)


def classify_error(error: BaseException, node_kind: str = "deterministic_module") -> tuple[str, ErrorTier]:
    code = getattr(error, "code", None)
    status = getattr(error, "status_code", None)
    message = str(error).lower()
    if status is None and getattr(error, "response", None) is not None:
        status = error.response.status_code
    if status in {401, 403}:
        code = "LLM_AUTH_ERROR"
    elif status == 429:
        code = "LLM_QUOTA_EXHAUSTED" if any(word in str(error).lower() for word in ("credit", "quota", "billing", "balance")) else "LLM_RATE_LIMIT_EXHAUSTED"
    elif status is not None and status >= 500:
        code = "LLM_RATE_LIMIT_EXHAUSTED"
    elif "credit_balance_exhausted" in message:
        code = "LLM_QUOTA_EXHAUSTED"
    elif "api" in message and "key" in message and any(word in message for word in ("not configured", "no api key")):
        code = "LLM_AUTH_ERROR"
    elif isinstance(error, MemoryError) or isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EDQUOT, errno.EACCES, errno.EROFS}:
        code = "OS_OOM_OR_DISK"
    elif isinstance(error, (SyntaxError, IndentationError, TabError)):
        code = "PY_SYNTAX_ERROR"
    elif isinstance(error, NameError):
        code = "PY_MISSING_STDLIB_IMPORT"
    elif isinstance(error, (ZeroDivisionError, OverflowError)):
        code = "PY_DOMAIN_ARITHMETIC_GUARD"
    elif isinstance(error, ValueError) and any(word in message for word in ("math domain error", "math range error", "nonfinite")):
        code = "PY_DOMAIN_ARITHMETIC_GUARD"
    if code and code.startswith("SANDBOX_ZIP_"):
        code = "SANDBOX_SECURITY_VIOLATION"
    code = code or ("SLOT_RUNTIME_LOGIC_ERROR" if node_kind == "llm_extension_slot" else "DETERMINISTIC_EMITTER_BUG")
    tier = ErrorTier.TIER0_REFLEX if code in TIER0_CODES else ErrorTier.TIER1_STRUCTURAL if code in TIER1_CODES else ErrorTier.TIER2_BOUNDED_LLM if code in TIER2_CODES else ErrorTier.CRITICAL_NON_RETRYABLE
    return code, tier

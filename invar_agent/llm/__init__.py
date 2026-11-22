"""Tiered LLM clients: provider adapters, structured errors, validation, telemetry."""
from .errors import LLMProviderError, extract_upstream_error, is_retryable_status
from .telemetry import (
    LLMCallRecord,
    LLMTelemetrySession,
    extract_usage_tokens,
    get_llm_telemetry_summary,
    llm_call_context,
    llm_telemetry_scope,
    record_llm_call,
    register_llm_telemetry_listener,
    reset_llm_telemetry,
)

__all__ = [
    "LLMProviderError",
    "extract_upstream_error",
    "is_retryable_status",
    "LLMCallRecord",
    "LLMTelemetrySession",
    "extract_usage_tokens",
    "get_llm_telemetry_summary",
    "llm_call_context",
    "llm_telemetry_scope",
    "record_llm_call",
    "register_llm_telemetry_listener",
    "reset_llm_telemetry",
]

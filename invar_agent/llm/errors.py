"""Structured LLM provider errors, upstream-body parsing, and retry policy."""
from __future__ import annotations

import json
import random
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

# API keys are redacted from every message, log line and API response.
_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
]

_MAX_MESSAGE_LEN = 500
_MAX_RETRIES = 2  # retries after the initial attempt (429/5xx/timeouts only)


def redact_secrets(text: Any, api_key: Optional[str] = None) -> str:
    """Return ``text`` as a string with API keys scrubbed out."""
    value = text if isinstance(text, str) else str(text)
    if api_key:
        value = value.replace(api_key, "sk-****")
    for pattern in _KEY_PATTERNS:
        value = pattern.sub("sk-****", value)
    return value


def _parse_error_body(body: Any) -> Tuple[str, Optional[str]]:
    """Extract (message, request_id) from OpenAI- or Google-style error bodies."""
    request_id: Optional[str] = None
    if isinstance(body, (dict, list)):
        data: Any = body
        if isinstance(data, dict):
            request_id = data.get("request_id") or data.get("requestId")
            err = data.get("error", data)
        else:
            err = data
        if isinstance(err, list):
            # Google-style: [{"error": {"code": 400, "message": ..., "status": ...}}]
            parts = []
            for item in err:
                message, item_request_id = _parse_error_body(item)
                request_id = request_id or item_request_id
                if message:
                    parts.append(message)
            return "; ".join(parts), request_id
        if isinstance(err, dict):
            message = err.get("message") or err.get("status") or ""
            if isinstance(message, dict):  # nested {"error": {"message": {"message": ...}}}
                message = message.get("message") or ""
            request_id = request_id or err.get("request_id") or err.get("requestId")
            return str(message), request_id
        return str(err), request_id
    if isinstance(body, str):
        try:
            return _parse_error_body(json.loads(body))
        except (json.JSONDecodeError, ValueError):
            return body, None
    return "", None


def extract_upstream_error(
    status_code: int,
    body: Any,
    headers: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Parse a provider error response into a sanitized (message, request_id)."""
    message, request_id = _parse_error_body(body)
    if headers:
        lowered = {k.lower(): v for k, v in headers.items()}
        request_id = request_id or lowered.get("x-request-id") or lowered.get("x-goog-request-id")
    message = redact_secrets(message, api_key).strip() or f"HTTP {status_code}"
    if len(message) > _MAX_MESSAGE_LEN:
        message = message[: _MAX_MESSAGE_LEN - 1] + "…"
    return message, request_id


class LLMProviderError(Exception):
    """Structured failure of an LLM provider call (HTTP or transport level)."""

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        upstream_message: str = "",
        request_id: Optional[str] = None,
        retryable: bool = False,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model = model or "unknown"
        self.status_code = status_code
        self.upstream_message = redact_secrets(upstream_message, api_key)[:_MAX_MESSAGE_LEN]
        self.request_id = request_id
        self.retryable = retryable
        super().__init__(self.describe())

    def describe(self) -> str:
        parts = [f"LLM call to {self.provider}/{self.model} failed"]
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        if self.upstream_message:
            parts.append(f": {self.upstream_message}")
        if self.request_id:
            parts.append(f" (request_id={self.request_id})")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status_code": self.status_code,
            "upstream_message": self.upstream_message,
            "request_id": self.request_id,
            "retryable": self.retryable,
        }


def is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


# ---------------------------------------------------------------------------
# Retry with exponential backoff + jitter (429/5xx/timeouts only)
# ---------------------------------------------------------------------------

def backoff_delay(attempt: int) -> float:
    """Seconds to wait before retry ``attempt`` (0-based): exp backoff + jitter."""
    return min(0.5 * (2 ** attempt), 8.0) + random.uniform(0.0, 0.25)


def sleep(seconds: float) -> None:
    """Patch point for tests."""
    time.sleep(seconds)


def call_with_retries(fn):
    """Call ``fn()`` up to 3 attempts; retries only when the raised
    :class:`LLMProviderError` is marked ``retryable`` (429/5xx/timeout)."""
    last: Optional[LLMProviderError] = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except LLMProviderError as exc:
            last = exc
            if exc.retryable and attempt < _MAX_RETRIES:
                sleep(backoff_delay(attempt))
                continue
            raise
    raise last if last else LLMProviderError("unknown")


# ---------------------------------------------------------------------------
# Unsupported-parameter detection (retry ONCE without optional params)
# ---------------------------------------------------------------------------

_UNSUPPORTED_PARAM_RE = re.compile(
    r"(unsupported|unknown|unrecognized|not (a )?(valid|supported)|deprecated).{0,60}(parameter|argument|field|option)"
    r"|(parameter|argument|field|option).{0,60}(unsupported|unknown|unrecognized|invalid|not supported)"
    r"|unknown name \W?(temperature|top_p)",
    re.IGNORECASE,
)


def is_unsupported_param_error(error: LLMProviderError) -> bool:
    """True when a 400 complains about an unsupported/unknown request parameter."""
    if error.status_code != 400:
        return False
    return bool(_UNSUPPORTED_PARAM_RE.search(error.upstream_message or ""))


# ---------------------------------------------------------------------------
# Last-error registry (surfaced by /api/status; sanitized at construction)
# ---------------------------------------------------------------------------

_last_error_lock = threading.Lock()
_last_error: Optional[Dict[str, Any]] = None


def record_llm_error(error: Any) -> None:
    """Record the most recent LLM failure (sanitized, timestamped)."""
    global _last_error
    if isinstance(error, LLMProviderError):
        payload: Dict[str, Any] = {
            "provider": error.provider,
            "model": error.model,
            "status_code": error.status_code,
            "retryable": error.retryable,
            "message": error.describe(),
        }
    elif isinstance(error, dict):
        payload = dict(error)
    else:
        payload = {"message": redact_secrets(str(error))}
    payload["at"] = datetime.now(timezone.utc).isoformat()
    with _last_error_lock:
        _last_error = payload


def get_last_llm_error() -> Optional[Dict[str, Any]]:
    with _last_error_lock:
        return dict(_last_error) if _last_error else None


def clear_last_llm_error() -> None:
    global _last_error
    with _last_error_lock:
        _last_error = None

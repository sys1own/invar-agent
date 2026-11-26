"""Pure structural JSON recovery for LLM structured outputs.

Strictly performs lossless structural cleanup (markdown fence stripping,
balanced brace extraction, trailing comma removal, single missing closing brace
repair, and DeepSeek `U+0020` whitespace-only empty-response detection).
NEVER mutates mathematical expressions, numeric constants, or domain semantics.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class EmptyWhitespaceResponseError(ValueError):
    """Raised when an LLM returns only whitespace (e.g. DeepSeek U+0020 stall)."""


@dataclass(frozen=True)
class JsonRecoveryResult:
    data: dict[str, Any]
    recovery_steps: tuple[str, ...]


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def is_whitespace_only_response(raw: str | None) -> bool:
    """Detect DeepSeek empty / U+0020 whitespace-only responses."""
    if raw is None:
        return True
    return len(raw.strip()) == 0


def _extract_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    if depth == 1:
        return text[start:] + "}"
    return None


def recover_json_object(raw: str) -> JsonRecoveryResult:
    """Parse a JSON object from raw LLM text using bounded structural recovery."""
    if is_whitespace_only_response(raw):
        raise EmptyWhitespaceResponseError(
            "LLM returned empty or whitespace-only (U+0020) response"
        )

    steps: list[str] = []
    candidate = raw.strip()

    # 1. Direct parse fast-path
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return JsonRecoveryResult(data=parsed, recovery_steps=())
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences if present
    fence_match = _FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()
        steps.append("strip_markdown_fence")
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return JsonRecoveryResult(data=parsed, recovery_steps=tuple(steps))
        except json.JSONDecodeError:
            pass

    # 3. Extract first balanced JSON object (also repairs single missing closing brace)
    extracted = _extract_balanced_object(candidate)
    if extracted is not None and extracted != candidate:
        if not candidate.rstrip().endswith("}") and extracted.endswith("}"):
            steps.append("close_single_missing_brace")
        else:
            steps.append("extract_balanced_object")
        candidate = extracted
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return JsonRecoveryResult(data=parsed, recovery_steps=tuple(steps))
        except json.JSONDecodeError:
            pass

    # 4. Strip trailing commas before '}' or ']'
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    if cleaned != candidate:
        steps.append("strip_trailing_commas")
        candidate = cleaned
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return JsonRecoveryResult(data=parsed, recovery_steps=tuple(steps))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Unrecoverable JSON object after steps {steps}: {candidate[:240]}")

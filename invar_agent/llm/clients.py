"""Tier-resolved LLM chat clients with structured errors and retry policy."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import httpx

from ..config import PROVIDER_KEY_ENVS, Tier, resolve_tier_model
from .errors import (
    LLMProviderError,
    call_with_retries,
    extract_upstream_error,
    is_retryable_status,
    is_unsupported_param_error,
)


@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


class LLMClient(Protocol):
    def complete(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        ...


# ---------------------------------------------------------------------------
# Provider endpoints and per-provider capability table
# ---------------------------------------------------------------------------

# API base per provider (used for both chat completions and /models discovery).
PROVIDER_API_BASES: Dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Capability table: which optional sampling parameters a provider accepts and
# how the max-token budget is named. Newer Gemini models reject sampling params,
# so payloads are built from this table and stripped once on an explicit
# "unsupported parameter" 400 (see ChatLLMClient.generate).
_PROVIDER_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "gemini": {"supports_temperature": True, "supports_top_p": True, "max_tokens_param": "max_tokens"},
    "openai": {"supports_temperature": True, "supports_top_p": True, "max_tokens_param": "max_tokens"},
    "openrouter": {"supports_temperature": True, "supports_top_p": True, "max_tokens_param": "max_tokens"},
    "deepseek": {"supports_temperature": False, "supports_top_p": False, "max_tokens_param": "max_tokens"},
}
_DEFAULT_CAPABILITIES: Dict[str, Any] = {"supports_temperature": True, "supports_top_p": True, "max_tokens_param": "max_tokens"}

# Remember (provider, model) pairs that rejected optional sampling params so we
# strip them preemptively for the rest of the process (no blind retries).
_STRIPPED_OPTIONAL_PARAMS: set = set()


def chat_url(provider: str) -> str:
    return f"{PROVIDER_API_BASES.get(provider, PROVIDER_API_BASES['openrouter'])}/chat/completions"


def models_url(provider: str) -> str:
    return f"{PROVIDER_API_BASES.get(provider, PROVIDER_API_BASES['openrouter'])}/models"


class ChatLLMClient:
    """OpenAI-compatible chat client bound to one resolved tier model."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tier: Tier = Tier.FAST,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.tier = tier
        self.api_key = api_key or os.environ.get(PROVIDER_KEY_ENVS.get(provider, ""), "")
        self.base_url = base_url or chat_url(provider)
        self._transport = transport  # injectable for tests (httpx.MockTransport)
        self._last_response: Optional[LLMResponse] = None

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _capabilities(self) -> Dict[str, Any]:
        return _PROVIDER_CAPABILITIES.get(self.provider, _DEFAULT_CAPABILITIES)

    def _build_payload(self, messages: List[Dict[str, str]], temperature: float,
                       max_tokens: int, strip_optional: bool = False) -> Dict[str, Any]:
        """Build the request payload from the provider capability table."""
        caps = self._capabilities()
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        payload[caps.get("max_tokens_param", "max_tokens")] = max_tokens
        if not strip_optional:
            if caps.get("supports_temperature", True):
                payload["temperature"] = temperature
            if caps.get("supports_top_p", True):
                payload["top_p"] = 1.0
        return payload

    def _post(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """POST once; raises :class:`LLMProviderError` with structured detail."""
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=120.0, transport=self._transport) as client:
                resp = client.post(self.base_url, headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise LLMProviderError(
                self.provider, self.model, status_code=None,
                upstream_message=f"Request timed out: {exc}", retryable=True,
                api_key=self.api_key,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(
                self.provider, self.model, status_code=None,
                upstream_message=f"Transport error: {exc}", retryable=True,
                api_key=self.api_key,
            ) from exc
        latency = (time.perf_counter() - start) * 1000.0
        if resp.status_code >= 400:
            message, request_id = extract_upstream_error(
                resp.status_code, resp.text, dict(resp.headers), api_key=self.api_key,
            )
            raise LLMProviderError(
                self.provider, self.model, status_code=resp.status_code,
                upstream_message=message, request_id=request_id,
                retryable=is_retryable_status(resp.status_code),
                api_key=self.api_key,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMProviderError(
                self.provider, self.model, status_code=resp.status_code,
                upstream_message="Invalid JSON in provider response",
                api_key=self.api_key,
            ) from exc
        return data, latency

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2,
                 max_tokens: int = 4096) -> str:
        """Multi-turn chat completion; returns the assistant message content."""
        if not self.api_key:
            raise RuntimeError(f"No API key configured for provider '{self.provider}'.")
        strip_optional = (self.provider, self.model) in _STRIPPED_OPTIONAL_PARAMS
        payload = self._build_payload(messages, temperature, max_tokens, strip_optional=strip_optional)
        try:
            data, latency = call_with_retries(lambda: self._post(payload))
        except LLMProviderError as exc:
            # Retry ONCE without optional params when the provider rejects them.
            if not strip_optional and is_unsupported_param_error(exc):
                _STRIPPED_OPTIONAL_PARAMS.add((self.provider, self.model))
                payload = self._build_payload(messages, temperature, max_tokens, strip_optional=True)
                data, latency = call_with_retries(lambda: self._post(payload))
            else:
                raise
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self._last_response = LLMResponse(
            content=choice,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=self.model,
            latency_ms=latency,
        )
        return choice

    def complete(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        """Single-turn completion compatible with the LLMClient protocol."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        content = self.generate(messages, temperature=temperature, max_tokens=max_tokens)
        if self._last_response is not None:
            return self._last_response
        return LLMResponse(content=content, model=self.model)


def get_llm_client(
    provider: Optional[str] = None,
    tier: Tier = Tier.FAST,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    override: Optional[str] = None,
) -> Optional[ChatLLMClient]:
    """Resolve a chat client for ``provider``/``tier`` via :func:`resolve_tier_model`.

    Returns ``None`` when no API key is available so callers can fall back to
    deterministic local behaviour. ``override`` is an explicit per-call model
    override (e.g. from ``/api/config`` tier_overrides) that wins over env,
    file config, and defaults.
    """
    provider_name = (provider or os.environ.get("INVAR_LLM_PROVIDER", "gemini")).lower()
    resolved_model = model or resolve_tier_model(provider_name, tier, override=override)
    if not resolved_model:
        return None
    key = (
        api_key
        or os.environ.get(PROVIDER_KEY_ENVS.get(provider_name, ""), "")
        or os.environ.get("INVAR_API_KEY", "")
    )
    if not key:
        return None
    if provider_name == "deepseek":
        # DeepSeek gets its own OpenAI-SDK adapter: it omits sampling params
        # (temperature/top_p) that DeepSeek thinking mode rejects, and resolves
        # the tier model through resolve_tier_model (deepseek-flash /
        # deepseek-v4-pro).
        from .deepseek import DeepSeekChatClient

        return DeepSeekChatClient(
            tier=tier,
            api_key=key,
            model=resolved_model,
            base_url=base_url,
        )
    return ChatLLMClient(
        provider=provider_name,
        model=resolved_model,
        api_key=key,
        base_url=base_url,
        tier=tier,
    )

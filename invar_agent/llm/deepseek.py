"""Native DeepSeek API adapter.

Wraps the OpenAI Python SDK against DeepSeek's OpenAI-compatible Chat
Completions API (``https://api.deepseek.com``) and routes every call through
:func:`invar_agent.config.resolve_tier_model` so the FAST tier always lands on
``deepseek-flash`` and the REASONING tier on ``deepseek-v4-pro``.

DeepSeek thinking mode rejects sampling parameters, so ``temperature`` /
``top_p`` are intentionally never sent — only ``model``, ``messages`` and
``max_tokens``.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from invar_agent.config import Tier, resolve_tier_model
from invar_agent.llm.errors import (
    LLMProviderError,
    call_with_retries,
    extract_upstream_error,
    is_retryable_status,
)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_PROVIDER = "deepseek"


def _llm_response(**kwargs: Any):
    """Build an LLMResponse without a hard import cycle with ``clients``."""
    from invar_agent.llm.clients import LLMResponse

    return LLMResponse(**kwargs)


class DeepSeekChatClient:
    """Tier-bound DeepSeek chat client compatible with ``ChatLLMClient``.

    The model is resolved once at construction from ``resolve_tier_model``
    (explicit override -> ``DEEPSEEK_MODEL_<TIER>`` env -> file config ->
    defaults), so a FAST client always talks to ``deepseek-flash`` and a
    REASONING client to ``deepseek-v4-pro``.
    """

    def __init__(
        self,
        tier: Tier = Tier.FAST,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self.tier = tier if isinstance(tier, Tier) else Tier(str(tier))
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or DEEPSEEK_BASE_URL).rstrip("/")
        # Explicit model wins; otherwise resolve through the tier router.
        self.model = model or resolve_tier_model(DEEPSEEK_PROVIDER, self.tier)
        self._client = client  # injectable OpenAI client (tests / custom wiring)
        self._last_response = None

    # -- OpenAI SDK access -----------------------------------------------------

    def _openai_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("DEEPSEEK_API_KEY not configured; cannot call DeepSeek.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    # -- error mapping ---------------------------------------------------------

    def _map_error(self, exc: Exception) -> Optional[LLMProviderError]:
        """Map OpenAI SDK / httpx exceptions to the shared LLMProviderError."""
        try:
            import openai  # noqa: PLC0415
        except ImportError:  # pragma: no cover - openai is a required dependency
            openai = None
        if openai is not None and isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None)
            response = getattr(exc, "response", None)
            if body is None and response is not None:
                body = getattr(response, "text", "") or str(exc)
            headers = dict(getattr(response, "headers", {}) or {})
            message, request_id = extract_upstream_error(
                status or 0, body, headers, api_key=self.api_key,
            )
            return LLMProviderError(
                DEEPSEEK_PROVIDER, self.model, status_code=status,
                upstream_message=message, request_id=request_id,
                retryable=is_retryable_status(status or 0), api_key=self.api_key,
            )
        if openai is not None and isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return LLMProviderError(
                DEEPSEEK_PROVIDER, self.model, status_code=None,
                upstream_message=f"{type(exc).__name__}: {exc}", retryable=True,
                api_key=self.api_key,
            )
        if isinstance(exc, httpx.TimeoutException):
            return LLMProviderError(
                DEEPSEEK_PROVIDER, self.model, status_code=None,
                upstream_message=f"Request timed out: {exc}", retryable=True,
                api_key=self.api_key,
            )
        if isinstance(exc, httpx.HTTPError):
            return LLMProviderError(
                DEEPSEEK_PROVIDER, self.model, status_code=None,
                upstream_message=f"Transport error: {exc}", retryable=True,
                api_key=self.api_key,
            )
        return None

    # -- chat interface (mirrors ChatLLMClient) ---------------------------------

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2,
                 max_tokens: int = 4096) -> str:
        """Multi-turn chat completion; returns the assistant message content.

        ``temperature`` is accepted for interface compatibility but deliberately
        NOT forwarded: DeepSeek thinking mode rejects sampling parameters.
        429/5xx/timeout failures are retried with backoff; every failure is
        raised as :class:`LLMProviderError` with the upstream detail preserved.
        """
        start = time.perf_counter()

        def _attempt():
            try:
                return self._openai_client().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # map SDK errors, keep the rest as-is
                mapped = self._map_error(exc)
                if mapped is not None:
                    raise mapped from exc
                raise

        response = call_with_retries(_attempt)
        # Thinking models spend completion tokens on reasoning first; when the
        # budget is exhausted the visible content comes back empty with
        # finish_reason="length". Retry with a doubled budget (bounded).
        budget = max_tokens
        for _ in range(2):
            choice0 = response.choices[0]
            if choice0.message.content and choice0.message.content.strip():
                break
            if getattr(choice0, "finish_reason", None) != "length" or budget >= 65536:
                break
            budget = min(budget * 2, 65536)
            response = call_with_retries(
                lambda: self._openai_client().chat.completions.create(
                    model=self.model, messages=messages, max_tokens=budget))
        latency = (time.perf_counter() - start) * 1000.0
        choice = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        self._last_response = _llm_response(
            content=choice,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=self.model,
            latency_ms=latency,
        )
        return choice

    def complete(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 4096, temperature: float = 0.0):
        """Single-turn completion compatible with the LLMClient protocol."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        content = self.generate(messages, max_tokens=max_tokens)
        if self._last_response is not None:
            return self._last_response
        return _llm_response(content=content, model=self.model)


def complete(prompt: str, tier: Tier = Tier.FAST, *, api_key: Optional[str] = None,
             system: Optional[str] = None, max_tokens: int = 4096,
             model: Optional[str] = None, client: Any = None):
    """One-shot DeepSeek completion; resolves the model via ``resolve_tier_model``."""
    adapter = DeepSeekChatClient(tier=tier, api_key=api_key, model=model, client=client)
    return adapter.complete(prompt, system=system, max_tokens=max_tokens)


def validate_api_key(api_key: str, base_url: Optional[str] = None,
                     client: Optional[httpx.Client] = None) -> Dict[str, Any]:
    """Validate a DeepSeek key via ``GET /models``.

    Returns ``{"valid": bool, "error": Optional[str], "models": [...]}``.
    ``client`` may be an ``httpx.Client`` (e.g. built on ``httpx.MockTransport``)
    for testing; otherwise a short-lived client is created.
    """
    base = (base_url or DEEPSEEK_BASE_URL).rstrip("/")
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        resp = http.get(f"{base}/models", headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as exc:
        return {"valid": False, "error": f"DeepSeek validation request failed: {exc}", "models": []}
    finally:
        if owns_client:
            http.close()

    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            data = {}
        models = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
        return {"valid": True, "error": None, "models": models}

    detail = ""
    try:
        payload = resp.json()
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            detail = str(err.get("message") or err)
        elif isinstance(payload, dict):
            detail = str(payload.get("message") or "")
    except ValueError:
        detail = resp.text[:200]
    message = f"DeepSeek rejected the API key (HTTP {resp.status_code})"
    if detail:
        message = f"{message}: {detail}"
    return {"valid": False, "error": message, "models": []}

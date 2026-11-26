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
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx

from invar_agent.config import Tier, resolve_tier_model
from invar_agent.llm.errors import (
    LLMProviderError,
    builder_transport_active,
    call_with_retries,
    extract_upstream_error,
    is_retryable_status,
)
from invar_agent.llm.telemetry import current_llm_context, extract_usage_tokens, record_llm_call

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
        self._injected_client = client is not None
        self._last_response = None

    # -- OpenAI SDK access -----------------------------------------------------

    def _openai_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("DEEPSEEK_API_KEY not configured; cannot call DeepSeek.")
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=httpx.Timeout(90.0, connect=20.0),
                max_retries=0,
            )
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
                 max_tokens: int = 4096, thinking: Optional[bool] = None,
                 json_object: bool = False) -> str:
        """Multi-turn chat completion; returns the assistant message content.

        ``temperature`` is accepted for interface compatibility but deliberately
        NOT forwarded: DeepSeek thinking mode rejects sampling parameters.
        429/5xx/timeout failures are retried with backoff; every failure is
        raised as :class:`LLMProviderError` with the upstream detail preserved.
        """
        start = time.perf_counter()

        extra: Dict[str, Any] = {}
        if json_object:
            extra["response_format"] = {"type": "json_object"}
        if thinking is False or (thinking is None and builder_transport_active()):
            # DeepSeek thinking mode burns the whole completion budget on
            # reasoning (observed: 1024/1024 tokens, empty content). Planning
            # calls disable it explicitly; the API accepts this on flash.
            extra["extra_body"] = {"thinking": {"type": "disabled"}}

        idle_timeout = float(os.environ.get("INVAR_DEEPSEEK_IDLE_TIMEOUT",
            os.environ.get("INVAR_DEEPSEEK_WALL_TIMEOUT", "180.0")))
        ctx = current_llm_context()
        tier_val = self.tier.value if isinstance(self.tier, Tier) else str(self.tier)
        est_input_chars = sum(len(str(m.get("content") or "")) for m in (messages or []))
        est_input_tokens = max(1, est_input_chars // 4)

        def _call_once(budget_val: int, extra_override: Optional[Dict[str, Any]] = None):
            req_extra = extra_override if extra_override is not None else extra
            try:
                client = self._openai_client()
                if not self._injected_client:
                    client = client.with_options(timeout=httpx.Timeout(idle_timeout, connect=20.0), max_retries=0)
                if builder_transport_active() and callable(getattr(client, "with_options", None)):
                    client = client.with_options(max_retries=0)
                if not self._injected_client:
                    import sys as _sys
                    try:
                        _sys.stderr.write(
                            f"[LLM_DISPATCH] phase={ctx.get('phase') or 'deepseek_generate'} "
                            f"tier={ctx.get('tier') or tier_val} model={self.model} "
                            f"est_input_tokens={est_input_tokens} max_tokens={budget_val} "
                            f"thinking={'disabled' if req_extra.get('extra_body') else 'auto'}\n"
                        )
                        _sys.stderr.flush()
                    except Exception:
                        pass
                    last_chunk_at = time.monotonic()
                    stream = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=budget_val,
                        stream=True,
                        stream_options={"include_usage": True},
                        **req_extra,
                    )
                    content_parts: List[str] = []
                    reasoning_parts: List[str] = []
                    finish_reason: Optional[str] = None
                    usage_obj: Any = None
                    try:
                        for chunk in stream:
                            now = time.monotonic()
                            if now - last_chunk_at > idle_timeout:
                                raise LLMProviderError(DEEPSEEK_PROVIDER, self.model,
                                    upstream_message=f"Streaming idle timeout after {idle_timeout:.0f}s without a chunk",
                                    retryable=True, api_key=self.api_key)
                            last_chunk_at = now
                            if getattr(chunk, "usage", None) is not None:
                                usage_obj = chunk.usage
                            choices = getattr(chunk, "choices", None) or ()
                            if choices:
                                ch = choices[0]
                                if getattr(ch, "finish_reason", None):
                                    finish_reason = ch.finish_reason
                                delta = getattr(ch, "delta", None)
                                text_delta = getattr(delta, "content", None) if delta is not None else None
                                if text_delta:
                                    content_parts.append(text_delta)
                                reason_delta = getattr(delta, "reasoning_content", None) if delta is not None else None
                                if reason_delta:
                                    reasoning_parts.append(reason_delta)
                    finally:
                        closer = getattr(stream, "close", None)
                        if callable(closer):
                            try:
                                closer()
                            except Exception:
                                pass
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="".join(content_parts),
                                    reasoning_content="".join(reasoning_parts),
                                ),
                                finish_reason=finish_reason,
                            )
                        ],
                        usage=usage_obj,
                    )
                return client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=budget_val,
                    **req_extra,
                )
            except Exception as exc:  # map SDK errors, keep the rest as-is
                if isinstance(exc, LLMProviderError):
                    raise
                mapped = self._map_error(exc)
                if mapped is not None:
                    raise mapped from exc
                raise

        response = call_with_retries(lambda: _call_once(max_tokens))
        # Thinking models spend completion tokens on reasoning first; when the
        # budget is exhausted the visible content comes back empty with
        # finish_reason="length". Retry at most ONCE with a bounded budget (<=16384),
        # disabling thinking on FAST tier so it never loops in empty chain-of-thought.
        budget = max_tokens
        max_retry_budget = 16384
        for _ in range(0 if builder_transport_active() else 1):
            choice0 = response.choices[0]
            _raw_content = choice0.message.content or ""
            _txt = _raw_content.strip()
            # DeepSeek U+0020 whitespace-only stall: non-empty whitespace with no visible payload.
            _whitespace_only = len(_raw_content) > 0 and len(_txt) == 0
            _truncated_json = (
                getattr(choice0, "finish_reason", None) == "length"
                and _txt.startswith(("{", "```"))
                and not _txt.endswith(("}", "```"))
            )
            if _txt and not _truncated_json:
                break
            if not _whitespace_only and (
                getattr(choice0, "finish_reason", None) != "length" or budget >= max_retry_budget
            ):
                break
            budget = min(max(budget * 2, 4096), max_retry_budget)
            retry_extra = dict(extra)
            if self.tier == Tier.FAST and not self._injected_client:
                retry_extra["extra_body"] = {"thinking": {"type": "disabled"}}
            response = call_with_retries(lambda b=budget, rx=retry_extra: _call_once(b, rx))
        latency = (time.perf_counter() - start) * 1000.0
        choice_msg = response.choices[0].message
        choice = getattr(choice_msg, "content", "") or ""
        usage = getattr(response, "usage", None)
        tok_meta = extract_usage_tokens(usage)
        if tok_meta["reasoning_tokens"] == 0:
            r_text = getattr(choice_msg, "reasoning_content", "") or ""
            if r_text:
                tok_meta["reasoning_tokens"] = max(1, len(r_text) // 4)
        rec = record_llm_call(
            model=self.model,
            tier=ctx.get("tier") or tier_val,
            phase=ctx.get("phase") or "deepseek_generate",
            prompt_tokens=tok_meta["prompt_tokens"],
            completion_tokens=tok_meta["completion_tokens"],
            reasoning_tokens=tok_meta["reasoning_tokens"],
            cache_hit_tokens=tok_meta["cache_hit_tokens"],
            latency_s=latency / 1000.0,
            attempt=ctx.get("attempt") or 1,
        )
        self._last_response = _llm_response(
            content=choice,
            input_tokens=tok_meta["prompt_tokens"],
            output_tokens=tok_meta["completion_tokens"],
            reasoning_tokens=tok_meta["reasoning_tokens"],
            cache_hit_tokens=tok_meta["cache_hit_tokens"],
            model=self.model,
            latency_ms=latency,
            latency_s=rec.latency_s,
            tier=rec.tier,
            phase=rec.phase,
            attempt=rec.attempt,
        )
        return choice

    def complete(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 4096, temperature: float = 0.0,
                 thinking: Optional[bool] = None):
        """Single-turn completion compatible with the LLMClient protocol."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        content = self.generate(messages, max_tokens=max_tokens, thinking=thinking)
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

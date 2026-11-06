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

    # -- chat interface (mirrors ChatLLMClient) ---------------------------------

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2,
                 max_tokens: int = 4096) -> str:
        """Multi-turn chat completion; returns the assistant message content.

        ``temperature`` is accepted for interface compatibility but deliberately
        NOT forwarded: DeepSeek thinking mode rejects sampling parameters.
        """
        start = time.perf_counter()
        response = self._openai_client().chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
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

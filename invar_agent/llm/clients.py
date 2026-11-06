"""Tiered LLM clients with token budget accounting."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import httpx

from ..config import Tier, resolve_tier_model
from ..orchestrator.router import CognitiveModelRouter, CognitiveTier, ModelRouteDecision, TaskEventType, TokenBudgetGuardrail


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


class TokenBudgetAccountant:
    """Tracks tiered token consumption against guardrail budgets."""

    def __init__(self, guardrail: Optional[TokenBudgetGuardrail] = None) -> None:
        self.budget = guardrail or TokenBudgetGuardrail()

    def record(self, tier: CognitiveTier, input_tokens: int, output_tokens: int) -> None:
        self.budget.record_usage(tier, input_tokens, output_tokens)

    def remaining(self, tier: CognitiveTier) -> int:
        if tier == CognitiveTier.TIER_2_PRO:
            return max(0, self.budget.max_pro_tokens - self.budget.used_pro_tokens)
        return max(0, self.budget.max_flash_tokens - self.budget.used_flash_tokens)


class _OpenRouterClient:
    """Minimal OpenRouter chat-completions client (used only when keyed)."""

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

    def complete(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not configured; cannot call remote model.")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        start = time.perf_counter()
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        latency = (time.perf_counter() - start) * 1000.0
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=choice,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=self.model,
            latency_ms=latency,
        )


class ProModelClient:
    def __init__(self, model: str = "gemini-2.5-pro", api_key: Optional[str] = None) -> None:
        self._client = _OpenRouterClient(model, api_key)
        self.model = model

    def complete(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        return self._client.complete(prompt, system, max_tokens, temperature)


class FlashModelClient:
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None) -> None:
        self._client = _OpenRouterClient(model, api_key)
        self.model = model

    def complete(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        return self._client.complete(prompt, system, max_tokens, temperature)


class TieredLLMClient:
    """Routes prompts through the CognitiveModelRouter and records budgets."""

    def __init__(
        self,
        router: Optional[CognitiveModelRouter] = None,
        pro_client: Optional[LLMClient] = None,
        flash_client: Optional[LLMClient] = None,
    ) -> None:
        self.router = router or CognitiveModelRouter()
        self.pro_client = pro_client or ProModelClient(self.router.pro_model)
        self.flash_client = flash_client or FlashModelClient(self.router.flash_model)
        self.accountant = TokenBudgetAccountant(self.router.budget)

    def complete(
        self,
        event_type: TaskEventType,
        prompt: str,
        system: Optional[str] = None,
        **route_kwargs: Any,
    ) -> tuple[ModelRouteDecision, LLMResponse]:
        decision = self.router.route(event_type, **route_kwargs)
        if decision.tier == CognitiveTier.TIER_0_REFLEX:
            raise RuntimeError("Tier-0 reflex path must be handled locally without an LLM call.")
        client = self.pro_client if decision.tier == CognitiveTier.TIER_2_PRO else self.flash_client
        response = client.complete(
            prompt,
            system=system,
            max_tokens=decision.max_output_tokens,
            temperature=decision.temperature,
        )
        # Rough token accounting when provider does not report usage.
        in_tokens = response.input_tokens or max(1, len(prompt) // 4)
        out_tokens = response.output_tokens or max(1, len(response.content) // 4)
        self.accountant.record(decision.tier, in_tokens, out_tokens)
        return decision, response


# ---------------------------------------------------------------------------
# Tier-resolved chat clients (copilot FAST / builder REASONING)
# ---------------------------------------------------------------------------

_PROVIDER_BASE_URLS: Dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}

_PROVIDER_KEY_ENVS: Dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class ChatLLMClient:
    """OpenAI-compatible chat client bound to one resolved tier model."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tier: Tier = Tier.FAST,
    ) -> None:
        self.provider = provider
        self.model = model
        self.tier = tier
        self.api_key = api_key or os.environ.get(_PROVIDER_KEY_ENVS.get(provider, ""), "")
        self.base_url = base_url or _PROVIDER_BASE_URLS.get(provider, _PROVIDER_BASE_URLS["openrouter"])

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2,
                 max_tokens: int = 4096) -> str:
        """Multi-turn chat completion; returns the assistant message content."""
        if not self.api_key:
            raise RuntimeError(f"No API key configured for provider '{self.provider}'.")
        start = time.perf_counter()
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                self.base_url,
                headers=self._headers(),
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        latency = (time.perf_counter() - start) * 1000.0
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
        response = getattr(self, "_last_response", None)
        if response is not None:
            return response
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
        or os.environ.get(_PROVIDER_KEY_ENVS.get(provider_name, ""), "")
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

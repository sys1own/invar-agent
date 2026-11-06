"""LLM provider and tier configuration shared by the copilot and the builder."""
from __future__ import annotations

import copy
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class Tier(str, Enum):
    """LLM routing tiers for model selection."""

    FAST = "fast"          # light model — copilot chat and lightweight sub-tasks
    REASONING = "reasoning"  # pro model — builder code generation and repair


DEFAULT_TIER_MODELS: Dict[str, Dict[str, str]] = {
    "gemini": {"fast": "gemini-2.5-flash", "reasoning": "gemini-2.5-pro"},
    "openai": {"fast": "gpt-4o-mini", "reasoning": "gpt-4o"},
    # DeepSeek retired deepseek-chat / deepseek-reasoner on 2026-07-24; the
    # current IDs are deepseek-flash (V4.1-Flash) and deepseek-v4-pro (V4-Pro).
    "deepseek": {"fast": "deepseek-flash", "reasoning": "deepseek-v4-pro"},
    "openrouter": {"fast": "openrouter/auto", "reasoning": "openrouter/auto"},
}

# Retired DeepSeek model IDs — never valid, kept only so tests can assert their absence.
RETIRED_DEEPSEEK_MODELS = frozenset({"deepseek-chat", "deepseek-reasoner"})

DEFAULT_CONFIG_PATH = Path.home() / ".invar-agent" / "config.json"


def _tier_env_name(tier: str) -> str:
    return f"INVAR_LLM_TIER_{tier.upper()}"


def _provider_tier_env_name(provider: str, tier: str) -> str:
    """Provider-specific override, e.g. ``DEEPSEEK_MODEL_FAST``."""
    return f"{provider.upper()}_MODEL_{tier.upper()}"


def _parse_tier_env(value: str, provider: str, tier: str) -> Optional[str]:
    """Parse an env override: JSON mapping ``{provider: model}`` or a raw model name."""
    value = value.strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, dict):
        candidate = parsed.get(provider)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        # Nested per-tier mapping: {"gemini": {"fast": "..."}}
        for key, val in parsed.items():
            if key == provider and isinstance(val, dict):
                model = val.get(tier)
                if isinstance(model, str) and model.strip():
                    return model.strip()
    return None


def load_file_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the optional JSON file config (``tier_models`` section)."""
    config_path = Path(path) if path else Path(os.environ.get("INVAR_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    try:
        if not config_path.is_file():
            return {}
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _file_config_model(file_config: Dict[str, Any], provider: str, tier: str) -> Optional[str]:
    tier_section = file_config.get("tier_models") or {}
    if not isinstance(tier_section, dict):
        return None
    provider_entry = tier_section.get(provider)
    if isinstance(provider_entry, dict):
        model = provider_entry.get(tier)
        if isinstance(model, str) and model.strip():
            return model.strip()
    # Flat keys such as "gemini_fast".
    flat = tier_section.get(f"{provider}_{tier}")
    if isinstance(flat, str) and flat.strip():
        return flat.strip()
    return None


def resolve_tier_model(
    provider: str,
    tier: str | Tier,
    file_config: Optional[Dict[str, Any]] = None,
    override: Optional[Any] = None,
) -> Optional[str]:
    """Return the model name for ``provider`` and ``tier``.

    Resolution precedence (first hit wins):
    1. Explicit override (a model string or an object with a ``model`` attribute).
    2. Provider-specific environment variable, e.g. ``DEEPSEEK_MODEL_FAST`` /
       ``DEEPSEEK_MODEL_REASONING`` (raw model name or JSON mapping).
    3. Generic environment variable ``INVAR_LLM_TIER_<TIER>`` (JSON mapping or raw model name).
    4. File config ``tier_models`` section.
    5. ``DEFAULT_TIER_MODELS``.
    """
    if override is not None:
        model = override if isinstance(override, str) else getattr(override, "model", None)
        if isinstance(model, str) and model.strip():
            return model.strip()

    provider = (provider or "").lower()
    tier_value = tier.value if isinstance(tier, Tier) else str(tier).lower()

    provider_env = os.environ.get(_provider_tier_env_name(provider, tier_value))
    if provider_env:
        model = _parse_tier_env(provider_env, provider, tier_value)
        if model:
            return model

    env_value = os.environ.get(_tier_env_name(tier_value))
    if env_value:
        model = _parse_tier_env(env_value, provider, tier_value)
        if model:
            return model

    config = file_config if file_config is not None else load_file_config()
    model = _file_config_model(config, provider, tier_value)
    if model:
        return model

    return copy.deepcopy(DEFAULT_TIER_MODELS).get(provider, {}).get(tier_value)

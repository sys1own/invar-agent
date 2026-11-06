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
    # Gemini 2.5 models are legacy (access-limited for new projects per the
    # deprecations doc); current GA IDs verified against the live
    # /v1beta/openai/models listing on 2026-11-06.
    "gemini": {"fast": "gemini-3.8-flash", "reasoning": "gemini-3.1-pro-preview"},
    "openai": {"fast": "gpt-5.4-mini", "reasoning": "gpt-5.5-pro"},
    # DeepSeek retired deepseek-chat / deepseek-reasoner on 2026-07-24; the
    # current IDs are deepseek-flash (V4.1-Flash) and deepseek-v4-pro (V4-Pro).
    "deepseek": {"fast": "deepseek-flash", "reasoning": "deepseek-v4-pro"},
    "openrouter": {"fast": "openrouter/auto", "reasoning": "openrouter/auto"},
}

# Ordered fallback candidates per provider/tier, used to auto-pick a model when
# the resolved default is missing from the provider's live /models listing.
TIER_MODEL_PREFERENCES: Dict[str, Dict[str, list]] = {
    "gemini": {
        "fast": ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                 "gemini-3.5-flash", "gemini-flash-latest", "gemini-2.5-flash"],
        "reasoning": ["gemini-3.1-pro-preview", "gemini-pro-latest", "gemini-2.5-pro"],
    },
    "openai": {
        "fast": ["gpt-5.4-mini", "gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini"],
        "reasoning": ["gpt-5.5-pro", "gpt-5.5", "gpt-5.4", "gpt-5", "gpt-4o"],
    },
    "deepseek": {"fast": ["deepseek-flash"], "reasoning": ["deepseek-v4-pro"]},
    "openrouter": {"fast": ["openrouter/auto"], "reasoning": ["openrouter/auto"]},
}

# Retired DeepSeek model IDs — never valid, kept only so tests can assert their absence.
RETIRED_DEEPSEEK_MODELS = frozenset({"deepseek-chat", "deepseek-reasoner"})

# Expected API-key prefixes, used only for soft mismatch warnings.
PROVIDER_KEY_PREFIXES: Dict[str, str] = {
    "gemini": "AIza",
    "openrouter": "sk-or-",
    "deepseek": "sk-",
    "openai": "sk-",
}

PROVIDER_KEY_ENVS: Dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
GENERIC_KEY_ENV = "INVAR_API_KEY"
PROVIDER_ENV_VAR = "INVAR_LLM_PROVIDER"

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
    """Return the model name for ``provider`` and ``tier`` (see precedence below)."""
    model, _source = resolve_tier_model_source(provider, tier, file_config=file_config, override=override)
    return model


def resolve_tier_model_source(
    provider: str,
    tier: str | Tier,
    file_config: Optional[Dict[str, Any]] = None,
    override: Optional[Any] = None,
) -> tuple[Optional[str], str]:
    """Like :func:`resolve_tier_model` but also reports where the value came from.

    Returns ``(model, source)`` with ``source`` one of ``"override"``, ``"env"``,
    ``"file"``, ``"default"`` or ``"unknown"``.

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
            return model.strip(), "override"

    provider = (provider or "").lower()
    tier_value = tier.value if isinstance(tier, Tier) else str(tier).lower()

    provider_env = os.environ.get(_provider_tier_env_name(provider, tier_value))
    if provider_env:
        model = _parse_tier_env(provider_env, provider, tier_value)
        if model:
            return model, "env"

    env_value = os.environ.get(_tier_env_name(tier_value))
    if env_value:
        model = _parse_tier_env(env_value, provider, tier_value)
        if model:
            return model, "env"

    config = file_config if file_config is not None else load_file_config()
    model = _file_config_model(config, provider, tier_value)
    if model:
        return model, "file"

    default = copy.deepcopy(DEFAULT_TIER_MODELS).get(provider, {}).get(tier_value)
    if default:
        return default, "default"
    return None, "unknown"


def detect_llm_from_env() -> Dict[str, Optional[str]]:
    """Auto-detect provider and API key from the environment.

    ``INVAR_LLM_PROVIDER`` picks the provider explicitly; otherwise exactly one
    provider key present in the environment selects it. Returns
    ``{"provider": ..., "api_key": ...}`` (either may be None).
    """
    provider_pref = (os.environ.get(PROVIDER_ENV_VAR) or "").strip().lower() or None
    keys = {p: (os.environ.get(env) or "").strip() for p, env in PROVIDER_KEY_ENVS.items()}
    present = {p for p, k in keys.items() if k}

    if provider_pref:
        api_key = keys.get(provider_pref) or (os.environ.get(GENERIC_KEY_ENV) or "").strip() or None
        return {"provider": provider_pref, "api_key": api_key}
    if len(present) == 1:
        only = next(iter(present))
        api_key = keys[only] or (os.environ.get(GENERIC_KEY_ENV) or "").strip() or None
        return {"provider": only, "api_key": api_key}
    return {"provider": None, "api_key": None}


def persist_config(
    provider: str,
    tier_overrides: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    path: Optional[str | Path] = None,
) -> Path:
    """Persist provider + explicit tier overrides to the config file (0600).

    The API key is only persisted when ``INVAR_PERSIST_KEY=1`` is set.
    """
    config_path = Path(path) if path else Path(os.environ.get("INVAR_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    payload: Dict[str, Any] = {"provider": provider}
    overrides = {k: v for k, v in (tier_overrides or {}).items() if isinstance(v, str) and v.strip()}
    if overrides:
        payload["tier_models"] = {provider: overrides}
    if api_key and os.environ.get("INVAR_PERSIST_KEY") == "1":
        payload["api_key"] = api_key
    config_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(config_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.chmod(str(config_path), 0o600)
    return config_path


def load_persisted_state(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load persisted provider/overrides (and key only with INVAR_PERSIST_KEY=1)."""
    data = load_file_config(path)
    api_key = data.get("api_key") if isinstance(data.get("api_key"), str) else None
    if os.environ.get("INVAR_PERSIST_KEY") != "1":
        api_key = None
    return {
        "provider": data.get("provider") if isinstance(data.get("provider"), str) else None,
        "api_key": api_key,
        "tier_models": data.get("tier_models") if isinstance(data.get("tier_models"), dict) else {},
    }

"""Live provider validation and model discovery (``GET {base}/models``)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from ..config import (
    PROVIDER_KEY_PREFIXES,
    TIER_MODEL_PREFERENCES,
    Tier,
    resolve_tier_model_source,
)
from .clients import models_url
from .errors import extract_upstream_error, redact_secrets

_USER_SOURCES = {"override", "env", "file"}


def _strip_model_prefix(model_id: str) -> str:
    return model_id.strip().removeprefix("models/")


def _extract_model_ids(payload: Any) -> List[str]:
    """Normalize OpenAI-style ``data`` and Google-style ``models`` listings."""
    ids: List[str] = []
    entries: List[Any] = []
    if isinstance(payload, dict):
        entries = payload.get("data") or payload.get("models") or []
    elif isinstance(payload, list):
        entries = payload
    for entry in entries:
        if isinstance(entry, str):
            ids.append(_strip_model_prefix(entry))
        elif isinstance(entry, dict):
            raw = entry.get("id") or entry.get("name") or ""
            if raw:
                ids.append(_strip_model_prefix(str(raw)))
    # de-duplicate, preserve order
    seen = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def _key_format_warning(provider: str, api_key: str) -> Optional[str]:
    prefix = PROVIDER_KEY_PREFIXES.get(provider)
    if prefix and api_key and not api_key.startswith(prefix):
        return (
            f"API key format does not match provider '{provider}' "
            f"(expected keys to start with '{prefix}'); check the key/provider match."
        )
    return None


def _pick_available(provider: str, tier: Tier, available: List[str]) -> Optional[str]:
    """First preference-list model present in ``available``."""
    available_set = set(available)
    for candidate in TIER_MODEL_PREFERENCES.get(provider, {}).get(tier.value, []):
        if candidate in available_set:
            return candidate
    return None


def validate_provider(
    provider: str,
    api_key: str,
    tier_overrides: Optional[Dict[str, str]] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Validate an API key and resolve tier models against the live catalog.

    Returns ``{"valid", "error", "models", "tier_models", "warnings"}``.
    ``tier_overrides`` are explicit per-tier model overrides (``{"fast": ...,
    "reasoning": ...}``); when the resolved model is missing from the provider's
    ``/models`` listing, user-configured values (override/env/file) produce a
    warning with suggestions while factory defaults are auto-picked from
    :data:`invar_agent.config.TIER_MODEL_PREFERENCES`.
    """
    provider = (provider or "").lower()
    warnings: List[str] = []
    result: Dict[str, Any] = {
        "valid": False,
        "error": None,
        "models": [],
        "tier_models": {},
        "auto_picked": {},
        "warnings": warnings,
    }
    if not provider or not api_key:
        result["error"] = "provider and api_key are required"
        return result

    format_warning = _key_format_warning(provider, api_key)
    if format_warning:
        warnings.append(format_warning)

    url = models_url(provider)
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        resp = http.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as exc:
        result["error"] = redact_secrets(f"{provider} validation request failed: {exc}", api_key)
        return result
    finally:
        if owns_client:
            http.close()

    if resp.status_code != 200:
        message, _request_id = extract_upstream_error(
            resp.status_code, resp.text, dict(resp.headers), api_key=api_key,
        )
        result["error"] = f"{provider} rejected the API key (HTTP {resp.status_code}): {message}"
        return result

    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    models = _extract_model_ids(payload)
    result["valid"] = True
    result["models"] = models

    overrides = tier_overrides or {}
    available = set(models)
    for tier in (Tier.FAST, Tier.REASONING):
        override = overrides.get(tier.value)
        model, source = resolve_tier_model_source(provider, tier, override=override)
        if not model:
            continue
        if model in available:
            result["tier_models"][tier.value] = model
            continue
        pick = _pick_available(provider, tier, models)
        if source in _USER_SOURCES:
            suggestions = [m for m in TIER_MODEL_PREFERENCES.get(provider, {}).get(tier.value, []) if m in available]
            hint = f"; suggestions: {', '.join(suggestions)}" if suggestions else ""
            warnings.append(
                f"{tier.value} model '{model}' (from {source}) is not in the provider's model list{hint}. "
                "Update the tier override or the provider's available models."
            )
            result["tier_models"][tier.value] = model
        elif pick:
            warnings.append(
                f"Default {tier.value} model '{model}' is not available for '{provider}'; "
                f"auto-selected '{pick}'."
            )
            result["tier_models"][tier.value] = pick
            result["auto_picked"][tier.value] = pick
        else:
            warnings.append(
                f"Default {tier.value} model '{model}' is not in the provider's model list "
                "and no preference-list fallback is available."
            )
            result["tier_models"][tier.value] = model
    return result

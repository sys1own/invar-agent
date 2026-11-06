"""Optional live smoke tests against real providers (marker: live).

Opt-in only: run with ``pytest -m live``. Also skipped unless
GEMINI_API_KEY or DEEPSEEK_API_KEY is set in the environment.
"""
from __future__ import annotations

import os

import pytest

from invar_agent.config import Tier
from invar_agent.llm.clients import get_llm_client
from invar_agent.llm.validate import validate_provider

pytestmark = pytest.mark.live

_LIVE = [
    ("gemini", "GEMINI_API_KEY"),
    ("deepseek", "DEEPSEEK_API_KEY"),
]


@pytest.mark.parametrize("provider,env", _LIVE)
def test_live_validate_and_fast_completion(provider, env, monkeypatch, tmp_path):
    key = os.environ.get(env)
    if not key:
        pytest.skip(f"{env} not set")
    monkeypatch.setenv("INVAR_CONFIG_PATH", str(tmp_path / "config.json"))
    verdict = validate_provider(provider, key)
    assert verdict["valid"] is True, verdict["error"]
    for tier in ("fast", "reasoning"):
        assert verdict["tier_models"][tier] in verdict["models"], verdict["warnings"]
    client = get_llm_client(provider, tier=Tier.FAST, api_key=key, override=verdict["tier_models"]["fast"])
    response = client.complete("Reply with exactly: OK", max_tokens=64)
    assert response.model == verdict["tier_models"]["fast"]

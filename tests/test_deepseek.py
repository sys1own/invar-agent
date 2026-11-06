"""DeepSeek native support: tier model routing, OpenAI-SDK adapter, key validation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import invar_agent.server as server
from invar_agent.config import DEFAULT_TIER_MODELS, RETIRED_DEEPSEEK_MODELS, Tier, resolve_tier_model
from invar_agent.core.agent import VerifiedAutonomousAgent
from invar_agent.copilot.agent import CopilotAgent
from invar_agent.llm.clients import get_llm_client
from invar_agent.llm.deepseek import DeepSeekChatClient, complete, validate_api_key
from invar_agent.server import create_app


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path_factory):
    """Never leak real provider keys, tier overrides or file config into these tests."""
    for var in (
        "GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
        "INVAR_API_KEY", "INVAR_LLM_PROVIDER", "INVAR_LLM_TIER_FAST", "INVAR_LLM_TIER_REASONING",
        "DEEPSEEK_MODEL_FAST", "DEEPSEEK_MODEL_REASONING",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("INVAR_CONFIG_PATH", str(tmp_path_factory.mktemp("cfg") / "config.json"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")


# ---------------------------------------------------------------------------
# Unit: model resolution
# ---------------------------------------------------------------------------

def test_deepseek_default_tier_models():
    assert DEFAULT_TIER_MODELS["deepseek"] == {"fast": "deepseek-flash", "reasoning": "deepseek-v4-pro"}
    assert resolve_tier_model("deepseek", "fast") == "deepseek-flash"
    assert resolve_tier_model("deepseek", Tier.FAST) == "deepseek-flash"
    assert resolve_tier_model("deepseek", Tier.REASONING) == "deepseek-v4-pro"


def test_no_retired_deepseek_model_ids_anywhere():
    """deepseek-chat / deepseek-reasoner were retired on 2026-07-24 (400 errors)."""
    for provider, tiers in DEFAULT_TIER_MODELS.items():
        for model in tiers.values():
            assert model not in RETIRED_DEEPSEEK_MODELS, f"{provider} still maps to retired {model}"


def test_deepseek_env_precedence(monkeypatch):
    # Provider-specific env var wins over the generic INVAR_LLM_TIER_* one.
    monkeypatch.setenv("INVAR_LLM_TIER_FAST", json.dumps({"deepseek": "generic-fast"}))
    monkeypatch.setenv("DEEPSEEK_MODEL_FAST", "deepseek-custom-fast")
    assert resolve_tier_model("deepseek", Tier.FAST) == "deepseek-custom-fast"
    # Explicit override beats every env var.
    assert resolve_tier_model("deepseek", Tier.FAST, override="explicit-model") == "explicit-model"
    # Generic env still applies when no provider-specific one is set.
    monkeypatch.delenv("DEEPSEEK_MODEL_FAST")
    assert resolve_tier_model("deepseek", Tier.FAST) == "generic-fast"


def test_deepseek_file_config_fallback(monkeypatch):
    monkeypatch.delenv("INVAR_LLM_TIER_FAST", raising=False)
    cfg = {"tier_models": {"deepseek": {"fast": "file-fast"}}}
    assert resolve_tier_model("deepseek", Tier.FAST, file_config=cfg) == "file-fast"
    # Flat key form also works.
    cfg_flat = {"tier_models": {"deepseek_fast": "file-fast-flat"}}
    assert resolve_tier_model("deepseek", Tier.FAST, file_config=cfg_flat) == "file-fast-flat"


# ---------------------------------------------------------------------------
# Unit: client adapter (mocked OpenAI client)
# ---------------------------------------------------------------------------

class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_fast_tier_client_uses_deepseek_flash():
    fake = _FakeOpenAIClient()
    client = DeepSeekChatClient(tier=Tier.FAST, api_key="sk-test", client=fake)
    assert client.model == "deepseek-flash"
    response = client.complete("hello")
    assert response.content == "ok"
    call = fake.chat.completions.calls[0]
    assert call["model"] == "deepseek-flash"
    # Thinking mode rejects sampling params — they must never be sent.
    assert "temperature" not in call
    assert "top_p" not in call


def test_reasoning_tier_client_uses_deepseek_v4_pro():
    fake = _FakeOpenAIClient()
    client = DeepSeekChatClient(tier=Tier.REASONING, api_key="sk-test", client=fake)
    assert client.model == "deepseek-v4-pro"
    client.generate([{"role": "user", "content": "hi"}])
    call = fake.chat.completions.calls[0]
    assert call["model"] == "deepseek-v4-pro"
    assert "temperature" not in call and "top_p" not in call


def test_module_level_complete_resolves_tier_model():
    fake = _FakeOpenAIClient()
    fast = complete("hi", tier=Tier.FAST, client=fake)
    assert fake.chat.completions.calls[-1]["model"] == "deepseek-flash"
    complete("hi", tier=Tier.REASONING, client=fake)
    assert fake.chat.completions.calls[-1]["model"] == "deepseek-v4-pro"
    assert fast.model == "deepseek-flash"


def test_explicit_model_override_wins():
    fake = _FakeOpenAIClient()
    client = DeepSeekChatClient(tier=Tier.FAST, api_key="sk-test", model="custom-model", client=fake)
    assert client.model == "custom-model"


def test_get_llm_client_routes_deepseek_to_adapter():
    fast = get_llm_client("deepseek", tier=Tier.FAST, api_key="sk-test")
    assert isinstance(fast, DeepSeekChatClient)
    assert fast.model == "deepseek-flash"
    pro = get_llm_client("deepseek", tier=Tier.REASONING, api_key="sk-test")
    assert isinstance(pro, DeepSeekChatClient)
    assert pro.model == "deepseek-v4-pro"
    # Other providers keep the generic HTTP chat client.
    gemini = get_llm_client("gemini", tier=Tier.FAST, api_key="k")
    assert not isinstance(gemini, DeepSeekChatClient)


def test_copilot_deepseek_uses_fast_adapter(tmp_path):
    agent = CopilotAgent(tmp_path, provider="deepseek", api_key="sk-test")
    client = agent._llm_client()
    assert isinstance(client, DeepSeekChatClient)
    assert client.model == "deepseek-flash"


def test_builder_healers_resolve_deepseek_tiers(tmp_path):
    agent = VerifiedAutonomousAgent(tmp_path, llm_provider="deepseek", llm_api_key="sk-test")
    healer = agent._healer_for()
    flash = healer.flash_healer._resolve_client()
    pro = healer.pro_healer._resolve_client()
    assert isinstance(flash, DeepSeekChatClient) and flash.model == "deepseek-flash"
    assert isinstance(pro, DeepSeekChatClient) and pro.model == "deepseek-v4-pro"


# ---------------------------------------------------------------------------
# Integration: API key validation (mocked GET /models)
# ---------------------------------------------------------------------------

def _models_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_validate_api_key_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/models"
        assert request.headers["authorization"] == "Bearer sk-good"
        return httpx.Response(200, json={"data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]})

    verdict = validate_api_key("sk-good", client=_models_client(handler))
    assert verdict["valid"] is True
    assert verdict["error"] is None
    assert "deepseek-flash" in verdict["models"]
    assert "deepseek-v4-pro" in verdict["models"]


def test_validate_api_key_invalid():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    verdict = validate_api_key("sk-bad", client=_models_client(handler))
    assert verdict["valid"] is False
    assert "401" in verdict["error"]
    assert "Invalid API key" in verdict["error"]


def test_validate_api_key_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    verdict = validate_api_key("sk-x", client=_models_client(handler))
    assert verdict["valid"] is False
    assert "boom" in verdict["error"]


# ---------------------------------------------------------------------------
# Integration: /api/config with DeepSeek
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
                "OPENROUTER_API_KEY", "INVAR_API_KEY", "INVAR_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("INVAR_CONFIG_PATH", str(tmp_path / "invar-config.json"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    monkeypatch.setattr(server, "_STATE", None)
    app = create_app(str(ws))
    return TestClient(app)


def _mock_validation(monkeypatch, valid: bool, error: str = "",
                     models=None, tier_models=None):
    """Mock the server-side validate_provider used by /api/config."""
    calls = []

    def fake_validate(provider, api_key, tier_overrides=None, **kwargs):
        calls.append({"provider": provider, "api_key": api_key})
        if valid:
            return {
                "valid": True, "error": None,
                "models": models if models is not None else ["deepseek-flash", "deepseek-v4-pro"],
                "tier_models": tier_models or {"fast": "deepseek-flash", "reasoning": "deepseek-v4-pro"},
                "warnings": [],
            }
        return {"valid": False, "error": error or "provider rejected the API key (HTTP 401)",
                "models": [], "tier_models": {}, "warnings": []}

    monkeypatch.setattr("invar_agent.server.validate_provider", fake_validate)
    return calls


def test_api_config_deepseek_invalid_key_rejected(client, monkeypatch):
    _mock_validation(monkeypatch, valid=False, error="DeepSeek rejected the API key (HTTP 401): Invalid API key")
    resp = client.post("/api/config", json={"provider": "deepseek", "api_key": "sk-bad"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "DeepSeek" in detail["error"]


def test_api_config_deepseek_valid_key_stores_tier_overrides(client, monkeypatch):
    _mock_validation(monkeypatch, valid=True)
    resp = client.post("/api/config", json={
        "provider": "deepseek",
        "api_key": "sk-good",
        "tier_overrides": {"fast": "deepseek-flash", "reasoning": "deepseek-v4-pro"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is True
    assert data["provider"] == "deepseek"
    assert data["api_key_set"] is True
    assert data["key_validated"] is True
    assert data["tier_models"] == {"fast": "deepseek-flash", "reasoning": "deepseek-v4-pro"}
    assert "available_models" in data and "warnings" in data
    state = server.get_state()
    assert state.provider == "deepseek"
    assert state.tier_overrides == {"fast": "deepseek-flash", "reasoning": "deepseek-v4-pro"}
    # The copilot must resolve the FAST adapter with the stored override.
    copilot = state.ensure_copilot()
    assert copilot.tier_overrides["fast"] == "deepseek-flash"
    # The key must never be echoed back or persisted to the config file.
    assert "sk-good" not in json.dumps(data)
    cfg_path = Path(os.environ["INVAR_CONFIG_PATH"])
    assert cfg_path.exists()
    assert "sk-good" not in cfg_path.read_text()


def test_api_config_validates_all_providers(client, monkeypatch):
    calls = _mock_validation(
        monkeypatch, valid=True,
        models=["gemini-3.8-flash", "gemini-3.1-pro-preview"],
        tier_models={"fast": "gemini-3.8-flash", "reasoning": "gemini-3.1-pro-preview"},
    )
    resp = client.post("/api/config", json={"provider": "gemini", "api_key": "AIza-test"})
    assert resp.status_code == 200
    # Non-DeepSeek providers are validated too (no more unvalidated storage).
    assert calls and calls[0]["provider"] == "gemini"
    data = resp.json()
    assert data["key_validated"] is True
    assert data["tier_models"]["fast"] == "gemini-3.8-flash"

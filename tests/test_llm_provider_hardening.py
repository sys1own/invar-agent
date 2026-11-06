"""Provider-layer hardening: structured errors, retries, validation, persistence."""
from __future__ import annotations

import json
import logging
import os
import stat
import sys
import types
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient

import invar_agent
import invar_agent.server as server
from invar_agent.config import (
    Tier,
    detect_llm_from_env,
    load_persisted_state,
    persist_config,
    resolve_tier_model,
)
from invar_agent.copilot.agent import CopilotAgent
from invar_agent.llm import clients as clients_mod
from invar_agent.llm import errors as errors_mod
from invar_agent.llm.clients import ChatLLMClient
from invar_agent.llm.errors import LLMProviderError, clear_last_llm_error, get_last_llm_error
from invar_agent.llm.validate import validate_provider
from invar_agent.server import create_app

SECRET = "AIzaSyD-secret-test-key-0123456789"
SK_SECRET = "sk-supersecretkey0123456789abcdef"

GOOGLE_400 = [{
    "error": {
        "code": 400,
        "message": "Model gemini-9-ultra is not found for API version v1beta.",
        "status": "INVALID_ARGUMENT",
    }
}]


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    for var in (
        "GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
        "INVAR_API_KEY", "INVAR_LLM_PROVIDER", "INVAR_LLM_TIER_FAST", "INVAR_LLM_TIER_REASONING",
        "GEMINI_MODEL_FAST", "GEMINI_MODEL_REASONING", "DEEPSEEK_MODEL_FAST",
        "DEEPSEEK_MODEL_REASONING", "INVAR_PERSIST_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("INVAR_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setattr(errors_mod, "sleep", lambda s: None)
    clients_mod._STRIPPED_OPTIONAL_PARAMS.clear()
    clear_last_llm_error()
    yield
    clear_last_llm_error()


def _chat_client(handler, provider="gemini", model="gemini-3.8-flash", api_key=SECRET):
    return ChatLLMClient(provider, model, api_key=api_key, transport=httpx.MockTransport(handler))


def _ok_body(text="hi"):
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _models_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture()
def app_client(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    server._STATE = None
    tc = TestClient(create_app(str(ws)))
    yield tc
    server._STATE = None


# (1) Google-style 400 body surfaces upstream_message in copilot reply + /api/status
def test_google_400_surfaces_in_copilot_reply_and_status(app_client):
    def handler(request):
        return httpx.Response(400, json=GOOGLE_400)

    state = server.get_state()
    state.set_api_key("gemini", SECRET)
    state.copilot = CopilotAgent(
        state.workspace_root, provider="gemini", api_key=SECRET,
        client_factory=lambda *a, **k: _chat_client(handler),
    )
    data = app_client.post("/api/copilot/chat", json={"message": "hello"}).json()
    assert data["llm_state"] == "provider_error"
    assert "gemini-9-ultra is not found" in data["reply"]
    assert "The LLM call to gemini/gemini-3.8-flash failed (HTTP 400)" in data["reply"]

    status = app_client.get("/api/status").json()
    last = status["llm"]["last_error"]
    assert last and "gemini-9-ultra is not found" in last["message"]
    assert last["status_code"] == 400 and last["at"]


# (2) not_configured vs provider_error
def test_not_configured_vs_provider_error(tmp_path):
    offline = CopilotAgent(tmp_path, provider="gemini", api_key="").chat("hello")
    assert offline["llm_state"] == "not_configured"
    assert "no llm provider is configured" in offline["reply"].lower()

    def failing(*a, **k):
        return _chat_client(lambda r: httpx.Response(401, json={"error": {"message": "API key not valid"}}))

    failed = CopilotAgent(tmp_path, provider="gemini", api_key=SECRET, client_factory=failing)
    result = failed.chat("Add a subtract function")
    assert result["llm_state"] == "provider_error"
    assert "No LLM provider is configured" not in result["reply"]
    assert "HTTP 401" in result["reply"] and "API key not valid" in result["reply"]
    assert "Test connection" in result["reply"]
    # Offline build-prompt draft is kept as a fallback.
    assert result["action"]["type"] == "SUGGEST_BUILD_PROMPT"


# (3) validate_provider: success, invalid key, auto-pick, explicit-override warning
def test_validate_provider_success_strips_models_prefix():
    def handler(request):
        assert request.url.path == "/v1beta/openai/models"
        return httpx.Response(200, json={"data": [
            {"id": "models/gemini-3.8-flash"}, {"id": "models/gemini-3.1-pro-preview"},
        ]})

    verdict = validate_provider("gemini", SECRET, client=_models_client(handler))
    assert verdict["valid"] is True
    assert verdict["models"] == ["gemini-3.8-flash", "gemini-3.1-pro-preview"]
    assert verdict["tier_models"] == {"fast": "gemini-3.8-flash", "reasoning": "gemini-3.1-pro-preview"}
    assert verdict["warnings"] == []


def test_validate_provider_invalid_key():
    def handler(request):
        return httpx.Response(400, json=[{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}}])

    verdict = validate_provider("gemini", SECRET, client=_models_client(handler))
    assert verdict["valid"] is False
    assert "API key not valid" in verdict["error"]
    assert "HTTP 400" in verdict["error"]


def test_validate_provider_auto_picks_missing_default():
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "models/gemini-3.6-flash"}, {"id": "models/gemini-pro-latest"}]})

    verdict = validate_provider("gemini", SECRET, client=_models_client(handler))
    assert verdict["tier_models"] == {"fast": "gemini-3.6-flash", "reasoning": "gemini-pro-latest"}
    assert verdict["auto_picked"] == {"fast": "gemini-3.6-flash", "reasoning": "gemini-pro-latest"}
    assert any("auto-selected 'gemini-3.6-flash'" in w for w in verdict["warnings"])


def test_validate_provider_explicit_override_missing_warns_with_suggestions():
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "models/gemini-3.8-flash"}, {"id": "models/gemini-3.1-pro-preview"}]})

    verdict = validate_provider("gemini", SECRET, tier_overrides={"fast": "gemini-nope"},
                                client=_models_client(handler))
    assert verdict["tier_models"]["fast"] == "gemini-nope"  # never silently replaced
    assert "fast" not in verdict["auto_picked"]
    warning = next(w for w in verdict["warnings"] if "gemini-nope" in w)
    assert "suggestions: gemini-3.8-flash" in warning


# (4) key-format mismatch warning (soft, not blocking)
def test_key_format_mismatch_warning():
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "gemini-3.8-flash"}]})

    verdict = validate_provider("gemini", SK_SECRET, client=_models_client(handler))
    assert verdict["valid"] is True
    assert any("does not match provider 'gemini'" in w and "AIza" in w for w in verdict["warnings"])

    ok = validate_provider("gemini", SECRET, client=_models_client(handler))
    assert not any("does not match" in w for w in ok["warnings"])


# (5) unsupported-parameter 400 -> retry once without optional params; other 4xx no retry
def test_unsupported_param_retries_once_without_sampling_params():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'temperature' is not supported with this model."}})
        return httpx.Response(200, json=_ok_body("stripped"))

    client = _chat_client(handler)
    assert client.generate([{"role": "user", "content": "x"}]) == "stripped"
    assert len(seen) == 2
    assert "temperature" in seen[0] and "top_p" in seen[0]
    assert "temperature" not in seen[1] and "top_p" not in seen[1]
    # Remembered for this provider+model: next call strips preemptively.
    client.generate([{"role": "user", "content": "y"}])
    assert len(seen) == 3 and "temperature" not in seen[2]


def test_other_4xx_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    with pytest.raises(LLMProviderError) as info:
        _chat_client(handler).generate([{"role": "user", "content": "x"}])
    assert len(calls) == 1
    assert info.value.status_code == 404 and info.value.retryable is False
    assert info.value.upstream_message == "model not found"


# (6) 429/5xx retried with backoff (sleep patched)
def test_429_and_5xx_retry_with_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr(errors_mod, "sleep", lambda s: sleeps.append(s))
    responses = iter([
        httpx.Response(429, json={"error": {"message": "rate limited"}}),
        httpx.Response(503, json={"error": {"message": "overloaded"}}),
        httpx.Response(200, json=_ok_body("finally")),
    ])
    assert _chat_client(lambda r: next(responses)).generate([{"role": "user", "content": "x"}]) == "finally"
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] * 1.2 - 0.25


def test_retry_gives_up_after_two_retries(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    with pytest.raises(LLMProviderError) as info:
        _chat_client(handler).generate([{"role": "user", "content": "x"}])
    assert len(calls) == 3
    assert info.value.retryable is True


def test_timeout_is_retryable():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=_ok_body("ok"))

    assert _chat_client(handler).generate([{"role": "user", "content": "x"}]) == "ok"
    assert len(calls) == 2


def test_deepseek_sdk_errors_map_to_provider_error(monkeypatch):
    openai = pytest.importorskip("openai")
    from invar_agent.llm.deepseek import DeepSeekChatClient

    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": "Model Not Exist"}})
    exc = openai.BadRequestError("Error code: 400", response=response, body={"error": {"message": "Model Not Exist"}})

    class _Completions:
        calls = 0

        def create(self, **kwargs):
            _Completions.calls += 1
            raise exc

    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))
    client = DeepSeekChatClient(tier=Tier.FAST, api_key=SK_SECRET, client=fake)
    with pytest.raises(LLMProviderError) as info:
        client.generate([{"role": "user", "content": "x"}])
    assert info.value.status_code == 400
    assert info.value.upstream_message == "Model Not Exist"
    assert _Completions.calls == 1  # 4xx: no retry


# (7) persistence round-trip, 0600 perms, env auto-detection
def test_persistence_round_trip_and_permissions(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "config.json"
    persist_config("deepseek", tier_overrides={"fast": "deepseek-flash", "reasoning": ""},
                   api_key=SK_SECRET, path=path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    raw = path.read_text()
    assert SK_SECRET not in raw  # key not persisted by default
    loaded = load_persisted_state(path)
    assert loaded["provider"] == "deepseek"
    assert loaded["tier_models"] == {"deepseek": {"fast": "deepseek-flash"}}
    assert loaded["api_key"] is None
    assert resolve_tier_model("deepseek", Tier.FAST, file_config=json.loads(raw)) == "deepseek-flash"

    monkeypatch.setenv("INVAR_PERSIST_KEY", "1")
    persist_config("deepseek", api_key=SK_SECRET, path=path)
    assert load_persisted_state(path)["api_key"] == SK_SECRET
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_env_auto_detection(monkeypatch):
    assert detect_llm_from_env() == {"provider": None, "api_key": None}
    monkeypatch.setenv("DEEPSEEK_API_KEY", SK_SECRET)
    assert detect_llm_from_env() == {"provider": "deepseek", "api_key": SK_SECRET}
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    assert detect_llm_from_env()["provider"] is None  # ambiguous without INVAR_LLM_PROVIDER
    monkeypatch.setenv("INVAR_LLM_PROVIDER", "gemini")
    assert detect_llm_from_env() == {"provider": "gemini", "api_key": SECRET}


def test_server_startup_loads_file_and_env(tmp_path, monkeypatch):
    persist_config("openrouter", path=os.environ["INVAR_CONFIG_PATH"])
    state = server.ServerState(tmp_path)
    assert state.provider == "openrouter" and state.api_key == ""

    monkeypatch.setenv("DEEPSEEK_API_KEY", SK_SECRET)
    state = server.ServerState(tmp_path)
    assert state.provider == "deepseek" and state.key_source == "env"
    assert state.api_key == SK_SECRET

    state = server.ServerState(tmp_path, provider_override="gemini")  # --provider flag
    assert state.provider == "gemini"


# (8) keys never appear in responses, logs, or exception strings
def test_keys_never_leak(app_client, monkeypatch, caplog):
    def echo_key(request):
        auth = request.headers.get("authorization", "")
        return httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {auth}"}})

    err = None
    with caplog.at_level(logging.DEBUG):
        try:
            _chat_client(echo_key, provider="openai", model="gpt-5.4-mini", api_key=SK_SECRET).generate(
                [{"role": "user", "content": "x"}])
        except LLMProviderError as exc:
            err = exc
        assert err is not None
        assert SK_SECRET not in str(err) and SK_SECRET not in json.dumps(err.to_dict())

        monkeypatch.setattr(server, "validate_provider",
                            lambda provider, api_key, tier_overrides=None, **kw: validate_provider(
                                provider, api_key, tier_overrides, client=_models_client(echo_key)))
        resp = app_client.post("/api/config", json={"provider": "openai", "api_key": SK_SECRET})
        assert resp.status_code == 400
        assert SK_SECRET not in resp.text
        assert "Incorrect API key provided" in resp.json()["detail"]["error"]

        state = server.get_state()
        state.set_api_key("openai", SK_SECRET)
        state.copilot = CopilotAgent(state.workspace_root, provider="openai", api_key=SK_SECRET,
                                     client_factory=lambda *a, **k: _chat_client(
                                         echo_key, provider="openai", model="gpt-5.4-mini", api_key=SK_SECRET))
        chat = app_client.post("/api/copilot/chat", json={"message": "hi"})
        assert SK_SECRET not in chat.text
        assert SK_SECRET not in app_client.get("/api/status").text
    assert SK_SECRET not in caplog.text
    assert SK_SECRET not in json.dumps(get_last_llm_error())


# (9) UI does not send tier_overrides; env tier overrides take effect
def test_ui_does_not_send_tier_overrides():
    html = (Path(invar_agent.__file__).parent / "static" / "index.html").read_text()
    assert "payload.tier_overrides" not in html
    assert "tier_overrides: {" not in html
    assert "JSON.stringify({provider: provider, api_key: key})" in html
    assert "/api/llm/test" in html


def test_env_tier_override_takes_effect_after_ui_save(app_client, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL_FAST", "deepseek-env-fast")

    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]})

    monkeypatch.setattr(server, "validate_provider",
                        lambda provider, api_key, tier_overrides=None, **kw: validate_provider(
                            provider, api_key, tier_overrides, client=_models_client(handler)))
    resp = app_client.post("/api/config", json={"provider": "deepseek", "api_key": SK_SECRET})
    assert resp.status_code == 200
    data = resp.json()
    assert data["tier_models"]["fast"] == "deepseek-env-fast"
    assert data["tier_models"]["reasoning"] == "deepseek-v4-pro"
    assert any("deepseek-env-fast" in w and "from env" in w for w in data["warnings"])
    status = app_client.get("/api/status").json()["llm"]
    assert status["tier_models"]["fast"] == "deepseek-env-fast"
    assert status["validated"] is True


def test_llm_test_endpoint_reports_structured_error(app_client, monkeypatch):
    assert app_client.post("/api/llm/test").status_code == 400  # not configured
    state = server.get_state()
    state.set_api_key("gemini", SECRET)

    def fake_get(provider, tier=None, api_key=None, override=None, **kw):
        return _chat_client(lambda r: httpx.Response(400, json=GOOGLE_400))

    monkeypatch.setattr("invar_agent.llm.clients.get_llm_client", fake_get)
    data = app_client.post("/api/llm/test").json()
    assert data["ok"] is False
    assert data["error"]["status_code"] == 400
    assert "gemini-9-ultra is not found" in data["error"]["upstream_message"]

    monkeypatch.setattr("invar_agent.llm.clients.get_llm_client",
                        lambda *a, **k: _chat_client(lambda r: httpx.Response(200, json=_ok_body("OK"))))
    ok = app_client.post("/api/llm/test").json()
    assert ok["ok"] is True and ok["model"] == "gemini-3.8-flash" and ok["latency_ms"] >= 0


# (10) _native_status false positive (namespace-package crate dir)
def test_native_status_requires_compiled_symbols(monkeypatch):
    fake_namespace = types.ModuleType("invar_agent._native")
    monkeypatch.setitem(sys.modules, "invar_agent._native", fake_namespace)
    monkeypatch.setattr(invar_agent, "_native", fake_namespace, raising=False)
    status = server._native_status()
    assert status["available"] is False
    assert status["modules"] == {name: False for name in
                                 ("PyGoiSolver", "PyHinEngine", "PyDeterministicHealer", "PyHisRegister")}

    full = types.ModuleType("invar_agent._native")
    for name in ("PyGoiSolver", "PyHinEngine", "PyDeterministicHealer", "PyHisRegister"):
        setattr(full, name, object)
    monkeypatch.setitem(sys.modules, "invar_agent._native", full)
    monkeypatch.setattr(invar_agent, "_native", full, raising=False)
    assert server._native_status()["available"] is True

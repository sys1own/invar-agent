"""Co-pilot port tests: ActionParser sanitisation, tier routing, and endpoints."""
from __future__ import annotations

import json
import sys
import textwrap

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import invar_agent.server as server
from invar_agent.config import Tier, resolve_tier_model
from invar_agent.core.agent import VerifiedAutonomousAgent
from invar_agent.copilot.action_parser import ActionParser, sanitize_builder_prompt
from invar_agent.copilot.agent import CopilotAgent, format_copilot_response
from invar_agent.healing.error_classifier import ClassifiedErrorPayload, FailureCategory
from invar_agent.healing.llm_healer import SurgicalLLMHealer
from invar_agent.server import create_app


# ---------------------------------------------------------------------------
# Unit: ActionParser sanitisation
# ---------------------------------------------------------------------------

def test_action_parser_sanitizes_build_prompt_fence():
    response = textwrap.dedent(
        """\
        Here is a plan for your workspace.

        ```build_prompt
        Here's a prompt: Add a subtract(a, b) function to calc.py that returns a - b,
        and extend test_calc.py with test_subtract asserting subtract(5, 3) == 2.
        Target: pure_python
        Acceleration: Standard Runtime (Bypass Bridge)
        ```
        """
    )
    parsed = ActionParser().parse(response)
    action = parsed["action"]
    assert action is not None
    prompt = sanitize_builder_prompt(action["clean_prompt"])
    assert prompt
    assert not prompt.lower().startswith("here's a prompt")
    assert "Here's a prompt" not in prompt
    assert "Target:" not in prompt and "Acceleration:" not in prompt
    assert "subtract" in prompt


def test_sanitize_builder_prompt_strips_meta_preamble():
    raw = "I've crafted a detailed build prompt:\nAdd mul(a, b) to calc.py."
    cleaned = sanitize_builder_prompt(raw)
    assert cleaned
    assert "crafted" not in cleaned.lower()
    assert cleaned.startswith("Add mul")


def test_format_copilot_response_returns_legacy_action():
    raw = json.dumps({
        "display_text": "Add a subtract helper to calc.py.",
        "action": {
            "type": "build",
            "clean_prompt": "Add subtract(a, b) to calc.py returning a - b.",
            "parameters": {"target": "pure_python"},
        },
    })
    reply, action = format_copilot_response(raw)
    assert "subtract" in reply.lower() or "Subtract" in reply
    assert action is not None
    assert action["params"]["prompt"].startswith("Add subtract")


# ---------------------------------------------------------------------------
# Unit: tier routing
# ---------------------------------------------------------------------------

def test_resolve_tier_model_defaults_and_precedence(monkeypatch):
    assert resolve_tier_model("gemini", Tier.FAST, file_config={}) == "gemini-3.8-flash"
    assert resolve_tier_model("gemini", Tier.REASONING, file_config={}) == "gemini-3.1-pro-preview"
    assert resolve_tier_model("openai", Tier.FAST, file_config={}) == "gpt-5.4-mini"
    assert resolve_tier_model("deepseek", Tier.FAST, file_config={}) == "deepseek-flash"
    assert resolve_tier_model("deepseek", Tier.REASONING, file_config={}) == "deepseek-v4-pro"

    # Explicit override wins over everything.
    assert resolve_tier_model("gemini", Tier.FAST, override="custom-model") == "custom-model"

    # Env var beats file config and defaults.
    monkeypatch.setenv("INVAR_LLM_TIER_FAST", json.dumps({"gemini": "env-flash"}))
    assert resolve_tier_model("gemini", Tier.FAST, file_config={}) == "env-flash"
    # ... but an explicit override still wins.
    assert resolve_tier_model("gemini", Tier.FAST, override="custom-model") == "custom-model"
    monkeypatch.delenv("INVAR_LLM_TIER_FAST")

    # File config beats defaults when no env/override is present.
    cfg = {"tier_models": {"gemini": {"fast": "file-flash"}}}
    assert resolve_tier_model("gemini", Tier.FAST, file_config=cfg) == "file-flash"


class _FakeChatClient:
    def __init__(self, reply: str):
        self.reply = reply
        self.messages_seen = []

    def generate(self, messages, temperature=0.2, max_tokens=None, **_):
        self.max_tokens_seen = max_tokens
        self.messages_seen.append(messages)
        return self.reply


def _llm_json_reply(prompt: str) -> str:
    return json.dumps({
        "display_text": "I can add that helper.",
        "action": {
            "type": "build",
            "clean_prompt": prompt,
            "parameters": {"target": "pure_python"},
        },
    })


def test_copilot_agent_uses_fast_tier(monkeypatch, tmp_path):
    calls = []

    def fake_factory(provider, tier=None, api_key=None, model=None):
        calls.append({"provider": provider, "tier": tier, "api_key": api_key, "model": model})
        return _FakeChatClient(_llm_json_reply("Add subtract(a, b) to calc.py returning a - b."))

    agent = CopilotAgent(tmp_path, provider="gemini", api_key="k", client_factory=fake_factory)
    result = agent.chat("Add a subtract function")

    assert calls, "copilot must resolve an LLM client"
    assert calls[0]["tier"] == Tier.FAST
    assert result["action"] is not None
    assert result["action"]["type"] == "SUGGEST_BUILD_PROMPT"
    assert result["action"]["params"]["prompt"].startswith("Add subtract")


def test_builder_defaults_to_reasoning_tier(tmp_path):
    agent = VerifiedAutonomousAgent(tmp_path)
    assert agent.tier == Tier.REASONING


def test_builder_healer_wiring_assigns_tiers(tmp_path):
    agent = VerifiedAutonomousAgent(tmp_path, llm_provider="gemini", llm_api_key="k")
    healer = agent._healer_for()
    # Tier-1 surgical repair -> FAST; Tier-2 architectural repair -> REASONING.
    assert healer.flash_healer.tier == Tier.FAST
    assert healer.pro_healer.tier == Tier.REASONING


def test_surgical_llm_healer_resolves_client_with_tier(monkeypatch):
    calls = []

    def fake_get_llm_client(provider, tier=None, api_key=None, model=None, **kwargs):
        calls.append({"provider": provider, "tier": tier})
        return _FakeChatClient(
            json.dumps({"ast_patches": [], "replacements": {"calc.py": "x = 1\n"}})
        )

    monkeypatch.setattr("invar_agent.llm.clients.get_llm_client", fake_get_llm_client)
    healer = SurgicalLLMHealer(tier=Tier.REASONING, provider="gemini", api_key="k")
    classified = ClassifiedErrorPayload(
        category=FailureCategory.SYNTAX_ERROR,
        error_code="SyntaxError",
        diagnostic_msg="SyntaxError: invalid syntax",
        failing_file="calc.py",
        failing_line=1,
    )
    healer.generate_ast_repair_patch(classified, {"calc.py": "def broken(:\n"})
    assert calls and calls[0]["tier"] == Tier.REASONING


# ---------------------------------------------------------------------------
# Integration: copilot endpoints + SSE + regression
# ---------------------------------------------------------------------------

def _sample_workspace(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    (tmp_path / "conftest.py").write_text("")
    return tmp_path


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Keep endpoint tests hermetic: never fall back to real provider keys.
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
                "OPENROUTER_API_KEY", "INVAR_API_KEY", "INVAR_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("INVAR_CONFIG_PATH", str(tmp_path / "invar-config.json"))
    ws = tmp_path / "ws"
    ws.mkdir()
    _sample_workspace(ws)
    monkeypatch.setattr(server, "_STATE", None)
    app = create_app(str(ws))
    return TestClient(app), ws


def test_copilot_chat_offline_fallback_suggests_build(client):
    tc, _ = client
    resp = tc.post("/api/copilot/chat", json={"message": "Add a subtract function", "history": []})
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    action = data.get("action")
    assert action is not None
    assert action["type"] == "SUGGEST_BUILD_PROMPT"
    assert "subtract" in action["params"]["prompt"].lower()


def test_copilot_chat_non_build_message_has_null_action(client):
    tc, _ = client
    resp = tc.post("/api/copilot/chat", json={"message": "hello there", "history": []})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("action") is None
    assert data.get("reply")


def test_copilot_run_build_streams_sse(client):
    tc, _ = client
    resp = tc.post("/api/copilot/run-build", json={"prompt": "inspect workspace"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "data: " in body
    assert '"type": "stage"' in body or '"type":"stage"' in body
    assert '"type": "done"' in body or '"type":"done"' in body


def test_copilot_run_build_requires_prompt(client):
    tc, _ = client
    resp = tc.post("/api/copilot/run-build", json={"prompt": "  "})
    assert resp.status_code == 400


def test_agent_stream_regression_nine_stages(client):
    """Regression: the builder's 9-stage pipeline still runs via /api/agent/stream."""
    tc, ws = client
    tc.post("/api/workspace/ingest", json={"path": str(ws)})
    node = {
        "node_id": "wf1", "wavefront_generation": 0, "assigned_tier": "TIER_1_FLASH",
        "description": "inspect", "read_set_paths": ["calc.py"], "write_set_paths": ["calc.py"],
        "target_ast_symbols": ["calc.add"],
        "output_ports": [{"port_name": "out", "symbol_contract": "calc.add"}],
        "verification_contract": {},
    }
    resp = tc.get("/api/agent/stream", params={
        "objective": "regression",
        "nodes": json.dumps([node]),
        "sandbox_cmd": json.dumps([sys.executable, "-m", "pytest", "-q"]),
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    for stage in ("Overlay_Mutate", "SMT_Concolic_Verify", "Sandbox_Execute",
                  "Reflex_Heal", "Adjoint_Commit_or_Rollback"):
        assert stage in body, f"missing stage event: {stage}"
    assert '"type": "done"' in body or '"type":"done"' in body

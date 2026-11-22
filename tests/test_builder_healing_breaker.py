from __future__ import annotations

import ast
import inspect
import json
import os
import shutil
import time
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from invar_agent.builder.circuit_breaker import BuildBreakerBudget, CircuitBreakerController, ProgressMetric, SlotBreakerState, compute_fault_fingerprint
from invar_agent.builder.error_taxonomy import BuildFault, CircuitAction, ErrorTier
from invar_agent.builder.fallback_registry import FallbackEmitterRegistry
from invar_agent.builder.healing_controller import HealingController
from invar_agent.builder.his_cache import digest
from invar_agent.builder.ingestion.pipeline import seal_ir
from invar_agent.builder.leaf_policy import LEAF_SYSTEM_PROMPT, validate_source_capabilities
from invar_agent.builder.state_machine import BuildRequest, BuilderStateMachine
from invar_agent.builder.uast_models import FALLBACK_EMITTERS, LLMExtensionSlotRequest
from invar_agent.llm.clients import ChatLLMClient, LLMResponse
from invar_agent.llm.deepseek import DeepSeekChatClient
from invar_agent.llm.errors import LLMProviderError, complete_for_builder
from test_builder_state_machine import NoLLM, make_graph, make_ir, offline, real_emitters


def port_for(graph):
    return next(node for node in graph.nodes if node.node_id == "port_core")


@pytest.mark.parametrize("source,capability,line", [
    ("import os", "import os", 1),
    ("from socket import socket", "import socket", 1),
    ("import _ctypes", "import _ctypes", 1),
    ("x=1\nx.__class__", "attribute __class__", 2),
    ("eval('1')", "eval", 1),
    ("f(shell=True)", "shell=True", 1),
    ("__import__('math')", "__import__", 1),
])
def test_leaf_capability_diagnostic_identifies_construct_and_line(source, capability, line):
    with pytest.raises(BuildFault) as caught:
        validate_source_capabilities(ast.parse(source), leaf=True, relpath="leaf.py")
    assert caught.value.code == "SANDBOX_SECURITY_VIOLATION"
    assert caught.value.details == dict(capability=capability, line=line, column=source.splitlines()[line-1].index("shell") if capability=="shell=True" else 0, relpath="leaf.py")
    assert f"leaf.py:{line}:" in str(caught.value)


def test_leaf_pure_stdlib_and_empty_module_are_allowed():
    for source in ("", "from __future__ import annotations", "import math,json,csv,io,html,statistics\nx=math.sqrt(4)\ns=io.StringIO('x')"):
        validate_source_capabilities(ast.parse(source), leaf=True, relpath="leaf.py")


@pytest.mark.parametrize("fence", ["", "python", "py", "Python"])
def test_leaf_response_contract_fences_and_private_redacted_capture(tmp_path, fence):
    ir=make_ir();graph=make_graph(ir);port=port_for(graph)
    node=next(node for node in graph.nodes if node.kind=="llm_extension_slot")
    source="def simulation_step(x,k):\n    return {'x': x}\n"
    class Client:
        api_key="private-test-provider-key"
        def complete(self, prompt, **kwargs):
            assert kwargs['system']==LEAF_SYSTEM_PROMPT
            assert "io.StringIO" in kwargs['system'] and "shell=True" in kwargs['system']
            assert json.loads(prompt)['entry_point_signature']=='def simulation_step(x, k):'
            assert json.loads(prompt)['required_output_keys']==['x']
            return LLMResponse(content=f"```{fence}\n{source}```\n", input_tokens=2, output_tokens=3)
    result=HealingController(Client(),diagnostics_dir=tmp_path).tier2("",node,port,graph,"",BuildBreakerBudget())
    assert result.source==source and result.tokens_used==5
    capture=next(tmp_path.glob('*.json'))
    if os.name != "nt":
        assert capture.stat().st_mode & 0o777 == 0o600
    assert json.loads(capture.read_text())['raw_response'].startswith('```')
    client=Client()
    client.complete=lambda *args,**kwargs: LLMResponse(content=client.api_key, input_tokens=0, output_tokens=0)
    HealingController(client,diagnostics_dir=tmp_path).tier2("",node,port,graph,"",BuildBreakerBudget())
    assert all(client.api_key not in path.read_text() for path in tmp_path.glob('*.json'))


def test_leaf_truncated_fence_is_not_silently_accepted():
    ir=make_ir();graph=make_graph(ir)
    class Client:
        def complete(self,*args,**kwargs):
            return LLMResponse(content="```python\ndef simulation_step(x,k):\n    return {'x':x}",input_tokens=0,output_tokens=0)
    node=next(node for node in graph.nodes if node.kind=='llm_extension_slot')
    result=HealingController(Client()).tier2('',node,port_for(graph),graph,'',BuildBreakerBudget())
    assert result.source.startswith('```python')


class StubbornClient:
    calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        return LLMResponse(content="def simulation_step(x,k):\n    raise ValueError('stubborn leaf')\n", input_tokens=5, output_tokens=8)


def decide(state, budget, *, message="failure", patch="A", count=2, tier=ErrorTier.TIER2_BOUNDED_LLM, kind="llm_extension_slot", code="SLOT_PYTEST_ASSERTION"):
    fingerprint = compute_fault_fingerprint(state.module_id, "Sandbox_Execute", code, "AssertionError", "leaf.py", message)
    metric = ProgressMetric(has_critical=int(tier == ErrorTier.CRITICAL_NON_RETRYABLE), hard_error_count=0, hin_violation_count=0,
                            z3_violation_count=0, failed_test_count=count, warning_count=0)
    return CircuitBreakerController().decide(module_id=state.module_id, node_kind=kind, round_index=1, tier=tier,
        fingerprint=fingerprint, metric=metric, candidate_ast_digest=digest("invar.his.body.v2", patch), slot_state=state,
        budget=budget, fallback_emitter_id="fb_svg_trajectory_viz")


def test_tier0_heals_syntax_and_missing_import_zero_tokens():
    controller = HealingController(NoLLM())
    source = "class Plot:\n    pass\n\ndef simulation_step(x,k):\n    if x:\n    return {'x': math.sqrt(x)}\n"
    start = time.perf_counter()
    result = controller.tier0(source)
    elapsed = time.perf_counter() - start
    assert elapsed < .005, elapsed
    assert result.tokens_used == 0 and "class Plot" in result.source
    namespace = {}
    exec(compile(result.source, "leaf.py", "exec"), namespace)
    assert namespace["simulation_step"](4, .1) == {"x": 2.0}
    routed = "class App:\n    def get(self, path):\n        return lambda fn: fn\napp=App()\n@app.get('/x')\ndef route(x):\n    return x / 0\n"
    sanitized = controller.tier0(routed).source
    assert any(isinstance(node, ast.ClassDef) for node in ast.parse(sanitized).body)
    assert "@app.get" in sanitized


def test_tier1_heals_his_signature_drift_zero_tokens():
    ir = make_ir()
    port = port_for(make_graph(ir))
    client = NoLLM()
    result = HealingController(client).tier1("def simulation_step(k,x):\n    return {'x': x}\n", port)
    namespace = {}
    exec(compile(result.source, "leaf.py", "exec"), namespace)
    assert list(inspect.signature(namespace["simulation_step"]).parameters) == ["x", "k"]
    assert namespace["simulation_step"](2, .1) == {"x": 2}
    assert result.frozen_his_digest == port.his_digest and result.tokens_used == 0 and client.calls == 0


@pytest.mark.asyncio
async def test_duplicate_fault_fingerprint_triggers_fallback(tmp_path):
    ir = make_ir()
    client = StubbornClient()
    state = await BuilderStateMachine(tmp_path, emitters=real_emitters(slot=None), llm_client=client).run(
        BuildRequest(build_id="duplicate", ir=ir, uast=make_graph(ir)))
    assert state.status == "COMPLETED_WITH_FALLBACKS", state.error
    assert client.calls == 2 and len(state.fallbacks) == 1
    assert state.fallbacks[0]["trip_reason"].startswith("DUPLICATE_FAULT_FINGERPRINT")
    assert state.fallbacks[0]["fallback_verified"]
    assert not list((Path(state.workspace).parent / "overlays").rglob("slot_custom_viz"))
    assert "def render_svg" in next(Path(state.workspace).rglob("slot_custom_viz.py")).read_text()
    assert state.total_llm_tokens_used == 26


def test_oscillating_ast_patch_triggers_fallback():
    state, budget = SlotBreakerState("slot"), BuildBreakerBudget()
    assert decide(state, budget, message="first", patch="A", count=3).action == CircuitAction.INVOKE_TIER2_LLM
    assert decide(state, budget, message="second", patch="B", count=2).action == CircuitAction.INVOKE_TIER2_LLM
    decision = decide(state, budget, message="third", patch="A", count=1)
    assert decision.reason_code.startswith("OSCILLATING_PATCH_DIGEST") and decision.action == CircuitAction.SUBSTITUTE_DETERMINISTIC_FALLBACK


def test_non_monotonic_progress_triggers_fallback():
    state, budget = SlotBreakerState("slot"), BuildBreakerBudget()
    decide(state, budget, message="first", patch="A", count=2)
    decision = decide(state, budget, message="second", patch="B", count=2)
    assert decision.reason_code.startswith("NON_MONOTONIC_PROGRESS")
    assert decision.action == CircuitAction.SUBSTITUTE_DETERMINISTIC_FALLBACK


@pytest.mark.asyncio
async def test_critical_z3_unsat_aborts_zero_llm_retries(tmp_path):
    ir = make_ir()
    ir.constants[0].bounds.lower = .5
    seal_ir(ir)
    client = NoLLM()
    state = await BuilderStateMachine(tmp_path, llm_client=client).run(BuildRequest(build_id="unsat", ir=ir))
    assert state.status == "ABORTED_CRITICAL" and state.error_code == "Z3_SPEC_UNSAT"
    assert not state.committed_waves and client.calls == 0
    assert "const:" in str(state.events)


@pytest.mark.asyncio
async def test_missing_gxx_switches_to_python_fallback(tmp_path, monkeypatch):
    ir = make_ir()
    original = shutil.which
    monkeypatch.setattr(shutil, "which", lambda command: None if command in {"g++", "clang++"} else original(command))
    client = NoLLM()
    state = await BuilderStateMachine(tmp_path, emitters=real_emitters(), llm_client=client).run(BuildRequest(build_id="python", ir=ir, uast=make_graph(ir)))
    assert state.status in {"COMPLETED", "COMPLETED_WITH_FALLBACKS"}, state.error
    assert state.python_fallback_active and not state.cabi_compiled and client.calls == 0


@pytest.mark.parametrize("status", [401, 429])
@pytest.mark.asyncio
async def test_llm_401_and_429_quota_substitutes_fallback_zero_retries(tmp_path, status):
    calls = []
    def provider(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "credit_balance_exhausted"}})
    client = ChatLLMClient("openai", "test-model", api_key="test-key", transport=httpx.MockTransport(provider))
    ir = make_ir()
    state = await BuilderStateMachine(tmp_path, emitters=real_emitters(slot=None), llm_client=client).run(
        BuildRequest(build_id=f"http_{status}", ir=ir, uast=make_graph(ir)))
    assert state.status == "COMPLETED_WITH_FALLBACKS", state.error
    assert len(calls) == 1 and state.fallbacks[0]["fallback_verified"]
    assert state.fallbacks[0]["trip_reason"] == "CRITICAL_SLOT_TRIP:" + ("LLM_AUTH_ERROR" if status == 401 else "LLM_QUOTA_EXHAUSTED")


@pytest.mark.asyncio
async def test_deterministic_module_never_calls_tier2(tmp_path):
    ir = make_ir()
    client = NoLLM()
    state = await BuilderStateMachine(tmp_path, emitters=real_emitters(sdk="raise RuntimeError('emitter bug')\n"), llm_client=client).run(
        BuildRequest(build_id="platform", ir=ir, uast=make_graph(ir)))
    assert state.status == "ABORTED_CRITICAL" and client.calls == 0
    assert "TIER2_FORBIDDEN_FOR_DETERMINISTIC_MODULE" in state.error
    sdk = next(node for node in make_graph(ir).nodes if node.node_id == "mod_sdk")
    with pytest.raises(BuildFault, match="restricted"):
        HealingController(client).tier2("", sdk, port_for(make_graph(ir)), make_graph(ir), "", BuildBreakerBudget())


@pytest.mark.parametrize("field,reason", [("used_global_tier2_repairs", "GLOBAL_TIER2_CAP_EXHAUSTED"), ("used_repair_tokens", "REPAIR_TOKEN_BUDGET_EXHAUSTED"), ("elapsed_sec", "WALL_CLOCK_BUDGET_EXHAUSTED")])
def test_global_repair_budgets(field, reason):
    budget = BuildBreakerBudget()
    setattr(budget, field, {"used_global_tier2_repairs": 6, "used_repair_tokens": 24000, "elapsed_sec": 300}[field])
    assert decide(SlotBreakerState("slot"), budget).reason_code == reason


def test_registered_fallbacks_and_plan_kind_aliases():
    ir = make_ir()
    port = port_for(make_graph(ir))
    registry = FallbackEmitterRegistry()
    for kind, emitter_id in FALLBACK_EMITTERS.items():
        request = LLMExtensionSlotRequest(slot_id=kind, intent_summary="Use the frozen contract")
        assert request.slot_kind == kind and request.fallback_emitter_id == emitter_id and request.slot_id.startswith("slot_")
        source = registry.emit(emitter_id, port, ir)
        if kind == "narrative_documentation":
            assert "0.1" in source and "decay.tex" in source
        else:
            namespace = {}
            exec(compile(source, "fallback.py", "exec"), namespace)
            assert namespace["simulation_step"](1, .1) == {"x": 1}
            assert namespace["render_svg"]({"x": [0, 1]}).startswith("<svg")
            assert namespace["adapt_rows"]([{"v": "150"}], si_scales={"v": .01}) == [{"v": 1.5}]
            with pytest.raises(ValueError):
                namespace["adapt_rows"]([{"v": "nan"}])
    with pytest.raises(ValueError, match="Unregistered"):
        registry.emit("invented", port, ir)


def test_builder_retry_context_does_not_change_research_policy(monkeypatch):
    monkeypatch.setattr("invar_agent.llm.errors.sleep", lambda delay: None)
    calls = []
    def provider(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"message": "RPM limit"}})
    client = ChatLLMClient("openai", "test-model", api_key="test-key", transport=httpx.MockTransport(provider))
    with pytest.raises(LLMProviderError):
        complete_for_builder(client, "test")
    assert len(calls) == 1
    with pytest.raises(LLMProviderError):
        client.complete("research keeps existing transport backoffs")
    assert len(calls) == 4


def test_builder_deepseek_budget_is_used_for_output_not_default_thinking():
    calls = []
    def provider(request):
        body = json.loads(request.content)
        calls.append(body)
        exhausted = body['messages'][-1]['content']=='prose' and body.get('thinking') != {'type':'disabled'}
        return httpx.Response(200,json={'id':'test','object':'chat.completion','created':1,'model':'deepseek-flash',
            'choices':[{'index':0,'message':{'role':'assistant','content':'' if exhausted else '{"proposals":[]}'},
                        'finish_reason':'length' if exhausted else 'stop'}],
            'usage':{'prompt_tokens':1,'completion_tokens':2048 if exhausted else 1,'total_tokens':2049 if exhausted else 2}})
    with httpx.Client(transport=httpx.MockTransport(provider)) as transport:
        sdk = OpenAI(api_key='test-key',base_url='https://example.invalid',http_client=transport)
        client = DeepSeekChatClient(api_key='test-key',client=sdk)
        client.complete('research')
        result = complete_for_builder(client,'prose',max_tokens=2048)
        complete_for_builder(client,'explicit reasoning',thinking=True)
    assert result.content == '{"proposals":[]}'
    assert 'thinking' not in calls[0] and 'thinking' not in calls[2]
    assert calls[1]['thinking'] == {'type':'disabled'} and len(calls)==3


def test_deepseek_sdk_cannot_retry_builder_quota_errors():
    calls = []
    def provider(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"message": "credit_balance_exhausted"}})
    with httpx.Client(transport=httpx.MockTransport(provider)) as transport:
        sdk = OpenAI(api_key="test-key", base_url="https://example.invalid", http_client=transport)
        client = DeepSeekChatClient(api_key="test-key", client=sdk)
        with pytest.raises(LLMProviderError):
            complete_for_builder(client, "test")
    assert len(calls) == 1


def test_tier1_coercion_clamp_and_invalid_dimension_gates():
    port = port_for(make_graph(make_ir()))
    healer = HealingController(NoLLM())
    code = "def simulation_step(k,x):\n    return {'x': x}\n"
    coerced = healer.tier1(code, port, error_code="HIN_SI_DIM_MISMATCH", details={"source_dimension": [0]*7, "target_dimension": [0]*7, "unit_scale": .01})
    namespace = {}
    exec(compile(coerced.source, "adapter.py", "exec"), namespace)
    assert namespace["simulation_step"](100, .1) == {"x": 1}
    clamped = healer.tier1("def simulation_step(x,k):\n    return {'x': -1}\n", port, error_code="BOUND_CLAMP_VIOLATION")
    exec(compile(clamped.source, "clamp.py", "exec"), namespace)
    assert namespace["simulation_step"](1, .1) == {"x": 0}
    with pytest.raises(BuildFault, match="Incommensurable"):
        healer.tier1(code, port, error_code="HIN_SI_DIM_MISMATCH", details={"source_dimension": [0]*7, "target_dimension": [1,0,0,0,0,0,0]})

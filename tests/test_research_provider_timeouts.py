import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from invar_agent.config import Tier
from invar_agent.llm import deepseek, errors
from invar_agent.llm.deepseek import DeepSeekChatClient
from invar_agent.llm.errors import LLMProviderError
from invar_agent.research import state_machine as sm
from invar_agent.research.epistemic import parse_run_certificate
from invar_agent.research.state_machine import ResearchStateMachine
from test_default_verification import model


class StreamingClient:
    def __init__(self, clock, mode):
        self.clock,self.mode = clock,mode
        self.chat = SimpleNamespace(completions=self)
        self.options = []
        self.streams = []

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        assert kwargs["stream"]
        owner = self

        class Stream:
            closed = False
            count = 0

            def __iter__(self): return self

            def __next__(self):
                if owner.mode == "empty": raise StopIteration
                if owner.mode == "transport_stall": raise httpx.ReadTimeout("No chunk received")
                if self.count == 3: raise StopIteration
                owner.clock[0] += 181 if owner.mode == "late_chunk" else 100
                self.count += 1
                delta = SimpleNamespace(content="done" if self.count == 3 else None,
                    reasoning_content="thinking" if self.count < 3 else None)
                return SimpleNamespace(choices=[SimpleNamespace(delta=delta,finish_reason="stop" if self.count == 3 else None)])

            def close(self): self.closed = True

        stream = Stream()
        self.streams.append(stream)
        return stream


@pytest.fixture
def streaming(monkeypatch):
    clock = [0.0]
    delays = []
    monkeypatch.setenv("INVAR_DEEPSEEK_IDLE_TIMEOUT","180")
    monkeypatch.setattr(deepseek,"time",SimpleNamespace(monotonic=lambda:clock[0],perf_counter=time.perf_counter))
    monkeypatch.setattr(errors,"sleep",delays.append)

    def make(mode):
        sdk = StreamingClient(clock,mode)
        client = DeepSeekChatClient(tier=Tier.REASONING,api_key="test",client=sdk)
        client._injected_client = False
        return client,sdk,delays
    return make


def test_reasoning_stream_can_progress_beyond_old_wall_deadline(streaming):
    client,sdk,delays = streaming("progress")
    assert client.generate([dict(role="user",content="study")],max_tokens=65536) == "done"
    assert len(sdk.streams) == 1 and sdk.streams[0].closed
    assert sdk.options[0]["timeout"].read == 180
    assert sdk.options[0]["max_retries"] == 0
    assert not delays


def test_empty_stream_closes_without_hanging(streaming):
    client,sdk,delays = streaming("empty")
    assert client.generate([dict(role="user",content="study")]) == ""
    assert len(sdk.streams) == 1 and sdk.streams[0].closed and not delays


@pytest.mark.parametrize("mode",["transport_stall","late_chunk"])
def test_stalled_stream_is_retried_and_closed(streaming,mode):
    client,sdk,delays = streaming(mode)
    with pytest.raises(LLMProviderError) as failure:
        client.generate([dict(role="user",content="study")],max_tokens=65536)
    assert failure.value.retryable
    assert len(sdk.streams) == 3 and all(stream.closed for stream in sdk.streams)
    assert len(delays) == 2 and delays[1] > delays[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("planning",[False,True])
async def test_stalled_provider_persists_terminal_inconclusive(streaming,monkeypatch,tmp_path,planning):
    client,sdk,delays = streaming("transport_stall")
    phase_delays = []

    async def sleep(delay): phase_delays.append(delay)

    monkeypatch.setattr(sm.asyncio,"sleep",sleep)

    async def complete(prompt):
        return await asyncio.to_thread(client.generate,[dict(role="user",content=prompt)],max_tokens=65536)

    machine = ResearchStateMachine(tmp_path/"research",workspace_root=tmp_path,llm_planner=complete)

    async def stages(state):
        # Exercise the same callback boundary used by refinement, including a
        # best-effort helper that swallows an exhausted provider error.
        try: await machine.llm_planner("refine")
        except LLMProviderError: pass
        await machine._emit("accelerator_log","next_phase",{})
        return state

    monkeypatch.setattr(machine,"_run_stages",stages)
    state = await machine.plan_and_run("Study x >= 2 with finite domains") if planning else await machine.run(model())
    assert state.status == state.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED"
    assert state.meaningfulness_certificate.failure_reason == "LLM_PROVIDER_RETRIES_EXHAUSTED"
    run_id = state.run_id if planning else state.blueprint.blueprint_id
    path = tmp_path/".invar"/"research_vault"/run_id/"meaningfulness_certificate.json"
    certificate = parse_run_certificate(path.read_text())
    assert certificate == state.meaningfulness_certificate
    tampered = json.loads(path.read_text())
    tampered["failure_reason"] = "changed"
    with pytest.raises(ValueError,match="receipt mismatch"): parse_run_certificate(tampered)
    assert len(sdk.streams) == 9 and all(stream.closed for stream in sdk.streams)
    assert len(phase_delays) == 2 and len(delays) == 6


@pytest.mark.asyncio
async def test_transient_research_provider_error_recovers_and_nonretryable_stops(tmp_path,monkeypatch):
    delays = []

    async def sleep(delay): delays.append(delay)

    monkeypatch.setattr(sm.asyncio,"sleep",sleep)
    calls = []

    async def transient(prompt):
        calls.append(prompt)
        if len(calls) < 3: raise LLMProviderError("stub",retryable=True)
        return "recovered"

    machine = ResearchStateMachine(tmp_path,llm_planner=transient,llm_fast=transient)
    assert machine.llm_planner is machine.llm_fast
    assert await machine.llm_fast("work") == "recovered"
    assert len(calls) == 3 and len(delays) == 2 and machine._provider_failure is None

    async def forbidden(prompt): raise LLMProviderError("stub",status_code=401,retryable=False)

    failed = ResearchStateMachine(tmp_path/"other",workspace_root=tmp_path/"other",llm_planner=forbidden)
    state = await failed.plan_and_run("Study a quantitative model")
    assert state.meaningfulness_certificate.failure_reason == "LLM_PROVIDER_ERROR"
    assert len(delays) == 2

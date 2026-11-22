from __future__ import annotations

import json
import errno
import os
from pathlib import Path

import pytest

from invar_agent.builder.his_cache import HISCache, interface_digest, port_signature
from invar_agent.builder.ingestion.equation_lowerer import check_z3, lower_and_verify_equations
from invar_agent.builder.ingestion.pipeline import seal_ir
from invar_agent.builder.ingestion.schemas import (
    CanonicalResearchIR, IntervalBoundModel, RawConstant, RawEquation, SIDimensionVectorModel,
    SourceSpan, StateVariableSpec,
)
from invar_agent.builder.planner import compile_builder_uast
from invar_agent.builder.state_machine import (
    BuildRequest, BuilderStage, BuilderStateMachine, EmitterRegistry, ModuleEmission,
)
from invar_agent.builder.uast_models import BuilderArchitectureSpec, BuilderProfile, BuilderUAST, DependencyEdgeModel


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_API", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "0")


class NoLLM:
    calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Unexpected LLM call")


def make_ir():
    source = SourceSpan(file="decay.tex", location="line:1", quote=r"\dot{x}=-k*x; k=0.1 s^-1")
    equations, constants, flags = lower_and_verify_equations([RawEquation(text=r"\dot{x}=-k*x", provenance=source)],
        [RawConstant(symbol="k", value="0.1", unit="1/s", provenance=source)])
    constants[0].bounds = IntervalBoundModel(lower=0, upper=1)
    ir = CanonicalResearchIR(ir_id="decay", source_mode="upload", title="Decay", equations=equations, constants=constants,
        state_variables=[StateVariableSpec(symbol="x", slot_index=0, initial_value=1, dimension=SIDimensionVectorModel(),
                                          bounds=IntervalBoundModel(lower=0, upper=100), provenance=source)],
        disambiguation_flags=flags, default_dt_si_s=.01, default_steps=100, user_approved=True)
    check_z3(ir)
    return seal_ir(ir)


def make_graph(ir):
    original = compile_builder_uast(ir, BuilderArchitectureSpec(project_name="decay", profiles=[BuilderProfile.INTERACTIVE_WORKBENCH]))
    selected = {"kernel_core", "port_core", "mod_sdk", "slot_custom_viz"}
    data = original.model_dump(mode="json")
    data["nodes"] = [node for node in data["nodes"] if node["node_id"] in selected]
    sdk = next(node for node in data["nodes"] if node["node_id"] == "mod_sdk")
    sdk.update(emitter_id="minimal_sdk", target_relpaths=["sdk.py", "test_sdk.py"], provided_port_ids=["port_core"])
    data["edges"] = [edge for edge in data["edges"] if edge["from_node"] in selected and edge["to_node"] in selected]
    return BuilderUAST.model_validate(data)


KERNEL = "def simulation_step(x, k):\n    return {'x': x * (1.0 - 0.01*k)}\n"
CPP = 'extern "C" double invar_step(double x, double k) { return x * (1.0 - 0.01*k); }\n'
KERNEL_TEST = '''import ctypes
from pathlib import Path
from kernel import simulation_step

def test_independent_equation_and_cabi_parity():
    assert abs(simulation_step(1.0, .1)['x'] - .999) < 1e-12
    library_path = Path('.invar_native/module.so')
    if library_path.exists():
        library = ctypes.CDLL(str(library_path.resolve()))
        library.invar_step.argtypes = [ctypes.c_double, ctypes.c_double]
        library.invar_step.restype = ctypes.c_double
        assert abs(library.invar_step(1.0, .1) - simulation_step(1.0, .1)['x']) < 1e-12
'''
SDK_TEST = "from sdk import simulation_step\ndef test_sdk_contract():\n    assert simulation_step(1., .1) == {'x': .999}\n"
VALID_SLOT = "def simulation_step(x, k):\n    return {'x': x}\n"


def real_emitters(*, slot=VALID_SLOT, sdk="from kernel import simulation_step\n"):
    registry = EmitterRegistry()
    registry.register("kernel", lambda node, context: ModuleEmission(
        {"kernel.py": KERNEL, "kernel.cpp": CPP, "test_kernel.py": KERNEL_TEST, "pinned.txt": "verified kernel"},
        pytest_paths=["test_kernel.py"], port_file="kernel.py"))
    registry.register("minimal_sdk", lambda node, context: ModuleEmission(
        {"sdk.py": sdk, "test_sdk.py": SDK_TEST}, pytest_paths=["test_sdk.py"], port_file="sdk.py"))
    if slot is not None:
        registry.register("slot", lambda node, context: ModuleEmission({node.target_relpath: slot}))
    return registry


@pytest.mark.asyncio
async def test_overlay_isolation_on_stage6_failure(tmp_path):
    ir = make_ir()
    observed = []
    async def progress(channel, payload):
        if payload["stage"] == "SMT_Concolic_Verify" and payload["module_id"] == "mod_sdk":
            workspace = tmp_path / ".invar/builds/isolation/workspace"
            observed.append((workspace / "pinned.txt").read_text())
            assert not (workspace / "sdk.py").exists()
            assert (tmp_path / ".invar/builds/isolation/overlays/round_1/mod_sdk/sdk.py").exists()
    client = NoLLM()
    machine = BuilderStateMachine(tmp_path, emitters=real_emitters(sdk="def simulation_step(x,k):\n    return {'x': -1.0}\n"), llm_client=client, progress_cb=progress)
    state = await machine.run(BuildRequest(build_id="isolation", ir=ir, uast=make_graph(ir)))
    assert state.status == "ABORTED_CRITICAL" and observed == ["verified kernel"]
    workspace = Path(state.workspace)
    assert (workspace / "pinned.txt").read_text() == "verified kernel" and not (workspace / "sdk.py").exists()
    assert state.committed_waves == [["kernel_core"], ["port_core"]] and client.calls == 0
    assert (workspace.parent / "partial_receipt.json").is_file()
    cache = HISCache(tmp_path, "isolation", workspace).load()
    assert set(cache.entries) == {"kernel_core", "port_core"}


@pytest.mark.asyncio
async def test_wavefront_atomic_commit_and_cabi_cleanup(tmp_path):
    ir = make_ir()
    events = []
    for index in range(3):
        machine = BuilderStateMachine(tmp_path, emitters=real_emitters(), progress_cb=lambda channel, payload: events.append((channel, payload)))
        state = await machine.run(BuildRequest(build_id=f"cabi_{index}", ir=ir, uast=make_graph(ir)))
        assert state.status == "COMPLETED", state.error
        assert state.cabi_compiled and state.unloaded_handles >= 2
        workspace = Path(state.workspace)
        assert (workspace.is_symlink() or (hasattr(workspace, "is_junction") and workspace.is_junction())) and (workspace / "sdk.py").is_file()
        assert not list(workspace.rglob("*.so"))
        assert not list(workspace.parent.rglob(".invar_native"))
        assert len(HISCache(tmp_path, state.build_id, workspace).load().entries) == 4
        assert len(state.receipt["merkle_root_blake3"]) == 64
    assert {channel for channel, _ in events} == {"log", "goi_stream", "accelerator_log"}
    assert all(payload["event_version"] == "2.0" and payload["mode"] == "builder" for _, payload in events)
    assert len(BuilderStage) == 9


@pytest.mark.asyncio
async def test_peer_failure_prevents_entire_wavefront_commit(tmp_path):
    ir = make_ir()
    graph = make_graph(ir)
    sdk = next(node for node in graph.nodes if node.node_id == "mod_sdk")
    peer = sdk.model_copy(deep=True, update={"node_id": "mod_bad", "emitter_id": "bad", "target_relpaths": ["bad.py"], "provided_port_ids": []})
    graph.nodes.append(peer)
    graph.edges.append(DependencyEdgeModel(from_node="port_core", to_node="mod_bad", edge_kind="data_port", port_id="port_core"))
    emitters = real_emitters()
    emitters.register("bad", lambda node, context: ModuleEmission({"bad.py": "raise RuntimeError('platform defect')\n"}))
    state = await BuilderStateMachine(tmp_path, emitters=emitters).run(BuildRequest(build_id="atomic", ir=ir, uast=graph))
    assert state.status == "ABORTED_CRITICAL"
    assert not (Path(state.workspace) / "sdk.py").exists() and not (Path(state.workspace) / "bad.py").exists()
    assert (Path(state.workspace) / "kernel.py").is_file()


@pytest.mark.asyncio
async def test_warm_cache_skips_verified_artifacts_without_llm(tmp_path):
    ir = make_ir()
    request = BuildRequest(build_id="cached", ir=ir, uast=make_graph(ir))
    first = await BuilderStateMachine(tmp_path, emitters=real_emitters()).run(request)
    assert first.status == "COMPLETED", first.error
    client = NoLLM()
    second = await BuilderStateMachine(tmp_path, llm_client=client).run(request)
    assert second.status == "COMPLETED", second.error
    assert sum(event["event"] == "CACHE_HIT_SKIPPED" for event in second.events) == 4 and client.calls == 0


@pytest.mark.asyncio
async def test_empty_ingestion_aborts_with_partial_receipt(tmp_path):
    state = await BuilderStateMachine(tmp_path).run(BuildRequest(build_id="empty"))
    assert state.status == "ABORTED_CRITICAL" and not state.committed_waves
    receipt = json.loads((tmp_path / ".invar/builds/empty/partial_receipt.json").read_text())
    assert receipt["status"] == "ABORTED_CRITICAL" and len(receipt["merkle_root_blake3"]) == 64


@pytest.mark.asyncio
async def test_stage9_copy_failure_preserves_previous_generation(tmp_path, monkeypatch):
    original = os.replace
    def fail_sdk_commit(source, target):
        if Path(target).name == "test_sdk.py":
            raise OSError(errno.ENOSPC, "simulated disk exhaustion")
        return original(source, target)
    monkeypatch.setattr("invar_agent.builder.state_machine.os.replace", fail_sdk_commit)
    ir = make_ir()
    state = await BuilderStateMachine(tmp_path, emitters=real_emitters()).run(BuildRequest(build_id="disk", ir=ir, uast=make_graph(ir)))
    assert state.status == "ABORTED_CRITICAL" and state.error_code == "OS_OOM_OR_DISK"
    workspace = Path(state.workspace)
    assert (workspace / "pinned.txt").is_file() and not (workspace / "sdk.py").exists()
    assert set(HISCache(tmp_path, "disk", workspace).load().entries) == {"kernel_core", "port_core"}
    assert not list((workspace.parent / "overlays").rglob("mod_sdk"))


@pytest.mark.asyncio
async def test_cross_port_z3_implication_rejects_incompatible_precondition(tmp_path):
    ir = make_ir()
    graph = make_graph(ir)
    core = next(node for node in graph.nodes if node.node_id == "port_core")
    strict = core.model_copy(deep=True, update={"node_id": "port_strict", "z3_preconditions": ["x >= 10"]})
    strict.his_digest = interface_digest(port_signature(strict))
    graph.nodes.append(strict)
    sdk = next(node for node in graph.nodes if node.node_id == "mod_sdk")
    sdk.required_port_ids, sdk.provided_port_ids = [strict.node_id], []
    client = NoLLM()
    state = await BuilderStateMachine(tmp_path, emitters=real_emitters(), llm_client=client).run(BuildRequest(build_id="contracts", ir=ir, uast=graph))
    assert state.status == "ABORTED_CRITICAL" and state.error_code == "Z3_SPEC_UNSAT" and client.calls == 0
    assert "counterexample" in str(state.events)


@pytest.mark.asyncio
async def test_cycle_aborts_before_emission_and_exposes_tarjan_witness(tmp_path):
    ir = make_ir()
    graph = make_graph(ir)
    graph.edges.append(DependencyEdgeModel(from_node="slot_custom_viz", to_node="mod_sdk", edge_kind="data_port"))
    client = NoLLM()
    state = await BuilderStateMachine(tmp_path, llm_client=client).run(BuildRequest(build_id="cycle", ir=ir, uast=graph))
    assert state.error_code == "GOI_IRREDUCIBLE_CYCLE" and not state.committed_waves and client.calls == 0
    assert state.receipt["verification"]["goi_spectral_radius"] >= 1.0
    assert "cycle_witness" in str(state.events)

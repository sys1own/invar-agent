from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from invar_agent import _fallback, _native
from invar_agent.builder.his_cache import (
    HISCache, HISCacheError, digest, interface_digest, node_fingerprints, port_signature, target_relpaths,
)
from invar_agent.builder.ingestion.equation_lowerer import check_z3, lower_and_verify_equations
from invar_agent.builder.ingestion.pipeline import seal_ir
from invar_agent.builder.ingestion.schemas import (
    CanonicalResearchIR, IntervalBoundModel, RawConstant, RawEquation,
    SIDimensionVectorModel, SourceSpan, StateVariableSpec,
)
from invar_agent.builder.planner import BuilderPlanner, compile_builder_uast, select_profiles
from invar_agent.builder.uast_models import (
    FALLBACK_EMITTERS, BuilderArchitectureSpec, BuilderProfile, BuilderUAST, DependencyEdgeModel,
    KernelNodeModel, LLMExtensionSlotNodeModel, LLMExtensionSlotRequest,
)
from invar_agent.builder.wavefront_scheduler import GoICriticalError, WavefrontScheduler
from invar_agent.llm.clients import LLMResponse
from invar_agent.uast.schemas import ResearchComputeUAST


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_API",
                 "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "INVAR_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "0")


@pytest.fixture
def research_ir():
    source = SourceSpan(file="decay.tex", location="line:1", quote=r"\dot{x} = -k*x; k=0.1 s^-1")
    equations, constants, flags = lower_and_verify_equations(
        [RawEquation(text=r"\dot{x} = -k*x", provenance=source)],
        [RawConstant(symbol="k", value="0.1", unit="1/s", provenance=source)])
    constants[0].bounds = IntervalBoundModel(lower=0, upper=1)
    ir = CanonicalResearchIR(ir_id="decay", source_mode="upload", title="Decay instrument", equations=equations,
        constants=constants, state_variables=[StateVariableSpec(symbol="x", slot_index=0, initial_value=1,
            dimension=SIDimensionVectorModel(), bounds=IntervalBoundModel(lower=0, upper=100), provenance=source)],
        disambiguation_flags=flags, default_dt_si_s=.01, default_steps=100, user_approved=True)
    check_z3(ir)
    return seal_ir(ir)


@pytest.fixture
def graph(research_ir):
    return compile_builder_uast(research_ir, BuilderArchitectureSpec(project_name="decay", profiles=list(BuilderProfile)))


def _node(graph, identifier):
    return next(node for node in graph.nodes if node.node_id == identifier)


def _checkpoint(root, graph):
    scheduler = WavefrontScheduler(root)
    plan = scheduler.plan(graph)
    cache = HISCache(root, "build_1")
    state = cache.new_state(graph, plan)
    for node in graph.nodes:
        for relpath in target_relpaths(node):
            path = root / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"verified fixture: {node.node_id}: {relpath}\n", encoding="utf-8")
        cache.record_verified(state, graph, node.node_id)
    cache.save(state)
    return scheduler, cache, cache.load()


class NoLLM:
    calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("LLM forbidden on a deterministic path")


class IntentClient:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def complete(self, prompt, **kwargs):
        self.calls.append((json.loads(prompt), kwargs))
        return LLMResponse(content=json.dumps(self.payload), input_tokens=5, output_tokens=8)


def test_builder_uast_serde_roundtrip(graph):
    assert {node.kind for node in graph.nodes} == {"kernel", "interface_port", "deterministic_module", "llm_extension_slot"}
    payload = graph.model_dump_json()
    assert _native.validate_builder_uast(payload) == payload
    assert BuilderUAST.model_validate_json(payload).model_dump_json() == payload
    port = _node(graph, "port_core")
    port.port_name = "step_λ"
    port.inputs[0].bounds.lower = 1e-9
    payload = graph.model_dump_json()
    assert "\\u03bb" in payload
    assert BuilderUAST.model_validate_json(payload).model_dump_json() == _native.validate_builder_uast(payload)
    for key in ("fallback_emitter_id", "frozen_his_digest"):
        data = graph.model_dump(mode="json")
        next(node for node in data["nodes"] if node["kind"] == "llm_extension_slot").pop(key)
        with pytest.raises(ValidationError):
            BuilderUAST.model_validate(data)
        with pytest.raises(ValueError):
            _native.validate_builder_uast(json.dumps(data))
    for mutate in (lambda data: data.update(extra=True),
                   lambda data: data["nodes"][0].update(extra=True),
                   lambda data: next(n for n in data["nodes"] if n["kind"] == "llm_extension_slot").update(max_tier2_retries=3),
                   lambda data: next(n for n in data["nodes"] if n["kind"] == "deterministic_module").update(target_relpaths=["../escape.py"])):
        data = graph.model_dump(mode="json")
        mutate(data)
        with pytest.raises(ValidationError):
            BuilderUAST.model_validate(data)


def test_canonical_eleven_module_nilpotency_and_wavefronts(graph):
    plan = WavefrontScheduler().plan(graph)
    assert len(graph.nodes) == 11
    assert plan.is_nilpotent and plan.spectral_radius == 0
    assert plan.wavefronts == [["kernel_core"], ["port_core"], ["mod_sdk"],
        ["mod_cli", "mod_data", "mod_sweep", "mod_tests", "slot_custom_viz"],
        ["mod_workbench_api"], ["mod_workbench_ui"], ["mod_receipt"]]
    order = plan.node_order
    matrix = np.zeros((len(order), len(order)))
    for edge in graph.edges:
        matrix[order.index(edge.to_node), order.index(edge.from_node)] = 1
    assert not np.any(np.linalg.matrix_power(matrix, len(order)))
    assert set(plan.rebuild_nodes) == set(order) and not plan.cached_nodes
    assert np.max(np.abs(np.asarray(plan.resolvent_flat).reshape(11, 11) - np.linalg.inv(np.eye(11) - matrix))) < 1e-12
    embedded = ResearchComputeUAST.model_validate_json(_node(graph, "kernel_core").research_compute_uast_json)
    assert embedded.governance.equations["x"]
    assert embedded.ontology.quantities["k"].default_value == .1


def test_body_only_edit_rebuilds_only_slot_and_receipt(tmp_path, graph):
    scheduler, cache, previous = _checkpoint(tmp_path, graph)
    before = cache.path.read_bytes()
    unchanged = scheduler.plan(graph, previous)
    assert not unchanged.rebuild_nodes and len(unchanged.cached_nodes) == 11
    modified = graph.model_copy(deep=True)
    _node(modified, "slot_custom_viz").intent_summary += " Use purple trajectory lines."
    assert node_fingerprints(modified)["slot_custom_viz"][0] == node_fingerprints(graph)["slot_custom_viz"][0]
    plan = scheduler.plan(modified, previous)
    assert set(plan.rebuild_nodes) == {"slot_custom_viz", "mod_receipt"}
    assert len(plan.cached_nodes) == 9
    assert sum(status == "CACHE_HIT_SKIPPED" for status in plan.node_status.values()) == 9
    assert plan.used_smw and plan.smw_capacitance_det == pytest.approx(1)
    assert cache.path.read_bytes() == before
    direct = np.asarray(unchanged.resolvent_flat).reshape(11, 11)
    column = unchanged.node_order.index("slot_custom_viz")
    assert {unchanged.node_order[i] for i in range(11) if direct[i, column] > 0} == {
        "slot_custom_viz", "mod_workbench_api", "mod_workbench_ui", "mod_receipt"}


def test_kernel_his_change_rebuilds_full_cone(tmp_path, graph):
    scheduler, _, previous = _checkpoint(tmp_path, graph)
    port = _node(graph, "port_core")
    port.inputs[0].bounds.upper = 200
    port.his_digest = interface_digest(port_signature(port))
    _node(graph, "kernel_core").his_digest = port.his_digest
    _node(graph, "slot_custom_viz").frozen_his_digest = port.his_digest
    plan = scheduler.plan(graph, previous)
    assert len(plan.rebuild_nodes) == 11 and not plan.cached_nodes
    assert plan.smw_capacitance_det == pytest.approx(1)


@pytest.mark.parametrize("warm", [False, True])
def test_cycle_edge_aborts_before_mutation(tmp_path, graph, research_ir, warm):
    client = NoLLM()
    BuilderPlanner(client).translate_intent(research_ir, profiles=list(BuilderProfile))
    scheduler, cache, previous = _checkpoint(tmp_path, graph)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    graph.edges.append(DependencyEdgeModel(from_node="mod_workbench_api", to_node="slot_custom_viz", edge_kind="data_port"))
    with pytest.raises(GoICriticalError) as error:
        scheduler.plan(graph, previous if warm else None)
    assert error.value.code == "GOI_IRREDUCIBLE_CYCLE" and error.value.llm_retries == 0
    assert set(error.value.cycle_witness) == {"mod_workbench_api", "slot_custom_viz"}
    assert error.value.capacitance_det == pytest.approx(0)
    assert error.value.spectral_radius >= 1 - 1e-12
    assert error.value.action == "ABORT_BUILD_CRITICAL" and client.calls == 0
    assert {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert cache.path.exists()


def test_goi_smw_rank2_exactness(tmp_path, graph, monkeypatch):
    scheduler, _, previous = _checkpoint(tmp_path, graph)
    graph.edges.extend([DependencyEdgeModel(from_node=a, to_node=b, edge_kind="data_port")
                        for a, b in (("mod_cli", "mod_tests"), ("mod_data", "mod_workbench_api"))])
    def no_full_inversion(*args, **kwargs):
        raise AssertionError("Compatible updates must use SMW, not full scheduling")
    monkeypatch.setattr(_native, "schedule_builder_wavefronts", no_full_inversion)
    plan = scheduler.plan(graph, previous)
    order = plan.node_order
    matrix = np.zeros((11, 11))
    for edge in graph.edges:
        matrix[order.index(edge.to_node), order.index(edge.from_node)] = 1
    assert plan.used_smw and plan.smw_capacitance_det == pytest.approx(1, abs=1e-12)
    assert np.max(np.abs(np.asarray(plan.resolvent_flat).reshape(11, 11) - np.linalg.inv(np.eye(11) - matrix))) < 1e-12
    next_state = previous.model_copy(deep=True, update={"resolvent_flat": plan.resolvent_flat, "edges": graph.edges.copy()})
    graph.edges = graph.edges[:-2]
    restored = scheduler.plan(graph, next_state)
    assert np.max(np.abs(np.asarray(restored.resolvent_flat) - previous.resolvent_flat)) < 1e-12


PROFILE_SUBSETS = [subset for size in range(1, 5) for subset in itertools.combinations(BuilderProfile, size)]


@pytest.mark.parametrize("profiles", PROFILE_SUBSETS)
def test_composable_profile_subsets(research_ir, profiles):
    spec = BuilderArchitectureSpec(project_name="subset", profiles=list(profiles))
    graph = compile_builder_uast(research_ir, spec)
    plan = WavefrontScheduler().plan(graph)
    identifiers = {node.node_id for node in graph.nodes}
    assert {"kernel_core", "port_core", "mod_sdk", "mod_tests", "mod_receipt"} <= identifiers
    assert ("mod_workbench_api" in identifiers) == (BuilderProfile.INTERACTIVE_WORKBENCH in profiles)
    assert (_node(graph, "mod_data").module_role == "calibration_engine") == (BuilderProfile.CALIBRATION_INVERSE_DESIGN in profiles)
    assert len([node for node in graph.nodes if isinstance(node, KernelNodeModel)]) == 1
    assert plan.is_nilpotent and len(plan.rebuild_nodes) == len(graph.nodes)


@pytest.mark.parametrize("profiles,kind,identifier", [
    ([1, 2, 3], "domain_visualization", "slot_custom_viz"),
    ([1, 4, 3], "bespoke_data_adapter", "slot_data_adapter"),
])
def test_requested_subset_leaf_contracts(research_ir, profiles, kind, identifier):
    selected = [list(BuilderProfile)[index - 1] for index in profiles]
    spec = BuilderArchitectureSpec(project_name="subset", profiles=selected, llm_extension_slots=[
        LLMExtensionSlotRequest(slot_id=identifier, slot_kind=kind, intent_summary="Use the verified x trajectory",
            fallback_emitter_id=FALLBACK_EMITTERS[kind], input_symbols=["x"], output_symbols=["x"])])
    graph = compile_builder_uast(research_ir, spec)
    slot = _node(graph, identifier)
    assert slot.frozen_his_digest == _node(graph, slot.frozen_port_id).his_digest
    assert len([node for node in graph.nodes if isinstance(node, LLMExtensionSlotNodeModel)]) == 1
    assert WavefrontScheduler().plan(graph).is_nilpotent
    expected_target = "mod_workbench_api" if kind == "domain_visualization" else "mod_data"
    assert any(edge.from_node == identifier and edge.to_node == expected_target for edge in graph.edges)


def test_profile_selector_and_scoped_intent_translation(research_ir):
    assert select_profiles("workbench, audit and calibration") == list(BuilderProfile)[1:]
    assert select_profiles("all four profiles") == list(BuilderProfile)
    assert select_profiles("profiles 1+4+3") == [list(BuilderProfile)[i] for i in (0, 2, 3)]
    assert select_profiles("hybrid_compute_library executable_paper_audit") == [list(BuilderProfile)[i] for i in (0, 2)]
    assert select_profiles("") == [BuilderProfile.HYBRID_COMPUTE_LIBRARY]
    no_llm = NoLLM()
    planner = BuilderPlanner(no_llm)
    planner.translate_intent(research_ir, "Interactive workbench with domain visualization")
    planner.translate_intent(research_ir, "", profiles=[BuilderProfile.CALIBRATION_INVERSE_DESIGN])
    assert no_llm.calls == 0 and planner.llm_tokens_used == 0
    baseline = BuilderPlanner().translate_intent(research_ir, "Make a specialized instrument for our laboratory")
    client = IntentClient({"llm_extension_slots": [{"slot_id": "slot_custom_viz", "slot_kind": "domain_visualization",
        "intent_summary": "Plot verified x trajectories", "fallback_emitter_id": "fb_svg_trajectory_viz", "input_symbols": ["x"]}]})
    planner = BuilderPlanner(client)
    translated = planner.translate_intent(research_ir, "Make a specialized instrument for our laboratory")
    assert planner.llm_tokens_used == 13 and len(client.calls) == 1
    assert "equations" not in client.calls[0][0] and client.calls[0][1]["temperature"] == 0
    before = compile_builder_uast(research_ir, baseline)
    after = planner.compile(research_ir, translated)
    assert [n.model_dump() for n in before.nodes] == [n.model_dump() for n in after.nodes if n.kind != "llm_extension_slot"]
    assert len(client.calls) == 1
    malicious = IntentClient({"enable_cpp_cabi": False, "nodes": [{"kind": "kernel"}]})
    with pytest.raises(ValidationError):
        BuilderPlanner(malicious).translate_intent(research_ir, "Make something tailored for our lab")
    with pytest.raises(ValueError):
        select_profiles("", [])
    unapproved = research_ir.model_copy(update={"user_approved": False})
    with pytest.raises(ValueError, match="IR_UNVERIFIED_FLAGGED"):
        compile_builder_uast(unapproved, baseline)


def test_his_cache_persistence_artifact_tamper_and_atomic_failure(tmp_path, graph, monkeypatch):
    scheduler, cache, previous = _checkpoint(tmp_path, graph)
    assert cache.path == tmp_path / ".invar/builds/build_1/his_cache.json"
    assert HISCache(tmp_path, "missing").load() is None
    assert cache.load().model_dump() == previous.model_dump()
    target = tmp_path / target_relpaths(_node(graph, "mod_sdk"))[0]
    target.write_text("tampered", encoding="utf-8")
    plan = scheduler.plan(graph, previous)
    assert set(plan.rebuild_nodes) == {"mod_sdk", "mod_receipt"}
    target.unlink()
    assert set(scheduler.plan(graph, previous).rebuild_nodes) == {"mod_sdk", "mod_receipt"}
    before = cache.path.read_bytes()
    def fail_replace(*args):
        raise OSError("simulated interrupted commit")
    monkeypatch.setattr("invar_agent.builder.his_cache.os.replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        cache.save(previous)
    assert cache.path.read_bytes() == before and not list(cache.path.parent.glob(".his_cache.*"))
    with pytest.raises(HISCacheError):
        HISCache(tmp_path, "../escape")
    previous.resolvent_flat[0] += .5
    with pytest.raises(HISCacheError, match="resolvent"):
        scheduler.plan(graph, previous)


def test_graph_changes_cache_safety_and_empty_graph(tmp_path, graph, research_ir):
    scheduler, cache, previous = _checkpoint(tmp_path, graph)
    spec = BuilderArchitectureSpec(project_name="decay", profiles=list(BuilderProfile), llm_extension_slots=[
        LLMExtensionSlotRequest(slot_id="slot_workflow", slot_kind="workflow_glue", intent_summary="Use a deterministic workflow",
                               fallback_emitter_id="fb_workflow_glue")])
    modified = compile_builder_uast(research_ir, spec)
    plan = scheduler.plan(modified, previous)
    assert not plan.used_smw and {"slot_workflow", "mod_cli", "mod_receipt"} <= set(plan.rebuild_nodes)
    assert "kernel_core" in plan.cached_nodes
    wrong_project = previous.model_copy(update={"project_id": "other"})
    with pytest.raises(HISCacheError, match="PROJECT_MISMATCH"):
        scheduler.plan(graph, wrong_project)
    cache.path.write_text("not json", encoding="utf-8")
    with pytest.raises(HISCacheError, match="HIS_CACHE_INVALID"):
        cache.load()
    empty = BuilderUAST(project_id="empty", canonical_ir_digest="a" * 64, active_profiles=[], nodes=[], edges=[])
    empty_plan = WavefrontScheduler().plan(empty)
    assert not empty_plan.wavefronts and not empty_plan.rebuild_nodes
    assert not empty_plan.cached_nodes and empty_plan.smw_capacitance_det == 1


def test_blake3_domain_separation(graph):
    assert digest("invar.his.interface.v2", {}) != digest("invar.his.body.v2", {})
    for node in graph.nodes:
        assert len(node_fingerprints(graph)[node.node_id][0]) == 64


def test_native_absent_graph_and_incremental_parity(tmp_path, graph, monkeypatch):
    scheduler, _, previous = _checkpoint(tmp_path, graph)
    expected = scheduler.plan(graph, previous)
    monkeypatch.setitem(sys.modules, "invar_agent._native", None)
    def native_forbidden(*args, **kwargs):
        raise AssertionError("Native backend must not execute while absent")
    for name in ("validate_builder_uast", "schedule_builder_wavefronts", "compute_builder_smw_delta", "compute_his_digest_v2"):
        monkeypatch.setattr(_native, name, native_forbidden)
    copied = BuilderUAST.model_validate_json(_fallback.validate_builder_uast(graph.model_dump_json()))
    actual = WavefrontScheduler(tmp_path).plan(copied, previous)
    assert actual.model_dump() == expected.model_dump()
    _node(copied, "slot_custom_viz").intent_summary += " Use green lines."
    assert set(scheduler.plan(copied, previous).rebuild_nodes) == {"slot_custom_viz", "mod_receipt"}

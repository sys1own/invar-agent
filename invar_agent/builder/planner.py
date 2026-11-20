"""Deterministic profile compilation with a leaf-only optional intent translator."""
from __future__ import annotations

import re

from pydantic import Field

from invar_agent.uast.schemas import ResearchComputeUAST
from invar_agent.llm.errors import complete_for_builder

from .his_cache import interface_digest, port_signature
from .ingestion.pipeline import seal_ir
from .ingestion.schemas import CanonicalResearchIR, EquationRole, StrictModel
from .uast_models import (
    FALLBACK_EMITTERS, BuilderArchitectureSpec, BuilderProfile, BuilderUAST, DependencyEdgeModel,
    DeterministicModuleNodeModel, InterfacePortNodeModel, KernelNodeModel,
    LLMExtensionSlotNodeModel, LLMExtensionSlotRequest, TypedPortFieldModel, canonical_json,
)


PROFILE_PATTERNS = {
    BuilderProfile.HYBRID_COMPUTE_LIBRARY: r"\b(?:hybrid_compute_library|sdk|library|hybrid compute|profile\s*1)\b",
    BuilderProfile.INTERACTIVE_WORKBENCH: r"\b(?:interactive_workbench|workbench|interactive|dashboard|profile\s*2)\b",
    BuilderProfile.EXECUTABLE_PAPER_AUDIT: r"\b(?:executable_paper_audit|audit|executable paper|reproducibility|profile\s*3)\b",
    BuilderProfile.CALIBRATION_INVERSE_DESIGN: r"\b(?:calibration_inverse_design|calibration|calibrate|inverse design|fitting|profile\s*4)\b",
}
SLOT_PATTERNS = {
    "bespoke_data_adapter": ("slot_data_adapter", r"\b(?:bespoke data adapter|custom data adapter|bespoke_data_adapter)\b"),
    "domain_visualization": ("slot_custom_viz", r"\b(?:domain visualization|custom visualization|domain_visualization)\b"),
    "workflow_glue": ("slot_workflow", r"\b(?:workflow glue|workflow_glue)\b"),
    "narrative_documentation": ("slot_narrative", r"\b(?:narrative documentation|narrative_documentation)\b"),
}


def _numbered_profiles(prompt):
    groups = re.findall(r"\bprofiles?\s+([1-4](?:\s*(?:[+,/&]|and)\s*[1-4])*)\b", prompt, re.I)
    return {list(BuilderProfile)[int(number) - 1] for group in groups for number in re.findall(r"[1-4]", group)}


def select_profiles(prompt: str = "", profiles: list[BuilderProfile | str] | None = None) -> list[BuilderProfile]:
    if profiles is not None:
        if not profiles:
            raise ValueError("Select at least one instrument profile")
        selected = {BuilderProfile(profile) for profile in profiles}
    elif re.search(r"\b(?:all (?:four|4) profiles|all profiles)\b", prompt, re.I):
        selected = set(BuilderProfile)
    else:
        selected = {profile for profile, pattern in PROFILE_PATTERNS.items() if re.search(pattern, prompt, re.I)}
        selected.update(_numbered_profiles(prompt))
        selected = selected or {BuilderProfile.HYBRID_COMPUTE_LIBRARY}
    return [profile for profile in BuilderProfile if profile in selected]


class _LeafIntentProposal(StrictModel):
    llm_extension_slots: list[LLMExtensionSlotRequest] = Field(default_factory=list, max_length=8)
    rationale: str = ""


class BuilderPlanner:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.llm_tokens_used = 0

    def translate_intent(self, ir: CanonicalResearchIR, prompt: str = "", *,
                         profiles: list[BuilderProfile | str] | None = None,
                         project_name: str | None = None) -> BuilderArchitectureSpec:
        selected = select_profiles(prompt, profiles)
        name = re.sub(r"[^a-z0-9]+", "_", ir.title.lower()).strip("_")[:49]
        if not name or not name[0].isalpha():
            name = "instrument_" + name[:38]
        if len(name) < 2:
            name += "_instrument"
        slots = [LLMExtensionSlotRequest(slot_id=identifier, slot_kind=kind, intent_summary=prompt.strip(),
                    fallback_emitter_id=FALLBACK_EMITTERS[kind])
                 for kind, (identifier, pattern) in SLOT_PATTERNS.items() if re.search(pattern, prompt, re.I)]
        explicit = profiles is not None or bool(slots) or bool(_numbered_profiles(prompt)) or any(re.search(pattern, prompt, re.I) for pattern in PROFILE_PATTERNS.values())
        explicit = explicit or bool(re.search(r"\ball (?:four|4)?\s*profiles\b", prompt, re.I))
        rationale = "Deterministic profile selection and frozen scientific interfaces"
        if self.llm_client is not None and prompt.strip() and not explicit:
            response = complete_for_builder(self.llm_client, canonical_json({"intent": prompt[:6000],
                "selected_profiles": [profile.value for profile in selected],
                "allowed_slots": FALLBACK_EMITTERS,
                "symbols": sorted({c.symbol for c in ir.constants} | {s.symbol for s in ir.state_variables})}),
                system="Return JSON with llm_extension_slots and rationale only. Shape optional leaf prompts and symbol selections. Never author kernels, deterministic modules, profiles, file paths or executable code.",
                max_tokens=2048, temperature=0.0)
            self.llm_tokens_used += max(0, response.input_tokens) + max(0, response.output_tokens)
            proposal = _LeafIntentProposal.model_validate_json(response.content)
            slots, rationale = proposal.llm_extension_slots, proposal.rationale
        return BuilderArchitectureSpec(project_name=project_name or name, profiles=selected,
            llm_extension_slots=slots, sweep_parameters=[c.symbol for c in ir.constants if c.is_tunable_parameter],
            calibration_target_symbols=[s.symbol for s in ir.state_variables] if BuilderProfile.CALIBRATION_INVERSE_DESIGN in selected else [],
            rationale=rationale)

    def compile(self, ir: CanonicalResearchIR, spec: BuilderArchitectureSpec) -> BuilderUAST:
        return compile_builder_uast(ir, spec)


def _research_compute_uast(ir: CanonicalResearchIR, spec: BuilderArchitectureSpec) -> dict:
    quantities = {}
    for item in [*ir.constants, *ir.state_variables]:
        powers = list(item.dimension.exponents)
        powers[0], powers[1] = powers[1], powers[0]
        bounds = item.bounds
        quantities[item.symbol] = dict(symbol=item.symbol, si_dimension=powers,
            semantic_tag=item.dimension.semantic_tag, dimension_unspecified=False,
            interval=[bounds.lower if bounds and bounds.lower is not None else -1e300,
                      bounds.upper if bounds and bounds.upper is not None else 1e300],
            unit=item.dimension.canonical_unit, is_state_variable=hasattr(item, "slot_index"),
            initial_value=getattr(item, "initial_value", None) or 0.0,
            default_value=getattr(item, "si_value", getattr(item, "initial_value", None)) or 0.0)
    payload = ResearchComputeUAST.model_validate(dict(title=ir.title, ontology={"quantities": quantities},
        governance={"equations": {e.lhs_symbol: e.sympy_expr for e in ir.equations if e.role == EquationRole.EVOLUTION_LAW and e.lhs_symbol and e.sympy_expr},
                    "observables": {e.lhs_symbol: e.sympy_expr for e in ir.equations if e.role == EquationRole.ALGEBRAIC_CLOSURE and e.lhs_symbol and e.sympy_expr}},
        invariants={"custom_smt_assertions": [e.canonical_latex for e in ir.equations if e.role == EquationRole.INVARIANT_CONTRACT]},
        dt=ir.default_dt_si_s if ir.default_dt_si_s is not None else .01,
        steps=ir.default_steps if ir.default_steps is not None else 1000))
    return payload.model_dump(mode="json")


def compile_builder_uast(ir: CanonicalResearchIR, spec: BuilderArchitectureSpec) -> BuilderUAST:
    spec = BuilderArchitectureSpec.model_validate(spec.model_dump(mode="json"))
    if not ir.user_approved or any(not f.resolved for f in ir.disambiguation_flags):
        raise ValueError("IR_UNVERIFIED_FLAGGED: researcher approval is required")
    if any(not e.hin_homogeneous or e.z3_satisfiable is not True for e in ir.equations):
        raise ValueError("IR_UNVERIFIED_FLAGGED: equations must pass HIN and Z3")
    if seal_ir(ir.model_copy(deep=True)).ir_blake3_digest != ir.ir_blake3_digest:
        raise ValueError("IR_DIGEST_MISMATCH")
    prefix = f"src/{spec.project_name}"
    states = sorted(ir.state_variables, key=lambda state: state.slot_index)
    inputs = [TypedPortFieldModel(name=item.symbol, scalar_type="f64", dimension=item.dimension, bounds=item.bounds)
              for item in [*states, *ir.constants]]
    outputs = [TypedPortFieldModel(name=item.symbol, scalar_type="f64", dimension=item.dimension, bounds=item.bounds)
               for item in states]
    for equation in ir.equations:
        if (equation.role == EquationRole.ALGEBRAIC_CLOSURE and equation.lhs_symbol and equation.dimension_lhs
                and equation.lhs_symbol not in {field.name for field in outputs}):
            outputs.append(TypedPortFieldModel(name=equation.lhs_symbol, scalar_type="f64", dimension=equation.dimension_lhs))
    if set(spec.sweep_parameters) - {constant.symbol for constant in ir.constants}:
        raise ValueError("Sweep parameters must be verified IR constants")
    if set(spec.calibration_target_symbols) - {field.name for field in outputs}:
        raise ValueError("Calibration targets must be verified IR outputs")
    port = InterfacePortNodeModel(node_id="port_core", port_name="simulation_step", inputs=inputs, outputs=outputs,
        z3_postconditions=[e.canonical_latex for e in ir.equations if e.role == EquationRole.INVARIANT_CONTRACT], his_digest="0" * 64)
    port.his_digest = interface_digest(port_signature(port))
    research = _research_compute_uast(ir, spec)
    kernel = KernelNodeModel(node_id="kernel_core", kernel_name="simulation_step", research_compute_uast_json=canonical_json(research),
        csel_expressions=[canonical_json(e.csel_ast) for e in ir.equations if e.csel_ast is not None],
        export_adjoint_jacobian=spec.enable_adjoint_jacobian, output_port_id=port.node_id, his_digest=port.his_digest)
    nodes, edges = [kernel, port], []

    def edge(source, target, kind="data_port"):
        edges.append(DependencyEdgeModel(from_node=source, to_node=target, edge_kind=kind,
                                        port_id=None if kind == "receipt_dep" else port.node_id))

    def module(identifier, role, emitter, paths, config=None):
        nodes.append(DeterministicModuleNodeModel(node_id=identifier, module_role=role, emitter_id=emitter,
            target_relpaths=paths, required_port_ids=[port.node_id], emitter_config={"package_name": spec.project_name, **(config or {})},
            his_digest=port.his_digest))

    workbench = BuilderProfile.INTERACTIVE_WORKBENCH in spec.profiles
    audit = BuilderProfile.EXECUTABLE_PAPER_AUDIT in spec.profiles
    calibration = BuilderProfile.CALIBRATION_INVERSE_DESIGN in spec.profiles
    sdk_paths = ["pyproject.toml", f"{prefix}/kernel_py.py", f"{prefix}/sdk.py"]
    if ir.research_reference is not None:
        sdk_paths.append(f'{prefix}/research_reference.py')
        sdk_paths.append(f'{prefix}/pinned_research.py')
    if spec.enable_cpp_cabi:
        sdk_paths += [f"{prefix}/native/kernel.cpp", f"{prefix}/native/kernel.h", f"{prefix}/native/kernel.so", f"{prefix}/cabi_runtime.py"]
    module("mod_sdk", "python_sdk", "hybrid_compute", sdk_paths, dict(enable_cpp_cabi=spec.enable_cpp_cabi,
        enable_adjoint_jacobian=spec.enable_adjoint_jacobian, research_compute_uast=research, pinned_kernel_code=ir.pinned_kernel_code))
    module("mod_cli", "cli_runner", "executable_paper" if audit else "hybrid_compute",
        [f"{prefix}/audit/{name}.py" for name in ("cli", "report_generator", "verify_receipt")] if audit else [f"{prefix}/cli.py"])
    module("mod_sweep", "sweep_pareto_runner", "hybrid_compute", [f"{prefix}/sweep.py"], {"sweep_parameters": spec.sweep_parameters})
    data_paths = [f"{prefix}/serializers.py"]
    if calibration:
        data_paths += [f"{prefix}/calibration/{name}.py" for name in ("data_loader", "fitter")]
    module("mod_data", "calibration_engine" if calibration else "data_serializer", "calibration" if calibration else "hybrid_compute",
        data_paths, {"datasets": ir.datasets_attached, "calibration_target_symbols": spec.calibration_target_symbols})
    tests = [f"tests/{name}.py" for name in ("test_kernel_contracts", "test_z3_invariants", "test_research_parity")]
    if workbench:
        tests.append("tests/test_workbench_api.py")
    if calibration:
        tests.append("tests/test_calibration_adjoint.py")
    module("mod_tests", "pytest_z3_suite", "hybrid_compute", tests)
    edge("kernel_core", "port_core")
    edge("port_core", "mod_sdk", "abi_link")
    for name in ("mod_cli", "mod_sweep", "mod_data", "mod_tests"):
        edge("mod_sdk", name, "verification_dep" if name == "mod_tests" else "data_port")
    requests = list(spec.llm_extension_slots)
    if workbench and not requests:
        requests = [LLMExtensionSlotRequest(slot_id="slot_custom_viz", slot_kind="domain_visualization",
            intent_summary="Render bounded simulation trajectories as an SVG visualization using the frozen SDK interface.",
            fallback_emitter_id="fb_svg_trajectory_viz")]
    symbols = {field.name for field in [*inputs, *outputs]}
    for request in requests:
        if (set(request.input_symbols) | set(request.output_symbols)) - symbols:
            raise ValueError("Extension symbols must come from the verified IR")
        selections = "" if not request.input_symbols and not request.output_symbols else "\nSymbol selections: " + canonical_json(
            {"inputs": request.input_symbols, "outputs": request.output_symbols})
        extension = ".md" if request.slot_kind == "narrative_documentation" else ".py"
        nodes.append(LLMExtensionSlotNodeModel(node_id=request.slot_id, slot_kind=request.slot_kind,
            target_relpath=f"{prefix}/extensions/{request.slot_id}{extension}", intent_summary=request.intent_summary + selections,
            frozen_port_id=port.node_id, frozen_his_digest=port.his_digest, fallback_emitter_id=request.fallback_emitter_id))
        edge("mod_sdk", request.slot_id)
        if request.slot_kind == "bespoke_data_adapter":
            edge(request.slot_id, "mod_data")
        elif request.slot_kind in {"workflow_glue", "narrative_documentation"}:
            edge(request.slot_id, "mod_cli")
    if workbench:
        module("mod_workbench_api", "fastapi_workbench", "workbench", [f"{prefix}/workbench/app.py"], {"surface": "api"})
        module("mod_workbench_ui", "fastapi_workbench", "workbench", [f"{prefix}/workbench/static/index.html"], {"surface": "ui"})
        visualizations = [request.slot_id for request in requests if request.slot_kind == "domain_visualization"]
        for source in visualizations or ["mod_sdk"]:
            edge(source, "mod_workbench_api")
        edge("mod_workbench_api", "mod_workbench_ui")
    module("mod_receipt", "receipt_builder", "receipts", ["provenance/build_receipt.json"],
        {"canonical_ir_digest": ir.ir_blake3_digest, "active_profiles": [p.value for p in spec.profiles]})
    outgoing = {dependency.from_node for dependency in edges}
    for node in nodes:
        if node.node_id not in outgoing and node.node_id != "mod_receipt":
            edge(node.node_id, "mod_receipt", "receipt_dep")
    return BuilderUAST(project_id=spec.project_name, canonical_ir_digest=ir.ir_blake3_digest,
        active_profiles=spec.profiles, nodes=nodes, edges=edges)

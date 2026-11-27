"""Typed architecture and Rust-compatible project graph boundaries."""
from __future__ import annotations

import importlib
import json
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .ingestion.schemas import IntervalBoundModel, SIDimensionVectorModel, StrictModel


def builder_backend():
    try:
        backend = importlib.import_module("invar_agent._native")
    except ImportError:
        from invar_agent import _fallback as backend
    return backend


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


class BuilderProfile(str, Enum):
    HYBRID_COMPUTE_LIBRARY = "hybrid_compute_library"
    INTERACTIVE_WORKBENCH = "interactive_workbench"
    EXECUTABLE_PAPER_AUDIT = "executable_paper_audit"
    CALIBRATION_INVERSE_DESIGN = "calibration_inverse_design"


class ProgramKind(str, Enum):
    """ARCH-005: Typed program kind derived strictly from equation roles."""
    ALGEBRAIC_MAP = "AlgebraicMap"
    ROOT_FIND = "RootFind"
    ODE = "ODE"
    DISCRETE_TIME_SYSTEM = "DiscreteTimeSystem"
    STOCHASTIC_SIMULATION = "StochasticSimulation"


DeterministicModuleRole = Literal[
    "package_metadata", "cpp_cabi_bridge", "python_fallback_kernel", "python_sdk",
    "cli_runner", "sweep_pareto_runner", "data_serializer", "fastapi_workbench",
    "calibration_engine", "audit_report", "pytest_z3_suite", "receipt_builder",
]
SlotKind = Literal["bespoke_data_adapter", "domain_visualization", "workflow_glue", "narrative_documentation"]
FALLBACK_EMITTERS = {
    "bespoke_data_adapter": "fb_csv_json_adapter",
    "domain_visualization": "fb_svg_trajectory_viz",
    "workflow_glue": "fb_workflow_glue",
    "narrative_documentation": "fb_markdown_narrative_doc",
}


class SlotOutputSpec(StrictModel):
    name: str
    scalar_type: Literal["f64", "i64", "bool", "str", "enum"]
    enum: list[str] | None = None
    range: tuple[float, float] | None = None


class SlotExample(StrictModel):
    inputs: dict[str, float | int | str | bool]
    expected: dict[str, float | int | str | bool]
    tolerance: float = 1e-9


class LLMExtensionSlotRequest(StrictModel):
    slot_id: str = Field(pattern=r"^slot_[a-z][a-z0-9_]*$")
    slot_kind: SlotKind
    intent_summary: str = Field(min_length=1)
    fallback_emitter_id: Literal["fb_csv_json_adapter", "fb_svg_trajectory_viz", "fb_workflow_glue", "fb_markdown_narrative_doc"]
    input_symbols: list[str] = Field(default_factory=list)
    output_symbols: list[str] = Field(default_factory=list)
    output_spec: list[SlotOutputSpec] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    examples: list[SlotExample] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def resolve_kind_alias(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            kind = data.get("slot_kind", data.get("slot_id"))
            if kind in FALLBACK_EMITTERS:
                data.setdefault("slot_kind", kind)
                if data.get("slot_id") in {None, kind}:
                    data["slot_id"] = "slot_custom_viz" if kind == "domain_visualization" else "slot_data_adapter" if kind == "bespoke_data_adapter" else "slot_" + kind
                data.setdefault("fallback_emitter_id", FALLBACK_EMITTERS[kind])
        return data

    @model_validator(mode="after")
    def matching_fallback(self):
        if self.fallback_emitter_id != FALLBACK_EMITTERS[self.slot_kind]:
            raise ValueError("Slot kind requires its registered deterministic fallback")
        return self


class BuilderArchitectureSpec(StrictModel):
    schema_version: Literal["builder_arch_spec.v1"] = "builder_arch_spec.v1"
    project_name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,48}$")
    profiles: list[BuilderProfile] = Field(min_length=1)
    enable_cpp_cabi: bool = True
    enable_adjoint_jacobian: bool = True
    llm_extension_slots: list[LLMExtensionSlotRequest] = Field(default_factory=list)
    sweep_parameters: list[str] = Field(default_factory=list)
    calibration_target_symbols: list[str] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def unique_selections(self):
        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError("Duplicate instrument profile")
        identifiers = [slot.slot_id for slot in self.llm_extension_slots]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate extension slot")
        return self


class TypedPortFieldModel(StrictModel):
    name: str
    scalar_type: Literal["f64", "i64", "bool", "vec_f64", "str", "table"]
    dimension: SIDimensionVectorModel
    bounds: IntervalBoundModel | None = None
    mell_linearity: Literal["Linear", "Bang", "Tensor"] = "Bang"


class InterfacePortNodeModel(StrictModel):
    kind: Literal["interface_port"] = "interface_port"
    node_id: str
    port_name: str
    inputs: list[TypedPortFieldModel]
    outputs: list[TypedPortFieldModel]
    z3_preconditions: list[str] = Field(default_factory=list)
    z3_postconditions: list[str] = Field(default_factory=list)
    his_digest: str


class KernelNodeModel(StrictModel):
    kind: Literal["kernel"] = "kernel"
    node_id: str
    kernel_name: str
    research_compute_uast_json: str
    csel_expressions: list[str]
    manifold_projectors: list[str] = Field(default_factory=list)
    export_adjoint_jacobian: bool = True
    output_port_id: str
    his_digest: str


class DeterministicModuleNodeModel(StrictModel):
    kind: Literal["deterministic_module"] = "deterministic_module"
    node_id: str
    module_role: DeterministicModuleRole
    emitter_id: str
    target_relpaths: list[str]
    required_port_ids: list[str] = Field(default_factory=list)
    provided_port_ids: list[str] = Field(default_factory=list)
    emitter_config: dict[str, Any] = Field(default_factory=dict)
    his_digest: str


class LLMExtensionSlotNodeModel(StrictModel):
    kind: Literal["llm_extension_slot"] = "llm_extension_slot"
    node_id: str
    slot_kind: SlotKind
    target_relpath: str
    intent_summary: str
    frozen_port_id: str
    frozen_his_digest: str
    fallback_emitter_id: str
    max_tier2_retries: int = Field(default=2, ge=0, le=2)


BuilderUASTNode = Annotated[
    KernelNodeModel | InterfacePortNodeModel | DeterministicModuleNodeModel | LLMExtensionSlotNodeModel,
    Field(discriminator="kind"),
]


class DependencyEdgeModel(StrictModel):
    from_node: str
    to_node: str
    edge_kind: Literal["data_port", "abi_link", "verification_dep", "receipt_dep"]
    port_id: str | None = None


class BuilderUAST(StrictModel):
    schema_version: Literal["builder_uast.v1"] = "builder_uast.v1"
    project_id: str
    canonical_ir_digest: str
    active_profiles: list[BuilderProfile]
    nodes: list[BuilderUASTNode]
    edges: list[DependencyEdgeModel]

    @model_validator(mode="after")
    def validate_native_contract(self):
        builder_backend().validate_builder_uast(canonical_json(self.model_dump(mode="json")))
        return self

    def model_dump_json(self, **kwargs) -> str:
        return builder_backend().validate_builder_uast(super().model_dump_json(**kwargs))


ProjectGraphUAST = BuilderUAST


class WavefrontScheduleResult(StrictModel):
    is_nilpotent: bool
    spectral_radius: float
    wavefronts: list[list[str]]
    resolvent_flat: list[float]
    node_order: list[str]
    cycle_witness: list[str]


class SmwDeltaResult(StrictModel):
    is_valid_dag: bool
    capacitance_det: float
    updated_resolvent_flat: list[float]
    affected_node_ids: list[str]


class WavefrontExecutionPlan(StrictModel):
    schema_version: Literal["builder_wavefront_plan.v1"] = "builder_wavefront_plan.v1"
    project_id: str
    uast_digest: str
    is_nilpotent: bool
    spectral_radius: float
    wavefronts: list[list[str]]
    node_order: list[str]
    resolvent_flat: list[float]
    rebuild_nodes: list[str]
    cached_nodes: list[str]
    smw_capacitance_det: float = 1.0
    used_smw: bool = False
    node_status: dict[str, Literal["PENDING", "CACHE_HIT_SKIPPED"]]

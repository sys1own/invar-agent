from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, allow_inf_nan=False)


class SourceSpan(StrictModel):
    file: str = Field(min_length=1)
    location: str
    quote: str = Field(min_length=1, max_length=2000)
    char_start: int | None = None
    char_end: int | None = None


class SIDimensionVectorModel(StrictModel):
    exponents: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)
    semantic_tag: int = Field(default=0, ge=0, le=255)
    canonical_unit: str = "1"


class IntervalBoundModel(StrictModel):
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True


class EquationRole(str, Enum):
    EVOLUTION_LAW = "EVOLUTION_LAW"
    ALGEBRAIC_CLOSURE = "ALGEBRAIC_CLOSURE"
    INVARIANT_CONTRACT = "INVARIANT_CONTRACT"
    BOUNDARY_CONDITION = "BOUNDARY_CONDITION"
    INITIAL_CONDITION = "INITIAL_CONDITION"
    PARAMETER_TABLE = "PARAMETER_TABLE"
    RUNTIME_PREDICATE = "RUNTIME_PREDICATE"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    WARNING = "WARNING"
    UNVERIFIED_FLAGGED = "UNVERIFIED_FLAGGED"
    REJECTED = "REJECTED"
    USER_OVERRIDDEN = "USER_OVERRIDDEN"


class ExtractedEquation(StrictModel):
    equation_id: str
    raw_latex: str
    canonical_latex: str
    sympy_expr: str | None = None
    csel_ast: dict[str, object] | None = None
    lhs_symbol: str | None = None
    referenced_symbols: list[str] = Field(default_factory=list)
    role: EquationRole
    dimension_lhs: SIDimensionVectorModel | None = None
    dimension_rhs: SIDimensionVectorModel | None = None
    hin_homogeneous: bool = False
    z3_satisfiable: bool | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED_FLAGGED
    diagnostics: list[str] = Field(default_factory=list)
    provenance: SourceSpan
    supporting_provenance: list[SourceSpan] = Field(default_factory=list)
    disambiguated_by_llm: bool = False


class ExtractedConstant(StrictModel):
    constant_id: str
    symbol: str
    name: str
    raw_value: str
    raw_unit: str = "1"
    si_value: float
    dimension: SIDimensionVectorModel
    bounds: IntervalBoundModel | None = None
    is_tunable_parameter: bool = True
    verification_status: VerificationStatus = VerificationStatus.VERIFIED
    provenance: SourceSpan
    disambiguated_by_llm: bool = False


class StateVariableSpec(StrictModel):
    symbol: str
    slot_index: int = Field(ge=0)
    description: str = ""
    initial_value: float | None = None
    dimension: SIDimensionVectorModel
    bounds: IntervalBoundModel | None = None
    provenance: SourceSpan


class DisambiguationTriggerKind(str, Enum):
    UNRESOLVED_SYMBOL = "UNRESOLVED_SYMBOL"
    PROSE_ONLY_CONSTANT = "PROSE_ONLY_CONSTANT"
    AMBIGUOUS_STATE_ORDERING = "AMBIGUOUS_STATE_ORDERING"
    IMPLICIT_BOUNDARY_CONDITION = "IMPLICIT_BOUNDARY_CONDITION"


class DisambiguationFlag(StrictModel):
    flag_id: str
    trigger: DisambiguationTriggerKind
    target_symbol_or_eq: str
    context_excerpt: str
    source_span: SourceSpan
    resolved: bool = False
    resolution_note: str | None = None


class DisambiguationProposal(StrictModel):
    flag_id: str
    proposal_kind: Literal["constant_binding", "equation_patch", "state_ordering", "boundary_condition"]
    symbol: str | None = None
    raw_value: str | None = None
    raw_unit: str | None = None
    latex_expression: str | None = None
    ordered_state_symbols: list[str] | None = None
    bounds: IntervalBoundModel | None = None
    provenance: SourceSpan


class DisambiguationResponse(StrictModel):
    proposals: list[DisambiguationProposal] = Field(default_factory=list)


class EquationOverridePatch(StrictModel):
    equation_id: str
    canonical_latex: str | None = None
    role: EquationRole | None = None
    drop: bool = False


class ConstantOverridePatch(StrictModel):
    constant_id: str
    si_value: float | None = None
    raw_unit: str | None = None
    bounds: IntervalBoundModel | None = None
    is_tunable_parameter: bool | None = None
    drop: bool = False


class IRReviewOverride(StrictModel):
    equation_patches: list[EquationOverridePatch] = Field(default_factory=list)
    constant_patches: list[ConstantOverridePatch] = Field(default_factory=list)
    state_symbol_order: list[str] | None = None
    time_horizon_si_s: float | None = None
    default_dt_si_s: float | None = None
    default_steps: int | None = None
    approve: bool = True


class RuntimePredicate(StrictModel):
    role: Literal['RUNTIME_PREDICATE'] = 'RUNTIME_PREDICATE'
    contract_id: str
    assertion: str
    tolerance: float = 0.0
    provenance: SourceSpan


class ReferenceInvariant(StrictModel):
    variable_name: str = ''
    assertion: str
    tolerance: float = 0.0


class ReferenceContract(StrictModel):
    contract_id: str
    initial_state: list[float]
    dt: float = Field(gt=0)
    steps: int = Field(ge=0, le=100000)
    invariants: list[ReferenceInvariant] = Field(default_factory=list)


class ResearchReference(StrictModel):
    integration_scheme: Literal['euler','midpoint','heun','rk4','semi_implicit'] = 'rk4'
    state_slots: dict[str, int]
    variable_map: dict[str, int] = Field(default_factory=dict)
    parameter_slots: dict[str, int] = Field(default_factory=dict)
    contracts: list[ReferenceContract]
    rhs_cross_checks: list[dict[str, object]] = Field(default_factory=list)


class CanonicalResearchIR(StrictModel):
    schema_version: Literal["canonical_research_ir.v1"] = "canonical_research_ir.v1"
    ir_id: str
    source_mode: Literal["vault", "upload", "hybrid"]
    vault_run_ids: list[str] = Field(default_factory=list)
    uploaded_files: list[str] = Field(default_factory=list)
    macro_domains: list[str] = Field(default_factory=list)
    title: str
    summary: str = ""
    state_variables: list[StateVariableSpec] = Field(default_factory=list)
    equations: list[ExtractedEquation] = Field(default_factory=list)
    constants: list[ExtractedConstant] = Field(default_factory=list)
    time_horizon_si_s: float | None = None
    default_dt_si_s: float | None = None
    default_steps: int | None = None
    datasets_attached: list[str] = Field(default_factory=list)
    pinned_kernel_code: str | None = None
    runtime_predicates: list[RuntimePredicate] = Field(default_factory=list)
    research_reference: ResearchReference | None = None
    disambiguation_flags: list[DisambiguationFlag] = Field(default_factory=list)
    llm_disambiguation_tokens_used: int = 0
    user_approved: bool = False
    ir_blake3_digest: str = ""


class RawEquation(StrictModel):
    text: str
    provenance: SourceSpan


class RawConstant(StrictModel):
    symbol: str
    value: str
    unit: str = "1"
    provenance: SourceSpan


class ParsedDocument(StrictModel):
    path: str
    text: str
    equations: list[RawEquation] = Field(default_factory=list)
    constants: list[RawConstant] = Field(default_factory=list)
    flags: list[DisambiguationFlag] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    function_signatures: list[str] = Field(default_factory=list)


class IngestionRequest(StrictModel):
    upload_paths: list[str] = Field(default_factory=list)
    vault_run_ids: list[str] = Field(default_factory=list)
    title: str = "Scientific instrument"

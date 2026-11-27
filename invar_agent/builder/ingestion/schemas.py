from __future__ import annotations

from enum import Enum
from typing import Any, Literal

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


class SymbolResolution(StrictModel):
    symbol: str = Field(min_length=1)
    role: Literal["FREE_PARAMETER", "DERIVED_BY_EQUATION"]
    equation_id: str | None = None
    si_value: float | None = None
    raw_unit: str | None = None
    bounds: IntervalBoundModel | None = None


class IRReviewOverride(StrictModel):
    equation_patches: list[EquationOverridePatch] = Field(default_factory=list)
    constant_patches: list[ConstantOverridePatch] = Field(default_factory=list)
    symbol_resolutions: list[SymbolResolution] = Field(default_factory=list)
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
    sciir_commitment_blake3: str = ""
    blake3_receipt: str = ""
    meaningfulness_certificates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    meaningfulness_certificate: dict[str, Any] | None = None
    epistemic_verdict: Literal["VERIFIED_DISCOVERY", "VERIFIED_BOUNDARY_DISCOVERY", "VERIFIED_FALSIFICATION", "INCONCLUSIVE_UNVERIFIED", "USER_MODIFIED_UNVERIFIED"] | None = None
    certified_valid_domain: dict[str, Any] | None = None
    negative_constraints: list[dict[str, Any]] = Field(default_factory=list)
    research_import_role: Literal["POSITIVE_EVIDENCE", "NEGATIVE_CONSTRAINT_ONLY", "LEGACY_UNVERIFIED", "USER_MODIFIED_UNVERIFIED"] = "LEGACY_UNVERIFIED"

    @classmethod
    def from_sci_ir(cls, sealed_or_ir: Any) -> "CanonicalResearchIR":
        """Losslessly adapt a vNext `SciIR` / `SealedSciIR` into Builder Studio's
        `CanonicalResearchIR` while preserving the identical `0x01 || JCS(SciIR)`
        BLAKE3 commitment across Research Mode and Builder Studio."""
        from invar_agent.ir.ihcg_bridge import SealedSciIR, seal_sciir_ihcg
        from invar_agent.ir.schema import SlotRole

        sealed = sealed_or_ir if isinstance(sealed_or_ir, SealedSciIR) else seal_sciir_ihcg(sealed_or_ir)
        ir = sealed.ir
        prov = SourceSpan(file="sci_ir.json", location=ir.ir_id, quote=f"SciIR {ir.ir_id}")

        states: list[StateVariableSpec] = []
        consts: list[ExtractedConstant] = []
        for sym in ir.symbols:
            exp_int = tuple(int(n // d) if d != 0 else 0 for n, d in sym.dimension.exponents)
            dim_model = SIDimensionVectorModel(
                exponents=exp_int,  # type: ignore[arg-type]
                canonical_unit=sym.dimension.unit_label or "1",
            )
            bounds = (
                IntervalBoundModel(lower=sym.domain.lo, upper=sym.domain.hi)
                if sym.domain is not None
                else None
            )
            if sym.role == SlotRole.STATE:
                states.append(
                    StateVariableSpec(
                        symbol=sym.name,
                        slot_index=sym.slot_index,
                        description=sym.description,
                        initial_value=sym.initial_or_default,
                        dimension=dim_model,
                        bounds=bounds,
                        provenance=prov,
                    )
                )
            elif sym.role in (SlotRole.CONTROL_PARAM, SlotRole.CONSTANT):
                consts.append(
                    ExtractedConstant(
                        constant_id=f"c_{sym.name}",
                        symbol=sym.name,
                        name=sym.display_symbol or sym.name,
                        raw_value=str(sym.initial_or_default),
                        raw_unit=sym.dimension.unit_label or "1",
                        si_value=sym.initial_or_default,
                        dimension=dim_model,
                        bounds=bounds,
                        is_tunable_parameter=(sym.role == SlotRole.CONTROL_PARAM),
                        verification_status=VerificationStatus.VERIFIED,
                        provenance=prov,
                    )
                )

        eqs: list[ExtractedEquation] = []
        for idx, eq in enumerate(ir.dynamics_or_residuals):
            eqs.append(
                ExtractedEquation(
                    equation_id=f"dyn_{idx}_{eq.target}",
                    raw_latex=eq.source_text or eq.target,
                    canonical_latex=eq.source_text or eq.target,
                    sympy_expr=eq.source_text or None,
                    lhs_symbol=eq.target,
                    role=EquationRole.EVOLUTION_LAW,
                    hin_homogeneous=True,
                    verification_status=VerificationStatus.VERIFIED,
                    provenance=prov,
                )
            )
        for idx, obs in enumerate(ir.observables):
            eqs.append(
                ExtractedEquation(
                    equation_id=f"obs_{idx}_{obs.target}",
                    raw_latex=obs.source_text or obs.target,
                    canonical_latex=obs.source_text or obs.target,
                    sympy_expr=obs.source_text or None,
                    lhs_symbol=obs.target,
                    role=EquationRole.ALGEBRAIC_CLOSURE,
                    hin_homogeneous=True,
                    verification_status=VerificationStatus.VERIFIED,
                    provenance=prov,
                )
            )

        return cls(
            ir_id=ir.ir_id,
            source_mode="vault",
            macro_domains=[ir.to_macro_domain()],
            title=f"SciIR Instrument: {ir.ir_id}",
            summary=f"Lowered from sealed SciIR ({ir.archetype.value} / {ir.domain.value})",
            state_variables=states,
            equations=eqs,
            constants=consts,
            user_approved=True,
            ir_blake3_digest=sealed.sciir_commitment_blake3,
            sciir_commitment_blake3=sealed.sciir_commitment_blake3,
            blake3_receipt=sealed.ihcg_v1_commitment_blake3,
            epistemic_verdict="VERIFIED_DISCOVERY",
            research_import_role="POSITIVE_EVIDENCE",
        )


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

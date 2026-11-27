"""Invar-Agent vNext Scientific Intermediate Representation (SciIR v2).

Exact Pydantic v2 mirror of `invar_agent/_native/src/sci_ir.rs`.
`CanonicalResearchIR` is an alias for `SciIR` so Research Mode and Builder
Studio share a single canonical representation and identical BLAKE3 commitment.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Archetype(str, Enum):
    ODE_PDE = "ODE_PDE"
    MONTE_CARLO = "MONTE_CARLO"
    FIXED_POINT = "FIXED_POINT"
    PARAM_SWEEP = "PARAM_SWEEP"
    ALGEBRAIC_CLOSURE = "ALGEBRAIC_CLOSURE"


class ScientificDomain(str, Enum):
    PHYSICS_THERMO = "PHYSICS_THERMO"
    BIOCHEM_SYSTEMS = "BIOCHEM_SYSTEMS"
    CLIMATE_EARTH = "CLIMATE_EARTH"
    ECOLOGY_EPIDEM = "ECOLOGY_EPIDEM"
    FLUID_CONTINUUM = "FLUID_CONTINUUM"
    QUANTUM_STATMECH = "QUANTUM_STATMECH"
    ENGINEERING_CTRL = "ENGINEERING_CTRL"
    ECON_COMPLEX = "ECON_COMPLEX"


# Lossless bijection with `invar_agent.research.extractor.MACRO_SCIENTIFIC_DOMAINS`
_VNEXT_TO_MACRO: dict[ScientificDomain, str] = {
    ScientificDomain.PHYSICS_THERMO: "PHYSICAL_SCIENCES_AND_ENG",
    ScientificDomain.BIOCHEM_SYSTEMS: "LIFE_SCIENCES_AND_BIOMEDICINE",
    ScientificDomain.CLIMATE_EARTH: "EARTH_CLIMATE_AND_ENVIRONMENT",
    ScientificDomain.ECOLOGY_EPIDEM: "EPIDEMIOLOGY_AND_ECOLOGY",
    ScientificDomain.FLUID_CONTINUUM: "CHEMICAL_AND_MATERIALS",
    ScientificDomain.QUANTUM_STATMECH: "MATHEMATICS_CS_AND_INFO_THEORY",
    ScientificDomain.ENGINEERING_CTRL: "STATISTICS_AND_CAUSAL_INFERENCE",
    ScientificDomain.ECON_COMPLEX: "ECONOMICS_AND_GAME_THEORY",
}

_MACRO_TO_VNEXT: dict[str, ScientificDomain] = {
    v: k for k, v in _VNEXT_TO_MACRO.items()
}


def to_macro_domain(domain: ScientificDomain | str) -> str:
    """Losslessly map a vNext ScientificDomain to the repository macro-domain."""
    if isinstance(domain, str):
        if domain in _MACRO_TO_VNEXT:
            return domain
        domain = ScientificDomain(domain)
    return _VNEXT_TO_MACRO[domain]


def from_macro_domain(macro: str | ScientificDomain) -> ScientificDomain:
    """Losslessly map a repository macro-domain to a vNext ScientificDomain."""
    if isinstance(macro, ScientificDomain):
        return macro
    if macro in _MACRO_TO_VNEXT:
        return _MACRO_TO_VNEXT[macro]
    try:
        return ScientificDomain(macro)
    except ValueError:
        return ScientificDomain.PHYSICS_THERMO


class SlotRole(str, Enum):
    STATE = "STATE"
    CONTROL_PARAM = "CONTROL_PARAM"
    CONSTANT = "CONSTANT"
    DERIVED_OBSERVABLE = "DERIVED_OBSERVABLE"
    INDEPENDENT_VAR = "INDEPENDENT_VAR"


class SiDimension(BaseModel):
    """7-component rational SI dimension vector [M, L, T, I, Theta, N, J] in Q^7."""

    model_config = ConfigDict(extra="forbid")

    exponents: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ] = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
    unit_label: str = "1"

    @field_validator("exponents")
    @classmethod
    def _nonzero_den(
        cls,
        v: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        if any(den <= 0 for _, den in v):
            raise ValueError("SiDimension denominators must be positive integers")
        return v

    @classmethod
    def dimensionless(cls, unit_label: str = "1") -> "SiDimension":
        return cls(unit_label=unit_label)


class IntervalBound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lo: float
    hi: float

    @model_validator(mode="after")
    def _validate_finite_and_ordered(self) -> "IntervalBound":
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise ValueError("IntervalBound lo/hi must be finite IEEE-754 floats")
        if self.lo > self.hi:
            raise ValueError(f"IntervalBound lo ({self.lo}) > hi ({self.hi})")
        return self


class SymbolDecl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=64)
    display_symbol: str = ""
    role: SlotRole
    slot_index: int = Field(default=0, ge=0)
    dimension: SiDimension = Field(default_factory=SiDimension.dimensionless)
    initial_or_default: float = 0.0
    domain: Optional[IntervalBound] = None
    discrete_values: Optional[list[float]] = None
    strictly_positive: bool = False
    pool_id: Optional[str] = None
    description: str = ""

    @model_validator(mode="after")
    def _validate_fields(self) -> "SymbolDecl":
        if not self.display_symbol:
            self.display_symbol = self.name
        if not math.isfinite(self.initial_or_default):
            raise ValueError(f"Non-finite initial_or_default for {self.name}")
        if self.discrete_values is not None:
            if not self.discrete_values or any(
                not math.isfinite(v) for v in self.discrete_values
            ):
                raise ValueError(f"Invalid discrete_values for {self.name}")
        return self


class ConservationPoolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_id: str
    member_states: list[str] = Field(..., min_length=1)
    stoichiometry: list[float] = Field(default_factory=list)
    expected_total: float = 1.0


class AssumptionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    source: str
    falsifiable_impact: str = ""


# Discriminated `ExprNode` union matching `invar_agent/_native/src/sci_ir.rs`
class ExprConst(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["const"] = "const"
    value: float

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("ExprConst value must be finite")
        return v


class ExprSym(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["sym"] = "sym"
    name: str
    role: SlotRole
    slot_index: int = 0


class ExprUnary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["unary"] = "unary"
    op: Literal["neg", "abs", "exp", "log", "sqrt", "sin", "cos", "tanh", "step"]
    arg: "ExprNode"


class ExprBinary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["binary"] = "binary"
    op: Literal["add", "sub", "mul", "div", "pow", "min", "max"]
    lhs: "ExprNode"
    rhs: "ExprNode"


class ExprIfThenElse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["if_then_else"] = "if_then_else"
    cond_op: Literal["lt", "le", "gt", "ge", "eq"]
    lhs: "ExprNode"
    rhs: "ExprNode"
    then_branch: "ExprNode"
    else_branch: "ExprNode"


class ExprRand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["rand"] = "rand"
    dist: Literal["uniform01", "normal01", "exponential1"]
    draw_stream_id: int = 0


ExprNode = Annotated[
    Union[ExprConst, ExprSym, ExprUnary, ExprBinary, ExprIfThenElse, ExprRand],
    Field(discriminator="kind"),
]

ExprUnary.model_rebuild()
ExprBinary.model_rebuild()
ExprIfThenElse.model_rebuild()


class LetBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dimension: SiDimension = Field(default_factory=SiDimension.dimensionless)
    expr: ExprNode
    source_text: str = ""


class EquationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    target_slot: int = 0
    expr: ExprNode
    source_text: str = ""
    reduction: Optional[Literal["TRAJECTORY_MAX", "TRAJECTORY_MIN", "TRAJECTORY_FINAL"]] = None


class ObligationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    quantifier: str = "FORALL"
    lhs_expr: ExprNode
    comparator: Literal["<=", "<", ">=", ">", "=="]
    rhs_expr: ExprNode
    tolerance: float = 1e-6
    epistemic_origin: str = "USER_PROMPT"
    source_text: str = ""

    @field_validator("tolerance")
    @classmethod
    def _valid_tol(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("Obligation tolerance must be finite and > 0")
        return v


# Discriminated `ArchetypeConfig` matching `sci_ir.rs`
class OdePdeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["ode_pde"] = "ode_pde"
    t_span: tuple[float, float] = (0.0, 100.0)
    sample_points: int = 200
    rtol: float = 1e-7
    atol: float = 1e-10
    preferred_solver: str = "AUTO"
    enable_buckingham_scaling: bool = True


class MonteCarloConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["monte_carlo"] = "monte_carlo"
    base_seed: int = 42
    replicates: int = 4096
    confidence_alpha: float = 0.05
    use_crn: bool = True
    observable_range_bounds: dict[str, IntervalBound] = Field(default_factory=dict)

    @field_validator("base_seed")
    @classmethod
    def _safe_seed(cls, v: int) -> int:
        if v < 0 or v > 9_007_199_254_740_991:
            raise ValueError("MonteCarlo base_seed must be in [0, 2^53 - 1]")
        return v


class FixedPointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fixed_point"] = "fixed_point"
    max_newton_iters: int = 100
    residual_tol: float = 1e-10
    require_krawczyk_proof: bool = True
    require_stability_cert: bool = True


class ParamSweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["param_sweep"] = "param_sweep"
    grid_strategy: str = "uniform"
    points_per_axis: int = 25
    bisection_tol: float = 1e-4


class AlgebraicClosureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["algebraic_closure"] = "algebraic_closure"
    shbt_federation_enabled: bool = True
    open_boundary_vars: list[str] = Field(default_factory=list)


ArchetypeConfig = Annotated[
    Union[
        OdePdeConfig,
        MonteCarloConfig,
        FixedPointConfig,
        ParamSweepConfig,
        AlgebraicClosureConfig,
    ],
    Field(discriminator="kind"),
]


class BuilderStudioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    target_languages: list[str] = Field(default_factory=lambda: ["python"])
    export_workbench_ui: bool = False
    artifact_manifest_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# Pass-1 / Pass-2 / Pass-3 Staged Synthesis Draft Payloads
# ---------------------------------------------------------------------------

class FrozenSymbolTable(BaseModel):
    """Pass-1 output frozen prior to Pass-2 equation synthesis."""

    model_config = ConfigDict(extra="forbid")

    rationale: str = ""
    ir_id: str
    archetype: Archetype
    domain: ScientificDomain
    symbols: list[SymbolDecl] = Field(default_factory=list)
    conservation_pools: list[ConservationPoolSpec] = Field(default_factory=list)
    assumptions_ledger: list[AssumptionEntry] = Field(default_factory=list)

    def assign_canonical_slot_indices(self) -> "FrozenSymbolTable":
        """Deterministically assign per-role contiguous slot indices."""
        counters: dict[SlotRole, int] = {r: 0 for r in SlotRole}
        updated: list[SymbolDecl] = []
        for sym in self.symbols:
            idx = counters[sym.role]
            counters[sym.role] = idx + 1
            updated.append(sym.model_copy(update={"slot_index": idx}))
        return self.model_copy(update={"symbols": updated})


class RawLetDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    expr_text: str
    dimension: Optional[SiDimension] = None


class RawEquationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    expr_text: str
    reduction: Optional[Literal["TRAJECTORY_MAX", "TRAJECTORY_MIN", "TRAJECTORY_FINAL"]] = None


class RawObligationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    quantifier: str = "FORALL"
    lhs_text: str
    comparator: Literal["<=", "<", ">=", ">", "=="]
    rhs_text: str
    tolerance: float = 1e-6
    epistemic_origin: str = "USER_PROMPT"
    source_text: str = ""


class ModelBodyDraft(BaseModel):
    """Pass-2 raw-expression output synthesized against a FrozenSymbolTable."""

    model_config = ConfigDict(extra="forbid")

    rationale: str = ""
    let_bindings: list[RawLetDraft] = Field(default_factory=list)
    dynamics_or_residuals: list[RawEquationDraft] = Field(default_factory=list)
    observables: list[RawEquationDraft] = Field(default_factory=list)
    obligations: list[RawObligationDraft] = Field(default_factory=list)
    config: ArchetypeConfig = Field(default_factory=OdePdeConfig)


class SingleSymbolRepairPayload(BaseModel):
    """Pass-3 bounded micro-repair payload (single equation or let-binding only)."""

    model_config = ConfigDict(extra="forbid")

    rationale: str = ""
    target_kind: Literal["let_binding", "dynamics_or_residuals", "observables"]
    target_name: str
    expr_text: str

    @field_validator("target_kind", mode="before")
    @classmethod
    def _normalize_target_kind(cls, v: Any) -> Any:
        if isinstance(v, str):
            low = v.strip().lower()
            aliases = {
                "observable": "observables",
                "observables": "observables",
                "let": "let_binding",
                "let_binding": "let_binding",
                "let_bindings": "let_binding",
                "dynamic": "dynamics_or_residuals",
                "dynamics": "dynamics_or_residuals",
                "residual": "dynamics_or_residuals",
                "residuals": "dynamics_or_residuals",
                "equation": "dynamics_or_residuals",
                "equations": "dynamics_or_residuals",
                "dynamics_or_residuals": "dynamics_or_residuals",
            }
            return aliases.get(low, v)
        return v


class SciIR(BaseModel):
    """Canonical Invar-Agent vNext Scientific IR (`invar.sciir.v2`)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["invar.sciir.v2"] = "invar.sciir.v2"
    ir_id: str
    archetype: Archetype
    domain: ScientificDomain
    user_prompt_digest: str
    assumptions_ledger: list[AssumptionEntry] = Field(default_factory=list)
    symbols: list[SymbolDecl] = Field(default_factory=list)
    conservation_pools: list[ConservationPoolSpec] = Field(default_factory=list)
    let_bindings: list[LetBinding] = Field(default_factory=list)
    dynamics_or_residuals: list[EquationSpec] = Field(default_factory=list)
    observables: list[EquationSpec] = Field(default_factory=list)
    obligations: list[ObligationSpec] = Field(default_factory=list)
    config: ArchetypeConfig = Field(default_factory=OdePdeConfig)
    builder_studio: BuilderStudioSpec = Field(default_factory=BuilderStudioSpec)
    ihcg_commitment_blake3: Optional[str] = None

    def to_macro_domain(self) -> str:
        return to_macro_domain(self.domain)

    def to_rust_json(self) -> str:
        """Serialize to compact JSON matching `invar_agent/_native/src/sci_ir.rs`."""
        return self.model_dump_json()


# Unified IR alias shared by Research Mode and Builder Studio
CanonicalResearchIR = SciIR

"""Pydantic v2 schemas for the Research Agent verification contracts."""

from __future__ import annotations

import math
import re
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _clean_text(value: Any, fallback: str = "") -> str:
    """Collapse whitespace and trim; never raise on odd LLM output."""
    if value is None:
        return fallback
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.strip().split())


def _slugify(value: Any, fallback: str = "unnamed", max_length: int = 64) -> str:
    """Slugify an identifier: lowercase, safe chars only, length-capped."""
    if not value or not isinstance(value, str):
        return fallback
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return (slug[:max_length] if slug else fallback)


class InvariantConstraint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variable_name: str
    assertion: str
    tolerance: float = Field(default=1e-6, ge=0.0)


class VerificationContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str

    @field_validator("contract_id", mode="before")
    @classmethod
    def _sanitize_contract_id(cls, value: Any) -> str:
        return _slugify(value, fallback="contract_unnamed")
    target_backend: Literal["rust_hin", "cpp", "cuda", "multi_variant"]
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    invariants: List[InvariantConstraint] = Field(min_length=1)
    max_memory_mb: int = Field(default=1024, le=8192)
    # Simulation parameters consumed by Sandbox_Execute.
    initial_state: Optional[List[float]] = Field(default=None, max_length=4096)
    initial_values: Optional[Dict[str, float]] = Field(default=None, max_length=4096)
    dt: float = Field(default=0.01, gt=0.0, le=1e8)
    steps: int = Field(default=1000, ge=1, le=100000)


class ModuleLayer(BaseModel):
    """A pinned, verified checkpoint layer of the unified research module."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_id: str
    stage: int = Field(ge=1)
    code: str
    bundle_hash: str
    contracts: List[VerificationContract] = Field(default_factory=list)
    baseline_invariants: List[InvariantConstraint] = Field(default_factory=list)
    telemetry: Dict[str, float] = Field(default_factory=dict)


class StageTelemetry(BaseModel):
    """Per-stage outcome recorded in the blueprint evolution history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: int = Field(ge=1)
    status: Literal["VERIFIED", "FALSIFIED", "DIVERGED", "ERROR", "REFUSED"]
    bundle_hash: str = ""
    telemetry: Dict[str, float] = Field(default_factory=dict)
    note: str = ""
    timestamp: float = 0.0


class FeasibilityVerdict(str, Enum):
    OPTIMAL_DIRECT = "OPTIMAL_DIRECT"
    REFORMULATED_SURROGATE = "REFORMULATED_SURROGATE"
    REFUSED_IMPOSSIBLE = "REFUSED_IMPOSSIBLE"
    REFUSED_UNSUITABLE = "REFUSED_UNSUITABLE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"

    @classmethod
    def _missing_(cls, value):
        # Case-insensitive coercion: LLM JSON often lowercases verdicts.
        for member in cls:
            if member.value.lower() == str(value).lower():
                return member
        return None


class FeasibilityAssessment(BaseModel):
    """Epistemic self-awareness verdict produced before any simulation runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: FeasibilityVerdict
    rationale: str
    violated_boundary: Optional[str] = None
    reformulation: Optional[str] = None
    questions: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    smt_unsat_core: Optional[List[str]] = None
    smt_unsat_core_items: List[Dict[str, Any]] = Field(default_factory=list)
    advisories: List[str] = Field(default_factory=list)
    assessed_by: Literal["heuristic", "heuristic_offline", "llm", "deterministic_smt"] = "heuristic_offline"

    @field_validator("verdict", mode="before")
    @classmethod
    def _coerce_verdict_case(cls, value):
        # LLM JSON often lowercases/mixed-cases verdicts; Enum._missing_ alone
        # is not consulted by pydantic's str-enum path, so normalise here.
        if isinstance(value, str):
            upper = value.upper()
            for member in FeasibilityVerdict:
                if member.value == upper:
                    return member
        return value

    @model_validator(mode="before")
    @classmethod
    def _feasible_verdicts_carry_no_refusal_fields(cls, data):
        # A passing verdict must never look like a refusal downstream:
        # OPTIMAL_DIRECT has no violated boundary and no reformulation.
        # REFORMULATED_SURROGATE keeps its reformulation (that IS the explicit
        # surrogate reduction) but has no violated boundary.
        if not isinstance(data, dict):
            return data
        verdict = data.get("verdict")
        if isinstance(verdict, FeasibilityVerdict):
            verdict = verdict.value
        verdict = str(verdict or "").upper()
        if verdict == FeasibilityVerdict.OPTIMAL_DIRECT.value:
            data = dict(data)
            data["violated_boundary"] = None
            data["reformulation"] = None
        elif verdict == FeasibilityVerdict.REFORMULATED_SURROGATE.value \
                and not str(data.get("reformulation") or "").strip():
            data = dict(data)
            data["reformulation"] = None
        return data


AssumptionKind = Literal[
    "GOVERNING_EQUATION",
    "SI_CONSTANT",
    "OPERATING_DOMAIN",
    "INITIAL_CONDITION",
    "HYPOTHESIS_FORMALIZATION",
    "NULL_BASELINE",
    "NOTE",
]

AssumptionSource = Literal[
    "USER_PROMPT",
    "LITERATURE",
    "REFERENCE_REGISTRY",
    "SHBT_REGISTRY",
    "LLM_DOMAIN_DEFAULT",
]


_VALID_ASSUMPTION_KINDS = {
    "GOVERNING_EQUATION",
    "SI_CONSTANT",
    "OPERATING_DOMAIN",
    "INITIAL_CONDITION",
    "HYPOTHESIS_FORMALIZATION",
    "NULL_BASELINE",
    "NOTE",
}
_VALID_ASSUMPTION_SOURCES = {
    "USER_PROMPT",
    "LITERATURE",
    "REFERENCE_REGISTRY",
    "SHBT_REGISTRY",
    "LLM_DOMAIN_DEFAULT",
}


def coerce_assumption_entry(raw: Any) -> Optional[Dict[str, str]]:
    """Deterministically normalize LLM-emitted assumption records into canonical FormalizationAssumption fields."""
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return {
            "kind": "GOVERNING_EQUATION",
            "symbol_or_field": "model",
            "value_or_expr": text[:400],
            "source": "LLM_DOMAIN_DEFAULT",
            "rationale": "",
        }
    if not isinstance(raw, dict):
        return None
    raw_kind = str(raw.get("kind") or raw.get("category") or raw.get("type") or "").strip().upper()
    if raw_kind not in _VALID_ASSUMPTION_KINDS:
        if any(k in raw for k in ("equation", "formula", "ode")):
            raw_kind = "GOVERNING_EQUATION"
        elif any(k in raw for k in ("bounds", "domain", "range")):
            raw_kind = "OPERATING_DOMAIN"
        elif any(k in raw for k in ("initial", "initial_value", "initial_condition")):
            raw_kind = "INITIAL_CONDITION"
        elif any(k in raw for k in ("null", "baseline")):
            raw_kind = "NULL_BASELINE"
        elif any(k in raw for k in ("value", "constant", "unit", "si_unit", "parameter")):
            raw_kind = "SI_CONSTANT"
        else:
            raw_kind = "HYPOTHESIS_FORMALIZATION"
    sym = ""
    for k in ("symbol_or_field", "symbol", "parameter", "observable", "field", "name", "variable", "id"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            sym = v.strip()
            break
    if not sym:
        sym = "model"
    val = ""
    for k in ("value_or_expr", "equation", "expression", "expr", "formula", "bounds", "domain"):
        v = raw.get(k)
        if v is not None and str(v).strip():
            val = str(v).strip()
            break
    if not val and raw.get("value") is not None:
        unit = str(raw.get("si_unit") or raw.get("unit") or "").strip()
        val = f"{raw['value']} {unit}".strip()
    if not val:
        for k in ("statement", "description", "rationale", "text", "note"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                val = v.strip()[:400]
                break
    if not val:
        return None
    raw_src = str(raw.get("source") or raw.get("provenance") or "").strip().upper()
    if raw_src not in _VALID_ASSUMPTION_SOURCES:
        if "USER" in raw_src or "PROMPT" in raw_src:
            raw_src = "USER_PROMPT"
        elif "SHBT" in raw_src:
            raw_src = "SHBT_REGISTRY"
        elif "REF" in raw_src or "CODATA" in raw_src or "REGISTRY" in raw_src:
            raw_src = "REFERENCE_REGISTRY"
        elif "LIT" in raw_src or "PAPER" in raw_src or "ARXIV" in raw_src:
            raw_src = "LITERATURE"
        else:
            raw_src = "LLM_DOMAIN_DEFAULT"
    rationale = ""
    for k in ("rationale", "statement", "description", "justification", "notes", "note"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            rationale = v.strip()[:600]
            break
    return {
        "kind": raw_kind,
        "symbol_or_field": sym[:120],
        "value_or_expr": val[:400],
        "source": raw_src,
        "rationale": rationale,
    }


class FormalizationAssumption(BaseModel):
    """Auditable record of an equation, constant, domain, or baseline introduced during NL->AST translation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: AssumptionKind
    symbol_or_field: str = Field(min_length=1)
    value_or_expr: str = Field(min_length=1)
    source: AssumptionSource = "LLM_DOMAIN_DEFAULT"
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_shape(cls, value: Any) -> Any:
        if isinstance(value, dict) and set(value.keys()) <= {"kind", "symbol_or_field", "value_or_expr", "source", "rationale"}:
            return value
        coerced = coerce_assumption_entry(value)
        return coerced if coerced is not None else value


def _validate_rational_si_dim(value: Any) -> List[Union[int, float]]:
    """R2-ARCH-006: Accept 7 integer or rational quarter/half-integer SI exponents
    (e.g. -0.5 for diffusive/volatility sqrt(time) scaling). Preserves int for whole numbers."""
    if value is None:
        return [0, 0, 0, 0, 0, 0, 0]
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError(
            "SI dimensions must be seven integer or half/quarter-integer exponents "
            "(e.g. [0, 0, -0.5, 0, 0, 0, 0] for diffusive/volatility 1/sqrt(s) scaling)."
        )
    out: List[Union[int, float]] = []
    for v in value:
        if isinstance(v, bool):
            raise ValueError("SI dimension exponents cannot be booleans.")
        if isinstance(v, int):
            out.append(v)
            continue
        if isinstance(v, Fraction):
            fval = float(v)
        elif isinstance(v, float):
            if not math.isfinite(v):
                raise ValueError("SI dimension exponents must be finite.")
            fval = v
        else:
            raise ValueError(
                f"Invalid SI dimension exponent {v!r}; must be an integer or multiple of 0.25 "
                "(e.g. -0.5 for diffusive/volatility sqrt(time) scaling)."
            )
        scaled = fval * 4.0
        if abs(scaled - round(scaled)) > 1e-9:
            raise ValueError(
                f"SI dimension exponent {v!r} must be an integer or multiple of 0.25 "
                "(declare e.g. -0.5 for a diffusive/volatility-like 1/sqrt(time) scaling)."
            )
        q = int(round(scaled))
        out.append(q // 4 if q % 4 == 0 else q / 4.0)
    return out


class TypedConstant(BaseModel):
    """Immutable typed SI physical or model constant (never required to occupy a mutable simulation slot)."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    si_unit: str = ""
    si_dim: List[Union[int, float]] = Field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0], min_length=7, max_length=7)
    source: AssumptionSource = "LLM_DOMAIN_DEFAULT"

    @field_validator("si_dim", mode="before")
    @classmethod
    def _int_dims(cls, value):
        return _validate_rational_si_dim(value)


class DerivedExpressionSpec(BaseModel):
    """Pure algebraic expression over typed_slots and constants (never required to occupy a mutable slot unless promoted)."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    si_unit: str = ""
    si_dim: List[Union[int, float]] = Field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0], min_length=7, max_length=7)
    promote_to_slot: bool = False

    @field_validator("si_dim", mode="before")
    @classmethod
    def _int_dims(cls, value):
        return _validate_rational_si_dim(value)


class TypedResearchSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    slot_idx: Optional[int] = Field(default=None, ge=0, strict=True)
    si_dim: List[Union[int, float]] = Field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0], min_length=7, max_length=7)
    role: Literal["CONTROL_PARAM", "DYNAMIC_STATE", "DERIVED_OBSERVABLE", "CONSERVATION_AUX", "TIME_STEP"]
    semantic_tag: int = Field(default=0, ge=0, le=255)
    unit_scale: tuple[int, int] = (1, 1)
    is_falsification_target: bool = False
    stochastic_role: Optional[Literal["UNIFORM", "SEED", "COUNTER", "CHANNEL"]] = None
    value: Optional[float] = Field(default=None, allow_inf_nan=False)
    prompt_symbol: Optional[str] = None
    si_unit: Optional[str] = None
    initial_value: Optional[float] = Field(default=None, allow_inf_nan=False)
    bounds: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)

    @field_validator("si_dim", mode="before")
    @classmethod
    def integer_dimensions(cls, value):
        return _validate_rational_si_dim(value)


class FormalResearchSpec(BaseModel):
    """Four-namespace Blueprint AST emitted by the LLM Interpreter and compiled by Tier-0 before IHCG sealing."""

    model_config = ConfigDict(extra="forbid")

    typed_slots: List[TypedResearchSlot] = Field(default_factory=list, max_length=4096)
    constants: Dict[str, TypedConstant] = Field(default_factory=dict, max_length=512)
    derived_expressions: Dict[str, DerivedExpressionSpec] = Field(default_factory=dict, max_length=512)
    equations: Dict[str, Any] = Field(default_factory=dict)
    sweep_domain: Dict[str, List[float]] = Field(default_factory=dict)
    hypothesis_obligations: List[Dict[str, Any]] = Field(default_factory=list)
    environmental_assumptions: List[str] = Field(default_factory=list)
    null_baseline: Optional[str] = None
    quantifier_mode: Literal["FORALL", "EXISTS", "BOUNDARY"] = "FORALL"
    archetype: Literal["ODE_PDE", "MONTE_CARLO", "FIXED_POINT", "PARAM_SWEEP"] = "ODE_PDE"
    mechanism_parameters: List[str] = Field(default_factory=list)
    assumptions_ledger: List[FormalizationAssumption] = Field(default_factory=list)


class ResearchBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blueprint_id: str

    @field_validator("blueprint_id", mode="before")
    @classmethod
    def _sanitize_blueprint_id(cls, value: Any) -> str:
        return _slugify(value, fallback="blueprint_unnamed")

    title: str = Field(min_length=5, max_length=256)

    @field_validator("title", mode="before")
    @classmethod
    def _normalize_title(cls, value: Any) -> str:
        if isinstance(value, str):
            cleaned = _clean_text(value)
            if len(cleaned) > 120:
                return cleaned[:117] + "..."
            return cleaned
        return _clean_text(value) if value is not None else "Untitled Research Blueprint"
    falsifiable_hypothesis: str
    research_prompt: str = ""
    search_queries: List[str]
    contracts: List[VerificationContract] = Field(min_length=1)
    code_template: str
    max_feedback_rounds: int = Field(default=3, ge=1, le=10)
    # -- cumulative accretion state (populated by the state machine) --------
    verified_layers: List[ModuleLayer] = Field(default_factory=list)
    target_research_stages: int = Field(default=1, ge=1, le=10)
    stage_objectives: List[str] = Field(default_factory=list, max_length=10)
    current_stage: int = Field(default=1, ge=1, le=10)
    evolution_history: List[StageTelemetry] = Field(default_factory=list)
    unified_module_code: Optional[str] = None
    feasibility: Optional[FeasibilityAssessment] = None
    # Named observables: human-readable metric (with units, e.g.
    # "magnetic_field_B_tesla") -> index into the final buf.data state vector.
    # Sandbox_Execute publishes each mapping as telemetry["obs_<name>"].
    variable_map: Dict[str, int] = Field(default_factory=dict, max_length=4096)
    typed_slots: List[TypedResearchSlot] = Field(default_factory=list, max_length=4096)
    slot_roles: Dict[str, Literal['read_only_parameter', 'evolving_state', 'diagnostic']] = Field(default_factory=dict, max_length=4096)
    # Zero-LLM deterministic extraction of quantitative literature evidence
    # (SI-normalized constants, LaTeX equations, benchmarks) attached by the
    # planner before blueprint synthesis and persisted into the vault.
    structured_evidence: Optional[StructuredResearchEvidence] = None
    # SI-normalized parameter dictionary derived from structured_evidence
    # (name -> si_value); consumed by the vault and the Builder context.
    extracted_constants: Dict[str, float] = Field(default_factory=dict, max_length=128)
    # Declarative ResearchComputeUAST specification (when synthesized via UAST/HIN pipeline).
    uast_spec: Optional[Dict[str, Any]] = None
    ihcg_evidence: Optional[Dict[str, Any]] = None
    # Four-namespace Blueprint AST extensions + transparent formalization assumptions ledger.
    constants: Dict[str, Any] = Field(default_factory=dict, max_length=512)
    derived_expressions: Dict[str, Any] = Field(default_factory=dict, max_length=512)
    assumptions_ledger: List[FormalizationAssumption] = Field(default_factory=list, max_length=512)
    formal_spec: Optional[FormalResearchSpec] = None


class ExtractedParameter(BaseModel):
    """A literature-derived physical/empirical constant, SI-normalized."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    symbol: str = ""
    raw_value: float
    raw_unit: str = ""
    si_value: float
    si_unit: str = ""
    uncertainty: Optional[float] = None
    source: str = ""
    confidence: float = 1.0


class ExtractedEquation(BaseModel):
    """A quantitative equation extracted verbatim from a literature source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lhs: str
    rhs_latex: str
    rhs_python_hint: str = ""
    variables: List[str] = Field(default_factory=list)
    description: str = ""
    source: str = ""


class BenchmarkAnchor(BaseModel):
    """An external quantitative benchmark to validate the simulation against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    input_regime: str = ""
    expected_value: Any = None  # float or list[float]
    unit: str = ""
    tolerance: float = 0.05
    source: str = ""


class StructuredResearchEvidence(BaseModel):
    """Deterministic, zero-LLM extraction of quantitative literature knowledge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    domains: List[str] = Field(default_factory=list)
    equations: List[ExtractedEquation] = Field(default_factory=list)
    parameters: Dict[str, ExtractedParameter] = Field(default_factory=dict)
    benchmarks: List[BenchmarkAnchor] = Field(default_factory=list)
    recommended_backend: Literal["rust_hin", "cpp", "multi_variant"] = "cpp"
    recommended_stepper: str = "rk4"
    suggested_invariants: List[str] = Field(default_factory=list)
    summary_markdown_table: str = ""


class SimulationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    blueprint_id: str
    contract_id: str
    status: Literal["VERIFIED", "FALSIFIED", "DIVERGED", "ERROR"]
    execution_time_ms: float
    telemetry: Dict[str, float]
    bundle_hash: str
    failure_counterexample: Optional[str] = None
    stage: Optional[int] = None


# ResearchBlueprint.structured_evidence is a forward reference resolved here
# (evidence models are defined above SimulationResult).
ResearchBlueprint.model_rebuild()

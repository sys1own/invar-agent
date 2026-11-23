"""Bounded, fail-closed scientific meaningfulness substrate.

The trusted boundary includes the formal evidence/prose binding, executable
runner, Z3, native rewrite rules, and native IEEE/libm interval extensions.
Native monotone libm enclosures retain an empirical 16-ULP assumption;
binomial intervals trust SciPy's beta quantile implementation.
GCI and variance consistency are reliability diagnostics, not analytic global
error proofs. No UNKNOWN result is promoted to a proof or a counterexample.
"""

from __future__ import annotations

import ast
import copy
import hmac
import json
import math
import os
import statistics
import sys
import tempfile
import time
from contextvars import ContextVar
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import z3
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator
from scipy.stats import beta

from invar_agent import _native
from invar_agent.algorithms.prng import invar_rand_u01, invar_splitmix64_stateless


TRUST_BOUNDARY_VERSION = "invar.tcb.phase2a.native-libm16.v1"
SMT_TIMEOUT_MS = 1500
MAX_BOXES = 256
MAX_PROBES = 512
MAX_VARIABLES = 32
_QUERY_DEADLINE = ContextVar("meaningfulness_query_deadline", default=math.inf)
Tier = Literal["QF_LRA", "QF_NRA", "MCCORMICK_QF_LRA", "WITNESS_PROBE", "UNKNOWN"]
Archetype = Literal["ODE_PDE", "MONTE_CARLO", "FIXED_POINT", "PARAM_SWEEP"]
Classification = Literal["VACUOUS", "TAUTOLOGICAL", "CONSTRAINING", "INFEASIBLE", "INFEASIBLE_UNDER_MODEL", "UNKNOWN"]
Bounds = dict[str, tuple[float, float]]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class EpistemicVerdict(str, Enum):
    VERIFIED_DISCOVERY = "VERIFIED_DISCOVERY"
    VERIFIED_BOUNDARY_DISCOVERY = "VERIFIED_BOUNDARY_DISCOVERY"
    VERIFIED_FALSIFICATION = "VERIFIED_FALSIFICATION"
    INCONCLUSIVE_UNVERIFIED = "INCONCLUSIVE_UNVERIFIED"


class ContractClass(str, Enum):
    VACUOUS = "VACUOUS"
    TAUTOLOGICAL = "TAUTOLOGICAL"
    CONSTRAINING = "CONSTRAINING"
    INFEASIBLE = "INFEASIBLE"
    INFEASIBLE_UNDER_MODEL = "INFEASIBLE_UNDER_MODEL"
    UNKNOWN = "UNKNOWN"


class SlotSpec(StrictModel):
    slot_idx: int = Field(ge=0)
    symbol: str = Field(min_length=1)
    si_dim: tuple[int, int, int, int, int, int, int]
    semantic_tag: int = Field(default=0, ge=0, le=255)
    role: Literal["CONTROL_PARAM", "DYNAMIC_STATE", "DERIVED_OBSERVABLE", "CONSERVATION_AUX"]
    is_falsification_target: bool = False
    stochastic_role: Literal["UNIFORM", "SEED", "COUNTER", "CHANNEL"] | None = None
    unit_scale: tuple[int,int] = (1,1)

    @field_validator("unit_scale")
    @classmethod
    def positive_scale(cls,value):
        if min(value) <= 0: raise ValueError("coordinate unit scales must be positive rationals")
        return value

    @model_serializer(mode="wrap")
    def serialize_slot(self, handler):
        data = handler(self)
        if self.stochastic_role is None: data.pop("stochastic_role",None)
        if self.unit_scale == (1,1): data.pop("unit_scale",None)
        return data


class Obligation(StrictModel):
    id: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    obligation_class: Literal["HYPOTHESIS_TARGET", "CONSERVATION_GUARD", "STAGE_COUPLING"]
    prompt_quote: str = Field(min_length=1)
    prompt_span: tuple[int, int]


class EquationSpec(StrictModel):
    expression: str
    si_dim: tuple[int, int, int, int, int, int, int]
    semantic_tag: int = Field(default=0, ge=0, le=255)
    kind: Literal["ALGEBRAIC", "ODE_RHS", "FIXED_POINT_RESIDUAL", "FIXED_POINT_MAP", "MC_SAMPLE", "TRAJECTORY_MAX", "TRAJECTORY_MIN"] = "ALGEBRAIC"
    state_symbol: str | None = None

    @model_serializer(mode="wrap")
    def serialize_equation(self, handler):
        data = handler(self)
        if self.kind == "ALGEBRAIC" and self.state_symbol is None:
            data.pop("kind",None); data.pop("state_symbol",None)
        return data


class IhcgGraph(StrictModel):
    slots: tuple[SlotSpec, ...]
    equations: dict[str, EquationSpec]
    environmental_assumptions: tuple[str, ...]
    obligations: tuple[Obligation, ...]
    omega: Bounds
    quantifier_mode: Literal["FORALL", "EXISTS", "BOUNDARY"]
    archetype: Archetype
    mechanism_parameters: tuple[str, ...] = ()
    control_values: dict[str, float] = Field(default_factory=dict)
    parent_ihcg_digest: str | None = None
    stage_chain_digests: tuple[str, ...] = ()
    perturbation_target: bool = False
    mc_difference_bounds: tuple[float, float] | None = None
    h_ihcg: str = ""

    def assert_sealed(self) -> None:
        if not hmac.compare_digest(self.h_ihcg, _native.py_compute_ihcg_digest(self.model_dump_json())):
            raise ValueError("IHCG_SEAL_MISMATCH")


class QueryResult(StrictModel):
    status: Literal["SAT", "UNSAT", "UNKNOWN"]
    z3_tier_used: Tier
    witness: dict[str, float] | None = None
    exact_model: str | None = None
    boxes_evaluated: int = 0
    probes_evaluated: int = 0
    reason: str | None = None


class ContractVacuityAudit(StrictModel):
    classification: Classification
    queries: dict[str, QueryResult]
    redundant_conjuncts: tuple[str, ...] = ()
    z3_tier_used: Tier

    @property
    def passed(self) -> bool:
        return self.classification == "CONSTRAINING" and not self.redundant_conjuncts


class EntailmentAudit(StrictModel):
    entailment_locked: bool
    queries: dict[str, QueryResult]
    z3_tier_used: Tier
    counterexample_if_weakened: dict[str, float] | None = None


class ContractAudit(StrictModel):
    entailment_locked: bool
    classification: Classification
    z3_logic_tier: Tier
    redundant_conjuncts: tuple[str, ...] = ()
    counterexample_if_weakened: dict[str, float] | None = None


class NontrivialityAudit(StrictModel):
    passed: bool
    claim: tuple[str, ...]
    contract_audit: ContractVacuityAudit
    failure_reason: str | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.passed != self.contract_audit.passed:
            raise ValueError("nontriviality must agree with Q1-Q5")
        return self


class PurityAudit(StrictModel):
    static_purity_passed: bool
    violations: tuple[str, ...] = ()
    runtime_stabilizer_hits: int = Field(default=0, ge=0)
    runtime_max_proj_residual: float = Field(default=0, ge=0)
    total_steps: int = Field(default=0, ge=0)
    preserved_steps: int = Field(default=0, ge=0)
    projected_steps: int = Field(default=0, ge=0)
    stabilized_steps: int = Field(default=0, ge=0)
    enforcement_ratio: float = Field(default=0, ge=0, le=1)
    domain_tag: str = Field(min_length=1)

    @property
    def passed(self) -> bool:
        return self.static_purity_passed and not self.violations

    @model_validator(mode="after")
    def counts(self):
        if self.preserved_steps + self.projected_steps + self.stabilized_steps != self.total_steps:
            raise ValueError("preserved/projected/stabilized categories must partition the steps")
        if self.enforcement_ratio != (self.projected_steps + self.stabilized_steps) / max(1, self.total_steps):
            raise ValueError("inconsistent enforcement ratio")
        return self


class GciAudit(StrictModel):
    passed: bool
    fine_value: float
    apparent_order_p: float | None
    gci_fine_abs: float = Field(ge=0)
    eps_peak_loc: float = Field(ge=0)
    safety_factor: float
    oscillatory: bool
    reason: str | None = None


class StatisticalReliabilityAudit(StrictModel):
    passed: bool
    variance_ratio: float = Field(ge=1)
    mean_drift_ratio: float = Field(default=0, ge=0)
    stream_keys: tuple[tuple[int, int, int], ...]
    failure_reason: str | None = None


class FidelityAudit(StrictModel):
    passed: bool
    algebraic_equiv_passed: bool
    si_and_semantic_tag_passed: bool
    convergence_passed: bool
    physical_checks_passed: bool
    apparent_order_p: float | None = None
    gci_fine_abs: float | None = Field(default=None, ge=0)
    secondary_gci_abs: float | None = Field(default=None, ge=0)
    multi_scheme_delta: float | None = Field(default=None, ge=0)
    convergence_reliability_ratio: float | None = Field(default=None, ge=0)
    scheme_delta_to_primary_gci: float | None = Field(default=None, ge=0)
    eps_peak_loc: float = Field(default=0, ge=0)
    epsilon_disc: float = Field(ge=0)
    epsilon_stat: float = Field(default=0, ge=0)
    epsilon_round: float = Field(default=0, ge=0)
    epsilon_total: float = Field(ge=0)
    statistical_audit: StatisticalReliabilityAudit | None = None
    failure_reason: str | None = None

    @property
    def alg_and_dim_passed(self) -> bool:
        return self.algebraic_equiv_passed and self.si_and_semantic_tag_passed

    @model_validator(mode="after")
    def consistent(self):
        if not math.isclose(self.epsilon_total, self.epsilon_disc + self.epsilon_stat + self.epsilon_round, rel_tol=1e-14, abs_tol=0):
            raise ValueError("epsilon_total must preserve the complete error budget")
        if self.passed and (not all((self.alg_and_dim_passed, self.convergence_passed, self.physical_checks_passed))
                            or self.failure_reason or (self.convergence_reliability_ratio or 0) > 1
                            or self.statistical_audit is not None and not self.statistical_audit.passed):
            raise ValueError("failed fidelity obligations cannot be certified")
        return self


ZERO_DISSIPATION_INVARIANT = 254
BOUNDARY_FLUX = 253
TRAJECTORY_MAXIMUM = 252
TRAJECTORY_MINIMUM = 251


class IdentifiabilityAudit(StrictModel):
    coupling_passed: bool
    dead_parameters: tuple[str, ...] = ()
    column_norms: dict[str, float] = Field(default_factory=dict)
    singular_values: tuple[float, ...] = ()
    brun_collinearity_gamma_k: float | None = None
    fim_condition_number: float | None = None
    equilibrium_restoring_force: float | None = None
    excluded_certified_invariants: tuple[str, ...] = ()


class SignificanceAudit(StrictModel):
    passed: bool
    null_baseline_description: str
    effect_size_estimate: float
    certified_lower_margin: float
    method: Literal["ODE_GCI_SEPARATION", "MC_EMPIRICAL_BERNSTEIN", "MC_CLOPPER_PEARSON", "FIXED_POINT_KRAWCZYK_HURWITZ", "BIFURCATION_LYAPUNOV_BRACKET", "PARETO_EPS_DOMINANCE"]
    confidence_level: float | None = Field(default=None, gt=0, lt=1)
    relative_effect_size: float = Field(ge=0)
    micro_effect_warning: bool
    domain_compression_bits: float | None = Field(default=None, ge=0)
    failure_reason: str | None = None
    win_rate_method: Literal["MC_CLOPPER_PEARSON"] | None = None
    win_rate_interval: tuple[float, float] | None = None
    replicate_count: int | None = Field(default=None, ge=2)


class UniquenessAudit(StrictModel):
    passed: bool
    cds_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cds_claim_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cds_full_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vault_class: Literal["NOVEL_FIRST_IN_VAULT", "EXACT_ALGEBRAIC_DUPLICATE", "VALID_DOMAIN_EXTENSION", "VALID_BOUND_SHARPENING", "VALID_COUPLED_EXTENSION", "VALID_REFUTATION_OF_PRIOR"] | None
    related_run_id: str | None = None
    reference_nontrivial: bool
    shbt_open_boundary_closure: bool = False

    @model_validator(mode="after")
    def classified_if_passed(self):
        if self.passed and self.vault_class is None: raise ValueError("passed uniqueness requires a proven Vault class")
        return self


class FalsificationCertificate(StrictModel):
    mode: Literal["MODE1_DEDUCTIVE_MUS", "MODE2A_COUNTEREXAMPLE", "MODE2B_EXHAUSTIVE_ENCLOSURE"]
    minimal_unsat_core: tuple[str, ...] | None = None
    proof_checker: Literal["z3_quickxplain", "cvc5_alethe_carcara"] | None = None
    proof_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    counterexample_theta: dict[str, float] | None = None
    signed_margin_at_witness: float | None = None
    certified_sup_upper_bound: float | None = None
    certified_subbox_count: int = Field(default=0, ge=0)
    boxes_evaluated: int = Field(default=0, ge=0)
    certified_refuted_volume_fraction: float = Field(default=0, ge=0, le=1)
    budget_exhausted: bool = False
    max_unresolved_upper_bound: float | None = None
    epsilon_total: float = Field(ge=0)
    epistemic_verdict: EpistemicVerdict = EpistemicVerdict.INCONCLUSIVE_UNVERIFIED
    ihcg_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fidelity_passed: bool = False
    reliability_passed: bool = False
    failure_reason: str | None = None
    witness_margin_interval: tuple[float, float] | None = None
    model_consistency_status: Literal["SAT", "UNSAT", "UNKNOWN"] | None = None
    core_minimal: bool = False
    solver_calls: int = Field(default=0, ge=0, le=64)
    proof_smt2: str | None = None
    proof_constraints: dict[str, str] = Field(default_factory=dict)
    external_checker_status: Literal["NOT_REQUESTED", "UNAVAILABLE", "VERIFIED", "FAILED"] = "NOT_REQUESTED"
    unresolved_bound_unbounded: bool = False
    effective_max_boxes: int = Field(default=0, ge=0, le=32768)
    enclosure_method: Literal["NATIVE_STRICT_F64_THRESHOLD", "LIPSCHITZ_GRID"] | None = None

    @property
    def passed(self) -> bool:
        if (self.budget_exhausted or self.failure_reason or not self.ihcg_digest or not self.fidelity_passed
                or not self.reliability_passed or self.epistemic_verdict != EpistemicVerdict.VERIFIED_FALSIFICATION):
            return False
        if self.mode == "MODE1_DEDUCTIVE_MUS":
            return bool(self.minimal_unsat_core and self.proof_checker and self.proof_digest and self.proof_smt2
                        and self.core_minimal and self.model_consistency_status == "SAT")
        if self.mode == "MODE2A_COUNTEREXAMPLE":
            return (self.counterexample_theta is not None and self.signed_margin_at_witness is not None
                    and self.witness_margin_interval is not None and self.witness_margin_interval[1] + self.epsilon_total < 0
                    and self.signed_margin_at_witness + self.epsilon_total < 0)
        return (self.certified_refuted_volume_fraction == 1 and self.certified_subbox_count > 0
                and self.certified_subbox_count <= self.boxes_evaluated <= self.effective_max_boxes
                and not self.unresolved_bound_unbounded and self.max_unresolved_upper_bound is None and self.certified_sup_upper_bound is not None
                and self.certified_sup_upper_bound + self.epsilon_total < 0)

    @model_validator(mode="after")
    def certified_evidence(self):
        if self.epistemic_verdict == EpistemicVerdict.VERIFIED_FALSIFICATION and not self.passed:
            raise ValueError("verified falsification requires closed evidence and passed fidelity/reliability")
        if self.witness_margin_interval is not None and self.witness_margin_interval[0] > self.witness_margin_interval[1]:
            raise ValueError("invalid witness margin enclosure")
        if self.passed and self.mode == "MODE1_DEDUCTIVE_MUS":
            payload = _json({"ihcg": self.ihcg_digest, "constraints": self.proof_constraints, "smt2": self.proof_smt2})
            if set(self.proof_constraints) != set(self.minimal_unsat_core) or not hmac.compare_digest(self.proof_digest, _native.py_compute_certificate_receipt(payload)):
                raise ValueError("deductive proof receipt mismatch")
        return self


class BoundaryCertificate(StrictModel):
    valid_witness_theta: dict[str, float]
    falsified_witness_theta: dict[str, float]
    valid_margin_lower: float
    falsified_margin_upper: float
    bracket_width: float = Field(ge=0)
    target_tolerance: float = Field(gt=0)
    epistemic_verdict: EpistemicVerdict | None = None
    ihcg_digest: str | None = None
    epsilon_total: float = Field(default=0, ge=0)
    evaluations: int = Field(default=0, ge=0, le=64)
    budget_exhausted: bool = False
    failure_reason: str | None = None
    boundary_parameter: str | None = None
    valid_witness_enclosure: Bounds | None = None
    falsified_witness_enclosure: Bounds | None = None

    @model_serializer(mode="wrap")
    def serialize_boundary(self, handler):
        data = handler(self)
        for key in ("boundary_parameter","valid_witness_enclosure","falsified_witness_enclosure"):
            if data.get(key) is None: data.pop(key,None)
        return data

    @model_validator(mode="after")
    def bracket(self):
        if not self.valid_witness_theta or self.valid_witness_theta.keys() != self.falsified_witness_theta.keys():
            raise ValueError("boundary witnesses must share a nonempty parameter space")
        keys = [self.boundary_parameter] if self.boundary_parameter else self.valid_witness_theta
        if any(k not in self.valid_witness_theta for k in keys): raise ValueError("boundary parameter is missing")
        distance = max(abs(self.valid_witness_theta[k]-self.falsified_witness_theta[k]) for k in keys)
        for point,box in ((self.valid_witness_theta,self.valid_witness_enclosure),(self.falsified_witness_theta,self.falsified_witness_enclosure)):
            if box is not None and (point.keys() != box.keys() or any(not box[s][0] <= v <= box[s][1] for s,v in point.items())):
                raise ValueError("boundary witness is outside its enclosure")
        if distance > self.bracket_width + 16*sys.float_info.epsilon*max(1, distance):
            raise ValueError("bracket width understates witness separation")
        return self

    @property
    def passed(self) -> bool:
        return (not self.budget_exhausted and not self.failure_reason and self.epistemic_verdict in {None, EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY}
                and self.valid_margin_lower > 0 > self.falsified_margin_upper and self.bracket_width <= self.target_tolerance)


class IhcgAnchor(StrictModel):
    ihcg_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage_chain_digests: tuple[str, ...]
    quantifier_mode: Literal["FORALL", "EXISTS", "BOUNDARY"]
    archetype: Archetype
    prompt_obligation_quotes: tuple[dict[str, str], ...]


class MeaningfulnessCertificate(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str = Field(min_length=1)
    epistemic_verdict: EpistemicVerdict
    ihcg_anchor: IhcgAnchor
    contract_audit: ContractAudit
    nontriviality_audit: NontrivialityAudit
    purity_audit: PurityAudit
    fidelity_audit: FidelityAudit
    identifiability_audit: IdentifiabilityAudit
    significance_audit: SignificanceAudit | None = None
    falsification_certificate: FalsificationCertificate | None = None
    boundary_certificate: BoundaryCertificate | None = None
    certified_valid_domain: Bounds | None = None
    uniqueness_audit: UniquenessAudit
    convergence_reliability_ratio: float | None = None
    preserved_vs_enforced: dict[str, PurityAudit]
    relative_effect_size: float | None = None
    domain_compression_bits: float | None = None
    z3_tier_used: Tier
    trust_boundary_version: Literal["invar.tcb.phase2a.native-libm16.v1"] = TRUST_BOUNDARY_VERSION
    uast_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    blake3_receipt: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_serializer(mode="wrap")
    def serialize_certificate(self, handler):
        data = handler(self)
        if data.get("certified_valid_domain") is None: data.pop("certified_valid_domain",None)
        return data

    @model_validator(mode="after")
    def gates(self):
        sig = self.significance_audit
        if (self.convergence_reliability_ratio != self.fidelity_audit.convergence_reliability_ratio
                or self.relative_effect_size != (sig.relative_effect_size if sig else None)
                or self.domain_compression_bits != (sig.domain_compression_bits if sig else None)):
            raise ValueError("certificate diagnostics must match their source audits")
        if self.preserved_vs_enforced.get(self.purity_audit.domain_tag) != self.purity_audit:
            raise ValueError("per-domain purity audit missing or changed")
        if self.epistemic_verdict != EpistemicVerdict.INCONCLUSIVE_UNVERIFIED:
            if not all(meaningfulness_gates(self).values()):
                raise ValueError("all eight meaningfulness gates are mandatory")
            if self.epistemic_verdict == EpistemicVerdict.VERIFIED_FALSIFICATION:
                if self.falsification_certificate is None or not self.falsification_certificate.passed:
                    raise ValueError("a closed falsification certificate is mandatory")
            elif not sig or not sig.passed:
                raise ValueError("discovery requires certified significance")
            if self.epistemic_verdict == EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY and (self.boundary_certificate is None or not self.boundary_certificate.passed):
                raise ValueError("a certified boundary bracket is mandatory")
            if self.falsification_certificate is not None and not math.isclose(self.falsification_certificate.epsilon_total, self.fidelity_audit.epsilon_total, rel_tol=1e-14, abs_tol=0):
                raise ValueError("falsification may not change the certified uncertainty")
            if self.falsification_certificate is not None and self.falsification_certificate.ihcg_digest != self.ihcg_anchor.ihcg_digest:
                raise ValueError("falsification certificate belongs to a different IHCG")
            if self.epistemic_verdict == EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY:
                boundary = self.boundary_certificate
                if boundary is None or boundary.ihcg_digest != self.ihcg_anchor.ihcg_digest or boundary.epsilon_total != self.fidelity_audit.epsilon_total:
                    raise ValueError("boundary certificate must preserve the IHCG and uncertainty")
        return self


def meaningfulness_gates(certificate: MeaningfulnessCertificate) -> dict[str, bool]:
    """The master eight-gate conjunction, including deductive Mode 1's shortcut."""
    c, f = certificate, certificate.fidelity_audit
    falsification = c.falsification_certificate
    mode1 = (c.epistemic_verdict == EpistemicVerdict.VERIFIED_FALSIFICATION and falsification is not None
             and falsification.mode == "MODE1_DEDUCTIVE_MUS" and falsification.passed
             and c.contract_audit.classification == "INFEASIBLE_UNDER_MODEL")
    reliable = ((f.convergence_reliability_ratio or 0) <= 1 and (f.statistical_audit is None or f.statistical_audit.passed)
                and f.failure_reason not in {"CONVERGENCE_ESTIMATOR_UNRELIABLE", "STATISTICAL_ESTIMATOR_UNRELIABLE"})
    if c.epistemic_verdict in {EpistemicVerdict.VERIFIED_FALSIFICATION, EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY}:
        reliable = reliable and (c.ihcg_anchor.archetype not in {"ODE_PDE", "PARAM_SWEEP"} or f.convergence_reliability_ratio is not None)
        reliable = reliable and (c.ihcg_anchor.archetype != "MONTE_CARLO" or f.statistical_audit is not None)
    if c.epistemic_verdict == EpistemicVerdict.VERIFIED_FALSIFICATION:
        epistemic = falsification is not None and falsification.passed
    else:
        epistemic = c.significance_audit is not None and c.significance_audit.passed
        if c.epistemic_verdict == EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY:
            epistemic = epistemic and c.boundary_certificate is not None and c.boundary_certificate.passed
    return dict(
        phi_lock=c.contract_audit.entailment_locked,
        phi_nonvac=c.contract_audit.classification in {"CONSTRAINING", "INFEASIBLE_UNDER_MODEL"} and not c.contract_audit.redundant_conjuncts,
        phi_pure=c.purity_audit.passed and all(a.passed for a in c.preserved_vs_enforced.values()),
        phi_fid=reliable and f.passed,
        phi_active=mode1 or c.identifiability_audit.coupling_passed,
        phi_epistemic=bool(epistemic),
        phi_novel=c.uniqueness_audit.passed and c.uniqueness_audit.reference_nontrivial and c.uniqueness_audit.vault_class != "EXACT_ALGEBRAIC_DUPLICATE",
        phi_nontrivial=c.nontriviality_audit.passed,
    )


_ARITIES = {"exp": 1, "log": 1, "sqrt": 1, "sin": 1, "cos": 1, "abs": 1,
            "logistic_sigmoid": 1, "kl_term": 2, "hill_eq": 3, "michaelis_menten": 3,
            "pow": 2, "min": 2, "max": 2, "implies": 2}


def _parse(text: str) -> ast.AST:
    if not isinstance(text, str) or len(text) > 32768:
        raise ValueError("formal expression exceeds the input budget")
    node = ast.parse(text, mode="eval").body
    nodes = list(ast.walk(node))
    if len(nodes) > 4096:
        raise ValueError("formal expression exceeds the AST budget")
    stack = [(node, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > 64: raise ValueError("formal expression exceeds the nesting budget")
        stack.extend((child, depth+1) for child in ast.iter_child_nodes(current))
    allowed = (ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
               ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Not,
               ast.And, ast.Or, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
    for n in nodes:
        if not isinstance(n, allowed):
            raise ValueError(f"unsupported formal syntax: {type(n).__name__}")
        if isinstance(n, ast.Constant) and (not isinstance(n.value, (bool, int, float)) or not math.isfinite(n.value)):
            raise ValueError("only finite real and boolean constants are supported")
        if isinstance(n, ast.Call) and (not isinstance(n.func, ast.Name) or n.func.id not in _ARITIES
                                       or len(n.args) != _ARITIES[n.func.id] or n.keywords):
            raise ValueError("unsupported formal function or arity")
    return node


def _variables(node: ast.AST) -> set[str]:
    calls = {id(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and id(n) not in calls}


def _bounds(domain: Mapping[str, Sequence[float]]) -> Bounds:
    if not isinstance(domain, Mapping):
        raise ValueError("ihcg_evidence.domain must map each slot symbol to [lower, upper]")
    if len(domain) > MAX_VARIABLES:
        raise ValueError("at most 32 bounded variables are supported")
    result = {}
    for name, pair in sorted(domain.items()):
        path = f"ihcg_evidence.domain[{name!r}]"
        if not isinstance(name, str) or not name.isidentifier() or name in _ARITIES:
            raise ValueError(f"{path}: invalid domain symbol")
        if isinstance(pair, (Mapping, str, bytes)) or not hasattr(pair, "__len__") or len(pair) != 2:
            raise ValueError(f"{path}: expected [lower, upper] numeric pair, not an object; encode a fixed value as [value, value]")
        try:
            lo, hi = map(float, pair)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path}: expected finite numeric bounds [lower, upper]") from error
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError(f"{path}: finite ordered domain bounds required")
        result[name] = (lo, hi)
    return result


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class IhcgCompiler:
    """Compile explicit formal evidence; never guess scientific prose semantics.

    Evidence supplies slots, equations, assumptions, quoted obligations and the
    quantifier mode. Each obligation is tied to an exact prompt character span.
    The prose-to-formula semantic binding remains a declared ingestion TCB.
    """

    def compile_from_prompt(self, prompt: str, evidence: Mapping[str, Any], domain: Mapping[str, Sequence[float]], archetype: Archetype) -> IhcgGraph:
        data = copy.deepcopy(dict(evidence))
        data.update(omega=_bounds(domain), archetype=archetype)
        graph = IhcgGraph.model_validate_json(_json(data))
        names = {s.symbol for s in graph.slots}
        if names != set(graph.omega) or len(names) != len(graph.slots) or len({s.slot_idx for s in graph.slots}) != len(names):
            raise ValueError(f"IHCG slots and Omega must be one-to-one; missing_domain={sorted(names-set(graph.omega))}; undeclared_domain={sorted(set(graph.omega)-names)}; symbols={[s.symbol for s in graph.slots]}; indices={[s.slot_idx for s in graph.slots]}")
        if not graph.obligations or not any(o.obligation_class == "HYPOTHESIS_TARGET" for o in graph.obligations):
            raise ValueError("at least one quoted primary hypothesis is required")
        if len({o.id for o in graph.obligations}) != len(graph.obligations):
            raise ValueError("duplicate obligation identifiers")
        for obligation in graph.obligations:
            start, end = obligation.prompt_span
            if not 0 <= start < end <= len(prompt) or prompt[start:end] != obligation.prompt_quote:
                raise ValueError("obligation quote does not match its immutable prompt span")
        expressions = [(f"environmental_assumptions[{i}]", e) for i,e in enumerate(graph.environmental_assumptions)]
        expressions += [(f"obligations[{i}].expression", o.expression) for i,o in enumerate(graph.obligations)]
        expressions += [(f"equations[{name!r}].expression", e.expression) for name,e in graph.equations.items()]
        for path, expression in expressions:
            try:
                variables = _variables(_parse(expression))
            except (ValueError, SyntaxError) as error:
                raise ValueError(f"ihcg_evidence.evidence.{path}: {error}; expression={expression!r}") from error
            if not variables <= names:
                raise ValueError(f"ihcg_evidence.evidence.{path}: undeclared IHCG symbol: {sorted(variables-names)}")
        controls = {s.symbol for s in graph.slots if s.role == "CONTROL_PARAM"}
        states = {s.symbol for s in graph.slots if s.role == "DYNAMIC_STATE"}
        evolved = [e.state_symbol for e in graph.equations.values() if e.kind == "ODE_RHS"]
        if len(evolved) != len(set(evolved)):
            raise ValueError("multiple evolution equations for one state are unsupported")
        for name,equation in graph.equations.items():
            if equation.kind == "ODE_RHS" and equation.state_symbol not in states:
                raise ValueError("ODE RHS must bind a declared dynamic state")
            if equation.kind in {"FIXED_POINT_RESIDUAL", "FIXED_POINT_MAP"} and equation.state_symbol not in states:
                raise ValueError("fixed-point equations must bind dynamic states")
            if equation.kind == "FIXED_POINT_MAP" and equation.si_dim != next(s.si_dim for s in graph.slots if s.symbol == equation.state_symbol):
                raise ValueError("fixed-point map dimensions must match the bound state")
            if equation.kind == "MC_SAMPLE" and equation.state_symbol not in names:
                raise ValueError("MC sample equations must bind an aggregate observable")
            if equation.kind in {"TRAJECTORY_MAX","TRAJECTORY_MIN"}:
                tag = TRAJECTORY_MAXIMUM if equation.kind == "TRAJECTORY_MAX" else TRAJECTORY_MINIMUM
                if (name not in names or equation.state_symbol not in states or equation.expression.strip() != equation.state_symbol
                        or next(s for s in graph.slots if s.symbol == name).semantic_tag != tag):
                    raise ValueError("trajectory reductions require a typed observable and a dynamic source")
        for slot in graph.slots:
            if slot.semantic_tag in {TRAJECTORY_MAXIMUM,TRAJECTORY_MINIMUM}:
                kind = "TRAJECTORY_MAX" if slot.semantic_tag == TRAJECTORY_MAXIMUM else "TRAJECTORY_MIN"
                if slot.symbol not in graph.equations or graph.equations[slot.symbol].kind != kind:
                    raise ValueError("trajectory semantic tags require matching reduction equations")
        if any(s.stochastic_role and s.role != "CONTROL_PARAM" for s in graph.slots):
            raise ValueError("stochastic inputs must be immutable control slots")
        if any(s.stochastic_role and s.symbol in graph.mechanism_parameters for s in graph.slots):
            raise ValueError("stochastic stream inputs cannot be mechanism interventions")
        if not set(graph.mechanism_parameters) <= controls or not set(graph.control_values) <= controls:
            raise ValueError("mechanism interventions and locked constants must be control parameters")
        if any(not graph.omega[k][0] <= v <= graph.omega[k][1] for k, v in graph.control_values.items()):
            raise ValueError("locked constant lies outside Omega")
        digest = _native.py_compute_ihcg_digest(graph.model_dump_json())
        return graph.model_copy(update={"h_ihcg": digest})

    def extend_stage(self, parent: IhcgGraph, prompt: str, evidence: Mapping[str, Any], domain: Mapping[str, Sequence[float]], archetype: Archetype) -> IhcgGraph:
        parent.assert_sealed()
        child = self.compile_from_prompt(prompt, evidence, domain, archetype)
        if child.quantifier_mode != parent.quantifier_mode or child.archetype != parent.archetype:
            raise ValueError("stage extension may not change the foundation's quantifier or archetype")
        if not set(parent.slots) <= set(child.slots) or any(child.omega.get(s) != bounds for s, bounds in parent.omega.items()):
            raise ValueError("stage extension may not change foundation slots or domains")
        if any(child.equations.get(name) != equation for name,equation in parent.equations.items()) or not set(parent.mechanism_parameters) <= set(child.mechanism_parameters):
            raise ValueError("stage extension may not remove foundation equations or mechanisms")
        if not all(o in child.obligations for o in parent.obligations) or not set(parent.environmental_assumptions) <= set(child.environmental_assumptions):
            raise ValueError("stage extension may not remove foundation obligations or assumptions")
        if any(child.control_values.get(k) != v for k, v in parent.control_values.items()):
            raise ValueError("stage extension may not change locked foundation constants")
        delta = child.model_dump(mode="json", exclude={"h_ihcg", "stage_chain_digests", "parent_ihcg_digest"})
        chain = _native.py_compute_ihcg_stage_chain(parent.stage_chain_digests[-1] if parent.stage_chain_digests else parent.h_ihcg, _json(delta))
        child = child.model_copy(update={"parent_ihcg_digest": parent.h_ihcg, "stage_chain_digests": (*parent.stage_chain_digests, chain)})
        return child.model_copy(update={"h_ihcg": _native.py_compute_ihcg_digest(child.model_dump_json())})


def _fraction(node: ast.AST) -> Fraction | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _fraction(node.operand)
        return (-value if isinstance(node.op, ast.USub) else value) if value is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        a, b = _fraction(node.left), _fraction(node.right)
        return a / b if a is not None and b else None
    return None


def _q(value: int | float | Fraction):
    value = Fraction(value)
    return z3.RealVal(f"{value.numerator}/{value.denominator}")


class _DomainError(ValueError):
    pass


class _DeadlineExceeded(Exception):
    pass


def _check_query_deadline():
    if time.monotonic() >= _QUERY_DEADLINE.get(): raise _DeadlineExceeded


class _IV:
    def __init__(self, lo: float, hi: float | None = None):
        self.lo, self.hi = float(lo), float(lo if hi is None else hi)
        if math.isnan(self.lo) or math.isnan(self.hi) or self.lo > self.hi:
            raise _DomainError("invalid interval")

    @classmethod
    def rational(cls, value: Fraction):
        x = float(value)
        return cls(math.nextafter(x, -math.inf) if Fraction(x) > value else x,
                   math.nextafter(x, math.inf) if Fraction(x) < value else x)

    def op(self, op: str, other: _IV | None = None):
        symbol = lambda name: {"kind": "symbol", "name": name}
        if other is None:
            expr = {"kind": "call", "func": op, "args": [symbol("a")]}
            bounds = {"a": (self.lo, self.hi)}
        else:
            expr = {"kind": "binary", "op": op, "left": symbol("a"), "right": symbol("b")}
            bounds = {"a": (self.lo, self.hi), "b": (other.lo, other.hi)}
        try:
            return _IV(*_native.evaluate_interval_f64(_json(expr), bounds))
        except ValueError as error:
            raise _DomainError(str(error)) from error

    def __add__(self, b): return self.op("+", b)
    def __sub__(self, b): return self.op("-", b)
    def __mul__(self, b): return self.op("*", b)
    def __truediv__(self, b):
        if b.lo <= 0 <= b.hi:
            raise _DomainError("division crosses zero")
        return self.op("/", b)
    def __neg__(self): return _IV(-self.hi, -self.lo)


def _trig(x: _IV, cosine: bool) -> _IV:
    native = x.op("cos" if cosine else "sin")
    if max(abs(x.lo), abs(x.hi)) > 8:
        return native
    # Taylor's theorem with |f^(k)| <= 1, evaluated by native interval arithmetic.
    degree = 48 if cosine else 49
    terms = range(degree if cosine else degree, -1 if cosine else 0, -2)
    square = x * x
    polynomial = _IV(0)
    for power in terms:
        coefficient = Fraction((-1) ** (power // 2), math.factorial(power))
        polynomial = polynomial * square + _IV.rational(coefficient)
    if not cosine:
        polynomial = polynomial * x
    magnitude = max(abs(x.lo), abs(x.hi))
    remainder = Fraction(magnitude) ** (degree + 1) / math.factorial(degree + 1)
    bound = _IV.rational(remainder).hi
    polynomial = polynomial + _IV(-bound, bound)
    return _IV(max(native.lo, polynomial.lo), min(native.hi, polynomial.hi))


def _expand(node: ast.AST) -> ast.AST:
    """Lift scientific operators to the common elementary interval substrate."""
    if not isinstance(node, ast.Call):
        return node
    name, args = node.func.id, node.args
    templates = {
        "logistic_sigmoid": "1 / (1 + exp(-a))",
        "kl_term": "max(0, a * log(a / max(b, 1e-30)) - a + max(b, 1e-30))",
        "hill_eq": "pow(a, c) / (pow(b, c) + pow(a, c))",
        "michaelis_menten": "b * a / (c + a)",
        "pow": "exp(b * log(a))",
    }
    if name not in templates:
        return node
    if name == "pow":
        exponent = _fraction(args[1])
        if exponent is not None and exponent.denominator == 1 and abs(exponent) <= 16:
            return ast.BinOp(left=args[0], op=ast.Pow(), right=args[1])
    replacements = dict(zip(("a", "b", "c"), args))
    class Substitute(ast.NodeTransformer):
        def visit_Name(self, n):
            return copy.deepcopy(replacements.get(n.id, n))
    return Substitute().visit(ast.parse(templates[name], mode="eval").body)


def _iv(node: ast.AST, box: Bounds) -> _IV:
    _check_query_deadline()
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _IV.rational(Fraction(node.value))
    if isinstance(node, ast.Name):
        return _IV(*box[node.id])
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)): raise _DomainError("boolean used as arithmetic")
        a = _iv(node.operand, box)
        return -a if isinstance(node.op, ast.USub) else a
    if isinstance(node, ast.BinOp):
        a = _iv(node.left, box)
        if isinstance(node.op, ast.Pow):
            exponent = _fraction(node.right)
            if exponent is not None and exponent.denominator == 1 and abs(exponent) <= 16:
                out = _IV(1)
                for _ in range(abs(exponent.numerator)):
                    out = out * a
                return _IV(1) / out if exponent < 0 else out
            return (_iv(node.right, box) * a.op("log")).op("exp")
        b = _iv(node.right, box)
        return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b}[type(node.op)]()
    if isinstance(node, ast.Call):
        if node.func.id == "kl_term":
            p, q = (_iv(a, box) for a in node.args)
            if p.hi <= 0: return _IV(max(q.lo, 1e-30), max(q.hi, 1e-30))
            if p.lo <= 0: raise _DomainError("KL box straddles its zero-mass branch")
        expanded = _expand(node)
        if expanded is not node:
            return _iv(expanded, box)
        args = [_iv(a, box) for a in node.args]
        name = node.func.id
        if name == "abs":
            a = args[0]
            return _IV(0 if a.lo <= 0 <= a.hi else min(abs(a.lo), abs(a.hi)), max(abs(a.lo), abs(a.hi)))
        if name in {"min", "max"}:
            f = min if name == "min" else max
            return _IV(f(a.lo for a in args), f(a.hi for a in args))
        if name in {"sin", "cos"}:
            return _trig(args[0], name == "cos")
        return args[0].op(name)
    raise _DomainError("non-arithmetic interval expression")


def _truth(node: ast.AST, box: Bounds) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BoolOp):
        values = [_truth(v, box) for v in node.values]
        if isinstance(node.op, ast.And):
            return False if False in values else (True if all(v is True for v in values) else None)
        return True if True in values else (False if all(v is False for v in values) else None)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _truth(node.operand, box)
        return None if value is None else not value
    if isinstance(node, ast.Call) and node.func.id == "implies":
        return _truth(ast.BoolOp(op=ast.Or(), values=[ast.UnaryOp(op=ast.Not(), operand=node.args[0]), node.args[1]]), box)
    if isinstance(node, ast.Compare):
        values = []
        for a, op, b in zip([node.left, *node.comparators], node.ops, node.comparators):
            x, y = _iv(a, box), _iv(b, box)
            same = ast.dump(a) == ast.dump(b)
            if isinstance(op, ast.Lt): v = True if x.hi < y.lo else (False if x.lo >= y.hi else None)
            elif isinstance(op, ast.LtE): v = True if x.hi <= y.lo or same else (False if x.lo > y.hi else None)
            elif isinstance(op, ast.Gt): v = True if x.lo > y.hi else (False if x.hi <= y.lo else None)
            elif isinstance(op, ast.GtE): v = True if x.lo >= y.hi or same else (False if x.hi < y.lo else None)
            else:
                v = True if same or x.lo == x.hi == y.lo == y.hi else (False if x.hi < y.lo or y.hi < x.lo else None)
                if isinstance(op, ast.NotEq): v = None if v is None else not v
            values.append(v)
        return False if False in values else (True if all(v is True for v in values) else None)
    raise _DomainError("expected a boolean predicate")


def _degree(node: ast.AST) -> int:
    if isinstance(node, ast.Constant): return 0
    if isinstance(node, ast.Name): return 1
    if isinstance(node, ast.UnaryOp): return _degree(node.operand)
    if isinstance(node, ast.BinOp):
        a, b = _degree(node.left), _degree(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)): return max(a, b)
        if isinstance(node.op, ast.Mult): return a + b
        if isinstance(node.op, ast.Div) and _fraction(node.right) not in (None, 0): return a
        if isinstance(node.op, ast.Div): return max(2,a,b)
        exponent = _fraction(node.right)
        if isinstance(node.op, ast.Pow) and exponent is not None and exponent.denominator == 1 and 0 <= exponent <= 6: return a * exponent.numerator
        return 1000
    if isinstance(node, ast.Call) and node.func.id == "implies": return max(map(_degree, node.args))
    if isinstance(node, ast.BoolOp): return max(map(_degree, node.values), default=0)
    if isinstance(node, ast.Compare): return max(map(_degree, [node.left, *node.comparators]))
    return 1000


class _Translator:
    def __init__(self, box: Bounds, outer: bool):
        self.box, self.outer = box, outer
        self.vars = {s: z3.Real(s) for s in box}
        self.constraints = [c for s, (lo, hi) in box.items() for c in (self.vars[s] >= _q(lo), self.vars[s] <= _q(hi))]
        self.serial = 0

    def fresh(self, iv: _IV):
        if not math.isfinite(iv.lo) or not math.isfinite(iv.hi): raise _DomainError("unbounded outer enclosure")
        self.serial += 1
        value = z3.Real(f"$invar_outer_{self.serial}")
        self.constraints.extend((value >= _q(iv.lo), value <= _q(iv.hi)))
        return value

    def product(self, a, b, x: _IV, y: _IV):
        w = self.fresh(x * y)
        xl, xu, yl, yu = map(_q, (x.lo, x.hi, y.lo, y.hi))
        self.constraints.extend((w >= xl*b + yl*a - xl*yl, w >= xu*b + yu*a - xu*yu,
                                 w <= xu*b + yl*a - xu*yl, w <= xl*b + yu*a - xl*yu))
        return w

    def term(self, node):
        _check_query_deadline()
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return _q(node.value)
        if isinstance(node, ast.Name): return self.vars[node.id]
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)): raise _DomainError("boolean used as arithmetic")
            a = self.term(node.operand)
            return -a if isinstance(node.op, ast.USub) else a
        if isinstance(node, ast.BinOp):
            a, b = self.term(node.left), self.term(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult):
                if not self.outer or _degree(node.left) == 0 or _degree(node.right) == 0: return a*b
                return self.product(a, b, _iv(node.left, self.box), _iv(node.right, self.box))
            if isinstance(node.op, ast.Div):
                constant = _fraction(node.right)
                if constant is not None and constant != 0: return a / _q(constant)
                denominator = _iv(node.right,self.box)
                if denominator.lo <= 0 <= denominator.hi: raise _DomainError("division domain crosses zero")
                if not self.outer: return a/b
                inverse = self.univariate("reciprocal", b, _iv(node.right, self.box))
                return self.product(a, inverse, _iv(node.left, self.box), _IV(1) / _iv(node.right, self.box))
            exponent = _fraction(node.right)
            if not self.outer: return a ** exponent.numerator
            if exponent is not None and exponent.denominator == 1 and 0 <= exponent <= 16:
                out, interval = _q(1), _IV(1)
                for _ in range(exponent.numerator):
                    out = self.product(out, a, interval, _iv(node.left, self.box))
                    interval = interval * _iv(node.left, self.box)
                return out
            return self.term(_parse(f"exp(({ast.unparse(node.right)}) * log({ast.unparse(node.left)}))"))
        if isinstance(node, ast.Call):
            if node.func.id == "kl_term":
                p = _iv(node.args[0], self.box)
                if p.hi <= 0: return self.term(_parse(f"max({ast.unparse(node.args[1])}, 1e-30)"))
                if p.lo <= 0: raise _DomainError("KL box straddles its zero-mass branch")
            expanded = _expand(node)
            if expanded is not node: return self.term(expanded)
            a = self.term(node.args[0])
            if node.func.id == "abs": return z3.If(a >= 0, a, -a)
            if node.func.id in {"min", "max"}:
                b = self.term(node.args[1]); return z3.If(a <= b, a, b) if node.func.id == "min" else z3.If(a >= b, a, b)
            return self.univariate(node.func.id, a, _iv(node.args[0], self.box))
        raise _DomainError("unsupported arithmetic translation")

    def univariate(self, name, argument, interval):
        if not self.outer: raise _DomainError("non-polynomial direct query")
        lo, hi = interval.lo, interval.hi
        if not math.isfinite(lo) or not math.isfinite(hi): raise _DomainError("finite nonlinear bounds required")
        def function(x):
            if name == "reciprocal": return _IV(1) / x
            return _trig(x, name == "cos") if name in {"sin", "cos"} else x.op(name)
        def derivative(x):
            if name == "exp": return x.op("exp")
            if name == "log": return _IV(1) / x
            if name == "sqrt": return _IV(0.5) / x.op("sqrt")
            if name == "sin": return _trig(x, True)
            if name == "cos": return -_trig(x, False)
            return -(_IV(1) / (x * x))
        value = self.fresh(function(interval))
        center = lo/2 + hi/2
        f = function(_IV(center))
        try: d = derivative(interval)
        except _DomainError: return value
        delta = argument - _q(center)
        self.constraints.extend((value >= _q(f.lo) + z3.If(delta >= 0, _q(d.lo)*delta, _q(d.hi)*delta),
                                 value <= _q(f.hi) + z3.If(delta >= 0, _q(d.hi)*delta, _q(d.lo)*delta)))
        if lo < hi and name in {"exp", "log", "sqrt"}:
            left, right, slope = function(_IV(lo)), function(_IV(hi)), derivative(_IV(center))
            if name == "exp":
                secant = _q(left.hi) + (_q(right.hi)-_q(left.hi)) * (argument-_q(lo)) / (_q(hi)-_q(lo))
                tangent = _q(f.lo) + z3.If(delta >= 0, _q(slope.lo)*delta, _q(slope.hi)*delta)
                self.constraints.extend((value <= secant, value >= tangent))
            else:
                secant = _q(left.lo) + (_q(right.lo)-_q(left.lo)) * (argument-_q(lo)) / (_q(hi)-_q(lo))
                tangent = _q(f.hi) + z3.If(delta >= 0, _q(slope.hi)*delta, _q(slope.lo)*delta)
                self.constraints.extend((value >= secant, value <= tangent))
        return value

    def formula(self, node):
        _check_query_deadline()
        if isinstance(node, ast.Constant) and isinstance(node.value, bool): return z3.BoolVal(node.value)
        if isinstance(node, ast.BoolOp):
            return (z3.And if isinstance(node.op, ast.And) else z3.Or)(*[self.formula(a) for a in node.values])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not): return z3.Not(self.formula(node.operand))
        if isinstance(node, ast.Call) and node.func.id == "implies": return z3.Implies(*map(self.formula, node.args))
        if isinstance(node, ast.Compare):
            values = []
            for a, op, b in zip([node.left, *node.comparators], node.ops, node.comparators):
                a, b = self.term(a), self.term(b)
                values.append({ast.Lt: lambda: a < b, ast.LtE: lambda: a <= b, ast.Gt: lambda: a > b,
                               ast.GtE: lambda: a >= b, ast.Eq: lambda: a == b, ast.NotEq: lambda: a != b}[type(op)]())
            return z3.And(*values)
        raise _DomainError("expected a logical predicate")


def _model_point(model, variables):
    point = {}
    for s, v in variables.items():
        value = model.eval(v, model_completion=True)
        if z3.is_rational_value(value): point[s] = float(value.as_fraction())
        else: point[s] = float(value.approx(20).as_fraction())
    return point


def _halton(index: int, base: int) -> float:
    value, scale = 0.0, 1.0
    while index:
        index, digit = divmod(index, base)
        scale /= base; value += digit * scale
    return value


def _witness_probe(node, box, deadline):
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131)
    count, best, best_score = 0, None, -math.inf
    def score(n, point):
        if isinstance(n, ast.BoolOp): return (min if isinstance(n.op, ast.And) else max)(score(a, point) for a in n.values)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not): return -score(n.operand, point)
        if isinstance(n, ast.Call) and n.func.id == "implies": return max(-score(n.args[0], point), score(n.args[1], point))
        if isinstance(n, ast.Constant) and isinstance(n.value, bool): return 1.0 if n.value else -1.0
        margins = []
        for a, op, b in zip([n.left, *n.comparators], n.ops, n.comparators):
            a, b = _iv(a, point), _iv(b, point); delta = (a.lo/2+a.hi/2)-(b.lo/2+b.hi/2)
            margins.append(-delta if isinstance(op, (ast.Lt, ast.LtE)) else (delta if isinstance(op, (ast.Gt, ast.GtE)) else (-abs(delta) if isinstance(op, ast.Eq) else abs(delta))))
        return min(margins)
    def check(point):
        nonlocal count, best, best_score
        count += 1
        singleton = {s: (v, v) for s, v in point.items()}
        try:
            if _truth(node, singleton) is True: return point
            candidate_score = score(node, singleton)
            if candidate_score > best_score: best, best_score = point.copy(), candidate_score
        except (ValueError, OverflowError, ZeroDivisionError):
            pass
        return None
    for k in range(1, MAX_PROBES + 1):
        if time.monotonic() >= deadline: break
        point = {s: lo + (hi-lo)*_halton(k, primes[i]) for i, (s, (lo, hi)) in enumerate(box.items())}
        if (found := check(point)) is not None: return found, count
    if best is not None:
        initial_scale = MAX_PROBES**(-1/max(1, len(box)))
        for sweep in range(8):
            for s, (lo, hi) in box.items():
                for sign in (-1, 1):
                    if time.monotonic() >= deadline: return None, count
                    point = best.copy(); point[s] = min(hi, max(lo, point[s] + sign*(hi-lo)*initial_scale/2**sweep))
                    if (found := check(point)) is not None: return found, count
    return None, count


def _solve_query(expression: str, domain: Mapping[str, Sequence[float]], *, witness: bool = False,
                timeout_ms: int = SMT_TIMEOUT_MS, max_boxes: int = MAX_BOXES, _deadline: float | None = None,
                prefer_linear_outer: bool = False) -> QueryResult:
    """Four-tier solver; outer-relaxation SAT is never an exact SAT result."""
    if not 0 < timeout_ms <= SMT_TIMEOUT_MS or not 0 < max_boxes <= MAX_BOXES: raise ValueError("SMT budget outside deterministic caps")
    deadline = min(_deadline or math.inf, time.monotonic() + timeout_ms/1000)
    box, node = _bounds(domain), _parse(expression)
    if not _variables(node) <= box.keys(): raise ValueError("undeclared query variable")
    def conjuncts(current):
        if isinstance(current,ast.BoolOp) and isinstance(current.op,ast.And):
            return [part for child in current.values for part in conjuncts(child)]
        return [current]
    clauses = conjuncts(node)
    eliminated = []
    for clause in tuple(clauses):
        if not (isinstance(clause,ast.Compare) and len(clause.ops) == 1 and isinstance(clause.ops[0],ast.Eq)):
            continue
        left,right = clause.left,clause.comparators[0]
        if not isinstance(left,ast.Name) or left.id not in box or left.id in _variables(right) or _degree(right) <= 6:
            continue
        name = left.id
        lo,hi = box.pop(name)
        replacement = copy.deepcopy(right)
        class Define(ast.NodeTransformer):
            def visit_Call(self,current):
                current.args = [self.visit(a) for a in current.args]
                return current
            def visit_Name(self,current):
                return copy.deepcopy(replacement) if current.id == name else current
        clauses = [Define().visit(part) for part in clauses if part is not clause]
        clauses.append(_parse(f"{lo!r} <= ({ast.unparse(replacement)}) <= {hi!r}"))
        eliminated.append(f"{name} = {ast.unparse(replacement)}")
    if eliminated:
        reduced = _and([ast.unparse(part) for part in clauses])
        remaining = int((deadline-time.monotonic())*1000)
        if remaining <= 0: return QueryResult(status="UNKNOWN",z3_tier_used="UNKNOWN",reason="TIME_BUDGET_EXHAUSTED")
        result = _solve_query(reduced,box,witness=witness,timeout_ms=min(timeout_ms,remaining),max_boxes=max_boxes,_deadline=deadline,prefer_linear_outer=prefer_linear_outer)
        return result.model_copy(update={"witness":None,"exact_model":"; ".join(eliminated)+"; "+(result.exact_model or "")})
    class Fixed(ast.NodeTransformer):
        def visit_Call(self,current):
            current.args = [self.visit(arg) for arg in current.args]
            return current
        def visit_Name(self,current):
            if current.id not in box: return current
            lo,hi = box[current.id]
            return ast.copy_location(ast.Constant(lo),current) if lo == hi else current
    node = Fixed().visit(node)
    degree = _degree(node)
    if degree <= 1 or not prefer_linear_outer and degree <= 6 and len(_variables(node)) <= 8:
        tier = "QF_LRA" if degree <= 1 else "QF_NRA"
        tr = _Translator(box, False)
        formula = tr.formula(node)
        solver = z3.SolverFor(tier)
        remaining = max(0, int((deadline-time.monotonic())*1000))
        if remaining:
            solver.set(timeout=min(500 if tier == "QF_LRA" else 1200, remaining), rlimit=1_000_000 if tier == "QF_LRA" else 5_000_000)
            solver.add(*tr.constraints, formula)
            answer = solver.check()
            if answer == z3.unsat: return QueryResult(status="UNSAT", z3_tier_used=tier)
            if answer == z3.sat:
                model = solver.model()
                return QueryResult(status="SAT", z3_tier_used=tier, witness=_model_point(model, tr.vars), exact_model=str(model))
    queue, evaluated, unresolved = [box], 0, False
    outer_deadline = deadline - (min(0.3, timeout_ms/2000) if witness else 0)
    while queue and evaluated < max_boxes and time.monotonic() < outer_deadline:
        current = queue.pop(); evaluated += 1
        candidate = None
        try:
            tr = _Translator(current, True); formula = tr.formula(node)
            solver = z3.SolverFor("QF_LRA")
            solver.set(timeout=max(1, min(100, int((outer_deadline-time.monotonic())*1000))), rlimit=1_000_000)
            solver.add(*tr.constraints, formula); answer = solver.check()
            if answer == z3.unsat: continue
            if answer == z3.sat:
                candidate = _model_point(solver.model(), tr.vars)
                if _truth(node, {s: (x, x) for s, x in candidate.items()}) is True:
                    return QueryResult(status="SAT", z3_tier_used="MCCORMICK_QF_LRA", witness=candidate, boxes_evaluated=evaluated)
        except (ValueError, OverflowError, ZeroDivisionError):
            pass
        if not current: unresolved = True; continue
        split = max(current, key=lambda s: (current[s][1]-current[s][0], s))
        lo, hi = current[split]; mid = lo/2 + hi/2
        if not lo < mid < hi: unresolved = True; continue
        left, right = current.copy(), current.copy(); left[split], right[split] = (lo, mid), (mid, hi)
        queue.extend((right, left))
    if not queue and not unresolved:
        return QueryResult(status="UNSAT", z3_tier_used="MCCORMICK_QF_LRA", boxes_evaluated=evaluated)
    if witness:
        found, probes = _witness_probe(node, box, deadline)
        if found is not None: return QueryResult(status="SAT", z3_tier_used="WITNESS_PROBE", witness=found, boxes_evaluated=evaluated, probes_evaluated=probes)
    else: probes = 0
    return QueryResult(status="UNKNOWN", z3_tier_used="UNKNOWN", boxes_evaluated=evaluated, probes_evaluated=probes,
                       reason="TIME_OR_SPATIAL_BUDGET_EXHAUSTED_OR_UNSUPPORTED_DOMAIN")


def solve_query(expression: str, domain: Mapping[str, Sequence[float]], *, witness: bool = False,
                timeout_ms: int = SMT_TIMEOUT_MS, max_boxes: int = MAX_BOXES, _deadline: float | None = None,
                prefer_linear_outer: bool = False) -> QueryResult:
    """Run a shared 1500 ms cooperative budget, including native witness work.

    Parsing, one native call, solver teardown and OS scheduling are not preemptible.
    """
    deadline = min(_deadline or math.inf, time.monotonic()+timeout_ms/1000)
    token = _QUERY_DEADLINE.set(deadline)
    try:
        if not 0 < timeout_ms <= SMT_TIMEOUT_MS or not 0 < max_boxes <= MAX_BOXES: raise ValueError("SMT budget outside deterministic caps")
        _check_query_deadline()
        return _solve_query(expression, domain, witness=witness, timeout_ms=timeout_ms, max_boxes=max_boxes, _deadline=deadline,prefer_linear_outer=prefer_linear_outer)
    except _DeadlineExceeded:
        return QueryResult(status="UNKNOWN", z3_tier_used="UNKNOWN", reason="TOTAL_TIME_BUDGET_EXHAUSTED")
    finally:
        _QUERY_DEADLINE.reset(token)


def _and(expressions: Sequence[str]) -> str:
    return " and ".join(f"({e})" for e in expressions) or "True"


def _worst_tier(results: Sequence[QueryResult]) -> Tier:
    order = ("QF_LRA", "QF_NRA", "MCCORMICK_QF_LRA", "WITNESS_PROBE", "UNKNOWN")
    return max((r.z3_tier_used for r in results), key=order.index, default="UNKNOWN")


def _classify(ihcg: IhcgGraph, claims: Sequence[str], model_eqs: Sequence[str] | None) -> ContractVacuityAudit:
    deadline = time.monotonic() + SMT_TIMEOUT_MS/1000
    def conjuncts(node):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return [part for child in node.values for part in conjuncts(child)]
        return [ast.unparse(node)]
    claims = tuple(part for claim in claims for part in conjuncts(_parse(claim)))
    base = _and(ihcg.environmental_assumptions)
    hypothesis = _and(claims)
    queries = {}
    def query(name, clauses, witness=True):
        queries[name] = solve_query(_and(clauses), ihcg.omega, witness=witness, _deadline=deadline)
    query("Q1", [base]); query("Q2", [base, hypothesis]); query("Q3", [base, f"not ({hypothesis})"])
    redundant = []
    for i, claim in enumerate(claims):
        key = f"Q4_{i}"; query(key, [base, *claims[:i], *claims[i+1:], f"not ({claim})"], False)
        if queries[key].status == "UNSAT": redundant.append(str(i))
    implications = []
    for claim in claims:
        implications.extend(n for n in ast.walk(_parse(claim)) if isinstance(n, ast.Call) and n.func.id == "implies")
    for i, implication in enumerate(implications): query(f"Q5_{i}", [base, ast.unparse(implication.args[0])])
    q1, q2, q3 = (queries[k].status for k in ("Q1", "Q2", "Q3"))
    if q1 == "UNSAT" or any(r.status == "UNSAT" for k, r in queries.items() if k.startswith("Q5_")): classification = "VACUOUS"
    elif q1 == "SAT" and q3 == "UNSAT": classification = "TAUTOLOGICAL"
    elif q1 == "SAT" and q2 == "UNSAT": classification = "INFEASIBLE"
    elif q1 == q2 == q3 == "SAT" and all(r.status != "UNKNOWN" for r in queries.values()): classification = "CONSTRAINING"
    else: classification = "UNKNOWN"
    if classification == "CONSTRAINING" and model_eqs is not None:
        query("MODEL_FEASIBILITY", [base, *model_eqs, hypothesis])
        if queries["MODEL_FEASIBILITY"].status == "UNSAT": classification = "INFEASIBLE_UNDER_MODEL"
        elif queries["MODEL_FEASIBILITY"].status == "UNKNOWN": classification = "UNKNOWN"
    return ContractVacuityAudit(classification=classification, queries=queries, redundant_conjuncts=tuple(redundant), z3_tier_used=_worst_tier(list(queries.values())))


def classify_contract_vacuity(ihcg: IhcgGraph, model_eqs: Sequence[str] | None = None) -> ContractVacuityAudit:
    ihcg.assert_sealed()
    return _classify(ihcg, tuple(o.expression for o in ihcg.obligations if o.obligation_class == "HYPOTHESIS_TARGET"), model_eqs)


def verify_entailment_lock(ihcg: IhcgGraph, compiled_contracts: Sequence[str]) -> EntailmentAudit:
    ihcg.assert_sealed()
    deadline = time.monotonic() + SMT_TIMEOUT_MS/1000
    base = [*ihcg.environmental_assumptions, *compiled_contracts]
    affine_compiled = _degree(_parse(_and(base))) <= 1
    queries = {o.id: solve_query(_and([*base, f"not ({o.expression})"]), ihcg.omega, _deadline=deadline,
        prefer_linear_outer=affine_compiled and _degree(_parse(o.expression)) > 1) for o in ihcg.obligations}
    feasibility = solve_query(_and(base), ihcg.omega, witness=True, _deadline=deadline)
    locked = all(q.status == "UNSAT" for q in queries.values()) and feasibility.status == "SAT"
    queries["COMPILED_FEASIBILITY"] = feasibility
    return EntailmentAudit(entailment_locked=locked, queries=queries,
                           z3_tier_used=_worst_tier(list(queries.values())),
                           counterexample_if_weakened=next((q.witness for q in queries.values() if q.status == "SAT"), None))


def verify_claim_nontriviality(ihcg: IhcgGraph, final_claims: Sequence[str]) -> NontrivialityAudit:
    """Q1-Q5 without M; novelty cannot rescue an environment-only deduction."""
    ihcg.assert_sealed()
    if not final_claims: raise ValueError("a nonempty final claim is required")
    audit = _classify(ihcg, tuple(final_claims), None)
    reason = None if audit.passed else ("TRIVIAL_DEDUCIBLE_WITHOUT_MODEL" if audit.queries["Q1"].status == "SAT" and audit.queries["Q3"].status == "UNSAT" else f"FINAL_CLAIM_{audit.classification}")
    return NontrivialityAudit(passed=audit.passed, claim=tuple(final_claims), contract_audit=audit, failure_reason=reason)


def audit_purity(wires: Mapping[str, Any], writes: Sequence[Mapping[str, Any]], *,
                 projection_drifts: Sequence[float], stabilization_hits: Sequence[bool], domain_tag: str) -> PurityAudit:
    if len(projection_drifts) != len(stabilization_hits) or any(not math.isfinite(d) or d < 0 for d in projection_drifts):
        raise ValueError("finite per-step drift and stabilization telemetry required")
    if any(type(hit) is not bool for hit in stabilization_hits): raise ValueError("boolean stabilization telemetry required")
    report = json.loads(_native.py_verify_observational_purity(_json(wires), _json(writes)))
    stabilized = sum(stabilization_hits)
    projected = sum(d > 1e-12 and not hit for d, hit in zip(projection_drifts, stabilization_hits))
    total = len(projection_drifts)
    return PurityAudit(static_purity_passed=report["valid"], violations=tuple(v["rule"] for v in report.get("violations", [])),
                       runtime_stabilizer_hits=stabilized, runtime_max_proj_residual=max(projection_drifts, default=0.0),
                       total_steps=total, preserved_steps=total-projected-stabilized, projected_steps=projected,
                       stabilized_steps=stabilized, enforcement_ratio=(projected+stabilized)/max(1, total), domain_tag=domain_tag)


def quadratic_peak(times: Sequence[float], values: Sequence[float], *, minimum: bool = False) -> tuple[float, float, float]:
    """Return extremum value, time, and |quadratic curvature| h^2 / 8."""
    if len(times) != len(values) or len(values) < 3 or any(not math.isfinite(x) for x in (*times, *values)):
        raise ValueError("at least three finite trajectory samples required")
    if any(b <= a for a, b in zip(times, times[1:])): raise ValueError("strictly increasing times required")
    sign = -1 if minimum else 1
    k = max(range(len(values)), key=lambda i: sign*values[i])
    if k in (0, len(values)-1): return float(values[k]), float(times[k]), 0.0
    x0, x1, x2 = times[k-1:k+2]; y0, y1, y2 = (sign*v for v in values[k-1:k+2])
    a = ((y2-y1)/(x2-x1)-(y1-y0)/(x1-x0))/(x2-x0)
    b = (y1-y0)/(x1-x0)+a*(x1-x0)
    if a >= 0: raise ValueError("extremum does not have negative interpolated curvature")
    offset = -b/(2*a)
    if not x0-x1 <= offset <= x2-x1: raise ValueError("interpolated peak lies outside its local bracket")
    h = max(x1-x0, x2-x1)
    return sign*(y1+b*offset+a*offset*offset), x1+offset, abs(2*a)*h*h/8


def roache_gci(fine: float, medium: float, coarse: float, *, refinement_ratio: float = 2.0,
               nominal_order: float = 4.0, eps_peak_loc: float = 0.0) -> GciAudit:
    if not all(map(math.isfinite, (fine, medium, coarse, refinement_ratio, nominal_order, eps_peak_loc))) or refinement_ratio <= 1.3 or nominal_order < 0.5 or eps_peak_loc < 0:
        raise ValueError("invalid three-grid convergence inputs")
    e21, e32 = medium-fine, coarse-medium
    floor = 100*sys.float_info.epsilon*max(1, abs(fine))
    if abs(e21) <= floor and abs(e32) <= floor:
        return GciAudit(passed=True, fine_value=fine, apparent_order_p=None, gci_fine_abs=floor,
                        eps_peak_loc=eps_peak_loc, safety_factor=3.0, oscillatory=False, reason="ROUNDING_FLOOR")
    if e21 == 0 or e32 == 0:
        return GciAudit(passed=False, fine_value=fine, apparent_order_p=None, gci_fine_abs=max(abs(e21), abs(e32), floor),
                        eps_peak_loc=eps_peak_loc, safety_factor=3.0, oscillatory=False, reason="NON_ASYMPTOTIC_CONVERGENCE")
    order = math.log(abs(e32/e21))/math.log(refinement_ratio)
    oscillatory = e32/e21 < 0
    suspect = oscillatory or abs(order-nominal_order)/nominal_order > 0.35
    effective = min(1.0, max(0.5, order)) if suspect else min(nominal_order, max(0.5, order))
    factor = 3.0 if suspect else 1.25
    error = max(floor, factor*abs(e21)/(refinement_ratio**effective-1))
    return GciAudit(passed=order > 0, fine_value=fine, apparent_order_p=order, gci_fine_abs=error,
                    eps_peak_loc=eps_peak_loc, safety_factor=factor, oscillatory=oscillatory,
                    reason=None if order > 0 else "NONCONVERGENT_GRIDS")


def convergence_reliability(primary: GciAudit, secondary: GciAudit) -> tuple[float, float]:
    delta = abs(primary.fine_value-secondary.fine_value)
    floor = 100*sys.float_info.epsilon*max(1, abs(primary.fine_value))
    return delta/max(primary.gci_fine_abs+secondary.gci_fine_abs, floor), delta/max(primary.gci_fine_abs, floor)


def dual_scheme_fidelity(primary: GciAudit, secondary: GciAudit, *, algebraic_equiv_passed: bool,
                         si_and_semantic_tag_passed: bool, physical_checks_passed: bool) -> FidelityAudit:
    rho, raw = convergence_reliability(primary, secondary)
    delta = abs(primary.fine_value-secondary.fine_value)
    rounding = 100*sys.float_info.epsilon*max(1, abs(primary.fine_value))
    disc = max(primary.gci_fine_abs, secondary.gci_fine_abs, delta, primary.eps_peak_loc, secondary.eps_peak_loc, rounding)
    reason = ("CONVERGENCE_ESTIMATOR_UNRELIABLE" if rho > 1 else
              "ALGEBRAIC_MISMATCH" if not algebraic_equiv_passed else "SI_OR_SEMANTIC_MISMATCH" if not si_and_semantic_tag_passed else
              "UNPHYSICAL_ARTIFACT" if not physical_checks_passed else "CONVERGENCE_FAILED" if not primary.passed or not secondary.passed else None)
    return FidelityAudit(passed=reason is None, algebraic_equiv_passed=algebraic_equiv_passed,
                         si_and_semantic_tag_passed=si_and_semantic_tag_passed, convergence_passed=primary.passed and secondary.passed and rho <= 1,
                         physical_checks_passed=physical_checks_passed, apparent_order_p=primary.apparent_order_p,
                         gci_fine_abs=primary.gci_fine_abs, secondary_gci_abs=secondary.gci_fine_abs, multi_scheme_delta=delta,
                         convergence_reliability_ratio=rho, scheme_delta_to_primary_gci=raw,
                         eps_peak_loc=max(primary.eps_peak_loc, secondary.eps_peak_loc), epsilon_disc=disc,
                         epsilon_round=rounding, epsilon_total=disc+rounding, failure_reason=reason)


def statistical_reliability(replicate_blocks: Sequence[Sequence[float]], stream_keys: Sequence[tuple[int, int, int]], *,
                            variance_ratio_limit: float = 4.0) -> StatisticalReliabilityAudit:
    """Conservative split-half/batch-means diagnostic, not a distributional proof."""
    if len(replicate_blocks) < 4 or len(replicate_blocks) != len(stream_keys) or variance_ratio_limit <= 1:
        raise ValueError("four independent replicate blocks and valid variance threshold required")
    hashes = [invar_splitmix64_stateless(*key) for key in stream_keys]
    if len(set(hashes)) != len(hashes): raise ValueError("replicate streams must be distinct SplitMix64 counter blocks")
    if any(len(b) < 4 or any(not math.isfinite(x) for x in b) for b in replicate_blocks): raise ValueError("finite replicate blocks of at least four observations required")
    sizes = {len(b) for b in replicate_blocks}
    if len(sizes) != 1: raise ValueError("replicate blocks must have equal sample counts")
    halves = [(b[:len(b)//2], b[len(b)//2:]) for b in replicate_blocks]
    first = statistics.mean(statistics.variance(a) for a, _ in halves)
    second = statistics.mean(statistics.variance(b) for _, b in halves)
    means = [statistics.mean(b) for b in replicate_blocks]
    half = len(means)//2
    va, vb = statistics.variance(means[:half]), statistics.variance(means[half:])
    floor = (100*sys.float_info.epsilon*max(1, *(abs(x) for b in replicate_blocks for x in b)))**2
    ratio = max(max(first, second, floor)/max(min(first, second), floor), max(va, vb, floor)/max(min(va, vb), floor))
    left = [x for a, _ in halves for x in a]; right = [x for _, b in halves for x in b]
    split_drift = abs(statistics.mean(left)-statistics.mean(right))/max(math.sqrt(floor), 5*math.sqrt(statistics.variance(left)/len(left)+statistics.variance(right)/len(right)))
    batch_drift = abs(statistics.mean(means[:half])-statistics.mean(means[half:]))/max(math.sqrt(floor), 5*math.sqrt(va/half+vb/(len(means)-half)))
    drift = max(split_drift, batch_drift)
    passed = ratio <= variance_ratio_limit and drift <= 1
    return StatisticalReliabilityAudit(passed=passed, variance_ratio=ratio, mean_drift_ratio=drift,
                                        stream_keys=tuple(stream_keys), failure_reason=None if passed else "STATISTICAL_ESTIMATOR_UNRELIABLE")


def run_mc_replicate_blocks(sampler: Callable[[float], float], *, seed: int, blocks: int = 8, block_size: int = 128) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[int, int, int], ...]]:
    if not 4 <= blocks <= 256 or not 4 <= block_size <= 65536: raise ValueError("MC replicate budget exceeded")
    deadline = time.monotonic()+2
    output, keys = [], []
    for block in range(blocks):
        keys.append((seed, block*block_size, 0)); values = []
        for j in range(block_size):
            if time.monotonic() >= deadline: raise TimeoutError("MC execution budget exhausted")
            value = float(sampler(invar_rand_u01(seed, block*block_size+j, 0)))
            if not math.isfinite(value): raise ValueError("nonfinite MC observation")
            values.append(value)
        output.append(tuple(values))
    return tuple(output), tuple(keys)


class GridRun(StrictModel):
    scheme: Literal["RK4", "HEUN"]
    times: tuple[float, ...]
    values: tuple[float, ...]
    physical_checks: dict[str, bool]
    stabilizer_hits: int = Field(default=0, ge=0)


def integrate_ode(rhs: Callable[[float, tuple[float, ...]], Sequence[float]], initial: Sequence[float], *,
                  start: float, stop: float, steps: int, scheme: Literal["RK4", "HEUN"] = "RK4") -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Independent reference schemes; external RHS calls must themselves be bounded."""
    if not 0 < steps <= 1_000_000 or scheme not in {"RK4", "HEUN"} or not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError("invalid integration grid")
    h = (stop-start)/steps; y = tuple(map(float, initial)); times, trajectory = [start], [y]
    if not y or any(not math.isfinite(x) for x in y): raise ValueError("finite nonempty state required")
    deadline = time.monotonic()+2
    def evaluate(t, state):
        result = tuple(map(float, rhs(t, state)))
        if len(result) != len(y) or any(not math.isfinite(x) for x in result): raise ValueError("invalid RHS result")
        return result
    def shift(state, slope, factor): return tuple(a+factor*b for a, b in zip(state, slope))
    for i in range(steps):
        if time.monotonic() >= deadline: raise TimeoutError("ODE execution budget exhausted")
        t = start+i*h; k1 = evaluate(t, y)
        if scheme == "HEUN":
            k2 = evaluate(t+h, shift(y, k1, h)); y = tuple(a+h*(b+c)/2 for a, b, c in zip(y, k1, k2))
        else:
            k2 = evaluate(t+h/2, shift(y, k1, h/2)); k3 = evaluate(t+h/2, shift(y, k2, h/2)); k4 = evaluate(t+h, shift(y, k3, h))
            y = tuple(a+h*(b+2*c+2*d+e)/6 for a, b, c, d, e in zip(y, k1, k2, k3, k4))
        if any(not math.isfinite(x) for x in y): raise ValueError("nonfinite trajectory")
        times.append(start+(i+1)*h); trajectory.append(y)
    return tuple(times), tuple(trajectory)


class JacobianAudit(StrictModel):
    passed: bool
    samples_checked: int
    maximum_inflation: float = Field(ge=0)
    failure_reason: str | None = None


def cross_check_krawczyk_jacobian(executable: Callable[[Sequence[float]], Sequence[float]], center: Sequence[float],
                                 krawczyk: Mapping[str, Any], third_derivative_bounds: Sequence[Sequence[float]], *,
                                 executable_rounding_bound: float, max_corners: int = 8) -> JacobianAudit:
    """Require [J_fd-eps_fd,J_fd+eps_fd] subset [J_g](Z), not mere overlap.

    Third partial derivative and executable absolute rounding bounds are caller
    obligations over Z, including FD stencils. They must not be fitted to FD data.
    """
    n = len(center)
    if not 0 < n <= 32 or not 2 <= max_corners <= 8 or executable_rounding_bound < 0 or not math.isfinite(executable_rounding_bound): raise ValueError("invalid FD certification budget")
    if len(third_derivative_bounds) != n or any(len(row) != n or any(x < 0 or not math.isfinite(x) for x in row) for row in third_derivative_bounds): raise ValueError("certified third partial bounds required")
    jac, root_box, original = krawczyk["interval_jacobian"], krawczyk["krawczyk_box"], krawczyk["box_z"]
    if any(len(matrix) != n for matrix in (jac, root_box, original)) or any(len(row) != n for row in jac): raise ValueError("Krawczyk dimension mismatch")
    intervals = [*original, *root_box, *(entry for row in jac for entry in row)]
    if any(not math.isfinite(v["lo"]) or not math.isfinite(v["hi"]) or v["lo"] > v["hi"] for v in intervals): raise ValueError("finite ordered Krawczyk enclosures required")
    if any(not original[j]["lo"] <= root_box[j]["lo"] <= root_box[j]["hi"] <= original[j]["hi"] or not original[j]["lo"] <= center[j] <= original[j]["hi"] for j in range(n)):
        raise ValueError("FD stencil anchors must be inside Z")
    points = [tuple(map(float, center))]
    masks = list(dict.fromkeys([0, (1 << n)-1, *(invar_splitmix64_stateless(0, i, 0) & ((1 << n)-1) for i in range(8))]))[:min(max_corners, 1 << n)]
    points.extend(tuple(root_box[j]["hi" if mask & (1 << j) else "lo"] for j in range(n)) for mask in masks)
    maximum = 0.0; deadline = time.monotonic()+2
    for index, point in enumerate(points):
        for j in range(n):
            if time.monotonic() >= deadline: raise TimeoutError("FD cross-check budget exhausted")
            h = min(sys.float_info.epsilon**(1/3)*max(1, abs(point[j])), (point[j]-original[j]["lo"])/2, (original[j]["hi"]-point[j])/2)
            if h <= 0: return JacobianAudit(passed=False, samples_checked=index, maximum_inflation=maximum, failure_reason="JACOBIAN_ENCLOSURE_MISMATCH")
            plus, minus = list(point), list(point); plus[j] += h; minus[j] -= h
            fp, fm = tuple(executable(plus)), tuple(executable(minus))
            if len(fp) != n or len(fm) != n or any(not math.isfinite(x) for x in (*fp, *fm)): raise ValueError("invalid executable residual")
            width = plus[j]-minus[j]
            for i in range(n):
                fd = (fp[i]-fm[i])/width
                eps = third_derivative_bounds[i][j]*max(plus[j]-point[j], point[j]-minus[j])**2/6
                eps += (2*executable_rounding_bound+8*sys.float_info.epsilon*max(1, abs(fp[i]), abs(fm[i])))/width
                # Rounding of the two stencil coordinates makes the central
                # formula slightly asymmetric; bound its first-derivative term.
                eps += max(abs(jac[i][j]["lo"]), abs(jac[i][j]["hi"])) * abs((plus[j]+minus[j])-2*point[j])/width
                maximum = max(maximum, eps)
                if fd-eps < jac[i][j]["lo"] or fd+eps > jac[i][j]["hi"]:
                    return JacobianAudit(passed=False, samples_checked=index+1, maximum_inflation=maximum, failure_reason="JACOBIAN_ENCLOSURE_MISMATCH")
    return JacobianAudit(passed=True, samples_checked=len(points), maximum_inflation=maximum)


def _csel(node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float): return {"kind": "literal", "value": float(node.value)}
    if isinstance(node, ast.Name): return {"kind": "symbol", "name": node.id}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return {"kind": "unary", "op": "-" if isinstance(node.op, ast.USub) else "+", "arg": _csel(node.operand)}
    if isinstance(node, ast.BinOp):
        return {"kind": "binary", "op": {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}[type(node.op)], "left": _csel(node.left), "right": _csel(node.right)}
    if isinstance(node, ast.Call) and node.func.id != "implies": return {"kind": "call", "func": node.func.id, "args": [_csel(a) for a in node.args]}
    raise ValueError("arithmetic CSEL required")


def _native_equation_domain_valid(node: ast.AST, box: Bounds) -> bool:
    """Restrict proof equations to domains where native healing is inactive."""
    try:
        for n in ast.walk(node):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
                d = _iv(n.right, box)
                if d.lo < 1e-12 and d.hi > -1e-12: return False
                value = _iv(n, box)
                if max(abs(value.lo), abs(value.hi)) > 1e300: return False
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) or isinstance(n, ast.Call) and n.func.id == "pow":
                a, b = (n.left, n.right) if isinstance(n, ast.BinOp) else n.args
                exponent = _fraction(b)
                if exponent is None or exponent.denominator != 1:
                    x, y = _iv(a, box), _iv(b, box)
                    # Near-integer exponent snapping is a healing operation.
                    if y.lo != y.hi or abs(y.lo-round(y.lo)) < 1e-12 or x.lo < 1e-30: return False
                    power = y * x.op("log")
                    if power.lo < -80 or power.hi > 80: return False
                value = _iv(n, box)
                if max(abs(value.lo), abs(value.hi)) > 1e300: return False
            if isinstance(n, ast.Call):
                name = n.func.id
                args = [_iv(a, box) for a in n.args]
                if name in {"exp", "logistic_sigmoid"} and (args[0].lo < -80 or args[0].hi > 80): return False
                if name == "log" and args[0].lo < 1e-30: return False
                if name == "sqrt" and args[0].lo < 0: return False
                if name == "hill_eq":
                    x, k, exponent = args
                    if x.lo <= 0 or k.lo < 1e-15 or exponent.lo != exponent.hi or exponent.lo != round(exponent.lo) or not 0 <= exponent.lo <= 16: return False
                    denominator = _iv(_parse(f"({ast.unparse(n.args[0])})**{int(exponent.lo)} + ({ast.unparse(n.args[1])})**{int(exponent.lo)}"), box)
                    if denominator.lo < 1e-12 or denominator.hi > 1e300: return False
                if name == "michaelis_menten" and (args[0].lo < 0 or args[2].lo < 1e-15 or (args[0]+args[2]).lo < 1e-12): return False
        return True
    except (ValueError, OverflowError, ZeroDivisionError):
        return False


def _equation_arithmetic(expression):
    node = _parse(expression)
    comparisons = [n for n in ast.walk(node) if isinstance(n, ast.Compare)]
    if not comparisons: return [node]
    return [ast.BinOp(left=a, op=ast.Sub(), right=b)
            for comparison in comparisons
            for a,b in zip([comparison.left,*comparison.comparators],comparison.comparators)]


def certify_implementation_fidelity(ihcg: IhcgGraph, uast: Mapping[str, Any], sandbox_runner: Callable[..., Any]) -> FidelityAudit:
    """Runner adapter: six GridRuns, MC blocks, or a fixed-point residual oracle.

    The runner must enforce its timeout inside the compiled worker. Python cannot
    preempt an arbitrary callback. No state-machine integration is performed here.
    """
    ihcg.assert_sealed(); deadline = time.monotonic()+2
    equations = uast.get("equations", {})
    alg = set(equations) == set(ihcg.equations) and bool(equations)
    for name, expected in ihcg.equations.items():
        if name not in equations: continue
        if any(isinstance(n, ast.Compare) for n in ast.walk(_parse(expected.expression))):
            alg = alg and ast.dump(_parse(expected.expression)) == ast.dump(_parse(equations[name]))
            continue
        result = json.loads(_native.py_check_algebraic_equivalence(_json(_csel(_parse(expected.expression))), _json(_csel(_parse(equations[name]))), iter_limit=30, node_limit=10000, time_limit_ms=5))
        alg = alg and result["equivalent"]
    quantities = [{"symbol": s.symbol, "si_dimension": s.si_dim, "interval": ihcg.omega[s.symbol],
                   "is_state_variable": s.role == "DYNAMIC_STATE", "semantic_tag": s.semantic_tag} for s in ihcg.slots]
    arithmetic = {}
    owners = {}
    for name, expression in equations.items():
        terms = _equation_arithmetic(expression)
        for i, term in enumerate(terms):
            key = f"{name}:{i}"
            arithmetic[key] = _csel(term)
            owners[key] = name
    reports = json.loads(_native.PyHolographicHin(_json(quantities)).verify_equations(_json(arithmetic)))
    dim = reports["all_valid"] and set(equations) == set(ihcg.equations)
    for report in reports["equations"]:
        expected = ihcg.equations.get(owners[report["equation_name"]])
        dim = dim and expected is not None and tuple(report["output_dimensions"]) == expected.si_dim and report["output_semantic_tag"] == expected.semantic_tag and not report["dimensional_warnings"]
    parameters = uast.get("parameters", {})
    locked = all(parameters.get(k) == v for k, v in ihcg.control_values.items())
    def failed(reason):
        return FidelityAudit(passed=False, algebraic_equiv_passed=bool(alg), si_and_semantic_tag_passed=bool(dim), convergence_passed=False,
                             physical_checks_passed=False, epsilon_disc=0.0, epsilon_total=0.0, failure_reason=reason)
    if not locked: return failed("CONTROL_PARAM_CHANGED")
    if not alg or not dim: return failed("ALGEBRAIC_MISMATCH" if not alg else "SI_OR_SEMANTIC_MISMATCH")
    if not all(_native_equation_domain_valid(term, ihcg.omega) for e in ihcg.equations.values() for term in _equation_arithmetic(e.expression)):
        return failed("NATIVE_HEALING_DOMAIN_MISMATCH")
    if ihcg.archetype in {"ODE_PDE", "PARAM_SWEEP"}:
        step = float(uast["step_size"])
        if step <= 0 or not math.isfinite(step): raise ValueError("positive base step size required")
        audits, physical = [], True
        guard_ids = {o.id for o in ihcg.obligations if o.obligation_class == "CONSERVATION_GUARD"}
        for scheme, nominal in (("RK4", 4.0), ("HEUN", 2.0)):
            values, peaks = [], []
            for factor in (0.5, 1, 2):
                remaining = int((deadline-time.monotonic())*1000)
                if remaining <= 0: return failed("EXECUTION_BUDGET_EXHAUSTED")
                run = sandbox_runner(copy.deepcopy(dict(uast)), scheme=scheme, step_size=step*factor, timeout_ms=remaining)
                if not isinstance(run, GridRun) or run.scheme != scheme or len(run.times) != len(run.values) or len(run.values) < 3:
                    raise ValueError("runner must return a tagged, nonempty GridRun")
                gaps = [b-a for a, b in zip(run.times, run.times[1:])]
                if any(not math.isclose(gap, step*factor, rel_tol=1e-9, abs_tol=1e-14) for gap in gaps): raise ValueError("runner grid does not match requested refinement")
                physical = physical and guard_ids <= run.physical_checks.keys() and all(run.physical_checks.values()) and run.stabilizer_hits == 0
                observable = uast.get("observable", "final")
                if observable in {"max", "min"}:
                    value, _, peak = quadratic_peak(run.times, run.values, minimum=observable == "min")
                elif observable == "final": value, peak = run.values[-1], 0.0
                else: raise ValueError("unsupported trajectory functional")
                values.append(value); peaks.append(peak)
            audits.append(roache_gci(*values, nominal_order=nominal, eps_peak_loc=peaks[0]))
        return dual_scheme_fidelity(*audits, algebraic_equiv_passed=True, si_and_semantic_tag_passed=True, physical_checks_passed=bool(physical))
    if ihcg.archetype == "MONTE_CARLO":
        evidence = sandbox_runner(copy.deepcopy(dict(uast)), mode="MONTE_CARLO", timeout_ms=max(1, int((deadline-time.monotonic())*1000)))
        reliability = statistical_reliability(evidence["replicate_blocks"], evidence["stream_keys"])
        error = float(evidence["epsilon_stat"])
        return FidelityAudit(passed=reliability.passed and evidence["physical_checks_passed"], algebraic_equiv_passed=True, si_and_semantic_tag_passed=True,
                             convergence_passed=reliability.passed, physical_checks_passed=evidence["physical_checks_passed"], epsilon_disc=0.0,
                             epsilon_stat=error, epsilon_total=error, statistical_audit=reliability, failure_reason=reliability.failure_reason)
    center, radius = uast["center"], float(uast["radius"])
    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node): return ast.Constant(value=parameters[node.id]) if node.id in parameters else node
    expressions = [_json(_csel(Substitute().visit(_parse(equations[k])))) for k in sorted(equations)]
    certificate = json.loads(_native.py_certify_krawczyk_fixed_point(expressions, list(center), radius))
    if not certificate["certified"]: return failed("KRAWCZYK_INCONCLUSIVE")
    audit = cross_check_krawczyk_jacobian(lambda point: sandbox_runner(copy.deepcopy(dict(uast)), point=point, timeout_ms=max(1, int((deadline-time.monotonic())*1000))),
                                        center, certificate, uast["third_derivative_bounds"], executable_rounding_bound=uast["executable_rounding_bound"])
    error = max(c["hi"]-c["lo"] for c in certificate["krawczyk_box"])
    return FidelityAudit(passed=audit.passed, algebraic_equiv_passed=True, si_and_semantic_tag_passed=True, convergence_passed=audit.passed,
                         physical_checks_passed=True, epsilon_disc=error, epsilon_total=error, failure_reason=audit.failure_reason)


class NullBaselineUast(StrictModel):
    uast: dict[str, Any]
    parameter_overrides: dict[str, float]
    description: str
    stage_idx: int = Field(ge=1)
    foundation_digest: str | None = None


def synthesize_null_baseline(uast: Mapping[str, Any], ihcg: IhcgGraph, stage_idx: int) -> NullBaselineUast:
    ihcg.assert_sealed()
    if stage_idx < 1: raise ValueError("stages are numbered from one")
    if stage_idx >= 2:
        foundation = uast.get("pinned_foundation")
        if not foundation or foundation.get("ihcg_digest") != ihcg.parent_ihcg_digest or not ihcg.parent_ihcg_digest:
            raise ValueError("a pinned foundation matching the parent IHCG is required")
        return NullBaselineUast(uast=copy.deepcopy(foundation["uast"]), parameter_overrides={}, description="Pinned prior-stage foundation", stage_idx=stage_idx, foundation_digest=ihcg.parent_ihcg_digest)
    if not ihcg.mechanism_parameters: raise ValueError("no declared mechanism is available for a deterministic null intervention")
    return NullBaselineUast(uast=copy.deepcopy(dict(uast)), parameter_overrides={k: 0.0 for k in ihcg.mechanism_parameters},
                            description="Zero declared mechanism controls; preserve all other parameters and random streams", stage_idx=stage_idx)


class SignificanceResult(StrictModel):
    observables: tuple[float, ...]
    fidelity: FidelityAudit
    null_baseline_description: str = "Declared null model"
    paired_samples: tuple[float, ...] = ()
    stream_keys: tuple[tuple[int, int, int], ...] = ()
    bernoulli_indicators: bool = False
    valid_domain: Bounds | None = None
    boundary: BoundaryCertificate | None = None
    pareto_points: tuple[tuple[float, ...], ...] = ()
    objective_errors: tuple[tuple[float, ...], ...] = ()
    krawczyk: dict[str, Any] | None = None
    stability_matrix: tuple[tuple[tuple[float, float], ...], ...] | None = None
    lyapunov_t: float | None = None
    lyapunov_2t: float | None = None
    lyapunov_disc_error: float = Field(default=0, ge=0)


def clopper_pearson_interval(successes: int, trials: int, *, confidence: float = 0.99) -> tuple[float, float]:
    """Two-sided exact binomial coverage, modulo SciPy beta quantile evaluation."""
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials or trials < 1 or not 0 < confidence < 1:
        raise ValueError("invalid binomial interval inputs")
    tail = (1-confidence)/2
    lower = 0.0 if successes == 0 else max(0.0, math.nextafter(float(beta.ppf(tail, successes, trials-successes+1)), -math.inf))
    upper = 1.0 if successes == trials else min(1.0, math.nextafter(float(beta.ppf(1-tail, successes+1, trials-successes)), math.inf))
    return lower, upper


def domain_compression_bits(domain: Mapping[str, Sequence[float]], valid_domain: Mapping[str, Sequence[float]]) -> float:
    full, valid = _bounds(domain), _bounds(valid_domain)
    if full.keys() != valid.keys(): raise ValueError("domain dimensions differ")
    bits = 0.0
    for s, (lo, hi) in full.items():
        a, b = valid[s]
        if not lo <= a <= b <= hi: raise ValueError("valid domain is not a subset of Omega")
        if lo == hi:
            if a != lo or b != hi: raise ValueError("fixed coordinate changed")
            continue
        if a == b: raise ValueError("zero-volume valid domain cannot receive finite compression bits")
        bits += math.log2(hi-lo)-math.log2(b-a)
    return max(0.0, bits)


def certify_archetype_significance(archetype: Archetype, res_m: SignificanceResult, res_m0: SignificanceResult,
                                  adjoint_res: IdentifiabilityAudit, ihcg: IhcgGraph) -> SignificanceAudit:
    ihcg.assert_sealed()
    if archetype != ihcg.archetype or not res_m.observables or len(res_m.observables) != len(res_m0.observables): raise ValueError("matching nonempty observable vectors required")
    effect = max(abs(a-b) for a, b in zip(res_m.observables, res_m0.observables))
    scale = max(*(abs(x) for x in (*res_m.observables, *res_m0.observables)), 100*sys.float_info.epsilon)
    relative = effect/scale
    uncertainty = res_m.fidelity.epsilon_total+res_m0.fidelity.epsilon_total
    margin = effect-5*uncertainty; confidence = None
    win_interval = None
    method = "ODE_GCI_SEPARATION"
    passed = res_m.fidelity.passed and res_m0.fidelity.passed and adjoint_res.coupling_passed
    if archetype == "MONTE_CARLO":
        method = "MC_EMPIRICAL_BERNSTEIN"; a, b = ihcg.mc_difference_bounds or (math.nan, math.nan)
        if any(e.kind == "MC_SAMPLE" for e in ihcg.equations.values()):
            from .stochastic_equilibrium import mc_difference_bounds
            a,b = mc_difference_bounds(ihcg)
        x, y = res_m.paired_samples, res_m0.paired_samples
        if len(x) < 2 or len(x) != len(y) or len(res_m.stream_keys) != len(x) or res_m.stream_keys != res_m0.stream_keys or len(set(res_m.stream_keys)) != len(x): raise ValueError("paired CRN samples and distinct shared stream keys required")
        if not math.isfinite(a) or not math.isfinite(b) or b <= a: raise ValueError("a priori bounded paired-difference range required")
        differences = [u-v for u, v in zip(x, y)]
        if any(not a <= d <= b for d in differences): raise ValueError("paired difference exceeds its a priori range")
        delta = 0.01/max(1, sum(o.obligation_class == "HYPOTHESIS_TARGET" for o in ihcg.obligations))
        # Split failure probability between EB and Hoeffding before taking min.
        n = len(x); width = b-a; log = math.log(8/delta)
        error = min(math.sqrt(2*statistics.variance(differences)*log/n)+7*width*log/(3*(n-1)), width*math.sqrt(math.log(4/delta)/(2*n)))
        effect = abs(statistics.mean(differences)); margin = effect-error-uncertainty
        relative = effect/scale; confidence = 1-delta
        samples = [e.state_symbol for e in ihcg.equations.values() if e.kind == "MC_SAMPLE"]
        claims = [o.expression for o in ihcg.obligations if o.obligation_class == "HYPOTHESIS_TARGET"]
        if len(samples) == len(claims) == 1:
            claim = _parse(claims[0])
            if isinstance(claim, ast.Compare) and len(claim.ops) == 1:
                left, right, op = claim.left, claim.comparators[0], claim.ops[0]
                orientation = None
                if isinstance(left, ast.Name) and left.id == samples[0] and samples[0] not in _variables(right):
                    orientation = 1 if isinstance(op,(ast.Gt,ast.GtE)) else -1 if isinstance(op,(ast.Lt,ast.LtE)) else None
                if isinstance(right, ast.Name) and right.id == samples[0] and samples[0] not in _variables(left):
                    orientation = -1 if isinstance(op,(ast.Gt,ast.GtE)) else 1 if isinstance(op,(ast.Lt,ast.LtE)) else None
                if orientation is not None:
                    win_interval = clopper_pearson_interval(sum(orientation*d > 0 for d in differences),n,confidence=1-delta/2)
        if res_m.bernoulli_indicators or res_m0.bernoulli_indicators:
            if not res_m.bernoulli_indicators or not res_m0.bernoulli_indicators or any(v not in (0, 1) for v in (*x, *y)):
                raise ValueError("both CRN vectors must contain declared Bernoulli indicators")
            lo, hi = clopper_pearson_interval(sum(int(v) for v in x), n, confidence=1-delta/2)
            lo0, hi0 = clopper_pearson_interval(sum(int(v) for v in y), n, confidence=1-delta/2)
            margin = max(lo-hi0, lo0-hi)-uncertainty
            method = "MC_CLOPPER_PEARSON"
        passed = passed and res_m.fidelity.statistical_audit is not None and res_m.fidelity.statistical_audit.passed and res_m0.fidelity.statistical_audit is not None and res_m0.fidelity.statistical_audit.passed
    elif archetype == "FIXED_POINT":
        method = "FIXED_POINT_KRAWCZYK_HURWITZ"
        if res_m.krawczyk is None or res_m.stability_matrix is None: raise ValueError("Krawczyk and interval stability evidence required")
        matrix = [[{"lo": lo, "hi": hi} for lo, hi in row] for row in res_m.stability_matrix]
        stability = json.loads(_native.py_certify_interval_stability(_json(matrix)))
        passed = passed and res_m.krawczyk.get("certified") is True and stability["certified"]
    elif archetype == "PARAM_SWEEP":
        method = "BIFURCATION_LYAPUNOV_BRACKET"
        if all(v is not None for v in (res_m.lyapunov_t, res_m.lyapunov_2t, res_m0.lyapunov_t, res_m0.lyapunov_2t)):
            lm, l0 = res_m.lyapunov_2t, res_m0.lyapunov_2t
            em = 3*abs(lm-res_m.lyapunov_t)+res_m.lyapunov_disc_error
            e0 = 3*abs(l0-res_m0.lyapunov_t)+res_m0.lyapunov_disc_error
            margin = max(min(lm-em, -l0-e0), min(l0-e0, -lm-em))
        elif res_m.boundary is not None:
            passed = passed and res_m.boundary.passed; margin = min(res_m.boundary.valid_margin_lower, -res_m.boundary.falsified_margin_upper)
        elif res_m.pareto_points and res_m0.pareto_points:
            method = "PARETO_EPS_DOMINANCE"
            if len(res_m.objective_errors) != len(res_m.pareto_points) or len(res_m0.objective_errors) != len(res_m0.pareto_points): raise ValueError("per-objective uncertainty bounds required")
            dimension = len(res_m.pareto_points[0])
            if not dimension or any(len(p) != dimension for p in (*res_m.pareto_points, *res_m0.pareto_points, *res_m.objective_errors, *res_m0.objective_errors)) or any(e < 0 for row in (*res_m.objective_errors, *res_m0.objective_errors) for e in row): raise ValueError("invalid Pareto evidence")
            # All objectives are declared minimization objectives by this adapter.
            margins = [max(min(b-a-5*(ea+eb) for a, b, ea, eb in zip(p, q, pe, qe)) for p, pe in zip(res_m.pareto_points, res_m.objective_errors)) for q, qe in zip(res_m0.pareto_points, res_m0.objective_errors)]
            margin = min(margins)
        else: passed = False
    passed = passed and margin > 0
    compression = domain_compression_bits(ihcg.omega, res_m.valid_domain) if res_m.valid_domain is not None else None
    micro = effect > 5*(res_m.fidelity.epsilon_disc+res_m0.fidelity.epsilon_disc) and relative < 1e-3 and not ihcg.perturbation_target
    return SignificanceAudit(passed=bool(passed), null_baseline_description=res_m0.null_baseline_description, effect_size_estimate=effect,
                             certified_lower_margin=margin, method=method, confidence_level=confidence, relative_effect_size=relative,
                             micro_effect_warning=micro, domain_compression_bits=compression, failure_reason=None if passed else "SIGNIFICANCE_UNCERTIFIED",
                             win_rate_method="MC_CLOPPER_PEARSON" if win_interval is not None else None,
                             win_rate_interval=win_interval,replicate_count=len(res_m.paired_samples) if archetype == "MONTE_CARLO" else None)


class HypothesisEvalResult(StrictModel):
    signed_margin: float
    epsilon_total: float = Field(ge=0)
    precision_refined: bool = False


class RouteDecision(StrictModel):
    action: Literal["REFLEX_HEAL", "TERMINATE_INCONCLUSIVE", "REFINE_PRECISION_ONCE", "RUN_MODE1_FALSIFICATION_CERTIFIER", "RUN_BOUNDARY_CERTIFIER", "RUN_MODE2A_COUNTEREXAMPLE_CERTIFIER", "RUN_MODE2B_EXHAUSTIVE_REFUTATION", "RUN_SIGNIFICANCE_AND_UNIQUENESS_GATE"]
    halt_healing: bool = False
    reason: str | None = None


def route_post_execution(ihcg: IhcgGraph, smt_audit: ContractAudit, purity_audit: PurityAudit,
                         fidelity: FidelityAudit, active_audit: IdentifiabilityAudit, eval_res: HypothesisEvalResult,
                         attempt: int, max_retries: int) -> RouteDecision:
    ihcg.assert_sealed()
    if not 0 <= attempt or not 0 <= max_retries <= 3: raise ValueError("retry budget must be bounded by three")
    def heal(reason): return RouteDecision(action="REFLEX_HEAL" if attempt < max_retries else "TERMINATE_INCONCLUSIVE", reason=reason)
    if not purity_audit.passed: return heal("PURITY_VIOLATION")
    if not smt_audit.entailment_locked or smt_audit.classification in {"VACUOUS", "TAUTOLOGICAL", "INFEASIBLE", "UNKNOWN"}: return heal(f"CONTRACT_{smt_audit.classification}")
    unreliable = (fidelity.convergence_reliability_ratio or 0) > 1 or fidelity.failure_reason == "CONVERGENCE_ESTIMATOR_UNRELIABLE"
    if unreliable: return RouteDecision(action="TERMINATE_INCONCLUSIVE" if eval_res.precision_refined else "REFINE_PRECISION_ONCE", reason="CONVERGENCE_ESTIMATOR_UNRELIABLE")
    if fidelity.failure_reason == "STATISTICAL_ESTIMATOR_UNRELIABLE" or fidelity.statistical_audit is not None and not fidelity.statistical_audit.passed:
        return RouteDecision(action="TERMINATE_INCONCLUSIVE", reason="STATISTICAL_ESTIMATOR_UNRELIABLE")
    if smt_audit.classification == "INFEASIBLE_UNDER_MODEL" and fidelity.passed and ihcg.quantifier_mode != "EXISTS":
        return RouteDecision(action="RUN_MODE1_FALSIFICATION_CERTIFIER", halt_healing=True)
    if not fidelity.passed: return heal(fidelity.failure_reason or "FIDELITY_UNCERTIFIED")
    if not active_audit.coupling_passed: return heal("DEGENERATE_UNCOUPLED_STAGE")
    if not math.isclose(eval_res.epsilon_total, fidelity.epsilon_total, rel_tol=1e-14): raise ValueError("evaluation may not change the certified uncertainty")
    margin, eps = eval_res.signed_margin, eval_res.epsilon_total
    if margin+eps < 0:
        action = {"BOUNDARY": "RUN_BOUNDARY_CERTIFIER", "FORALL": "RUN_MODE2A_COUNTEREXAMPLE_CERTIFIER", "EXISTS": "RUN_MODE2B_EXHAUSTIVE_REFUTATION"}[ihcg.quantifier_mode]
        return RouteDecision(action=action, halt_healing=True)
    if abs(margin) <= eps:
        return RouteDecision(action="TERMINATE_INCONCLUSIVE" if eval_res.precision_refined else "REFINE_PRECISION_ONCE", reason="MARGIN_WITHIN_UNCERTAINTY" if eval_res.precision_refined else None)
    return RouteDecision(action="RUN_BOUNDARY_CERTIFIER" if ihcg.quantifier_mode == "BOUNDARY" else "RUN_SIGNIFICANCE_AND_UNIQUENESS_GATE", halt_healing=True)


def build_meaningfulness_certificate(*, run_id: str, verdict: EpistemicVerdict, ihcg: IhcgGraph,
                                      contract_audit: ContractAudit, nontriviality_audit: NontrivialityAudit,
                                      purity_audit: PurityAudit, fidelity_audit: FidelityAudit,
                                      identifiability_audit: IdentifiabilityAudit, uniqueness_audit: UniquenessAudit,
                                      uast_digest: str, significance_audit: SignificanceAudit | None = None,
                                      falsification_certificate: FalsificationCertificate | None = None,
                                      boundary_certificate: BoundaryCertificate | None = None,
                                      certified_valid_domain: Bounds | None = None,
                                      preserved_vs_enforced: Mapping[str, PurityAudit] | None = None) -> MeaningfulnessCertificate:
    ihcg.assert_sealed()
    # Re-run the eighth gate against the actual sealed environment; callers
    # cannot substitute an audit from an unrelated claim or easier domain.
    checked = verify_claim_nontriviality(ihcg, nontriviality_audit.claim)
    if checked.passed != nontriviality_audit.passed or checked.failure_reason != nontriviality_audit.failure_reason:
        raise ValueError("nontriviality audit is not bound to this IHCG")
    anchor = IhcgAnchor(ihcg_digest=ihcg.h_ihcg, stage_chain_digests=ihcg.stage_chain_digests, quantifier_mode=ihcg.quantifier_mode,
                        archetype=ihcg.archetype, prompt_obligation_quotes=tuple({"quote": o.prompt_quote, "csel": o.expression, "class": o.obligation_class} for o in ihcg.obligations))
    certificate = MeaningfulnessCertificate(run_id=run_id, epistemic_verdict=verdict, ihcg_anchor=anchor, contract_audit=contract_audit,
        nontriviality_audit=checked, purity_audit=purity_audit, fidelity_audit=fidelity_audit, identifiability_audit=identifiability_audit,
        significance_audit=significance_audit, falsification_certificate=falsification_certificate, boundary_certificate=boundary_certificate,
        certified_valid_domain=certified_valid_domain,
        uniqueness_audit=uniqueness_audit, convergence_reliability_ratio=fidelity_audit.convergence_reliability_ratio,
        preserved_vs_enforced=dict(preserved_vs_enforced or {purity_audit.domain_tag: purity_audit}),
        relative_effect_size=significance_audit.relative_effect_size if significance_audit else None,
        domain_compression_bits=significance_audit.domain_compression_bits if significance_audit else None,
        z3_tier_used=_worst_tier([*checked.contract_audit.queries.values(), QueryResult(status="UNKNOWN", z3_tier_used=contract_audit.z3_logic_tier)]),
        uast_digest=uast_digest, blake3_receipt="0"*64)
    receipt = _native.py_compute_certificate_receipt(certificate.model_dump_json())
    return certificate.model_copy(update={"blake3_receipt": receipt})


def verify_certificate_receipt(certificate: MeaningfulnessCertificate) -> bool:
    validated = MeaningfulnessCertificate.model_validate_json(certificate.model_dump_json())
    return hmac.compare_digest(validated.blake3_receipt, _native.py_compute_certificate_receipt(validated.model_dump_json()))


def write_meaningfulness_certificate(certificate: MeaningfulnessCertificate, path: Path) -> None:
    if not verify_certificate_receipt(certificate): raise ValueError("certificate receipt mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name); handle.write(certificate.model_dump_json(indent=2)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)

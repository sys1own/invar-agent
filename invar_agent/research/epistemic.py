"""Pipeline evidence boundary: legacy execution success is not a discovery."""
from __future__ import annotations

import ast
import hmac
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

EvidenceStatus = Literal["CERTIFIED", "OBSERVED_UNVERIFIED", "VIOLATED", "NOT_COMPUTED"]


def is_verified_epistemic_verdict(verdict_or_cert: Any) -> bool:
    """True iff the certificate or verdict string is a certified VERIFIED_* outcome."""
    if verdict_or_cert is None:
        return False
    if hasattr(verdict_or_cert, "epistemic_verdict"):
        raw = getattr(verdict_or_cert, "epistemic_verdict")
    elif isinstance(verdict_or_cert, Mapping) and "epistemic_verdict" in verdict_or_cert:
        raw = verdict_or_cert.get("epistemic_verdict")
    else:
        raw = verdict_or_cert
    val = getattr(raw, "value", raw)
    return str(val or "").startswith("VERIFIED_")


def compute_evidence_status(
    verdict_or_cert: Any,
    value: Any,
    *,
    violated: bool = False,
) -> EvidenceStatus:
    """Single authoritative function deriving metric EvidenceStatus from the certificate."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "NOT_COMPUTED"
    if violated:
        return "VIOLATED"
    if is_verified_epistemic_verdict(verdict_or_cert):
        return "CERTIFIED"
    return "OBSERVED_UNVERIFIED"

from pydantic import Field

from invar_agent import _native
from . import meaningfulness as m
from .falsification import (MarginSpec, WitnessEvaluation, certify_phase_boundary,
    run_mode1_deductive_falsification, run_mode2a_counterexample_falsification,
    run_mode2b_exhaustive_refutation)
from .uniqueness import (CanonicalSignatureEngine, ReferenceRegistry,
    check_intra_vault_uniqueness, check_reference_nontriviality,
    combine_uniqueness_audits, load_verified_vault_entries, si_error_budgets)


TRUST_BOUNDARY = (
    "Z3 per-logic soundness", "egg rewrite-axiom soundness",
    "IntervalF64 IEEE-754 outward rounding", "HIN interval typing soundness",
    "adjoint.rs symbolic derivative & interval-extension soundness",
    "[1. Ingest] prose→FormalResearchSpec AST binding (audited via assumptions_ledger)",
)


class IncompleteRunCertificate(m.StrictModel):
    schema_version: str = "meaningfulness_incomplete.v1"
    run_id: str
    epistemic_verdict: m.EpistemicVerdict = m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED
    hypothesis_quote: str
    failure_reason: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    trust_boundary_version: str = "invar.tcb.v1"
    blake3_receipt: str = Field(pattern=r"^[0-9a-f]{64}$")


def incomplete_certificate(run_id, hypothesis, reason, *, diagnostics=None):
    certificate = IncompleteRunCertificate(run_id=run_id, hypothesis_quote=hypothesis, failure_reason=reason, diagnostics=diagnostics or {}, blake3_receipt="0"*64)
    return certificate.model_copy(update={"blake3_receipt": _native.py_compute_certificate_receipt(certificate.model_dump_json())})


def parse_run_certificate(raw):
    if hasattr(raw, "model_dump"): raw = raw.model_dump(mode="json")
    if isinstance(raw, str): raw = json.loads(raw)
    if raw.get("schema_version") == "meaningfulness_incomplete.v1":
        certificate = IncompleteRunCertificate.model_validate_json(json.dumps(raw, allow_nan=False))
        if certificate.epistemic_verdict != m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED:
            raise ValueError("incomplete evidence cannot certify a terminal discovery")
        signed = certificate.model_dump(mode="json")
        if "diagnostics" not in raw: signed.pop("diagnostics")
        valid = hmac.compare_digest(certificate.blake3_receipt, _native.py_compute_certificate_receipt(json.dumps(signed, allow_nan=False)))
    else:
        certificate = m.MeaningfulnessCertificate.model_validate_json(json.dumps(raw, allow_nan=False))
        valid = m.verify_certificate_receipt(certificate)
    if not valid: raise ValueError("meaningfulness certificate receipt mismatch")
    return certificate


def validate_certificate_artifacts(raw, module_code, blueprint):
    certificate = parse_run_certificate(raw)
    if not isinstance(certificate, m.MeaningfulnessCertificate): return certificate
    from .accretion import code_hash
    if code_hash(module_code) != certificate.uast_digest: raise ValueError("certificate executable digest mismatch")
    pack = blueprint.get("ihcg_evidence")
    if not pack: raise ValueError("certificate IHCG source binding is missing")
    from ..schemas.research import ResearchBlueprint
    graph = compile_blueprint_ihcg(ResearchBlueprint.model_validate(blueprint),stage=len(certificate.ihcg_anchor.stage_chain_digests)+1)
    if graph.h_ihcg != certificate.ihcg_anchor.ihcg_digest: raise ValueError("certificate IHCG source binding mismatch")
    return certificate


def compile_blueprint_ihcg(blueprint,*,stage=1):
    from .feasibility import compile_ingest_ihcg
    pack = blueprint.ihcg_evidence
    if not pack: return None
    root = {key:value for key,value in pack.items() if key not in {"extensions","execution","notes"}}
    prompt = blueprint.research_prompt or blueprint.falsifiable_hypothesis
    graph = compile_ingest_ihcg(prompt, **root)
    graphs = [graph]
    for extension in pack.get("extensions",()):
        graph = m.IhcgCompiler().extend_stage(graph,prompt,extension["evidence"],extension["domain"],extension["archetype"])
        audit = m.classify_contract_vacuity(graph)
        if not audit.passed: raise ValueError(f"IHCG_INGEST_REJECTED: {audit.classification}")
        graphs.append(graph)
    if not 1 <= stage <= len(graphs): raise ValueError("IHCG stage binding is missing")
    for index,graph in enumerate(graphs,1):
        stage_arg = index if len(graphs) > 1 else None
        compiled = formal_contracts(blueprint, graph, stage=stage_arg)
        lock = verify_blueprint_entailment_lock(blueprint, graph, stage=stage_arg)
        if not lock.entailment_locked:
            raise ValueError(f"IHCG_INGEST_REJECTED: Phi_lock; compiled_contracts={compiled}; query_statuses={dict((k,v.status) for k,v in lock.queries.items())}; counterexample={lock.counterexample_if_weakened}")
    return graphs[stage-1]


def _is_liveness_telemetry_guard(assertion: str) -> bool:
    text = re.sub(r"\s+", "", str(assertion or "")).replace('"', "'")
    if "variants[" in text and "state[" not in text and "buf.data[" not in text:
        return True
    if "telemetry[" in text and "state[" not in text and "buf.data[" not in text:
        return True
    return text in {
        "telemetry['finite']==1.0",
        "telemetry['finite']==1",
        "telemetry['finite']>0",
        "telemetry['finite']>0.5",
        "telemetry['finite']>=1",
        "telemetry['finite']>=1.0",
        "bool(telemetry['finite'])",
    }


def _formal_contracts_for(blueprint, graph, contracts):
    names = {s.symbol for s in graph.slots}
    slots = {slot.slot_idx: slot.symbol for slot in graph.slots}
    for name, index in blueprint.variable_map.items():
        if name in names and slots.get(index) != name:
            raise ValueError(f'compiled contract slot binding conflict: {name} at {index}')
    class Bind(ast.NodeTransformer):
        def __init__(self, initial_state):
            self.initial_state = list(initial_state or ())

        def visit_Subscript(self, node):
            base = ast.unparse(node.value)
            if base in {"state", "buf.data", "data"} and isinstance(node.slice, ast.Constant) and node.slice.value in slots:
                return ast.Name(id=slots[node.slice.value], ctx=ast.Load())
            if base in {"initial", "initial_state"} and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int and 0 <= node.slice.value < len(self.initial_state):
                return ast.Constant(value=float(self.initial_state[node.slice.value]))
            raise ValueError(f"compiled contract has no sealed symbolic binding: {ast.unparse(node)}; variable_map={blueprint.variable_map}; sealed_symbols={sorted(names)}")
    all_pairs = [(contract, invariant) for contract in contracts for invariant in contract.invariants]
    non_liveness = [(c, inv) for c, inv in all_pairs if not _is_liveness_telemetry_guard(inv.assertion)]
    selected_pairs = non_liveness if non_liveness else all_pairs
    return tuple(ast.unparse(ast.fix_missing_locations(Bind(contract.initial_state).visit(ast.parse(invariant.assertion, mode="eval").body)))
                 for contract, invariant in selected_pairs)


def formal_contracts(blueprint, graph,*,stage=None):
    contracts = blueprint.contracts
    if stage is not None:
        from .state_machine import _contracts_through_stage
        contracts = _contracts_through_stage(blueprint,stage)
    return _formal_contracts_for(blueprint, graph, contracts)


def verify_blueprint_entailment_lock(blueprint, graph, *, stage=None) -> m.EntailmentAudit:
    """Verify Phi_lock for single-scenario or multi-regime blueprint contracts.

    When multiple contracts test distinct parameter regimes or horizons (e.g.
    R0=1.5 vs R0=4.0), their naive conjunction is UNSAT even though every
    contract is individually feasible and entails its obligations. In that case
    we verify:
      1. First try full conjunction (standard path; unchanged when SAT).
      2. If UNSAT with len(contracts) > 1, check each contract individually
         (preferring the final/primary contract first). If any single feasible
         contract entails all obligations, Phi_lock holds.
      3. Otherwise, require EVERY contract to be individually feasible (SAT),
         ALL HYPOTHESIS_TARGET obligations to be jointly entailed by one
         feasible contract, and EVERY remaining obligation to be entailed by
         at least one feasible contract.
    """
    contracts = blueprint.contracts
    if stage is not None:
        from .state_machine import _contracts_through_stage
        contracts = _contracts_through_stage(blueprint, stage)
    compiled_all = _formal_contracts_for(blueprint, graph, contracts)
    lock = m.verify_entailment_lock(graph, compiled_all)
    if lock.entailment_locked or len(contracts) <= 1:
        return lock
    feas_q = lock.queries.get("COMPILED_FEASIBILITY")
    if feas_q is None or feas_q.status != "UNSAT":
        return lock
    per_contract: list[m.EntailmentAudit] = []
    for contract in reversed(contracts):
        c_compiled = _formal_contracts_for(blueprint, graph, (contract,))
        c_lock = m.verify_entailment_lock(graph, c_compiled)
        if c_lock.entailment_locked:
            return c_lock
        per_contract.append(c_lock)
    # Require every contract to be individually feasible (SAT).
    if not all(c_lock.queries.get("COMPILED_FEASIBILITY") and c_lock.queries["COMPILED_FEASIBILITY"].status == "SAT" for c_lock in per_contract):
        return lock
    hyp_ids = [o.id for o in graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET"]
    # All HYPOTHESIS_TARGET obligations must be jointly entailed by at least one single feasible contract.
    if hyp_ids and not any(all(c_lock.queries[oid].status == "UNSAT" for oid in hyp_ids) for c_lock in per_contract):
        return lock
    merged_queries: dict[str, m.QueryResult] = {}
    for o in graph.obligations:
        entailed = next((c_lock.queries[o.id] for c_lock in per_contract if c_lock.queries[o.id].status == "UNSAT"), None)
        if entailed is None:
            return lock
        merged_queries[o.id] = entailed
    merged_queries["COMPILED_FEASIBILITY"] = per_contract[0].queries["COMPILED_FEASIBILITY"]
    return m.EntailmentAudit(
        entailment_locked=True,
        queries=merged_queries,
        z3_tier_used=m._worst_tier(list(merged_queries.values())),
        counterexample_if_weakened=None,
    )


@dataclass(frozen=True)
class ExecutionEvidence:
    executable_digest: str
    formal_uast: Mapping[str, Any]
    purity: m.PurityAudit
    identifiability: m.IdentifiabilityAudit
    evaluation: WitnessEvaluation
    sandbox_runner: Callable[..., Any] | None = None
    fidelity: m.FidelityAudit | None = None
    comparison: tuple[m.SignificanceResult, m.SignificanceResult] | None = None
    theta_star: Mapping[str, float] | None = None
    margin_spec: MarginSpec = MarginSpec()
    boundary_runner: Callable[..., WitnessEvaluation] | None = None
    boundary_tolerance: float | None = None
    boundary_parameter: str | None = None
    fixed_values: Mapping[str, float] | None = None
    compared_run_id: str | None = None


def evaluate_execution(blueprint, graph, evidence, *, executable_digest, vault, attempt, precision_refined, preserved):
    if not isinstance(evidence, ExecutionEvidence) or evidence.executable_digest != executable_digest:
        raise ValueError("execution evidence does not describe this executable")
    claims = tuple(o.expression for o in graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET")
    lock = verify_blueprint_entailment_lock(blueprint, graph)
    from .falsification import _model_constraints
    vacuity = m.classify_contract_vacuity(
        graph,
        list(_model_constraints(graph).values()),
        candidate_witness=evidence.theta_star,
        krawczyk=evidence.formal_uast.get("krawczyk") if isinstance(evidence.formal_uast, Mapping) else None,
    )
    contract = m.ContractAudit(entailment_locked=lock.entailment_locked, classification=vacuity.classification,
        z3_logic_tier=vacuity.z3_tier_used, redundant_conjuncts=vacuity.redundant_conjuncts)
    nontrivial = m.verify_claim_nontriviality(graph, claims)
    fidelity = evidence.fidelity
    if fidelity is None:
        if evidence.sandbox_runner is None: raise ValueError("compiled fidelity runner is missing")
        fidelity = m.certify_implementation_fidelity(graph, evidence.formal_uast, evidence.sandbox_runner)
    fidelity = m.FidelityAudit.model_validate_json(fidelity.model_dump_json())
    evaluation = evidence.evaluation.model_copy(update={"precision_refined": precision_refined})
    decision = m.route_post_execution(graph, contract, evidence.purity, fidelity, evidence.identifiability,
                                    evaluation, min(attempt, 3), 3)
    proof = boundary = significance = None
    verdict = m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED
    action = decision.action
    if action == "RUN_MODE1_FALSIFICATION_CERTIFIER":
        proof = run_mode1_deductive_falsification(graph, fidelity=fidelity)
    elif action == "RUN_MODE2A_COUNTEREXAMPLE_CERTIFIER" and evidence.theta_star is not None:
        proof = run_mode2a_counterexample_falsification(graph, evidence.theta_star, evaluation, fidelity)
    elif action == "RUN_MODE2B_EXHAUSTIVE_REFUTATION":
        proof = run_mode2b_exhaustive_refutation(graph, evidence.margin_spec, fidelity)
    elif action == "RUN_BOUNDARY_CERTIFIER" and evidence.boundary_runner and evidence.boundary_tolerance:
        boundary = certify_phase_boundary(graph, evidence.boundary_runner, evidence.boundary_tolerance,
            fidelity=fidelity, parameter=evidence.boundary_parameter, fixed_values=evidence.fixed_values)
    if proof and proof.passed: verdict = m.EpistemicVerdict.VERIFIED_FALSIFICATION
    if proof and proof.mode == "MODE1_DEDUCTIVE_MUS" and attempt:
        verdict = m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED
        decision = m.RouteDecision(action="TERMINATE_INCONCLUSIVE", reason="DEDUCTIVE_CERTIFICATION_REQUIRES_ZERO_HEAL_COUNT")
    if evidence.comparison is not None:
        primary, baseline = evidence.comparison
        primary = primary.model_copy(update={"fidelity": fidelity, "boundary": boundary})
        if boundary is not None and boundary.passed:
            from .default_verification import certified_valid_domain
            primary = primary.model_copy(update={"valid_domain":certified_valid_domain(graph,boundary)})
        significance = m.certify_archetype_significance(graph.archetype, primary, baseline, evidence.identifiability, graph)
        if significance.passed and proof is None:
            if boundary is not None and boundary.passed: verdict = m.EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY
            elif action == "RUN_SIGNIFICANCE_AND_UNIQUENESS_GATE": verdict = m.EpistemicVerdict.VERIFIED_DISCOVERY
    signature = CanonicalSignatureEngine().compute_cds(graph, evidence.formal_uast, verdict)
    disc_si,total_si = si_error_budgets(graph,fidelity.epsilon_disc,fidelity.epsilon_total)
    novel = check_intra_vault_uniqueness(signature, graph, verdict, load_verified_vault_entries(vault),
        epsilon_disc=disc_si, epsilon_total=total_si, active_audit=evidence.identifiability,
        significance_audit=significance, compared_run_id=evidence.compared_run_id)
    reference = check_reference_nontriviality(signature, graph, verdict, ReferenceRegistry(), significance_audit=significance,active_audit=evidence.identifiability)
    uniqueness = m.UniquenessAudit(passed=novel.passed and reference.passed, cds_model_hash=signature.model_hash,
        cds_claim_hash=signature.claim_hash, cds_full_hash=signature.full_hash, vault_class=novel.vault_class,
        related_run_id=novel.related_run_id, reference_nontrivial=reference.passed,
        shbt_open_boundary_closure=reference.shbt_open_boundary_closure)
    kwargs = dict(run_id=blueprint.blueprint_id, ihcg=graph, contract_audit=contract, nontriviality_audit=nontrivial,
        purity_audit=evidence.purity, fidelity_audit=fidelity, identifiability_audit=evidence.identifiability,
        uniqueness_audit=uniqueness, significance_audit=significance, falsification_certificate=proof,
        boundary_certificate=boundary, uast_digest=executable_digest, preserved_vs_enforced=preserved)
    if evidence.comparison is not None and boundary is not None and boundary.passed:
        kwargs["certified_valid_domain"] = primary.valid_domain
    # Check the conjunction before choosing a verified terminal, not after publishing it.
    provisional = m.build_meaningfulness_certificate(verdict=m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED, **kwargs)
    if verdict != m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED:
        candidate = provisional.model_copy(update={"epistemic_verdict": verdict})
        if not all(m.meaningfulness_gates(candidate).values()): verdict = m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED
    if verdict.value != json.loads(signature.domain_and_verdict)["verdict"]:
        signature = CanonicalSignatureEngine().compute_cds(graph, evidence.formal_uast, verdict)
        kwargs["uniqueness_audit"] = uniqueness.model_copy(update={"cds_full_hash": signature.full_hash})
    certificate = m.build_meaningfulness_certificate(verdict=verdict, **kwargs)
    if not nontrivial.passed: decision = m.RouteDecision(action="TERMINATE_INCONCLUSIVE", reason=nontrivial.failure_reason)
    return certificate, signature, decision.model_copy(update={"halt_healing": verdict == m.EpistemicVerdict.VERIFIED_FALSIFICATION})

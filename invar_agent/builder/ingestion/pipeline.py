"""Typed upload/vault ingestion and transactional researcher review."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .archive_guard import IngestionError, validate_and_unpack_upload
from .equation_lowerer import (FormalGateError, _native, check_z3, equation_parts,
                               lower_and_verify_equations, normalize_to_si, unit_dimension)
from .parsers import parse_document
from .schemas import (CanonicalResearchIR, EquationRole, ExtractedConstant, IngestionRequest, IRReviewOverride,
                      RawConstant, RawEquation, SIDimensionVectorModel, SourceSpan, StateVariableSpec,
                      VerificationStatus)
from .vault_adapter import load_vault_bundle


def seal_ir(ir: CanonicalResearchIR) -> CanonicalResearchIR:
    payload = ir.model_dump(mode="json", exclude={"ir_blake3_digest", "blake3_receipt"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    ir.ir_blake3_digest = _native().compute_his_digest_v2("invar.ir.canonical.v1", canonical)
    native = _native()
    if hasattr(native, "py_compute_certificate_receipt"):
        ir.blake3_receipt = native.py_compute_certificate_receipt(ir.model_dump_json())
    elif ir.meaningfulness_certificates or ir.meaningfulness_certificate is not None or ir.epistemic_verdict is not None:
        raise FormalGateError("NATIVE_CERTIFICATE_RECEIPT_REQUIRED", "Certified research requires the native BLAKE3 receipt binding")
    else:
        ir.blake3_receipt = ""
    return ir


def reverify_ir(ir: CanonicalResearchIR) -> CanonicalResearchIR:
    equations, constants, flags = lower_and_verify_equations(
        [RawEquation(text=e.canonical_latex, provenance=e.provenance) for e in ir.equations],
        [RawConstant(symbol=c.symbol, value=str(c.si_value), unit=c.dimension.canonical_unit,
                     provenance=c.provenance) for c in ir.constants],
        {"state_symbols": [s.symbol for s in ir.state_variables],
         "boundary_conditions": {s.symbol: s.bounds.model_dump() for s in ir.state_variables if s.bounds is not None},
         "symbol_dimensions": {s.symbol: s.dimension.model_dump() for s in [*ir.constants, *ir.state_variables]}})
    if len(equations) != len(ir.equations) or len(constants) != len(ir.constants):
        raise FormalGateError("IR_OVERRIDE_INVALID", "Review changed an equation into a parameter; use a constant patch")
    for old, checked in zip(ir.constants, constants):
        checked.constant_id = old.constant_id
        checked.name = old.name
        checked.raw_value, checked.raw_unit = old.raw_value, old.raw_unit
        checked.bounds = old.bounds
        checked.is_tunable_parameter = old.is_tunable_parameter
        checked.disambiguated_by_llm = old.disambiguated_by_llm
        checked.verification_status = old.verification_status
        checked.dimension = old.dimension.model_copy(deep=True)
    for old, checked in zip(ir.equations, equations):
        checked.equation_id = old.equation_id
        checked.raw_latex = old.raw_latex
        checked.role = old.role
        checked.supporting_provenance = old.supporting_provenance
        checked.disambiguated_by_llm = old.disambiguated_by_llm
    ir.equations, ir.constants = equations, constants
    unresolved = {(flag.trigger, flag.target_symbol_or_eq) for flag in flags}
    for flag in ir.disambiguation_flags:
        if flag.trigger.value in {"UNRESOLVED_SYMBOL", "IMPLICIT_BOUNDARY_CONDITION"}:
            flag.resolved = (flag.trigger, flag.target_symbol_or_eq) not in unresolved
    for flag in flags:
        if not any(f.trigger == flag.trigger and f.target_symbol_or_eq == flag.target_symbol_or_eq for f in ir.disambiguation_flags):
            flag.flag_id = f"flag:{len(ir.disambiguation_flags)}"
            ir.disambiguation_flags.append(flag)
    check_z3(ir)
    return seal_ir(ir)


def apply_ir_override(ir: CanonicalResearchIR, override: IRReviewOverride) -> CanonicalResearchIR:
    candidate = ir.model_copy(deep=True)
    for patch in override.equation_patches:
        target = next((e for e in candidate.equations if e.equation_id == patch.equation_id), None)
        if target is None:
            raise FormalGateError("IR_OVERRIDE_INVALID", patch.equation_id)
        if patch.drop:
            candidate.equations.remove(target)
            continue
        if patch.canonical_latex is not None:
            target.canonical_latex = patch.canonical_latex
        if patch.role is not None:
            target.role = patch.role
    override_invalidates_cert = bool(override.equation_patches)
    cert_domain = candidate.certified_valid_domain if isinstance(candidate.certified_valid_domain, dict) else None
    for patch in override.constant_patches:
        target = next((c for c in candidate.constants if c.constant_id == patch.constant_id), None)
        if target is None:
            raise FormalGateError("IR_OVERRIDE_INVALID", patch.constant_id)
        if patch.drop:
            candidate.constants.remove(target)
            override_invalidates_cert = True
            continue
        if patch.si_value is not None:
            target.si_value = patch.si_value
            if cert_domain is None or target.symbol not in cert_domain:
                override_invalidates_cert = True
            else:
                bounds_pair = cert_domain[target.symbol]
                if (
                    not isinstance(bounds_pair, (list, tuple))
                    or len(bounds_pair) != 2
                    or not (float(bounds_pair[0]) <= float(patch.si_value) <= float(bounds_pair[1]))
                ):
                    override_invalidates_cert = True
        if patch.raw_unit is not None:
            _, unit = normalize_to_si(1, patch.raw_unit)
            target.dimension = unit_dimension(unit)
            target.dimension.canonical_unit = unit
            target.raw_unit = patch.raw_unit
            override_invalidates_cert = True
        if patch.bounds is not None:
            target.bounds = patch.bounds
        if patch.is_tunable_parameter is not None:
            target.is_tunable_parameter = patch.is_tunable_parameter
        target.verification_status = VerificationStatus.USER_OVERRIDDEN
    if override_invalidates_cert and (
        candidate.research_import_role == "POSITIVE_EVIDENCE"
        or str(candidate.epistemic_verdict or "").startswith("VERIFIED_")
    ):
        # ARCH-012 (BUG-020): Override outside certified_valid_domain downgrades certificate role.
        candidate.research_import_role = "USER_MODIFIED_UNVERIFIED"
        candidate.epistemic_verdict = "USER_MODIFIED_UNVERIFIED"
    resolved_derived_symbols: set[str] = set()
    for resolution in override.symbol_resolutions:
        sym = resolution.symbol.strip()
        if resolution.role == "DERIVED_BY_EQUATION":
            resolved_derived_symbols.add(sym)
            target_eq = None
            if resolution.equation_id:
                target_eq = next((e for e in candidate.equations if e.equation_id == resolution.equation_id), None)
                if target_eq is None:
                    raise FormalGateError("IR_OVERRIDE_INVALID", f"Unknown equation_id {resolution.equation_id!r} for symbol {sym!r}")
            else:
                target_eq = next(
                    (e for e in candidate.equations if e.lhs_symbol == sym or e.canonical_latex.strip().startswith(f"{sym} ") or e.canonical_latex.strip().startswith(f"{sym}=")),
                    None,
                )
            if target_eq is not None and not target_eq.lhs_symbol:
                target_eq.lhs_symbol = sym
            for flag in candidate.disambiguation_flags:
                if flag.trigger.value == "UNRESOLVED_SYMBOL" and flag.target_symbol_or_eq == sym:
                    flag.resolved = True
                    flag.resolution_note = f"Researcher confirmed {sym} is derived by equation"
        elif resolution.role == "FREE_PARAMETER":
            existing_const = next((c for c in candidate.constants if c.symbol == sym), None)
            if existing_const is not None:
                existing_const.is_tunable_parameter = True
                if resolution.si_value is not None:
                    existing_const.si_value = resolution.si_value
                if resolution.bounds is not None:
                    existing_const.bounds = resolution.bounds
                if resolution.raw_unit is not None:
                    _, unit = normalize_to_si(1, resolution.raw_unit)
                    existing_const.dimension = unit_dimension(unit)
                    existing_const.dimension.canonical_unit = unit
                    existing_const.raw_unit = resolution.raw_unit
                existing_const.verification_status = VerificationStatus.USER_OVERRIDDEN
            else:
                unit_str = resolution.raw_unit or "1"
                if resolution.raw_unit:
                    _, unit_str = normalize_to_si(1, resolution.raw_unit)
                dim = unit_dimension(unit_str)
                dim.canonical_unit = unit_str
                val = float(resolution.si_value) if resolution.si_value is not None else (
                    0.5 * (resolution.bounds.lower + resolution.bounds.upper)
                    if resolution.bounds and resolution.bounds.lower is not None and resolution.bounds.upper is not None
                    else (resolution.bounds.lower if resolution.bounds and resolution.bounds.lower is not None else 1.0)
                )
                prov = candidate.equations[0].provenance if candidate.equations else SourceSpan(file="override", location="symbol_resolution", quote=sym)
                candidate.constants.append(
                    ExtractedConstant(
                        constant_id=f"constant:{len(candidate.constants)}",
                        name=sym,
                        symbol=sym,
                        raw_value=str(val),
                        raw_unit=resolution.raw_unit or "1",
                        si_value=val,
                        dimension=dim,
                        bounds=resolution.bounds,
                        is_tunable_parameter=True,
                        verification_status=VerificationStatus.USER_OVERRIDDEN,
                        provenance=prov,
                    )
                )
            for flag in candidate.disambiguation_flags:
                if flag.trigger.value == "UNRESOLVED_SYMBOL" and flag.target_symbol_or_eq == sym:
                    flag.resolved = True
                    flag.resolution_note = f"Researcher confirmed {sym} is a free/tunable parameter"
    if override.state_symbol_order is not None:
        states = {s.symbol: s for s in candidate.state_variables}
        if len(set(override.state_symbol_order)) != len(states) or set(override.state_symbol_order) != set(states):
            raise FormalGateError("IR_OVERRIDE_INVALID", "State order must be a permutation")
        candidate.state_variables = [states[name].model_copy(update={"slot_index": i}) for i, name in enumerate(override.state_symbol_order)]
        for flag in candidate.disambiguation_flags:
            if flag.trigger.value == "AMBIGUOUS_STATE_ORDERING":
                flag.resolved = True
                flag.resolution_note = "Researcher supplied a verified state permutation"
    for name in ("time_horizon_si_s", "default_dt_si_s", "default_steps"):
        value = getattr(override, name)
        if value is not None:
            if value <= 0 or (name == "default_steps" and not isinstance(value, int)):
                raise FormalGateError("IR_OVERRIDE_INVALID", name)
            setattr(candidate, name, value)
    candidate.user_approved = override.approve
    reverify_ir(candidate)
    if resolved_derived_symbols:
        for flag in candidate.disambiguation_flags:
            if flag.trigger.value == "UNRESOLVED_SYMBOL" and flag.target_symbol_or_eq in resolved_derived_symbols:
                flag.resolved = True
                flag.resolution_note = f"Researcher confirmed {flag.target_symbol_or_eq} is derived by equation"
    if override.approve and (any(not flag.resolved for flag in candidate.disambiguation_flags)
                            or any(not e.hin_homogeneous or e.z3_satisfiable is not True for e in candidate.equations)):
        raise FormalGateError("IR_UNVERIFIED_FLAGGED", "Unresolved flags require review")
    return seal_ir(candidate)


class CanonicalResearchIngestionPipeline:
    def __init__(self, workspace_root: Path, *, llm_client=None):
        self.workspace_root = Path(workspace_root)
        self.llm_client = llm_client

    def ingest(self, request: IngestionRequest) -> CanonicalResearchIR:
        if not request.upload_paths and not request.vault_run_ids:
            raise IngestionError("INGESTION_EMPTY", "Provide uploads or vault IDs")
        ir = CanonicalResearchIR(ir_id=uuid4().hex,
            source_mode="hybrid" if request.upload_paths and request.vault_run_ids else "vault" if request.vault_run_ids else "upload",
            title=request.title)
        positive_bundles = 0
        for run_id in request.vault_run_ids:
            bundle = load_vault_bundle(self.workspace_root, run_id)
            ir.vault_run_ids.append(run_id)
            ir.meaningfulness_certificates.update(bundle.meaningfulness_certificates)
            if ir.meaningfulness_certificate is None and bundle.meaningfulness_certificate is not None:
                ir.meaningfulness_certificate = bundle.meaningfulness_certificate
            if ir.epistemic_verdict is None and bundle.epistemic_verdict is not None:
                ir.epistemic_verdict = bundle.epistemic_verdict
            if ir.certified_valid_domain is None and bundle.certified_valid_domain is not None:
                ir.certified_valid_domain = bundle.certified_valid_domain
            if bundle.research_import_role == "NEGATIVE_CONSTRAINT_ONLY":
                ir.negative_constraints.extend(bundle.negative_constraints)
                if not request.upload_paths and positive_bundles == 0:
                    ir.research_import_role = "NEGATIVE_CONSTRAINT_ONLY"
            else:
                positive_bundles += 1
                ir.research_import_role = bundle.research_import_role
                ir.negative_constraints.extend(bundle.negative_constraints)
            ir.equations.extend(bundle.equations)
            ir.constants.extend(bundle.constants)
            ir.state_variables.extend(bundle.state_variables)
            ir.disambiguation_flags.extend(bundle.disambiguation_flags)
            ir.runtime_predicates.extend(bundle.runtime_predicates)
            if bundle.research_reference is not None:
                if ir.research_reference is not None:
                    raise IngestionError('VAULT_MULTIPLE_REFERENCES_UNSUPPORTED', 'Build each pinned research reference separately')
                ir.research_reference = bundle.research_reference
            ir.macro_domains.extend(bundle.macro_domains)
            ir.pinned_kernel_code = "\n\n".join(filter(None, [ir.pinned_kernel_code, bundle.pinned_kernel_code]))
            for name in ("time_horizon_si_s", "default_dt_si_s", "default_steps"):
                if getattr(ir, name) is None:
                    setattr(ir, name, getattr(bundle, name))
        if ir.negative_constraints and not ir.equations and not request.upload_paths:
            raise IngestionError(
                "NEGATIVE_CONSTRAINT_ONLY",
                f"Vault run(s) {request.vault_run_ids} are VERIFIED_FALSIFICATION (negative constraint only) and cannot be admitted for positive executable profiles",
                details={
                    "epistemic_verdict": ir.epistemic_verdict or "VERIFIED_FALSIFICATION",
                    "research_import_role": "NEGATIVE_CONSTRAINT_ONLY",
                },
            )
        if ir.meaningfulness_certificates:
            from ...research.epistemic import parse_run_certificate
            certificates = [parse_run_certificate(raw) for raw in ir.meaningfulness_certificates.values()]
            if all(c.epistemic_verdict.value in {"VERIFIED_DISCOVERY", "VERIFIED_BOUNDARY_DISCOVERY"} for c in certificates):
                ir.research_import_role = "POSITIVE_EVIDENCE"
                if len(certificates) == 1:
                    ir.meaningfulness_certificate = certificates[0].model_dump(mode="json")
                    ir.epistemic_verdict = certificates[0].epistemic_verdict.value
                    ir.certified_valid_domain = getattr(certificates[0],"certified_valid_domain",None)
        raw_eqs, raw_constants, source_text = [], [], []
        for index, upload in enumerate(request.upload_paths):
            paths = validate_and_unpack_upload(Path(upload), self.workspace_root / ".invar" / "builder_ingestion" / ir.ir_id / str(index))
            for path in paths:
                doc = parse_document(path)
                source_text.append(doc.text)
                if doc.function_signatures:
                    ir.summary += "\nStatic signatures: " + "; ".join(doc.function_signatures)
                ir.uploaded_files.append(str(path))
                ir.datasets_attached.extend(doc.datasets)
                ir.disambiguation_flags.extend(doc.flags)
                raw_eqs.extend(doc.equations)
                raw_constants.extend(doc.constants)
        if raw_eqs or raw_constants:
            equations, constants, flags = lower_and_verify_equations(raw_eqs, raw_constants, {"source_text": "\n".join(source_text)})
            ir.equations.extend(equations)
            ir.constants.extend(constants)
            ir.disambiguation_flags.extend(flags)
        unique = {}
        for constant in ir.constants:
            old = unique.get(constant.symbol)
            if old and (old.si_value != constant.si_value or old.dimension.exponents != constant.dimension.exponents):
                raise FormalGateError("Z3_SPEC_UNSAT", f"Conflicting definitions of {constant.symbol}")
            unique[constant.symbol] = constant
        ir.constants = list(unique.values())
        for index, constant in enumerate(ir.constants):
            constant.constant_id = f"constant:{index}"
        for index, equation in enumerate(ir.equations):
            equation.equation_id = f"equation:{index}"
            if equation.role == EquationRole.EVOLUTION_LAW and equation.lhs_symbol and equation.lhs_symbol not in {s.symbol for s in ir.state_variables}:
                state_dim = SIDimensionVectorModel()
                if equation.dimension_lhs is not None:
                    exps = list(equation.dimension_lhs.exponents)
                    exps[2] = exps[2] + 1
                    state_dim = SIDimensionVectorModel(exponents=tuple(exps))
                ir.state_variables.append(StateVariableSpec(symbol=equation.lhs_symbol, slot_index=len(ir.state_variables),
                    dimension=state_dim, provenance=equation.provenance))
        for index, flag in enumerate(ir.disambiguation_flags):
            flag.flag_id = f"flag:{index}"
        states = {state.symbol: state for state in ir.state_variables}
        for equation in ir.equations:
            if equation.role == EquationRole.INITIAL_CONDITION and equation.lhs_symbol in states:
                _, _, value = equation_parts(equation.canonical_latex)
                if value.is_number:
                    states[equation.lhs_symbol].initial_value = float(value)
        if not (ir.constants or ir.equations or ir.datasets_attached or ir.pinned_kernel_code or ir.disambiguation_flags):
            raise IngestionError("INGESTION_EMPTY", "No scientific content was extracted")
        if request.upload_paths and request.vault_run_ids:
            reverify_ir(ir)
        else:
            check_z3(ir)
        if self.llm_client is not None and any(not f.resolved for f in ir.disambiguation_flags):
            from .disambiguator import disambiguate_ir
            ir = disambiguate_ir(ir, self.llm_client,
                failure_capture_dir=self.workspace_root/'.invar/builder/diagnostics/disambiguation')
        return seal_ir(ir)

    @staticmethod
    def apply_override(ir: CanonicalResearchIR, override: IRReviewOverride) -> CanonicalResearchIR:
        return apply_ir_override(ir, override)

"""Bounded LLM proposals behind provenance, dimensions and SMT rejection gates."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from invar_agent.llm.errors import complete_for_builder, redact_secrets

from .equation_lowerer import FormalGateError, normalize_constant
from .pipeline import reverify_ir, seal_ir
from .schemas import CanonicalResearchIR, DisambiguationProposal, DisambiguationResponse, RawConstant


def _normalized(text: str) -> str:
    return " ".join(text.split())


def disambiguate_ir(ir: CanonicalResearchIR, llm_client, *, failure_capture_dir: Path | None = None) -> CanonicalResearchIR:
    flags = [flag for flag in ir.disambiguation_flags if not flag.resolved
             and not flag.source_span.location.startswith("vault:")]
    if not flags:
        return ir.model_copy(deep=True)
    eligible_ids = {flag.flag_id for flag in flags}
    prompt = json.dumps({"flags": [{"flag_id": f.flag_id, "trigger": f.trigger.value,
        "target": f.target_symbol_or_eq, "file": f.source_span.file, "location": f.source_span.location,
        "excerpt": f.context_excerpt[:1500]} for f in flags]}, ensure_ascii=True)
    schema = json.dumps(DisambiguationResponse.model_json_schema(), ensure_ascii=True)
    response = complete_for_builder(llm_client, prompt, system="Return JSON matching this schema: " + schema + ". Propose only flagged items; quote the cited excerpt verbatim. Never invent values or units. raw_value must be a JSON string preserving the quoted spelling, not a JSON number. Omit items without sufficient evidence; never emit placeholder proposals. A single supported binding can resolve multiple related flags.", max_tokens=2048, temperature=0.0)
    result = ir.model_copy(deep=True)
    result.llm_disambiguation_tokens_used += max(0, response.input_tokens) + max(0, response.output_tokens)
    failed = False

    def capture_failure():
        if failure_capture_dir is None:
            return
        directory = Path(failure_capture_dir)
        directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        path = directory/(uuid4().hex+'.json')
        key = getattr(llm_client,'api_key',None)
        payload = dict(ir_id=ir.ir_id,input_tokens=response.input_tokens,output_tokens=response.output_tokens,
            raw_response=redact_secrets(response.content,api_key=key),
            diagnostics=[redact_secrets(f.resolution_note or '',api_key=key) for f in result.disambiguation_flags])
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as stream:
            json.dump(payload,stream,ensure_ascii=True,indent=2)
        for flag in result.disambiguation_flags:
            if not flag.resolved:
                flag.resolution_note = (flag.resolution_note or 'DISAMBIGUATION_UNRESOLVED')+f'; response_capture={path}'
    try:
        content = response.content.strip()
        fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
        decoded = json.loads(fenced[1] if fenced else content)
        if not isinstance(decoded, dict) or not isinstance(decoded.get('proposals', []), list):
            raise ValueError('Expected a JSON object containing a proposals array')
        DisambiguationResponse.model_validate({**decoded, 'proposals': []})
    except (ValidationError, ValueError) as exc:
        diagnostic = ', '.join(f"{'.'.join(map(str,item['loc']))}:{item['type']}" for item in exc.errors(include_input=False)[:4]) if isinstance(exc, ValidationError) else 'json_envelope_invalid'
        for flag in result.disambiguation_flags:
            if not flag.resolved:
                flag.resolution_note = f"DISAMBIGUATION_SCHEMA_REJECTED: {diagnostic}"
        capture_failure()
        return seal_ir(result)
    proposals = []
    for raw in decoded.get('proposals', []):
        try:
            proposals.append(DisambiguationProposal.model_validate(raw))
        except ValidationError as exc:
            failed = True
            diagnostic = ', '.join(f"{'.'.join(map(str,item['loc']))}:{item['type']}" for item in exc.errors(include_input=False)[:4])
            target = raw.get('flag_id') if isinstance(raw, dict) else None
            for flag in result.disambiguation_flags:
                if not flag.resolved and (target is None or flag.flag_id == target):
                    flag.resolution_note = f"DISAMBIGUATION_SCHEMA_REJECTED: {diagnostic}"
    for proposal in proposals:
        if proposal.flag_id not in eligible_ids:
            continue
        flag = next((f for f in result.disambiguation_flags if f.flag_id == proposal.flag_id and not f.resolved), None)
        if flag is None:
            continue
        candidate = result.model_copy(deep=True)
        candidate_flag = next(f for f in candidate.disambiguation_flags if f.flag_id == flag.flag_id)
        try:
            kinds = {"UNRESOLVED_SYMBOL": {"constant_binding", "equation_patch"},
                     "PROSE_ONLY_CONSTANT": {"constant_binding"},
                     "AMBIGUOUS_STATE_ORDERING": {"state_ordering"},
                     "IMPLICIT_BOUNDARY_CONDITION": {"boundary_condition"}}
            if proposal.proposal_kind not in kinds[flag.trigger.value]:
                raise FormalGateError("DISAMBIGUATION_SCOPE_REJECTED", "Proposal does not address this trigger")
            span = proposal.provenance
            if (span.file != flag.source_span.file or span.location != flag.source_span.location
                    or not _normalized(span.quote) or _normalized(span.quote) not in _normalized(flag.context_excerpt[:1500])):
                raise FormalGateError("DISAMBIGUATION_PROVENANCE_REJECTED", "Quote does not occur at the flagged source location")
            if proposal.proposal_kind == "constant_binding":
                if not proposal.symbol or proposal.raw_value is None or proposal.raw_unit is None:
                    raise FormalGateError("DISAMBIGUATION_SCHEMA_REJECTED", "Missing binding fields")
                if flag.target_symbol_or_eq and proposal.symbol != flag.target_symbol_or_eq:
                    raise FormalGateError("DISAMBIGUATION_SCOPE_REJECTED", proposal.symbol)
                constant = normalize_constant(RawConstant(symbol=proposal.symbol, value=proposal.raw_value,
                    unit=proposal.raw_unit, provenance=span), f"constant:{len(candidate.constants)}")
                # A quoted value is not permission to invent another value or unit.
                binding = rf"(?<![\w.+-]){re.escape(proposal.raw_value)}\s*{re.escape(proposal.raw_unit)}(?![\w/^*])"
                if not re.search(binding, span.quote):
                    raise FormalGateError("DISAMBIGUATION_PROVENANCE_REJECTED", "Binding absent from quote")
                old = next((c for c in candidate.constants if c.symbol == proposal.symbol), None)
                if old:
                    if old.dimension.exponents != constant.dimension.exponents or old.dimension.semantic_tag != constant.dimension.semantic_tag:
                        raise FormalGateError("HIN_SI_DIM_MISMATCH", proposal.symbol)
                    constant.constant_id = old.constant_id
                    candidate.constants.remove(old)
                constant.disambiguated_by_llm = True
                constant.bounds = proposal.bounds
                candidate.constants.append(constant)
            elif proposal.proposal_kind in {"equation_patch", "boundary_condition"}:
                target = next((e for e in candidate.equations if e.equation_id == flag.target_symbol_or_eq), None)
                if target is None:
                    raise FormalGateError("DISAMBIGUATION_SCOPE_REJECTED", "No flagged equation")
                if proposal.proposal_kind == "boundary_condition":
                    state = next((s for s in candidate.state_variables if s.symbol == target.lhs_symbol), None)
                    if state is None or proposal.bounds is None:
                        raise FormalGateError("DISAMBIGUATION_SCOPE_REJECTED", "No typed state bounds")
                    for bound in (proposal.bounds.lower, proposal.bounds.upper):
                        if bound is not None and not re.search(rf"(?<![\w.+-]){re.escape(format(bound, 'g'))}(?![\w.])", span.quote):
                            raise FormalGateError("DISAMBIGUATION_PROVENANCE_REJECTED", "Boundary absent from quote")
                    state.bounds = proposal.bounds
                elif not proposal.latex_expression:
                    raise FormalGateError("DISAMBIGUATION_SCHEMA_REJECTED", "Missing equation")
                if proposal.latex_expression:
                    target.canonical_latex = proposal.latex_expression
                target.disambiguated_by_llm = True
            else:
                order = proposal.ordered_state_symbols or []
                states = {s.symbol: s for s in candidate.state_variables}
                if len(order) != len(states) or set(order) != set(states):
                    raise FormalGateError("DISAMBIGUATION_SCOPE_REJECTED", "Invalid state permutation")
                candidate.state_variables = [states[name].model_copy(update={"slot_index": i}) for i, name in enumerate(order)]
            reverify_ir(candidate)
            if (proposal.proposal_kind == "boundary_condition"
                    or proposal.proposal_kind == "constant_binding" and flag.trigger.value == "UNRESOLVED_SYMBOL") and not candidate_flag.resolved:
                raise FormalGateError("DISAMBIGUATION_UNVERIFIED", "Target remains unresolved")
            if proposal.proposal_kind == "equation_patch" and not next(e for e in candidate.equations if e.equation_id == target.equation_id).hin_homogeneous:
                raise FormalGateError("DISAMBIGUATION_UNVERIFIED", "Equation remains unverified")
            candidate_flag.resolved = True
            candidate_flag.resolution_note = "Provenance, HIN and Z3 gates passed"
            result = candidate
        except (ValueError, TypeError) as exc:
            failed = True
            flag.resolution_note = f"DISAMBIGUATION_REJECTED: {exc}"
    if failed or any(not f.resolved and f.flag_id in eligible_ids for f in result.disambiguation_flags):
        capture_failure()
    return seal_ir(result)

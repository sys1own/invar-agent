"""Literal formula ingestion; no guessed domains, SI units, or paraphrased spans."""
from __future__ import annotations

import ast
import copy
import re

from . import meaningfulness as m


def _quote_tokens(text):
    operators = {"≥":">=", "≤":"<=", "≠":"!=", "−":"-"}
    return [(operators.get(match[0],match[0]),match.start(),match.end()) for match in re.finditer(
        r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[^\W\d]\w*|>=|<=|==|!=|\*\*|\S",text)]


def _anchor_evidence(prompt,pack):
    pack = copy.deepcopy(pack)
    tokens = _quote_tokens(prompt)
    values = [token[0] for token in tokens]
    for evidence in [pack.get("evidence",{}),*(extension.get("evidence",{}) for extension in pack.get("extensions",()))]:
        for obligation in evidence.get("obligations",()):
            quote = obligation.get("prompt_quote","")
            span = obligation.get("prompt_span",())
            if not isinstance(quote,str) or not isinstance(span,(list,tuple)):
                raise ValueError("obligation quote and prompt span must be explicit text and offsets")
            if len(span) == 2 and all(type(v) is int for v in span):
                start,end = span
                if 0 <= start < end <= len(prompt) and prompt[start:end] == quote: continue
            needle = [token[0] for token in _quote_tokens(quote)]
            matches = [(tokens[i][1],tokens[i+len(needle)-1][2]) for i in range(len(tokens)-len(needle)+1)
                if needle and values[i:i+len(needle)] == needle]
            if len(matches) > 1 and len(span) == 2 and all(type(v) is int for v in span):
                matches = [(start,end) for start,end in matches if start <= span[0] < end or start < span[1] <= end]
            if len(matches) != 1: raise ValueError("obligation quote has no unique immutable prompt span")
            start,end = matches[0]
            obligation.update(prompt_quote=prompt[start:end],prompt_span=[start,end])
    return pack


def attach_formal_evidence(blueprint):
    prompt = blueprint.research_prompt or blueprint.falsifiable_hypothesis
    if blueprint.ihcg_evidence is not None:
        return blueprint.model_copy(update={"ihcg_evidence":_anchor_evidence(prompt,blueprint.ihcg_evidence)})
    bounds = {match[1]:(float(match[2]),float(match[3])) for match in re.finditer(
        r"\b([A-Za-z_]\w*)\s+in\s*\[\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\]",prompt)}
    claims, assumptions, equations = [], [], {}
    for match in re.finditer(r"(?im)^\s*(claim|hypothesis|assume|model|evolution)\s*:\s*([^\n]+)",prompt):
        label, formula = match[1].lower(),match[2].strip().rstrip(".")
        if label in {"claim","hypothesis"}:
            start = match.start(2)+len(match[2])-len(match[2].lstrip())
            claims.append(dict(id=f"h{len(claims)}",expression=formula,obligation_class="HYPOTHESIS_TARGET",
                prompt_quote=prompt[start:start+len(formula)],prompt_span=[start,start+len(formula)]))
        elif label == "assume": assumptions.append(formula)
        else:
            item = re.fullmatch(r"(?:d([A-Za-z_]\w*)/dt|([A-Za-z_]\w*))\s*=\s*([^=].*)",formula)
            if item is None: return blueprint
            state,name,rhs = item.groups()
            key = f"d_{state}_dt" if state else name
            equations[key] = dict(expression=rhs,si_dim=[0]*7,**(dict(kind="ODE_RHS",state_symbol=state) if state else {}))
    if not claims:
        formula = blueprint.falsifiable_hypothesis.strip()
        start = prompt.find(formula)
        if start >= 0:
            try: node = m._parse(formula)
            except (ValueError,SyntaxError): node = None
            if isinstance(node,(ast.Compare,ast.BoolOp)):
                claims.append(dict(id="h0",expression=formula,obligation_class="HYPOTHESIS_TARGET",
                    prompt_quote=formula,prompt_span=[start,start+len(formula)]))
    if not claims or not bounds: return blueprint
    for match in re.finditer(r"(?m)^\s*([A-Za-z_]\w*)\s*=(?!=)\s*([^\n]+)$",prompt):
        name,rhs = match[1],match[2].strip()
        if name not in equations:
            try: m._parse(rhs)
            except (ValueError,SyntaxError): return blueprint
            equations[name] = dict(expression=rhs,si_dim=[0]*7)
    if not claims or not equations or not bounds: return blueprint
    dimensions = {name:[int(v) for v in match[2].split(",")] for match in re.finditer(
        r"\b([A-Za-z_]\w*)\s*:\s*SI\s*\[([\d,\s+-]+)\]",prompt) for name in [match[1]]}
    dimensionless = re.search(r"\b(?:all variables|all quantities) are dimensionless\b",prompt,re.I)
    if not dimensionless and set(dimensions) != set(bounds): return blueprint
    for name in bounds: dimensions.setdefault(name,[0]*7)
    if any(len(v) != 7 for v in dimensions.values()): return blueprint
    if not set(bounds) <= blueprint.variable_map.keys(): return blueprint
    slots = []
    states = {e["state_symbol"] for e in equations.values() if e.get("kind") == "ODE_RHS"}
    for name in bounds:
        role = "DYNAMIC_STATE" if name in states else "DERIVED_OBSERVABLE" if name in equations else "CONTROL_PARAM"
        slots.append(dict(slot_idx=blueprint.variable_map[name],symbol=name,si_dim=dimensions[name],role=role))
    for name,equation in equations.items():
        if equation.get("kind") == "ODE_RHS":
            dims = list(dimensions[equation["state_symbol"]]); dims[2] -= 1
            equation["si_dim"] = dims
        else: equation["si_dim"] = dimensions.get(name,[0]*7)
    mechanism = re.search(r"(?im)^\s*mechanism\s*:\s*([A-Za-z_][\w, ]*)$",prompt)
    quantifier = re.search(r"(?im)^\s*quantifier\s*:\s*(FORALL|EXISTS|BOUNDARY)\s*$",prompt)
    archetype = re.search(r"(?im)^\s*archetype\s*:\s*(ODE_PDE|PARAM_SWEEP|MONTE_CARLO|FIXED_POINT)\s*$",prompt)
    pack = dict(evidence=dict(slots=slots,equations=equations,environmental_assumptions=assumptions,obligations=claims,
        mechanism_parameters=[s.strip() for s in mechanism[1].split(",")] if mechanism else [],
        quantifier_mode=quantifier[1].upper() if quantifier else "FORALL"),domain=bounds,archetype=archetype[1].upper() if archetype else "ODE_PDE")
    # Validate literal spans and vocabulary before adding anything to a blueprint.
    m.IhcgCompiler().compile_from_prompt(prompt,**pack)
    return blueprint.model_copy(update={"ihcg_evidence":pack})

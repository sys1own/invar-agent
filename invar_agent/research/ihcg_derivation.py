"""Derive bounded contract coordinates without guessing scientific obligations."""

import ast
import copy
import json
import math
import re
from fractions import Fraction

from . import meaningfulness as m
from .formal_evidence import _quote_tokens
from .ihcg_normalization import normalize_ihcg_evidence


_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_UNITS = {
    'm': ((0,1,0,0,0,0,0),1), 'km': ((0,1,0,0,0,0,0),1000),
    's': ((0,0,1,0,0,0,0),1), 'ms': ((0,0,1,0,0,0,0),Fraction(1,1000)),
    'kg': ((1,0,0,0,0,0,0),1), 'g': ((1,0,0,0,0,0,0),Fraction(1,1000)),
    'K': ((0,0,0,0,1,0,0),1), 'J': ((1,2,-2,0,0,0,0),1),
    'W': ((1,2,-3,0,0,0,0),1), 'kW': ((1,2,-3,0,0,0,0),1000),
    'Pa': ((1,-1,-2,0,0,0,0),1),
    'm/s': ((0,1,-1,0,0,0,0),1), 'km/s': ((0,1,-1,0,0,0,0),1000),
}


def extract_prompt_domains(prompt, symbols):
    facts = {}
    for symbol in symbols:
        name = rf"(?<!\w){re.escape(symbol)}(?!\w)"
        patterns = [
            (rf"{name}\s+in\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]", False),
            (rf"{name}\s+(?:between|from)\s+({_NUMBER})\s+(?:and|to)\s+({_NUMBER})(?!\w|\.\d)", False),
            (rf"{name}\s*(?:=(?!=)|is fixed at)\s*({_NUMBER})(?!\w|\.\d)", True),
        ]
        candidates = []
        for pattern, point in patterns:
            for match in re.finditer(pattern,prompt):
                if re.search(r"\b(?:at|when|initially|if|not|except|outside)\s*$",prompt[max(0,match.start()-20):match.start()],re.I): continue
                if re.match(r'\s*[* /+^(-]',prompt[match.end():].lstrip(' ')): continue
                pair = [float(match[1]),float(match[1] if point else match[2])]
                if not all(map(math.isfinite,pair)) or pair[0] > pair[1]:
                    raise ValueError(f'prompt domain for {symbol}: finite ordered bounds required')
                unit = re.match(r'\s+([A-Za-zµμ]+(?:/[A-Za-z]+)?)',prompt[match.end():])
                token = unit[1] if unit else None
                if token in {'and','or','with','at','is','to','for','while','in'}: token = None
                candidates.append(dict(bounds=pair,literals=[match[1],match[1] if point else match[2]],unit=token,
                    quote=match[0],span=[match.start(),match.end()],point=point))
        if len({tuple(item['bounds']) for item in candidates}) > 1:
            raise ValueError(f'ambiguous exact prompt domains for {symbol}: {candidates}')
        if candidates: facts[symbol] = candidates[0]
    return facts


def derive_ihcg_evidence(question, blueprint, raw):
    pack = copy.deepcopy(raw)
    if not isinstance(pack,dict) or not isinstance(pack.get('evidence'),dict):
        return normalize_ihcg_evidence(pack)
    evidence = pack['evidence']
    audit = {}
    typed = blueprint.get('typed_slots') or evidence.get('slots',[])
    if not isinstance(typed,(list,tuple)) or any(not isinstance(slot,dict) for slot in typed):
        raise ValueError('typed_slots: expected an array of typed slot objects')
    slots, fixed, aliases = [], {}, {}
    for original in typed:
        slot = copy.deepcopy(original)
        name = slot.get('symbol')
        if not isinstance(name,str) or not name.isidentifier(): raise ValueError(f'typed_slots: invalid symbol {name!r}')
        value, alias = slot.pop('value',None), slot.pop('prompt_symbol',None)
        index = blueprint.get('variable_map',{}).get(name,slot.get('slot_idx'))
        if slot.get('slot_idx') is not None and index != slot['slot_idx']:
            raise ValueError(f'typed slot binding conflict for {name}')
        slot['slot_idx'] = index
        slots.append(slot)
        aliases[name] = alias or name
        if value is not None:
            if slot.get('role') != 'CONTROL_PARAM': raise ValueError(f'only a fixed CONTROL_PARAM may declare value: {name}')
            fixed[name] = value
    evidence['slots'] = slots
    raw_domain = pack.get('domain',{})
    if not isinstance(raw_domain,dict): raise ValueError('ihcg_evidence.domain: expected an object')
    pack['domain'] = {}
    pack = normalize_ihcg_evidence(pack)
    evidence = pack['evidence']
    notes = pack.get('notes',[])
    if not isinstance(notes,list) or any(not isinstance(note,str) for note in notes):
        raise ValueError('ihcg_evidence.notes: expected an array of text')
    obligations = evidence.get('obligations',[])
    if not isinstance(obligations,(list,tuple)) or any(not isinstance(item,dict) or not isinstance(item.get('expression'),str) for item in obligations):
        raise ValueError('ihcg_evidence.evidence.obligations: expected executable obligation objects')
    obligation_text = {tuple(token[0] for token in _quote_tokens(item.get(key,''))) for item in obligations for key in ('expression','prompt_quote')}
    assumptions, moved = [], []
    environment = evidence.get('environmental_assumptions',[])
    if not isinstance(environment,(list,tuple)): raise ValueError('environmental_assumptions: expected an array')
    for index, expression in enumerate(environment):
        try: m._parse(expression)
        except (ValueError,SyntaxError):
            if not isinstance(expression,str): raise ValueError(f'environmental_assumptions[{index}]: expected text')
            if tuple(token[0] for token in _quote_tokens(expression)) in obligation_text:
                raise ValueError(f'environmental_assumptions[{index}] is a sealed obligation and cannot be moved to notes')
            notes.append(expression)
            moved.append(dict(index=index,text=expression,reason='NON_EXECUTABLE_ENVIRONMENT_NOTE'))
        else: assumptions.append(expression)
    evidence['environmental_assumptions'] = assumptions
    if notes: pack['notes'] = list(dict.fromkeys(notes))
    if moved: audit['environment_notes'] = moved
    declared = {slot['symbol'] for slot in slots}
    if len(declared) != len(slots): raise ValueError('typed_slots: duplicate symbol declarations')
    mechanisms = evidence.get('mechanism_parameters',[])
    if not isinstance(mechanisms,(list,tuple)) or any(not isinstance(name,str) for name in mechanisms):
        raise ValueError('ihcg_evidence.evidence.mechanism_parameters: expected symbol strings')
    references = set(mechanisms)
    references.update(slot['symbol'] for slot in slots if slot.get('stochastic_role'))
    expressions = [item['expression'] for item in obligations]+assumptions
    for index,item in enumerate(obligations):
        try: m._parse(item['expression'])
        except (ValueError,SyntaxError) as error:
            raise ValueError(f'ihcg_evidence.evidence.obligations[{index}].expression: {error}') from error
    definitions = {}
    for name, equation in evidence.get('equations',{}).items():
        node = m._parse(equation['expression'])
        expressions.append(equation['expression'])
        if name in declared: references.add(name)
        if equation.get('state_symbol'): references.add(equation['state_symbol'])
        if name in declared and equation.get('kind','ALGEBRAIC') == 'ALGEBRAIC' and not isinstance(node,(ast.Compare,ast.BoolOp)):
            definitions[name] = node
    for expression in expressions: references.update(m._variables(m._parse(expression)))
    if references-declared:
        raise ValueError(f'missing typed slots for referenced symbols: {sorted(references-declared)}')
    selected = [slot for slot in slots if slot['symbol'] in references]
    for index,slot in enumerate(selected):
        try: m.SlotSpec.model_validate_json(json.dumps(slot,allow_nan=False))
        except ValueError as error: raise ValueError(f'typed_slots[{index}]: {error}') from error
    if len(selected) != len(slots): audit['unreferenced_slots'] = [slot for slot in slots if slot['symbol'] not in references]
    evidence['slots'] = selected
    if 'control_values' in evidence:
        evidence['control_values'] = {name:value for name,value in evidence['control_values'].items() if name in references}
    pack['domain'] = {name:bounds for name,bounds in raw_domain.items() if name in references}
    pack = normalize_ihcg_evidence(pack)
    domain = pack['domain']
    prompt_facts = extract_prompt_domains(question,{aliases[name] for name in references})
    for name in references:
        fact = prompt_facts.get(aliases[name])
        if fact is not None:
            slot = next(slot for slot in selected if slot['symbol'] == name)
            if fact['point'] and slot['role'] != 'CONTROL_PARAM': continue
            bounds = fact['bounds']
            if fact['unit']:
                if fact['unit'] not in _UNITS:
                    audit.setdefault('unresolved_units',{})[name] = fact
                    continue
                dimensions, scale = _UNITS[fact['unit']]
                if tuple(slot['si_dim']) != dimensions: raise ValueError(f'prompt unit and typed SI dimensions disagree for {name}: {fact["unit"]}')
                coordinate = slot.get('unit_scale',(1,1))
                factor = Fraction(scale)*Fraction(coordinate[1],coordinate[0])
                bounds = [m._IV.rational(Fraction(fact['literals'][0])*factor).lo,
                          m._IV.rational(Fraction(fact['literals'][1])*factor).hi]
            domain[name] = bounds
            audit.setdefault('prompt_domains',{})[name] = fact
        elif name in fixed or name in blueprint.get('extracted_constants',{}):
            slot = next(slot for slot in selected if slot['symbol'] == name)
            if slot['role'] == 'CONTROL_PARAM':
                value = fixed.get(name,blueprint.get('extracted_constants',{}).get(name))
                if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value): raise ValueError(f'invalid fixed constant {name}')
                domain[name] = [value,value]
                audit.setdefault('fixed_constants',{})[name] = value
    free_missing = references-set(definitions)-set(domain)
    if free_missing: raise ValueError(f'missing finite domains for free symbols: {sorted(free_missing)}')
    pending = {name:node for name,node in definitions.items() if name not in domain}
    resolved = set(domain)
    while pending:
        progressed = False
        for name,node in list(pending.items()):
            if not m._variables(node) <= resolved: continue
            try: enclosure = m._iv(node,domain)
            except (ValueError,KeyError,ZeroDivisionError,OverflowError) as error:
                raise ValueError(f'cannot derive a finite domain for {name}: {error}') from error
            if not math.isfinite(enclosure.lo) or not math.isfinite(enclosure.hi):
                raise ValueError(f'cannot derive a finite domain for {name}: unbounded interval')
            domain[name] = [enclosure.lo,enclosure.hi]
            audit.setdefault('derived_domains',{})[name] = domain[name]
            pending.pop(name); resolved.add(name); progressed = True
        if not progressed:
            missing = set(pending)-set(domain)
            if missing: raise ValueError(f'missing finite domains for cyclic or unresolved symbols: {sorted(missing)}')
            break
    if audit:
        history = pack.setdefault('execution',{}).setdefault('normalization_source',{})
        history['derivation'] = {**history.get('derivation',{}),**audit}
        history['derivation'].setdefault('provided_domains',copy.deepcopy(raw_domain))
    return pack

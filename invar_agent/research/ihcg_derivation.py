"""Derive bounded contract coordinates across the four-namespace Blueprint AST."""

import ast
import copy
import json
import math
import re
from fractions import Fraction

from . import meaningfulness as m
from .blueprint_compiler import (
    STANDARD_SI_CONSTANTS,
    _collect_assumptions_ledger,
    _extract_module_scalar_constants,
    _find_bijective_aliases,
    _normalize_constants_namespace,
    _normalize_derived_expressions,
    _resolve_derived_topological,
    _substitute_ast_symbols,
)
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
            (rf"{name}\s+(?:in|across|over)\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]", False),
            (rf"{name}\s+(?:between|from)\s+({_NUMBER})\s+(?:and|to)\s+({_NUMBER})(?!\w|\.\d)", False),
            (rf"{name}\s*(?:=(?!=)|is fixed at)\s*({_NUMBER})(?!\w|\.\d)", True),
        ]
        candidates = []
        for pattern, point in patterns:
            for match in re.finditer(pattern,prompt):
                if re.search(r"\b(?:at|when|initially|if|not|except|outside)\s*$",prompt[max(0,match.start()-20):match.start()],re.I): continue
                if re.search(r"[+\-*/^=<>]\s*$",prompt[max(0,match.start()-10):match.start()]): continue
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
    if not isinstance(pack, dict) or not isinstance(pack.get('evidence'), dict):
        return normalize_ihcg_evidence(pack)
    evidence = pack['evidence']
    audit = {}

    # Four-namespace inputs: typed_slots, constants, derived_expressions, assumptions_ledger.
    assumptions_ledger = _collect_assumptions_ledger(blueprint, pack)
    constants = _normalize_constants_namespace(blueprint, pack)
    derived_exprs = _normalize_derived_expressions(blueprint, pack)
    code_constants = _extract_module_scalar_constants(str(blueprint.get('code_template') or ''))

    # Strip non-IHCG helper keys before strict schema validation.
    pack.pop('constants', None)
    pack.pop('derived_expressions', None)
    pack.pop('assumptions_ledger', None)
    evidence.pop('constants', None)
    evidence.pop('derived_expressions', None)
    evidence.pop('assumptions_ledger', None)

    typed = blueprint.get('typed_slots') or evidence.get('slots', [])
    if not isinstance(typed, (list, tuple)) or any(not isinstance(slot, dict) for slot in typed):
        raise ValueError('typed_slots: expected an array of typed slot objects')
    slots, fixed, aliases, slot_bounds = [], {}, {}, {}
    variable_map = dict(blueprint.get('variable_map') or {})
    used_indices = {v for v in variable_map.values() if isinstance(v, int)}
    next_idx = (max(used_indices) + 1) if used_indices else 0

    for original in typed:
        slot = copy.deepcopy(original)
        name = slot.get('symbol')
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f'typed_slots: invalid symbol {name!r}')
        if slot.get('role') == 'TIME_STEP':
            slot['role'] = 'DYNAMIC_STATE'
        value, alias = slot.pop('value', None), slot.pop('prompt_symbol', None)
        slot.pop('si_unit', None)
        slot.pop('initial_value', None)
        bounds = slot.pop('bounds', None)
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            try:
                lo, hi = float(bounds[0]), float(bounds[1])
                if math.isfinite(lo) and math.isfinite(hi) and lo <= hi:
                    slot_bounds[name] = [lo, hi]
            except (ValueError, TypeError):
                pass
        index = variable_map.get(name, slot.get('slot_idx'))
        if slot.get('slot_idx') is not None and index != slot['slot_idx']:
            raise ValueError(f'typed slot binding conflict for {name}')
        if index is None:
            index = next_idx
            next_idx += 1
            variable_map[name] = index
        slot['slot_idx'] = index
        slots.append(slot)
        aliases[name] = alias or name
        if value is not None:
            if slot.get('role') != 'CONTROL_PARAM':
                raise ValueError(f'only a fixed CONTROL_PARAM may declare value: {name}')
            fixed[name] = value
    evidence['slots'] = slots
    raw_domain = pack.get('domain', {})
    if not isinstance(raw_domain, dict):
        raise ValueError('ihcg_evidence.domain: expected an object')
    pack['domain'] = {}
    pack = normalize_ihcg_evidence(pack)
    evidence = pack['evidence']
    notes = pack.get('notes', [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError('ihcg_evidence.notes: expected an array of text')
    obligations = evidence.get('obligations', [])
    if not isinstance(obligations, (list, tuple)) or any(not isinstance(item, dict) or not isinstance(item.get('expression'), str) for item in obligations):
        raise ValueError('ihcg_evidence.evidence.obligations: expected executable obligation objects')
    obligation_text = {tuple(token[0] for token in _quote_tokens(item.get(key, ''))) for item in obligations for key in ('expression', 'prompt_quote')}
    assumptions, moved = [], []
    environment = evidence.get('environmental_assumptions', [])
    if not isinstance(environment, (list, tuple)):
        raise ValueError('environmental_assumptions: expected an array')
    for index, expression in enumerate(environment):
        try:
            m._parse(expression)
        except (ValueError, SyntaxError):
            if not isinstance(expression, str):
                raise ValueError(f'environmental_assumptions[{index}]: expected text')
            if tuple(token[0] for token in _quote_tokens(expression)) in obligation_text:
                raise ValueError(f'environmental_assumptions[{index}] is a sealed obligation and cannot be moved to notes')
            notes.append(expression)
            moved.append(dict(index=index, text=expression, reason='NON_EXECUTABLE_ENVIRONMENT_NOTE'))
        else:
            assumptions.append(expression)
    evidence['environmental_assumptions'] = assumptions
    if notes:
        pack['notes'] = list(dict.fromkeys(notes))
    if moved:
        audit['environment_notes'] = moved
    declared = {slot['symbol'] for slot in slots}
    if len(declared) != len(slots):
        raise ValueError('typed_slots: duplicate symbol declarations')

    # Inline pure derived_expressions (from namespace 3 or non-slot ALGEBRAIC helper equations).
    equations = evidence.get('equations', {})
    inline_derived = {
        sym: spec for sym, spec in derived_exprs.items()
        if sym not in declared and not spec.get('promote_to_slot')
    }
    for eq_name, eq_obj in list(equations.items()):
        if eq_name not in declared and isinstance(eq_obj, dict) and eq_obj.get('kind', 'ALGEBRAIC') == 'ALGEBRAIC':
            expr_str = eq_obj.get('expression', '')
            if isinstance(expr_str, str):
                try:
                    eq_node = m._parse(expr_str)
                except (ValueError, SyntaxError):
                    eq_node = None
                if eq_node is not None and not isinstance(eq_node, (ast.Compare, ast.BoolOp)):
                    inline_derived.setdefault(eq_name, {
                        'symbol': eq_name, 'expression': expr_str,
                        'si_unit': '', 'si_dim': list(eq_obj.get('si_dim') or [0] * 7),
                        'promote_to_slot': False,
                    })
                    equations.pop(eq_name, None)

    if inline_derived:
        ref_pack = set()
        for eq_obj in equations.values():
            if isinstance(eq_obj, dict) and isinstance(eq_obj.get('expression'), str):
                try: ref_pack.update(m._variables(m._parse(eq_obj['expression'])))
                except (ValueError, SyntaxError): pass
        for ob in obligations:
            if isinstance(ob, dict) and isinstance(ob.get('expression'), str):
                try: ref_pack.update(m._variables(m._parse(ob['expression'])))
                except (ValueError, SyntaxError): pass
        for expr in evidence.get('environmental_assumptions', []):
            if isinstance(expr, str):
                try: ref_pack.update(m._variables(m._parse(expr)))
                except (ValueError, SyntaxError): pass
        needed = {}
        frontier = ref_pack & set(inline_derived.keys())
        while frontier:
            sym = frontier.pop()
            if sym in needed: continue
            needed[sym] = inline_derived[sym]
            try:
                frontier.update((m._variables(m._parse(inline_derived[sym]['expression'])) & set(inline_derived.keys())) - set(needed.keys()))
            except (ValueError, SyntaxError):
                pass
        if needed:
            expanded_derived, derived_diags = _resolve_derived_topological(needed)
            if derived_diags:
                raise ValueError(derived_diags[0].message)
            for eq_obj in equations.values():
                if isinstance(eq_obj, dict) and isinstance(eq_obj.get('expression'), str):
                    eq_obj['expression'] = _substitute_ast_symbols(eq_obj['expression'], expanded_derived)
            for ob in obligations:
                ob['expression'] = _substitute_ast_symbols(ob['expression'], expanded_derived)
            evidence['environmental_assumptions'] = [
                _substitute_ast_symbols(expr, expanded_derived) for expr in evidence['environmental_assumptions']
            ]
            assumptions = evidence['environmental_assumptions']
            audit['inlined_derived_expressions'] = sorted(expanded_derived.keys())

    mechanisms = evidence.get('mechanism_parameters', [])
    if not isinstance(mechanisms, (list, tuple)) or any(not isinstance(name, str) for name in mechanisms):
        raise ValueError('ihcg_evidence.evidence.mechanism_parameters: expected symbol strings')

    def _collect_refs():
        refs = set(mechanisms)
        refs.update(slot['symbol'] for slot in slots if slot.get('stochastic_role'))
        exprs = [item['expression'] for item in obligations] + assumptions
        for index, item in enumerate(obligations):
            try:
                m._parse(item['expression'])
            except (ValueError, SyntaxError) as error:
                raise ValueError(f'ihcg_evidence.evidence.obligations[{index}].expression: {error}') from error
        defs = {}
        for name, equation in equations.items():
            node = m._parse(equation['expression'])
            exprs.append(equation['expression'])
            if name in declared:
                refs.add(name)
            if equation.get('state_symbol'):
                refs.add(equation['state_symbol'])
            if name in declared and equation.get('kind', 'ALGEBRAIC') == 'ALGEBRAIC' and not isinstance(node, (ast.Compare, ast.BoolOp)):
                defs[name] = node
        for expression in exprs:
            refs.update(m._variables(m._parse(expression)))
        if blueprint.get('typed_slots'):
            slot_by_idx = {slot.get('slot_idx'): slot['symbol'] for slot in slots if isinstance(slot.get('slot_idx'), int)}
            for contract in (blueprint.get('contracts') or ()):
                for inv in (contract.get('invariants') or ()):
                    assertion = inv.get('assertion') if isinstance(inv, dict) else getattr(inv, 'assertion', '')
                    for idx_str in re.findall(r'(?:state|buf\.data|data)\[(\d+)\]', str(assertion or '')):
                        sym = slot_by_idx.get(int(idx_str))
                        if sym in declared:
                            refs.add(sym)
        return refs, defs

    references, definitions = _collect_refs()

    # Bijective alias resolution (e.g. Delta_p -> delta_p, atp -> atp_mol_m3).
    unresolved = references - declared - set(constants.keys())
    found_aliases = _find_bijective_aliases(unresolved, declared | set(constants.keys()))
    if found_aliases:
        alias_ast = {src: ast.Name(id=tgt, ctx=ast.Load()) for src, tgt in found_aliases.items()}
        for eq_obj in equations.values():
            eq_obj['expression'] = _substitute_ast_symbols(eq_obj['expression'], alias_ast)
            if eq_obj.get('state_symbol') in found_aliases:
                eq_obj['state_symbol'] = found_aliases[eq_obj['state_symbol']]
        for ob in obligations:
            ob['expression'] = _substitute_ast_symbols(ob['expression'], alias_ast)
        evidence['environmental_assumptions'] = [
            _substitute_ast_symbols(expr, alias_ast) for expr in evidence['environmental_assumptions']
        ]
        assumptions = evidence['environmental_assumptions']
        if mechanisms:
            evidence['mechanism_parameters'] = [found_aliases.get(p, p) for p in mechanisms]
            mechanisms = evidence['mechanism_parameters']
        audit['resolved_aliases'] = found_aliases
        references, definitions = _collect_refs()

    # Auto-populate unambiguous CODATA SI constants or module-level code_template constants.
    for sym in sorted(references - declared - set(constants.keys())):
        if sym in STANDARD_SI_CONSTANTS:
            val, unit, dims = STANDARD_SI_CONSTANTS[sym]
            constants[sym] = dict(symbol=sym, value=val, si_unit=unit, si_dim=list(dims), source='REFERENCE_REGISTRY')
            assumptions_ledger.append(dict(
                kind='SI_CONSTANT', symbol_or_field=sym, value_or_expr=f'{val} {unit}',
                source='REFERENCE_REGISTRY', rationale=f'Auto-populated CODATA 2018 SI constant {sym}.',
            ))
        elif sym in code_constants:
            val = code_constants[sym]
            constants[sym] = dict(symbol=sym, value=val, si_unit='', si_dim=[0] * 7, source='LLM_DOMAIN_DEFAULT')
            assumptions_ledger.append(dict(
                kind='SI_CONSTANT', symbol_or_field=sym, value_or_expr=str(val),
                source='LLM_DOMAIN_DEFAULT', rationale=f'Bound module-level constant {sym} from code_template.',
            ))
        elif sym == 'pi':
            constants['pi'] = dict(symbol='pi', value=math.pi, si_unit='', si_dim=[0] * 7, source='REFERENCE_REGISTRY')

    # Bind referenced constants: dimensional -> locked CONTROL_PARAM slot; dimensionless -> inline literal.
    inline_const_ast = {}
    for sym in sorted((references & set(constants.keys())) - declared):
        c_info = constants[sym]
        c_val = float(c_info['value'])
        c_dim = list(c_info.get('si_dim') or [0] * 7)
        if any(d != 0 for d in c_dim) and len(slots) < m.MAX_VARIABLES:
            if sym in variable_map:
                new_idx = variable_map[sym]
            else:
                new_idx = next_idx
                next_idx += 1
                variable_map[sym] = new_idx
            slots.append(dict(symbol=sym, slot_idx=new_idx, si_dim=c_dim, role='CONTROL_PARAM', semantic_tag=0))
            declared.add(sym)
            aliases[sym] = sym
            fixed[sym] = c_val
            evidence.setdefault('control_values', {})[sym] = c_val
            audit.setdefault('promoted_constants', {})[sym] = c_val
        else:
            inline_const_ast[sym] = ast.Constant(value=c_val)
            audit.setdefault('inlined_constants', {})[sym] = c_val

    if inline_const_ast:
        for eq_obj in equations.values():
            eq_obj['expression'] = _substitute_ast_symbols(eq_obj['expression'], inline_const_ast)
        for ob in obligations:
            ob['expression'] = _substitute_ast_symbols(ob['expression'], inline_const_ast)
        evidence['environmental_assumptions'] = [
            _substitute_ast_symbols(expr, inline_const_ast) for expr in evidence['environmental_assumptions']
        ]
        assumptions = evidence['environmental_assumptions']
        references, definitions = _collect_refs()

    if references - declared:
        raise ValueError(f'missing typed slots for referenced symbols: {sorted(references - declared)}')
    selected = [slot for slot in slots if slot['symbol'] in references]
    # Derive trajectory reduction state_symbol and semantic_tag when blueprint left them unspecified.
    dynamic_names = {slot['symbol'] for slot in selected if slot.get('role') == 'DYNAMIC_STATE'}
    for eq_name, equation in equations.items():
        kind = equation.get('kind')
        if kind in {'ODE_RHS', 'FIXED_POINT_RESIDUAL', 'FIXED_POINT_MAP'} and not equation.get('state_symbol') and eq_name in dynamic_names:
            equation['state_symbol'] = eq_name
            audit.setdefault('derived_state_symbols', {})[eq_name] = eq_name
        if kind in {'TRAJECTORY_MAX', 'TRAJECTORY_MIN'}:
            if not equation.get('state_symbol') and str(equation.get('expression', '')).strip() in dynamic_names:
                equation['state_symbol'] = str(equation.get('expression', '')).strip()
                audit.setdefault('derived_trajectory_sources', {})[eq_name] = equation['state_symbol']
            if equation.get('state_symbol') in dynamic_names:
                target_slot = next((s for s in selected if s['symbol'] == eq_name and s.get('role') == 'DERIVED_OBSERVABLE'), None)
                if target_slot is not None:
                    expected_tag = m.TRAJECTORY_MAXIMUM if kind == 'TRAJECTORY_MAX' else m.TRAJECTORY_MINIMUM
                    if target_slot.get('semantic_tag', 0) == 0:
                        target_slot['semantic_tag'] = expected_tag
                        audit.setdefault('derived_trajectory_tags', {})[eq_name] = expected_tag
    for index, slot in enumerate(selected):
        try:
            m.SlotSpec.model_validate_json(json.dumps(slot, allow_nan=False))
        except ValueError as error:
            raise ValueError(f'typed_slots[{index}]: {error}') from error
    if len(selected) != len(slots):
        audit['unreferenced_slots'] = [slot for slot in slots if slot['symbol'] not in references]
    evidence['slots'] = selected
    if assumptions_ledger:
        evidence['assumptions_ledger'] = assumptions_ledger
    if 'control_values' in evidence:
        evidence['control_values'] = {name: value for name, value in evidence['control_values'].items() if name in references}
    pack['domain'] = {name: bounds for name, bounds in raw_domain.items() if name in references}
    for name, bnd in slot_bounds.items():
        if name in references and name not in pack['domain']:
            pack['domain'][name] = bnd
    pack = normalize_ihcg_evidence(pack)
    domain = pack['domain']

    # Non-blocking fallback: only consult literal prompt domains for symbols missing an explicit AST domain.
    missing_for_prompt = {aliases[name] for name in references if name not in domain}
    prompt_facts = {}
    if missing_for_prompt and isinstance(question, str) and question.strip():
        try:
            prompt_facts = extract_prompt_domains(question, missing_for_prompt)
        except ValueError:
            prompt_facts = {}
    for name in references:
        if name in domain:
            continue
        fact = prompt_facts.get(aliases[name])
        if fact is not None:
            slot = next(slot for slot in selected if slot['symbol'] == name)
            if fact['point'] and slot['role'] != 'CONTROL_PARAM':
                continue
            bounds = fact['bounds']
            if fact['unit']:
                if fact['unit'] not in _UNITS:
                    audit.setdefault('unresolved_units', {})[name] = fact
                    continue
                dimensions, scale = _UNITS[fact['unit']]
                if tuple(slot['si_dim']) != dimensions:
                    audit.setdefault('unit_dim_mismatch_skipped', {})[name] = fact
                    continue
                coordinate = slot.get('unit_scale', (1, 1))
                factor = Fraction(scale) * Fraction(coordinate[1], coordinate[0])
                bounds = [m._IV.rational(Fraction(fact['literals'][0]) * factor).lo,
                          m._IV.rational(Fraction(fact['literals'][1]) * factor).hi]
            domain[name] = bounds
            audit.setdefault('prompt_domains', {})[name] = fact
        elif name in fixed or name in constants or name in blueprint.get('extracted_constants', {}):
            slot = next(slot for slot in selected if slot['symbol'] == name)
            if slot['role'] == 'CONTROL_PARAM':
                value = fixed.get(name, constants[name]['value'] if name in constants else blueprint.get('extracted_constants', {}).get(name))
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    raise ValueError(f'invalid fixed constant {name}')
                domain[name] = [value, value]
                audit.setdefault('fixed_constants', {})[name] = value
    free_missing = references - set(definitions) - set(domain)
    if free_missing:
        raise ValueError(f'missing finite domains for free symbols: {sorted(free_missing)}')
    for ctrl_name, ctrl_val in (evidence.get('control_values') or {}).items():
        if ctrl_name in domain and isinstance(ctrl_val, (int, float)) and not isinstance(ctrl_val, bool) and math.isfinite(ctrl_val):
            lo, hi = domain[ctrl_name]
            if not (lo <= ctrl_val <= hi):
                if math.isclose(ctrl_val, lo, rel_tol=1e-5, abs_tol=1e-12) or math.isclose(ctrl_val, hi, rel_tol=1e-5, abs_tol=1e-12):
                    domain[ctrl_name] = [min(lo, float(ctrl_val)), max(hi, float(ctrl_val))]
                    audit.setdefault('rounded_control_enclosures', {})[ctrl_name] = domain[ctrl_name]
    pending = {name: node for name, node in definitions.items() if name not in domain}
    resolved = set(domain)
    while pending:
        progressed = False
        for name, node in list(pending.items()):
            if not m._variables(node) <= resolved:
                continue
            try:
                enclosure = m._iv(node, domain)
            except (ValueError, KeyError, ZeroDivisionError, OverflowError) as error:
                raise ValueError(f'cannot derive a finite domain for {name}: {error}') from error
            if not math.isfinite(enclosure.lo) or not math.isfinite(enclosure.hi):
                raise ValueError(f'cannot derive a finite domain for {name}: unbounded interval')
            domain[name] = [enclosure.lo, enclosure.hi]
            audit.setdefault('derived_domains', {})[name] = domain[name]
            pending.pop(name)
            resolved.add(name)
            progressed = True
        if not progressed:
            missing = set(pending) - set(domain)
            if missing:
                raise ValueError(f'missing finite domains for cyclic or unresolved symbols: {sorted(missing)}')
            break
    if audit:
        history = pack.setdefault('execution', {}).setdefault('normalization_source', {})
        history['derivation'] = {**history.get('derivation', {}), **audit}
        history['derivation'].setdefault('provided_domains', copy.deepcopy(raw_domain))
    return pack

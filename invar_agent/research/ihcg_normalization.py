"""Lossless encoding normalization, never scientific repair or unit inference."""

import ast
import copy
import re


ARCHETYPE_ALIASES = {"ODE": "ODE_PDE", "PDE": "ODE_PDE", "ODE/PDE": "ODE_PDE", "MC": "MONTE_CARLO"}
EQUATION_KIND_ALIASES = {"ODE": "ODE_RHS", "ALGEBRAIC_EXPRESSION": "ALGEBRAIC"}
_PRESENTATION = {"latex", "display_latex"}
_ASSIGNMENT = re.compile(r"^\s*(d([^\W\d]\w*)/dt|[^\W\d]\w*)\s*=(?!=)\s*(.+)$", re.S)
_MATH_PREFIX = re.compile(r"\bmath\.(log|exp|sqrt|sin|cos|fabs)\s*\(")
_LN_CALL = re.compile(r"\bln\s*\(")


from .expr_frontend import normalize_expression as _fe_normalize


def _normalize_formal_expr(
    expr: str,
    *,
    declared_symbols=(),
    domain=None,
    si_dims=None,
    quantifier_mode=None,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Lossless mathematical syntax normalization via the deterministic token-level front end.

    Returns ``(canonical_or_original, transforms, binder_assumptions)``.
    """
    if not isinstance(expr, str):
        return expr, (), ()
    res = _fe_normalize(
        expr,
        declared_symbols=declared_symbols,
        domain=domain,
        si_dims=si_dims,
        quantifier_mode=quantifier_mode,
    )
    if res.ok and res.canonical is not None:
        return res.canonical, res.transforms, res.binder_assumptions
    return expr, (), ()


def normalize_ihcg_evidence(raw):
    if not isinstance(raw, dict):
        raise ValueError("ihcg_evidence: expected an object")
    pack = copy.deepcopy(raw)
    source = {}

    def remember(path, value): source[path] = copy.deepcopy(value)

    def presentation(obj, path):
        for key in sorted(_PRESENTATION & obj.keys()):
            remember(path+'.'+key, obj.pop(key))

    if isinstance(pack.get('archetype'), str) and pack['archetype'] in ARCHETYPE_ALIASES:
        remember('archetype', pack['archetype'])
        pack['archetype'] = ARCHETYPE_ALIASES[pack['archetype']]
    evidence = pack.get('evidence')
    if not isinstance(evidence, dict):
        raise ValueError('ihcg_evidence.evidence: expected an object')
    presentation(pack, 'ihcg_evidence')
    presentation(evidence, 'evidence')
    slots = evidence.get('slots', [])
    if isinstance(slots, list):
        for i, slot in enumerate(slots):
            if isinstance(slot, dict): presentation(slot, f'evidence.slots[{i}]')
    dimensions = {s.get('symbol'): s.get('si_dim') for s in slots if isinstance(s, dict)} if isinstance(slots, list) else {}
    roles = {s.get('symbol'): s.get('role') for s in slots if isinstance(s, dict)} if isinstance(slots, list) else {}
    declared_symbols = set(dimensions.keys())
    raw_domain = pack.get('domain') if isinstance(pack.get('domain'), dict) else {}
    q_mode = evidence.get('quantifier_mode')
    frontend_transforms = {}
    extra_binder_assumptions = []

    def _record_fe(path, original, transforms):
        remember(path, original)
        if transforms:
            frontend_transforms[path] = list(transforms)

    obligations = evidence.get('obligations', [])
    if isinstance(obligations, list):
        for i, obligation in enumerate(obligations):
            if isinstance(obligation, dict):
                presentation(obligation, f'evidence.obligations[{i}]')
                if isinstance(obligation.get('expression'), str):
                    norm_expr, transforms, binder_assumps = _normalize_formal_expr(
                        obligation['expression'],
                        declared_symbols=declared_symbols,
                        domain=raw_domain,
                        si_dims=dimensions,
                        quantifier_mode=q_mode,
                    )
                    if norm_expr != obligation['expression'] or transforms:
                        _record_fe(f'evidence.obligations[{i}].expression', obligation['expression'], transforms)
                        obligation['expression'] = norm_expr
                    extra_binder_assumptions.extend(binder_assumps)
    environment = evidence.get('environmental_assumptions', [])
    if isinstance(environment, list):
        for i, expr in enumerate(environment):
            if isinstance(expr, str):
                norm_expr, transforms, binder_assumps = _normalize_formal_expr(
                    expr,
                    declared_symbols=declared_symbols,
                    domain=raw_domain,
                    si_dims=dimensions,
                    quantifier_mode=q_mode,
                )
                if norm_expr != expr or transforms:
                    _record_fe(f'evidence.environmental_assumptions[{i}]', expr, transforms)
                    environment[i] = norm_expr
                extra_binder_assumptions.extend(binder_assumps)
        for ba in extra_binder_assumptions:
            if ba not in environment:
                environment.append(ba)
    elif extra_binder_assumptions:
        evidence['environmental_assumptions'] = list(dict.fromkeys(extra_binder_assumptions))
    domain = pack.get('domain')
    if isinstance(domain, dict):
        for name, bounds in list(domain.items()):
            path = f'ihcg_evidence.domain[{name!r}]'
            if isinstance(bounds, dict):
                original = copy.deepcopy(bounds)
                presentation(bounds, path)
                keys = set(bounds)
                pair = next((p for p in [('lower', 'upper'), ('min', 'max')] if keys-set(p) <= {'dimensionless'} and set(p) <= keys), None)
                if pair is None:
                    raise ValueError(path+': ambiguous bounds object; expected exactly lower/upper or min/max')
                if 'dimensionless' in bounds:
                    dims = dimensions.get(name)
                    if not isinstance(bounds['dimensionless'], bool) or not isinstance(dims, (list, tuple)) or len(dims) != 7 or bounds['dimensionless'] != all(v == 0 for v in dims):
                        raise ValueError(path+': dimensionless flag conflicts with, or lacks, explicit slot SI dimensions')
                remember(path, original)
                domain[name] = [bounds[pair[0]], bounds[pair[1]]]
    equations = evidence.get('equations')
    if isinstance(equations, dict):
        normalized = {}
        for key, raw_equation in equations.items():
            path = f'ihcg_evidence.evidence.equations[{key!r}]'
            equation = {'expression': raw_equation} if isinstance(raw_equation, str) else copy.deepcopy(raw_equation)
            if not isinstance(equation, dict):
                raise ValueError(path+': expected an equation string or object')
            presentation(equation, path)
            kind = equation.get('kind')
            if isinstance(kind, str) and kind in EQUATION_KIND_ALIASES: equation['kind'] = EQUATION_KIND_ALIASES[kind]
            expression = equation.get('expression')
            match = _ASSIGNMENT.fullmatch(expression) if isinstance(expression, str) else None
            target = key
            dims = dimensions.get(key)
            if match:
                lhs, state, rhs = match.groups()
                if state:
                    if roles.get(state) != 'DYNAMIC_STATE': raise ValueError(path+': derivative must name a declared DYNAMIC_STATE')
                    if equation.get('kind', 'ODE_RHS') != 'ODE_RHS' or equation.get('state_symbol', state) != state:
                        raise ValueError(path+': derivative conflicts with explicit kind/state_symbol')
                    dims = dimensions.get(state)
                    if isinstance(dims, (list, tuple)) and len(dims) == 7 and all(isinstance(v,int) and not isinstance(v,bool) for v in dims):
                        dims = list(dims); dims[2] -= 1
                    equation.update(kind='ODE_RHS', state_symbol=state)
                else:
                    if lhs not in dimensions or (key in dimensions and key != lhs) or equation.get('kind', 'ALGEBRAIC') != 'ALGEBRAIC':
                        raise ValueError(path+': assignment has an ambiguous or conflicting output symbol')
                    target, dims = lhs, dimensions[lhs]
                norm_rhs, transforms, _ = _normalize_formal_expr(
                    rhs,
                    declared_symbols=declared_symbols,
                    domain=raw_domain,
                    si_dims=dimensions,
                    quantifier_mode=q_mode,
                )
                if transforms:
                    frontend_transforms[path + '.expression'] = list(transforms)
                equation['expression'] = norm_rhs
            elif isinstance(expression, str):
                norm_expr, transforms, _ = _normalize_formal_expr(
                    expression,
                    declared_symbols=declared_symbols,
                    domain=raw_domain,
                    si_dims=dimensions,
                    quantifier_mode=q_mode,
                )
                if norm_expr != expression or transforms:
                    _record_fe(path + '.expression', expression, transforms)
                    expression = norm_expr
                    equation['expression'] = expression
                try: boolean = isinstance(ast.parse(expression, mode='eval').body, (ast.Compare, ast.BoolOp))
                except SyntaxError: boolean = False
                if boolean: dims = [0]*7
            if 'si_dim' not in equation:
                if equation.get('kind') == 'ODE_RHS':
                    state = equation.get('state_symbol')
                    dims = dimensions.get(state)
                    if roles.get(state) != 'DYNAMIC_STATE' or not isinstance(dims, (list, tuple)) or len(dims) != 7 or not all(isinstance(v,int) and not isinstance(v,bool) for v in dims):
                        raise ValueError(path+': missing si_dim and no declared derivative state dimensions')
                    dims = list(dims); dims[2] -= 1
                elif equation.get('kind', 'ALGEBRAIC') != 'ALGEBRAIC':
                    dims = None
                if not isinstance(dims, (list, tuple)) or len(dims) != 7:
                    raise ValueError(path+': missing si_dim; no uniquely declared output/state SI dimensions')
                equation['si_dim'] = list(dims)
            if target in normalized: raise ValueError(path+': duplicate normalized output '+target)
            normalized[target] = equation
            if target != key or equation != raw_equation: remember(path, raw_equation)
        evidence['equations'] = normalized
    if source or frontend_transforms:
        execution = pack.setdefault('execution', {})
        if not isinstance(execution, dict): raise ValueError('ihcg_evidence.execution: expected an object')
        history = execution.setdefault('normalization_source', {})
        if not isinstance(history, dict): raise ValueError('ihcg_evidence.execution.normalization_source: expected an object')
        for path, value in source.items():
            if path in history and history[path] != value: raise ValueError('normalization source collision: '+path)
            history[path] = value
        if frontend_transforms:
            existing_tf = history.setdefault('expr_frontend_transforms', {})
            if isinstance(existing_tf, dict):
                existing_tf.update(frontend_transforms)
    return pack

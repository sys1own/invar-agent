"""Deterministic recovery and independent validation of declared ODE models."""
from __future__ import annotations

import ast
import re

from ..runtime.research_reference import ReferenceError, run_reference, integrate_rhs, evaluate_predicate


NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
DERIVATIVE = re.compile(r'\bd([A-Za-z_]\w*)/dt\s*=\s*([^,;\n]+)')


def text_evolutions(text):
    output = {}
    for match in DERIVATIVE.finditer(text or ''):
        rhs = re.split(r'\s+(?:with|where|and|for|preserves|enforcing|subject|on)\b', match[2], maxsplit=1)[0].strip().rstrip('.')
        rhs = re.sub(r'(?<=[\w)])\s+(?=[A-Za-z_(])', '*', rhs)
        output[match[1]] = rhs
    return output


def _slot(node):
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
        if isinstance(node.value, ast.Name) or (isinstance(node.value, ast.Attribute) and node.value.attr == 'data'):
            return node.slice.value
    return None


def _key(name):
    return re.sub('[^a-z0-9]', '', name.lower())


def structured_evidence(blueprint, metadata):
    primary, secondary = blueprint.get('structured_evidence') or {}, metadata.get('structured_evidence') or {}
    return {**secondary, **primary, 'equations':primary.get('equations') or secondary.get('equations') or [],
            'parameters':{**(secondary.get('parameters') or {}), **(primary.get('parameters') or {})}}


def _model_texts(blueprint, metadata):
    containers = (('blueprint.json',blueprint),('metadata.json',metadata))
    fields = ('research_prompt',) if any(container.get('research_prompt') for _,container in containers) else (
        'falsifiable_hypothesis','description','question','governing_equations')
    return [(filename,field,text) for filename,container in containers for field in fields
            for value in [container.get(field) or ''] for text in (value if isinstance(value,list) else [value])]


def symbol_slots(names, blueprint, code):
    candidates = dict(blueprint.get('variable_map') or {})
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign) and _slot(node.value) is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    candidates.setdefault(target.id, _slot(node.value))
    slots = {}
    for name in names:
        ranked = []
        for candidate, index in candidates.items():
            if type(index) is not int or index < 0:
                continue
            score = 100 if candidate == name else 90 if _key(candidate) == _key(name) else 70 if name in candidate.split('_') else 50 if len(name)>1 and _key(candidate).startswith(_key(name)) else 0
            if score:
                ranked.append((-score, index))
        if ranked:
            slots[name] = min(ranked)[1]
    return slots


def parameter_bindings(blueprint, metadata):
    from .extractor import normalize_to_si
    from ..builder.ingestion.equation_lowerer import unit_dimension
    evidence = structured_evidence(blueprint, metadata)
    parameters = evidence.get('parameters') or {}
    output = {}
    for key, item in parameters.items():
        if item.get('si_value') is not None:
            filename = 'blueprint.json' if key in ((blueprint.get('structured_evidence') or {}).get('parameters') or {}) else 'metadata.json'
            output[item.get('symbol') or key] = dict(value=item['si_value'], unit=item.get('si_unit') or '1', source='structured', key=key, file=filename)
    for key, value in (metadata.get('extracted_constants') or blueprint.get('extracted_constants') or {}).items():
        symbol = (parameters.get(key) or {}).get('symbol') or next((name for name in output if _key(name)==_key(key)), key)
        if symbol not in output:
            output[symbol] = dict(value=value, unit='1', source='stored', key=key)
        elif output[symbol]['value'] != value:
            raise ReferenceError('VAULT_CONSTANT_MISMATCH', f'{symbol}: stored {value}, structured {output[symbol]["value"]}')
    texts = _model_texts(blueprint,metadata)
    text = '\n'.join(str(value) for _,_,value in texts)
    unit_token = r'(?:1|[A-Za-zµμ]+)(?:\^[+-]?\d+)?(?:[/*](?:\([^)]*\)|[A-Za-zµμ]+(?:\^[+-]?\d+)?))*'
    for match in re.finditer(rf'(?<![\w/])([A-Za-z_]\w*)\s*=\s*({NUMBER})(?:\s+({unit_token}))?', text):
        name, value, unit = match[1], float(match[2]), match[3] or '1'
        try:
            unit_dimension(unit)
        except ValueError:
            unit = '1'
        value, canonical = normalize_to_si(value, unit)
        symbol = next((old for old in output if _key(old)==_key(name)), name)
        old = output.get(symbol)
        if old and abs(old['value']-value) > max(abs(value)*1e-12, 1e-30):
            raise ReferenceError('VAULT_CONSTANT_MISMATCH', f'{symbol}: stored {old["value"]}, text {value}')
        if old is None or old['unit'] in ('', '1') and unit != '1':
            filename, field = next((filename,field) for filename,field,value in texts if match[0] in str(value))
            output[symbol] = dict(value=value, unit=canonical, source='text', key=field, file=filename)
    return output


class _SymbolicKernel:
    def __init__(self, code, slots):
        import sympy as sp
        self.sp = sp
        self.functions = {node.name: node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
        self.slots = {index: sp.Symbol(name) for name, index in slots.items()}
        self.globals = {}
        self.block([node for node in ast.parse(code).body if isinstance(node, (ast.Assign, ast.AnnAssign))], self.globals)

    def expression(self, node, environment, depth=0):
        sp = self.sp
        if depth > 24:
            raise ValueError('Symbolic call depth exceeded')
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and isinstance(environment.get(node.value.id), list) and isinstance(node.slice, ast.Constant):
            return environment[node.value.id][node.slice.value]
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and _slot(node) is None:
            return self.expression(node.value, environment, depth)[node.slice.value]
        if _slot(node) is not None:
            return environment.get(('slot', _slot(node)), self.slots.get(_slot(node), sp.Symbol(f'_slot_{_slot(node)}')))
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return sp.true if node.value else sp.false
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name):
            return environment.get(node.id, sp.Symbol(node.id))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return self.expression(node.operand, environment, depth) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.Compare):
            left = self.expression(node.left,environment,depth)
            comparisons = []
            relations = {ast.Gt:sp.Gt,ast.GtE:sp.Ge,ast.Lt:sp.Lt,ast.LtE:sp.Le,ast.Eq:sp.Eq,ast.NotEq:sp.Ne}
            for operation,right_node in zip(node.ops,node.comparators):
                right = self.expression(right_node,environment,depth)
                comparisons.append(relations[type(operation)](left,right))
                left = right
            return sp.And(*comparisons)
        if isinstance(node,ast.BoolOp):
            return (sp.And if isinstance(node.op,ast.And) else sp.Or)(*(self.expression(value,environment,depth) for value in node.values))
        if isinstance(node,ast.IfExp):
            return sp.Piecewise((self.expression(node.body,environment,depth),self.expression(node.test,environment,depth)),
                                (self.expression(node.orelse,environment,depth),True))
        if isinstance(node, ast.BinOp):
            left, right = self.expression(node.left, environment, depth), self.expression(node.right, environment, depth)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow): return left ** right
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'math' else ''
            arguments = [self.expression(arg, environment, depth) for arg in node.args]
            if name in {'_invar_sdiv', 'safe_div', 'div_eps'} and len(arguments)==2:
                return arguments[0]/arguments[1]
            if name in {'sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'min', 'max'}:
                return {'abs': sp.Abs, 'min': sp.Min, 'max': sp.Max}.get(name, getattr(sp, name, None))(*arguments)
            function = environment.get(('function',name),self.functions.get(name))
            if function is not None:
                local = dict(environment)
                local.update({arg.arg: value for arg, value in zip(function.args.args, arguments)})
                result = self.block(function.body, local, depth+1)
                environment.update({key:value for key,value in local.items() if isinstance(key,tuple) and key[0]=='slot'})
                return result
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.expression(item, environment, depth) for item in node.elts]
        raise ValueError('Unsupported symbolic expression: '+ast.unparse(node))

    def block(self, body, environment, depth=0):
        if depth > 24:
            raise ValueError('Symbolic call depth exceeded')
        for node in body:
            if isinstance(node, ast.FunctionDef):
                environment[('function',node.name)] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                try:
                    value = self.expression(node.value, environment, depth)
                except ValueError:
                    continue
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    if isinstance(target, ast.Name): environment[target.id] = value
                    elif _slot(target) is not None: environment[('slot', _slot(target))] = value
            elif isinstance(node, ast.AugAssign):
                value = self.expression(ast.BinOp(left=node.target,op=node.op,right=node.value),environment,depth)
                if isinstance(node.target,ast.Name): environment[node.target.id] = value
                elif _slot(node.target) is not None: environment[('slot',_slot(node.target))] = value
            elif isinstance(node, ast.Return) and node.value is not None:
                return self.expression(node.value, environment, depth)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and (call.func.id in self.functions or ('function',call.func.id) in environment):
                    self.expression(call, environment, depth+1)
        return None

    def updates(self, states):
        driver = self.functions.get('simulation_step')
        if driver is None:
            return {}
        environment = {**self.globals, 'dt':self.sp.Symbol('_dt')}
        if len(driver.args.args)>1:
            environment[driver.args.args[1].arg] = environment['dt']
        self.block(driver.body,environment)
        return {name:environment[('slot',index)] for index,symbol in self.slots.items()
                if (name:=str(symbol)) in states and ('slot',index) in environment}

    def derivatives(self, states):
        sp = self.sp
        if 'simulation_step' in self.functions:
            updates = self.updates(states)
            return {name:sp.diff(value,sp.Symbol('_dt')).subs(sp.Symbol('_dt'),0)
                    for name,value in updates.items()}, 'simulation_step'
        ordered = [symbol for index,symbol in sorted(self.slots.items()) if str(symbol) in states]
        parameters = [symbol for index,symbol in sorted(self.slots.items()) if str(symbol) not in states]
        for function in self.functions.values():
            if any(isinstance(node, ast.Return) and isinstance(node.value, (ast.List, ast.Tuple)) for node in function.body):
                environment = {}
                arguments = function.args.args
                for arg in arguments:
                    if arg.arg in {'t', 'time'}:
                        environment[arg.arg] = sp.Symbol('t')
                    elif arg.arg in {'params', 'parameters', 'p'}:
                        environment[arg.arg] = parameters
                    else:
                        environment[arg.arg] = ordered
                try:
                    result = self.block(function.body, environment)
                    if isinstance(result, list) and len(result)==len(states):
                        return dict(zip(map(str,ordered), result)), function.name
                except ValueError:
                    pass
        return {}, None


def recover_evolution(blueprint, metadata, code):
    from ..builder.ingestion.equation_lowerer import equation_parts
    evidence = structured_evidence(blueprint, metadata)
    laws, origins = {}, {}
    for item in evidence.get('equations') or []:
        text = f"{item.get('lhs', '')} = {item.get('rhs_latex', '')}"
        if item.get('role') == 'EVOLUTION_LAW' and not re.search(r'\bd\w+/dt|\\dot|\\frac\{d', text):
            text = f"d{item['lhs']}/dt = {item['rhs_latex']}"
        if re.search(r'\bd\w+/dt|\\dot|\\frac\{d', text):
            left, _, right = equation_parts(text)
            laws[str(left)] = right
            filename = 'blueprint.json' if (blueprint.get('structured_evidence') or {}).get('equations') else 'metadata.json'
            origins[str(left)] = ('structured', item['rhs_latex'], filename)
    text_laws = {}
    for filename,key,text in _model_texts(blueprint,metadata):
        for state,rhs in text_evolutions(str(text)).items():
            text_laws[state] = equation_parts(f'd{state}/dt = {rhs}')[2]
            if state not in laws:
                laws[state] = text_laws[state]
                origins[state] = ('text',key,filename)
    states = list(blueprint.get('state_order') or blueprint.get('state_symbols') or laws)
    if not states:
        tree = ast.parse(code)
        candidates = [name for name in (blueprint.get('variable_map') or {})]
        slots = symbol_slots(candidates, blueprint, code)
        assigned = {_slot(target) for node in ast.walk(tree) if isinstance(node, ast.Assign) for target in node.targets if _slot(target) is not None}
        states = [name for name in candidates if slots.get(name) in assigned]
        if not states:
            states = candidates
    parameters = parameter_bindings(blueprint, metadata)
    contracts = blueprint.get('contracts') or metadata.get('contracts_summary') or []
    initial = (contracts[0] if contracts else {}).get('initial_state') or []
    slots = symbol_slots(states + list(parameters), blueprint, code)
    subscripts = [node for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Subscript) and _slot(node) is not None]
    readonly = {_slot(node) for node in subscripts if isinstance(node.ctx,ast.Load)} - {_slot(node) for node in subscripts if isinstance(node.ctx,ast.Store)}
    for name,item in parameters.items():
        if name not in slots:
            candidates = [index for index in readonly-set(slots.values()) if index<len(initial) and initial[index]==item['value']]
            if len(candidates)==1:
                slots[name] = candidates[0]
    missing_slots = [state for state in states if state not in slots]
    if missing_slots:
        raise ReferenceError('VAULT_MISSING_EVOLUTION_LAW', 'Cannot resolve state slots: '+', '.join(missing_slots), {'missing_states': missing_slots})
    symbolic, function = _SymbolicKernel(code, slots).derivatives(states)
    import sympy as sp
    unresolved = set().union(*(expression.free_symbols for expression in symbolic.values())) - {sp.Symbol(name) for name in [*states,*parameters,'t']}
    if unresolved:
        missing = sorted(map(str,unresolved))
        raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED','Cannot resolve executed RHS symbols: '+', '.join(missing),{'unresolved_symbols':missing})
    for state, expression in symbolic.items():
        if state not in laws:
            laws[state], origins[state] = expression, ('kernel', function)
    missing = [state for state in states if state not in laws]
    if not states or missing:
        missing = missing or list(blueprint.get('variable_map') or {'<unknown>': 0})
        raise ReferenceError('VAULT_MISSING_EVOLUTION_LAW', 'Missing evolution laws for: '+', '.join(missing), {'missing_states': missing})
    for name, item in parameters.items():
        if name in slots and slots[name] < len(initial) and initial[slots[name]] != item['value']:
            raise ReferenceError('VAULT_CONSTANT_MISMATCH', f'{name}: initial slot {initial[slots[name]]} != {item["value"]}')
    environment = {name: item['value'] for name, item in parameters.items()}
    environment.update({name: initial[slots[name]] if slots[name]<len(initial) else 1. for name in states})
    comparisons = []
    for factor in (1., .75, 1.125):
        point = {**environment, **{name: value*factor if value else .001*factor for name, value in environment.items() if name in states}, 't': 0.}
        for state in text_laws.keys() & symbolic.keys():
            actual, expected = float(symbolic[state].subs(point)), float(text_laws[state].subs(point))
            if abs(actual-expected) > 1e-10:
                raise ReferenceError('VAULT_REFERENCE_MISMATCH', f'{state}: stated RHS {expected}, symbolic RHS {actual}', dict(state=state, step=0, actual=actual, expected=expected, absolute_error=abs(actual-expected)))
            comparisons.append(dict(state=state, actual=actual, expected=expected, point=point))
    return dict(states=states, laws=laws, origins=origins, parameters=parameters, slots=slots,
                symbolic=symbolic, symbolic_function=function, comparisons=comparisons, contracts=contracts)


def check_evolution_reference(blueprint, metadata, code, *, recovered=None):
    import math
    import sympy as sp
    recovered = recovered or recover_evolution(blueprint, metadata, code)
    if not recovered['contracts']:
        raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', 'No reference initial state/time contract')
    names = recovered['states']
    parameters = recovered['parameters']
    arguments = [sp.Symbol(name) for name in [*names, *parameters, 't']]
    expressions = [recovered['laws'][name] for name in names]
    numerical_rhs = sp.lambdify(arguments, expressions, modules='math', dummify=True, cse=True)
    rhs_scales = sp.lambdify(arguments,[sum(sp.Abs(term) for term in sp.Add.make_args(sp.expand(expression)))
        for expression in expressions],modules='math',dummify=True,cse=True)
    symbolic = recovered['symbolic']
    if set(names)-set(symbolic):
        raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', 'Cannot independently recover the executed RHS for '+', '.join(set(names)-set(symbolic)))
    actual_rhs = sp.lambdify(arguments,[symbolic[name] for name in names],modules='math',dummify=True,cse=True)
    updates = _SymbolicKernel(code,recovered['slots']).updates(names)
    numerical_update = sp.lambdify([*arguments,sp.Symbol('_dt')],[updates[name] for name in names],modules='math',dummify=True,cse=True)
    scores = {scheme:0. for scheme in ('euler','midpoint','heun','rk4')}

    def check_rhs(values, parameters, t, step, contract_id):
        try:
            expected = list(map(float,numerical_rhs(*values,*parameters,t)))
            actual = list(map(float,actual_rhs(*values,*parameters,t)))
            scales = list(map(float,rhs_scales(*values,*parameters,t)))
        except (ValueError,OverflowError,ZeroDivisionError,TypeError) as error:
            raise ReferenceError('VAULT_REFERENCE_UNDEFINED',f'Cannot evaluate RHS: {error}',
                dict(states=names,step=step,contract_id=contract_id)) from error
        for name,a,b,scale in zip(names,actual,expected,scales):
            tolerance = max(1e-12*scale,8*math.ulp(scale))
            if not math.isfinite(a) or not math.isfinite(b) or not math.isfinite(scale) or abs(a-b)>tolerance:
                details = dict(state=name,step=step,contract_id=contract_id,
                    actual=a if math.isfinite(a) else None,expected=b if math.isfinite(b) else None,
                    absolute_error=abs(a-b) if math.isfinite(a) and math.isfinite(b) else None,
                    tolerance=tolerance if math.isfinite(tolerance) else None,comparison='rhs')
                raise ReferenceError('VAULT_REFERENCE_MISMATCH',f'{name}: executed RHS {a}, stated RHS {b}',details)
        return expected

    first = recovered['contracts'][0]
    probes = []
    for factor in (1.,.75,1.125):
        values = [first['initial_state'][recovered['slots'][name]]*factor or .001*factor for name in names]
        params = [first['initial_state'][recovered['slots'][name]] if name in recovered['slots'] else item['value'] for name,item in parameters.items()]
        for multiple in (1.,.5,2.):
            dt = first['dt']*multiple
            check_rhs(values,params,0.,0,first['contract_id'])
            for trial in (params,[p*1.125 if p else .001 for p in params]):
                actual = numerical_update(*values,*trial,0.,dt)
                for scheme in scores:
                    try:
                        expected = integrate_rhs(lambda y,p,t:actual_rhs(*y,*p,t),values,trial,dt,0.,scheme)
                        error = max(abs(float(a)-float(b))/(1e-12+1e-11*abs(float(b))) for a,b in zip(actual,expected))
                    except (ValueError,OverflowError,ZeroDivisionError,TypeError):
                        error = math.inf
                    scores[scheme] = max(scores[scheme],error if math.isfinite(error) else math.inf)
            probes.append((values,params,dt))
    scheme = min(scores,key=scores.get)
    if scores[scheme]>1.:
        raise ReferenceError('VAULT_INTEGRATOR_UNSUPPORTED','Cannot identify the executed integration scheme; supported: Euler, midpoint, Heun, RK4',
            {'scheme_residuals':{name:score if math.isfinite(score) else None for name,score in scores.items()}})
    recovered['integration_scheme'] = scheme
    for original in recovered['contracts']:
        contract = {**original, 'invariants': []}
        if len(names)*contract['steps'] > 10000000:
            raise ReferenceError('VAULT_REFERENCE_LIMIT', 'Independent ODE reference exceeds the validation work budget')
        result = run_reference(code, contract, sample_steps=list(range(1,contract['steps']+1)),
            sample_slots=[recovered['slots'][name] for name in names])
        state = [contract['initial_state'][recovered['slots'][name]] for name in names]
        parameter_values = [contract['initial_state'][recovered['slots'][name]]
            if name in recovered['slots'] and recovered['slots'][name]<len(contract['initial_state'])
            else item['value'] for name,item in parameters.items()]

        def rhs(values, time):
            return [float(value) for value in numerical_rhs(*values,*parameter_values,time)]

        dt = contract['dt']
        for step in range(1,contract['steps']+1):
            t = (step-1)*dt
            try:
                check_rhs(state,parameter_values,t,step,contract['contract_id'])
                expected_state = integrate_rhs(lambda y,p,t:rhs(y,t),state,parameter_values,dt,t,scheme)
            except ReferenceError:
                raise
            except (ValueError,OverflowError,ZeroDivisionError,TypeError) as error:
                raise ReferenceError('VAULT_REFERENCE_UNDEFINED',
                    f'Declared evolution is undefined at step {step}: {error}',
                    dict(states=names,step=step,contract_id=contract['contract_id'])) from error
            for index,(name,expected) in enumerate(zip(names,expected_state)):
                actual = result['samples'][str(step)][index]
                if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual-expected)>1e-10:
                    details = dict(state=name,step=step,contract_id=contract['contract_id'],
                        actual=actual if math.isfinite(actual) else None,
                        expected=expected if math.isfinite(expected) else None,
                        absolute_error=abs(actual-expected) if math.isfinite(actual) and math.isfinite(expected) else None,
                        tolerance=1e-10)
                    raise ReferenceError('VAULT_REFERENCE_MISMATCH',
                        f'{name} diverges at step {step}: pinned={actual}, ODE ({scheme})={expected}',details)
            state = result['samples'][str(step)]
    for values,params,dt in probes:
        initial = list(first['initial_state'])
        for name,value in zip(names,values): initial[recovered['slots'][name]] = value
        admissible = True
        for invariant in first.get('invariants',[]):
            tree = ast.parse(invariant['assertion'],mode='eval')
            if not isinstance(tree.body,ast.Compare) or any(isinstance(node,(ast.Call,ast.Attribute)) for node in ast.walk(tree)):
                continue
            def context(state):
                return {'state':state,'data':state,'trace_min':state,'trace_max':state,
                        'initial':first['initial_state'],'initial_state':first['initial_state'],
                        **{name:state[index] for name,index in recovered['slots'].items() if index<len(state)}}
            try:
                if evaluate_predicate(tree,context(first['initial_state'])) and not evaluate_predicate(tree,context(initial)):
                    admissible = False
                    break
            except (KeyError,IndexError,TypeError,ValueError):
                continue
        if not admissible:
            continue
        sample = run_reference(code,{**first,'initial_state':initial,'steps':1,'dt':dt,'invariants':[]},
            sample_steps=[1],sample_slots=[recovered['slots'][name] for name in names])['samples']['1']
        expected = integrate_rhs(lambda y,p,t:numerical_rhs(*y,*p,t),values,params,dt,0.,scheme)
        for name,a,b in zip(names,sample,expected):
            if not math.isfinite(a) or abs(a-b)>1e-10:
                details = dict(state=name,step=1,actual=a,expected=float(b),absolute_error=abs(a-b),tolerance=1e-10,comparison='sampled_step')
                raise ReferenceError('VAULT_REFERENCE_MISMATCH',f'{name}: sampled {scheme} step disagrees with stated RHS',details)
    return recovered

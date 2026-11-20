"""Bounded, standard-library-only execution of persisted research references."""
from __future__ import annotations

import ast
import json
import math
import os
import operator
import re
import subprocess
import sys
import time
from types import SimpleNamespace


class ReferenceError(ValueError):
    status_code = 422

    def __init__(self, code, message, details=None):
        self.code, self.details = code, details
        super().__init__(message)


class _Vector(list):
    def _compare(self, other, operation):
        values = other if isinstance(other, (list, tuple)) else [other]*len(self)
        return _Vector(operation(a,b) for a,b in zip(self, values))

    def __ge__(self, other): return self._compare(other, lambda a,b:a>=b)
    def __gt__(self, other): return self._compare(other, lambda a,b:a>b)
    def __le__(self, other): return self._compare(other, lambda a,b:a<=b)
    def __lt__(self, other): return self._compare(other, lambda a,b:a<b)


def _finite(value):
    return _Vector(math.isfinite(item) for item in value) if isinstance(value, (list, tuple)) else math.isfinite(value)


def validate_predicate(expression):
    tree = ast.parse(expression, mode='eval')
    allowed = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
               ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.Pow, ast.Mod, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
               ast.Gt, ast.GtE, ast.Name, ast.Load, ast.Store, ast.Constant,
               ast.Subscript, ast.Slice, ast.Call, ast.Attribute, ast.List,
               ast.Tuple, ast.GeneratorExp, ast.ListComp, ast.comprehension)
    functions = {'abs', 'min', 'max', 'sum', 'len', 'all', 'any', 'isfinite', 'range'}
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ReferenceError('VAULT_RUNTIME_PREDICATE_UNSUPPORTED', ast.dump(node))
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ReferenceError('VAULT_RUNTIME_PREDICATE_UNSUPPORTED', node.id)
        if isinstance(node, ast.Attribute):
            buffer = isinstance(node.value,ast.Name) and node.value.id=='buf' and node.attr in {'data','size'}
            if not buffer and (node.attr not in functions or not isinstance(node.value, ast.Name) or node.value.id not in {'math', 'np', 'numpy'}):
                raise ReferenceError('VAULT_RUNTIME_PREDICATE_UNSUPPORTED', ast.unparse(node))
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            if name not in functions or node.keywords:
                raise ReferenceError('VAULT_RUNTIME_PREDICATE_UNSUPPORTED', name)
    return tree


def evaluate_predicate(node, context):
    if isinstance(node, ast.Expression): return evaluate_predicate(node.body, context)
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Name): return context[node.id]
    if isinstance(node, ast.Attribute): return getattr(context[node.value.id], node.attr)
    if isinstance(node, ast.Subscript): return evaluate_predicate(node.value, context)[evaluate_predicate(node.slice, context)]
    if isinstance(node, ast.Slice):
        return slice(*(evaluate_predicate(value, context) if value is not None else None for value in (node.lower,node.upper,node.step)))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [evaluate_predicate(item, context) for item in node.elts]
    if isinstance(node, ast.Call):
        return evaluate_predicate(node.func, context)(*(evaluate_predicate(arg, context) for arg in node.args))
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            result = evaluate_predicate(value, context)
            if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result:
                return result
        return result
    if isinstance(node, ast.UnaryOp):
        return {ast.Not:operator.not_,ast.USub:operator.neg,ast.UAdd:operator.pos}[type(node.op)](evaluate_predicate(node.operand, context))
    if isinstance(node, ast.BinOp):
        function = {ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow,ast.Mod:operator.mod}[type(node.op)]
        return function(evaluate_predicate(node.left, context), evaluate_predicate(node.right, context))
    if isinstance(node, ast.Compare):
        left = evaluate_predicate(node.left, context)
        for op, right_node in zip(node.ops, node.comparators):
            right = evaluate_predicate(right_node, context)
            function = {ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}[type(op)]
            comparison = function(left, right)
            if isinstance(comparison, _Vector) and len(node.ops)==1:
                return comparison
            if not comparison: return False
            left = right
        return True
    if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
        def generate(index, values):
            if index == len(node.generators):
                yield evaluate_predicate(node.elt, values)
                return
            generator = node.generators[index]
            if not isinstance(generator.target, ast.Name) or generator.is_async:
                raise ReferenceError('VAULT_RUNTIME_PREDICATE_UNSUPPORTED', 'Unsupported comprehension')
            for item in evaluate_predicate(generator.iter, values):
                local = {**values, generator.target.id:item}
                if all(evaluate_predicate(condition, local) for condition in generator.ifs):
                    yield from generate(index+1, local)
        return list(generate(0, context))
    raise ReferenceError('VAULT_RUNTIME_PREDICATE_UNSUPPORTED', type(node).__name__)


def _constant_value(node, constants):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.Name) and node.id in constants:
        value = constants[node.id]
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _constant_value(node.operand, constants) * (-1 if isinstance(node.op, ast.USub) else 1)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left, right = _constant_value(node.left, constants), _constant_value(node.right, constants)
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', 'Non-numeric constant arithmetic')
        if isinstance(node.op, ast.Pow) and abs(right) > 1024:
            raise ReferenceError('VAULT_REFERENCE_LIMIT', 'Constant exponent exceeds budget')
        if isinstance(node.op, ast.Add): value = left + right
        elif isinstance(node.op, ast.Sub): value = left - right
        elif isinstance(node.op, ast.Mult): value = left * right
        elif isinstance(node.op, ast.Div): value = left / right
        else: value = left ** right
    else:
        raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', 'Non-static module constant')
    if type(value) not in (int, float) or not math.isfinite(value) or isinstance(value, int) and value.bit_length() > 16384:
        raise ReferenceError('VAULT_REFERENCE_LIMIT', 'Invalid numeric constant')
    return value


def _safe_namespace(code):
    tree = ast.parse(code)
    forbidden = (ast.ClassDef, ast.With, ast.AsyncWith, ast.Global, ast.Nonlocal,
                 ast.ImportFrom, ast.Lambda, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', type(node).__name__)
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', node.attr)
        if isinstance(node, ast.Import) and any(alias.name != 'math' for alias in node.names):
            raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', 'Only math imports are permitted')
    if any(not isinstance(node, (ast.FunctionDef, ast.Import, ast.Assign, ast.AnnAssign)) for node in tree.body):
        raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', 'Unsupported module-level statement')
    constants = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = _constant_value(node.value, constants)
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
            node.value = ast.copy_location(ast.parse(repr(value), mode='eval').body, node.value)
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', 'Decorators are not permitted')
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    ast.literal_eval(default)

    def importer(name, *args, **kwargs):
        if name != 'math':
            raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', name)
        return math

    safe = {name: getattr(__builtins__, name) if not isinstance(__builtins__, dict) else __builtins__[name]
            for name in ('abs', 'min', 'max', 'sum', 'len', 'all', 'any', 'range',
                         'float', 'int', 'bool', 'list', 'tuple', 'dict', 'enumerate',
                         'zip', 'pow', 'ValueError', 'OverflowError', 'ZeroDivisionError')}
    safe['__import__'] = importer
    namespace = {'__builtins__': safe, 'math': math}
    ast.fix_missing_locations(tree)
    exec(compile(tree, '<pinned-research-reference>', 'exec'), namespace)
    return namespace


def _run(payload):
    namespace = _safe_namespace(payload['code'])
    function = namespace.get('simulation_step')
    if not callable(function):
        raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', 'Missing simulation_step(buf, dt)')
    contract = payload['contract']
    initial = list(contract['initial_state'])
    indices = [node.slice.value for node in ast.walk(ast.parse(payload['code']))
               if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
               and type(node.slice.value) is int and node.slice.value >= 0]
    size = max(len(initial), max(indices, default=0)+1)
    if size > 4096 or not 0 <= contract['steps'] <= 100000:
        raise ReferenceError('VAULT_REFERENCE_LIMIT', 'Reference exceeds buffer/step limits')
    initial.extend([0.] * (size-len(initial)))
    for slot, value in payload.get('overrides', {}).items():
        initial[int(slot)] = value
    buffer = SimpleNamespace(data=initial.copy(), size=size)
    low, high = initial.copy(), initial.copy()
    sample_slots = payload.get('sample_slots')
    if sample_slots is not None and any(type(index) is not int or not 0 <= index < size for index in sample_slots):
        raise ReferenceError('VAULT_REFERENCE_LIMIT', 'Invalid reference sample slot')

    def snapshot():
        return buffer.data.copy() if sample_slots is None else [buffer.data[index] for index in sample_slots]

    samples = {'0': snapshot()}
    requested = set(payload.get('sample_steps', [1, 2, contract['steps']]))
    started = time.perf_counter()
    for step in range(1, contract['steps']+1):
        function(buffer, contract['dt'])
        low = [min(a, b) for a, b in zip(low, buffer.data)]
        high = [max(a, b) for a, b in zip(high, buffer.data)]
        if step in requested:
            samples[str(step)] = snapshot()
    finite = all(math.isfinite(value) for value in buffer.data)
    telemetry = {'finite': float(finite), 'steps': contract['steps'], 'dt': contract['dt'],
                 'final_min': min(buffer.data), 'final_max': max(buffer.data),
                 'final_l2': math.sqrt(sum(value*value for value in buffer.data)),
                 'final_mean': sum(buffer.data)/len(buffer.data)}
    telemetry.update(execution_time_ms=(time.perf_counter()-started)*1000,
        initial_l2=math.sqrt(sum(value*value for value in initial)), final_norm=telemetry['final_l2'],
        max_abs_change=max(abs(a-b) for a,b in zip(buffer.data, initial)),
        energy_drift=max(abs(a-b) for a,b in zip(buffer.data, initial)))
    arrays = SimpleNamespace(all=all, any=any, isfinite=_finite, min=min, max=max, sum=sum)
    vector = _Vector(buffer.data)
    context = {'state': vector, 'data': vector, 'buf': SimpleNamespace(data=vector,size=size),
               'initial': _Vector(initial), 'initial_state': _Vector(initial), 'trace_max': _Vector(high), 'trace_min': _Vector(low),
               'np': arrays, 'numpy': arrays,
               'telemetry': telemetry, 'math': math, 'abs': abs, 'min': min, 'max': max,
               'sum': sum, 'len': len, 'all': all, 'any': any, 'isfinite': math.isfinite, 'range': range}
    for name, index in payload.get('variable_map', {}).items():
        if type(index) is int and 0 <= index < size:
            telemetry[f'obs_{name}'] = buffer.data[index]
            if name not in context:
                context[name] = buffer.data[index]
    failures = []
    for invariant in contract.get('invariants', []):
        assertion = invariant.get('assertion', '')
        if not assertion:
            continue
        indices = set(int(index) for index in re.findall(r'(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]', assertion))
        label = invariant.get('variable_name')
        if label and len(indices)==1 and max(indices)<size:
            telemetry.setdefault(f'obs_{label}',buffer.data[next(iter(indices))])
        context['tol'] = invariant.get('tolerance',0.)
        tree = validate_predicate(assertion)
        passed = evaluate_predicate(tree, context)
        if not (all(passed) if isinstance(passed, (list, tuple)) else passed):
            failures.append(assertion)
    return {'samples': samples, 'state': buffer.data, 'initial': initial,
            'phase_hits': namespace.get('_invar_phase_hits', []),
            'trace_max': high, 'trace_min': low, 'telemetry': telemetry, 'failures': failures}


def run_reference(code, contract, *, overrides=None, sample_steps=None, variable_map=None, sample_slots=None):
    payload = dict(code=code, contract=contract, overrides=overrides or {}, sample_steps=sample_steps or [1, 2, contract['steps']])
    payload['variable_map'] = variable_map or {}
    payload['sample_slots'] = sample_slots
    if getattr(sys, '_invar_builder_sandbox', False):
        return _run(payload)
    environment = {key: value for key, value in os.environ.items()
                   if key in {'PATH', 'SYSTEMROOT', 'WINDIR', 'HOME', 'LANG'}}
    process = subprocess.run([sys.executable, '-I', __file__], input=json.dumps(payload),
                             text=True, capture_output=True, timeout=35, env=environment)
    if process.returncode:
        raise ReferenceError('VAULT_REFERENCE_EXECUTION_FAILED', process.stderr[-2000:])
    result = json.loads(process.stdout)
    if 'error' in result:
        raise ReferenceError(result['code'], result['error'], result.get('details'))
    return result


def integrate_rhs(rhs, state, parameters, dt, t=0., scheme='rk4'):
    a = rhs(state, parameters, t)
    if scheme == 'euler':
        return [x+dt*v for x,v in zip(state,a)]
    if scheme == 'midpoint':
        b = rhs([x+dt*v/2 for x,v in zip(state,a)], parameters, t+dt/2)
        return [x+dt*v for x,v in zip(state,b)]
    if scheme == 'heun':
        b = rhs([x+dt*v for x,v in zip(state,a)], parameters, t+dt)
        return [x+dt*(v+w)/2 for x,v,w in zip(state,a,b)]
    if scheme != 'rk4':
        raise ReferenceError('VAULT_INTEGRATOR_UNSUPPORTED', scheme)
    b = rhs([x+dt*v/2 for x,v in zip(state,a)], parameters, t+dt/2)
    c = rhs([x+dt*v/2 for x,v in zip(state,b)], parameters, t+dt/2)
    d = rhs([x+dt*v for x,v in zip(state,c)], parameters, t+dt)
    return [x+dt*(v+2*w+2*z+q)/6 for x,v,w,z,q in zip(state,a,b,c,d)]


def check_model_reference(model, advance, *, rhs=None, parameters=None, initial=None, steps=None, dt=None):
    reference = model['ir'].get('research_reference')
    if not reference:
        return {}
    states = model['states']
    output = {}
    for original in reference['contracts']:
        if steps is not None and (original['steps'] != steps or original['dt'] != dt):
            continue
        contract = {**original, 'initial_state': list(original['initial_state'])}
        supplied = {**model['defaults'], **(parameters or {}), **(initial or {})}
        for name, slot in {**reference['parameter_slots'], **reference['state_slots']}.items():
            if name in supplied:
                contract['initial_state'][slot] = supplied[name]
        result = run_reference(model['ir']['pinned_kernel_code'], contract,variable_map=reference.get('variable_map'))
        if result['failures']:
            raise ReferenceError('VAULT_RUNTIME_GUARD_FAILED', str(result['failures']))
        state = [contract['initial_state'][reference['state_slots'][name]] for name in states]
        for step in range(1, contract['steps']+1):
            values = [supplied.get(name,model['defaults'][name]) for name in model['parameters']]
            scheme = reference.get('integration_scheme','rk4')
            if rhs is None and scheme != 'rk4':
                raise ReferenceError('VAULT_INTEGRATOR_UNSUPPORTED', 'RHS required for pinned '+scheme+' parity')
            state = (integrate_rhs(rhs,state,values,contract['dt'],(step-1)*contract['dt'],scheme)
                     if rhs is not None else advance(state,values,contract['dt'],(step-1)*contract['dt']))
            sample = result['samples'].get(str(step))
            if sample is not None:
                for name, actual in zip(states, state):
                    expected = sample[reference['state_slots'][name]]
                    if not math.isfinite(actual) or abs(actual-expected) > 1e-10:
                        details = dict(state=name, step=step, actual=actual, expected=expected,
                                       absolute_error=abs(actual-expected), tolerance=1e-10)
                        raise ReferenceError('VAULT_REFERENCE_MISMATCH', str(details), details)
        output[contract['contract_id']] = True
    return output


if __name__ == '__main__':
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
        resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
    except ImportError:
        pass
    try:
        print(json.dumps(_run(json.load(sys.stdin)), allow_nan=False))
    except Exception as error:
        print(json.dumps({'error': str(error), 'code': getattr(error, 'code', 'VAULT_REFERENCE_EXECUTION_FAILED'),
                          'details': getattr(error, 'details', None)}))

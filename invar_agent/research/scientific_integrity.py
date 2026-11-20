"""Dataflow and execution checks for scientific observations and protocols."""
from __future__ import annotations

import ast
import copy
import functools
import json
import math
import re
import subprocess
import sys

from .slot_schema import _state_slot, _state_slots
from .evolution import check_evolution_reference
from ..runtime.research_reference import ReferenceError, run_reference


SCIENTIFIC_CONTRACT = """Scientific outputs must be computed from simulated states, never set or clipped to success thresholds.
Contract bounds are predicates, not assignments. Explicitly compute every contracted observable.
If no parameter setting satisfies the discovery predicates, report no solution, not fabricated success.
Match the requested physical horizon exactly. Every time-phased protocol must visit every phase
strictly inside that horizon. A named integration variant must execute that actual numerical scheme;
use explicit RHS functions and real stage evaluations, not aliases or relabelled Euler updates.
Keep derived observables separate from physical state and immutable input parameters.
Reserve the predicate symbol tol for numerical zero-check bands, not physical bounds."""


_NONNEGATIVE_PROOF = '''
import json, sys
import sympy as s
symbols = {}
def build(n):
    op, *args = n
    if op == 'number': return s.Rational(args[0])
    if op == 'symbol': return symbols.setdefault(args[0], s.Symbol(args[0], real=True))
    values = [build(arg) for arg in args]
    if op == 'add': return values[0] + values[1]
    if op == 'sub': return values[0] - values[1]
    if op == 'mul': return values[0] * values[1]
    if op == 'pow': return values[0] ** values[1]
    if op == 'div': return values[0] / values[1]
    raise ValueError(op)
expression = build(json.load(sys.stdin))
if not symbols:
    result = expression.is_nonnegative is True
else:
    variables = list(symbols.values())
    polynomial = s.Poly(expression.expand(), *variables)
    result = all(c >= 0 and all(power % 2 == 0 for power in powers) for powers,c in polynomial.terms())
    if not result and polynomial.total_degree() <= 2:
        matrix = s.zeros(len(variables) + 1)
        for powers, coefficient in polynomial.terms():
            indices = [i+1 for i,power in enumerate(powers) for _ in range(power)]
            indices = ([0,0] if not indices else [0,indices[0]] if len(indices)==1 else indices)
            i,j = indices
            if i == j: matrix[i,j] += coefficient
            else:
                matrix[i,j] += coefficient/2
                matrix[j,i] += coefficient/2
        result = matrix.is_positive_semidefinite is True
print('true' if result else 'false')
'''


@functools.lru_cache(maxsize=256)
def _bounded_nonnegative_proof(payload):
    try:
        result = subprocess.run([sys.executable, '-c', _NONNEGATIVE_PROOF], input=payload,
                                text=True, capture_output=True, timeout=1.0)
        return result.returncode == 0 and result.stdout.strip() == 'true'
    except (OSError, subprocess.TimeoutExpired):
        return False


def prove_nonnegative(expression, definitions=None):
    definitions = definitions or {}
    if isinstance(expression, ast.Name):
        values = definitions.get(expression.id, [])
        nonzero = [value for value in values if not (isinstance(value, ast.Constant) and value.value == 0)]
        if len(values) > 1 and len(nonzero) == 1:
            expression = nonzero[0]
    mutable = {name for name, values in definitions.items() if len(values) != 1}
    arithmetic = (ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
    definitions = {name: values for name, values in definitions.items()
                   if len(values) == 1 and all(isinstance(node, arithmetic)
                       and not (isinstance(node, ast.Name) and node.id in mutable)
                       for node in ast.walk(values[0]))}
    atoms, fuel = {}, [128]

    def encode(node, seen=frozenset()):
        fuel[0] -= 1
        if fuel[0] < 0:
            raise ValueError('proof size budget')
        if isinstance(node, ast.Name) and node.id not in seen:
            values = definitions.get(node.id, [])
            if len(values) == 1:
                return encode(values[0], seen | {node.id})
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if len(str(node.value)) > 32:
                raise ValueError('proof constant budget')
            return ['number', str(node.value)], 1, 0
        if isinstance(node, (ast.Name, ast.Subscript)):
            key = ast.dump(node, include_attributes=False)
            if len(atoms) >= 12 and key not in atoms:
                raise ValueError('proof symbol budget')
            return ['symbol', atoms.setdefault(key, f'x{len(atoms)}')], 1, 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value, terms, degree = encode(node.operand, seen)
            return (['mul', ['number', '-1'], value] if isinstance(node.op, ast.USub) else value), terms, degree
        if not isinstance(node, ast.BinOp):
            raise ValueError('non-polynomial proof')
        left, lt, ld = encode(node.left, seen)
        right, rt, rd = encode(node.right, seen)
        op = {ast.Add:'add', ast.Sub:'sub', ast.Mult:'mul', ast.Div:'div', ast.Pow:'pow'}[type(node.op)]
        if op == 'pow':
            if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int or not 0 <= node.right.value <= 8:
                raise ValueError('proof power budget')
            terms, degree = lt ** node.right.value, ld * node.right.value
        elif op == 'div':
            if right[0] != 'number' or float(right[1]) == 0:
                raise ValueError('nonconstant proof denominator')
            terms, degree = lt, ld
        else:
            terms, degree = (lt + rt, max(ld,rd)) if op in {'add','sub'} else (lt * rt, ld + rd)
        if terms > 256 or degree > 8:
            raise ValueError('proof expansion budget')
        return [op, left, right], terms, degree

    try:
        payload, _, _ = encode(expression)
        return _bounded_nonnegative_proof(json.dumps(payload, separators=(',', ':')))
    except (ValueError, KeyError, OverflowError, RecursionError):
        return False


def _redundant_zero_guard(node, definitions):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'max' and len(node.args) == 2 and not node.keywords):
        return False
    for index in (0,1):
        constant = node.args[index]
        if isinstance(constant, ast.Constant) and type(constant.value) in (int,float) and constant.value == 0:
            return prove_nonnegative(node.args[1-index], definitions)
    return False


def observed_roundoff_scales(code, run):
    flow = output_dataflow(code)
    result = {}
    vectors = [run.get(key) or [] for key in ('state', 'initial', 'trace_min', 'trace_max')]
    for function in flow['functions'].values():
        definitions, stored = {}, {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        definitions.setdefault(target.id, []).append(node.value)
                    elif _state_slot(target) is not None and isinstance(node.value, ast.Name):
                        stored.setdefault(node.value.id, set()).add(_state_slot(target))

        def terms(node, seen=frozenset()):
            if len(seen) > 24:
                return None
            slot = _state_slot(node)
            slots = {slot} if slot is not None else stored.get(node.id, set()) if isinstance(node, ast.Name) else set()
            if slots:
                values = [abs(float(vector[index])) for index in slots for vector in vectors if index < len(vector)]
                return [max(values)] if values and all(math.isfinite(value) for value in values) else None
            if isinstance(node, ast.Name) and node.id not in seen and len(definitions.get(node.id, [])) == 1:
                return terms(definitions[node.id][0], seen | {node.id})
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                left, right = terms(node.left, seen), terms(node.right, seen)
                return left + right if left is not None and right is not None else None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'abs', 'max', 'min'}:
                values = [terms(arg, seen) for arg in node.args]
                return [item for value in values for item in value] if all(value is not None for value in values) else None
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return [abs(float(node.value))]
            return None

        for node in ast.walk(function):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                slot = _state_slot(target)
                if slot not in flow['derived']:
                    continue
                expression = node.value
                seen = set()
                while isinstance(expression, ast.Name) and expression.id not in seen and len(definitions.get(expression.id, [])) == 1:
                    seen.add(expression.id)
                    expression = definitions[expression.id][0]
                if not any(isinstance(item, ast.BinOp) and isinstance(item.op, ast.Sub) for item in ast.walk(expression)):
                    continue
                values = terms(expression)
                if values:
                    allowance = max(8, 2 * len(values)) * math.ulp(max(values))
                    result[slot] = max(result.get(slot, 0.0), allowance)
    return result


def evaluate_roundoff_predicate(expression, env, allowances):
    class Comparisons(ast.NodeTransformer):
        def visit_GeneratorExp(self, node):
            return node

        def visit_ListComp(self, node):
            return node

        def visit_Compare(self, node):
            if len(node.ops) != 1:
                return node
            left = eval(compile(ast.Expression(node.left), '<roundoff>', 'eval'), {'__builtins__': {}, **env})
            right = eval(compile(ast.Expression(node.comparators[0]), '<roundoff>', 'eval'), {'__builtins__': {}, **env})
            if type(left) not in (float, int) or type(right) not in (float, int) or not all(math.isfinite(value) for value in (left,right)):
                return node
            arithmetic = any(isinstance(item, ast.BinOp) for side in (node.left, node.comparators[0]) for item in ast.walk(side))
            if isinstance(node.ops[0], ast.Eq) and arithmetic:
                values = []
                def add_terms(side):
                    if isinstance(side, ast.BinOp) and isinstance(side.op, (ast.Add, ast.Sub)):
                        add_terms(side.left); add_terms(side.right)
                    else:
                        values.append(abs(float(eval(compile(ast.Expression(side), '<roundoff>', 'eval'), {'__builtins__': {}, **env}))))
                add_terms(node.left); add_terms(node.comparators[0])
                allowance = max(8, 2 * len(values)) * math.ulp(max(values))
                return ast.copy_location(ast.Constant(abs(left-right) <= allowance), node)
            observed = node.left
            if isinstance(observed, ast.Call) and isinstance(observed.func, ast.Name) and observed.func.id == 'abs' and len(observed.args) == 1:
                observed = observed.args[0]
            allowance = allowances.get(_state_slot(observed), 0.0)
            tolerance = abs(float(env.get('tol', 0.0)))
            bound = node.comparators[0]
            zero_band = isinstance(bound, ast.Name) and bound.id == 'tol'
            zero_band |= (isinstance(bound, ast.UnaryOp) and isinstance(bound.op, ast.USub)
                          and isinstance(bound.operand, ast.Name) and bound.operand.id == 'tol')
            if allowance and isinstance(node.ops[0], ast.Eq) and isinstance(bound, ast.Constant) and right == 0:
                return ast.copy_location(ast.Constant(abs(left) <= allowance), node)
            if allowance and (zero_band and abs(right) == tolerance and tolerance <= 2 * allowance or isinstance(bound, ast.Constant) and right == 0):
                if isinstance(node.ops[0], ast.LtE) and right >= 0:
                    return ast.copy_location(ast.Constant(left <= right + allowance), node)
                if isinstance(node.ops[0], ast.GtE) and right <= 0:
                    return ast.copy_location(ast.Constant(left >= right - allowance), node)
            return node
    try:
        tree = Comparisons().visit(ast.parse(expression, mode='eval'))
        return bool(eval(compile(ast.fix_missing_locations(tree), '<roundoff>', 'eval'), {'__builtins__': {}, **env}))
    except (ArithmeticError, LookupError, TypeError, NameError, ValueError):
        return False


def output_dataflow(code):
    tree = ast.parse(code) if isinstance(code, str) else code
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    reachable = set()
    queue = ['simulation_step']
    while queue:
        name = queue.pop()
        if name in reachable or name not in functions:
            continue
        reachable.add(name)
        queue.extend(node.func.id for node in ast.walk(functions[name])
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name))
    writes, reads, graph, clocks = set(), set(), {}, set()
    scopes = {}
    extrema = set()
    dynamic = False
    for name in reachable:
        fn = functions[name]
        bindings = {}

        def dependencies(expr):
            result = _state_slots(expr)
            for node in ast.walk(expr):
                if isinstance(node, ast.Name):
                    result |= bindings.get(node.id, set())
            return result

        assignments = [n if isinstance(n, ast.Assign) else ast.Assign(
            targets=[n.target], value=ast.BinOp(left=n.target, op=n.op, right=n.value))
            for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AugAssign))]
        step_slots = {n.value.value for n in assignments if isinstance(n.value, ast.Constant)
                      and type(n.value.value) is int and any(isinstance(t, ast.Name) and t.id == '_invar_step_slot' for t in n.targets)}
        for _ in range(len(assignments) + 1):
            changed = False
            for stmt in assignments:
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        old = bindings.get(target.id, set())
                        value = old | dependencies(stmt.value)
                        if value != old:
                            bindings[target.id] = value
                            changed = True
            if not changed:
                break
        reads |= _state_slots(fn)
        scopes[name] = bindings
        for stmt in assignments:
            for target in stmt.targets:
                slot = _state_slot(target)
                if slot is None:
                    if isinstance(target, ast.Subscript) and ast.unparse(target.value) == 'buf.data':
                        dynamic = True
                    continue
                writes.add(slot)
                deps = dependencies(stmt.value)
                graph.setdefault(slot, set()).update(deps)
                if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id in {'min', 'max'} and slot in deps and deps - {slot}:
                    extrema.add(slot)
                expr = stmt.value
                if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)):
                    if isinstance(expr.right, ast.Name) and expr.right.id == 'dt' and dependencies(expr.left) == {slot}:
                        clocks.add(slot)
                    if slot in step_slots and isinstance(expr.right, ast.Constant) and expr.right.value == 1:
                        clocks.add(slot)
    physical = set()
    for start in writes:
        pending = list(graph.get(start, ()))
        seen = set()
        while pending:
            slot = pending.pop()
            if slot == start:
                physical.add(start)
                break
            if slot not in seen:
                seen.add(slot)
                pending.extend(graph.get(slot, ()))
    physical -= extrema
    return dict(tree=tree, functions=functions, reachable=reachable, writes=writes,
                reads=reads, physical=physical, derived=writes-physical, clocks=clocks,
                dynamic_writes=dynamic, scopes=scopes, dependencies=graph)


def protected_observation_slots(tree, contracts=(), variable_map=None, physical_slots=(), roles=None):
    flow = output_dataflow(tree)
    protected = {slot for slot in flow['derived'] - set(physical_slots) if flow['dependencies'].get(slot)}
    labels = list((variable_map or {}).items())
    required = set()
    for contract in contracts:
        value = contract if isinstance(contract, dict) else contract.model_dump()
        for invariant in value.get('invariants') or []:
            slots = _state_slots(ast.parse(invariant['assertion'], mode='eval'))
            required |= slots
            labels.extend((invariant.get('variable_name', ''), slot) for slot in slots)
            if len(slots) == 1 and re.search(r'(?:^|_)conservation(?:_|$)', invariant.get('variable_name', ''), re.I):
                protected |= slots
    protected |= {slot for name, slot in labels if re.search(
        r'(?:^|_)(?:peak|min|max|corridor|ratio|error|drift|residual|diagnostic|outcome|threshold)(?:_|$)', name, re.I)}
    protected |= {slot for name, slot in (variable_map or {}).items() if (roles or {}).get(name) == 'diagnostic'}
    return protected | flow['clocks'] | (required - flow['writes'])


def physical_projection_contracts(tree, contracts, variable_map=None, physical_slots=(), roles=None):
    protected = protected_observation_slots(tree, contracts, variable_map, physical_slots, roles)
    result = []
    for contract in contracts:
        value = copy.deepcopy(contract if isinstance(contract, dict) else contract.model_dump())
        kept = []
        for inv in value.get('invariants') or []:
            try:
                expr = ast.parse(inv.get('assertion', ''), mode='eval').body
            except SyntaxError:
                continue
            parts = expr.values if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And) else [expr]
            parts = [part for part in parts if not _state_slots(part) & protected
                     and not any(isinstance(compare, ast.Compare)
                                 and not any(_state_slot(side) is not None for side in [compare.left, *compare.comparators])
                                 and any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and _state_slots(node) for node in ast.walk(compare))
                                 for compare in ast.walk(part))]
            if parts:
                predicate = parts[0] if len(parts) == 1 else ast.BoolOp(op=ast.And(), values=parts)
                kept.append({**inv, 'assertion': ast.unparse(predicate)})
        value['invariants'] = kept
        result.append(value)
    return result


def validate_observables(code, contracts, variable_map, roles, physical_slots=()):
    flow = output_dataflow(code)
    protected = protected_observation_slots(code, contracts, variable_map, physical_slots) | {slot for name, slot in variable_map.items()
        if roles.get(name) == 'diagnostic' or re.search(r'^(?:peak|min|max|corridor)_|_(?:ratio|peak|residual)$', name, re.I)}
    required = set()
    for contract in contracts:
        value = contract if isinstance(contract, dict) else contract.model_dump()
        for inv in value.get('invariants') or []:
            required |= _state_slots(ast.parse(inv['assertion'], mode='eval'))
    parameters = {slot for name, slot in variable_map.items() if roles.get(name) == 'read_only_parameter'}
    missing = required - flow['writes'] - parameters
    if missing and not flow['dynamic_writes']:
        raise ValueError(f'SCIENTIFIC_INTEGRITY: uncomputed contract observables at slots {sorted(missing)}')
    for name in flow['reachable']:
        fn = flow['functions'][name]
        definitions = {}
        for argument in fn.args.args:
            definitions[argument.arg] = [ast.Name(id=f'_input_{argument.arg}', ctx=ast.Load()),
                                         ast.Name(id=f'_input_{argument.arg}', ctx=ast.Load())]
        physical_locals = set()
        clamped_locals = set()
        zero_guard_locals = set()
        for stmt in ast.walk(fn):
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                definitions.setdefault(stmt.target.id, []).append(ast.BinOp(left=stmt.target, op=stmt.op, right=stmt.value))
            if isinstance(stmt, ast.For):
                for target in ast.walk(stmt.target):
                    if isinstance(target, ast.Name):
                        definitions.setdefault(target.id, []).extend([ast.Constant(None), ast.Constant(None)])
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        definitions.setdefault(target.id, []).append(stmt.value)
                    elif _state_slot(target) in flow['physical'] - protected and isinstance(stmt.value, ast.Name):
                        physical_locals.add(stmt.value.id)
            if isinstance(stmt, ast.If):
                tested_names = {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
                for child in stmt.body + stmt.orelse:
                    if isinstance(child, ast.Assign) and isinstance(child.value, ast.Constant):
                        clamped_locals.update(t.id for t in child.targets if isinstance(t, ast.Name) and t.id in tested_names)
                        if (child.value.value == 0 and isinstance(stmt.test, ast.Compare)
                                and isinstance(stmt.test.left, ast.Name) and len(stmt.test.ops) == 1
                                and isinstance(stmt.test.ops[0], (ast.Lt, ast.LtE))
                                and isinstance(stmt.test.comparators[0], ast.Constant)
                                and stmt.test.comparators[0].value == 0):
                            zero_guard_locals.update(t.id for t in child.targets if isinstance(t, ast.Name) and t.id == stmt.test.left.id)

        def clipped(expr, seen=None):
            seen = seen or set()
            for node in ast.walk(expr):
                if isinstance(node, ast.Call):
                    callee = ast.unparse(node.func).lower()
                    if ('clamp' in callee or callee.endswith('clip') or callee in {'min', 'max'}
                            and any(isinstance(arg, ast.Constant) for arg in node.args)
                            and not _redundant_zero_guard(node, definitions)):
                        return True
                if isinstance(node, ast.Name) and node.id not in physical_locals and node.id not in seen:
                    if node.id in clamped_locals:
                        values = list({ast.dump(value): value for value in definitions.get(node.id, [])
                                       if not (isinstance(value, ast.Constant) and value.value == 0)}.values())
                        if node.id not in zero_guard_locals or len(values) != 1 or not prove_nonnegative(values[0], definitions):
                            return True
                    if any(clipped(value, seen | {node.id}) for value in definitions.get(node.id, [])):
                        return True
            return False

        conditional_flags = set()
        bindings = flow['scopes'][name]
        for node in ast.walk(flow['functions'][name]):
            if isinstance(node, ast.If):
                control = _state_slots(node.test)
                for item in ast.walk(node.test):
                    if isinstance(item, ast.Name):
                        control |= bindings.get(item.id, set())
                for stmt in node.body + node.orelse:
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and stmt.value.value in (0, 1):
                        conditional_flags.update(_state_slot(t) for t in stmt.targets if control - {_state_slot(t)})
        for node in ast.walk(flow['functions'][name]):
            if isinstance(node, ast.If):
                tested = _state_slots(node.test) & protected
                for stmt in node.body + node.orelse:
                    if isinstance(stmt, ast.Assign) and any(_state_slot(t) in tested for t in stmt.targets):
                        if not _state_slots(stmt.value) and isinstance(stmt.value, (ast.Constant, ast.BinOp)):
                            raise ValueError(f'SCIENTIFIC_INTEGRITY: projected diagnostic at line {node.lineno}')
            if isinstance(node, ast.Assign) and any(_state_slot(t) in protected for t in node.targets):
                diagnostic = {slot for label, slot in variable_map.items() if roles.get(label) == 'diagnostic'
                    or re.search(r'^(?:peak|min|max|corridor)_|_(?:ratio|peak|residual)$', label, re.I)}
                if isinstance(node.value, ast.Constant) and any(_state_slot(t) in required & diagnostic for t in node.targets) and not any(_state_slot(t) in conditional_flags for t in node.targets):
                    raise ValueError(f'SCIENTIFIC_INTEGRITY: diagnostic is a constant, not a state-dependent computation at line {node.lineno}')
                if clipped(node.value):
                    raise ValueError(f'SCIENTIFIC_INTEGRITY: clipped diagnostic expression at line {node.lineno}')
                if isinstance(node.value, ast.Call):
                    name = ast.unparse(node.value.func).lower()
                    if 'clamp' in name or name.endswith('clip'):
                        raise ValueError(f'SCIENTIFIC_INTEGRITY: clamped diagnostic at line {node.lineno}')
                    if name in {'min', 'max'} and any(isinstance(arg, ast.Constant) for arg in node.value.args) and not _redundant_zero_guard(node.value, definitions):
                        raise ValueError(f'SCIENTIFIC_INTEGRITY: clipped diagnostic at line {node.lineno}')
    return flow


def instrument_phases(code):
    flow = output_dataflow(code)
    tree = flow['tree']
    labels = []
    for name in flow['reachable']:
        fn = flow['functions'][name]
        aliases = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and _state_slots(node.value) & flow['clocks']:
                aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            is_time = bool(_state_slots(node.test) & flow['clocks']) or any(
                isinstance(n, ast.Name) and n.id in aliases for n in ast.walk(node.test))
            if is_time and any(isinstance(n, ast.Assign) for stmt in node.body + node.orelse for n in ast.walk(stmt)):
                labels.append(f'{name}:{node.lineno}: {ast.unparse(node.test)}')
                node.test = ast.Call(func=ast.Name(id='_invar_phase_check', ctx=ast.Load()),
                    args=[ast.Constant(len(labels)-1), node.test], keywords=[])
    if labels:
        if '_invar_phase_hits' in code or '_invar_phase_check' in code:
            raise ValueError('SCIENTIFIC_INTEGRITY: reserved phase-audit symbol')
        helper = ast.parse('def _invar_phase_check(index, value):\n    _invar_phase_hits[2 * index + int(bool(value))] += 1\n    return value\n')
        tree.body = ast.parse(f'_invar_phase_hits = {[0] * (2*len(labels))!r}').body + helper.body + tree.body
        ast.fix_missing_locations(tree)
    return ast.unparse(tree), labels


def validate_protocol(code, contract, question):
    if not re.search(r'\b(protocol|phase|phases|pulse|switch|piecewise)\b', question or '', re.I):
        return {}
    instrumented, labels = instrument_phases(code)
    if not labels:
        if re.search(r'\b(phases|switch|piecewise)\b', question or '', re.I):
            raise ValueError('SCIENTIFIC_INTEGRITY: no auditable time-phase conditions')
        return {}
    value = contract if isinstance(contract, dict) else contract.model_dump()
    result = run_reference(instrumented, {**value, 'invariants': []}, sample_steps=[value['steps']])
    hits = result['phase_hits']
    coverage = {label: {'false': hits[2*i], 'true': hits[2*i+1]} for i, label in enumerate(labels)}
    missing = [label for label, counts in coverage.items() if not all(counts.values())]
    if missing:
        raise ValueError(f'SCIENTIFIC_INTEGRITY: unreachable protocol phases within dt*steps={value["dt"]*value["steps"]}: {missing}; coverage={coverage}')
    return coverage


def validate_variant_schemes(code, blueprint, contracts):
    tree = ast.parse(code)
    entries = [node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.args.args and node.args.args[0].arg == 'buf'
               and re.search(r'_(rk4|euler|midpoint|heun)$', node.name)]
    if not contracts:
        return {}
    original = contracts[0] if isinstance(contracts[0], dict) else contracts[0].model_dump()
    summaries = {}
    for entry in entries:
        variant = copy.deepcopy(tree)
        for node in ast.walk(variant):
            if isinstance(node, ast.FunctionDef) and node.name == 'simulation_step':
                node.name = '_invar_benchmark_base'
            elif isinstance(node, ast.Name) and node.id == 'simulation_step':
                node.id = '_invar_benchmark_base'
        variant.body.extend(ast.parse(f'def simulation_step(buf, dt):\n    {entry}(buf, dt)\n').body)
        source = ast.unparse(ast.fix_missing_locations(variant))
        flow = output_dataflow(source)
        physical = flow['physical']
        if not physical or len(physical) > 32:
            raise ValueError(f'SCIENTIFIC_INTEGRITY: cannot verify scheme label {entry}: unsupported state shape')
        slots = {name: slot for name, slot in blueprint.variable_map.items() if slot in flow['reads']}
        for slot in flow['reads']:
            if slot not in slots.values():
                slots[f'state_{slot}'] = slot
        initial = list(original.get('initial_state') or [])
        initial += [0.0] * (max(flow['reads'], default=-1) + 1 - len(initial))
        parameters = {name: {'si_value': initial[slot], 'si_unit': '1'} for name, slot in slots.items() if slot not in physical}
        model = {'variable_map': slots, 'state_order': [name for name, slot in slots.items() if slot in physical],
                 'structured_evidence': {'parameters': parameters},
                 'contracts': [{**original, 'steps': 1, 'initial_state': initial, 'invariants': []}]}
        try:
            recovered = check_evolution_reference(model, {}, source)
            actual = recovered['integration_scheme']
        except (ReferenceError, ValueError, TypeError, KeyError) as error:
            raise ValueError(f'SCIENTIFIC_INTEGRITY: cannot verify integration scheme {entry}: {error}') from error
        claimed = entry.rsplit('_', 1)[1]
        if claimed != actual:
            raise ValueError(f'SCIENTIFIC_INTEGRITY: integration label {entry} claims {claimed}, executed scheme is {actual}')
        summaries[entry] = actual
    return summaries

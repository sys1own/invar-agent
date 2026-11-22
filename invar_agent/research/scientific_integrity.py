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

from .slot_schema import (
    ROLE_CONTROL_PARAM,
    ROLE_DIAGNOSTIC,
    ROLE_STATE,
    ROLE_TIME_STEP,
    _state_slot,
    _state_slots,
    classify_slot_roles,
)
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
from fractions import Fraction
import json, sys

sym_idx = {}

def add_poly(a, b, sign=1):
    out = dict(a)
    for k, v in b.items():
        nv = out.get(k, Fraction(0)) + sign * v
        if nv:
            out[k] = nv
        else:
            out.pop(k, None)
    return out

def mul_poly(a, b):
    out = {}
    for k1, v1 in a.items():
        for k2, v2 in b.items():
            d = {}
            for var, p in k1:
                d[var] = d.get(var, 0) + p
            for var, p in k2:
                d[var] = d.get(var, 0) + p
            nk = tuple(sorted((var, p) for var, p in d.items() if p))
            nv = out.get(nk, Fraction(0)) + v1 * v2
            if nv:
                out[nk] = nv
            else:
                out.pop(nk, None)
    return out

def pow_poly(a, exp):
    if exp == 0:
        return {(): Fraction(1)}
    res = {(): Fraction(1)}
    base = a
    e = exp
    while e > 0:
        if e & 1:
            res = mul_poly(res, base)
        e >>= 1
        if e:
            base = mul_poly(base, base)
    return res

def build(n):
    op, *args = n
    if op == 'number':
        v = Fraction(str(args[0]))
        return {(): v} if v else {}
    if op == 'symbol':
        idx = sym_idx.setdefault(args[0], len(sym_idx))
        return {((idx, 1),): Fraction(1)}
    vals = [build(arg) for arg in args]
    if op == 'add':
        return add_poly(vals[0], vals[1], 1)
    if op == 'sub':
        return add_poly(vals[0], vals[1], -1)
    if op == 'mul':
        return mul_poly(vals[0], vals[1])
    if op == 'div':
        if list(vals[1].keys()) != [()]:
            raise ValueError('nonconstant div')
        denom = vals[1][()]
        return {k: v / denom for k, v in vals[0].items()}
    if op == 'pow':
        if vals[1] == {}:
            return {(): Fraction(1)}
        if list(vals[1].keys()) != [()] or vals[1][()].denominator != 1 or vals[1][()] < 0:
            raise ValueError('bad pow')
        return pow_poly(vals[0], int(vals[1][()]))
    raise ValueError(op)

def is_psd_rational(mat):
    n = len(mat)
    a = [[mat[i][j] for j in range(n)] for i in range(n)]
    for k in range(n):
        if a[k][k] < 0:
            return False
        if a[k][k] == 0:
            if any(a[k][j] != 0 or a[j][k] != 0 for j in range(k, n)):
                return False
            continue
        piv = a[k][k]
        for i in range(k + 1, n):
            if a[i][k] == 0:
                continue
            fac = a[i][k] / piv
            for j in range(k + 1, n):
                a[i][j] -= fac * a[k][j]
            a[i][k] = Fraction(0)
            a[k][i] = Fraction(0)
    return True

poly = build(json.load(sys.stdin))
if not sym_idx:
    result = poly.get((), Fraction(0)) >= 0
else:
    result = all(v >= 0 and all(p % 2 == 0 for _, p in k) for k, v in poly.items())
    max_deg = max((sum(p for _, p in k) for k in poly), default=0)
    if not result and max_deg <= 2:
        dim = len(sym_idx) + 1
        mat = [[Fraction(0) for _ in range(dim)] for _ in range(dim)]
        for k, coeff in poly.items():
            indices = [v + 1 for v, p in k for _ in range(p)]
            if not indices:
                i = j = 0
            elif len(indices) == 1:
                i, j = 0, indices[0]
            else:
                i, j = indices[0], indices[1]
            if i == j:
                mat[i][j] += coeff
            else:
                half = coeff / 2
                mat[i][j] += half
                mat[j][i] += half
        result = is_psd_rational(mat)
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
    scopes = {name: {} for name in reachable}
    return_deps = {name: [] for name in reachable}
    extrema = set()
    dynamic = False
    for _pass in range(len(reachable) + 2):
        any_scope_change = False
        for name in reachable:
            fn = functions[name]
            bindings = scopes[name]

            def dependencies(expr, _b=bindings):
                result = _state_slots(expr)
                for node in ast.walk(expr):
                    if isinstance(node, ast.Name):
                        result |= _b.get(node.id, set())
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in return_deps:
                        for rset in return_deps[node.func.id]:
                            result |= rset
                return result

            assignments = [n if isinstance(n, ast.Assign) else ast.Assign(
                targets=[n.target], value=ast.BinOp(left=n.target, op=n.op, right=n.value))
                for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AugAssign))]

            def _bind_target(target, rhs_node):
                if isinstance(target, ast.Name):
                    old = bindings.get(target.id, set())
                    value = old | dependencies(rhs_node)
                    if value != old:
                        bindings[target.id] = value
                        return True
                    return False
                if isinstance(target, (ast.Tuple, ast.List)):
                    if isinstance(rhs_node, (ast.Tuple, ast.List)) and len(target.elts) == len(rhs_node.elts):
                        any_c = False
                        for sub_t, sub_v in zip(target.elts, rhs_node.elts):
                            if _bind_target(sub_t, sub_v):
                                any_c = True
                        return any_c
                    if isinstance(rhs_node, ast.Call) and isinstance(rhs_node.func, ast.Name) and rhs_node.func.id in return_deps:
                        rlist = return_deps[rhs_node.func.id]
                        any_c = False
                        for idx, sub_t in enumerate(target.elts):
                            if isinstance(sub_t, ast.Name):
                                old = bindings.get(sub_t.id, set())
                                rset = rlist[idx] if idx < len(rlist) else (set().union(*rlist) if rlist else dependencies(rhs_node))
                                value = old | rset
                                if value != old:
                                    bindings[sub_t.id] = value
                                    any_c = True
                        return any_c
                return False

            for _ in range(len(assignments) + 1):
                changed = False
                for stmt in assignments:
                    for target in stmt.targets:
                        if _bind_target(target, stmt.value):
                            changed = True
                            any_scope_change = True
                if not changed:
                    break

            # Propagate argument dependencies to callees and collect return dependencies
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in scopes:
                    callee_fn = functions[node.func.id]
                    callee_scope = scopes[node.func.id]
                    for param, arg_expr in zip(callee_fn.args.args, node.args):
                        if param.arg == 'buf':
                            continue
                        old_p = callee_scope.get(param.arg, set())
                        new_p = old_p | dependencies(arg_expr)
                        if new_p != old_p:
                            callee_scope[param.arg] = new_p
                            any_scope_change = True
                elif isinstance(node, ast.Return) and node.value is not None:
                    if isinstance(node.value, (ast.Tuple, ast.List)):
                        cur_ret = [dependencies(elt) for elt in node.value.elts]
                    else:
                        cur_ret = [dependencies(node.value)]
                    prev_ret = return_deps.get(name) or []
                    if len(cur_ret) > len(prev_ret):
                        prev_ret = prev_ret + [set() for _ in range(len(cur_ret) - len(prev_ret))]
                    merged_ret = [a | b for a, b in zip(prev_ret, cur_ret)] + prev_ret[len(cur_ret):]
                    if merged_ret != return_deps.get(name):
                        return_deps[name] = merged_ret
                        any_scope_change = True
        if not any_scope_change:
            break

    for name in reachable:
        fn = functions[name]
        bindings = scopes[name]

        def dependencies(expr, _b=bindings):
            result = _state_slots(expr)
            for node in ast.walk(expr):
                if isinstance(node, ast.Name):
                    result |= _b.get(node.id, set())
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in return_deps:
                    for rset in return_deps[node.func.id]:
                        result |= rset
            return result

        assignments = [n if isinstance(n, ast.Assign) else ast.Assign(
            targets=[n.target], value=ast.BinOp(left=n.target, op=n.op, right=n.value))
            for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AugAssign))]
        step_slots = {n.value.value for n in assignments if isinstance(n.value, ast.Constant)
                      and type(n.value.value) is int and any(isinstance(t, ast.Name) and t.id == '_invar_step_slot' for t in n.targets)}
        if_extrema_vars = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
                if isinstance(node.test.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                    cmp_names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                    for child in node.body:
                        if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name) and isinstance(child.value, ast.Name):
                            if (
                                child.targets[0].id in cmp_names
                                and child.value.id in cmp_names
                                and child.targets[0].id != child.value.id
                                and bool(bindings.get(child.value.id))
                                and not child.value.id.isupper()
                            ):
                                if_extrema_vars.add(child.targets[0].id)

        def _record_slot_target(target, rhs_node):
            nonlocal dynamic
            slot = _state_slot(target)
            if slot is not None:
                writes.add(slot)
                deps = dependencies(rhs_node)
                graph.setdefault(slot, set()).update(deps)
                if isinstance(rhs_node, ast.Name) and rhs_node.id in if_extrema_vars and slot in deps and (deps - {slot}):
                    extrema.add(slot)
                if (
                    isinstance(rhs_node, ast.Call)
                    and isinstance(rhs_node.func, ast.Name)
                    and rhs_node.func.id in {'min', 'max'}
                    and len(rhs_node.args) >= 2
                    and not any(isinstance(arg, ast.Constant) for arg in rhs_node.args)
                ):
                    arg_deps = [dependencies(arg) for arg in rhs_node.args]
                    if any(d == {slot} for d in arg_deps) and any(d - {slot} for d in arg_deps):
                        extrema.add(slot)
                if isinstance(rhs_node, ast.BinOp) and isinstance(rhs_node.op, (ast.Add, ast.Sub)):
                    if isinstance(rhs_node.right, ast.Name) and rhs_node.right.id == 'dt' and dependencies(rhs_node.left) == {slot}:
                        clocks.add(slot)
                    if slot in step_slots and isinstance(rhs_node.right, ast.Constant) and rhs_node.right.value == 1:
                        clocks.add(slot)
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(rhs_node, (ast.Tuple, ast.List)) and len(target.elts) == len(rhs_node.elts):
                for sub_t, sub_v in zip(target.elts, rhs_node.elts):
                    _record_slot_target(sub_t, sub_v)
            elif isinstance(target, ast.Subscript) and ast.unparse(target.value) == 'buf.data':
                dynamic = True

        reads |= _state_slots(fn)
        for stmt in assignments:
            for target in stmt.targets:
                _record_slot_target(target, stmt.value)
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
    # Promote algebraic conservation-complement species (e.g. C = K - A - B)
    # and co-summands of additive pool sums into physical state (never DIAGNOSTIC).
    for fn in functions.values():
        loc_exprs = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                loc_exprs[node.targets[0].id] = node.value
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt_s = _state_slot(node.targets[0])
            if tgt_s is None or tgt_s in clocks or tgt_s in extrema:
                continue
            rhs = node.value
            for _ in range(3):
                if isinstance(rhs, ast.Name) and rhs.id in loc_exprs:
                    rhs = loc_exprs[rhs.id]
                else:
                    break
            if isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.Sub):
                if (graph.get(tgt_s, set()) & physical) and not any(
                    isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in {'abs', 'min', 'max'}
                    for c in ast.walk(rhs)
                ):
                    left = rhs.left
                    while isinstance(left, ast.BinOp) and isinstance(left.op, ast.Sub):
                        left = left.left
                    if (
                        (isinstance(left, ast.Constant) and isinstance(left.value, (int, float)) and left.value > 0)
                        or (isinstance(left, ast.Name) and left.id.isupper())
                    ):
                        physical.add(tgt_s)
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
    protected |= {slot for name, slot in (variable_map or {}).items() if (roles or {}).get(name) in ('diagnostic', ROLE_DIAGNOSTIC)}
    shared_roles = classify_slot_roles(
        variable_map, tree, contracts, explicit_roles=roles, physical_slots=physical_slots,
    )
    mapped_slots = {int(s) for s in (variable_map or {}).values()}
    state_role_slots = {slot for slot in mapped_slots if shared_roles.get(slot) == ROLE_STATE}
    protected -= state_role_slots
    protected |= {
        slot for slot, role in shared_roles.items()
        if role in (ROLE_CONTROL_PARAM, ROLE_TIME_STEP)
        or (role == ROLE_DIAGNOSTIC and (slot in mapped_slots or slot in (flow['derived'] - set(physical_slots))))
    }
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
    slot_names = {}
    for label, slot in (variable_map or {}).items():
        slot_names.setdefault(int(slot), str(label))
    shared_roles = classify_slot_roles(
        variable_map, flow['tree'], contracts, explicit_roles=roles, physical_slots=physical_slots,
    )
    protected = protected_observation_slots(code, contracts, variable_map, physical_slots, roles) | {
        slot for name, slot in (variable_map or {}).items()
        if (roles or {}).get(name) in ('diagnostic', ROLE_DIAGNOSTIC)
        or shared_roles.get(slot) in (ROLE_DIAGNOSTIC, ROLE_CONTROL_PARAM, ROLE_TIME_STEP)
        or re.search(r'^(?:peak|min|max|corridor)_|_(?:ratio|peak|residual)$', name, re.I)
    }
    required = set()
    for contract in contracts:
        value = contract if isinstance(contract, dict) else contract.model_dump()
        for inv in value.get('invariants') or []:
            inv_slots = _state_slots(ast.parse(inv['assertion'], mode='eval'))
            required |= inv_slots
            for s in inv_slots:
                if inv.get('variable_name'):
                    slot_names.setdefault(s, str(inv['variable_name']))
    parameters = {
        slot for name, slot in (variable_map or {}).items()
        if (roles or {}).get(name) in ('read_only_parameter', ROLE_CONTROL_PARAM)
        or shared_roles.get(slot) == ROLE_CONTROL_PARAM
    }
    missing = required - flow['writes'] - parameters
    if missing and not flow['dynamic_writes']:
        missing_desc = ', '.join(f"{slot_names.get(s, f'slot_{s}')} (buf.data[{s}])" for s in sorted(missing))
        raise ValueError(
            f'SCIENTIFIC_INTEGRITY: uncomputed contract observables at slots {sorted(missing)} ({missing_desc}). '
            f'Explicitly compute and assign each observable to its buf.data[i] slot inside simulation_step.'
        )
    if len(contracts) > 1 and (flow['writes'] or flow['dynamic_writes']) and variable_map:
        mapped_ctrl = {
            int(slot): str(name) for name, slot in variable_map.items()
            if shared_roles.get(int(slot)) == ROLE_CONTROL_PARAM
        }
        for c_slot, c_name in sorted(mapped_ctrl.items()):
            distinct_vals = set()
            for c_item in contracts:
                c_val = c_item if isinstance(c_item, dict) else c_item.model_dump()
                init_vec = c_val.get('initial_state') or []
                if c_slot < len(init_vec):
                    try:
                        distinct_vals.add(round(float(init_vec[c_slot]), 12))
                    except (TypeError, ValueError):
                        pass
            if len(distinct_vals) > 1 and c_slot not in flow['reads']:
                raise ValueError(
                    f"SCIENTIFIC_INTEGRITY: unused swept control parameter '{c_name}' (buf.data[{c_slot}]) in simulation_step; "
                    f"the active integrator ignores parameter-sweep variations ({sorted(distinct_vals)[:4]}) and must not replace governing ODEs with fixed target relaxation."
                )
    diagnostic_slots = {
        slot for label, slot in (variable_map or {}).items()
        if (roles or {}).get(label) in ('diagnostic', ROLE_DIAGNOSTIC)
        or shared_roles.get(slot) == ROLE_DIAGNOSTIC
        or re.search(r'^(?:peak|min|max|corridor|best)_|_(?:ratio|peak|residual)$', label, re.I)
    }
    # Slots governed by an exact algebraic equilibrium / gauge / flatness equality
    # (e.g. `abs(state[i] - 1.0) <= 1e-8` or `state[i] == 1.0`) may legitimately hold
    # that equilibrium value across steps without being rejected as a "constant cheat".
    _exact_eq_re = re.compile(
        r"abs\(\s*(?:state|buf\.data|data)\[(\d+)\]\s*-\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\)\s*<=?\s*([0-9.eE+\-]+)"
        r"|(?:state|buf\.data|data)\[(\d+)\]\s*==\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
    )
    exact_equilibrium_slots = set()
    for _c in (contracts or ()):
        _invs = getattr(_c, 'invariants', None) if not isinstance(_c, dict) else _c.get('invariants')
        for _inv in (_invs or ()):
            _asrt = str(getattr(_inv, 'assertion', '') if not isinstance(_inv, dict) else _inv.get('assertion', ''))
            for _m in _exact_eq_re.finditer(_asrt):
                if _m.group(1) is not None:
                    try:
                        if float(_m.group(2)) <= 1e-5:
                            exact_equilibrium_slots.add(int(_m.group(1)))
                    except ValueError:
                        pass
                elif _m.group(3) is not None:
                    exact_equilibrium_slots.add(int(_m.group(3)))
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
                    elif (
                        _state_slot(target) in (flow['physical'] | {s for s, r in shared_roles.items() if r == ROLE_STATE})
                        and _state_slot(target) not in diagnostic_slots
                        and isinstance(stmt.value, ast.Name)
                    ):
                        physical_locals.add(stmt.value.id)
            if isinstance(stmt, ast.If):
                tested_names = {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
                for child in stmt.body + stmt.orelse:
                    if isinstance(child, ast.Assign) and isinstance(child.value, ast.Constant):
                        clamped_locals.update(t.id for t in child.targets if isinstance(t, ast.Name) and t.id in tested_names)
                        if (isinstance(child.value.value, (int, float))
                                and 0.0 <= float(child.value.value) <= 1e-6
                                and isinstance(stmt.test, ast.Compare)
                                and isinstance(stmt.test.left, ast.Name) and len(stmt.test.ops) == 1
                                and isinstance(stmt.test.ops[0], (ast.Lt, ast.LtE))
                                and isinstance(stmt.test.comparators[0], ast.Constant)
                                and isinstance(stmt.test.comparators[0].value, (int, float))
                                and 0.0 <= float(stmt.test.comparators[0].value) <= 1e-6):
                            zero_guard_locals.update(t.id for t in child.targets if isinstance(t, ast.Name) and t.id == stmt.test.left.id)

        def _is_tiny_eps_floor(call_node: ast.Call) -> bool:
            """Return True for `max(expr, eps)` denominator guards with 0 <= eps <= 1e-6."""
            callee = ast.unparse(call_node.func).lower()
            if callee != 'max' or len(call_node.args) != 2:
                return False
            consts = [
                float(a.value) for a in call_node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, (int, float))
            ]
            return len(consts) == 1 and 0.0 <= consts[0] <= 1e-6

        def clipped(expr, seen=None):
            seen = seen or set()
            for node in ast.walk(expr):
                if isinstance(node, ast.Call):
                    callee = ast.unparse(node.func).lower()
                    if ('clamp' in callee or callee.endswith('clip') or callee in {'min', 'max'}
                            and any(isinstance(arg, ast.Constant) for arg in node.args)
                            and not _redundant_zero_guard(node, definitions)
                            and not _is_tiny_eps_floor(node)):
                        return True
                if isinstance(node, ast.Name) and node.id not in physical_locals and node.id not in seen:
                    if node.id in clamped_locals:
                        if node.id in zero_guard_locals:
                            # Tiny epsilon denominator guard (e.g. if pool < 1e-12: pool = 1e-12)
                            # or non-negativity guard on a provably non-negative / physical expression.
                            continue
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
        def _diag_err(kind: str, node_obj: ast.AST, target_slots: set[int]) -> ValueError:
            s_idx = min(target_slots) if target_slots else -1
            vname = slot_names.get(s_idx, f'slot_{s_idx}') if s_idx >= 0 else 'unknown'
            snippet = ast.unparse(node_obj).splitlines()[0][:160]
            slot_ref = f'buf.data[{s_idx}]' if s_idx >= 0 else 'diagnostic slot'
            return ValueError(
                f"SCIENTIFIC_INTEGRITY: {kind} at line {node_obj.lineno} on '{vname}' ({slot_ref}): `{snippet}`. "
                f"Do not clamp, project, or reset diagnostic slot '{vname}' ({slot_ref}); "
                f"enforce positivity/stability in the upstream state ODE integrator (e.g., Patankar/semi-implicit) "
                f"and compute '{vname}' purely from final state."
            )

        nonconst_diag_slots = set()
        for _rname in flow['reachable']:
            for _rnode in ast.walk(flow['functions'][_rname]):
                if isinstance(_rnode, ast.Assign) and not isinstance(_rnode.value, ast.Constant):
                    for _rt in _rnode.targets:
                        _rs = _state_slot(_rt)
                        if _rs is not None:
                            nonconst_diag_slots.add(_rs)
                elif isinstance(_rnode, ast.AugAssign):
                    _rs = _state_slot(_rnode.target)
                    if _rs is not None:
                        nonconst_diag_slots.add(_rs)
        _ulp_deadband_assigns = set()
        for node in ast.walk(flow['functions'][name]):
            if isinstance(node, ast.If):
                # Exempt sub-ULP zero-flush deadbands (`if abs(buf.data[i]) <= 1e-11: buf.data[i] = 0.0`)
                # emitted by the reversibility pass for machine-roundoff noise.
                if (
                    not node.orelse
                    and len(node.body) == 1
                    and isinstance(node.body[0], ast.Assign)
                    and isinstance(node.body[0].value, ast.Constant)
                    and node.body[0].value.value == 0.0
                    and isinstance(node.test, ast.Compare)
                    and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], (ast.Lt, ast.LtE))
                    and isinstance(node.test.left, ast.Call)
                    and isinstance(node.test.left.func, ast.Name)
                    and node.test.left.func.id == 'abs'
                    and isinstance(node.test.comparators[0], ast.Constant)
                    and isinstance(node.test.comparators[0].value, (int, float))
                    and float(node.test.comparators[0].value) <= 1e-10
                ):
                    _ulp_deadband_assigns.add(id(node.body[0]))
                    continue
                # Exempt NaN/Inf finite-value safety guards (`if not math.isfinite(buf.data[i]): buf.data[i] = 0.0`)
                # emitted by the sanitizer or defensive kernels (never a physical threshold clamp).
                if (
                    not node.orelse
                    and len(node.body) == 1
                    and isinstance(node.body[0], ast.Assign)
                    and isinstance(node.body[0].value, ast.Constant)
                    and node.body[0].value.value == 0.0
                    and isinstance(node.test, ast.UnaryOp)
                    and isinstance(node.test.op, ast.Not)
                    and isinstance(node.test.operand, ast.Call)
                    and ast.unparse(node.test.operand.func) in ('math.isfinite', 'isfinite')
                ):
                    _ulp_deadband_assigns.add(id(node.body[0]))
                    continue
                tested = (_state_slots(node.test) & protected) - exact_equilibrium_slots
                for stmt in node.body + node.orelse:
                    if isinstance(stmt, ast.Assign):
                        hit_slots = {_state_slot(t) for t in stmt.targets} & tested
                        if hit_slots and not _state_slots(stmt.value) and isinstance(stmt.value, (ast.Constant, ast.BinOp)):
                            raise _diag_err('projected diagnostic', node, hit_slots)
            if isinstance(node, ast.Assign) and id(node) not in _ulp_deadband_assigns:
                assigned_prot = {_state_slot(t) for t in node.targets} & protected
                if not assigned_prot:
                    continue
                assigned_contracted = assigned_prot & required
                diagnostic = {
                    slot for label, slot in (variable_map or {}).items()
                    if (roles or {}).get(label) in ('diagnostic', ROLE_DIAGNOSTIC)
                    or shared_roles.get(slot) == ROLE_DIAGNOSTIC
                    or re.search(r'^(?:peak|min|max|corridor)_|_(?:ratio|peak|residual)$', label, re.I)
                }
                if (
                    isinstance(node.value, ast.Constant)
                    and (assigned_contracted & diagnostic)
                    and not (assigned_prot & conditional_flags)
                    and not (assigned_contracted & nonconst_diag_slots)
                    and not (assigned_contracted & exact_equilibrium_slots)
                ):
                    raise _diag_err('diagnostic is a constant, not a state-dependent computation', node, assigned_prot)
                if assigned_contracted:
                    if clipped(node.value):
                        raise _diag_err('clipped diagnostic expression', node, assigned_contracted)
                    if isinstance(node.value, ast.Call):
                        callee_name = ast.unparse(node.value.func).lower()
                        if 'clamp' in callee_name or callee_name.endswith('clip'):
                            raise _diag_err('clamped diagnostic', node, assigned_contracted)
                        if callee_name in {'min', 'max'} and any(isinstance(arg, ast.Constant) for arg in node.value.args) and not _redundant_zero_guard(node.value, definitions):
                            raise _diag_err('clipped diagnostic', node, assigned_contracted)
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
        # Filter flow['physical'] via flow-sensitive entry-read check so algebraic
        # conservation complements written without reading their entry buffer
        # slot (e.g. amp_new = TOTAL - atp_new - adp_new) are not misclassified
        # as differential states during scheme verification.
        from .evolution import _entry_read_slots
        entry_reads = _entry_read_slots(source)
        vm_bp = getattr(blueprint, 'variable_map', None) or {}
        roles_bp = getattr(blueprint, 'slot_roles', None) or {}
        shared_roles = classify_slot_roles(
            vm_bp,
            source,
            contracts,
            explicit_roles=roles_bp,
            physical_slots=flow['physical'],
        )
        physical = (flow['physical'] - flow['clocks'])
        if entry_reads:
            physical = (physical & entry_reads) or physical
        physical = {
            s for s in (physical or flow['physical'])
            if shared_roles.get(s, ROLE_STATE) == ROLE_STATE
        }
        if not physical or len(physical) > 32:
            raise ValueError(f'SCIENTIFIC_INTEGRITY: cannot verify scheme label {entry}: unsupported state shape')
        slots = {name: slot for name, slot in vm_bp.items() if slot in flow['reads']}
        for slot in flow['reads']:
            if slot not in slots.values():
                slots[f'state_{slot}'] = slot
        initial = list(original.get('initial_state') or [])
        initial += [0.0] * (max(flow['reads'], default=-1) + 1 - len(initial))
        parameters = {
            name: {'si_value': initial[slot], 'si_unit': '1'}
            for name, slot in slots.items()
            if slot not in physical and slot not in flow['clocks']
        }
        model = {'variable_map': slots, 'state_order': [name for name, slot in slots.items() if slot in physical],
                 'structured_evidence': {'parameters': parameters},
                 'contracts': [{**original, 'steps': 1, 'initial_state': initial, 'invariants': []}]}
        try:
            recovered = check_evolution_reference(model, {}, source)
            actual = recovered['integration_scheme']
        except (ReferenceError, ValueError, TypeError, KeyError, NameError, ArithmeticError) as error:
            raise ValueError(f'SCIENTIFIC_INTEGRITY: cannot verify integration scheme {entry}: {error}') from error
        claimed = entry.rsplit('_', 1)[1]
        if claimed != actual:
            raise ValueError(f'SCIENTIFIC_INTEGRITY: integration label {entry} claims {claimed}, executed scheme is {actual}')
        summaries[entry] = actual
    return summaries

"""Deterministic recovery and independent validation of declared ODE models."""
from __future__ import annotations

import ast
import re

from ..runtime.research_reference import ReferenceError, run_reference, integrate_rhs, evaluate_predicate, _safe_namespace


NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
DERIVATIVE = re.compile(r'\bd([A-Za-z_]\w*)/dt\s*=\s*([^,;\n]+)')


def text_evolutions(text):
    output = {}
    for match in DERIVATIVE.finditer(text or ''):
        rhs = re.split(r'\s+(?:with|where|and|for|preserves|enforcing|subject|on)\b', match[2], maxsplit=1)[0].strip().rstrip('.')
        rhs = re.sub(r'(?<=[\w)])\s+(?=[A-Za-z_(])', '*', rhs)
        output[match[1]] = rhs
    return output


def _const_int_eval(expr, int_env):
    if isinstance(expr, ast.Constant) and type(expr.value) is int:
        return expr.value
    if isinstance(expr, ast.Name) and expr.id in int_env:
        return int_env[expr.id]
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
        val = _const_int_eval(expr.operand, int_env)
        return (-val if isinstance(expr.op, ast.USub) else val) if val is not None else None
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult)):
        lval = _const_int_eval(expr.left, int_env)
        rval = _const_int_eval(expr.right, int_env)
        if lval is None or rval is None:
            return None
        if isinstance(expr.op, ast.Add):
            return lval + rval
        if isinstance(expr.op, ast.Sub):
            return lval - rval
        return lval * rval
    return None


def _slot(node, int_env=None):
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) or (isinstance(node.value, ast.Attribute) and node.value.attr == 'data'):
            if isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
                return node.slice.value
            if int_env:
                idx = _const_int_eval(node.slice, int_env)
                if type(idx) is int and idx >= 0:
                    return idx
    return None


def _function_slots_rw(tree):
    """Return ``(written, readonly_loads)`` resolving local integer index aliases."""
    written: set[int] = set()
    loads: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            int_env: dict[str, int] = {}
            for st in node.body:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    iv = _const_int_eval(st.value, int_env)
                    if iv is not None:
                        int_env[st.targets[0].id] = iv
            for sub in ast.walk(node):
                if isinstance(sub, ast.Subscript):
                    idx = _slot(sub, int_env)
                    if idx is not None:
                        if isinstance(sub.ctx, ast.Store):
                            written.add(idx)
                        elif isinstance(sub.ctx, ast.Load):
                            loads.add(idx)
    return written, loads - written


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


def _entry_read_slots(code: str) -> set[int]:
    """Return buffer slot indices read before unconditional overwrite in ``simulation_step``."""
    tree = ast.parse(code)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if 'simulation_step' not in funcs:
        return set()
    reads, overwritten = set(), set()

    def scan(stmts, seen=frozenset(), conditional=False, int_env=None):
        int_env = dict(int_env or {})
        for st in stmts:
            if isinstance(st, (ast.Assign, ast.AugAssign)):
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    iv = _const_int_eval(st.value, int_env)
                    if iv is not None:
                        int_env[st.targets[0].id] = iv
                val = st.value if isinstance(st, ast.Assign) else st
                for sub in ast.walk(val):
                    sr = _slot(sub, int_env)
                    if sr is not None and sr not in overwritten:
                        reads.add(sr)
                    elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in funcs and sub.func.id not in seen:
                        scan(funcs[sub.func.id].body, seen | {sub.func.id}, conditional)
                if not conditional:
                    tgts = st.targets if isinstance(st, ast.Assign) else [st.target]
                    for tg in tgts:
                        sw = _slot(tg, int_env)
                        if sw is not None:
                            overwritten.add(sw)
            elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Name):
                for a in st.value.args:
                    for sub in ast.walk(a):
                        sr = _slot(sub, int_env)
                        if sr is not None and sr not in overwritten:
                            reads.add(sr)
                fn_name = st.value.func.id
                if fn_name in funcs and fn_name not in seen:
                    scan(funcs[fn_name].body, seen | {fn_name}, conditional)
            else:
                for sub in ast.walk(st):
                    sr = _slot(sub, int_env)
                    if sr is not None and sr not in overwritten:
                        reads.add(sr)
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in funcs and sub.func.id not in seen:
                        scan(funcs[sub.func.id].body, seen | {sub.func.id}, True)

    scan(funcs['simulation_step'].body, {'simulation_step'}, False)
    return reads


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
            sym = item.get('symbol') or key
            out_key = sym if sym not in output or output[sym]['value'] == item['si_value'] else key
            output[out_key] = dict(value=item['si_value'], unit=item.get('si_unit') or '1', source='structured', key=key, file=filename)
    for key, value in (metadata.get('extracted_constants') or blueprint.get('extracted_constants') or {}).items():
        param_entry = parameters.get(key) or {}
        param_sym = param_entry.get('symbol') or ''
        if key in parameters and param_entry.get('si_value') is not None:
            if param_entry['si_value'] != value:
                if param_entry.get('source') == 'user_prompt':
                    # Verified stored vault constant (e.g. SHBT rational 23/33) takes precedence
                    # over regex prose extraction (which can match `a + b = 1.0` as `b = 1.0`).
                    out_k = param_sym or key
                    if out_k in output:
                        output[out_k]['value'] = value
                    continue
                raise ReferenceError('VAULT_CONSTANT_MISMATCH', f'{param_sym or key}: stored {value}, structured {param_entry["si_value"]}')
            continue
        symbol = param_sym or next((name for name in output if _key(name)==_key(key)), key)
        if symbol not in output:
            output[symbol] = dict(value=value, unit='1', source='stored', key=key)
        elif output[symbol]['value'] != value:
            if output[symbol].get('source') == 'structured':
                output[symbol]['value'] = value
                continue
            raise ReferenceError('VAULT_CONSTANT_MISMATCH', f'{symbol}: stored {value}, structured {output[symbol]["value"]}')
    texts = _model_texts(blueprint,metadata)
    text = '\n'.join(str(value) for _,_,value in texts)
    unit_token = r'(?:1|[A-Za-zµμ]+)(?:\^[+-]?\d+)?(?:[/*](?:\([^)]*\)|[A-Za-zµμ]+(?:\^[+-]?\d+)?))*'
    for match in re.finditer(rf'(?<![\w/])([A-Za-z_]\w*)\s*=\s*({NUMBER})(?:\s+({unit_token}))?', text):
        prefix_ctx = text[max(0, match.start() - 4):match.start()].rstrip()
        if prefix_ctx.endswith(('+', '-', '*')):
            continue
        if text[match.end(2):].lstrip().startswith('/'):
            continue
        name, raw_val, unit = match[1], float(match[2]), match[3] or '1'
        value_si, canonical = normalize_to_si(raw_val, unit)
        try:
            unit_dimension(canonical)
            value = value_si
        except ValueError:
            value, canonical = normalize_to_si(raw_val, '1')
        symbol = next((old for old in output if _key(old)==_key(name)), name)
        old = output.get(symbol)
        if old is not None:
            if old.get('source') == 'text':
                continue
            if len(name) <= 2 and old.get('unit', '1') != canonical:
                continue
            if abs(old['value']-value) > max(abs(value)*1e-12, 1e-30):
                # Do not flag a sweep probe value mentioned in prose (e.g. "at R_0 = 3.0"
                # for a free sweep parameter with domain [2.5, 4.0]) as a constant mismatch.
                ihcg_dom = (blueprint.get('ihcg_evidence') or {}).get('domain') or {}
                dom_bounds = ihcg_dom.get(symbol) or ihcg_dom.get(name) or next(
                    (b for k, b in ihcg_dom.items() if _key(k) == _key(name) or _key(k).endswith(_key(name))),
                    None,
                )
                if (isinstance(dom_bounds, (list, tuple)) and len(dom_bounds) == 2
                        and float(dom_bounds[0]) < float(dom_bounds[1])
                        and float(dom_bounds[0]) <= value <= float(dom_bounds[1])):
                    continue
                raise ReferenceError('VAULT_CONSTANT_MISMATCH', f'{symbol}: stored {old["value"]}, text {value}')
        if old is None or old['unit'] in ('', '1') and unit != '1':
            filename, field = next((filename,field) for filename,field,value in texts if match[0] in str(value))
            output[symbol] = dict(value=value, unit=canonical, source='text', key=field, file=filename)
    return output


def _is_zero_expr(value):
    if isinstance(value, (int, float)):
        return value == 0
    return bool(getattr(value, 'is_Number', False) and value == 0)


class _Dual:
    """First-order tangent number v + dv * eps (eps^2 = 0) for fast O(1)-depth d/dt|_0."""
    __slots__ = ('val', 'deriv')

    def __init__(self, val, deriv=0):
        self.val = val
        self.deriv = deriv

    @staticmethod
    def lift(value, sp):
        if isinstance(value, _Dual):
            return value
        return _Dual(value, sp.Integer(0))

    @staticmethod
    def value_of(value):
        return value.val if isinstance(value, _Dual) else value

    def __add__(self, other):
        o = other if isinstance(other, _Dual) else _Dual(other, 0)
        if _is_zero_expr(self.deriv) and _is_zero_expr(o.deriv):
            return _Dual(self.val + o.val, 0)
        return _Dual(self.val + o.val, self.deriv + o.deriv)

    def __radd__(self, other):
        return _Dual(other, 0).__add__(self)

    def __sub__(self, other):
        o = other if isinstance(other, _Dual) else _Dual(other, 0)
        if _is_zero_expr(self.deriv) and _is_zero_expr(o.deriv):
            return _Dual(self.val - o.val, 0)
        return _Dual(self.val - o.val, self.deriv - o.deriv)

    def __rsub__(self, other):
        return _Dual(other, 0).__sub__(self)

    def __neg__(self):
        return _Dual(-self.val, 0 if _is_zero_expr(self.deriv) else -self.deriv)

    def __mul__(self, other):
        o = other if isinstance(other, _Dual) else _Dual(other, 0)
        sd_z = _is_zero_expr(self.deriv)
        od_z = _is_zero_expr(o.deriv)
        if sd_z and od_z:
            return _Dual(self.val * o.val, 0)
        sv_z = _is_zero_expr(self.val)
        ov_z = _is_zero_expr(o.val)
        t1 = 0 if (sv_z or od_z) else self.val * o.deriv
        t2 = 0 if (ov_z or sd_z) else o.val * self.deriv
        return _Dual(self.val * o.val, t1 + t2)

    def __rmul__(self, other):
        return _Dual(other, 0).__mul__(self)

    def __truediv__(self, other):
        o = other if isinstance(other, _Dual) else _Dual(other, 0)
        sd_z = _is_zero_expr(self.deriv)
        od_z = _is_zero_expr(o.deriv)
        if sd_z and od_z:
            return _Dual(self.val / o.val, 0)
        if od_z:
            return _Dual(self.val / o.val, self.deriv / o.val)
        num = (0 if sd_z else self.deriv * o.val) - (0 if _is_zero_expr(self.val) else self.val * o.deriv)
        return _Dual(self.val / o.val, num / (o.val * o.val))

    def __rtruediv__(self, other):
        return _Dual(other, 0).__truediv__(self)

    def __pow__(self, other, sp=None):
        o = other if isinstance(other, _Dual) else _Dual(other, 0)
        sd_z = _is_zero_expr(self.deriv)
        od_z = _is_zero_expr(o.deriv)
        p = self.val ** o.val
        if sd_z and od_z:
            return _Dual(p, 0)
        if od_z:
            return _Dual(p, o.val * (self.val ** (o.val - 1)) * self.deriv)
        raise ValueError('Dual exponent requires constant/non-dt exponent or explicit log')


def _unwrap_piecewise_expr(expr, sp):
    """Strip nested Piecewise inside relational conditions cleanly when SymPy hits NaN comparisons."""
    def _unwrap(node):
        if isinstance(node, sp.Piecewise) and node.args:
            return _unwrap(node.args[-1][0])
        return node
    try:
        return expr.replace(lambda e: isinstance(e, sp.Piecewise), _unwrap)
    except (TypeError, ValueError, AttributeError):
        return expr


def _safe_piecewise(tu, cond_val, eu, sp):
    if isinstance(cond_val, bool):
        return tu if cond_val else eu
    if cond_val is sp.S.true:
        return tu
    if cond_val is sp.S.false:
        return eu
    try:
        return sp.Piecewise((tu, cond_val), (eu, True), evaluate=False)
    except TypeError:
        clean_cond = _unwrap_piecewise_expr(cond_val, sp)
        try:
            return sp.Piecewise((tu, clean_cond), (eu, True), evaluate=False)
        except TypeError:
            return eu


def _safe_cmp_piecewise(tu, lhs, op_name, rhs, eu, sp):
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        py_ops = {
            'ge': lhs >= rhs,
            'le': lhs <= rhs,
            'gt': lhs > rhs,
            'lt': lhs < rhs,
        }
        return tu if py_ops[op_name] else eu
    ops = {'ge': sp.Ge, 'le': sp.Le, 'gt': sp.Gt, 'lt': sp.Lt}
    fn = ops[op_name]
    try:
        return _safe_piecewise(tu, fn(lhs, rhs, evaluate=False), eu, sp)
    except TypeError:
        try:
            return _safe_piecewise(
                tu,
                fn(_unwrap_piecewise_expr(lhs, sp), _unwrap_piecewise_expr(rhs, sp), evaluate=False),
                eu,
                sp,
            )
        except TypeError:
            return eu


def _safe_minmax(cls, *args):
    if all(isinstance(a, (int, float)) or getattr(a, 'is_Number', False) for a in args):
        return cls(*args)
    if len(args) == 2 and args[0] == args[1]:
        return args[0]
    return cls(*args, evaluate=False)


def _apply_dual_math(name, arguments, sp):
    has_dual = any(isinstance(a, _Dual) for a in arguments)
    if not has_dual:
        if name in {'_invar_sdiv', '_sdiv', 'sdiv', 'safe_div', 'div_eps'} and len(arguments) == 2:
            return arguments[0] / arguments[1]
        if name in {'logistic_sigmoid', 'sigmoid'} and len(arguments) == 1:
            return 1 / (1 + sp.exp(-arguments[0]))
        if name in {'clamp', 'clip'} and len(arguments) == 3:
            return _safe_minmax(sp.Min, _safe_minmax(sp.Max, arguments[0], arguments[1]), arguments[2])
        if name == 'min':
            return _safe_minmax(sp.Min, *arguments)
        if name == 'max':
            return _safe_minmax(sp.Max, *arguments)
        if name in {'abs', 'fabs'} and len(arguments) == 1:
            arg = arguments[0]
            if isinstance(arg, (int, float)) or getattr(arg, 'is_Number', False):
                return sp.Abs(arg)
            return sp.Abs(arg, evaluate=False)
        return getattr(sp, name)(*arguments)

    duals = [_Dual.lift(a, sp) for a in arguments]
    vals = [d.val for d in duals]
    derivs = [d.deriv for d in duals]
    if all(_is_zero_expr(d) for d in derivs):
        res = _apply_dual_math(name, vals, sp)
        return _Dual(res, sp.Integer(0))
    if name in {'_invar_sdiv', '_sdiv', 'sdiv', 'safe_div', 'div_eps'} and len(duals) == 2:
        return duals[0] / duals[1]
    if name in {'logistic_sigmoid', 'sigmoid'} and len(duals) == 1:
        sig = 1 / (1 + sp.exp(-vals[0]))
        return _Dual(sig, sig * (1 - sig) * derivs[0])
    if name in {'clamp', 'clip'} and len(duals) == 3:
        x, dx = vals[0], derivs[0]
        lo, hi = vals[1], vals[2]
        try:
            cond_clamp = sp.Or(x < lo, x > hi)
        except TypeError:
            xu = _unwrap_piecewise_expr(x, sp)
            cond_clamp = sp.Or(xu < _unwrap_piecewise_expr(lo, sp), xu > _unwrap_piecewise_expr(hi, sp))
        return _Dual(
            _safe_minmax(sp.Min, _safe_minmax(sp.Max, x, lo), hi),
            _safe_piecewise(sp.Integer(0), cond_clamp, dx, sp),
        )
    if name == 'exp' and len(duals) == 1:
        ev = sp.exp(vals[0])
        return _Dual(ev, ev * derivs[0])
    if name == 'log' and len(duals) == 1:
        return _Dual(sp.log(vals[0]), derivs[0] / vals[0])
    if name == 'log10' and len(duals) == 1:
        return _Dual(sp.log(vals[0], 10), derivs[0] / (vals[0] * sp.log(10)))
    if name == 'sqrt' and len(duals) == 1:
        sv = sp.sqrt(vals[0])
        return _Dual(sv, derivs[0] / (2 * sv))
    if name == 'sin' and len(duals) == 1:
        return _Dual(sp.sin(vals[0]), sp.cos(vals[0]) * derivs[0])
    if name == 'cos' and len(duals) == 1:
        return _Dual(sp.cos(vals[0]), -sp.sin(vals[0]) * derivs[0])
    if name == 'tan' and len(duals) == 1:
        tv = sp.tan(vals[0])
        return _Dual(tv, (1 + tv ** 2) * derivs[0])
    if name == 'tanh' and len(duals) == 1:
        tv = sp.tanh(vals[0])
        return _Dual(tv, (1 - tv ** 2) * derivs[0])
    if name == 'sinh' and len(duals) == 1:
        return _Dual(sp.sinh(vals[0]), sp.cosh(vals[0]) * derivs[0])
    if name == 'cosh' and len(duals) == 1:
        return _Dual(sp.cosh(vals[0]), sp.sinh(vals[0]) * derivs[0])
    if name in {'abs', 'fabs'} and len(duals) == 1:
        u, du = vals[0], derivs[0]
        abs_u = sp.Abs(u) if (isinstance(u, (int, float)) or getattr(u, 'is_Number', False)) else sp.Abs(u, evaluate=False)
        return _Dual(abs_u, _safe_cmp_piecewise(du, u, 'ge', 0, -du, sp))
    if name == 'min' and len(duals) >= 2:
        cur = duals[0]
        for nxt in duals[1:]:
            u, du = cur.val, cur.deriv
            v, dv = nxt.val, nxt.deriv
            m_val = _safe_minmax(sp.Min, u, v)
            m_der = du if du == dv else _safe_cmp_piecewise(du, u, 'le', v, dv, sp)
            cur = _Dual(m_val, m_der)
        return cur
    if name == 'max' and len(duals) >= 2:
        cur = duals[0]
        for nxt in duals[1:]:
            u, du = cur.val, cur.deriv
            v, dv = nxt.val, nxt.deriv
            m_val = _safe_minmax(sp.Max, u, v)
            m_der = du if du == dv else _safe_cmp_piecewise(du, u, 'ge', v, dv, sp)
            cur = _Dual(m_val, m_der)
        return cur
    raise ValueError(f'Unsupported dual function: {name}')


def _merge_conditional(cond_val, then_val, else_val, sp):
    if then_val == else_val:
        return then_val
    if isinstance(then_val, _Dual) or isinstance(else_val, _Dual):
        tu, tdu = _Dual.lift(then_val, sp).val, _Dual.lift(then_val, sp).deriv
        eu, edu = _Dual.lift(else_val, sp).val, _Dual.lift(else_val, sp).deriv
        mv = tu if tu == eu else _safe_piecewise(tu, cond_val, eu, sp)
        md = tdu if tdu == edu else _safe_piecewise(tdu, cond_val, edu, sp)
        return _Dual(mv, md)
    return _safe_piecewise(then_val, cond_val, else_val, sp)


class _SymbolicKernel:
    def __init__(self, code, slots):
        import sympy as sp
        self.sp = sp
        self.functions = {node.name: node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
        self.slots = {index: sp.Symbol(name) for name, index in slots.items()}
        self.initial_slots = {}
        self.globals = {}
        self.block([node for node in ast.parse(code).body if isinstance(node, (ast.Assign, ast.AnnAssign))], self.globals)

    def _resolve_slot(self, node, environment, depth):
        static = _slot(node)
        if static is not None:
            return static
        if isinstance(node, ast.Subscript) and (
            (isinstance(node.value, ast.Attribute) and node.value.attr == 'data')
            or (isinstance(node.value, ast.Name) and node.value.id in ('buf', 'data', 'state') and not isinstance(environment.get(node.value.id), (list, tuple)))
        ):
            try:
                idx_val = _Dual.value_of(self.expression(node.slice, environment, depth))
                if (getattr(idx_val, 'is_Integer', False) or type(idx_val) is int) and int(idx_val) >= 0:
                    return int(idx_val)
            except ValueError:
                return None
        return None

    def expression(self, node, environment, depth=0):
        sp = self.sp
        if depth > 24:
            raise ValueError('Symbolic call depth exceeded')
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and isinstance(environment.get(node.value.id), (list, tuple)):
            idx_val = node.slice.value if isinstance(node.slice, ast.Constant) else _Dual.value_of(self.expression(node.slice, environment, depth))
            if getattr(idx_val, 'is_Integer', False) or type(idx_val) is int:
                return environment[node.value.id][int(idx_val)]
        dyn_slot = self._resolve_slot(node, environment, depth)
        if dyn_slot is not None:
            key = ('slot', dyn_slot)
            if key in environment:
                return environment[key]
            if key in self.initial_slots:
                return self.initial_slots[key]
            return self.slots.get(dyn_slot, sp.Symbol(f'_slot_{dyn_slot}'))
        if isinstance(node, ast.Subscript):
            idx_val = node.slice.value if isinstance(node.slice, ast.Constant) else _Dual.value_of(self.expression(node.slice, environment, depth))
            if getattr(idx_val, 'is_Integer', False) or type(idx_val) is int:
                return self.expression(node.value, environment, depth)[int(idx_val)]
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return sp.true if node.value else sp.false
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'math':
            if node.attr == 'pi':
                return sp.pi
            if node.attr == 'e':
                return sp.E
            if node.attr == 'tau':
                return 2 * sp.pi
        if isinstance(node, ast.Name):
            return environment.get(node.id, sp.Symbol(node.id))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            val = self.expression(node.operand, environment, depth)
            return -val if isinstance(node.op, ast.USub) else val
        if isinstance(node, ast.Compare):
            left = _Dual.value_of(self.expression(node.left, environment, depth))
            comparisons = []
            relations = {ast.Gt: sp.Gt, ast.GtE: sp.Ge, ast.Lt: sp.Lt, ast.LtE: sp.Le, ast.Eq: sp.Eq, ast.NotEq: sp.Ne}
            for operation, right_node in zip(node.ops, node.comparators):
                right = _Dual.value_of(self.expression(right_node, environment, depth))
                rel_fn = relations[type(operation)]
                try:
                    cmp_expr = rel_fn(left, right)
                except TypeError:
                    cmp_expr = rel_fn(_unwrap_piecewise_expr(left, sp), _unwrap_piecewise_expr(right, sp), evaluate=False)
                if isinstance(cmp_expr, sp.Piecewise):
                    cmp_expr = rel_fn(_unwrap_piecewise_expr(left, sp), _unwrap_piecewise_expr(right, sp), evaluate=False)
                comparisons.append(cmp_expr)
                left = right
            return comparisons[0] if len(comparisons) == 1 else sp.And(*comparisons)
        if isinstance(node, ast.BoolOp):
            vals = []
            for value in node.values:
                v = self.expression(value, environment, depth)
                if isinstance(v, sp.Piecewise):
                    v = _unwrap_piecewise_expr(v, sp)
                vals.append(v)
            return (sp.And if isinstance(node.op, ast.And) else sp.Or)(*vals)
        if isinstance(node, ast.IfExp):
            cond_val = _Dual.value_of(self.expression(node.test, environment, depth))
            if cond_val in (sp.true, True):
                return self.expression(node.body, environment, depth)
            if cond_val in (sp.false, False):
                return self.expression(node.orelse, environment, depth)
            return _merge_conditional(
                cond_val,
                self.expression(node.body, environment, depth),
                self.expression(node.orelse, environment, depth),
                sp,
            )
        if isinstance(node, ast.BinOp):
            left, right = self.expression(node.left, environment, depth), self.expression(node.right, environment, depth)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult):
                rv = _Dual.value_of(right)
                lv = _Dual.value_of(left)
                if isinstance(left, list) and getattr(rv, 'is_Integer', False) and 0 <= int(rv) <= 128:
                    return list(left) * int(rv)
                if isinstance(right, list) and getattr(lv, 'is_Integer', False) and 0 <= int(lv) <= 128:
                    return list(right) * int(lv)
                return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow):
                if isinstance(left, _Dual) or isinstance(right, _Dual):
                    dl, dr = _Dual.lift(left, sp), _Dual.lift(right, sp)
                    if _is_zero_expr(dr.deriv):
                        return dl.__pow__(dr, sp)
                    return _Dual(dl.val ** dr.val, (dl.val ** dr.val) * (dr.val * dl.deriv / dl.val + sp.log(dl.val) * dr.deriv))
                return left ** right
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'math' else ''
            if name == 'len' and len(node.args) == 1:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Attribute) and arg0.attr == 'data':
                    return sp.Integer(max(max(self.slots.keys(), default=0) + 1, 64))
                try:
                    val0 = self.expression(arg0, environment, depth)
                    if isinstance(val0, (list, tuple)):
                        return sp.Integer(len(val0))
                except ValueError:
                    pass
                return sp.Integer(max(max(self.slots.keys(), default=0) + 1, 64))
            if name in {'isfinite', 'isnan', 'isinf'} and len(node.args) == 1:
                return sp.true if name == 'isfinite' else sp.false
            arguments = [self.expression(arg, environment, depth) for arg in node.args]
            if name in {'float', 'int'} and len(arguments) == 1:
                return arguments[0]
            if name == 'pow' and len(arguments) == 2:
                left, right = arguments
                if isinstance(left, _Dual) or isinstance(right, _Dual):
                    dl, dr = _Dual.lift(left, sp), _Dual.lift(right, sp)
                    if _is_zero_expr(dr.deriv):
                        return dl.__pow__(dr, sp)
                    return _Dual(dl.val ** dr.val, (dl.val ** dr.val) * (dr.val * dl.deriv / dl.val + sp.log(dl.val) * dr.deriv))
                return left ** right
            if name in {'_invar_sdiv', '_sdiv', 'sdiv', 'safe_div', 'div_eps', 'logistic_sigmoid', 'sigmoid', 'clamp', 'clip', 'sin', 'cos', 'tan', 'tanh', 'sinh', 'cosh', 'exp', 'log', 'log10', 'sqrt', 'abs', 'fabs', 'min', 'max'}:
                return _apply_dual_math(name, arguments, sp)
            function = environment.get(('function', name), self.functions.get(name))
            if function is not None:
                local = dict(environment)
                local.update({arg.arg: value for arg, value in zip(function.args.args, arguments)})
                result = self.block(function.body, local, depth + 1)
                environment.update({key: value for key, value in local.items() if isinstance(key, tuple) and key[0] == 'slot'})
                return result
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.expression(item, environment, depth) for item in node.elts]
        raise ValueError('Unsupported symbolic expression: ' + ast.unparse(node))

    def block(self, body, environment, depth=0):
        if depth > 24:
            raise ValueError('Symbolic call depth exceeded')
        for idx_stmt, node in enumerate(body):
            if isinstance(node, ast.FunctionDef):
                environment[('function', node.name)] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                try:
                    value = self.expression(node.value, environment, depth)
                except ValueError:
                    # NF-05: Fail closed if the assignment contains an unsupported Call expression
                    if any(isinstance(sub, ast.Call) for sub in ast.walk(node.value)):
                        raise ValueError(f"SYMBOLIC_EXPRESSION_UNSUPPORTED: {ast.unparse(node.value)}")
                    continue
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    def _assign_target(tgt, val):
                        dyn_s = self._resolve_slot(tgt, environment, depth)
                        if isinstance(tgt, ast.Name):
                            environment[tgt.id] = val
                        elif dyn_s is not None:
                            environment[('slot', dyn_s)] = val
                        elif isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(val, (list, tuple)) and len(tgt.elts) == len(val):
                            for sub_t, sub_v in zip(tgt.elts, val):
                                _assign_target(sub_t, sub_v)
                        elif isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and isinstance(environment.get(tgt.value.id), list):
                            try:
                                idx_v = tgt.slice.value if isinstance(tgt.slice, ast.Constant) else _Dual.value_of(self.expression(tgt.slice, environment, depth))
                                if getattr(idx_v, 'is_Integer', False) or type(idx_v) is int:
                                    buf_list = list(environment[tgt.value.id])
                                    buf_list[int(idx_v)] = val
                                    environment[tgt.value.id] = buf_list
                            except (ValueError, IndexError):
                                pass
                    _assign_target(target, value)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                items = None
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                    try:
                        r_args = [int(_Dual.value_of(self.expression(a, environment, depth))) for a in node.iter.args]
                        if 1 <= len(r_args) <= 3:
                            rng = range(*r_args)
                            if len(rng) <= 64:
                                items = list(rng)
                    except (ValueError, TypeError):
                        items = None
                else:
                    try:
                        seq = self.expression(node.iter, environment, depth)
                        if isinstance(seq, (list, tuple)) and len(seq) <= 64:
                            items = list(seq)
                    except ValueError:
                        items = None
                if items is not None:
                    for item in items:
                        environment[node.target.id] = self.sp.Integer(item) if isinstance(item, int) else item
                        res = self.block(node.body, environment, depth + 1)
                        if res is not None:
                            return res
            elif isinstance(node, ast.AugAssign):
                value = self.expression(ast.BinOp(left=node.target, op=node.op, right=node.value), environment, depth)
                dyn_s = self._resolve_slot(node.target, environment, depth)
                if isinstance(node.target, ast.Name): environment[node.target.id] = value
                elif dyn_s is not None: environment[('slot', dyn_s)] = value
            elif isinstance(node, ast.If):
                # Skip single-branch zero-floor / finiteness guards (e.g. `if buf.data[0] < 0.0: buf.data[0] = 0.0`,
                # `if not math.isfinite(buf.data[0]): buf.data[0] = 0.0`) so symbolic ODE recovery captures the
                # smooth interior evolution law without 2^N Piecewise blowup, while preserving non-zero clamps.
                if (
                    not node.orelse
                    and len(node.body) == 1
                    and isinstance(node.body[0], ast.Assign)
                    and len(node.body[0].targets) == 1
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, (int, float))
                    and abs(float(node.body[0].value.value)) <= 1e-12
                ):
                    tgt_s = ast.unparse(node.body[0].targets[0])
                    if tgt_s.startswith('buf.data[') and tgt_s in ast.unparse(node.test):
                        continue
                # Take the interior positive-denominator branch for `if pool > eps:` guards (0 < eps <= 1e-7)
                if (
                    not node.orelse
                    and isinstance(node.test, ast.Compare)
                    and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], (ast.Gt, ast.GtE))
                    and isinstance(node.test.comparators[0], ast.Constant)
                    and isinstance(node.test.comparators[0].value, (int, float))
                    and 0.0 < float(node.test.comparators[0].value) <= 1e-7
                ):
                    res = self.block(node.body, environment, depth + 1)
                    if res is not None:
                        return res
                    continue
                try:
                    cond_val = _Dual.value_of(self.expression(node.test, environment, depth))
                except ValueError:
                    cond_val = None
                if cond_val in (self.sp.true, True):
                    res = self.block(node.body, environment, depth + 1)
                    if res is not None:
                        return res
                elif cond_val in (self.sp.false, False):
                    res = self.block(node.orelse, environment, depth + 1)
                    if res is not None:
                        return res
                elif cond_val is not None:
                    then_env = dict(environment)
                    then_res = self.block(node.body, then_env, depth + 1)
                    if then_res is not None:
                        else_body = node.orelse if node.orelse else body[idx_stmt + 1:]
                        else_res = self.block(else_body, dict(environment), depth + 1) if else_body else None
                        if else_res is not None:
                            return _merge_conditional(cond_val, then_res, else_res, self.sp)
                    elif not node.orelse:
                        for k, tv in then_env.items():
                            if isinstance(k, tuple) and k[0] != 'slot':
                                continue
                            if k not in environment:
                                environment[k] = tv
                            elif environment[k] != tv and not isinstance(tv, list):
                                environment[k] = _merge_conditional(cond_val, tv, environment[k], self.sp)
                        continue
                    else:
                        else_env = dict(environment)
                        else_res = self.block(node.orelse, else_env, depth + 1)
                        if else_res is None:
                            for k in (set(then_env) | set(else_env)):
                                if isinstance(k, tuple) and k[0] != 'slot':
                                    continue
                                tv = then_env.get(k, environment.get(k))
                                ev = else_env.get(k, environment.get(k))
                                if tv is not None and ev is not None and not isinstance(tv, list) and not isinstance(ev, list):
                                    environment[k] = _merge_conditional(cond_val, tv, ev, self.sp)
                                elif tv is not None and k not in environment:
                                    environment[k] = tv
                            continue
            elif isinstance(node, ast.Return) and node.value is not None:
                return self.expression(node.value, environment, depth)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and (call.func.id in self.functions or ('function', call.func.id) in environment):
                    self.expression(call, environment, depth + 1)
        return None

    def updates(self, states):
        driver = self.functions.get('simulation_step')
        if driver is None:
            return {}
        environment = {**self.globals, 'dt': self.sp.Symbol('_dt')}
        if len(driver.args.args) > 1:
            environment[driver.args.args[1].arg] = environment['dt']
        self.block(driver.body, environment)
        return {name: environment[('slot', index)] for index, symbol in self.slots.items()
                if (name := str(symbol)) in states and ('slot', index) in environment}

    def zero_dt_profile(self, states):
        """Return (rhs_dict, zero_dt_updates_dict, written_slots) via O(steps) dual evaluation without _dt blowup."""
        sp = self.sp
        driver = self.functions.get('simulation_step')
        if driver is None:
            return {}, {}, set()
        self.initial_slots = {('slot', index): _Dual(symbol, sp.Integer(0)) for index, symbol in self.slots.items()}
        environment = {**self.globals, 'dt': _Dual(sp.Integer(0), sp.Integer(1))}
        if len(driver.args.args) > 1:
            environment[driver.args.args[1].arg] = environment['dt']
        try:
            self.block(driver.body, environment)
        finally:
            self.initial_slots = {}
        rhs, vals, written = {}, {}, set()
        for index, symbol in self.slots.items():
            key = ('slot', index)
            if key not in environment:
                continue
            written.add(str(symbol))
            if str(symbol) not in states:
                continue
            v = environment[key]
            dval = v.deriv if isinstance(v, _Dual) else sp.Integer(0)
            rhs[str(symbol)] = sp.Integer(0) if _is_zero_expr(dval) else dval
            vals[str(symbol)] = _Dual.value_of(v)
        return rhs, vals, written

    def derivatives(self, states):
        sp = self.sp
        if 'simulation_step' in self.functions:
            driver = self.functions['simulation_step']
            self.initial_slots = {('slot', index): _Dual(symbol, sp.Integer(0)) for index, symbol in self.slots.items()}
            environment = {**self.globals, 'dt': _Dual(sp.Integer(0), sp.Integer(1))}
            if len(driver.args.args) > 1:
                environment[driver.args.args[1].arg] = environment['dt']
            try:
                self.block(driver.body, environment)
                # Detect clock slot(s) incremented by dt (buf.data[k] = buf.data[k] + dt) not in states
                clock_symbols = set()
                for key, val in environment.items():
                    if not (isinstance(key, tuple) and key[0] == 'slot'):
                        continue
                    index = key[1]
                    symbol = self.slots.get(index, sp.Symbol(f'_slot_{index}'))
                    if str(symbol) not in states and isinstance(val, _Dual) and val.val == symbol and val.deriv == 1:
                        clock_symbols.add(index)
                if any(str(self.slots.get(idx)) != 't' for idx in clock_symbols):
                    self.initial_slots = {
                        ('slot', index): _Dual(sp.Symbol('t') if index in clock_symbols else symbol, sp.Integer(0))
                        for index, symbol in self.slots.items()
                    }
                    for index in clock_symbols:
                        self.initial_slots[('slot', index)] = _Dual(sp.Symbol('t'), sp.Integer(0))
                    environment = {**self.globals, 'dt': _Dual(sp.Integer(0), sp.Integer(1))}
                    if len(driver.args.args) > 1:
                        environment[driver.args.args[1].arg] = environment['dt']
                    self.block(driver.body, environment)
            finally:
                self.initial_slots = {}
            out = {}
            for index, symbol in self.slots.items():
                name = str(symbol)
                if name in states and ('slot', index) in environment:
                    val = environment[('slot', index)]
                    dval = val.deriv if isinstance(val, _Dual) else sp.Integer(0)
                    out[name] = sp.Integer(0) if _is_zero_expr(dval) else dval
            return out, 'simulation_step'
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
        lhs_raw = str(item.get('lhs', '') or '')
        rhs_raw = str(item.get('rhs_latex', '') or '')
        text = f"{lhs_raw} = {rhs_raw}"
        if item.get('role') == 'EVOLUTION_LAW' and not re.search(r'\bd\w+/dt|\\dot|\\frac\{d', lhs_raw):
            text = f"d{lhs_raw}/dt = {rhs_raw}"
        if item.get('role') == 'EVOLUTION_LAW' or re.search(r'\bd\w+/dt|\\dot|\\frac\{d', lhs_raw):
            try:
                left, _, right = equation_parts(text)
            except ValueError:
                continue
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
        from .slot_schema import ROLE_DIAGNOSTIC, ROLE_CONTROL_PARAM, ROLE_TIME_STEP, classify_slot_roles
        from .scientific_integrity import output_dataflow
        tree = ast.parse(code)
        flow = output_dataflow(tree)
        candidates = [name for name in (blueprint.get('variable_map') or {})]
        slots = symbol_slots(candidates, blueprint, code)
        assigned = {_slot(target) for node in ast.walk(tree) if isinstance(node, ast.Assign) for target in node.targets if _slot(target) is not None}
        # Flow-sensitive entry-read filter: a genuine dynamical state MUST read its own entry slot
        # before being unconditionally overwritten inside simulation_step; algebraic diagnostics
        # / instantaneous derivatives (buf.data[k] = rhs(...) written before read) are not ODE states.
        _entry_reads = _entry_read_slots(code)
        shared_roles = classify_slot_roles(
            blueprint.get('variable_map'), tree, blueprint.get('contracts') or (),
            physical_slots=flow.get('physical'),
        )
        states = [
            name for name in candidates
            if slots.get(name) in assigned
            and slots.get(name) not in flow.get('clocks', set())
            and (not _entry_reads or slots.get(name) in _entry_reads)
            and shared_roles.get(slots.get(name)) not in (ROLE_DIAGNOSTIC, ROLE_CONTROL_PARAM, ROLE_TIME_STEP)
        ]
        if not states:
            states = [name for name in candidates if slots.get(name) in assigned and slots.get(name) not in flow.get('clocks', set())]
        if not states:
            states = candidates
    parameters = parameter_bindings(blueprint, metadata)
    contracts = blueprint.get('contracts') or metadata.get('contracts_summary') or []
    initial = (contracts[0] if contracts else {}).get('initial_state') or []
    tree_ast = ast.parse(code)
    written_slots, readonly = _function_slots_rw(tree_ast)
    slots = symbol_slots(states, blueprint, code)
    param_slots = symbol_slots(list(parameters), blueprint, code)
    for pname, pslot in param_slots.items():
        if pname not in slots and pslot not in written_slots and pslot not in slots.values():
            slots[pname] = pslot
    for name,item in parameters.items():
        if name not in slots:
            candidates = [index for index in readonly-set(slots.values()) if index<len(initial) and initial[index]==item['value']]
            if len(candidates)==1:
                slots[name] = candidates[0]
    for vname, vslot in (blueprint.get('variable_map') or {}).items():
        try:
            idx = int(vslot)
        except (TypeError, ValueError):
            continue
        if idx in readonly and vname not in slots and idx not in slots.values():
            slots[vname] = idx
            parameters.setdefault(
                vname,
                dict(
                    value=float(initial[idx]) if idx < len(initial) else 0.0,
                    unit='1',
                    source='variable_map',
                    key=vname,
                ),
            )
    missing_slots = [state for state in states if state not in slots]
    if missing_slots:
        raise ReferenceError('VAULT_MISSING_EVOLUTION_LAW', 'Cannot resolve state slots: '+', '.join(missing_slots), {'missing_states': missing_slots})
    for vname, vslot in (blueprint.get('variable_map') or {}).items():
        if str(vname).lower() in ('t', 'time', 'time_s', 't_s') and 't' not in slots:
            try:
                t_idx = int(vslot)
                if t_idx not in slots.values():
                    slots['t'] = t_idx
            except (TypeError, ValueError):
                pass
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
    ihcg_domain = (blueprint.get('ihcg_evidence') or {}).get('domain') or {}
    ihcg_vmap_inv = {idx: sym for sym, idx in (blueprint.get('variable_map') or {}).items()}
    for name, item in parameters.items():
        if name in slots and slots[name] < len(initial):
            iv_f = float(initial[slots[name]])
            pv_f = float(item['value'])
            if abs(iv_f - pv_f) > 1e-6 * max(1.0, abs(iv_f), abs(pv_f)):
                slot_sym = ihcg_vmap_inv.get(slots[name], name)
                bounds = ihcg_domain.get(slot_sym) or ihcg_domain.get(name)
                if (isinstance(bounds, (list, tuple)) and len(bounds) == 2
                        and float(bounds[0]) <= iv_f <= float(bounds[1])):
                    item = {**item, 'value': iv_f}
                    parameters[name] = item
                    continue
                raise ReferenceError('VAULT_CONSTANT_MISMATCH', f'{name}: initial slot {initial[slots[name]]} != {item["value"]}')
    environment = {name: item['value'] for name, item in parameters.items()}
    environment.update({name: initial[slots[name]] if slots[name]<len(initial) else 1. for name in states})
    comparisons = []
    for factor in (1., .75, 1.125):
        point = {**environment, **{name: value*factor if value else .001*factor for name, value in environment.items() if name in states}, 't': 0.}
        for state in text_laws.keys() & symbolic.keys():
            try:
                actual, expected = float(symbolic[state].subs(point)), float(text_laws[state].subs(point))
            except (TypeError, ValueError):
                continue
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
    math_shim = [
        {
            're': lambda z: float(getattr(z, 'real', z)),
            'im': lambda z: float(getattr(z, 'imag', 0.0)),
            'Abs': abs,
            'Min': min,
            'Max': max,
            'sign': lambda x: (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)),
        },
        'math',
    ]
    def _safe_lambdify(args, exprs, *, cse=True):
        arg_set = set(args)
        alias_subs = {}
        for e in exprs:
            for fs in getattr(e, 'free_symbols', ()):
                if fs not in arg_set and fs not in alias_subs:
                    matches = [a for a in args if a.name.endswith(f'_{fs.name}') or _key(a.name) == _key(fs.name)]
                    if len(matches) == 1:
                        alias_subs[fs] = matches[0]
        cleaned = []
        for e in exprs:
            if alias_subs and hasattr(e, 'xreplace'):
                e = e.xreplace(alias_subs)
            if hasattr(e, 'has') and e.has(sp.ITE):
                e = e.replace(sp.ITE, lambda c, t, f: sp.Or(sp.And(c, t), sp.And(sp.Not(c), f)))
            cleaned.append(e)
        try:
            return sp.lambdify(args, cleaned, modules=math_shim, dummify=True, cse=cse)
        except TypeError:
            def _eval_fallback(*vals):
                subs = dict(zip(args, vals))
                return [float(e.xreplace(subs).evalf()) for e in cleaned]
            return _eval_fallback

    _use_cse_num = all(sp.count_ops(e) <= 5000 for e in expressions)
    numerical_rhs = _safe_lambdify(arguments, expressions, cse=_use_cse_num)
    symbolic = recovered['symbolic']
    if set(names)-set(symbolic):
        raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', 'Cannot independently recover the executed RHS for '+', '.join(set(names)-set(symbolic)))
    sym_expressions = [symbolic[name] for name in names]
    if sym_expressions == expressions:
        actual_rhs = numerical_rhs
        rhs_scales = None
    else:
        rhs_scales = _safe_lambdify(
            arguments,
            [
                sp.Abs(expression, evaluate=False)
                if (expression.has(sp.Piecewise) or sp.count_ops(expression) > 150)
                else sum(sp.Abs(term) for term in sp.Add.make_args(sp.expand(expression)))
                for expression in expressions
            ],
            cse=_use_cse_num,
        )
        _use_cse_sym = all(sp.count_ops(e) <= 5000 for e in sym_expressions)
        actual_rhs = _safe_lambdify(arguments, sym_expressions, cse=_use_cse_sym)
    first = recovered['contracts'][0]
    param_names = list(parameters.keys())
    _direct_unstripped = False
    try:
        from types import SimpleNamespace

        _stripped_finite_guard = False

        class _StripFiniteGuards(ast.NodeTransformer):
            def visit_If(self, node):
                nonlocal _stripped_finite_guard
                node = self.generic_visit(node)
                if any(
                    (isinstance(sub, ast.Name) and sub.id == 'isfinite')
                    or (isinstance(sub, ast.Attribute) and sub.attr == 'isfinite')
                    for sub in ast.walk(node.test)
                ):
                    _stripped_finite_guard = True
                    return None
                return node

        probe_tree = _StripFiniteGuards().visit(ast.parse(code))
        ast.fix_missing_locations(probe_tree)
        direct_ns = _safe_namespace(ast.unparse(probe_tree))
        _direct_unstripped = not _stripped_finite_guard
        direct_ns.setdefault('_invar_sdiv', lambda a, b: a / b)
        direct_ns.setdefault('safe_div', lambda a, b: a / b)
        direct_ns.setdefault('div_eps', lambda a, b: a / b)
        direct_ns.setdefault('logistic_sigmoid', lambda x: 1.0 / (1.0 + math.exp(-x)))
        direct_ns.setdefault('sigmoid', lambda x: 1.0 / (1.0 + math.exp(-x)))
        direct_ns.setdefault('clamp', lambda x, lo, hi: min(max(x, lo), hi))
        direct_ns.setdefault('clip', lambda x, lo, hi: min(max(x, lo), hi))
        step_fn = direct_ns.get('simulation_step')
        if not callable(step_fn):
            raise ReferenceError('VAULT_REFERENCE_UNSUPPORTED', 'Missing simulation_step(buf, dt)')
        slot_indices = [
            node.slice.value
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and type(node.slice.value) is int
            and node.slice.value >= 0
        ]
        buf_size = max(
            len(first['initial_state']),
            max(recovered['slots'].values(), default=0) + 1,
            max(slot_indices, default=0) + 1,
        )

        def numerical_update(*args):
            st_vals = args[:len(names)]
            tr_vals = args[len(names):len(names) + len(param_names)]
            dt_val = float(args[-1])
            buf_data = list(first['initial_state']) + [0.0] * max(0, buf_size - len(first['initial_state']))
            for p_name, p_val in zip(param_names, tr_vals):
                if p_name in recovered['slots']:
                    buf_data[recovered['slots'][p_name]] = float(p_val)
            for s_name, s_val in zip(names, st_vals):
                buf_data[recovered['slots'][s_name]] = float(s_val)
            buf = SimpleNamespace(data=buf_data, size=buf_size)
            step_fn(buf, dt_val)
            return [buf.data[recovered['slots'][s_name]] for s_name in names]
    except (ReferenceError, SyntaxError, ValueError, TypeError):
        updates = _SymbolicKernel(code, recovered['slots']).updates(names)
        numerical_update = sp.lambdify([*arguments, sp.Symbol('_dt')], [updates[name] for name in names], modules=math_shim, dummify=True, cse=True)
    scores = {scheme:0. for scheme in ('euler','midpoint','heun','rk4')}

    def check_rhs(values, parameters, t, step, contract_id):
        try:
            expected = list(map(float,numerical_rhs(*values,*parameters,t)))
            if rhs_scales is None:
                for name, b in zip(names, expected):
                    if not math.isfinite(b):
                        raise ReferenceError('VAULT_REFERENCE_MISMATCH', f'{name}: non-finite RHS', dict(state=name, step=step, contract_id=contract_id, actual=None, expected=None, absolute_error=None, tolerance=None, comparison='rhs'))
                return expected
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

    def _dyn_mask(st_vals, tr_vals):
        try:
            phi0 = numerical_update(*st_vals, *tr_vals, 0., 0.0)
            m = [abs(float(a) - float(b)) <= 1e-12 for a, b in zip(phi0, st_vals)]
            return m if any(m) else [True] * len(st_vals)
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            return [True] * len(st_vals)

    for factor in (1.,.75,1.125):
        values = [first['initial_state'][recovered['slots'][name]]*factor or .001*factor for name in names]
        params = [first['initial_state'][recovered['slots'][name]] if name in recovered['slots'] else item['value'] for name,item in parameters.items()]
        for multiple in (1.,.5,2.):
            dt = first['dt']*multiple
            check_rhs(values,params,0.,0,first['contract_id'])
            for trial in (params,[p*1.125 if p else .001 for p in params]):
                dmask = _dyn_mask(values, trial)
                actual = numerical_update(*values,*trial,0.,dt)
                for scheme in scores:
                    try:
                        expected = integrate_rhs(lambda y,p,t:actual_rhs(*y,*p,t),values,trial,dt,0.,scheme)
                        error = max(
                            abs(float(a)-float(b))/(1e-12+1e-11*abs(float(b)))
                            for a, b, ok in zip(actual, expected, dmask) if ok
                        )
                    except (ValueError,OverflowError,ZeroDivisionError,TypeError):
                        error = math.inf
                    scores[scheme] = max(scores[scheme],error if math.isfinite(error) else math.inf)
            probes.append((values,params,dt,_dyn_mask(values, params)))
    min_err = min(scores.values())
    tied = [name for name in ('rk4', 'heun', 'midpoint', 'euler') if scores[name] <= max(min_err + 1e-12, 1e-9)]
    if len(tied) > 1:
        code_low = code or ''
        if 'rk4' in tied and re.search(r'(?:/\s*6\b|\bk4\b)', code_low):
            scheme = 'rk4'
        elif 'heun' in tied and re.search(r'(?:/\s*2\b|\bk2\b)', code_low):
            scheme = 'heun'
        elif 'midpoint' in tied and re.search(r'(?:0\.5\s*\*|/\s*2\b)', code_low):
            scheme = 'midpoint'
        else:
            scheme = min(scores, key=scores.get)
    else:
        scheme = min(scores, key=scores.get)
    if scores[scheme]>1. and any(not ok for _, _, _, dm in probes for ok in dm):
        scheme = 'semi_implicit'
    elif scores[scheme]>1.:
        # Check mathematical 1st-order consistency for Patankar / linear semi-implicit steppers
        # (denominator involving dt such as 1 + dt * D or multi-stage operator split):
        # lim_{d->0} ||Phi_d(y) - (y + d*f(y))|| / d == 0 with O(d^2) halving decay.
        dt_probe = min(float(first['dt']) if first['dt'] > 0 else 1e-3, 1e-4)
        semi_ok = dt_probe > 0.0
        for factor in (1., .75, .875):
            if not semi_ok:
                break
            p_vals = [first['initial_state'][recovered['slots'][name]]*factor or .001*factor for name in names]
            p_pars = [first['initial_state'][recovered['slots'][name]] if name in recovered['slots'] else item['value'] for name,item in parameters.items()]
            for tr in (p_pars, [p*1.125 if p else .001 for p in p_pars]):
                try:
                    dmask = _dyn_mask(p_vals, tr)
                    f_y = list(map(float, actual_rhs(*p_vals, *tr, 0.)))
                    u1 = numerical_update(*p_vals, *tr, 0., dt_probe)
                    u2 = numerical_update(*p_vals, *tr, 0., 0.5 * dt_probe)
                    e1 = max(
                        abs(float(u) - (float(y) + dt_probe * float(fy)))
                        / (dt_probe * (1.0 + abs(float(fy))) + 1e-12)
                        for y, fy, u, ok in zip(p_vals, f_y, u1, dmask) if ok
                    )
                    e2 = max(
                        abs(float(u) - (float(y) + 0.5 * dt_probe * float(fy)))
                        / (0.5 * dt_probe * (1.0 + abs(float(fy))) + 1e-12)
                        for y, fy, u, ok in zip(p_vals, f_y, u2, dmask) if ok
                    )
                    if not (math.isfinite(e1) and math.isfinite(e2) and e1 <= 0.05 and e2 <= 0.65 * e1 + 1e-9):
                        semi_ok = False
                        break
                except (ValueError, OverflowError, ZeroDivisionError, TypeError):
                    semi_ok = False
                    break
        if semi_ok:
            scheme = 'semi_implicit'
        else:
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
        if scheme == 'semi_implicit':
            # Verify trajectory finiteness, RHS consistency, and bounded O(dt^2) step consistency
            # against two half-steps at sampled checkpoints rather than explicit RK4 parity.
            from .scientific_integrity import output_dataflow
            _semi_flow = output_dataflow(code)
            _semi_clocks = set(_semi_flow.get('clocks') or ())
            check_rhs(state, parameter_values, 0., 1, contract['contract_id'])
            sample_indices = {1, max(1, contract['steps'] // 2), contract['steps']}
            for step in range(1, contract['steps'] + 1):
                row = result['samples'][str(step)]
                if any(not math.isfinite(v) for v in row):
                    bad = next(i for i, v in enumerate(row) if not math.isfinite(v))
                    raise ReferenceError('VAULT_REFERENCE_MISMATCH', f'{names[bad]} non-finite at step {step}', dict(state=names[bad], step=step, contract_id=contract['contract_id']))
                if step in sample_indices:
                    t = (step - 1) * dt
                    check_rhs(state, parameter_values, t, step, contract['contract_id'])
                    base_state = list(contract['initial_state'])
                    for c_slot in _semi_clocks:
                        if c_slot < len(base_state):
                            base_state[c_slot] = t
                    for idx_n, s_name in enumerate(names):
                        base_state[recovered['slots'][s_name]] = state[idx_n]
                    half = run_reference(code, {**contract, 'initial_state': base_state, 'steps': 2, 'dt': 0.5 * dt},
                        sample_steps=[2], sample_slots=[recovered['slots'][name] for name in names])['samples']['2']
                    for idx_n, (s_name, act_v, half_v) in enumerate(zip(names, row, half)):
                        scale_v = max(1.0, abs(state[idx_n]), abs(half_v))
                        if not math.isfinite(half_v) or abs(act_v - half_v) > max(0.1 * scale_v, 1e-6):
                            raise ReferenceError('VAULT_REFERENCE_MISMATCH',
                                f'{s_name} diverges at step {step}: full={act_v}, two_half={half_v}',
                                dict(state=s_name, step=step, contract_id=contract['contract_id'], actual=act_v, expected=half_v))
                state = row
            continue
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
    if scheme != 'semi_implicit':
        probe_jobs = []
        sample_slots = [recovered['slots'][name] for name in names]
        for values,params,dt,dmask in probes:
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
            if admissible:
                probe_jobs.append((values, params, dt, initial, dmask))
        if probe_jobs:
            from concurrent.futures import ThreadPoolExecutor

            def _run_probe(job):
                _vals, _pars, _dt, _init, _dmask = job
                return run_reference(
                    code,
                    {**first, 'initial_state': _init, 'steps': 1, 'dt': _dt, 'invariants': []},
                    sample_steps=[1],
                    sample_slots=sample_slots,
                )['samples']['1']

            if _direct_unstripped:
                probe_samples = [
                    numerical_update(*_vals, *_pars, 0., _dt)
                    for _vals, _pars, _dt, _init, _dmask in probe_jobs
                ]
            elif len(probe_jobs) == 1:
                probe_samples = [_run_probe(probe_jobs[0])]
            else:
                with ThreadPoolExecutor(max_workers=min(4, len(probe_jobs))) as pool:
                    probe_samples = list(pool.map(_run_probe, probe_jobs))
            for (values, params, dt, _init, dmask), sample in zip(probe_jobs, probe_samples):
                expected = integrate_rhs(lambda y,p,t:numerical_rhs(*y,*p,t),values,params,dt,0.,scheme)
                for name,a,b,ok in zip(names,sample,expected,dmask):
                    if not ok:
                        continue
                    if not math.isfinite(a) or abs(a-b)>1e-10:
                        details = dict(state=name,step=1,actual=a,expected=float(b),absolute_error=abs(a-b),tolerance=1e-10,comparison='sampled_step')
                        raise ReferenceError('VAULT_REFERENCE_MISMATCH',f'{name}: sampled {scheme} step disagrees with stated RHS',details)
    return recovered

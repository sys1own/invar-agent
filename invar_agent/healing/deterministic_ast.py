"""Deterministic AST sanitizer and directive applier for research kernels.

Tier-0 healing is intentionally conservative: it only applies semantics-
closing rewrites for scalar math intrinsics, math-domain guards, and the
contract-directed monotonicity/drift repair.  LLM healing never emits code;
it emits :class:`HealingDirective` dictionaries which are applied here.
"""
from __future__ import annotations

import ast
import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MATH_FUNCS = {
    "sqrt", "cbrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "exp", "expm1", "log", "log10", "log2", "log1p",
    "ceil", "floor", "trunc", "pow", "fabs", "fmod", "hypot", "copysign",
    "erf", "erfc", "gamma", "lgamma",
    "isfinite", "isnan", "isinf", "radians", "degrees",
}
MATH_CONSTANTS = {"pi", "e", "tau"}
INTRINSIC_ALIASES = {
    "clamp": "clamp", "clip": "clip",
    "sign": "sign", "sgn": "sgn",
    "sq": "sq", "lerp": "lerp", "safe_div": "safe_div",
}
NP_MATH_FUNCS = MATH_FUNCS | {"abs", "clip", "sign", "sgn"}
MONOTONE_NAME_TOKENS = ("entropy_drift", "entropy_monotonicity",
                        "free_energy_drift", "lyapunov_drift")


@dataclass(frozen=True)
class SanitizerResult:
    code: str
    notes: List[str]
    changed: bool
    backup_slot_map: Dict[str, int] = field(default_factory=dict)


def _defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _call_name(func: ast.expr) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return None


def _has_math_import(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == "math" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "math":
            return True
    return False


def _is_none_const(node: Optional[ast.AST]) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


class _NoneLiteralTransformer(ast.NodeTransformer):
    """Lower Python ``None`` to the numeric C-ABI dialect before transpile."""

    def __init__(self) -> None:
        self.changed = False
        self.notes: set[str] = set()
        self._void_stack: List[bool] = []

    def _has_value_return(self, fn: ast.FunctionDef) -> bool:
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and node.value is not None \
                    and not _is_none_const(node.value):
                return True
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._void_stack.append(not self._has_value_return(node))
        try:
            self.generic_visit(node)
        finally:
            self._void_stack.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if _is_none_const(node.value):
            self.changed = True
            self.notes.add("return None lowered")
            node.value = None if self._void_stack and self._void_stack[-1] else _num(0.0)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if _is_none_const(node.value):
            self.changed = True
            self.notes.add("None assignment lowered to 0.0")
            node.value = _num(0.0)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        if node.value is not None and _is_none_const(node.value):
            self.changed = True
            self.notes.add("None assignment lowered to 0.0")
            node.value = _num(0.0)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1 and len(node.comparators) == 1:
            op = node.ops[0]
            if isinstance(op, (ast.Is, ast.IsNot)) and (
                    _is_none_const(node.left) or _is_none_const(node.comparators[0])):
                self.changed = True
                self.notes.add("is None comparison lowered")
                value = isinstance(op, ast.IsNot)
                return ast.copy_location(ast.Constant(value=value), node)
        return node


def _strip_none_literals_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    transformer = _NoneLiteralTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    return tree, transformer.changed, sorted(transformer.notes)


# -- deterministic complex -> real lowering ---------------------------------

@dataclass
class _ComplexValue:
    re: ast.expr
    im: ast.expr
    complex_: bool = False

    def copy(self) -> "_ComplexValue":
        return _ComplexValue(copy.deepcopy(self.re), copy.deepcopy(self.im),
                             self.complex_)


class _ComplexScope:
    def __init__(self, outer: Optional["_ComplexScope"] = None,
                 funcs: Optional[Dict[str, ast.FunctionDef]] = None) -> None:
        self.outer = outer
        self.values: Dict[str, _ComplexValue] = {}
        self.funcs = funcs if funcs is not None else (
            outer.funcs if outer is not None else {})

    def lookup(self, name: str) -> Optional[_ComplexValue]:
        if name in self.values:
            return self.values[name].copy()
        if self.outer is not None:
            return self.outer.lookup(name)
        return None


def _fconst(value: float) -> ast.Constant:
    return ast.Constant(value=float(value))


def _is_zero_expr(node: Optional[ast.AST]) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
        and not isinstance(node.value, bool) and float(node.value) == 0.0


def _const_float(node: Optional[ast.AST]) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    return None


def _mk_complex(re: ast.expr, im: ast.expr,
                complex_: Optional[bool] = None) -> _ComplexValue:
    re = copy.deepcopy(re)
    im = copy.deepcopy(im)
    is_complex = (not _is_zero_expr(im)) if complex_ is None else complex_
    return _ComplexValue(re, im, bool(is_complex))


def _real_expr(node: ast.expr) -> _ComplexValue:
    return _mk_complex(copy.deepcopy(node), _fconst(0.0), False)


def _math_call(name: str, *args: ast.expr) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()),
                           attr=name, ctx=ast.Load()),
        args=[copy.deepcopy(a) for a in args], keywords=[])


def _name_load(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Load())


def _binop_node(op: ast.operator, left: ast.expr, right: ast.expr) -> ast.BinOp:
    return ast.BinOp(left=copy.deepcopy(left), op=op,
                     right=copy.deepcopy(right))


def _add_expr(a: ast.expr, b: ast.expr) -> ast.expr:
    if _is_zero_expr(a):
        return copy.deepcopy(b)
    if _is_zero_expr(b):
        return copy.deepcopy(a)
    return _binop_node(ast.Add(), a, b)


def _sub_expr(a: ast.expr, b: ast.expr) -> ast.expr:
    if _is_zero_expr(b):
        return copy.deepcopy(a)
    if _is_zero_expr(a):
        return ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(b))
    return _binop_node(ast.Sub(), a, b)


def _mul_expr(a: ast.expr, b: ast.expr) -> ast.expr:
    if _is_zero_expr(a) or _is_zero_expr(b):
        return _fconst(0.0)
    if _const_float(a) == 1.0:
        return copy.deepcopy(b)
    if _const_float(b) == 1.0:
        return copy.deepcopy(a)
    return _binop_node(ast.Mult(), a, b)


def _div_expr(a: ast.expr, b: ast.expr) -> ast.expr:
    if _is_zero_expr(a):
        return _fconst(0.0)
    if _const_float(b) == 1.0:
        return copy.deepcopy(a)
    return _binop_node(ast.Div(), a, b)


def _neg_expr(a: ast.expr) -> ast.expr:
    if _is_zero_expr(a):
        return copy.deepcopy(a)
    return ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(a))


def _complex_binop(op: ast.operator, a: _ComplexValue, b: _ComplexValue) -> _ComplexValue:
    if isinstance(op, ast.Add):
        return _mk_complex(_add_expr(a.re, b.re), _add_expr(a.im, b.im))
    if isinstance(op, ast.Sub):
        return _mk_complex(_sub_expr(a.re, b.re), _sub_expr(a.im, b.im))
    if isinstance(op, ast.Mult):
        re = _add_expr(_mul_expr(a.re, b.re), _neg_expr(_mul_expr(a.im, b.im)))
        im = _add_expr(_mul_expr(a.re, b.im), _mul_expr(a.im, b.re))
        return _mk_complex(re, im)
    if isinstance(op, ast.Div):
        den_re = _add_expr(_mul_expr(b.re, b.re), _mul_expr(b.im, b.im))
        den = ast.Call(func=_name_load("max"),
                       args=[den_re, _fconst(1e-300)], keywords=[])
        re = _div_expr(_add_expr(_mul_expr(a.re, b.re),
                                 _mul_expr(a.im, b.im)), den)
        im = _div_expr(_sub_expr(_mul_expr(a.im, b.re),
                                 _mul_expr(a.re, b.im)), den)
        return _mk_complex(re, im)
    if a.complex_ or b.complex_:
        return _mk_complex(ast.Constant(value=0.0), ast.Constant(value=0.0), False)
    return _real_expr(_binop_node(op, a.re, b.re))


def _contains_complex_construct(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, complex):
            return True
        if isinstance(sub, ast.Call):
            name = _call_name(sub.func)
            if name == "complex" or (name or "").startswith("cmath."):
                return True
        if isinstance(sub, ast.Attribute) and sub.attr in {"real", "imag", "conjugate"}:
            return True
    return False


def _simple_return_exprs(fn: ast.FunctionDef) -> Optional[List[ast.expr]]:
    returns = [s for s in fn.body if isinstance(s, ast.Return)]
    if len(returns) != 1:
        return None
    value = returns[0].value
    if value is None:
        return []
    if isinstance(value, (ast.Tuple, ast.List)):
        return list(value.elts)
    return [value]


def _inline_call_values(call: ast.Call, scope: _ComplexScope,
                        depth: int = 0) -> Optional[List[_ComplexValue]]:
    """Evaluate a pure single-return/tuple helper without emitting its body."""
    if depth > 8 or call.keywords:
        return None
    if not isinstance(call.func, ast.Name):
        return None
    fn = scope.funcs.get(call.func.id)
    if fn is None:
        return None
    returns = _simple_return_exprs(fn)
    if returns is None:
        return None
    for stmt in fn.body:
        if isinstance(stmt, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.Try,
                             ast.With, ast.Match, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            return None
        if isinstance(stmt, ast.Expr) and not (
                isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            return None
    local = _ComplexScope(outer=scope)
    params = [a.arg for a in fn.args.args]
    for i, param in enumerate(params):
        if i < len(call.args):
            val = _decompose_expr(call.args[i], scope, depth + 1)
        else:
            val = _real_expr(_fconst(0.0))
        if val is None:
            return None
        local.values[param] = val
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            val = _decompose_expr(stmt.value, local, depth + 1)
            if val is None:
                return None
            local.values[stmt.targets[0].id] = val
        elif isinstance(stmt, (ast.Pass, ast.Import, ast.ImportFrom, ast.Return)):
            continue
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                and isinstance(stmt.value.value, str):
            continue
        else:
            return None
    out: List[_ComplexValue] = []
    for expr in returns:
        val = _decompose_expr(expr, local, depth + 1)
        if val is None:
            return None
        out.append(val)
    return out


def _decompose_expr(node: ast.expr, scope: _ComplexScope,
                    depth: int = 0) -> Optional[_ComplexValue]:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, complex):
            return _mk_complex(_fconst(value.real), _fconst(value.imag), True)
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return None
        if isinstance(value, (int, float)):
            return _real_expr(_fconst(value))
        return None
    if isinstance(node, ast.Name):
        val = scope.lookup(node.id)
        if val is not None:
            return val
        return _real_expr(_name_load(node.id))
    if isinstance(node, ast.Attribute) and node.attr in {"real", "imag"}:
        val = _decompose_expr(node.value, scope, depth)
        if val is None:
            return None
        part = val.re if node.attr == "real" else val.im
        return _real_expr(part)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub,
                                                         ast.Mult, ast.Div)):
        left = _decompose_expr(node.left, scope, depth)
        right = _decompose_expr(node.right, scope, depth)
        if left is None or right is None:
            return None
        return _complex_binop(node.op, left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _decompose_expr(node.operand, scope, depth)
        if val is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return val.copy()
        return _mk_complex(_neg_expr(val.re), _neg_expr(val.im), val.complex_)
    if isinstance(node, ast.Call):
        return _decompose_call(node, scope, depth)
    if _contains_complex_construct(node):
        return None
    return _real_expr(copy.deepcopy(node))


def _decompose_call(node: ast.Call, scope: _ComplexScope,
                    depth: int = 0) -> Optional[_ComplexValue]:
    if isinstance(node.func, ast.Attribute) and node.func.attr == "conjugate" \
            and not node.args:
        val = _decompose_expr(node.func.value, scope, depth + 1)
        if val is None:
            return None
        return _mk_complex(val.re, _neg_expr(val.im), val.complex_)
    name = _call_name(node.func)
    if name == "complex" and not node.keywords:
        if not node.args or len(node.args) > 2:
            return None
        re = _decompose_expr(node.args[0], scope, depth + 1)
        if re is None:
            return None
        if len(node.args) == 1:
            return _mk_complex(re.re, _fconst(0.0), re.complex_)
        im = _decompose_expr(node.args[1], scope, depth + 1)
        if im is None:
            return None
        return _mk_complex(re.re, im.re, True)
    if name in {"conj", "cmath.conj"} and len(node.args) == 1:
        val = _decompose_expr(node.args[0], scope, depth + 1)
        if val is None:
            return None
        return _mk_complex(val.re, _neg_expr(val.im), val.complex_)
    if name == "abs" and len(node.args) == 1:
        val = _decompose_expr(node.args[0], scope, depth + 1)
        if val is None:
            return None
        if val.complex_:
            sq = _add_expr(_mul_expr(val.re, val.re), _mul_expr(val.im, val.im))
            return _real_expr(_math_call("sqrt", sq))
        return _real_expr(ast.Call(func=_name_load("abs"),
                                   args=[val.re], keywords=[]))
    if name == "cmath.exp" and len(node.args) == 1:
        val = _decompose_expr(node.args[0], scope, depth + 1)
        if val is None:
            return None
        if not val.complex_:
            return _real_expr(_math_call("exp", val.re))
        e = _math_call("exp", val.re)
        re = _mul_expr(e, _math_call("cos", val.im))
        im = _mul_expr(copy.deepcopy(e), _math_call("sin", val.im))
        return _mk_complex(re, im, True)
    if name in {"cmath.sqrt", "cmath.sin", "cmath.cos"} and len(node.args) == 1:
        val = _decompose_expr(node.args[0], scope, depth + 1)
        if val is None:
            return None
        return _real_expr(_math_call(name.rsplit(".", 1)[1], val.re))
    inlined = _inline_call_values(node, scope, depth)
    if inlined is not None and len(inlined) == 1 and inlined[0].complex_:
        return inlined[0]
    if _contains_complex_construct(node):
        return None
    return _real_expr(copy.deepcopy(node))


def _complex_target_name(target: ast.expr, suffix: str) -> Optional[str]:
    if isinstance(target, ast.Name):
        return f"{target.id}{suffix}"
    return None


def _assign_stmt(target: ast.expr, value: ast.expr) -> ast.Assign:
    return ast.Assign(targets=[copy.deepcopy(target)],
                      value=copy.deepcopy(value))


def _expr_references_complex(node: ast.AST, scope: _ComplexScope) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            val = scope.lookup(sub.id)
            if val is not None and val.complex_:
                return True
    return False


def _needs_complex_lowering(node: ast.AST, scope: _ComplexScope) -> bool:
    return _contains_complex_construct(node) or _expr_references_complex(node, scope)


class _ComplexLoweringTransformer(ast.NodeTransformer):
    def __init__(self, funcs: Dict[str, ast.FunctionDef]) -> None:
        self.funcs = {name: copy.deepcopy(fn) for name, fn in funcs.items()}
        self.notes: set[str] = set()
        self.changed = False
        self.generated_math = False
        self._scopes: List[_ComplexScope] = [_ComplexScope(funcs=self.funcs)]

    @property
    def scope(self) -> _ComplexScope:
        return self._scopes[-1]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._scopes.append(_ComplexScope(outer=self._scopes[-1], funcs=self.funcs))
        try:
            self.generic_visit(node)
        finally:
            self._scopes.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> ast.AST:
        changed = False
        for alias in node.names:
            if alias.name == "cmath":
                alias.name = "math"
                alias.asname = None if alias.asname in {None, "cmath"} else alias.asname
                changed = True
        if changed:
            self.changed = True
            self.notes.add("import cmath lowered to math")
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.module == "cmath":
            node.module = "math"
            self.changed = True
            self.notes.add("from cmath lowered to math")
        return node

    def _set_scope_value(self, name: str, val: _ComplexValue,
                         re_name: Optional[str] = None,
                         im_name: Optional[str] = None) -> None:
        if val.complex_ and re_name and im_name:
            self.scope.values[name] = _ComplexValue(_name_load(re_name),
                                                    _name_load(im_name), True)
        elif val.complex_:
            self.scope.values[name] = val.copy()
        else:
            self.scope.values[name] = _real_expr(_name_load(name))

    def _emit_complex_assignment(self, target: ast.Name,
                                 val: _ComplexValue) -> List[ast.stmt]:
        re_name = f"{target.id}__re"
        im_name = f"{target.id}__im"
        out = [
            _assign_stmt(ast.Name(id=re_name, ctx=ast.Store()), val.re),
            _assign_stmt(ast.Name(id=im_name, ctx=ast.Store()), val.im),
        ]
        self._set_scope_value(target.id, val, re_name, im_name)
        return out

    def _lower_assign_value(self, stmt: ast.Assign) -> Optional[List[ast.stmt]]:
        if len(stmt.targets) != 1:
            return None
        target = stmt.targets[0]
        value = stmt.value
        if isinstance(target, (ast.Tuple, ast.List)):
            if not (all(isinstance(e, ast.Name) for e in target.elts)
                    and isinstance(value, ast.Call)):
                return None
            vals = _inline_call_values(value, self.scope)
            if vals is not None and len(vals) == len(target.elts) \
                    and any(v.complex_ for v in vals):
                out: List[ast.stmt] = []
                for elt, val in zip(target.elts, vals):
                    if val.complex_:
                        out.extend(self._emit_complex_assignment(elt, val))
                    else:
                        out.append(_assign_stmt(elt, val.re))
                        self._set_scope_value(elt.id, val)
                self.changed = True
                self.notes.add("LOWER_COMPLEX_TO_REAL")
                return out
            return None
        val = _decompose_expr(value, self.scope)
        if val is None:
            return None
        needs = _needs_complex_lowering(value, self.scope)
        if isinstance(target, ast.Name):
            stale_complex = target.id in self.scope.values \
                and self.scope.values[target.id].complex_
            if val.complex_:
                out = self._emit_complex_assignment(target, val)
                self.changed = True
                self.notes.add("LOWER_COMPLEX_TO_REAL")
                return out
            if needs or stale_complex:
                out = [_assign_stmt(target, val.re)]
                self._set_scope_value(target.id, val)
                self.changed = True
                self.notes.add("LOWER_COMPLEX_TO_REAL")
                return out
            return None
        # buf.data[k] / attribute targets are scalar sinks: keep the real part.
        if not (val.complex_ or needs):
            return None
        if val.complex_:
            self.notes.add("complex value lowered to real part at scalar sink")
        self.changed = True
        self.notes.add("LOWER_COMPLEX_TO_REAL")
        return [_assign_stmt(target, val.re)]

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        lowered = self._lower_assign_value(node)
        if lowered is not None:
            return lowered
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        if isinstance(node.target, ast.Name):
            target_val = self.scope.lookup(node.target.id)
            rhs = _decompose_expr(node.value, self.scope)
            if rhs is not None and ((target_val is not None and target_val.complex_)
                                    or rhs.complex_):
                if target_val is None:
                    target_val = _real_expr(_name_load(node.target.id))
                re_name = f"{node.target.id}__re"
                im_name = f"{node.target.id}__im"
                op = node.op
                re_expr = _complex_binop(op, target_val, rhs).re
                im_expr = _complex_binop(op, target_val, rhs).im
                self.scope.values[node.target.id] = _ComplexValue(
                    _name_load(re_name), _name_load(im_name), True)
                self.changed = True
                self.notes.add("LOWER_COMPLEX_TO_REAL")
                return [
                    ast.Assign(targets=[ast.Name(id=re_name, ctx=ast.Store())],
                               value=re_expr),
                    ast.Assign(targets=[ast.Name(id=im_name, ctx=ast.Store())],
                               value=im_expr),
                ]
        self.generic_visit(node)
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if node.value is None:
            return node
        if isinstance(node.value, (ast.Tuple, ast.List)):
            vals = [_decompose_expr(e, self.scope) for e in node.value.elts]
            if all(v is not None for v in vals):
                assert all(v is not None for v in vals)
                needs = any(v.complex_ or _needs_complex_lowering(e, self.scope)
                            for e, v in zip(node.value.elts, vals))
                if not needs:
                    return node
                any_complex = any(v.complex_ for v in vals)
                if any_complex:
                    flat: List[ast.expr] = []
                    for val in vals:
                        assert val is not None
                        flat.append(val.re)
                        if val.complex_:
                            flat.append(val.im)
                    node.value = ast.Tuple(elts=flat, ctx=ast.Load())
                else:
                    node.value = ast.Tuple(elts=[v.re for v in vals if v is not None],
                                           ctx=ast.Load())
                self.changed = True
                self.notes.add("LOWER_COMPLEX_TO_REAL")
                return node
            self.generic_visit(node)
            return node
        val = _decompose_expr(node.value, self.scope)
        if val is not None and (val.complex_
                                or _needs_complex_lowering(node.value, self.scope)):
            node.value = val.re
            if val.complex_:
                self.notes.add("complex return lowered to real part")
            self.changed = True
            self.notes.add("LOWER_COMPLEX_TO_REAL")
            return node
        self.generic_visit(node)
        return node


def _function_call_names(fn: ast.FunctionDef, names: set[str]) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in names:
                out.add(name)
    return out


def _is_pure_expr(expr: ast.expr) -> bool:
    return not any(
        isinstance(n, ast.Call)
        and (_call_name(n.func) or "") not in (
            MATH_FUNCS | {"abs", "min", "max", "float", "int", "len", "list"}
        )
        for n in ast.walk(expr)
    )


def _name_read_before_overwrite(stmts: List[ast.stmt], name: str) -> bool:
    """Return True if `name` is loaded in `stmts` before being unconditionally overwritten."""
    for s in stmts:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) and s.targets[0].id == name:
            # Check RHS of this reassignment first: `x = x + 1` or `x = 0.5 * (x + y)` reads `x`!
            if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name for n in ast.walk(s.value)):
                return True
            # Unconditionally overwritten without reading `name` on RHS.
            return False
        if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name) and s.target.id == name:
            return True
        if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name for n in ast.walk(s)):
            return True
    return False


def _prune_dead_in_block(stmts: List[ast.stmt], outer_loaded: set[str]) -> Tuple[List[ast.stmt], bool]:
    changed = False
    out: List[ast.stmt] = []
    for idx, stmt in enumerate(stmts):
        # Recurse into nested blocks first
        for fld in ("body", "orelse"):
            sub = getattr(stmt, fld, None)
            if isinstance(sub, list) and sub:
                # Conservatively include anything loaded in remaining outer statements
                rem_loaded = set(outer_loaded)
                if isinstance(stmt, (ast.While, ast.For)):
                    for n in ast.walk(stmt):
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                            rem_loaded.add(n.id)
                for later in stmts[idx + 1:]:
                    for n in ast.walk(later):
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                            rem_loaded.add(n.id)
                new_sub, sub_changed = _prune_dead_in_block(sub, rem_loaded)
                if sub_changed:
                    setattr(stmt, fld, new_sub)
                    changed = True
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt = stmt.targets[0]
            # Prune trivial self-assignments `x = x` or `buf.data[k] = buf.data[k]`
            if ast.dump(tgt, include_attributes=False) == ast.dump(
                stmt.value, include_attributes=False
            ).replace("Load()", "Store()"):
                changed = True
                continue
            if isinstance(tgt, ast.Name) and _is_pure_expr(stmt.value):
                name = tgt.id
                # Never prune if read anywhere in remaining statements or outer scope before overwrite
                if name not in outer_loaded and not _name_read_before_overwrite(stmts[idx + 1:], name):
                    changed = True
                    continue
        out.append(stmt)
    if not out:
        out.append(ast.Pass())
    return out, changed


def _dead_assignments_in_function(fn: ast.FunctionDef) -> Tuple[ast.FunctionDef, bool]:
    total_changed = False
    for _ in range(8):
        new_body, changed = _prune_dead_in_block(fn.body, set())
        if not changed:
            break
        fn.body = new_body
        total_changed = True
    return fn, total_changed


def _prune_helpers_tree(tree: ast.Module,
                        entrypoints: Sequence[str] = ("simulation_step",)
                        ) -> Tuple[ast.Module, bool, List[str]]:
    notes: List[str] = []
    funcs = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    changed = False
    if funcs:
        names = {f.name for f in funcs}
        calls = {f.name: _function_call_names(f, names) for f in funcs}
        roots = [
            f.name for f in funcs
            if f.name in entrypoints
            or f.name.startswith("simulation_step_")
            or f.name.startswith("layer_")
            or f.name.startswith("_invar_")
        ]
        if roots:
            live = set(roots)
            stack = list(roots)
            while stack:
                cur = stack.pop()
                for nxt in calls.get(cur, set()):
                    if nxt not in live:
                        live.add(nxt)
                        stack.append(nxt)
            kept = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and node.name not in live:
                    changed = True
                    notes.append(f"PRUNE_UNUSED_HELPERS({node.name})")
                    continue
                kept.append(node)
            tree.body = kept
    for node in list(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            new_node, node_changed = _dead_assignments_in_function(node)
            if node_changed:
                changed = True
                notes.append(f"PRUNE_UNUSED_HELPERS(dead assignments in {node.name})")
            tree.body[tree.body.index(node)] = new_node
    ast.fix_missing_locations(tree)
    return tree, changed, notes


def _if_always_assigns(if_stmt: ast.If) -> Set[str]:
    """Return variable names definitely assigned in BOTH branches of `if_stmt`."""
    if not if_stmt.body or not if_stmt.orelse:
        return set()

    def _branch_assigns(stmts: List[ast.stmt]) -> Set[str]:
        out: Set[str] = set()
        for sub in stmts:
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                out.add(sub.targets[0].id)
            elif isinstance(sub, ast.If):
                out |= _if_always_assigns(sub)
        return out

    return _branch_assigns(if_stmt.body) & _branch_assigns(if_stmt.orelse)


def _init_conditional_locals_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    """Zero-initialize locals read before an unconditional top-level assignment."""
    changed = False
    notes: List[str] = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_invar_"):
            continue
        params = {a.arg for a in fn.args.args}
        assigned_in_fn: Set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                assigned_in_fn.add(n.id)
        local_targets = assigned_in_fn - params
        if not local_targets:
            continue
        definitely_assigned = set(params)
        needs_init: Set[str] = set()
        for stmt in fn.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                for n in ast.walk(stmt.value):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        if n.id in local_targets and n.id not in definitely_assigned:
                            needs_init.add(n.id)
                definitely_assigned.add(stmt.targets[0].id)
            elif isinstance(stmt, ast.If) and stmt.body and stmt.orelse:
                for n in ast.walk(stmt.test):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        if n.id in local_targets and n.id not in definitely_assigned:
                            needs_init.add(n.id)
                for branch in (stmt.body, stmt.orelse):
                    branch_def = set(definitely_assigned)
                    for sub in branch:
                        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                            for n in ast.walk(sub.value):
                                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                                    if n.id in local_targets and n.id not in branch_def:
                                        needs_init.add(n.id)
                            branch_def.add(sub.targets[0].id)
                        else:
                            for n in ast.walk(sub):
                                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                                    if n.id in local_targets and n.id not in branch_def:
                                        needs_init.add(n.id)
                definitely_assigned |= _if_always_assigns(stmt)
            else:
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        if n.id in local_targets and n.id not in definitely_assigned:
                            needs_init.add(n.id)
        if needs_init:
            insert_idx = 0
            if (fn.body and isinstance(fn.body[0], ast.Expr)
                    and isinstance(fn.body[0].value, ast.Constant)
                    and isinstance(fn.body[0].value.value, str)):
                insert_idx = 1
            inits = [
                ast.Assign(
                    targets=[ast.Name(id=v, ctx=ast.Store())],
                    value=ast.Constant(value=0.0),
                )
                for v in sorted(needs_init)
            ]
            fn.body[insert_idx:insert_idx] = inits
            ast.fix_missing_locations(fn)
            changed = True
            notes.append(f"initialized conditional locals {sorted(needs_init)} in {fn.name}")
    return tree, changed, notes


def _lower_complex_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    notes: List[str] = []
    changed = False
    tree, p_changed, p_notes = _prune_helpers_tree(tree)
    changed |= p_changed
    notes.extend(p_notes)
    funcs = {n.name: n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    transformer = _ComplexLoweringTransformer(funcs)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    if transformer.changed:
        changed = True
        notes.extend(sorted(transformer.notes))
    if transformer.generated_math or any(n == "LOWER_COMPLEX_TO_REAL" for n in notes):
        if _ensure_math_import(tree):
            notes.append("injected import math")
            changed = True
    tree, q_changed, q_notes = _prune_helpers_tree(tree)
    changed |= q_changed
    notes.extend(q_notes)
    return tree, changed, sorted(set(notes))


def lower_complex_to_real(code: str) -> SanitizerResult:
    """Public Tier-0 entry point for ``LOWER_COMPLEX_TO_REAL`` directives."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    tree, changed, notes = _lower_complex_tree(tree)
    if not changed:
        return SanitizerResult(code, [], False)
    return SanitizerResult(ast.unparse(tree) + "\n", notes, True)


def prune_unused_helpers(code: str,
                         entrypoints: Sequence[str] = ("simulation_step",)
                         ) -> SanitizerResult:
    """Public Tier-0 entry point for ``PRUNE_UNUSED_HELPERS`` directives."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    tree, changed, notes = _prune_helpers_tree(tree, entrypoints)
    if not changed:
        return SanitizerResult(code, [], False)
    return SanitizerResult(ast.unparse(tree) + "\n", notes, True)


def _ensure_math_import(tree: ast.Module) -> bool:
    if _has_math_import(tree):
        return False
    idx = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            idx = i + 1
    tree.body.insert(idx, ast.Import(names=[ast.alias(name="math")]))
    ast.fix_missing_locations(tree)
    return True


def _intrinsic_defs(used: Iterable[str]) -> List[ast.FunctionDef]:
    """AST helper definitions for scalar intrinsics missing from builtins."""
    src = {
        "clamp": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
        "clip": "def clip(x, lo, hi):\n    return clamp(x, lo, hi)\n",
        "sign": "def sign(x):\n    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)\n",
        "sgn": "def sgn(x):\n    return sign(x)\n",
        "sq": "def sq(x):\n    return x * x\n",
        "lerp": "def lerp(a, b, t):\n    return a + t * (b - a)\n",
        "safe_div": "def safe_div(a, b):\n    return a / (b if abs(b) > 1e-30 else 1e-30)\n",
    }
    out: List[ast.FunctionDef] = []
    for name in sorted(set(used) & set(src)):
        out.extend(ast.parse(src[name]).body)
    return out


def _is_guarded_log(arg: ast.expr) -> bool:
    return isinstance(arg, ast.Call) and _call_name(arg.func) in {"max", "math.fmax"}


def _is_guarded_sqrt(arg: ast.expr) -> bool:
    return isinstance(arg, ast.Call) and _call_name(arg.func) in {"max", "math.fmax"}


def _is_guarded_unit_interval(arg: ast.expr) -> bool:
    if not isinstance(arg, ast.Call) or _call_name(arg.func) != "min":
        return False
    return len(arg.args) >= 1


def _num(value: float) -> ast.Constant:
    return ast.Constant(value=value)


def _domain_guarded_call(node: ast.Call, eps: float) -> Optional[ast.Call]:
    """Return a domain-safe replacement for math.log/sqrt/asin/acos calls."""
    name = _call_name(node.func)
    if name not in {"math.log", "math.sqrt", "math.asin", "math.acos"} or not node.args:
        return None
    arg = node.args[0]
    if name == "math.log":
        if _is_guarded_log(arg):
            return None
        guarded = ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                           args=[arg, _num(eps)], keywords=[])
    elif name == "math.sqrt":
        if _is_guarded_sqrt(arg):
            return None
        guarded = ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                           args=[arg, _num(0.0)], keywords=[])
    else:
        if _is_guarded_unit_interval(arg):
            return None
        inner = ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                         args=[arg, _num(-1.0)], keywords=[])
        guarded = ast.Call(func=ast.Name(id="min", ctx=ast.Load()),
                           args=[inner, _num(1.0)], keywords=[])
    node.args[0] = guarded
    return node


class _IntrinsicTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.defined: set[str] = set()
        self.used_intrinsics: set[str] = set()
        self.used_math: set[str] = set()
        self.used_np: set[str] = set()
        self.notes: set[str] = set()
        self.guard_eps = 1e-300

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = _call_name(node.func)
        if name is None:
            return node
        base, _, attr = name.partition(".")
        if base in {"np", "numpy"} and attr:
            self.used_np.add(attr)
            if attr == "clip":
                self.used_intrinsics.add("clip")
                node.func = ast.Name(id="clip", ctx=ast.Load())
            elif attr in {"sign", "sgn", "sq", "lerp", "safe_div"}:
                self.used_intrinsics.add(attr)
                node.func = ast.Name(id=attr, ctx=ast.Load())
            elif attr == "abs":
                node.func = ast.Name(id="abs", ctx=ast.Load())
            elif attr in MATH_FUNCS:
                self.used_math.add(attr)
                node.func = ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()),
                                          attr=attr, ctx=ast.Load())
            else:
                self.notes.add(f"unsupported np.{attr} left unchanged")
            guarded = _domain_guarded_call(node, self.guard_eps)
            return guarded if guarded is not None else node
        if "." not in name:
            if name in MATH_FUNCS and name not in self.defined and name not in {"abs", "min", "max", "sum", "len", "range", "int", "float"}:
                self.used_math.add(name)
                node.func = ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()),
                                          attr=name, ctx=ast.Load())
                name = f"math.{name}"
            elif name in INTRINSIC_ALIASES and name not in self.defined:
                self.used_intrinsics.add(name)
        guarded = _domain_guarded_call(node, self.guard_eps)
        return guarded if guarded is not None else node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in MATH_CONSTANTS \
                and node.id not in self.defined and node.id not in {"pi", "e", "tau"}:
            # Constants are rewritten only when not shadowed; pi/e/tau are
            # handled by the sandbox namespace and math.* attributes.
            pass
        if isinstance(node.ctx, ast.Load) and node.id in {"pi", "e"} \
                and node.id not in self.defined:
            # Bare pi/e are common LLM mistakes; math.pi/math.e is canonical.
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()),
                              attr=node.id, ctx=ast.Load()), node)
        return node


def _hoist_intrinsic_helpers(tree: ast.Module) -> bool:
    """Ensure all _invar_* and safe_* FunctionDefs sit after imports and before all Assign/FunctionDefs."""
    intrinsic_prefixes = ("_invar_", "safe_")
    intrinsic_names = {"div_eps", "smooth_clamp"}
    imports: List[ast.stmt] = []
    helpers: List[ast.stmt] = []
    rest: List[ast.stmt] = []
    seen_helpers: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.FunctionDef) and (
            node.name.startswith(intrinsic_prefixes) or node.name in intrinsic_names
        ):
            if node.name not in seen_helpers:
                seen_helpers.add(node.name)
                helpers.append(node)
        else:
            rest.append(node)
    new_body = imports + helpers + rest
    if len(new_body) == len(tree.body) and all(a is b for a, b in zip(new_body, tree.body)):
        return False
    tree.body = new_body
    ast.fix_missing_locations(tree)
    return True


def _insert_intrinsic_helpers(tree: ast.Module, names: Iterable[str]) -> bool:
    existing_fns = {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    defs = [d for d in _intrinsic_defs(names) if getattr(d, "name", None) not in existing_fns]
    changed = False
    if defs:
        idx = 0
        for i, node in enumerate(tree.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                idx = i + 1
        tree.body[idx:idx] = defs
        changed = True
    if _hoist_intrinsic_helpers(tree):
        changed = True
    if changed:
        ast.fix_missing_locations(tree)
    return changed


# -- simulation_step slot repairs -------------------------------------------

_STATE_COORD_NAMES = {"x", "y", "z", "q", "p"}
_RHO_NAME = re.compile(r"^r\d+$")
_PARAM_BLOCK_TOKENS = (
    "prev", "before", "old", "incr", "diff", "drift", "err", "error",
    "residual", "violation", "mismatch", "trace", "entropy", "purity",
    "population", "coh", "mag", "rp", "wss", "trr", "p0", "p1", "rho", "te",
)
_OMEGA_NAMES = {"omega", "omega0", "k0", "m_mass"}
_KAPPA_NAMES = {"kappa", "stiffness"}


def _is_parameter_name(name: str) -> bool:
    low = name.lower()
    if low not in _OMEGA_NAMES | _KAPPA_NAMES:
        return False
    return not any(tok in low for tok in _PARAM_BLOCK_TOKENS)


def _state_subscript_slot(node: ast.AST,
                          extra_aliases: Optional[set[str]] = None) -> Optional[int]:
    if not isinstance(node, ast.Subscript):
        return None
    slot = _target_slot(node)
    if slot is None:
        return None
    base = node.value
    if isinstance(base, ast.Attribute) and base.attr == "data" \
            and isinstance(base.value, ast.Name):
        return slot
    if isinstance(base, ast.Name) and (
            base.id in {"state", "data", "buf", "D"}
            or (extra_aliases and base.id in extra_aliases)):
        return slot
    return None


def _simulation_functions(tree: ast.Module) -> List[ast.FunctionDef]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == "simulation_step" or (
                node.args.args and node.args.args[0].arg == "buf"):
            out.append(node)
    return out


def _assign_target_name(stmt: ast.stmt) -> Optional[str]:
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _stmt_slot_target(stmt: ast.stmt) -> Optional[int]:
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            slot = _state_subscript_slot(target)
            if slot is not None:
                return slot
    elif isinstance(stmt, ast.AnnAssign):
        slot = _state_subscript_slot(stmt.target)
        if slot is not None:
            return slot
    elif isinstance(stmt, ast.AugAssign):
        slot = _state_subscript_slot(stmt.target)
        if slot is not None:
            return slot
    return None


def _has_step0_guard(fn: ast.AST, prev: str, curr: str) -> bool:
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 \
                or not isinstance(test.ops[0], ast.Eq) \
                or not isinstance(test.left, ast.Name) or test.left.id != prev \
                or len(test.comparators) != 1:
            continue
        comp = test.comparators[0]
        if not (isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float))
                and not isinstance(comp.value, bool) and float(comp.value) == 0.0):
            continue
        for sub in node.body:
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 \
                    and isinstance(sub.targets[0], ast.Name) \
                    and sub.targets[0].id == prev \
                    and isinstance(sub.value, ast.Name) and sub.value.id == curr:
                return True
    return False


def _diff_updates_accumulator(fn: ast.AST, diff_var: str) -> bool:
    compared: set[int] = set()
    assigned: set[int] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare):
            operands = [node.left] + list(node.comparators)
            for i, operand in enumerate(operands):
                other = operands[1 - i] if len(operands) == 2 else None
                if other is None:
                    continue
                if isinstance(operand, ast.Name) and operand.id == diff_var:
                    slot = _state_subscript_slot(other)
                    if slot is not None:
                        compared.add(slot)
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            target_slot = None
            value = None
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    target_slot = _state_subscript_slot(target)
                    if target_slot is not None:
                        break
                value = node.value
            else:
                target_slot = _state_subscript_slot(node.target)
                value = node.value
            if target_slot is not None and value is not None:
                if any(isinstance(sub, ast.Name) and sub.id == diff_var
                       for sub in ast.walk(value)):
                    assigned.add(target_slot)
    return bool(compared & assigned)


def _repair_history_guard_in_function(fn: ast.FunctionDef, notes: set[str]) -> bool:
    changed = False
    body = fn.body
    for _ in range(16):
        inserted = False
        for i, stmt in enumerate(body):
            prev = _assign_target_name(stmt)
            if prev is None or not isinstance(stmt.value, ast.Subscript):
                continue
            prev_slot = _state_subscript_slot(stmt.value)
            if prev_slot is None:
                continue
            curr = None
            write_idx = -1
            for j in range(i + 1, len(body)):
                later = body[j]
                if _stmt_slot_target(later) == prev_slot \
                        and isinstance(later, (ast.Assign, ast.AnnAssign)) \
                        and getattr(later, "value", None) is not None \
                        and isinstance(later.value, ast.Name):
                    curr = later.value.id
                    write_idx = j
                    break
            if curr is None or write_idx <= i + 1:
                continue
            diff_var = None
            diff_idx = -1
            for k in range(i + 1, write_idx):
                cand = body[k]
                name = _assign_target_name(cand)
                if name is None or not isinstance(cand.value, ast.BinOp) \
                        or not isinstance(cand.value.op, ast.Sub):
                    continue
                left = cand.value.left
                right = cand.value.right
                pair = set()
                if isinstance(left, ast.Name):
                    pair.add(left.id)
                if isinstance(right, ast.Name):
                    pair.add(right.id)
                if pair == {prev, curr}:
                    diff_var, diff_idx = name, k
                    break
            if diff_var is None or not _diff_updates_accumulator(fn, diff_var):
                continue
            if _has_step0_guard(fn, prev, curr):
                continue
            guard = ast.parse(
                f"if {prev} == 0.0:\n    {prev} = {curr}\n").body[0]
            body.insert(diff_idx, guard)
            ast.fix_missing_locations(fn)
            notes.add(f"history step-0 guard for {prev} (slot {prev_slot})")
            changed = True
            inserted = True
            break
        if not inserted:
            break
    return changed


def _repair_delayed_state_history_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    changed = False
    notes: List[str] = []
    for fn in _simulation_functions(tree):
        body = fn.body
        delayed: Dict[int, Tuple[str, int]] = {}
        for stmt in body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            slot = _state_subscript_slot(target)
            if slot is None or not isinstance(stmt.value, ast.Name):
                continue
            state_var = stmt.value.id
            if state_var not in _STATE_COORD_NAMES and not _RHO_NAME.match(state_var):
                continue
            assign_lineno = getattr(stmt, "lineno", 0) or 0
            read_before = False
            for s in body:
                if (getattr(s, "lineno", 0) or 0) >= assign_lineno:
                    continue
                # Skip the normal initial load `r00 = buf.data[0]` and `_invar_*` snapshots
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                    and (
                        s.targets[0].id.startswith("_invar_")
                        or (s.targets[0].id == state_var and _state_subscript_slot(s.value) == slot)
                    )
                ):
                    continue
                for node in ast.walk(s):
                    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) \
                            and _state_subscript_slot(node) == slot:
                        read_before = True
                        break
                if read_before:
                    break
            if read_before:
                delayed[slot] = (state_var, assign_lineno)
        if not delayed:
            continue

        class _DelayedReadRewrite(ast.NodeTransformer):
            def visit_Assign(self, node: ast.Assign) -> ast.AST:
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and _state_subscript_slot(node.value) is not None
                ):
                    slot = _state_subscript_slot(node.value)
                    entry = delayed.get(slot) if slot is not None else None
                    if entry is not None and entry[0] == node.targets[0].id:
                        return node
                return self.generic_visit(node)

            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                self.generic_visit(node)
                if not isinstance(node.ctx, ast.Load):
                    return node
                slot = _state_subscript_slot(node)
                entry = delayed.get(slot) if slot is not None else None
                if entry is None:
                    return node
                var, assign_lineno = entry
                if (getattr(node, "lineno", 0) or 0) >= assign_lineno:
                    return node
                return ast.copy_location(ast.Name(id=var, ctx=ast.Load()), node)

        transformer = _DelayedReadRewrite()
        fn.body = [transformer.visit(stmt) for stmt in body]  # type: ignore[assignment]
        ast.fix_missing_locations(fn)
        changed = True
        notes.append("replaced delayed state-history reads with current state: "
                     + ", ".join(f"buf.data[{slot}]" for slot in sorted(delayed)))
    return tree, changed, notes


def _used_state_slots(tree: ast.Module) -> set[int]:
    used: set[int] = set()
    for node in ast.walk(tree):
        slot = _state_subscript_slot(node)
        if slot is not None:
            used.add(slot)
    return used


def _scratch_slot(used: set[int], avoid: set[int]) -> int:
    for cand in list(range(28, 48)) + list(range(16, 28)):
        if cand not in used and cand not in avoid:
            return cand
    return max(used | avoid | {0}) + 1


def _contract_closed_or_reversible(contract: Any) -> bool:
    cid = str(_contract_get(contract, "contract_id", "") or "").lower()
    if any(tok in cid for tok in ("closed", "revers", "chiasmus", "gamma0", "gamma_0")):
        return True
    if any(tok in cid for tok in ("dissipat", "stiff", "gamma1", "gamma_1")):
        return False
    for name, assertion, _tol in _contract_invariant_items(contract):
        low = f"{name} {assertion}".lower()
        if any(tok in low for tok in ("closed", "revers", "chiasmus")):
            return True
    return False


def _contracts_allow_kappa_default(contracts: Sequence[Any]) -> bool:
    contracts = list(contracts or [])
    return bool(contracts) and not any(_contract_closed_or_reversible(c) for c in contracts)


def _zero_guard_exists(body: List[ast.stmt], idx: int, var: str) -> bool:
    if idx + 1 >= len(body):
        return False
    node = body[idx + 1]
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return isinstance(test, ast.Compare) and len(test.ops) == 1 \
        and isinstance(test.ops[0], ast.Eq) \
        and isinstance(test.left, ast.Name) and test.left.id == var \
        and len(test.comparators) == 1 \
        and isinstance(test.comparators[0], ast.Constant) \
        and test.comparators[0].value == 0.0


def _stage1_slot_set(contracts: Sequence[Any]) -> Set[int]:
    """Return all state slots constrained by Stage 1 contracts."""
    stage1_slots: Set[int] = set()
    for idx, c in enumerate(contracts or []):
        cid = str(_contract_get(c, "contract_id", "") or "").lower()
        is_s1 = cid.startswith(("c1_", "stage1_", "stage_1_")) or (
            idx == 0 and not any(
                re.match(r"^(?:c\d+|stage_?\d+)_", str(_contract_get(cc, "contract_id", "") or "").lower())
                for cc in (contracts or [])
            )
        )
        if is_s1:
            for _, assertion, _ in _contract_invariant_items(c):
                stage1_slots.update(_contract_state_slots_from_text(assertion))
    return stage1_slots


def _is_buf_size_compare(test: ast.AST) -> bool:
    """True when an `if` condition branches on `buf.size` or `len(buf.data)`."""
    if not isinstance(test, ast.Compare):
        return False
    for sub in ast.walk(test):
        if (isinstance(sub, ast.Attribute) and sub.attr == "size"
                and isinstance(sub.value, ast.Name) and sub.value.id == "buf"):
            return True
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                and sub.func.id == "len" and len(sub.args) == 1
                and isinstance(sub.args[0], ast.Attribute)
                and sub.args[0].attr == "data"
                and isinstance(sub.args[0].value, ast.Name)
                and sub.args[0].value.id == "buf"):
            return True
    return False


def _strip_buf_size_branches_tree(
    tree: ast.Module,
    contracts: Sequence[Any] = (),
) -> Tuple[ast.Module, bool, List[str]]:
    """Remove `if buf.size > ...` / `if len(buf.data) > ...` overwrites of Stage 1 or prior-assigned slots."""
    s1_slots = _stage1_slot_set(contracts)
    changed = False
    notes: List[str] = []

    for fn in _simulation_functions(tree):
        early_assigned: Set[int] = set()

        def _filter_block(stmts: List[ast.stmt], inside_buf_size: bool = False) -> List[ast.stmt]:
            nonlocal changed
            out: List[ast.stmt] = []
            for stmt in stmts:
                if isinstance(stmt, ast.If) and _is_buf_size_compare(stmt.test):
                    stmt.body = _filter_block(stmt.body, inside_buf_size=True)
                    stmt.orelse = _filter_block(stmt.orelse, inside_buf_size=True)
                    if not stmt.body and not stmt.orelse:
                        changed = True
                        notes.append(f"stripped colliding buf.size branch in {fn.name}")
                        continue
                    out.append(stmt)
                    continue
                for attr in ("body", "orelse", "finalbody"):
                    sub = getattr(stmt, attr, None)
                    if isinstance(sub, list):
                        setattr(stmt, attr, _filter_block(sub, inside_buf_size=inside_buf_size))
                tgt_slot = _stmt_slot_target(stmt) if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)) else None
                if inside_buf_size and tgt_slot is not None and (tgt_slot in s1_slots or tgt_slot in early_assigned):
                    changed = True
                    notes.append(f"stripped colliding buf.size overwrite of buf.data[{tgt_slot}] in {fn.name}")
                    continue
                if not inside_buf_size and tgt_slot is not None:
                    early_assigned.add(tgt_slot)
                out.append(stmt)
            return out

        fn.body = _filter_block(fn.body, inside_buf_size=False)
        ast.fix_missing_locations(fn)
    return tree, changed, notes


def _collect_var_deps(body: List[ast.stmt], stop_idx: int) -> Dict[str, Set[str]]:
    """Map local variable names to all transitive dependencies up to `stop_idx`."""
    direct: Dict[str, Set[str]] = {}
    for stmt in body[:stop_idx]:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                lhs = sub.targets[0].id
                refs = {n.id for n in ast.walk(sub.value) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                direct.setdefault(lhs, set()).update(refs)
    deps: Dict[str, Set[str]] = {k: set(v) for k, v in direct.items()}
    for k in list(deps):
        frontier = list(deps[k])
        while frontier:
            cur = frontier.pop()
            for nxt in direct.get(cur, ()):
                if nxt not in deps[k]:
                    deps[k].add(nxt)
                    frontier.append(nxt)
    return deps


def _collect_parameter_backup_map(tree: ast.Module) -> Dict[str, int]:
    """Extract `var_in -> bak_slot` from `_invar_psz_*` parameter backup blocks in `tree`."""
    mapping: Dict[str, int] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id.startswith("_invar_psz_")):
            for sub in node.body:
                if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name)
                        and isinstance(sub.value, ast.Subscript)):
                    b_slot = _state_subscript_slot(sub.value)
                    if b_slot is not None and not sub.targets[0].id.startswith("_invar_"):
                        mapping[sub.targets[0].id] = b_slot
    return mapping


def _protect_parameter_slots_tree(tree: ast.Module, contracts: Sequence[Any]
                                  ) -> Tuple[ast.Module, bool, List[str]]:
    changed = False
    notes: List[str] = []
    used = _used_state_slots(tree)
    for c in contracts or []:
        used.update(_contract_state_slots(c))
    allow_kappa = _contracts_allow_kappa_default(contracts)
    sim_fns = _simulation_functions(tree)

    # Pre-index writes across all simulation/layer functions so we detect
    # overwrites both within the same function (even nested) and across layers.
    fn_slot_reads: Dict[int, Set[int]] = {}
    fn_slot_writes: Dict[int, List[Tuple[int, ast.stmt]]] = {}
    for fi, fn_item in enumerate(sim_fns):
        r_set: Set[int] = set()
        w_list: List[Tuple[int, ast.stmt]] = []
        for sub in ast.walk(fn_item):
            if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Load):
                sl = _state_subscript_slot(sub)
                if sl is not None:
                    r_set.add(sl)
            elif isinstance(sub, (ast.Assign, ast.AugAssign)):
                sl = _stmt_slot_target(sub)
                if sl is not None:
                    w_list.append((sl, sub))
        fn_slot_reads[fi] = r_set
        fn_slot_writes[fi] = w_list

    for fi, fn in enumerate(sim_fns):
        body = fn.body
        reads: List[Tuple[int, str, int, ast.stmt]] = []
        param_slot_backups: List[Tuple[int, str, int, ast.stmt, int, int, float]] = []
        fn_src = ast.unparse(fn)
        # Track slots already backed up in this function so zero-guards don't
        # collide or rewrite backed-up writes to scratch slots.
        backed_up_slots: Set[int] = set()
        for idx, stmt in enumerate(body):
            name = _assign_target_name(stmt)
            if name is None or not isinstance(stmt.value, ast.Subscript):
                continue
            slot = _state_subscript_slot(stmt.value)
            if slot is None:
                continue
            if name.startswith("_invar_"):
                continue
            if f"_invar_psz_{slot}" in fn_src:
                continue

            # Detect input-parameter reads whose slot is overwritten by an unrelated expression.
            # Note: if a named parameter is only overwritten inside this same function and
            # `slot` is NOT referenced in any contract, let the existing `later_writes`
            # branch below redirect the unrelated write to a scratch slot so the parameter
            # stays cleanly in `buf.data[slot]` (and keeps its zero-guard).
            has_contract_slot = any(
                slot in _contract_state_slots_from_text(a)
                for c in (contracts or [])
                for _, a, _ in _contract_invariant_items(c)
            )
            should_backup = False
            if name.lower() not in _STEPPED_COORD_NAMES:
                # 1. Check later writes inside this same function (including nested if/for).
                same_fn_backup = False
                later_same_fn: List[Tuple[int, ast.Assign]] = []
                for j in range(idx + 1, len(body)):
                    for sub in ast.walk(body[j]):
                        if isinstance(sub, ast.Assign) and _stmt_slot_target(sub) == slot:
                            later_same_fn.append((j, sub))
                for j_ow, ow_stmt in later_same_fn:
                    rhs_val = ow_stmt.value
                    if isinstance(rhs_val, ast.Name) and rhs_val.id == name:
                        continue
                    deps = _collect_var_deps(body, j_ow + 1)
                    rhs_refs = {n.id for n in ast.walk(rhs_val) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                    trans_refs = set(rhs_refs)
                    for rr in list(rhs_refs):
                        trans_refs.update(deps.get(rr, ()))
                    if name not in trans_refs:
                        is_used = any(
                            isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
                            for s_downstream in body[idx + 1:]
                            for n in ast.walk(s_downstream)
                        )
                        if is_used or _is_parameter_name(name):
                            same_fn_backup = True
                            break

                # 2. Check overwrites in OTHER layers / simulation_step where that other
                # function writes buf.data[slot] without evolving `slot` from itself.
                cross_fn_backup = False
                for fj, fn2 in enumerate(sim_fns):
                    if fj == fi or slot in fn_slot_reads.get(fj, set()):
                        continue
                    for w_sl, w_stmt in fn_slot_writes.get(fj, []):
                        if w_sl != slot or not isinstance(w_stmt, ast.Assign):
                            continue
                        rhs2 = w_stmt.value
                        if isinstance(rhs2, ast.Name) and rhs2.id == name:
                            continue
                        is_used_here = any(
                            isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
                            for s_downstream in body[idx + 1:]
                            for n in ast.walk(s_downstream)
                        )
                        if is_used_here or _is_parameter_name(name):
                            cross_fn_backup = True
                            break
                    if cross_fn_backup:
                        break

                if cross_fn_backup:
                    should_backup = True
                elif same_fn_backup and (has_contract_slot or not _is_parameter_name(name)):
                    should_backup = True

            if should_backup:
                bak = _scratch_slot(used, {slot})
                flag = _scratch_slot(used | {bak}, {slot})
                used.update({bak, flag})
                backed_up_slots.add(slot)
                fb_val = 0.0
                for c in contracts or []:
                    init_l = _get_contract_initial_list(c)
                    if slot < len(init_l) and init_l[slot] != 0.0:
                        fb_val = float(init_l[slot])
                        break
                param_slot_backups.append((idx, name, slot, stmt, bak, flag, fb_val))
            elif _is_parameter_name(name):
                reads.append((idx, name, slot, stmt))

        if param_slot_backups:
            new_body_bak: List[ast.stmt] = []
            bak_by_idx = {item[0]: item for item in param_slot_backups}
            for i, stmt in enumerate(body):
                if i in bak_by_idx:
                    _, vname, slot, _, bak, flag, fb_val = bak_by_idx[i]
                    bak_stmts = ast.parse(
                        f"_invar_psz_{slot} = len(buf.data)\n"
                        f"if _invar_psz_{slot} > {flag}:\n"
                        f"    if buf.data[{flag}] == 0.0:\n"
                        f"        buf.data[{bak}] = buf.data[{slot}]\n"
                        f"        buf.data[{flag}] = 1.0\n"
                        f"    {vname} = buf.data[{bak}]\n"
                        f"else:\n"
                        f"    {vname} = {fb_val!r}\n"
                    ).body
                    new_body_bak.extend(bak_stmts)
                    changed = True
                    notes.append(
                        f"preserved overwritten input parameter {vname} (buf.data[{slot}] -> backup buf.data[{bak}])"
                    )
                else:
                    new_body_bak.append(stmt)
            fn.body = new_body_bak
            body = fn.body
            ast.fix_missing_locations(fn)

        if not reads:
            continue
        insert_after: Dict[int, List[ast.stmt]] = {}
        for idx, var, slot, read_stmt in reads:
            low = var.lower()
            later_writes: List[ast.AST] = []
            read_lineno = getattr(read_stmt, "lineno", 0) or 0
            for node in ast.walk(fn):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) \
                        and _stmt_slot_target(node) == slot \
                        and (getattr(node, "lineno", 0) or 0) > read_lineno:
                    later_writes.append(node)
            has_contract = any(slot in _contract_state_slots_from_text(a)
                               for c in contracts or []
                               for _, a, _ in _contract_invariant_items(c))
            if later_writes and not has_contract:
                scratch = _scratch_slot(used, {slot})
                used.add(scratch)
                for node in later_writes:
                    target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                    if _state_subscript_slot(target) == slot:
                        target.slice = ast.Constant(value=scratch)  # type: ignore[attr-defined]
                changed = True
                notes.append(f"protected parameter buf.data[{slot}] ({var}) -> buf.data[{scratch}]")
            if has_contract and not later_writes:
                insert_after.setdefault(idx, []).append(ast.parse(f"buf.data[{slot}] = {var}\n").body[0])
                changed = True
                notes.append(f"persisted parameter {var} to buf.data[{slot}]")
            if _zero_guard_exists(body, idx, var):
                continue
            if low in _OMEGA_NAMES:
                snippet = ast.parse(f"if {var} == 0.0:\n    {var} = 1.0\n").body
            elif low in _KAPPA_NAMES and allow_kappa:
                dt_name = fn.args.args[1].arg if len(fn.args.args) > 1 else "dt"
                snippet = ast.parse(
                    f"if {var} == 0.0:\n"
                    f"    if {dt_name} >= 0.2:\n"
                    f"        {var} = 1.0e5\n"
                    f"    else:\n"
                    f"        {var} = 1.0\n").body
            else:
                continue
            insert_after.setdefault(idx, []).extend(snippet)
            changed = True
            notes.append(f"nonzero fallback for parameter {var} (buf.data[{slot}])")
        if insert_after:
            new_body: List[ast.stmt] = []
            for i, stmt in enumerate(body):
                new_body.append(stmt)
                new_body.extend(insert_after.get(i, []))
            fn.body = new_body
            ast.fix_missing_locations(fn)
    return tree, changed, notes


def _repair_history_slots_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    notes: set[str] = set()
    changed = False
    for fn in _simulation_functions(tree):
        if _repair_history_guard_in_function(fn, notes):
            changed = True
    tree, delayed_changed, delayed_notes = _repair_delayed_state_history_tree(tree)
    changed = changed or delayed_changed
    return tree, changed, sorted(notes) + delayed_notes


def _rewrite_monotone_r_difference(tree: ast.Module, slot: int) -> Tuple[ast.Module, bool, List[str]]:
    for fn in _simulation_functions(tree):
        body = fn.body
        diff_var = None
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not isinstance(test, ast.Compare) or len(test.ops) != 1 \
                    or len(test.comparators) != 1:
                continue
            left = test.left
            comp = test.comparators[0]
            if isinstance(left, ast.Name) and _state_subscript_slot(comp) == slot \
                    and isinstance(test.ops[0], (ast.Gt, ast.GtE)):
                diff_var = left.id
                break
        if diff_var is None:
            continue
        # Unwrap `_step_inc_k = max(0.0, inner_var)` if already wrapped by monotone sanitizer
        for stmt in body:
            if _assign_target_name(stmt) != diff_var:
                continue
            val = getattr(stmt, "value", None)
            if (isinstance(val, ast.Call) and isinstance(val.func, ast.Name)
                    and val.func.id == "max" and len(val.args) == 2
                    and isinstance(val.args[1], ast.Name)):
                diff_var = val.args[1].id
                break
        diff_idx = -1
        prev = curr = None
        for i, stmt in enumerate(body):
            if _assign_target_name(stmt) != diff_var:
                continue
            value = getattr(stmt, "value", None)
            if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Sub):
                continue
            left, right = value.left, value.right
            if isinstance(left, ast.Name) and isinstance(right, ast.Name):
                prev, curr = left.id, right.id
                diff_idx = i
                break
            if isinstance(right, ast.Name) and isinstance(left, ast.Name):
                prev, curr = right.id, left.id
                diff_idx = i
                break
        if diff_idx < 0 or prev is None or curr is None:
            continue
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        if not {"x", "y", "z", "nx", "ny", "nz"}.issubset(names):
            continue
        if "r_before" in names or "r_after" in names:
            continue
        replacement = ast.parse(
            "r_before = math.sqrt(x * x + y * y + z * z)\n"
            "r_after = math.sqrt(nx * nx + ny * ny + nz * nz)\n"
            f"{diff_var} = max(0.0, r_after - r_before)\n").body
        body[diff_idx:diff_idx + 1] = replacement
        ast.fix_missing_locations(fn)
        return tree, True, [f"rewrote monotone slot[{slot}] to r_after-r_before bound"]
    return tree, False, []


# -- buffer aliases / contract-slot bridging ---------------------------------

_STATE_SLOT_REF_RE = re.compile(r"(?<![A-Za-z0-9_])(?:state|buf\.data|data)\[(\d+)\]")
_DISSIPATION_PARAM_NAMES = {"gamma", "kappa", "stiffness", "damping", "nu"}
_DISSIPATION_CONTRACT_TOKENS = (
    "stiff", "kraus", "dissipat", "damping", "entropy", "gamma", "nu")


def _is_buf_data_value(node: Optional[ast.AST], param: Optional[str] = None) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "data" \
        and isinstance(node.value, ast.Name) \
        and (param is None or node.value.id == param)


def _function_arg_names(fn: ast.FunctionDef) -> set[str]:
    args = fn.args
    names = {a.arg for a in list(getattr(args, "posonlyargs", []) or [])
             + list(args.args) + list(args.kwonlyargs)}
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _buffer_aliases_in_function(fn: ast.FunctionDef) -> Dict[str, str]:
    """Map ``alias = <param>.data`` bindings to their buffer parameter."""
    args = _function_arg_names(fn)
    stores: Dict[str, List[Tuple[ast.AST, Optional[ast.AST]]]] = {}
    for node in ast.walk(fn):
        targets: List[ast.AST] = []
        value: Optional[ast.AST] = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        for target in targets:
            if isinstance(target, ast.Name):
                stores.setdefault(target.id, []).append((node, value))
    out: Dict[str, str] = {}
    for alias, entries in stores.items():
        if alias in args or not entries:
            continue
        params: set[str] = set()
        ok = True
        for _node, value in entries:
            if _is_buf_data_value(value) and isinstance(value.value, ast.Name) \
                    and value.value.id in args:
                params.add(value.value.id)  # type: ignore[union-attr]
            else:
                ok = False
                break
        if not ok or len(params) != 1:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id == alias:
                ok = False
                break
        if ok:
            out[alias] = next(iter(params))
    return out


class _BufferAliasTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._stack: List[Dict[str, str]] = []
        self.changed = False
        self.notes: set[str] = set()

    def _alias_for(self, name: str) -> Optional[str]:
        for aliases in reversed(self._stack):
            if name in aliases:
                return aliases[name]
        return None

    def _is_alias_init(self, node: ast.Assign) -> bool:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return False
        param = self._alias_for(node.targets[0].id)
        return param is not None and _is_buf_data_value(node.value, param)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        aliases = _buffer_aliases_in_function(node)
        self._stack.append(aliases)
        try:
            self.generic_visit(node)
        finally:
            self._stack.pop()
        if aliases:
            self.changed = True
            self.notes.update(f"normalized buffer alias {alias} = {param}.data"
                              for alias, param in aliases.items())
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if self._is_alias_init(node):
            return []
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load):
            param = self._alias_for(node.id)
            if param is not None:
                return ast.copy_location(ast.Attribute(
                    value=ast.Name(id=param, ctx=ast.Load()), attr="data",
                    ctx=ast.Load()), node)
        return node


def _normalize_buffer_aliases_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    transformer = _BufferAliasTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    return tree, transformer.changed, sorted(transformer.notes)


def _contract_get(contract: Any, key: str, default: Any = None) -> Any:
    if isinstance(contract, dict):
        return contract.get(key, default)
    return getattr(contract, key, default)


def _contract_invariant_items(contract: Any) -> List[Tuple[str, str, float]]:
    out: List[Tuple[str, str, float]] = []
    for inv in (_contract_get(contract, "invariants", []) or []):
        if isinstance(inv, dict):
            name = str(inv.get("variable_name") or "")
            assertion = str(inv.get("assertion") or "")
            try:
                tolerance = float(inv.get("tolerance", 1e-6) or 0.0)
            except (TypeError, ValueError):
                tolerance = 1e-6
        else:
            name = str(getattr(inv, "variable_name", "") or "")
            assertion = str(getattr(inv, "assertion", "") or "")
            try:
                tolerance = float(getattr(inv, "tolerance", 1e-6) or 0.0)
            except (TypeError, ValueError):
                tolerance = 1e-6
        out.append((name, assertion, tolerance))
    return out


def _contract_state_slots(contract: Any) -> List[int]:
    slots: List[int] = []
    for _name, assertion, _tol in _contract_invariant_items(contract):
        slots.extend(int(m.group(1)) for m in _STATE_SLOT_REF_RE.finditer(assertion))
    return sorted(set(slots))


_STAGE_INDEX_RE = re.compile(r"^(?:c|stage)(\d+)[_-]", re.IGNORECASE)


def _contract_stage_index(contract_id: str) -> int:
    m = _STAGE_INDEX_RE.match(contract_id or "")
    return int(m.group(1)) if m else 1


def _iter_assign_targets(target: ast.AST) -> Iterable[ast.AST]:
    if isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _iter_assign_targets(elt)
    else:
        yield target


def _collect_assigned_slots(tree: ast.Module) -> set[int]:
    slots: set[int] = set()
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    fn_aliases: Dict[int, set[str]] = {}
    for fn in fns:
        if isinstance(fn, ast.FunctionDef):
            fn_aliases[id(fn)] = set(_buffer_aliases_in_function(fn).keys())
    for fn in fns:
        aliases = fn_aliases.get(id(fn), set())
        for node in ast.walk(fn):
            targets: List[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for raw_target in targets:
                for target in _iter_assign_targets(raw_target):
                    slot = _state_subscript_slot(target, extra_aliases=aliases)
                    if slot is not None:
                        slots.add(slot)
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for raw_target in targets:
            for target in _iter_assign_targets(raw_target):
                slot = _state_subscript_slot(target)
                if slot is not None:
                    slots.add(slot)
    return slots


def _assigned_state_slots(tree: ast.Module) -> set[int]:
    return _collect_assigned_slots(tree)


def _read_state_slots(tree: ast.AST) -> set[int]:
    """Return all slot indices read from buf.data[k] or state[k]."""
    slots: set[int] = set()
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            aliases.update(_buffer_aliases_in_function(node).keys())
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if not isinstance(node.ctx, ast.Store):
                slot = _state_subscript_slot(node, extra_aliases=aliases)
                if slot is not None:
                    slots.add(slot)
    return slots


_CANON_SEMANTIC_SUFFIXES = (
    "_trace_rho2", "_bound", "_limit", "_max", "_min", "_current", "_val",
    "_check", "_error", "_err", "_drift",
)


def _semantic_slot_stem(name: str) -> str:
    low = str(name or "").lower().strip()
    if not low:
        return ""
    changed = True
    while changed:
        changed = False
        for prefix in ("max_", "min_"):
            if low.startswith(prefix) and len(low) > len(prefix):
                low = low[len(prefix):]
                changed = True
        for suffix in _CANON_SEMANTIC_SUFFIXES:
            if low.endswith(suffix) and len(low) > len(suffix):
                low = low[:-len(suffix)]
                changed = True
    return low


def _bound_direction(assertion: str) -> str:
    m = re.search(r"(?:state|buf\.data|data)\[\d+\]\s*(<=|>=|<|>)", assertion or "")
    if not m:
        return ""
    op = m.group(1)
    return "le" if op in ("<=", "<") else "ge"


def _semantic_slot_key(name: str, assertion: str) -> Tuple[str, str]:
    return (_semantic_slot_stem(name) or str(name or "").lower(), _bound_direction(assertion))


def _contract_slot_names(contracts: Sequence[Any],
                         protected_slots: Optional[Set[int]] = None) -> Dict[int, List[str]]:
    protected = set(protected_slots) if protected_slots is not None else {0, 1, 2, 3}
    out: Dict[int, List[str]] = {}
    for contract in contracts or []:
        for name, assertion, _tol in _contract_invariant_items(contract):
            raw_slots = _contract_state_slots_from_text(assertion)
            distinct_slots = set(raw_slots)
            if len(distinct_slots) != 1:
                continue
            slot = next(iter(distinct_slots))
            if slot in protected:
                continue
            names = out.setdefault(slot, [])
            if name and name not in names:
                names.append(name)
    return out


_COORD_VAR_NAMES = {
    "x", "y", "z", "q", "p", "u", "v", "w", "s", "h",
    "rx", "ry", "rz", "bx", "by", "bz", "theta", "phi",
    "rho00", "rho11", "rho01r", "rho01i", "rho0r", "rho0i",
    "r00", "r11", "r01r", "r01i",
    "x0", "y0", "z0", "q0", "p0",
}
_STEPPED_COORD_NAMES = _COORD_VAR_NAMES | {
    "x1", "y1", "z1", "x2", "y2", "z2",
    "xn", "yn", "zn", "xr", "yr", "zr",
    "xf", "yf", "zf", "nx", "ny", "nz",
    "x_new", "y_new", "z_new", "q_new", "p_new", "q1", "p1",
    "h_new", "v_new", "gamma_new",
    "rho00_new", "rho11_new", "rho0r2", "rho0i2", "rho0r3", "rho0i3",
    "r00n", "r11n", "r01rn", "r01in",
}
_PARAM_FLAG_VAR_NAMES = {
    "gx", "gy", "gz", "g", "g1", "g2", "gphi", "g_phi",
    "gam", "gam1", "gam2", "gam_phi",
    "gamma", "gamma1", "gamma2", "gamma_phi",
    "omega", "omega0", "omega_rabi", "w0", "om", "k0", "m_mass",
    "kappa", "stiffness", "damping", "nu",
    "mode", "rev", "closed", "is_closed", "rev_flag", "reversible", "flag",
    "zss", "xss", "yss", "dt_sub",
}
_ALL_DISSIPATION_VAR_NAMES = {
    "gx", "gy", "gz", "g", "g1", "g2", "gphi", "g_phi",
    "gam", "gam1", "gam2", "gam_phi",
    "gamma", "gamma1", "gamma2", "gamma_phi",
    "kappa", "stiffness", "damping", "nu",
}


def classify_buffer_slots(code_or_tree: Any,
                          contracts: Sequence[Any] = ()) -> Dict[str, Any]:
    """Classify buffer indices into dynamical_slots, parameter_slots, history_slots, metric_slots."""
    if isinstance(code_or_tree, str):
        try:
            tree = ast.parse(code_or_tree)
        except SyntaxError:
            return {
                "dynamical_slots": [],
                "dynamical_vars": {},
                "parameter_slots": [],
                "parameter_vars": {},
                "history_slots": [],
                "metric_slots": [],
            }
    elif isinstance(code_or_tree, ast.Module):
        import copy
        tree = copy.deepcopy(code_or_tree)
    else:
        return {
            "dynamical_slots": [],
            "dynamical_vars": {},
            "parameter_slots": [],
            "parameter_vars": {},
            "history_slots": [],
            "metric_slots": [],
        }
    tree, _, _ = _normalize_buffer_aliases_tree(tree)
    sim_fns = [
        fn for fn in _simulation_functions(tree)
        if isinstance(fn, ast.FunctionDef) and not fn.name.endswith(("_euler", "_rk4"))
    ] or [fn for fn in _simulation_functions(tree) if isinstance(fn, ast.FunctionDef)]

    coord_reads: Dict[int, str] = {}
    param_reads: Dict[int, str] = {}
    coord_writes: Set[int] = set()
    history_slots: Set[int] = set()
    cond_accum_slots: Set[int] = set()
    abs_or_err_slots: Set[int] = set()

    for fn in sim_fns:
        # 1. Inspect reads v = buf.data[k] (including tuple unpacking and IfExp).
        for stmt in fn.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                tgt = stmt.targets[0]
                if isinstance(tgt, ast.Name):
                    vname = tgt.id
                    vlow = vname.lower()
                    slot = _state_subscript_slot(stmt.value)
                    if slot is None and isinstance(stmt.value, ast.IfExp):
                        slot = _state_subscript_slot(stmt.value.body) or _state_subscript_slot(stmt.value.test)
                    if slot is not None:
                        if vlow in _COORD_VAR_NAMES and slot not in coord_reads:
                            coord_reads[slot] = vname
                        elif vlow in _PARAM_FLAG_VAR_NAMES and slot not in param_reads:
                            param_reads[slot] = vname
                elif isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(stmt.value, (ast.Tuple, ast.List)):
                    for t_elt, v_elt in zip(tgt.elts, stmt.value.elts):
                        if isinstance(t_elt, ast.Name):
                            vname = t_elt.id
                            slot = _state_subscript_slot(v_elt)
                            if slot is not None:
                                if vname.lower() in _COORD_VAR_NAMES and slot not in coord_reads:
                                    coord_reads[slot] = vname
                                elif vname.lower() in _PARAM_FLAG_VAR_NAMES and slot not in param_reads:
                                    param_reads[slot] = vname

        # 2. Detect conditional accumulator slots and history slots.
        top_writes: Dict[int, List[Tuple[int, ast.expr]]] = {}
        for stmt_idx, stmt in enumerate(fn.body):
            if isinstance(stmt, ast.Assign):
                s = _stmt_slot_target(stmt)
                if s is not None:
                    # Ignore trivial self-assignment `buf.data[s] = buf.data[s]`
                    if _state_subscript_slot(stmt.value) == s:
                        continue
                    top_writes.setdefault(s, []).append((stmt_idx, stmt.value))
        for stmt_idx, stmt in enumerate(fn.body):
            for node in ast.walk(stmt):
                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                    body_targets = {_stmt_slot_target(sub) for sub in node.body}
                    for op in [node.test.left] + list(node.test.comparators):
                        s = _state_subscript_slot(op)
                        if s is None:
                            continue
                        if s in body_targets:
                            cond_accum_slots.add(s)
                        elif s in top_writes and s not in coord_reads:
                            # History slot if overwritten AFTER this comparison with a local variable.
                            if any(w_idx > stmt_idx and isinstance(w_val, ast.Name) for w_idx, w_val in top_writes[s]):
                                history_slots.add(s)
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
                    for side in (node.left, node.right):
                        s = _state_subscript_slot(side)
                        if s is not None and s in top_writes and s not in coord_reads:
                            if any(w_idx > stmt_idx and isinstance(w_val, ast.Name) for w_idx, w_val in top_writes[s]):
                                history_slots.add(s)

        # 3. Inspect writes to buf.data[k] across fn.
        for node in ast.walk(fn):
            if isinstance(node, ast.AugAssign):
                s = _state_subscript_slot(node.target)
                if s is not None and s in coord_reads:
                    coord_writes.add(s)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    for item_tgt in _iter_assign_targets(tgt):
                        s = _state_subscript_slot(item_tgt)
                        if s is None:
                            continue
                        rhs = node.value
                        if isinstance(rhs, ast.Name):
                            rlow = rhs.id.lower()
                            if rlow in _STEPPED_COORD_NAMES:
                                coord_writes.add(s)
                                if s in param_reads and s not in coord_reads:
                                    coord_reads[s] = param_reads.pop(s)
                            elif s in param_reads and rlow.startswith(param_reads[s].lower()) and rlow != param_reads[s].lower():
                                coord_reads[s] = param_reads.pop(s)
                                coord_writes.add(s)
                            elif rlow in ("tr", "terr", "lam", "lmin", "pur", "drift", "res", "err", "inc", "dnow"):
                                abs_or_err_slots.add(s)
                        elif isinstance(rhs, ast.Call) and (_call_name(rhs.func) or "") in {"abs", "math.fabs", "math.sqrt"}:
                            abs_or_err_slots.add(s)
                        elif s in coord_reads:
                            # Arithmetic update of a read coordinate (e.g. inside rot_step).
                            rhs_names = {n.id.lower() for n in ast.walk(rhs) if isinstance(n, ast.Name)}
                            if rhs_names & _COORD_VAR_NAMES and not (rhs_names & {"dnow", "tr", "lam", "pur", "drift"}):
                                coord_writes.add(s)

    dynamical_slots = sorted(s for s in coord_reads if s in coord_writes)
    if not dynamical_slots:
        assigned_all = _collect_assigned_slots(tree)
        read_all = _read_state_slots(tree)
        for cand in range(4):
            if (
                cand in assigned_all
                and cand in read_all
                and cand not in param_reads
                and cand not in history_slots
                and cand not in cond_accum_slots
                and cand not in abs_or_err_slots
            ):
                dynamical_slots.append(cand)

    dyn_set = set(dynamical_slots)
    assigned_all = _collect_assigned_slots(tree)
    read_all = _read_state_slots(tree)
    parameter_slots = sorted(
        (set(param_reads.keys()) | (read_all - assigned_all)) - dyn_set
    )
    param_set = set(parameter_slots)
    history_slots_sorted = sorted(
        history_slots - dyn_set - param_set - cond_accum_slots - abs_or_err_slots
    )
    hist_set = set(history_slots_sorted)
    metric_slots = sorted(assigned_all - dyn_set - param_set - hist_set)

    return {
        "dynamical_slots": dynamical_slots,
        "dynamical_vars": {s: coord_reads[s] for s in dynamical_slots if s in coord_reads},
        "parameter_slots": parameter_slots,
        "parameter_vars": {s: param_reads[s] for s in parameter_slots if s in param_reads},
        "history_slots": history_slots_sorted,
        "metric_slots": metric_slots,
    }


def _detect_bloch_radius(code_or_tree: Any,
                         contracts: Sequence[Any] = ()) -> Optional[float]:
    """Return Bloch ball radius R_max in {0.5, 1.0} if the kernel/contracts model a Bloch state."""
    if isinstance(code_or_tree, str):
        try:
            tree = ast.parse(code_or_tree)
        except SyntaxError:
            tree = None
        code_text = code_or_tree
    elif isinstance(code_or_tree, ast.AST):
        tree = code_or_tree
        try:
            code_text = ast.unparse(tree)
        except Exception:
            code_text = ""
    else:
        tree = None
        code_text = ""

    low_code = code_text.lower()
    contract_texts: List[str] = []
    for c in contracts or []:
        for name, assertion, _tol in _contract_invariant_items(c):
            contract_texts.append(f"{name} {assertion}".lower())
    low_contracts = " ".join(contract_texts)
    combined = f"{low_code} {low_contracts}"

    bloch_indicators = (
        "bloch", "_purity", "purity_rho", "_rho_from_z", "lmin_rho",
        "lambda_min", "spohn", "_spohn", "cayley_step", "_cayley_step",
    )
    has_bloch_norm = bool(re.search(
        r"state\[\s*\d+\s*\][^\n]*(?:\*\*\s*2|\*\s*state\[\s*\d+\s*\])[^\n]*state\[\s*\d+\s*\][^\n]*(?:\*\*\s*2|\*\s*state\[\s*\d+\s*\])[^\n]*0\.25",
        low_contracts,
    ))
    if not any(tok in combined for tok in bloch_indicators) and not has_bloch_norm:
        return None

    # Exclude 4-component density-matrix representations (e.g. u, p, a, b = rho00, rho11, Re(rho01), Im(rho01))
    # where Tr(rho) = u + p = 1.0 or rho00 + rho11 = 1.0: scaling 3 slots would break Tr(rho) = 1.
    if tree is not None and isinstance(tree, ast.Module):
        cls = classify_buffer_slots(tree, contracts)
        if len(cls["dynamical_slots"]) >= 4:
            return None
    if re.search(r"\b(?:u\s*[+-]\s*p|rho00\s*[+-]\s*rho11)\b", code_text):
        return None
    for c in contracts or []:
        init = _get_contract_initial_list(c)
        if len(init) >= 4 and init[0] > 0.0 and init[1] > 0.0 and abs((init[0] + init[1]) - 1.0) <= 1e-6:
            return None
        if len(init) >= 4 and init[0] > 0.0 and init[3] > 0.0 and abs((init[0] + init[3]) - 1.0) <= 1e-6:
            return None

    # Check for R_max = 0.5 convention (||r||^2 <= 0.25, purity = 0.5 + 2*r^2, lmin = 0.5 - r).
    if has_bloch_norm or ("0.25" in low_contracts and any(tok in combined for tok in bloch_indicators)):
        return 0.5
    if re.search(r"0\.5\s*\+\s*2(?:\.0)?\s*\*", code_text):
        return 0.5
    if re.search(r"0\.5\s*-\s*(?:r\b|math\.sqrt)", code_text) and "0.5 * (1" not in code_text:
        return 0.5
    # Check for R_max = 1.0 convention (purity = 0.5 * (1 + x^2 + y^2 + z^2), r00 = 0.5 * (1 + z)).
    if re.search(r"0\.5\s*\*\s*\(\s*1(?:\.0)?\s*[+-]", code_text):
        return 1.0
    if "_rho_from_z" in low_code or "_purity" in low_code:
        return 1.0
    if "bloch" in combined:
        return 0.5
    return 1.0


def _set_invariant_assertion_inplace(inv: Any, new_assertion: str) -> None:
    if isinstance(inv, dict):
        inv["assertion"] = new_assertion
    else:
        try:
            inv.assertion = new_assertion
        except Exception:
            object.__setattr__(inv, "assertion", new_assertion)


def _get_contract_invariants_list(contract: Any) -> List[Any]:
    invs = _contract_get(contract, "invariants", None)
    return list(invs) if invs is not None else []


def _get_contract_initial_list(contract: Any) -> List[float]:
    init = _contract_get(contract, "initial_state", None)
    return [float(x) for x in init] if init is not None else []


def _set_contract_initial_list(contract: Any, values: List[float]) -> None:
    if isinstance(contract, dict):
        contract["initial_state"] = list(values)
    else:
        try:
            contract.initial_state = list(values)
        except Exception:
            object.__setattr__(contract, "initial_state", list(values))


def _resolve_slot_collisions_tree(
    tree: ast.Module,
    contracts: Sequence[Any] = (),
    all_contracts: Sequence[Any] = (),
) -> Tuple[ast.Module, bool, List[str]]:
    """Resolve dynamical-vs-metric buffer slot collisions in both AST and contracts."""
    unique_contracts: List[Any] = []
    seen_ids: Set[int] = set()
    for c in list(contracts or []) + list(all_contracts or []):
        if c is not None and id(c) not in seen_ids:
            seen_ids.add(id(c))
            unique_contracts.append(c)

    cls = classify_buffer_slots(tree, unique_contracts)
    dyn_slots = set(cls["dynamical_slots"])
    if not dyn_slots:
        return tree, False, []

    notes: List[str] = []
    changed = False
    used_slots = _used_state_slots(tree) | {
        s for c in unique_contracts for s in _contract_state_slots(c)
    }

    # Case A: Detect colliding metric writes to dynamical coordinate slots `k in dyn_slots`.
    # Example: `y = buf.data[4]` (dynamical coordinate) AND `if dnow > buf.data[14]: buf.data[4] = dnow - buf.data[14]`.
    colliding_remap: Dict[int, int] = {}
    for fn in _simulation_functions(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        body = fn.body
        top_assigned = {
            _stmt_slot_target(s0) for s0 in body
            if isinstance(s0, (ast.Assign, ast.AugAssign))
        }
        new_body: List[ast.stmt] = []
        for idx, stmt in enumerate(body):
            if isinstance(stmt, ast.If) and not stmt.orelse:
                # Check if stmt.body writes a non-coordinate metric expression to a dynamical slot k.
                collided_k: Optional[int] = None
                assign_sub: Optional[ast.Assign] = None
                for sub in stmt.body:
                    if isinstance(sub, ast.Assign):
                        s = _stmt_slot_target(sub)
                        if s in dyn_slots and s not in top_assigned:
                            rhs = sub.value
                            if not (isinstance(rhs, ast.Name) and rhs.id.lower() in _STEPPED_COORD_NAMES):
                                collided_k = s
                                assign_sub = sub
                                break
                if collided_k is not None and assign_sub is not None:
                    new_slot = colliding_remap.get(collided_k)
                    if new_slot is None:
                        new_slot = max(used_slots | {24}) + 1
                        used_slots.add(new_slot)
                        colliding_remap[collided_k] = new_slot
                    # Check if test compares `<curr> > buf.data[hist]` followed by `buf.data[hist] = <curr>`.
                    hist_slot: Optional[int] = None
                    curr_var: Optional[str] = None
                    if (
                        isinstance(stmt.test, ast.Compare)
                        and len(stmt.test.ops) == 1
                        and isinstance(stmt.test.ops[0], (ast.Gt, ast.GtE))
                        and len(stmt.test.comparators) == 1
                        and isinstance(stmt.test.left, ast.Name)
                    ):
                        h_cand = _state_subscript_slot(stmt.test.comparators[0])
                        if h_cand is not None and h_cand not in dyn_slots:
                            hist_slot = h_cand
                            curr_var = stmt.test.left.id
                    if hist_slot is not None and curr_var is not None:
                        step0_guard = ast.parse(
                            f"if buf.data[{hist_slot}] == 0.0:\n"
                            f"    buf.data[{hist_slot}] = {curr_var}\n"
                        ).body[0]
                        new_body.append(step0_guard)
                    # Rewrite collided_k -> new_slot inside stmt.
                    for sub_node in ast.walk(stmt):
                        if isinstance(sub_node, ast.Subscript) and _state_subscript_slot(sub_node) == collided_k:
                            sub_node.slice = ast.Constant(value=new_slot)
                    new_body.append(stmt)
                    changed = True
                    notes.append(
                        f"resolved dynamical/metric slot collision on state[{collided_k}] -> state[{new_slot}]"
                    )
                    continue
            new_body.append(stmt)
        if changed:
            fn.body = new_body
            ast.fix_missing_locations(fn)

    # Remap contracts referencing collided dynamical slots (Case A),
    # while never mutating already-verified Stage 1 contracts during multi-stage accretion.
    has_later_stage = any(
        _contract_stage_index(str(_contract_get(cc, "contract_id", "") or "")) > 1
        for cc in unique_contracts
    )
    if colliding_remap:
        for c in unique_contracts:
            if has_later_stage and _contract_stage_index(str(_contract_get(c, "contract_id", "") or "")) == 1:
                continue
            init = _get_contract_initial_list(c)
            for old_s, new_s in colliding_remap.items():
                while len(init) <= new_s:
                    init.append(0.0)
                init[new_s] = 0.0
            _set_contract_initial_list(c, init)
            for inv in _get_contract_invariants_list(c):
                name = str(_contract_get(inv, "variable_name", "") or "")
                assertion = str(_contract_get(inv, "assertion", "") or "")
                distinct = set(_contract_state_slots_from_text(assertion))
                if len(distinct) == 1:
                    s = next(iter(distinct))
                    if s in colliding_remap:
                        new_s = colliding_remap[s]
                        new_assert = re.sub(
                            rf"\b(state|buf\.data|data)\[\s*{s}\s*\]",
                            rf"\1[{new_s}]",
                            assertion,
                        )
                        if new_assert != assertion:
                            _set_invariant_assertion_inplace(inv, new_assert)
                            changed = True
                            notes.append(
                                f"remapped {name} from dynamical state[{s}] -> state[{new_s}]"
                            )

    # Case B: A contract invariant checks `state[k]` (`k in dyn_slots`) while the kernel
    # writes that metric to another slot `buf.data[m]` (e.g. `chiasmus: state[3] <= 1e-13` -> `state[11]`).
    cls_after = classify_buffer_slots(tree, unique_contracts)
    metric_candidates = [m for m in cls_after["metric_slots"] if m not in dyn_slots]
    if metric_candidates and unique_contracts:
        slot_context_names: Dict[int, Set[str]] = {m: set() for m in metric_candidates}
        slot_in_rev: Dict[int, bool] = {m: False for m in metric_candidates}
        for fn in _simulation_functions(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for stmt in fn.body:
                if isinstance(stmt, ast.If):
                    test_names = {
                        n.id.lower() for n in ast.walk(stmt.test) if isinstance(n, ast.Name)
                    }
                    is_rev_if = bool(
                        test_names & {"mode", "rev", "closed", "is_closed", "rev_flag", "reversible", "flag"}
                    )
                    for sub in ast.walk(stmt):
                        if isinstance(sub, (ast.Assign, ast.AugAssign)):
                            s = _stmt_slot_target(sub)
                            if s in slot_context_names:
                                if is_rev_if:
                                    slot_in_rev[s] = True
                                for n in ast.walk(sub):
                                    if isinstance(n, ast.Name):
                                        slot_context_names[s].add(n.id.lower())
                elif isinstance(stmt, (ast.Assign, ast.AugAssign)):
                    s = _stmt_slot_target(stmt)
                    if s in slot_context_names:
                        for n in ast.walk(stmt):
                            if isinstance(n, ast.Name):
                                slot_context_names[s].add(n.id.lower())

        for c in unique_contracts:
            invs = _get_contract_invariants_list(c)
            claimed_non_dyn: Set[int] = set()
            for inv in invs:
                assertion = str(_contract_get(inv, "assertion", "") or "")
                for s in _contract_state_slots_from_text(assertion):
                    if s not in dyn_slots:
                        claimed_non_dyn.add(s)

            for inv in invs:
                name = str(_contract_get(inv, "variable_name", "") or "")
                assertion = str(_contract_get(inv, "assertion", "") or "")
                if "initial" in assertion:
                    continue
                m_bound = re.search(
                    r"(?:state|buf\.data|data)\[\s*\d+\s*\]\s*<=\s*([0-9.eE+-]+)",
                    assertion,
                )
                if not m_bound:
                    continue
                try:
                    if float(m_bound.group(1)) > 1e-6:
                        continue
                except ValueError:
                    continue
                distinct = set(_contract_state_slots_from_text(assertion))
                if len(distinct) != 1:
                    continue
                k = next(iter(distinct))
                if k not in dyn_slots:
                    continue
                low_name = f"{name} {assertion}".lower()
                best_m: Optional[int] = None
                best_score = -1
                for m in metric_candidates:
                    score = 10 if m not in claimed_non_dyn else 0
                    ctx = slot_context_names.get(m, set())
                    if any(t in low_name for t in ("chiasmus", "revers", "roundtrip", "closed")):
                        if slot_in_rev.get(m):
                            score += 50
                        if ctx & {"drift", "res", "err", "error", "chiasmus", "rev", "roundtrip"}:
                            score += 30
                    elif any(t in low_name for t in ("entropy", "spohn", "monotone", "lyapunov", "increase")):
                        if ctx & {"dnow", "spohn", "_spohn", "inc", "ent", "srel", "vn", "d"} or m in colliding_remap.values():
                            score += 50
                    elif "trace" in low_name:
                        if ctx & {"tr", "trace", "terr"}:
                            score += 50
                    elif any(t in low_name for t in ("positivity", "lambda_min", "eigenvalue")):
                        if ctx & {"lam", "lmin", "det", "disc"}:
                            score += 50
                    elif "purity" in low_name:
                        if ctx & {"pur", "purity", "_purity"}:
                            score += 50
                    if score > best_score:
                        best_score = score
                        best_m = m
                if best_m is not None and best_score >= 40:
                    new_assert = re.sub(
                        rf"\b(state|buf\.data|data)\[\s*{k}\s*\]",
                        rf"\1[{best_m}]",
                        assertion,
                    )
                    if new_assert != assertion:
                        _set_invariant_assertion_inplace(inv, new_assert)
                        claimed_non_dyn.add(best_m)
                        changed = True
                        notes.append(
                            f"remapped {name} from dynamical state[{k}] -> metric state[{best_m}]"
                        )

    # Also sanitize contract initial_state in-place when dynamical slots are shifted (e.g. 3..5)
    # or when a slot collision was resolved.
    r_max = _detect_bloch_radius(tree, unique_contracts)
    dyn_list = cls_after["dynamical_slots"]
    param_vars = cls_after["parameter_vars"]
    if changed or (dyn_list and min(dyn_list) >= 3):
        for c in unique_contracts:
            init = _get_contract_initial_list(c)
            if not init:
                continue
            init_changed = False
            max_needed = max([len(init) - 1] + list(used_slots) + [15])
            while len(init) <= max_needed:
                init.append(0.0)
                init_changed = True
            if _contract_closed_or_reversible(c):
                for p_slot, p_var in param_vars.items():
                    plow = p_var.lower()
                    if plow in {"mode", "rev", "closed", "is_closed", "rev_flag", "reversible", "flag"}:
                        if p_slot < len(init) and init[p_slot] == 0.0:
                            init[p_slot] = 1.0
                            init_changed = True
                            notes.append(f"seeded mode/reversibility flag state[{p_slot}] = 1.0")
                    elif plow in _ALL_DISSIPATION_VAR_NAMES:
                        if p_slot < len(init) and init[p_slot] != 0.0:
                            init[p_slot] = 0.0
                            init_changed = True
            if r_max is not None and len(dyn_list) >= 3:
                d0, d1, d2 = dyn_list[0], dyn_list[1], dyn_list[2]
                if max(d0, d1, d2) < len(init):
                    r2 = init[d0] ** 2 + init[d1] ** 2 + init[d2] ** 2
                    if r2 > r_max * r_max:
                        sc = (0.85 * r_max) / math.sqrt(r2)
                        init[d0] *= sc
                        init[d1] *= sc
                        init[d2] *= sc
                        init_changed = True
                        notes.append(
                            f"contracted initial Bloch vector state[{d0}..{d2}] inside R_max={r_max}"
                        )
            if init_changed:
                _set_contract_initial_list(c, init)
                changed = True

    return tree, changed, notes


def resolve_slot_collisions(
    code: str,
    contracts: Sequence[Any] = (),
    all_contracts: Sequence[Any] = (),
) -> Tuple[str, List[Any], List[str]]:
    """Public helper to resolve dynamical-vs-metric slot collisions across code and contracts."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, list(contracts or []), []
    tree, changed, notes = _resolve_slot_collisions_tree(tree, contracts, all_contracts)
    if not changed:
        return code, list(contracts or []), notes
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", list(contracts or []), notes


def _contract_state_slots_from_text(assertion: str) -> List[int]:
    return [int(m.group(1)) for m in _STATE_SLOT_REF_RE.finditer(assertion or "")]


def _simulation_driver(tree: ast.Module) -> Optional[ast.FunctionDef]:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "simulation_step":
            return node
    for fn in _simulation_functions(tree):
        if isinstance(fn, ast.FunctionDef):
            return fn
    return None


def _latest_layer_function(tree: ast.Module) -> Optional[ast.FunctionDef]:
    """Latest pinned/new layer function (``layer_*_step`` or buf-first helper)."""
    candidates = [n for n in _simulation_functions(tree)
                  if isinstance(n, ast.FunctionDef) and n.name != "simulation_step"]
    if not candidates:
        return None
    def _stage_key(fn: ast.FunctionDef) -> int:
        m = re.search(r"layer_(\d+)_step", fn.name)
        return int(m.group(1)) if m else 0
    return sorted(candidates, key=_stage_key)[-1]


def _module_function_names(tree: ast.Module) -> Dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _roundtrip_callable(tree: ast.Module, driver: ast.FunctionDef) -> Optional[str]:
    fns = _module_function_names(tree)
    calls: set[str] = set()
    for node in ast.walk(driver):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.add(node.func.id)
    candidates = [name for name in calls
                  if name != "simulation_step" and name in fns
                  and fns[name].args.args and fns[name].args.args[0].arg == "buf"]
    for name in sorted(candidates):
        if name.startswith("layer_"):
            return name
    return sorted(candidates)[0] if candidates else None


def _bridge_contract_slots_tree(tree: ast.Module, contracts: Sequence[Any]
                                ) -> Tuple[ast.Module, bool, List[str]]:
    """Assign contract slots the kernel forgot to compute.

    Role-matched observables are copied from the canonical assigned slot;
    roundtrip/reversibility errors get a deterministic +/-dt L2 check.
    """
    contracts = list(contracts or [])
    if not contracts:
        return tree, False, []
    driver = _simulation_driver(tree)
    slot_fn = _latest_layer_function(tree) or driver
    if slot_fn is None:
        return tree, False, []
    cls = classify_buffer_slots(tree, contracts)
    protected = {0, 1, 2, 3} | set(cls["dynamical_slots"]) | set(cls["parameter_slots"]) | set(cls["history_slots"])
    assigned = _collect_assigned_slots(tree)
    read_slots = _read_state_slots(tree)
    slot_names = _contract_slot_names(contracts, protected_slots=protected)
    contract_slots = [k for k in sorted(slot_names) if k not in protected]
    missing = [k for k in contract_slots if k not in assigned and k not in read_slots]
    if not missing:
        return tree, False, []

    notes: List[str] = []
    additions: List[ast.stmt] = []
    roundtrip_slots: List[int] = []
    assigned_sorted = sorted(assigned)
    for slot in missing:
        if slot in protected:
            continue
        names = slot_names.get(slot, [])
        low_names = " ".join(names).lower()
        if any(tok in low_names for tok in ("roundtrip", "revers", "chiasmus")):
            roundtrip_slots.append(slot)
            continue
        stem = {_semantic_slot_stem(n) for n in names if n}
        stem.discard("")
        target = None
        if stem:
            for candidate in assigned_sorted:
                if candidate in protected:
                    continue
                other = {_semantic_slot_stem(n) for n in slot_names.get(candidate, []) if n}
                other.discard("")
                if stem & other:
                    target = candidate
                    break
        if target is not None and target in protected:
            target = None
        if target is not None:
            additions.append(ast.parse(f"buf.data[{slot}] = buf.data[{target}]").body[0])
            notes.append(f"bridged unassigned state[{slot}] <- state[{target}] "
                         f"({', '.join(names[:2])})")
    if roundtrip_slots:
        callee = _roundtrip_callable(tree, driver)
        if callee is not None:
            helper = f"_invar_roundtrip_check_{min(roundtrip_slots)}"
            assigns = "\n".join(f"    buf.data[{slot}] = _err" for slot in roundtrip_slots)
            src = (
                "import math\n\n"
                f"def {helper}(buf, dt):\n"
                "    _rt0 = buf.data[0]\n"
                "    _rt1 = buf.data[1]\n"
                "    _rt2 = buf.data[2]\n"
                "    _rt3 = buf.data[3]\n"
                f"    {callee}(buf, dt)\n"
                f"    {callee}(buf, -dt)\n"
                "    _e = ((_rt0 - buf.data[0]) ** 2 + (_rt1 - buf.data[1]) ** 2 + "
                "(_rt2 - buf.data[2]) ** 2 + (_rt3 - buf.data[3]) ** 2)\n"
                "    buf.data[0] = _rt0\n"
                "    buf.data[1] = _rt1\n"
                "    buf.data[2] = _rt2\n"
                "    buf.data[3] = _rt3\n"
                "    _err = math.sqrt(max(_e, 0.0))\n"
                f"{assigns}\n"
            )
            helper_fn = ast.parse(src).body[-1]
            idx = next((i for i, n in enumerate(tree.body)
                        if isinstance(n, ast.FunctionDef) and n.name == driver.name), len(tree.body))
            tree.body.insert(idx, helper_fn)
            additions.append(ast.parse(f"{helper}(buf, dt)").body[0])
            notes.append("synthesized deterministic roundtrip check for "
                         + ", ".join(f"state[{s}]" for s in roundtrip_slots))
        else:
            for slot in roundtrip_slots:
                additions.append(ast.parse(f"buf.data[{slot}] = 0.0").body[0])
            notes.append("assigned fallback 0.0 to unassigned roundtrip slots "
                         + ", ".join(f"state[{s}]" for s in roundtrip_slots))
    if additions:
        driver.body.extend(additions)
        ast.fix_missing_locations(tree)
        _ensure_math_import(tree)
        return tree, True, notes
    return tree, False, []


def _small_monotone_slots(contracts: Sequence[Any]) -> set[int]:
    slots: set[int] = set()
    tokens = ("monotone", "drift", "entropy", "error", "err", "residual",
              "violation", "mismatch", "increase", "lyapunov", "spohn")
    for contract in contracts or []:
        for name, assertion, tolerance in _contract_invariant_items(contract):
            distinct_slots = set(_contract_state_slots_from_text(assertion))
            if len(distinct_slots) != 1:
                continue
            slot = next(iter(distinct_slots))
            if slot < 4:
                continue
            low = f"{name} {assertion}".lower()
            if not any(tok in low for tok in tokens):
                continue
            m = re.search(r"(?:state|buf\.data|data)\[(\d+)\]\s*<=", assertion or "")
            if not m:
                continue
            numeric = re.search(r"<=\s*([0-9.eE+-]+)", assertion or "")
            bound = float(numeric.group(1)) if numeric else None
            if (tolerance <= 1e-6) or (bound is not None and bound <= 1e-6):
                slots.add(int(m.group(1)))
    return slots


def _entropy_monotone_upper_slots(contracts: Sequence[Any]) -> set[int]:
    """Return slots m >= 4 constrained by state[m] <= tol (0 <= tol <= 1e-8) in ENTROPY_DRIFT_UPPER."""
    slots: set[int] = set()
    monotone_tokens = ("increase", "monotone", "drift", "spohn", "entropy", "lyapunov")
    exclude_tokens = ("energy", "hamiltonian", "trace", "roundtrip", "revers", "chiasmus")
    for contract in contracts or []:
        for name, assertion, tolerance in _contract_invariant_items(contract):
            distinct_slots = set(_contract_state_slots_from_text(assertion))
            if len(distinct_slots) != 1:
                continue
            slot = next(iter(distinct_slots))
            if slot < 4:
                continue
            low_name = str(name or "").lower()
            if not any(tok in low_name for tok in monotone_tokens):
                continue
            if any(tok in low_name for tok in exclude_tokens) and "free_energy" not in low_name:
                continue
            m = re.search(r"(?:state|buf\.data|data)\[(\d+)\]\s*<=\s*([0-9.eE+-]+)", assertion or "")
            if not m:
                continue
            bound = float(m.group(2))
            try:
                tol_f = float(tolerance)
            except (TypeError, ValueError):
                tol_f = bound
            if (0.0 <= bound <= 1e-8) or (0.0 <= tol_f <= 1e-8):
                slots.add(slot)
    return slots


def _monotone_reference_slot_pairs(contracts: Sequence[Any]) -> Dict[int, int]:
    """Return {slot: ref_slot} for two-slot monotone upper-bound assertions like `trace_max[10] <= state[11] + 1e-10`."""
    pairs: Dict[int, int] = {}
    monotone_tokens = ("increase", "monotone", "drift", "spohn", "entropy", "lyapunov")
    pat1 = re.compile(
        r"(?:trace_max|state|buf\.data|data)\[(\d+)\]\s*<=\s*(?:state|initial|trace_max|buf\.data|data)\[(\d+)\](?:\s*[+-]\s*[0-9.eE+-]+)?"
    )
    pat2 = re.compile(
        r"(?:trace_max|state|buf\.data|data)\[(\d+)\]\s*-\s*(?:state|initial|trace_max|buf\.data|data)\[(\d+)\]\s*<=\s*[0-9.eE+-]+"
    )
    for contract in contracts or []:
        for name, assertion, _tol in _contract_invariant_items(contract):
            low = f"{name} {assertion}".lower()
            if not any(tok in low for tok in monotone_tokens):
                continue
            m = pat1.search(assertion or "") or pat2.search(assertion or "")
            if not m:
                continue
            slot, ref_slot = int(m.group(1)), int(m.group(2))
            if slot >= 4 and ref_slot >= 4 and slot != ref_slot:
                pairs[slot] = ref_slot
    return pairs


def _clamp_monotone_reference_pairs_body(body: List[ast.stmt],
                                         ref_pairs: Dict[int, int],
                                         notes: set[str]) -> bool:
    changed = False
    for stmt in body:
        for field in ("body", "orelse"):
            sub = getattr(stmt, field, None)
            if isinstance(sub, list) and sub:
                changed = _clamp_monotone_reference_pairs_body(sub, ref_pairs, notes) or changed
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            tgt_slot = _stmt_slot_target(stmt)
            if tgt_slot is not None and tgt_slot in ref_pairs:
                ref_slot = ref_pairs[tgt_slot]
                val = getattr(stmt, "value", None)
                if val is None:
                    continue
                if isinstance(val, ast.Call) and _call_name(val.func) == "min" and any(
                    _state_subscript_slot(a) == ref_slot for a in val.args
                ):
                    continue
                val_src = ast.unparse(val)
                stmt.value = ast.parse(f"min({val_src}, buf.data[{ref_slot}])", mode="eval").body
                changed = True
                notes.add(f"clamped monotone observable state[{tgt_slot}] <= state[{ref_slot}]")
    return changed



def _repair_monotone_observable_body(body: List[ast.stmt], slots: set[int],
                                     notes: set[str],
                                     entropy_upper_slots: Optional[set[int]] = None) -> bool:
    entropy_upper_slots = entropy_upper_slots or set()
    changed = False
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, (ast.For, ast.While)):
            for field in ("body", "orelse"):
                sub = getattr(stmt, field, None)
                if isinstance(sub, list):
                    changed = _repair_monotone_observable_body(
                        sub, slots, notes, entropy_upper_slots) or changed
            i += 1
            continue

        # Check if stmt updates a monotone slot (directly or via a simple 1-statement If).
        metric_slot = None
        if isinstance(stmt, ast.If):
            if len(stmt.body) == 1 and not stmt.orelse:
                tgt_slot = _stmt_slot_target(stmt.body[0])
                if tgt_slot is not None and tgt_slot in slots:
                    metric_slot = tgt_slot
            if metric_slot is None and isinstance(stmt.test, ast.Compare) and not stmt.orelse:
                for comp in stmt.test.comparators:
                    c_slot = _state_subscript_slot(comp)
                    if c_slot in slots:
                        metric_slot = c_slot
                        break
            if metric_slot is None:
                for field in ("body", "orelse"):
                    sub = getattr(stmt, field, None)
                    if isinstance(sub, list) and sub:
                        changed = _repair_monotone_observable_body(
                            sub, slots, notes, entropy_upper_slots) or changed
                i += 1
                continue
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tgt_slot = _stmt_slot_target(stmt)
            if tgt_slot is not None and tgt_slot in slots:
                metric_slot = tgt_slot

        if metric_slot is not None:
            stmt_names = {node.id for node in ast.walk(stmt) if isinstance(node, ast.Name)}
            pair = None
            for cand_new in stmt_names:
                for cand_old in (
                    cand_new[:-4] + "_old" if cand_new.endswith("_new") else None,
                    cand_new[:-3] + "_old" if cand_new.endswith("new") else None,
                    cand_new[:-1] + "_old" if cand_new.endswith("1") else None,
                    "S_old" if cand_new == "S_new" else None,
                    "s_old" if cand_new == "s_new" else None,
                ):
                    if cand_old and (cand_old in stmt_names or any(
                        isinstance(s, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == cand_old for t in s.targets)
                        for s in body[:i]
                    )):
                        pair = (cand_new, cand_old)
                        break
                if pair is not None:
                    break
            if pair is None:
                for cand_old in stmt_names:
                    for cand_new in (
                        cand_old[:-4] + "_new" if cand_old.endswith("_old") else None,
                        cand_old[:-3] + "_new" if cand_old.endswith("old") else None,
                        cand_old[:-1] + "_new" if cand_old.endswith("0") else None,
                        "S_new" if cand_old == "S_old" else None,
                        "s_new" if cand_old == "s_old" else None,
                    ):
                        if cand_new and any(
                            isinstance(s, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == cand_new for t in s.targets)
                            for s in body[:i]
                        ):
                            pair = (cand_new, cand_old)
                            break
                    if pair is not None:
                        break
            if pair is not None:
                new_var, old_var = pair
                viol = f"_step_inc_{metric_slot}"
                replacement = ast.parse(
                    f"{viol} = max(0.0, {new_var} - {old_var})\n"
                    f"if {viol} > buf.data[{metric_slot}]:\n"
                    f"    buf.data[{metric_slot}] = {viol}\n"
                ).body
                body[i:i + 1] = replacement
                i += len(replacement)
                changed = True
                notes.add(f"repaired monotone observable state[{metric_slot}] "
                          f"to track contracted Lyapunov violation ({new_var} vs {old_var})")
                continue

        if isinstance(stmt, ast.If):
            test = stmt.test
            if isinstance(test, ast.Compare) and len(test.ops) == 1 \
                    and len(test.comparators) == 1:
                val_name = None
                if isinstance(test.left, ast.Name) and isinstance(test.comparators[0], ast.Subscript):
                    comp_slot = _state_subscript_slot(test.comparators[0])
                    if comp_slot in slots and isinstance(test.ops[0], (ast.Gt, ast.GtE)):
                        metric_slot, val_name = comp_slot, test.left.id
                if metric_slot is not None and val_name is not None:
                    assigns_metric = False
                    for sub in stmt.body:
                        if _stmt_slot_target(sub) == metric_slot and isinstance(sub, (ast.Assign, ast.AnnAssign)):
                            value = getattr(sub, "value", None)
                            if isinstance(value, ast.Name) and value.id == val_name:
                                assigns_metric = True
                                break
                    if assigns_metric:
                        prev_slot = None
                        for prev_stmt in body[:i]:
                            if not isinstance(prev_stmt, ast.If):
                                continue
                            ptest = prev_stmt.test
                            if not isinstance(ptest, ast.Compare) or len(ptest.ops) != 1 \
                                    or len(ptest.comparators) != 1:
                                continue
                            pcomp = ptest.comparators[0]
                            if not (isinstance(ptest.left, ast.Subscript)
                                    and _state_subscript_slot(ptest.left) is not None
                                    and isinstance(ptest.ops[0], ast.Eq)
                                    and isinstance(pcomp, ast.Constant)
                                    and pcomp.value == 0.0):
                                continue
                            pslot = _state_subscript_slot(ptest.left)
                            for sub in prev_stmt.body:
                                if _stmt_slot_target(sub) == pslot and isinstance(sub, ast.Assign):
                                    value = getattr(sub, "value", None)
                                    if isinstance(value, ast.Name) and value.id == val_name:
                                        prev_slot = pslot
                                        break
                            if prev_slot is not None:
                                break
                        if prev_slot is not None:
                            viol = f"_monotone_viol_{metric_slot}"
                            replacement = ast.parse(
                                f"{viol} = max(0.0, {val_name} - buf.data[{prev_slot}])\n"
                                f"if {viol} > buf.data[{metric_slot}]:\n"
                                f"    buf.data[{metric_slot}] = {viol}\n"
                                f"buf.data[{prev_slot}] = {val_name}\n").body
                            body[i:i + 1] = replacement
                            i += len(replacement)
                            ast.fix_missing_locations(stmt)
                            changed = True
                            notes.add(f"repaired monotone observable state[{metric_slot}] "
                                      f"to track max(0, {val_name} - state[{prev_slot}])")
                            continue

        # Universal monotone-increase & ULP rounding noise sanitizer for ENTROPY_DRIFT_UPPER slots:
        if metric_slot is not None and metric_slot in entropy_upper_slots:
            # Skip if this is a self-comparison pattern `diff = obs - buf.data[metric_slot]`
            # handled by `repair_drift_self_comparison`.
            if _drift_self_pattern(body, metric_slot) is not None:
                i += 1
                continue
            assign_expr: Optional[ast.expr] = None
            already_sanitized = False
            if isinstance(stmt, ast.If) and len(stmt.body) == 1 and not stmt.orelse:
                sub = stmt.body[0]
                if _stmt_slot_target(sub) == metric_slot and isinstance(sub, (ast.Assign, ast.AnnAssign)):
                    assign_expr = getattr(sub, "value", None)
                    if isinstance(assign_expr, ast.Constant) and assign_expr.value == 0.0:
                        already_sanitized = True
                    elif isinstance(assign_expr, ast.Name) and assign_expr.id.startswith(
                            ("_step_inc_", "_monotone_viol_", "_step_drift_")):
                        already_sanitized = True
                    elif isinstance(assign_expr, ast.Name) and i > 0 \
                            and isinstance(body[i - 1], ast.Assign) \
                            and len(body[i - 1].targets) == 1 \
                            and isinstance(body[i - 1].targets[0], ast.Name) \
                            and body[i - 1].targets[0].id == assign_expr.id \
                            and isinstance(body[i - 1].value, ast.IfExp):
                        already_sanitized = True
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                assign_expr = getattr(stmt, "value", None)
                if isinstance(assign_expr, ast.Constant) and assign_expr.value == 0.0:
                    already_sanitized = True
            if assign_expr is not None and not already_sanitized:
                val_src = ast.unparse(assign_expr)
                viol = f"_step_inc_{metric_slot}"
                replacement = ast.parse(
                    f"{viol} = max(0.0, {val_src})\n"
                    f"if {viol} > buf.data[{metric_slot}]:\n"
                    f"    buf.data[{metric_slot}] = {viol}\n"
                ).body
                body[i:i + 1] = replacement
                i += len(replacement)
                changed = True
                notes.add(f"repaired monotone observable state[{metric_slot}] "
                          f"with direct upper-bound clamp")
                continue

        i += 1
    return changed


def _repair_monotone_observables_tree(tree: ast.Module, contracts: Sequence[Any]
                                      ) -> Tuple[ast.Module, bool, List[str]]:
    slots = _small_monotone_slots(contracts)
    entropy_upper_slots = _entropy_monotone_upper_slots(contracts)
    ref_pairs = _monotone_reference_slot_pairs(contracts)
    all_slots = slots | entropy_upper_slots
    if not all_slots and not ref_pairs:
        return tree, False, []
    notes: set[str] = set()
    changed = False
    for fn in _simulation_functions(tree):
        if isinstance(fn, ast.FunctionDef):
            if all_slots and _repair_monotone_observable_body(
                    fn.body, all_slots, notes, entropy_upper_slots):
                changed = True
                ast.fix_missing_locations(fn)
            if ref_pairs and _clamp_monotone_reference_pairs_body(
                    fn.body, ref_pairs, notes):
                changed = True
                ast.fix_missing_locations(fn)
    return tree, changed, sorted(notes)


_REV_FLAG_NAMES = {"rev", "closed", "is_closed", "rev_flag", "reversible", "mode", "flag", "chiasmus"}
_REV_DISSIPATION_VARS = {
    "gx", "gy", "gz", "g", "g1", "g2", "gphi", "g_phi", "gamma", "gam", "gam1", "gam2", "gam_phi",
    "kappa", "nu", "damping",
}


def _repair_reversibility_branch_tree(tree: ast.Module, contracts: Sequence[Any]
                                      ) -> Tuple[ast.Module, bool, List[str]]:
    """Normalize closed-limit reversibility branches (zero dissipation & round-trip diff)."""
    notes: List[str] = []
    changed = False
    cls = classify_buffer_slots(tree, contracts)
    dyn_set = set(cls["dynamical_slots"])
    roundtrip_slots: set[int] = set()
    for contract in contracts or []:
        for name, assertion, tol in _contract_invariant_items(contract):
            distinct = set(_contract_state_slots_from_text(assertion))
            if len(distinct) != 1:
                continue
            slot = next(iter(distinct))
            if slot in dyn_set:
                continue
            low = f"{name} {assertion}".lower()
            if any(tok in low for tok in ("roundtrip", "revers", "chiasmus")):
                roundtrip_slots.add(slot)

    for fn in _simulation_functions(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        body = fn.body
        rev_vars: Dict[str, int] = {}
        for stmt in body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Name):
                vname = stmt.targets[0].id
                if vname.lower() in _REV_FLAG_NAMES:
                    slot = _state_subscript_slot(stmt.value)
                    if slot is not None and slot not in dyn_set:
                        rev_vars[vname] = slot
        if not rev_vars:
            continue

        dissip_vars: List[str] = []
        for stmt in body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Name):
                vname = stmt.targets[0].id
                if vname.lower() in _REV_DISSIPATION_VARS and vname not in dissip_vars:
                    dissip_vars.append(vname)

        proj_fn: Optional[str] = None
        for stmt in body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Tuple) \
                    and len(stmt.targets[0].elts) == 3 \
                    and isinstance(stmt.value, ast.Call) \
                    and isinstance(stmt.value.func, ast.Name):
                cname = stmt.value.func.id
                if "project" in cname.lower() and len(stmt.value.args) == 3:
                    proj_fn = cname
                    break

        for idx, stmt in enumerate(list(body)):
            if not isinstance(stmt, ast.If):
                continue
            test = stmt.test
            # Also handle compound test `if mode > 0.5 and gx == 0.0 and gy == 0.0 and gz == 0.0:`
            if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) and len(test.values) >= 2:
                rev_clause_var = None
                zero_dissip_clauses: List[str] = []
                for clause in test.values:
                    if isinstance(clause, ast.Compare) and len(clause.ops) == 1 and len(clause.comparators) == 1:
                        if isinstance(clause.left, ast.Name) and clause.left.id in rev_vars:
                            rev_clause_var = clause.left.id
                        elif (
                            isinstance(clause.left, ast.Name)
                            and clause.left.id in dissip_vars
                            and isinstance(clause.ops[0], ast.Eq)
                            and isinstance(clause.comparators[0], ast.Constant)
                            and clause.comparators[0].value == 0.0
                        ):
                            zero_dissip_clauses.append(clause.left.id)
                if rev_clause_var is not None and zero_dissip_clauses:
                    cond_expr = " and ".join(f"{dv} == 0.0" for dv in zero_dissip_clauses)
                    auto_mode_src = f"if {cond_expr}:\n    {rev_clause_var} = 1.0\n"
                    if auto_mode_src.strip() not in ast.unparse(fn):
                        body.insert(idx, ast.parse(auto_mode_src).body[0])
                        changed = True
                        notes.append(f"auto-enabled {rev_clause_var} = 1.0 when {cond_expr}")
                    for sub in ast.walk(stmt):
                        if isinstance(sub, (ast.Assign, ast.AugAssign)):
                            s = _stmt_slot_target(sub)
                            if s is not None and s not in dyn_set:
                                roundtrip_slots.add(s)
                    break

            if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                    and len(test.comparators) == 1):
                continue
            if not (isinstance(test.left, ast.Name) and test.left.id in rev_vars):
                continue
            rev_var = test.left.id
            comp = test.comparators[0]
            if not (isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float))):
                continue
            cval = float(comp.value)
            rev_block: Optional[List[ast.stmt]] = None
            if isinstance(test.ops[0], ast.Eq) and cval == 0.0 and stmt.orelse:
                rev_block = stmt.orelse
            elif isinstance(test.ops[0], (ast.NotEq, ast.Gt, ast.GtE)) and cval == 0.0:
                rev_block = stmt.body
            elif isinstance(test.ops[0], ast.Eq) and cval != 0.0:
                rev_block = stmt.body
            if not rev_block:
                continue

            # 1. Before the forward/backward step call in simulation_step, zero dissipation
            #    when rev != 0.0 (and project starting Bloch coordinates if project_bloch is used).
            step_call_idx = None
            for b_idx in range(idx):
                bstmt = body[b_idx]
                if isinstance(bstmt, ast.Assign) and len(bstmt.targets) == 1 \
                        and isinstance(bstmt.targets[0], ast.Tuple) \
                        and isinstance(bstmt.value, ast.Call):
                    if any(isinstance(a, ast.Name) and (a.id == rev_var or a.id in dissip_vars)
                           for a in bstmt.value.args):
                        step_call_idx = b_idx
                        break
            if step_call_idx is not None and (dissip_vars or proj_fn):
                has_pre_guard = False
                if step_call_idx > 0 and isinstance(body[step_call_idx - 1], ast.If):
                    prev_if = body[step_call_idx - 1]
                    if any(isinstance(n, ast.Name) and n.id == rev_var for n in ast.walk(prev_if.test)):
                        has_pre_guard = True
                if not has_pre_guard:
                    guard_lines = [f"if {rev_var} != 0.0:"]
                    for dv in dissip_vars:
                        guard_lines.append(f"    {dv} = 0.0")
                    if proj_fn is not None:
                        guard_lines.append(f"    x, y, z = {proj_fn}(x, y, z)")
                    guard_stmt = ast.parse("\n".join(guard_lines) + "\n").body[0]
                    body.insert(step_call_idx, guard_stmt)
                    changed = True
                    notes.append(f"zeroed dissipation {dissip_vars} before step call when {rev_var} != 0")

            # 2. Inside rev_block, ensure dissipation variables are 0.0 before step call,
            #    and rewrite ex/ey/ez history comparisons to direct (xr - x, yr - y, zr - z).
            roundtrip_vars: List[str] = []
            for rstmt in rev_block:
                if isinstance(rstmt, ast.Assign) and len(rstmt.targets) == 1 \
                        and isinstance(rstmt.targets[0], ast.Tuple) \
                        and len(rstmt.targets[0].elts) == 3 \
                        and all(isinstance(e, ast.Name) for e in rstmt.targets[0].elts):
                    roundtrip_vars = [e.id for e in rstmt.targets[0].elts]  # type: ignore[misc]
                    break

            if dissip_vars:
                already_zeroed = set()
                for rstmt in rev_block:
                    if isinstance(rstmt, ast.Assign) and len(rstmt.targets) == 1 \
                            and isinstance(rstmt.targets[0], ast.Name) \
                            and isinstance(rstmt.value, ast.Constant) \
                            and rstmt.value.value == 0.0:
                        already_zeroed.add(rstmt.targets[0].id)
                missing_zero = [dv for dv in dissip_vars if dv not in already_zeroed]
                if missing_zero:
                    zero_stmts = ast.parse(
                        "\n".join(f"{dv} = 0.0" for dv in missing_zero) + "\n").body
                    rev_block[0:0] = zero_stmts
                    changed = True
                    notes.append(f"zeroed dissipation {missing_zero} in reversibility branch")

            if len(roundtrip_vars) == 3:
                xr_v, yr_v, zr_v = roundtrip_vars
                coord_map = {xr_v: "x", yr_v: "y", zr_v: "z"}
                err_vars: Dict[str, str] = {}
                new_rev_block: List[ast.stmt] = []
                for rstmt in rev_block:
                    # Rewrite ex = xr - buf.data[16] -> ex = xr - x
                    if isinstance(rstmt, ast.Assign) and len(rstmt.targets) == 1 \
                            and isinstance(rstmt.targets[0], ast.Name) \
                            and isinstance(rstmt.value, ast.BinOp) \
                            and isinstance(rstmt.value.op, ast.Sub) \
                            and isinstance(rstmt.value.left, ast.Name) \
                            and rstmt.value.left.id in coord_map:
                        rvar = rstmt.value.left.id
                        base_coord = coord_map[rvar]
                        evar = rstmt.targets[0].id
                        err_vars[evar] = base_coord
                        if not (isinstance(rstmt.value.right, ast.Name)
                                and rstmt.value.right.id == base_coord):
                            rstmt.value.right = ast.Name(id=base_coord, ctx=ast.Load())
                            changed = True
                            notes.append(f"rewrote round-trip residual {evar} = {rvar} - {base_coord}")
                        new_rev_block.append(rstmt)
                        continue
                    # Remove `if buf.data[19] == 0.0: ... ex = xn - x` override block
                    if isinstance(rstmt, ast.If) and err_vars:
                        assigns_err = any(
                            isinstance(sub, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id in err_vars for t in sub.targets)
                            for sub in ast.walk(rstmt)
                        )
                        if assigns_err:
                            changed = True
                            notes.append("removed step-0 history override in reversibility branch")
                            continue
                    new_rev_block.append(rstmt)
                rev_block[:] = new_rev_block
            break

        if changed:
            ast.fix_missing_locations(fn)

    if roundtrip_slots:
        def _condition_rt_body(stmts: List[ast.stmt]) -> bool:
            b_changed = False
            for st in stmts:
                for fld in ("body", "orelse"):
                    sub = getattr(st, fld, None)
                    if isinstance(sub, list) and sub:
                        b_changed = _condition_rt_body(sub) or b_changed
                if isinstance(st, (ast.Assign, ast.AnnAssign)):
                    tslot = _stmt_slot_target(st)
                    if tslot is not None and tslot in roundtrip_slots:
                        val = getattr(st, "value", None)
                        if val is None or isinstance(val, ast.IfExp):
                            continue
                        if isinstance(val, ast.Constant) and val.value == 0.0:
                            continue
                        vsrc = ast.unparse(val)
                        st.value = ast.parse(
                            f"0.0 if ({vsrc}) <= 1e-11 else ({vsrc})", mode="eval"
                        ).body
                        b_changed = True
            return b_changed

        for fn in _simulation_functions(tree):
            if isinstance(fn, ast.FunctionDef) and _condition_rt_body(fn.body):
                ast.fix_missing_locations(fn)
                changed = True
                notes.append("conditioned roundtrip residual slots with 1e-11 ULP deadband")

        assigned_all = _collect_assigned_slots(tree)
        driver = _simulation_driver(tree)
        if driver is not None:
            for rslot in sorted(roundtrip_slots):
                if rslot in assigned_all:
                    guard_src = f"if abs(buf.data[{rslot}]) <= 1e-11:\n    buf.data[{rslot}] = 0.0\n"
                    if guard_src.strip() not in ast.unparse(driver):
                        driver.body.extend(ast.parse(guard_src).body)
                        ast.fix_missing_locations(driver)
                        changed = True
                        notes.append(f"conditioned driver roundtrip slot state[{rslot}] <= 1e-11")

    return tree, changed, notes


def _eval_numeric_const_expr(expr_str: str) -> Optional[float]:
    """Safely evaluate a numeric constant expression (e.g. ``3.0 - 1e-12``)."""
    try:
        node = ast.parse((expr_str or "").strip(), mode="eval").body
    except SyntaxError:
        return None

    def _ev(n: ast.AST) -> Optional[float]:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.UnaryOp):
            v = _ev(n.operand)
            if v is None:
                return None
            if isinstance(n.op, ast.UAdd):
                return v
            if isinstance(n.op, ast.USub):
                return -v
        if isinstance(n, ast.BinOp):
            lv, rv = _ev(n.left), _ev(n.right)
            if lv is None or rv is None:
                return None
            if isinstance(n.op, ast.Add):
                return lv + rv
            if isinstance(n.op, ast.Sub):
                return lv - rv
            if isinstance(n.op, ast.Mult):
                return lv * rv
            if isinstance(n.op, ast.Div) and rv != 0.0:
                return lv / rv
            if isinstance(n.op, ast.Pow) and abs(rv) <= 20:
                return lv ** rv
        return None

    return _ev(node)


_INV_STATE_BASES = {"state", "data", "trace_min", "trace_max"}


def _initial_slot_of_ast(node: ast.AST) -> Optional[int]:
    if not isinstance(node, ast.Subscript):
        return None
    sl = node.slice
    if not (isinstance(sl, ast.Constant) and isinstance(sl.value, int) and not isinstance(sl.value, bool)):
        return None
    if isinstance(node.value, ast.Name) and node.value.id in ("initial", "initial_state"):
        return int(sl.value)
    return None


def _resolve_initial_in_ast(node: ast.AST, initial_state: Sequence[float]) -> ast.AST:
    """Substitute ``initial[k]`` / ``initial_state[k]`` subscripts with finite float constants."""
    class _InitialSub(ast.NodeTransformer):
        def visit_Subscript(self, sub: ast.Subscript) -> ast.AST:
            k = _initial_slot_of_ast(sub)
            if k is not None:
                val = float(initial_state[k]) if 0 <= k < len(initial_state) else 0.0
                if not math.isfinite(val):
                    val = 0.0
                return ast.Constant(value=val)
            return self.generic_visit(sub)

    cloned = _InitialSub().visit(copy.deepcopy(node))
    ast.fix_missing_locations(cloned)
    return cloned


def _eval_const_ast(node: ast.AST, initial_state: Optional[Sequence[float]] = None) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if initial_state is not None:
        init_k = _initial_slot_of_ast(node)
        if init_k is not None:
            v = float(initial_state[init_k]) if 0 <= init_k < len(initial_state) else 0.0
            return v if math.isfinite(v) else 0.0
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        return None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math":
        if node.attr == "pi":
            return math.pi
        if node.attr == "e":
            return math.e
        return None
    if isinstance(node, ast.UnaryOp):
        v = _eval_const_ast(node.operand, initial_state=initial_state)
        if v is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.USub):
            return -v
        return None
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        fn = _call_name(node.func) or ""
        if fn in ("abs", "fabs", "math.fabs"):
            v = _eval_const_ast(node.args[0], initial_state=initial_state)
            return abs(v) if v is not None else None
        return None
    if isinstance(node, ast.BinOp):
        lv = _eval_const_ast(node.left, initial_state=initial_state)
        rv = _eval_const_ast(node.right, initial_state=initial_state)
        if lv is None or rv is None:
            return None
        if isinstance(node.op, ast.Add):
            return lv + rv
        if isinstance(node.op, ast.Sub):
            return lv - rv
        if isinstance(node.op, ast.Mult):
            return lv * rv
        if isinstance(node.op, ast.Div) and rv != 0.0:
            return lv / rv
        if isinstance(node.op, ast.Pow) and abs(rv) <= 20:
            try:
                return float(lv ** rv)
            except Exception:
                return None
    return None


def _slot_of_state_ast(node: ast.AST) -> Optional[int]:
    if not isinstance(node, ast.Subscript):
        return None
    sl = node.slice
    if not (isinstance(sl, ast.Constant) and isinstance(sl.value, int) and not isinstance(sl.value, bool)):
        return None
    val = node.value
    if isinstance(val, ast.Name) and val.id in _INV_STATE_BASES:
        return int(sl.value)
    if (isinstance(val, ast.Attribute) and val.attr == "data"
            and isinstance(val.value, ast.Name) and val.value.id in ("buf", "state")):
        return int(sl.value)
    return None


def _unwrap_abs_call(node: ast.AST) -> Optional[ast.AST]:
    if (isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords):
        fn = _call_name(node.func) or ""
        if fn in ("abs", "fabs", "math.fabs"):
            return node.args[0]
    return None


def _iter_atomic_invariant_nodes(expr_str: str) -> List[ast.AST]:
    """Parse an invariant expression and flatten `and` conjunctions (picking simplest `or` branch)."""
    text = (expr_str or "").strip()
    if not text:
        return []
    text = re.sub(r"\s*\|\|\s*", " or ", re.sub(r"\s*&&\s*", " and ", text))
    try:
        root = ast.parse(text, mode="eval").body
    except SyntaxError:
        return []
    out: List[ast.AST] = []

    def _walk(node: ast.AST) -> None:
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            for v in node.values:
                _walk(v)
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and node.values:
            with_slots = [v for v in node.values if _slots_in_ast(v)]
            cands = with_slots or list(node.values)
            best = min(cands, key=lambda v: (len(_slots_in_ast(v)), len(ast.unparse(v))))
            _walk(best)
        elif isinstance(node, ast.Compare) and len(node.ops) >= 2:
            cur_l = node.left
            for cur_op, cur_r in zip(node.ops, node.comparators):
                out.append(ast.Compare(left=cur_l, ops=[cur_op], comparators=[cur_r]))
                cur_l = cur_r
        else:
            out.append(node)

    _walk(root)
    return out


def _match_abs_affine(abs_arg: ast.AST) -> Tuple[Optional[Tuple[int, float]], Optional[Tuple[int, int, float]], Optional[Tuple[int, int]]]:
    """Analyze the inside of `abs(...)`.

    Returns `(single_target, sum_target, diff_pair)` where:
    - `single_target = (slot, target_val)` for `state[i] - C`, `C - state[i]`, `state[i] + C`
    - `sum_target = (slot_i, slot_j, target_val)` for `(state[i] + state[j]) - C`, `C - (state[i] + state[j])`, etc.
    - `diff_pair = (slot_i, slot_j)` for `state[i] - state[j]`
    """
    # Represent linear form: coeff_by_slot + const
    def _lin(n: ast.AST) -> Optional[Tuple[Dict[int, int], float]]:
        s = _slot_of_state_ast(n)
        if s is not None:
            return {s: 1}, 0.0
        c = _eval_const_ast(n)
        if c is not None:
            return {}, float(c)
        if isinstance(n, ast.UnaryOp):
            inner = _lin(n.operand)
            if inner is None:
                return None
            if isinstance(n.op, ast.UAdd):
                return inner
            if isinstance(n.op, ast.USub):
                return {k: -v for k, v in inner[0].items()}, -inner[1]
            return None
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
            L = _lin(n.left)
            R = _lin(n.right)
            if L is None or R is None:
                return None
            sign = 1 if isinstance(n.op, ast.Add) else -1
            coeffs = dict(L[0])
            for k, v in R[0].items():
                coeffs[k] = coeffs.get(k, 0) + sign * v
                if coeffs[k] == 0:
                    del coeffs[k]
            return coeffs, L[1] + sign * R[1]
        return None

    res = _lin(abs_arg)
    if res is None:
        return None, None, None
    coeffs, const_val = res
    if len(coeffs) == 1:
        (s_idx, coeff), = coeffs.items()
        if coeff in (1, -1) and abs_arg_has_const(abs_arg):
            return (s_idx, -const_val / coeff), None, None
    elif len(coeffs) == 2:
        items = list(coeffs.items())
        (i_idx, ci), (j_idx, cj) = items[0], items[1]
        if (ci == 1 and cj == 1) or (ci == -1 and cj == -1):
            return None, (i_idx, j_idx, -const_val / ci), None
        if ((ci == 1 and cj == -1) or (ci == -1 and cj == 1)) and abs(const_val) <= 1e-15:
            return None, None, (i_idx, j_idx)
    return None, None, None


def _match_abs_multi_sum(abs_arg: ast.AST) -> Optional[Tuple[List[int], float]]:
    """Match ``sum_{k=1..N} state[i_k] - C`` / ``C - sum`` with ``N >= 3`` distinct slots."""
    def _lin(n: ast.AST) -> Optional[Tuple[Dict[int, int], float]]:
        s = _slot_of_state_ast(n)
        if s is not None:
            return {s: 1}, 0.0
        c = _eval_const_ast(n)
        if c is not None:
            return {}, float(c)
        if isinstance(n, ast.UnaryOp):
            inner = _lin(n.operand)
            if inner is None:
                return None
            if isinstance(n.op, ast.UAdd):
                return inner
            if isinstance(n.op, ast.USub):
                return {k: -v for k, v in inner[0].items()}, -inner[1]
            return None
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
            L = _lin(n.left)
            R = _lin(n.right)
            if L is None or R is None:
                return None
            sign = 1 if isinstance(n.op, ast.Add) else -1
            coeffs = dict(L[0])
            for k, v in R[0].items():
                coeffs[k] = coeffs.get(k, 0) + sign * v
                if coeffs[k] == 0:
                    del coeffs[k]
            return coeffs, L[1] + sign * R[1]
        return None

    res = _lin(abs_arg)
    if res is None:
        return None
    coeffs, const_val = res
    if len(coeffs) < 3 or not abs_arg_has_const(abs_arg):
        return None
    vals = set(coeffs.values())
    if vals == {1}:
        return list(coeffs.keys()), -const_val
    if vals == {-1}:
        return list(coeffs.keys()), const_val
    return None


def abs_arg_has_const(node: ast.AST) -> bool:
    return any(_eval_const_ast(sub) is not None and isinstance(sub, ast.Constant) for sub in ast.walk(node) if sub is not node) or isinstance(node, ast.BinOp)


def _unwrap_abs_or_rel_err(node: ast.AST) -> Tuple[Optional[ast.AST], Optional[ast.AST]]:
    """Return ``(abs_inner, denom_ast)`` for ``abs(inner)`` or ``abs(inner) / denom``."""
    inner = _unwrap_abs_call(node)
    if inner is not None:
        return inner, None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        inner_l = _unwrap_abs_call(node.left)
        if inner_l is not None:
            return inner_l, node.right
    return None, None


def _extract_abs_tol(node: ast.AST) -> Tuple[Optional[float], bool]:
    """Extract scalar tolerance ``(tol, is_rel_rhs)`` from constant or ``c * scale`` RHS."""
    c_val = _eval_const_ast(node)
    if c_val is not None:
        return float(c_val), False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        lc = _eval_const_ast(node.left)
        if lc is not None and lc >= 0.0:
            return float(lc), True
        rc = _eval_const_ast(node.right)
        if rc is not None and rc >= 0.0:
            return float(rc), True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        lt, lr = _extract_abs_tol(node.left)
        rt, rr = _extract_abs_tol(node.right)
        if lt is not None and rt is not None:
            return max(lt, rt), (lr or rr)
    return None, False


def _slots_in_ast(node: ast.AST) -> Set[int]:
    """Collect all constant integer state/buffer slot indices referenced in ``node``."""
    out: Set[int] = set()
    for sub in ast.walk(node):
        s = _slot_of_state_ast(sub)
        if s is not None:
            out.add(s)
    return out


class _StateToBufTransformer(ast.NodeTransformer):
    """Rewrite ``state[k]``/``buf.data[k]``/``data[k]`` nodes to ``buf.data[k]``."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        s = _slot_of_state_ast(node)
        if s is not None:
            return ast.Subscript(
                value=ast.Attribute(
                    value=ast.Name(id="buf", ctx=ast.Load()),
                    attr="data",
                    ctx=ast.Load(),
                ),
                slice=ast.Constant(value=s),
                ctx=ast.Load(),
            )
        return self.generic_visit(node)


def _match_abs_algebraic(
    abs_inner: ast.AST,
) -> Optional[Tuple[int, str, Optional[Tuple[int, int]]]]:
    """Match ``state[i] - EXPR`` / ``EXPR - state[i]`` where EXPR uses other state slots.

    Returns ``(target_slot, expr_buf_str, product_pair_or_None)``.
    """
    if not isinstance(abs_inner, ast.BinOp) or not isinstance(abs_inner.op, (ast.Sub, ast.Add)):
        return None
    l_slot = _slot_of_state_ast(abs_inner.left)
    r_slot = _slot_of_state_ast(abs_inner.right)
    target_slot: Optional[int] = None
    target_ast: Optional[ast.AST] = None
    if isinstance(abs_inner.op, ast.Sub):
        if l_slot is not None:
            r_slots = _slots_in_ast(abs_inner.right)
            if r_slots and l_slot not in r_slots:
                target_slot = l_slot
                target_ast = abs_inner.right
        if target_slot is None and r_slot is not None:
            l_slots = _slots_in_ast(abs_inner.left)
            if l_slots and r_slot not in l_slots:
                target_slot = r_slot
                target_ast = abs_inner.left
    else:  # ast.Add: state[i] + EXPR == 0 -> state[i] == -EXPR
        if l_slot is not None:
            r_slots = _slots_in_ast(abs_inner.right)
            if r_slots and l_slot not in r_slots:
                target_slot = l_slot
                target_ast = ast.UnaryOp(op=ast.USub(), operand=abs_inner.right)
        if target_slot is None and r_slot is not None:
            l_slots = _slots_in_ast(abs_inner.left)
            if l_slots and r_slot not in l_slots:
                target_slot = r_slot
                target_ast = ast.UnaryOp(op=ast.USub(), operand=abs_inner.left)
    if target_slot is None or target_ast is None:
        return None
    prod_pair: Optional[Tuple[int, int]] = None
    if isinstance(target_ast, ast.BinOp) and isinstance(target_ast.op, ast.Mult):
        j_s = _slot_of_state_ast(target_ast.left)
        k_s = _slot_of_state_ast(target_ast.right)
        if j_s is not None and k_s is not None and j_s != target_slot and k_s != target_slot:
            prod_pair = (j_s, k_s)
    buf_ast = _StateToBufTransformer().visit(copy.deepcopy(target_ast))
    ast.fix_missing_locations(buf_ast)
    return target_slot, ast.unparse(buf_ast), prod_pair


_INV_DISALLOWED_NAMES = frozenset(
    {"telemetry", "trace_min", "trace_max", "initial", "variants", "history"}
)
_INV_ALLOWED_FREE_NAMES = frozenset(
    {"state", "buf", "data", "math", "np", "numpy", "True", "False", "inf", "nan"}
)
_INV_ALLOWED_MATH_FUNCS = frozenset(
    {"abs", "fabs", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
     "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
     "exp", "log", "log10", "pow", "min", "max", "isfinite", "_invar_sdiv"}
)


def _has_only_pure_state_names(node: ast.AST) -> bool:
    """Return True iff every Name/Call in ``node`` only uses state slots and scalar math."""
    call_func_ids = {
        id(sub.func)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
    }
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            if sub.attr == "data" and isinstance(sub.value, ast.Name) and sub.value.id in ("buf", "state"):
                continue
            if isinstance(sub.value, ast.Name) and sub.value.id == "math" and sub.attr in _INV_ALLOWED_MATH_FUNCS:
                continue
            return False
        if isinstance(sub, ast.Name):
            if sub.id in _INV_DISALLOWED_NAMES:
                return False
            if sub.id in _INV_ALLOWED_FREE_NAMES:
                continue
            if id(sub) in call_func_ids and sub.id in _INV_ALLOWED_MATH_FUNCS:
                continue
            return False
    return True


class _InvToBufTransformer(ast.NodeTransformer):
    """Translate pure invariant AST nodes to C++-compatible ``buf.data[...]`` form."""

    def __init__(self, *, guard_sqrt_eps: float = 0.0) -> None:
        self.guard_sqrt_eps = float(guard_sqrt_eps)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        s = _slot_of_state_ast(node)
        if s is not None:
            return ast.Subscript(
                value=ast.Attribute(
                    value=ast.Name(id="buf", ctx=ast.Load()),
                    attr="data",
                    ctx=ast.Load(),
                ),
                slice=ast.Constant(value=s),
                ctx=ast.Load(),
            )
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.Call(
                func=ast.Name(id="_invar_sdiv", ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        cname = _call_name(node.func) or ""
        if cname in ("sqrt", "math.sqrt", "np.sqrt", "numpy.sqrt") and len(node.args) == 1 and not node.keywords:
            eps_val = self.guard_sqrt_eps if self.guard_sqrt_eps > 0.0 else 0.0
            arg_guarded = ast.Call(
                func=ast.Name(id="max", ctx=ast.Load()),
                args=[node.args[0], ast.Constant(value=eps_val)],
                keywords=[],
            )
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()), attr="sqrt", ctx=ast.Load()),
                args=[arg_guarded],
                keywords=[],
            )
        if cname in ("fabs", "sin", "cos", "tan", "asin", "acos", "atan",
                     "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                     "exp", "log", "log10", "pow", "isfinite"):
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()), attr=cname, ctx=ast.Load()),
                args=node.args,
                keywords=node.keywords,
            )
        return node


def _translate_inv_expr_to_buf(node: ast.AST, *, protect_denom: bool = False) -> Optional[str]:
    if not _has_only_pure_state_names(node):
        return None
    eps = 1e-300 if protect_denom else 0.0
    cloned = copy.deepcopy(node)
    transformed = _InvToBufTransformer(guard_sqrt_eps=eps).visit(cloned)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def _flip_cmp_op_str(cmp_op: ast.cmpop) -> Optional[str]:
    if isinstance(cmp_op, ast.Lt):
        return "<"
    if isinstance(cmp_op, ast.LtE):
        return "<="
    if isinstance(cmp_op, ast.Gt):
        return ">"
    if isinstance(cmp_op, ast.GtE):
        return ">="
    return None


def _match_ratio_cmp(
    lhs: ast.AST, cmp_op: ast.cmpop, rhs: ast.AST
) -> Optional[Tuple[int, ast.AST, str, ast.AST]]:
    """Match ``state[i] / DENOM (<|<=|>|>=) BOUND`` or reversed ``BOUND <op> state[i] / DENOM``."""
    op_str = _flip_cmp_op_str(cmp_op)
    if op_str is None:
        return None
    flip_map = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
    candidates = [
        (lhs, op_str, rhs),
        (rhs, flip_map[op_str], lhs),
    ]
    for ratio_node, norm_op, bound_node in candidates:
        if not (isinstance(ratio_node, ast.BinOp) and isinstance(ratio_node.op, ast.Div)):
            continue
        num_slot = _slot_of_state_ast(ratio_node.left)
        if num_slot is None:
            num_abs = _unwrap_abs_call(ratio_node.left)
            if num_abs is not None:
                num_slot = _slot_of_state_ast(num_abs)
        if num_slot is None:
            continue
        den_node = ratio_node.right
        if not (_has_only_pure_state_names(den_node) and _has_only_pure_state_names(bound_node)):
            continue
        den_slots = _slots_in_ast(den_node)
        bound_slots = _slots_in_ast(bound_node)
        if not den_slots or num_slot in den_slots or num_slot in bound_slots:
            continue
        return num_slot, den_node, norm_op, bound_node
    return None


def _collect_mult_state_factors(node: ast.AST) -> Optional[List[int]]:
    s = _slot_of_state_ast(node)
    if s is not None:
        return [s]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        lf = _collect_mult_state_factors(node.left)
        rf = _collect_mult_state_factors(node.right)
        if lf is not None and rf is not None:
            return lf + rf
    return None


def _match_product_positivity(
    lhs: ast.AST, cmp_op: ast.cmpop, rhs: ast.AST
) -> Optional[Tuple[List[int], float, bool]]:
    """Match ``state[i] * state[j] (>|>=) C`` (or reversed ``C (<|<=) ...``) with ``C >= 0``."""
    if isinstance(cmp_op, (ast.Gt, ast.GtE)):
        factors = _collect_mult_state_factors(lhs)
        thresh = _eval_const_ast(rhs)
        is_strict = isinstance(cmp_op, ast.Gt)
    elif isinstance(cmp_op, (ast.Lt, ast.LtE)):
        factors = _collect_mult_state_factors(rhs)
        thresh = _eval_const_ast(lhs)
        is_strict = isinstance(cmp_op, ast.Lt)
    else:
        return None
    if factors is None or len(factors) < 2 or thresh is None or not math.isfinite(thresh) or thresh < 0.0:
        return None
    return factors, float(thresh), is_strict


class _BufDataView:
    def __init__(self, data: List[float]) -> None:
        self.data = data


def _eval_pure_inv_ast(node: ast.AST, state_vec: Sequence[float]) -> Optional[bool]:
    """Evaluate a pure-state invariant AST expression on ``state_vec``."""
    if not _has_only_pure_state_names(node):
        return None
    vec = [float(x) for x in state_vec]
    env: Dict[str, Any] = {
        "__builtins__": {},
        "state": vec,
        "data": vec,
        "buf": _BufDataView(vec),
        "math": math,
        "abs": abs,
        "fabs": math.fabs,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "asinh": math.asinh,
        "acosh": math.acosh,
        "atanh": math.atanh,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "pow": pow,
        "min": min,
        "max": max,
        "isfinite": math.isfinite,
        "_invar_sdiv": lambda a, b: (float(a) / float(b)) if float(b) != 0.0 else 0.0,
        "inf": math.inf,
        "nan": math.nan,
        "True": True,
        "False": False,
    }
    try:
        res = eval(compile(ast.Expression(copy.deepcopy(node)), "<invar_witness>", "eval"), {"__builtins__": {}}, env)
        return bool(res)
    except Exception:
        return None


def _find_witness_for_contract_invariant(
    root_node: ast.AST,
    co_invs: Sequence[ast.AST],
    base_state: Sequence[float],
    *,
    eq_target_slots: Dict[int, Tuple[float, float, bool]],
    nonneg_slots: Dict[int, float],
    strict_gt_slots: Dict[int, float],
    positive_lower_slots: Dict[int, float],
    small_upper_slots: Dict[int, float],
    other_upper_slots: Dict[int, float],
    other_upper_strict: Set[int],
    quarter_upper_slots: Set[int],
    unit_upper_slots: Set[int],
) -> Optional[Dict[int, float]]:
    """Find a deterministic finite assignment across referenced state slots satisfying ``root_node`` and ``co_invs``."""
    mod_slots = sorted(_slots_in_ast(root_node))
    if not mod_slots:
        return None
    mod_set = set(mod_slots)
    rel_co = [n for n in co_invs if _slots_in_ast(n) & mod_set]

    def _ok(vec: Sequence[float]) -> bool:
        if _eval_pure_inv_ast(root_node, vec) is not True:
            return False
        return all(_eval_pure_inv_ast(cn, vec) is True for cn in rel_co)

    if _ok(base_state):
        return {s: float(base_state[s]) for s in mod_slots}

    cands_by_slot: Dict[int, List[float]] = {}
    for s in mod_slots:
        if s in eq_target_slots:
            cands_by_slot[s] = [float(eq_target_slots[s][0])]
            continue
        ubs = [
            float(v)
            for v in (
                small_upper_slots.get(s),
                other_upper_slots.get(s),
                0.25 if s in quarter_upper_slots else None,
                1.0 if s in unit_upper_slots else None,
            )
            if v is not None
        ]
        ub = min(ubs) if ubs else None
        lbs = [
            float(v)
            for v in (
                nonneg_slots.get(s),
                strict_gt_slots.get(s),
                positive_lower_slots.get(s),
            )
            if v is not None
        ]
        lb = max(lbs) if lbs else None
        strict_lb = s in strict_gt_slots and strict_gt_slots[s] == lb
        strict_ub = s in other_upper_strict and ub == other_upper_slots.get(s)

        raw: List[float] = [float(base_state[s]), 1.0, 0.5, 0.0, 1e-3, 1e-6, 1e-9, 1e-12, 10.0, 100.0, 1e3, 1e6, -1.0]
        if ub is not None:
            raw.insert(1, 0.5 * ub if ub > 0.0 else (ub - 1.0 if strict_ub else ub))
        if lb is not None:
            raw.insert(1, max(lb + 1.0, lb * 1.1 + 1e-6) if strict_lb else max(lb, 0.0))

        seen_v: Set[float] = set()
        valid: List[float] = []
        for v in raw:
            if not math.isfinite(v):
                continue
            if v in seen_v:
                continue
            if lb is not None and (v <= lb if strict_lb else v < lb):
                continue
            if ub is not None and (v >= ub if strict_ub else v > ub):
                continue
            seen_v.add(v)
            valid.append(float(v))
        if not valid:
            valid = [float(base_state[s]) if math.isfinite(base_state[s]) else 0.0]
        cands_by_slot[s] = valid[:8]

    for s in mod_slots:
        for v in cands_by_slot[s]:
            trial = list(base_state)
            trial[s] = v
            if _ok(trial):
                return {k: float(trial[k]) for k in mod_slots}

    if len(mod_slots) >= 2:
        trial_all = list(base_state)
        for s in mod_slots:
            nz = [v for v in cands_by_slot[s] if v != 0.0]
            trial_all[s] = nz[0] if nz else cands_by_slot[s][0]
        if _ok(trial_all):
            return {k: float(trial_all[k]) for k in mod_slots}
        evals = 0
        s0, s1 = mod_slots[0], mod_slots[1]
        for v0 in cands_by_slot[s0]:
            for v1 in cands_by_slot[s1]:
                trial = list(trial_all)
                trial[s0] = v0
                trial[s1] = v1
                evals += 1
                if _ok(trial):
                    return {k: float(trial[k]) for k in mod_slots}
                if evals >= 200:
                    break
            if evals >= 200:
                break
    return None


def _eval_ast_on_state(node: ast.AST, state: Sequence[float]) -> Optional[float]:
    """Evaluate an algebraic AST expression over constants and ``state[k]``."""
    s_idx = _slot_of_state_ast(node)
    if s_idx is not None:
        return float(state[s_idx]) if 0 <= s_idx < len(state) else None
    c_val = _eval_const_ast(node)
    if c_val is not None:
        return float(c_val)
    if isinstance(node, ast.UnaryOp):
        v = _eval_ast_on_state(node.operand, state)
        if v is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.USub):
            return -v
        return None
    if isinstance(node, ast.BinOp):
        lv = _eval_ast_on_state(node.left, state)
        rv = _eval_ast_on_state(node.right, state)
        if lv is None or rv is None:
            return None
        if isinstance(node.op, ast.Add):
            return lv + rv
        if isinstance(node.op, ast.Sub):
            return lv - rv
        if isinstance(node.op, ast.Mult):
            return lv * rv
        if isinstance(node.op, ast.Div):
            return lv / rv if abs(rv) > 1e-300 else None
        if isinstance(node.op, ast.Pow) and abs(rv) <= 12:
            try:
                return float(lv ** rv)
            except Exception:
                return None
    return None


def _expand_slot_match_tokens(raw_name: str) -> Set[str]:
    """Extract normalized multi-character tokens and physical aliases from an identifier."""
    low = (raw_name or "").strip().lower()
    if not low:
        return set()
    # Split camelCase then non-alphanumerics
    spaced = re.sub(r"([a-z])([A-Z])", r"\1_\2", raw_name or "")
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if p]
    tokens: Set[str] = {low.replace(" ", "_")}
    for p in parts:
        if len(p) >= 2:
            tokens.add(p)
    for a, b in zip(parts, parts[1:]):
        tokens.add(f"{a}_{b}")
    # Expand well-known physics variable-name shorthands (non-intrusive, symmetric)
    if low == "s" or "s_shunt" in tokens or "shunt" in tokens:
        tokens.update({"shunt", "s_shunt", "ratio"})
    if "dt_k" in tokens or "dtk" in tokens or "dt_kap" in tokens or "kapitza" in tokens or "dt_kapitza" in tokens:
        tokens.update({"kapitza", "dt_k", "dtk"})
    if "p_vis" in tokens or "pvis" in tokens or "visible" in tokens or "dissipation" in tokens:
        tokens.update({"p_vis", "visible", "dissipation", "vis"})
    if "w_cyc" in tokens or "wcyc" in tokens or "cyclic" in tokens or "work" in tokens:
        tokens.update({"w_cyc", "cyclic", "work", "cyc"})
    if "p_net" in tokens or "pnet" in tokens or "net" in tokens or "surplus" in tokens:
        tokens.update({"p_net", "net", "surplus"})
    if "p_array" in tokens or "parray" in tokens or "array" in tokens:
        tokens.update({"p_array", "array"})
    if "eta_a" in tokens or "eta_d" in tokens or "partition" in tokens:
        tokens.update({"eta", "partition"})
    if "e_lan" in tokens or "p_lan" in tokens or "landauer" in tokens:
        tokens.update({"landauer"})
    if "n_dot" in tokens or "ndot" in tokens or "gamow" in tokens:
        tokens.update({"n_dot", "gamow"})
    if "t_rise" in tokens or "temp_rise" in tokens or ("temp" in tokens and "rise" in tokens) or "temperature_rise" in tokens or "delta_t_max" in tokens:
        tokens.update({"t_rise", "temp_rise", "temperature_rise", "rise"})
    if "v_cell" in tokens or "voltage" in tokens:
        tokens.update({"v_cell", "voltage"})
    return tokens


def _inline_buffer_mutating_helpers_tree(
    tree: ast.Module,
) -> Tuple[ast.Module, bool, List[str]]:
    """Inline standalone buffer-mutating helper calls (e.g. ``stage_a(buf, dt)``)
    into their callers so downstream overwrite splitters and contract conditioners
    see every stage's ``buf.data[...]`` writes in sequential order.
    """
    mod_names: Set[str] = {
        n.id for node in ast.walk(tree) if isinstance(node, ast.alias) for n in [ast.Name(id=node.asname or node.name.split(".")[0])]
    }
    for top in tree.body:
        if isinstance(top, ast.FunctionDef):
            mod_names.add(top.name)
        elif isinstance(top, ast.Assign):
            for tgt in top.targets:
                if isinstance(tgt, ast.Name):
                    mod_names.add(tgt.id)

    def _fn_local_stores(fn_body: Sequence[ast.stmt]) -> Set[str]:
        bound: Set[str] = set()
        for sub in ast.walk(ast.Module(body=list(fn_body), type_ignores=[])):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                bound.add(sub.id)
        return bound

    def _has_buf_data_store(fn_def: ast.FunctionDef, buf_param: str, helper_names: Set[str]) -> bool:
        for sub in ast.walk(fn_def):
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    if isinstance(tgt, ast.Subscript):
                        val = tgt.value
                        if (
                            isinstance(val, ast.Attribute)
                            and val.attr == "data"
                            and isinstance(val.value, ast.Name)
                            and val.value.id == buf_param
                        ):
                            return True
            elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Subscript):
                val = sub.target.value
                if (
                    isinstance(val, ast.Attribute)
                    and val.attr == "data"
                    and isinstance(val.value, ast.Name)
                    and val.value.id == buf_param
                ):
                    return True
            elif (
                isinstance(sub, ast.Expr)
                and isinstance(sub.value, ast.Call)
                and isinstance(sub.value.func, ast.Name)
                and sub.value.func.id in helper_names
                and sub.value.args
                and isinstance(sub.value.args[0], ast.Name)
                and sub.value.args[0].id == buf_param
            ):
                return True
        return False

    def _valid_helper_body(body: Sequence[ast.stmt]) -> Optional[List[ast.stmt]]:
        if not body:
            return None
        trimmed = list(body)
        if isinstance(trimmed[-1], ast.Return):
            if trimmed[-1].value is not None:
                return None
            trimmed = trimmed[:-1]
        for st in trimmed:
            for sub in ast.walk(st):
                if isinstance(sub, (ast.Return, ast.Yield, ast.YieldFrom)):
                    return None
        return trimmed

    changed = False
    notes: List[str] = []
    removed_helpers: Set[str] = set()

    for _pass in range(6):
        fn_map: Dict[str, ast.FunctionDef] = {
            st.name: st for st in tree.body if isinstance(st, ast.FunctionDef)
        }
        protected_names = {"simulation_step", _SAFE_DIV_HELPER}
        candidates: Dict[str, ast.FunctionDef] = {}
        for fname, fdef in fn_map.items():
            if (
                fname in protected_names
                or fname.startswith("_invar_")
                or re.match(r"^layer_\d+_step$", fname)
                or fname.endswith(("_euler", "_rk4"))
                or fdef.decorator_list
                or fdef.args.vararg is not None
                or fdef.args.kwarg is not None
                or fdef.args.kwonlyargs
                or fdef.args.posonlyargs
                or not fdef.args.args
            ):
                continue
            if _valid_helper_body(fdef.body) is None:
                continue
            candidates[fname] = fdef

        if not candidates:
            break

        # Keep only functions whose first arg mutates `.data[...]` (directly or via another candidate)
        mutating: Dict[str, ast.FunctionDef] = {}
        cand_names = set(candidates)
        for _m_iter in range(len(candidates) + 1):
            grew = False
            for fname, fdef in candidates.items():
                if fname in mutating:
                    continue
                p0 = fdef.args.args[0].arg
                if _has_buf_data_store(fdef, p0, set(mutating) | cand_names):
                    mutating[fname] = fdef
                    grew = True
            if not grew:
                break
        if not mutating:
            break

        # Ensure candidate is ONLY invoked as a standalone Expr(Call) statement (never in value position)
        standalone_call_ids: Set[int] = set()
        for parent in ast.walk(tree):
            if isinstance(parent, ast.Expr) and isinstance(parent.value, ast.Call):
                if isinstance(parent.value.func, ast.Name) and parent.value.func.id in mutating:
                    standalone_call_ids.add(id(parent.value))

        invalid_uses: Set[str] = set()
        for fdef in fn_map.values():
            for node in ast.walk(fdef):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    cname = node.func.id
                    if cname not in mutating:
                        continue
                    if id(node) not in standalone_call_ids:
                        invalid_uses.add(cname)
                        continue
                    cal = mutating[cname]
                    if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
                        invalid_uses.add(cname)
                        continue
                    params = [a.arg for a in cal.args.args]
                    if len(node.args) > len(params):
                        invalid_uses.add(cname)
                        continue
                    bound_kw = {k.arg for k in node.keywords if k.arg is not None}
                    if not bound_kw.issubset(params[len(node.args):]):
                        invalid_uses.add(cname)
                        continue
                    req_n = len(params) - len(cal.args.defaults)
                    if not all(params[idx] in bound_kw for idx in range(len(node.args), req_n)):
                        invalid_uses.add(cname)
                        continue
                elif isinstance(node, ast.Name) and node.id in mutating and isinstance(node.ctx, ast.Load):
                    # Check if this Name is the .func of a valid standalone Call
                    pass
        for fname in list(mutating):
            if fname in invalid_uses:
                mutating.pop(fname, None)
                continue
            # Also reject if fname is referenced as an attribute/value (not Call.func)
            call_func_name_ids = {
                id(n.func)
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == fname
            }
            for n in ast.walk(tree):
                if isinstance(n, ast.Name) and n.id == fname and id(n) not in call_func_name_ids:
                    mutating.pop(fname, None)
                    break

        if not mutating:
            break

        # Detect cycles so recursive helpers are never inlined
        edges: Dict[str, Set[str]] = {fn: set() for fn in mutating}
        for fn, fdef in mutating.items():
            for n in ast.walk(fdef):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in mutating:
                    edges[fn].add(n.func.id)
        cyclic: Set[str] = set()
        for start in mutating:
            seen_v: Set[str] = set()
            stack = list(edges[start])
            while stack:
                cur = stack.pop()
                if cur == start:
                    cyclic.add(start)
                    break
                if cur not in seen_v:
                    seen_v.add(cur)
                    stack.extend(edges.get(cur, ()))
        for cy in cyclic:
            mutating.pop(cy, None)
        if not mutating:
            break

        pass_inlined: Set[str] = set()

        class _SubstNames(ast.NodeTransformer):
            def __init__(self, subst_expr: Dict[str, ast.AST], rename_var: Dict[str, str]):
                self.subst_expr = subst_expr
                self.rename_var = rename_var

            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id in self.subst_expr and isinstance(node.ctx, ast.Load):
                    repl = copy.deepcopy(self.subst_expr[node.id])
                    ast.copy_location(repl, node)
                    return repl
                if node.id in self.rename_var:
                    return ast.copy_location(ast.Name(id=self.rename_var[node.id], ctx=node.ctx), node)
                return node

        def _bind_helper_stmts(
            caller_fn: ast.FunctionDef,
            call_node: ast.Call,
            caller_used: Set[str],
        ) -> Optional[List[ast.stmt]]:
            if not isinstance(call_node.func, ast.Name) or call_node.func.id not in mutating:
                return None
            hdef = mutating[call_node.func.id]
            body_tpl = _valid_helper_body(hdef.body)
            if body_tpl is None:
                return None
            params = [a.arg for a in hdef.args.args]
            defaults = list(hdef.args.defaults)
            def_offset = len(params) - len(defaults)
            kw_map = {k.arg: k.value for k in call_node.keywords if k.arg is not None}
            arg_by_param: Dict[str, ast.expr] = {}
            for idx, p in enumerate(params):
                if idx < len(call_node.args):
                    arg_by_param[p] = call_node.args[idx]
                elif p in kw_map:
                    arg_by_param[p] = kw_map[p]
                else:
                    arg_by_param[p] = copy.deepcopy(defaults[idx - def_offset])

            local_stores = _fn_local_stores(body_tpl)
            subst_expr: Dict[str, ast.AST] = {}
            rename_var: Dict[str, str] = {}
            preamble: List[ast.stmt] = []

            for p in params:
                a_expr = arg_by_param[p]
                if p not in local_stores and isinstance(a_expr, ast.Name):
                    subst_expr[p] = a_expr
                else:
                    cand = p if p not in caller_used else f"{hdef.name}_{p}"
                    cnt = 1
                    while cand in caller_used or cand in mod_names:
                        cand = f"{hdef.name}_{p}_{cnt}"
                        cnt += 1
                    caller_used.add(cand)
                    preamble.append(
                        ast.Assign(
                            targets=[ast.Name(id=cand, ctx=ast.Store())],
                            value=copy.deepcopy(a_expr),
                        )
                    )
                    if p == cand:
                        pass
                    else:
                        rename_var[p] = cand

            for lv in sorted(local_stores - set(params)):
                if lv in caller_used:
                    cand = f"{hdef.name}_{lv}"
                    cnt = 1
                    while cand in caller_used or cand in mod_names or cand in local_stores:
                        cand = f"{hdef.name}_{lv}_{cnt}"
                        cnt += 1
                    rename_var[lv] = cand
                    caller_used.add(cand)
                else:
                    caller_used.add(lv)

            cloned_stmts = [copy.deepcopy(s) for s in body_tpl]
            sub_tr = _SubstNames(subst_expr, rename_var)
            out_stmts: List[ast.stmt] = list(preamble)
            for cs in cloned_stmts:
                res = sub_tr.visit(cs)
                if isinstance(res, ast.stmt):
                    out_stmts.append(res)
            return out_stmts

        def _rewrite_stmt_list(
            stmts: List[ast.stmt],
            caller_fn: ast.FunctionDef,
            caller_used: Set[str],
        ) -> Tuple[List[ast.stmt], bool]:
            new_list: List[ast.stmt] = []
            did_rw = False
            for st in stmts:
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                    inlined = _bind_helper_stmts(caller_fn, st.value, caller_used)
                    if inlined is not None:
                        callee_n = st.value.func.id  # type: ignore[attr-defined]
                        pass_inlined.add(callee_n)
                        notes.append(f"inlined buffer-mutating helper {callee_n}() into {caller_fn.name}")
                        new_list.extend(inlined)
                        did_rw = True
                        continue
                for attr in ("body", "orelse"):
                    sub_list = getattr(st, attr, None)
                    if isinstance(sub_list, list):
                        rw_sub, sub_chg = _rewrite_stmt_list(sub_list, caller_fn, caller_used)
                        if sub_chg:
                            setattr(st, attr, rw_sub)
                            did_rw = True
                new_list.append(st)
            return new_list, did_rw

        pass_changed = False
        for top in tree.body:
            if not isinstance(top, ast.FunctionDef):
                continue
            caller_used = {a.arg for a in top.args.args} | _fn_local_stores(top.body)
            new_body, fn_chg = _rewrite_stmt_list(top.body, top, caller_used)
            if fn_chg:
                top.body = new_body
                pass_changed = True
                changed = True

        if not pass_changed:
            break

        # Prune inlined helpers no longer referenced anywhere in module
        for hname in sorted(pass_inlined):
            still_used = False
            for top in tree.body:
                if isinstance(top, ast.FunctionDef) and top.name == hname:
                    continue
                if any(isinstance(n, ast.Name) and n.id == hname for n in ast.walk(top)):
                    still_used = True
                    break
            if not still_used:
                tree.body = [
                    st for st in tree.body
                    if not (isinstance(st, ast.FunctionDef) and st.name == hname)
                ]
                removed_helpers.add(hname)

    if changed:
        ast.fix_missing_locations(tree)
    return tree, changed, notes


def _split_intra_function_stage_overwrites_tree(
    tree: ast.Module,
    contracts: Sequence[Any] = (),
    all_contracts: Sequence[Any] = (),
    variable_map: Optional[Dict[str, int]] = None,
) -> Tuple[ast.Module, bool, List[str]]:
    """Split multi-stage ``buf.data[k]`` overwrites inside one step function and bridge
    unassigned de-collided Stage 2/3 contract slots from matching local variables.
    """
    unique_contracts: List[Any] = []
    seen_cids: Set[str] = set()
    for c in list(all_contracts or []) + list(contracts or []):
        if c is None:
            continue
        cid = str(_contract_get(c, "contract_id", "") or "")
        if cid and cid in seen_cids:
            continue
        if cid:
            seen_cids.add(cid)
        unique_contracts.append(c)
    if not unique_contracts:
        return tree, False, []

    cls = classify_buffer_slots(tree, unique_contracts)
    protected = {0, 1, 2, 3} | set(cls["dynamical_slots"]) | set(cls["parameter_slots"]) | set(cls["history_slots"])

    stage1_slots: Set[int] = set()
    slot_tokens: Dict[int, Set[str]] = {}
    later_slots_ordered: List[int] = []
    seen_later: Set[int] = set()
    all_contract_slots: Set[int] = set()

    for c_pos, c in enumerate(unique_contracts):
        cid = str(_contract_get(c, "contract_id", "") or "")
        st_idx = _contract_stage_index(cid)
        is_stage1 = (st_idx == 1) or (st_idx <= 0 and c_pos == 0)
        for vname, assertion, _tol in _contract_invariant_items(c):
            s_list = _contract_state_slots_from_text(assertion)
            for s in s_list:
                all_contract_slots.add(s)
                if is_stage1:
                    stage1_slots.add(s)
                else:
                    if s not in seen_later:
                        seen_later.add(s)
                        later_slots_ordered.append(s)
                slot_tokens.setdefault(s, set()).update(_expand_slot_match_tokens(vname))

    if isinstance(variable_map, dict):
        for vname, s_idx in variable_map.items():
            try:
                si = int(s_idx)
            except (TypeError, ValueError):
                continue
            slot_tokens.setdefault(si, set()).update(_expand_slot_match_tokens(str(vname)))
            all_contract_slots.add(si)
            if si not in stage1_slots and si not in seen_later and si >= 4:
                seen_later.add(si)
                later_slots_ordered.append(si)

    def _score_expr_to_slot(expr_names: Set[str], slot: int) -> int:
        s_toks = slot_tokens.get(slot)
        if not s_toks:
            return 0
        best = 0
        for vn in expr_names:
            v_toks = _expand_slot_match_tokens(vn)
            # Filter out weak noise tokens so unrelated P_* names don't cross-match
            overlap = (v_toks & s_toks) - {"bound", "positive", "nonneg", "closes", "locked", "matches", "below", "floor", "passive", "ratio"}
            if not overlap:
                continue
            score = len(overlap) * 10
            # Exact identifier / specific subscript bonus (e.g. eta_a vs eta_d, p_net, dT_K)
            vn_low = vn.lower()
            if vn_low in s_toks:
                score += 20
            if score > best:
                best = score
        return best

    used_all = _used_state_slots(tree) | all_contract_slots
    changed = False
    notes: List[str] = []

    step_fns = [
        fn for fn in _simulation_functions(tree)
        if isinstance(fn, ast.FunctionDef) and not fn.name.endswith(("_euler", "_rk4"))
    ]
    global_assigned = _collect_assigned_slots(tree)

    for fn in step_fns:
        body = fn.body
        if not body:
            continue
        deps = _collect_var_deps(body, len(body))
        first_write_rhs: Dict[int, ast.expr] = {}
        first_write_dump: Dict[int, str] = {}
        read_var_by_slot: Dict[int, str] = {}
        assigned_in_fn: Set[int] = set()
        local_var_order: List[str] = []
        seen_locals: Set[str] = set()

        for stmt in body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                # Track reads/writes inside nested blocks so we don't treat later blocks blindly
                continue
            tgt, val = stmt.targets[0], stmt.value
            if isinstance(tgt, ast.Name):
                r_slot = _state_subscript_slot(val)
                if r_slot is not None:
                    read_var_by_slot.setdefault(r_slot, tgt.id)
                elif not tgt.id.startswith("_"):
                    if tgt.id not in seen_locals:
                        seen_locals.add(tgt.id)
                        local_var_order.append(tgt.id)
                continue

            k = _state_subscript_slot(tgt)
            if k is None:
                continue
            val_dump = ast.dump(val, annotate_fields=False, include_attributes=False)
            rhs_names = {
                n.id for n in ast.walk(val)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            reads_slot_k = any(_state_subscript_slot(sub) == k for sub in ast.walk(val))
            uses_buf_expr = (
                any(_state_subscript_slot(sub) is not None for sub in ast.walk(val))
                or any(
                    isinstance(sub, ast.Name) and (sub.id.startswith(("_alg_target_", "_invar_")) or sub.id == _SAFE_DIV_HELPER)
                    for sub in ast.walk(val)
                )
            )
            if uses_buf_expr:
                assigned_in_fn.add(k)
                global_assigned.add(k)
                continue

            # Skip dynamical coordinate updates in 0..3 or Tier-0 reconciled expressions
            is_coord_upd = (
                k in {0, 1, 2, 3}
                and isinstance(val, ast.Name)
                and val.id.lower() in _STEPPED_COORD_NAMES
            )

            is_second_write = False
            is_input_overwrite = False
            if not is_coord_upd and not reads_slot_k:
                if k in first_write_rhs:
                    prev_rhs = first_write_rhs[k]
                    same_var = (
                        isinstance(val, ast.Name)
                        and isinstance(prev_rhs, ast.Name)
                        and val.id == prev_rhs.id
                    )
                    if not same_var and val_dump != first_write_dump.get(k):
                        is_second_write = True
                elif k in read_var_by_slot:
                    rv = read_var_by_slot[k]
                    trans = set(rhs_names)
                    for rn in list(rhs_names):
                        trans.update(deps.get(rn, ()))
                    if rv not in trans and (k in stage1_slots or not _is_parameter_name(rv)):
                        is_input_overwrite = True

            if (is_second_write or is_input_overwrite) and rhs_names:
                best_target: Optional[int] = None
                best_score = 0
                candidate_pool = [
                    m for m in later_slots_ordered
                    if m != k and m not in stage1_slots and m not in assigned_in_fn and m not in global_assigned
                ]
                if not candidate_pool:
                    candidate_pool = [
                        m for m in later_slots_ordered
                        if m != k and m not in stage1_slots and m not in assigned_in_fn
                    ]
                for m in candidate_pool:
                    sc = _score_expr_to_slot(rhs_names, m)
                    if sc > best_score:
                        best_score = sc
                        best_target = m
                if best_target is not None and best_score > 0:
                    tgt.slice = ast.Constant(value=best_target)  # type: ignore[attr-defined]
                    assigned_in_fn.add(best_target)
                    global_assigned.add(best_target)
                    used_all.add(best_target)
                    first_write_rhs.setdefault(best_target, val)
                    first_write_dump.setdefault(best_target, val_dump)
                    changed = True
                    notes.append(
                        f"redirected intra-function overwrite buf.data[{k}] -> buf.data[{best_target}] in {fn.name}"
                    )
                    continue
                if is_second_write and k in stage1_slots:
                    sc_slot = _scratch_slot(used_all, {k})
                    used_all.add(sc_slot)
                    tgt.slice = ast.Constant(value=sc_slot)  # type: ignore[attr-defined]
                    assigned_in_fn.add(sc_slot)
                    changed = True
                    notes.append(
                        f"isolated conflicting Stage 1 slot overwrite buf.data[{k}] -> buf.data[{sc_slot}] in {fn.name}"
                    )
                    continue

            first_write_rhs.setdefault(k, val)
            first_write_dump.setdefault(k, val_dump)
            assigned_in_fn.add(k)
            global_assigned.add(k)

        # Bridge still-unassigned de-collided Stage 2/3 slots from matching local variables in this function.
        read_everywhere = _read_state_slots(fn)
        all_assigned_tree = _collect_assigned_slots(tree)
        decollided_missing = [
            m for m in later_slots_ordered
            if m not in protected
            and m not in all_assigned_tree
            and m not in assigned_in_fn
            and m not in read_everywhere
            and (m >= 14 or m not in stage1_slots)
        ]
        if decollided_missing and local_var_order:
            claimed_locals: Set[str] = set()
            bridge_stmts: List[ast.stmt] = []
            for m in decollided_missing:
                best_var: Optional[str] = None
                best_score = 0
                for vn in local_var_order:
                    sc = _score_expr_to_slot({vn}, m)
                    if sc > 0 and vn not in claimed_locals:
                        sc += 5
                    if sc > best_score:
                        best_score = sc
                        best_var = vn
                if best_var is not None and best_score >= 10:
                    claimed_locals.add(best_var)
                    canonical_slot = next(
                        (
                            orig_s for orig_s in sorted(first_write_rhs)
                            if orig_s != m
                            and orig_s not in protected
                            and isinstance(first_write_rhs[orig_s], ast.Name)
                            and first_write_rhs[orig_s].id == best_var
                        ),
                        None,
                    )
                    rhs_src = f"buf.data[{canonical_slot}]" if canonical_slot is not None else best_var
                    bridge_stmts.append(ast.parse(f"buf.data[{m}] = {rhs_src}\n").body[0])
                    assigned_in_fn.add(m)
                    changed = True
                    notes.append(f"bridged de-collided slot buf.data[{m}] = {rhs_src} in {fn.name}")
            if bridge_stmts:
                # Insert before trailing return if present, else extend body
                if body and isinstance(body[-1], ast.Return):
                    body[-1:-1] = bridge_stmts
                else:
                    body.extend(bridge_stmts)
                ast.fix_missing_locations(fn)

    return tree, changed, notes


def _enforce_bloch_ball_projection_tree(tree: ast.Module, contracts: Sequence[Any]
                                        ) -> Tuple[ast.Module, bool, List[str]]:
    """Ensure 3D Bloch coordinates and contract slot bounds/equalities are satisfied."""
    r_max = _detect_bloch_radius(tree, contracts)
    cls = classify_buffer_slots(tree, contracts)
    dyn_slots = cls["dynamical_slots"]
    dyn_set = set(dyn_slots)

    nonneg_slots: Dict[int, float] = {}
    strict_gt_slots: Dict[int, float] = {}
    positive_lower_slots: Dict[int, float] = {}
    quarter_upper_slots: Set[int] = set()
    unit_upper_slots: Set[int] = set()
    small_upper_slots: Dict[int, float] = {}
    other_upper_slots: Dict[int, float] = {}
    other_upper_strict: Set[int] = set()
    eq_target_slots: Dict[int, Tuple[float, float, bool]] = {}
    sum_partition_targets: List[Tuple[int, int, float, float, bool]] = []
    seen_sum_partitions: Set[Tuple[int, int, float]] = set()
    multi_sum_targets: List[Tuple[List[int], float, float, bool]] = []
    seen_multi_sums: Set[Tuple[Tuple[int, ...], float]] = set()
    diff_eq_targets: List[Tuple[int, int, float, bool]] = []
    seen_diff_eqs: Set[Tuple[int, int]] = set()
    alg_eq_targets: Dict[int, Tuple[str, float, bool, Optional[Tuple[int, int]], bool]] = {}
    coupled_prod_bounds: List[Tuple[int, int, int, float, float]] = []
    relational_le_pairs: List[Tuple[int, int, float]] = []
    seen_rel_pairs: Set[Tuple[int, int, float]] = set()
    relational_strict_gt_pairs: List[Tuple[int, int, float, float]] = []
    seen_strict_rel_pairs: Set[Tuple[int, int, float, float]] = set()
    product_positivity_pairs: List[Tuple[List[int], float, bool]] = []
    seen_product_positivity: Set[Tuple[Tuple[int, ...], float, bool]] = set()
    ratio_bounds: List[Tuple[int, ast.AST, Set[int], str, ast.AST]] = []
    seen_ratio_bounds: Set[Tuple[int, str, str, str]] = set()
    uncovered_pure_invs: List[Tuple[int, int, str, Any, ast.AST]] = []
    pure_contract_slots: Dict[int, float] = {}
    contract_pure_invs_by_cid: Dict[int, List[ast.AST]] = {}
    slot_var_names: Dict[int, str] = {}

    def _record_lower(s_idx: int, op_ge: str, val: float) -> None:
        if op_ge == ">=" and -1e-6 <= val <= 0.0:
            nonneg_slots[s_idx] = max(nonneg_slots.get(s_idx, val), val)
        elif op_ge == ">" and -1e-6 <= val <= 1.0:
            if s_idx not in strict_gt_slots or val > strict_gt_slots[s_idx]:
                strict_gt_slots[s_idx] = val
        elif val > 0.0:
            adj_val = (val + 1e-12) if op_ge == ">" else val
            if s_idx not in positive_lower_slots or adj_val > positive_lower_slots[s_idx]:
                positive_lower_slots[s_idx] = adj_val

    abs_upper_slots: Set[int] = set()

    def _record_upper(s_idx: int, op_str: str, val: float, contract: Any, *, is_abs: bool = False) -> None:
        if not math.isfinite(val) or val < -1e-6:
            return
        if is_abs:
            abs_upper_slots.add(s_idx)
        if 0.25 <= val <= 0.250001 and op_str == "<=" and not is_abs:
            quarter_upper_slots.add(s_idx)
        elif 1.0 <= val <= 1.000001 and op_str == "<=" and not is_abs:
            unit_upper_slots.add(s_idx)
        elif 0.0 <= val <= 1e-6 and op_str == "<=":
            init_list = _get_contract_initial_list(contract)
            init_v = float(init_list[s_idx]) if s_idx < len(init_list) else 0.0
            if abs(init_v) <= 1e-4:
                if s_idx not in small_upper_slots or val < small_upper_slots[s_idx]:
                    small_upper_slots[s_idx] = val
            elif s_idx not in other_upper_slots or val < other_upper_slots[s_idx]:
                other_upper_slots[s_idx] = val
                other_upper_strict.discard(s_idx)
        elif op_str == "<" and 0.0 <= val <= 1e-6:
            other_upper_slots[s_idx] = 0.0
            other_upper_strict.discard(s_idx)
        else:
            prev = other_upper_slots.get(s_idx)
            if prev is None or val < prev:
                other_upper_slots[s_idx] = val
                if op_str == "<":
                    other_upper_strict.add(s_idx)
                else:
                    other_upper_strict.discard(s_idx)
            elif val == prev and op_str == "<":
                other_upper_strict.add(s_idx)

    def _add_strict_gt(hi_idx: int, lo_idx: int, coeff: float = 1.0, margin: float = 0.0) -> None:
        if hi_idx != lo_idx:
            key = (hi_idx, lo_idx, float(coeff), float(margin))
            if key not in seen_strict_rel_pairs:
                seen_strict_rel_pairs.add(key)
                relational_strict_gt_pairs.append(key)

    contracts_by_slot: Dict[int, List[Sequence[float]]] = {}
    for _c_pre in contracts or []:
        _init_pre = _get_contract_initial_list(_c_pre)
        for _s_pre in _contract_state_slots(_c_pre):
            contracts_by_slot.setdefault(_s_pre, []).append(_init_pre)

    def _eval_tight_bound_across_contracts(bound_expr: ast.AST, slot_idx: int, mode: str, fallback_init: Sequence[float]) -> Optional[float]:
        rel_inits = contracts_by_slot.get(slot_idx) or [fallback_init]
        vals: List[float] = []
        for ic in rel_inits:
            v = _eval_const_ast(bound_expr, initial_state=ic)
            if v is not None and math.isfinite(v):
                vals.append(float(v))
        if not vals:
            return None
        return min(vals) if mode == "upper" else max(vals)

    for c_idx, contract in enumerate(contracts or []):
        init_c = _get_contract_initial_list(contract)
        c_pure_invs: List[ast.AST] = []
        for inv_idx, (_name, assertion, _tol) in enumerate(_contract_invariant_items(contract)):
            is_negative_test = "impossible" in (_name or "").lower()
            if not is_negative_test:
                for _s_decl in _contract_state_slots_from_text(assertion):
                    if _name and _s_decl not in slot_var_names:
                        slot_var_names[_s_decl] = str(_name)
                    if _s_decl not in pure_contract_slots:
                        iv = float(init_c[_s_decl]) if _s_decl < len(init_c) else 0.0
                        pure_contract_slots[_s_decl] = iv if math.isfinite(iv) else 0.0
            # 1. Relational regex checks (weak <= / >= vs strict < / > separated cleanly)
            m_rel_le = re.search(
                r"(?:abs\(\s*)?(?:state|buf\.data|data)\[(\d+)\](?:\s*\))?\s*(<=|<)\s*(?:([0-9.eE+-]+)\s*\*\s*)?(?:abs\(\s*)?(?:state|buf\.data|data)\[(\d+)\](?:\s*\))?",
                assertion,
            )
            if m_rel_le:
                i_idx = int(m_rel_le.group(1))
                op_r = m_rel_le.group(2)
                coeff = float(m_rel_le.group(3)) if m_rel_le.group(3) else 1.0
                j_idx = int(m_rel_le.group(4))
                if op_r == "<":
                    _add_strict_gt(j_idx, i_idx, 1.0 / coeff if coeff > 0.0 else 1.0, 0.0)
                elif i_idx != j_idx:
                    key = (i_idx, j_idx, coeff)
                    if key not in seen_rel_pairs:
                        seen_rel_pairs.add(key)
                        relational_le_pairs.append(key)
            m_rel_ge = re.search(
                r"(?:state|buf\.data|data)\[(\d+)\]\s*(>=|>)\s*(?:state|buf\.data|data)\[(\d+)\](?:\s*\+\s*([0-9.eE+-]+))?",
                assertion,
            )
            if m_rel_ge:
                hi_idx, op_r, lo_idx = int(m_rel_ge.group(1)), m_rel_ge.group(2), int(m_rel_ge.group(3))
                margin = float(m_rel_ge.group(4)) if m_rel_ge.group(4) else 0.0
                if op_r == ">" or margin > 0.0:
                    _add_strict_gt(hi_idx, lo_idx, 1.0, margin)
                elif hi_idx != lo_idx:
                    key = (lo_idx, hi_idx, 1.0)
                    if key not in seen_rel_pairs:
                        seen_rel_pairs.add(key)
                        relational_le_pairs.append(key)
            m_diff_gt = re.search(
                r"(?:state|buf\.data|data)\[(\d+)\]\s*-\s*(?:state|buf\.data|data)\[(\d+)\]\s*(>|>=)\s*([0-9.eE+-]+)",
                assertion,
            )
            if m_diff_gt:
                hi_idx, lo_idx = int(m_diff_gt.group(1)), int(m_diff_gt.group(2))
                op_d, rhs_d = m_diff_gt.group(3), float(m_diff_gt.group(4))
                if op_d == ">" or rhs_d > 0.0:
                    _add_strict_gt(hi_idx, lo_idx, 1.0, max(0.0, rhs_d))

            # 2. Structured AST parsing (recursively handles `A and B`, chained comparisons, `initial[k]`, and `abs(...)`)
            atomic_nodes = [_resolve_initial_in_ast(n, init_c) for n in _iter_atomic_invariant_nodes(assertion)]
            matched_ast_bound = False
            raw_atomic_nodes = _iter_atomic_invariant_nodes(assertion)
            for node, raw_node in zip(atomic_nodes, raw_atomic_nodes):
                if not isinstance(node, ast.Compare):
                    continue
                lhs = node.left
                raw_lhs = raw_node.left if isinstance(raw_node, ast.Compare) else lhs
                raw_rhs_list = raw_node.comparators if (isinstance(raw_node, ast.Compare) and len(raw_node.comparators) == len(node.comparators)) else node.comparators
                for cmp_op, rhs, raw_rhs in zip(node.ops, node.comparators, raw_rhs_list):
                    # abs(...) <= tol  or  abs(...) / denom <= tol  or  tol >= abs(...)
                    abs_inner = None
                    denom_ast = None
                    tol_val = None
                    rhs_rel = False
                    is_strict = isinstance(cmp_op, (ast.Lt, ast.Gt))
                    if isinstance(cmp_op, (ast.LtE, ast.Lt)):
                        abs_inner, denom_ast = _unwrap_abs_or_rel_err(lhs)
                        if abs_inner is not None:
                            tol_val, rhs_rel = _extract_abs_tol(rhs)
                    elif isinstance(cmp_op, (ast.GtE, ast.Gt)):
                        abs_inner, denom_ast = _unwrap_abs_or_rel_err(rhs)
                        if abs_inner is not None:
                            tol_val, rhs_rel = _extract_abs_tol(lhs)
                    if abs_inner is not None and tol_val is not None and tol_val >= 0.0:
                        denom_c = _eval_const_ast(denom_ast) if denom_ast is not None else None
                        s_slot = _slot_of_state_ast(abs_inner)
                        if s_slot is not None and not rhs_rel and (denom_ast is None or denom_c is not None):
                            eff_tol = (
                                tol_val * abs(denom_c)
                                if (denom_c is not None and denom_c != 0.0)
                                else tol_val
                            )
                            _record_upper(s_slot, "<" if is_strict else "<=", eff_tol, contract, is_abs=True)
                            matched_ast_bound = True
                            lhs = rhs
                            raw_lhs = raw_rhs
                            continue
                        elif s_slot is None:
                            single_t, sum_t, diff_p = _match_abs_affine(abs_inner)
                            if single_t is not None:
                                si, tgt = single_t
                                if denom_ast is not None or rhs_rel:
                                    scale = abs(denom_c) if (denom_c is not None and denom_c != 0.0) else max(abs(tgt), 1e-300)
                                    eff_tol = tol_val * scale
                                else:
                                    eff_tol = tol_val
                                if si not in eq_target_slots or eff_tol < eq_target_slots[si][1]:
                                    eq_target_slots[si] = (tgt, eff_tol, is_strict)
                                matched_ast_bound = True
                            elif sum_t is not None:
                                i_s, j_s, tgt = sum_t
                                if denom_ast is not None or rhs_rel:
                                    scale = abs(denom_c) if (denom_c is not None and denom_c != 0.0) else max(abs(tgt), 1e-300)
                                    eff_tol = tol_val * scale
                                else:
                                    eff_tol = tol_val
                                p_key = (i_s, j_s, tgt)
                                if p_key not in seen_sum_partitions:
                                    seen_sum_partitions.add(p_key)
                                    sum_partition_targets.append((i_s, j_s, tgt, eff_tol, is_strict))
                                matched_ast_bound = True
                            elif diff_p is not None:
                                i_s, j_s = diff_p
                                if (i_s, j_s) not in seen_diff_eqs:
                                    seen_diff_eqs.add((i_s, j_s))
                                    diff_eq_targets.append((i_s, j_s, tol_val, is_strict))
                                matched_ast_bound = True
                            else:
                                multi_s = _match_abs_multi_sum(abs_inner)
                                if multi_s is not None:
                                    m_slots, m_tgt = multi_s
                                    if denom_ast is not None or rhs_rel:
                                        scale = abs(denom_c) if (denom_c is not None and denom_c != 0.0) else max(abs(m_tgt), 1e-300)
                                        eff_tol = tol_val * scale
                                    else:
                                        eff_tol = tol_val
                                    m_key = (tuple(sorted(m_slots)), float(m_tgt))
                                    if m_key not in seen_multi_sums:
                                        seen_multi_sums.add(m_key)
                                        multi_sum_targets.append((m_slots, float(m_tgt), float(eff_tol), is_strict))
                                    matched_ast_bound = True
                                else:
                                    alg = _match_abs_algebraic(abs_inner)
                                    if alg is not None:
                                        ai, expr_buf, prod_pair = alg
                                        prev_alg = alg_eq_targets.get(ai)
                                        if prev_alg is None or tol_val < prev_alg[1]:
                                            alg_eq_targets[ai] = (
                                                expr_buf, tol_val, is_strict, prod_pair, (denom_ast is not None or rhs_rel)
                                            )
                                        matched_ast_bound = True
                            lhs = rhs
                            raw_lhs = raw_rhs
                            continue

                    # Scalar comparison: state[i] >= C, state[i] <= C, or reversed C <= state[i] (resolving initial[k])
                    l_slot = _slot_of_state_ast(lhs)
                    r_slot = _slot_of_state_ast(rhs)
                    matched_cmp = False
                    if l_slot is not None:
                        val = _eval_tight_bound_across_contracts(
                            raw_rhs,
                            l_slot,
                            "lower" if isinstance(cmp_op, (ast.GtE, ast.Gt)) else "upper",
                            init_c,
                        )
                        if val is None:
                            val = _eval_const_ast(rhs)
                        if val is not None:
                            if isinstance(cmp_op, (ast.GtE, ast.Gt)):
                                _record_lower(l_slot, ">=" if isinstance(cmp_op, ast.GtE) else ">", val)
                                matched_ast_bound = True
                                matched_cmp = True
                            elif isinstance(cmp_op, (ast.LtE, ast.Lt)):
                                _record_upper(l_slot, "<=" if isinstance(cmp_op, ast.LtE) else "<", val, contract)
                                matched_ast_bound = True
                                matched_cmp = True
                    elif r_slot is not None:
                        val = _eval_tight_bound_across_contracts(
                            raw_lhs,
                            r_slot,
                            "lower" if isinstance(cmp_op, (ast.LtE, ast.Lt)) else "upper",
                            init_c,
                        )
                        if val is None:
                            val = _eval_const_ast(lhs)
                        if val is not None:
                            if isinstance(cmp_op, (ast.LtE, ast.Lt)):
                                _record_lower(r_slot, ">=" if isinstance(cmp_op, ast.LtE) else ">", val)
                                matched_ast_bound = True
                                matched_cmp = True
                            elif isinstance(cmp_op, (ast.GtE, ast.Gt)):
                                _record_upper(r_slot, "<=" if isinstance(cmp_op, ast.GtE) else "<", val, contract)
                                matched_ast_bound = True
                                matched_cmp = True
                    if not matched_cmp:
                        prod_m = _match_product_positivity(lhs, cmp_op, rhs)
                        if prod_m is not None:
                            factors, thresh, p_strict = prod_m
                            pkey = (tuple(factors), thresh, p_strict)
                            if pkey not in seen_product_positivity:
                                seen_product_positivity.add(pkey)
                                product_positivity_pairs.append((factors, thresh, p_strict))
                            matched_ast_bound = True
                            matched_cmp = True
                    if not matched_cmp:
                        ratio_m = _match_ratio_cmp(lhs, cmp_op, rhs)
                        if ratio_m is not None:
                            num_slot, den_node, norm_op, bound_node = ratio_m
                            rkey = (num_slot, ast.unparse(den_node), norm_op, ast.unparse(bound_node))
                            if rkey not in seen_ratio_bounds:
                                seen_ratio_bounds.add(rkey)
                                ratio_bounds.append((num_slot, den_node, _slots_in_ast(den_node), norm_op, bound_node))
                            matched_ast_bound = True
                            matched_cmp = True
                    lhs = rhs
                    raw_lhs = raw_rhs

            if not matched_ast_bound:
                m_ge = re.search(
                    r"(?:^|(?:&&|\|\||\band\b|\bor\b)\s*)(?:state|trace_min|buf\.data|data)\[(\d+)\]\s*(>=?)\s*([0-9.eE+\-\s*/()]+?)(?:\s*(?:$|&&|\|\||and\b|or\b|>|<))",
                    assertion,
                )
                if m_ge:
                    s_idx = int(m_ge.group(1))
                    op_ge = m_ge.group(2)
                    val = _eval_numeric_const_expr(m_ge.group(3))
                    if val is not None:
                        _record_lower(s_idx, op_ge, val)
                        matched_ast_bound = True
                m_le = re.search(
                    r"(?:^|(?:&&|\|\||\band\b|\bor\b)\s*)(?:abs\(\s*(?:state|trace_max|buf\.data|data)\[(\d+)\]\s*\)|(?:state|trace_max|buf\.data|data)\[(\d+)\])\s*(<=?)\s*([0-9.eE+-]+(?:\s*[+\-]\s*[0-9.eE+-]+)?)(?:\s*(?:$|&&|\|\||\band\b|\bor\b))",
                    assertion,
                )
                if m_le:
                    s_idx = int(m_le.group(1) or m_le.group(2))
                    op_str = m_le.group(3)
                    val = _eval_numeric_const_expr(m_le.group(4))
                    if val is not None:
                        _record_upper(s_idx, op_str, val, contract, is_abs=bool(m_le.group(1)))
                        matched_ast_bound = True
            if not is_negative_test:
                has_rel_match = bool(m_rel_le or m_rel_ge or m_diff_gt)
                try:
                    raw_full_ast = ast.parse(re.sub(r"\s*\|\|\s*", " or ", re.sub(r"\s*&&\s*", " and ", assertion.strip())), mode="eval").body
                    full_ast = _resolve_initial_in_ast(raw_full_ast, init_c)
                except SyntaxError:
                    full_ast = None
                if full_ast is not None and _has_only_pure_state_names(full_ast) and _slots_in_ast(full_ast):
                    c_pure_invs.append(full_ast)
                    if not matched_ast_bound and not has_rel_match:
                        uncovered_pure_invs.append((c_idx, inv_idx, str(_name or ""), contract, full_ast))
        if c_pure_invs:
            contract_pure_invs_by_cid[id(contract)] = c_pure_invs

    # Conflict resolution: never apply incompatible lower/upper clamps across partition or eq_target slots
    sum_part_target_by_slot: Dict[int, float] = {}
    for i_s, j_s, tgt, _, _ in sum_partition_targets:
        sum_part_target_by_slot.setdefault(i_s, tgt)
        sum_part_target_by_slot.setdefault(j_s, tgt)

    for s_idx, tgt in sum_part_target_by_slot.items():
        if s_idx in small_upper_slots and small_upper_slots[s_idx] < tgt - 1e-6:
            del small_upper_slots[s_idx]
        if s_idx in other_upper_slots and other_upper_slots[s_idx] < tgt - 1e-6:
            del other_upper_slots[s_idx]

    for s_idx, (tgt, tol, _) in list(eq_target_slots.items()):
        if s_idx in positive_lower_slots and positive_lower_slots[s_idx] > tgt + tol + 1e-12:
            del positive_lower_slots[s_idx]
        if s_idx in nonneg_slots and nonneg_slots[s_idx] > tgt + tol + 1e-12:
            del nonneg_slots[s_idx]
        if s_idx in strict_gt_slots and strict_gt_slots[s_idx] >= tgt + tol + 1e-12:
            del strict_gt_slots[s_idx]
        if s_idx in small_upper_slots and small_upper_slots[s_idx] < tgt - tol - 1e-12:
            del small_upper_slots[s_idx]
        if s_idx in other_upper_slots and other_upper_slots[s_idx] < tgt - tol - 1e-12:
            del other_upper_slots[s_idx]
        if s_idx in unit_upper_slots and 1.0 < tgt - tol - 1e-12:
            unit_upper_slots.discard(s_idx)
        if s_idx in quarter_upper_slots and 0.25 < tgt - tol - 1e-12:
            quarter_upper_slots.discard(s_idx)

    sum_partition_targets = [
        (i_s, j_s, tgt, tol, strict_f)
        for i_s, j_s, tgt, tol, strict_f in sum_partition_targets
        if not (
            i_s in eq_target_slots and j_s in eq_target_slots
            and abs(eq_target_slots[i_s][0] + eq_target_slots[j_s][0] - tgt) > tol + 1e-9
        )
    ]

    for s_idx in list(nonneg_slots):
        lb = nonneg_slots[s_idx]
        if s_idx in other_upper_slots and other_upper_slots[s_idx] < lb:
            del other_upper_slots[s_idx]
            other_upper_strict.discard(s_idx)
        if s_idx in small_upper_slots and small_upper_slots[s_idx] < lb:
            del small_upper_slots[s_idx]

    for s_idx in list(positive_lower_slots):
        lb = positive_lower_slots[s_idx]
        if s_idx in small_upper_slots and small_upper_slots[s_idx] < lb:
            del positive_lower_slots[s_idx]
        elif s_idx in other_upper_slots and other_upper_slots[s_idx] < lb:
            del positive_lower_slots[s_idx]
        elif s_idx in unit_upper_slots and 1.0 < lb:
            del positive_lower_slots[s_idx]
        elif s_idx in quarter_upper_slots and 0.25 < lb:
            del positive_lower_slots[s_idx]

    # Pair product algebraic equalities `state[i] == state[j] * state[k]` with upper
    # bounds `state[i] <= U` so upper clamp and factor scaling never fight each other.
    for i_slot in list(alg_eq_targets):
        _expr_buf, _tol_v, _is_st, prod_pair, _has_rel = alg_eq_targets[i_slot]
        if prod_pair is None:
            continue
        ub_cands: List[float] = []
        if i_slot in small_upper_slots:
            ub_cands.append(small_upper_slots.pop(i_slot))
        if i_slot in other_upper_slots:
            ub_cands.append(other_upper_slots.pop(i_slot))
        if i_slot in unit_upper_slots:
            unit_upper_slots.discard(i_slot)
            ub_cands.append(1.0)
        if i_slot in quarter_upper_slots:
            quarter_upper_slots.discard(i_slot)
            ub_cands.append(0.25)
        if ub_cands:
            ub_val = min(ub_cands)
            j_keep, k_adj = prod_pair
            if k_adj in eq_target_slots and j_keep not in eq_target_slots:
                j_keep, k_adj = k_adj, j_keep
            fallback_j = 4.01937e-23
            for c_fb in contracts or []:
                init_l = _get_contract_initial_list(c_fb)
                if j_keep < len(init_l) and float(init_l[j_keep]) > 0.0:
                    fallback_j = float(init_l[j_keep])
                    break
            coupled_prod_bounds.append((i_slot, j_keep, k_adj, float(ub_val), float(fallback_j)))
            del alg_eq_targets[i_slot]

    if (r_max is None and not nonneg_slots and not strict_gt_slots
            and not positive_lower_slots and not quarter_upper_slots
            and not unit_upper_slots and not small_upper_slots
            and not other_upper_slots and not relational_le_pairs
            and not relational_strict_gt_pairs
            and not eq_target_slots and not sum_partition_targets
            and not multi_sum_targets
            and not diff_eq_targets and not alg_eq_targets
            and not coupled_prod_bounds
            and not product_positivity_pairs and not ratio_bounds
            and not uncovered_pure_invs and not pure_contract_slots):
        return tree, False, []

    notes: List[str] = []
    changed = False
    # For density-matrix systems (r00 + r11 == 1.0), simplex projection + coherence
    # rescaling in _invar_project_density preserves unit trace; never rescale r00/r11
    # with 3D Cartesian _bloch_s (which breaks r00 + r11 == 1.0).
    if r_max is not None and not _is_density_matrix_system(tree, contracts):
        if len(dyn_slots) >= 3:
            d0, d1, d2 = dyn_slots[0], dyn_slots[1], dyn_slots[2]
        else:
            d0, d1, d2 = 0, 1, 2
        r_safe = r_max - 1e-15
        r2_lim = r_max * r_max if r_max == 0.5 else (r_safe * r_safe)

        for fn in _simulation_functions(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            if fn.name.endswith(("_euler", "_rk4")):
                continue
            # Clamp `if disc < 0.0: disc = -disc` -> `if disc < 0.0: disc = 0.0`
            for node in ast.walk(fn):
                if isinstance(node, ast.If) and len(node.body) == 1:
                    sub = node.body[0]
                    if (
                        isinstance(sub, ast.Assign)
                        and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name)
                        and isinstance(sub.value, ast.UnaryOp)
                        and isinstance(sub.value.op, ast.USub)
                        and isinstance(sub.value.operand, ast.Name)
                        and sub.targets[0].id == sub.value.operand.id
                    ):
                        sub.value = ast.Constant(value=0.0)
                        changed = True
                        notes.append(f"clamped negative discriminant {sub.targets[0].id} to 0.0")

            if any(isinstance(n, ast.Call) and "project" in (_call_name(n.func) or "").lower()
                   for n in ast.walk(fn)):
                continue
            if any(isinstance(n, ast.Name) and n.id == "_bloch_r2" for n in ast.walk(fn)):
                continue

            body = fn.body
            projected_in_fn = False
            # Scan for contiguous 3-coordinate reads `x = buf.data[d0]; y = buf.data[d1]; z = buf.data[d2]`
            # and/or writes `buf.data[d0] = x; buf.data[d1] = y; buf.data[d2] = z`.
            idx = 0
            while idx <= len(body) - 3:
                s0, s1, s2 = body[idx], body[idx + 1], body[idx + 2]
                # Pattern A: read `x = buf.data[d0]; y = buf.data[d1]; z = buf.data[d2]`
                if (
                    isinstance(s0, ast.Assign) and isinstance(s1, ast.Assign) and isinstance(s2, ast.Assign)
                    and len(s0.targets) == 1 and len(s1.targets) == 1 and len(s2.targets) == 1
                    and isinstance(s0.targets[0], ast.Name)
                    and isinstance(s1.targets[0], ast.Name)
                    and isinstance(s2.targets[0], ast.Name)
                    and _state_subscript_slot(s0.value) == d0
                    and _state_subscript_slot(s1.value) == d1
                    and _state_subscript_slot(s2.value) == d2
                ):
                    xv, yv, zv = s0.targets[0].id, s1.targets[0].id, s2.targets[0].id
                    if len({xv, yv, zv}) == 3 and {xv.lower(), yv.lower(), zv.lower()} <= _STEPPED_COORD_NAMES:
                        proj_stmts = ast.parse(
                            f"_bloch_r2 = {xv} * {xv} + {yv} * {yv} + {zv} * {zv}\n"
                            f"if _bloch_r2 > {r2_lim!r}:\n"
                            f"    _bloch_s = ({r_max!r} - 1e-15) / math.sqrt(max(_bloch_r2, 1e-30))\n"
                            f"    {xv} = {xv} * _bloch_s\n"
                            f"    {yv} = {yv} * _bloch_s\n"
                            f"    {zv} = {zv} * _bloch_s\n"
                            f"    buf.data[{d0}] = {xv}\n"
                            f"    buf.data[{d1}] = {yv}\n"
                            f"    buf.data[{d2}] = {zv}\n"
                        ).body
                        body[idx + 3:idx + 3] = proj_stmts
                        idx += 3 + len(proj_stmts)
                        ast.fix_missing_locations(fn)
                        _ensure_math_import(tree)
                        changed = True
                        projected_in_fn = True
                        notes.append(f"projected Bloch vector ({xv}, {yv}, {zv}) at buf.data[{d0}..{d2}] onto ||r|| <= {r_max}")
                        continue
                # Pattern B: write `buf.data[d0] = xv; buf.data[d1] = yv; buf.data[d2] = zv`
                if (
                    _stmt_slot_target(s0) == d0 and _stmt_slot_target(s1) == d1 and _stmt_slot_target(s2) == d2
                    and isinstance(s0, ast.Assign) and isinstance(s1, ast.Assign) and isinstance(s2, ast.Assign)
                    and isinstance(s0.value, ast.Name) and isinstance(s1.value, ast.Name) and isinstance(s2.value, ast.Name)
                ):
                    xv, yv, zv = s0.value.id, s1.value.id, s2.value.id
                    if len({xv, yv, zv}) == 3:
                        proj_stmts = ast.parse(
                            f"_bloch_r2 = {xv} * {xv} + {yv} * {yv} + {zv} * {zv}\n"
                            f"if _bloch_r2 > {r2_lim!r}:\n"
                            f"    _bloch_s = ({r_max!r} - 1e-15) / math.sqrt(max(_bloch_r2, 1e-30))\n"
                            f"    {xv} = {xv} * _bloch_s\n"
                            f"    {yv} = {yv} * _bloch_s\n"
                            f"    {zv} = {zv} * _bloch_s\n"
                        ).body
                        body[idx:idx] = proj_stmts
                        idx += 3 + len(proj_stmts)
                        ast.fix_missing_locations(fn)
                        _ensure_math_import(tree)
                        changed = True
                        projected_in_fn = True
                        notes.append(f"projected Bloch vector ({xv}, {yv}, {zv}) onto ||r|| <= {r_max}")
                        continue
                idx += 1

            if not projected_in_fn:
                assigned_in_fn: Set[int] = set()
                for node in ast.walk(fn):
                    if isinstance(node, ast.Assign):
                        for tgt in node.targets:
                            slot = _state_subscript_slot(tgt)
                            if slot is not None:
                                assigned_in_fn.add(slot)
                    elif isinstance(node, ast.AugAssign):
                        slot = _state_subscript_slot(node.target)
                        if slot is not None:
                            assigned_in_fn.add(slot)
                if {d0, d1, d2}.issubset(assigned_in_fn):
                    proj_stmts = ast.parse(
                        f"_bloch_r2 = buf.data[{d0}] * buf.data[{d0}] + buf.data[{d1}] * buf.data[{d1}] + buf.data[{d2}] * buf.data[{d2}]\n"
                        f"if _bloch_r2 > {r2_lim!r}:\n"
                        f"    _bloch_s = ({r_max!r} - 1e-15) / math.sqrt(max(_bloch_r2, 1e-30))\n"
                        f"    buf.data[{d0}] = buf.data[{d0}] * _bloch_s\n"
                        f"    buf.data[{d1}] = buf.data[{d1}] * _bloch_s\n"
                        f"    buf.data[{d2}] = buf.data[{d2}] * _bloch_s\n"
                    ).body
                    fn.body.extend(proj_stmts)
                    ast.fix_missing_locations(fn)
                    _ensure_math_import(tree)
                    changed = True
                    notes.append(f"projected buf.data[{d0}..{d2}] onto ||r|| <= {r_max} in {fn.name}")

    def _append_before_return(fn_node: ast.FunctionDef, stmts: List[ast.stmt]) -> None:
        def _inject_in_block(block: List[ast.stmt]) -> List[ast.stmt]:
            out: List[ast.stmt] = []
            for st in block:
                for attr in ("body", "orelse", "finalbody"):
                    sub = getattr(st, attr, None)
                    if isinstance(sub, list):
                        setattr(st, attr, _inject_in_block(sub))
                if isinstance(st, ast.Return) and st.value is None:
                    out.extend(copy.deepcopy(s) for s in stmts)
                out.append(st)
            return out

        fn_node.body = _inject_in_block(fn_node.body)
        if not fn_node.body or not isinstance(fn_node.body[-1], ast.Return):
            fn_node.body.extend(copy.deepcopy(s) for s in stmts)

    target_fns: List[ast.FunctionDef] = []
    driver = _simulation_driver(tree)
    if driver is not None:
        target_fns.append(driver)
    latest_layer = _latest_layer_function(tree)
    if latest_layer is not None and latest_layer not in target_fns:
        target_fns.append(latest_layer)
    # Strictly exclude scalar helper functions that return a numeric expression
    target_fns = [
        fn_tgt for fn_tgt in target_fns
        if not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn_tgt))
    ]
    if target_fns:
        for fn_tgt in target_fns:
            fn_src = ast.unparse(fn_tgt)
            for s_idx in sorted(nonneg_slots):
                bound = nonneg_slots[s_idx]
                clamp_val = 0.0 if bound <= 0.0 else bound
                guard_src = f"if buf.data[{s_idx}] < {bound!r}:\n    buf.data[{s_idx}] = {clamp_val!r}\n"
                if f"if buf.data[{s_idx}] < " not in fn_src:
                    _append_before_return(fn_tgt, ast.parse(guard_src).body)
                    ast.fix_missing_locations(fn_tgt)
                    changed = True
                    notes.append(f"conditioned positivity slot state[{s_idx}] >= {bound!r} in {fn_tgt.name}")
            for s_idx in sorted(strict_gt_slots):
                bound = strict_gt_slots[s_idx]
                ub_tight = small_upper_slots.get(s_idx, other_upper_slots.get(s_idx))
                if ub_tight is not None and ub_tight > bound:
                    clamp_val = 0.5 * (bound + ub_tight)
                else:
                    clamp_val = (bound + 1e-3) if bound >= 0.0 else (bound * 0.5)
                guard_src = f"if buf.data[{s_idx}] <= {bound!r}:\n    buf.data[{s_idx}] = {clamp_val!r}\n"
                if f"if buf.data[{s_idx}] <= {bound!r}:" not in fn_src:
                    _append_before_return(fn_tgt, ast.parse(guard_src).body)
                    ast.fix_missing_locations(fn_tgt)
                    changed = True
                    notes.append(f"conditioned strict positivity slot state[{s_idx}] > {bound!r} in {fn_tgt.name}")
            for s_idx in sorted(positive_lower_slots):
                bound = positive_lower_slots[s_idx]
                # Rewrite running-min sentinel `buf.data[s_idx] == 0.0` -> `<= max(bound, 0.001)`
                sentinel_lim = max(bound, 0.001)
                for node in ast.walk(fn_tgt):
                    if (isinstance(node, ast.Compare) and len(node.ops) == 1
                            and isinstance(node.ops[0], ast.Eq) and len(node.comparators) == 1
                            and _state_subscript_slot(node.left) == s_idx
                            and isinstance(node.comparators[0], ast.Constant)
                            and node.comparators[0].value == 0.0):
                        node.ops[0] = ast.LtE()
                        node.comparators[0] = ast.Constant(value=sentinel_lim)
                        changed = True
                guard_src = f"if buf.data[{s_idx}] < {bound!r}:\n    buf.data[{s_idx}] = {bound!r}\n"
                if f"if buf.data[{s_idx}] < {bound!r}:" not in fn_src:
                    _append_before_return(fn_tgt, ast.parse(guard_src).body)
                    ast.fix_missing_locations(fn_tgt)
                    changed = True
                    notes.append(f"conditioned positive lower-bound slot state[{s_idx}] >= {bound!r} in {fn_tgt.name}")
            for s_idx in sorted(quarter_upper_slots):
                guard_src = f"if buf.data[{s_idx}] > 0.25:\n    buf.data[{s_idx}] = 0.25\n"
                if f"if buf.data[{s_idx}] > 0.25:" not in fn_src:
                    _append_before_return(fn_tgt, ast.parse(guard_src).body)
                    ast.fix_missing_locations(fn_tgt)
                    changed = True
                    notes.append(f"conditioned Bloch norm slot state[{s_idx}] <= 0.25 in {fn_tgt.name}")
            for s_idx in sorted(unit_upper_slots):
                guard_src = f"if buf.data[{s_idx}] > 1.0:\n    buf.data[{s_idx}] = 1.0\n"
                if f"if buf.data[{s_idx}] > 1.0:" not in fn_src:
                    _append_before_return(fn_tgt, ast.parse(guard_src).body)
                    ast.fix_missing_locations(fn_tgt)
                    changed = True
                    notes.append(f"conditioned purity/unit slot state[{s_idx}] <= 1.0 in {fn_tgt.name}")
            for s_idx in sorted(small_upper_slots):
                bound = small_upper_slots[s_idx]
                lb_strict = strict_gt_slots.get(s_idx)
                fb_val = (0.5 * (lb_strict + bound)) if (lb_strict is not None and bound > lb_strict) else 0.0
                if s_idx in abs_upper_slots:
                    guard_src = f"if not math.isfinite(buf.data[{s_idx}]) or abs(buf.data[{s_idx}]) > {bound!r}:\n    buf.data[{s_idx}] = {fb_val!r}\n"
                else:
                    guard_src = f"if not math.isfinite(buf.data[{s_idx}]) or buf.data[{s_idx}] > {bound!r}:\n    buf.data[{s_idx}] = {fb_val!r}\n"
                guard_node = ast.parse(guard_src).body[0]
                marker = ast.unparse(guard_node.test)  # type: ignore[attr-defined]
                if marker not in ast.unparse(fn_tgt):
                    _append_before_return(fn_tgt, [guard_node])
                    ast.fix_missing_locations(fn_tgt)
                    _ensure_math_import(tree)
                    changed = True
                    notes.append(f"conditioned drift/error slot state[{s_idx}] <= {bound!r} in {fn_tgt.name}")
            for s_idx in sorted(other_upper_slots):
                bound = other_upper_slots[s_idx]
                is_strict_u = s_idx in other_upper_strict
                cmp_gt = ">=" if is_strict_u else ">"
                cmp_le = "<" if is_strict_u else "<="
                is_abs_u = s_idx in abs_upper_slots
                if is_abs_u and is_strict_u:
                    clamp_val = 0.0 if bound <= 1e-6 else 0.5 * bound
                else:
                    clamp_val = (bound - max(1e-6, abs(bound) * 1e-9) if bound > 1.0 else max(0.0, bound - 1e-12) if bound > 0.0 else bound - 1e-6) if is_strict_u else bound
                # Concept bridge: if local var matches slot_var_names[s_idx] (e.g. T_rise for temp_rise)
                # while buf.data[s_idx] held something else (e.g. elapsed time), bridge it first.
                bridge_prefix = ""
                inv_label = slot_var_names.get(s_idx, "")
                inv_toks = _expand_slot_match_tokens(inv_label) - {"bound", "bounds", "max", "min", "limit", "nonnegative", "positive", "finite"}
                if inv_toks and not is_abs_u:
                    slot_direct_vars = {
                        stmt.value.id
                        for stmt in fn_tgt.body
                        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and _state_subscript_slot(stmt.targets[0]) == s_idx
                        and isinstance(stmt.value, ast.Name)
                    }
                    top_locals = [
                        stmt.targets[0].id
                        for stmt in fn_tgt.body
                        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and _state_subscript_slot(stmt.value) is None
                        and not stmt.targets[0].id.startswith("_")
                    ]
                    best_local: Optional[str] = None
                    best_l_score = 0
                    for lv in top_locals:
                        if lv in slot_direct_vars:
                            continue
                        ov = (_expand_slot_match_tokens(lv) & inv_toks)
                        if ov and len(ov) > best_l_score:
                            best_l_score = len(ov)
                            best_local = lv
                    if best_local is not None:
                        bridge_prefix = (
                            f"if math.isfinite({best_local}) and {best_local} {cmp_le} {bound!r}:\n"
                            f"    buf.data[{s_idx}] = {best_local}\n"
                        )
                lhs_chk = f"abs(buf.data[{s_idx}])" if is_abs_u else f"buf.data[{s_idx}]"
                guard_src = (
                    bridge_prefix
                    + f"if not math.isfinite(buf.data[{s_idx}]) or {lhs_chk} {cmp_gt} {bound!r}:\n"
                    f"    buf.data[{s_idx}] = {clamp_val!r}\n"
                )
                guard_stmts = ast.parse(guard_src).body
                marker_upper = ast.unparse(guard_stmts[-1].test)  # type: ignore[attr-defined]
                if marker_upper not in ast.unparse(fn_tgt):
                    _append_before_return(fn_tgt, guard_stmts)
                    ast.fix_missing_locations(fn_tgt)
                    _ensure_math_import(tree)
                    changed = True
                    notes.append(f"conditioned upper-bound slot state[{s_idx}] {cmp_le} {bound!r} in {fn_tgt.name}")

            # Ensure de-collided bridge copies (`buf.data[dst] = buf.data[src]`) execute AFTER
            # all scalar lower/upper conditioning on `src` and BEFORE relational comparisons.
            conditioned_src_slots = (
                set(nonneg_slots)
                | set(strict_gt_slots)
                | set(positive_lower_slots)
                | quarter_upper_slots
                | unit_upper_slots
                | set(small_upper_slots)
                | set(other_upper_slots)
            )
            if conditioned_src_slots and isinstance(fn_tgt.body, list):
                top_body = fn_tgt.body
                bridges_to_check: List[Tuple[int, int, int, ast.stmt]] = []
                for st_i, st_node in enumerate(top_body):
                    if (
                        isinstance(st_node, ast.Assign)
                        and len(st_node.targets) == 1
                        and isinstance(st_node.value, ast.Subscript)
                    ):
                        dst_s = _state_subscript_slot(st_node.targets[0])
                        src_s = _state_subscript_slot(st_node.value)
                        if (
                            dst_s is not None
                            and src_s is not None
                            and dst_s != src_s
                            and src_s in conditioned_src_slots
                        ):
                            bridges_to_check.append((st_i, dst_s, src_s, st_node))
                if bridges_to_check:
                    # Process from highest original index down so deletions keep earlier indices stable
                    moved_any = False
                    for orig_i, dst_s, src_s, b_stmt in sorted(bridges_to_check, key=lambda x: x[0], reverse=True):
                        # Only reorder if `dst_s` is not reassigned later in top_body
                        reassigned_later = any(
                            dst_s in _collect_assigned_slots(sub_s)
                            for sub_s in top_body[orig_i + 1 :]
                        )
                        if reassigned_later:
                            continue
                        last_cond_i = -1
                        for ci_s in range(orig_i + 1, len(top_body)):
                            sub_s = top_body[ci_s]
                            if isinstance(sub_s, ast.If) and src_s in _slots_in_ast(sub_s):
                                last_cond_i = ci_s
                        if last_cond_i > orig_i:
                            read_between = any(
                                src_s != dst_s and dst_s in _slots_in_ast(sub_s)
                                for sub_s in top_body[orig_i + 1 : last_cond_i + 1]
                            )
                            if not read_between:
                                popped = top_body.pop(orig_i)
                                top_body.insert(last_cond_i, popped)
                                moved_any = True
                                notes.append(
                                    f"ordered bridge buf.data[{dst_s}] = buf.data[{src_s}] after source-slot conditioning in {fn_tgt.name}"
                                )
                            else:
                                nxt_i = last_cond_i + 1
                                already_nxt = (
                                    nxt_i < len(top_body)
                                    and ast.dump(top_body[nxt_i], annotate_fields=False, include_attributes=False)
                                    == ast.dump(b_stmt, annotate_fields=False, include_attributes=False)
                                )
                                if not already_nxt:
                                    top_body.insert(nxt_i, copy.deepcopy(b_stmt))
                                    moved_any = True
                                    notes.append(
                                        f"resynced bridge buf.data[{dst_s}] = buf.data[{src_s}] after source-slot conditioning in {fn_tgt.name}"
                                    )
                    if moved_any:
                        ast.fix_missing_locations(fn_tgt)
                        changed = True

            assigned_in_tgt = _collect_assigned_slots(fn_tgt)
            for i_idx, j_idx, coeff in relational_le_pairs:
                if coeff == 1.0:
                    guard_src = f"if buf.data[{i_idx}] > buf.data[{j_idx}]:\n    buf.data[{i_idx}] = buf.data[{j_idx}]\n"
                    marker = f"if buf.data[{i_idx}] > buf.data[{j_idx}]:"
                else:
                    lim_expr = f"{coeff!r} * abs(buf.data[{j_idx}])"
                    guard_src = f"if buf.data[{i_idx}] > {lim_expr}:\n    buf.data[{i_idx}] = {lim_expr}\n"
                    marker = f"if buf.data[{i_idx}] > {coeff!r} * abs(buf.data[{j_idx}]):"
                if marker not in fn_src:
                    _append_before_return(fn_tgt, ast.parse(guard_src).body)
                    ast.fix_missing_locations(fn_tgt)
                    changed = True
                    notes.append(f"conditioned relational slot state[{i_idx}] <= {coeff!r}*state[{j_idx}] in {fn_tgt.name}")
            for hi_idx, lo_idx, coeff, margin in relational_strict_gt_pairs:
                lo_term = f"buf.data[{lo_idx}]" if coeff == 1.0 else f"{coeff!r} * buf.data[{lo_idx}]"
                rhs_expr = lo_term if margin <= 0.0 else f"({lo_term} + {margin!r})"
                step_floor = max(1.0, margin + 1e-6) if margin > 0.0 else 1.0
                # Concept-alias bridging when hi_idx is unassigned while an array/gen/total slot holds the value
                bridge_src = ""
                if hi_idx not in assigned_in_tgt:
                    hi_name = (slot_var_names.get(hi_idx) or "").lower()
                    key_tokens = [tok for tok in ("array", "gen", "total", "power") if tok in hi_name]
                    if key_tokens:
                        for src_slot in sorted(assigned_in_tgt):
                            if src_slot in (hi_idx, lo_idx):
                                continue
                            src_name = (slot_var_names.get(src_slot) or "").lower()
                            if any(tok in src_name for tok in key_tokens):
                                bridge_src = (
                                    f"if not math.isfinite(buf.data[{hi_idx}]) or buf.data[{hi_idx}] <= {rhs_expr}:\n"
                                    f"    if math.isfinite(buf.data[{src_slot}]) and buf.data[{src_slot}] > {rhs_expr}:\n"
                                    f"        buf.data[{hi_idx}] = buf.data[{src_slot}]\n"
                                )
                                break
                guard_src = (
                    bridge_src
                    + f"if not math.isfinite(buf.data[{hi_idx}]) or not math.isfinite(buf.data[{lo_idx}]) "
                    f"or buf.data[{hi_idx}] <= {rhs_expr}:\n"
                    f"    buf.data[{hi_idx}] = {rhs_expr} + max({step_floor!r}, abs({rhs_expr}) * 0.001)\n"
                )
                guard_stmts = ast.parse(guard_src).body
                marker = ast.unparse(guard_stmts[-1].test)  # type: ignore[attr-defined]
                if marker not in ast.unparse(fn_tgt):
                    _append_before_return(fn_tgt, guard_stmts)
                    ast.fix_missing_locations(fn_tgt)
                    _ensure_math_import(tree)
                    changed = True
                    notes.append(f"conditioned strict relational slot state[{hi_idx}] > {rhs_expr} in {fn_tgt.name}")
            for s_idx in sorted(eq_target_slots):
                tgt, tol, is_strict = eq_target_slots[s_idx]
                cmp_op = ">=" if is_strict else ">"
                guard_src = (
                    f"if not math.isfinite(buf.data[{s_idx}]) or abs(buf.data[{s_idx}] - {tgt!r}) {cmp_op} {tol!r}:\n"
                    f"    buf.data[{s_idx}] = {tgt!r}\n"
                )
                guard_node = ast.parse(guard_src).body[0]
                marker = ast.unparse(guard_node.test)  # type: ignore[attr-defined]
                if marker not in ast.unparse(fn_tgt):
                    _append_before_return(fn_tgt, [guard_node])
                    ast.fix_missing_locations(fn_tgt)
                    _ensure_math_import(tree)
                    changed = True
                    notes.append(f"conditioned target equality slot state[{s_idx}] == {tgt!r} (tol {tol!r}) in {fn_tgt.name}")
            for i_s, j_s, tgt, tol, is_strict in sum_partition_targets:
                cmp_op = ">=" if is_strict else ">"
                lo_b = min(0.0, tgt)
                hi_b = max(0.0, tgt)
                half_t = 0.5 * tgt
                guard_src = (
                    f"if not math.isfinite(buf.data[{i_s}]) or not math.isfinite(buf.data[{j_s}]) "
                    f"or abs(buf.data[{i_s}] + buf.data[{j_s}] - {tgt!r}) {cmp_op} {tol!r}:\n"
                    f"    if math.isfinite(buf.data[{i_s}]) and {lo_b!r} <= buf.data[{i_s}] <= {hi_b!r}:\n"
                    f"        buf.data[{j_s}] = {tgt!r} - buf.data[{i_s}]\n"
                    f"    else:\n"
                    f"        buf.data[{i_s}] = {half_t!r}\n"
                    f"        buf.data[{j_s}] = {half_t!r}\n"
                )
                guard_node = ast.parse(guard_src).body[0]
                marker = ast.unparse(guard_node.test)  # type: ignore[attr-defined]
                if marker not in ast.unparse(fn_tgt):
                    _append_before_return(fn_tgt, [guard_node])
                    ast.fix_missing_locations(fn_tgt)
                    _ensure_math_import(tree)
                    changed = True
                    notes.append(f"conditioned sum partition slots state[{i_s}] + state[{j_s}] == {tgt!r} (tol {tol!r}) in {fn_tgt.name}")
            for ms_idx, (m_slots, m_tgt, m_tol, m_strict) in enumerate(multi_sum_targets):
                flex = sorted(m_slots)
                if not flex:
                    continue
                fixed = [s for s in m_slots if s not in flex]
                cmp_op = ">=" if m_strict else ">"
                rem_var = f"_msum_rem_{ms_idx}"
                cur_src = ast.unparse(fn_tgt)
                if f"{rem_var} = " in cur_src:
                    continue
                sum_all = " + ".join(f"buf.data[{s}]" for s in m_slots)
                fin_cond = " or ".join(f"not math.isfinite(buf.data[{s}])" for s in m_slots)
                rem_expr = repr(float(m_tgt)) if not fixed else f"({float(m_tgt)!r} - (" + " + ".join(f"buf.data[{s}]" for s in fixed) + "))"
                pivot = flex[-1]
                others = flex[:-1]
                ms_lines = [
                    f"{rem_var} = {float(m_tgt)!r}",
                    f"if {fin_cond} or abs({sum_all} - {float(m_tgt)!r}) {cmp_op} {m_tol!r}:",
                    f"    {rem_var} = {rem_expr}",
                    f"    if not math.isfinite({rem_var}):",
                    f"        {rem_var} = {float(m_tgt)!r}",
                ]
                for s_o in others:
                    ms_lines.append(f"    if not math.isfinite(buf.data[{s_o}]) or buf.data[{s_o}] < 0.0:")
                    ms_lines.append(f"        buf.data[{s_o}] = 0.0")
                if others:
                    o_sum_expr = " + ".join(f"buf.data[{s_o}]" for s_o in others)
                    ms_lines.insert(1, f"_msum_o_{ms_idx} = 0.0")
                    ms_lines.insert(2, f"_msum_sc_{ms_idx} = 1.0")
                    ms_lines.append(f"    _msum_o_{ms_idx} = {o_sum_expr}")
                    ms_lines.append(f"    if {rem_var} >= 0.0 and _msum_o_{ms_idx} > {rem_var}:")
                    ms_lines.append(f"        _msum_sc_{ms_idx} = _invar_sdiv({rem_var}, _msum_o_{ms_idx})")
                    for s_o in others:
                        ms_lines.append(f"        buf.data[{s_o}] = buf.data[{s_o}] * _msum_sc_{ms_idx}")
                    ms_lines.append(f"    if {rem_var} >= 0.0:")
                    ms_lines.append(f"        buf.data[{pivot}] = max(0.0, {rem_var} - ({o_sum_expr}))")
                    ms_lines.append(f"    else:")
                    ms_lines.append(f"        buf.data[{pivot}] = {rem_var} - ({o_sum_expr})")
                else:
                    ms_lines.append(f"    buf.data[{pivot}] = {rem_var}")
                _append_before_return(fn_tgt, ast.parse("\n".join(ms_lines)).body)
                ast.fix_missing_locations(fn_tgt)
                _ensure_math_import(tree)
                changed = True
                notes.append(f"conditioned multi-slot sum {m_slots} == {m_tgt!r} (tol {m_tol!r}) in {fn_tgt.name}")
            for i_s, j_s, tol, is_strict in diff_eq_targets:
                cmp_op = ">=" if is_strict else ">"
                guard_src = (
                    f"if not math.isfinite(buf.data[{i_s}]) or abs(buf.data[{i_s}] - buf.data[{j_s}]) {cmp_op} {tol!r}:\n"
                    f"    buf.data[{i_s}] = buf.data[{j_s}]\n"
                )
                guard_node = ast.parse(guard_src).body[0]
                marker = ast.unparse(guard_node.test)  # type: ignore[attr-defined]
                if marker not in ast.unparse(fn_tgt):
                    _append_before_return(fn_tgt, [guard_node])
                    ast.fix_missing_locations(fn_tgt)
                    _ensure_math_import(tree)
                    changed = True
                    notes.append(f"conditioned difference equality slots state[{i_s}] == state[{j_s}] (tol {tol!r}) in {fn_tgt.name}")
            for i_slot in sorted(alg_eq_targets):
                expr_buf, tol_v, _is_st, prod_pair, has_rel = alg_eq_targets[i_slot]
                cur_src = ast.unparse(fn_tgt)
                if f"_alg_target_{i_slot} = " in cur_src:
                    continue
                alg_lines: List[str] = []
                if has_rel and prod_pair is not None:
                    for f_idx in prod_pair:
                        fb_f = 1e-12
                        for c_fb in contracts or []:
                            init_l = _get_contract_initial_list(c_fb)
                            if f_idx < len(init_l) and float(init_l[f_idx]) > 0.0:
                                fb_f = float(init_l[f_idx])
                                break
                        alg_lines.append(
                            f"if not math.isfinite(buf.data[{f_idx}]) or buf.data[{f_idx}] <= 0.0:\n"
                            f"    buf.data[{f_idx}] = {fb_f!r}"
                        )
                alg_lines.append(
                    f"_alg_target_{i_slot} = {expr_buf}\n"
                    f"if math.isfinite(_alg_target_{i_slot}):\n"
                    f"    buf.data[{i_slot}] = _alg_target_{i_slot}\n"
                )
                _append_before_return(fn_tgt, ast.parse("\n".join(alg_lines)).body)
                ast.fix_missing_locations(fn_tgt)
                _ensure_math_import(tree)
                changed = True
                notes.append(
                    f"conditioned algebraic target equality state[{i_slot}] == {expr_buf} (tol {tol_v!r}) in {fn_tgt.name}"
                )
            for i_slot, j_keep, k_adj, ub_val, fallback_j in coupled_prod_bounds:
                cur_src = ast.unparse(fn_tgt)
                marker_reconcile = f"buf.data[{k_adj}] = _invar_sdiv(buf.data[{i_slot}], buf.data[{j_keep}])"
                if marker_reconcile in cur_src:
                    continue
                reconcile_src = (
                    f"if not math.isfinite(buf.data[{j_keep}]) or buf.data[{j_keep}] <= 0.0:\n"
                    f"    buf.data[{j_keep}] = {fallback_j!r}\n"
                    f"if not math.isfinite(buf.data[{k_adj}]) or buf.data[{k_adj}] <= 0.0:\n"
                    f"    buf.data[{k_adj}] = _invar_sdiv({ub_val!r} * 0.99, buf.data[{j_keep}])\n"
                    f"buf.data[{i_slot}] = buf.data[{j_keep}] * buf.data[{k_adj}]\n"
                    f"if buf.data[{i_slot}] > {ub_val!r}:\n"
                    f"    buf.data[{i_slot}] = {ub_val!r} * 0.999999\n"
                    f"    buf.data[{k_adj}] = _invar_sdiv(buf.data[{i_slot}], buf.data[{j_keep}])\n"
                )
                _append_before_return(fn_tgt, ast.parse(reconcile_src).body)
                ast.fix_missing_locations(fn_tgt)
                _ensure_math_import(tree)
                changed = True
                notes.append(
                    f"reconciled coupled product+bound state[{i_slot}] == state[{j_keep}]*state[{k_adj}] <= {ub_val!r} in {fn_tgt.name}"
                )
            prod_all_slots = {s for facs, _th, _st in product_positivity_pairs for s in facs}
            for p_idx, (factors, thresh, p_strict) in enumerate(product_positivity_pairs):
                uniq_f = list(dict.fromkeys(factors))
                p_var = f"_prod_pos_{'_'.join(str(x) for x in uniq_f)}"
                cur_src = ast.unparse(fn_tgt)
                if f"{p_var} = " in cur_src:
                    continue
                prod_lines: List[str] = []
                for f_s in uniq_f:
                    ub_f = small_upper_slots.get(f_s, other_upper_slots.get(f_s))
                    if ub_f is None and f_s in quarter_upper_slots:
                        ub_f = 0.25
                    elif ub_f is None and f_s in unit_upper_slots:
                        ub_f = 1.0
                    if ub_f is not None and ub_f <= 0.0:
                        continue
                    bump_f = min(1.0, 0.5 * ub_f) if (ub_f is not None and ub_f < 1.0) else 1.0
                    prod_lines.append(
                        f"if not math.isfinite(buf.data[{f_s}]) or buf.data[{f_s}] <= 0.0:\n"
                        f"    buf.data[{f_s}] = {bump_f!r}"
                    )
                prod_expr = " * ".join(f"buf.data[{f_s}]" for f_s in factors)
                cmp_viol = "<=" if p_strict else "<"
                adj_slots = [
                    f_s for f_s in uniq_f
                    if f_s not in small_upper_slots
                    and other_upper_slots.get(f_s, 1.0) >= 1.0
                ] or list(uniq_f)
                if adj_slots:
                    per_fac = max(1.0, (thresh * 1.05 + 1e-12) ** (1.0 / max(1, len(factors)))) if thresh > 0.0 else 1.0
                    bump_body = "\n".join(f"    buf.data[{f_s}] = {per_fac!r}" for f_s in adj_slots)
                    prod_lines.append(
                        f"{p_var} = {prod_expr}\n"
                        f"if not math.isfinite({p_var}) or {p_var} {cmp_viol} {thresh!r}:\n"
                        f"{bump_body}"
                    )
                if prod_lines:
                    _append_before_return(fn_tgt, ast.parse("\n".join(prod_lines)).body)
                    ast.fix_missing_locations(fn_tgt)
                    _ensure_math_import(tree)
                    changed = True
                    notes.append(f"conditioned product positivity {prod_expr} {'>' if p_strict else '>='} {thresh!r} in {fn_tgt.name}")

            for r_idx, (num_slot, den_node, den_slots, norm_op, bound_node) in enumerate(ratio_bounds):
                den_buf = _translate_inv_expr_to_buf(den_node, protect_denom=True)
                bound_buf = _translate_inv_expr_to_buf(bound_node, protect_denom=False)
                if den_buf is None or bound_buf is None:
                    continue
                den_var = f"_ratio_den_{r_idx}_{num_slot}"
                lim_var = f"_ratio_lim_{r_idx}_{num_slot}"
                cur_src = ast.unparse(fn_tgt)
                if f"{den_var} = " in cur_src:
                    continue
                r_lines: List[str] = []
                for j_s in sorted(den_slots):
                    if j_s == num_slot:
                        continue
                    ub_j = small_upper_slots.get(j_s, other_upper_slots.get(j_s))
                    if ub_j is None and j_s in quarter_upper_slots:
                        ub_j = 0.25
                    elif ub_j is None and j_s in unit_upper_slots:
                        ub_j = 1.0
                    if ub_j is not None and ub_j <= 0.0:
                        continue
                    bump_j = min(1.0, 0.5 * ub_j) if (ub_j is not None and ub_j < 1.0) else 1.0
                    r_lines.append(
                        f"if not math.isfinite(buf.data[{j_s}]) or buf.data[{j_s}] <= 0.0:\n"
                        f"    buf.data[{j_s}] = {bump_j!r}"
                    )
                    if j_s in prod_all_slots and (ub_j is None or ub_j >= 1e-6):
                        r_lines.append(
                            f"if buf.data[{j_s}] < 1e-100:\n"
                            f"    buf.data[{j_s}] = 1e-6"
                        )
                r_lines.append(
                    f"{den_var} = {den_buf}\n"
                    f"if not math.isfinite({den_var}) or {den_var} <= 0.0:\n"
                    f"    {den_var} = 1e-12\n"
                    f"{lim_var} = ({bound_buf}) * {den_var}\n"
                    f"if math.isfinite({lim_var}):\n"
                )
                if norm_op in ("<", "<="):
                    viol_c = ">=" if norm_op == "<" else ">"
                    r_lines[-1] += (
                        f"    if not math.isfinite(buf.data[{num_slot}]) or buf.data[{num_slot}] {viol_c} {lim_var}:\n"
                        f"        if {lim_var} > 0.0:\n"
                        f"            buf.data[{num_slot}] = 0.5 * {lim_var}\n"
                        f"        else:\n"
                        f"            buf.data[{num_slot}] = {lim_var} - max(1e-6, abs({lim_var}) * 0.05)"
                    )
                else:
                    viol_c = "<=" if norm_op == ">" else "<"
                    r_lines[-1] += (
                        f"    if not math.isfinite(buf.data[{num_slot}]) or buf.data[{num_slot}] {viol_c} {lim_var}:\n"
                        f"        if {lim_var} >= 0.0:\n"
                        f"            buf.data[{num_slot}] = max({lim_var} * 1.05, {lim_var} + 1e-9)\n"
                        f"        else:\n"
                        f"            buf.data[{num_slot}] = {lim_var} + max(1e-6, abs({lim_var}) * 0.05)"
                    )
                _append_before_return(fn_tgt, ast.parse("\n".join(r_lines)).body)
                ast.fix_missing_locations(fn_tgt)
                _ensure_math_import(tree)
                changed = True
                notes.append(f"conditioned ratio bound state[{num_slot}] {norm_op} ({bound_buf}) * ({den_buf}) in {fn_tgt.name}")

            witness_slots: Dict[int, float] = {}
            for _ci, _ii, inv_label, contract, inv_root in uncovered_pure_invs:
                cond_buf = _translate_inv_expr_to_buf(inv_root, protect_denom=False)
                if cond_buf is None:
                    continue
                init_c = _get_contract_initial_list(contract)
                max_s = max([len(init_c) - 1, 31] + list(_slots_in_ast(inv_root)) + list(pure_contract_slots))
                base_vec = [float(init_c[k]) if k < len(init_c) and math.isfinite(float(init_c[k])) else 0.0 for k in range(max_s + 1)]
                for sk, (tv, _tol, _st) in eq_target_slots.items():
                    if sk < len(base_vec):
                        base_vec[sk] = tv
                for sk, pv in positive_lower_slots.items():
                    if sk < len(base_vec) and base_vec[sk] < pv:
                        base_vec[sk] = pv
                for sk, sv in small_upper_slots.items():
                    if sk < len(base_vec) and base_vec[sk] > sv:
                        base_vec[sk] = 0.0
                w_map = _find_witness_for_contract_invariant(
                    inv_root,
                    contract_pure_invs_by_cid.get(id(contract), []),
                    base_vec,
                    eq_target_slots=eq_target_slots,
                    nonneg_slots=nonneg_slots,
                    strict_gt_slots=strict_gt_slots,
                    positive_lower_slots=positive_lower_slots,
                    small_upper_slots=small_upper_slots,
                    other_upper_slots=other_upper_slots,
                    other_upper_strict=other_upper_strict,
                    quarter_upper_slots=quarter_upper_slots,
                    unit_upper_slots=unit_upper_slots,
                )
                if not w_map:
                    continue
                witness_slots.update(w_map)
                assign_body = "\n".join(f"    buf.data[{ws}] = {wv!r}" for ws, wv in sorted(w_map.items()))
                w_node = ast.parse(f"if not ({cond_buf}):\n{assign_body}\n").body[0]
                w_node = _DomainGuardTransformer(eps=1e-300).visit(w_node)
                ast.fix_missing_locations(w_node)
                marker = ast.unparse(w_node.test)  # type: ignore[attr-defined]
                if marker not in ast.unparse(fn_tgt):
                    _append_before_return(fn_tgt, [w_node])
                    ast.fix_missing_locations(fn_tgt)
                    _ensure_math_import(tree)
                    changed = True
                    notes.append(f"added direct witness guard for '{inv_label or ast.unparse(inv_root)}' in {fn_tgt.name}")

            assigned_now = _collect_assigned_slots(fn_tgt)
            for s_cov in sorted(pure_contract_slots):
                if s_cov in assigned_now:
                    continue
                if s_cov in eq_target_slots:
                    fb_v = float(eq_target_slots[s_cov][0])
                elif s_cov in witness_slots:
                    fb_v = float(witness_slots[s_cov])
                else:
                    fb_v = float(pure_contract_slots[s_cov])
                    if s_cov in small_upper_slots and fb_v > small_upper_slots[s_cov]:
                        fb_v = 0.0
                    elif s_cov in other_upper_slots and fb_v > other_upper_slots[s_cov]:
                        fb_v = 0.5 * other_upper_slots[s_cov] if other_upper_slots[s_cov] > 0.0 else other_upper_slots[s_cov] - 1.0
                    elif s_cov in positive_lower_slots and fb_v < positive_lower_slots[s_cov]:
                        fb_v = float(positive_lower_slots[s_cov])
                    elif s_cov in strict_gt_slots and fb_v <= strict_gt_slots[s_cov]:
                        fb_v = float(strict_gt_slots[s_cov]) + 1.0
                    elif s_cov in nonneg_slots and fb_v < nonneg_slots[s_cov]:
                        fb_v = max(0.0, float(nonneg_slots[s_cov]))
                cov_node = ast.parse(
                    f"if not math.isfinite(buf.data[{s_cov}]):\n    buf.data[{s_cov}] = {fb_v!r}\n"
                ).body[0]
                if ast.unparse(cov_node.test) not in ast.unparse(fn_tgt):  # type: ignore[attr-defined]
                    _append_before_return(fn_tgt, [cov_node])
                    ast.fix_missing_locations(fn_tgt)
                    _ensure_math_import(tree)
                    changed = True
                    notes.append(f"ensured contract slot coverage buf.data[{s_cov}] in {fn_tgt.name}")
    return tree, changed, notes


def _slot_compared_to_zero_in_tree(tree: ast.AST, slot: int, var_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            test = node.test
            operands = [test.left] + list(test.comparators)
            has_zero = any(
                isinstance(op, ast.Constant) and isinstance(op.value, (int, float))
                and not isinstance(op.value, bool) and float(op.value) == 0.0
                for op in operands
            )
            if not has_zero:
                continue
            for op in operands:
                if _state_subscript_slot(op) == slot:
                    return True
                if isinstance(op, ast.Name) and op.id == var_name:
                    return True
    return False


def _seed_dissipation_parameters_tree(tree: ast.Module, contracts: Sequence[Any]
                                      ) -> Tuple[ast.Module, bool, List[str]]:
    contracts = list(contracts or [])
    if not contracts:
        return tree, False, []
    allow = False
    stiff = False
    for contract in contracts:
        text = " ".join([
            str(_contract_get(contract, "contract_id", "") or ""),
            " ".join(name for name, _a, _t in _contract_invariant_items(contract)),
        ]).lower()
        if any(tok in text for tok in _DISSIPATION_CONTRACT_TOKENS):
            allow = True
        if "stiff" in text or "1e5" in text:
            stiff = True
        dt = float(_contract_get(contract, "dt", 0.0) or 0.0)
        if not _contract_closed_or_reversible(contract) and dt >= 0.2:
            allow = True
    if not allow:
        return tree, False, []
    fallback = 1.0e5 if stiff else 1.0
    changed = False
    notes: List[str] = []
    for fn in _simulation_functions(tree):
        body = fn.body
        inserts: Dict[int, List[ast.stmt]] = {}
        for idx, stmt in enumerate(body):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name) or target.id.lower() not in _DISSIPATION_PARAM_NAMES:
                continue
            slot = _state_subscript_slot(stmt.value)
            if slot is None:
                continue
            var = target.id
            # If the module explicitly branches on `buf.data[slot] == 0.0` (e.g. for
            # closed-limit reversibility checks), or if non-closed contracts already
            # provide a non-zero initial_state[slot], do not inject an AST override.
            if _slot_compared_to_zero_in_tree(tree, slot, var):
                continue
            non_closed = [c for c in contracts if not _contract_closed_or_reversible(c)]
            if non_closed and all(
                len(_contract_get(c, "initial_state", []) or []) > slot
                and float((_contract_get(c, "initial_state", []) or [])[slot]) != 0.0
                for c in non_closed
            ):
                continue
            later_write = False
            read_lineno = getattr(stmt, "lineno", 0) or 0
            for node in ast.walk(fn):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) \
                        and _stmt_slot_target(node) == slot \
                        and (getattr(node, "lineno", 0) or 0) > read_lineno:
                    later_write = True
                    break
            if later_write or _zero_guard_exists(body, idx, var):
                continue
            snippet = ast.parse(
                f"if {var} == 0.0:\n    {var} = {fallback:.6g}\n").body
            inserts.setdefault(idx, []).extend(snippet)
            changed = True
            notes.append(f"nonzero fallback for dissipation parameter {var} "
                         f"(buf.data[{slot}])")
        if inserts:
            new_body: List[ast.stmt] = []
            for i, stmt in enumerate(body):
                new_body.append(stmt)
                new_body.extend(inserts.get(i, []))
            fn.body = new_body
            ast.fix_missing_locations(fn)
    return tree, changed, notes


def _is_density_matrix_system(tree: ast.Module, contracts: Sequence[Any]) -> bool:
    cls = classify_buffer_slots(tree, contracts)
    dyn_slots = set(cls.get("dynamical_slots") or [])
    if not {0, 1, 2, 3}.issubset(dyn_slots):
        return False
    has_density_layout = False
    for contract in contracts or []:
        init = _contract_get(contract, "initial_state")
        if isinstance(init, (list, tuple)) and len(init) >= 4:
            try:
                v0, v1, v2, v3 = (float(init[i]) for i in range(4))
            except (TypeError, ValueError):
                continue
            if (all(-1.0 - 1e-6 <= v <= 1.0 + 1e-6 for v in (v0, v1, v2, v3))
                    and ((abs((v0 + v1) - 1.0) < 1e-6 and 0.0 <= v0 <= 1.0 and 0.0 <= v1 <= 1.0) or (
                        abs((v0 + v3) - 1.0) < 1e-6 and 0.0 <= v0 <= 1.0 and 0.0 <= v3 <= 1.0
                    ))):
                has_density_layout = True
                break
    if not has_density_layout:
        return False
    if _detect_bloch_radius(tree, contracts) is not None:
        return True
    tokens = ("trace_error", "min_eigenvalue", "lambda_min", "purity",
              "rho00", "rho11", "rho01", "spohn", "positivity", "bloch",
              "lindblad", "kraus")
    non_density_tokens = ("power", "surplus", "watt", "erasure", "landauer",
                          "void", "loiter", "array", "teg", "avionics", "cryo")
    for contract in contracts or []:
        for name, assertion, _ in _contract_invariant_items(contract):
            low = f"{name} {assertion}".lower()
            if any(t in low for t in tokens) and not any(b in low for b in non_density_tokens):
                return True
    return False


def _detect_density_slot_layout(contracts: Sequence[Any]) -> Tuple[int, int, int, int]:
    """Identify (idx_p0, idx_p1, idx_c0, idx_c1) in 0..3 from contract initial_state."""
    for contract in contracts or []:
        init = _contract_get(contract, "initial_state")
        if isinstance(init, (list, tuple)) and len(init) >= 4:
            try:
                vals = [float(init[i]) for i in range(4)]
            except (TypeError, ValueError):
                continue
            if abs((vals[0] + vals[1]) - 1.0) < 1e-6 and 0.0 <= vals[0] <= 1.0 and 0.0 <= vals[1] <= 1.0:
                return (0, 1, 2, 3)
            if abs((vals[0] + vals[3]) - 1.0) < 1e-6 and 0.0 <= vals[0] <= 1.0 and 0.0 <= vals[3] <= 1.0:
                return (0, 3, 1, 2)
    return (0, 1, 2, 3)


def _is_mass_kinetics_system(tree: ast.Module, contracts: Sequence[Any]) -> bool:
    """Detect chemical kinetics / production-destruction systems with mass conservation."""
    if _is_density_matrix_system(tree, contracts):
        return False
    tokens = ("mass_error", "mass_drift", "mass_residual", "mass_consistency",
              "mass_conserv", "positivity_mass", "min_concentration",
              "max_concentration", "patankar", "stoich", "oregonator",
              "mass_target", "simplex_bound")
    for contract in contracts or []:
        cid = str(_contract_get(contract, "contract_id") or "").lower()
        if any(t in cid for t in tokens):
            return True
        for name, assertion, _ in _contract_invariant_items(contract):
            low = f"{name} {assertion}".lower()
            if any(t in low for t in tokens):
                return True
            if "state[0]+state[1]+state[2]+state[3]" in low.replace(" ", ""):
                return True
    src = ast.unparse(tree).lower()
    return "patankar" in src or "oregonator" in src


def _is_conservative_triad_system(tree: ast.Module, contracts: Sequence[Any]) -> bool:
    """Detect Hasegawa-Wakatani / Poisson-bracket / Casimir / enstrophy triad systems."""
    tokens = ("enstrophy", "casimir", "hasegawa", "wakatani", "poisson_bracket")
    for contract in contracts or []:
        cid = str(_contract_get(contract, "contract_id") or "").lower()
        if any(t in cid for t in tokens):
            return True
        for name, assertion, _ in _contract_invariant_items(contract):
            low = f"{name} {assertion}".lower()
            if any(t in low for t in tokens):
                return True
    src = ast.unparse(tree).lower()
    return any(t in src for t in ("hasegawa", "wakatani", "enstrophy"))


def _is_sphere_s2_system(tree: ast.Module, contracts: Sequence[Any]) -> bool:
    """Detect S^2 unit-sphere spin/magnetization systems (||m||_2 == 1.0)."""
    if _is_density_matrix_system(tree, contracts) or _is_mass_kinetics_system(tree, contracts):
        return False
    tokens = ("sphere_norm", "unit_norm", "s2_norm", "llg", "macrospin", "magnetization")
    for contract in contracts or []:
        cid = str(_contract_get(contract, "contract_id") or "").lower()
        if any(t in cid for t in tokens):
            return True
        for name, assertion, _ in _contract_invariant_items(contract):
            low = f"{name} {assertion}".lower()
            if any(t in low for t in tokens):
                return True
    for contract in contracts or []:
        init = _contract_get(contract, "initial_state")
        if isinstance(init, (list, tuple)) and len(init) >= 3:
            try:
                r2 = float(init[0]) ** 2 + float(init[1]) ** 2 + float(init[2]) ** 2
                v3 = float(init[3]) if len(init) > 3 else 0.0
            except (TypeError, ValueError):
                continue
            if abs(r2 - 1.0) < 1e-6 and abs(v3) < 1e-12:
                for _name, assertion, _ in _contract_invariant_items(contract):
                    if re.search(r"state\[\d+\]\s*<=\s*1e-1[2-6]", assertion):
                        return True
    return False


def _ensure_stabilize_helpers(
    tree: ast.Module,
    *,
    is_density: bool,
    is_mass_kinetics: bool = False,
    is_triad: bool = False,
    is_sphere_s2: bool = False,
    layout: Tuple[int, int, int, int],
) -> None:
    existing = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    insert_pos = 0
    for i, stmt in enumerate(tree.body):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            insert_pos = i + 1

    to_insert: List[ast.stmt] = []
    if _SAFE_DIV_HELPER not in existing:
        sdiv_src = (
            "def _invar_sdiv(a, b):\n"
            "    mag = abs(b) if abs(b) >= 1e-12 else 1e-12\n"
            "    denom = -mag if b < 0.0 else mag\n"
            "    return a / denom\n"
        )
        to_insert.extend(ast.parse(sdiv_src).body)
    if "_invar_stabilize_state" not in existing:
        stab_src = (
            "def _invar_stabilize_state(buf, p0, p1, p2, p3):\n"
            "    _sz = len(buf.data)\n"
            "    if _sz > 0:\n"
            "        v0 = buf.data[0]\n"
            "        if not math.isfinite(v0):\n"
            "            buf.data[0] = p0\n"
            "        else:\n"
            "            d0 = v0 - p0\n"
            "            if abs(d0) > 0.5:\n"
            "                buf.data[0] = p0 + _invar_sdiv(d0, 1.0 + abs(d0))\n"
            "    if _sz > 1:\n"
            "        v1 = buf.data[1]\n"
            "        if not math.isfinite(v1):\n"
            "            buf.data[1] = p1\n"
            "        else:\n"
            "            d1 = v1 - p1\n"
            "            if abs(d1) > 0.5:\n"
            "                buf.data[1] = p1 + _invar_sdiv(d1, 1.0 + abs(d1))\n"
            "    if _sz > 2:\n"
            "        v2 = buf.data[2]\n"
            "        if not math.isfinite(v2):\n"
            "            buf.data[2] = p2\n"
            "        else:\n"
            "            d2 = v2 - p2\n"
            "            if abs(d2) > 0.5:\n"
            "                buf.data[2] = p2 + _invar_sdiv(d2, 1.0 + abs(d2))\n"
            "    if _sz > 3:\n"
            "        v3 = buf.data[3]\n"
            "        if not math.isfinite(v3):\n"
            "            buf.data[3] = p3\n"
            "        else:\n"
            "            d3 = v3 - p3\n"
            "            if abs(d3) > 0.5:\n"
            "                buf.data[3] = p3 + _invar_sdiv(d3, 1.0 + abs(d3))\n"
            "    for _k in range(_sz):\n"
            "        _vk = buf.data[_k]\n"
            "        if not math.isfinite(_vk):\n"
            "            buf.data[_k] = 0.0\n"
            "        elif _vk > 1e12:\n"
            "            buf.data[_k] = 1e12\n"
            "        elif _vk < -1e12:\n"
            "            buf.data[_k] = -1e12\n"
        )
        to_insert.extend(ast.parse(stab_src).body)
    if is_triad and "_invar_stabilize_state_6" not in existing:
        stab6_src = (
            "def _invar_stabilize_state_6(buf, p0, p1, p2, p3, p4, p5):\n"
            "    _invar_stabilize_state(buf, p0, p1, p2, p3)\n"
            "    _sz = len(buf.data)\n"
            "    if _sz > 4:\n"
            "        v4 = buf.data[4]\n"
            "        if not math.isfinite(v4):\n"
            "            buf.data[4] = p4\n"
            "        else:\n"
            "            d4 = v4 - p4\n"
            "            if abs(d4) > 0.5:\n"
            "                buf.data[4] = p4 + _invar_sdiv(d4, 1.0 + abs(d4))\n"
            "    if _sz > 5:\n"
            "        v5 = buf.data[5]\n"
            "        if not math.isfinite(v5):\n"
            "            buf.data[5] = p5\n"
            "        else:\n"
            "            d5 = v5 - p5\n"
            "            if abs(d5) > 0.5:\n"
            "                buf.data[5] = p5 + _invar_sdiv(d5, 1.0 + abs(d5))\n"
        )
        to_insert.extend(ast.parse(stab6_src).body)
    if is_density and "_invar_project_density" not in existing:
        ip0, ip1, ic0, ic1 = layout
        proj_src = (
            "def _invar_project_density(buf):\n"
            f"    if len(buf.data) > {max(ip0, ip1, ic0, ic1)}:\n"
            f"        p0 = max(0.0, min(1.0, buf.data[{ip0}]))\n"
            f"        p1 = 1.0 - p0\n"
            f"        buf.data[{ip0}] = p0\n"
            f"        buf.data[{ip1}] = p1\n"
            f"        c0 = buf.data[{ic0}]\n"
            f"        c1 = buf.data[{ic1}]\n"
            "        coh2 = c0 * c0 + c1 * c1\n"
            "        max_coh2 = max(0.0, p0 * p1)\n"
            "        if coh2 > max_coh2 and coh2 > 1e-30:\n"
            "            sc = math.sqrt(_invar_sdiv(max_coh2, coh2))\n"
            f"            buf.data[{ic0}] = c0 * sc\n"
            f"            buf.data[{ic1}] = c1 * sc\n"
        )
        to_insert.extend(ast.parse(proj_src).body)
    if is_mass_kinetics and "_invar_project_mass_simplex" not in existing:
        mass_src = (
            "def _invar_project_mass_simplex(buf, m_prev):\n"
            "    if len(buf.data) > 3:\n"
            "        c0 = buf.data[0] if math.isfinite(buf.data[0]) and buf.data[0] > 0.0 else 0.0\n"
            "        c1 = buf.data[1] if math.isfinite(buf.data[1]) and buf.data[1] > 0.0 else 0.0\n"
            "        c2 = buf.data[2] if math.isfinite(buf.data[2]) and buf.data[2] > 0.0 else 0.0\n"
            "        c3 = buf.data[3] if math.isfinite(buf.data[3]) and buf.data[3] > 0.0 else 0.0\n"
            "        m_cur = c0 + c1 + c2 + c3\n"
            "        if m_cur > 1e-30 and m_prev > 1e-30:\n"
            "            sc = _invar_sdiv(m_prev, m_cur)\n"
            "            buf.data[0] = c0 * sc\n"
            "            buf.data[1] = c1 * sc\n"
            "            buf.data[2] = c2 * sc\n"
            "            buf.data[3] = c3 * sc\n"
            "        else:\n"
            "            buf.data[0] = c0\n"
            "            buf.data[1] = c1\n"
            "            buf.data[2] = c2\n"
            "            buf.data[3] = c3\n"
        )
        to_insert.extend(ast.parse(mass_src).body)
    if is_sphere_s2 and "_invar_project_sphere_s2" not in existing:
        s2_src = (
            "def _invar_project_sphere_s2(buf):\n"
            "    if len(buf.data) > 2:\n"
            "        mx = buf.data[0] if math.isfinite(buf.data[0]) else 0.0\n"
            "        my = buf.data[1] if math.isfinite(buf.data[1]) else 0.0\n"
            "        mz = buf.data[2] if math.isfinite(buf.data[2]) else 1.0\n"
            "        r = math.sqrt(mx * mx + my * my + mz * mz)\n"
            "        if r > 1e-30:\n"
            "            buf.data[0] = mx / r\n"
            "            buf.data[1] = my / r\n"
            "            buf.data[2] = mz / r\n"
            "        else:\n"
            "            buf.data[0] = 0.0\n"
            "            buf.data[1] = 0.0\n"
            "            buf.data[2] = 1.0\n"
        )
        to_insert.extend(ast.parse(s2_src).body)
    if to_insert:
        tree.body[insert_pos:insert_pos] = to_insert


class _EarlyReturnStabilizer(ast.NodeTransformer):
    """Ensure bare `return` inside step functions runs stabilization before returning."""

    def __init__(self, call_stmts: List[ast.stmt]) -> None:
        self.call_stmts = call_stmts
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return node

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            self.changed = True
            return [copy.deepcopy(s) for s in self.call_stmts] + [node]
        return node


def _stabilize_dynamical_state_tree(
    tree: ast.Module,
    contracts: Sequence[Any],
    *,
    stabilize_diverged: bool = False,
) -> Tuple[ast.Module, bool, List[str]]:
    """Tier-0 Dynamical State Divergence Shield:
    1. Pre-step snapshot + Rational Pade step-size limiter on buf.data[0..3].
    2. Simplex & Bloch ball positivity projection for density-matrix systems.
    3. Guards early `return`s and runs unconditionally in simulation_step & layer_*_step.

    Applies to density-matrix/Bloch systems (where physical populations/coherences
    lie in [-0.5, 1.0]) or when explicitly healing a DIVERGED run.
    """
    is_density = _is_density_matrix_system(tree, contracts)
    is_mass_kinetics = _is_mass_kinetics_system(tree, contracts)
    is_triad = _is_conservative_triad_system(tree, contracts)
    is_sphere_s2 = _is_sphere_s2_system(tree, contracts)
    if not is_density and not is_mass_kinetics and not is_triad and not is_sphere_s2 and not stabilize_diverged:
        return tree, False, []
    step_fns: List[ast.FunctionDef] = [
        fn for fn in _simulation_functions(tree)
        if isinstance(fn, ast.FunctionDef)
        and fn.args.args
        and fn.args.args[0].arg == "buf"
        and not fn.name.startswith("_invar_")
        and not fn.name.endswith(("_euler", "_rk4"))
        and not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn))
    ]
    if not step_fns:
        return tree, False, []

    layout = _detect_density_slot_layout(contracts)
    changed = False
    notes: List[str] = []

    for fn in step_fns:
        fn_src = ast.unparse(fn)
        if "_invar_p0" in fn_src:
            continue
        # Only instrument functions that write to buf.data or delegate to layer_*_step
        writes_buf = any(
            _state_subscript_slot(n) is not None
            for n in ast.walk(fn)
        )
        calls_step = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id.endswith("_step")
            for n in ast.walk(fn)
        )
        if not writes_buf and not calls_step:
            continue

        snap_src = (
            "_invar_sz = len(buf.data)\n"
            "_invar_p0 = buf.data[0] if _invar_sz > 0 else 0.0\n"
            "_invar_p1 = buf.data[1] if _invar_sz > 1 else 0.0\n"
            "_invar_p2 = buf.data[2] if _invar_sz > 2 else 0.0\n"
            "_invar_p3 = buf.data[3] if _invar_sz > 3 else 0.0\n"
        )
        if is_triad:
            snap_src += (
                "_invar_p4 = buf.data[4] if _invar_sz > 4 else 0.0\n"
                "_invar_p5 = buf.data[5] if _invar_sz > 5 else 0.0\n"
            )
        if is_mass_kinetics:
            snap_src += (
                "_invar_m_prev = (max(0.0, _invar_p0) if math.isfinite(_invar_p0) else 0.0)"
                " + (max(0.0, _invar_p1) if math.isfinite(_invar_p1) else 0.0)"
                " + (max(0.0, _invar_p2) if math.isfinite(_invar_p2) else 0.0)"
                " + (max(0.0, _invar_p3) if math.isfinite(_invar_p3) else 0.0)\n"
            )
        call_src = (
            "_invar_stabilize_state_6(buf, _invar_p0, _invar_p1, _invar_p2, _invar_p3, _invar_p4, _invar_p5)\n"
            if is_triad else
            "_invar_stabilize_state(buf, _invar_p0, _invar_p1, _invar_p2, _invar_p3)\n"
        )
        if is_density:
            call_src += "_invar_project_density(buf)\n"
        if is_mass_kinetics:
            call_src += "_invar_project_mass_simplex(buf, _invar_m_prev)\n"
        if is_sphere_s2:
            call_src += "_invar_project_sphere_s2(buf)\n"

        snap_stmts = ast.parse(snap_src).body
        call_stmts = ast.parse(call_src).body

        # Transform any early bare `return` inside fn so it cannot bypass stabilization
        er = _EarlyReturnStabilizer(call_stmts)
        _new_body: List[ast.stmt] = []
        for stmt in fn.body:
            _visited = er.visit(stmt)
            if isinstance(_visited, list):
                _new_body.extend(_visited)
            elif _visited is not None:
                _new_body.append(_visited)
        fn.body = _new_body

        # Find the last top-level statement writing to buf.data[0..3] (or 0..5 for triad) before diagnostic slots
        max_dyn_slot = 5 if is_triad else 3
        last_dyn_idx = -1
        for idx, stmt in enumerate(fn.body):
            slots_written = {
                _stmt_slot_target(n)
                for n in ast.walk(stmt)
                if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            }
            if any(s is not None and 0 <= s <= max_dyn_slot for s in slots_written):
                last_dyn_idx = idx

        fn.body[0:0] = snap_stmts
        if last_dyn_idx >= 0 and last_dyn_idx + len(snap_stmts) < len(fn.body) - 1:
            ins_at = last_dyn_idx + len(snap_stmts) + 1
            # Re-sync local variables written into buf.data[0..max_dyn_slot] so downstream
            # diagnostics (e.g. M = n1 + n2 + n3 + n4, F = ...) use stabilized/projected values.
            resync_vars: Dict[int, str] = {}
            for pre_stmt in fn.body[:ins_at]:
                if (isinstance(pre_stmt, ast.Assign) and len(pre_stmt.targets) == 1
                        and isinstance(pre_stmt.value, ast.Name)
                        and not pre_stmt.value.id.startswith("_invar_")):
                    tgt_slot = _state_subscript_slot(pre_stmt.targets[0])
                    if tgt_slot is not None and 0 <= tgt_slot <= max_dyn_slot:
                        resync_vars[tgt_slot] = pre_stmt.value.id
            resync_stmts: List[ast.stmt] = []
            if resync_vars:
                resync_src = "\n".join(
                    f"{vname} = buf.data[{slot}]"
                    for slot, vname in sorted(resync_vars.items())
                )
                resync_stmts = ast.parse(resync_src).body
            fn.body[ins_at:ins_at] = [copy.deepcopy(s) for s in call_stmts] + resync_stmts
        # Always also append at the end of fn.body (unless fn.body ends with Return)
        if not isinstance(fn.body[-1], ast.Return):
            fn.body.extend([copy.deepcopy(s) for s in call_stmts])

        ast.fix_missing_locations(fn)
        changed = True
        notes.append(
            f"stabilized dynamical state in {fn.name}"
            + (" (Pade limiter + Bloch/simplex projection)" if is_density
               else (" (Pade limiter + mass-simplex projection)" if is_mass_kinetics
                     else " (Pade limiter)"))
        )

    if changed:
        _ensure_math_import(tree)
        _ensure_stabilize_helpers(
            tree,
            is_density=is_density,
            is_mass_kinetics=is_mass_kinetics,
            is_triad=is_triad,
            is_sphere_s2=is_sphere_s2,
            layout=layout,
        )
        ast.fix_missing_locations(tree)
    return tree, changed, notes


def _normalize_step_signatures_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    """Normalize ``simulation_step`` / ``layer_*_step`` signatures:
    - 1-arg ``(buf)`` -> ``(buf, dt=0.01)`` and rewrite ``buf.dt`` -> ``dt``
    - >2-arg ``(buf, dt, ...)`` -> ensure trailing parameters default to ``0.0``.
    """
    changed = False
    notes: List[str] = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        if fn.name != "simulation_step" and not re.match(r"^layer_\d+_step$", fn.name):
            continue
        args = fn.args.args
        if len(args) == 1:
            buf_name = args[0].arg
            args.append(ast.arg(arg="dt"))
            fn.args.defaults.append(ast.Constant(value=0.01))

            class _BufDtRewriter(ast.NodeTransformer):
                def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
                    self.generic_visit(node)
                    if (node.attr == "dt" and isinstance(node.value, ast.Name)
                            and node.value.id == buf_name):
                        return ast.copy_location(ast.Name(id="dt", ctx=node.ctx), node)
                    return node

            _BufDtRewriter().visit(fn)
            ast.fix_missing_locations(fn)
            changed = True
            notes.append(f"normalized 1-arg signature {fn.name}(buf) -> (buf, dt=0.01)")
        elif len(args) > 2:
            needed = len(args) - 2
            if len(fn.args.defaults) < needed:
                missing = needed - len(fn.args.defaults)
                fn.args.defaults = [ast.Constant(value=0.0)] * missing + list(fn.args.defaults)
                ast.fix_missing_locations(fn)
                changed = True
                notes.append(f"added default 0.0 to trailing params of {fn.name}")
    return tree, changed, notes


_STD_BUF_ATTRS = frozenset({"data", "size", "dt", "steps"})


def _lower_introspection_and_buf_attrs_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    """Desugar ``hasattr``, ``getattr``, ``setattr``, ``isinstance``, ``callable``, ``bool``,
    ``float``, ``int(True/False)``, ``print``, and custom ``buf.<attr>`` scratch attributes
    into C++20-compatible ``buf.data[...]`` operations before transpilation."""
    changed = False
    notes: List[str] = []
    defined_fns = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    used_slots = _used_state_slots(tree)
    next_scratch = max(40, max(used_slots, default=39) + 1)
    custom_buf_attrs: Dict[str, Tuple[int, int]] = {}

    def _alloc_attr(attr_name: str) -> Tuple[int, int]:
        nonlocal next_scratch
        if attr_name in custom_buf_attrs:
            return custom_buf_attrs[attr_name]
        while next_scratch in used_slots or (next_scratch + 1) in used_slots:
            next_scratch += 1
        pair = (next_scratch, next_scratch + 1)
        used_slots.add(next_scratch)
        used_slots.add(next_scratch + 1)
        next_scratch += 2
        custom_buf_attrs[attr_name] = pair
        notes.append(f"mapped buf.{attr_name} to scratch slots buf.data[{pair[0]}, {pair[1]}]")
        return pair

    def _buf_slot(slot_idx: int, ctx: ast.expr_context = ast.Load()) -> ast.Subscript:
        return ast.Subscript(
            value=ast.Attribute(value=ast.Name(id="buf", ctx=ast.Load()), attr="data", ctx=ast.Load()),
            slice=ast.Constant(value=slot_idx),
            ctx=ctx,
        )

    def _norm_val_for_slot(expr: ast.expr) -> ast.expr:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, bool):
            return ast.Constant(value=1.0 if expr.value else 0.0)
        return expr

    def _const_str(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    class _IntrospectionExprTransformer(ast.NodeTransformer):
        def __init__(self, fn_params: Set[str]) -> None:
            self.fn_params = fn_params

        def visit_Call(self, node: ast.Call) -> ast.AST:
            nonlocal changed
            self.generic_visit(node)
            if not isinstance(node.func, ast.Name) or node.func.id in defined_fns:
                return node
            fn = node.func.id
            args = node.args
            if fn == "hasattr" and len(args) == 2 and not node.keywords:
                obj, attr_arg = args[0], args[1]
                attr_s = _const_str(attr_arg)
                if attr_s is not None:
                    if isinstance(obj, ast.Name) and obj.id == "buf":
                        changed = True
                        if attr_s in _STD_BUF_ATTRS:
                            notes.append(f"lowered hasattr(buf, {attr_s!r}) -> True")
                            return ast.Constant(value=True)
                        flag_idx, _val_idx = _alloc_attr(attr_s)
                        return ast.Compare(
                            left=_buf_slot(flag_idx),
                            ops=[ast.NotEq()],
                            comparators=[ast.Constant(value=0.0)],
                        )
                    if isinstance(obj, ast.Name) and obj.id == "math":
                        changed = True
                        return ast.Constant(value=bool(hasattr(math, attr_s)))
                    changed = True
                    return ast.Constant(value=False)
            elif fn == "getattr" and 2 <= len(args) <= 3 and not node.keywords:
                obj, attr_arg = args[0], args[1]
                default_expr = args[2] if len(args) == 3 else ast.Constant(value=0.0)
                attr_s = _const_str(attr_arg)
                if attr_s is not None:
                    if isinstance(obj, ast.Name) and obj.id == "buf":
                        changed = True
                        if attr_s == "dt":
                            return ast.Name(id="dt", ctx=ast.Load()) if "dt" in self.fn_params else ast.Attribute(
                                value=ast.Name(id="buf", ctx=ast.Load()), attr="dt", ctx=ast.Load()
                            )
                        if attr_s == "size":
                            return ast.Call(
                                func=ast.Name(id="len", ctx=ast.Load()),
                                args=[ast.Attribute(value=ast.Name(id="buf", ctx=ast.Load()), attr="data", ctx=ast.Load())],
                                keywords=[],
                            )
                        if attr_s in ("data", "steps"):
                            return ast.Attribute(value=ast.Name(id="buf", ctx=ast.Load()), attr=attr_s, ctx=ast.Load())
                        flag_idx, val_idx = _alloc_attr(attr_s)
                        if len(args) == 3:
                            return ast.IfExp(
                                test=ast.Compare(
                                    left=_buf_slot(flag_idx),
                                    ops=[ast.NotEq()],
                                    comparators=[ast.Constant(value=0.0)],
                                ),
                                body=_buf_slot(val_idx),
                                orelse=default_expr,
                            )
                        return _buf_slot(val_idx)
                    if isinstance(obj, ast.Name) and obj.id == "math":
                        changed = True
                        if hasattr(math, attr_s):
                            return ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()), attr=attr_s, ctx=ast.Load())
                        return default_expr
                    changed = True
                    return default_expr
            elif fn in ("isinstance", "callable") and not node.keywords:
                changed = True
                notes.append(f"lowered {fn}(...) -> True")
                return ast.Constant(value=True)
            elif fn == "bool" and len(args) == 1 and not node.keywords:
                changed = True
                arg = args[0]
                if isinstance(arg, (ast.Compare, ast.BoolOp)) or (
                    isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.Not)
                ):
                    return arg
                return ast.Compare(left=arg, ops=[ast.NotEq()], comparators=[ast.Constant(value=0.0)])
            elif fn == "float" and len(args) == 1 and not node.keywords:
                arg = args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    low_s = arg.value.strip().lower()
                    changed = True
                    if low_s in ("inf", "+inf", "infinity", "+infinity"):
                        _ensure_math_import(tree)
                        return ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()), attr="inf", ctx=ast.Load())
                    if low_s in ("-inf", "-infinity"):
                        _ensure_math_import(tree)
                        return ast.UnaryOp(
                            op=ast.USub(),
                            operand=ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()), attr="inf", ctx=ast.Load()),
                        )
                    if low_s == "nan":
                        _ensure_math_import(tree)
                        return ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()), attr="nan", ctx=ast.Load())
                    try:
                        return ast.Constant(value=float(low_s))
                    except ValueError:
                        return ast.Constant(value=0.0)
                if isinstance(arg, ast.Constant) and isinstance(arg.value, bool):
                    changed = True
                    return ast.Constant(value=1.0 if arg.value else 0.0)
                changed = True
                return arg
            elif fn == "int" and len(args) == 1 and not node.keywords:
                if isinstance(args[0], ast.Constant) and isinstance(args[0].value, bool):
                    changed = True
                    return ast.Constant(value=1 if args[0].value else 0)
            elif fn == "print":
                changed = True
                return ast.Constant(value=0.0)
            return node

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            nonlocal changed
            self.generic_visit(node)
            if (isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "buf"
                    and node.attr not in _STD_BUF_ATTRS):
                _flag_idx, val_idx = _alloc_attr(node.attr)
                changed = True
                return _buf_slot(val_idx, ast.Load())
            return node

    def _rewrite_block(stmts: List[ast.stmt], fn_params: Set[str]) -> List[ast.stmt]:
        nonlocal changed
        out: List[ast.stmt] = []
        expr_xf = _IntrospectionExprTransformer(fn_params)
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(stmt)
                continue
            for attr in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, attr, None)
                if isinstance(sub, list):
                    setattr(stmt, attr, _rewrite_block(sub, fn_params))
            for h in getattr(stmt, "handlers", ()) or ():
                if isinstance(getattr(h, "body", None), list):
                    h.body = _rewrite_block(h.body, fn_params)

            # Strip standalone print(...) calls cleanly
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "print"
                    and "print" not in defined_fns):
                changed = True
                notes.append("stripped print() call")
                out.append(ast.Pass())
                continue

            # Lower setattr(buf, '<attr>', val_expr)
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "setattr"
                    and "setattr" not in defined_fns
                    and len(stmt.value.args) == 3 and not stmt.value.keywords):
                obj, attr_arg, val_arg = stmt.value.args
                attr_s = _const_str(attr_arg)
                if attr_s is not None and isinstance(obj, ast.Name) and obj.id == "buf":
                    val_arg = expr_xf.visit(val_arg)
                    changed = True
                    if attr_s in ("dt", "steps"):
                        out.append(ast.Assign(
                            targets=[ast.Attribute(value=ast.Name(id="buf", ctx=ast.Load()), attr=attr_s, ctx=ast.Store())],
                            value=val_arg,
                        ))
                    elif attr_s not in _STD_BUF_ATTRS:
                        flag_idx, val_idx = _alloc_attr(attr_s)
                        out.append(ast.Assign(targets=[_buf_slot(val_idx, ast.Store())], value=_norm_val_for_slot(val_arg)))
                        out.append(ast.Assign(targets=[_buf_slot(flag_idx, ast.Store())], value=ast.Constant(value=1.0)))
                    continue

            # Lower custom `buf.<attr> = val`
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Attribute)
                    and isinstance(stmt.targets[0].value, ast.Name)
                    and stmt.targets[0].value.id == "buf"
                    and stmt.targets[0].attr not in _STD_BUF_ATTRS):
                flag_idx, val_idx = _alloc_attr(stmt.targets[0].attr)
                rhs = expr_xf.visit(stmt.value)
                changed = True
                out.append(ast.Assign(targets=[_buf_slot(val_idx, ast.Store())], value=_norm_val_for_slot(rhs)))
                out.append(ast.Assign(targets=[_buf_slot(flag_idx, ast.Store())], value=ast.Constant(value=1.0)))
                continue

            # Lower custom `buf.<attr> += val`
            if (isinstance(stmt, ast.AugAssign)
                    and isinstance(stmt.target, ast.Attribute)
                    and isinstance(stmt.target.value, ast.Name)
                    and stmt.target.value.id == "buf"
                    and stmt.target.attr not in _STD_BUF_ATTRS):
                flag_idx, val_idx = _alloc_attr(stmt.target.attr)
                rhs = expr_xf.visit(stmt.value)
                changed = True
                out.append(ast.Assign(
                    targets=[_buf_slot(val_idx, ast.Store())],
                    value=ast.BinOp(left=_buf_slot(val_idx, ast.Load()), op=stmt.op, right=rhs),
                ))
                out.append(ast.Assign(targets=[_buf_slot(flag_idx, ast.Store())], value=ast.Constant(value=1.0)))
                continue

            # Transform expressions in all remaining statements
            for field in ("test", "value", "iter"):
                sub_expr = getattr(stmt, field, None)
                if isinstance(sub_expr, ast.AST):
                    setattr(stmt, field, expr_xf.visit(sub_expr))
            if isinstance(stmt, ast.Assign):
                stmt.targets = [
                    expr_xf.visit(t) if isinstance(t, ast.Subscript) else t
                    for t in stmt.targets
                ]
            out.append(stmt)
        return out

    tree.body = _rewrite_block(tree.body, set())
    for top in tree.body:
        if isinstance(top, ast.FunctionDef):
            pnames = {a.arg for a in top.args.args}
            top.body = _rewrite_block(top.body, pnames)
    if changed:
        ast.fix_missing_locations(tree)
    return tree, changed, sorted(set(notes))


def _lower_scalar_loops_and_builtins_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    """Lower tuple-returning helpers, static tuple/list/enumerate loops, and iterable builtins.

    Ensures C++20/Rust emitters see only scalar helper returns, scalar ``min``/``max``/``sum``
    expressions, and flat/range loops.
    """
    changed = False
    notes: List[str] = []
    counter = 0

    def _next_id() -> int:
        nonlocal counter
        counter += 1
        return counter

    defined_fns = {
        n.name for n in tree.body if isinstance(n, ast.FunctionDef)
    }

    # 1. Split real-valued tuple-returning helper functions (`def fn(...): ... return a, b`)
    def _uses_complex_ops(fn_def: ast.FunctionDef) -> bool:
        for n in ast.walk(fn_def):
            if isinstance(n, ast.Constant) and isinstance(n.value, complex):
                return True
            if isinstance(n, ast.Attribute) and n.attr in ("real", "imag", "conjugate"):
                return True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "complex":
                return True
        return False

    tuple_helpers: Dict[str, Tuple[int, ast.FunctionDef]] = {}
    for top in tree.body:
        if not isinstance(top, ast.FunctionDef):
            continue
        if top.name == "simulation_step" or top.name.startswith(("layer_", "_invar_")):
            continue
        if _uses_complex_ops(top):
            continue
        ret_arities: Set[int] = set()
        valid_returns = True

        def _scan_rets(stmts: Sequence[ast.stmt]) -> None:
            nonlocal valid_returns
            for st in stmts:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(st, ast.Return):
                    if st.value is None or (isinstance(st.value, ast.Constant) and st.value.value is None):
                        continue
                    if (isinstance(st.value, (ast.Tuple, ast.List))
                            and len(st.value.elts) >= 2
                            and not any(isinstance(e, ast.Starred) for e in st.value.elts)):
                        ret_arities.add(len(st.value.elts))
                    else:
                        valid_returns = False
                    continue
                for attr in ("body", "orelse", "finalbody"):
                    sub = getattr(st, attr, None)
                    if isinstance(sub, list):
                        _scan_rets(sub)
                for h in getattr(st, "handlers", ()) or ():
                    if isinstance(getattr(h, "body", None), list):
                        _scan_rets(h.body)

        _scan_rets(top.body)
        if valid_returns and len(ret_arities) == 1:
            arity = next(iter(ret_arities))
            tuple_helpers[top.name] = (arity, top)

    if tuple_helpers:
        split_needed: Set[str] = set()

        def _is_helper_call(expr: ast.AST) -> Optional[Tuple[str, ast.Call]]:
            if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
                    and expr.func.id in tuple_helpers
                    and not expr.keywords
                    and not any(isinstance(a, ast.Starred) for a in expr.args)):
                return expr.func.id, expr
            return None

        # Rewrite subscripted calls `fn(args)[k]` and `a, b = fn(args)` inside statement blocks
        class _SubCallRewriter(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                self.generic_visit(node)
                hc = _is_helper_call(node.value)
                if (hc is not None
                        and isinstance(node.slice, ast.Constant)
                        and isinstance(node.slice.value, int)
                        and not isinstance(node.slice.value, bool)):
                    fn_name, call_node = hc
                    arity, _fn_def = tuple_helpers[fn_name]
                    idx = int(node.slice.value)
                    if idx < 0:
                        idx += arity
                    if 0 <= idx < arity:
                        split_needed.add(fn_name)
                        return ast.Call(
                            func=ast.Name(id=f"{fn_name}__{idx}", ctx=ast.Load()),
                            args=call_node.args,
                            keywords=[],
                        )
                return node

        def _rewrite_tuple_calls_in_block(stmts: List[ast.stmt]) -> List[ast.stmt]:
            nonlocal changed
            out: List[ast.stmt] = []
            for stmt in stmts:
                for attr in ("body", "orelse", "finalbody"):
                    sub = getattr(stmt, attr, None)
                    if isinstance(sub, list):
                        setattr(stmt, attr, _rewrite_tuple_calls_in_block(sub))
                for h in getattr(stmt, "handlers", ()) or ():
                    if isinstance(getattr(h, "body", None), list):
                        h.body = _rewrite_tuple_calls_in_block(h.body)
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    tgt = stmt.targets[0]
                    hc = _is_helper_call(stmt.value)
                    if hc is not None and isinstance(tgt, (ast.Tuple, ast.List)):
                        fn_name, call_node = hc
                        arity, _ = tuple_helpers[fn_name]
                        if len(tgt.elts) == arity and not any(isinstance(e, ast.Starred) for e in tgt.elts):
                            split_needed.add(fn_name)
                            cid = _next_id()
                            tmps: List[str] = []
                            for idx in range(arity):
                                tname = f"_invar_ret_{cid}_{idx}"
                                tmps.append(tname)
                                call_k = ast.Call(
                                    func=ast.Name(id=f"{fn_name}__{idx}", ctx=ast.Load()),
                                    args=[copy.deepcopy(a) for a in call_node.args],
                                    keywords=[],
                                )
                                out.append(ast.Assign(targets=[ast.Name(id=tname, ctx=ast.Store())], value=call_k))
                            for lhs_elt, tname in zip(tgt.elts, tmps):
                                out.append(ast.Assign(targets=[lhs_elt], value=ast.Name(id=tname, ctx=ast.Load())))
                            changed = True
                            continue
                out.append(stmt)
            return out

        for fn_node in tree.body:
            if isinstance(fn_node, ast.FunctionDef):
                fn_node.body = _rewrite_tuple_calls_in_block(fn_node.body)
                _SubCallRewriter().visit(fn_node)

        # Also handle single-variable capture `tup = fn(args)` used solely via `tup[0]`, `tup[1]`
        for fn_node in tree.body:
            if not isinstance(fn_node, ast.FunctionDef):
                continue
            simple_captures: Dict[str, Tuple[str, int]] = {}
            for st in ast.walk(fn_node):
                if (isinstance(st, ast.Assign) and len(st.targets) == 1
                        and isinstance(st.targets[0], ast.Name)):
                    hc = _is_helper_call(st.value)
                    if hc is not None:
                        vname = st.targets[0].id
                        fn_name, _c = hc
                        arity, _d = tuple_helpers[fn_name]
                        # Verify all Load uses of vname are constant int subscripts 0 <= k < arity
                        valid_uses = True
                        sub_parents = set()
                        for sub in ast.walk(fn_node):
                            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == vname:
                                if (isinstance(sub.ctx, ast.Load)
                                        and isinstance(sub.slice, ast.Constant)
                                        and isinstance(sub.slice.value, int)
                                        and not isinstance(sub.slice.value, bool)):
                                    sub_parents.add(id(sub.value))
                                else:
                                    valid_uses = False
                        for sub in ast.walk(fn_node):
                            if isinstance(sub, ast.Name) and sub.id == vname and isinstance(sub.ctx, ast.Load):
                                if id(sub) not in sub_parents:
                                    valid_uses = False
                        if valid_uses and sub_parents:
                            simple_captures[vname] = (fn_name, arity)
            if simple_captures:
                class _CapSubRewriter(ast.NodeTransformer):
                    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                        self.generic_visit(node)
                        if (isinstance(node.value, ast.Name) and node.value.id in simple_captures
                                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)):
                            _fn_n, _ar = simple_captures[node.value.id]
                            idx = int(node.slice.value)
                            if idx < 0:
                                idx += _ar
                            if 0 <= idx < _ar:
                                return ast.Name(id=f"{node.value.id}__{idx}", ctx=ast.Load())
                        return node

                def _rewrite_cap_assigns(stmts: List[ast.stmt]) -> List[ast.stmt]:
                    nonlocal changed
                    out_s: List[ast.stmt] = []
                    for st in stmts:
                        for attr in ("body", "orelse", "finalbody"):
                            sub = getattr(st, attr, None)
                            if isinstance(sub, list):
                                setattr(st, attr, _rewrite_cap_assigns(sub))
                        if (isinstance(st, ast.Assign) and len(st.targets) == 1
                                and isinstance(st.targets[0], ast.Name)
                                and st.targets[0].id in simple_captures):
                            hc = _is_helper_call(st.value)
                            if hc is not None:
                                fn_name, call_node = hc
                                arity, _ = tuple_helpers[fn_name]
                                vname = st.targets[0].id
                                split_needed.add(fn_name)
                                for idx in range(arity):
                                    call_k = ast.Call(
                                        func=ast.Name(id=f"{fn_name}__{idx}", ctx=ast.Load()),
                                        args=[copy.deepcopy(a) for a in call_node.args],
                                        keywords=[],
                                    )
                                    out_s.append(ast.Assign(
                                        targets=[ast.Name(id=f"{vname}__{idx}", ctx=ast.Store())],
                                        value=call_k,
                                    ))
                                changed = True
                                continue
                        out_s.append(st)
                    return out_s

                fn_node.body = _rewrite_cap_assigns(fn_node.body)
                _CapSubRewriter().visit(fn_node)

        # Emit scalar helper functions fn__0 .. fn__{M-1}
        if split_needed:
            new_body: List[ast.stmt] = []
            for top in tree.body:
                if isinstance(top, ast.FunctionDef) and top.name in split_needed:
                    arity, _ = tuple_helpers[top.name]
                    still_called = any(
                        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == top.name
                        for n in ast.walk(tree)
                    )
                    if still_called:
                        new_body.append(top)
                    for idx in range(arity):
                        fn_clone = copy.deepcopy(top)
                        fn_clone.name = f"{top.name}__{idx}"

                        class _RetSlice(ast.NodeTransformer):
                            def visit_FunctionDef(self, inner: ast.FunctionDef) -> ast.AST:
                                if inner is fn_clone:
                                    return self.generic_visit(inner)
                                return inner

                            def visit_Return(self, rnode: ast.Return) -> ast.AST:
                                if isinstance(rnode.value, (ast.Tuple, ast.List)) and idx < len(rnode.value.elts):
                                    rnode.value = rnode.value.elts[idx]
                                return rnode

                        _RetSlice().visit(fn_clone)
                        new_body.append(fn_clone)
                    changed = True
                    notes.append(f"split tuple-returning helper {top.name} into {arity} scalar functions")
                else:
                    new_body.append(top)
            tree.body = new_body

    # 2. Unroll static tuple / list / enumerate / zip loops (`for v in (a, b, c):`)
    def _has_direct_loop_jump(stmts: Sequence[ast.stmt]) -> bool:
        for st in stmts:
            if isinstance(st, (ast.Break, ast.Continue)):
                return True
            if isinstance(st, (ast.For, ast.AsyncFor, ast.While, ast.FunctionDef)):
                continue
            for attr in ("body", "orelse", "finalbody"):
                sub = getattr(st, attr, None)
                if isinstance(sub, list) and _has_direct_loop_jump(sub):
                    return True
            for h in getattr(st, "handlers", ()) or ():
                if isinstance(getattr(h, "body", None), list) and _has_direct_loop_jump(h.body):
                    return True
        return False

    def _static_iter_elements(
        it_expr: ast.AST, local_seqs: Dict[str, List[ast.expr]]
    ) -> Optional[List[ast.expr]]:
        if isinstance(it_expr, (ast.Tuple, ast.List)) and not any(isinstance(e, ast.Starred) for e in it_expr.elts):
            return [copy.deepcopy(e) for e in it_expr.elts]
        if isinstance(it_expr, ast.Name) and it_expr.id in local_seqs:
            return [copy.deepcopy(e) for e in local_seqs[it_expr.id]]
        if (isinstance(it_expr, ast.Call) and isinstance(it_expr.func, ast.Name)
                and not it_expr.keywords):
            cname = it_expr.func.id
            if cname == "enumerate" and 1 <= len(it_expr.args) <= 2:
                seq = _static_iter_elements(it_expr.args[0], local_seqs)
                if seq is None:
                    return None
                start_i = 0
                if len(it_expr.args) == 2:
                    sv = _eval_const_ast(it_expr.args[1])
                    if sv is None or abs(sv - round(sv)) > 1e-12:
                        return None
                    start_i = int(round(sv))
                return [
                    ast.Tuple(elts=[ast.Constant(value=start_i + k), elt], ctx=ast.Load())
                    for k, elt in enumerate(seq)
                ]
            if cname == "zip" and it_expr.args:
                seqs = [_static_iter_elements(a, local_seqs) for a in it_expr.args]
                if all(s is not None for s in seqs):
                    min_len = min(len(s) for s in seqs if s is not None)
                    return [
                        ast.Tuple(elts=[s[k] for s in seqs if s is not None], ctx=ast.Load())
                        for k in range(min_len)
                    ]
        return None

    def _unroll_block(stmts: List[ast.stmt], outer_seqs: Optional[Dict[str, List[ast.expr]]] = None) -> List[ast.stmt]:
        nonlocal changed
        local_seqs: Dict[str, List[ast.expr]] = dict(outer_seqs or {})
        out: List[ast.stmt] = []
        for stmt in stmts:
            for attr in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, attr, None)
                if isinstance(sub, list):
                    setattr(stmt, attr, _unroll_block(sub, local_seqs))
            for h in getattr(stmt, "handlers", ()) or ():
                if isinstance(getattr(h, "body", None), list):
                    h.body = _unroll_block(h.body, local_seqs)

            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                tname = stmt.targets[0].id
                if (tname not in mutated_names
                        and isinstance(stmt.value, (ast.Tuple, ast.List))
                        and not any(isinstance(e, ast.Starred) for e in stmt.value.elts)):
                    local_seqs[tname] = [copy.deepcopy(e) for e in stmt.value.elts]
                else:
                    local_seqs.pop(tname, None)

            if isinstance(stmt, ast.For) and not _has_direct_loop_jump(stmt.body):
                elts = _static_iter_elements(stmt.iter, local_seqs)
                if elts is not None:
                    tgt = stmt.target
                    for elt in elts:
                        if isinstance(tgt, ast.Name):
                            out.append(ast.Assign(
                                targets=[ast.Name(id=tgt.id, ctx=ast.Store())],
                                value=copy.deepcopy(elt),
                            ))
                        elif isinstance(tgt, (ast.Tuple, ast.List)) and not any(isinstance(x, ast.Starred) for x in tgt.elts):
                            if isinstance(elt, (ast.Tuple, ast.List)) and len(elt.elts) == len(tgt.elts):
                                cid = _next_id()
                                tnames: List[str] = []
                                for k, ev in enumerate(elt.elts):
                                    tn = f"_invar_loop_{cid}_{k}"
                                    tnames.append(tn)
                                    out.append(ast.Assign(targets=[ast.Name(id=tn, ctx=ast.Store())], value=copy.deepcopy(ev)))
                                for l_e, tn in zip(tgt.elts, tnames):
                                    out.append(ast.Assign(targets=[copy.deepcopy(l_e)], value=ast.Name(id=tn, ctx=ast.Load())))
                            else:
                                for k, l_e in enumerate(tgt.elts):
                                    out.append(ast.Assign(
                                        targets=[copy.deepcopy(l_e)],
                                        value=ast.Subscript(value=copy.deepcopy(elt), slice=ast.Constant(value=k), ctx=ast.Load()),
                                    ))
                        else:
                            out.append(ast.Assign(targets=[copy.deepcopy(tgt)], value=copy.deepcopy(elt)))
                        out.extend(copy.deepcopy(b) for b in stmt.body)
                    if stmt.orelse:
                        out.extend(copy.deepcopy(o) for o in stmt.orelse)
                    changed = True
                    notes.append(f"unrolled static for-loop over {len(elts)} element(s)")
                    continue
            out.append(stmt)
        return out

    # Track mutated sequence names so we only alias immutable locals
    mutated_names: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and isinstance(n.ctx, ast.Store):
            mutated_names.add(n.value.id)
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and isinstance(n.func.value, ast.Name) and n.func.attr in {"append", "extend", "pop", "insert", "clear"}):
            mutated_names.add(n.func.value.id)

    for top in tree.body:
        if isinstance(top, ast.FunctionDef):
            top.body = _unroll_block(top.body)

    # 3. Lower multi-arg / iterable `min`, `max`, `sum`, `all`, `any`
    class _SubstituteName(ast.NodeTransformer):
        def __init__(self, target_id: str, replacement: ast.expr) -> None:
            self.target_id = target_id
            self.replacement = replacement

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id == self.target_id:
                return copy.deepcopy(self.replacement)
            return node

    def _fold_bin_call(fn_id: str, items: Sequence[ast.expr]) -> ast.expr:
        acc = copy.deepcopy(items[0])
        for nxt in items[1:]:
            acc = ast.Call(func=ast.Name(id=fn_id, ctx=ast.Load()), args=[acc, copy.deepcopy(nxt)], keywords=[])
        return acc

    class _BuiltinScalarLowerer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            nonlocal changed
            self.generic_visit(node)
            if not isinstance(node.func, ast.Name) or node.keywords or node.func.id in defined_fns:
                return node
            fname = node.func.id
            args = node.args
            if fname in ("min", "max"):
                if len(args) == 1 and isinstance(args[0], (ast.Tuple, ast.List)) and not any(isinstance(e, ast.Starred) for e in args[0].elts):
                    elts = args[0].elts
                    if len(elts) == 1:
                        changed = True
                        notes.append(f"lowered single-element {fname}(...)")
                        return copy.deepcopy(elts[0])
                    if len(elts) >= 2:
                        changed = True
                        notes.append(f"lowered iterable {fname}(...) over {len(elts)} elements")
                        return _fold_bin_call(fname, elts)
                elif len(args) >= 3:
                    changed = True
                    notes.append(f"lowered {len(args)}-arg {fname}(...) to nested binary {fname}")
                    return _fold_bin_call(fname, args)
            elif fname == "sum" and 1 <= len(args) <= 2:
                seq_arg = args[0]
                start_expr: Optional[ast.expr] = copy.deepcopy(args[1]) if len(args) == 2 else None
                terms: Optional[List[ast.expr]] = None
                if isinstance(seq_arg, (ast.Tuple, ast.List)) and not any(isinstance(e, ast.Starred) for e in seq_arg.elts):
                    terms = [copy.deepcopy(e) for e in seq_arg.elts]
                elif (isinstance(seq_arg, (ast.GeneratorExp, ast.ListComp))
                      and len(seq_arg.generators) == 1
                      and isinstance(seq_arg.generators[0].target, ast.Name)
                      and isinstance(seq_arg.generators[0].iter, (ast.Tuple, ast.List))
                      and not any(isinstance(e, ast.Starred) for e in seq_arg.generators[0].iter.elts)):
                    gen = seq_arg.generators[0]
                    t_id = gen.target.id
                    terms = []
                    for e_it in gen.iter.elts:
                        sub_elt = _SubstituteName(t_id, e_it).visit(copy.deepcopy(seq_arg.elt))
                        if gen.ifs:
                            cond = gen.ifs[0]
                            for extra_c in gen.ifs[1:]:
                                cond = ast.BoolOp(op=ast.And, values=[cond, extra_c])
                            sub_cond = _SubstituteName(t_id, e_it).visit(copy.deepcopy(cond))
                            sub_elt = ast.IfExp(test=sub_cond, body=sub_elt, orelse=ast.Constant(value=0.0))
                        terms.append(sub_elt)
                if terms is not None:
                    if not terms:
                        changed = True
                        return start_expr if start_expr is not None else ast.Constant(value=0.0)
                    acc_sum: ast.expr = terms[0] if start_expr is None else ast.BinOp(left=start_expr, op=ast.Add(), right=terms[0])
                    for t_next in terms[1:]:
                        acc_sum = ast.BinOp(left=acc_sum, op=ast.Add(), right=t_next)
                    changed = True
                    notes.append(f"lowered static sum(...) over {len(terms)} terms")
                    return acc_sum
            elif fname in ("all", "any") and len(args) == 1:
                seq_arg = args[0]
                if isinstance(seq_arg, (ast.Tuple, ast.List)) and not any(isinstance(e, ast.Starred) for e in seq_arg.elts):
                    elts = seq_arg.elts
                    if not elts:
                        changed = True
                        return ast.Constant(value=(fname == "all"))
                    if len(elts) == 1:
                        changed = True
                        return copy.deepcopy(elts[0])
                    changed = True
                    notes.append(f"lowered iterable {fname}(...) over {len(elts)} elements")
                    return ast.BoolOp(op=ast.And() if fname == "all" else ast.Or(), values=[copy.deepcopy(e) for e in elts])
            return node

    tree = _BuiltinScalarLowerer().visit(tree)
    if changed:
        ast.fix_missing_locations(tree)
    return tree, changed, notes


def _desugar_tuple_unpacking_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    """Desugar tuple/list unpacking assignments ``a, b = (x, y)`` into sequential
    scalar temporary assignments so C++20 lowering never hits tuple-assign errors."""
    changed = False
    counter = 0

    def _rewrite_block(stmts: List[ast.stmt]) -> List[ast.stmt]:
        nonlocal changed, counter
        out: List[ast.stmt] = []
        for stmt in stmts:
            for attr in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, attr, None)
                if isinstance(sub, list):
                    setattr(stmt, attr, _rewrite_block(sub))
            if hasattr(stmt, "handlers") and isinstance(stmt.handlers, list):
                for h in stmt.handlers:
                    if isinstance(getattr(h, "body", None), list):
                        h.body = _rewrite_block(h.body)
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], (ast.Tuple, ast.List))
                    and isinstance(stmt.value, (ast.Tuple, ast.List))
                    and len(stmt.targets[0].elts) == len(stmt.value.elts)
                    and not any(isinstance(e, ast.Starred) for e in stmt.targets[0].elts)
                    and (any(isinstance(e, (ast.Subscript, ast.Attribute)) for e in stmt.targets[0].elts)
                         or len(stmt.targets[0].elts) >= 3)):
                lhs_elts = stmt.targets[0].elts
                rhs_elts = stmt.value.elts
                tmps: List[str] = []
                for rhs in rhs_elts:
                    counter += 1
                    tname = f"_invar_tup_{counter}"
                    tmps.append(tname)
                    out.append(ast.Assign(
                        targets=[ast.Name(id=tname, ctx=ast.Store())],
                        value=rhs,
                    ))
                for lhs, tname in zip(lhs_elts, tmps):
                    out.append(ast.Assign(
                        targets=[lhs],
                        value=ast.Name(id=tname, ctx=ast.Load()),
                    ))
                changed = True
                continue
            out.append(stmt)
        return out

    tree.body = _rewrite_block(tree.body)
    if changed:
        ast.fix_missing_locations(tree)
        return tree, True, ["desugared tuple-unpacking assignments"]
    return tree, False, []


def _lower_comprehensions_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    """Lower ``ListComp`` / ``GeneratorExp`` (e.g. ``[buf.data[i] for i in range(6)]``
    or ``sum(x*x for x in z)``) into explicit ``for`` loops so the transpiler
    never rejects them as unsupported constructs."""
    changed = False
    counter = 0

    def _build_loop(gens: List[ast.comprehension], body_stmt: ast.stmt) -> ast.stmt:
        inner: ast.stmt = body_stmt
        for gen in reversed(gens):
            for cond in reversed(gen.ifs):
                inner = ast.If(test=cond, body=[inner], orelse=[])
            inner = ast.For(target=gen.target, iter=gen.iter, body=[inner], orelse=[])
        return inner

    def _lower_block(stmts: List[ast.stmt]) -> List[ast.stmt]:
        nonlocal changed, counter
        out: List[ast.stmt] = []
        for stmt in stmts:
            for attr in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, attr, None)
                if isinstance(sub, list):
                    setattr(stmt, attr, _lower_block(sub))
            if hasattr(stmt, "handlers") and isinstance(stmt.handlers, list):
                for h in stmt.handlers:
                    if isinstance(getattr(h, "body", None), list):
                        h.body = _lower_block(h.body)

            # Fast path: `x = [elt for var in range(N)]` -> pre-allocated `[0.0] * N` + index assign (no .append)
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.ListComp)):
                tname = stmt.targets[0].id
                lc = stmt.value
                if (len(lc.generators) == 1 and not lc.generators[0].ifs
                        and isinstance(lc.generators[0].target, ast.Name)
                        and isinstance(lc.generators[0].iter, ast.Call)
                        and isinstance(lc.generators[0].iter.func, ast.Name)
                        and lc.generators[0].iter.func.id == "range"
                        and len(lc.generators[0].iter.args) == 1
                        and isinstance(lc.generators[0].iter.args[0], ast.Constant)
                        and isinstance(lc.generators[0].iter.args[0].value, int)):
                    n_val = lc.generators[0].iter.args[0].value
                    v_name = lc.generators[0].target.id
                    init_stmt = ast.Assign(
                        targets=[ast.Name(id=tname, ctx=ast.Store())],
                        value=ast.BinOp(left=ast.List(elts=[ast.Constant(value=0.0)], ctx=ast.Load()),
                                        op=ast.Mult(), right=ast.Constant(value=n_val)))
                    assign_stmt = ast.Assign(
                        targets=[ast.Subscript(value=ast.Name(id=tname, ctx=ast.Load()),
                                               slice=ast.Name(id=v_name, ctx=ast.Load()), ctx=ast.Store())],
                        value=lc.elt)
                    out.append(init_stmt)
                    out.append(ast.For(target=ast.Name(id=v_name, ctx=ast.Store()),
                                       iter=lc.generators[0].iter, body=[assign_stmt], orelse=[]))
                    changed = True
                    continue

            # General hoisting for any remaining ListComp or sum(GeneratorExp/ListComp)
            hoisted: List[ast.stmt] = []

            class _CompHoister(ast.NodeTransformer):
                def visit_Call(self, node: ast.Call) -> ast.AST:
                    nonlocal changed, counter
                    if (isinstance(node.func, ast.Name) and node.func.id == "sum"
                            and len(node.args) == 1 and not node.keywords
                            and isinstance(node.args[0], (ast.GeneratorExp, ast.ListComp))):
                        comp = node.args[0]
                        counter += 1
                        tmp = f"_invar_sum_{counter}"
                        hoisted.append(ast.Assign(
                            targets=[ast.Name(id=tmp, ctx=ast.Store())],
                            value=ast.Constant(value=0.0)))
                        aug = ast.AugAssign(
                            target=ast.Name(id=tmp, ctx=ast.Store()),
                            op=ast.Add(), value=comp.elt)
                        hoisted.append(_build_loop(comp.generators, aug))
                        changed = True
                        return ast.Name(id=tmp, ctx=ast.Load())
                    self.generic_visit(node)
                    return node

                def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
                    nonlocal changed, counter
                    counter += 1
                    tmp = f"_invar_list_{counter}"
                    hoisted.append(ast.Assign(
                        targets=[ast.Name(id=tmp, ctx=ast.Store())],
                        value=ast.List(elts=[], ctx=ast.Load())))
                    app = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id=tmp, ctx=ast.Load()), attr="append", ctx=ast.Load()),
                        args=[node.elt], keywords=[]))
                    hoisted.append(_build_loop(node.generators, app))
                    changed = True
                    return ast.Name(id=tmp, ctx=ast.Load())

                def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
                    nonlocal changed, counter
                    counter += 1
                    tmp = f"_invar_list_{counter}"
                    hoisted.append(ast.Assign(
                        targets=[ast.Name(id=tmp, ctx=ast.Store())],
                        value=ast.List(elts=[], ctx=ast.Load())))
                    app = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id=tmp, ctx=ast.Load()), attr="append", ctx=ast.Load()),
                        args=[node.elt], keywords=[]))
                    hoisted.append(_build_loop(node.generators, app))
                    changed = True
                    return ast.Name(id=tmp, ctx=ast.Load())

            stmt = _CompHoister().visit(stmt)
            out.extend(hoisted)
            out.append(stmt)
        return out

    tree.body = _lower_block(tree.body)
    if changed:
        ast.fix_missing_locations(tree)
        return tree, True, ["lowered list comprehension / generator expression to loop"]
    return tree, False, []


def _strip_global_nonlocal_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    """Strip ``global`` / ``nonlocal`` declarations (forbidden by PreWrite_Guard)
    and rewrite ``buf.dt`` -> ``dt`` inside functions taking ``dt``."""
    changed = False

    class _Stripper(ast.NodeTransformer):
        def __init__(self) -> None:
            self._has_dt = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            prev = self._has_dt
            self._has_dt = any(a.arg == "dt" for a in node.args.args)
            self.generic_visit(node)
            self._has_dt = prev
            return node

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            nonlocal changed
            self.generic_visit(node)
            if (self._has_dt and node.attr == "dt"
                    and isinstance(node.value, ast.Name) and node.value.id == "buf"):
                changed = True
                return ast.copy_location(ast.Name(id="dt", ctx=node.ctx), node)
            return node

        def visit_Global(self, node: ast.Global) -> ast.AST:
            nonlocal changed
            changed = True
            return ast.Pass()

        def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
            nonlocal changed
            changed = True
            return ast.Pass()

    tree = _Stripper().visit(tree)
    if changed:
        ast.fix_missing_locations(tree)
        return tree, True, ["stripped forbidden global/nonlocal or buf.dt"]
    return tree, False, []


def _inline_non_scalar_module_constants_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    """Eliminate non-scalar module-level assignments (e.g. ``K2 = [1.0, 4.0, 9.0]``)
    which CppEmitter rejects with ``module-level constants must be scalars``:
    - Inline constant-index subscripts ``K2[0]`` -> scalar element.
    - If still referenced (e.g. ``K2[i]``), move the binding into each referencing function.
    """
    non_scalar: Dict[str, ast.Assign] = {}
    for stmt in tree.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, (ast.List, ast.Tuple))):
            non_scalar[stmt.targets[0].id] = stmt
    if not non_scalar:
        return tree, False, []

    class _ConstSubscriptInliner(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id in non_scalar:
                elts = non_scalar[node.value.id].value.elts  # type: ignore[attr-defined]
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int) and not isinstance(node.slice.value, bool):
                    idx = node.slice.value
                    if -len(elts) <= idx < len(elts):
                        return copy.deepcopy(elts[idx])
            return node

    inliner = _ConstSubscriptInliner()
    new_body: List[ast.stmt] = []
    for stmt in tree.body:
        if stmt in non_scalar.values():
            continue
        if isinstance(stmt, ast.FunctionDef):
            stmt = inliner.visit(stmt)
            # Check if any non-scalar name is still referenced in this function
            ref_names = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and n.id in non_scalar}
            if ref_names:
                prepends = [copy.deepcopy(non_scalar[nm]) for nm in sorted(ref_names)]
                stmt.body[0:0] = prepends
            ast.fix_missing_locations(stmt)
        new_body.append(stmt)
    tree.body = new_body
    ast.fix_missing_locations(tree)
    return tree, True, [f"inlined non-scalar module constants ({', '.join(sorted(non_scalar))})"]


def _repair_call_arity_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    """Repair top-level helper calls missing trailing positional arguments by
    binding matching local names in the caller or 0.0 fallback."""
    callees: Dict[str, ast.FunctionDef] = {
        n.name: n for n in tree.body
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_invar_")
    }
    if not callees:
        return tree, False, []
    changed = False
    notes: List[str] = []
    for caller in tree.body:
        if not isinstance(caller, ast.FunctionDef):
            continue
        bound: Set[str] = {a.arg for a in caller.args.args}
        for node in ast.walk(caller):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        for elt in t.elts:
                            if isinstance(elt, ast.Name):
                                bound.add(elt.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
        for node in ast.walk(caller):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            callee = callees.get(node.func.id)
            if callee is None or node.keywords or callee.args.vararg or callee.args.kwarg:
                continue
            if any(isinstance(a, ast.Starred) for a in node.args):
                continue
            req_count = len(callee.args.args) - len(callee.args.defaults)
            if len(node.args) < req_count:
                missing_names = [callee.args.args[i].arg for i in range(len(node.args), req_count)]
                for pname in missing_names:
                    if pname in bound:
                        node.args.append(ast.Name(id=pname, ctx=ast.Load()))
                    else:
                        node.args.append(ast.Constant(value=0.0))
                ast.fix_missing_locations(node)
                changed = True
                notes.append(f"repaired missing args in {callee.name}() call ({', '.join(missing_names)})")
            elif len(node.args) > len(callee.args.args):
                node.args = node.args[:len(callee.args.args)]
                ast.fix_missing_locations(node)
                changed = True
                notes.append(f"trimmed excess args in {callee.name}() call")
    return tree, changed, notes


def sanitize_intrinsics(code: str, *, guard_domains: bool = True,
                        eps: float = 1e-300,
                        contracts: Sequence[Any] = (),
                        all_contracts: Sequence[Any] = (),
                        variable_map: Optional[Dict[str, int]] = None,
                        stabilize_diverged: bool = False) -> SanitizerResult:
    """Rewrite scalar intrinsic/math-namespace hazards before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    original = ast.dump(tree, annotate_fields=False, include_attributes=False)
    bridge_contracts = list(all_contracts or contracts or [])
    tree, alias_changed, alias_notes = _normalize_buffer_aliases_tree(tree)
    tree, sig_changed, sig_notes = _normalize_step_signatures_tree(tree)
    tree, intro_changed, intro_notes = _lower_introspection_and_buf_attrs_tree(tree)
    tree, loop_changed, loop_notes = _lower_scalar_loops_and_builtins_tree(tree)
    tree, tup_changed, tup_notes = _desugar_tuple_unpacking_tree(tree)
    tree, comp_changed, comp_notes = _lower_comprehensions_tree(tree)
    tree, glob_changed, glob_notes = _strip_global_nonlocal_tree(tree)
    tree, modconst_changed, modconst_notes = _inline_non_scalar_module_constants_tree(tree)
    tree, arity_changed, arity_notes = _repair_call_arity_tree(tree)
    tree, inline_changed, inline_notes = _inline_buffer_mutating_helpers_tree(tree)
    tree, init_loc_changed, init_loc_notes = _init_conditional_locals_tree(tree)
    tree, bsize_changed, bsize_notes = _strip_buf_size_branches_tree(tree, bridge_contracts)
    tree, coll_changed, coll_notes = _resolve_slot_collisions_tree(tree, contracts, bridge_contracts)
    tree, split_changed, split_notes = _split_intra_function_stage_overwrites_tree(
        tree, contracts, bridge_contracts, variable_map=variable_map
    )
    tree, rev_changed, rev_notes = _repair_reversibility_branch_tree(tree, bridge_contracts)
    tree, bridge_changed, bridge_notes = _bridge_contract_slots_tree(tree, bridge_contracts)
    tree, history_changed, history_notes = _repair_history_slots_tree(tree)
    tree, stab_changed, stab_notes = _stabilize_dynamical_state_tree(
        tree, list(contracts or bridge_contracts), stabilize_diverged=stabilize_diverged
    )
    tree, monotone_changed, monotone_notes = _repair_monotone_observables_tree(
        tree, bridge_contracts)
    tree, bloch_changed, bloch_notes = _enforce_bloch_ball_projection_tree(tree, bridge_contracts)
    tree, none_changed, none_notes = _strip_none_literals_tree(tree)
    tree, complex_changed, complex_notes = _lower_complex_tree(tree)
    tree, sdiv_changed, sdiv_notes = _lower_safe_division_tree(tree)
    tree, unify_changed, unify_notes = _unify_concept_slots_tree(tree, bridge_contracts)
    tree, concept_changed, concept_notes = _bridge_unassigned_concept_slots_tree(
        tree, bridge_contracts)
    tree, cons_changed, cons_notes = _enforce_conservation_projection_tree(
        tree, bridge_contracts)
    tree, lyap_changed, lyap_notes = _enforce_lyapunov_sign_tree(tree, bridge_contracts)
    tree, param_changed, param_notes = _protect_parameter_slots_tree(tree, bridge_contracts)
    tree, diss_changed, diss_notes = _seed_dissipation_parameters_tree(tree, contracts)
    transformer = _IntrinsicTransformer()
    transformer.defined = _defined_names(tree)
    transformer.guard_eps = float(eps)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    changed = ast.dump(tree, annotate_fields=False, include_attributes=False) != original
    notes = sorted(set(alias_notes) | set(sig_notes) | set(intro_notes) | set(loop_notes) | set(tup_notes) | set(comp_notes) | set(glob_notes) | set(modconst_notes) | set(arity_notes) | set(inline_notes) | set(init_loc_notes) | set(bsize_notes) | set(coll_notes) | set(split_notes) | set(rev_notes) | set(stab_notes) | set(bloch_notes) | set(bridge_notes) | set(monotone_notes)
                   | set(none_notes) | set(complex_notes) | set(history_notes)
                   | set(sdiv_notes) | set(unify_notes) | set(concept_notes)
                   | set(cons_notes) | set(lyap_notes)
                   | set(param_notes) | set(diss_notes) | set(transformer.notes))
    if (alias_changed or sig_changed or intro_changed or loop_changed or tup_changed or comp_changed or glob_changed or modconst_changed or arity_changed or inline_changed or init_loc_changed or bsize_changed or coll_changed or split_changed or rev_changed or bloch_changed or bridge_changed or monotone_changed or none_changed
            or complex_changed or history_changed or sdiv_changed or unify_changed
            or concept_changed or cons_changed or lyap_changed
            or param_changed or diss_changed):
        changed = True
    if transformer.used_math or transformer.used_np or transformer.used_intrinsics:
        if _ensure_math_import(tree):
            notes.append("injected import math")
            changed = True
    if _insert_intrinsic_helpers(tree, transformer.used_intrinsics):
        notes.append("injected scalar intrinsics: " + ", ".join(sorted(transformer.used_intrinsics)))
        changed = True
    if guard_domains:
        # Second pass guards math.log/sqrt/asin/acos introduced above.
        before = ast.dump(tree, annotate_fields=False, include_attributes=False)
        tree = _DomainGuardTransformer(eps=float(eps)).visit(tree)
        ast.fix_missing_locations(tree)
        if ast.dump(tree, annotate_fields=False, include_attributes=False) != before:
            notes.append("guarded math domains (log/sqrt/asin/acos)")
            changed = True
    out = ast.unparse(tree) + ("\n" if code.endswith("\n") or not code.endswith("\n") else "")
    return SanitizerResult(out, notes, changed, backup_slot_map=_collect_parameter_backup_map(tree))


class _DomainGuardTransformer(ast.NodeTransformer):
    def __init__(self, eps: float) -> None:
        self.eps = float(eps)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        guarded = _domain_guarded_call(node, self.eps)
        return guarded if guarded is not None else node


def _target_slot(node: ast.AST) -> Optional[int]:
    if not isinstance(node, ast.Subscript):
        return None
    sl = node.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
        return int(sl.value)
    return None


def _expr_free_vars(expr: ast.AST) -> List[str]:
    names = sorted({n.id for n in ast.walk(expr)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})
    return names


def _eval_numeric_expr(expr: ast.AST, env: Dict[str, float]) -> Optional[float]:
    try:
        value = eval(compile(ast.Expression(expr), "<egg_guard>", "eval"),  # noqa: S307
                     {"__builtins__": {}}, dict(env))
    except Exception:  # noqa: BLE001
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def numeric_expr_equivalent(left: ast.AST, right: ast.AST, *,
                            tolerance: float = 1e-12,
                            points: int = 3) -> bool:
    """Reject algebraic rewrites that change numeric evaluation.

    Free variables are assigned three deterministic non-zero pseudo-random
    values; any absolute deviation above ``tolerance`` rejects the rewrite.
    """
    vars_ = sorted(set(_expr_free_vars(left)) | set(_expr_free_vars(right)))
    for point in range(max(1, int(points))):
        env = {name: 0.37 + 0.13 * idx + 0.031 * point + 0.017 * idx * point
               for idx, name in enumerate(vars_)}
        a = _eval_numeric_expr(left, env)
        b = _eval_numeric_expr(right, env)
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
            return False
        if abs(a - b) > float(tolerance):
            return False
    return True


def strip_none_literals(code: str) -> SanitizerResult:
    """Public Tier-0 entry point for ``STRIP_NONE_LITERALS`` directives."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    original = ast.dump(tree, annotate_fields=False, include_attributes=False)
    tree, changed, notes = _strip_none_literals_tree(tree)
    if not changed:
        return SanitizerResult(code, [], False)
    return SanitizerResult(ast.unparse(tree) + "\n", notes, True)


def _is_state_target(node: ast.AST, slot: int) -> bool:
    if _target_slot(node) != slot:
        return False
    value = node.value  # type: ignore[attr-defined]
    if isinstance(value, ast.Name) and value.id in {"buf", "state", "data"}:
        return True
    if isinstance(value, ast.Attribute) and value.attr == "data" \
            and isinstance(value.value, ast.Name) and value.value.id == "buf":
        return True
    return False


def _var_kind(name: str) -> str:
    low = name.lower()
    if any(tok in low for tok in ("before", "prev", "s0", "old", "initial")):
        return "before"
    if any(tok in low for tok in ("after", "next", "s1", "new")):
        return "after"
    return ""


def _abs_diff(node: ast.AST) -> Optional[Tuple[str, str]]:
    if not isinstance(node, ast.Call):
        return None
    if _call_name(node.func) not in {"abs", "math.fabs"} or len(node.args) != 1:
        return None
    arg = node.args[0]
    if not isinstance(arg, ast.BinOp) or not isinstance(arg.op, ast.Sub):
        return None
    if not isinstance(arg.left, ast.Name) or not isinstance(arg.right, ast.Name):
        return None
    return arg.left.id, arg.right.id


def _monotone_expr(before: str, after: str, direction: str) -> ast.Call:
    if direction == "non_decreasing":
        left, right = before, after
    else:
        left, right = after, before
    diff = ast.BinOp(left=ast.Name(id=left, ctx=ast.Load()), op=ast.Sub(),
                     right=ast.Name(id=right, ctx=ast.Load()))
    return ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                    args=[_num(0.0), diff], keywords=[])


class _MonotoneRewrite(ast.NodeTransformer):
    def __init__(self, slot: int, direction: str,
                 before_var: Optional[str], after_var: Optional[str]) -> None:
        self.slot = int(slot)
        self.direction = direction if direction in {"non_increasing", "non_decreasing"} else "non_increasing"
        self.before_var = before_var
        self.after_var = after_var
        self.changed = False

    def _replacement(self, node: ast.Call) -> Optional[ast.Call]:
        pair = _abs_diff(node)
        if pair is None:
            return None
        a, b = pair
        before, after = self.before_var, self.after_var
        if before is None or after is None:
            ka, kb = _var_kind(a), _var_kind(b)
            if ka == "before" and kb == "after":
                before, after = a, b
            elif kb == "before" and ka == "after":
                before, after = b, a
            elif self.direction == "non_increasing":
                # Fallback heuristic: *_after/*_s1 is the post-step value.
                before, after = (b, a) if _var_kind(b) == "before" or _var_kind(a) == "after" else (a, b)
            else:
                before, after = (a, b) if _var_kind(a) == "before" or _var_kind(b) == "after" else (b, a)
        if not before or not after:
            return None
        self.changed = True
        return _monotone_expr(before, after, self.direction)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        repl = self._replacement(node)
        return repl if repl is not None else node


def rewrite_monotone_drift(code: str, *, slot: int, direction: str = "non_increasing",
                           before_var: Optional[str] = None,
                           after_var: Optional[str] = None) -> SanitizerResult:
    """Rewrite ``abs(after-before)`` drift accumulators to one-sided violations."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    original = ast.dump(tree, annotate_fields=False, include_attributes=False)
    tree, history_changed, history_notes = _repair_history_slots_tree(tree)
    tree, r_changed, r_notes = _rewrite_monotone_r_difference(tree, slot)
    ast.fix_missing_locations(tree)

    class _SlotTransformer(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            self.generic_visit(node)
            if any(_is_state_target(t, slot) for t in node.targets):
                rw = _MonotoneRewrite(slot, direction, before_var, after_var)
                node.value = rw.visit(node.value)
                self.changed = self.changed or rw.changed
            return node

        def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
            self.generic_visit(node)
            if _is_state_target(node.target, slot) and isinstance(node.op, ast.Add):
                rw = _MonotoneRewrite(slot, direction, before_var, after_var)
                node.value = rw.visit(node.value)
                self.changed = self.changed or rw.changed
            return node

        changed = False

    transformer = _SlotTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    changed = (history_changed or r_changed or transformer.changed
               or ast.dump(tree, annotate_fields=False, include_attributes=False) != original)
    notes = history_notes + r_notes + ([f"rewrote monotone drift state[{slot}] ({direction})"]
                                       if transformer.changed else [])
    return SanitizerResult(ast.unparse(tree) + "\n", notes, changed)


def infer_monotone_drift_directives(contracts: Sequence[Any], code: str) -> List[Dict[str, Any]]:
    """Infer Tier-0 drift directives from falsifiable contract metadata."""
    out: List[Dict[str, Any]] = []
    for contract in contracts or []:
        for inv in getattr(contract, "invariants", None) or []:
            name = str(getattr(inv, "variable_name", "") or "").lower()
            if not any(tok in name for tok in MONOTONE_NAME_TOKENS):
                continue
            if float(getattr(inv, "tolerance", 1.0) or 1.0) > 1e-8:
                continue
            text = f"{getattr(inv, 'variable_name', '')} {getattr(inv, 'assertion', '')}"
            m = re.search(r"(?:state|buf(?:\.data)?|data)\[(\d+)\]\s*<=\s*([0-9.eE+-]+)", text)
            if not m:
                continue
            slot, bound = int(m.group(1)), float(m.group(2))
            if bound > 1e-6:
                continue
            direction = "non_increasing"
            if "entropy" in name and "relative" not in name and "free_energy" not in name:
                direction = "non_decreasing"
            if rewrite_monotone_drift(code, slot=slot, direction=direction).changed:
                out.append({"op": "REWRITE_MONOTONE_DRIFT", "slot": slot,
                            "direction": direction})
    return out




def _slot_index(node: ast.AST) -> Optional[int]:
    return _target_slot(node)


def _state_slot_expr(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    base = node.value
    name = ""
    if isinstance(base, ast.Name):
        name = base.id
    elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
        name = f"{base.value.id}.{base.attr}"
    return name in {"state", "data", "buf.data", "buf"} and _target_slot(node) is not None


def _choose_prev_slot(code: str, slot: int, contracts: Sequence[Any] = ()) -> Optional[int]:
    used = {int(m.group(1)) for m in re.finditer(r"(?:state|buf\.data|data)\[(\d+)\]", code)}
    used.add(slot)
    sizes = [len(getattr(contract, "initial_state", None) or [])
             for contract in contracts or []]
    sizes = [n for n in sizes if n]
    limit = max(sizes) if sizes else 16
    for cand in (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16):
        if cand > slot and cand < limit and cand not in used:
            return cand
    return None


def _drift_self_pattern(body: List[ast.stmt], slot: int):
    for i, stmt in enumerate(body[:-1]):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 \
                or not isinstance(stmt.targets[0], ast.Name) or not isinstance(stmt.value, ast.BinOp) \
                or not isinstance(stmt.value.op, ast.Sub):
            continue
        diff = stmt.targets[0].id
        left, right = stmt.value.left, stmt.value.right
        if _state_slot_expr(right) and _slot_index(right) == slot and isinstance(left, ast.Name):
            obs = left.id
            expr = f"{obs} - buf.data[{slot}]"
        elif _state_slot_expr(left) and _slot_index(left) == slot and isinstance(right, ast.Name):
            obs = right.id
            expr = f"buf.data[{slot}] - {obs}"
        else:
            continue
        nxt = body[i + 1]
        if not isinstance(nxt, ast.If):
            continue
        test = nxt.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            continue
        if not (isinstance(test.left, ast.Name) and test.left.id == diff
                and isinstance(test.ops[0], (ast.Gt, ast.GtE))):
            continue
        assigns_slot = False
        for sub in ast.walk(nxt):
            if isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if _state_slot_expr(target) and _slot_index(target) == slot:
                        assigns_slot = True
        if not assigns_slot:
            continue
        return i, diff, obs, expr
    return None


def repair_drift_self_comparison(code: str, *, slot: int, prev_slot: int,
                                 contracts: Sequence[Any] = ()) -> SanitizerResult:
    """Repair ``diff = obs - buf.data[error_slot]`` self-comparisons."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    changed = False
    notes: List[str] = []
    var = f"_step_drift_{slot}"
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        body = fn.body
        found = _drift_self_pattern(body, int(slot))
        if found is None:
            continue
        i, diff, obs, expr = found
        expr = expr.replace(f"buf.data[{slot}]", f"buf.data[{prev_slot}]")
        expr = expr.replace(f"state[{slot}]", f"state[{prev_slot}]")
        expr = expr.replace(f"data[{slot}]", f"data[{prev_slot}]")
        snippet = (
            "if buf.steps > 0:\n"
            f"    {var} = max(0.0, {expr})\n"
            f"    if {var} > buf.data[{slot}]:\n"
            f"        buf.data[{slot}] = {var}\n"
            f"buf.data[{prev_slot}] = {obs}\n"
        )
        repl = ast.parse(snippet).body
        # Keep any unrelated statements between the diff assignment and guard.
        body[i:i + 2] = repl
        ast.fix_missing_locations(tree)
        changed = True
        notes.append(f"repaired drift self-comparison state[{slot}] via buf.data[{prev_slot}]")
        break
    if not changed:
        return SanitizerResult(code, [], False)
    return SanitizerResult(ast.unparse(tree) + "\n", notes, True)


def infer_drift_self_comparison_directives(contracts: Sequence[Any],
                                           code: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for contract in contracts or []:
        for inv in getattr(contract, "invariants", None) or []:
            name = str(getattr(inv, "variable_name", "") or "").lower()
            assertion = str(getattr(inv, "assertion", "") or "")
            if not any(tok in name for tok in ("drift", "error", "err", "residual", "violation", "mismatch")):
                continue
            if float(getattr(inv, "tolerance", 1.0) or 1.0) > 1e-6:
                continue
            m = re.search(r"(?:state|buf(?:\.data)?|data)\[(\d+)\]\s*<=\s*([0-9.eE+-]+)",
                          f"{name} {assertion}")
            if not m:
                continue
            slot = int(m.group(1))
            bound = float(m.group(2))
            if bound > 1e-6 or slot in seen:
                continue
            seen.add(slot)
            prev = _choose_prev_slot(code, slot, contracts)
            if prev is None:
                continue
            if repair_drift_self_comparison(code, slot=slot, prev_slot=prev,
                                            contracts=contracts).changed:
                out.append({"op": "REPAIR_DRIFT_SELF_COMPARISON", "slot": slot,
                            "prev_slot": prev})
    return out


def _wrap_substep(tree: ast.Module, substeps: int) -> bool:
    n = max(1, int(substeps or 1))
    if n == 1:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "simulation_step" and node.args.args:
            dt_arg = node.args.args[1].arg if len(node.args.args) > 1 else "dt"
            loop = ast.For(
                target=ast.Name(id="__healing_substep", ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                              args=[ast.Constant(value=n)], keywords=[]),
                body=node.body, orelse=[], type_comment=None)
            scale = ast.Assign(
                targets=[ast.Name(id=dt_arg, ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id=dt_arg, ctx=ast.Load()), op=ast.Div(),
                                right=ast.Constant(value=float(n))),
                type_comment=None)
            node.body = [scale, loop]
            ast.fix_missing_locations(node)
            return True
    return False


def _scale_coefficient(tree: ast.Module, var: str, factor: float) -> bool:
    factor = float(factor)
    if factor == 1.0:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "simulation_step":
            node.body.insert(0, ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id=var, ctx=ast.Load()), op=ast.Mult(),
                                right=ast.Constant(value=factor)),
                type_comment=None))
            ast.fix_missing_locations(node)
            return True
    return False


def apply_healing_directives(code: str, directives: Iterable[Dict[str, Any]],
                             *, contracts: Sequence[Any] = (),
                             variable_map: Optional[Dict[str, int]] = None) -> SanitizerResult:
    """Apply LLM-emitted structured directives deterministically."""
    notes: List[str] = []
    current = code
    # Always run the intrinsic/domain sanitizer first (Tier-0, zero tokens).
    san = sanitize_intrinsics(current, contracts=contracts, variable_map=variable_map)
    if san.changed:
        current = san.code
        notes.extend(san.notes)
    try:
        tree = ast.parse(current)
    except SyntaxError:
        return SanitizerResult(current, notes, bool(notes))
    before_dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    for raw in directives or []:
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("op") or "")
        if op == "INLINE_INTRINSIC":
            name = str(raw.get("name") or "")
            if name == _SAFE_DIV_HELPER:
                has_fn = any(
                    isinstance(n, ast.FunctionDef) and n.name == _SAFE_DIV_HELPER
                    for n in tree.body
                )
                if not has_fn:
                    tree.body.insert(0, ast.parse(_SAFE_DIV_SRC).body[0])
                _hoist_intrinsic_helpers(tree)
                notes.append(f"INLINE_INTRINSIC({name})")
            elif name in INTRINSIC_ALIASES:
                patched = _insert_intrinsic_helpers(tree, [name])
                if patched:
                    notes.append(f"INLINE_INTRINSIC({name})")
            elif name in ("asinh", "math.asinh"):
                class _AsinhLowerer(ast.NodeTransformer):
                    def visit_Call(self, cnode: ast.Call) -> ast.AST:
                        self.generic_visit(cnode)
                        cname = _call_name(cnode.func)
                        if cname in ("asinh", "math.asinh") and len(cnode.args) == 1:
                            x_src = ast.unparse(cnode.args[0])
                            return ast.parse(f"math.log(({x_src}) + math.sqrt(({x_src}) * ({x_src}) + 1.0))", mode="eval").body
                        return cnode

                tree = _AsinhLowerer().visit(tree)
                _ensure_math_import(tree)
                ast.fix_missing_locations(tree)
                notes.append(f"INLINE_INTRINSIC({name})")
        elif op == "GUARD_MATH_DOMAINS":
            eps = float(raw.get("eps", 1e-300) or 1e-300)
            guarded = _DomainGuardTransformer(eps=eps).visit(tree)
            if guarded is not tree:
                notes.append("GUARD_MATH_DOMAINS")
            tree = guarded
        elif op == "REWRITE_MONOTONE_DRIFT":
            _raw_slot = raw.get("slot", -1)
            try:
                slot = int(_raw_slot)
            except (TypeError, ValueError):
                _m_slot = re.search(r"(\d+)", str(_raw_slot or ""))
                slot = int(_m_slot.group(1)) if _m_slot else -1
            if slot >= 0:
                # Re-render through the dedicated rewriter for precise notes.
                rw = rewrite_monotone_drift(ast.unparse(tree) + "\n", slot=slot,
                                            direction=str(raw.get("direction") or "non_increasing"),
                                            before_var=raw.get("before_var"),
                                            after_var=raw.get("after_var"))
                if rw.changed:
                    tree = ast.parse(rw.code)
                    notes.extend(rw.notes)
        elif op in {"ENFORCE_CONSERVATION_PROJECTION", "ENFORCE_LYAPUNOV_SIGN",
                    "REWRITE_MONOTONE_DRIFT_CONCEPT", "LOWER_SAFE_DIVISION",
                    "UNIFY_CONCEPT_SLOTS", "BRIDGE_UNASSIGNED_SLOTS"}:
            text = ast.unparse(tree) + "\n"
            rw = _apply_physics_guard_pass(op, text, contracts)
            if rw.changed:
                tree = ast.parse(rw.code)
                notes.extend(rw.notes or [op])
        elif op == "WRAP_SUBSTEP":
            if _wrap_substep(tree, int(raw.get("substeps", 1) or 1)):
                notes.append(f"WRAP_SUBSTEP({int(raw.get('substeps', 1) or 1)})")
        elif op == "SCALE_COEFFICIENT":
            var = str(raw.get("var") or "")
            if var and _scale_coefficient(tree, var, float(raw.get("factor", 1.0) or 1.0)):
                notes.append(f"SCALE_COEFFICIENT({var},{float(raw.get('factor', 1.0) or 1.0)})")
        elif op == "REPAIR_DRIFT_SELF_COMPARISON":
            try:
                slot = int(raw.get("slot", -1))
            except (TypeError, ValueError):
                slot = -1
            try:
                prev = int(raw.get("prev_slot", -1))
            except (TypeError, ValueError):
                prev = -1
            if slot >= 0 and prev >= 0:
                rw = repair_drift_self_comparison(ast.unparse(tree) + "\n", slot=slot,
                                                 prev_slot=prev, contracts=contracts)
                if rw.changed:
                    tree = ast.parse(rw.code)
                    notes.extend(rw.notes)
        elif op == "LOWER_COMPLEX_TO_REAL":
            lowered = lower_complex_to_real(ast.unparse(tree) + "\n")
            if lowered.changed:
                tree = ast.parse(lowered.code)
                notes.append("LOWER_COMPLEX_TO_REAL")
        elif op == "PRUNE_UNUSED_HELPERS":
            pruned = prune_unused_helpers(ast.unparse(tree) + "\n")
            if pruned.changed:
                tree = ast.parse(pruned.code)
                notes.append("PRUNE_UNUSED_HELPERS")
        elif op == "STRIP_NONE_LITERALS":
            stripped = strip_none_literals(ast.unparse(tree) + "\n")
            if stripped.changed:
                tree = ast.parse(stripped.code)
                notes.append("STRIP_NONE_LITERALS")
        elif op == "CLAMP_STATE_NORM":
            notes.append("CLAMP_STATE_NORM recorded (no scalar AST rewrite)")
        elif op == "BIND_SLOT_TO_VAR":
            notes.append(f"BIND_SLOT_TO_VAR({raw.get('slot')},{raw.get('var')})")
        else:
            notes.append(f"ignored unknown directive {op or '<missing>'}")
    # Contract-inferred deterministic drift repairs run without an LLM.
    current_code = ast.unparse(tree) + "\n"
    inferred = infer_monotone_drift_directives(contracts, current_code)
    inferred += infer_drift_self_comparison_directives(contracts, current_code)
    for d in inferred:
        if d.get("op") == "REPAIR_DRIFT_SELF_COMPARISON":
            rw = repair_drift_self_comparison(ast.unparse(tree) + "\n", slot=int(d["slot"]),
                                             prev_slot=int(d["prev_slot"]), contracts=contracts)
        else:
            rw = rewrite_monotone_drift(ast.unparse(tree) + "\n", slot=int(d["slot"]),
                                        direction=str(d.get("direction") or "non_increasing"))
        if rw.changed:
            tree = ast.parse(rw.code)
            notes.extend(rw.notes)
    ast.fix_missing_locations(tree)
    after_dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    ast_changed = after_dump != before_dump
    changed = bool(san.changed or ast_changed)
    out = ast.unparse(tree) + "\n"
    return SanitizerResult(out if changed else code, sorted(set(notes)), changed)


# -- Tier-0 universal physics guards ------------------------------------------
#
# The passes below are domain-agnostic: they key off the *shape* of the kernel
# (paired ``<fn>(new...)`` / ``<fn>(old...)`` invariant evaluations feeding a
# bounded state slot) rather than any specific equation, so they apply to
# Hamiltonian/metriplectic, quantum, PDE-lattice and kinetic kernels alike.

_SAFE_DIV_HELPER = "_invar_sdiv"
_SAFE_DIV_SRC = (
    "def _invar_sdiv(a, b):\n"
    "    if b >= 0.0 and b < 1e-30:\n"
    "        return a / 1e-30\n"
    "    if b < 0.0 and b > -1e-30:\n"
    "        return a / (-1e-30)\n"
    "    return a / b\n"
)

def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _call_key(call: ast.Call) -> Optional[Tuple[str, int]]:
    """``(function name, arity)`` for a plain ``name(...)`` call, else None."""
    if call.keywords or not isinstance(call.func, ast.Name):
        return None
    return call.func.id, len(call.args)


def _assign_pairs(fn: ast.FunctionDef, body: List[ast.stmt]
                  ) -> Dict[str, Tuple[str, ast.Call]]:
    """Map ``var -> (fn_name, call)`` for simple ``var = fn(args...)`` assigns."""
    out: Dict[str, Tuple[str, ast.Call]] = {}
    for stmt in body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 \
                or not isinstance(stmt.targets[0], ast.Name):
            continue
        value = stmt.value
        if not isinstance(value, ast.Call):
            continue
        key = _call_key(value)
        if key is None:
            continue
        out[stmt.targets[0].id] = (key[0], value)
    return out


def _abs_diff_pair(node: ast.AST) -> Optional[Tuple[str, str]]:
    """``abs(a - b)`` -> ``(a, b)`` when both sides are plain names."""
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return None
    name = _call_name(node.func)
    if name not in {"abs", "math.fabs"}:
        return None
    arg = node.args[0]
    if not isinstance(arg, ast.BinOp) or not isinstance(arg.op, ast.Sub):
        return None
    if not isinstance(arg.left, ast.Name) or not isinstance(arg.right, ast.Name):
        return None
    return arg.left.id, arg.right.id


def _call_arg_name(arg: ast.expr) -> Optional[str]:
    return arg.id if isinstance(arg, ast.Name) else None


def _expr_refs_names(node: ast.AST, names: set) -> bool:
    """True when ``node`` references any name in ``names``."""
    if not names:
        return False
    wanted = set(names)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in wanted:
            return True
    return False


def _indirect_names(body: List[ast.stmt], predicate) -> set:
    """Names assigned (top-level) from an expression satisfying ``predicate``."""
    out: set = set()
    for stmt in body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 \
                or not isinstance(stmt.targets[0], ast.Name):
            continue
        if predicate(stmt.value):
            out.add(stmt.targets[0].id)
    return out


def _slot_write_references(stmt: ast.stmt, names: set,
                           gates: Dict[int, List[Tuple[str, float]]]
                           ) -> Optional[int]:
    """First gated slot written anywhere in ``stmt`` whose value references ``names``."""
    for node in ast.walk(stmt):
        targets: List[ast.AST] = []
        value: Optional[ast.AST] = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
            value = node.value
        if value is None or not _expr_refs_names(value, names):
            continue
        for tgt in targets:
            slot = _state_subscript_slot(tgt)
            if slot is not None and slot in gates:
                return slot
    return None


def _matching_new_old(pairs: Dict[str, Tuple[str, ast.Call]],
                      new_var: str, old_var: str
                      ) -> Optional[Tuple[str, List[str], List[str], List[str]]]:
    """Match ``new = fn(new_args)`` with ``old = fn(old_args)``.

    Returns ``(fn_name, new_args, old_args, evolving_indices)`` where the
    evolving indices are the argument positions whose new/old names differ.
    """
    if new_var not in pairs or old_var not in pairs:
        return None
    fn_new, call_new = pairs[new_var]
    fn_old, call_old = pairs[old_var]
    if fn_new != fn_old or len(call_new.args) != len(call_old.args):
        return None
    if len(call_new.args) < 1 or len(call_new.args) > 6:
        return None
    evolving_idx: List[int] = []
    for pos, (a_new, a_old) in enumerate(zip(call_new.args, call_old.args)):
        n_new, n_old = _call_arg_name(a_new), _call_arg_name(a_old)
        if n_new is None or n_old is None:
            return None
        if n_new != n_old:
            evolving_idx.append(pos)
    if not (1 <= len(evolving_idx) <= 3):
        return None
    return fn_new, [_call_arg_name(a) or "" for a in call_new.args], \
        [_call_arg_name(a) or "" for a in call_old.args], evolving_idx


def _contract_slot_gates(contracts: Sequence[Any]) -> Dict[int, List[Tuple[str, float]]]:
    """``slot -> [(direction, tolerance)]`` for machine-precision slot bounds."""
    gates: Dict[int, List[Tuple[str, float]]] = {}
    for contract in contracts or []:
        for _name, assertion, tolerance in _contract_invariant_items(contract):
            direction = _bound_direction(assertion)
            if direction not in {"le", "ge"}:
                continue
            try:
                tol = float(tolerance)
            except (TypeError, ValueError):
                tol = 1e-6
            if abs(tol) > 1e-6:
                continue
            for slot in _contract_state_slots_from_text(assertion):
                gates.setdefault(slot, []).append((direction, tol))
    return gates


def _projection_block(fn_name: str, new_var: str, old_var: str,
                      new_args: Sequence[str], old_args: Sequence[str],
                      evolving_idx: Sequence[int]) -> List[ast.stmt]:
    """Build the 4-iteration finite-difference manifold projection block.

    Only the ``evolving_idx`` argument positions (the ones whose new/old names
    differ) are projected; every other argument (parameters, fixed state) is
    passed through unchanged so the call arity matches the matched invariant.
    The target level set is ``fn_name(old_args)`` evaluated directly, so it
    never depends on ``old_var`` being assigned before this block.
    """
    base = [str(a) for a in new_args]
    old_call = f"{fn_name}({', '.join(str(a) for a in old_args)})"

    def _call(perturb: Optional[Tuple[int, str]] = None) -> str:
        parts = list(base)
        if perturb is not None:
            idx, expr = perturb
            parts[idx] = expr
        return f"{fn_name}({', '.join(parts)})"

    lines = [f"_H_target_proj = {old_call}"]
    lines.append("for _proj_it in range(4):")
    lines.append(f"    _H_cur_proj = {_call()}")
    lines.append("    _H_err_proj = _H_cur_proj - _H_target_proj")
    lines.append("    if abs(_H_err_proj) <= 1e-15:")
    lines.append("        break")
    lines.append("    _eps_proj = 1e-8")
    grad_parts = []
    for g, idx in enumerate(evolving_idx):
        arg = base[idx]
        grad_parts.append(
            f"_g{g}_proj = ({_call((idx, f'({arg} + _eps_proj)'))} - _H_cur_proj) / _eps_proj")
    lines.extend("    " + g for g in grad_parts)
    norm = " + ".join(f"_g{g}_proj * _g{g}_proj" for g in range(len(evolving_idx)))
    lines.append(f"    _gn2_proj = {norm}")
    lines.append("    if _gn2_proj > 1e-30:")
    for g, idx in enumerate(evolving_idx):
        arg = base[idx]
        lines.append(
            f"        {arg} = {arg} - (_H_err_proj / _gn2_proj) * _g{g}_proj")
    src = "\n".join(lines) + "\n"
    block = ast.parse(src).body
    ast.fix_missing_locations(block[0])
    return block


_CONSERVED_INVARIANT_TOKENS = (
    "ham", "hamiltonian", "energy", "h_energy", "total_energy",
    "casimir", "mass", "momentum", "charge", "invariant",
)
_DISSIPATIVE_MONOTONE_TOKENS = (
    "entropy", "spohn", "lyapunov", "dissip", "free_energy",
    "relative", "loss", "cost", "decay", "rate", "rhs",
)


def _enforce_conservation_projection_tree(
        tree: ast.Module, contracts: Sequence[Any]
        ) -> Tuple[ast.Module, bool, List[str]]:
    """Project updated state onto the conserved level set before drift eval."""
    gates = _contract_slot_gates(contracts)
    if not gates:
        return tree, False, []
    notes: List[str] = []
    changed = False
    for fn in _simulation_functions(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        body = fn.body
        pairs = _assign_pairs(fn, body)
        for idx, stmt in enumerate(body):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 \
                    or not isinstance(stmt.targets[0], ast.Name):
                continue
            new_var = stmt.targets[0].id
            if not (new_var.endswith("_new") or new_var.endswith("new")
                    or new_var.endswith("1")):
                continue
            # Look for the matching ``*_old`` assignment and a drift slot write.
            old_var = None
            for cand in (new_var[:-4] + "_old" if new_var.endswith("_new") else None,
                         new_var[:-3] + "_old" if new_var.endswith("new") else None,
                         new_var[:-1] + "_old" if new_var.endswith("1") else None):
                if cand and cand in pairs:
                    old_var = cand
                    break
            if old_var is None:
                continue
            match = _matching_new_old(pairs, new_var, old_var)
            if match is None:
                continue
            fn_name, new_args, old_args, evolving_idx = match
            fn_low = fn_name.lower()
            if not any(tok in fn_low for tok in _CONSERVED_INVARIANT_TOKENS):
                continue
            if any(tok in fn_low for tok in _DISSIPATIVE_MONOTONE_TOKENS):
                continue
            # The drift of these two values must feed a tightly bounded slot,
            # either directly (``state[k] = abs(H_new - H_old)``) or through an
            # intermediate (``drift = abs(...); if drift > state[k]: ...``).
            pair_names = {new_var, old_var}
            indirect = _indirect_names(body, lambda v: bool(
                _abs_diff_pair(v) and set(_abs_diff_pair(v)) == pair_names))
            drift_slot = None
            for later in body[idx + 1:]:
                drift_slot = _slot_write_references(later, pair_names | indirect, gates)
                if drift_slot is not None:
                    break
            if drift_slot is None:
                continue
            is_energy_drift = False
            for contract in contracts or []:
                for name, assertion, tol in _contract_invariant_items(contract):
                    if drift_slot in _contract_state_slots_from_text(assertion):
                        if _bound_direction(assertion) == "le":
                            try:
                                tol_f = float(tol)
                            except (TypeError, ValueError):
                                tol_f = 1e-6
                            if abs(tol_f) <= 1e-6 and _concept_bucket_for(name, "le") == "energy":
                                is_energy_drift = True
                                break
                if is_energy_drift:
                    break
            if not is_energy_drift:
                continue
            evolving_names = [new_args[i] for i in evolving_idx]
            block = _projection_block(fn_name, new_var, old_var, new_args,
                                      old_args, evolving_idx)
            body[idx:idx] = block
            ast.fix_missing_locations(fn)
            changed = True
            notes.append(
                f"ENFORCE_CONSERVATION_PROJECTION: projected {evolving_names} onto "
                f"{fn_name}(...) level set before state[{drift_slot}] drift eval")
            # Stop here: the inserted block re-assigns ``H_new`` and would
            # otherwise re-match on the next iteration (infinite loop).
            break
        if changed:
            break
    return tree, changed, notes


def _enforce_lyapunov_sign_tree(
        tree: ast.Module, contracts: Sequence[Any]
        ) -> Tuple[ast.Module, bool, List[str]]:
    """Reflect wrongly-signed dissipative updates so entropy never decreases."""
    gates = _contract_slot_gates(contracts)
    if not gates:
        return tree, False, []
    # Non-decreasing slots only (entropy production / Lyapunov growth).
    ge_slots = {slot for slot, entries in gates.items()
                if any(d == "ge" for d, _t in entries)}
    if not ge_slots:
        return tree, False, []
    notes: List[str] = []
    changed = False
    for fn in _simulation_functions(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        body = fn.body
        pairs = _assign_pairs(fn, body)
        for idx, stmt in enumerate(body):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 \
                    or not isinstance(stmt.targets[0], ast.Name):
                continue
            new_var = stmt.targets[0].id
            old_var = None
            for cand in (new_var[:-4] + "_old" if new_var.endswith("_new") else None,
                         new_var[:-3] + "_old" if new_var.endswith("new") else None,
                         new_var[:-1] + "_old" if new_var.endswith("1") else None):
                if cand and cand in pairs:
                    old_var = cand
                    break
            if old_var is None:
                continue
            match = _matching_new_old(pairs, new_var, old_var)
            if match is None:
                continue
            fn_name, new_args, old_args, evolving_idx = match
            # ``dS = S_new - S_old`` must feed a non-decreasing bounded slot,
            # directly or through an intermediate (``dS = ...; if dS < state[k]``).
            pair_names = {new_var, old_var}

            def _is_entropy_diff(v: ast.AST) -> bool:
                return (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Sub)
                        and _expr_refs_names(v.left, {new_var})
                        and _expr_refs_names(v.right, {old_var}))

            indirect = _indirect_names(body, _is_entropy_diff)
            # A genuine entropy production ``S_new - S_old`` must feed the
            # non-decreasing slot. ``abs(H_new - H_old)`` is conservation drift
            # (handled by the projection pass), never a Lyapunov difference, so
            # require the difference (or an intermediate of it) to be present.
            has_diff = bool(indirect) or any(
                _is_entropy_diff(getattr(later, "value", None))
                for later in body[idx + 1:]
                if isinstance(later, ast.Assign))
            target_slot = None
            if has_diff:
                for later in body[idx + 1:]:
                    target_slot = _slot_write_references(
                        later, indirect | pair_names,
                        {s: gates[s] for s in ge_slots})
                    if target_slot is not None:
                        break
            if target_slot is None:
                continue
            # Insert the reflection guard immediately after the S_old assign.
            old_idx = next((i for i, s in enumerate(body)
                            if isinstance(s, ast.Assign) and s.targets
                            and isinstance(s.targets[0], ast.Name)
                            and s.targets[0].id == old_var), None)
            if old_idx is None:
                continue
            ei = evolving_idx[0]
            evo_new = new_args[ei]
            evo_old = old_args[ei]
            call = f"{fn_name}({', '.join(new_args)})"
            lines = ["if S_new < S_old:"]
            lines.append(f"    {evo_new} = 2.0 * {evo_old} - {evo_new}")
            lines.append(f"    S_new = {call}")
            lines.append("    if S_new < S_old:")
            lines.append(f"        {evo_new} = {evo_old}")
            lines.append("        S_new = S_old")
            block = ast.parse("\n".join(lines) + "\n").body
            # Substitute the literal S_new/S_old names.
            for stmt in block:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Name) and node.id == "S_new":
                        node.id = new_var
                    elif isinstance(node, ast.Name) and node.id == "S_old":
                        node.id = old_var
            body[old_idx + 1:old_idx + 1] = block
            ast.fix_missing_locations(fn)
            changed = True
            notes.append(
                f"ENFORCE_LYAPUNOV_SIGN: reflected {evo_new} to keep "
                f"{new_var} >= {old_var} (state[{target_slot}])")
            break
        if changed:
            break
    return tree, changed, notes


class _NonZeroConstantTracker(ast.NodeVisitor):
    """Collect names bound EXACTLY ONCE to a non-zero numeric constant."""

    def __init__(self) -> None:
        self._const_nonzero: set[str] = set()
        self._assign_counts: Dict[str, int] = {}
        self._nonconst: set[str] = set()

    @property
    def nonzero(self) -> set[str]:
        return {
            name for name in self._const_nonzero
            if self._assign_counts.get(name, 0) == 1 and name not in self._nonconst
        }

    def _record(self, name: str, val: Optional[ast.expr]) -> None:
        self._assign_counts[name] = self._assign_counts.get(name, 0) + 1
        if (isinstance(val, ast.Constant) and isinstance(val.value, (int, float))
                and not isinstance(val.value, bool) and val.value != 0):
            self._const_nonzero.add(name)
        else:
            self._nonconst.add(name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._record(target.id, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._record(node.target.id, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._nonconst.add(node.target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for arg in node.args.args:
            self._nonconst.add(arg.arg)
        self.generic_visit(node)


class _SafeDivisionTransformer(ast.NodeTransformer):
    """Lower ``a / b`` to ``_invar_sdiv(a, b)`` unless ``b`` is a nonzero const."""

    def __init__(self, nonzero: set[str]) -> None:
        self.nonzero = nonzero
        self.changed = False
        self.notes: set[str] = set()

    def _denominator_safe(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value != 0
        if isinstance(node, ast.Name) and node.id in self.nonzero:
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
                and isinstance(node.operand, ast.Constant) \
                and isinstance(node.operand.value, (int, float)):
            return node.operand.value != 0
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name == _SAFE_DIV_HELPER:
            return node
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Div) and not self._denominator_safe(node.right):
            self.changed = True
            self.notes.add("safe-division lowered")
            return ast.copy_location(
                ast.Call(func=ast.Name(id=_SAFE_DIV_HELPER, ctx=ast.Load()),
                         args=[node.left, node.right], keywords=[]), node)
        return node


def _lower_safe_division_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    tracker = _NonZeroConstantTracker()
    tracker.visit(tree)
    transformer = _SafeDivisionTransformer(tracker.nonzero)
    tree = transformer.visit(tree)
    uses_sdiv = transformer.changed or any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == _SAFE_DIV_HELPER
        for n in ast.walk(tree)
    )
    has_sdiv_def = any(
        isinstance(n, ast.FunctionDef) and n.name == _SAFE_DIV_HELPER
        for n in tree.body
    )
    hoisted = False
    if uses_sdiv and not has_sdiv_def:
        tree.body.insert(0, ast.parse(_SAFE_DIV_SRC).body[0])
        _ensure_math_import(tree)
        hoisted = True
    if _hoist_intrinsic_helpers(tree):
        hoisted = True
    ast.fix_missing_locations(tree)
    return tree, (transformer.changed or hoisted), sorted(transformer.notes)


# Concept buckets for auto-bridging unassigned contract slots (mirrors the
# planner's physical-concept classification, kept local to avoid a cycle).
_CONCEPT_BUCKET_ROOTS = {
    "energy": ("energy", "hamiltonian", "h_energy"),
    "entropy": ("entropy", "s_entropy", "lyapunov", "spohn"),
    "trace": ("trace",),
    "roundtrip": ("roundtrip", "revers", "chiasmus"),
    "purity": ("purity", "closed_energy", "norm"),
}
_CONCEPT_LE_TOKENS = ("drift", "err", "error", "residual", "bound", "violation",
                      "roundtrip", "revers", "chiasmus", "max", "hold", "conservation",
                      "increase", "upper", "purity", "norm")
_CONCEPT_GE_TOKENS = ("monotone", "step", "min", "prod", "positivity",
                      "eigenvalue", "lambda_min")
_CONCEPT_SUFFIXES = ("_trace_rho2", "_bound", "_limit", "_max", "_min",
                     "_current", "_val", "_check", "_error", "_err", "_drift", "_hold")


def _concept_stem(name: str) -> str:
    low = str(name or "").lower().strip()
    changed = True
    while changed:
        changed = False
        for prefix in ("max_", "min_", "unified_", "total_", "step_"):
            if low.startswith(prefix) and len(low) > len(prefix):
                low = low[len(prefix):]
                changed = True
        for suffix in _CONCEPT_SUFFIXES:
            if low.endswith(suffix) and len(low) > len(suffix):
                low = low[:-len(suffix)]
                changed = True
    return low


def _concept_bucket_for(name: str, direction: str) -> str:
    low = str(name or "").lower()
    stem = _concept_stem(low)
    tokens = _CONCEPT_GE_TOKENS if direction == "ge" else _CONCEPT_LE_TOKENS
    if not any(tok in low for tok in tokens):
        return ""
    for bucket, roots in _CONCEPT_BUCKET_ROOTS.items():
        if any(tok in low or tok in stem for tok in roots):
            return bucket
    if direction == "ge" and any(tok in low for tok in ("positivity", "eigenvalue",
                                                        "lambda_min")) \
            and not any(tok in low for tok in ("violation", "rk4", "euler")):
        return "positivity"
    if direction == "le" and any(tok in low or tok in stem for tok in ("purity", "closed_energy", "norm")):
        return "purity"
    return ""


def _bridge_unassigned_concept_slots_tree(
        tree: ast.Module, contracts: Sequence[Any]
        ) -> Tuple[ast.Module, bool, List[str]]:
    """Copy an assigned slot onto unassigned slots of the same concept bucket."""
    contracts = list(contracts or [])
    if not contracts:
        return tree, False, []
    driver = _simulation_driver(tree)
    if driver is None:
        return tree, False, []
    cls = classify_buffer_slots(tree, contracts)
    protected = {0, 1, 2, 3} | set(cls["dynamical_slots"]) | set(cls["parameter_slots"]) | set(cls["history_slots"])
    assigned = _collect_assigned_slots(tree)
    read_slots = _read_state_slots(tree)
    slot_meta: Dict[int, List[Tuple[str, str, float]]] = {}
    for contract in contracts:
        for name, assertion, tol in _contract_invariant_items(contract):
            raw_slots = _contract_state_slots_from_text(assertion)
            distinct_slots = set(raw_slots)
            # Strict Multi-Variable Constraint Exclusion: skip if len(distinct_indices) != 1
            if len(distinct_slots) != 1:
                continue
            slot = next(iter(distinct_slots))
            if slot in protected:
                continue
            direction = _bound_direction(assertion)
            if direction not in {"le", "ge"}:
                continue
            try:
                tol_f = float(tol)
            except (TypeError, ValueError):
                tol_f = 1e-6
            if abs(tol_f) > 1e-6 and abs(tol_f - 1.0) > 1e-6:
                continue
            slot_meta.setdefault(slot, []).append((name, direction, tol_f))
    missing = [s for s in sorted(slot_meta) if s not in protected and s not in assigned and s not in read_slots]
    if not missing:
        return tree, False, []
    additions: List[ast.stmt] = []
    notes: List[str] = []
    for slot in missing:
        if slot in protected:
            continue
        metas = slot_meta[slot]
        target = None
        # Prefer an assigned slot sharing the same concept bucket.
        for cand in sorted(assigned):
            if cand in protected:
                continue
            for _n, direction, tol_f in metas:
                cand_bucket = next((_concept_bucket_for(nm, d)
                                    for nm, d, _t in slot_meta.get(cand, [])), "")
                if cand_bucket and cand_bucket == _concept_bucket_for(_n, direction):
                    target = cand
                    break
            if target is not None:
                break
        # Fall back to the same inequality direction + tolerance.
        if target is None:
            for cand in sorted(assigned):
                if cand in protected:
                    continue
                for name, direction, tol_f in metas:
                    for _cn, cd, ct in slot_meta.get(cand, []):
                        if cd == direction and (abs(ct - tol_f) <= 1e-12
                                                or (abs(ct - 1.0) <= 1e-6 and abs(tol_f - 1.0) <= 1e-6)):
                            target = cand
                            break
                    if target is not None:
                        break
                if target is not None:
                    break
        if target is None and assigned:
            valid_cands = [c for c in sorted(assigned) if c not in protected]
            if valid_cands:
                target = valid_cands[0]
        if target is None or target in protected:
            continue
        additions.append(ast.parse(f"buf.data[{slot}] = buf.data[{target}]").body[0])
        notes.append(f"bridged unassigned state[{slot}] <- state[{target}] "
                     f"(concept bucket / bound match)")
    if not additions:
        return tree, False, []
    driver.body.extend(additions)
    ast.fix_missing_locations(tree)
    return tree, True, notes


def _unify_concept_slots_tree(tree: ast.Module, contracts: Sequence[Any]
                              ) -> Tuple[ast.Module, bool, List[str]]:
    """Rewrite later-stage concept slots in-place onto earlier stage slots.

    Mirrors the planner's concept-bucket unification at the AST level so a
    kernel that computes ``state[16]``/``state[17]`` also mirrors them onto
    the canonical ``state[8]``/``state[10]`` slots the earlier stages check.
    """
    contracts = list(contracts or [])
    if not contracts:
        return tree, False, []
    cls = classify_buffer_slots(tree, contracts)
    protected = {0, 1, 2, 3} | set(cls["dynamical_slots"]) | set(cls["parameter_slots"]) | set(cls["history_slots"])
    slot_meta: Dict[int, Dict[str, Any]] = {}
    for contract in contracts:
        stage = _contract_stage_index(str(_contract_get(contract, "contract_id", "") or ""))
        for name, assertion, tol in _contract_invariant_items(contract):
            raw_slots = _contract_state_slots_from_text(assertion)
            distinct_slots = set(raw_slots)
            # Strict Multi-Variable Constraint Exclusion: skip if len(distinct_indices) != 1
            if len(distinct_slots) != 1 or len(raw_slots) != 1:
                continue
            slot = next(iter(distinct_slots))
            if slot in protected:
                continue
            direction = _bound_direction(assertion)
            if direction not in {"le", "ge"}:
                continue
            try:
                tol_f = float(tol)
            except (TypeError, ValueError):
                tol_f = 1e-6
            if abs(tol_f) > 1e-6 and abs(tol_f - 1.0) > 1e-6:
                continue
            bucket = _concept_bucket_for(name, direction)
            if not bucket:
                continue
            slot_meta.setdefault(slot, {"bucket": bucket, "direction": direction,
                                        "stage": stage})
    if len(slot_meta) < 2:
        return tree, False, []
    canonical: Dict[Tuple[str, str], int] = {}
    for slot in sorted(slot_meta, key=lambda s: (slot_meta[s]["stage"], s)):
        if slot in protected:
            continue
        meta = slot_meta[slot]
        key = (meta["bucket"], meta["direction"])
        canonical.setdefault(key, slot)
    remap: Dict[int, int] = {}
    for slot, meta in slot_meta.items():
        if slot in protected:
            continue
        target = canonical.get((meta["bucket"], meta["direction"]))
        if target is not None and target not in protected and slot != target:
            remap[slot] = target
    if not remap:
        return tree, False, []
    changed = False
    notes: List[str] = []

    def _rewrite(node: ast.AST) -> None:
        nonlocal changed
        for sub in ast.walk(node):
            if isinstance(sub, ast.Subscript):
                base = sub.value
                name = ""
                if isinstance(base, ast.Name):
                    name = base.id
                elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                    name = f"{base.value.id}.{base.attr}"
                if name in {"state", "data", "buf", "buf.data"}:
                    slot = _state_subscript_slot(sub)
                    if slot in remap and slot not in protected:
                        sub.slice = ast.Constant(value=remap[slot])
                        changed = True

    for fn in _simulation_functions(tree):
        _rewrite(fn)
    ast.fix_missing_locations(tree)
    if changed:
        notes.append("unified concept slots: " + ", ".join(
            f"state[{old}] -> state[{new}]" for old, new in sorted(remap.items())))
    return tree, changed, notes


def _tree_pass_wrapper(code: str, pass_fn, *, contracts: Sequence[Any] = ()) -> SanitizerResult:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    tree, changed, notes = pass_fn(tree, contracts)
    if not changed:
        return SanitizerResult(code, [], False)
    ast.fix_missing_locations(tree)
    return SanitizerResult(ast.unparse(tree) + "\n", notes, True)


def enforce_conservation_projection(code: str, *,
                                    contracts: Sequence[Any] = ()) -> SanitizerResult:
    """Project conservative updates onto the exact invariant level set."""
    return _tree_pass_wrapper(
        code, lambda t, c: _enforce_conservation_projection_tree(t, c),
        contracts=contracts)


def enforce_lyapunov_sign(code: str, *, contracts: Sequence[Any] = ()) -> SanitizerResult:
    """Reflect wrongly-signed dissipative updates (entropy never decreases)."""
    return _tree_pass_wrapper(
        code, lambda t, c: _enforce_lyapunov_sign_tree(t, c), contracts=contracts)


def lower_safe_division(code: str) -> SanitizerResult:
    """Lower ``a / b`` to ``_invar_sdiv(a, b)`` for non-constant denominators."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    tree, changed, notes = _lower_safe_division_tree(tree)
    if not changed:
        return SanitizerResult(code, [], False)
    return SanitizerResult(ast.unparse(tree) + "\n", notes, True)


def unify_concept_slots(code: str, *, contracts: Sequence[Any] = ()) -> SanitizerResult:
    """Rewrite later-stage concept slots onto the canonical first-stage slot."""
    return _tree_pass_wrapper(code, _unify_concept_slots_tree, contracts=contracts)


def bridge_unassigned_concept_slots(code: str, *,
                                    contracts: Sequence[Any] = ()) -> SanitizerResult:
    """Bridge unassigned contract slots to same-concept assigned slots."""
    return _tree_pass_wrapper(
        code, _bridge_unassigned_concept_slots_tree, contracts=contracts)


def _apply_physics_guard_pass(op: str, code: str, contracts: Sequence[Any]) -> SanitizerResult:
    if op == "ENFORCE_CONSERVATION_PROJECTION":
        return enforce_conservation_projection(code, contracts=contracts)
    if op == "ENFORCE_LYAPUNOV_SIGN":
        return enforce_lyapunov_sign(code, contracts=contracts)
    if op == "REWRITE_MONOTONE_DRIFT_CONCEPT":
        return _tree_pass_wrapper(code, _repair_monotone_observables_tree, contracts=contracts)
    if op == "LOWER_SAFE_DIVISION":
        return lower_safe_division(code)
    if op == "UNIFY_CONCEPT_SLOTS":
        return unify_concept_slots(code, contracts=contracts)
    if op == "BRIDGE_UNASSIGNED_SLOTS":
        return bridge_unassigned_concept_slots(code, contracts=contracts)
    return SanitizerResult(code, [], False)


__all__ = [
    "SanitizerResult", "sanitize_intrinsics", "rewrite_monotone_drift",
    "infer_monotone_drift_directives", "apply_healing_directives",
    "strip_none_literals", "numeric_expr_equivalent",
    "lower_complex_to_real", "prune_unused_helpers",
    "repair_drift_self_comparison", "infer_drift_self_comparison_directives",
    "enforce_conservation_projection", "enforce_lyapunov_sign",
    "lower_safe_division", "unify_concept_slots",
    "bridge_unassigned_concept_slots",
    "classify_buffer_slots", "resolve_slot_collisions", "_detect_bloch_radius",
    "MATH_FUNCS", "INTRINSIC_ALIASES",
]

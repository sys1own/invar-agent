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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MATH_FUNCS = {
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",
    "cosh", "tanh", "exp", "log", "log10", "log2", "log1p", "expm1", "ceil",
    "floor", "pow", "fabs", "fmod", "hypot", "copysign", "trunc", "isfinite",
    "isnan", "isinf", "radians", "degrees",
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


def _dead_assignments_in_function(fn: ast.FunctionDef) -> Tuple[ast.FunctionDef, bool]:
    loaded = {n.id for n in ast.walk(fn)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    class _Dead(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                pure = not any(isinstance(n, ast.Call) for n in ast.walk(node.value))
                if name not in loaded and pure:
                    self.changed = True
                    return None  # type: ignore[return-value]
            return node

    transformer = _Dead()
    new_fn = transformer.visit(fn)
    return new_fn, transformer.changed


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
        roots = [e for e in entrypoints if e in names]
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
        "clamp": "def clamp(x, lo, hi):\n    return min(max(x, lo), hi)\n",
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


def _insert_intrinsic_helpers(tree: ast.Module, names: Iterable[str]) -> bool:
    defs = _intrinsic_defs(names)
    if not defs:
        return False
    idx = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            idx = i + 1
    tree.body[idx:idx] = defs
    ast.fix_missing_locations(tree)
    return True


def sanitize_intrinsics(code: str, *, guard_domains: bool = True,
                        eps: float = 1e-300) -> SanitizerResult:
    """Rewrite scalar intrinsic/math-namespace hazards before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    original = ast.dump(tree, annotate_fields=False, include_attributes=False)
    tree, none_changed, none_notes = _strip_none_literals_tree(tree)
    tree, complex_changed, complex_notes = _lower_complex_tree(tree)
    transformer = _IntrinsicTransformer()
    transformer.defined = _defined_names(tree)
    transformer.guard_eps = float(eps)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    changed = ast.dump(tree, annotate_fields=False, include_attributes=False) != original
    notes = sorted(set(none_notes) | set(complex_notes) | set(transformer.notes))
    if none_changed or complex_changed:
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
    return SanitizerResult(out, notes, changed)


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
    changed = transformer.changed or ast.dump(tree, annotate_fields=False,
                                              include_attributes=False) != original
    notes = [f"rewrote monotone drift state[{slot}] ({direction})"] if transformer.changed else []
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
                             *, contracts: Sequence[Any] = ()) -> SanitizerResult:
    """Apply LLM-emitted structured directives deterministically."""
    notes: List[str] = []
    current = code
    # Always run the intrinsic/domain sanitizer first (Tier-0, zero tokens).
    san = sanitize_intrinsics(current)
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
            if name in INTRINSIC_ALIASES:
                patched = _insert_intrinsic_helpers(tree, [name])
                if patched:
                    notes.append(f"INLINE_INTRINSIC({name})")
        elif op == "GUARD_MATH_DOMAINS":
            eps = float(raw.get("eps", 1e-300) or 1e-300)
            guarded = _DomainGuardTransformer(eps=eps).visit(tree)
            if guarded is not tree:
                notes.append("GUARD_MATH_DOMAINS")
            tree = guarded
        elif op == "REWRITE_MONOTONE_DRIFT":
            slot = int(raw.get("slot", -1))
            if slot >= 0:
                # Re-render through the dedicated rewriter for precise notes.
                rw = rewrite_monotone_drift(ast.unparse(tree) + "\n", slot=slot,
                                            direction=str(raw.get("direction") or "non_increasing"),
                                            before_var=raw.get("before_var"),
                                            after_var=raw.get("after_var"))
                if rw.changed:
                    tree = ast.parse(rw.code)
                    notes.extend(rw.notes)
        elif op == "WRAP_SUBSTEP":
            if _wrap_substep(tree, int(raw.get("substeps", 1) or 1)):
                notes.append(f"WRAP_SUBSTEP({int(raw.get('substeps', 1) or 1)})")
        elif op == "SCALE_COEFFICIENT":
            var = str(raw.get("var") or "")
            if var and _scale_coefficient(tree, var, float(raw.get("factor", 1.0) or 1.0)):
                notes.append(f"SCALE_COEFFICIENT({var},{float(raw.get('factor', 1.0) or 1.0)})")
        elif op == "REPAIR_DRIFT_SELF_COMPARISON":
            slot = int(raw.get("slot", -1))
            prev = int(raw.get("prev_slot", -1))
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


__all__ = [
    "SanitizerResult", "sanitize_intrinsics", "rewrite_monotone_drift",
    "infer_monotone_drift_directives", "apply_healing_directives",
    "strip_none_literals", "numeric_expr_equivalent",
    "lower_complex_to_real", "prune_unused_helpers",
    "repair_drift_self_comparison", "infer_drift_self_comparison_directives",
    "MATH_FUNCS", "INTRINSIC_ALIASES",
]

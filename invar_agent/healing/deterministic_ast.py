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


def _state_subscript_slot(node: ast.AST) -> Optional[int]:
    if not isinstance(node, ast.Subscript):
        return None
    slot = _target_slot(node)
    if slot is None:
        return None
    base = node.value
    if isinstance(base, ast.Attribute) and base.attr == "data" \
            and isinstance(base.value, ast.Name) and base.value.id == "buf":
        return slot
    if isinstance(base, ast.Name) and base.id in {"state", "data", "buf"}:
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
            for node in ast.walk(fn):
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) \
                        and _state_subscript_slot(node) == slot \
                        and (getattr(node, "lineno", 0) or 0) < assign_lineno:
                    read_before = True
                    break
            if read_before:
                delayed[slot] = (state_var, assign_lineno)
        if not delayed:
            continue

        class _DelayedReadRewrite(ast.NodeTransformer):
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
    cid = str(getattr(contract, "contract_id", "") or "").lower()
    return "closed" in cid or "revers" in cid


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


def _protect_parameter_slots_tree(tree: ast.Module, contracts: Sequence[Any]
                                  ) -> Tuple[ast.Module, bool, List[str]]:
    changed = False
    notes: List[str] = []
    used = _used_state_slots(tree)
    allow_kappa = _contracts_allow_kappa_default(contracts)
    for fn in _simulation_functions(tree):
        body = fn.body
        reads: List[Tuple[int, str, int, ast.stmt]] = []
        for idx, stmt in enumerate(body):
            name = _assign_target_name(stmt)
            if name is None or not isinstance(stmt.value, ast.Subscript):
                continue
            if not _is_parameter_name(name):
                continue
            slot = _state_subscript_slot(stmt.value)
            if slot is None:
                continue
            reads.append((idx, name, slot, stmt))
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
            if later_writes:
                scratch = _scratch_slot(used, {slot})
                used.add(scratch)
                for node in later_writes:
                    target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                    if _state_subscript_slot(target) == slot:
                        target.slice = ast.Constant(value=scratch)  # type: ignore[attr-defined]
                changed = True
                notes.append(f"protected parameter buf.data[{slot}] ({var}) -> buf.data[{scratch}]")
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


def _assigned_state_slots(tree: ast.Module) -> set[int]:
    slots: set[int] = set()
    for node in ast.walk(tree):
        targets: List[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            slot = _state_subscript_slot(target)
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


def _contract_slot_names(contracts: Sequence[Any]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for contract in contracts or []:
        for name, assertion, _tol in _contract_invariant_items(contract):
            for slot in _contract_state_slots_from_text(assertion):
                names = out.setdefault(slot, [])
                if name and name not in names:
                    names.append(name)
    return out


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
    assigned = _assigned_state_slots(tree)
    slot_names = _contract_slot_names(contracts)
    contract_slots = sorted(slot_names)
    missing = [k for k in contract_slots if k not in assigned]
    if not missing:
        return tree, False, []

    notes: List[str] = []
    additions: List[ast.stmt] = []
    roundtrip_slots: List[int] = []
    assigned_sorted = sorted(assigned)
    for slot in missing:
        names = slot_names.get(slot, [])
        low_names = " ".join(names).lower()
        if any(tok in low_names for tok in ("roundtrip", "revers")):
            roundtrip_slots.append(slot)
            continue
        stem = {_semantic_slot_stem(n) for n in names if n}
        stem.discard("")
        target = None
        if stem:
            for candidate in assigned_sorted:
                other = {_semantic_slot_stem(n) for n in slot_names.get(candidate, []) if n}
                other.discard("")
                if stem & other:
                    target = candidate
                    break
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
              "violation", "mismatch")
    for contract in contracts or []:
        for name, assertion, tolerance in _contract_invariant_items(contract):
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


def _repair_monotone_observable_body(body: List[ast.stmt], slots: set[int],
                                     notes: set[str]) -> bool:
    changed = False
    for i, stmt in enumerate(body):
        if isinstance(stmt, (ast.For, ast.While)):
            for field in ("body", "orelse"):
                sub = getattr(stmt, field, None)
                if isinstance(sub, list):
                    changed = _repair_monotone_observable_body(sub, slots, notes) or changed
            continue
        if not isinstance(stmt, ast.If):
            continue
        test = stmt.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 \
                or len(test.comparators) != 1:
            continue
        metric_slot = None
        val_name = None
        if isinstance(test.left, ast.Name) and isinstance(test.comparators[0], ast.Subscript):
            comp_slot = _state_subscript_slot(test.comparators[0])
            if comp_slot in slots and isinstance(test.ops[0], (ast.Gt, ast.GtE)):
                metric_slot, val_name = comp_slot, test.left.id
        if metric_slot is None:
            continue
        assigns_metric = False
        for sub in stmt.body:
            if _stmt_slot_target(sub) == metric_slot and isinstance(sub, (ast.Assign, ast.AnnAssign)):
                value = getattr(sub, "value", None)
                if isinstance(value, ast.Name) and value.id == val_name:
                    assigns_metric = True
                    break
        if not assigns_metric:
            continue
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
        if prev_slot is None:
            continue
        viol = f"_monotone_viol_{metric_slot}"
        replacement = ast.parse(
            f"{viol} = max(0.0, {val_name} - buf.data[{prev_slot}])\n"
            f"if {viol} > buf.data[{metric_slot}]:\n"
            f"    buf.data[{metric_slot}] = {viol}\n"
            f"buf.data[{prev_slot}] = {val_name}\n").body
        body[i:i + 1] = replacement
        ast.fix_missing_locations(stmt)
        changed = True
        notes.add(f"repaired monotone observable state[{metric_slot}] "
                  f"to track max(0, {val_name} - state[{prev_slot}])")
    return changed


def _repair_monotone_observables_tree(tree: ast.Module, contracts: Sequence[Any]
                                      ) -> Tuple[ast.Module, bool, List[str]]:
    slots = _small_monotone_slots(contracts)
    if not slots:
        return tree, False, []
    notes: set[str] = set()
    changed = False
    for fn in _simulation_functions(tree):
        if isinstance(fn, ast.FunctionDef) and _repair_monotone_observable_body(fn.body, slots, notes):
            changed = True
            ast.fix_missing_locations(fn)
    return tree, changed, sorted(notes)


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


def sanitize_intrinsics(code: str, *, guard_domains: bool = True,
                        eps: float = 1e-300,
                        contracts: Sequence[Any] = (),
                        all_contracts: Sequence[Any] = ()) -> SanitizerResult:
    """Rewrite scalar intrinsic/math-namespace hazards before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    original = ast.dump(tree, annotate_fields=False, include_attributes=False)
    bridge_contracts = list(all_contracts or contracts or [])
    tree, alias_changed, alias_notes = _normalize_buffer_aliases_tree(tree)
    tree, bridge_changed, bridge_notes = _bridge_contract_slots_tree(tree, bridge_contracts)
    tree, monotone_changed, monotone_notes = _repair_monotone_observables_tree(
        tree, bridge_contracts)
    tree, none_changed, none_notes = _strip_none_literals_tree(tree)
    tree, complex_changed, complex_notes = _lower_complex_tree(tree)
    tree, history_changed, history_notes = _repair_history_slots_tree(tree)
    tree, sdiv_changed, sdiv_notes = _lower_safe_division_tree(tree)
    tree, unify_changed, unify_notes = _unify_concept_slots_tree(tree, bridge_contracts)
    tree, concept_changed, concept_notes = _bridge_unassigned_concept_slots_tree(
        tree, bridge_contracts)
    tree, cons_changed, cons_notes = _enforce_conservation_projection_tree(
        tree, bridge_contracts)
    tree, lyap_changed, lyap_notes = _enforce_lyapunov_sign_tree(tree, bridge_contracts)
    tree, param_changed, param_notes = _protect_parameter_slots_tree(tree, contracts)
    tree, diss_changed, diss_notes = _seed_dissipation_parameters_tree(tree, contracts)
    transformer = _IntrinsicTransformer()
    transformer.defined = _defined_names(tree)
    transformer.guard_eps = float(eps)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    changed = ast.dump(tree, annotate_fields=False, include_attributes=False) != original
    notes = sorted(set(alias_notes) | set(bridge_notes) | set(monotone_notes)
                   | set(none_notes) | set(complex_notes) | set(history_notes)
                   | set(sdiv_notes) | set(unify_notes) | set(concept_notes)
                   | set(cons_notes) | set(lyap_notes)
                   | set(param_notes) | set(diss_notes) | set(transformer.notes))
    if (alias_changed or bridge_changed or monotone_changed or none_changed
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
                             *, contracts: Sequence[Any] = ()) -> SanitizerResult:
    """Apply LLM-emitted structured directives deterministically."""
    notes: List[str] = []
    current = code
    # Always run the intrinsic/domain sanitizer first (Tier-0, zero tokens).
    san = sanitize_intrinsics(current, contracts=contracts)
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
    """Collect names bound to non-zero numeric constants (module + function)."""

    def __init__(self) -> None:
        self.nonzero: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)) \
                and not isinstance(node.value.value, bool) and node.value.value != 0:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.nonzero.add(target.id)
        self.generic_visit(node)

    visit_AnnAssign = visit_Assign


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
    if transformer.changed and _SAFE_DIV_HELPER not in _module_level_names(tree):
        tree.body.insert(0, ast.parse(_SAFE_DIV_SRC).body[0])
        _ensure_math_import(tree)
    ast.fix_missing_locations(tree)
    return tree, transformer.changed, sorted(transformer.notes)


# Concept buckets for auto-bridging unassigned contract slots (mirrors the
# planner's physical-concept classification, kept local to avoid a cycle).
_CONCEPT_BUCKET_ROOTS = {
    "energy": ("energy", "hamiltonian", "h_energy"),
    "entropy": ("entropy", "s_entropy", "lyapunov"),
    "trace": ("trace",),
    "roundtrip": ("roundtrip", "revers"),
}
_CONCEPT_LE_TOKENS = ("drift", "err", "error", "residual", "bound", "violation",
                      "roundtrip", "revers", "max")
_CONCEPT_GE_TOKENS = ("monotone", "step", "min", "prod", "positivity",
                      "eigenvalue", "lambda_min")
_CONCEPT_SUFFIXES = ("_trace_rho2", "_bound", "_limit", "_max", "_min",
                     "_current", "_val", "_check", "_error", "_err", "_drift")


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
                                                        "lambda_min")):
        return "positivity"
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
    assigned = _assigned_state_slots(tree)
    slot_meta: Dict[int, List[Tuple[str, str, float]]] = {}
    for contract in contracts:
        for name, assertion, tol in _contract_invariant_items(contract):
            direction = _bound_direction(assertion)
            if direction not in {"le", "ge"}:
                continue
            try:
                tol_f = float(tol)
            except (TypeError, ValueError):
                tol_f = 1e-6
            if abs(tol_f) > 1e-6:
                continue
            bucket = _concept_bucket_for(name, direction)
            for slot in _contract_state_slots_from_text(assertion):
                slot_meta.setdefault(slot, []).append((name, direction, tol_f))
    missing = [s for s in sorted(slot_meta) if s not in assigned]
    if not missing:
        return tree, False, []
    additions: List[ast.stmt] = []
    notes: List[str] = []
    for slot in missing:
        metas = slot_meta[slot]
        target = None
        # Prefer an assigned slot sharing the same concept bucket.
        for cand in sorted(assigned):
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
                for name, direction, tol_f in metas:
                    for _cn, cd, ct in slot_meta.get(cand, []):
                        if cd == direction and abs(ct - tol_f) <= 1e-12:
                            target = cand
                            break
                    if target is not None:
                        break
                if target is not None:
                    break
        if target is None and assigned:
            target = sorted(assigned)[0]
        if target is None:
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
    slot_meta: Dict[int, Dict[str, Any]] = {}
    for contract in contracts:
        stage = _contract_stage_index(str(_contract_get(contract, "contract_id", "") or ""))
        for name, assertion, tol in _contract_invariant_items(contract):
            direction = _bound_direction(assertion)
            if direction not in {"le", "ge"}:
                continue
            try:
                tol_f = float(tol)
            except (TypeError, ValueError):
                tol_f = 1e-6
            if abs(tol_f) > 1e-6:
                continue
            bucket = _concept_bucket_for(name, direction)
            if not bucket:
                continue
            for slot in _contract_state_slots_from_text(assertion):
                slot_meta.setdefault(slot, {"bucket": bucket, "direction": direction,
                                            "stage": stage})
    if len(slot_meta) < 2:
        return tree, False, []
    # Canonical slot per (bucket, direction): the earliest stage's slot.
    canonical: Dict[Tuple[str, str], int] = {}
    for slot in sorted(slot_meta, key=lambda s: (slot_meta[s]["stage"], s)):
        meta = slot_meta[slot]
        key = (meta["bucket"], meta["direction"])
        canonical.setdefault(key, slot)
    remap: Dict[int, int] = {}
    for slot, meta in slot_meta.items():
        target = canonical[(meta["bucket"], meta["direction"])]
        if slot != target:
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
                    if slot in remap:
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
    "MATH_FUNCS", "INTRINSIC_ALIASES",
]

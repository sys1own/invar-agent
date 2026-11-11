"""Normalize Python source into the UAST dialect used by the research agent.

Dialect node types: module, function_declaration, binding, reference, literal,
binop, unaryop, call, compare, if, loop_for, loop_while, attribute, subscript,
list, dict, set.
"""

from __future__ import annotations

import ast
from typing import List, Optional

IO_MODULES = {
    "requests", "urllib", "socket", "os", "subprocess", "pathlib", "io",
    "ftplib", "http", "sys",
}
IO_NAMES = {"open", "print", "input", "exec", "eval", "compile", "exit", "quit", "__import__"}
SAFE_STD_MODULES = {"io", "sys", "time", "math"}
SAFE_BUILTINS = {"print", "len", "range", "abs", "min", "max", "sum", "enumerate"}
MATH_CONSTANTS = {"pi": 3.141592653589793, "e": 2.718281828459045,
                  "tau": 6.283185307179586, "inf": float("inf")}
MATH_FUNCS = {
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",
    "cosh", "tanh", "exp", "log", "log10", "log2", "log1p", "expm1", "ceil",
    "floor", "pow", "fabs", "fmod", "hypot", "copysign", "trunc", "isfinite",
    "isnan", "isinf", "radians", "degrees",
}
INTRINSIC_ALIASES = {"clamp": "clamp", "clip": "clamp", "sign": "sign",
                     "sgn": "sign", "sq": "sq", "lerp": "lerp",
                     "safe_div": "safe_div"}
NP_MATH_FUNCS = MATH_FUNCS | {"abs", "clip", "sign", "sgn"}


class UnsupportedError(ValueError):
    def __init__(self, feature: str, node=None):
        super().__init__(f"unsupported construct: {feature}")
        self.feature = feature
        self.node = node


def python_source_to_uast(source: str) -> dict:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for inner in ast.walk(node):
                if inner is not node and isinstance(
                        inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    raise UnsupportedError("nested closures", node=inner)
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsupportedError("global/nonlocal", node=node)
    _lower_expr.local_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    children: List[dict] = []
    for stmt in tree.body:
        node = _lower_stmt(stmt)
        if node is not None:
            children.append(node)
    return {"type": "module", "children": children}


def _is_io_call(expr: ast.Call) -> bool:
    if isinstance(expr.func, ast.Name) and expr.func.id in IO_NAMES:
        return True
    if (isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id in IO_MODULES):
        return True
    return False


def _lower_stmt(stmt: ast.stmt) -> Optional[dict]:
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return None
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        raise UnsupportedError("with statements", node=stmt)
    if isinstance(stmt, (ast.Try, getattr(ast, "TryStar", ast.Try))):
        raise UnsupportedError("try/except", node=stmt)
    if isinstance(stmt, ast.ClassDef):
        raise UnsupportedError("classes/dataclasses", node=stmt)
    if isinstance(stmt, (ast.Pass, ast.Break, ast.Continue)):
        return {"type": type(stmt).__name__.lower()}
    if isinstance(stmt, (ast.AsyncFor, ast.AsyncFunctionDef, ast.Match)):
        raise UnsupportedError(type(stmt).__name__, node=stmt)
    if isinstance(stmt, ast.FunctionDef):
        params = [a.arg for a in stmt.args.args]
        body = [n for n in (_lower_stmt(s) for s in stmt.body) if n is not None]
        return {"type": "function_declaration", "name": stmt.name,
                "params": params, "body": body}
    if isinstance(stmt, ast.Assign) and stmt.targets:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return {"type": "binding", "name": target.id,
                    "value": _lower_expr(stmt.value)}
        if isinstance(target, (ast.Tuple, ast.List)) and all(
                isinstance(e, ast.Name) for e in target.elts):
            names = [e.id for e in target.elts]
            if (isinstance(stmt.value, (ast.Tuple, ast.List))
                    and len(stmt.value.elts) == len(target.elts)
                    and not any(isinstance(v, ast.Starred) for v in stmt.value.elts)):
                # Simultaneous assignment: a, b = x, y
                # (emitters lower this through temporaries, so a, b = b, a
                # swaps are 100% safe).
                return {"type": "multi_binding", "names": names,
                        "values": [_lower_expr(v) for v in stmt.value.elts]}
            if not isinstance(stmt.value, (ast.Tuple, ast.List)):
                if isinstance(stmt.value, ast.Call):
                    # Call-site tuple unpacking: x, y, z = fn(...)
                    return {"type": "tuple_unbind", "names": names,
                            "value": _lower_expr(stmt.value)}
                # Unpacking from an array/indexable: q, p, s = state
                # -> q = state[0]; p = state[1]; s = state[2] (via temps).
                value = _lower_expr(stmt.value)
                if value is not None:
                    return {"type": "multi_binding", "names": names,
                            "values": [{"type": "subscript", "value": value,
                                        "slice": {"type": "literal", "value": i}}
                                       for i in range(len(names))]}
            raise UnsupportedError("tuple unpacking of non-tuple values", node=stmt)
        # Subscript/attribute targets: buf.data[i] = ...
        lowered_target = _lower_expr(target)
        if lowered_target is not None:
            return {"type": "binding", "target": lowered_target,
                    "value": _lower_expr(stmt.value)}
    if isinstance(stmt, ast.Return):
        if isinstance(stmt.value, ast.Tuple):
            # Multi-value return: return a, b, c -> UAST tuple node so
            # emitters can lower it to std::tuple / std::make_tuple.
            return {"type": "return",
                    "value": {"type": "tuple",
                              "elements": [_lower_expr(e) for e in stmt.value.elts]}}
        if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            return {"type": "return", "value": None}
        return {"type": "return",
                "value": _lower_expr(stmt.value) if stmt.value is not None else None}
    if isinstance(stmt, ast.If):
        return _lower_if(stmt)
    if isinstance(stmt, ast.For):
        body = [n for n in (_lower_stmt(s) for s in stmt.body) if n is not None]
        return {"type": "loop_for", "target": _lower_expr(stmt.target),
                "iter": _lower_expr(stmt.iter), "body": body}
    if isinstance(stmt, ast.While):
        body = [n for n in (_lower_stmt(s) for s in stmt.body) if n is not None]
        return {"type": "loop_while", "condition": _lower_expr(stmt.test),
                "body": body}
    if isinstance(stmt, ast.Expr):
        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            return None  # docstring
        value = _lower_expr(stmt.value)
        return {"type": "expr", "value": value} if value is not None else None
    if isinstance(stmt, ast.AugAssign):
        op = _binop_name(stmt.op)
        target_expr = _lower_expr(stmt.target)
        if isinstance(stmt.target, ast.Name):
            return {"type": "binding", "name": stmt.target.id,
                    "value": {"type": "binop", "op": op, "left": target_expr,
                              "right": _lower_expr(stmt.value)}}
        if target_expr is not None:
            return {"type": "binding", "target": target_expr,
                    "value": {"type": "binop", "op": op, "left": target_expr,
                              "right": _lower_expr(stmt.value)}}
    return None


def _binop_name(op: ast.operator) -> str:
    return {
        ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
        ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
        ast.LShift: "<<", ast.RShift: ">>",
    }.get(type(op), "?")


def _unaryop_name(op: ast.unaryop) -> str:
    return {ast.UAdd: "+", ast.USub: "-", ast.Not: "not", ast.Invert: "~"}.get(type(op), "?")


def _cmp_op_name(op: ast.cmpop) -> str:
    return {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
            ast.Gt: ">", ast.GtE: ">=", ast.In: "in", ast.NotIn: "not_in",
            ast.Is: "is", ast.IsNot: "is_not"}.get(type(op), "compare")


def _lower_if(stmt: ast.If) -> dict:
    then_body = [n for n in (_lower_stmt(s) for s in stmt.body) if n is not None]
    else_body = [n for n in (_lower_stmt(s) for s in stmt.orelse) if n is not None]
    return {"type": "if", "condition": _lower_expr(stmt.test),
            "then": then_body, "else": else_body}


def _lower_expr(expr: Optional[ast.expr]) -> Optional[dict]:
    if expr is None:
        return None
    if isinstance(expr, ast.NamedExpr):
        raise UnsupportedError("walrus operator", node=expr)
    if isinstance(expr, (ast.Await, ast.Yield, ast.YieldFrom)):
        raise UnsupportedError("async/yield expressions", node=expr)
    if isinstance(expr, ast.Constant):
        # Last-resort complex fallback: Tier-0 should have lowered complex
        # arithmetic already, but never crash on a stray ``1j`` literal.
        if isinstance(expr.value, complex):
            value = expr.value
            return {"type": "literal",
                    "value": float(value.real if value.real != 0.0 else value.imag)}
        # Python ``None`` is not a numeric literal.  Return-position ``None``
        # is handled by the statement lowering above; every expression use is
        # scalarized so emitters never see an unsupported ``None`` literal.
        return {"type": "literal", "value": 0.0 if expr.value is None else expr.value}
    if isinstance(expr, ast.Name):
        return {"type": "reference", "name": expr.id}
    if isinstance(expr, ast.Call):
        if _is_io_call(expr):
            raise UnsupportedError("io/print calls", node=expr)
        args = [_lower_expr(a) for a in expr.args]
        func = expr.func
        local = getattr(_lower_expr, "local_functions", set())
        callee_name = func.id if isinstance(func, ast.Name) else None
        call_type = "user_function_call" if callee_name in local else "call"
        func_node = _lower_expr(func)
        if isinstance(func_node, dict) and func_node.get("type") == "reference":
            fname = str(func_node.get("name") or "")
            if isinstance(func, ast.Name) and func.id in MATH_FUNCS \
                    and func.id not in local:
                func_node = {"type": "reference", "name": f"math.{func.id}"}
            elif isinstance(func, ast.Name) and func.id in INTRINSIC_ALIASES \
                    and func.id not in local:
                func_node = {"type": "reference",
                             "name": INTRINSIC_ALIASES[func.id]}
            elif fname.startswith(("np.", "numpy.")):
                attr = fname.split(".", 1)[1]
                if attr in {"clip", "sign", "sgn"}:
                    func_node = {"type": "reference",
                                 "name": {"clip": "clamp", "sgn": "sign"}.get(attr, attr)}
                elif attr in {"sq", "lerp", "safe_div"}:
                    func_node = {"type": "reference", "name": attr}
                elif attr in NP_MATH_FUNCS:
                    func_node = {"type": "reference", "name": f"math.{attr}"}
        return {"type": call_type, "function": func_node,
                "argument": args[0] if args else None, "arguments": args}
    if isinstance(expr, ast.IfExp):
        return {"type": "if", "condition": _lower_expr(expr.test),
                "then": [_lower_expr(expr.body)], "else": [_lower_expr(expr.orelse)]}
    if isinstance(expr, ast.BinOp):
        return {"type": "binop", "op": _binop_name(expr.op),
                "left": _lower_expr(expr.left), "right": _lower_expr(expr.right)}
    if isinstance(expr, ast.UnaryOp):
        return {"type": "unaryop", "op": _unaryop_name(expr.op),
                "operand": _lower_expr(expr.operand)}
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and isinstance(expr.ops[0], ast.In):
        return {"type": "set_member", "collection": _lower_expr(expr.comparators[0]),
                "element": _lower_expr(expr.left)}
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
        return {"type": "compare", "op": _cmp_op_name(expr.ops[0]),
                "left": _lower_expr(expr.left), "right": _lower_expr(expr.comparators[0])}
    if isinstance(expr, ast.Compare):
        # Chained comparison a < b < c  ->  (a < b) and (b < c)
        parts = []
        left = expr.left
        for op, right in zip(expr.ops, expr.comparators):
            parts.append({"type": "compare", "op": _cmp_op_name(op),
                          "left": _lower_expr(left), "right": _lower_expr(right)})
            left = right
        return {"type": "call", "function": {"type": "reference", "name": "__boolop_and__"},
                "argument": {"type": "literal", "value": parts}}
    if isinstance(expr, ast.BoolOp):
        op_name = "and" if isinstance(expr.op, ast.And) else "or"
        return {"type": "call", "function": {"type": "reference", "name": f"__boolop_{op_name}__"},
                "argument": {"type": "literal", "value": [_lower_expr(v) for v in expr.values]}}
    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name) and expr.value.id == "math" and expr.attr in MATH_CONSTANTS:
            return {"type": "literal", "value": MATH_CONSTANTS[expr.attr]}
        base = _lower_expr(expr.value)
        base_name = base.get("name") if base and base.get("type") == "reference" else None
        if base_name is not None:
            return {"type": "reference", "name": f"{base_name}.{expr.attr}"}
        return {"type": "attribute", "value": base, "attr": expr.attr}
    if isinstance(expr, ast.Subscript):
        return {"type": "subscript", "value": _lower_expr(expr.value),
                "slice": _lower_expr(expr.slice)}
    if isinstance(expr, ast.Dict):
        return {"type": "dict",
                "pairs": [{"key": _lower_expr(k), "value": _lower_expr(v)}
                          for k, v in zip(expr.keys, expr.values) if k is not None]}
    if isinstance(expr, ast.Set):
        return {"type": "set", "elements": [_lower_expr(e) for e in expr.elts]}
    if isinstance(expr, ast.List):
        return {"type": "list", "elements": [_lower_expr(e) for e in expr.elts]}
    if isinstance(expr, ast.ListComp):
        # Only the 2-D allocation idiom is supported:
        #   [[0.0] * n for _ in range(m)]
        gen = expr.generators[0] if len(expr.generators) == 1 else None
        elt = expr.elt
        if (gen is not None and not gen.ifs and isinstance(gen.iter, ast.Call)
                and isinstance(gen.iter.func, ast.Name) and gen.iter.func.id == "range"
                and len(gen.iter.args) == 1 and isinstance(elt, ast.BinOp)
                and isinstance(elt.op, ast.Mult) and isinstance(elt.left, ast.List)
                and len(elt.left.elts) == 1):
            return {"type": "list2d_alloc", "rows": _lower_expr(gen.iter.args[0]),
                    "cols": _lower_expr(elt.right),
                    "fill": _lower_expr(elt.left.elts[0])}
        raise UnsupportedError("list comprehensions", node=expr)
    if isinstance(expr, (ast.Tuple, ast.List)):
        # Tuple/list literals are lowered to the emitter's list/array dialect;
        # indexing t[0] then works through the normal subscript node. In
        # return position the caller wraps them in a UAST ``tuple`` node.
        if any(isinstance(e, ast.Starred) for e in expr.elts):
            raise UnsupportedError("starred expressions", node=expr)
        return {"type": "list",
                "elements": [_lower_expr(e) for e in expr.elts]}
    if isinstance(expr, ast.Lambda):
        raise UnsupportedError("lambda expressions", node=expr)
    return None


__all__ = ["python_source_to_uast", "UnsupportedError"]

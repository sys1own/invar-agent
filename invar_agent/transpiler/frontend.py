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


class UnsupportedError(ValueError):
    def __init__(self, feature: str, node=None):
        super().__init__(f"unsupported construct: {feature}")
        self.feature = feature
        self.node = node


def python_source_to_uast(source: str) -> dict:
    tree = ast.parse(source)
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
    if isinstance(stmt, (ast.Try, getattr(ast, "TryStar", ()))):
        body = [n for n in (_lower_stmt(s) for s in stmt.body) if n is not None]
        return body[-1] if body else None
    if isinstance(stmt, (ast.Yield, ast.YieldFrom)):
        raise UnsupportedError("generators", node=stmt)
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
            return {"type": "binding",
                    "name": [e.id for e in target.elts],
                    "value": _lower_expr(stmt.value)}
        # Subscript/attribute targets: buf.data[i] = ...
        lowered_target = _lower_expr(target)
        if lowered_target is not None:
            return {"type": "binding", "target": lowered_target,
                    "value": _lower_expr(stmt.value)}
    if isinstance(stmt, ast.Return):
        return _lower_expr(stmt.value) if stmt.value is not None else None
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
        return _lower_expr(stmt.value)
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
        return {"type": "literal", "value": expr.value}
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
        return {"type": call_type, "function": _lower_expr(func),
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
        raise UnsupportedError("list comprehensions", node=expr)
    if isinstance(expr, ast.Lambda):
        raise UnsupportedError("lambda expressions", node=expr)
    return None


__all__ = ["python_source_to_uast", "UnsupportedError"]

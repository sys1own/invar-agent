"""Unified Expression AST `invar.expr.v1` (UPG-006).

Single canonical expression representation shared between Python and Rust (`expr_ast.rs`),
with exact rational literals, total lossless SciIR adapters, formal Python AST bridge,
and multi-dialect rendering (`formal`, `exec_py`, `sci_text`, `csel`, `cpp`, `rust`, `latex`).
"""
from __future__ import annotations

import ast
import json
import math
from fractions import Fraction
from typing import Annotated, Any, Dict, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field


class _N(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Const(_N):
    kind: Literal["const"] = "const"
    num: str
    den: str = "1"

    def as_fraction(self) -> Fraction:
        return Fraction(int(self.num), int(self.den))

    def as_float(self) -> float:
        return float(self.as_fraction())


class Sym(_N):
    kind: Literal["sym"] = "sym"
    name: str


class Unary(_N):
    kind: Literal["unary"] = "unary"
    op: Literal["neg"] = "neg"
    arg: "Expr"


class Binary(_N):
    kind: Literal["binary"] = "binary"
    op: Literal["add", "sub", "mul", "div", "pow"]
    lhs: "Expr"
    rhs: "Expr"


class Call(_N):
    kind: Literal["call"] = "call"
    fn: str
    args: Tuple["Expr", ...]


class Cond(_N):
    kind: Literal["cond"] = "cond"
    pred: "Pred"
    then_expr: "Expr"
    else_expr: "Expr"


class Rand(_N):
    kind: Literal["rand"] = "rand"
    dist: Literal["uniform01", "normal01", "exponential1"]
    draw_stream_id: int = 0


class Cmp(_N):
    kind: Literal["cmp"] = "cmp"
    op: Literal["lt", "le", "gt", "ge", "eq", "ne"]
    lhs: "Expr"
    rhs: "Expr"


class BoolN(_N):
    kind: Literal["bool"] = "bool"
    op: Literal["and", "or", "not"]
    args: Tuple["Pred", ...]


Expr = Annotated[
    Union[Const, Sym, Unary, Binary, Call, Cond, Rand],
    Field(discriminator="kind"),
]
Pred = Annotated[
    Union[Cmp, BoolN],
    Field(discriminator="kind"),
]

Unary.model_rebuild()
Binary.model_rebuild()
Call.model_rebuild()
Cond.model_rebuild()
Cmp.model_rebuild()
BoolN.model_rebuild()


def const_from_fraction(f: Fraction) -> Const:
    return Const(num=str(f.numerator), den=str(f.denominator))


def const_from_number(v: Union[int, float, str, Fraction]) -> Const:
    if isinstance(v, Fraction):
        return const_from_fraction(v)
    if isinstance(v, int):
        return Const(num=str(v), den="1")
    if isinstance(v, str):
        return const_from_fraction(Fraction(v.strip()))
    f = Fraction(str(v)) if math.isfinite(float(v)) else Fraction(0)
    return const_from_fraction(f)


def parse_expr(text: str, dialect: str = "formal") -> Expr:
    """Parse an expression string in the given dialect via the native Rust Pratt parser."""
    from invar_agent import _native

    raw = json.loads(_native.py_expr_parse(text, dialect))
    return _expr_from_dict(raw)


def canonicalize(expr: Expr) -> Expr:
    """Canonicalize an `invar.expr.v1` AST identically to Rust `expr_ast::canonicalize`."""
    if isinstance(expr, Const):
        f = expr.as_fraction()
        return Const(num=str(f.numerator), den=str(f.denominator))
    if isinstance(expr, (Sym, Rand)):
        return expr
    if isinstance(expr, Unary):
        inner = canonicalize(expr.arg)
        if isinstance(inner, Const):
            f = -inner.as_fraction()
            return Const(num=str(f.numerator), den=str(f.denominator))
        return Unary(op=expr.op, arg=inner)
    if isinstance(expr, Binary):
        cl = canonicalize(expr.lhs)
        cr = canonicalize(expr.rhs)
        if expr.op == "div" and isinstance(cl, Const) and isinstance(cr, Const):
            den_f = cr.as_fraction()
            if den_f != 0:
                q = cl.as_fraction() / den_f
                return Const(num=str(q.numerator), den=str(q.denominator))
        return Binary(op=expr.op, lhs=cl, rhs=cr)
    if isinstance(expr, Cond):
        return Cond(
            pred=canonicalize_pred(expr.pred),
            then_expr=canonicalize(expr.then_expr),
            else_expr=canonicalize(expr.else_expr),
        )
    if isinstance(expr, Call):
        cargs = tuple(canonicalize(a) for a in expr.args)
        fn = {"ln": "log", "fabs": "abs"}.get(expr.fn, expr.fn)
        if fn == "pow" and len(cargs) == 2:
            return Binary(op="pow", lhs=cargs[0], rhs=cargs[1])
        if fn == "clamp" and len(cargs) == 3:
            x, lo, hi = cargs
            return Call(fn="min", args=(hi, Call(fn="max", args=(lo, x))))
        if fn == "step" and len(cargs) == 1:
            return Cond(
                pred=Cmp(op="ge", lhs=cargs[0], rhs=Const(num="0", den="1")),
                then_expr=Const(num="1", den="1"),
                else_expr=Const(num="0", den="1"),
            )
        if fn in ("min", "max") and len(cargs) >= 2:
            return _lower_nary_binary_tree(fn, cargs)
        return Call(fn=fn, args=cargs)
    raise TypeError(f"Unsupported Expr node: {type(expr).__name__}")


def canonicalize_pred(pred: Pred) -> Pred:
    if isinstance(pred, Cmp):
        return Cmp(op=pred.op, lhs=canonicalize(pred.lhs), rhs=canonicalize(pred.rhs))
    if isinstance(pred, BoolN):
        return BoolN(op=pred.op, args=tuple(canonicalize_pred(a) for a in pred.args))
    raise TypeError(f"Unsupported Pred node: {type(pred).__name__}")


def _lower_nary_binary_tree(fn: str, args: Sequence[Expr]) -> Expr:
    items = list(args)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return Call(fn=fn, args=(items[0], items[1]))
    mid = len(items) // 2
    return Call(
        fn=fn,
        args=(
            _lower_nary_binary_tree(fn, items[:mid]),
            _lower_nary_binary_tree(fn, items[mid:]),
        ),
    )


def render_expr(expr: Expr, dialect: str = "formal") -> str:
    """Render an `invar.expr.v1` AST to the requested dialect via the native Rust renderer."""
    from invar_agent import _native

    return _native.py_expr_render(expr.model_dump_json(), dialect)


def _expr_from_dict(d: Dict[str, Any]) -> Expr:
    k = d.get("kind")
    if k == "const":
        return Const(num=str(d["num"]), den=str(d.get("den", "1")))
    if k == "sym":
        return Sym(name=str(d["name"]))
    if k == "unary":
        return Unary(op=d["op"], arg=_expr_from_dict(d["arg"]))
    if k == "binary":
        return Binary(op=d["op"], lhs=_expr_from_dict(d["lhs"]), rhs=_expr_from_dict(d["rhs"]))
    if k == "call":
        return Call(fn=str(d["fn"]), args=tuple(_expr_from_dict(a) for a in d["args"]))
    if k == "cond":
        return Cond(
            pred=_pred_from_dict(d["pred"]),
            then_expr=_expr_from_dict(d["then_expr"]),
            else_expr=_expr_from_dict(d["else_expr"]),
        )
    if k == "rand":
        return Rand(dist=d["dist"], draw_stream_id=int(d.get("draw_stream_id", 0)))
    raise ValueError(f"Unknown Expr kind: {k!r}")


def _pred_from_dict(d: Dict[str, Any]) -> Pred:
    k = d.get("kind")
    if k == "cmp":
        return Cmp(op=d["op"], lhs=_expr_from_dict(d["lhs"]), rhs=_expr_from_dict(d["rhs"]))
    if k == "bool":
        return BoolN(op=d["op"], args=tuple(_pred_from_dict(a) for a in d["args"]))
    raise ValueError(f"Unknown Pred kind: {k!r}")


# ---------------------------------------------------------------------------
# Total, lossless adapters between SciIR ExprNode dicts and `invar.expr.v1`
# ---------------------------------------------------------------------------

_SCIIR_UNARY_TO_FN = {
    "abs": "abs",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "sin": "sin",
    "cos": "cos",
    "tanh": "tanh",
    "step": "step",
}
_FN_TO_SCIIR_UNARY = {v: k for k, v in _SCIIR_UNARY_TO_FN.items()}


def sciir_to_expr(node: Mapping[str, Any]) -> Expr:
    """Lossless adapter from SciIR `ExprNode` JSON dict -> `invar.expr.v1` `Expr`."""
    kind = node.get("kind")
    if kind == "const":
        exact = node.get("exact")
        if exact:
            return const_from_fraction(Fraction(str(exact)))
        return const_from_number(float(node["value"]))
    if kind == "sym":
        return Sym(name=str(node["name"]))
    if kind == "unary":
        op = str(node["op"])
        arg = sciir_to_expr(node["arg"])
        if op == "neg":
            return Unary(op="neg", arg=arg)
        if op in _SCIIR_UNARY_TO_FN:
            return Call(fn=_SCIIR_UNARY_TO_FN[op], args=(arg,))
        raise ValueError(f"Unsupported SciIR unary op: {op!r}")
    if kind == "binary":
        op = str(node["op"])
        lhs = sciir_to_expr(node["lhs"])
        rhs = sciir_to_expr(node["rhs"])
        if op in ("add", "sub", "mul", "div", "pow"):
            return Binary(op=op, lhs=lhs, rhs=rhs)
        if op in ("min", "max"):
            return Call(fn=op, args=(lhs, rhs))
        raise ValueError(f"Unsupported SciIR binary op: {op!r}")
    if kind == "call":
        return Call(fn=str(node["func"]), args=tuple(sciir_to_expr(a) for a in node["args"]))
    if kind == "if_then_else":
        return Cond(
            pred=Cmp(
                op=str(node["cond_op"]),
                lhs=sciir_to_expr(node["lhs"]),
                rhs=sciir_to_expr(node["rhs"]),
            ),
            then_expr=sciir_to_expr(node["then_branch"]),
            else_expr=sciir_to_expr(node["else_branch"]),
        )
    if kind == "rand":
        return Rand(dist=node["dist"], draw_stream_id=int(node.get("draw_stream_id", 0)))
    raise ValueError(f"Unknown SciIR ExprNode kind: {kind!r}")


def expr_to_sciir(
    expr: Expr,
    symbol_meta: Optional[Mapping[str, Tuple[str, int]]] = None,
    *,
    include_exact: bool = False,
) -> Dict[str, Any]:
    """Total lossless adapter from `invar.expr.v1` `Expr` -> SciIR `ExprNode` JSON dict."""
    sym_table = symbol_meta or {}
    if isinstance(expr, Const):
        f = expr.as_fraction()
        out: Dict[str, Any] = {"kind": "const", "value": float(f)}
        if include_exact and f.denominator != 1:
            out["exact"] = f"{f.numerator}/{f.denominator}"
        return out
    if isinstance(expr, Sym):
        role, idx = sym_table.get(expr.name, ("STATE", 0))
        return {"kind": "sym", "name": expr.name, "role": role, "slot_index": int(idx)}
    if isinstance(expr, Unary):
        return {
            "kind": "unary",
            "op": "neg",
            "arg": expr_to_sciir(expr.arg, sym_table, include_exact=include_exact),
        }
    if isinstance(expr, Binary):
        return {
            "kind": "binary",
            "op": expr.op,
            "lhs": expr_to_sciir(expr.lhs, sym_table, include_exact=include_exact),
            "rhs": expr_to_sciir(expr.rhs, sym_table, include_exact=include_exact),
        }
    if isinstance(expr, Call):
        if expr.fn in _FN_TO_SCIIR_UNARY and len(expr.args) == 1:
            return {
                "kind": "unary",
                "op": _FN_TO_SCIIR_UNARY[expr.fn],
                "arg": expr_to_sciir(expr.args[0], sym_table, include_exact=include_exact),
            }
        if expr.fn in ("min", "max") and len(expr.args) == 2:
            return {
                "kind": "binary",
                "op": expr.fn,
                "lhs": expr_to_sciir(expr.args[0], sym_table, include_exact=include_exact),
                "rhs": expr_to_sciir(expr.args[1], sym_table, include_exact=include_exact),
            }
        return {
            "kind": "call",
            "func": expr.fn,
            "args": [expr_to_sciir(a, sym_table, include_exact=include_exact) for a in expr.args],
        }
    if isinstance(expr, Cond):
        return _cond_to_sciir(expr.pred, expr.then_expr, expr.else_expr, sym_table, include_exact)
    if isinstance(expr, Rand):
        return {"kind": "rand", "dist": expr.dist, "draw_stream_id": expr.draw_stream_id}
    raise TypeError(f"Unsupported Expr node: {type(expr).__name__}")


def _cond_to_sciir(
    pred: Pred,
    then_e: Expr,
    else_e: Expr,
    sym_table: Mapping[str, Tuple[str, int]],
    include_exact: bool,
) -> Dict[str, Any]:
    if isinstance(pred, Cmp):
        return {
            "kind": "if_then_else",
            "cond_op": pred.op,
            "lhs": expr_to_sciir(pred.lhs, sym_table, include_exact=include_exact),
            "rhs": expr_to_sciir(pred.rhs, sym_table, include_exact=include_exact),
            "then_branch": expr_to_sciir(then_e, sym_table, include_exact=include_exact),
            "else_branch": expr_to_sciir(else_e, sym_table, include_exact=include_exact),
        }
    if isinstance(pred, BoolN):
        if pred.op == "not" and len(pred.args) == 1:
            return _cond_to_sciir(pred.args[0], else_e, then_e, sym_table, include_exact)
        if pred.op == "and":
            acc = then_e
            for sub in reversed(pred.args):
                acc = Cond(pred=sub, then_expr=acc, else_expr=else_e)
            return expr_to_sciir(acc, sym_table, include_exact=include_exact)
        if pred.op == "or":
            acc = else_e
            for sub in reversed(pred.args):
                acc = Cond(pred=sub, then_expr=then_e, else_expr=acc)
            return expr_to_sciir(acc, sym_table, include_exact=include_exact)
    raise ValueError(f"Unsupported predicate in Cond -> SciIR: {pred!r}")


# ---------------------------------------------------------------------------
# Formal Python AST bridge & AST substitution (replaces regex substitution)
# ---------------------------------------------------------------------------

_CMP_TO_AST = {
    "lt": ast.Lt,
    "le": ast.LtE,
    "gt": ast.Gt,
    "ge": ast.GtE,
    "eq": ast.Eq,
    "ne": ast.NotEq,
}
_BIN_TO_AST = {
    "add": ast.Add,
    "sub": ast.Sub,
    "mul": ast.Mult,
    "div": ast.Div,
    "pow": ast.Pow,
}


def _pred_to_formal_ast(pred: Pred) -> ast.expr:
    if isinstance(pred, Cmp):
        return ast.Compare(
            left=expr_to_formal_ast(pred.lhs),
            ops=[_CMP_TO_AST[pred.op]()],
            comparators=[expr_to_formal_ast(pred.rhs)],
        )
    if isinstance(pred, BoolN):
        if pred.op == "not":
            return ast.UnaryOp(op=ast.Not(), operand=_pred_to_formal_ast(pred.args[0]))
        op_cls = ast.And if pred.op == "and" else ast.Or
        return ast.BoolOp(op=op_cls(), values=[_pred_to_formal_ast(a) for a in pred.args])
    raise TypeError(f"Unsupported Pred node: {type(pred).__name__}")


def expr_to_formal_ast(expr: Expr) -> ast.expr:
    """Convert an `invar.expr.v1` `Expr` into a formal Python `ast.expr`."""
    if isinstance(expr, Const):
        f = expr.as_fraction()
        if f.denominator == 1:
            node: ast.expr = ast.Constant(value=int(f.numerator))
        else:
            node = ast.BinOp(
                left=ast.Constant(value=int(f.numerator)),
                op=ast.Div(),
                right=ast.Constant(value=int(f.denominator)),
            )
        return ast.fix_missing_locations(node)
    if isinstance(expr, Sym):
        return ast.Name(id=expr.name, ctx=ast.Load())
    if isinstance(expr, Unary):
        return ast.UnaryOp(op=ast.USub(), operand=expr_to_formal_ast(expr.arg))
    if isinstance(expr, Binary):
        return ast.BinOp(
            left=expr_to_formal_ast(expr.lhs),
            op=_BIN_TO_AST[expr.op](),
            right=expr_to_formal_ast(expr.rhs),
        )
    if isinstance(expr, Call):
        return ast.Call(
            func=ast.Name(id=expr.fn, ctx=ast.Load()),
            args=[expr_to_formal_ast(a) for a in expr.args],
            keywords=[],
        )
    if isinstance(expr, Cond):
        return ast.IfExp(
            test=_pred_to_formal_ast(expr.pred),
            body=expr_to_formal_ast(expr.then_expr),
            orelse=expr_to_formal_ast(expr.else_expr),
        )
    if isinstance(expr, Rand):
        return ast.Call(
            func=ast.Name(id=f"rand_{expr.dist}", ctx=ast.Load()),
            args=[ast.Constant(value=expr.draw_stream_id)],
            keywords=[],
        )
    raise TypeError(f"Unsupported Expr node: {type(expr).__name__}")


def substitute_expr(expr: Expr, env: Mapping[str, Expr]) -> Expr:
    """Capture-free AST substitution of symbols in `expr`."""
    if isinstance(expr, Sym):
        return env.get(expr.name, expr)
    if isinstance(expr, (Const, Rand)):
        return expr
    if isinstance(expr, Unary):
        return Unary(op=expr.op, arg=substitute_expr(expr.arg, env))
    if isinstance(expr, Binary):
        return Binary(
            op=expr.op,
            lhs=substitute_expr(expr.lhs, env),
            rhs=substitute_expr(expr.rhs, env),
        )
    if isinstance(expr, Call):
        return Call(fn=expr.fn, args=tuple(substitute_expr(a, env) for a in expr.args))
    if isinstance(expr, Cond):
        return Cond(
            pred=_substitute_pred(expr.pred, env),
            then_expr=substitute_expr(expr.then_expr, env),
            else_expr=substitute_expr(expr.else_expr, env),
        )
    return expr


def _substitute_pred(pred: Pred, env: Mapping[str, Expr]) -> Pred:
    if isinstance(pred, Cmp):
        return Cmp(
            op=pred.op,
            lhs=substitute_expr(pred.lhs, env),
            rhs=substitute_expr(pred.rhs, env),
        )
    if isinstance(pred, BoolN):
        return BoolN(op=pred.op, args=tuple(_substitute_pred(a, env) for a in pred.args))
    return pred


def expr_free_symbols(expr: Expr) -> Set[str]:
    """Return all free symbol names referenced in `expr`."""
    out: Set[str] = set()
    _collect_syms(expr, out)
    return out


def _collect_syms(expr: Expr, out: Set[str]) -> None:
    if isinstance(expr, Sym):
        out.add(expr.name)
    elif isinstance(expr, Unary):
        _collect_syms(expr.arg, out)
    elif isinstance(expr, Binary):
        _collect_syms(expr.lhs, out)
        _collect_syms(expr.rhs, out)
    elif isinstance(expr, Call):
        for a in expr.args:
            _collect_syms(a, out)
    elif isinstance(expr, Cond):
        _collect_pred_syms(expr.pred, out)
        _collect_syms(expr.then_expr, out)
        _collect_syms(expr.else_expr, out)


def _collect_pred_syms(pred: Pred, out: Set[str]) -> None:
    if isinstance(pred, Cmp):
        _collect_syms(pred.lhs, out)
        _collect_syms(pred.rhs, out)
    elif isinstance(pred, BoolN):
        for a in pred.args:
            _collect_pred_syms(a, out)

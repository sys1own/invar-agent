"""Closed Symbolic Expression Language (CSEL) parser, validator, and totalizing lowerer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

CSEL_INTRINSICS: Set[str] = {
    "safe_div",
    "safe_sqrt",
    "safe_log",
    "safe_exp",
    "safe_pow",
    "safe_sin",
    "safe_cos",
    "safe_tan",
    "safe_tanh",
    "safe_acos",
    "safe_asin",
    "smooth_clamp",
    "where",
    "abs",
    "min",
    "max",
    "log1p",
    "expm1",
}

BARE_TO_SAFE_CALL: Dict[str, str] = {
    "sqrt": "safe_sqrt",
    "log": "safe_log",
    "ln": "safe_log",
    "exp": "safe_exp",
    "pow": "safe_pow",
    "sin": "safe_sin",
    "cos": "safe_cos",
    "tan": "safe_tan",
    "tanh": "safe_tanh",
    "acos": "safe_acos",
    "asin": "safe_asin",
    "fabs": "abs",
}

FORBIDDEN_CSEL_TOKENS: Set[str] = {
    "import",
    "from",
    "def",
    "class",
    "lambda",
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "while",
    "for",
    "try",
    "except",
    "with",
    "yield",
    "return",
    "global",
    "nonlocal",
}


class CselSyntaxError(ValueError):
    """Raised when an expression violates the Closed Symbolic Expression Language grammar."""


@dataclass(frozen=True)
class _Token:
    kind: str  # NUM, ID, OP, LPAREN, RPAREN, COMMA, EOF
    value: str
    pos: int


_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<NUM>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<ID>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<OP>\*\*|[+\-*/^])"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<COMMA>,)"
    r"|(?P<BAD>.)"
    r")"
)


def tokenize_csel(text: str) -> List[_Token]:
    tokens: List[_Token] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        val = m.group(kind) if kind is not None else ""
        if kind == "BAD":
            raise CselSyntaxError(f"Illegal character {val!r} at position {m.start()} in CSEL: {text!r}")
        if kind == "ID" and val in FORBIDDEN_CSEL_TOKENS:
            raise CselSyntaxError(f"Forbidden imperative keyword {val!r} in CSEL: {text!r}")
        if kind is not None:
            tokens.append(_Token(kind=kind, value=val, pos=m.start()))
    tokens.append(_Token(kind="EOF", value="", pos=len(text)))
    return tokens


class _CselParser:
    def __init__(self, tokens: List[_Token], source: str) -> None:
        self.tokens = tokens
        self.source = source
        self.idx = 0

    def peek(self) -> _Token:
        return self.tokens[self.idx]

    def consume(self, expected_kind: str | None = None) -> _Token:
        tok = self.peek()
        if expected_kind is not None and tok.kind != expected_kind:
            raise CselSyntaxError(
                f"Expected {expected_kind}, got {tok.kind} ({tok.value!r}) at pos {tok.pos} in {self.source!r}"
            )
        self.idx += 1
        return tok

    def parse(self) -> Dict[str, Any]:
        node = self.parse_expr()
        if self.peek().kind != "EOF":
            tok = self.peek()
            raise CselSyntaxError(
                f"Unexpected trailing token {tok.value!r} at pos {tok.pos} in {self.source!r}"
            )
        return node

    def parse_expr(self) -> Dict[str, Any]:
        node = self.parse_term()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.consume().value
            right = self.parse_term()
            node = {"kind": "binary", "op": op, "left": node, "right": right}
        return node

    def parse_term(self) -> Dict[str, Any]:
        node = self.parse_factor()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/"):
            op = self.consume().value
            right = self.parse_factor()
            if op == "/":
                # Totalize division automatically into safe_div
                node = {"kind": "call", "func": "safe_div", "args": [node, right]}
            else:
                node = {"kind": "binary", "op": op, "left": node, "right": right}
        return node

    def parse_factor(self) -> Dict[str, Any]:
        if self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.consume().value
            arg = self.parse_factor()
            if op == "+":
                return arg
            return {"kind": "unary", "op": "-", "arg": arg}
        node = self.parse_primary()
        if self.peek().kind == "OP" and self.peek().value in ("^", "**"):
            self.consume()
            exp = self.parse_factor()
            # Totalize exponentiation automatically into safe_pow
            return {"kind": "call", "func": "safe_pow", "args": [node, exp]}
        return node

    def parse_primary(self) -> Dict[str, Any]:
        tok = self.peek()
        if tok.kind == "NUM":
            self.consume()
            return {"kind": "literal", "value": float(tok.value)}
        if tok.kind == "LPAREN":
            self.consume("LPAREN")
            inner = self.parse_expr()
            self.consume("RPAREN")
            return inner
        if tok.kind == "ID":
            self.consume()
            if self.peek().kind == "LPAREN":
                self.consume("LPAREN")
                args: List[Dict[str, Any]] = []
                if self.peek().kind != "RPAREN":
                    args.append(self.parse_expr())
                    while self.peek().kind == "COMMA":
                        self.consume("COMMA")
                        args.append(self.parse_expr())
                self.consume("RPAREN")
                func = BARE_TO_SAFE_CALL.get(tok.value, tok.value)
                if func not in CSEL_INTRINSICS:
                    raise CselSyntaxError(
                        f"Unknown or unauthorized function {tok.value!r} in CSEL; allowed: {sorted(CSEL_INTRINSICS)}"
                    )
                return {"kind": "call", "func": func, "args": args}
            return {"kind": "symbol", "name": tok.value}
        raise CselSyntaxError(
            f"Unexpected token {tok.kind} ({tok.value!r}) at pos {tok.pos} in {self.source!r}"
        )


def parse_csel_to_ast(expr: str) -> Dict[str, Any]:
    """Parse a CSEL expression string into a totalized `CselExpr` JSON-compatible dict."""
    text = (expr or "").strip()
    if not text:
        raise CselSyntaxError("Empty CSEL expression")
    tokens = tokenize_csel(text)
    return _CselParser(tokens, text).parse()


def collect_csel_symbols(ast_node: Dict[str, Any], out: Set[str] | None = None) -> Set[str]:
    if out is None:
        out = set()
    kind = ast_node.get("kind")
    if kind == "symbol":
        out.add(str(ast_node["name"]))
    elif kind == "unary":
        collect_csel_symbols(ast_node["arg"], out)
    elif kind == "binary":
        collect_csel_symbols(ast_node["left"], out)
        collect_csel_symbols(ast_node["right"], out)
    elif kind == "call":
        for a in ast_node.get("args", []):
            collect_csel_symbols(a, out)
    return out


def csel_ast_to_str(ast_node: Dict[str, Any]) -> str:
    """Render a `CselExpr` dict back to canonical CSEL string."""
    kind = ast_node.get("kind")
    if kind == "literal":
        val = float(ast_node["value"])
        if abs(val - round(val)) < 1e-12 and abs(val) < 1e15:
            return f"{val:.1f}"
        return repr(val)
    if kind == "symbol":
        return str(ast_node["name"])
    if kind == "unary":
        return f"(-{csel_ast_to_str(ast_node['arg'])})"
    if kind == "binary":
        return f"({csel_ast_to_str(ast_node['left'])} {ast_node['op']} {csel_ast_to_str(ast_node['right'])})"
    if kind == "call":
        args = ", ".join(csel_ast_to_str(a) for a in ast_node.get("args", []))
        return f"{ast_node['func']}({args})"
    return "0.0"


def csel_ast_to_python(ast_node: Dict[str, Any]) -> str:
    """Render a `CselExpr` dict into a v9_transpiler_friendly Python expression using totalized helpers."""
    kind = ast_node.get("kind")
    if kind == "literal":
        val = float(ast_node["value"])
        if abs(val - round(val)) < 1e-12 and abs(val) < 1e15:
            return f"{val:.1f}"
        return repr(val)
    if kind == "symbol":
        return str(ast_node["name"])
    if kind == "unary":
        return f"(-{csel_ast_to_python(ast_node['arg'])})"
    if kind == "binary":
        l = csel_ast_to_python(ast_node["left"])
        r = csel_ast_to_python(ast_node["right"])
        op = ast_node["op"]
        if op == "/":
            return f"safe_div({l}, {r})"
        if op in ("^", "**"):
            return f"safe_pow({l}, {r})"
        return f"({l} {op} {r})"
    if kind == "call":
        func = ast_node["func"]
        py_func = {
            "log1p": "safe_log1p",
            "expm1": "safe_expm1",
            "where": "safe_where",
        }.get(func, func)
        args = ", ".join(csel_ast_to_python(a) for a in ast_node.get("args", []))
        return f"{py_func}({args})"
    return "0.0"

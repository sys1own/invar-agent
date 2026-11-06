"""Lower UAST to C++ (Clang++/G++ -O3 -std=c++20 -fPIC -shared)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .base import EmitterError, qual_name, render_literal

_CPP_BINOPS = {"+": "+", "-": "-", "*": "*", "/": "/", "//": "/", "%": "%",
               "**": "pow", "&": "&", "|": "|", "^": "^", "<<": "<<", ">>": ">>"}
_CPP_CMPS = {"==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
             "in": "in", "not_in": "not_in", "is": "==", "is_not": "!="}


class CppEmitter:
    """Emits a self-contained C++20 simulation kernel with a C-ABI
    ``extern "C" void execute_step(SimulationBuffer*)`` entry point."""

    def __init__(self, buffer_struct: str = "SimulationBuffer"):
        self.buffer_struct = buffer_struct

    # -- expressions -----------------------------------------------------
    def emit_expr(self, node: Optional[Dict[str, Any]]) -> str:
        if node is None:
            return "nullptr"
        t = node.get("type")
        if t == "literal":
            return render_literal(node.get("value"), "cpp")
        if t == "reference":
            name = str(node.get("name", ""))
            if name.startswith("buf."):
                return f"buf->{name.split('.', 1)[1]}"
            return qual_name(name, "cpp")
        if t == "binop":
            op = node.get("op", "?")
            left = self.emit_expr(node.get("left"))
            right = self.emit_expr(node.get("right"))
            if op == "**":
                return f"std::pow({left}, {right})"
            if op == "//":
                return f"std::floor({left} / ({right}))"
            sym = _CPP_BINOPS.get(op)
            if sym is None:
                raise EmitterError(f"unsupported binop {op!r}")
            return f"({left} {sym} {right})"
        if t == "unaryop":
            op = node.get("op", "?")
            operand = self.emit_expr(node.get("operand"))
            if op == "-":
                return f"(-{operand})"
            if op == "+":
                return operand
            if op == "not":
                return f"(!{operand})"
            raise EmitterError(f"unsupported unaryop {op!r}")
        if t == "call":
            fn = node.get("function") or {}
            args = node.get("arguments") or []
            if not args and node.get("argument") is not None:
                args = [node["argument"]]
            name = fn.get("name") if fn.get("type") == "reference" else None
            if name == "range" and args:
                return self.emit_expr(args[0])
            if name == "len" and args:
                return f"({self.emit_expr(args[0])}.size())"
            if name and name.startswith("__boolop_"):
                vals = args[0].get("value", []) if args and args[0].get("type") == "literal" else []
                op = "&&" if "and" in name else "||"
                return "(" + f" {op} ".join(self.emit_expr(v) for v in vals) + ")"
            if name and name.startswith("__unary_"):
                return self.emit_expr(node.get("argument"))
            target = qual_name(str(name or ""), "cpp") if name else self.emit_expr(fn)
            rendered = ", ".join(self.emit_expr(a) for a in args)
            return f"{target}({rendered})"
        if t == "compare":
            op = node.get("op", "==")
            left = self.emit_expr(node.get("left"))
            right = self.emit_expr(node.get("right"))
            sym = _CPP_CMPS.get(op)
            if sym in ("in", "not_in"):
                raise EmitterError("membership tests are not supported in C++ emitter")
            if sym is None:
                raise EmitterError(f"unsupported compare {op!r}")
            return f"({left} {sym} {right})"
        if t == "if":
            cond = self.emit_expr(node.get("condition"))
            then = self._stmt_value(node.get("then"))
            other = self._stmt_value(node.get("else"))
            return f"({then} if {cond} else {other})" if False else f"({cond} ? {then} : {other})"
        if t == "attribute":
            value = node.get("value") or {}
            # buf.data / buf.dt / buf.size / buf.steps become struct member access.
            if value.get("type") == "reference" and value.get("name") == "buf":
                return f"buf->{node.get('attr')}"
            rendered = self.emit_expr(value)
            return f"{rendered}.{node.get('attr')}"
        if t == "subscript":
            value = self.emit_expr(node.get("value"))
            slc = self.emit_expr(node.get("slice"))
            return f"{value}[{slc}]"
        if t == "list":
            items = node.get("elements") or []
            return "std::vector<double>{" + ", ".join(self.emit_expr(i) for i in items) + "}"
        raise EmitterError(f"unsupported expression type {t!r}")

    def _stmt_value(self, stmts) -> str:
        if isinstance(stmts, dict):
            stmts = [stmts]
        if not stmts:
            return "nullptr"
        last = stmts[-1]
        if isinstance(last, dict) and last.get("type") in ("return", "binding", "if"):
            if last.get("type") == "return":
                return self.emit_expr(last.get("value") if "value" in last else last)
            if last.get("type") == "binding":
                return self.emit_expr(last.get("value"))
            return self.emit_expr(last)
        return self.emit_expr(last)

    # -- statements ------------------------------------------------------
    def emit_stmt(self, node: Dict[str, Any], indent: int = 1) -> str:
        pad = "    " * indent
        t = node.get("type")
        if t == "binding":
            name = node.get("name")
            value = self.emit_expr(node.get("value"))
            target = node.get("target")
            if target is not None:
                return f"{pad}{self.emit_expr(target)} = {value};\n"
            if isinstance(name, list):
                names = ", ".join(name)
                vals = value.strip("std::vector<double>{}").split(", ") if value.startswith("std::vector") else [value]
                return f"{pad}auto [{names}] = {value};"
            return f"{pad}double {name} = {value};"
        if t == "return":
            val = node.get("value", node)
            if isinstance(val, dict) and val.get("type") not in (None,):
                return f"{pad}return {self.emit_expr(val)};"
            return f"{pad}return {self.emit_expr(val)};"
        if t == "function_declaration":
            return self._emit_function(node, indent)
        if t == "loop_for":
            target = node.get("target") or {}
            var = target.get("name") if isinstance(target, dict) else str(target)
            iter_expr = self.emit_expr(node.get("iter"))
            body = "".join(self.emit_stmt(s, indent + 1) for s in (node.get("body") or []) if s)
            return f"{pad}for (int {var} = 0; {var} < static_cast<int>({iter_expr}); ++{var}) {{\n{body}{pad}}}\n"
        if t == "loop_while":
            cond = self.emit_expr(node.get("condition"))
            body = "".join(self.emit_stmt(s, indent + 1) for s in (node.get("body") or []) if s)
            return f"{pad}while ({cond}) {{\n{body}{pad}}}\n"
        if t == "if":
            cond = self.emit_expr(node.get("condition"))
            then_body = "".join(self.emit_stmt(s, indent + 1) for s in (node.get("then") or []) if s)
            else_body = "".join(self.emit_stmt(s, indent + 1) for s in (node.get("else") or []) if s)
            else_part = f" else {{\n{else_body}{pad}}}" if else_body else ""
            return f"{pad}if ({cond}) {{\n{then_body}{pad}}}{else_part}\n"
        if t == "expr":
            return f"{pad}{self.emit_expr(node.get('value'))};\n"
        raise EmitterError(f"unsupported statement type {t!r}")

    def _emit_function(self, node: Dict[str, Any], indent: int) -> str:
        pad = "    " * indent
        name = node.get("name", "fn")
        params = node.get("params") or []
        body = "".join(self.emit_stmt(s, indent + 1) for s in (node.get("body") or []) if s)
        if not body.strip():
            body = f"{pad}    return 0;\n"
        param_list = ", ".join(f"double {p}" for p in params) if params else ""
        return (f"{pad}double {name}({param_list}) {{\n{body}{pad}}}\n")

    # -- whole module ----------------------------------------------------
    def emit_module(self, uast: Dict[str, Any]) -> str:
        children = uast.get("children") or []
        funcs = [c for c in children if isinstance(c, dict) and c.get("type") == "function_declaration"]
        step_fn = next((f for f in funcs if f.get("name") == "simulation_step"), None)
        other_funcs = [f for f in funcs if f is not step_fn]
        # Module-level bindings become globals.
        globals_ = [c for c in children if isinstance(c, dict) and c.get("type") == "binding"]
        lines = [
            "// Generated by invar-agent transpiler (CppEmitter)",
            "#include <cstdint>",
            "#include <cmath>",
            "#include <vector>",
            "",
            "extern \"C\" {",
            "typedef struct SimulationBuffer {",
            "    size_t size;",
            "    double dt;",
            "    uint32_t steps;",
            "    double* data;",
            "} SimulationBuffer;",
            "",
        ]
        for g in globals_:
            lines.append(f"static double g_{g.get('name')} = {self.emit_expr(g.get('value'))};")
        lines.append("}")
        lines.append("")
        for f in other_funcs:
            lines.append(self.emit_stmt(f, indent=0).rstrip())
            lines.append("")
        if step_fn is not None:
            # Inline the step function body into the C-ABI entry point; the
            # Python signature `simulation_step(buf, dt)` maps buf -> SimulationBuffer*.
            body_stmts = [s for s in (step_fn.get("body") or []) if s]
            body = "".join(self.emit_stmt(s, indent=1) for s in body_stmts)
            if not body.strip():
                body = "    (void)buf;\n"
            lines.append("extern \"C\" void execute_step(SimulationBuffer* buf) {")
            # Local alias so Python-level `dt` parameter references resolve.
            lines.append("    const double dt = buf->dt;")
            lines.append(body.rstrip("\n"))
            lines.append("    buf->steps += 1;")
            lines.append("}")
        return "\n".join(lines) + "\n"

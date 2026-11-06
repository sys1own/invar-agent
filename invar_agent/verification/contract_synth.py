"""Contract synthesizer: derives VerificationContract obligations from source heuristics."""
from __future__ import annotations

import ast
import re
from typing import Optional

from invar_agent.contracts import VerificationContract


class ContractSynthesizer:
    """Heuristically synthesizes pre/post conditions for a function source."""

    _DIV_RE = re.compile(r"(/|//|%)")
    _COMPARE_RE = re.compile(r"(<=|>=|==|!=|<|>)")

    def synthesize_pre_post_conditions(self, func_source: str) -> VerificationContract:
        preconditions: list[str] = []
        postconditions: list[str] = []
        chiasmus_pairs: list[tuple[str, str]] = []
        concolic_targets: list[str] = []
        sandbox_cmd: Optional[str] = None

        try:
            tree = ast.parse(func_source)
        except SyntaxError:
            return VerificationContract()

        fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if fn is None:
            return VerificationContract()

        arg_names = [a.arg for a in fn.args.args]
        # Preconditions from guarding comparisons on arguments.
        for node in ast.walk(fn):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
                if node.left.id in arg_names and len(node.ops) == 1 and len(node.comparators) == 1:
                    comp = node.ops[0]
                    right = node.comparators[0]
                    if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)):
                        op = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}[type(comp)]
                        preconditions.append(f"{node.left.id} {op} {right.value}")
            if isinstance(node, ast.Assert):
                # Convert simple `assert x >= 0` into a postcondition clause.
                clause = self._assert_to_clause(node.test)
                if clause:
                    postconditions.append(clause)

        # Division without guard -> require non-zero divisor precondition.
        for node in ast.walk(fn):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                divisor = node.right
                if isinstance(divisor, ast.Name) and divisor.id in arg_names:
                    preconditions.append(f"{divisor.id} != 0")
                    concolic_targets.append(divisor.id)

        # Chiasmus pairs from acquire/release naming conventions.
        calls = {self._dotted(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)}
        for name in sorted(calls):
            tail = name.split(".")[-1]
            if tail == "acquire":
                base = name.rsplit(".", 1)[0]
                release = f"{base}.release"
                if release in calls or any(r.endswith(".release") and r.rsplit(".", 1)[0] == base for r in calls):
                    chiasmus_pairs.append((name, release))

        # Default sandbox verification command.
        sandbox_cmd = "pytest -q"

        # Deduplicate preserving order.
        preconditions = list(dict.fromkeys(preconditions))
        postconditions = list(dict.fromkeys(postconditions))
        return VerificationContract(
            foge_guard_pre=["file_exists('target')"] if False else [],
            smt_preconditions=preconditions,
            smt_postconditions=postconditions,
            chiasmus_pairs=chiasmus_pairs,
            concolic_targets=list(dict.fromkeys(concolic_targets)),
            sandbox_verification_cmd=sandbox_cmd,
        )

    def _assert_to_clause(self, test: ast.expr) -> Optional[str]:
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
            op = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}.get(type(test.ops[0]))
            left = self._expr_to_name(test.left)
            right = self._expr_to_name(test.comparators[0])
            if left and right and op:
                return f"{left} {op} {right}"
        return None

    def _expr_to_name(self, node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return str(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            l = self._expr_to_name(node.left)
            r = self._expr_to_name(node.right)
            if l and r:
                sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}[type(node.op)]
                return f"({l} {sym} {r})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs" and node.args:
            inner = self._expr_to_name(node.args[0])
            if inner:
                return f"Abs({inner})"
        return None

    def _dotted(self, node: ast.expr) -> str:
        parts: list[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))

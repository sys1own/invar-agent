"""Tier-0 Z3 SMT consistency check of verification-contract invariants.

Each invariant is a small Python boolean expression over sandbox observables
(``telemetry['k']``, ``state[i]``, ``initial[i]``, ``trace_max[i]``, ...).
We translate the conjunction of a contract's invariants into Z3 over the
reals, treating every distinct observable term as a free variable. If the
conjunction is UNSAT the contract is self-contradictory and no simulation
can ever satisfy it, so the run is refused *before* any code executes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:  # pragma: no cover - import guard
    import z3  # type: ignore
    HAVE_Z3 = True
except Exception:  # noqa: BLE001
    z3 = None  # type: ignore
    HAVE_Z3 = False


@dataclass
class SMTReport:
    satisfiable: bool
    checked: int
    untranslatable: List[str] = field(default_factory=list)
    unsat_core: List[str] = field(default_factory=list)
    engine: str = "z3"


class _Translator:
    def __init__(self) -> None:
        self.vars: Dict[str, "z3.ArithRef"] = {}
        self._fresh = 0

    def _var(self, key: str):
        if key not in self.vars:
            self.vars[key] = z3.Real(key)
        return self.vars[key]

    def _opaque(self):
        self._fresh += 1
        return z3.Real(f"__opaque_{self._fresh}")

    def term(self, node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return z3.RealVal(repr(float(node.value)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.term(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self.term(node.operand)
        if isinstance(node, ast.BinOp):
            l, r = self.term(node.left), self.term(node.right)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                return l / r
            return self._opaque()
        if isinstance(node, (ast.Subscript, ast.Name, ast.Attribute)):
            return self._var(ast.unparse(node))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "abs" and len(node.args) == 1:
                x = self.term(node.args[0])
                return z3.If(x >= 0, x, -x)
            if name in ("min", "max") and len(node.args) == 2:
                a, b = self.term(node.args[0]), self.term(node.args[1])
                return z3.If(a <= b, a, b) if name == "min" else z3.If(a >= b, a, b)
            if name == "len":
                v = self._var(ast.unparse(node))
                return v
            return self._var(ast.unparse(node))
        raise ValueError(f"untranslatable term {ast.unparse(node)!r}")

    def formula(self, node: ast.AST):
        if isinstance(node, ast.BoolOp):
            parts = [self.formula(v) for v in node.values]
            return z3.And(*parts) if isinstance(node.op, ast.And) else z3.Or(*parts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return z3.Not(self.formula(node.operand))
        if isinstance(node, ast.Compare):
            parts = []
            left = self.term(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = self.term(comp)
                if isinstance(op, ast.Lt):
                    parts.append(left < right)
                elif isinstance(op, ast.LtE):
                    parts.append(left <= right)
                elif isinstance(op, ast.Gt):
                    parts.append(left > right)
                elif isinstance(op, ast.GtE):
                    parts.append(left >= right)
                elif isinstance(op, ast.Eq):
                    parts.append(left == right)
                elif isinstance(op, ast.NotEq):
                    parts.append(left != right)
                else:
                    raise ValueError("unsupported comparison")
                left = right
            return z3.And(*parts) if len(parts) > 1 else parts[0]
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return z3.BoolVal(node.value)
        raise ValueError(f"untranslatable formula {ast.unparse(node)!r}")


def check_invariants_consistent(assertions: List[str],
                                timeout_ms: int = 2000) -> SMTReport:
    """Return whether the conjunction of ``assertions`` is satisfiable."""
    if not HAVE_Z3:
        return SMTReport(satisfiable=True, checked=0,
                         untranslatable=list(assertions), engine="unavailable")
    tr = _Translator()
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    untranslatable: List[str] = []
    labels: Dict[str, str] = {}
    checked = 0
    for idx, text in enumerate(assertions):
        try:
            node = ast.parse(text, mode="eval").body
            f = tr.formula(node)
        except Exception:  # noqa: BLE001 - leave to concolic post-run check
            untranslatable.append(text)
            continue
        label = f"inv_{idx}"
        labels[label] = text
        solver.assert_and_track(f, z3.Bool(label))
        checked += 1
    if checked == 0:
        return SMTReport(satisfiable=True, checked=0, untranslatable=untranslatable)
    result = solver.check()
    if result == z3.unsat:
        core = [labels.get(str(c), str(c)) for c in solver.unsat_core()]
        return SMTReport(satisfiable=False, checked=checked,
                         untranslatable=untranslatable, unsat_core=core)
    # sat or unknown: unknown is treated as "not refuted".
    return SMTReport(satisfiable=True, checked=checked, untranslatable=untranslatable)

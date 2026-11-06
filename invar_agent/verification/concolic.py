"""Bounded concolic (concrete + symbolic) execution engine with z3 path solving."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import z3

from .smt_solver import SMTStatus, SMTSolverBridge, _z3_to_python


@dataclass
class PathCondition:
    expr: z3.BoolRef
    taken: bool
    description: str = ""

    def as_constraint(self) -> z3.BoolRef:
        return self.expr if self.taken else z3.Not(self.expr)

    def negated(self) -> z3.BoolRef:
        return z3.Not(self.as_constraint())


class PathConstraintCollector:
    def __init__(self) -> None:
        self.conditions: list[PathCondition] = []

    def record(self, expr: z3.BoolRef, taken: bool, description: str = "") -> None:
        self.conditions.append(PathCondition(expr=expr, taken=taken, description=description))

    def as_constraints(self) -> list[z3.BoolRef]:
        return [c.as_constraint() for c in self.conditions]


class SymbolicShadowValue:
    """Concrete value carrying a z3 symbolic expression and a constraint sink."""

    __slots__ = ("concrete", "expr", "collector", "name")

    def __init__(
        self,
        concrete: Any,
        expr: z3.ExprRef,
        collector: PathConstraintCollector,
        name: str = "",
    ) -> None:
        self.concrete = concrete
        self.expr = expr
        self.collector = collector
        self.name = name

    # -- arithmetic -------------------------------------------------------------

    def _binop(self, other: "SymbolicShadowValue", op: str) -> "SymbolicShadowValue":
        other_c = other.concrete if isinstance(other, SymbolicShadowValue) else other
        other_e = other.expr if isinstance(other, SymbolicShadowValue) else _to_z3(other)
        if op == "+":
            c, e = self.concrete + other_c, self.expr + other_e
        elif op == "-":
            c, e = self.concrete - other_c, self.expr - other_e
        elif op == "*":
            c, e = self.concrete * other_c, self.expr * other_e
        elif op == "/":
            c, e = self.concrete / other_c, self.expr / other_e
        elif op == "//":
            c, e = self.concrete // other_c, z3.Int2Int(0) if False else self.expr  # symbolic div unsupported
        else:
            raise ValueError(f"Unsupported op {op}")
        return SymbolicShadowValue(c, e, self.collector)

    def __add__(self, other): return self._binop(other, "+")
    def __radd__(self, other): return self._binop(other, "+")
    def __sub__(self, other): return self._binop(other, "-")
    def __rsub__(self, other): return self._binop(other, "-") if False else SymbolicShadowValue(other - self.concrete, _to_z3(other) - self.expr, self.collector)
    def __mul__(self, other): return self._binop(other, "*")
    def __rmul__(self, other): return self._binop(other, "*")
    def __truediv__(self, other): return self._binop(other, "/")
    def __floordiv__(self, other): return self._binop(other, "//")
    def __neg__(self): return SymbolicShadowValue(-self.concrete, -self.expr, self.collector)

    # -- comparisons (record path constraints, return concrete bool) -------------

    def _compare(self, other: "SymbolicShadowValue", op: str) -> bool:
        other_c = other.concrete if isinstance(other, SymbolicShadowValue) else other
        other_e = other.expr if isinstance(other, SymbolicShadowValue) else _to_z3(other)
        sym = {
            "==": self.expr == other_e,
            "!=": self.expr != other_e,
            "<": self.expr < other_e,
            "<=": self.expr <= other_e,
            ">": self.expr > other_e,
            ">=": self.expr >= other_e,
        }[op]
        concrete = {
            "==": self.concrete == other_c,
            "!=": self.concrete != other_c,
            "<": self.concrete < other_c,
            "<=": self.concrete <= other_c,
            ">": self.concrete > other_c,
            ">=": self.concrete >= other_c,
        }[op]
        self.collector.record(sym, bool(concrete), description=f"{self.name} {op} {getattr(other, 'name', other_c)}")
        return bool(concrete)

    def __eq__(self, other): return self._compare(other, "==")
    def __ne__(self, other): return self._compare(other, "!=")
    def __lt__(self, other): return self._compare(other, "<")
    def __le__(self, other): return self._compare(other, "<=")
    def __gt__(self, other): return self._compare(other, ">")
    def __ge__(self, other): return self._compare(other, ">=")

    def __bool__(self):
        return bool(self.concrete)

    def __int__(self):
        return int(self.concrete)

    def __repr__(self):
        return f"Shadow({self.concrete!r}, {self.name})"


def _to_z3(value: Any) -> z3.ExprRef:
    if isinstance(value, SymbolicShadowValue):
        return value.expr
    if isinstance(value, bool):
        return z3.BoolVal(value)
    if isinstance(value, int):
        return z3.IntVal(value)
    if isinstance(value, float):
        return z3.RealVal(value)
    return z3.IntVal(0)


def negate_and_solve_branch(
    path_prefix: list[PathCondition],
    target: PathCondition,
    input_vars: dict[str, z3.ExprRef],
    timeout_ms: int = 2000,
) -> Optional[dict[str, Any]]:
    """Solve ``path_prefix AND NOT(target.as_taken)`` for new concrete inputs."""
    bridge = SMTSolverBridge(timeout_ms)
    constraints = [c.as_constraint() for c in path_prefix] + [target.negated()]
    result = bridge.check_sat(constraints, timeout_ms=timeout_ms, track=input_vars)
    if result.status == SMTStatus.SAT_COUNTEREXAMPLE and result.witness:
        return result.witness
    return None


@dataclass
class ConcolicFinding:
    inputs: dict[str, Any]
    exception: Optional[str] = None
    traceback_tail: str = ""
    path_conditions: list[str] = field(default_factory=list)


@dataclass
class ConcolicReport:
    branches_explored: int = 0
    findings: list[ConcolicFinding] = field(default_factory=list)
    explored_inputs: list[dict[str, Any]] = field(default_factory=list)


class ConcolicExecutor:
    """Bounded DFS concolic exploration of a Python callable."""

    def __init__(self, max_explorations: int = 32, solver_timeout_ms: int = 2000) -> None:
        self.max_explorations = max_explorations
        self.solver_timeout_ms = solver_timeout_ms

    def explore(
        self,
        func: Callable[..., Any],
        input_names: list[str],
        seed_values: dict[str, Any],
        max_depth: int = 6,
    ) -> ConcolicReport:
        report = ConcolicReport()
        input_vars = {name: z3.Int(name) for name in input_names}
        seen: set[tuple] = set()
        queue: list[tuple[list[PathCondition], dict[str, Any]]] = [([], dict(seed_values))]

        while queue and report.branches_explored < self.max_explorations:
            prefix, inputs = queue.pop(0)
            key = tuple(sorted((k, inputs.get(k)) for k in input_names))
            if key in seen:
                continue
            seen.add(key)
            report.explored_inputs.append(dict(inputs))

            collector = PathConstraintCollector()
            shadows = {
                name: SymbolicShadowValue(inputs.get(name, 0), input_vars.get(name, z3.IntVal(0)), collector, name=name)
                for name in input_names
            }
            exc: Optional[BaseException] = None
            try:
                func(**shadows)
            except Exception as e:  # noqa: BLE001 - concolic harness captures all
                exc = e

            if exc is not None:
                report.findings.append(
                    ConcolicFinding(
                        inputs={k: _concrete(v) for k, v in inputs.items()},
                        exception=type(exc).__name__,
                        traceback_tail=str(exc),
                        path_conditions=[c.description for c in collector.conditions],
                    )
                )

            # Fork on each recorded branch condition.
            for i, cond in enumerate(collector.conditions):
                if report.branches_explored >= self.max_explorations:
                    break
                report.branches_explored += 1
                candidate_prefix = list(prefix) + collector.conditions[:i]
                witness = negate_and_solve_branch(
                    candidate_prefix, cond, input_vars, timeout_ms=self.solver_timeout_ms
                )
                if witness is None:
                    continue
                new_inputs = dict(inputs)
                for name in input_names:
                    if name in witness and isinstance(witness[name], int):
                        new_inputs[name] = witness[name]
                new_key = tuple(sorted((k, new_inputs.get(k)) for k in input_names))
                if new_key in seen:
                    continue
                negated_cond = PathCondition(cond.expr, not cond.taken, cond.description)
                queue.append((candidate_prefix + [negated_cond], new_inputs))
        return report


def _concrete(value: Any) -> Any:
    if isinstance(value, SymbolicShadowValue):
        return value.concrete
    return value

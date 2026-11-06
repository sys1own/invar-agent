"""Z3 SMT solver bridge with strict millisecond timeouts and unsat-core tracking."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import z3


class SMTStatus(str, Enum):
    UNSAT_PROVEN = "UNSAT_PROVEN"
    SAT_COUNTEREXAMPLE = "SAT_COUNTEREXAMPLE"
    UNKNOWN = "UNKNOWN"
    TIMEOUT = "TIMEOUT"


@dataclass
class SMTResult:
    status: SMTStatus
    witness: Optional[dict[str, Any]] = None
    unsat_core: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    diagnostics: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == SMTStatus.UNSAT_PROVEN


_TOKEN_RE = re.compile(r"\s*(<=|>=|==|!=|[-+*/()<>]|\(|\)|[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*)")


class SMTSolverBridge:
    """Thin, deterministic wrapper over z3 with per-query millisecond timeouts."""

    def __init__(self, default_timeout_ms: int = 5000) -> None:
        self.default_timeout_ms = default_timeout_ms
        self._solver = z3.Solver()
        self._declared: dict[str, z3.ExprRef] = {}

    # -- environment management -------------------------------------------------

    def reset(self) -> None:
        self._solver = z3.Solver()
        self._declared = {}

    def declare(self, name: str, sort: str = "Int") -> z3.ExprRef:
        if name in self._declared:
            return self._declared[name]
        if sort == "Real":
            var = z3.Real(name)
        else:
            var = z3.Int(name)
        self._declared[name] = var
        return var

    def env(self) -> dict[str, z3.ExprRef]:
        return dict(self._declared)

    # -- clause compilation -----------------------------------------------------

    def compile_clause(self, clause: str, local_env: Optional[dict[str, Any]] = None) -> z3.BoolRef:
        """Compile a predicate string like ``result >= 0`` or ``x != 0`` to z3.

        Supported operators: ==, !=, <, <=, >, >=.  Operands may be integer
        literals, declared variables, or simple calls ``Abs(x)`` / ``len(x)``.
        """
        env: dict[str, Any] = dict(self._declared)
        if local_env:
            env.update(local_env)
        tokens = [m.group(1) for m in _TOKEN_RE.finditer(clause) if m.group(1)]
        if not tokens:
            raise ValueError(f"Empty SMT clause: {clause!r}")

        # Split at the top-level comparison operator.
        cmp_idx = None
        cmp_op = None
        depth = 0
        for i, tok in enumerate(tokens):
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth -= 1
            elif depth == 0 and tok in ("==", "!=", "<=", ">=", "<", ">"):
                cmp_idx, cmp_op = i, tok
                break
        if cmp_idx is None or cmp_idx == 0 or cmp_idx == len(tokens) - 1:
            raise ValueError(f"SMT clause needs exactly one top-level comparison: {clause!r}")
        left_tokens = tokens[:cmp_idx]
        right_tokens = tokens[cmp_idx + 1 :]
        left = _parse_full(left_tokens, env, self)
        right = _parse_full(right_tokens, env, self)
        return _compare(cmp_op, left, right)

    # -- solving -----------------------------------------------------------------

    def check_sat(
        self,
        constraints: list[z3.BoolRef],
        timeout_ms: Optional[int] = None,
        track: Optional[dict[str, z3.ExprRef]] = None,
    ) -> SMTResult:
        """Check satisfiability of ``constraints`` (conjoined).

        ``track`` optionally maps witness names to expressions to extract from
        the model when SAT.
        """
        timeout = timeout_ms if timeout_ms is not None else self.default_timeout_ms
        solver = z3.Solver()
        solver.set("timeout", max(1, int(timeout)))
        for c in constraints:
            solver.add(c)
        start = time.perf_counter()
        res = solver.check()
        elapsed = (time.perf_counter() - start) * 1000.0
        if res == z3.sat:
            model = solver.model()
            witness: dict[str, Any] = {}
            targets = track if track is not None else self._declared
            for name, expr in targets.items():
                try:
                    val = model.eval(expr, model_completion=True)
                    witness[name] = _z3_to_python(val)
                except Exception:
                    witness[name] = None
            return SMTResult(SMTStatus.SAT_COUNTEREXAMPLE, witness=witness, elapsed_ms=elapsed)
        if res == z3.unsat:
            core = [str(c) for c in solver.unsat_core()] if solver.assertions() else []
            return SMTResult(SMTStatus.UNSAT_PROVEN, unsat_core=core, elapsed_ms=elapsed)
        if res == z3.unknown:
            return SMTResult(SMTStatus.TIMEOUT, elapsed_ms=elapsed, diagnostics=["z3 returned unknown (timeout)"])
        return SMTResult(SMTStatus.UNKNOWN, elapsed_ms=elapsed)

    def prove_postcondition(
        self,
        preconditions: list[str],
        postconditions: list[str],
        local_env: Optional[dict[str, Any]] = None,
        timeout_ms: Optional[int] = None,
    ) -> SMTResult:
        """Prove ``preconditions => postconditions`` by checking
        ``preconditions AND NOT(postconditions)`` for unsatisfiability."""
        self.reset()
        pre: list[z3.BoolRef] = []
        for clause in preconditions:
            pre.append(self.compile_clause(clause, local_env))
        # Negate each postcondition and conjoin: violation exists if all hold.
        violations: list[z3.BoolRef] = []
        for clause in postconditions:
            violations.append(z3.Not(self.compile_clause(clause, local_env)))
        constraints = pre + violations
        track = dict(self._declared)
        if local_env:
            for k, v in local_env.items():
                if isinstance(v, z3.ExprRef):
                    track.setdefault(k, v)
        return self.check_sat(constraints, timeout_ms=timeout_ms, track=track)


def _next_is_expr(tokens: list[str], pos: int) -> bool:
    return pos < len(tokens) and tokens[pos] not in (")", "==", "!=", "<=", ">=", "<", ">"
    ) and not re.fullmatch(r"\d+\.?\d*", tokens[pos] or "")


def _arith(op: str, a: z3.ExprRef, b: z3.ExprRef) -> z3.ExprRef:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a / b


def _compare(op: str, a: z3.ExprRef, b: z3.ExprRef) -> z3.BoolRef:
    return {
        "==": a == b,
        "!=": a != b,
        "<": a < b,
        "<=": a <= b,
        ">": a > b,
        ">=": a >= b,
    }[op]


def _parse_full(tokens: list[str], env: dict[str, Any], bridge: SMTSolverBridge) -> z3.ExprRef:
    pos = 0

    def peek() -> Optional[str]:
        return tokens[pos] if pos < len(tokens) else None

    def parse_primary() -> z3.ExprRef:
        nonlocal pos
        tok = peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        pos += 1
        if tok == "(":
            node = parse_sum()
            if peek() == ")":
                pos += 1
            return node
        if re.fullmatch(r"\d+\.?\d*", tok):
            return z3.RealVal(float(tok)) if "." in tok else z3.IntVal(int(tok))
        if tok == "-":
            return -parse_primary()
        if tok == "+":
            return parse_primary()
        if peek() == "(":
            pos += 1
            arg = parse_sum()
            if peek() == ")":
                pos += 1
            if tok == "Abs":
                return z3.Abs(arg)
            if tok == "len":
                return arg
            raise ValueError(f"Unsupported SMT function: {tok}")
        if tok in env:
            val = env[tok]
            if isinstance(val, bool):
                raise ValueError("Boolean used as arithmetic operand")
            if isinstance(val, int):
                return z3.IntVal(val)
            if isinstance(val, float):
                return z3.RealVal(val)
            return val
        return bridge.declare(tok, "Int")

    def parse_prod() -> z3.ExprRef:
        nonlocal pos
        node = parse_primary()
        while peek() in ("*", "/"):
            op = tokens[pos]
            pos += 1
            rhs = parse_primary()
            node = node * rhs if op == "*" else node / rhs
        return node

    def parse_sum() -> z3.ExprRef:
        nonlocal pos
        node = parse_prod()
        while peek() in ("+", "-"):
            op = tokens[pos]
            pos += 1
            rhs = parse_prod()
            node = node + rhs if op == "+" else node - rhs
        return node

    result = parse_sum()
    if pos != len(tokens):
        raise ValueError(f"Could not parse full expression near token {tokens[pos]!r}")
    return result


def _z3_to_python(val: z3.ExprRef) -> Any:
    try:
        if z3.is_int_value(val):
            return val.as_long()
        if z3.is_rational_value(val):
            return val.as_fraction()
        if z3.is_true(val):
            return True
        if z3.is_false(val):
            return False
        return str(val)
    except Exception:
        return str(val)

"""Offline translation validation for trusted, branch-free simulation kernels.

Arithmetic is checked to 8 ULP. Libm kernels use 32 ULP or 2e-14 absolute
error near cancellation/zero. Rust exp/log/sqrt endpoints are expanded by
16 ULP (an empirical platform-libm assumption, not a correct-rounding proof);
sin/cos use their full mathematical codomain. No interval is fitted to the
observed Python or C++ result. Unsupported interval operations fail closed.
"""

from __future__ import annotations

import ast
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from invar_agent import _native
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
from invar_agent.transpiler.frontend import python_source_to_uast


def ulp_distance(a: float, b: float) -> int:
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("ULP distance requires finite outputs")
    def ordered(x):
        bits = struct.unpack(">Q", struct.pack(">d", x))[0]
        return (1 << 63) - (bits & ((1 << 63) - 1)) if bits >> 63 else (1 << 63) + bits
    return abs(ordered(a) - ordered(b))


class _Interval:
    def __init__(self, lo, hi=None):
        self.bounds = (float(lo), float(lo if hi is None else hi))

    def _op(self, op, other=None):
        symbol = lambda name: {"kind": "symbol", "name": name}
        if other is None:
            expr = {"kind": "call", "func": op, "args": [symbol("a")]}
            bounds = {"a": self.bounds}
        else:
            other = other if isinstance(other, _Interval) else _Interval(other)
            expr = {"kind": "binary", "op": op, "left": symbol("a"), "right": symbol("b")}
            bounds = {"a": self.bounds, "b": other.bounds}
        return _Interval(*_native.evaluate_interval_f64(json.dumps(expr), bounds))

    def __add__(self, other): return self._op("+", other)
    def __sub__(self, other): return self._op("-", other)
    def __mul__(self, other): return self._op("*", other)
    def __truediv__(self, other): return self._op("/", other)
    def __pow__(self, other):
        if not isinstance(other,int) or not 0 <= other <= 16: raise ValueError("unsupported interval exponent")
        result = _Interval(1.0)
        for _ in range(other): result = result*self
        return result
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return _Interval(other) - self
    def __rmul__(self, other): return self * other
    def __rtruediv__(self, other): return _Interval(other) / self
    def __neg__(self): return _Interval(-self.bounds[1], -self.bounds[0])
    def __bool__(self): raise TypeError("interval-dependent control flow is unsupported")
    def __eq__(self, other): raise TypeError("interval-dependent comparison is unsupported")
    def __ne__(self, other): raise TypeError("interval-dependent comparison is unsupported")


class _Buffer:
    def __init__(self, data, dt):
        self.data = list(data)
        self.size = len(data)
        self.dt = dt
        self.steps = 0

    def __getitem__(self, index): return self.data[index]
    def __setitem__(self, index, value): self.data[index] = value


@dataclass(frozen=True)
class DifferentialResult:
    python: tuple[float, ...]
    cpp: tuple[float, ...]
    intervals: tuple[tuple[float, float], ...]
    ulps: tuple[int, ...]


def check_kernel(source: str, initial_state: list[float], *, dt: float,
                 steps: int, cache_dir: Path, transcendental: bool = False) -> DifferentialResult:
    """Execute trusted source independently as floats, lowered C-ABI, and intervals.

    Fixed-count loops are supported; data-dependent interval branches are not.
    Monte Carlo callers supply identical pre-generated samples in the buffer.
    Empty state is valid. Compilation/execution/domain errors propagate.
    """
    if steps < 0 or not math.isfinite(dt) or not all(map(math.isfinite, initial_state)):
        raise ValueError("finite inputs and nonnegative steps required")
    uast = python_source_to_uast(source)
    cpp = DynamicCABIBridge(cache_dir).run(
        cpp_code=CppEmitter().emit_module(uast), initial_state=initial_state,
        dt=dt, steps=steps, entry="execute_step")
    namespace = {"math": math}
    exec(compile(source, "<differential-reference>", "exec"), namespace)
    ref = _Buffer(initial_state, dt)
    for _ in range(steps):
        namespace["simulation_step"](ref, dt)

    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    interval_math = SimpleNamespace(**{
        name: (lambda x, name=name: (x if isinstance(x, _Interval) else _Interval(x))._op(name))
        for name in ("exp", "log", "sqrt", "sin", "cos")})
    namespace = {"math": interval_math}
    exec(compile(tree, "<differential-interval>", "exec"), namespace)
    iv = _Buffer([_Interval(x) for x in initial_state], _Interval(dt))
    for _ in range(steps):
        namespace["simulation_step"](iv, _Interval(dt))
    intervals = tuple(x.bounds if isinstance(x, _Interval) else (float(x), float(x)) for x in iv.data)
    outputs = tuple(cpp["state"])
    assert len(ref.data) == len(outputs) == len(intervals)
    distances = tuple(ulp_distance(a, b) for a, b in zip(ref.data, outputs))
    for i, (a, b, (lo, hi), ulps) in enumerate(zip(ref.data, outputs, intervals, distances)):
        assert lo <= a <= hi and lo <= b <= hi, (i, a, b, lo, hi)
        assert ulps <= (32 if transcendental else 8) or (
            transcendental and abs(a - b) <= 2e-14), (i, a, b, ulps)
    return DifferentialResult(tuple(ref.data), outputs, intervals, distances)

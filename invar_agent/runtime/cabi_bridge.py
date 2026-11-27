"""Dynamic C-ABI bridge: compile C++ kernels and execute them via ctypes.

Compilation uses Clang++ ``-O3 -std=c++20 -fPIC -shared`` when available,
falling back to G++. The zero-copy :class:`SimulationBuffer` ctypes struct
mirrors the C ``SimulationBuffer`` layout exactly.

Execution is isolated: each run happens in a short-lived subprocess with
RLIMIT_CPU / RLIMIT_AS limits derived from the verification contract, so a
kernel with an infinite loop or unbounded allocation cannot hang or exhaust
the server. ``-ffast-math`` is deliberately NOT used: IEEE NaN/Inf semantics
are required for divergence detection.
"""

from __future__ import annotations

from collections import OrderedDict
import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

_EXPORTED_SYMBOLS_CACHE_MAX = 512
_EXPORTED_SYMBOLS_CACHE: "OrderedDict[tuple[str, int, int], tuple[str, ...]]" = OrderedDict()
_EXPORTED_SYMBOLS_LOCK = threading.Lock()


def _direct_python_executable() -> str:
    """Resolve the direct base CPython executable for stdlib-only C-ABI runner
    subprocesses, avoiding the Windows .venv redirector double-spawn."""
    from .accel import accel_enabled
    if not accel_enabled():
        return sys.executable
    base = getattr(sys, "_base_executable", None)
    if base and base != sys.executable:
        base_path = Path(base)
        if base_path.is_file():
            try:
                if Path(sys.base_prefix).resolve() in base_path.resolve().parents:
                    return str(base_path)
            except OSError:
                pass
    if os.name == "nt":
        candidate = Path(sys.base_prefix) / "python.exe"
        if candidate.is_file():
            return str(candidate)
    return sys.executable

try:
    import resource
except ImportError:  # Windows has no POSIX resource module
    resource = None


class SimulationBuffer(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("dt", ctypes.c_double),
        ("steps", ctypes.c_uint32),
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("diag", ctypes.POINTER(ctypes.c_double)),
    ]


class KernelExecutionError(RuntimeError):
    """Raised when a kernel fails, diverges, times out or exceeds memory."""

    def __init__(self, message: str, *, timed_out: bool = False,
                 returncode: Optional[int] = None, compile_error: bool = False):
        super().__init__(message)
        self.timed_out = timed_out
        self.returncode = returncode
        # True when the C++ toolchain rejected the kernel. Kernels that already
        # passed PreWrite_Guard may fall back to the deterministic Python runner.
        self.compile_error = compile_error


_RUNNER_SOURCE = r'''
import ctypes, json, math, os, sys, time

if hasattr(os, "add_dll_directory"):
    for _p in os.environ.get("PATH", "").split(os.pathsep):
        if _p and os.path.isdir(_p):
            try:
                os.add_dll_directory(_p)
            except OSError:
                pass

so_path, entry, initial_json, dt, steps = (
    sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))

class SimulationBuffer(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("dt", ctypes.c_double),
                ("steps", ctypes.c_uint32), ("data", ctypes.POINTER(ctypes.c_double)),
                ("diag", ctypes.POINTER(ctypes.c_double))]

initial = json.loads(initial_json)
n = len(initial)
alloc_n = max(n, 64)
padded = list(initial) + [0.0] * (alloc_n - n)
data = (ctypes.c_double * alloc_n)(*padded)
diag = (ctypes.c_double * 8)()
buf = SimulationBuffer(size=n, dt=dt, steps=0,
                       data=ctypes.cast(data, ctypes.POINTER(ctypes.c_double)), diag=diag)
lib = ctypes.CDLL(so_path)
fn = getattr(lib, entry)
fn.argtypes = [ctypes.POINTER(SimulationBuffer)]
fn.restype = None
trace_min = list(initial)
trace_max = list(initial)
trace_slots = json.loads(sys.argv[6]) if len(sys.argv) > 6 else []
trace_values = []
first_nonfinite = -1
t0 = time.perf_counter()
for step in range(steps):
    diag[6] = diag[7] = 0.0
    fn(ctypes.byref(buf))
    if trace_slots:
        trace_values.append([data[i] for i in trace_slots])
    diag[0] += 1
    diag[3 if diag[7] else 2 if diag[6] > 1e-12 else 1] += 1
    diag[5] = max(diag[5], diag[6])
    for i in range(n):
        v = data[i]
        if v < trace_min[i]:
            trace_min[i] = v
        if v > trace_max[i]:
            trace_max[i] = v
        if first_nonfinite < 0 and not math.isfinite(v):
            first_nonfinite = step
    if first_nonfinite >= 0:
        break
state = list(data)[:n]
finite = all(math.isfinite(v) for v in state)
print(json.dumps({"state": state, "steps": int(buf.steps), "finite": finite,
                  "diag": list(diag),
                  "execution_time_ms": (time.perf_counter() - t0) * 1000.0,
                  "trace_min": trace_min, "trace_max": trace_max,
                  "trace_values": trace_values,
                  "first_nonfinite_step": first_nonfinite}))
'''


# -- deterministic Python kernel fallback -------------------------------------
# Used when the C++ toolchain rejects a kernel that already passed PreWrite_Guard
# AST validation: the same numerical physics runs in an isolated Python
# subprocess instead of burning Tier-2 LLM retries on a compiler diagnostic.
_PY_RUNNER_SOURCE = r'''
import json, math, sys, time


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


def clip(x, lo, hi):
    return clamp(x, lo, hi)


def sign(x):
    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


def sgn(x):
    return sign(x)


def sq(x):
    return x * x


def lerp(a, b, t):
    return a + t * (b - a)


def safe_div(a, b):
    return a / (b if abs(b) > 1e-30 else 1e-30)


_MASK64 = (1 << 64) - 1


def invar_splitmix64_stateless(seed, step, channel):
    s = int(seed) & _MASK64
    st = int(step) & _MASK64
    ch = int(channel) & _MASK64
    z = (s + ((st * 0x9E3779B97F4A7C15) & _MASK64) + ((ch * 0xBF58476D1CE4E5B9) & _MASK64)) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def invar_rand_u01(seed, step, channel):
    val = invar_splitmix64_stateless(seed, step, channel)
    u = (val >> 11) * (1.0 / 9007199254740992.0)
    return 1e-15 if u <= 1e-15 else (1.0 - 1e-15 if u >= 1.0 - 1e-15 else u)


def invar_rand_normal(seed, step, channel):
    ch = int(channel) & _MASK64
    u1 = invar_rand_u01(seed, step, (ch * 2) & _MASK64)
    u2 = invar_rand_u01(seed, step, (ch * 2 + 1) & _MASK64)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(6.2831853071795864769 * u2)


def invar_rand_poisson(seed, step, lam):
    if lam <= 0.0:
        return 0.0
    if lam < 30.0:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L and k < 256:
            k += 1
            p *= invar_rand_u01(seed, step, 1000 + k)
        return float(k - 1)
    z = invar_rand_normal(seed, step, 999)
    res = math.floor(lam + math.sqrt(lam) * z + 0.5)
    return 0.0 if res < 0.0 else res


def invar_rand_binomial(seed, step, n, p):
    if n <= 0.0 or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return math.floor(n)
    if n < 64.0:
        hits = 0
        ni = int(n)
        for i in range(ni):
            if invar_rand_u01(seed, step, 2000 + i) < p:
                hits += 1
        return float(hits)
    mu = n * p
    sigma = math.sqrt(n * p * (1.0 - p))
    z = invar_rand_normal(seed, step, 1999)
    val = math.floor(mu + sigma * z + 0.5)
    n_floor = math.floor(n)
    return 0.0 if val < 0.0 else (n_floor if val > n_floor else val)


def welford_mean_update(mean, x, k):
    return mean + (x - mean) / (1.0 if k < 1.0 else k)


def welford_m2_update(m2, mean_old, mean_new, x):
    return m2 + (x - mean_old) * (x - mean_new)


def welford_cov_update(c_xy, mean_x_old, x, mean_y_new, y):
    return c_xy + (x - mean_x_old) * (y - mean_y_new)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x * 0.70710678118654752440))


def norm_ppf(p):
    pc = 1e-15 if p <= 1e-15 else (1.0 - 1e-15 if p >= 1.0 - 1e-15 else p)
    a1, a2, a3, a4, a5, a6 = -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
    b1, b2, b3, b4, b5 = -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01
    c1, c2, c3, c4, c5, c6 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
    d1, d2, d3, d4 = 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00
    p_low = 0.02425
    p_high = 1.0 - p_low
    if pc < p_low:
        q = math.sqrt(-2.0 * math.log(pc))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif pc <= p_high:
        q = pc - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - pc))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


def logistic_sigmoid(x):
    xc = 80.0 if x > 80.0 else (-80.0 if x < -80.0 else x)
    return 1.0 / (1.0 + math.exp(-xc))


def logit(p):
    pc = 1e-15 if p < 1e-15 else (1.0 - 1e-15 if p > 1.0 - 1e-15 else p)
    return math.log(pc / (1.0 - pc))


def hill_eq(c, ec50, n):
    if c <= 0.0:
        return 0.0
    cn = (c if c > 0.0 else 0.0) ** n
    kn = (ec50 if ec50 > 1e-15 else 1e-15) ** n
    return cn / (cn + kn + 1e-30)


def michaelis_menten(s, vmax, km):
    sc = s if s > 0.0 else 0.0
    return (vmax * sc) / ((km if km > 1e-15 else 1e-15) + sc)


def kl_term(p, q):
    qc = q if q > 1e-30 else 1e-30
    if p <= 0.0:
        return qc
    val = p * math.log(p / qc) - p + qc
    return val if val > 0.0 else 0.0


def shannon_term(p):
    if p <= 0.0:
        return 0.0
    return -p * math.log(p)


def gammainc_reg(a, x):
    ac = a if a > 1e-12 else 1e-12
    if x <= 0.0:
        return 0.0
    ratio = (x / ac) ** (1.0 / 3.0)
    mu = 1.0 - 1.0 / (9.0 * ac)
    sigma = math.sqrt(1.0 / (9.0 * ac))
    return norm_cdf((ratio - mu) / (sigma if sigma > 1e-15 else 1e-15))


def betainc_reg(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ac = a if a > 1e-12 else 1e-12
    bc = b if b > 1e-12 else 1e-12
    mean = ac / (ac + bc)
    var = (ac * bc) / ((ac + bc) * (ac + bc) * (ac + bc + 1.0))
    sigma = math.sqrt(var)
    return norm_cdf((x - mean) / (sigma if sigma > 1e-15 else 1e-15))


for _fn_name, _fn_obj in [
    ("invar_splitmix64_stateless", invar_splitmix64_stateless),
    ("invar_rand_u01", invar_rand_u01),
    ("invar_rand_normal", invar_rand_normal),
    ("invar_rand_poisson", invar_rand_poisson),
    ("invar_rand_binomial", invar_rand_binomial),
    ("welford_mean_update", welford_mean_update),
    ("welford_m2_update", welford_m2_update),
    ("welford_cov_update", welford_cov_update),
    ("norm_cdf", norm_cdf),
    ("norm_ppf", norm_ppf),
    ("logistic_sigmoid", logistic_sigmoid),
    ("logit", logit),
    ("hill_eq", hill_eq),
    ("michaelis_menten", michaelis_menten),
    ("kl_term", kl_term),
    ("shannon_term", shannon_term),
    ("gammainc_reg", gammainc_reg),
    ("betainc_reg", betainc_reg),
]:
    setattr(math, _fn_name, _fn_obj)


class _NumpyShim:
    pi = math.pi
    e = math.e

    def __getattr__(self, name):
        if name == "clip":
            return clip
        if name == "abs":
            return abs
        if name in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
                    "atan2", "sinh", "cosh", "tanh", "exp", "log", "log2",
                    "log10", "floor", "ceil", "fabs", "isfinite", "isnan",
                    "isinf"):
            return getattr(math, name)
        raise AttributeError(name)


np = _NumpyShim()
numpy = np

py_path, entry, initial_json, dt, steps = (
    sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))


class SimulationBuffer:
    """Lightweight buf wrapper exposing .data/.size/.dt/.steps to kernels."""

    def __init__(self, data, dt):
        self.data = list(data)
        self.size = len(self.data)
        self.dt = dt
        self.steps = 0
        self.diag = [0.0] * 8

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value


src = open(py_path, encoding="utf-8").read()
namespace = {"__name__": "invar_kernel_fallback", "math": math,
             "clamp": clamp, "clip": clip, "sign": sign, "sgn": sgn,
             "sq": sq, "lerp": lerp, "safe_div": safe_div,
             "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
             "tan": math.tan, "asin": math.asin, "acos": math.acos,
             "atan": math.atan, "atan2": math.atan2, "sinh": math.sinh,
             "cosh": math.cosh, "tanh": math.tanh, "exp": math.exp,
             "log": math.log, "log2": math.log2, "log10": math.log10,
             "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs,
             "isfinite": math.isfinite, "isnan": math.isnan,
             "isinf": math.isinf, "pi": math.pi, "e": math.e,
             "np": np, "numpy": numpy}
exec(compile(src, py_path, "exec"), namespace)  # noqa: S102 - guard-passed kernel
fn = namespace.get(entry)
if fn is None:
    raise SystemExit(f"entry function {entry!r} not found in module")

initial = [float(v) for v in json.loads(initial_json)]
orig_n = len(initial)
if "[62]" in src and len(initial) < 64:
    initial = initial + [0.0] * (64 - len(initial))
buf = SimulationBuffer(initial, dt)
trace_min = list(initial[:orig_n])
trace_max = list(initial[:orig_n])
trace_slots = json.loads(sys.argv[6]) if len(sys.argv) > 6 else []
trace_values = []
first_nonfinite = -1
start = time.perf_counter()
for step in range(steps):
    prev_steps = buf.steps
    buf.diag[6] = buf.diag[7] = 0.0
    fn(buf, dt)
    if trace_slots:
        trace_values.append([buf.data[i] for i in trace_slots])
    buf.diag[0] += 1
    buf.diag[3 if buf.diag[7] else 2 if buf.diag[6] > 1e-12 else 1] += 1
    buf.diag[5] = max(buf.diag[5], buf.diag[6])
    if buf.steps == prev_steps:
        buf.steps += 1
    for i in range(orig_n):
        v = buf.data[i]
        if v < trace_min[i]:
            trace_min[i] = v
        if v > trace_max[i]:
            trace_max[i] = v
        if first_nonfinite < 0 and not math.isfinite(v):
            first_nonfinite = step
    if first_nonfinite >= 0:
        break
run_ms = (time.perf_counter() - start) * 1000.0
state = [float(v) for v in buf.data[:orig_n]]
finite = all(math.isfinite(v) for v in state)
print(json.dumps({"state": state, "steps": int(buf.steps), "finite": finite,
                  "diag": buf.diag,
                  "trace_values": trace_values,
                  "trace_min": trace_min, "trace_max": trace_max,
                  "first_nonfinite_step": first_nonfinite,
                  "execution_time_ms": run_ms}))
'''


CUSTOM_DISCRETE_MAP = "custom_discrete_map"
RK4_ODE = "rk4_ode"
ARCHETYPE_DYNAMICAL_STEPPER = "archetype_dynamical_stepper"
ARCHETYPE_MONTE_CARLO_ENSEMBLE = "archetype_monte_carlo_ensemble"
ARCHETYPE_EQUILIBRIUM_FIXED_POINT = "archetype_equilibrium_fixed_point"
ARCHETYPE_PARAMETER_BIFURCATION_SWEEP = "archetype_parameter_bifurcation_sweep"
EXECUTION_ARCHETYPES = (
    ARCHETYPE_DYNAMICAL_STEPPER,
    ARCHETYPE_MONTE_CARLO_ENSEMBLE,
    ARCHETYPE_EQUILIBRIUM_FIXED_POINT,
    ARCHETYPE_PARAMETER_BIFURCATION_SWEEP,
)

# Kernel execution modes. ``rk4_ode`` is the classical ODE trajectory mode;
# ``custom_discrete_map`` executes the researcher's exact discrete update law
# z[k+1] = Phi_dt(z[k]) (discrete gradients, symplectic/metriplectic maps,
# automata) — the bridge loop calls the exported step function exactly once
# per step in both modes, so a custom map is a first-class kernel, never
# wrapped in a hardcoded RK4 stepper.
KERNEL_MODES = (RK4_ODE, CUSTOM_DISCRETE_MAP) + EXECUTION_ARCHETYPES


class DynamicCABIBridge:
    """Compiles C++ source to a shared library and drives entry points."""

    def __init__(self, workspace: Path, compiler: Optional[str] = None):
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.compiler = compiler or self._detect_compiler()
        self._cache: Dict[str, Path] = {}

    @staticmethod
    def _detect_compiler() -> str:
        for cand in ("clang++", "g++", "c++"):
            if shutil.which(cand):
                return cand
        raise RuntimeError("no C++ compiler found (clang++/g++/c++)")

    @staticmethod
    def cache_key(cpp_code: str) -> str:
        return hashlib.blake2b(cpp_code.encode(), digest_size=16).hexdigest()

    def compile(self, cpp_code: str) -> Path:
        from .accel import accel_enabled
        key = self.cache_key(cpp_code)
        if accel_enabled():
            cached = self._cache.get(key)
            if cached and cached.exists():
                return cached
        cpp_path = (self.workspace / f"kernel_{key}.cpp").resolve()
        so_path = (self.workspace / f"kernel_{key}.so").resolve()
        cpp_path.write_text(cpp_code, encoding="utf-8")
        cmd = [self.compiler, "-O3", "-std=c++20", "-fPIC", "-shared",
               "-Wl,--no-undefined",
               str(cpp_path), "-o", str(so_path)]
        if "clang" in self.compiler:
            cmd[1:1] = ["-march=native"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except OSError as exc:
            cpp_path.unlink(missing_ok=True)
            raise KernelExecutionError(
                f"C-ABI compilation failed ({self.compiler}):\n{exc}",
                compile_error=True) from exc
        if proc.returncode != 0:
            cpp_path.unlink(missing_ok=True)
            raise KernelExecutionError(
                f"C-ABI compilation failed ({self.compiler}):\n"
                f"{proc.stderr.strip()[:2000]}", compile_error=True)
        if not so_path.is_file():
            raise KernelExecutionError(
                f"C-ABI compiler did not produce shared object: {so_path}",
                compile_error=True)
        self._cache[key] = so_path
        return so_path

    def discard(self, cpp_code: str) -> None:
        """Unlink the compiled artifacts of a rejected candidate.

        Used by Adjoint_Commit_or_Rollback: only verified stages keep their
        ``.cpp``/``.so``/``.dylib`` kernels in the bridge workspace.
        """
        key = self.cache_key(cpp_code)
        self._cache.pop(key, None)
        for suffix in (".cpp", ".so", ".dylib"):
            try:
                (self.workspace / f"kernel_{key}{suffix}").unlink(missing_ok=True)
            except OSError:
                pass

    def exported_symbols(self, so_path: Path) -> List[str]:
        """List exported function symbols via ``nm -D`` (empty if unavailable).

        Results are memoized in a bounded thread-safe LRU keyed by
        ``(os.path.realpath(path), stat.st_size, stat.st_mtime_ns)`` so any
        modified binary misses immediately while repeated executions of the
        same compiled library avoid redundant ``nm`` subprocess spawns.
        """
        nm = shutil.which("nm")
        if nm is None:
            return []
        cache_key: Optional[tuple[str, int, int]] = None
        try:
            real_path = os.path.realpath(str(so_path))
            st = os.stat(real_path)
            cache_key = (real_path, st.st_size, st.st_mtime_ns)
        except OSError:
            cache_key = None

        from .accel import accel_enabled
        use_cache = accel_enabled()
        if use_cache and cache_key is not None:
            with _EXPORTED_SYMBOLS_LOCK:
                cached = _EXPORTED_SYMBOLS_CACHE.get(cache_key)
                if cached is not None:
                    _EXPORTED_SYMBOLS_CACHE.move_to_end(cache_key)
                    return list(cached)

        proc = subprocess.run([nm, "-D", "--defined-only", str(so_path)],
                              capture_output=True, text=True, timeout=30)
        symbols: List[str] = []
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("T", "W"):
                symbols.append(parts[2])
        frozen = tuple(symbols)
        if use_cache and cache_key is not None:
            with _EXPORTED_SYMBOLS_LOCK:
                _EXPORTED_SYMBOLS_CACHE[cache_key] = frozen
                _EXPORTED_SYMBOLS_CACHE.move_to_end(cache_key)
                while len(_EXPORTED_SYMBOLS_CACHE) > _EXPORTED_SYMBOLS_CACHE_MAX:
                    _EXPORTED_SYMBOLS_CACHE.popitem(last=False)
        return list(frozen)

    @staticmethod
    def resolve_entry(entry: str, kernel_mode: str = RK4_ODE) -> str:
        """Normalise the exported entry symbol for a kernel mode.

        Custom discrete-map kernels may export ``execute_custom_step`` or
        ``step_custom`` in addition to the canonical ``execute_step``; both
        name the same per-step update law z[k+1] = Phi_dt(z[k]).
        """
        if kernel_mode not in KERNEL_MODES:
            raise ValueError(f"unknown kernel_mode {kernel_mode!r}")
        if entry in ("execute_custom_step", "step_custom"):
            return "execute_custom_step"
        return entry

    def run(self, cpp_code: str, initial_state: List[float], dt: float,
            steps: int, entry: str = "execute_step",
            timeout_seconds: float = 30.0,
            max_memory_mb: int = 1024,
            isolated: bool = True,
            kernel_mode: str = RK4_ODE,
            trace_slots: tuple[int, ...] = ()) -> Dict[str, object]:
        """Compile (cached) and run ``steps`` iterations of ``entry``.

        Returns ``state``, ``steps``, ``execution_time_ms`` (kernel wall time
        inside the sandbox), ``run_ms`` (subprocess wall time), ``finite`` and
        ``bundle_hash``. Raises :class:`KernelExecutionError` on compile
        failure, missing entry point, timeout, memory exhaustion or a
        non-finite final state is reported via ``finite=False`` instead.

        ``kernel_mode`` selects the stepping archetype: ``rk4_ode`` (default)
        or ``custom_discrete_map`` for researcher-defined discrete update
        laws. Both modes drive the exported C++ step function once per step;
        no RK4 wrapper is ever injected around a custom map.
        """
        entry = self.resolve_entry(entry, kernel_mode)
        if any(type(i) is not int or not 0 <= i < len(initial_state) for i in trace_slots):
            raise ValueError("trace slots must address the initial state")
        if trace_slots and not isolated:
            raise ValueError("trajectory capture requires isolated execution")
        start = time.perf_counter()
        so_path = self.compile(cpp_code)
        compile_ms = (time.perf_counter() - start) * 1000.0
        if not isolated:
            return self._run_inprocess(so_path, entry, initial_state, dt, steps,
                                       compile_ms, self.cache_key(cpp_code))
        return self._run_subprocess(so_path, entry, initial_state, dt, steps,
                                    timeout_seconds, max_memory_mb, compile_ms,
                                    self.cache_key(cpp_code), trace_slots)

    # -- in-process (tests / trusted callers) ------------------------------
    def _run_inprocess(self, so_path: Path, entry: str, initial_state,
                       dt, steps, compile_ms, key) -> Dict[str, object]:
        so_path = Path(so_path).resolve()
        if not so_path.is_file():
            raise KernelExecutionError(f"compiled kernel missing: {so_path}")
        import os
        if hasattr(os, "add_dll_directory"):
            for _p in os.environ.get("PATH", "").split(os.pathsep):
                if _p and os.path.isdir(_p):
                    try:
                        os.add_dll_directory(_p)
                    except OSError:
                        pass
        lib = ctypes.CDLL(str(so_path))
        try:
            fn = getattr(lib, entry, None)
            if fn is None:
                raise KernelExecutionError(f"entry point {entry} not exported")
            fn.argtypes = [ctypes.POINTER(SimulationBuffer)]
            fn.restype = None
            n = len(initial_state)
            alloc_n = max(n, 64)
            padded = list(initial_state) + [0.0] * (alloc_n - n)
            data = (ctypes.c_double * alloc_n)(*padded)
            diag = (ctypes.c_double * 8)()
            buf = SimulationBuffer(size=n, dt=dt, steps=0, data=data, diag=diag)
            run_start = time.perf_counter()
            for _ in range(steps):
                diag[6] = diag[7] = 0.0
                fn(ctypes.byref(buf))
                diag[0] += 1
                diag[3 if diag[7] else 2 if diag[6] > 1e-12 else 1] += 1
                diag[5] = max(diag[5], diag[6])
            run_ms = (time.perf_counter() - run_start) * 1000.0
            state = list(data)[:n]
            finite = all(math.isfinite(v) for v in state)
            return {
                "state": state,
                "diag": list(diag),
                "steps": int(buf.steps),
                "finite": finite,
                "execution_time_ms": run_ms,
                "run_ms": run_ms + compile_ms,
                "compile_ms": compile_ms,
                "bundle_hash": key,
            }
        finally:
            handle = getattr(lib, "_handle", None)
            del lib
            if handle is not None:
                try:
                    import _ctypes
                    if os.name == "nt":
                        _ctypes.FreeLibrary(handle)
                    else:
                        _ctypes.dlclose(handle)
                except Exception:  # noqa: BLE001
                    pass

    def cleanup_scratch(self) -> None:
        """Remove temporary C-ABI build files in cache_dir to prevent disk/handle leaks."""
        cache = getattr(self, "cache_dir", None)
        if cache is None or not Path(cache).is_dir():
            return
        for p in Path(cache).iterdir():
            if p.is_file() and p.suffix in (".cpp", ".so", ".dll", ".pyd", ".exp", ".lib", ".o"):
                try:
                    p.unlink()
                except OSError:
                    pass

    # -- isolated subprocess (Sandbox_Execute) ------------------------------
    def _run_subprocess(self, so_path: Path, entry: str, initial_state,
                        dt, steps, timeout_seconds, max_memory_mb,
                        compile_ms, key, trace_slots=()) -> Dict[str, object]:
        so_path = Path(so_path).resolve()
        if not so_path.is_file():
            raise KernelExecutionError(f"compiled kernel missing: {so_path}")
        symbols = self.exported_symbols(so_path)
        if symbols and entry not in symbols:
            raise KernelExecutionError(
                f"entry point {entry} not exported by kernel "
                f"(available: {', '.join(symbols[:8])})")
        run_dir = Path(tempfile.mkdtemp(prefix="invar-kernel-"))
        try:
            runner = run_dir / "runner.py"
            runner.write_text(_RUNNER_SOURCE, encoding="utf-8")
            cpu_limit = max(5, int(math.ceil(timeout_seconds)) + 5)
            mem_bytes = max(256, int(max_memory_mb)) * 1024 * 1024 + 512 * 1024 * 1024
            # 64-bit CPython + libstdc++ virtual mappings exceed 256MB even
            # with tiny physical RSS; floor RLIMIT_AS so dlopen() never
            # fails with ENOMEM ("failed to map segment from shared object").
            as_bytes = max(mem_bytes, 2048 * 1024 * 1024)

            def _limits() -> None:
                if resource is not None:
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 5))
                    resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))

            preexec = _limits if (sys.platform != "win32" and resource is not None) else None
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    [_direct_python_executable(), "-I", str(runner), str(so_path), entry,
                     json.dumps([float(v) for v in initial_state]), str(float(dt)), str(int(steps)), json.dumps(trace_slots)],
                    capture_output=True, text=True, timeout=timeout_seconds + 10,
                    preexec_fn=preexec, cwd=str(run_dir))
            except subprocess.TimeoutExpired as exc:
                raise KernelExecutionError(
                    f"kernel execution timed out after {timeout_seconds:.1f}s",
                    timed_out=True) from exc
            wall_ms = (time.perf_counter() - start) * 1000.0
        finally:
            # The runner script and any subprocess scratch are ephemeral.
            shutil.rmtree(run_dir, ignore_errors=True)
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-1500:]
            lowered = tail.lower()
            if "memoryerror" in lowered or "cannot allocate" in lowered \
                    or "bad_alloc" in lowered:
                raise KernelExecutionError(
                    f"kernel exceeded memory budget ({max_memory_mb} MB):\n{tail}")
            raise KernelExecutionError(
                f"kernel execution failed (rc={proc.returncode}):\n{tail}",
                returncode=proc.returncode)
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise KernelExecutionError(
                f"kernel produced no valid result: {(proc.stdout or '')[-500:]}") from exc
        state = [float(v) for v in payload["state"]]
        finite = bool(payload.get("finite", True)) and all(
            math.isfinite(v) for v in state)
        exec_ms = max(0.0, float(payload.get("execution_time_ms", wall_ms)))
        return {
            "state": state,
            "trace_values": payload.get("trace_values", []),
            "steps": int(payload.get("steps", steps)),
            "diag": [float(v) for v in payload["diag"]],
            "finite": finite,
            "trace_min": [float(v) for v in payload.get("trace_min", state)],
            "trace_max": [float(v) for v in payload.get("trace_max", state)],
            "first_nonfinite_step": int(payload.get("first_nonfinite_step", -1)),
            "execution_time_ms": exec_ms,
            "run_ms": wall_ms,
            "compile_ms": compile_ms,
            "bundle_hash": key,
        }

    # -- deterministic Python fallback (Sandbox_Execute) ----------------------
    @staticmethod
    def python_entry_name(entry: str) -> str:
        """Map a C-ABI entry symbol back to the module-level Python function."""
        if entry in ("execute_step", "execute_custom_step", "step_custom"):
            return "simulation_step"
        if entry.startswith("execute_"):
            return entry[len("execute_"):]
        return entry

    def run_python(self, python_code: str, initial_state: List[float], dt: float,
                   steps: int, entry: str = "execute_step",
                   timeout_seconds: float = 30.0,
                   max_memory_mb: int = 1024,
                   kernel_mode: str = RK4_ODE,
                   trace_slots: tuple[int, ...] = ()) -> Dict[str, object]:
        """Execute the guard-passed Python kernel in an isolated subprocess.

        Deterministic fallback for C-ABI compilation failures: runs
        ``simulation_step(buf, dt)`` (or the mapped entry function) for exactly
        ``steps`` iterations against a :class:`SimulationBuffer` wrapper and
        returns the same result shape as :meth:`run`.
        """
        start = time.perf_counter()
        if any(type(i) is not int or not 0 <= i < len(initial_state) for i in trace_slots):
            raise ValueError("trace slots must address the initial state")
        entry = self.resolve_entry(entry, kernel_mode)
        func_name = self.python_entry_name(entry)
        key = hashlib.blake2b(python_code.encode(), digest_size=16).hexdigest()
        run_dir = Path(tempfile.mkdtemp(prefix="invar-pykernel-"))
        try:
            kernel_path = run_dir / "kernel.py"
            kernel_path.write_text(python_code, encoding="utf-8")
            runner = run_dir / "runner.py"
            runner.write_text(_PY_RUNNER_SOURCE, encoding="utf-8")
            cpu_limit = max(5, int(math.ceil(timeout_seconds)) + 5)
            mem_bytes = max(256, int(max_memory_mb)) * 1024 * 1024 + 512 * 1024 * 1024
            as_bytes = max(mem_bytes, 2048 * 1024 * 1024)

            def _limits() -> None:
                if resource is not None:
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 5))
                    resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))

            preexec = _limits if (sys.platform != "win32" and resource is not None) else None
            try:
                proc = subprocess.run(
                    [sys.executable, str(runner), str(kernel_path), func_name,
                     json.dumps([float(v) for v in initial_state]),
                     str(float(dt)), str(int(steps)), json.dumps(trace_slots)],
                    capture_output=True, text=True, timeout=timeout_seconds + 10,
                    preexec_fn=preexec, cwd=str(run_dir))
            except subprocess.TimeoutExpired as exc:
                raise KernelExecutionError(
                    f"python kernel fallback timed out after {timeout_seconds:.1f}s",
                    timed_out=True) from exc
            wall_ms = (time.perf_counter() - start) * 1000.0
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-1500:]
            lowered = tail.lower()
            if "memoryerror" in lowered or "cannot allocate" in lowered:
                raise KernelExecutionError(
                    f"python kernel fallback exceeded memory budget "
                    f"({max_memory_mb} MB):\n{tail}")
            raise KernelExecutionError(
                f"python kernel fallback failed (rc={proc.returncode}):\n{tail}",
                returncode=proc.returncode)
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise KernelExecutionError(
                f"python kernel fallback produced no valid result: "
                f"{(proc.stdout or '')[-500:]}") from exc
        state = [float(v) for v in payload["state"]]
        finite = bool(payload.get("finite", True)) and all(
            math.isfinite(v) for v in state)
        run_ms = max(0.0, float(payload.get("execution_time_ms", 0.0)))
        return {
            "state": state,
            "trace_values": payload.get("trace_values", []),
            "steps": int(payload.get("steps", steps)),
            "diag": [float(v) for v in payload["diag"]],
            "finite": finite,
            "trace_min": [float(v) for v in payload.get("trace_min", state)],
            "trace_max": [float(v) for v in payload.get("trace_max", state)],
            "first_nonfinite_step": int(payload.get("first_nonfinite_step", -1)),
            "execution_time_ms": run_ms,
            "run_ms": wall_ms,
            "compile_ms": 0.0,
            "fallback": "python",
            "bundle_hash": key,
        }

    # -- legacy ctypes helpers (kept for API compatibility) ------------------
    def load(self, so_path: Path):
        lib = ctypes.CDLL(str(so_path))
        lib.execute_step.argtypes = [ctypes.POINTER(SimulationBuffer)]
        lib.execute_step.restype = None
        return lib

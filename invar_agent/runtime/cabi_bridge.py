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

import ctypes
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

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
import ctypes, json, math, sys

so_path, entry, initial_json, dt, steps = (
    sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))

class SimulationBuffer(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("dt", ctypes.c_double),
                ("steps", ctypes.c_uint32), ("data", ctypes.POINTER(ctypes.c_double))]

initial = json.loads(initial_json)
n = len(initial)
data = (ctypes.c_double * n)(*initial)
buf = SimulationBuffer(size=n, dt=dt, steps=0,
                       data=ctypes.cast(data, ctypes.POINTER(ctypes.c_double)))
lib = ctypes.CDLL(so_path)
fn = getattr(lib, entry)
fn.argtypes = [ctypes.POINTER(SimulationBuffer)]
fn.restype = None
trace_min = list(initial)
trace_max = list(initial)
first_nonfinite = -1
for step in range(steps):
    fn(ctypes.byref(buf))
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
state = list(data)
finite = all(math.isfinite(v) for v in state)
print(json.dumps({"state": state, "steps": int(buf.steps), "finite": finite,
                  "trace_min": trace_min, "trace_max": trace_max,
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
buf = SimulationBuffer(initial, dt)
trace_min = list(initial)
trace_max = list(initial)
first_nonfinite = -1
start = time.perf_counter()
for step in range(steps):
    fn(buf, dt)
    for i in range(buf.size):
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
state = [float(v) for v in buf.data]
finite = all(math.isfinite(v) for v in state)
print(json.dumps({"state": state, "steps": int(buf.steps), "finite": finite,
                  "trace_min": trace_min, "trace_max": trace_max,
                  "first_nonfinite_step": first_nonfinite,
                  "execution_time_ms": run_ms}))
'''


CUSTOM_DISCRETE_MAP = "custom_discrete_map"
RK4_ODE = "rk4_ode"

# Kernel execution modes. ``rk4_ode`` is the classical ODE trajectory mode;
# ``custom_discrete_map`` executes the researcher's exact discrete update law
# z[k+1] = Phi_dt(z[k]) (discrete gradients, symplectic/metriplectic maps,
# automata) — the bridge loop calls the exported step function exactly once
# per step in both modes, so a custom map is a first-class kernel, never
# wrapped in a hardcoded RK4 stepper.
KERNEL_MODES = (RK4_ODE, CUSTOM_DISCRETE_MAP)


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
        key = self.cache_key(cpp_code)
        cached = self._cache.get(key)
        if cached and cached.exists():
            return cached
        cpp_path = (self.workspace / f"kernel_{key}.cpp").resolve()
        so_path = (self.workspace / f"kernel_{key}.so").resolve()
        cpp_path.write_text(cpp_code)
        cmd = [self.compiler, "-O3", "-std=c++20", "-fPIC", "-shared",
               "-Wl,--no-undefined",
               str(cpp_path), "-o", str(so_path)]
        if "clang" in self.compiler:
            cmd[1:1] = ["-march=native"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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
        """List exported function symbols via ``nm -D`` (empty if unavailable)."""
        nm = shutil.which("nm")
        if nm is None:
            return []
        proc = subprocess.run([nm, "-D", "--defined-only", str(so_path)],
                              capture_output=True, text=True, timeout=30)
        symbols = []
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("T", "W"):
                symbols.append(parts[2])
        return symbols

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
            kernel_mode: str = RK4_ODE) -> Dict[str, object]:
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
        start = time.perf_counter()
        so_path = self.compile(cpp_code)
        compile_ms = (time.perf_counter() - start) * 1000.0
        if not isolated:
            return self._run_inprocess(so_path, entry, initial_state, dt, steps,
                                       compile_ms, self.cache_key(cpp_code))
        return self._run_subprocess(so_path, entry, initial_state, dt, steps,
                                    timeout_seconds, max_memory_mb, compile_ms,
                                    self.cache_key(cpp_code))

    # -- in-process (tests / trusted callers) ------------------------------
    def _run_inprocess(self, so_path: Path, entry: str, initial_state,
                       dt, steps, compile_ms, key) -> Dict[str, object]:
        so_path = Path(so_path).resolve()
        if not so_path.is_file():
            raise KernelExecutionError(f"compiled kernel missing: {so_path}")
        lib = ctypes.CDLL(str(so_path))
        fn = getattr(lib, entry, None)
        if fn is None:
            raise KernelExecutionError(f"entry point {entry} not exported")
        fn.argtypes = [ctypes.POINTER(SimulationBuffer)]
        fn.restype = None
        n = len(initial_state)
        data = (ctypes.c_double * n)(*initial_state)
        buf = SimulationBuffer(size=n, dt=dt, steps=0, data=data)
        run_start = time.perf_counter()
        for _ in range(steps):
            fn(ctypes.byref(buf))
        run_ms = (time.perf_counter() - run_start) * 1000.0
        state = list(data)
        finite = all(math.isfinite(v) for v in state)
        return {
            "state": state,
            "steps": int(buf.steps),
            "finite": finite,
            "execution_time_ms": run_ms,
            "run_ms": run_ms + compile_ms,
            "compile_ms": compile_ms,
            "bundle_hash": key,
        }

    # -- isolated subprocess (Sandbox_Execute) ------------------------------
    def _run_subprocess(self, so_path: Path, entry: str, initial_state,
                        dt, steps, timeout_seconds, max_memory_mb,
                        compile_ms, key) -> Dict[str, object]:
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
            runner.write_text(_RUNNER_SOURCE)
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
                    [sys.executable, str(runner), str(so_path), entry,
                     json.dumps([float(v) for v in initial_state]), str(float(dt)), str(int(steps))],
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
        run_ms = wall_ms - compile_ms
        return {
            "state": state,
            "steps": int(payload.get("steps", steps)),
            "finite": finite,
            "trace_min": [float(v) for v in payload.get("trace_min", state)],
            "trace_max": [float(v) for v in payload.get("trace_max", state)],
            "first_nonfinite_step": int(payload.get("first_nonfinite_step", -1)),
            "execution_time_ms": run_ms,
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
                   kernel_mode: str = RK4_ODE) -> Dict[str, object]:
        """Execute the guard-passed Python kernel in an isolated subprocess.

        Deterministic fallback for C-ABI compilation failures: runs
        ``simulation_step(buf, dt)`` (or the mapped entry function) for exactly
        ``steps`` iterations against a :class:`SimulationBuffer` wrapper and
        returns the same result shape as :meth:`run`.
        """
        start = time.perf_counter()
        entry = self.resolve_entry(entry, kernel_mode)
        func_name = self.python_entry_name(entry)
        key = hashlib.blake2b(python_code.encode(), digest_size=16).hexdigest()
        run_dir = Path(tempfile.mkdtemp(prefix="invar-pykernel-"))
        try:
            kernel_path = run_dir / "kernel.py"
            kernel_path.write_text(python_code)
            runner = run_dir / "runner.py"
            runner.write_text(_PY_RUNNER_SOURCE)
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
                     str(float(dt)), str(int(steps))],
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
        run_ms = float(payload.get("execution_time_ms", 0.0))
        return {
            "state": state,
            "steps": int(payload.get("steps", steps)),
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

"""ARCH-005 Real Execution Isolation: AST allow-list + OS-enforced child process sandbox.

Eliminates in-process `exec` of generated or client-supplied code and replaces
fragile deny-lists with a strict AST allow-list plus OS process/memory/CPU limits
(Windows Job Objects / POSIX rlimits + strict audit hook + minimal environment).
"""
from __future__ import annotations

import ast
from collections import deque
import gc
import inspect
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, TypeVar

_T = TypeVar("_T")


class SolverBusyError(RuntimeError):
    """Raised when the Z3 actor queue is saturated."""
    code = "SOLVER_BUSY"


class SolverTimeoutError(TimeoutError):
    """Raised when a Z3 actor task exceeds its deadline."""
    code = "SOLVER_TIMEOUT"


_Z3_LOCK_DEPTH = threading.local()


class _TrackedZ3Lock:
    """Legacy reentrant lock wrapper that tracks per-thread acquisition depth
    so run_on_z3_thread can assert no caller holds it across the actor boundary."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        ok = self._lock.acquire(blocking, timeout)
        if ok:
            _Z3_LOCK_DEPTH.depth = getattr(_Z3_LOCK_DEPTH, "depth", 0) + 1
        return ok

    def release(self) -> None:
        self._lock.release()
        _Z3_LOCK_DEPTH.depth = max(0, getattr(_Z3_LOCK_DEPTH, "depth", 0) - 1)

    def __enter__(self) -> "_TrackedZ3Lock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


# Process-wide dedicated single-thread actor for Z3 (ARCH-003: lock-free actor).
# On Windows, Z3's global C++ context is thread-affine and crashes with
# access violations / `b'unreachable'` / 0xC0000374 heap corruption when
# Z3 objects are created, used, or finalized across multiple OS threads.
Z3_GLOBAL_LOCK = _TrackedZ3Lock()
_Z3_THREAD_NAME = "invar-z3-worker"
_Z3_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix=_Z3_THREAD_NAME)
_Z3_WORKER_TID: Optional[int] = None
_Z3_CALL_DEPTH = threading.local()
_Z3_DEFERRED_RELEASES: deque[tuple[Callable[[Any], Any], Any]] = deque()
_Z3_DEFERRED_COUNT = 0
_Z3_FINALIZERS_INSTALLED = False
_Z3_INSTALL_LOCK = threading.Lock()
_Z3_DRAIN_SCHEDULED = False
_Z3_PENDING_LOCK = threading.Lock()
_Z3_PENDING_TIMESTAMPS: Dict[int, float] = {}
_Z3_TASK_SEQ = 0


def get_z3_solver_status() -> Dict[str, Any]:
    """Return live Z3 actor queue telemetry for /api/status."""
    now = time.monotonic()
    with _Z3_PENDING_LOCK:
        pending = len(_Z3_PENDING_TIMESTAMPS)
        oldest = min(_Z3_PENDING_TIMESTAMPS.values()) if _Z3_PENDING_TIMESTAMPS else None
    queue_age_s = round(max(0.0, now - oldest), 3) if oldest is not None else 0.0
    return {
        "queue_age_s": queue_age_s,
        "pending": pending,
        "deferred_releases": _Z3_DEFERRED_COUNT,
    }


def get_z3_worker_tid() -> Optional[int]:
    """Return the OS thread ident of the dedicated Z3 worker, or None if not yet started."""
    return _Z3_WORKER_TID


def is_on_z3_thread() -> bool:
    """Return True if the caller is currently executing on the dedicated Z3 worker thread."""
    tid = _Z3_WORKER_TID
    if tid is not None:
        return threading.get_ident() == tid
    return threading.current_thread().name.startswith(_Z3_THREAD_NAME)


def get_z3_deferred_release_count() -> int:
    """Return total count of Z3 finalizers intercepted on foreign threads and deferred."""
    return _Z3_DEFERRED_COUNT


def _drain_z3_deferred_releases() -> int:
    """Execute any deferred Z3 C-handle releases on the dedicated Z3 worker thread."""
    global _Z3_DRAIN_SCHEDULED
    drained = 0
    while _Z3_DEFERRED_RELEASES:
        try:
            orig_del, obj = _Z3_DEFERRED_RELEASES.popleft()
        except IndexError:
            break
        try:
            orig_del(obj)
        except Exception:
            pass
        drained += 1
    _Z3_DRAIN_SCHEDULED = False
    return drained


def _install_z3_finalizer_confinement() -> None:
    """Wrap Z3 wrapper ``__del__`` methods so C-level ``Z3_*_dec_ref`` calls never
    execute on a foreign thread (e.g. during cyclic GC on MainThread)."""
    global _Z3_FINALIZERS_INSTALLED
    if _Z3_FINALIZERS_INSTALLED:
        return
    with _Z3_INSTALL_LOCK:
        if _Z3_FINALIZERS_INSTALLED:
            return
        try:
            import z3  # type: ignore
        except Exception:
            _Z3_FINALIZERS_INSTALLED = True
            return

        for _name, cls in inspect.getmembers(z3, inspect.isclass):
            if "__del__" not in cls.__dict__:
                continue
            orig_del = cls.__dict__["__del__"]
            if getattr(orig_del, "_invar_z3_confined", False):
                continue

            def _make_confined_del(orig: Callable[[Any], Any]) -> Callable[[Any], None]:
                def _confined_del(self: Any) -> None:
                    global _Z3_DEFERRED_COUNT
                    worker_tid = _Z3_WORKER_TID
                    if (
                        worker_tid is None
                        or threading.get_ident() == worker_tid
                        or sys.is_finalizing()
                    ):
                        orig(self)
                        return
                    # Lock-free handoff only: never call _Z3_EXECUTOR.submit()
                    # from inside __del__, because cyclic GC can trigger while
                    # ThreadPoolExecutor._shutdown_lock is held on this thread.
                    _Z3_DEFERRED_COUNT += 1
                    _Z3_DEFERRED_RELEASES.append((orig, self))

                _confined_del._invar_z3_confined = True  # type: ignore[attr-defined]
                return _confined_del

            try:
                setattr(cls, "__del__", _make_confined_del(orig_del))
            except (TypeError, AttributeError):
                pass
        _Z3_FINALIZERS_INSTALLED = True


def _clear_exception_frames(exc: BaseException) -> None:
    """Clear all local variables from exception tracebacks while still on the Z3 thread
    so no frame holding Z3 objects can cross the thread boundary."""
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        tb = cur.__traceback__
        if tb is not None:
            traceback.clear_frames(tb)
            cur.__traceback__ = None
        cur = cur.__cause__ or cur.__context__


def _run_confined(fn: Callable[..., _T], args: tuple[Any, ...], kwargs: dict[str, Any]) -> _T:
    global _Z3_WORKER_TID
    if _Z3_WORKER_TID is None:
        _Z3_WORKER_TID = threading.get_ident()
    _install_z3_finalizer_confinement()

    depth = getattr(_Z3_CALL_DEPTH, "depth", 0)
    _Z3_CALL_DEPTH.depth = depth + 1
    if depth == 0:
        _drain_z3_deferred_releases()
    try:
        return fn(*args, **kwargs)
    except BaseException as exc:
        _clear_exception_frames(exc)
        raise
    finally:
        _Z3_CALL_DEPTH.depth = depth
        if depth == 0:
            _drain_z3_deferred_releases()
            if gc.isenabled():
                gc.collect(0)
            _drain_z3_deferred_releases()


def run_on_z3_thread(
    fn: Callable[..., _T],
    *args: Any,
    deadline: Optional[float] = None,
    **kwargs: Any,
) -> _T:
    """Execute a Z3 computation on the dedicated single-threaded Z3 OS actor.

    ARCH-003: Lock-free actor submission. Raises RuntimeError if called from a
    non-Z3 thread while holding the legacy ``Z3_GLOBAL_LOCK``, and raises
    ``SolverTimeoutError`` (SOLVER_TIMEOUT) if ``deadline`` (monotonic seconds)
    expires.
    """
    global _Z3_TASK_SEQ
    if is_on_z3_thread():
        return _run_confined(fn, args, kwargs)
    if getattr(_Z3_LOCK_DEPTH, "depth", 0) > 0:
        raise RuntimeError(
            "Z3_LOCK_INVERSION: cannot submit to Z3 actor while holding Z3_GLOBAL_LOCK"
        )
    with _Z3_PENDING_LOCK:
        _Z3_TASK_SEQ += 1
        task_id = _Z3_TASK_SEQ
        _Z3_PENDING_TIMESTAMPS[task_id] = time.monotonic()

    def _tracked_run() -> _T:
        try:
            return _run_confined(fn, args, kwargs)
        finally:
            with _Z3_PENDING_LOCK:
                _Z3_PENDING_TIMESTAMPS.pop(task_id, None)

    future = _Z3_EXECUTOR.submit(_tracked_run)
    timeout: Optional[float] = None
    if deadline is not None:
        timeout = max(0.0, float(deadline) - time.monotonic())
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        future.cancel()
        raise SolverTimeoutError("SOLVER_TIMEOUT: Z3 actor deadline exceeded") from exc


def _direct_python_executable() -> str:
    """Resolve the direct CPython executable (bypassing uv trampolines on Windows)
    so ActiveProcessLimit=1 in a Win32 Job Object allows Python itself to run
    while blocking all child process creation."""
    if os.name == "nt":
        candidate = Path(sys.base_prefix) / "python.exe"
        if candidate.is_file():
            return str(candidate)
    return sys.executable


class IsolationViolation(ValueError):
    """Raised when candidate code violates the AST allow-list or OS sandbox."""


ALLOWED_MATH_ATTRS: frozenset[str] = frozenset(
    name for name in dir(math) if not name.startswith("_")
)

ALLOWED_BUILTIN_CALLS: frozenset[str] = frozenset(
    {"abs", "min", "max", "int", "float", "bool", "range", "len", "round", "sum", "pow"}
)

ALLOWED_LITERAL_NAMES: frozenset[str] = frozenset({"True", "False", "None"})

ALLOWED_BUF_ATTRS: frozenset[str] = frozenset({"data", "size"})

ALLOWED_AST_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Return,
    ast.If,
    ast.For,
    ast.While,
    ast.Expr,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Import,
    ast.ImportFrom,
    ast.alias,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.IfExp,
    ast.Tuple,
    ast.List,
    ast.ListComp,
    ast.comprehension,
    ast.Load,
    ast.Store,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

FORBIDDEN_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "INVAR_API_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "GITHUB_TOKEN",
)


def _collect_assigned_names(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, ast.arg):
            names.add(child.arg)
        elif isinstance(child, ast.FunctionDef):
            names.add(child.name)
    return names


def validate_kernel_ast_allowlist(
    code: str,
    *,
    require_simulation_step: bool = True,
    extra_allowed_names: Optional[Iterable[str]] = None,
) -> List[str]:
    """Strict AST allow-list validator for candidate scientific kernels.

    Every AST node must belong to `ALLOWED_AST_NODES`. Every attribute access
    must target `math.<public>` or `buf.{data,size}` (zero dunder/private access).
    Every loaded name must resolve to declared parameters, locals, top-level
    module definitions, `math` (or imported `math` symbols), or safe scalar builtins.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]

    violations: List[str] = []
    imported_math_names: Set[str] = set()
    module_defined: Set[str] = set()

    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            module_defined.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                module_defined |= _collect_assigned_names(tgt)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            module_defined.add(stmt.target.id)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name != "math":
                    violations.append(f"forbidden import: {alias.name}")
                else:
                    imported_math_names.add(alias.asname or "math")
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.module != "math" or (stmt.level or 0) != 0:
                violations.append(f"forbidden import: {stmt.module}")
            else:
                for alias in stmt.names:
                    if alias.name == "*" or alias.name not in ALLOWED_MATH_ATTRS:
                        violations.append(f"forbidden math import: {alias.name}")
                    else:
                        imported_math_names.add(alias.asname or alias.name)

    # Walk all nodes for structural allow-list & dunder blocking
    forbidden_call_names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            if isinstance(node, ast.ClassDef):
                violations.append(f"forbidden construct: class {node.name}")
            elif isinstance(node, ast.Try):
                violations.append("forbidden construct: try/except")
            elif isinstance(node, ast.With):
                violations.append("forbidden construct: with")
            elif isinstance(node, ast.Lambda):
                violations.append("forbidden construct: lambda")
            else:
                violations.append(f"forbidden construct: {type(node).__name__}")
            continue

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "math":
                    violations.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module != "math" or (node.level or 0) != 0:
                violations.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars", "breakpoint", "input"}:
                violations.append(f"forbidden call: {node.func.id}()")
                forbidden_call_names.add(node.func.id)
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                violations.append(f"forbidden dunder name: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                violations.append(f"forbidden attribute access: .{node.attr}")
            elif isinstance(node.value, ast.Name):
                owner = node.value.id
                if owner in imported_math_names or owner == "math":
                    if node.attr not in ALLOWED_MATH_ATTRS:
                        violations.append(f"forbidden math attribute: math.{node.attr}")
                elif owner == "buf":
                    if node.attr not in ALLOWED_BUF_ATTRS:
                        violations.append(
                            f"unsupported buffer attribute buf.{node.attr} at line {node.lineno} (ONLY buf.data[i] and buf.size are supported)"
                        )
                else:
                    violations.append(f"forbidden attribute owner: {owner}.{node.attr}")
            else:
                violations.append(f"forbidden chained attribute access: .{node.attr}")
        elif isinstance(node, ast.While):
            if (
                isinstance(node.test, ast.Constant)
                and node.test.value is True
                and not any(isinstance(n, ast.Break) for n in ast.walk(node))
            ):
                violations.append("unbounded while True loop without break")

    # Nested functions forbidden
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        for inner in ast.walk(fn):
            if inner is not fn and isinstance(inner, ast.FunctionDef):
                violations.append(f"forbidden construct: nested function {inner.name}")

    # Name resolution check per function & module level
    base_allowed = (
        ALLOWED_BUILTIN_CALLS
        | ALLOWED_LITERAL_NAMES
        | {"math"}
        | imported_math_names
        | module_defined
        | forbidden_call_names
        | set(extra_allowed_names or ())
    )
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        fn_locals = _collect_assigned_names(fn)
        allowed = base_allowed | fn_locals
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in allowed:
                    violations.append(f"unresolved or forbidden name: {node.id}")

    # Module-level non-function expressions
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in base_allowed:
                    violations.append(f"unresolved or forbidden name: {node.id}")

    if require_simulation_step and not any(
        isinstance(n, ast.FunctionDef) and n.name == "simulation_step"
        for n in tree.body
    ):
        violations.append("module must define simulation_step(buf, dt)")

    return list(dict.fromkeys(violations))


# -- OS Job Object enforcement on Windows ------------------------------------

class _WindowsJobObject:
    """Thin Win32 Job Object wrapper enforcing CPU, memory, and process-count caps."""

    def __init__(
        self,
        *,
        memory_limit_mb: int = 512,
        cpu_time_limit_s: int = 15,
        active_process_limit: int = 1,
    ) -> None:
        self._handle = None
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("WriteTransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
            JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
            JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_PROCESS_TIME
            if active_process_limit > 0:
                flags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                info.BasicLimitInformation.ActiveProcessLimit = active_process_limit
            if memory_limit_mb > 0:
                flags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
                mem_bytes = memory_limit_mb * 1024 * 1024
                info.ProcessMemoryLimit = mem_bytes
                info.JobMemoryLimit = mem_bytes
            info.BasicLimitInformation.PerProcessUserTimeLimit = int(cpu_time_limit_s * 10_000_000)
            info.BasicLimitInformation.LimitFlags = flags

            JobObjectExtendedLimitInformation = 9
            ok = kernel32.SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                kernel32.CloseHandle(job)
                return
            self._handle = job
        except Exception:
            self._handle = None

    def assign(self, proc: subprocess.Popen) -> bool:
        if not self._handle or os.name != "nt":
            return False
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            PROCESS_SET_QUOTA = 0x0100
            PROCESS_TERMINATE = 0x0001
            h_proc = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, proc.pid)
            if not h_proc:
                return False
            try:
                return bool(kernel32.AssignProcessToJobObject(self._handle, h_proc))
            finally:
                kernel32.CloseHandle(h_proc)
        except Exception:
            return False

    def close(self) -> None:
        if self._handle and os.name == "nt":
            try:
                import ctypes

                ctypes.windll.kernel32.CloseHandle(self._handle)
            except Exception:
                pass
            self._handle = None

    def __enter__(self) -> "_WindowsJobObject":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def attach_os_limits(
    proc: subprocess.Popen,
    *,
    memory_limit_mb: int = 512,
    cpu_time_limit_s: int = 30,
    active_process_limit: int = 4,
) -> _WindowsJobObject:
    """Attach OS-level resource limits to an already-spawned child process."""
    job = _WindowsJobObject(
        memory_limit_mb=memory_limit_mb,
        cpu_time_limit_s=cpu_time_limit_s,
        active_process_limit=active_process_limit,
    )
    job.assign(proc)
    return job


def _sanitized_child_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    for key in ("SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "COMSPEC", "TEMP", "TMP"):
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for secret_key in FORBIDDEN_ENV_KEYS:
        env.pop(secret_key, None)
    return env


def run_isolated_kernel_smoke(
    code: str,
    *,
    dt: float = 0.01,
    timeout_s: float = 8.0,
    memory_limit_mb: int = 256,
    validate_ast: bool = True,
    mode: str = "kernel_smoke",
) -> Dict[str, Any]:
    """Execute a 1-step kernel smoke test in an OS-isolated child process.

    Never executes candidate code in the server interpreter.
    """
    if validate_ast:
        violations = validate_kernel_ast_allowlist(
            code, require_simulation_step=(mode == "kernel_smoke")
        )
        if violations:
            raise IsolationViolation("; ".join(violations))

    max_idx = max([int(m) for m in re.findall(r"buf\.data\[(\d+)\]", code)] + [31])
    buf_size = min(4096, max_idx + 1)
    worker_path = Path(__file__).resolve().parent / "_smoke_worker.py"
    payload = json.dumps(
        {
            "code": code,
            "mode": mode,
            "dt": dt,
            "buf_size": buf_size,
            "timeout_s": min(timeout_s, 6.0),
            "memory_mb": memory_limit_mb,
        },
        allow_nan=False,
    )

    with tempfile.TemporaryDirectory(prefix="invar_iso_") as empty_cwd:
        with _WindowsJobObject(
            memory_limit_mb=memory_limit_mb,
            cpu_time_limit_s=max(3, int(timeout_s) + 2),
            active_process_limit=1,
        ) as job:
            proc = subprocess.Popen(
                [_direct_python_executable(), "-I", str(worker_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=empty_cwd,
                env=_sanitized_child_env(),
            )
            job.assign(proc)
            try:
                stdout, stderr = proc.communicate(payload, timeout=timeout_s + 2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise TimeoutError(
                    "python_smoke exceeded safety step/time budget (possible infinite loop)"
                )

    for line in reversed((stdout or "").splitlines()):
        if line.startswith("INVAR_ISOLATION="):
            result = json.loads(line[len("INVAR_ISOLATION=") :])
            if not result.get("ok"):
                err_type = result.get("error_type", "RuntimeError")
                err_msg = result.get("error", "isolated smoke failed")
                if err_type == "TimeoutError":
                    raise TimeoutError(err_msg)
                if err_type == "PermissionError":
                    raise PermissionError(err_msg)
                raise RuntimeError(f"{err_type}: {err_msg}")
            return result

    raise RuntimeError(
        f"isolated smoke worker terminated abnormally (rc={proc.returncode}): {(stderr or '')[:500]}"
    )

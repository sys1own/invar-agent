"""Killable process-based step worker with Windows Job Object / POSIX process-group cleanup (UPG-018)."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import importlib
import multiprocessing
import os
import pickle
import queue
import signal
import sys
import traceback
from typing import Any


class StepTimeout(RuntimeError):
    """Raised when a step exceeds its wall-clock budget and its worker tree is killed."""


def _resolve_factory(factory_path: str) -> Any:
    if ":" not in factory_path:
        raise ValueError(f"Invalid factory path {factory_path!r}; expected 'module:callable'")
    mod_name, attr_name = factory_path.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr_name)
    return obj() if callable(obj) else obj


def _worker_main(factory_path: str, req_q: Any, resp_q: Any) -> None:
    if os.name != "nt":
        try:
            os.setsid()
        except OSError:
            pass
    # Block until parent has attached Job Object (Windows) or recorded PGID (POSIX)
    msg = req_q.get()
    if msg != ("__go__", None):
        return
    try:
        target = _resolve_factory(factory_path)
        resp_q.put(("__ready__", os.getpid()))
    except BaseException as exc:  # noqa: BLE001
        resp_q.put(("__init_error__", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        return

    while True:
        try:
            item = req_q.get()
        except (EOFError, OSError):
            break
        if item is None or item[0] == "__shutdown__":
            break
        method_name, payload_bytes = item
        try:
            args, kwargs = pickle.loads(payload_bytes)
            fn = getattr(target, method_name)
            result = fn(*args, **kwargs)
            resp_q.put(("ok", pickle.dumps(result)))
        except BaseException as exc:  # noqa: BLE001
            resp_q.put(("err", pickle.dumps((type(exc).__name__, str(exc), traceback.format_exc()))))


# Windows Job Object helpers (KILL_ON_JOB_CLOSE only, no memory/process limits)
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JobObjectExtendedLimitInformation = 9
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
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


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _attach_kill_on_close_job(pid: int) -> int | None:
    if os.name != "nt":
        return None
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.OpenProcess.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        job,
        _JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        kernel32.CloseHandle(job)
        return None
    proc_h = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, int(pid))
    if not proc_h:
        kernel32.CloseHandle(job)
        return None
    try:
        if not kernel32.AssignProcessToJobObject(job, proc_h):
            kernel32.CloseHandle(job)
            return None
    finally:
        kernel32.CloseHandle(proc_h)
    return int(job)


def _close_job_handle(job_handle: int | None) -> None:
    if os.name == "nt" and job_handle:
        try:
            ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(job_handle))
        except Exception:
            pass


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = wintypes.HANDLE
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, int(pid))
        if not h:
            return False
        try:
            code = wintypes.DWORD(0)
            if kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
                return code.value == 259  # STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class StepWorker:
    """Persistent spawn()ed process that runs picklable callables and can be killed with its children."""

    def __init__(self, factory: str):
        self.factory = factory
        self._ctx = multiprocessing.get_context("spawn")
        self._proc: Any = None
        self._req_q: Any = None
        self._resp_q: Any = None
        self._job_handle: int | None = None
        self.pid: int | None = None

    def warm(self, timeout_s: float = 30.0) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._cleanup_handles()
        req_q = self._ctx.Queue()
        resp_q = self._ctx.Queue()
        proc = self._ctx.Process(
            target=_worker_main,
            args=(self.factory, req_q, resp_q),
            daemon=False,
        )
        proc.start()
        assert proc.pid is not None
        self.pid = int(proc.pid)
        job = _attach_kill_on_close_job(self.pid)
        self._proc = proc
        self._req_q = req_q
        self._resp_q = resp_q
        self._job_handle = job
        req_q.put(("__go__", None))
        try:
            status, detail = resp_q.get(timeout=timeout_s)
        except queue.Empty as exc:
            self._kill_tree()
            raise StepTimeout(f"StepWorker failed to warm within {timeout_s:.1f}s") from exc
        if status != "__ready__":
            self._kill_tree()
            raise RuntimeError(f"StepWorker initialization failed: {detail}")

    def run(self, method: str, payload: bytes, timeout_s: float) -> bytes:
        if timeout_s <= 0.0:
            raise StepTimeout("Step budget exhausted before dispatch")
        self.warm()
        assert self._req_q is not None and self._resp_q is not None
        self._req_q.put((method, payload))
        try:
            status, body = self._resp_q.get(timeout=timeout_s)
        except queue.Empty as exc:
            self._kill_tree()
            raise StepTimeout(f"Step {method!r} exceeded timeout ({timeout_s:.2f}s)") from exc
        if status == "ok":
            return body
        exc_name, exc_msg, tb = pickle.loads(body)
        raise RuntimeError(f"StepWorker {method} raised {exc_name}: {exc_msg}\n{tb}")

    def _kill_tree(self) -> None:
        job = self._job_handle
        self._job_handle = None
        proc = self._proc
        pid = self.pid
        if os.name == "nt":
            if job is not None:
                _close_job_handle(job)
            if proc is not None and proc.is_alive():
                try:
                    proc.kill()
                except Exception:
                    pass
        else:
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    if proc is not None and proc.is_alive():
                        proc.kill()
        if proc is not None:
            try:
                proc.join(timeout=5.0)
            except Exception:
                pass
        self._cleanup_handles()

    def _cleanup_handles(self) -> None:
        if self._job_handle is not None:
            _close_job_handle(self._job_handle)
            self._job_handle = None
        for q in (self._req_q, self._resp_q):
            if q is not None:
                try:
                    q.close()
                    q.cancel_join_thread()
                except Exception:
                    pass
        self._req_q = None
        self._resp_q = None
        self._proc = None

    def close(self) -> None:
        proc = self._proc
        if proc is not None and proc.is_alive():
            try:
                if self._req_q is not None:
                    self._req_q.put(("__shutdown__", None))
                proc.join(timeout=2.0)
            except Exception:
                pass
            if proc.is_alive():
                self._kill_tree()
                return
        self._cleanup_handles()

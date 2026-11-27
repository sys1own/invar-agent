"""Isolated child-process smoke worker for candidate kernel execution.

No secrets, no network, no filesystem writes, no subprocess spawning,
and a strictly restricted builtins dictionary.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict


class _BufView:
    __slots__ = ("data", "size")

    def __init__(self, data: list[float]) -> None:
        self.data = data
        self.size = len(data)


def _apply_posix_rlimits(memory_mb: int, cpu_s: int) -> None:
    if os.name == "nt":
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ValueError, OSError):
            pass
        mem_bytes = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
    except Exception:
        pass


def _install_strict_audit_hook() -> None:
    def _audit(event: str, args: tuple[Any, ...]) -> None:
        if event in {
            "open",
            "socket.connect",
            "socket.bind",
            "socket.getaddrinfo",
            "socket.sendto",
            "socket.socket",
            "ctypes.dlopen",
            "ctypes.dlsym",
            "winreg.OpenKey",
            "winreg.CreateKey",
            "winreg.SetValue",
        } or event.startswith(
            (
                "subprocess.",
                "os.exec",
                "os.spawn",
                "os.system",
                "os.fork",
                "os.popen",
                "os.remove",
                "os.rmdir",
                "os.mkdir",
                "os.rename",
                "os.chmod",
                "os.symlink",
                "os.link",
                "shutil.",
                "webbrowser.",
                "urllib.",
                "http.",
            )
        ):
            raise PermissionError(f"ISOLATION_DENIED: {event}")

    sys.addaudithook(_audit)


def _safe_import(name: str, globals_=None, locals_=None, fromlist=(), level: int = 0):
    if name == "math" and level == 0:
        return math
    raise ImportError(f"ISOLATION_IMPORT_DENIED: {name}")


SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "bool": bool,
    "range": range,
    "len": len,
    "round": round,
    "sum": sum,
    "pow": pow,
    "True": True,
    "False": False,
    "None": None,
    "__import__": _safe_import,
}


def main() -> int:
    raw = sys.stdin.read()
    payload = json.loads(raw)
    code: str = str(payload["code"])
    mode: str = str(payload.get("mode", "kernel_smoke"))
    dt: float = float(payload.get("dt", 0.01))
    buf_size: int = max(32, min(4096, int(payload.get("buf_size", 64))))
    timeout_s: float = float(payload.get("timeout_s", 6.0))
    memory_mb: int = int(payload.get("memory_mb", 256))
    cpu_s: int = max(2, int(timeout_s) + 2)

    _apply_posix_rlimits(memory_mb, cpu_s)
    _install_strict_audit_hook()

    ns: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "math": math}
    t0 = time.perf_counter()
    steps = 0

    def _trace(frame, event, arg):
        nonlocal steps
        steps += 1
        if steps > 25_000_000 or ((steps & 4095) == 0 and (time.perf_counter() - t0) > timeout_s):
            raise TimeoutError("python_smoke exceeded safety step/time budget (possible infinite loop)")
        return _trace

    try:
        compiled = compile(code, "<isolated_smoke>", "exec")
        sys.settrace(_trace)
        try:
            exec(compiled, ns, ns)
            if mode == "kernel_smoke":
                fn = ns.get("simulation_step")
                if callable(fn):
                    buf = _BufView([0.5] * buf_size)
                    fn(buf, dt)
        finally:
            sys.settrace(None)
        out = {"ok": True, "elapsed_s": round(time.perf_counter() - t0, 4)}
    except BaseException as exc:
        out = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }

    sys.stdout.write("INVAR_ISOLATION=" + json.dumps(out, allow_nan=False) + "\n")
    sys.stdout.flush()
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

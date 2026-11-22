"""Isolated builder verifier entry point. No agent/provider imports are permitted."""
from __future__ import annotations

import _ctypes
import ctypes
import gc
import importlib
import importlib.util
import inspect
import json
import math
import os
import socket
import sys
import tempfile
import traceback
from pathlib import Path


def main(config):
    root = Path(config["root"]).resolve()
    sandbox_tmp = root / ".sandbox_tmp"
    sandbox_tmp.mkdir(exist_ok=True)
    tempfile.tempdir = str(sandbox_tmp)
    sys.path[:0] = [str(root / "src"), str(root)]
    handles = []
    cdll = ctypes.CDLL

    def tracking_cdll(*args, **kwargs):
        library = cdll(*args, **kwargs)
        handles.append(library)
        return library

    ctypes.CDLL = tracking_cdll
    ctypes.cdll._dlltype = tracking_cdll
    in_socketpair = 0
    orig_socketpair = socket.socketpair

    def tracking_socketpair(*args, **kwargs):
        nonlocal in_socketpair
        in_socketpair += 1
        try:
            return orig_socketpair(*args, **kwargs)
        finally:
            in_socketpair -= 1

    socket.socketpair = tracking_socketpair

    def audit(event, args):
        if event in {'socket.connect', 'socket.bind'} and in_socketpair > 0 and len(args) >= 2 and isinstance(args[1], tuple) and args[1][:1] in {("127.0.0.1",), ("::1",)}:
            return
        if event in {'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'socket.sendto'} or event.startswith(("subprocess.", "os.exec", "os.spawn", "os.system", "os.fork")):
            raise PermissionError("SANDBOX_SECURITY_VIOLATION: external execution/network forbidden")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            mode, flags = args[1:3]
            writing = bool(isinstance(mode, str) and any(letter in mode for letter in "wax+")) or isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            raw_p = os.fsdecode(args[0])
            is_devnull = raw_p == os.devnull or raw_p.lower() in {"nul", r"\\.\nul", os.path.abspath(os.devnull).lower()}
            if writing and not is_devnull and not Path(raw_p).resolve().is_relative_to(root):
                raise PermissionError("SANDBOX_SECURITY_VIOLATION: write outside overlay")
        if event in {"os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.chmod", "os.symlink", "os.link"}:
            for path in args[:2] if event in {"os.rename", "os.symlink", "os.link"} else args[:1]:
                if isinstance(path, (str, bytes, os.PathLike)) and not Path(os.fsdecode(path)).resolve().is_relative_to(root):
                    raise PermissionError("SANDBOX_SECURITY_VIOLATION: mutation outside overlay")

    sys.addaudithook(audit)
    sys._invar_builder_sandbox = True
    result = dict(exit_code=0, failed_test_count=0, passed_test_count=0, pytest_ran=False, unloaded_handles=0)
    try:
        for library in config.get("libraries", []):
            tracking_cdll(library)
        for relpath in config.get("imports", []):
            parts = list(Path(relpath).with_suffix("").parts)
            if parts[0] == "src":
                module = importlib.import_module(".".join(parts[1:]))
            else:
                spec = importlib.util.spec_from_file_location("_invar_candidate", root / relpath)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
            port = config.get("port")
            if port and relpath == config.get("primary"):
                function = getattr(module, port["port_name"])
                if list(inspect.signature(function).parameters) != [field["name"] for field in port["inputs"]]:
                    raise ValueError("HIS_CONTRACT_DRIFT: parameter signature")
                for case in config["cases"]:
                    value = function(**case)
                    if not isinstance(value, dict) or set(value) != {field["name"] for field in port["outputs"]}:
                        raise ValueError("HIS_CONTRACT_DRIFT: output schema")
                    for field in port["outputs"]:
                        item = value[field["name"]]
                        scalar = field["scalar_type"]
                        valid = {"f64": isinstance(item, (int, float)) and not isinstance(item, bool), "i64": type(item) is int,
                            "bool": type(item) is bool, "str": isinstance(item, str), "vec_f64": isinstance(item, list), "table": isinstance(item, (list, dict))}[scalar]
                        if not valid:
                            raise ValueError("HIS_CONTRACT_DRIFT: output type " + field["name"])
                        numbers = item if scalar == "vec_f64" else [item] if scalar in {"f64", "i64"} else []
                        for number in numbers:
                            if not isinstance(number, (int, float)) or not math.isfinite(number):
                                raise ValueError("PY_DOMAIN_ARITHMETIC_GUARD: nonfinite output")
                            bound = field.get("bounds")
                            if bound and ((bound["lower"] is not None and (number < bound["lower"] or number == bound["lower"] and not bound["lower_inclusive"])) or (bound["upper"] is not None and (number > bound["upper"] or number == bound["upper"] and not bound["upper_inclusive"]))):
                                raise ValueError("BOUND_CLAMP_VIOLATION: " + field["name"])
        if config.get("pytest_paths"):
            import pytest
            failures = []
            class Reports:
                def pytest_runtest_logreport(self, report):
                    if report.when == "call" and report.passed:
                        result["passed_test_count"] += 1
                    if report.failed:
                        result["failed_test_count"] += 1
                        failures.append(report.longreprtext)
            result["pytest_ran"] = True
            result["exit_code"] = int(pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", *config["pytest_paths"]], plugins=[Reports()]))
            if result["exit_code"]:
                result["error_code"] = "SLOT_PYTEST_ASSERTION" if result["exit_code"] == 1 else "PYTEST_INTERNAL_OR_USAGE"
                result["message"] = "\n".join(failures)[-4000:] or "Generated pytest exit " + str(result["exit_code"])
    except BaseException as exc:
        message = str(exc)
        code = "OS_OOM_OR_DISK" if isinstance(exc, MemoryError) else next((code for code in ("HIS_CONTRACT_DRIFT", "BOUND_CLAMP_VIOLATION", "PY_DOMAIN_ARITHMETIC_GUARD", "SANDBOX_SECURITY_VIOLATION") if code in message),
                    "PY_MISSING_STDLIB_IMPORT" if isinstance(exc, NameError) else "PY_DOMAIN_ARITHMETIC_GUARD" if isinstance(exc, (ZeroDivisionError, OverflowError)) or isinstance(exc, ValueError) and "math domain" in message else "SLOT_RUNTIME_LOGIC_ERROR")
        result.update(exit_code=1, error_code=code, message=message, traceback=traceback.format_exc()[-6000:])
    finally:
        for library in reversed(handles):
            if library._handle:
                (_ctypes.FreeLibrary if os.name == "nt" else _ctypes.dlclose)(library._handle)
                library._handle = 0
                result["unloaded_handles"] += 1
        handles.clear()
        gc.collect()
    return result


if __name__ == "__main__":
    configuration = json.loads(sys.stdin.read())
    if os.name != "nt":
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 ** 3,) * 2)
    print("INVAR_SANDBOX=" + json.dumps(main(configuration), allow_nan=False))

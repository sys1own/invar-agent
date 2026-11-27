"""Pytest configuration for xdist worker isolation and optional Z3 thread-confinement probing."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any

WORKER = os.environ.get("PYTEST_XDIST_WORKER")
_SCRATCH: list[Path] = []
_Z3_PROBE_ORIGINALS: dict[type, Any] = {}
_Z3_FOREIGN_RELEASES: list[tuple[str, int, str]] = []
_Z3_INTENTIONAL_RACE_DEFERRED = 0


def _write_progress_event(event: str, nodeid: str) -> None:
    base = os.environ.get("INVAR_TEST_PROGRESS", "").strip()
    if not base:
        return
    suffix = WORKER or "main"
    path = Path(f"{base}.{suffix}.jsonl")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event, "nodeid": nodeid, "t": time.time()}) + "\n")
    except OSError:
        pass


def pytest_runtest_logstart(nodeid, location):
    _write_progress_event("start", str(nodeid))


def pytest_runtest_logfinish(nodeid, location):
    _write_progress_event("finish", str(nodeid))


def record_intentional_z3_race_deferred(delta: int) -> None:
    global _Z3_INTENTIONAL_RACE_DEFERRED
    _Z3_INTENTIONAL_RACE_DEFERRED += max(0, int(delta))


def _install_z3_probe() -> None:
    """When INVAR_Z3_THREAD_PROBE=1, record any Z3 __del__ called off the Z3 worker thread."""
    try:
        import z3  # type: ignore
        from invar_agent.runtime.isolation import get_z3_worker_tid
    except Exception:
        return

    for cls in (z3.AstRef, z3.Solver, z3.ModelRef, z3.AstVector):
        if cls in _Z3_PROBE_ORIGINALS:
            continue
        orig = cls.__del__
        _Z3_PROBE_ORIGINALS[cls] = orig

        def _make_probe(c: type, fn: Any):
            def _probed_del(self: Any):
                if os.environ.get("_INVAR_Z3_INTENTIONAL_RACE") != "1":
                    worker_tid = get_z3_worker_tid()
                    cur_tid = threading.get_ident()
                    if worker_tid is not None and cur_tid != worker_tid:
                        _Z3_FOREIGN_RELEASES.append(
                            (c.__name__, cur_tid, threading.current_thread().name)
                        )
                return fn(self)

            return _probed_del

        cls.__del__ = _make_probe(cls, orig)


def pytest_configure(config):
    if os.environ.get("INVAR_Z3_THREAD_PROBE") == "1":
        _install_z3_probe()

    if not WORKER:
        return
    for name in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    root = Path(tempfile.mkdtemp(prefix=f"invar_ws_{WORKER}_"))
    _SCRATCH.append(root)
    os.environ["INVAR_WORKSPACE_DIR"] = str(root / "workspace")
    os.environ["INVAR_CONFIG_PATH"] = str(root / "config.json")
    from hypothesis import database, settings

    db = Path(config.rootpath) / ".hypothesis" / "xdist" / WORKER
    db.mkdir(parents=True, exist_ok=True)
    settings.register_profile(
        f"invar_{WORKER}",
        database=database.DirectoryBasedExampleDatabase(str(db)),
    )
    settings.load_profile(f"invar_{WORKER}")


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("INVAR_Z3_THREAD_PROBE") == "1":
        from invar_agent.runtime.isolation import get_z3_deferred_release_count

        deferred = max(0, get_z3_deferred_release_count() - _Z3_INTENTIONAL_RACE_DEFERRED)
        if _Z3_FOREIGN_RELEASES or deferred:
            print(
                f"\n[INVAR_Z3_THREAD_PROBE] FAILURE: "
                f"foreign_releases={len(_Z3_FOREIGN_RELEASES)} "
                f"deferred_releases={deferred} "
                f"sample={_Z3_FOREIGN_RELEASES[:5]}"
            )
            session.exitstatus = 1
        else:
            print(
                "\n[INVAR_Z3_THREAD_PROBE] OK: 0 foreign-thread Z3 releases, "
                "0 unexpected deferred releases"
            )


def pytest_unconfigure(config):
    for cls, orig in list(_Z3_PROBE_ORIGINALS.items()):
        try:
            cls.__del__ = orig
        except Exception:
            pass
    _Z3_PROBE_ORIGINALS.clear()
    for root in _SCRATCH:
        shutil.rmtree(root, ignore_errors=True)
    _SCRATCH.clear()

"""Regression and stress tests for P0 Z3 cross-thread object-lifetime confinement."""
from __future__ import annotations

from contextlib import contextmanager
import gc
import threading
import time
from typing import Iterator

import pytest
import z3

from invar_agent.research.meaningfulness import solve_query
from invar_agent.research.smt import check_invariants_consistent
from invar_agent.runtime.isolation import (
    _Z3_EXECUTOR,
    get_z3_deferred_release_count,
    get_z3_worker_tid,
    run_on_z3_thread,
)
from invar_agent.verification.concolic import ConcolicExecutor
from invar_agent.verification.smt_solver import SMTStatus, SMTSolverBridge


_PROBED_Z3_CLASSES = (z3.AstRef, z3.Solver, z3.ModelRef, z3.AstVector)


@contextmanager
def z3_release_thread_probe() -> Iterator[list[tuple[str, int, str]]]:
    """Wrap Z3 ``__del__`` hooks and record any finalizer invoked off the Z3 worker thread.

    Records ``(class_name, thread_ident, thread_name)`` whenever a Z3 wrapper's
    ``__del__`` fires on a thread other than ``get_z3_worker_tid()`` while the
    Z3 worker is active.
    """
    foreign_releases: list[tuple[str, int, str]] = []
    originals: dict[type, object] = {}

    for cls in _PROBED_Z3_CLASSES:
        orig = cls.__del__
        originals[cls] = orig

        def _make_probe(c: type, fn):
            def _probed_del(self):
                worker_tid = get_z3_worker_tid()
                cur_tid = threading.get_ident()
                if worker_tid is not None and cur_tid != worker_tid:
                    foreign_releases.append(
                        (c.__name__, cur_tid, threading.current_thread().name)
                    )
                return fn(self)

            return _probed_del

        cls.__del__ = _make_probe(cls, orig)

    try:
        yield foreign_releases
    finally:
        for cls, orig in originals.items():
            cls.__del__ = orig


def test_z3_cyclic_gc_concurrency_stress(monkeypatch):
    """Exact §4.1 cyclic-GC concurrency stress test: concurrent solve_query + make_cycles + gc.collect()."""
    # Ensure Z3 worker is initialized and finalizer confinement is active.
    res0 = solve_query("x + y > 1", {"x": (0.0, 2.0), "y": (0.0, 2.0)}, witness=True)
    assert res0.status == "SAT"
    assert get_z3_worker_tid() is not None

    deferred_before = get_z3_deferred_release_count()
    monkeypatch.setenv("_INVAR_Z3_INTENTIONAL_RACE", "1")

    def make_cycles(n: int = 400) -> int:
        x = z3.Real("x")
        for i in range(n):
            h: dict[str, object] = {"ast": x * i + i}
            h["self"] = h
        return n

    errors: list[BaseException] = []
    stop = threading.Event()
    queries_done = 0

    def solver_loop() -> None:
        nonlocal queries_done
        try:
            for k in range(80):
                if stop.is_set():
                    break
                r = solve_query(
                    f"x + y > {0.5 + (k % 5) * 0.1}",
                    {"x": (0.0, 2.0), "y": (0.0, 2.0)},
                    witness=True,
                )
                assert r.status == "SAT"
                queries_done += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            stop.set()

    def gc_race_loop() -> None:
        try:
            for _ in range(12):
                if stop.is_set():
                    break
                run_on_z3_thread(make_cycles, 400)
                pending = _Z3_EXECUTOR.submit(make_cycles, 1500)
                gc.collect()
                pending.result()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            stop.set()

    t_solver = threading.Thread(target=solver_loop, name="z3-stress-solver")
    t_gc = threading.Thread(target=gc_race_loop, name="z3-stress-gc")
    t_solver.start()
    t_gc.start()
    t_solver.join(timeout=60)
    t_gc.join(timeout=60)
    assert not t_solver.is_alive(), "solver thread hung"
    assert not t_gc.is_alive(), "gc race thread hung"
    assert not errors, f"Z3 concurrency stress raised: {errors!r}"
    assert queries_done >= 80
    # Drain any deferred releases and flush intentional cyclic garbage on the Z3 thread.
    run_on_z3_thread(gc.collect)
    try:
        import tests.conftest as _conf

        _conf.record_intentional_z3_race_deferred(
            get_z3_deferred_release_count() - deferred_before
        )
    except Exception:
        pass


def test_z3_zero_foreign_releases_across_all_entry_points():
    """Verify 0 foreign-thread Z3 releases across all guarded entry points + exception paths."""
    # Warm up worker so get_z3_worker_tid() is active before probing.
    run_on_z3_thread(lambda: None)
    assert get_z3_worker_tid() is not None

    before_deferred = get_z3_deferred_release_count()

    with z3_release_thread_probe() as foreign_releases:
        # 1. research.meaningfulness.solve_query
        r1 = solve_query("x * x + y * y <= 1", {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}, witness=True)
        assert r1.status == "SAT"
        r2 = solve_query("x > 5 and x < 1", {"x": (0.0, 10.0)})
        assert r2.status == "UNSAT"

        # 2. research.smt.check_invariants_consistent (previously unguarded)
        rep_sat = check_invariants_consistent(["state[0] >= 0", "state[0] <= 10"])
        assert rep_sat.satisfiable
        rep_unsat = check_invariants_consistent(["state[0] > 5", "state[0] < 1"])
        assert not rep_unsat.satisfiable

        # 3. verification.smt_solver.SMTSolverBridge (previously unguarded)
        bridge = SMTSolverBridge(default_timeout_ms=2000)
        smt_res = bridge.prove_postcondition(["x > 5"], ["x > 0"], timeout_ms=2000)
        assert smt_res.status == SMTStatus.UNSAT_PROVEN
        del bridge

        # 4. verification.concolic.ConcolicExecutor (previously unguarded)
        def target(x):
            if x == 7:
                return 1 / 0
            return 0

        concolic_rep = ConcolicExecutor(max_explorations=8, solver_timeout_ms=2000).explore(
            target, ["x"], {"x": 0}
        )
        assert any(f.exception == "ZeroDivisionError" for f in concolic_rep.findings)

        # 5. Exception escaping Z3 worker must not leak Z3 locals in traceback frames.
        def _raise_with_z3_locals():
            s = z3.Solver()
            v = z3.Real("leak_test")
            s.add(v > 0)
            raise RuntimeError("intentional_z3_worker_error")

        with pytest.raises(RuntimeError, match="intentional_z3_worker_error"):
            run_on_z3_thread(_raise_with_z3_locals)

        # Force cyclic GC on MainThread to expose any leaked Z3 references.
        gc.collect()

    assert foreign_releases == [], f"Detected foreign-thread Z3 releases: {foreign_releases}"
    assert get_z3_deferred_release_count() == before_deferred

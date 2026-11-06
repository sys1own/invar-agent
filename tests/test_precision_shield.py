"""Precision Shield tests: SMT, concolic, chiasmus, holographic parity."""
from __future__ import annotations

import z3

from invar_agent.verification.chiasmus import ChiasmusSymmetryVerifier
from invar_agent.verification.concolic import ConcolicExecutor
from invar_agent.verification.holographic import HolographicVerifier
from invar_agent.verification.smt_solver import SMTStatus, SMTSolverBridge


def test_smt_proves_abs_result_nonnegative():
    bridge = SMTSolverBridge(default_timeout_ms=2000)
    result = bridge.prove_postcondition(
        [],
        ["result >= 0"],
        local_env={"result": z3.Abs(z3.Int("x"))},
        timeout_ms=2000,
    )
    assert result.status == SMTStatus.UNSAT_PROVEN


def test_smt_finds_discount_counterexample_witness():
    bridge = SMTSolverBridge(default_timeout_ms=2000)
    result = bridge.prove_postcondition(
        ["discount > price"],
        ["price - discount >= 0"],
        local_env={"price": z3.Int("price"), "discount": z3.Int("discount")},
        timeout_ms=2000,
    )
    assert result.status == SMTStatus.SAT_COUNTEREXAMPLE
    witness = result.witness or {}
    assert witness.get("price") is not None and witness.get("discount") is not None
    assert witness["discount"] > witness["price"]
    assert witness["price"] - witness["discount"] < 0


def test_concolic_negates_branch_and_uncovers_zerodivision():
    def buggy(x):
        if x == 42:
            return 1 / 0
        return 1

    executor = ConcolicExecutor(max_explorations=16, solver_timeout_ms=2000)
    report = executor.explore(buggy, ["x"], {"x": 0})
    assert report.branches_explored >= 1
    panics = [f for f in report.findings if f.exception == "ZeroDivisionError"]
    assert panics, f"expected ZeroDivisionError finding, got {report.findings}"
    assert panics[0].inputs.get("x") == 42


def test_chiasmus_accepts_balanced_and_rejects_early_return():
    verifier = ChiasmusSymmetryVerifier()
    pairs = [("lock.acquire", "lock.release")]

    balanced = (
        "def worker(lock):\n"
        "    lock.acquire()\n"
        "    do_work()\n"
        "    lock.release()\n"
    )
    result_ok = verifier.verify_mirror_invariants(balanced, pairs)
    assert result_ok.symmetric, result_ok.diagnostics

    bypass = (
        "def worker(lock, cond):\n"
        "    lock.acquire()\n"
        "    if cond:\n"
        "        return\n"
        "    lock.release()\n"
    )
    result_bad = verifier.verify_mirror_invariants(bypass, pairs)
    assert not result_bad.symmetric
    assert result_bad.violations

    try_finally = (
        "def worker(lock, cond):\n"
        "    lock.acquire()\n"
        "    try:\n"
        "        if cond:\n"
        "            return\n"
        "        do_work()\n"
        "    finally:\n"
        "        lock.release()\n"
    )
    result_tf = verifier.verify_mirror_invariants(try_finally, pairs)
    assert result_tf.symmetric, result_tf.diagnostics


def test_holographic_parity_stable_and_sensitive():
    verifier = HolographicVerifier()
    v1 = {"pkg/a.py": "def api(x):\n    return x + 1\n"}
    v2 = {"pkg/a.py": "def api(x):\n    return x + 12345\n"}
    v3 = {"pkg/a.py": "def api(x, y):\n    return x + y\n"}
    assert verifier.encode_bulk_ast_to_boundary(v1) == verifier.encode_bulk_ast_to_boundary(v2)
    assert verifier.encode_bulk_ast_to_boundary(v1) != verifier.encode_bulk_ast_to_boundary(v3)

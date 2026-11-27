#!/usr/bin/env python3
"""Unified two-lane offline test runner for invar-agent.

Runs:
  1. ``cargo test`` (unless ``--pytest-only`` is passed)
  2. Parallel pytest lane: ``pytest -m "not live and not serial" -n auto --maxprocesses=<workers> --dist loadgroup``
  3. Serial pytest lane:   ``pytest -m "serial and not live" -n 0``

Verifies that both lanes pass with 0 failures and prints a concise timing summary.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parent.parent
_COLLECTED_RE = re.compile(r"collected\s+(\d+)\s+items(?:\s*/\s*(\d+)\s+deselected)?")
_SUMMARY_RE = re.compile(r"(\d+)\s+passed")


def _default_workers() -> int:
    raw = os.environ.get("INVAR_TEST_WORKERS", "").strip()
    if raw:
        try:
            val = int(raw)
            if val >= 1:
                return val
        except ValueError:
            pass
    return 6


def _run_step(label: str, cmd: list[str], env: dict[str, str]) -> tuple[int, float, str]:
    print(f"\n=== [{label}] {' '.join(cmd)} ===", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    rc = proc.wait()
    elapsed = time.perf_counter() - t0
    return rc, elapsed, "".join(lines)


def _count_selected(pytest_exe: str, marker_expr: str, env: dict[str, str]) -> int:
    proc = subprocess.run(
        [pytest_exe, "--collect-only", "-q", "-m", marker_expr],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return -1
    # Last summary line looks like: "1234/1265 tests collected (31 deselected) in ..."
    m = re.search(r"(\d+)(?:/\d+)?\s+tests?\s+collected", proc.stdout)
    if m:
        return int(m.group(1))
    m2 = _COLLECTED_RE.search(proc.stdout)
    if m2:
        total = int(m2.group(1))
        desel = int(m2.group(2) or 0)
        return total - desel
    return -1


def _parse_passed(output: str) -> int:
    matches = _SUMMARY_RE.findall(output)
    return int(matches[-1]) if matches else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run invar-agent offline test suite (two-lane).")
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help="Max pytest-xdist worker processes for parallel lane (default: INVAR_TEST_WORKERS or 6).",
    )
    parser.add_argument(
        "--pytest-only",
        action="store_true",
        help="Skip cargo test and run only the two pytest lanes.",
    )
    parser.add_argument(
        "--verify-partition",
        action="store_true",
        help="Verify exact test count partition: parallel + serial == total offline.",
    )
    parser.add_argument(
        "--extra-pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to both pytest lanes.",
    )
    args = parser.parse_args(argv)

    env = os.environ.copy()
    env.pop("PYTHONUTF8", None)

    pytest_exe = str(Path(sys.executable).parent / ("pytest.exe" if os.name == "nt" else "pytest"))
    if not Path(pytest_exe).is_file():
        pytest_exe = sys.executable
        pytest_prefix = [sys.executable, "-m", "pytest"]
    else:
        pytest_prefix = [pytest_exe]

    total_start = time.perf_counter()
    cargo_s: float | None = None
    cargo_rc = 0

    if not args.pytest_only:
        cargo_rc, cargo_s, _ = _run_step("cargo test", ["cargo", "test"], env)
        if cargo_rc != 0:
            print(f"\n[FAIL] cargo test exited with {cargo_rc} after {cargo_s:.2f}s")
            return cargo_rc

    if args.verify_partition and pytest_prefix == [pytest_exe]:
        n_total = _count_selected(pytest_exe, "not live", env)
        n_par = _count_selected(pytest_exe, "not live and not serial", env)
        n_ser = _count_selected(pytest_exe, "serial and not live", env)
        print(
            f"\n[partition] total_offline={n_total}, parallel={n_par}, serial={n_ser}, "
            f"sum={n_par + n_ser}",
            flush=True,
        )
        if n_total <= 0 or n_par + n_ser != n_total:
            print("[FAIL] Test partition mismatch!", file=sys.stderr)
            return 2

    par_cmd = [
        *pytest_prefix,
        "-m",
        "not live and not serial",
        "-n",
        "auto",
        f"--maxprocesses={max(1, args.workers)}",
        "--dist",
        "loadgroup",
        *args.extra_pytest_args,
    ]
    par_rc, par_s, par_out = _run_step(f"pytest parallel (max {args.workers} workers)", par_cmd, env)

    ser_cmd = [
        *pytest_prefix,
        "-m",
        "serial and not live",
        "-n",
        "0",
        *args.extra_pytest_args,
    ]
    ser_rc, ser_s, ser_out = _run_step("pytest serial (-n 0)", ser_cmd, env)

    total_s = time.perf_counter() - total_start
    par_passed = _parse_passed(par_out)
    ser_passed = _parse_passed(ser_out)

    print("\n================ OFFLINE TEST SUMMARY ================")
    if cargo_s is not None:
        print(f"  Cargo test    : {cargo_s:7.2f} s  (exit {cargo_rc})")
    print(f"  Parallel lane : {par_s:7.2f} s  (exit {par_rc}, {par_passed} passed, workers<={args.workers})")
    print(f"  Serial lane   : {ser_s:7.2f} s  (exit {ser_rc}, {ser_passed} passed)")
    print(f"  Total wall    : {total_s:7.2f} s  ({total_s / 60.0:.2f} min, {par_passed + ser_passed} pytest passed)")
    print("======================================================")

    if par_rc != 0:
        return par_rc
    if ser_rc != 0:
        return ser_rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

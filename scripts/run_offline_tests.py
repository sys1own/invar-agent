#!/usr/bin/env python3
"""Unified two-lane offline test runner for invar-agent (UPG-018 non-hanging watchdog).

Runs:
  1. ``cargo test`` (unless ``--pytest-only`` is passed)
  2. Parallel pytest lane: ``pytest -m "not live and not serial" -n auto --maxprocesses=<workers> --dist loadgroup``
  3. Serial pytest lane:   ``pytest -m "serial and not live" -n 0``

Monitors child stdout on a background thread with an idle watchdog
(`INVAR_TEST_IDLE_TIMEOUT`, default 300s). On idle timeout or unreported tests,
prints started-but-unfinished node IDs, kills the process tree, and exits with code 3.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
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


def _idle_timeout_s() -> float:
    raw = os.environ.get("INVAR_TEST_IDLE_TIMEOUT", "").strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return 300.0


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _read_progress_files(progress_base: Path) -> tuple[set[str], set[str]]:
    started: set[str] = set()
    finished: set[str] = set()
    parent = progress_base.parent
    prefix = progress_base.name + "."
    if not parent.is_dir():
        return started, finished
    for p in parent.iterdir():
        if p.name.startswith(prefix) and p.name.endswith(".jsonl"):
            try:
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    nid = rec.get("nodeid")
                    ev = rec.get("event")
                    if nid:
                        if ev == "start":
                            started.add(nid)
                        elif ev == "finish":
                            finished.add(nid)
            except (OSError, json.JSONDecodeError):
                continue
    return started, finished


def _run_step(
    label: str,
    cmd: list[str],
    env: dict[str, str],
    *,
    progress_base: Path | None = None,
    idle_timeout_s: float = 300.0,
) -> tuple[int, float, str, bool]:
    print(f"\n=== [{label}] {' '.join(cmd)} ===", flush=True)
    t0 = time.perf_counter()
    step_env = dict(env)
    step_env["PYTHONUNBUFFERED"] = "1"
    if progress_base is not None:
        step_env["INVAR_TEST_PROGRESS"] = str(progress_base)

    popen_kwargs: dict = dict(
        cwd=str(REPO_ROOT),
        env=step_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **popen_kwargs)

    lines: list[str] = []
    last_activity = [time.monotonic()]
    eof_event = threading.Event()

    def _reader() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                last_activity[0] = time.monotonic()
                sys.stdout.write(line)
                sys.stdout.flush()
                lines.append(line)
        finally:
            eof_event.set()

    reader_t = threading.Thread(target=_reader, daemon=True)
    reader_t.start()

    timed_out = False
    while not eof_event.wait(timeout=1.0):
        if proc.poll() is not None:
            eof_event.wait(timeout=5.0)
            break
        if time.monotonic() - last_activity[0] > idle_timeout_s:
            timed_out = True
            print(
                f"\n[WATCHDOG] {label} idle for > {idle_timeout_s:.1f}s — killing process tree!",
                file=sys.stderr,
                flush=True,
            )
            if progress_base is not None:
                started, finished = _read_progress_files(progress_base)
                unfinished = sorted(started - finished)
                print(
                    f"[WATCHDOG] Started-but-unfinished nodes ({len(unfinished)}): {unfinished[:30]}",
                    file=sys.stderr,
                    flush=True,
                )
            _kill_process_tree(proc)
            break

    reader_t.join(timeout=5.0)
    rc = 3 if timed_out else proc.wait()
    elapsed = time.perf_counter() - t0
    return rc, elapsed, "".join(lines), timed_out


def _collect_nodeids(pytest_prefix: list[str], marker_expr: str, env: dict[str, str]) -> list[str] | None:
    proc = subprocess.run(
        [*pytest_prefix, "--collect-only", "-q", "-m", marker_expr],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return None
    nodeids: list[str] = []
    for line in proc.stdout.splitlines():
        s = line.strip()
        if "::" in s and not s.startswith(("=", "-", "WARNING")):
            nodeids.append(s)
    return nodeids


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
        help="Verify exact test count partition and that every collected node reports completion.",
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
    env["PYTHONUNBUFFERED"] = "1"
    idle_limit = _idle_timeout_s()

    pytest_exe = str(Path(sys.executable).parent / ("pytest.exe" if os.name == "nt" else "pytest"))
    if not Path(pytest_exe).is_file():
        pytest_prefix = [sys.executable, "-m", "pytest"]
    else:
        pytest_prefix = [pytest_exe]

    total_start = time.perf_counter()
    cargo_s: float | None = None
    cargo_rc = 0

    if not args.pytest_only:
        cargo_rc, cargo_s, _, cargo_to = _run_step("cargo test", ["cargo", "test"], env, idle_timeout_s=idle_limit)
        if cargo_to or cargo_rc != 0:
            print(f"\n[FAIL] cargo test exited with {cargo_rc} after {cargo_s:.2f}s")
            return 3 if cargo_to else cargo_rc

    collected_total: list[str] | None = None
    if args.verify_partition:
        collected_total = _collect_nodeids(pytest_prefix, "not live", env)
        collected_par = _collect_nodeids(pytest_prefix, "not live and not serial", env)
        collected_ser = _collect_nodeids(pytest_prefix, "serial and not live", env)
        if collected_total is None or collected_par is None or collected_ser is None:
            print("[FAIL] Failed to collect test node IDs!", file=sys.stderr)
            return 2
        n_total, n_par, n_ser = len(collected_total), len(collected_par), len(collected_ser)
        print(
            f"\n[partition] total_offline={n_total}, parallel={n_par}, serial={n_ser}, "
            f"sum={n_par + n_ser}",
            flush=True,
        )
        if n_total <= 0 or n_par + n_ser != n_total or (set(collected_par) | set(collected_ser)) != set(collected_total):
            print("[FAIL] Test partition mismatch!", file=sys.stderr)
            return 2

    prog_dir = Path(tempfile.mkdtemp(prefix="invar_test_progress_"))
    try:
        par_prog = prog_dir / "par"
        ser_prog = prog_dir / "ser"

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
        par_rc, par_s, par_out, par_to = _run_step(
            f"pytest parallel (max {args.workers} workers)",
            par_cmd,
            env,
            progress_base=par_prog,
            idle_timeout_s=idle_limit,
        )

        ser_cmd = [
            *pytest_prefix,
            "-m",
            "serial and not live",
            "-n",
            "0",
            *args.extra_pytest_args,
        ]
        ser_rc, ser_s, ser_out, ser_to = _run_step(
            "pytest serial (-n 0)",
            ser_cmd,
            env,
            progress_base=ser_prog,
            idle_timeout_s=idle_limit,
        )

        par_started, par_finished = _read_progress_files(par_prog)
        ser_started, ser_finished = _read_progress_files(ser_prog)
        all_finished = par_finished | ser_finished
        unreported: list[str] = []
        if args.verify_partition and collected_total is not None and not args.extra_pytest_args:
            unreported = sorted(set(collected_total) - all_finished)

        total_s = time.perf_counter() - total_start
        par_passed = _parse_passed(par_out)
        ser_passed = _parse_passed(ser_out)

        print("\n================ OFFLINE TEST SUMMARY ================")
        if cargo_s is not None:
            print(f"  Cargo test    : {cargo_s:7.2f} s  (exit {cargo_rc})")
        print(f"  Parallel lane : {par_s:7.2f} s  (exit {par_rc}, {par_passed} passed, workers<={args.workers})")
        print(f"  Serial lane   : {ser_s:7.2f} s  (exit {ser_rc}, {ser_passed} passed)")
        if args.verify_partition and collected_total is not None:
            print(f"  Reported      : {len(all_finished)}/{len(collected_total)} finished, {len(unreported)} unreported")
        print(f"  Total wall    : {total_s:7.2f} s  ({total_s / 60.0:.2f} min, {par_passed + ser_passed} pytest passed)")
        print("======================================================")

        if par_to or ser_to:
            return 3
        if unreported:
            print(f"[FAIL] {len(unreported)} collected tests never reported completion: {unreported[:20]}", file=sys.stderr)
            return 3
        if par_rc != 0:
            return par_rc
        if ser_rc != 0:
            return ser_rc
        return 0
    finally:
        shutil.rmtree(prog_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

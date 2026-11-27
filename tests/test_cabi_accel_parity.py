"""Parity & cache-invalidation tests for Phase 2 cabi_bridge.py accelerations."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

import invar_agent.runtime.cabi_bridge as cabi_mod
from invar_agent.runtime.cabi_bridge import (
    DynamicCABIBridge,
    _EXPORTED_SYMBOLS_CACHE,
    _direct_python_executable,
)


HARMONIC_CPP = r"""
#include <cmath>
#include <cstddef>
#include <cstdint>

typedef struct SimulationBuffer {
    size_t size;
    double dt;
    uint32_t steps;
    double* data;
    double* diag;
} SimulationBuffer;

extern "C" void execute_step(SimulationBuffer* buf) {
    double q = buf->data[0];
    double p = buf->data[1];
    double dt = buf->dt;
    double k1_q = p;
    double k1_p = -q;
    double k2_q = p + 0.5 * dt * k1_p;
    double k2_p = -(q + 0.5 * dt * k1_q);
    double k3_q = p + 0.5 * dt * k2_p;
    double k3_p = -(q + 0.5 * dt * k2_q);
    double k4_q = p + dt * k3_p;
    double k4_p = -(q + dt * k3_q);
    buf->data[0] = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q);
    buf->data[1] = p + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p);
    buf->steps += 1;
}
"""

CUSTOM_MAP_CPP = r"""
#include <cmath>
#include <cstddef>
#include <cstdint>

typedef struct SimulationBuffer {
    size_t size;
    double dt;
    uint32_t steps;
    double* data;
    double* diag;
} SimulationBuffer;

extern "C" void execute_custom_step(SimulationBuffer* buf) {
    double x = buf->data[0];
    double y = buf->data[1];
    double dt = buf->dt;
    buf->data[0] = x * std::cos(dt) - y * std::sin(dt);
    buf->data[1] = x * std::sin(dt) + y * std::cos(dt);
    buf->steps += 1;
}
"""


def test_exported_symbols_lru_memoization_and_stat_invalidation(monkeypatch, tmp_path):
    """Verify exported_symbols memoizes hits, returns fresh lists, and misses on size/mtime change."""
    _EXPORTED_SYMBOLS_CACHE.clear()
    fake_so = tmp_path / "libtest.so"
    fake_so.write_bytes(b"ELF_V1")

    nm_calls: list[list[str]] = []
    nm_output = {"stdout": "0000000000001000 T execute_step\n0000000000001040 W helper_fn\n"}

    monkeypatch.setattr(cabi_mod.shutil, "which", lambda name: "/usr/bin/nm" if name == "nm" else None)

    def _fake_run(cmd, **kwargs):
        nm_calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=nm_output["stdout"], stderr="")

    monkeypatch.setattr(cabi_mod.subprocess, "run", _fake_run)

    bridge = DynamicCABIBridge(tmp_path, compiler="clang++")
    first = bridge.exported_symbols(fake_so)
    assert first == ["execute_step", "helper_fn"]
    assert len(nm_calls) == 1

    # Mutating the returned list must not corrupt the cached entry.
    first.append("mutated_symbol")

    second = bridge.exported_symbols(fake_so)
    assert second == ["execute_step", "helper_fn"]
    assert len(nm_calls) == 1  # cache hit, no extra nm subprocess

    # Changing file size must immediately miss.
    nm_output["stdout"] = "0000000000001000 T execute_step\n0000000000001080 T execute_v2\n"
    fake_so.write_bytes(b"ELF_V2_LONGER_PAYLOAD")
    third = bridge.exported_symbols(fake_so)
    assert third == ["execute_step", "execute_v2"]
    assert len(nm_calls) == 2

    # Changing mtime_ns (even with identical size) must immediately miss.
    st = os.stat(fake_so)
    os.utime(fake_so, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))
    nm_output["stdout"] = "0000000000001000 T execute_v3\n"
    fourth = bridge.exported_symbols(fake_so)
    assert fourth == ["execute_v3"]
    assert len(nm_calls) == 3


@pytest.mark.parametrize(
    ("cpp_code", "entry", "kernel_mode", "initial_state", "dt", "steps", "trace_slots"),
    [
        (HARMONIC_CPP, "execute_step", "rk4_ode", [1.0, 0.0], 0.05, 40, (0, 1)),
        (CUSTOM_MAP_CPP, "execute_custom_step", "custom_discrete_map", [1.0, 0.25], 0.1, 25, (0,)),
    ],
)
def test_cabi_direct_python_executable_bit_exact_parity(
    monkeypatch,
    tmp_path,
    cpp_code,
    entry,
    kernel_mode,
    initial_state,
    dt,
    steps,
    trace_slots,
):
    """Bit-exact parity between sys.executable and _direct_python_executable() across C-ABI kernels."""
    bridge = DynamicCABIBridge(tmp_path)
    if bridge.compiler is None:
        pytest.skip("No C++ compiler available")

    direct_py = _direct_python_executable()
    assert Path(direct_py).is_file()

    # Run with direct base interpreter (default path).
    res_direct = bridge.run(
        cpp_code,
        initial_state=initial_state,
        dt=dt,
        steps=steps,
        entry=entry,
        kernel_mode=kernel_mode,
        trace_slots=trace_slots,
    )

    # Run with sys.executable via monkeypatch.
    monkeypatch.setattr(cabi_mod, "_direct_python_executable", lambda: sys.executable)
    res_sys = bridge.run(
        cpp_code,
        initial_state=initial_state,
        dt=dt,
        steps=steps,
        entry=entry,
        kernel_mode=kernel_mode,
        trace_slots=trace_slots,
    )

    assert res_direct["finite"] is True
    assert res_direct["state"] == res_sys["state"]
    assert res_direct["diag"] == res_sys["diag"]
    assert res_direct["trace_min"] == res_sys["trace_min"]
    assert res_direct["trace_max"] == res_sys["trace_max"]
    assert res_direct["trace_values"] == res_sys["trace_values"]
    assert res_direct["steps"] == res_sys["steps"]
    assert res_direct["first_nonfinite_step"] == res_sys["first_nonfinite_step"]

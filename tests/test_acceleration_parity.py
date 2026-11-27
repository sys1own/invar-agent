"""BUG-040: INVAR_ACCEL_CACHE=0 lossless acceleration kill-switch parity tests."""
from __future__ import annotations

from pathlib import Path

from invar_agent.runtime.accel import accel_enabled
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
from invar_agent.runtime.isolation import run_isolated_kernel_smoke


def test_accel_enabled_flag_parsing(monkeypatch):
    monkeypatch.delenv("INVAR_ACCEL_CACHE", raising=False)
    assert accel_enabled() is True
    for off in ("0", "false", "FALSE", "off", "no"):
        monkeypatch.setenv("INVAR_ACCEL_CACHE", off)
        assert accel_enabled() is False
    monkeypatch.setenv("INVAR_ACCEL_CACHE", "1")
    assert accel_enabled() is True


def test_isolation_parity_with_and_without_accel_cache(monkeypatch, tmp_path: Path):
    code = (
        "def simulation_step(buf, dt):\n"
        "    buf.data[0] = buf.data[0] + dt * (-0.5 * buf.data[0])\n"
    )
    monkeypatch.setenv("INVAR_ACCEL_CACHE", "1")
    out1 = run_isolated_kernel_smoke(code, dt=0.1)

    monkeypatch.setenv("INVAR_ACCEL_CACHE", "0")
    out0 = run_isolated_kernel_smoke(code, dt=0.1)

    assert out1["ok"] is True
    assert out0["ok"] is True
    assert out1.get("sample") == out0.get("sample")
    bridge = DynamicCABIBridge(tmp_path / "cabi")
    assert bridge.workspace == tmp_path / "cabi"

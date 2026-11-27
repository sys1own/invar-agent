"""Picklable slow emitter factory for killable wall-clock preemption tests (UPG-018)."""
from __future__ import annotations

import subprocess
import sys
import time

from invar_agent.builder.emitters.base import EmitterRegistry, ModuleEmission


def _sleep_forever_emit(node, context):
    # Spawn a grandchild that sleeps as well so Job Object / process-group cleanup is exercised
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        time.sleep(60.0)
    finally:
        proc.kill()
    return ModuleEmission({})


def slow_emitters() -> EmitterRegistry:
    reg = EmitterRegistry()
    reg.process_factory = "tests._slow_emitters:slow_emitters"
    for name in ("kernel", "hybrid_compute", "workbench", "executable_paper", "calibration", "receipts"):
        reg.register(name, _sleep_forever_emit)
    return reg

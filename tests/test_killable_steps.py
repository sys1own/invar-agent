"""UPG-018 (BUG-030, NF-16) tests: killable StepWorker and picklable default_emitters."""
from __future__ import annotations

import pickle
import time

import pytest

from invar_agent.builder.emitters import default_emitters
from invar_agent.runtime.killable import StepTimeout, StepWorker, is_pid_alive


def test_default_emitters_is_picklable_and_has_factory():
    reg = default_emitters()
    assert reg.process_factory == "invar_agent.builder.emitters:default_emitters"
    blob = pickle.dumps(reg)
    unpickled = pickle.loads(blob)
    assert set(unpickled.emitters.keys()) == set(reg.emitters.keys())


@pytest.mark.serial
@pytest.mark.timeout(60)
def test_step_worker_kills_sleeping_process_on_timeout():
    worker = StepWorker("tests._slow_emitters:slow_emitters")
    try:
        worker.warm(timeout_s=30.0)
        pid = worker.pid
        assert pid is not None and is_pid_alive(pid)
        t0 = time.monotonic()
        from types import SimpleNamespace
        with pytest.raises(StepTimeout):
            worker.run("emit", pickle.dumps(((SimpleNamespace(kind="kernel"), None), {})), timeout_s=1.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 8.0
        assert not is_pid_alive(pid)
    finally:
        worker.close()

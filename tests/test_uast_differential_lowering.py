"""Multi-domain translation checks; MC samples are fixed common random inputs."""

import json
import math

import pytest

from invar_agent import _native
from invar_agent.research.differential_check import check_kernel, ulp_distance


KERNELS = {
    "ode": ("""def simulation_step(buf, dt):
    x = buf[0]
    k1 = -0.3 * x
    k2 = -0.3 * (x + 0.5 * dt * k1)
    k3 = -0.3 * (x + 0.5 * dt * k2)
    k4 = -0.3 * (x + dt * k3)
    buf[0] = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
""", [1.0], False),
    "monte_carlo": ("""def simulation_step(buf, dt):
    mean = 0.0
    m2 = 0.0
    for i in range(1, 9):
        x = buf[i]
        delta = x - mean
        mean = mean + delta / float(i)
        m2 = m2 + delta * (x - mean)
    buf[0] = mean + m2 / 7.0
""", [0.0, 0.125, 0.75, 0.375, 0.875, 0.0625, 0.5, 0.625, 0.25], False),
    "equilibrium": ("""def simulation_step(buf, dt):
    x = buf[0]
    for i in range(4):
        x = x - (x * x - 2.0) / (2.0 * x)
    buf[0] = x
""", [1.5], False),
    "transcendental": ("""import math
def simulation_step(buf, dt):
    x = buf[0]
    buf[0] = math.exp(-x) + math.log(1.0 + x) + math.sqrt(x)
    buf[1] = math.sin(x) + math.cos(x)
""", [0.5, 0.0], True),
    "simplex": ("""def simulation_step(buf, dt):
    total = 0.0
    for i in range(3):
        total = total + buf[i]
    for i in range(3):
        buf[i] = buf[i] / total
""", [0.2, 0.3, 0.6], False),
}


@pytest.mark.parametrize("domain", KERNELS)
@pytest.mark.parametrize("steps,scale", [(1, 1.0), (3, 0.5), (5, 1.25)])
def test_multi_domain_lowering(tmp_path, domain, steps, scale):
    source, initial, libm = KERNELS[domain]
    result = check_kernel(source, [x * scale for x in initial], dt=0.03125,
                          steps=steps, cache_dir=tmp_path, transcendental=libm)
    assert len(result.python) == len(initial)


@pytest.mark.parametrize("cube",["x**3","pow(x,3)","x*x*x"])
def test_cubic_rhs_differential_lowering(tmp_path,cube):
    source = f"def simulation_step(buf,dt):\n    x = buf.data[0]\n    buf.data[0] = x + dt*(x-x*x-0.01*({cube}))\n"
    report = check_kernel(source,[0.7],dt=.01,steps=40,cache_dir=tmp_path)
    assert len(report.python) == 1


@pytest.mark.parametrize("cube",["x**3","pow(x,3)","x*x*x"])
def test_cubic_rhs_differential_lowering(tmp_path,cube):
    source = f"def simulation_step(buf,dt):\n    x = buf.data[0]\n    buf.data[0] = x + dt*(x-x*x-0.01*({cube}))\n"
    result = check_kernel(source,[0.7],dt=.01,steps=40,cache_dir=tmp_path)
    assert len(result.python) == 1


def test_empty_kernel_and_zero_steps(tmp_path):
    source = "def simulation_step(buf, dt):\n    buf.steps = buf.steps + 1\n"
    assert check_kernel(source, [], dt=0.1, steps=1, cache_dir=tmp_path).intervals == ()
    result = check_kernel(source, [2.0], dt=0.1, steps=0, cache_dir=tmp_path)
    assert result.intervals == ((2.0, 2.0),)


def test_invalid_inputs_and_unsupported_interval_operation(tmp_path):
    with pytest.raises(ValueError, match="finite inputs"):
        check_kernel(KERNELS["ode"][0], [math.nan], dt=0.1, steps=1, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        _native.evaluate_interval_f64(json.dumps({"kind": "call", "func": "lgamma",
            "args": [{"kind": "literal", "value": 1.0}]}), {})
    with pytest.raises(ValueError, match="invalid interval"):
        _native.evaluate_interval_f64('{"kind":"symbol","name":"x"}', {"x": (2.0, 1.0)})


def test_ulp_ordering():
    assert ulp_distance(-0.0, 0.0) == 0
    for x in [-1.0, 0.0, 1.0]:
        assert ulp_distance(x, math.nextafter(x, math.inf)) == 1


def test_checker_rejects_corrupted_lowering(tmp_path, monkeypatch):
    from invar_agent.research.differential_check import CppEmitter
    original = CppEmitter.emit_module
    def corrupted(self, uast):
        return original(self, uast).replace("0.3", "0.4")
    monkeypatch.setattr(CppEmitter, "emit_module", corrupted)
    with pytest.raises(AssertionError):
        check_kernel(KERNELS["ode"][0], [1.0], dt=0.1, steps=1, cache_dir=tmp_path)


def test_interval_branch_fails_closed(tmp_path):
    source = "def simulation_step(buf, dt):\n    if buf[0] == 1.0:\n        buf[0] = 2.0\n"
    with pytest.raises(TypeError, match="interval-dependent comparison"):
        check_kernel(source, [1.0], dt=0.1, steps=1, cache_dir=tmp_path)

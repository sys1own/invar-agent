"""UPG-009 (BUG-022): Formal model fidelity checker & compiler-owned equilibrium analytics."""
from __future__ import annotations

import pytest

from invar_agent.research.claims import SpanRef
from invar_agent.research.fidelity import (
    StatedEquation,
    check_expression_equivalence,
    check_model_fidelity,
    compute_2x2_interval_eigenvalues,
    compute_equilibrium_analytics,
    scan_math_spans,
)
from invar_agent.research.meaningfulness import ForcingSchedule, ForcingSegment


def test_scan_math_spans_and_equivalence_ladder():
    prompt = "Consider the ODE dx/dt = a - (b + 1) * x + x**2 * y and dy/dt = b * x - x**2 * y."
    spans = scan_math_spans(prompt)
    assert len(spans) == 2

    # x**2.0 vs x*x (where e-graph alone fails, rational cancel succeeds!)
    status_eq, w_eq = check_expression_equivalence("x**2.0", "x*x", {"x": (0.5, 2.0)})
    assert status_eq == "EQUIVALENT" and w_eq is None

    # Genuine mismatch caught by 64 Halton point interval refutation with witness
    status_mis, w_mis = check_expression_equivalence("x**2 + 1", "x**2 + 2", {"x": (0.0, 5.0)})
    assert status_mis == "MISMATCH" and w_mis is not None and "x" in w_mis


def test_fidelity_missing_forcing_and_mismatch():
    prompt = "Model dC/dt = u - k * C"
    idx = prompt.index("dC/dt")
    se = StatedEquation(
        id="eq1",
        kind="DERIVATIVE",
        lhs="C",
        rhs_text="u - k * C",
        span=SpanRef(start=idx, end=len(prompt), quote=prompt[idx:]),
        symbol_map={},
    )
    sched = ForcingSchedule(
        symbols=("u__0",),
        segments=(
            ForcingSegment(t_start="0", t_end="10", values={"u__0": "0"}),
            ForcingSegment(t_start="10", t_end="40", values={"u__0": "1"}),
            ForcingSegment(t_start="40", t_end="100", values={"u__0": "0"}),
        ),
    )
    findings_ok = check_model_fidelity(
        prompt,
        [se],
        {"C": "u - k * C"},
        {"C": (0.0, 5.0), "u": (0.0, 1.0), "k": (0.05, 0.8)},
        forcing_schedule=sched,
        stated_forcing_segments=[("u", "10", "40", "1")],
    )
    assert findings_ok == []

    findings_miss = check_model_fidelity(
        prompt,
        [se],
        {"C": "u - k * C"},
        {"C": (0.0, 5.0), "u": (0.0, 1.0), "k": (0.05, 0.8)},
        forcing_schedule=None,
        stated_forcing_segments=[("u", "10", "40", "1")],
    )
    assert any(f.code == "FIDELITY_MISSING_FORCING" for f in findings_miss)


def test_r6_brusselator_compiler_owned_analytics():
    # R6 Brusselator-like: dx/dt = a - (b + 1)*x + x^2*y, dy/dt = b*x - x^2*y with a=2, b=3.5
    # Equilibrium: x* = a = 2, y* = b/a = 1.75
    # Jacobian at (2, 1.75): [[b - 1, a^2], [-b, -a^2]] = [[2.5, 4], [-3.5, -4]]
    # tr = -1.5, det = 4 => lambda = -0.75 +/- i * sqrt(4 - 0.5625) = -0.75 +/- 1.854050i
    # Sensitivities wrt b: dx*/db = 0, dy*/db = 1/a = 0.5
    res = compute_equilibrium_analytics(
        state_names=["x", "y"],
        rhs_exprs={"x": "a - (b + 1) * x + x**2 * y", "y": "b * x - x**2 * y"},
        params={"a": 2.0, "b": 3.5},
        initial_guess={"x": 1.8, "y": 1.5},
        sensitivity_params=["b"],
    )
    assert res["ledger_tag"] == "ANALYTIC_COMPILER_OWNED"
    assert abs(res["equilibrium"]["eq__x"] - 2.0) < 1e-9
    assert abs(res["equilibrium"]["eq__y"] - 1.75) < 1e-9
    eigs = res["eigenvalues"]
    assert eigs["complex"] is True and eigs["stable"] is True
    assert eigs["eig0__re"][0] <= -0.75 <= eigs["eig0__re"][1]
    assert eigs["eig0__im"][0] <= 1.8540496217739157 <= eigs["eig0__im"][1]
    assert abs(res["sensitivities"]["sens__x__b"] - 0.0) < 1e-9
    assert abs(res["sensitivities"]["sens__y__b"] - 0.5) < 1e-9

    # Multiplicity unresolved when discriminant straddles 0
    with pytest.raises(ValueError, match="EIGENVALUE_MULTIPLICITY_UNRESOLVED"):
        compute_2x2_interval_eigenvalues([[(-1.01, -0.99), (0.0, 0.0)], [(0.0, 0.0), (-1.01, -0.99)]])

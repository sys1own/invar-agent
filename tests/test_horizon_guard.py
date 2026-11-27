"""UPG-005 (BUG-007) tests: guard applicability for temporal vs non-temporal models."""
from __future__ import annotations

import math

from invar_agent.research.planner import (
    _kernel_reads_dt,
    _reconcile_time_horizon,
    program_has_temporal_equations,
)


def test_algebraic_param_sweep_skips_horizon_reconciliation():
    bp = {
        "code_template": "def simulation_step(buf, dt):\n    buf.data[1] = buf.data[0] * 15.0\n",
        "contracts": [{"contract_id": "c1", "dt": 0.05, "steps": 200}],
        "ihcg_evidence": {
            "archetype": "PARAM_SWEEP",
            "evidence": {
                "equations": {
                    "net_savings": {"kind": "ALGEBRAIC", "expression": "capex - annual * 15"},
                }
            },
        },
    }
    assert not program_has_temporal_equations(bp)
    assert not _kernel_reads_dt(bp["code_template"])
    out = _reconcile_time_horizon(bp, "Evaluate over a 15-year horizon")
    assert out["contracts"][0]["dt"] == 0.05
    assert out["contracts"][0]["steps"] == 200


def test_temporal_ode_reconciles_months_and_quarters():
    bp = {
        "code_template": "def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0] - dt * 1e-7 * buf.data[0]\n",
        "contracts": [{"contract_id": "c1", "dt": 1000.0, "steps": 100}],
        "ihcg_evidence": {
            "archetype": "ODE_PDE",
            "evidence": {
                "equations": {
                    "d_x_dt": {"kind": "ODE_RHS", "expression": "-k * x", "state_symbol": "x"},
                }
            },
        },
    }
    assert program_has_temporal_equations(bp)
    assert _kernel_reads_dt(bp["code_template"])
    out_mo = _reconcile_time_horizon(bp, "Simulate over 6 months")
    assert math.isclose(out_mo["contracts"][0]["dt"] * out_mo["contracts"][0]["steps"], 6.0 * 2629746.0, rel_tol=1e-9)

    out_q = _reconcile_time_horizon(bp, "Simulate over 2 quarters")
    assert math.isclose(out_q["contracts"][0]["dt"] * out_q["contracts"][0]["steps"], 2.0 * 7889238.0, rel_tol=1e-9)


def test_bare_dict_fallback_uses_dt_read():
    bare_temporal = {
        "code_template": "def simulation_step(buf, dt):\n    buf.data[0] += dt * buf.data[1]\n",
        "contracts": [{"contract_id": "c1", "dt": 1.0, "steps": 10}],
    }
    bare_algebraic = {
        "code_template": "def simulation_step(buf, dt):\n    buf.data[0] = buf.data[1] * 2.0\n",
        "contracts": [{"contract_id": "c1", "dt": 1.0, "steps": 10}],
    }
    assert program_has_temporal_equations(bare_temporal)
    assert not program_has_temporal_equations(bare_algebraic)

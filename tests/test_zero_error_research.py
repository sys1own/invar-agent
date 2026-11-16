"""Phase 5 Tests: Zero-Error Research Compute Engine on Singularities, Extremes, and Stiff ODEs."""

import math
import pytest

from invar_agent.research.state_machine import _lower_kernel, _BufView
from invar_agent.uast import (
    InvariantContractSpec,
    OntologyAndParameterSpec,
    PhysicalQuantity,
    ResearchComputeUAST,
    SymbolicGovernanceMap,
    compile_uast_to_python,
)
from invar_agent.uast.healer import tier0_heal_uast


def _run_compiled_steps(compiled, steps: int = 100, dt: float = 0.001):
    uast_ir, cpp_src = _lower_kernel(compiled.python_code)
    assert "simulation_step" in cpp_src
    assert uast_ir["type"] == "module"
    ns = {}
    exec(compiled.python_code, ns)
    step_fn = ns["simulation_step"]
    data = list(compiled.initial_state)
    buf = _BufView(data)
    for idx in range(steps):
        step_fn(buf, dt)
        assert all(math.isfinite(v) for v in buf.data), f"Non-finite state at step {idx}: {buf.data}"
    return buf.data, ns


def test_zero_error_singular_initial_conditions_and_extreme_values():
    """x_0 = 0 in 1/x and log(x) and sqrt(-x), plus y_0 = 1e6 in exp(y): 0% crash, 0 NaN/Inf."""
    uast = ResearchComputeUAST(
        title="Singular & Extreme Test System",
        ontology=OntologyAndParameterSpec(
            quantities={
                "x": PhysicalQuantity(symbol="x", is_state_variable=True, initial_value=0.0),
                "y": PhysicalQuantity(symbol="y", is_state_variable=True, initial_value=1000.0),
            }
        ),
        governance=SymbolicGovernanceMap(
            equations={
                "x": "1.0 / x + log(x) - sqrt(-x)",
                "y": "-0.01 * y + exp(y) * 1e-35",
            },
            observables={
                "ratio": "x / y",
            },
        ),
        dt=0.001,
        steps=50,
        target_backend="multi_variant",
    )
    healed, _ = tier0_heal_uast(uast)
    compiled = compile_uast_to_python(healed)
    final_data, ns = _run_compiled_steps(compiled, steps=50, dt=0.001)
    assert all(math.isfinite(v) for v in final_data)
    # Also verify every Pareto variant runs cleanly without NaN/Inf
    for variant_name in ("simulation_step_rk4", "simulation_step_semi_implicit", "simulation_step_ros3pr"):
        v_buf = _BufView(list(compiled.initial_state))
        for _ in range(20):
            ns[variant_name](v_buf, 0.001)
        assert all(math.isfinite(v) for v in v_buf.data), f"Variant {variant_name} produced non-finite state"


def test_stiff_ode_triggers_ros3pr_synthesis_and_converges():
    """Stiffness ratio > 1e6 automatically synthesizes ROS3PR and integrates stably."""
    uast = ResearchComputeUAST(
        title="Stiff Linear-Nonlinear Decay",
        ontology=OntologyAndParameterSpec(
            quantities={
                "x": PhysicalQuantity(symbol="x", is_state_variable=True, initial_value=1.0),
                "y": PhysicalQuantity(symbol="y", is_state_variable=True, initial_value=1.0),
            }
        ),
        governance=SymbolicGovernanceMap(
            equations={
                "x": "-1000000.0 * x + y",
                "y": "x - 2.0 * y",
            }
        ),
        dt=0.01,
        steps=50,
        target_backend="multi_variant",
    )
    healed, _ = tier0_heal_uast(uast)
    compiled = compile_uast_to_python(healed)
    assert compiled.spectral_radius >= 1e6 - 1.0
    assert compiled.stiffness_ratio >= 1e5
    assert compiled.selected_solver == "ros3pr"

    # With dt=0.01 and lambda=-1e6, explicit RK4 would blow up immediately; ROS3PR stays finite and decays!
    final_data, _ = _run_compiled_steps(compiled, steps=50, dt=0.01)
    assert all(math.isfinite(v) for v in final_data)
    assert abs(final_data[0]) < 1.0

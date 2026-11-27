"""UPG-004 (BUG-004, 018, 019, 020): Diagnostics, derive_archetype, MATHEMATICS, StateFamily, Functionals."""
from __future__ import annotations

import math
import pytest

from invar_agent.ir.diagnostics import Diagnostic, LoweringError
from invar_agent.ir.schema import (
    Archetype,
    EquationSemantics,
    EquationSpec,
    SciIR,
    ScientificDomain,
    SiDimension,
    SlotRole,
    SymbolDecl,
    derive_archetype,
    from_macro_domain,
    to_macro_domain,
)
from invar_agent.ir.state_family import BoundarySpec, StateFamily, expand_state_family
from invar_agent.research.functionals import (
    FunctionalError,
    evaluate_functional,
)


def test_diagnostics_and_lowering_error():
    d = Diagnostic(code="TEST_ERR", field="equations[0]", message="bad rhs", symbol="x")
    err = LoweringError([d])
    assert len(err.diagnostics) == 1
    assert "TEST_ERR" in str(err)


def test_mathematics_domain_mapping():
    assert to_macro_domain(ScientificDomain.MATHEMATICS) == "MATHEMATICS_CS_AND_INFO_THEORY"
    assert from_macro_domain("MATHEMATICS_CS_AND_INFO_THEORY") == ScientificDomain.QUANTUM_STATMECH


def test_derive_archetype_rules():
    ir = SciIR(
        ir_id="arch_test",
        domain=ScientificDomain.MATHEMATICS,
        user_prompt_digest="0" * 64,
        archetype=Archetype.PARAM_SWEEP,
        symbols=[
            SymbolDecl(
                name="x",
                display_symbol="x",
                role=SlotRole.STATE,
                slot_index=0,
                dimension=SiDimension(),
                initial_or_default=1.0,
            )
        ],
        dynamics_or_residuals=[
            EquationSpec(
                target="x",
                expr={"kind": "sym", "name": "x", "role": "STATE", "slot_index": 0},
                semantics=EquationSemantics.DERIVATIVE,
            )
        ],
    )
    arch, diags = derive_archetype(ir, "TIME_EVOLUTION")
    assert arch == Archetype.ODE_PDE
    assert any(d.code == "ARCHETYPE_PROMOTION" for d in diags)

    arch_eq, _ = derive_archetype(ir, "EQUILIBRIUM")
    assert arch_eq == Archetype.FIXED_POINT


def test_state_family_16_node_dirichlet_heat_rod_and_budget():
    # 1D copper rod: L=1.0 m, alpha=1.11e-4 m^2/s, T(0)=100 C, T(1)=20 C, T(x,0)=20 C
    # At t=600 s, midpoint T(0.5, 600s) ~= 61.33 C (using steady-state linear 60 C + transient)
    fam = StateFamily(
        name="T",
        shape=[16],
        length_symbols=["L"],
        coordinate_symbols=["x"],
        initial_text="20.0",
        rhs_text="alpha * lap(T)",
        boundaries=[
            BoundarySpec(axis=0, side="LOW", kind="DIRICHLET", value_text="100.0"),
            BoundarySpec(axis=0, side="HIGH", kind="DIRICHLET", value_text="20.0"),
        ],
    )
    exp = expand_state_family(fam, length_values={"L": 1.0})
    assert len(exp["slots"]) == 16
    assert exp["slots"][0]["role"] == "CONSTANT" and exp["slots"][0]["initial_or_default"] == 100.0
    assert exp["slots"][15]["role"] == "CONSTANT" and exp["slots"][15]["initial_or_default"] == 20.0
    assert len(exp["equations"]) == 14

    # Budget cap MAX_VARIABLES = 32
    too_big = fam.model_copy(update={"shape": [33]})
    with pytest.raises(ValueError, match="STATE_FAMILY_EXCEEDS_VERIFIER_BUDGET"):
        expand_state_family(too_big, length_values={"L": 1.0})


def test_state_family_neumann_and_periodic():
    fam_neu = StateFamily(
        name="u",
        shape=[5],
        length_symbols=["L"],
        coordinate_symbols=["x"],
        initial_text="1.0",
        rhs_text="lap(u)",
        boundaries=[
            BoundarySpec(axis=0, side="LOW", kind="NEUMANN", value_text="0.0"),
            BoundarySpec(axis=0, side="HIGH", kind="NEUMANN", value_text="0.0"),
        ],
    )
    exp_n = expand_state_family(fam_neu, length_values={"L": 1.0})
    assert len(exp_n["equations"]) == 5
    assert "u__1" in exp_n["equations"][0]["expr_text"]

    fam_per = fam_neu.model_copy(
        update={
            "boundaries": [
                BoundarySpec(axis=0, side="LOW", kind="PERIODIC"),
                BoundarySpec(axis=0, side="HIGH", kind="PERIODIC"),
            ]
        }
    )
    exp_p = expand_state_family(fam_per, length_values={"L": 1.0})
    assert abs(exp_p["h"][0] - 0.2) < 1e-12
    assert "u__4" in exp_p["equations"][0]["expr_text"]


def test_trajectory_functionals_max_peak_abs_final_and_hermite_period():
    dt = 0.01
    times = [i * dt for i in range(4001)]
    omega = 2.0 * math.pi / 5.0  # exact period T = 5.0 s
    samples = [math.sin(omega * t) for t in times]
    derivs = [omega * math.cos(omega * t) for t in times]

    est_max = evaluate_functional("MAX", times, samples)
    assert abs(est_max.value - 1.0) < 1e-4

    est_pa = evaluate_functional("PEAK_ABS", times, [-2.0 * s for s in samples])
    assert abs(est_pa.value - 2.0) < 1e-4

    est_fin = evaluate_functional("FINAL", times, samples)
    assert est_fin.eps_peak_loc == 0.0

    est_per = evaluate_functional("PERIOD", times, samples, derivatives=derivs, level=0.0)
    assert abs(est_per.value - 5.0) < 1e-5
    assert "PERIOD_INTERPOLATION_BOUND_HEURISTIC" in est_per.advisories

    # Insufficient cycles (< 3)
    with pytest.raises(FunctionalError, match="INSUFFICIENT_CYCLES"):
        evaluate_functional("PERIOD", times[:600], samples[:600], derivatives=derivs[:600], level=0.0)

    # Misaligned window
    with pytest.raises(FunctionalError, match="WINDOW_NOT_GRID_ALIGNED"):
        evaluate_functional("MAX", times, samples, window=(0.003, 1.0), dt=0.01)

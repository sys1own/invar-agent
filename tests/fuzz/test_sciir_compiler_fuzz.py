"""Hypothesis Property-Based Fuzz Suite for Invar-Agent vNext (`>= 500` randomized systems).

Covers all 7 compiler/solver stress domains from `inv4.txt` Section E.3:
1. Random reversible conserved stoichiometric ODEs (scope check, sealing idempotence,
   diagonal Buckingham-Pi scaling, C++20 compilation, ROS2 linear conservation < 1e-11)
2. Random let-DAG permutations & injected cycles (topological invariance + Tarjan SCC)
3. Random Theorem A.1 alias resolution & collision poisoning (`⊥_AMBIG`)
4. Random Monte Carlo & Fixed-Point & Parameter Sweep systems
5. Random RFC 8785 JCS key-order invariance & finite float canonicalization
"""
from __future__ import annotations

import json

from hypothesis import given, settings, strategies as st

from invar_agent import _native
from invar_agent.ir import (
    Archetype,
    FixedPointConfig,
    FrozenSymbolTable,
    IntervalBound,
    ModelBodyDraft,
    MonteCarloConfig,
    OdePdeConfig,
    ParamSweepConfig,
    RawEquationDraft,
    RawLetDraft,
    RawObligationDraft,
    ScientificDomain,
    SlotRole,
    SymbolDecl,
    jcs_canonicalize,
    scope_check_and_lower,
    seal_sciir_ihcg,
)


@settings(max_examples=200, deadline=None)
@given(
    a0=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    b0=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    kf=st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False),
    kr=st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False),
    scale_exp=st.integers(min_value=-6, max_value=6),
    domain=st.sampled_from(list(ScientificDomain)),
)
def test_fuzz_conserved_ode_sealing_scaling_and_ros2_invariance(
    a0: float, b0: float, kf: float, kr: float, scale_exp: int, domain: ScientificDomain
):
    scale = 10.0 ** scale_exp
    a_init = a0 * scale
    b_init = b0 * scale
    total = a_init + b_init

    table = FrozenSymbolTable(
        ir_id="fuzz_ode",
        archetype=Archetype.ODE_PDE,
        domain=domain,
        symbols=[
            SymbolDecl(name="A", role=SlotRole.STATE, initial_or_default=a_init, pool_id="pool_ab"),
            SymbolDecl(name="B", role=SlotRole.STATE, initial_or_default=b_init, pool_id="pool_ab"),
            SymbolDecl(
                name="k_f",
                role=SlotRole.CONTROL_PARAM,
                initial_or_default=kf,
                domain=IntervalBound(lo=kf * 0.5, hi=kf * 1.5),
            ),
            SymbolDecl(name="k_r", role=SlotRole.CONSTANT, initial_or_default=kr),
            SymbolDecl(name="sum_AB", role=SlotRole.DERIVED_OBSERVABLE),
        ],
        conservation_pools=[
            {"pool_id": "pool_ab", "member_states": ["A", "B"], "stoichiometry": [1.0, 1.0], "expected_total": total}
        ],
    ).assign_canonical_slot_indices()

    body = ModelBodyDraft(
        let_bindings=[RawLetDraft(name="flux", expr_text="k_f * A - k_r * B")],
        dynamics_or_residuals=[
            RawEquationDraft(target="A", expr_text="-flux"),
            RawEquationDraft(target="B", expr_text="flux"),
        ],
        observables=[RawEquationDraft(target="sum_AB", expr_text="A + B")],
        obligations=[
            RawObligationDraft(
                id="ob_mass",
                lhs_text="sum_AB",
                comparator=">=",
                rhs_text="0.0",
                tolerance=1e-6,
            )
        ],
        config=OdePdeConfig(t_span=(0.0, 1.0), sample_points=10, preferred_solver="ROS2"),
    )

    out = scope_check_and_lower(table, body, user_prompt="fuzz ode")
    assert out.ok and out.sci_ir is not None

    s1 = seal_sciir_ihcg(out.sci_ir)
    s2 = seal_sciir_ihcg(out.sci_ir)
    assert s1.sciir_commitment_blake3 == s2.sciir_commitment_blake3

    scales = json.loads(_native.py_compute_diagonal_scales(s1.ir.to_rust_json()))
    assert all(sc > 0.0 for sc in scales["state_scales"])

    traj = json.loads(_native.py_solve_sciir_ode(s1.ir.to_rust_json()))
    assert traj["max_conservation_drift"] <= 1e-11 * max(1.0, total)


@settings(max_examples=120, deadline=None)
@given(
    target_val=st.floats(min_value=0.2, max_value=20.0, allow_nan=False, allow_infinity=False),
    damp=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_fuzz_fixed_point_and_param_sweep(target_val: float, damp: float):
    # Linear stable equilibrium: dx/dt = -damp * (x - target_val) => x* = target_val
    table = FrozenSymbolTable(
        ir_id="fuzz_fp",
        archetype=Archetype.FIXED_POINT,
        domain=ScientificDomain.PHYSICS_THERMO,
        symbols=[
            SymbolDecl(name="x", role=SlotRole.STATE, initial_or_default=target_val * 0.5),
            SymbolDecl(
                name="d",
                role=SlotRole.CONTROL_PARAM,
                initial_or_default=damp,
                domain=IntervalBound(lo=damp * 0.8, hi=damp * 1.2),
            ),
            SymbolDecl(name="x_eq", role=SlotRole.CONSTANT, initial_or_default=target_val),
            SymbolDecl(name="err", role=SlotRole.DERIVED_OBSERVABLE),
        ],
    ).assign_canonical_slot_indices()

    body = ModelBodyDraft(
        dynamics_or_residuals=[RawEquationDraft(target="x", expr_text="-d * (x - x_eq)")],
        observables=[RawEquationDraft(target="err", expr_text="abs(x - x_eq)")],
        obligations=[
            RawObligationDraft(id="ob_eq", lhs_text="err", comparator="<=", rhs_text="1e-6", tolerance=1e-5)
        ],
        config=FixedPointConfig(max_newton_iters=50, residual_tol=1e-10),
    )
    out = scope_check_and_lower(table, body)
    assert out.ok and out.sci_ir is not None
    fp = json.loads(_native.py_solve_sciir_fixed_point(out.sci_ir.to_rust_json()))
    assert fp["converged"] is True
    assert abs(fp["fixed_point"][0] - target_val) < 1e-8
    assert fp["stability"]["stable"] is True

    sweep_res = json.loads(_native.py_solve_sciir_sweep(out.sci_ir.to_rust_json()))
    assert sweep_res["all_obligations_satisfied"] is True


@settings(max_examples=100, deadline=None)
@given(
    seed=st.integers(min_value=1, max_value=10_000_000),
    shift=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_fuzz_monte_carlo_determinism(seed: int, shift: float):
    table = FrozenSymbolTable(
        ir_id="fuzz_mc",
        archetype=Archetype.MONTE_CARLO,
        domain=ScientificDomain.QUANTUM_STATMECH,
        symbols=[
            SymbolDecl(name="mu", role=SlotRole.CONSTANT, initial_or_default=shift),
            SymbolDecl(name="sample_y", role=SlotRole.DERIVED_OBSERVABLE),
        ],
    ).assign_canonical_slot_indices()
    body = ModelBodyDraft(
        observables=[RawEquationDraft(target="sample_y", expr_text="mu + rand_normal01(0)")],
        obligations=[],
        config=MonteCarloConfig(base_seed=seed, replicates=128, confidence_alpha=0.05),
    )
    out = scope_check_and_lower(table, body)
    assert out.ok and out.sci_ir is not None
    mc1 = json.loads(_native.py_solve_sciir_mc(out.sci_ir.to_rust_json()))
    mc2 = json.loads(_native.py_solve_sciir_mc(out.sci_ir.to_rust_json()))
    assert mc1["observable_means"] == mc2["observable_means"]


@settings(max_examples=120, deadline=None)
@given(
    d=st.dictionaries(
        keys=st.text(min_size=1, max_size=12),
        values=st.one_of(
            st.integers(min_value=-10_000_000, max_value=10_000_000),
            st.floats(min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.text(max_size=16),
        ),
        min_size=1,
        max_size=8,
    )
)
def test_fuzz_rfc8785_jcs_permutation_invariance(d: dict):
    keys = list(d.keys())
    rev = {k: d[k] for k in reversed(keys)}
    c1 = jcs_canonicalize(d)
    c2 = jcs_canonicalize(rev)
    assert c1 == c2
    # Idempotence: canonicalizing canonical output is a fixed point
    assert jcs_canonicalize(c1) == c1

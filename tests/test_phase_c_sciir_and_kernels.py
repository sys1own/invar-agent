"""Phase C Hard Gate: Unified Typed SciIR, Equation Front-End Parity, Running-Max Binding, & Rational Fixed Point."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import scipy.optimize as opt
import numpy as np

from invar_agent.ir.lower_uast import emit_python_kernel_from_sciir, sciir_to_research_blueprint
from invar_agent.ir.ihcg_bridge import seal_sciir_ihcg
from invar_agent.ir.schema import (
    Archetype,
    EquationSpec,
    ExprBinary,
    ExprConst,
    ExprSym,
    IntervalBound,
    ObligationSpec,
    ParamSweepConfig,
    SciIR,
    ScientificDomain,
    SiDimension,
    SlotRole,
    SymbolDecl,
)
from invar_agent.research import meaningfulness as m
from invar_agent.research.default_verification import _Kernel
from invar_agent.research.expr_frontend import normalize_expression
from invar_agent.research.ihcg_derivation import derive_ihcg_evidence
from invar_agent.research.stochastic_equilibrium import _Roundoff
from invar_agent.runtime.isolation import run_isolated_kernel_smoke, validate_kernel_ast_allowlist
from invar_agent.schemas.research import ResearchBlueprint, VerificationContract


# ── 1. BUG-004 Equation & Obligation Front-End Parity ───────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("min(a, b, c)", "min(a, min(b, c))"),
        ("max(a, b, c)", "max(a, max(b, c))"),
        ("math.fabs(a - b)", "abs(a - b)"),
        ("a if a > b else b", "max(a, b)"),
        ("a if a < b else b", "min(a, b)"),
        ("a if a >= 0 else -a", "abs(a)"),
    ],
)
def test_equation_frontend_parity(raw: str, expected: str) -> None:
    res = normalize_expression(raw, declared_symbols={"a", "b", "c"})
    assert res.ok, f"Front-end failed on {raw!r}: {res.diagnostics}"
    assert res.canonical == expected


# ── 2. BUG-002 NOTE Ledger Entries Never Sealed ─────────────────────────────

def test_note_ledger_never_sealed() -> None:
    bp = {
        "blueprint_id": "note_test",
        "title": "Note Test",
        "research_prompt": "Test note filtering",
        "falsifiable_hypothesis": "y > 0",
        "variable_map": {"x": 0, "y": 1},
        "contracts": [
            {
                "contract_id": "c1",
                "target_backend": "cpp",
                "initial_state": [1.0, 2.0],
                "dt": 0.1,
                "steps": 10,
                "invariants": [{"variable_name": "y", "assertion": "state[1] > 0", "tolerance": 1e-6}],
            }
        ],
    }
    raw = {
        "archetype": "PARAM_SWEEP",
        "domain": {"x": [1.0, 2.0], "y": [0.0, 10.0]},
        "evidence": {
            "slots": [
                {"symbol": "x", "slot_idx": 0, "si_dim": [0]*7, "role": "CONTROL_PARAM"},
                {"symbol": "y", "slot_idx": 1, "si_dim": [0]*7, "role": "DERIVED_OBSERVABLE"},
            ],
            "equations": {"y": {"kind": "ALGEBRAIC", "expression": "2 * x", "si_dim": [0]*7}},
            "obligations": [{"id": "h0", "expression": "y > 0", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "y > 0"}],
            "environmental_assumptions": [],
            "assumptions_ledger": [
                {"kind": "NOTE", "symbol": "x", "statement": "informational note"},
                {"kind": "GOVERNING_EQUATION", "symbol": "y", "statement": "y = 2*x"},
            ],
            "quantifier_mode": "FORALL",
            "mechanism_parameters": ["x"],
        },
    }
    derived = derive_ihcg_evidence("Test note filtering", bp, raw)
    ledger = derived["evidence"].get("assumptions_ledger") or []
    assert all(entry.get("kind") != "NOTE" for entry in ledger)


# ── 3. BUG-007 All Running-Max Idioms Bind Cleanly ──────────────────────────

@pytest.mark.parametrize(
    "peak_update",
    [
        "    if I_new > buf.data[2]:\n        buf.data[2] = I_new\n",
        "    peak = buf.data[2]\n    if I_new > peak:\n        peak = I_new\n        buf.data[2] = peak\n",
        "    peak = buf.data[2]\n    peak = max(peak, I_new)\n    buf.data[2] = peak\n",
        "    peak = buf.data[2]\n    peak = I_new if I_new > peak else peak\n    buf.data[2] = peak\n",
    ],
)
def test_running_max_forms_bind(peak_update: str) -> None:
    evidence = {
        "slots": [
            {"symbol": "I", "slot_idx": 0, "si_dim": [0]*7, "role": "DYNAMIC_STATE"},
            {"symbol": "r", "slot_idx": 1, "si_dim": [0, 0, -1, 0, 0, 0, 0], "role": "CONTROL_PARAM"},
            {"symbol": "I_max", "slot_idx": 2, "si_dim": [0]*7, "role": "DERIVED_OBSERVABLE", "semantic_tag": m.TRAJECTORY_MAXIMUM},
        ],
        "equations": {
            "dI_dt": {"kind": "ODE_RHS", "state_symbol": "I", "expression": "-r * I", "si_dim": [0, 0, -1, 0, 0, 0, 0]},
            "I_max": {"kind": "TRAJECTORY_MAX", "state_symbol": "I", "expression": "I", "si_dim": [0]*7},
        },
        "obligations": [
            {"id": "h0", "expression": "I_max <= 0.5", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "I_max <= 0.5"},
        ],
        "environmental_assumptions": [],
        "quantifier_mode": "FORALL",
        "mechanism_parameters": ["r"],
    }
    domain = {"I": [0.0, 1.0], "r": [0.1, 1.0], "I_max": [0.0, 1.0]}
    graph = m.IhcgCompiler().compile_from_prompt("I_max <= 0.5", evidence, domain, "ODE_PDE")
    code = (
        "import math\n"
        "def simulation_step(buf, dt):\n"
        "    I = buf.data[0]\n"
        "    r = buf.data[1]\n"
        "    I_new = I + dt * (-r * I)\n"
        "    buf.data[0] = I_new\n"
        + peak_update
    )
    bp = ResearchBlueprint(
        blueprint_id="running_max_bind",
        title="Running Max Bind",
        research_prompt="I_max <= 0.5",
        falsifiable_hypothesis="I_max <= 0.5",
        search_queries=["sir"],
        variable_map={"I": 0, "r": 1, "I_max": 2},
        code_template=code,
        contracts=[
            VerificationContract(
                contract_id="c1",
                target_backend="cpp",
                initial_state=[0.3, 0.2, 0.3],
                dt=0.1,
                steps=20,
                invariants=[{"variable_name": "I_max", "assertion": "state[2] <= 0.5", "tolerance": 1e-6}],
            )
        ],
    )
    kernel = _Kernel(None, SimpleNamespace(ihcg=graph, blueprint=bp, research_stage=1), code)
    assert "I_max" in kernel.temporal


# ── 4. BUG-028 Rational Fixed Point (Lotka-Volterra Mutualism) ──────────────

def test_rational_fixed_point_mutualism_rounding_and_eigenvalues() -> None:
    """Certify Lotka-Volterra mutualism equilibrium x*=35.082800, y*=25.505164,
    eigenvalues -4.2949, -2.7843, and rational saturation rounding envelope."""
    r1, K1, a12 = 1.2, 10.0, 0.6
    r2, K2, a21 = 0.9, 8.0, 0.5
    h = 0.2

    def interior_eq(xy):
        x, y = xy
        return [
            1.0 - x / K1 + a12 * y / (1.0 + h * y),
            1.0 - y / K2 + a21 * x / (1.0 + h * x),
        ]

    def rhs(xy):
        x, y = xy
        return [
            r1 * x * (1.0 - x / K1 + a12 * y / (1.0 + h * y)),
            r2 * y * (1.0 - y / K2 + a21 * x / (1.0 + h * x)),
        ]

    sol = opt.root(interior_eq, [35.0, 25.0], tol=1e-13)
    assert sol.success
    x_star, y_star = float(sol.x[0]), float(sol.x[1])
    assert abs(x_star - 35.082800) < 1e-5
    assert abs(y_star - 25.505164) < 1e-5

    # Verify Jacobian eigenvalues
    eps = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        dx = np.zeros(2)
        dx[j] = eps
        J[:, j] = (np.array(rhs(sol.x + dx)) - np.array(rhs(sol.x - dx))) / (2 * eps)
    eigs = sorted(np.real(np.linalg.eigvals(J)))
    assert abs(eigs[0] - (-4.2949)) < 1e-3
    assert abs(eigs[1] - (-2.7843)) < 1e-3

    # Verify _RoundingEnvelope handles rational saturation (1 + h*y)**(-1) cleanly (BUG-028)
    code = (
        "def simulation_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    y = buf.data[1]\n"
        "    h = buf.data[2]\n"
        "    sat_y = (1.0 + h * y) ** -1\n"
        "    sat_x = (1.0 + h * x) ** -1\n"
        "    buf.data[0] = x + dt * (1.5 * x * (1.0 - x / 20.0 + 0.8 * y * sat_y))\n"
        "    buf.data[1] = y + dt * (1.2 * y * (1.0 - y / 15.0 + 0.7 * x * sat_x))\n"
    )
    env = _Roundoff(
        code,
        {"x": 0, "y": 1, "h": 2},
        {"x": (34.0, 36.0), "y": (24.5, 26.5), "h": (0.04, 0.06)},
        dt=0.01,
    )
    assert math.isfinite(env.values[0][1]) and env.values[0][1] < 1e-9
    assert math.isfinite(env.values[1][1]) and env.values[1][1] < 1e-9


# ── 5. ARCH-001 / ARCH-004 Deterministic Kernel Emission from SciIR ─────────

def test_sciir_deterministic_kernel_emission_readonly_params() -> None:
    ir = SciIR(
        ir_id="carnot_sciir",
        user_prompt_digest="0" * 64,
        domain=ScientificDomain.PHYSICS_THERMO,
        archetype=Archetype.PARAM_SWEEP,
        symbols=[
            SymbolDecl(name="T_hot", role=SlotRole.CONTROL_PARAM, slot_index=0, dimension=SiDimension(exponents=((0, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1), (0, 1)), unit_label="K"), initial_or_default=600.0, domain=IntervalBound(lo=600.0, hi=600.0)),
            SymbolDecl(name="T_cold", role=SlotRole.CONTROL_PARAM, slot_index=1, dimension=SiDimension(exponents=((0, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1), (0, 1)), unit_label="K"), initial_or_default=300.0, domain=IntervalBound(lo=300.0, hi=300.0)),
            SymbolDecl(name="eta", role=SlotRole.DERIVED_OBSERVABLE, slot_index=2, dimension=SiDimension(unit_label="1"), initial_or_default=0.5, domain=IntervalBound(lo=0.0, hi=1.0)),
        ],
        dynamics_or_residuals=[
            EquationSpec(
                target="eta",
                target_slot=2,
                expr=ExprBinary(
                    op="sub",
                    lhs=ExprConst(value=1.0),
                    rhs=ExprBinary(
                        op="div",
                        lhs=ExprSym(name="T_cold", role=SlotRole.CONTROL_PARAM),
                        rhs=ExprSym(name="T_hot", role=SlotRole.CONTROL_PARAM),
                    ),
                ),
            ),
        ],
        obligations=[
            ObligationSpec(
                id="h0",
                quantifier="FORALL",
                lhs_expr=ExprSym(name="eta", role=SlotRole.DERIVED_OBSERVABLE),
                rhs_expr=ExprConst(value=0.65),
                comparator=">=",
            ),
        ],
        config=ParamSweepConfig(),
    )
    code = emit_python_kernel_from_sciir(ir)
    # Control params (slots 0 and 1) MUST NEVER be written back to buf.data (prevents BUG-027!)
    assert "buf.data[0] =" not in code
    assert "buf.data[1] =" not in code
    assert "buf.data[2] = eta" in code
    assert validate_kernel_ast_allowlist(code) == []
    res = run_isolated_kernel_smoke(code, dt=0.01, validate_ast=True)
    assert res["ok"] is True

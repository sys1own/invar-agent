"""Phase E Hard Gate: Builder Symbol Identity, Dimensional ODE, Algebraic, P_0 Heuristic & Workbench Contracts."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import sympy as sp

from invar_agent.builder.emitters.workbench import APP as WORKBENCH_APP
from invar_agent.builder.ingestion.equation_lowerer import (
    classify_equation,
    equation_parts,
    lower_and_verify_equations,
    unit_dimension,
)
from invar_agent.builder.ingestion.schemas import EquationRole, RawConstant, RawEquation, SourceSpan


def _span(text: str = "spec") -> SourceSpan:
    return SourceSpan(file="spec.md", location="line 1", quote=text, char_start=0, char_end=len(text))


# ── 1. BUG-016 Strict Symbol Identity ────────────────────────────────────────

@pytest.mark.parametrize("sym", ["e", "E", "pi", "I", "S", "N", "lambda", "gamma"])
def test_builder_symbol_identity(sym: str) -> None:
    """Every user-declared identifier (`e`, `E`, `pi`, `I`, `S`, `N`, `lambda`, `gamma`)
    evaluates strictly to the user's numeric value — NEVER silently substituted with
    math.e, math.pi, or sp.I."""
    raw_consts = [
        RawConstant(symbol="k", value="2.0", unit="1/s", provenance=_span("k=2")),
        RawConstant(symbol=sym, value="0.85", unit="1", provenance=_span(f"{sym}=0.85")),
    ]
    raw_eqs = [
        RawEquation(text=f"dx/dt = -k * {sym} * x", provenance=_span(f"dx/dt = -k*{sym}*x")),
    ]
    equations, constants, flags = lower_and_verify_equations(raw_eqs, raw_consts, {"state_symbols": ["x"]})
    assert not [f for f in flags if not f.resolved], f"Unexpected unresolved flags for {sym}: {flags}"
    assert len(equations) == 1
    assert sym in equations[0].referenced_symbols
    lhs, _rel, rhs = equation_parts(equations[0].canonical_latex, declared_symbols={"k", sym, "x"})
    assert sp.Symbol(sym) in rhs.free_symbols, f"Symbol {sym!r} was silently substituted in {rhs}"
    val = float(rhs.subs({sp.Symbol("k"): 2.0, sp.Symbol(sym): 0.85, sp.Symbol("x"): 1.0}))
    assert val == pytest.approx(-1.7, abs=1e-12), f"Expected -1.7 for {sym}=0.85, got {val}"


# ── 2. BUG-018 Dimensional ODE States & Compound SI Units ────────────────────

def test_builder_dimensional_ode_and_compound_units() -> None:
    # Compound unit W/(m²·K) -> kg * s^-3 * K^-1 = (0, 1, -3, 0, -1, 0, 0)
    u_dim = unit_dimension("W/(m²·K)")
    assert u_dim.exponents == (0, 1, -3, 0, -1, 0, 0)

    # Free-fall dv/dt = g with g in m/s^2 -> v inferred as m/s without HIN_SI_DIM_MISMATCH
    eqs, consts, flags = lower_and_verify_equations(
        [RawEquation(text="dv/dt = g", provenance=_span("dv/dt = g"))],
        [RawConstant(symbol="g", value="9.81", unit="m/s^2", provenance=_span("g=9.81"))],
    )
    assert len(eqs) == 1 and eqs[0].hin_homogeneous is True
    assert not flags

    # Newton cooling dT/dt = -k * (T - T_env) with T_env in K and k in 1/s -> T inferred as K
    eqs2, consts2, flags2 = lower_and_verify_equations(
        [RawEquation(text="dT/dt = -k * (T - T_env)", provenance=_span("cooling"))],
        [
            RawConstant(symbol="k", value="0.05", unit="1/s", provenance=_span("k")),
            RawConstant(symbol="T_env", value="300.0", unit="K", provenance=_span("T_env")),
        ],
    )
    assert len(eqs2) == 1 and eqs2[0].hin_homogeneous is True
    assert not flags2


# ── 3. BUG-036 P_0 Algebraic Formula Not Misclassified as INITIAL_CONDITION ─

def test_p0_algebraic_formula_not_initial_condition() -> None:
    assert classify_equation("P_0 = 1 / (1 + rho + rho**2 / 2)") == EquationRole.ALGEBRAIC_CLOSURE
    assert classify_equation("x_0 = 1.5") == EquationRole.INITIAL_CONDITION
    assert classify_equation("x(0) = 1.5") == EquationRole.INITIAL_CONDITION


# ── 4. BUG-035 Workbench UTF-8 Root & Strict Sweep Schema ───────────────────

def test_workbench_app_template_uses_utf8_and_strict_sweep_schema() -> None:
    assert "read_text(encoding='utf-8')" in WORKBENCH_APP
    assert "extra='forbid'" in WORKBENCH_APP
    assert "class SweepRequest(BaseModel):" in WORKBENCH_APP

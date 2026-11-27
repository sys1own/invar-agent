"""HOTFIX-C (NF-03) tests: refuse non-autonomous ODE_PDE referencing INDEPENDENT_VAR."""
from __future__ import annotations

import pytest

from invar_agent.ir.lower_uast import emit_python_kernel_from_sciir
from invar_agent.ir.schema import (
    Archetype,
    FrozenSymbolTable,
    ModelBodyDraft,
    OdePdeConfig,
    RawEquationDraft,
    ScientificDomain,
    SiDimension,
    SlotRole,
    SymbolDecl,
)
from invar_agent.ir.typechecker import scope_check_and_lower


def test_nonautonomous_ode_referencing_independent_var_is_refused():
    dimless = SiDimension.dimensionless()
    time_dim = SiDimension(exponents=[(0, 1), (0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1)], unit_label="s")
    table = FrozenSymbolTable(
        ir_id="nonauto_t",
        archetype=Archetype.ODE_PDE,
        domain=ScientificDomain.PHYSICS_THERMO,
        symbols=[
            SymbolDecl(name="x", display_symbol="x", role=SlotRole.STATE, slot_index=0, dimension=dimless, initial_or_default=0.0),
            SymbolDecl(name="t", display_symbol="t", role=SlotRole.INDEPENDENT_VAR, slot_index=1, dimension=time_dim, initial_or_default=0.0),
        ],
    )
    body = ModelBodyDraft(
        dynamics_or_residuals=[RawEquationDraft(target="x", expr_text="t")],
        config=OdePdeConfig(t_span=(0.0, 1.0), sample_points=20),
    )
    outcome = scope_check_and_lower(table, body)
    assert outcome.ok and outcome.sci_ir is not None
    with pytest.raises(ValueError, match="NONAUTONOMOUS_TIME_UNSUPPORTED"):
        emit_python_kernel_from_sciir(outcome.sci_ir)


def test_autonomous_ode_with_unused_independent_var_is_allowed():
    dimless = SiDimension.dimensionless()
    time_dim = SiDimension(exponents=[(0, 1), (0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1)], unit_label="s")
    table = FrozenSymbolTable(
        ir_id="auto_t_unused",
        archetype=Archetype.ODE_PDE,
        domain=ScientificDomain.PHYSICS_THERMO,
        symbols=[
            SymbolDecl(name="x", display_symbol="x", role=SlotRole.STATE, slot_index=0, dimension=dimless, initial_or_default=1.0),
            SymbolDecl(name="t", display_symbol="t", role=SlotRole.INDEPENDENT_VAR, slot_index=1, dimension=time_dim, initial_or_default=0.0),
        ],
    )
    body = ModelBodyDraft(
        dynamics_or_residuals=[RawEquationDraft(target="x", expr_text="-x")],
        config=OdePdeConfig(t_span=(0.0, 1.0), sample_points=20),
    )
    outcome = scope_check_and_lower(table, body)
    assert outcome.ok and outcome.sci_ir is not None
    code = emit_python_kernel_from_sciir(outcome.sci_ir)
    assert "def simulation_step(buf, dt):" in code

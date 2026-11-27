"""UPG-008 (BUG-010, NF-06, 12, 17, 18) tests: unit parser, SI order, literal polymorphism, neutral time."""
from __future__ import annotations

from fractions import Fraction
import pytest

from invar_agent.builder.ingestion.equation_lowerer import FormalGateError, unit_dimension
from invar_agent.builder.ingestion.schemas import CanonicalResearchIR
from invar_agent.ir.ihcg_bridge import seal_sciir_ihcg
from invar_agent.ir.lower_uast import sciir_to_research_blueprint
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
from invar_agent.ir.units import builder_to_sci, parse_unit, sci_to_builder
from invar_agent.research.reference_annotations import ANNOTATIONS


def test_unit_parser_and_order_converters():
    assert parse_unit("mol/m^3").dim == (0, -3, 0, 0, 0, 1, 0)
    assert parse_unit("H").dim == (1, 2, -2, -2, 0, 0, 0)
    assert parse_unit("Ohm").dim == (1, 2, -3, -2, 0, 0, 0)
    assert parse_unit("m s^-1").dim == (0, 1, -1, 0, 0, 0, 0)
    assert parse_unit("mM").dim == (0, -3, 0, 0, 0, 1, 0)
    assert parse_unit("mM").scale == Fraction(1, 1)  # 1 mmol/L = 1 mol/m^3
    assert parse_unit("kWh").dim == (1, 2, -2, 0, 0, 0, 0)
    assert parse_unit("degC").to_si_rational(25) == Fraction(29815, 100)
    # Round-trip sci <-> builder
    v = (1, 2, -3, -1, 0, 0, 0)
    assert builder_to_sci(sci_to_builder(v)) == v
    assert sci_to_builder((0, 1, -1, 0, 0, 0, 0)) == (1, 0, -1, 0, 0, 0, 0)


def test_nf06_reference_annotations_in_sci_order():
    _, _, sb_dims = ANNOTATIONS[("fundamental_constants", 2)]
    assert sb_dims["T"] == (0, 0, 0, 0, 1, 0, 0)
    assert sb_dims["sigma_SB"] == (1, 0, -3, 0, -4, 0, 0)
    _, _, rl_dims = ANNOTATIONS[("relativistic_plasma", 1)]
    assert rl_dims["q"] == (0, 0, 1, 1, 0, 0, 0)
    assert rl_dims["B"] == (1, 0, -2, -1, 0, 0, 0)


def test_nf12_builder_unit_dimension_rejects_fractional_and_supports_new_units():
    assert unit_dimension("m s^-1").exponents == (1, 0, -1, 0, 0, 0, 0)
    assert unit_dimension("H").exponents == (2, 1, -2, -2, 0, 0, 0)
    assert unit_dimension("Ohm").exponents == (2, 1, -3, -2, 0, 0, 0)
    with pytest.raises(FormalGateError) as exc:
        unit_dimension("m^0.5")
    assert exc.value.code == "HIN_FRACTIONAL_DIMENSION"


def test_nf17_literal_polymorphism_and_rhs_target_check():
    # -k * (T - 293.15) with T in K and k in 1/s must PASS check_dimensions=True
    table = FrozenSymbolTable(
        ir_id="cooling_ok",
        archetype=Archetype.ODE_PDE,
        domain=ScientificDomain.PHYSICS_THERMO,
        symbols=[
            SymbolDecl(name="T", display_symbol="T", role=SlotRole.STATE, slot_index=0, dimension=SiDimension(unit_label="K"), initial_or_default=350.0),
            SymbolDecl(name="k", display_symbol="k", role=SlotRole.CONSTANT, slot_index=1, dimension=SiDimension(unit_label="1/s"), initial_or_default=0.1),
        ],
    )
    body = ModelBodyDraft(
        dynamics_or_residuals=[RawEquationDraft(target="T", expr_text="-k * (T - 293.15)")],
        config=OdePdeConfig(t_span=(0.0, 10.0), sample_points=50),
    )
    ok = scope_check_and_lower(table, body, check_dimensions=True)
    assert ok.ok, f"Literal polymorphism failed: {ok.diagnostics}"

    # dT/dt = -T (missing 1/s rate) must FAIL check_dimensions=True
    bad_body = ModelBodyDraft(
        dynamics_or_residuals=[RawEquationDraft(target="T", expr_text="-T")],
        config=OdePdeConfig(t_span=(0.0, 10.0), sample_points=50),
    )
    bad = scope_check_and_lower(table, bad_body, check_dimensions=True)
    assert not bad.ok
    assert any(d.get("code") == "DIMENSIONAL_INCONSISTENCY" for d in bad.diagnostics)


def test_symbol_mismatch_still_fails():
    table = FrozenSymbolTable(
        ir_id="sym_mismatch",
        archetype=Archetype.PARAM_SWEEP,
        domain=ScientificDomain.PHYSICS_THERMO,
        symbols=[
            SymbolDecl(name="x", display_symbol="x", role=SlotRole.CONTROL_PARAM, slot_index=0, dimension=SiDimension(unit_label="m"), initial_or_default=1.0),
            SymbolDecl(name="t_val", display_symbol="t", role=SlotRole.CONTROL_PARAM, slot_index=1, dimension=SiDimension(unit_label="s"), initial_or_default=1.0),
            SymbolDecl(name="y", display_symbol="y", role=SlotRole.DERIVED_OBSERVABLE, slot_index=2, dimension=SiDimension(unit_label="m"), initial_or_default=0.0),
        ],
    )
    body = ModelBodyDraft(
        observables=[RawEquationDraft(target="y", expr_text="x + t_val")],
    )
    res = scope_check_and_lower(table, body, check_dimensions=True)
    assert not res.ok
    assert any(d.get("code") == "DIMENSIONAL_INCONSISTENCY" for d in res.diagnostics)


def test_neutral_time_dimension_for_dimensionless_brusselator():
    table = FrozenSymbolTable(
        ir_id="brusselator_dimless",
        archetype=Archetype.ODE_PDE,
        domain=ScientificDomain.BIOCHEM_SYSTEMS,
        symbols=[
            SymbolDecl(name="x", display_symbol="x", role=SlotRole.STATE, slot_index=0, dimension=SiDimension(unit_label="1"), initial_or_default=1.0),
            SymbolDecl(name="y", display_symbol="y", role=SlotRole.STATE, slot_index=1, dimension=SiDimension(unit_label="1"), initial_or_default=1.0),
            SymbolDecl(name="a", display_symbol="a", role=SlotRole.CONSTANT, slot_index=2, dimension=SiDimension(unit_label="1"), initial_or_default=1.0),
            SymbolDecl(name="b", display_symbol="b", role=SlotRole.CONSTANT, slot_index=3, dimension=SiDimension(unit_label="1"), initial_or_default=3.0),
        ],
    )
    body = ModelBodyDraft(
        dynamics_or_residuals=[
            RawEquationDraft(target="x", expr_text="a - (b + 1) * x + x * x * y"),
            RawEquationDraft(target="y", expr_text="b * x - x * x * y"),
        ],
        config=OdePdeConfig(t_span=(0.0, 10.0), sample_points=100),
    )
    res = scope_check_and_lower(table, body, check_dimensions=True)
    assert res.ok and res.sci_ir is not None, f"Dimensionless Brusselator failed: {res.diagnostics}"
    bp = sciir_to_research_blueprint(seal_sciir_ihcg(res.sci_ir), user_prompt="Brusselator")
    eqs = bp["ihcg_evidence"]["evidence"]["equations"]
    assert eqs["d_x_dt"]["si_dim"] == [0, 0, 0, 0, 0, 0, 0]
    assert eqs["d_y_dt"]["si_dim"] == [0, 0, 0, 0, 0, 0, 0]

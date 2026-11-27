"""Round 2 Phase B regression & Hypothesis property tests:
- R2-ARCH-005 / R2-BUG-008: Builder free/derived parameter resolution & lhs_symbol extraction
- R2-ARCH-006 / R2-BUG-005: Rational/half-integer SI dimension exponents
- R2-BUG-006: Arbitrary-depth nested max()/min() expression parsing (depths 2..32)
- R2-BUG-009: Builder Studio CLI subcommands (`invar-agent builder ...`)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from invar_agent.builder.ingestion.equation_lowerer import equation_parts, lower_and_verify_equations
from invar_agent.builder.ingestion.pipeline import CanonicalResearchIngestionPipeline, apply_ir_override
from invar_agent.builder.ingestion.schemas import (
    IRReviewOverride,
    IngestionRequest,
    IntervalBoundModel,
    RawConstant,
    RawEquation,
    SourceSpan,
    SymbolResolution,
)
from invar_agent.research import meaningfulness as m
from invar_agent.research.expr_frontend import normalize_expression
from invar_agent.schemas.research import DerivedExpressionSpec, TypedConstant, TypedResearchSlot


def test_mmc_lambda_a_over_mu_parses_lhs_symbol_cleanly():
    """R2-BUG-008: `a = \\lambda_a / mu` must strip backslash before `_` and parse lhs_symbol='a'."""
    lhs, rel, rhs = equation_parts(r"a = \lambda_a / mu")
    assert str(lhs) == "a"
    assert rel == "="
    assert {str(s) for s in rhs.free_symbols} == {"lambda_a", "mu"}

    span = SourceSpan(file="mmc.md", location="line:1", quote="a = lambda_a / mu")
    eqs, consts, flags = lower_and_verify_equations(
        [
            RawEquation(text="a = lambda_a / mu", provenance=span),
            RawEquation(text="rho = a / c", provenance=span),
        ],
        [
            RawConstant(symbol="lambda_a", value="10", unit="1/s", provenance=span),
            RawConstant(symbol="mu", value="3", unit="1/s", provenance=span),
            RawConstant(symbol="c", value="4", unit="1", provenance=span),
        ],
    )
    assert eqs[0].lhs_symbol == "a"
    assert not eqs[0].diagnostics
    assert eqs[1].lhs_symbol == "rho"
    assert not [f for f in flags if f.target_symbol_or_eq == "a"]


def test_ir_override_symbol_resolutions_free_and_derived(tmp_path: Path):
    """R2-ARCH-005: IRReviewOverride.symbol_resolutions resolves both FREE_PARAMETER and DERIVED_BY_EQUATION."""
    spec_file = tmp_path / "bs_spec.md"
    spec_file.write_text(
        "# Black-Scholes Fee Model\n"
        "Fee multiplier $e = 0.0025$.\n"
        "$$a = S / K$$\n"
        "$$net = (1 - e) * a$$\n",
        encoding="utf-8",
    )
    pipeline = CanonicalResearchIngestionPipeline(tmp_path)
    ir = pipeline.ingest(IngestionRequest(upload_paths=[str(spec_file)], title="BS Fee"))
    # S and K are free parameters; e=0.0025 is a user constant (never Euler's e)
    unresolved_syms = {f.target_symbol_or_eq for f in ir.disambiguation_flags if not f.resolved}
    assert {"S", "K"} <= unresolved_syms

    override = IRReviewOverride(
        symbol_resolutions=[
            SymbolResolution(symbol="S", role="FREE_PARAMETER", si_value=100.0, bounds=IntervalBoundModel(lower=50.0, upper=200.0)),
            SymbolResolution(symbol="K", role="FREE_PARAMETER", si_value=100.0, bounds=IntervalBoundModel(lower=50.0, upper=200.0)),
            SymbolResolution(symbol="a", role="DERIVED_BY_EQUATION"),
            SymbolResolution(symbol="net", role="DERIVED_BY_EQUATION"),
        ],
        approve=True,
    )
    approved = apply_ir_override(ir, override)
    assert approved.user_approved is True
    assert all(f.resolved for f in approved.disambiguation_flags)
    const_map = {c.symbol: c for c in approved.constants}
    assert const_map["S"].is_tunable_parameter is True
    assert const_map["K"].is_tunable_parameter is True
    assert const_map["e"].si_value == pytest.approx(0.0025)


@settings(max_examples=40, deadline=None)
@given(
    dims=st.lists(
        st.sampled_from([-1, -0.5, 0, 0.5, 1, -0.25, 0.25]),
        min_size=7,
        max_size=7,
    )
)
def test_hypothesis_fractional_si_dim_exponents_accepted(dims):
    """R2-ARCH-006 / R2-BUG-005: Rational half/quarter-integer SI exponents are accepted across schemas."""
    slot = TypedResearchSlot(symbol="sigma", slot_idx=0, si_dim=dims, role="CONTROL_PARAM")
    const = TypedConstant(symbol="sigma_c", value=0.2, si_dim=dims)
    derived = DerivedExpressionSpec(symbol="vol", expression="sigma", si_dim=dims)
    ihcg_slot = m.SlotSpec(slot_idx=0, symbol="sigma", si_dim=dims, role="CONTROL_PARAM")
    ihcg_eq = m.EquationSpec(expression="sigma", si_dim=dims)

    assert len(slot.si_dim) == 7
    assert len(const.si_dim) == 7
    assert len(derived.si_dim) == 7
    assert len(ihcg_slot.si_dim) == 7
    assert len(ihcg_eq.si_dim) == 7


def test_invalid_fractional_si_dim_gives_repair_hint():
    with pytest.raises(ValueError, match="0.25"):
        TypedResearchSlot(symbol="bad", slot_idx=0, si_dim=[0, 0, 0.333, 0, 0, 0, 0], role="CONTROL_PARAM")


@settings(max_examples=25, deadline=None)
@given(
    depth=st.integers(min_value=2, max_value=32),
    fn=st.sampled_from(["max", "min"]),
)
def test_hypothesis_deeply_nested_max_min_roundtrip(depth: int, fn: str):
    """R2-BUG-006: Nested max/min chains of depth 2..32 parse and round-trip cleanly."""
    syms = [f"u_x{i}" for i in range(depth)]
    expr = syms[-1]
    for sym in reversed(syms[:-1]):
        expr = f"{fn}({sym}, {expr})"

    norm = normalize_expression(expr, declared_symbols=syms)
    assert norm.ok, f"Failed at depth={depth}: {norm.diagnostics}"
    assert norm.canonical is not None
    parsed = m._parse(norm.canonical)
    assert parsed is not None

    # Variadic form must also fold cleanly into binary nests
    variadic = f"{fn}({', '.join(syms)})"
    norm_var = normalize_expression(variadic, declared_symbols=syms)
    assert norm_var.ok, f"Variadic failed at depth={depth}: {norm_var.diagnostics}"
    assert norm_var.canonical is not None
    assert m._parse(norm_var.canonical) is not None


def test_builder_cli_ingest_subcommand(tmp_path: Path, capsys):
    """R2-BUG-009: `invar-agent builder ingest` drives Builder Studio's CanonicalResearchIngestionPipeline."""
    from invar_agent.cli import main

    spec = tmp_path / "decay.md"
    spec.write_text("$$dx/dt = -k * x$$\nGiven $k = 0.5\\,\\mathrm{1/s}$ and $x(0) = 1.0$.\n", encoding="utf-8")
    out_ir = tmp_path / "ir.json"
    rc = main(["builder", "ingest", str(tmp_path), "--upload", str(spec), "--title", "DecayCLI", "--out", str(out_ir)])
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["schema_version"] == "canonical_research_ir.v1"
    assert captured["title"] == "DecayCLI"
    assert out_ir.is_file()

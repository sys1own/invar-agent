"""Phase 5 Tests: Holographic HIN, Reverse-Mode Adjoint Jacobian, E-Graph & Tier-0 Healer."""

import json
import pytest

from invar_agent._native import (
    PyAdjointHin,
    PyHolographicHin,
    csel_egraph_optimize,
    desugar_math_to_csel,
    resolve_symbol_fuzzy,
)
from invar_agent.uast import (
    OntologyAndParameterSpec,
    PhysicalQuantity,
    ResearchComputeUAST,
    SymbolicGovernanceMap,
    parse_csel_to_ast,
)
from invar_agent.uast.healer import (
    build_counterexample_diagnostic_card,
    format_single_hole_repair_prompt,
    tier0_heal_uast,
)


def test_holographic_hin_mell_promotion_dereliction_and_smart_dimensions():
    quantities = [
        {
            "symbol": "x",
            "si_dimension": [0, 1, 0, 0, 0, 0, 0],
            "interval": [-10.0, 10.0],
            "is_state_variable": True,
            "dimension_unspecified": False,
        },
        {
            "symbol": "unused_param",
            "si_dimension": [1, 0, 0, 0, 0, 0, 0],
            "interval": [0.0, 100.0],
            "is_state_variable": False,
            "dimension_unspecified": False,
        },
    ]
    hin = PyHolographicHin(json.dumps(quantities))
    # Literal `1.0` unifies with `x` (Length) without false-positive dimensional error;
    # `x` used twice triggers `!x (x2)`; `unused_param` triggers `?unused_param`.
    eqs = {"x": parse_csel_to_ast("x * x + 1.0")}
    rep = json.loads(hin.verify_equations(json.dumps(eqs)))
    assert rep["all_valid"] is True
    eq_rep = rep["equations"][0]
    assert "!x (x2)" in eq_rep["promotion_agents"]
    assert "?unused_param" in eq_rep["dereliction_agents"]

    # Unbound wire must be strictly caught by MELL verification
    bad_eqs = {"x": parse_csel_to_ast("x + hallucinated_var")}
    bad_rep = json.loads(hin.verify_equations(json.dumps(bad_eqs)))
    assert bad_rep["all_valid"] is False
    assert bad_rep["fault_node_id"] == "x"
    assert "hallucinated_var" in bad_rep["equations"][0]["unbound_symbols"]


def test_adjoint_hin_symbolic_jacobian_and_stiffness_detection():
    eqs = {
        "x": parse_csel_to_ast("-1000000.0 * x + y"),
        "y": parse_csel_to_ast("x - y"),
    }
    adj = PyAdjointHin(["x", "y"], json.dumps(eqs))
    res = json.loads(adj.analyze(json.dumps({"x": 1.0, "y": 0.0})))
    jac = res["jacobian"]
    assert jac[0][0] == pytest.approx(-1e6)
    assert jac[0][1] == pytest.approx(1.0)
    assert jac[1][0] == pytest.approx(1.0)
    assert jac[1][1] == pytest.approx(-1.0)
    assert res["spectral_radius"] >= 1e6 - 1.0
    assert res["stiffness_ratio"] >= 1e5
    assert res["recommended_solver"] == "ros3pr"


def test_egraph_cancellation_and_stiff_imex_rules():
    # safe_log(1 + x) -> log1p(x)
    ast_log = parse_csel_to_ast("safe_log(1 + x)")
    opt_log = json.loads(csel_egraph_optimize(json.dumps(ast_log)))
    assert opt_log["expr"]["func"] == "log1p"

    # safe_exp(x) - 1 -> expm1(x)
    ast_exp = parse_csel_to_ast("safe_exp(x) - 1")
    opt_exp = json.loads(csel_egraph_optimize(json.dumps(ast_exp)))
    assert opt_exp["expr"]["func"] == "expm1"

    # Stiff linear decay separation -> imex_split
    ast_stiff = parse_csel_to_ast("-5000 * x + y")
    opt_stiff = json.loads(csel_egraph_optimize(json.dumps(ast_stiff)))
    assert opt_stiff["has_imex_split"] is True


def test_tier0_symbol_auto_resolution_and_codata_injection():
    uast = ResearchComputeUAST(
        ontology=OntologyAndParameterSpec(
            quantities={
                "initial_velocity": PhysicalQuantity(
                    symbol="initial_velocity", is_state_variable=True, initial_value=10.0
                ),
                "damping": PhysicalQuantity(symbol="damping", default_value=0.1),
            }
        ),
        governance=SymbolicGovernanceMap(
            # Uses CamelCase typo `InitialVelocity`, Levenshtein typo `dampng`, and CODATA `g`
            equations={"initial_velocity": "-dampng * InitialVelocity - g"}
        ),
    )
    healed, logs = tier0_heal_uast(uast)
    assert "g" in healed.ontology.quantities
    assert healed.ontology.quantities["g"].default_value == pytest.approx(9.80665)
    assert any("resolved_symbol(InitialVelocity->initial_velocity)" in l for l in logs)
    assert any("resolved_symbol(dampng->damping)" in l for l in logs)
    assert any("injected_codata(g=" in l for l in logs)


def test_counterexample_diagnostic_card_and_single_hole_prompt():
    uast = ResearchComputeUAST(
        ontology=OntologyAndParameterSpec(
            quantities={"x": PhysicalQuantity(symbol="x", is_state_variable=True, initial_value=1.0)}
        ),
        governance=SymbolicGovernanceMap(equations={"x": "x + missing_term"}),
    )
    card = build_counterexample_diagnostic_card(
        uast,
        "x",
        smt_assertion_failed="x >= 0",
    )
    prompt = format_single_hole_repair_prompt(card)
    assert "node f_k (x)" in prompt
    assert '"target_node_id": "x"' in prompt
    assert "missing_term" in prompt

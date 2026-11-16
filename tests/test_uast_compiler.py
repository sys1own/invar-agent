"""Phase 5 Tests: Declarative ResearchComputeUAST, CSEL Parser, and Deterministic Lowering Compiler."""

import pytest

from invar_agent.research.state_machine import shacl_chiasmus_firewall, v9_transpiler_violations
from invar_agent.transpiler.frontend import python_source_to_uast
from invar_agent.uast import (
    CselSyntaxError,
    InvariantContractSpec,
    OntologyAndParameterSpec,
    PhysicalQuantity,
    ResearchComputeUAST,
    SymbolicGovernanceMap,
    apply_perturbation_delta,
    compile_uast_to_python,
    parse_csel_to_ast,
)
from invar_agent.uast.schemas import OperatorMutation, UASTPerturbationDelta
from invar_agent.uast.healer import coerce_llm_json_to_uast, tier0_heal_uast
from invar_agent.uast.totalized_math import (
    safe_acos,
    safe_asin,
    safe_div,
    safe_exp,
    safe_log,
    safe_pow,
    safe_sqrt,
)


def test_totalized_operators_continuity_and_edge_cases():
    # Continuous safe_div across zero: copysign(eps, 0.0) == +eps
    assert safe_div(1.0, 0.0) == pytest.approx(1e12)
    assert safe_div(1.0, -0.0) == pytest.approx(1e12)
    assert safe_div(1.0, 1e-15) == pytest.approx(1e12)
    assert safe_div(1.0, -1e-15) == pytest.approx(-1e12)
    assert safe_div(6.0, -2.0) == pytest.approx(-3.0)

    # safe_pow preserves exact integer powers on negative bases and 0
    assert safe_pow(-2.0, 2.0) == pytest.approx(4.0)
    assert safe_pow(-2.0, 3.0) == pytest.approx(-8.0)
    assert safe_pow(0.0, 2.0) == pytest.approx(0.0)
    assert safe_pow(-4.0, 0.5) == pytest.approx(2.0)

    # safe_exp clamps exponent to [-80, 80] so squaring never overflows f64
    assert safe_exp(1000.0) == pytest.approx(safe_exp(80.0))
    assert (safe_exp(1000.0) ** 2) < 1e300

    # safe_log, safe_sqrt, safe_acos, safe_asin
    assert safe_log(-10.0) == pytest.approx(safe_log(1e-30))
    assert safe_sqrt(-9.0) == 0.0
    assert safe_acos(5.0) == 0.0
    assert safe_asin(-5.0) == pytest.approx(-1.5707963267948966)


@pytest.mark.parametrize(
    "bad_expr",
    [
        "__import__('os').system('ls')",
        "eval('1+1')",
        "lambda x: x + 1",
        "while True: pass",
        "os.path.join('a', 'b')",
        "exec('x=1')",
        "x[0] + 1",
    ],
)
def test_csel_rejects_imperative_and_unsafe_constructs(bad_expr):
    with pytest.raises(CselSyntaxError):
        parse_csel_to_ast(bad_expr)


def test_tier0_desugars_latex_implicit_mul_and_compiles_cleanly():
    raw_json = """
    ```json
    {
      "title": "Damped Harmonic Oscillator",
      "hypothesis": "Energy decays monotonically under linear damping.",
      "ontology": {
        "quantities": {
          "x": {"symbol": "x", "si_dimension": [0, 1, 0, 0, 0, 0, 0], "is_state_variable": true, "initial_value": 1.0},
          "v": {"symbol": "v", "si_dimension": [0, 1, -1, 0, 0, 0, 0], "is_state_variable": true, "initial_value": 0.0},
          "k": {"symbol": "k", "si_dimension": [1, 0, -2, 0, 0, 0, 0], "default_value": 4.0},
          "m": {"symbol": "m", "si_dimension": [1, 0, 0, 0, 0, 0, 0], "default_value": 1.0},
          "γ": {"symbol": "gamma", "si_dimension": [1, 0, -1, 0, 0, 0, 0], "default_value": 0.2}
        }
      },
      "governance": {
        "equations": {
          "x": "v",
          "v": "\\\\frac{-k x}{m} - 2\\\\gamma\\\\cdot v + 0.0*\\\\left(\\\\sin(t)\\\\right)"
        },
        "observables": {
          "energy_joules": "0.5*m*v^2 + 0.5*k*x^2"
        }
      },
      "invariants": {
        "custom_smt_assertions": ["energy_joules >= 0.0"]
      },
      "dt": 0.01,
      "steps": 200,
      "target_backend": "multi_variant"
    }
    ```
    """
    uast = coerce_llm_json_to_uast(raw_json)
    healed, logs = tier0_heal_uast(uast)
    assert any("desugared_math_to_csel" in l for l in logs)

    compiled = compile_uast_to_python(healed)
    assert shacl_chiasmus_firewall(compiled.python_code) == []
    assert v9_transpiler_violations(compiled.python_code) == []
    # Also must lower cleanly to UAST for CppEmitter
    uast_ir = python_source_to_uast(compiled.python_code)
    assert uast_ir["type"] == "module"


def test_perturbation_delta_splicing():
    base = ResearchComputeUAST(
        ontology=OntologyAndParameterSpec(
            quantities={
                "x": PhysicalQuantity(symbol="x", is_state_variable=True, initial_value=1.0),
                "v": PhysicalQuantity(symbol="v", is_state_variable=True, initial_value=0.0),
                "k": PhysicalQuantity(symbol="k", default_value=2.0),
            }
        ),
        governance=SymbolicGovernanceMap(equations={"x": "v", "v": "-k * x"}),
        invariants=InvariantContractSpec(),
    )
    delta = UASTPerturbationDelta(
        add_quantities={
            "mu": PhysicalQuantity(symbol="mu", default_value=0.5),
        },
        add_terms={"v": "-mu * (x^2 - 1.0) * v"},
        modify_terms=[],
    )
    evolved = apply_perturbation_delta(base, delta)
    assert "mu" in evolved.ontology.quantities
    assert "mu" in evolved.governance.equations["v"]
    compiled = compile_uast_to_python(evolved)
    assert v9_transpiler_violations(compiled.python_code) == []

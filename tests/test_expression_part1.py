"""UPG-006 Part 1 (BUG-008, 027, NF-05, 11) tests: float exponents, Builder grammar, CSEL, adjoint, symbolic kernel."""
from __future__ import annotations

import json
import pytest
import sympy as sp

from invar_agent import _native
from invar_agent.builder.ingestion.equation_lowerer import (
    FormalGateError,
    lower_and_verify_equations,
    to_csel,
)
from invar_agent.builder.ingestion.schemas import RawEquation, SourceSpan
from invar_agent.ir.lower_uast import _render_expr_to_python
from invar_agent.research.evolution import _SymbolicKernel
from invar_agent.research.meaningfulness import _parse
from invar_agent.research.stochastic_equilibrium import _Roundoff, UnsupportedVerification


def test_bug008_integral_float_exponents_render_and_bind():
    rendered = _render_expr_to_python(
        {"kind": "binary", "op": "pow", "lhs": {"kind": "sym", "name": "x"}, "rhs": {"kind": "const", "value": 2.0}},
        {"x"},
    )
    assert rendered == "(x ** 2)"

    ro = _Roundoff("def simulation_step(buf, dt):\n    pass\n", {"x": 0}, {"x": (1.0, 2.0)}, 0.1)
    env = {"x": ((1.0, 2.0), 0.0)}
    val2, err2 = ro.expression(_parse("x ** 2.0"), env, 0)
    assert val2[0] >= 0.99 and val2[1] <= 4.01 and err2 >= 0.0
    valm1, errm1 = ro.expression(_parse("x ** -1.0"), env, 0)
    assert valm1[0] >= 0.49 and valm1[1] <= 1.01 and errm1 >= 0.0
    with pytest.raises(UnsupportedVerification):
        ro.expression(_parse("x ** 0.5"), env, 0)


def test_bug027_constant_vs_equation_grammar_and_sci_notation():
    prov = SourceSpan(file="spec.tex", location="L1", quote="test")
    eqs, consts, _flags = lower_and_verify_equations(
        [
            RawEquation(text=r"y = 2 \cdot x", provenance=prov),
            RawEquation(text=r"k_B = 1.380649 \times 10^{-23} J/K", provenance=prov),
            RawEquation(text=r"g = 9.81 m/s^2", provenance=prov),
        ],
        [],
        {"state_symbols": ["x", "y"]},
    )
    assert len(eqs) == 1
    assert eqs[0].lhs_symbol == "y"
    const_map = {c.symbol: c for c in consts}
    assert "k_B" in const_map and abs(const_map["k_B"].si_value - 1.380649e-23) < 1e-28
    assert "g" in const_map and abs(const_map["g"].si_value - 9.81) < 1e-9


def test_nf11_builder_csel_minmax_and_piecewise_fail_closed():
    x = sp.Symbol("x")
    csel_min = to_csel(sp.Min(x, sp.Integer(2)))
    assert csel_min["kind"] == "call" and csel_min["func"] == "min" and len(csel_min["args"]) == 2

    pw = sp.Piecewise((x, x > 0), (0, True))
    with pytest.raises(FormalGateError) as exc:
        to_csel(pw)
    assert exc.value.code == "CSEL_PIECEWISE_UNSUPPORTED"


def test_nf11_native_adjoint_minmax_and_unknown_call_fails_closed():
    # min(x, 2) - 3*x has derivative -2 for x < 2 and -3 for x > 2
    eq = {
        "x": {
            "kind": "binary",
            "op": "-",
            "left": {"kind": "call", "func": "min", "args": [{"kind": "symbol", "name": "x"}, {"kind": "literal", "value": 2.0}]},
            "right": {"kind": "binary", "op": "*", "left": {"kind": "literal", "value": 3.0}, "right": {"kind": "symbol", "name": "x"}},
        }
    }
    adj = _native.PyAdjointHin(["x"], json.dumps(eq))
    r_lo = json.loads(adj.analyze(json.dumps({"x": 1.0})))
    r_hi = json.loads(adj.analyze(json.dumps({"x": 3.0})))
    assert abs(r_lo["jacobian"][0][0] - (-2.0)) < 1e-9
    assert abs(r_hi["jacobian"][0][0] - (-3.0)) < 1e-9

    bad_eq = {"x": {"kind": "call", "func": "unknown_special_fn", "args": [{"kind": "symbol", "name": "x"}]}}
    bad_adj = _native.PyAdjointHin(["x"], json.dumps(bad_eq))
    with pytest.raises(ValueError, match="ADJOINT_UNSUPPORTED_FUNCTION"):
        bad_adj.analyze(json.dumps({"x": 1.0}))


def test_nf05_symbolic_kernel_fails_closed_on_unsupported_call():
    code = "import math\ndef simulation_step(buf, dt):\n    buf.data[0] = math.lgamma(buf.data[0])\n"
    sk = _SymbolicKernel(code, {"x": 0})
    with pytest.raises(ValueError, match="SYMBOLIC_EXPRESSION_UNSUPPORTED"):
        sk.block(sk.functions["simulation_step"].body, {"dt": sp.Symbol("dt")}, 0)

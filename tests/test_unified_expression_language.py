"""UPG-006 Full + NF-08: Unified Expression AST, FunctionSpec, NF-19 roundoff, Cond, ForcingSchedule."""
from __future__ import annotations

import json
import math
from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st

from invar_agent import _native
from invar_agent.ir.expr_ast import (
    Binary,
    Call,
    Cmp,
    Cond,
    Const,
    Sym,
    canonicalize,
    expr_to_formal_ast,
    expr_to_sciir,
    parse_expr,
    render_expr,
    sciir_to_expr,
)
from invar_agent.ir.function_registry import (
    DomainError,
    FUNCTION_REGISTRY,
    FunctionSpec,
    RegistryError,
    core_formal_functions,
    register,
)
from invar_agent.ir.lower_uast import compile_time_forcing_schedule
from invar_agent.research import meaningfulness as m
from invar_agent.research.stochastic_equilibrium import _Roundoff
from invar_agent.research.uniqueness import _scale_fractional_si_dims


def test_python_rust_function_registry_parity():
    native_specs = json.loads(_native.py_function_registry_spec())
    native_map = {s["parse_name"]: s for s in native_specs}
    py_core = core_formal_functions()
    assert set(native_map.keys()) == set(py_core.keys())
    for name, nspec in native_map.items():
        pspec = py_core[name]
        assert pspec.min_arity == nspec["min_arity"]
        assert pspec.max_arity == nspec["max_arity"]
        assert pspec.py_name == nspec["py_name"]
        assert pspec.cpp_name == nspec["cpp_name"]
        assert pspec.rust_name == nspec["rust_name"]
        assert pspec.csel_name == nspec["csel_name"]


def test_function_spec_register_enforces_six_members():
    bad = FunctionSpec(
        parse_name="_incomplete_fn_test",
        arity=(1, 1),
        eval=lambda x: x,
        interval_ext=None,
        roundoff_bound=None,
        symbolic_derivative=None,
        dimension_rule=None,
        py_name="x",
        cpp_name="x",
        rust_name="x",
        domain=None,
        formal_supported=True,
    )
    with pytest.raises(RegistryError):
        register(bad)


def test_expr_ast_python_rust_canonicalize_and_sciir_roundtrip():
    src = "ln(x) + clamp(y, 0.1, 1.5) + step(t - 10) + min(a, b, c)"
    e = parse_expr(src, "formal")
    py_c = canonicalize(e)
    rs_c_json = _native.py_expr_canonicalize(e.model_dump_json())
    assert json.loads(py_c.model_dump_json()) == json.loads(rs_c_json)

    for dialect in ("formal", "exec_py", "sci_text"):
        rendered = render_expr(py_c, dialect)
        reparsed = canonicalize(parse_expr(rendered, dialect))
        assert reparsed == py_c

    sciir = expr_to_sciir(py_c, include_exact=True)
    back = canonicalize(sciir_to_expr(sciir))
    assert back == py_c


@given(
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, deadline=None)
def test_hypothesis_expr_roundtrip(a: float, b: float):
    a_r = round(a, 3)
    b_r = round(b, 3)
    src = f"(x + {a_r}) * (y - {b_r}) + (1 if x >= 0 else 0)"
    e = canonicalize(parse_expr(src, "formal"))
    for d in ("formal", "exec_py", "sci_text"):
        out = canonicalize(parse_expr(render_expr(e, d), d))
        assert out == e


def test_sqrt_bound_is_not_image_width():
    """NF-19: On [1, 2] with e_in=1e-12, sqrt error must be ~O(1e-12), NOT image width 0.414."""
    ro = _Roundoff("def simulation_step(buf, dt):\n    pass\n", {"x": 0}, {"x": (1.0, 2.0)}, 0.1)
    (lo, hi), err = ro.expression(m._parse("sqrt(x)"), {"x": ((1.0, 2.0), 1e-12)}, 0)
    assert lo >= 0.999 and hi <= 1.415
    assert err < 1e-10, f"NF-19 violated: sqrt error {err} should be ~1e-12, not image width 0.414"


def test_tan_pole_rejection_and_safe_interval():
    with pytest.raises(m._DomainError, match="TAN_POLE_IN_DOMAIN"):
        m._iv(m._parse("tan(x)"), {"x": (1.0, 2.0)})
    with pytest.raises(m._DomainError, match="TAN_POLE_IN_DOMAIN"):
        m._iv(m._parse("tan(x)"), {"x": (-2.0, 2.0)})
    iv = m._iv(m._parse("tan(x)"), {"x": (-0.5, 0.5)})
    assert iv.lo < math.tan(-0.5) + 1e-12 and iv.hi > math.tan(0.5) - 1e-12


def test_cond_narrowed_box_and_jump_rounding():
    # Narrowed box prevents division by zero in branch `1/x if x >= 1 else 0` over x in [-1, 2]
    iv = m._iv(m._parse("1 / x if x >= 1 else 0"), {"x": (-1.0, 2.0)})
    assert iv.lo <= 0.0 and iv.hi >= 1.0

    # Ambiguous conditional in _Roundoff adds jump term
    ro = _Roundoff("def simulation_step(buf, dt):\n    pass\n", {"x": 0}, {"x": (-1.0, 2.0)}, 0.1)
    (_lo, _hi), err = ro.expression(m._parse("10 if x >= 0 else 0"), {"x": ((-1.0, 2.0), 0.0)}, 0)
    assert err >= 10.0


def test_r5_forcing_schedule_extraction_and_state_dependent_rejection():
    rhs = "(1 if (10 <= t and t <= 40) else 0) - k * C"
    smooth_rhs, sched, domains = compile_time_forcing_schedule(
        rhs, "t", 0.0, 100.0, state_symbols={"C"}
    )
    assert "u__0" in smooth_rhs
    assert sched is not None
    assert len(sched.segments) == 3
    assert [(s.t_start, s.t_end, s.values["u__0"]) for s in sched.segments] == [
        ("0", "10", "0"),
        ("10", "40", "1"),
        ("40", "100", "0"),
    ]
    assert domains["u__0"] == (0.0, 1.0)

    with pytest.raises(ValueError, match="PIECEWISE_STATE_DEPENDENT_UNSUPPORTED"):
        compile_time_forcing_schedule("1 if C >= 2 else 0", "t", 0.0, 100.0, state_symbols={"C"})


def test_nf08_fractional_si_dim_lcm_scaling():
    eqs = [{"expr": "a - b", "symbols": {"a": {"si_dim": [0, 1, Fraction(1, 2), 0, 0, 0, 0]}}}]
    scaled = _scale_fractional_si_dims(eqs)
    assert scaled[0]["symbols"]["a"]["si_dim"] == [0, 2, 1, 0, 0, 0, 0]
    # Integer system is untouched
    int_eqs = [{"expr": "a - b", "symbols": {"a": {"si_dim": [0, 1, -1, 0, 0, 0, 0]}}}]
    assert _scale_fractional_si_dims(int_eqs) is int_eqs

"""UPG-015 (BUG-014, 029, NF-14): Grounded proposals, Colebrook ROOT_FIND, Butterworth DISCRETE_TIME_SYSTEM."""
from __future__ import annotations

import math
import pytest

from invar_agent.builder.ingestion.schemas import EquationProposal, ProposedSymbol
from invar_agent.builder.solvers_ext import (
    analyze_discrete_iir,
    emit_discrete_iir_python,
    emit_discrete_iir_rust,
    emit_root_find_cpp,
    emit_root_find_python,
    emit_root_find_rust,
    parse_discrete_difference_equation,
    solve_safeguarded_newton,
    validate_equation_proposal,
)


def test_upg015_grounded_equation_proposal_and_ungrounded_rejections():
    src = ["The Darcy friction factor obeys 1/sqrt(f) + 2*log10(eps/(3.7*D) + 2.51/(Re*sqrt(f))) = 0."]
    good = EquationProposal(
        latex="1/sqrt(f) + 2*log10(eps/(3.7*D) + 2.51/(Re*sqrt(f))) = 0",
        quote_span="1/sqrt(f) + 2*log10(eps/(3.7*D) + 2.51/(Re*sqrt(f))) = 0",
        symbols=[ProposedSymbol(name="f", role="OUTPUT", unit="1")],
    )
    ok, err = validate_equation_proposal(good, src)
    assert ok is True and err is None

    # Hallucinated numeric constant 9.99 not in quote
    bad_num = good.model_copy(update={"latex": "1/sqrt(f) + 9.99*log10(eps/(3.7*D) + 2.51/(Re*sqrt(f))) = 0"})
    ok2, err2 = validate_equation_proposal(bad_num, src)
    assert ok2 is False and "PROPOSAL_UNGROUNDED" in (err2 or "")

    # Hallucinated quote span not in source
    bad_q = good.model_copy(update={"quote_span": "not in source document"})
    ok3, err3 = validate_equation_proposal(bad_q, src)
    assert ok3 is False and "PROPOSAL_UNGROUNDED" in (err3 or "")


def test_upg015_colebrook_white_root_find_and_emitters():
    eps_over_d = 0.00075  # audit roughness ratio (eps/D = 7.5e-4)
    for Re, expected_f in ((1e5, 0.021295417), (1e6, 0.018708791)):
        def _f(f: float, _Re=Re) -> float:
            return 1.0 / math.sqrt(f) + 2.0 * math.log10(eps_over_d / 3.7 + 2.51 / (_Re * math.sqrt(f)))

        def _df(f: float, _Re=Re) -> float:
            sf = math.sqrt(f)
            arg = eps_over_d / 3.7 + 2.51 / (_Re * sf)
            return -0.5 / (f * sf) - (2.0 / math.log(10.0)) * (2.51 / (2.0 * _Re * f * sf)) / arg

        cert = solve_safeguarded_newton(_f, _df, 0.005, 0.1, symbol="f", equation_id="colebrook")
        assert abs(cert.root - expected_f) < 1e-8, f"Re={Re}: got {cert.root}, expected {expected_f}"
        assert abs(cert.residual) < 1e-12

    # Missing sign change raises ROOT_BRACKET_REQUIRED
    with pytest.raises(ValueError, match="ROOT_BRACKET_REQUIRED"):
        solve_safeguarded_newton(lambda x: x * x + 1.0, lambda x: 2.0 * x, 0.0, 1.0)

    py_src = emit_root_find_python("solve_f", "f - 0.02", "1.0", "f", ["Re"])
    cpp_src = emit_root_find_cpp("solve_f", "f - 0.02", "1.0", "f", ["Re"])
    rs_src = emit_root_find_rust("solve_f", "f - 0.02", "1.0", "f", ["Re"])
    assert "ROOT_BRACKET_REQUIRED" in py_src and "std::nextafter" in cpp_src and "f64::NAN" in rs_src


def test_upg015_butterworth_discrete_time_system_and_emitters():
    b = [0.020083365564211, 0.040166731128424, 0.020083365564211]
    a = [1.0, -1.561018075800718, 0.641351538057563]
    # Cutoff normalized frequency w_c = 0.1 * pi (fc/fs = 0.05 -> 2*pi*0.05 = 0.1*pi)
    wc = 0.1 * math.pi
    res = analyze_discrete_iir(b, a, step_samples=64, omegas=(0.0, wc, math.pi))
    assert res["status"] == "STABLE"
    assert res["pole_radius"] < 1.0
    assert abs(res["frequency_response"][0.0] - 1.0) < 1e-6
    assert abs(res["frequency_response"][wc] - 0.7071068) < 1e-4
    assert abs(res["frequency_response"][math.pi]) < 1e-9
    assert res["peak_index"] == 14
    assert abs(res["overshoot_pct"] - 4.4447) < 0.01

    # Discrete lag parsing & ambiguity rejection
    parsed = parse_discrete_difference_equation("y[n] = 0.5*x[n] + 0.5*x[n-1] - 0.2*y[n-1]")
    assert parsed["is_discrete"] is True and len(parsed["terms"]) == 4
    with pytest.raises(ValueError, match="DISCRETE_LAG_AMBIGUOUS"):
        parse_discrete_difference_equation("y = 0.5*x_1 - 0.2*y_1")

    py_fn = emit_discrete_iir_python("butter2", b, a)
    rs_fn = emit_discrete_iir_rust("butter2", b, a)
    ns: dict = {}
    exec(py_fn, ns)
    out = ns["butter2"]([1.0] * 64)
    assert abs(out[14] - res["step_response"][14]) < 1e-12
    assert "pub fn butter2" in rs_fn

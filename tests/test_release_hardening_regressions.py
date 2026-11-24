"""Deterministic offline regression tests for universal release-hardening fixes."""
from __future__ import annotations

import json
import pytest

from invar_agent.research import meaningfulness as m
from invar_agent.research.epistemic import formal_contracts
from invar_agent.research.feasibility import override_refusal_for_computational_benchmark
from invar_agent.research.ihcg_derivation import derive_ihcg_evidence, extract_prompt_domains
from invar_agent.research.ihcg_normalization import normalize_ihcg_evidence
from invar_agent.research.planner import parse_blueprint
from invar_agent.schemas.research import FeasibilityAssessment, FeasibilityVerdict


def test_direct_contradiction_refusal_is_not_overridden_to_surrogate():
    question = (
        "Formulate a thermal equilibrium model with a single positive scalar temperature T "
        "satisfying both T >= 500.0 K and T <= 300.0 K simultaneously under standard "
        "thermodynamic conservation laws."
    )
    assessment = FeasibilityAssessment(
        verdict=FeasibilityVerdict.REFUSED_IMPOSSIBLE,
        rationale="A single scalar temperature cannot simultaneously satisfy T >= 500 K and T <= 300 K; the feasible set is the empty set.",
        violated_boundary="mutually exclusive inequality bounds",
    )
    res = override_refusal_for_computational_benchmark(question, assessment)
    assert res.verdict == FeasibilityVerdict.REFUSED_IMPOSSIBLE


def test_parse_blueprint_validates_typed_slots_unit_scale():
    payload = {
        "blueprint_id": "bp_bad_scale",
        "title": "Bad unit scale",
        "falsifiable_hypothesis": "x >= 0",
        "search_queries": [],
        "code_template": "def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0]\n",
        "variable_map": {"x": 0},
        "typed_slots": [
            {"symbol": "x", "slot_idx": 0, "si_dim": [0] * 7, "role": "CONTROL_PARAM", "unit_scale": [1, 0]}
        ],
        "contracts": [
            {
                "contract_id": "c1",
                "target_backend": "cpp",
                "initial_state": [1.0],
                "dt": 0.1,
                "steps": 10,
                "invariants": [{"variable_name": "x", "assertion": "state[0] >= 0", "tolerance": 1e-6}],
            }
        ],
    }
    with pytest.raises(ValueError, match="positive rationals"):
        parse_blueprint(payload)


def test_normalize_ihcg_evidence_lossless_math_syntax():
    raw = {
        "archetype": "ODE_PDE",
        "domain": {"x": [0.1, 2.0], "y": [-5.0, 5.0]},
        "evidence": {
            "slots": [
                {"symbol": "x", "slot_idx": 0, "si_dim": [0] * 7, "role": "CONTROL_PARAM"},
                {"symbol": "y", "slot_idx": 1, "si_dim": [0] * 7, "role": "DERIVED_OBSERVABLE"},
            ],
            "equations": {"y": {"expression": "math.log(x) + ln(x) + x^2", "kind": "ALGEBRAIC", "si_dim": [0] * 7}},
            "environmental_assumptions": ["math.fabs(x) > 0"],
            "obligations": [
                {"id": "h1", "expression": "y >= -10", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "y >= -10"}
            ],
            "quantifier_mode": "EXISTS",
            "mechanism_parameters": ["x"],
            "control_values": {"x": 1.0},
        },
    }
    norm = normalize_ihcg_evidence(raw)
    assert norm["evidence"]["equations"]["y"]["expression"] == "log(x) + log(x) + x**2"
    assert norm["evidence"]["environmental_assumptions"] == ["abs(x) > 0"]


def test_extract_prompt_domains_supports_across_and_over():
    facts = extract_prompt_domains("Sweep R_0 across [2.5, 4.0] and p_v over [0.0, 1.0].", {"R_0", "p_v"})
    assert facts["R_0"]["bounds"] == [2.5, 4.0]
    assert facts["p_v"]["bounds"] == [0.0, 1.0]


def test_derive_ihcg_evidence_derives_trajectory_tag_and_rounds_control_enclosure():
    blueprint = {
        "variable_map": {"R0": 0, "I": 1, "I_max": 2},
        "typed_slots": [
            {"symbol": "R0", "slot_idx": 0, "si_dim": [0] * 7, "role": "CONTROL_PARAM"},
            {"symbol": "I", "slot_idx": 1, "si_dim": [0] * 7, "role": "DYNAMIC_STATE"},
            {"symbol": "I_max", "slot_idx": 2, "si_dim": [0] * 7, "role": "DERIVED_OBSERVABLE", "semantic_tag": 0},
        ],
    }
    raw = {
        "archetype": "ODE_PDE",
        "domain": {"R0": [0.30303, 0.30303], "I": [0.0, 1.0], "I_max": [0.0, 1.0]},
        "evidence": {
            "equations": {
                "dI": {"expression": "0.1 * (R0 - 1.0) * I", "kind": "ODE_RHS", "state_symbol": "I", "si_dim": [0, 0, -1, 0, 0, 0, 0]},
                "I_max": {"expression": "I", "kind": "TRAJECTORY_MAX", "state_symbol": "I", "si_dim": [0] * 7},
            },
            "environmental_assumptions": ["I >= 0"],
            "obligations": [
                {"id": "h1", "expression": "I_max <= 0.15", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "I_max <= 0.15"}
            ],
            "quantifier_mode": "FORALL",
            "mechanism_parameters": ["R0"],
            "control_values": {"R0": 10.0 / 33.0},
        },
    }
    derived = derive_ihcg_evidence("Test I_max <= 0.15", blueprint, raw)
    imax_slot = next(s for s in derived["evidence"]["slots"] if s["symbol"] == "I_max")
    assert imax_slot["semantic_tag"] == m.TRAJECTORY_MAXIMUM
    lo, hi = derived["domain"]["R0"]
    assert lo <= 10.0 / 33.0 <= hi


def test_formal_contracts_skips_liveness_telemetry_guard_when_math_contracts_present():
    bp = parse_blueprint({
        "blueprint_id": "bp_tele",
        "title": "Telemetry guard test",
        "falsifiable_hypothesis": "x >= 1",
        "search_queries": [],
        "code_template": "def simulation_step(buf, dt):\n    buf.data[0] = 2.0\n",
        "variable_map": {"x": 0},
        "contracts": [
            {
                "contract_id": "c1",
                "target_backend": "cpp",
                "initial_state": [2.0],
                "dt": 0.1,
                "steps": 5,
                "invariants": [
                    {"variable_name": "finite", "assertion": "telemetry['finite'] == 1.0", "tolerance": 0.0},
                    {"variable_name": "x", "assertion": "state[0] >= 1.0", "tolerance": 1e-6},
                ],
            }
        ],
    })
    graph = m.IhcgCompiler().compile_from_prompt(
        "Verify x >= 1",
        evidence={
            "slots": [{"symbol": "x", "slot_idx": 0, "si_dim": [0] * 7, "role": "DERIVED_OBSERVABLE"}],
            "equations": {"x": {"expression": "2.0", "kind": "ALGEBRAIC", "si_dim": [0] * 7}},
            "environmental_assumptions": [],
            "obligations": [
                {"id": "h1", "expression": "x >= 1", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "x >= 1", "prompt_span": [7, 13]}
            ],
            "quantifier_mode": "EXISTS",
            "mechanism_parameters": [],
            "control_values": {},
        },
        domain={"x": [0.0, 5.0]},
        archetype="ODE_PDE",
    )
    compiled = formal_contracts(bp, graph)
    assert compiled == ("x >= 1.0",)


def test_compile_ingest_ihcg_accepts_conservation_guard_obligations():
    from invar_agent.research.feasibility import compile_ingest_ihcg
    graph = compile_ingest_ihcg(
        "Verify x >= 1 while conserving s + r == 1",
        evidence={
            "slots": [
                {"symbol": "x", "slot_idx": 0, "si_dim": [0] * 7, "role": "DERIVED_OBSERVABLE"},
                {"symbol": "s", "slot_idx": 1, "si_dim": [0] * 7, "role": "DYNAMIC_STATE"},
                {"symbol": "r", "slot_idx": 2, "si_dim": [0] * 7, "role": "DYNAMIC_STATE"},
            ],
            "equations": {"x": {"expression": "s + 2 * r", "kind": "ALGEBRAIC", "si_dim": [0] * 7}},
            "environmental_assumptions": ["s >= 0", "r >= 0"],
            "obligations": [
                {"id": "h1", "expression": "x >= 1", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "x >= 1", "prompt_span": [7, 13]},
                {"id": "g1", "expression": "s + r == 1", "obligation_class": "CONSERVATION_GUARD", "prompt_quote": "s + r == 1", "prompt_span": [31, 41]},
            ],
            "quantifier_mode": "EXISTS",
            "mechanism_parameters": [],
            "control_values": {},
        },
        domain={"x": [0.0, 3.0], "s": [0.0, 1.0], "r": [0.0, 1.0]},
        archetype="ODE_PDE",
    )
    assert graph.h_ihcg


def test_derive_ihcg_evidence_preserves_slots_referenced_by_contracts():
    blueprint = {
        "variable_map": {"x": 0, "aux_guard": 1},
        "typed_slots": [
            {"symbol": "x", "slot_idx": 0, "si_dim": [0] * 7, "role": "DERIVED_OBSERVABLE"},
            {"symbol": "aux_guard", "slot_idx": 1, "si_dim": [0] * 7, "role": "DERIVED_OBSERVABLE"},
        ],
        "contracts": [
            {"contract_id": "c1", "invariants": [{"variable_name": "aux_guard", "assertion": "state[1] >= 0"}]}
        ],
    }
    raw = {
        "archetype": "ODE_PDE",
        "domain": {"x": [0.0, 2.0], "aux_guard": [0.0, 1.0]},
        "evidence": {
            "equations": {"x": {"expression": "1.5", "kind": "ALGEBRAIC", "si_dim": [0] * 7}},
            "environmental_assumptions": [],
            "obligations": [
                {"id": "h1", "expression": "x >= 1", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "x >= 1"}
            ],
            "quantifier_mode": "EXISTS",
            "mechanism_parameters": [],
            "control_values": {},
        },
    }
    derived = derive_ihcg_evidence("Verify x >= 1", blueprint, raw)
    assert {s["symbol"] for s in derived["evidence"]["slots"]} == {"x", "aux_guard"}


def test_formal_contracts_substitutes_initial_state_subscripts():
    bp = parse_blueprint({
        "blueprint_id": "bp_init_sub",
        "title": "Initial state subscript test",
        "falsifiable_hypothesis": "x >= 0.5",
        "search_queries": [],
        "code_template": "def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0]\n",
        "variable_map": {"x": 0},
        "contracts": [
            {
                "contract_id": "c1",
                "target_backend": "cpp",
                "initial_state": [0.5],
                "dt": 0.1,
                "steps": 5,
                "invariants": [{"variable_name": "x", "assertion": "state[0] >= initial[0]", "tolerance": 1e-6}],
            }
        ],
    })
    graph = m.IhcgCompiler().compile_from_prompt(
        "Verify x >= 0.5",
        evidence={
            "slots": [{"symbol": "x", "slot_idx": 0, "si_dim": [0] * 7, "role": "DERIVED_OBSERVABLE"}],
            "equations": {"x": {"expression": "1.0", "kind": "ALGEBRAIC", "si_dim": [0] * 7}},
            "environmental_assumptions": [],
            "obligations": [
                {"id": "h1", "expression": "x >= 0.5", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "x >= 0.5", "prompt_span": [7, 15]}
            ],
            "quantifier_mode": "EXISTS",
            "mechanism_parameters": [],
            "control_values": {},
        },
        domain={"x": [0.0, 2.0]},
        archetype="ODE_PDE",
    )
    compiled = formal_contracts(bp, graph)
    assert compiled == ("x >= 0.5",)


def test_derive_ihcg_evidence_infers_trajectory_state_symbol_from_expression():
    blueprint = {
        "variable_map": {"I": 0, "I_peak": 1},
        "typed_slots": [
            {"symbol": "I", "slot_idx": 0, "si_dim": [0] * 7, "role": "DYNAMIC_STATE"},
            {"symbol": "I_peak", "slot_idx": 1, "si_dim": [0] * 7, "role": "DERIVED_OBSERVABLE", "semantic_tag": 0},
        ],
    }
    raw = {
        "archetype": "ODE_PDE",
        "domain": {"I": [0.0, 1.0], "I_peak": [0.0, 1.0]},
        "evidence": {
            "equations": {
                "dI": {"expression": "-0.1 * I", "kind": "ODE_RHS", "state_symbol": "I", "si_dim": [0, 0, -1, 0, 0, 0, 0]},
                "I_peak": {"expression": "I", "kind": "TRAJECTORY_MAX", "si_dim": [0] * 7},
            },
            "environmental_assumptions": ["I >= 0"],
            "obligations": [
                {"id": "h1", "expression": "I_peak <= 0.5", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "I_peak <= 0.5"}
            ],
            "quantifier_mode": "EXISTS",
            "mechanism_parameters": [],
            "control_values": {},
        },
    }
    derived = derive_ihcg_evidence("Check I_peak <= 0.5", blueprint, raw)
    assert derived["evidence"]["equations"]["I_peak"]["state_symbol"] == "I"
    slot = next(s for s in derived["evidence"]["slots"] if s["symbol"] == "I_peak")
    assert slot["semantic_tag"] == m.TRAJECTORY_MAXIMUM


def test_z3_degree_handles_abs_min_max_in_qf_lra_and_qf_nra():
    res_lra = m.solve_query("abs(x + y - 1) <= 0.01 and x >= 0.4", {"x": (0.0, 1.0), "y": (0.0, 1.0)})
    assert res_lra.status == "SAT" and res_lra.z3_tier_used == "QF_LRA"
    res_nra = m.solve_query("abs(x + y - 1) == e and b == r * g and e <= 0.01", {"x": (0.0, 1.0), "y": (0.0, 1.0), "e": (0.0, 1.0), "b": (0.1, 0.8), "r": (1.5, 4.0), "g": (0.05, 0.2)})
    assert res_nra.status == "SAT" and res_nra.z3_tier_used == "QF_NRA"


def test_equation_lowerer_converts_prose_lark_error_to_value_error():
    from invar_agent.builder.ingestion.equation_lowerer import parse_expression
    with pytest.raises(ValueError):
        parse_expression("1.0 governed by dS/dt")


def test_multi_regime_contracts_verify_phi_lock_soundly():
    from invar_agent.research.epistemic import compile_blueprint_ihcg, verify_blueprint_entailment_lock
    from invar_agent.schemas.research import ResearchBlueprint
    prompt = "Claim: I_max <= 0.15 for R0 in [1.5, 4.0]."
    quote = "I_max <= 0.15"
    start = prompt.index(quote)
    bp = ResearchBlueprint(
        blueprint_id="multi_regime_lock",
        title="Multi-regime Phi_lock test",
        research_prompt=prompt,
        falsifiable_hypothesis=quote,
        search_queries=[],
        code_template="def simulation_step(buf, dt):\n    buf.data[1] = 0.05 * buf.data[0]\n",
        variable_map={"R0": 0, "I_max": 1},
        slot_roles={"R0": "read_only_parameter", "I_max": "diagnostic"},
        ihcg_evidence={
            "archetype": "PARAM_SWEEP",
            "domain": {"R0": [1.5, 4.0], "I_max": [0.0, 1.0]},
            "evidence": {
                "slots": [
                    {"symbol": "R0", "slot_idx": 0, "si_dim": [0] * 7, "role": "CONTROL_PARAM"},
                    {"symbol": "I_max", "slot_idx": 1, "si_dim": [0] * 7, "role": "DERIVED_OBSERVABLE"},
                ],
                "equations": {"I_max": {"expression": "0.05 * R0", "si_dim": [0] * 7}},
                "environmental_assumptions": ["R0 >= 1.5"],
                "obligations": [
                    {"id": "h1", "expression": "I_max <= 0.15", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": quote, "prompt_span": [start, start + len(quote)]}
                ],
                "quantifier_mode": "FORALL",
                "mechanism_parameters": ["R0"],
            },
        },
        contracts=[
            {
                "contract_id": "c1_low",
                "target_backend": "cpp",
                "initial_state": [1.5, 0.0],
                "dt": 1.0,
                "steps": 1,
                "invariants": [
                    {"variable_name": "R0", "assertion": "abs(state[0] - 1.5) <= 1e-9"},
                    {"variable_name": "I_max", "assertion": "state[1] <= 0.15"},
                ],
            },
            {
                "contract_id": "c2_high",
                "target_backend": "cpp",
                "initial_state": [4.0, 0.0],
                "dt": 1.0,
                "steps": 1,
                "invariants": [
                    {"variable_name": "R0", "assertion": "abs(state[0] - 4.0) <= 1e-9"},
                    {"variable_name": "I_max", "assertion": "state[1] <= 0.15"},
                ],
            },
        ],
    )
    graph = compile_blueprint_ihcg(bp)
    assert verify_blueprint_entailment_lock(bp, graph).entailment_locked
    # Weakening both contracts so neither entails I_max <= 0.15 MUST still fail Phi_lock.
    weak = bp.model_copy(deep=True)
    weak.contracts[0].invariants[1] = weak.contracts[0].invariants[1].model_copy(update={"assertion": "state[1] <= 0.25"})
    weak.contracts[1].invariants[1] = weak.contracts[1].invariants[1].model_copy(update={"assertion": "state[1] <= 0.25"})
    assert not verify_blueprint_entailment_lock(weak, graph).entailment_locked


@pytest.mark.asyncio
async def test_mc_ensemble_finite_sample_certifies_empirical_bernstein(tmp_path):
    from invar_agent.research.epistemic import parse_run_certificate
    from invar_agent.research.state_machine import ResearchStateMachine
    from invar_agent.schemas.research import ResearchBlueprint

    prompt = "Finite-sample Monte Carlo test. Claim: mean_diff > 0.1"
    quote = "mean_diff > 0.1"
    start = prompt.index(quote)
    code = (
        "def simulation_step(buf, dt):\n"
        "    effect = buf.data[0]\n"
        "    seed = int(buf.data[1])\n"
        "    total = 0.0\n"
        "    for i in range(40):\n"
        "        u = invar_rand_u01(seed, i, 1)\n"
        "        total += effect * effect + 0.05 * (u - 0.5)\n"
        "    buf.data[2] = total / 40.0\n"
    )
    bp = ResearchBlueprint(
        blueprint_id="mc_ensemble_reg",
        title="MC Ensemble Regression",
        research_prompt=prompt,
        falsifiable_hypothesis=quote,
        search_queries=[],
        code_template=code,
        variable_map={"effect": 0, "seed": 1, "mean_diff": 2},
        slot_roles={"effect": "read_only_parameter", "seed": "read_only_parameter", "mean_diff": "diagnostic"},
        ihcg_evidence={
            "archetype": "MONTE_CARLO",
            "domain": {"effect": [0.8, 0.8], "seed": [42.0, 42.0], "mean_diff": [-2.0, 2.0]},
            "evidence": {
                "slots": [
                    {"symbol": "effect", "slot_idx": 0, "si_dim": [0] * 7, "role": "CONTROL_PARAM"},
                    {"symbol": "seed", "slot_idx": 1, "si_dim": [0] * 7, "role": "CONTROL_PARAM", "stochastic_role": "SEED"},
                    {"symbol": "mean_diff", "slot_idx": 2, "si_dim": [0] * 7, "role": "DERIVED_OBSERVABLE"},
                ],
                "equations": {"mean_diff": {"expression": "effect * effect", "si_dim": [0] * 7}},
                "environmental_assumptions": [],
                "obligations": [
                    {"id": "h1", "expression": quote, "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": quote, "prompt_span": [start, start + len(quote)]}
                ],
                "quantifier_mode": "EXISTS",
                "mechanism_parameters": ["effect"],
                "mc_difference_bounds": [-1.0, 1.0],
            },
        },
        contracts=[
            {
                "contract_id": "c1",
                "target_backend": "cpp",
                "initial_state": [0.8, 42.0, 0.0],
                "dt": 1.0,
                "steps": 1,
                "invariants": [{"variable_name": "mean_diff", "assertion": "state[2] > 0.1"}],
            }
        ],
        max_feedback_rounds=1,
    )
    state = await ResearchStateMachine(tmp_path, workspace_root=tmp_path).run(bp)
    cert = parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == "VERIFIED_DISCOVERY", cert.model_dump_json(indent=2)
    assert cert.fidelity_audit.statistical_audit is not None and cert.fidelity_audit.statistical_audit.passed
    assert cert.significance_audit.method == "MC_EMPIRICAL_BERNSTEIN"
    assert cert.significance_audit.certified_lower_margin > 0.0
    assert cert.identifiability_audit.coupling_passed
    assert cert.identifiability_audit.dead_parameters == ()


def test_diagnostic_mechanism_parameter_error_names_offending_symbol():
    prompt = "Check x > 0"
    with pytest.raises(ValueError, match="bad_state"):
        m.IhcgCompiler().compile_from_prompt(
            prompt,
            {
                "slots": [
                    {"symbol": "p", "slot_idx": 0, "si_dim": [0] * 7, "role": "CONTROL_PARAM"},
                    {"symbol": "bad_state", "slot_idx": 1, "si_dim": [0] * 7, "role": "DYNAMIC_STATE"},
                ],
                "equations": {"bad_state": {"expression": "p", "si_dim": [0] * 7}},
                "environmental_assumptions": [],
                "obligations": [
                    {"id": "h1", "expression": "bad_state > 0", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "x > 0", "prompt_span": [6, 11]}
                ],
                "quantifier_mode": "EXISTS",
                "mechanism_parameters": ["bad_state"],
            },
            {"p": [0.0, 1.0], "bad_state": [-1.0, 2.0]},
            "ODE_PDE",
        )


def test_variadic_min_max_lossless_folding():
    from invar_agent.research.ihcg_normalization import _normalize_formal_expr
    assert _normalize_formal_expr("min(a, b, c, d)") == "min(a, min(b, min(c, d)))"
    assert _normalize_formal_expr("max(x, y, z)") == "max(x, max(y, z))"


def test_instrument_purity_domain_guard_does_not_false_positive_on_valid_log(tmp_path):
    from invar_agent.healing.deterministic_ast import instrument_purity, purity_audit_from_diag
    from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
    code = (
        "import math\n"
        "def simulation_step(buf, dt):\n"
        "    buf.data[1] = math.log(max(buf.data[0], 1e-300))\n"
    )
    inst = instrument_purity(code)
    assert instrument_purity(inst) == inst
    bridge = DynamicCABIBridge(tmp_path)
    # Valid positive input: max(2.0, 1e-300) does not clamp -> 0 projected steps.
    ok = bridge.run_python(inst, [2.0, 0.0], 0.1, 3)
    audit_ok = purity_audit_from_diag(ok["diag"], domain_tag="test/log_ok", static_purity_passed=True)
    assert audit_ok.projected_steps == 0 and audit_ok.stabilized_steps == 0
    # Negative input: max(-1.0, 1e-300) DOES clamp -> projected_steps == 3!
    clamped = bridge.run_python(inst, [-1.0, 0.0], 0.1, 3)
    audit_clamped = purity_audit_from_diag(clamped["diag"], domain_tag="test/log_clamp", static_purity_passed=True)
    assert audit_clamped.projected_steps == 3


def test_roundoff_supports_abs_min_max():
    from invar_agent.research.stochastic_equilibrium import _Roundoff
    code = (
        "def simulation_step(buf, dt):\n"
        "    buf.data[2] = abs(buf.data[0] - buf.data[1])\n"
        "    buf.data[3] = min(buf.data[0], buf.data[1])\n"
    )
    r = _Roundoff(code, {"a": 0, "b": 1, "c": 2, "d": 3}, {"a": (0.2, 0.8), "b": (0.1, 0.9), "c": (0.0, 1.0), "d": (0.0, 1.0)}, 0.1)
    assert r.values[2][0] == (0.0, pytest.approx(0.7))
    assert r.values[3][0] == (0.1, 0.8)


def test_roundoff_sound_interval_hull_if_branch_pruning():
    from invar_agent.research.stochastic_equilibrium import _Roundoff
    code = (
        "import math\n"
        "def simulation_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    y = 0.0\n"
        "    if x > 0.0:\n"
        "        y = -x * math.log(x)\n"
        "    buf.data[1] = y\n"
    )
    r = _Roundoff(code, {"x": 0, "y": 1}, {"x": (0.1, 1.0), "y": (0.0, 1.0)}, 0.1)
    lo, hi = r.values[1][0]
    assert lo <= 0.0 and hi > 0.0 and r.values[1][1] >= 0.0


def test_buckingham_pi_interval_proven_nonzero_denominator():
    import json
    from invar_agent import _native
    # m in [1000, 1000] -> m - 1 = 999 > 0 (proven nonzero); x in [0, 2] -> x - 1 may be 0 (rejected).
    ok_eq = [{
        "expr": "(/ 1 (- m 1))",
        "symbols": {
            "m": {"role": "CONTROL_PARAM", "si_dim": [0]*7, "semantic_tag": 0, "unit_scale": [1, 1], "positive": True, "domain": [1000.0, 1000.0]},
        },
    }]
    res = json.loads(_native.py_compute_alpha_canonical_system(json.dumps(ok_eq), 64))
    assert res["canonical_system"]
    bad_eq = [{
        "expr": "(/ 1 (- x 1))",
        "symbols": {
            "x": {"role": "CONTROL_PARAM", "si_dim": [0]*7, "semantic_tag": 0, "unit_scale": [1, 1], "positive": False, "domain": [0.0, 2.0]},
        },
    }]
    with pytest.raises(ValueError, match="nonzero denominator"):
        _native.py_compute_alpha_canonical_system(json.dumps(bad_eq), 64)


def test_roache_gci_nanoscale_rounding_floor():
    # Nanoscale physical observables (e.g. 1e-15 W) must not be swamped by a 2.22e-14 floor.
    audit = m.roache_gci(1e-15, 1e-15, 1e-15, nominal_order=4.0)
    assert audit.passed
    assert audit.gci_fine_abs < 1e-25


def test_eliminate_explicit_model_eqs_reduces_nonlinear_feasibility():
    omega = {"R0": (2.5, 4.0), "e": (0.85, 0.85), "pv": (0.0, 1.0), "R_eff": (0.0, 5.0), "p_star": (0.0, 1.0)}
    model_eqs = ["(p_star) == ((1.0 - 1.0 / R0) / e)", "(R_eff) == (R0 * (1.0 - e * pv))"]
    expr, red_omega = m._eliminate_explicit_model_eqs(omega, model_eqs, "R0 >= 2.5", "R_eff <= 1.0")
    assert "R_eff" not in red_omega and "p_star" not in red_omega
    q = m.solve_query(expr, red_omega, witness=True)
    assert q.status == "SAT"


def test_cpp_emitter_64bit_int64_splitmix64_parity(tmp_path):
    import types
    from invar_agent.research.state_machine import _lower_kernel
    from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
    code = (
        "MASK = 0xFFFFFFFFFFFFFFFF\n"
        "GOLD = 0x9E3779B97F4A7C15\n"
        "def sm64(s):\n"
        "    s = (s + GOLD) & MASK\n"
        "    z = s\n"
        "    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK\n"
        "    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK\n"
        "    return (z ^ (z >> 31)) & MASK\n"
        "def simulation_step(buf, dt):\n"
        "    s = int(buf.data[0])\n"
        "    s = sm64(s)\n"
        "    buf.data[1] = (float(s >> 11) + 0.5) / 9007199254740992.0\n"
    )
    ns = {}
    exec(code, ns)
    b = types.SimpleNamespace(data=[123456789.0, 0.0])
    ns["simulation_step"](b, 1.0)
    _, cpp = _lower_kernel(code)
    assert "static int64_t sm64(int64_t s)" in cpp
    res = DynamicCABIBridge(tmp_path).run(cpp, [123456789.0, 0.0], 1.0, 1, timeout_seconds=30)
    assert res["state"][1] == pytest.approx(b.data[1], rel=0.0, abs=1e-15)


def test_domain_guard_does_not_wrap_positive_constants():
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics
    code = (
        "import math\n"
        "def simulation_step(buf, dt):\n"
        "    buf.data[1] = buf.data[0] * math.log(2.0) + math.sqrt(3.0)\n"
    )
    out = sanitize_intrinsics(code).code
    assert "math.log(2.0)" in out and "math.sqrt(3.0)" in out
    assert "max(2.0" not in out and "max(3.0" not in out


def test_builder_reverify_ir_preserves_constant_si_dimensions():
    from invar_agent.builder.ingestion.pipeline import reverify_ir
    from invar_agent.builder.ingestion.schemas import (
        CanonicalResearchIR, ExtractedConstant, ExtractedEquation, EquationRole,
        SIDimensionVectorModel, SourceSpan, StateVariableSpec,
    )
    span = SourceSpan(file="test.json", location="test", quote="E = k * T")
    ir = CanonicalResearchIR(
        ir_id="dim_test", source_mode="vault", vault_run_ids=["r1"], title="Dim Test",
        state_variables=[StateVariableSpec(symbol="E", slot_index=0, initial_value=1.0, dimension=SIDimensionVectorModel(exponents=(2, 1, -2, 0, 0, 0, 0)), provenance=span)],
        constants=[
            ExtractedConstant(constant_id="c0", name="Boltzmann constant", symbol="k", raw_value="1.38e-23", raw_unit="1", si_value=1.38e-23, dimension=SIDimensionVectorModel(exponents=(2, 1, -2, 0, -1, 0, 0)), provenance=span),
            ExtractedConstant(constant_id="c1", name="Temperature", symbol="T", raw_value="300.0", raw_unit="1", si_value=300.0, dimension=SIDimensionVectorModel(exponents=(0, 0, 0, 0, 1, 0, 0)), provenance=span),
        ],
        equations=[ExtractedEquation(equation_id="equation:0", raw_latex="E = k * T", canonical_latex="E = k * T", role=EquationRole.ALGEBRAIC_CLOSURE, provenance=span)],
    )
    reverify_ir(ir)
    assert tuple(ir.constants[0].dimension.exponents) == (2, 1, -2, 0, -1, 0, 0)
    # Second reverify must STILL pass with preserved SI exponents!
    reverify_ir(ir)
    assert ir.equations[0].hin_homogeneous is True


def test_roundoff_floor_and_ceil_sound_interval():
    import math
    from invar_agent.research.stochastic_equilibrium import _Roundoff
    code = (
        "import math\n"
        "def simulation_step(buf, dt):\n"
        "    buf.data[1] = math.floor(buf.data[0] + 0.5) + math.ceil(buf.data[0])\n"
    )
    rp = _Roundoff(code, {"x": 0, "y": 1}, {"x": (1.2, 2.3), "y": (0.0, 10.0)}, 0.1)
    (lo, hi), err = rp.values[1]
    assert lo == pytest.approx(3.0) and hi == pytest.approx(5.0)
    assert err >= 0.0 and math.isfinite(err)


def test_csel_large_polynomial_canonicalization_ac_ring():
    import json
    from invar_agent import _native
    # Build two algebraically identical 20+ term polynomials in different parenthesizations/orders.
    poly_a = "((a + b + c + d) * (a + 2*b - c + 3*d) * (b + c + d))"
    poly_b = "((d + c + b) * (3*d - c + 2*b + a) * (d + c + b + a))"
    tag = {"role": "DYNAMIC_STATE", "si_dim": [0, 0, 0, 0, 0, 0, 0], "semantic_tag": 0, "unit_scale": [1, 1], "positive": True}
    syms = {k: tag for k in ("a", "b", "c", "d")}
    eq_a = [{"expr": m._json(m._csel(m._parse(poly_a))), "symbols": syms}]
    eq_b = [{"expr": m._json(m._csel(m._parse(poly_b))), "symbols": syms}]
    res_a = json.loads(_native.py_compute_alpha_canonical_system(m._json(eq_a), max_aut_branches=64))
    res_b = json.loads(_native.py_compute_alpha_canonical_system(m._json(eq_b), max_aut_branches=64))
    assert res_a["saturated"] is True and res_b["saturated"] is True
    assert res_a["canonical_system"] == res_b["canonical_system"]


def test_fixed_point_ode_kernel_krawczyk_hurwitz_and_ift_direct_partials(tmp_path):
    from invar_agent.research import epistemic as ep
    from invar_agent.research.state_machine import ResearchStateMachine, ResearchRunState
    from invar_agent.research.stochastic_equilibrium import derive_evidence
    from invar_agent.research.default_verification import UnsupportedVerification
    from invar_agent.research.vault import ResearchVaultManager
    # 2D damped linear oscillator + algebraic closure x3 = 1 - x1 - x2 with direct-parameter observable.
    code = (
        "def simulation_step(buf, dt):\n"
        "    t = buf.data[0]\n"
        "    x1 = buf.data[1]\n"
        "    x2 = buf.data[2]\n"
        "    alpha = buf.data[4]\n"
        "    mu = buf.data[5]\n"
        "    f1 = (0.5 - x1) * (alpha + mu) + (x2 - 0.25)\n"
        "    f2 = (0.25 - x2) * (alpha + 2.0 * mu) - (x1 - 0.5)\n"
        "    nx1 = x1 + dt * f1\n"
        "    nx2 = x2 + dt * f2\n"
        "    nx3 = 1.0 - nx1 - nx2\n"
        "    buf.data[0] = t + dt\n"
        "    buf.data[1] = nx1\n"
        "    buf.data[2] = nx2\n"
        "    buf.data[3] = nx3\n"
        "    buf.data[6] = 0.0 - 2.0 * alpha - 3.0 * mu\n"
        "    buf.data[7] = alpha * (nx1 + nx2)\n"
        "    if abs(f1) > buf.data[8]:\n"
        "        buf.data[8] = abs(f1)\n"
    )
    from invar_agent.schemas.research import ResearchBlueprint
    bp = ResearchBlueprint.model_validate({
        "blueprint_id": "fp_ode_reg",
        "title": "Fixed Point ODE Kernel Regression",
        "falsifiable_hypothesis": "Damped equilibrium has negative trace.",
        "search_queries": [],
        "code_template": code,
        "variable_map": {"time_s": 0, "x1": 1, "x2": 2, "x3": 3, "alpha": 4, "mu": 5, "tr_j": 6, "payoff": 7, "peak_res": 8},
        "contracts": [{
            "contract_id": "c1",
            "target_backend": "cpp",
            "initial_state": [0.0, 0.4, 0.3, 0.3, 0.4, 0.2, 0.0, 0.0, 0.0],
            "dt": 0.05,
            "steps": 20,
            "invariants": [{"variable_name": "tr_j", "assertion": "state[6] <= -0.1", "tolerance": 1e-9}],
        }],
    })
    from invar_agent import _native
    raw_ihcg = m.IhcgGraph(
        archetype="FIXED_POINT",
        quantifier_mode="EXISTS",
        slots=(
            m.SlotSpec(symbol="time_s", slot_idx=0, role="DYNAMIC_STATE", si_dim=(0,)*7),
            m.SlotSpec(symbol="x1", slot_idx=1, role="DYNAMIC_STATE", si_dim=(0,)*7),
            m.SlotSpec(symbol="x2", slot_idx=2, role="DYNAMIC_STATE", si_dim=(0,)*7),
            m.SlotSpec(symbol="x3", slot_idx=3, role="DYNAMIC_STATE", si_dim=(0,)*7),
            m.SlotSpec(symbol="alpha", slot_idx=4, role="CONTROL_PARAM", si_dim=(0,)*7),
            m.SlotSpec(symbol="mu", slot_idx=5, role="CONTROL_PARAM", si_dim=(0,)*7),
            m.SlotSpec(symbol="tr_j", slot_idx=6, role="DERIVED_OBSERVABLE", si_dim=(0,)*7),
            m.SlotSpec(symbol="payoff", slot_idx=7, role="DERIVED_OBSERVABLE", si_dim=(0,)*7),
        ),
        omega={
            "time_s": (0.0, 100.0), "x1": (0.01, 0.99), "x2": (0.01, 0.99), "x3": (0.01, 0.99),
            "alpha": (0.4, 0.4), "mu": (0.2, 0.2), "tr_j": (-10.0, 10.0), "payoff": (-5.0, 5.0),
        },
        control_values={"alpha": 0.4, "mu": 0.2},
        mechanism_parameters=("alpha", "mu"),
        equations={
            "time_s": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="time_s", expression="time_s", si_dim=(0,)*7),
            "x1": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x1", expression="(0.5 - x1) * (alpha + mu) + (x2 - 0.25)", si_dim=(0,)*7),
            "x2": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x2", expression="(0.25 - x2) * (alpha + 2.0 * mu) - (x1 - 0.5)", si_dim=(0,)*7),
            "x3": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x3", expression="x3 - (1.0 - x1 - x2)", si_dim=(0,)*7),
            "tr_j": m.EquationSpec(kind="ALGEBRAIC", expression="0.0 - 2.0 * alpha - 3.0 * mu", si_dim=(0,)*7),
            "payoff": m.EquationSpec(kind="ALGEBRAIC", expression="alpha * (x1 + x2)", si_dim=(0,)*7),
        },
        obligations=(m.Obligation(id="o1", obligation_class="HYPOTHESIS_TARGET", expression="tr_j <= -0.1", prompt_quote="Damped equilibrium has negative trace.", prompt_span=(0, 39)),),
        environmental_assumptions=(),
        h_ihcg="",
    )
    ihcg = raw_ihcg.model_copy(update={"h_ihcg": _native.py_compute_ihcg_digest(raw_ihcg.model_dump_json())})
    sm = ResearchStateMachine(str(tmp_path))
    st = ResearchRunState(blueprint=bp, ihcg=ihcg, research_stage=1)
    ev = derive_evidence(sm, st, code, phase="execution")
    assert ev.fidelity.passed is True
    assert ev.identifiability.coupling_passed is True
    assert ev.identifiability.dead_parameters == ()
    cert, _, route = ep.evaluate_execution(
        bp, ihcg, ev, executable_digest=ev.executable_digest,
        vault=ResearchVaultManager(tmp_path / "vault"), attempt=1, precision_refined=False, preserved={},
    )
    assert cert.epistemic_verdict == m.EpistemicVerdict.VERIFIED_DISCOVERY
    assert cert.significance_audit.method == "FIXED_POINT_KRAWCZYK_HURWITZ"
    assert cert.significance_audit.passed is True
    # Zero-bypass check: mutating the executable RHS must still fail closed with ORACLE_EXECUTABLE_MISMATCH.
    tampered_code = code.replace("f1 = (0.5 - x1) * (alpha + mu) + (x2 - 0.25)", "f1 = (0.1 - x1) * (alpha + mu) + (x2 - 0.25)")
    with pytest.raises(UnsupportedVerification, match="ORACLE_EXECUTABLE_MISMATCH"):
        derive_evidence(sm, st, tampered_code, phase="execution")


def test_iv_even_power_and_sqrt_subnormal_nonnegative():
    box = {"x": (0.3332, 0.3334), "y": (0.3332, 0.3334), "z": (0.3332, 0.3334)}
    iv = m._iv(m._parse("sqrt((x - 1/3)**2 + (y - 1/3)**2 + (z - 1/3)**2)"), box)
    assert iv.lo >= 0.0
    assert iv.hi > 0.0 and iv.hi < 1e-3
    pt_iv = m._iv(m._parse("sqrt((x - 1/3)**2 + (y - 1/3)**2 + (z - 1/3)**2)"), {"x": (1/3, 1/3), "y": (1/3, 1/3), "z": (1/3, 1/3)})
    assert pt_iv.lo >= 0.0


def test_zero_dt_profile_avoids_rk4_cubic_blowup():
    import time
    from invar_agent.research.default_verification import _SymbolicKernel
    code = (
        "def p1(x1, x2, x3, a): return (1.0 + a) * x2 - x3\n"
        "def p2(x1, x2, x3, a): return -x1 + (1.0 + a) * x3\n"
        "def p3(x1, x2, x3, a): return (1.0 + a) * x1 - x2\n"
        "def g(buf, x1, x2, x3, idx):\n"
        "    a, mu = buf.data[3], buf.data[4]\n"
        "    f1, f2, f3 = p1(x1, x2, x3, a), p2(x1, x2, x3, a), p3(x1, x2, x3, a)\n"
        "    phi = x1 * f1 + x2 * f2 + x3 * f3\n"
        "    if idx == 0: return x1 * (f1 - phi) + mu * (1.0/3.0 - x1)\n"
        "    if idx == 1: return x2 * (f2 - phi) + mu * (1.0/3.0 - x2)\n"
        "    return x3 * (f3 - phi) + mu * (1.0/3.0 - x3)\n"
        "def simulation_step(buf, dt):\n"
        "    x1, x2, x3 = buf.data[0], buf.data[1], buf.data[2]\n"
        "    k11, k12 = g(buf, x1, x2, x3, 0), g(buf, x1, x2, x3, 1)\n"
        "    k21, k22 = g(buf, x1+0.5*dt*k11, x2+0.5*dt*k12, 1.0-(x1+0.5*dt*k11)-(x2+0.5*dt*k12), 0), g(buf, x1+0.5*dt*k11, x2+0.5*dt*k12, 1.0-(x1+0.5*dt*k11)-(x2+0.5*dt*k12), 1)\n"
        "    k31, k32 = g(buf, x1+0.5*dt*k21, x2+0.5*dt*k22, 1.0-(x1+0.5*dt*k21)-(x2+0.5*dt*k22), 0), g(buf, x1+0.5*dt*k21, x2+0.5*dt*k22, 1.0-(x1+0.5*dt*k21)-(x2+0.5*dt*k22), 1)\n"
        "    k41, k42 = g(buf, x1+dt*k31, x2+dt*k32, 1.0-(x1+dt*k31)-(x2+dt*k32), 0), g(buf, x1+dt*k31, x2+dt*k32, 1.0-(x1+dt*k31)-(x2+dt*k32), 1)\n"
        "    x1n = x1 + dt*(k11 + 2.0*k21 + 2.0*k31 + k41)/6.0\n"
        "    x2n = x2 + dt*(k12 + 2.0*k22 + 2.0*k32 + k42)/6.0\n"
        "    x3n = 1.0 - x1n - x2n\n"
        "    s = x1n + x2n + x3n\n"
        "    buf.data[0], buf.data[1], buf.data[2] = x1n / s, x2n / s, x3n / s\n"
    )
    slots = {"x1": 0, "x2": 1, "x3": 2, "a": 3, "mu": 4}
    t0 = time.monotonic()
    rhs, vals, written = _SymbolicKernel(code, slots).zero_dt_profile(set(slots))
    assert time.monotonic() - t0 < 2.0
    assert {"x1", "x2", "x3"} <= set(rhs)
    assert "a" not in written and "mu" not in written


def test_fixed_point_manifold_conservation_rk4_krawczyk_hurwitz_and_z3_feasibility(tmp_path):
    from invar_agent import _native
    from invar_agent.research import epistemic as ep
    from invar_agent.research.state_machine import ResearchStateMachine, ResearchRunState
    from invar_agent.research.stochastic_equilibrium import derive_evidence
    from invar_agent.research.vault import ResearchVaultManager
    from invar_agent.schemas.research import ResearchBlueprint
    code = (
        "import math\n"
        "def p1(x1, x2, x3, a): return (1.0 + a) * x2 - x3\n"
        "def p2(x1, x2, x3, a): return -x1 + (1.0 + a) * x3\n"
        "def p3(x1, x2, x3, a): return (1.0 + a) * x1 - x2\n"
        "def g1(x1, x2, x3, a, mu):\n"
        "    f1, f2, f3 = p1(x1, x2, x3, a), p2(x1, x2, x3, a), p3(x1, x2, x3, a)\n"
        "    return x1 * (f1 - (x1 * f1 + x2 * f2 + x3 * f3)) + mu * (1.0 / 3.0 - x1)\n"
        "def g2(x1, x2, x3, a, mu):\n"
        "    f1, f2, f3 = p1(x1, x2, x3, a), p2(x1, x2, x3, a), p3(x1, x2, x3, a)\n"
        "    return x2 * (f2 - (x1 * f1 + x2 * f2 + x3 * f3)) + mu * (1.0 / 3.0 - x2)\n"
        "def simulation_step(buf, dt):\n"
        "    x1, x2, x3, alpha, mu = buf.data[0], buf.data[1], buf.data[2], buf.data[3], buf.data[4]\n"
        "    k1a, k1b = g1(x1, x2, x3, alpha, mu), g2(x1, x2, x3, alpha, mu)\n"
        "    x1a, x2a = x1 + 0.5 * dt * k1a, x2 + 0.5 * dt * k1b\n"
        "    x3a = 1.0 - x1a - x2a\n"
        "    k2a, k2b = g1(x1a, x2a, x3a, alpha, mu), g2(x1a, x2a, x3a, alpha, mu)\n"
        "    x1b, x2b = x1 + 0.5 * dt * k2a, x2 + 0.5 * dt * k2b\n"
        "    x3b = 1.0 - x1b - x2b\n"
        "    k3a, k3b = g1(x1b, x2b, x3b, alpha, mu), g2(x1b, x2b, x3b, alpha, mu)\n"
        "    x1c, x2c = x1 + dt * k3a, x2 + dt * k3b\n"
        "    x3c = 1.0 - x1c - x2c\n"
        "    k4a, k4b = g1(x1c, x2c, x3c, alpha, mu), g2(x1c, x2c, x3c, alpha, mu)\n"
        "    x1n = x1 + dt * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0\n"
        "    x2n = x2 + dt * (k1b + 2.0 * k2b + 2.0 * k3b + k4b) / 6.0\n"
        "    x3n = 1.0 - x1n - x2n\n"
        "    s = x1n + x2n + x3n\n"
        "    x1, x2, x3 = x1n / s, x2n / s, x3n / s\n"
        "    buf.data[0], buf.data[1], buf.data[2] = x1, x2, x3\n"
        "    buf.data[5] = -(x1 * math.log(x1) + x2 * math.log(x2) + x3 * math.log(x3))\n"
        "    buf.data[6] = alpha / 3.0 + 2.0 * mu\n"
        "    buf.data[7] = (alpha * alpha + 3.0 * alpha + 3.0) / 9.0 + mu * mu + alpha * mu / 3.0\n"
    )
    bp = ResearchBlueprint.model_validate({
        "blueprint_id": "fp_rps_manifold_reg",
        "title": "RPS Manifold Fixed Point Regression",
        "falsifiable_hypothesis": "Interior Nash equilibrium is stable with positive entropy.",
        "search_queries": [],
        "code_template": code,
        "variable_map": {"x1": 0, "x2": 1, "x3": 2, "alpha": 3, "mu": 4, "H": 5, "trA": 6, "detB": 7},
        "contracts": [{
            "contract_id": "c1",
            "target_backend": "cpp",
            "initial_state": [0.34, 0.33, 0.33, 0.25, 0.12, 0.0, 0.0, 0.0],
            "dt": 0.05,
            "steps": 400,
            "invariants": [
                {"variable_name": "H", "assertion": "state[5] >= 1.0", "tolerance": 1e-9},
                {"variable_name": "trA", "assertion": "state[6] >= 0.1", "tolerance": 1e-9},
                {"variable_name": "detB", "assertion": "state[7] >= 0.2", "tolerance": 1e-9},
            ],
        }],
    })
    phi_expr = "x1 * ((1.0 + alpha) * x2 - x3) + x2 * (-x1 + (1.0 + alpha) * x3) + x3 * ((1.0 + alpha) * x1 - x2)"
    raw_ihcg = m.IhcgGraph(
        archetype="FIXED_POINT",
        quantifier_mode="EXISTS",
        slots=(
            m.SlotSpec(symbol="x1", slot_idx=0, role="DYNAMIC_STATE", si_dim=(0,)*7),
            m.SlotSpec(symbol="x2", slot_idx=1, role="DYNAMIC_STATE", si_dim=(0,)*7),
            m.SlotSpec(symbol="x3", slot_idx=2, role="DYNAMIC_STATE", si_dim=(0,)*7),
            m.SlotSpec(symbol="alpha", slot_idx=3, role="CONTROL_PARAM", si_dim=(0,)*7),
            m.SlotSpec(symbol="mu", slot_idx=4, role="CONTROL_PARAM", si_dim=(0,)*7),
            m.SlotSpec(symbol="H", slot_idx=5, role="DERIVED_OBSERVABLE", si_dim=(0,)*7),
            m.SlotSpec(symbol="trA", slot_idx=6, role="DERIVED_OBSERVABLE", si_dim=(0,)*7),
            m.SlotSpec(symbol="detB", slot_idx=7, role="DERIVED_OBSERVABLE", si_dim=(0,)*7),
        ),
        omega={
            "x1": (1e-6, 0.999999), "x2": (1e-6, 0.999999), "x3": (1e-6, 0.999999),
            "alpha": (0.25, 0.25), "mu": (0.12, 0.12),
            "H": (0.0, 2.0), "trA": (0.0, 5.0), "detB": (0.0, 5.0),
        },
        control_values={"alpha": 0.25, "mu": 0.12},
        mechanism_parameters=("alpha", "mu"),
        equations={
            "x1": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x1", expression=f"x1 * (((1.0 + alpha) * x2 - x3) - ({phi_expr})) + mu * (1.0 / 3.0 - x1)", si_dim=(0,)*7),
            "x2": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x2", expression=f"x2 * ((-x1 + (1.0 + alpha) * x3) - ({phi_expr})) + mu * (1.0 / 3.0 - x2)", si_dim=(0,)*7),
            "x3": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x3", expression=f"x3 * (((1.0 + alpha) * x1 - x2) - ({phi_expr})) + mu * (1.0 / 3.0 - x3)", si_dim=(0,)*7),
            "H": m.EquationSpec(kind="ALGEBRAIC", expression="-(x1 * log(x1) + x2 * log(x2) + x3 * log(x3))", si_dim=(0,)*7),
            "trA": m.EquationSpec(kind="ALGEBRAIC", expression="alpha / 3.0 + 2.0 * mu", si_dim=(0,)*7),
            "detB": m.EquationSpec(kind="ALGEBRAIC", expression="(alpha**2 + 3.0 * alpha + 3.0) / 9.0 + mu**2 + alpha * mu / 3.0", si_dim=(0,)*7),
        },
        obligations=(
            m.Obligation(id="o1", obligation_class="HYPOTHESIS_TARGET", expression="H >= 1.0 and trA >= 0.1 and detB >= 0.2", prompt_quote="Interior Nash equilibrium is stable with positive entropy.", prompt_span=(0, 56)),
        ),
        environmental_assumptions=("x1 + x2 + x3 == 1.0",),
        h_ihcg="",
    )
    ihcg = raw_ihcg.model_copy(update={"h_ihcg": _native.py_compute_ihcg_digest(raw_ihcg.model_dump_json())})
    sm = ResearchStateMachine(str(tmp_path))
    st = ResearchRunState(blueprint=bp, ihcg=ihcg, research_stage=1)
    ev = derive_evidence(sm, st, code, phase="execution")
    assert ev.fidelity.passed is True
    assert ev.identifiability.coupling_passed is True
    assert ev.identifiability.dead_parameters == ()
    assert ev.identifiability.equilibrium_restoring_force is not None and ev.identifiability.equilibrium_restoring_force > 0.1
    cert, _, _ = ep.evaluate_execution(
        bp, ihcg, ev, executable_digest=ev.executable_digest,
        vault=ResearchVaultManager(tmp_path / "vault"), attempt=1, precision_refined=False, preserved={},
    )
    assert cert.epistemic_verdict == m.EpistemicVerdict.VERIFIED_DISCOVERY
    assert cert.contract_audit.classification == "CONSTRAINING"
    assert cert.significance_audit.method == "FIXED_POINT_KRAWCZYK_HURWITZ"
    assert cert.significance_audit.passed is True
    assert cert.purity_audit.enforcement_ratio == 0.0


def test_fixed_point_ode_kernel_preserves_canonical_invar_sdiv(tmp_path):
    """Ensure _StripDiagIf never mutates helper functions like _invar_sdiv (preventing GUARDED_DIVISION_IMPLEMENTATION_UNBOUND)."""
    from invar_agent import _native
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics
    from invar_agent.research import epistemic as ep
    from invar_agent.research.state_machine import ResearchStateMachine, ResearchRunState
    from invar_agent.research.stochastic_equilibrium import derive_evidence
    from invar_agent.research.vault import ResearchVaultManager
    from invar_agent.schemas.research import ResearchBlueprint
    raw_code = (
        "def simulation_step(buf, dt):\n"
        "    x1 = buf.data[0]\n"
        "    x2 = buf.data[1]\n"
        "    alpha = buf.data[2]\n"
        "    denom = 1.0 + x1 + x2\n"
        "    f1 = (0.4 + alpha - x1) / denom + (x2 - 0.3)\n"
        "    f2 = (0.3 - x2) - (x1 - (0.4 + alpha))\n"
        "    buf.data[0] = x1 + dt * f1\n"
        "    buf.data[1] = x2 + dt * f2\n"
        "    buf.data[3] = x1 + x2\n"
    )
    san_code = sanitize_intrinsics(raw_code).code
    assert "def _invar_sdiv" in san_code
    bp = ResearchBlueprint.model_validate({
        "blueprint_id": "fp_sdiv_reg",
        "title": "Fixed Point Guarded Division Regression",
        "falsifiable_hypothesis": "Sum x1 + x2 is at least 0.9 at equilibrium.",
        "search_queries": [],
        "code_template": san_code,
        "variable_map": {"x1": 0, "x2": 1, "alpha": 2, "xsum": 3},
        "contracts": [{
            "contract_id": "c1",
            "target_backend": "cpp",
            "initial_state": [0.85, 0.25, 0.5, 1.1],
            "dt": 0.05,
            "steps": 40,
            "invariants": [{"variable_name": "xsum", "assertion": "state[3] >= 0.9", "tolerance": 1e-9}],
        }],
    })
    raw_ihcg = m.IhcgGraph(
        archetype="FIXED_POINT",
        quantifier_mode="EXISTS",
        slots=(
            m.SlotSpec(symbol="x1", slot_idx=0, role="DYNAMIC_STATE", si_dim=(0,)*7),
            m.SlotSpec(symbol="x2", slot_idx=1, role="DYNAMIC_STATE", si_dim=(0,)*7),
            m.SlotSpec(symbol="alpha", slot_idx=2, role="CONTROL_PARAM", si_dim=(0,)*7),
            m.SlotSpec(symbol="xsum", slot_idx=3, role="DERIVED_OBSERVABLE", si_dim=(0,)*7),
        ),
        omega={"x1": (0.01, 0.99), "x2": (0.01, 0.99), "alpha": (0.5, 0.5), "xsum": (0.0, 2.0)},
        control_values={"alpha": 0.5},
        mechanism_parameters=("alpha",),
        equations={
            "x1": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x1", expression="(0.4 + alpha - x1) / (1.0 + x1 + x2) + (x2 - 0.3)", si_dim=(0,)*7),
            "x2": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x2", expression="(0.3 - x2) - (x1 - (0.4 + alpha))", si_dim=(0,)*7),
            "xsum": m.EquationSpec(kind="ALGEBRAIC", expression="x1 + x2", si_dim=(0,)*7),
        },
        obligations=(
            m.Obligation(id="o1", obligation_class="HYPOTHESIS_TARGET", expression="xsum >= 0.9", prompt_quote="Sum x1 + x2 is at least 0.9 at equilibrium.", prompt_span=(0, 44)),
        ),
        environmental_assumptions=(),
        h_ihcg="",
    )
    ihcg = raw_ihcg.model_copy(update={"h_ihcg": _native.py_compute_ihcg_digest(raw_ihcg.model_dump_json())})
    sm = ResearchStateMachine(str(tmp_path))
    st = ResearchRunState(blueprint=bp, ihcg=ihcg, research_stage=1)
    ev = derive_evidence(sm, st, san_code, phase="execution")
    assert ev.fidelity.passed is True
    cert, _, _ = ep.evaluate_execution(
        bp, ihcg, ev, executable_digest=ev.executable_digest,
        vault=ResearchVaultManager(tmp_path / "vault"), attempt=1, precision_refined=False, preserved={},
    )
    assert cert.epistemic_verdict == m.EpistemicVerdict.VERIFIED_DISCOVERY
    assert cert.purity_audit.enforcement_ratio == 0.0


def test_default_verification_expression_normalizes_variadic_max_min():
    from invar_agent.research.default_verification import _expression
    import sympy as sp
    a, b, c = sp.Symbol("a"), sp.Symbol("b"), sp.Symbol("c")
    expr = sp.Max(a, b, c) + sp.Min(a, b) + sp.Abs(a - b)
    norm = _expression(expr)
    assert "Max(" not in norm and "Min(" not in norm and "Abs(" not in norm
    iv = m._iv(m._parse(norm), {"a": (1.0, 1.0), "b": (3.0, 3.0), "c": (5.0, 5.0)})
    assert iv.lo == pytest.approx(8.0) and iv.hi == pytest.approx(8.0)


@pytest.mark.asyncio
async def test_evaluate_epistemic_routes_recoverable_algebraic_mismatch_to_reflex_heal(tmp_path):
    import invar_agent._native as _native
    from invar_agent.healing.structural import is_structural_failure
    from invar_agent.research.state_machine import ResearchBlueprint, ResearchRunState, ResearchStateMachine
    from invar_agent.research.stochastic_equilibrium import UnsupportedVerification, derive_evidence

    bad_code = "\n".join([
        "import math",
        "def simulation_step(buf, dt):",
        "    x = buf.data[0]",
        "    alpha = buf.data[2]",
        "    buf.data[0] = x + dt * (0.1 + alpha - x)",
        "    buf.data[1] = -x * math.log(x + 1e-12)",
    ])
    good_code = bad_code.replace("x + 1e-12", "x")
    bp = ResearchBlueprint.model_validate({
        "blueprint_id": "bp_heal_mismatch",
        "title": "Entropy Fix",
        "falsifiable_hypothesis": "Entropy at interior x*=0.37 exceeds 0.355",
        "research_prompt": "Find x* solving 0.1 + alpha - x = 0 with H = -x*log(x) >= 0.355.",
        "search_queries": [],
        "variable_map": {"x": 0, "H": 1, "alpha": 2},
        "slot_roles": {"x": "evolving_state", "H": "diagnostic"},
        "code_template": bad_code,
        "contracts": [{
            "contract_id": "c1",
            "target_backend": "cpp",
            "initial_state": [0.35, 0.367, 0.27],
            "dt": 0.05,
            "steps": 40,
            "invariants": [{"variable_name": "H", "assertion": "state[1] >= 0.355", "tolerance": 1e-9}],
        }],
    })
    raw_ihcg = m.IhcgGraph(
        archetype="FIXED_POINT",
        quantifier_mode="EXISTS",
        slots=(
            m.SlotSpec(symbol="x", slot_idx=0, role="DYNAMIC_STATE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="H", slot_idx=1, role="DERIVED_OBSERVABLE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="alpha", slot_idx=2, role="CONTROL_PARAM", si_dim=(0,) * 7),
        ),
        omega={"x": (0.05, 0.9), "H": (0.0, 1.0), "alpha": (0.27, 0.27)},
        control_values={"alpha": 0.27},
        mechanism_parameters=("alpha",),
        equations={
            "x": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x", expression="0.1 + alpha - x", si_dim=(0,) * 7),
            "H": m.EquationSpec(kind="ALGEBRAIC", expression="-x * log(x)", si_dim=(0,) * 7),
        },
        obligations=(
            m.Obligation(id="o1", obligation_class="HYPOTHESIS_TARGET", expression="H >= 0.355", prompt_quote="H >= 0.355", prompt_span=(0, 10)),
        ),
        environmental_assumptions=(),
        h_ihcg="",
    )
    ihcg = raw_ihcg.model_copy(update={"h_ihcg": _native.py_compute_ihcg_digest(raw_ihcg.model_dump_json())})
    sm = ResearchStateMachine(str(tmp_path))
    st = ResearchRunState(blueprint=bp, ihcg=ihcg, research_stage=1, heal_count=0)

    with pytest.raises(UnsupportedVerification, match=r"ALGEBRAIC_EXECUTABLE_MISMATCH: slot='H'"):
        derive_evidence(sm, st, bad_code, phase="execution")

    decision = await sm._evaluate_epistemic(st, phase="post_execution", code=bad_code)
    assert decision is not None
    assert decision.action == "REFLEX_HEAL"
    assert decision.reason.startswith("EXECUTABLE_IHCG_MISMATCH: ALGEBRAIC_EXECUTABLE_MISMATCH: slot='H'")
    assert is_structural_failure(decision.reason) is True

    st.heal_count = 2
    exhausted = await sm._evaluate_epistemic(st, phase="post_execution", code=bad_code)
    assert exhausted is None
    assert st.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED"

    st.heal_count = 1
    good_decision = await sm._evaluate_epistemic(st, phase="post_execution", code=good_code)
    assert good_decision is not None
    assert st.epistemic_verdict == "VERIFIED_DISCOVERY"


def test_auto_populate_variable_map_preserves_ihcg_suffix_names():
    from invar_agent.research.state_machine import ResearchBlueprint, _auto_populate_variable_map

    bp = ResearchBlueprint.model_validate({
        "blueprint_id": "bp_suffix",
        "title": "Suffix Preserve",
        "falsifiable_hypothesis": "Eigenvalue bound stays strictly negative.",
        "search_queries": [],
        "variable_map": {"x": 0, "eigenvalue_real_part_bound": 1},
        "code_template": "def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0]\n    buf.data[1] = -0.1\n",
        "contracts": [{
            "contract_id": "c1",
            "target_backend": "cpp",
            "initial_state": [0.5, -0.1],
            "dt": 0.05,
            "steps": 10,
            "invariants": [{"variable_name": "eigenvalue_real_part", "assertion": "state[1] <= -0.02", "tolerance": 1e-9}],
        }],
        "ihcg_evidence": {
            "evidence": {
                "slots": [
                    {"symbol": "x", "slot_idx": 0, "role": "DYNAMIC_STATE", "si_dim": [0] * 7},
                    {"symbol": "eigenvalue_real_part_bound", "slot_idx": 1, "role": "DERIVED_OBSERVABLE", "si_dim": [0] * 7},
                ]
            }
        },
    })
    _auto_populate_variable_map(bp)
    assert bp.variable_map.get("eigenvalue_real_part_bound") == 1


@pytest.mark.asyncio
async def test_fixed_point_null_ablation_symbolic_cancellation_certifies_rps_equilibrium(tmp_path):
    import invar_agent._native as _native
    from invar_agent.research.state_machine import ResearchBlueprint, ResearchRunState, ResearchStateMachine

    code = "\n".join([
        "def d1(x1, x2, x3, a, mu):",
        "    return x1 * (-x2 + (1.0 + a) * x3) + mu * (1.0 / 3.0 - x1)",
        "def d2(x1, x2, x3, a, mu):",
        "    return x2 * (x1 - x3) + mu * (1.0 / 3.0 - x2)",
        "def d3(x1, x2, x3, a, mu):",
        "    return x3 * (-(1.0 + a) * x1 + x2) + mu * (1.0 / 3.0 - x3)",
        "def simulation_step(buf, dt):",
        "    x1 = buf.data[0]",
        "    x2 = buf.data[1]",
        "    x3 = buf.data[2]",
        "    a = buf.data[4]",
        "    mu = buf.data[5]",
        "    buf.data[0] = x1 + dt * d1(x1, x2, x3, a, mu)",
        "    buf.data[1] = x2 + dt * d2(x1, x2, x3, a, mu)",
        "    buf.data[2] = x3 + dt * d3(x1, x2, x3, a, mu)",
        "    buf.data[3] = (-x2 + (1.0 + a) * x3 - (1.0 + a) * x1 - mu) + (x1 - x3 + x2 - mu)",
        "    buf.data[6] = x2 - x3",
    ])
    bp = ResearchBlueprint.model_validate({
        "blueprint_id": "bp_rps_null_simp",
        "title": "RPS Null Simplification",
        "falsifiable_hypothesis": "Asymmetry alpha and mutation mu make trace <= -0.05 and x2 - x3 >= 0.05",
        "research_prompt": "Test trace <= -0.05 and diff >= 0.05 at fixed point.",
        "search_queries": [],
        "variable_map": {"x1": 0, "x2": 1, "x3": 2, "trace": 3, "alpha": 4, "mu": 5, "diff": 6},
        "code_template": code,
        "contracts": [{
            "contract_id": "c1",
            "target_backend": "cpp",
            "initial_state": [0.3164196548, 0.3826177045, 0.3009626407, -0.323, 0.25, 0.12, 0.0816],
            "dt": 0.05,
            "steps": 40,
            "invariants": [
                {"variable_name": "trace", "assertion": "state[3] <= -0.05", "tolerance": 1e-9},
                {"variable_name": "diff", "assertion": "state[6] >= 0.05", "tolerance": 1e-9},
            ],
        }],
    })
    raw_ihcg = m.IhcgGraph(
        archetype="FIXED_POINT",
        quantifier_mode="EXISTS",
        slots=(
            m.SlotSpec(symbol="x1", slot_idx=0, role="DYNAMIC_STATE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="x2", slot_idx=1, role="DYNAMIC_STATE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="x3", slot_idx=2, role="DYNAMIC_STATE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="trace", slot_idx=3, role="DERIVED_OBSERVABLE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="alpha", slot_idx=4, role="CONTROL_PARAM", si_dim=(0,) * 7),
            m.SlotSpec(symbol="mu", slot_idx=5, role="CONTROL_PARAM", si_dim=(0,) * 7),
            m.SlotSpec(symbol="diff", slot_idx=6, role="DERIVED_OBSERVABLE", si_dim=(0,) * 7),
        ),
        omega={"x1": (0.05, 0.9), "x2": (0.05, 0.9), "x3": (0.05, 0.9), "trace": (-2.0, 2.0), "diff": (-1.0, 1.0), "alpha": (0.25, 0.25), "mu": (0.12, 0.12)},
        control_values={"alpha": 0.25, "mu": 0.12},
        mechanism_parameters=("alpha", "mu"),
        equations={
            "x1": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x1", expression="x1 * (-x2 + (1.0 + alpha) * x3) + mu * (1.0 / 3.0 - x1)", si_dim=(0,) * 7),
            "x2": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x2", expression="x2 * (x1 - x3) + mu * (1.0 / 3.0 - x2)", si_dim=(0,) * 7),
            "x3": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x3", expression="x3 * (-(1.0 + alpha) * x1 + x2) + mu * (1.0 / 3.0 - x3)", si_dim=(0,) * 7),
            "trace": m.EquationSpec(kind="ALGEBRAIC", expression="(-x2 + (1.0 + alpha) * x3 - (1.0 + alpha) * x1 - mu) + (x1 - x3 + x2 - mu)", si_dim=(0,) * 7),
            "diff": m.EquationSpec(kind="ALGEBRAIC", expression="x2 - x3", si_dim=(0,) * 7),
        },
        obligations=(
            m.Obligation(id="o1", obligation_class="HYPOTHESIS_TARGET", expression="trace <= -0.05 and diff >= 0.05", prompt_quote="trace <= -0.05 and diff >= 0.05", prompt_span=(0, 31)),
        ),
        environmental_assumptions=(),
        h_ihcg="",
    )
    ihcg = raw_ihcg.model_copy(update={"h_ihcg": _native.py_compute_ihcg_digest(raw_ihcg.model_dump_json())})
    sm = ResearchStateMachine(str(tmp_path))
    st = ResearchRunState(blueprint=bp, ihcg=ihcg, research_stage=1)
    await sm._evaluate_epistemic(st, phase="post_execution", code=code)
    assert st.epistemic_verdict == "VERIFIED_DISCOVERY"
    assert st.meaningfulness_certificate.significance_audit.method == "FIXED_POINT_KRAWCZYK_HURWITZ"
    assert st.meaningfulness_certificate.significance_audit.certified_lower_margin > 0.0
    assert st.meaningfulness_certificate.purity_audit.enforcement_ratio == 0.0


@pytest.mark.asyncio
async def test_fixed_point_ode_kernel_accepts_sanitizer_domain_guards_when_domain_proven(tmp_path):
    import invar_agent._native as _native
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics
    from invar_agent.research.state_machine import ResearchBlueprint, ResearchRunState, ResearchStateMachine

    raw_code = "\n".join([
        "import math",
        "def simulation_step(buf, dt):",
        "    x = buf.data[0]",
        "    alpha = buf.data[3]",
        "    r = (0.1 + alpha) - x",
        "    buf.data[0] = x + dt * r",
        "    buf.data[1] = -x * math.log(x)",
        "    buf.data[2] = math.sqrt(r * r)",
    ])
    san_code = sanitize_intrinsics(raw_code).code
    assert "max(" in san_code
    bp = ResearchBlueprint.model_validate({
        "blueprint_id": "bp_san_guard",
        "title": "Sanitized Sqrt Log Regression",
        "falsifiable_hypothesis": "Entropy H >= 0.355 and residual l2 <= 0.05 at x*=0.37.",
        "research_prompt": "Test H >= 0.355.",
        "search_queries": [],
        "variable_map": {"x": 0, "H": 1, "res": 2, "alpha": 3},
        "code_template": san_code,
        "contracts": [{
            "contract_id": "c1",
            "target_backend": "cpp",
            "initial_state": [0.37, 0.367, 0.0, 0.27],
            "dt": 0.05,
            "steps": 40,
            "invariants": [
                {"variable_name": "H", "assertion": "state[1] >= 0.355", "tolerance": 1e-9},
                {"variable_name": "res", "assertion": "state[2] <= 0.05", "tolerance": 1e-9},
            ],
        }],
    })
    raw_ihcg = m.IhcgGraph(
        archetype="FIXED_POINT",
        quantifier_mode="EXISTS",
        slots=(
            m.SlotSpec(symbol="x", slot_idx=0, role="DYNAMIC_STATE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="H", slot_idx=1, role="DERIVED_OBSERVABLE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="res", slot_idx=2, role="DERIVED_OBSERVABLE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="alpha", slot_idx=3, role="CONTROL_PARAM", si_dim=(0,) * 7),
        ),
        omega={"x": (0.05, 0.9), "H": (0.0, 1.0), "res": (0.0, 2.0), "alpha": (0.27, 0.27)},
        control_values={"alpha": 0.27},
        mechanism_parameters=("alpha",),
        equations={
            "x": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x", expression="0.1 + alpha - x", si_dim=(0,) * 7),
            "H": m.EquationSpec(kind="ALGEBRAIC", expression="-x * log(x)", si_dim=(0,) * 7),
            "res": m.EquationSpec(kind="ALGEBRAIC", expression="sqrt(((0.1 + alpha) - x) ** 2)", si_dim=(0,) * 7),
        },
        obligations=(
            m.Obligation(id="o1", obligation_class="HYPOTHESIS_TARGET", expression="H >= 0.355 and res <= 0.05", prompt_quote="H >= 0.355", prompt_span=(0, 10)),
        ),
        environmental_assumptions=(),
        h_ihcg="",
    )
    ihcg = raw_ihcg.model_copy(update={"h_ihcg": _native.py_compute_ihcg_digest(raw_ihcg.model_dump_json())})
    sm = ResearchStateMachine(str(tmp_path))
    st = ResearchRunState(blueprint=bp, ihcg=ihcg, research_stage=1)
    await sm._evaluate_epistemic(st, phase="post_execution", code=san_code)
    assert st.epistemic_verdict == "VERIFIED_DISCOVERY"
    assert st.meaningfulness_certificate.purity_audit.enforcement_ratio == 0.0


@pytest.mark.asyncio
async def test_fixed_point_res_name_subs_and_negative_discriminant_sqrt_guard(tmp_path):
    import invar_agent._native as _native
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics
    from invar_agent.research.state_machine import ResearchBlueprint, ResearchRunState, ResearchStateMachine

    raw_code = "\n".join([
        "import math",
        "def simulation_step(buf, dt):",
        "    x = buf.data[0]",
        "    y = buf.data[1]",
        "    alpha = buf.data[6]",
        "    f1 = (0.1 + alpha) - x + 0.5 * (y - 0.5)",
        "    f2 = (0.5 - y) - 0.5 * (x - (0.1 + alpha))",
        "    buf.data[0] = x + dt * f1",
        "    buf.data[1] = y + dt * f2",
        "    buf.data[2] = f1",
        "    buf.data[3] = -x * math.log(x)",
        "    tr = -2.0",
        "    det = 1.25",
        "    disc = tr * tr - 4.0 * det",
        "    eig_val = 0.5 * (tr + math.sqrt(max(0.0, disc)))",
        "    buf.data[4] = eig_val",
        "    buf.data[5] = y - x",
    ])
    san_code = sanitize_intrinsics(raw_code).code
    bp = ResearchBlueprint.model_validate({
        "blueprint_id": "bp_res_name_sqrt",
        "title": "Residual Name Subs and Complex Discriminant",
        "falsifiable_hypothesis": "At x*=0.37, y*=0.5, |f1|<=1e-3, H>=0.355, eig_real<=-0.5.",
        "research_prompt": "Test F1<=1e-3, H>=0.355, eig_real<=-0.5.",
        "search_queries": [],
        "variable_map": {"x": 0, "y": 1, "F1": 2, "H": 3, "eig": 4, "diff": 5, "alpha": 6},
        "code_template": san_code,
        "contracts": [{
            "contract_id": "c1",
            "target_backend": "cpp",
            "initial_state": [0.37, 0.5, 0.0, 0.367, -1.0, 0.13, 0.27],
            "dt": 0.05,
            "steps": 40,
            "invariants": [
                {"variable_name": "F1", "assertion": "abs(state[2]) <= 1e-3", "tolerance": 1e-9},
                {"variable_name": "H", "assertion": "state[3] >= 0.355", "tolerance": 1e-9},
                {"variable_name": "eig", "assertion": "state[4] <= -0.5", "tolerance": 1e-9},
                {"variable_name": "diff", "assertion": "state[5] >= -0.05", "tolerance": 1e-9},
            ],
        }],
    })
    raw_ihcg = m.IhcgGraph(
        archetype="FIXED_POINT",
        quantifier_mode="EXISTS",
        slots=(
            m.SlotSpec(symbol="x", slot_idx=0, role="DYNAMIC_STATE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="y", slot_idx=1, role="DYNAMIC_STATE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="F1", slot_idx=2, role="DERIVED_OBSERVABLE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="H", slot_idx=3, role="DERIVED_OBSERVABLE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="eig", slot_idx=4, role="DERIVED_OBSERVABLE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="diff", slot_idx=5, role="DERIVED_OBSERVABLE", si_dim=(0,) * 7),
            m.SlotSpec(symbol="alpha", slot_idx=6, role="CONTROL_PARAM", si_dim=(0,) * 7),
        ),
        omega={"x": (0.05, 0.9), "y": (0.05, 0.9), "F1": (-2.0, 2.0), "H": (0.0, 1.0), "eig": (-3.0, 3.0), "diff": (-1.0, 1.0), "alpha": (0.27, 0.27)},
        control_values={"alpha": 0.27},
        mechanism_parameters=("alpha",),
        equations={
            "F1": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="x", expression="0.1 + alpha - x + 0.5 * (y - 0.5)", si_dim=(0,) * 7),
            "F2": m.EquationSpec(kind="FIXED_POINT_RESIDUAL", state_symbol="y", expression="(0.5 - y) - 0.5 * (x - (0.1 + alpha))", si_dim=(0,) * 7),
            "H": m.EquationSpec(kind="ALGEBRAIC", expression="-x * log(x)", si_dim=(0,) * 7),
            "eig": m.EquationSpec(kind="ALGEBRAIC", expression="0.5 * (-2.0 + sqrt(max(0.0, (-2.0) ** 2 - 4.0 * 1.25)))", si_dim=(0,) * 7),
            "diff": m.EquationSpec(kind="ALGEBRAIC", expression="y - x", si_dim=(0,) * 7),
        },
        obligations=(
            m.Obligation(id="o1", obligation_class="HYPOTHESIS_TARGET", expression="abs(F1) <= 1e-3 and H >= 0.355 and eig <= -0.5", prompt_quote="abs(F1) <= 1e-3 and H >= 0.355", prompt_span=(0, 32)),
        ),
        environmental_assumptions=(),
        h_ihcg="",
    )
    ihcg = raw_ihcg.model_copy(update={"h_ihcg": _native.py_compute_ihcg_digest(raw_ihcg.model_dump_json())})
    sm = ResearchStateMachine(str(tmp_path))
    st = ResearchRunState(blueprint=bp, ihcg=ihcg, research_stage=1)
    await sm._evaluate_epistemic(st, phase="post_execution", code=san_code)
    assert st.epistemic_verdict == "VERIFIED_DISCOVERY"
    assert st.meaningfulness_certificate.fidelity_audit.epsilon_disc < 1e-3
    assert st.meaningfulness_certificate.purity_audit.enforcement_ratio == 0.0








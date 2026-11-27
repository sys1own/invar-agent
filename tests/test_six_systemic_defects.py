"""Deterministic regression tests locking all 6 universal systemic fixes."""

import asyncio
import json
from pathlib import Path
import pytest

from invar_agent.healing.deterministic_ast import sanitize_intrinsics
from invar_agent.healing.structural import is_structural_failure
from invar_agent.research.accretion import is_noop_layer_wrapper
from invar_agent.research.extractor import (
    QuantitativeInsightExtractor,
    extract_parameters_from_text,
)
from invar_agent.research.planner import canonicalize_contract_dicts
from invar_agent.research.report import rank_literature_by_relevance
from invar_agent.research.scientific_integrity import validate_observables
from invar_agent.research.search import (
    LiteratureSearchClient,
    _channels_for_prompt,
    is_numerical_method_query,
)
from invar_agent.research.slot_schema import (
    ROLE_CONTROL_PARAM,
    ROLE_DIAGNOSTIC,
    ROLE_STATE,
    ROLE_TIME_STEP,
    audit_contract_alignment,
    classify_variable_roles,
    compile_initial_state_from_values,
    validate_contract_faithfulness,
)
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.schemas.research import (
    InvariantConstraint,
    ResearchBlueprint,
    VerificationContract,
)


def test_fix1_unified_role_classifier_and_no_projected_diagnostic_deadlock():
    vm = {
        "pool_a": 0,
        "pool_b": 1,
        "x_state": 2,
        "g_ctrl": 3,
        "time_s": 4,
        "ratio_diag": 5,
        "peak_x": 6,
    }
    code = (
        "def layer_1_step(buf, dt):\n"
        "    a = buf.data[0]\n"
        "    b = buf.data[1]\n"
        "    g = buf.data[3]\n"
        "    buf.data[0] = max(0.0, a - dt * g * a + dt * b)\n"
        "    buf.data[1] = max(0.0, 10.0 - buf.data[0])\n"
        "    buf.data[2] = buf.data[2] + dt * (a - g * buf.data[2])\n"
        "    buf.data[4] = buf.data[4] + dt\n"
        "    buf.data[5] = buf.data[0] / (buf.data[0] + buf.data[1] + 1e-12)\n"
        "    if buf.data[2] > buf.data[6]:\n"
        "        buf.data[6] = buf.data[2]\n"
        "\ndef simulation_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
    )
    roles = classify_variable_roles(vm, code)
    assert roles["time_s"] == ROLE_TIME_STEP
    assert roles["g_ctrl"] == ROLE_CONTROL_PARAM
    assert roles["ratio_diag"] == ROLE_DIAGNOSTIC
    assert roles["peak_x"] == ROLE_DIAGNOSTIC
    assert roles["x_state"] == ROLE_STATE

    contracts = [
        VerificationContract(
            contract_id="c2_finiteness",
            target_backend="cpp",
            initial_state=[5.0, 5.0, 1.0, 0.2, 0.0, 0.5, 1.0],
            dt=0.01,
            steps=10,
            invariants=[
                InvariantConstraint(
                    variable_name="finiteness_c2",
                    assertion="math.isfinite(state[2]) and math.isfinite(state[5]) and math.isfinite(state[6])",
                )
            ],
        )
    ]
    san = sanitize_intrinsics(
        code,
        contracts=contracts,
        all_contracts=[c.model_dump() for c in contracts],
        variable_map=vm,
    )
    # Tier-0 must never trigger SCIENTIFIC_INTEGRITY by resetting diagnostic slots
    validate_observables(san.code, contracts, vm, {})

    # Bad manual projection on a diagnostic slot produces actionable diagnostic + structural routing
    bad_code = code.replace(
        "def simulation_step(buf, dt):",
        "    buf.data[5] = max(0.0, buf.data[5] - 1.0)\n\ndef simulation_step(buf, dt):",
    )
    with pytest.raises(ValueError) as exc_info:
        validate_observables(bad_code, contracts, vm, {})
    msg = str(exc_info.value)
    assert "SCIENTIFIC_INTEGRITY" in msg
    assert "buf.data[5]" in msg or "state[5]" in msg
    assert "ratio_diag" in msg
    assert is_structural_failure(msg)


@pytest.mark.asyncio
async def test_fix1_stage_rollback_purges_rolled_back_contracts(tmp_path: Path):
    bp = ResearchBlueprint(
        blueprint_id="rollback_purge_test",
        title="Rollback Contract Purge",
        research_prompt="Verify stage rollback purges stage 2 contracts before stage 3.",
        falsifiable_hypothesis="Stage 2 failure never leaks into Stage 3.",
        search_queries=["rollback stage contract purge"],
        target_research_stages=3,
        max_feedback_rounds=1,
        stage_objectives=["Stage 1 baseline", "Stage 2 doomed", "Stage 3 valid"],
        contracts=[
            VerificationContract(
                contract_id="c1_base",
                target_backend="cpp",
                initial_state=[1.0, 2.0, 1.0],
                dt=0.01,
                steps=5,
                invariants=[InvariantConstraint(variable_name="x_pos", assertion="state[0] >= 0.0")],
            ),
            VerificationContract(
                contract_id="c2_impossible",
                target_backend="cpp",
                initial_state=[1.0, 2.0, 1.0],
                dt=0.01,
                steps=5,
                invariants=[
                    InvariantConstraint(
                        variable_name="finiteness_c2_fail",
                        assertion="state[2] > 0.9",
                    )
                ],
            ),
            VerificationContract(
                contract_id="c3_valid",
                target_backend="cpp",
                initial_state=[1.0, 2.0, 1.0],
                dt=0.01,
                steps=5,
                invariants=[InvariantConstraint(variable_name="y_pos", assertion="state[1] >= 0.0")],
            ),
        ],
        variable_map={"x_val": 0, "y_val": 1, "ratio_diag": 2},
        code_template=(
            "def simulation_step(buf, dt):\n"
            "    buf.data[0] = buf.data[0] + 0.01 * dt\n"
            "    buf.data[1] = buf.data[1] + 0.02 * dt\n"
            "    buf.data[2] = buf.data[0] / (buf.data[0] + buf.data[1])\n"
        ),
    )

    seen_stage3_contracts = []

    async def fake_synth(**kw):
        s = kw["stage"]
        if s == 2:
            # Stage 2 fails its unsatisfied diagnostic target (0.333 < 0.9) and rolls back
            return {
                "code": (
                    "def layer_2_step(buf, dt):\n"
                    "    layer_1_step(buf, dt)\n"
                    "    buf.data[2] = buf.data[0] / (buf.data[0] + buf.data[1])\n"
                    "\ndef simulation_step(buf, dt):\n    layer_2_step(buf, dt)\n"
                ),
                "contracts": [],
                "rationale": "stage 2",
            }
        seen_stage3_contracts.extend([c["contract_id"] if isinstance(c, dict) else c.contract_id for c in (kw.get("contracts") or [])])
        return {
            "code": (
                "def layer_3_step(buf, dt):\n"
                "    layer_1_step(buf, dt)\n"
                "    buf.data[1] = buf.data[1] + 0.002 * dt\n"
                "\ndef simulation_step(buf, dt):\n    layer_3_step(buf, dt)\n"
            ),
            "contracts": [],
            "rationale": "stage 3",
        }

    sm = ResearchStateMachine(tmp_path, stage_synthesizer=fake_synth)
    state = await sm.run(bp)
    assert 2 in state.rolled_back_stages
    remaining_ids = {c.contract_id for c in state.blueprint.contracts}
    assert "c2_impossible" not in remaining_ids
    assert "c3_valid" in remaining_ids
    assert any(h.stage == 3 and h.status == "VERIFIED" for h in state.blueprint.evolution_history)


def test_fix2_named_initial_values_and_alignment_audit():
    vm = {"pot_v": 0, "time_s": 1, "g_leak": 2, "ros_uM": 3, "peak_ros_uM": 4}
    iv = {"g_leak": 0.0012, "pot_v": 0.165, "time_s": 0.0, "peak_ros_uM": 1.0, "ros_uM": 1.0}
    vec = compile_initial_state_from_values(vm, iv)
    assert vec == [0.165, 0.0, 0.0012, 1.0, 1.0]

    # Unaligned shifted TIME_STEP or zeroed positive control param fails audit
    bad_time = [{"contract_id": "c1", "initial_state": [0.165, 0.001, 0.0, 1.0, 1.0], "invariants": []}]
    with pytest.raises(ValueError, match="SLOT_ALIGNMENT_ERROR"):
        audit_contract_alignment(bad_time, vm, prompt="Simulate g_leak in [1e-4, 1.2e-3] over 0 to 600 s")

    # Extrema invariant bound to instantaneous slot gets rebound deterministically during canonicalization
    raw_contracts = [
        {
            "contract_id": "c1",
            "initial_values": iv,
            "dt": 0.1,
            "steps": 100,
            "invariants": [
                {"variable_name": "peak_ros_nonneg", "assertion": "state[3] >= 0.0", "tolerance": 1e-6}
            ],
        }
    ]
    canon, _, _ = canonicalize_contract_dicts(raw_contracts, vm)
    assert canon[0]["initial_state"] == vec
    assert canon[0]["invariants"][0]["assertion"] == "state[4] >= 0.0"


def test_fix3_post_projection_diagnostic_resync():
    code = (
        "def layer_1_step(buf, dt):\n"
        "    a = buf.data[0]\n"
        "    b = buf.data[1]\n"
        "    c = buf.data[2]\n"
        "    buf.data[0] = a - dt * 100.0 * a\n"
        "    buf.data[1] = b + dt * 50.0 * a\n"
        "    buf.data[2] = c + dt * 50.0 * a\n"
        "    buf.data[3] = (buf.data[0] + 0.5 * buf.data[1]) / 12.0\n"
        "\ndef simulation_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
    )
    vm = {"a": 0, "b": 1, "c": 2, "aec": 3}
    contracts = [
        VerificationContract(
            contract_id="c1_pool",
            target_backend="cpp",
            initial_state=[1.0, 11.0, 0.0, 0.541666],
            dt=0.1,
            steps=5,
            invariants=[
                InvariantConstraint(
                    variable_name="pool_nonneg",
                    assertion="state[0] >= 0.0 and state[1] >= 0.0 and state[2] >= 0.0",
                ),
                InvariantConstraint(variable_name="pool_sum", assertion="abs((state[0] + state[1] + state[2]) - 12.0) <= 1e-6"),
                InvariantConstraint(variable_name="aec_diag", assertion="state[3] >= 0.0"),
            ],
        )
    ]
    san = sanitize_intrinsics(
        code,
        contracts=contracts,
        all_contracts=[c.model_dump() for c in contracts],
        variable_map=vm,
    )
    # Execute sanitized step in Python and verify buf.data[3] matches FINAL buf.data[0..1]
    ns = {}
    exec(san.code, ns)

    class Buf:
        def __init__(self, d):
            self.data = list(d)

    buf = Buf([1.0, 11.0, 0.0, 0.541666])
    ns["layer_1_step"](buf, 0.1)
    # ARCH-001: sanitize_intrinsics preserves exact kernel dynamics without injecting silent projection.
    assert abs(buf.data[0] - (1.0 - 0.1 * 100.0 * 1.0)) < 1e-12
    assert abs((buf.data[0] + buf.data[1] + buf.data[2]) - 12.0) < 1e-9
    assert abs(buf.data[3] - (buf.data[0] + 0.5 * buf.data[1]) / 12.0) < 1e-12
    expected_aec = (buf.data[0] + 0.5 * buf.data[1]) / 12.0
    assert abs(buf.data[3] - expected_aec) < 1e-12


def test_fix4_contract_faithfulness_and_noop_layer_rejection():
    vm = {"delta_psi": 0, "g_leak": 1}
    hollow_band = [
        {
            "contract_id": "c1",
            "initial_state": [0.15, 0.001],
            "invariants": [{"variable_name": "delta_psi_band", "assertion": "state[0] >= 0.0"}],
        }
    ]
    with pytest.raises(ValueError, match="CONTRACT_FAITHFULNESS_ERROR"):
        validate_contract_faithfulness(hollow_band, vm)

    dup_sweep = [
        {
            "contract_id": "c1",
            "initial_state": [0.15, 0.001],
            "invariants": [{"variable_name": "delta_psi_band", "assertion": "0.10 <= state[0] <= 0.20"}],
        },
        {
            "contract_id": "c2",
            "initial_state": [0.15, 0.001],
            "invariants": [{"variable_name": "delta_psi_band", "assertion": "0.10 <= state[0] <= 0.20"}],
        },
    ]
    with pytest.raises(ValueError, match="CONTRACT_FAITHFULNESS_ERROR"):
        validate_contract_faithfulness(
            dup_sweep,
            vm,
            prompt="Sweep g_leak in [1e-4, 1.2e-3] S/m^2.",
            require_sweep_coverage=True,
        )

    noop_code = (
        "def layer_1_step(buf, dt):\n"
        "    buf.data[0] = buf.data[0] - dt * buf.data[1]\n\n"
        "def layer_3_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n\n"
        "def simulation_step(buf, dt):\n"
        "    layer_3_step(buf, dt)\n"
    )
    bad, reason = is_noop_layer_wrapper(noop_code, 3)
    assert bad and "NOOP_ACCRETION_LAYER" in reason


def test_fix5_scoping_ranking_and_equation_extraction():
    num_q = "Patankar positivity-preserving semi-implicit Euler ODE solver"
    assert is_numerical_method_query(num_q)
    channels = _channels_for_prompt(num_q)
    assert "medrxiv" not in channels and "pubmed" not in channels
    assert "arxiv" in channels

    ranked = rank_literature_by_relevance(
        [
            {"source": "medrxiv", "stage": 2, "title": "Post-burn contracture reconstruction randomized trial", "summary": "clinical", "url": "https://example.org/burn"},
            {"source": "pubmed", "stage": 1, "title": "Mitochondrial OXPHOS adenine nucleotide dynamics", "summary": "ATP ADP membrane potential", "url": "https://example.org/mito", "equations": ["dV/dt = -I_leak/C_m"]},
        ],
        prompt="Mitochondrial OXPHOS adenine nucleotide membrane potential",
    )
    assert ranked[0]["url"] == "https://example.org/mito"

    ext = QuantitativeInsightExtractor()
    text = (
        "The voltage satisfies \\( dV/dt = -(I_res - I_leak) / C_m \\), "
        "and energy charge AEC = (ATP + 0.5 * ADP) / A_tot, where p = 0.05 in n = 12."
    )
    eqs = ext.equations_from_text(text, source="unit")
    eq_strs = [f"{e.lhs} = {e.rhs_latex}" for e in eqs]
    assert any("dV" in s or "I_res" in s or "ATP" in s for s in eq_strs)
    assert not any(s.startswith("p =") or s.startswith("n =") for s in eq_strs)


def test_fix6_participle_stopwords_and_preceding_symbol_binding():
    prompt = (
        "Delta_Psi (initialized at 165 mV across a 7.5 nm bilayer with "
        "specific capacitance C_m = 1.0 uF/cm^2, dropping toward 0); "
        "Delta_pH (baseline 0.5 units at T = 310.15 K); "
        "g_leak (spanning 1e-4 to 1.2e-3 S/m^2)."
    )
    params = extract_parameters_from_text(prompt, source="user_prompt")
    names = {p.name for p in params}
    assert "initialized" not in names
    assert "spanning" not in names
    assert "dropping" not in names
    by_name = {p.name: p for p in params}
    assert "delta_psi" in by_name
    assert by_name["delta_psi"].symbol == "Delta_Psi"
    assert abs(by_name["delta_psi"].si_value - 0.165) < 1e-9 and by_name["delta_psi"].si_unit == "V"
    assert "t" in by_name
    assert by_name["t"].symbol == "T"
    assert abs(by_name["t"].si_value - 310.15) < 1e-9 and by_name["t"].si_unit == "K"
    assert "gradient_delta_ph" not in by_name and "delta_ph" not in by_name
    assert "bilayer" in by_name
    assert abs(by_name["bilayer"].si_value - 7.5e-9) < 1e-16 and by_name["bilayer"].si_unit == "m"
    assert "specific_capacitance_c_m" in by_name
    assert abs(by_name["specific_capacitance_c_m"].si_value - 0.01) < 1e-9 and by_name["specific_capacitance_c_m"].si_unit == "F/m^2"


def test_shared_variable_map_corridor_no_phantom_clone_and_shared_symbol_constants():
    vm = {"pot_v": 0, "g_unc": 1, "g_leak": 2}
    raw_contracts = [
        {
            "contract_id": "c1_baseline",
            "initial_values": {"pot_v": 0.165, "g_unc": 0.0, "g_leak": 0.0001},
            "dt": 0.05,
            "steps": 100,
            "invariants": [{"variable_name": "finite_v", "assertion": "state[0] >= 0.0"}],
        },
        {
            "contract_id": "c2b_corridor",
            "initial_values": {"pot_v": 0.165, "g_unc": 0.0015, "g_leak": 0.001},
            "dt": 0.05,
            "steps": 100,
            "invariants": [{"variable_name": "pot_corridor_v", "assertion": "state[0] >= 0.115 and state[0] <= 0.155"}],
        },
    ]
    canon, out_vm, _ = canonicalize_contract_dicts(raw_contracts, vm)
    assert "state[0]" in canon[1]["invariants"][0]["assertion"]
    assert 3 not in out_vm.values()
    assert canon[0]["initial_state"][1] == pytest.approx(0.0)
    assert canon[1]["initial_state"][1] == pytest.approx(0.0015)

    from invar_agent.research.evolution import parameter_bindings
    from invar_agent.runtime.research_reference import ReferenceError as RefError

    bp = {
        "structured_evidence": {
            "parameters": {
                "potential_delta_psi": {"symbol": "Delta_Psi", "si_value": 0.165, "si_unit": "V"},
                "hyperpolarized_delta_psi": {"symbol": "Delta_Psi", "si_value": 0.15, "si_unit": "V"},
            }
        },
        "extracted_constants": {"potential_delta_psi": 0.165, "hyperpolarized_delta_psi": 0.15},
    }
    bindings = parameter_bindings(bp, {})
    assert bindings["Delta_Psi"]["value"] == pytest.approx(0.165)
    assert bindings["hyperpolarized_delta_psi"]["value"] == pytest.approx(0.15)

    bp_bad = {
        **bp,
        "extracted_constants": {"potential_delta_psi": 0.25, "hyperpolarized_delta_psi": 0.15},
    }
    with pytest.raises(RefError) as excinfo:
        parameter_bindings(bp_bad, {})
    assert excinfo.value.code == "VAULT_CONSTANT_MISMATCH"


def test_uncontracted_ode_rate_slot_with_constitutive_saturation_not_flagged_clipped():
    code = (
        "def compute_rates(buf, t, v):\n"
        "    gate = 1.0 - 1.5 * v\n"
        "    if gate < 0.0:\n"
        "        gate = 0.0\n"
        "    if gate > 1.0:\n"
        "        gate = 1.0\n"
        "    buf.data[1] = -0.1 * gate * v\n"
        "def simulation_step(buf, dt):\n"
        "    compute_rates(buf, 0.0, buf.data[0])\n"
        "    buf.data[0] = buf.data[0] + dt * buf.data[1]\n"
    )
    vm = {"v": 0, "dv_dt": 1}
    contracts = [VerificationContract(contract_id="c1", target_backend="cpp", initial_state=[0.2, 0.0], dt=0.05, steps=5,
                                      invariants=[InvariantConstraint(variable_name="v_nonneg", assertion="state[0] >= 0.0")])]
    validate_observables(code, contracts, vm, {"v": "physical", "dv_dt": "diagnostic"})


def test_parameter_bindings_normalizes_prefixed_non_si_units_in_text():
    from invar_agent.research.evolution import parameter_bindings
    from invar_agent.research.extractor import normalize_to_si

    si_cm, _ = normalize_to_si(1.0, "uF/cm^2")
    bp = {
        "falsifiable_hypothesis": "Membrane capacitance C_m = 1.0 uF/cm^2 stabilizes dV/dt.",
        "structured_evidence": {"parameters": {"specific_capacitance_c_m": {"symbol": "C_m", "si_value": si_cm, "si_unit": "F/m^2"}}},
        "extracted_constants": {"specific_capacitance_c_m": si_cm},
    }
    bindings = parameter_bindings(bp, {})
    assert bindings["C_m"]["value"] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_final_heal_attempt_invokes_tier2_refine_on_falsified(tmp_path):
    called = []

    async def fast_llm(prompt):
        return json.dumps({"fixable": True, "directives": []})

    async def planner_llm(prompt):
        called.append(prompt)
        return json.dumps({
            "code": (
                "def layer_1_step(buf, dt):\n"
                "    buf.data[0] = buf.data[0] * 0.5\n"
                "\ndef layer_2_step(buf, dt):\n"
                "    buf.data[0] = buf.data[0] + dt * 1.0\n"
                "\ndef simulation_step(buf, dt):\n"
                "    layer_2_step(buf, dt)\n"
            )
        })

    sm = ResearchStateMachine(tmp_path / "res", llm_fast=fast_llm, llm_planner=planner_llm)
    c2 = VerificationContract(
        contract_id="c2_target",
        target_backend="cpp",
        initial_state=[0.1, 0.0],
        dt=0.1,
        steps=10,
        invariants=[InvariantConstraint(variable_name="target_v", assertion="state[0] >= 0.5")],
    )
    out = await sm._heal_and_refine(
        stage=2,
        round_idx=2,
        objective="reach >= 0.5",
        candidate_code=(
            "def layer_1_step(buf, dt):\n"
            "    buf.data[0] = buf.data[0] * 0.5\n"
            "\ndef layer_2_step(buf, dt):\n"
            "    layer_1_step(buf, dt)\n"
            "\ndef simulation_step(buf, dt):\n"
            "    layer_2_step(buf, dt)\n"
        ),
        contracts=[c2],
        failure="FALSIFIED on c2_target: state[0] >= 0.5 observed 0.05",
        attempt=2,
        max_attempts=3,
        pinned_functions=["layer_1_step"],
    )
    assert called, "attempt=2 with max_attempts=3 (Attempt 3/3) must invoke Tier-2 refine_stage_code"
    assert out["applied_ops"] == ["TIER2_REFINE_STAGE_CODE"]
    assert "layer_2_step" in out["code"] and out["terminated"] is False


def test_clock_slot_never_auto_inits_from_final_time_equality_and_prose_indices():
    from invar_agent.research.state_machine import _guard_contract_initial_state
    from invar_agent.research.evolution import parameter_bindings

    code = (
        "def simulation_step(buf, dt):\n"
        "    t_next = buf.data[0] + dt\n"
        "    buf.data[0] = t_next\n"
        "    buf.data[1] = buf.data[1] * 0.99\n"
    )
    c = VerificationContract(
        contract_id="c1",
        target_backend="cpp",
        initial_state=[0.0, 1.0],
        dt=0.1,
        steps=6000,
        invariants=[InvariantConstraint(variable_name="time_s", assertion="abs(state[0] - 600.0) <= 0.1")],
    )
    init, warns = _guard_contract_initial_state(code, c, variable_map={"time_s": 0, "x": 1})
    assert init[0] == 0.0
    assert not any("state[0]" in w for w in warns)

    bp = {
        "falsifiable_hypothesis": "Pool A uses K = 3 species and Pool B uses K = 2 species.",
        "structured_evidence": {"parameters": {}},
        "extracted_constants": {},
    }
    assert parameter_bindings(bp, {})["K"]["value"] == pytest.approx(3.0)


def test_evolution_reference_handles_exp_abs_lambdify_without_re_nameerror():
    from invar_agent.research.evolution import check_evolution_reference

    code = (
        "import math\n"
        "def simulation_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    k = math.exp(40.0 * x) - 1.0\n"
        "    buf.data[0] = x - dt * k * 1e-6\n"
    )
    model = {
        "variable_map": {"x": 0},
        "state_order": ["x"],
        "structured_evidence": {"parameters": {}},
        "contracts": [{"contract_id": "c1", "dt": 0.01, "steps": 2, "initial_state": [0.15], "invariants": []}],
    }
    res = check_evolution_reference(model, {}, code)
    assert res["integration_scheme"] == "euler"


def test_discovery_corridor_no_upper_or_lower_clamp_and_readonly_vmap_slots():
    code = (
        "def simulation_step(buf, dt):\n"
        "    v = buf.data[0]\n"
        "    g = buf.data[1]\n"
        "    buf.data[0] = v - dt * g * v\n"
    )
    c = VerificationContract(
        contract_id="c2_corridor",
        target_backend="cpp",
        initial_state=[0.165, 0.001],
        dt=0.05,
        steps=10,
        invariants=[InvariantConstraint(variable_name="pot_corridor", assertion="0.115 <= state[0] <= 0.155")],
    )
    san = sanitize_intrinsics(code, contracts=[c], variable_map={"v": 0, "g_unc": 1})
    assert "0.115" not in san.code and "0.155" not in san.code

    from invar_agent.research.evolution import recover_evolution
    rec = recover_evolution(
        {
            "variable_map": {"v": 0, "g_unc": 1},
            "state_order": ["v"],
            "contracts": [c.model_dump()],
        },
        {},
        code,
    )
    assert "g_unc" in rec["parameters"] and rec["slots"]["g_unc"] == 1


def test_symbolic_kernel_handles_indexed_if_selector_and_rk4_tiebreak():
    from invar_agent.research.evolution import check_evolution_reference

    code = (
        "def deriv(x, idx):\n"
        "    if idx == 0:\n"
        "        return 0.0\n"
        "    return 1.0\n"
        "def simulation_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    k1 = deriv(x, 0)\n"
        "    k2 = deriv(x + 0.5 * dt * k1, 0)\n"
        "    k3 = deriv(x + 0.5 * dt * k2, 0)\n"
        "    k4 = deriv(x + dt * k3, 0)\n"
        "    buf.data[0] = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0\n"
    )
    model = {
        "variable_map": {"x": 0},
        "state_order": ["x"],
        "structured_evidence": {"parameters": {}},
        "contracts": [{"contract_id": "c1", "dt": 0.1, "steps": 2, "initial_state": [1.0], "invariants": []}],
    }
    res = check_evolution_reference(model, {}, code)
    assert res["integration_scheme"] == "rk4"


def test_dynamic_loop_and_variant_writes_preserve_role_state_and_piecewise_helper():
    from invar_agent.research.slot_schema import classify_variable_roles, to_blueprint_role, validate_slot_writes
    from invar_agent.research.evolution import _SymbolicKernel

    code = (
        "def force_profile(t):\n"
        "    if t <= 240.0:\n"
        "        return 1.0 - 0.5 * t / 240.0\n"
        "    elif t <= 360.0:\n"
        "        return 0.5 + 0.5 * (t - 240.0) / 120.0\n"
        "    else:\n"
        "        return 1.0\n"
        "def simulation_step(buf, dt):\n"
        "    t = buf.data[0]\n"
        "    f = force_profile(t)\n"
        "    idx = [1, 2]\n"
        "    for i in idx:\n"
        "        buf.data[i] = buf.data[i] - dt * buf.data[3] * f\n"
        "    buf.data[0] = t + dt\n"
        "def simulation_step_semi_implicit(buf, dt):\n"
        "    t = buf.data[0]\n"
        "    buf.data[1] = buf.data[1] / (1.0 + dt * buf.data[3])\n"
        "    buf.data[0] = t + dt\n"
    )
    vm = {"time_s": 0, "delta_psi_V": 1, "delta_pH": 2, "g_leak": 3}
    v_roles = classify_variable_roles(vm, code, ())
    assert v_roles["delta_psi_V"] == ROLE_STATE
    assert v_roles["delta_pH"] == ROLE_STATE
    assert v_roles["g_leak"] == ROLE_CONTROL_PARAM
    bp_roles = {k: to_blueprint_role(v) for k, v in v_roles.items()}
    validate_slot_writes(code, vm, bp_roles)

    pw_code = (
        "def force_profile(t):\n"
        "    if t <= 240.0:\n"
        "        return 1.0 - 0.5 * t / 240.0\n"
        "    elif t <= 360.0:\n"
        "        return 0.5 + 0.5 * (t - 240.0) / 120.0\n"
        "    else:\n"
        "        return 1.0\n"
        "def simulation_step(buf, dt):\n"
        "    t = buf.data[0]\n"
        "    f = force_profile(t)\n"
        "    buf.data[1] = buf.data[1] - dt * buf.data[2] * f\n"
        "    buf.data[0] = t + dt\n"
    )
    derivs, _fn = _SymbolicKernel(pw_code, {"time_s": 0, "delta_psi_V": 1, "g_leak": 2}).derivatives(["delta_psi_V"])
    assert "delta_psi_V" in derivs and derivs["delta_psi_V"] is not None


def test_unused_swept_control_param_clamp_wrapped_noop_and_scratch_vmap_isolation():
    from invar_agent.research.state_machine import _auto_populate_variable_map
    from invar_agent.schemas.research import ResearchBlueprint

    bad_swept_code = (
        "def simulation_step(buf, dt):\n"
        "    buf.data[0] = buf.data[0] + dt * (0.145 - buf.data[0]) / 30.0\n"
    )
    vm = {"v": 0, "g_unc": 1}
    contracts = [
        VerificationContract(contract_id="c1", target_backend="cpp", initial_state=[0.165, 0.0], dt=0.1, steps=5,
                             invariants=[InvariantConstraint(variable_name="v", assertion="state[0] >= 0.0")]),
        VerificationContract(contract_id="c2", target_backend="cpp", initial_state=[0.165, 0.001], dt=0.1, steps=5,
                             invariants=[InvariantConstraint(variable_name="v", assertion="state[0] >= 0.0")]),
    ]
    with pytest.raises(ValueError, match="unused swept control parameter"):
        validate_observables(bad_swept_code, contracts, vm, {"v": "physical", "g_unc": "read_only_parameter"})

    padded_noop = (
        "def layer_2_step(buf, dt):\n"
        "    buf.data[0] = buf.data[0] * 0.99\n"
        "def layer_3_step(buf, dt):\n"
        "    layer_2_step(buf, dt)\n"
        "    if buf.data[0] < 0.0:\n"
        "        buf.data[0] = 0.0\n"
        "def simulation_step(buf, dt):\n"
        "    layer_3_step(buf, dt)\n"
    )
    bad_noop, reason_noop = is_noop_layer_wrapper(padded_noop, 3)
    assert bad_noop and "NOOP_ACCRETION_LAYER" in reason_noop

    scratch_code = (
        "def simulation_step(buf, dt):\n"
        "    r_ak = 0.1 * buf.data[0]\n"
        "    buf.data[25] = r_ak\n"
        "    buf.data[0] = buf.data[0] - dt * buf.data[1] * buf.data[25]\n"
    )
    bp = ResearchBlueprint(
        blueprint_id="bp_scratch",
        title="Scratch Isolation",
        falsifiable_hypothesis="H",
        search_queries=["ode"],
        code_template=scratch_code,
        variable_map={"v": 0, "g_unc": 1},
        contracts=contracts,
    )
    _auto_populate_variable_map(bp, scratch_code)
    assert "r_ak" not in bp.variable_map and 25 not in bp.variable_map.values()


def test_validate_variant_schemes_handles_rk4_range_loop_and_list_buffers():
    from types import SimpleNamespace
    from invar_agent.research.scientific_integrity import validate_variant_schemes

    code = (
        "def deriv(sel, x, y, k):\n"
        "    if sel == 0:\n"
        "        return -k * x\n"
        "    return k * x\n"
        "def simulation_step_rk4(buf, dt):\n"
        "    x0 = buf.data[0]\n"
        "    y0 = buf.data[1]\n"
        "    k = buf.data[2]\n"
        "    k1 = [0.0] * 2\n"
        "    k2 = [0.0] * 2\n"
        "    k3 = [0.0] * 2\n"
        "    k4 = [0.0] * 2\n"
        "    for i in range(2):\n"
        "        k1[i] = deriv(i, x0, y0, k)\n"
        "    for i in range(2):\n"
        "        k2[i] = deriv(i, x0 + 0.5 * dt * k1[0], y0 + 0.5 * dt * k1[1], k)\n"
        "    for i in range(2):\n"
        "        k3[i] = deriv(i, x0 + 0.5 * dt * k2[0], y0 + 0.5 * dt * k2[1], k)\n"
        "    for i in range(2):\n"
        "        k4[i] = deriv(i, x0 + dt * k3[0], y0 + dt * k3[1], k)\n"
        "    buf.data[0] = x0 + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])\n"
        "    buf.data[1] = y0 + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])\n"
        "def simulation_step(buf, dt):\n"
        "    simulation_step_rk4(buf, dt)\n"
    )
    c = {"contract_id": "c1", "initial_state": [1.0, 0.0, 0.2], "dt": 0.1, "steps": 5, "invariants": []}
    assert validate_variant_schemes(code, SimpleNamespace(variable_map={"x": 0, "y": 1, "k": 2}), [c]) == {
        "simulation_step_rk4": "rk4"
    }


@pytest.mark.serial
def test_nested_nonlinear_rk4_piecewise_exp_completes_under_one_second():
    import time
    from types import SimpleNamespace
    from invar_agent.research.scientific_integrity import validate_variant_schemes

    code = (
        "import math\n"
        "A = 1.0\n"
        "def force(t):\n"
        "    if t <= 10.0:\n"
        "        return math.exp(-t / 5.0)\n"
        "    return 1.0 - 0.5 * math.exp(-(t - 10.0) / 5.0)\n"
        "def rate(p, h, f, g):\n"
        "    dp = p + 0.06 * h\n"
        "    gate = max(0.0, min(1.0, (dp - 0.1) / 0.05))\n"
        "    return A * f * gate * (math.exp(min(10.0, max(-10.0, dp))) - g)\n"
        "def deriv(sel, p, h, a, d, n, r, t, g):\n"
        "    f = force(t)\n"
        "    v = rate(p, h, f, g)\n"
        "    if sel == 0:\n"
        "        return v - g * p\n"
        "    if sel == 1:\n"
        "        return 0.5 * v - 0.1 * h\n"
        "    if sel == 2:\n"
        "        return v - 0.2 * a\n"
        "    if sel == 3:\n"
        "        return -v + 0.2 * a\n"
        "    if sel == 4:\n"
        "        return f - v - 0.1 * n\n"
        "    return max(0.0, dp := (p + 0.06 * h)) * 0.01 - 0.5 * r\n"
    ).replace("max(0.0, dp := (p + 0.06 * h))", "max(0.0, p + 0.06 * h)")
    code += (
        "def simulation_step_rk4(buf, dt):\n"
        "    t0 = buf.data[6]\n"
        "    g = buf.data[7]\n"
        "    p0, h0, a0, d0, n0, r0 = buf.data[0], buf.data[1], buf.data[2], buf.data[3], buf.data[4], buf.data[5]\n"
        "    k1 = [0.0] * 6\n"
        "    k2 = [0.0] * 6\n"
        "    k3 = [0.0] * 6\n"
        "    k4 = [0.0] * 6\n"
        "    for i in range(6):\n"
        "        k1[i] = deriv(i, p0, h0, a0, d0, n0, r0, t0, g)\n"
        "    for i in range(6):\n"
        "        k2[i] = deriv(i, p0 + 0.5*dt*k1[0], h0 + 0.5*dt*k1[1], a0 + 0.5*dt*k1[2], d0 + 0.5*dt*k1[3], n0 + 0.5*dt*k1[4], r0 + 0.5*dt*k1[5], t0 + 0.5*dt, g)\n"
        "    for i in range(6):\n"
        "        k3[i] = deriv(i, p0 + 0.5*dt*k2[0], h0 + 0.5*dt*k2[1], a0 + 0.5*dt*k2[2], d0 + 0.5*dt*k2[3], n0 + 0.5*dt*k2[4], r0 + 0.5*dt*k2[5], t0 + 0.5*dt, g)\n"
        "    for i in range(6):\n"
        "        k4[i] = deriv(i, p0 + dt*k3[0], h0 + dt*k3[1], a0 + dt*k3[2], d0 + dt*k3[3], n0 + dt*k3[4], r0 + dt*k3[5], t0 + dt, g)\n"
        "    buf.data[0] = p0 + dt / 6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])\n"
        "    buf.data[1] = h0 + dt / 6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])\n"
        "    buf.data[2] = a0 + dt / 6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])\n"
        "    buf.data[3] = d0 + dt / 6.0 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3])\n"
        "    buf.data[4] = n0 + dt / 6.0 * (k1[4] + 2.0*k2[4] + 2.0*k3[4] + k4[4])\n"
        "    buf.data[5] = r0 + dt / 6.0 * (k1[5] + 2.0*k2[5] + 2.0*k3[5] + k4[5])\n"
        "    buf.data[6] = t0 + dt\n"
        "def simulation_step(buf, dt):\n"
        "    simulation_step_rk4(buf, dt)\n"
    )
    vm = {"p": 0, "h": 1, "a": 2, "d": 3, "n": 4, "r": 5, "t": 6, "g": 7}
    c = {"contract_id": "c1", "initial_state": [0.16, 0.4, 2.0, 1.0, 1.5, 0.02, 0.0, 0.05], "dt": 0.05, "steps": 3, "invariants": []}
    t0 = time.perf_counter()
    res = validate_variant_schemes(code, SimpleNamespace(variable_map=vm), [c])
    elapsed = time.perf_counter() - t0
    assert res == {"simulation_step_rk4": "rk4"}
    assert elapsed < 3.0


def test_semi_implicit_patankar_builder_ingest_redundant_noop_and_prompt_equations():
    from invar_agent.research.evolution import check_evolution_reference
    from invar_agent.research.extractor import QuantitativeInsightExtractor
    from invar_agent.runtime.research_reference import ReferenceError

    code_patankar = (
        "def simulation_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    y = buf.data[1]\n"
        "    k = buf.data[2]\n"
        "    buf.data[0] = (x + dt * (k * y)) / (1.0 + dt * k)\n"
        "    buf.data[1] = (y + dt * (k * x)) / (1.0 + dt * k)\n"
    )
    bp_pat = {
        "variable_map": {"x": 0, "y": 1, "k": 2},
        "state_order": ["x", "y"],
        "structured_evidence": {"parameters": {"k": {"si_value": 0.5, "si_unit": "1/s"}}},
        "contracts": [{"contract_id": "c1", "initial_state": [2.0, 1.0, 0.5], "dt": 0.05, "steps": 4, "invariants": []}],
    }
    rec = check_evolution_reference(bp_pat, {}, code_patankar)
    assert rec["integration_scheme"] == "semi_implicit"

    bad_patankar = (
        "def simulation_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    y = buf.data[1]\n"
        "    k = buf.data[2]\n"
        "    buf.data[0] = (x + dt * (2.0 * k * y)) / (1.0 + dt * k)\n"
        "    buf.data[1] = (y + dt * (k * x)) / (1.0 + dt * k)\n"
    )
    bad_patankar_explicit_rhs = {
        **bp_pat,
        "falsifiable_hypothesis": "dx/dt = k*y - k*x; dy/dt = k*x - k*y; k = 0.5 1/s",
    }
    with pytest.raises(ReferenceError):
        check_evolution_reference(bad_patankar_explicit_rhs, {}, bad_patankar)

    redundant_noop = (
        "def layer_1_step(buf, dt):\n"
        "    buf.data[0] = buf.data[0] + dt\n"
        "    buf.data[1] = 2.0 * buf.data[0]\n"
        "def layer_2_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
        "    buf.data[1] = 2.0 * buf.data[0]\n"
        "def simulation_step(buf, dt):\n"
        "    layer_2_step(buf, dt)\n"
    )
    bad, reason = is_noop_layer_wrapper(redundant_noop, 2)
    assert bad and "NOOP_ACCRETION_LAYER" in reason

    ex = QuantitativeInsightExtractor(use_registry=False)
    ev = ex.extract("Enforce A + B + C = 12.0 mM and dV/dt = (I_a - I_b) / C_m with C_m = 1 uF/cm^2.")
    assert len(ev.equations) >= 2

    from invar_agent.research.slot_schema import validate_contract_slots
    vm_pos = {"atp": 3, "adp": 4}
    c_pos = [
        {"contract_id": "c1", "initial_state": [0, 0, 0, 1.0, 1.0], "dt": 0.1, "steps": 2,
         "invariants": [{"variable_name": "atp_nonnegative", "assertion": "state[3] >= 0.0"}]},
        {"contract_id": "c2", "initial_state": [0, 0, 0, 1.0, 1.0], "dt": 0.1, "steps": 2,
         "invariants": [{"variable_name": "semi_implicit_atp_positivity", "assertion": "state[3] >= 0.0"}]},
    ]
    validate_contract_slots(c_pos, variable_map=vm_pos)
    c_bad = c_pos + [
        {"contract_id": "c3", "initial_state": [0, 0, 0, 1.0, 1.0], "dt": 0.1, "steps": 2,
         "invariants": [{"variable_name": "ros_nonnegative", "assertion": "state[3] >= 0.0"}]},
    ]
    with pytest.raises(ValueError, match="SLOT_SCHEMA_ERROR"):
        validate_contract_slots(c_bad, variable_map=vm_pos)


@pytest.mark.asyncio
async def test_dynamic_offset_slot_evolution_and_noop_accretion_direct_refine(tmp_path):
    from invar_agent.research.evolution import check_evolution_reference

    code = (
        "def layer_1_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    k = buf.data[1]\n"
        "    o = 10\n"
        "    buf.data[o] = -k * x\n"
        "    buf.data[o + 1] = 2.0 * x\n"
        "    r = buf.data[10]\n"
        "    buf.data[0] = (x + dt * r) / (1.0 + dt * 0.5)\n"
        "\ndef simulation_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
    )
    bp = {
        "blueprint_id": "dyn_off",
        "title": "Dynamic Scratch Offset Test",
        "falsifiable_hypothesis": "x stays nonnegative",
        "variable_map": {"x": 0, "k": 1, "scratch_r": 10, "scratch_y": 11},
        "contracts": [
            {
                "contract_id": "c1",
                "target_backend": "cpp",
                "initial_state": [1.0, 0.2] + [0.0] * 12,
                "dt": 0.05,
                "steps": 4,
                "invariants": [{"variable_name": "x_nonneg", "assertion": "state[0] >= 0.0"}],
            }
        ],
    }
    rec = check_evolution_reference(bp, {}, code)
    assert rec["integration_scheme"] == "semi_implicit"
    assert "scratch_r" not in rec["parameters"]

    fast_called = []
    planner_called = []

    async def fast_llm(prompt):
        fast_called.append(prompt)
        # Simulate Tier-1 returning an unchanged no-op wrapper -> must escalate to Tier-2 planner_llm
        return json.dumps({
            "code": (
                "def layer_1_step(buf, dt):\n"
                "    buf.data[0] = buf.data[0] * 0.99\n"
                "\ndef layer_2_step(buf, dt):\n"
                "    layer_1_step(buf, dt)\n"
                "\ndef simulation_step(buf, dt):\n"
                "    layer_2_step(buf, dt)\n"
            )
        })

    async def planner_llm(prompt):
        planner_called.append(prompt)
        return json.dumps({
            "code": (
                "def layer_1_step(buf, dt):\n"
                "    buf.data[0] = buf.data[0] * 0.99\n"
                "\ndef layer_2_step(buf, dt):\n"
                "    layer_1_step(buf, dt)\n"
                "    buf.data[2] = buf.data[0] * buf.data[1]\n"
                "\ndef simulation_step(buf, dt):\n"
                "    layer_2_step(buf, dt)\n"
            )
        })

    sm = ResearchStateMachine(tmp_path / "res_noop", llm_fast=fast_llm, llm_planner=planner_llm)
    c2 = VerificationContract(
        contract_id="c2_ext",
        target_backend="cpp",
        initial_state=[1.0, 2.0, 0.0],
        dt=0.1,
        steps=5,
        invariants=[InvariantConstraint(variable_name="scaled", assertion="state[2] >= 0.0")],
    )
    out = await sm._heal_and_refine(
        stage=2,
        round_idx=1,
        objective="compute scaled metric",
        candidate_code=(
            "def layer_1_step(buf, dt):\n"
            "    buf.data[0] = buf.data[0] * 0.99\n"
            "\ndef layer_2_step(buf, dt):\n"
            "    layer_1_step(buf, dt)\n"
            "\ndef simulation_step(buf, dt):\n"
            "    layer_2_step(buf, dt)\n"
        ),
        contracts=[c2],
        failure="PreWrite_Guard: NOOP_ACCRETION_LAYER: layer_2_step is a bare forwarding wrapper",
        attempt=1,
        max_attempts=3,
        pinned_functions=["layer_1_step"],
    )
    assert fast_called and planner_called
    assert out["applied_ops"] == ["STRUCTURAL_API_REPAIR"]
    assert "buf.data[2] = buf.data[0] * buf.data[1]" in out["code"]


@pytest.mark.asyncio
async def test_algebraic_complement_midpoint_evolution_and_variant_residual_state_mask(tmp_path):
    from invar_agent.research.evolution import check_evolution_reference

    code = (
        "TOTAL = 10.0\n"
        "def layer_1_step(buf, dt):\n"
        "    a = buf.data[0]\n"
        "    b = buf.data[1]\n"
        "    k = buf.data[2]\n"
        "    rhs1 = -0.1 * a + 0.05 * b - 0.01 * a * a\n"
        "    a_m = a + 0.5 * dt * rhs1\n"
        "    b_m = b + 0.5 * dt * (-rhs1)\n"
        "    rhs2 = -0.1 * a_m + 0.05 * b_m - 0.01 * a_m * a_m\n"
        "    buf.data[0] = a + dt * rhs2\n"
        "    buf.data[1] = TOTAL - buf.data[0]\n"
        "\ndef layer_2_step(buf, dt):\n"
        "    a0 = buf.data[0]\n"
        "    b0 = buf.data[1]\n"
        "    layer_1_step(buf, dt)\n"
        "    buf.data[3] = abs(buf.data[0] - a0) + abs(buf.data[1] - b0)\n"
        "\ndef simulation_step(buf, dt):\n"
        "    layer_2_step(buf, dt)\n"
    )
    bp = {
        "blueprint_id": "alg_comp_mid",
        "title": "Algebraic Complement Midpoint Test",
        "falsifiable_hypothesis": "da/dt = -0.1*a + 0.05*b - 0.01*a*a; db/dt = 0.1*a - 0.05*b + 0.01*a*a",
        "variable_map": {"a": 0, "b": 1, "k_ctrl": 2, "step_residual": 3},
        "contracts": [
            {
                "contract_id": "c1",
                "target_backend": "multi_variant",
                "initial_state": [7.0, 3.0, 1.0, 0.0],
                "dt": 0.1,
                "steps": 4,
                "invariants": [{"variable_name": "a_nonneg", "assertion": "state[0] >= 0.0"}],
            }
        ],
    }
    rec = check_evolution_reference(bp, {}, code)
    assert rec["integration_scheme"] == "midpoint"

    sm = ResearchStateMachine(tmp_path / "res_var")
    class _FakeBridge:
        def run(self, *args, **kwargs):
            # Variant matches ROLE_STATE slots [0, 1] closely, differs on auxiliary slot [3]
            return {"state": [6.999, 3.001, 1.0, 99.0], "execution_time_ms": 1.0, "finite": True}
    sm.bridge = _FakeBridge()
    c = VerificationContract(
        contract_id="c1",
        target_backend="multi_variant",
        initial_state=[7.0, 3.0, 1.0, 0.0],
        dt=0.1,
        steps=4,
        invariants=[InvariantConstraint(variable_name="a_nonneg", assertion="state[0] >= 0.0")],
    )
    vars_out = await sm._run_variants(
        "", ["simulation_step", "simulation_step_alt"], [7.0, 3.0, 1.0, 0.0], c,
        {"state": [6.999, 3.001, 1.0, 0.0], "execution_time_ms": 1.0, "finite": True},
        variable_map={"a": 0, "b": 1, "k_ctrl": 2, "step_residual": 3},
    )
    assert vars_out["alt"]["residual"] < 1e-9


def test_auto_populate_variable_map_no_state_lock_and_planner_slot_roles_preserve():
    from invar_agent.research.planner import parse_blueprint
    from invar_agent.research.state_machine import _auto_populate_variable_map

    bp = parse_blueprint({
        "blueprint_id": "slot_role_lock_test",
        "title": "Slot role sanitization and non-latching",
        "falsifiable_hypothesis": "x evolves and reaction_rate tracks flux",
        "search_queries": ["reaction kinetics"],
        "code_template": "def simulation_step(buf, dt):\n    buf.data[2] = buf.data[0] * buf.data[1]\n",
        "variable_map": {"x_conc": 0, "g_leak": 1, "reaction_rate": 2, "y_conc": 3},
        "slot_roles": {
            "x_conc": "evolving_state",
            "g_leak": "read_only_parameter",
            "reaction_rate": "read_only_parameter",
        },
        "contracts": [
            {
                "contract_id": "c1",
                "target_backend": "cpp",
                "initial_state": [5.0, 0.01, 0.0, 2.0],
                "dt": 0.1,
                "steps": 10,
                "invariants": [{"variable_name": "x_pos", "assertion": "state[0] >= 0.0"}],
            }
        ],
    })
    assert bp.slot_roles["reaction_rate"] == "diagnostic"
    assert bp.slot_roles["g_leak"] == "read_only_parameter"

    # Simulate broken attempt omitting writes to x_conc and y_conc
    broken_stub = (
        "def simulation_step(buf, dt):\n"
        "    buf.data[2] = buf.data[0] * buf.data[1]\n"
    )
    _auto_populate_variable_map(bp, broken_stub)
    assert bp.slot_roles["x_conc"] == "evolving_state"
    assert bp.slot_roles.get("y_conc") != "read_only_parameter"


def test_extractor_ignores_inequality_thresholds_and_bare_prose_symbol():
    from invar_agent.research.extractor import extract_parameters_from_text

    prompt = (
        "Simulate at T = 310.15 K while peak reperfusion ROS concentration below 2.0 uM "
        "and membrane potential exceeding 150 mV are monitored against threshold <= 0.5 mM."
    )
    params = extract_parameters_from_text(prompt, source="user_prompt")
    names = {p.name for p in params}
    symbols = {p.symbol.lower() for p in params if p.symbol}
    assert "t" in names
    assert all("2.0" not in str(p.raw_value) and abs(p.raw_value - 2.0) > 1e-9 for p in params)
    assert all(abs(p.raw_value - 150.0) > 1e-9 for p in params)
    assert "concentration" not in symbols
    assert "threshold" not in symbols


def test_validate_variant_schemes_ignores_running_extrema_diagnostic():
    from invar_agent.research.scientific_integrity import validate_variant_schemes
    from types import SimpleNamespace

    code = (
        "def rhs_x(x, k):\n"
        "    return -k * x + 0.05 * x * x\n"
        "def simulation_step_rk4(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    k = buf.data[1]\n"
        "    k1 = rhs_x(x, k)\n"
        "    k2 = rhs_x(x + 0.5 * dt * k1, k)\n"
        "    k3 = rhs_x(x + 0.5 * dt * k2, k)\n"
        "    k4 = rhs_x(x + dt * k3, k)\n"
        "    x_new = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)\n"
        "    buf.data[0] = x_new\n"
        "    if x_new > buf.data[2]:\n"
        "        buf.data[2] = x_new\n"
        "def simulation_step(buf, dt):\n"
        "    simulation_step_rk4(buf, dt)\n"
    )
    bp = SimpleNamespace(
        variable_map={"x_conc": 0, "k_decay": 1, "peak_x_conc": 2},
        slot_roles={"x_conc": "evolving_state", "k_decay": "read_only_parameter", "peak_x_conc": "diagnostic"},
    )
    contracts = [
        {
            "contract_id": "c1",
            "initial_state": [1.0, 0.2, 0.0],
            "dt": 0.05,
            "steps": 5,
            "invariants": [{"variable_name": "peak_x_conc", "assertion": "state[2] <= 5.0"}],
        }
    ]
    res = validate_variant_schemes(code, bp, contracts)
    assert res == {"simulation_step_rk4": "rk4"}


def test_tier0_does_not_clamp_existential_trace_min_dip():
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics
    from invar_agent.runtime.research_reference import _safe_namespace
    from types import SimpleNamespace

    code = (
        "def simulation_step(buf, dt):\n"
        "    t = buf.data[0] + dt\n"
        "    buf.data[0] = t\n"
        "    if t < 5.0:\n"
        "        buf.data[1] = 0.05\n"
        "    else:\n"
        "        buf.data[1] = 1.0\n"
    )
    contracts = [
        {
            "contract_id": "c_dip_and_recover",
            "initial_state": [0.0, 1.0],
            "dt": 1.0,
            "steps": 10,
            "invariants": [
                {"variable_name": "signal_dip", "assertion": "trace_min[1] <= 0.1"},
                {"variable_name": "signal_recover", "assertion": "state[1] >= 0.95"},
            ],
        }
    ]
    san = sanitize_intrinsics(
        code,
        contracts=contracts,
        variable_map={"time_s": 0, "signal": 1},
        slot_roles={"time_s": "evolving_state", "signal": "diagnostic"},
    )
    assert all("upper-bound slot state[1] <= 0.1" not in note for note in san.notes)
    ns = _safe_namespace(san.code)
    buf = SimpleNamespace(data=[0.0, 1.0])
    for _ in range(10):
        ns["simulation_step"](buf, 1.0)
    assert buf.data[1] >= 0.95

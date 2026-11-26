"""Unit and regression tests for the Four-Namespace Blueprint Compiler, NL-to-AST decoupling,
Tier-1 targeted AST repair convergence, and post-seal 8-gate anti-gaming preservation.
"""

from __future__ import annotations

import copy
import json

import pytest

from invar_agent.research import meaningfulness as m
from invar_agent.research.blueprint_compiler import compile_and_preflight_blueprint
from invar_agent.research.epistemic import compile_blueprint_ihcg, verify_blueprint_entailment_lock
from invar_agent.research.falsification import run_mode1_deductive_falsification
from invar_agent.research.formalization import formalize_blueprint
from invar_agent.research.planner import parse_blueprint
from invar_agent.schemas.research import FormalResearchSpec, ResearchBlueprint


def _oxphos_four_namespace_payload():
    """Realistic multi-pool OXPHOS blueprint separating typed_slots, constants, and derived_expressions."""
    typed_slots = [
        {"symbol": "atp_mol_m3", "slot_idx": 0, "si_dim": [0, -3, 0, 0, 0, 1, 0], "role": "DYNAMIC_STATE", "si_unit": "mol/m^3", "initial_value": 9.0, "bounds": [0.1, 12.0]},
        {"symbol": "adp", "slot_idx": 1, "si_dim": [0, -3, 0, 0, 0, 1, 0], "role": "DYNAMIC_STATE", "si_unit": "mol/m^3", "initial_value": 2.5, "bounds": [0.05, 12.0]},
        {"symbol": "amp", "slot_idx": 2, "si_dim": [0, -3, 0, 0, 0, 1, 0], "role": "DYNAMIC_STATE", "si_unit": "mol/m^3", "initial_value": 0.5, "bounds": [0.01, 12.0]},
        {"symbol": "delta_psi_v", "slot_idx": 3, "si_dim": [1, 2, -3, -1, 0, 0, 0], "role": "DYNAMIC_STATE", "si_unit": "V", "initial_value": 0.165, "bounds": [0.08, 0.22]},
        {"symbol": "g_leak", "slot_idx": 4, "si_dim": [-1, -4, 3, 2, 0, 0, 0], "role": "CONTROL_PARAM", "si_unit": "S/m^2", "initial_value": 5e-4, "bounds": [1e-4, 1.2e-3]},
        {"symbol": "g_unc", "slot_idx": 5, "si_dim": [-1, -4, 3, 2, 0, 0, 0], "role": "CONTROL_PARAM", "si_unit": "S/m^2", "initial_value": 8e-4, "bounds": [0.0, 2.5e-3]},
        {"symbol": "ros_uM", "slot_idx": 6, "si_dim": [0, 0, 0, 0, 0, 0, 0], "role": "DERIVED_OBSERVABLE", "si_unit": "uM", "initial_value": 0.2, "bounds": [0.0, 20.0]},
    ]
    constants = {
        "C_m": {"symbol": "C_m", "value": 0.01, "si_unit": "F/m^2", "si_dim": [-1, -4, 4, 2, 0, 0, 0], "source": "USER_PROMPT"},
        "F_far": {"symbol": "F_far", "value": 96485.33212, "si_unit": "C/mol", "si_dim": [0, 0, 1, 1, 0, -1, 0], "source": "REFERENCE_REGISTRY"},
        "R_gas": {"symbol": "R_gas", "value": 8.314462618, "si_unit": "J/(mol*K)", "si_dim": [1, 2, -2, 0, -1, -1, 0], "source": "REFERENCE_REGISTRY"},
        "T_K": {"symbol": "T_K", "value": 310.15, "si_unit": "K", "si_dim": [0, 0, 0, 0, 1, 0, 0], "source": "USER_PROMPT"},
        "POOL_AD": {"symbol": "POOL_AD", "value": 12.0, "si_unit": "mol/m^3", "si_dim": [0, -3, 0, 0, 0, 1, 0], "source": "USER_PROMPT"},
        "POOL_RED": {"symbol": "POOL_RED", "value": 3.0, "si_unit": "mol/m^3", "si_dim": [0, -3, 0, 0, 0, 1, 0], "source": "USER_PROMPT"},
    }
    derived_expressions = {
        "delta_p": {"symbol": "delta_p", "expression": "delta_psi_v + (2.303 * R_gas * T_K / F_far) * 0.5", "si_unit": "V", "si_dim": [1, 2, -3, -1, 0, 0, 0]},
        "aec_ratio": {"symbol": "aec_ratio", "expression": "(atp_mol_m3 + 0.5 * adp) / POOL_AD", "si_unit": "", "si_dim": [0, 0, 0, 0, 0, 0, 0]},
    }
    blueprint = {
        "blueprint_id": "oxphos_four_ns",
        "title": "Mitochondrial OXPHOS Mild Uncoupling Corridor",
        "falsifiable_hypothesis": "Mild uncoupling keeps peak ROS below 2.0 uM while preserving AEC >= 0.78.",
        "research_prompt": "Does mild mitochondrial uncoupling prevent ROS bursts without killing ATP?",
        "search_queries": ["mitochondrial uncoupling ROS"],
        "variable_map": {s["symbol"]: s["slot_idx"] for s in typed_slots},
        "typed_slots": typed_slots,
        "constants": constants,
        "derived_expressions": derived_expressions,
        "assumptions_ledger": [
            {
                "kind": "SI_CONSTANT",
                "symbol_or_field": "T_K",
                "value_or_expr": "310.15 K",
                "source": "LLM_DOMAIN_DEFAULT",
                "rationale": "Standard human physiological body temperature (37 C).",
            },
            {
                "kind": "OPERATING_DOMAIN",
                "symbol_or_field": "g_unc",
                "value_or_expr": "[0.0, 2.5e-3] S/m^2",
                "source": "LITERATURE",
                "rationale": "Physiological mild-to-severe protonophore conductance range.",
            },
        ],
        "code_template": (
            "def simulation_step(buf, dt):\n"
            "    buf.data[6] = 10.0 * buf.data[3] * (1.0 / (1.0 + 1000.0 * buf.data[5]))\n"
        ),
        "contracts": [
            {
                "contract_id": "c1_corridor",
                "target_backend": "cpp",
                "initial_state": [9.0, 2.5, 0.5, 0.165, 5e-4, 8e-4, 0.2],
                "dt": 0.1,
                "steps": 10,
                "invariants": [
                    {"variable_name": "ros_uM", "assertion": "state[6] < 2.0", "tolerance": 1e-6},
                    {"variable_name": "aec_bound", "assertion": "(state[0] + 0.5 * state[1]) / 12.0 >= 0.78", "tolerance": 1e-6},
                ],
            }
        ],
    }
    ihcg_pack = {
        "archetype": "ODE_PDE",
        "domain": {
            "atp_mol_m3": [0.1, 12.0],
            "adp": [0.05, 12.0],
            "amp": [0.01, 12.0],
            "delta_psi_v": [0.08, 0.22],
            "g_leak": [1e-4, 1.2e-3],
            "g_unc": [0.0, 2.5e-3],
            "ros_uM": [0.0, 20.0],
        },
        "evidence": {
            "equations": {
                "ros_uM": {
                    "expression": "10.0 * delta_p * (1.0 / (1.0 + 1000.0 * g_unc))",
                    "si_dim": [0, 0, 0, 0, 0, 0, 0],
                },
                "pool_conservation": {
                    "expression": "atp_mol_m3 + adp + amp == POOL_AD",
                    "si_dim": [0, 0, 0, 0, 0, 0, 0],
                },
            },
            "environmental_assumptions": ["g_unc >= 0.0"],
            "obligations": [
                {
                    "id": "h_ros",
                    "expression": "ros_uM < 2.0",
                    "obligation_class": "HYPOTHESIS_TARGET",
                    "prompt_quote": "prevent ROS bursts",
                },
                {
                    "id": "h_aec",
                    "expression": "aec_ratio >= 0.78",
                    "obligation_class": "HYPOTHESIS_TARGET",
                    "prompt_quote": "without killing ATP",
                },
            ],
            "quantifier_mode": "EXISTS",
            "mechanism_parameters": ["g_unc"],
        },
    }
    return blueprint, ihcg_pack


def test_four_namespace_resolution_compiles_without_missing_typed_slots():
    """Constants (C_m, F_far, R_gas, T_K, POOL_AD, POOL_RED) and derived_expressions (delta_p, aec_ratio)
    must compile cleanly without ever raising 'missing typed slots for referenced symbols'."""
    blueprint, ihcg_pack = _oxphos_four_namespace_payload()

    # Validate FormalResearchSpec schema accepts all four namespaces cleanly.
    spec = FormalResearchSpec(
        typed_slots=blueprint["typed_slots"],
        constants=blueprint["constants"],
        derived_expressions=blueprint["derived_expressions"],
        equations=ihcg_pack["evidence"]["equations"],
        sweep_domain=ihcg_pack["domain"],
        hypothesis_obligations=ihcg_pack["evidence"]["obligations"],
        environmental_assumptions=ihcg_pack["evidence"]["environmental_assumptions"],
        quantifier_mode="EXISTS",
        archetype="ODE_PDE",
        mechanism_parameters=["g_unc"],
        assumptions_ledger=blueprint["assumptions_ledger"],
    )
    assert "R_gas" in spec.constants and "delta_p" in spec.derived_expressions

    res = compile_and_preflight_blueprint(blueprint, ihcg_pack, question=blueprint["research_prompt"])
    assert res.ok, f"Unexpected preflight diagnostics: {res.error_summary()}"
    assert res.compiled_graph is not None
    res.compiled_graph.assert_sealed()
    assert len(res.compiled_graph.h_ihcg) == 64
    # Derived expressions were inlined cleanly (not required as mutable buffer slots).
    slot_names = {s.symbol for s in res.compiled_graph.slots}
    assert "delta_p" not in slot_names
    assert "aec_ratio" not in slot_names
    # Referenced dimensional constants were bound cleanly as locked CONTROL_PARAMs.
    for c_sym in ("R_gas", "T_K", "F_far", "POOL_AD"):
        assert c_sym in slot_names
        assert res.compiled_graph.omega[c_sym][0] == res.compiled_graph.omega[c_sym][1]
    # Assumptions ledger is sealed into the IHCG!
    assert len(res.compiled_graph.assumptions_ledger) >= 2


def test_casual_prose_prompt_seals_without_literal_quote_span_rejection():
    """A 1-sentence casual prompt with zero equations/units must seal cleanly into IHCG
    with valid BLAKE3 digest and zero literal quote-span rejection."""
    casual_prompt = "Does mild mitochondrial uncoupling prevent ROS bursts without killing ATP?"
    blueprint, ihcg_pack = _oxphos_four_namespace_payload()
    blueprint["research_prompt"] = casual_prompt
    # Paraphrased / non-literal quotes must NEVER block deterministic sealing.
    ihcg_pack["evidence"]["obligations"][0]["prompt_quote"] = "mild uncoupling keeps reactive oxygen species low"
    ihcg_pack["evidence"]["obligations"][1]["prompt_quote"] = "cellular ATP energy charge remains preserved"

    res = compile_and_preflight_blueprint(blueprint, ihcg_pack, question=casual_prompt)
    assert res.ok, res.error_summary()
    bp = ResearchBlueprint.model_validate(res.compiled_blueprint)
    graph = compile_blueprint_ihcg(bp)
    assert graph is not None
    graph.assert_sealed()
    assert len(graph.h_ihcg) == 64
    assert all(ob.prompt_span is None for ob in graph.obligations)
    assert len(graph.assumptions_ledger) >= 2


@pytest.mark.asyncio
async def test_tier1_targeted_repair_converges_on_malformed_spec():
    """Feed a deliberately malformed FormalResearchSpec (2 missing symbols + unclosed initial pool sum)
    into the compiler + mocked Tier-1 repair loop and confirm automatic convergence on attempt 2."""
    blueprint, valid_pack = _oxphos_four_namespace_payload()
    # Deliberately break attempt 1: remove POOL_AD from constants and reference an undeclared symbol 'mystery_k',
    # AND set initial pool sum inconsistent (9.0 + 2.5 + 9.0 != 12.0).
    broken_bp = copy.deepcopy(blueprint)
    broken_bp["contracts"][0]["initial_state"][2] = 9.0  # sum = 20.5 != 12.0
    broken_pack = copy.deepcopy(valid_pack)
    broken_pack["evidence"]["equations"]["ros_uM"]["expression"] = "mystery_k * delta_p + ghost_bias"

    # First verify Tier-0 preflight catches the undeclared symbols cleanly.
    preflight1 = compile_and_preflight_blueprint(broken_bp, broken_pack, question=broken_bp["research_prompt"])
    assert not preflight1.ok
    codes1 = {d.code for d in preflight1.diagnostics}
    assert "UNRESOLVED_SYMBOL" in codes1
    assert {"mystery_k", "ghost_bias"} <= {d.symbol for d in preflight1.diagnostics}

    # Verify Tier-0 preflight catches the unclosed initial pool sum when symbols are valid.
    preflight_pool = compile_and_preflight_blueprint(broken_bp, valid_pack, question=broken_bp["research_prompt"])
    assert not preflight_pool.ok
    assert any(d.code == "INITIAL_POOL_MISMATCH" for d in preflight_pool.diagnostics)

    # Now run the bounded Tier-1 repair loop on a blueprint with valid initial state + malformed attempt 1 pack.
    repair_prompts = []

    async def initial_caller(prompt: str) -> str:
        return json.dumps(broken_pack)

    async def tier1_repair_caller(prompt: str) -> str:
        repair_prompts.append(prompt)
        return json.dumps(valid_pack)

    repaired_bp = await formalize_blueprint(
        blueprint["research_prompt"],
        blueprint,
        initial_caller,
        repair_complete=tier1_repair_caller,
    )
    assert len(repair_prompts) == 1
    assert "UNRESOLVED_SYMBOL" in repair_prompts[0]
    assert "mystery_k" in repair_prompts[0]
    graph = compile_blueprint_ihcg(repaired_bp)
    assert graph is not None
    graph.assert_sealed()


def test_anti_gaming_preserved_post_seal_and_carnot_false_claim_routes_to_falsification():
    """After IHCG sealing:
    (a) Weakening a contract below the sealed obligation still fails Phi_lock.
    (b) A genuine false claim (Carnot eta >= 0.65 at 300/600 K) passes preflight and routes to VERIFIED_FALSIFICATION."""
    carnot_prompt = "Can a reversible heat engine working between a 300 K cold bath and a 600 K hot bath achieve at least 65% thermal efficiency?"
    names = ["Tc", "Th", "Qin", "W", "Qout", "eta"]
    energy = [1, 2, -2, 0, 0, 0, 0]
    temp = [0, 0, 0, 0, 1, 0, 0]
    dims = [temp, temp, energy, energy, energy, [0] * 7]

    carnot_bp = {
        "blueprint_id": "carnot_casual",
        "title": "Carnot Efficiency Ceiling Check",
        "falsifiable_hypothesis": "eta >= 0.65",
        "research_prompt": carnot_prompt,
        "search_queries": ["Carnot efficiency"],
        "variable_map": dict(zip(names, range(6))),
        "typed_slots": [
            {"symbol": n, "slot_idx": i, "role": "CONTROL_PARAM" if i < 3 else "DERIVED_OBSERVABLE", "si_dim": dims[i]}
            for i, n in enumerate(names)
        ],
        "code_template": (
            "def simulation_step(buf, dt):\n"
            "    buf.data[3] = buf.data[2] * (1.0 - buf.data[0] / buf.data[1])\n"
            "    buf.data[4] = buf.data[2] - buf.data[3]\n"
            "    buf.data[5] = buf.data[3] / buf.data[2]\n"
        ),
        "contracts": [
            {
                "contract_id": "c1_carnot",
                "target_backend": "cpp",
                "initial_state": [300.0, 600.0, 100.0, 0.0, 0.0, 0.0],
                "dt": 0.125,
                "steps": 1,
                "invariants": [{"variable_name": "eta", "assertion": "state[5] >= 0.65"}],
            }
        ],
    }
    carnot_pack = {
        "archetype": "ODE_PDE",
        "domain": {"Tc": [300.0, 300.0], "Th": [600.0, 600.0], "Qin": [1.0, 200.0], "W": [0.0, 200.0], "Qout": [0.0, 200.0], "eta": [0.0, 1.0]},
        "evidence": {
            "equations": {
                "W": {"expression": "Qin * (1 - Tc / Th)", "si_dim": energy},
                "Qout": {"expression": "Qin - W", "si_dim": energy},
                "eta": {"expression": "W / Qin", "si_dim": [0] * 7},
                "first_law": {"expression": "Qin == W + Qout", "si_dim": [0] * 7},
                "second_law": {"expression": "Qout / Tc >= Qin / Th", "si_dim": [0] * 7},
            },
            "environmental_assumptions": [],
            "obligations": [
                {
                    "id": "h_carnot",
                    "expression": "eta >= 0.65",
                    "obligation_class": "HYPOTHESIS_TARGET",
                    "prompt_quote": "at least 65% thermal efficiency",
                }
            ],
            "quantifier_mode": "EXISTS",
            "mechanism_parameters": [],
        },
    }

    # 1. Casual false claim passes preflight & seals cleanly (NOT rejected as vacuous/unanchored).
    preflight = compile_and_preflight_blueprint(carnot_bp, carnot_pack, question=carnot_prompt)
    assert preflight.ok, preflight.error_summary()
    bp = ResearchBlueprint.model_validate(preflight.compiled_blueprint)
    graph = compile_blueprint_ihcg(bp)
    assert graph is not None
    graph.assert_sealed()

    # 2. Attempting to weaken the contract post-seal STILL fails Phi_lock!
    weakened_bp = copy.deepcopy(preflight.compiled_blueprint)
    weakened_bp["contracts"][0]["invariants"][0]["assertion"] = "state[5] >= 0.45"
    weak_model = ResearchBlueprint.model_validate(weakened_bp)
    lock = verify_blueprint_entailment_lock(weak_model, graph)
    assert not lock.entailment_locked

    # 3. Deterministic engine certifies Mode 1 VERIFIED_FALSIFICATION on the sealed false claim!
    fidelity = m.FidelityAudit(
        passed=True,
        algebraic_equiv_passed=True,
        si_and_semantic_tag_passed=True,
        convergence_passed=True,
        physical_checks_passed=True,
        convergence_reliability_ratio=0.0,
        epsilon_disc=0.0,
        epsilon_total=0.0,
    )
    cert = run_mode1_deductive_falsification(graph, fidelity=fidelity)
    assert cert.passed
    assert cert.epistemic_verdict == m.EpistemicVerdict.VERIFIED_FALSIFICATION
    assert cert.mode == "MODE1_DEDUCTIVE_MUS"


def test_assumptions_ledger_coerces_llm_schema_drift():
    """LLM-emitted assumptions_ledger entries using statement/equation/parameter/unit keys
    must normalize cleanly into canonical FormalizationAssumption objects without 502."""
    bp, pack = _oxphos_four_namespace_payload()
    bp["assumptions_ledger"] = [
        {"id": "A1", "statement": "Carnot efficiency law", "equation": "eta = 1 - T_c / T_h"},
        {"id": "A2", "statement": "Cold bath fixed at 300 K", "parameter": "T_c", "value": 300.0, "unit": "K"},
    ]
    bp["typed_slots"][0]["description"] = "Benign display metadata"
    parsed = parse_blueprint(bp)
    assert len(parsed.assumptions_ledger) == 2
    assert parsed.assumptions_ledger[0].kind == "GOVERNING_EQUATION"
    assert parsed.assumptions_ledger[1].kind == "SI_CONSTANT"
    assert parsed.assumptions_ledger[1].symbol_or_field == "T_c"
    assert "300.0 K" in parsed.assumptions_ledger[1].value_or_expr


def test_sequential_dummy_semantic_tags_normalized_without_touching_reserved_tags():
    """Dummy sequential semantic_tag IDs (1..250) on slots when equations use semantic_tag=0
    are normalized to 0 so PyHolographicHin does not fail with spurious SI_OR_SEMANTIC_MISMATCH,
    while reserved tags (251..254) are strictly preserved."""
    bp, pack = _oxphos_four_namespace_payload()
    for i, slot in enumerate(bp["typed_slots"]):
        slot["semantic_tag"] = i + 1  # dummy sequential IDs 1..7
    res = compile_and_preflight_blueprint(bp, pack, question=bp["research_prompt"])
    assert res.ok, res.error_summary()
    assert all(s.semantic_tag == 0 for s in res.compiled_graph.slots)


def test_unreferenced_blueprint_derived_expressions_do_not_block_formalization():
    """Unreferenced helper entries in blueprint.derived_expressions (e.g. calling code_template helpers)
    must not block compilation when ihcg_pack equations do not reference them."""
    bp, pack = _oxphos_four_namespace_payload()
    bp["derived_expressions"]["unused_helper_call"] = {
        "symbol": "unused_helper_call",
        "expression": "custom_code_helper(delta_psi_v)",
        "si_dim": [0] * 7,
    }
    res = compile_and_preflight_blueprint(bp, pack, question=bp["research_prompt"])
    assert res.ok, res.error_summary()
    res.compiled_graph.assert_sealed()


def test_preflight_catches_degenerate_point_domain_on_mechanism_parameter():
    """A zero-width point domain [v, v] on a declared mechanism_parameter is caught in Tier-0 preflight
    before sealing so Tier-1 repair can widen it cleanly."""
    bp, pack = _oxphos_four_namespace_payload()
    pack["domain"]["g_unc"] = [8e-4, 8e-4]
    res = compile_and_preflight_blueprint(bp, pack, question=bp["research_prompt"])
    assert not res.ok
    assert any(d.code == "DEGENERATE_MECHANISM_DOMAIN" and d.symbol == "g_unc" for d in res.diagnostics)


def test_trajectory_max_tag_derived_even_when_slot_had_dummy_semantic_tag():
    """TRAJECTORY_MAX tag (252) must be derived cleanly even if the LLM initially placed a dummy
    sequential semantic_tag (1..250) on the observable slot."""
    bp, pack = _oxphos_four_namespace_payload()
    bp["typed_slots"][6]["semantic_tag"] = 7  # dummy sequential tag on ros_uM
    pack["evidence"]["equations"]["ros_uM"] = {
        "expression": "delta_psi_v",
        "si_dim": [1, 2, -3, -1, 0, 0, 0],
        "kind": "TRAJECTORY_MAX",
        "state_symbol": "delta_psi_v",
    }
    bp["typed_slots"][6]["si_dim"] = [1, 2, -3, -1, 0, 0, 0]
    res = compile_and_preflight_blueprint(bp, pack, question=bp["research_prompt"])
    assert res.ok, res.error_summary()
    ros_slot = next(s for s in res.compiled_graph.slots if s.symbol == "ros_uM")
    assert ros_slot.semantic_tag == m.TRAJECTORY_MAXIMUM

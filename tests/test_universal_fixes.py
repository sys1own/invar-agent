"""Regression tests for universal multi-stage, C-ABI intrinsic, de-aliasing, SHBT, extractor, query hygiene, and report fixes."""

from __future__ import annotations

import ast
import pytest

from invar_agent.healing.deterministic_ast import apply_healing_directives, sanitize_intrinsics
from invar_agent.latex_normalizer import normalize_report_latex
from invar_agent.research.accretion import merge_accretive_module
from invar_agent.research.extractor import (
    classify_registry_domains,
    extract_parameters_from_text,
    registry_evidence,
)
from invar_agent.research.planner import canonicalize_contract_dicts
from invar_agent.research.reporter import generate_comprehensive_research_report
from invar_agent.research.search import strip_internal_diagnostic_tokens
from invar_agent.research.shbt_federation import resolve_shbt_solvability
from invar_agent.schemas.research import ResearchBlueprint
from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
from invar_agent.transpiler.emitters.rust_emitter import RustEmitter
from invar_agent.transpiler.frontend import python_source_to_uast


def test_invar_sdiv_hoisted_before_top_level_constants_python_accretion_cpp_rust():
    stage1_src = (
        "import math\n\n"
        "ETA_A = 10.0 / 33.0\n"
        "ETA_D = 23.0 / 33.0\n"
        "S_SHUNT = (ETA_A / (ETA_A + ETA_D)) ** 2\n\n"
        "def layer_1_step(buf, dt):\n"
        "    buf.data[0] = S_SHUNT\n\n"
        "def simulation_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
    )
    san1 = sanitize_intrinsics(stage1_src)
    assert "_invar_sdiv" in san1.code
    # Must execute cleanly in Python (def _invar_sdiv before S_SHUNT = _invar_sdiv(...))
    ns1 = {}
    exec(compile(san1.code, "<stage1>", "exec"), ns1)
    assert abs(ns1["S_SHUNT"] - 100.0 / 1089.0) < 1e-12

    # Verify AST order: _invar_sdiv FunctionDef appears strictly before S_SHUNT Assign
    tree1 = ast.parse(san1.code)
    fn_idx = next(i for i, n in enumerate(tree1.body) if isinstance(n, ast.FunctionDef) and n.name == "_invar_sdiv")
    assign_idx = next(i for i, n in enumerate(tree1.body) if isinstance(n, ast.Assign))
    assert fn_idx < assign_idx

    # Accretion Stage 2 & Stage 3 must preserve _invar_sdiv above module constants
    pinned1 = san1.code.replace("def simulation_step(buf, dt):\n    layer_1_step(buf, dt)\n", "")
    stage2_cand = (
        "def layer_2_step(buf, dt):\n"
        "    buf.data[1] = buf.data[0] * 2.0\n\n"
        "def simulation_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
        "    layer_2_step(buf, dt)\n"
    )
    merged2 = merge_accretive_module(pinned1, stage2_cand)
    ns2 = {}
    exec(compile(merged2.code, "<stage2>", "exec"), ns2)
    assert abs(ns2["S_SHUNT"] - 100.0 / 1089.0) < 1e-12

    # INLINE_INTRINSIC healing on a broken module with _invar_sdiv defined after Assign
    broken = "import math\nX = _invar_sdiv(1.0, 2.0)\ndef _invar_sdiv(a, b):\n    return a / b\ndef simulation_step(buf, dt):\n    buf.data[0] = X\n"
    healed = apply_healing_directives(broken, [{"op": "INLINE_INTRINSIC", "name": "_invar_sdiv"}])
    ns_h = {}
    exec(compile(healed.code, "<healed>", "exec"), ns_h)
    assert ns_h["X"] == 0.5

    # C++20 emitter emits _invar_sdiv in prelude BEFORE static const double S_SHUNT and without duplicates
    uast = python_source_to_uast(san1.code)
    cpp = CppEmitter().emit_module(uast)
    assert cpp.index("constexpr inline double _invar_sdiv(") < cpp.index("static const double S_SHUNT")
    assert cpp.count("double _invar_sdiv(") == 1

    # Rust emitter also emits _invar_sdiv without duplicates
    rs_uast = python_source_to_uast(
        "def f(a, b):\n    return _invar_sdiv(a, b)\n"
        "def _invar_sdiv(a, b):\n    return a / b\n"
    )
    rs = RustEmitter().emit_module(rs_uast)
    assert rs.count("fn _invar_sdiv(") == 1


def test_multi_contract_disjoint_state_dealiasing():
    contracts = [
        {
            "contract_id": "c1_gamow_ledger",
            "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 999054.0],
            "dt": 0.01,
            "steps": 10,
            "invariants": [
                {"variable_name": "gamow_enhancement", "assertion": "state[5] >= 1.0", "tolerance": 1e-9},
                {"variable_name": "P_gen_W", "assertion": "abs(state[6] - 999054.0) <= 1e-3", "tolerance": 1e-3},
            ],
        },
        {
            "contract_id": "c3_stinespring_shunt",
            "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0, 10.0 / 33.0, 23.0 / 33.0],
            "dt": 0.01,
            "steps": 10,
            "invariants": [
                {"variable_name": "eta_A", "assertion": "abs(state[5] - 10.0/33.0) <= 1e-12", "tolerance": 1e-12},
                {"variable_name": "eta_D", "assertion": "abs(state[6] - 23.0/33.0) <= 1e-12", "tolerance": 1e-12},
            ],
        },
    ]
    canon, vm, notes = canonicalize_contract_dicts(contracts)
    c1_slots = canon[0]["invariants"]
    c3_slots = canon[1]["invariants"]
    assert "state[5]" in c1_slots[0]["assertion"]
    assert "state[6]" in c1_slots[1]["assertion"]
    # c3's eta_A and eta_D must be remapped away from state[5] and state[6]!
    assert "state[5]" not in c3_slots[0]["assertion"]
    assert "state[6]" not in c3_slots[1]["assertion"]
    assert vm["eta_A"] != vm["gamow_enhancement"]
    assert vm["eta_D"] != vm["P_gen_W"]


def test_shbt_resolver_activates_all_four_repos_without_acronyms():
    prompt = (
        "Connect standard continuum physics (Gamow WKB, TEG, Fourier-Kapitza thermal resistance, "
        "Landauer floor at 4.2 K) with periodic Floquet-Adler-Wiser dynamic dielectric screening "
        "(U_eff = 352.48 eV) across a 1,800-module solid-state nickel-hydrogen lattice-assisted "
        "nuclear array (555.03 W net DC per cell) to service a continuous 906.00 kW holographic "
        "boundary Landauer erasure debt while maintaining a Kapitza temperature drop of 3.546 K "
        "and at least 11.79 K superconducting quench headroom below MgB2; suppress 10.9989 nW "
        "raw erasure load down to <= 1.01 nW sensible cryostat heat by shunting erasure entropy "
        "across an open bipartite Stinespring boundary register partitioned into a 10:23 "
        "visible-to-dark capacity ratio under symplectic Heegaard-Floer boundary relabeling."
    )
    res = resolve_shbt_solvability(prompt)
    assert res.should_activate is True
    repos = {r.repo for r in res.matched_repos}
    assert {"sys1own/shbt-precision", "sys1own/shbt-cf", "sys1own/shbt-recon", "sys1own/shbt-swarm"}.issubset(repos)
    assert "sys1own/shbt-warp" not in repos

    warp_res = resolve_shbt_solvability("Simulate SHBT-MAR-312X Alcubierre warp drive metric and housekeeping surplus")
    assert warp_res.should_activate is True
    assert "sys1own/shbt-warp" in {r.repo for r in warp_res.matched_repos}

    assert resolve_shbt_solvability("Simulate modified Patankar Oregonator stiff chemical kinetics in SI units").should_activate is False
    assert resolve_shbt_solvability("Verify Cayley LLG macrospin Bloch sphere norm conservation").should_activate is False
    assert resolve_shbt_solvability("Simulate a 1D damped harmonic oscillator in SI units").should_activate is False


def test_extractor_filters_cryo_tps_and_stopword_params():
    cryo_prompt = (
        "Couple Fourier-Kapitza interfacial thermal boundary resistance across a 4.2 K loop "
        "to service a continuous 906.00 kW Landauer debt with at least 11.79 K quench headroom "
        "and 10:23 Stinespring partitioning."
    )
    domains = classify_registry_domains(cryo_prompt)
    assert "materials_thermophysics" not in domains
    reg_params = registry_evidence(cryo_prompt)[0]
    names_reg = {p.name for p in reg_params}
    assert "pica_density" not in names_reg
    assert "tungsten_density" not in names_reg

    extracted = extract_parameters_from_text(cryo_prompt, source="user_prompt")
    names = {p.name for p in extracted}
    assert not (names & {"10", "at", "least", "continuous", "system_physics"})


def test_search_query_hygiene_and_report_formatting():
    dirty = "CABI fallback zero-step integration PreWrite_Guard state[5] buf.data _invar_sdiv Landauer erasure"
    cleaned = strip_internal_diagnostic_tokens(dirty)
    assert "cabi" not in cleaned.lower()
    assert "fallback" not in cleaned.lower()
    assert "zero-step" not in cleaned.lower()
    assert "state[5]" not in cleaned
    assert "_invar_sdiv" not in cleaned
    assert "Landauer erasure" in cleaned

    bp = ResearchBlueprint(
        blueprint_id="test_rep",
        title="Test Report",
        falsifiable_hypothesis="H0",
        search_queries=["landauer erasure"],
        code_template="def simulation_step(buf, dt):\n    pass\n",
        variable_map={"S_SHUNT": 0, "S_shunt": 0, "MgB2_margin_K": 1, "P_gen_W": 2},
        contracts=[{
            "contract_id": "c1",
            "target_backend": "cpp",
            "initial_state": [0.0, 0.0, 0.0],
            "dt": 0.01,
            "steps": 1,
            "invariants": [{"variable_name": "S_shunt", "assertion": "state[0] >= 0.0", "tolerance": 1e-9}],
        }],
    )
    tel = {"obs_S_SHUNT": 0.0, "obs_S_shunt": 0.0918274, "obs_MgB2_margin_K": 11.79, "obs_P_gen_W": 999054.0}
    rep = generate_comprehensive_research_report(bp, tel, [])
    assert "| W |" in rep
    assert "| K |" in rep
    # Only one S_shunt row (deduplicated)
    assert rep.lower().count("s shunt") == 1

    # Prose equation title in table must NOT be wrapped in $...$
    md = "| Equation | Meaning |\n| --- | --- |\n| Topological Stinespring ledger partitioning | Split |\n"
    norm = normalize_report_latex(md)
    assert "$Topological" not in norm

    # Code assertions and dict counterexamples in table notes must NEVER be wrapped in $...$
    note_md = (
        "| Stage | Status | Bundle hash | Note |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| 1 | FAILED | `deadbeef` | — FALSIFIED: abs(state[6] - 999054.0) <= 1.0 observed {'state[6]': 0.535} |\n"
    )
    note_norm = normalize_report_latex(note_md)
    assert "$abs(state[6]" not in note_norm
    assert "${'state[6]'" not in note_norm


def test_multi_contract_slot_decollision_on_low_and_partition_slots():
    contracts = [
        {
            "contract_id": "c1_baseline_ledger",
            "initial_state": [1800.0, 10.0 / 33.0, 23.0 / 33.0, 0.0, 0.0, 0.0, 999054.0],
            "dt": 0.01,
            "steps": 5,
            "invariants": [
                {"variable_name": "eta_partition", "assertion": "abs(state[1] + state[2] - 1.0) <= 1e-12", "tolerance": 1e-12},
                {"variable_name": "P_array_check", "assertion": "abs(state[6] - 999054.0) <= 1.0", "tolerance": 1.0},
            ],
        },
        {
            "contract_id": "c3_stinespring_shunt_audit",
            "initial_state": [0.0, 0.0, 1.01e-9, 0.0, 0.0, 0.0, 0.535385],
            "dt": 0.01,
            "steps": 5,
            "invariants": [
                {"variable_name": "visible_power_bound", "assertion": "state[2] <= 1.01e-9", "tolerance": 1e-12},
                {"variable_name": "eta_TEG", "assertion": "state[6] >= 0.0 and state[6] <= 1.0", "tolerance": 1e-9},
            ],
        },
    ]
    vm_in = {"eta_partition": 1, "P_array_check": 6, "visible_power_bound": 2, "eta_TEG": 6}
    canon, vm, _ = canonicalize_contract_dicts(contracts, vm_in)
    c1_inv = canon[0]["invariants"]
    c3_inv = canon[1]["invariants"]
    assert "state[1]" in c1_inv[0]["assertion"] and "state[2]" in c1_inv[0]["assertion"]
    assert "state[6]" in c1_inv[1]["assertion"]
    # c3's visible_power_bound and eta_TEG must be moved off state[2] and state[6]
    assert "state[2]" not in c3_inv[0]["assertion"]
    assert "state[6]" not in c3_inv[1]["assertion"]
    assert vm["visible_power_bound"] != 2
    assert vm["eta_TEG"] != 6
    assert canon[1]["initial_state"][vm["eta_TEG"]] == pytest.approx(0.535385)


def test_tier0_conditions_abs_target_sum_partition_and_compound_invariants():
    kernel = (
        "def simulation_step(buf, dt):\n"
        "    buf.data[1] = 0.25\n"
        "    buf.data[2] = 0.25\n"
        "    buf.data[4] = 5.0\n"
        "    buf.data[5] = 3.0\n"
        "    buf.data[6] = 0.535\n"
        "    buf.data[7] = -2.0\n"
        "    buf.data[17] = 1.5\n"
        "    buf.data[18] = 0.0\n"
    )
    contracts = [
        {
            "contract_id": "c1_test",
            "initial_state": [0.0] * 20,
            "dt": 0.01,
            "steps": 5,
            "invariants": [
                {"variable_name": "eta_partition", "assertion": "abs(state[1] + state[2] - 1.0) <= 1e-12", "tolerance": 1e-12},
                {"variable_name": "P_array_check", "assertion": "abs(state[6] - 999054.0) <= 1.0", "tolerance": 1.0},
                {"variable_name": "shunt_S", "assertion": "abs(state[18] - 100.0/1089.0) <= 1e-12", "tolerance": 1e-12},
                {"variable_name": "parity", "assertion": "abs(state[4] - state[5]) <= 1e-9", "tolerance": 1e-9},
                {"variable_name": "unit_band", "assertion": "state[17] >= 0.0 and state[17] <= 1.0", "tolerance": 1e-9},
                {"variable_name": "finite_lb", "assertion": "math.isfinite(state[7]) and state[7] >= 1.0", "tolerance": 1e-9},
            ],
        }
    ]
    san = sanitize_intrinsics(kernel, contracts=contracts)
    # ARCH-001: sanitize_intrinsics does NOT inject invariant-forcing clamps.
    assert san.changed is False
    san2 = sanitize_intrinsics(san.code, contracts=contracts)
    assert san2.changed is False


def test_strip_buf_size_and_protect_overwritten_input_param():
    kernel = (
        "def simulation_step(buf, dt):\n"
        "    p_cell = buf.data[1]\n"
        "    p_arr = 1800.0 * p_cell\n"
        "    buf.data[6] = p_arr\n"
        "    eta_a = 10.0 / 33.0\n"
        "    buf.data[1] = eta_a\n"
        "    if buf.size > 10:\n"
        "        buf.data[6] = 0.535385\n"
    )
    contracts = [
        {
            "contract_id": "c1_ledger",
            "initial_state": [1800.0, 555.03, 0.0, 0.0, 0.0, 0.0, 999054.0] + [0.0] * 15,
            "dt": 0.01,
            "steps": 5,
            "invariants": [
                {"variable_name": "P_array", "assertion": "abs(state[6] - 999054.0) <= 1.0", "tolerance": 1.0},
            ],
        }
    ]
    san = sanitize_intrinsics(kernel, contracts=contracts)
    # ARCH-001: sanitize_intrinsics preserves exact kernel dynamics without stripping
    # executable branches or redirecting overwritten input slots.
    assert "0.535385" in san.code
    assert san.backup_slot_map == {}


def test_extractor_filters_shbt_prompt_noise():
    noisy = (
        "1,800-module LANR array (555.03 W covering 999,054 W) service 906.00 kW debt "
        "with Kapitza drop 3.546 K and MgB2 T_op >= 11.79 K superconducting quench headroom "
        "while suppress visible sensory reset dissipation at 4.2 K <= 1.01 nW."
    )
    params = extract_parameters_from_text(noisy, source="user_prompt")
    names = {p.name for p in params}
    for bad in ("999054_w_covering", "546_k", "service", "covering", "drop", "cell", "accuracy", "visible_sensory_reset_dissipation"):
        assert bad not in names
    for p in params:
        assert not p.name[0].isdigit()
        assert p.symbol not in (">=", "<=", ">", "<", "=", "==")


def test_strict_relational_inequality_auto_init_and_sanitizer():
    from invar_agent.research.state_machine import _guard_contract_initial_state
    from invar_agent.schemas.research import VerificationContract

    kernel = (
        "def layer_1_step(buf, dt):\n"
        "    buf.data[17] = 906000.0\n\n"
        "def layer_2_step(buf, dt):\n"
        "    pass\n\n"
        "def simulation_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
        "    layer_2_step(buf, dt)\n"
    )
    vc = VerificationContract.model_validate({
        "contract_id": "c2_teg_kapitza",
        "target_backend": "cpp",
        "initial_state": [0.0] * 17 + [906000.0, 0.0, 0.0],
        "dt": 0.01,
        "steps": 5,
        "invariants": [
            {"variable_name": "array_above_debt", "assertion": "state[16] > state[17]", "tolerance": 0.0},
        ],
    })
    init, _warns = _guard_contract_initial_state(kernel, vc)
    assert init[16] == 0.0 and init[17] == 906000.0

    san = sanitize_intrinsics(kernel, contracts=[vc])
    san2 = sanitize_intrinsics(san.code, contracts=[vc])
    assert san2.changed is False

    class _Buf:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(self.data)

    ns = {}
    exec(compile(san.code, "<strict_rel>", "exec"), ns)
    b = _Buf(list(init) + [0.0] * 10)
    b.data[16] = 906000.0
    b.data[17] = 906000.0
    ns["simulation_step"](b, 0.01)
    assert b.data[16] == b.data[17]


def test_overwritten_input_slots_backed_up_and_vmap_updated():
    from invar_agent.research.state_machine import _apply_backup_slots_to_vmap

    kernel = (
        "def layer_1_step(buf, dt):\n"
        "    n_cell = buf.data[6]\n"
        "    u_eff = buf.data[7]\n"
        "    buf.data[3] = n_cell * 555.03\n"
        "    buf.data[4] = u_eff * 1.0\n"
        "    dT_Kapitza = 3.546\n"
        "    headroom = 31.254\n"
        "    buf.data[6] = dT_Kapitza\n"
        "    buf.data[7] = headroom\n\n"
        "def simulation_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
    )
    contracts = [
        {
            "contract_id": "c1_test",
            "target_backend": "cpp",
            "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1800.0, 5.64735e-17] + [0.0] * 12,
            "dt": 0.01,
            "steps": 5,
            "invariants": [
                {"variable_name": "dT_Kapitza", "assertion": "abs(state[6] - 3.546) <= 0.01", "tolerance": 0.01},
                {"variable_name": "headroom", "assertion": "state[7] >= 11.79", "tolerance": 1e-6},
            ],
        }
    ]
    bp = ResearchBlueprint(
        blueprint_id="backup_vmap_test",
        title="Backup Vmap Test",
        falsifiable_hypothesis="H",
        search_queries=["test"],
        code_template=kernel,
        variable_map={"n_cell": 6, "u_eff": 7, "dT_Kapitza": 6, "headroom": 7},
        contracts=contracts,
    )
    san = sanitize_intrinsics(kernel, contracts=contracts, all_contracts=contracts)
    # ARCH-001: name-based parameter slot backup redirection is permanently removed.
    assert san.backup_slot_map == {}
    assert bp.variable_map["dT_Kapitza"] == 6
    assert bp.variable_map["headroom"] == 7


def test_relevance_filtered_shbt_and_arxiv_generic_tokens():
    from invar_agent.research.extractor import QuantitativeInsightExtractor
    prompt = (
        "Floquet-Adler-Wiser dynamic dielectric screening U_eff = 352.48 eV across "
        "1,800-module Ni-H LANR array (555.03 W) servicing 906.00 kW Landauer debt, "
        "Kapitza 3.546 K, MgB2 11.79 K, clamping 10.9989 nW to <= 1.01 nW at 4.2 K via "
        "10:23 bipartite Stinespring boundary."
    )
    ev = QuantitativeInsightExtractor().extract(
        prompt,
        [{"source": "arxiv", "title": "ArXiv test", "summary": "with m = 1 and servicing = 906000 W"}],
    )
    assert "void_loiter_power_w" not in ev.parameters
    assert "sram_arena_bytes" not in ev.parameters
    assert "dark_os_conductor" not in ev.parameters
    assert "symplectic_shunt_ratio" in ev.parameters
    assert "floquet_screening_ev" in ev.parameters
    assert "m" not in ev.parameters
    assert "servicing" not in ev.parameters


def test_comma_numbers_and_reserved_math_idents():
    from invar_agent.research.extractor import _parse_number, extract_parameters_from_text
    assert _parse_number("999,054") == pytest.approx(999054.0)
    assert _parse_number("93,054") == pytest.approx(93054.0)
    assert _parse_number("1,800") == pytest.approx(1800.0)
    assert _parse_number("3,14") == pytest.approx(3.14)

    text = (
        "1,800-module array with P_array = 999,054 W and P_net = 93,054 W; "
        "Landauer factor ln(2) gives P_lan = 10.9989 nW and ln2 = 10.9989."
    )
    extracted = {p.name: p.si_value for p in extract_parameters_from_text(text, source="user_prompt")}
    assert extracted.get("p_array") == pytest.approx(999054.0)
    assert extracted.get("p_net") == pytest.approx(93054.0)
    assert "ln2" not in extracted
    assert "ln_2" not in extracted


def test_full_initial_state_slot_decollision():
    c1 = {
        "contract_id": "c1_baseline_closed_system",
        "target_backend": "cpp",
        "initial_state": [0.0, 0.0, 0.0, 0.0, 2.736e8, 2.736e8, 4.01937e-23, 1.0997e-14] + [0.0] * 8,
        "dt": 0.01,
        "steps": 5,
        "invariants": [
            {
                "variable_name": "landauer_floor_matches",
                "assertion": "abs(state[7] - state[6] * state[5]) / (state[6] * state[5]) < 1e-9",
                "tolerance": 1e-9,
            },
        ],
    }
    c2 = {
        "contract_id": "c2_ledger_teg_kapitza",
        "target_backend": "cpp",
        "initial_state": [0.0, 0.0, 0.0, 0.0, 999054.0] + [0.0] * 11,
        "dt": 0.01,
        "steps": 5,
        "invariants": [
            {
                "variable_name": "P_array_matches",
                "assertion": "abs(state[4] - 999054.0) < 1e-3",
                "tolerance": 1e-3,
            },
        ],
    }
    canon, vm, notes = canonicalize_contract_dicts([c1, c2])
    c2_assert = canon[1]["invariants"][0]["assertion"]
    assert "state[4]" not in c2_assert, f"c2 must de-collide state[4] against c1.initial_state[4]: {canon[1]}"
    assert canon[1]["initial_state"][4] == pytest.approx(2.736e8)
    new_slot = vm["P_array_matches"]
    assert new_slot != 4
    assert canon[1]["initial_state"][new_slot] == pytest.approx(999054.0)
    assert canon[0]["initial_state"][new_slot] == pytest.approx(999054.0)
    assert len(canon[0]["initial_state"]) == len(canon[1]["initial_state"])


def test_intra_function_overwrite_splitter_and_relative_error_product_reconcile():
    from invar_agent.research.state_machine import _eval_invariant, _guard_contract_initial_state
    from invar_agent.schemas.research import VerificationContract

    kernel = (
        "def simulation_step(buf, dt):\n"
        "    # Stage 1: closed-system baseline\n"
        "    N_dot = 2.736e8\n"
        "    E_lan = 4.01937e-23\n"
        "    P_lan = N_dot * E_lan\n"
        "    buf.data[5] = N_dot\n"
        "    buf.data[6] = E_lan\n"
        "    buf.data[7] = 1.5e-8\n"  # deliberately exceeds U and disagrees with N_dot*E_lan
        "    buf.data[12] = 0.0\n"   # deliberately 0 instead of 1.09989e-08\n"
        "    # Stage 2: coupled LANR + TEG + Kapitza (overwrites 4..7!)\n"
        "    P_array = 1800.0 * 555.03\n"
        "    P_net = P_array - 906000.0\n"
        "    dT_K = 3.546\n"
        "    buf.data[4] = P_array\n"
        "    buf.data[5] = P_net\n"
        "    buf.data[6] = dT_K\n"
        "    # Stage 3: Stinespring symplectic shunt (overwrites 4..8!)\n"
        "    eta_A = 10.0 / 33.0\n"
        "    eta_D = 23.0 / 33.0\n"
        "    S = (eta_A / (eta_A + eta_D)) ** 2\n"
        "    P_vis = 1.01e-9\n"
        "    W_cyc = 0.0\n"
        "    buf.data[4] = eta_A\n"
        "    buf.data[5] = eta_D\n"
        "    buf.data[6] = S\n"
        "    buf.data[7] = P_vis\n"
        "    buf.data[8] = W_cyc\n"
    )
    contracts = [
        VerificationContract.model_validate({
            "contract_id": "c1_baseline_closed_system",
            "target_backend": "cpp",
            "initial_state": [0.0, 0.0, 0.0, 0.0, 2.736e8, 2.736e8, 4.01937e-23, 1.0997e-14] + [0.0] * 16,
            "dt": 0.01,
            "steps": 5,
            "invariants": [
                {"variable_name": "landauer_floor_matches", "assertion": "abs(state[7] - state[6] * state[5]) / (state[6] * state[5]) < 1e-9", "tolerance": 1e-9},
                {"variable_name": "landauer_upper_bound", "assertion": "state[7] <= 1.09989e-08 * 1.01", "tolerance": 1e-12},
            ],
        }),
        VerificationContract.model_validate({
            "contract_id": "c2_ledger_teg_kapitza",
            "target_backend": "cpp",
            "initial_state": [0.0] * 24,
            "dt": 0.01,
            "steps": 5,
            "invariants": [
                {"variable_name": "P_array_matches", "assertion": "abs(state[14] - 999054.0) < 1e-3", "tolerance": 1e-3},
                {"variable_name": "P_net_nonneg", "assertion": "state[15] >= 0.0", "tolerance": 1e-9},
                {"variable_name": "kapitza_drop_positive", "assertion": "state[16] > 0.0", "tolerance": 0.0},
            ],
        }),
        VerificationContract.model_validate({
            "contract_id": "c3_stinespring_shunt_audit",
            "target_backend": "cpp",
            "initial_state": [0.0] * 24,
            "dt": 0.01,
            "steps": 5,
            "invariants": [
                {"variable_name": "P_vis_rel_match", "assertion": "abs(state[12] - 1.09989e-08) / 1.09989e-08 < 1e-6", "tolerance": 1e-6},
                {"variable_name": "partition_closes", "assertion": "abs(state[17] + state[18] - 1.0) <= 1e-12", "tolerance": 1e-12},
                {"variable_name": "shunt_ratio_locked", "assertion": "abs(state[19] - (100.0 / 1089.0)) <= 1e-9", "tolerance": 1e-9},
                {"variable_name": "visible_dissipation_below_floor", "assertion": "state[20] <= 1.01e-9", "tolerance": 1e-12},
                {"variable_name": "cyclic_work_passive", "assertion": "state[21] <= 1e-12", "tolerance": 1e-12},
            ],
        }),
    ]
    san = sanitize_intrinsics(kernel, contracts=contracts, all_contracts=contracts)
    san2 = sanitize_intrinsics(san.code, contracts=contracts, all_contracts=contracts)
    assert san2.changed is False

    class _Buf:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(self.data)

    ns = {}
    exec(compile(san.code, "<split_rel>", "exec"), ns)
    falsified = {'landauer_floor_matches', 'P_array_matches', 'kapitza_drop_positive',
                 'P_vis_rel_match', 'partition_closes', 'shunt_ratio_locked'}
    for vc in contracts:
        init, _w = _guard_contract_initial_state(san.code, vc, all_contracts=contracts)
        b = _Buf(list(init) + [0.0] * 8)
        for _ in range(5):
            ns["simulation_step"](b, 0.01)
        for inv in vc.invariants:
            assert _eval_invariant(inv.assertion, b.data, {}, {"tol": inv.tolerance}) == (inv.variable_name not in falsified), (
                f"{vc.contract_id}.{inv.variable_name} failed: {inv.assertion}, state={b.data[:24]}"
            )


def test_cpp20_all_math_functions_and_clamp_transpile_and_execute(tmp_path):
    import math
    from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
    from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
    from invar_agent.transpiler.frontend import python_source_to_uast

    src = """import math

def simulation_step(buf, dt):
    x = 0.5
    buf.data[0] = math.asinh(x)
    buf.data[1] = math.acosh(2.0)
    buf.data[2] = math.atanh(0.25)
    buf.data[3] = math.sinh(x) + math.cosh(x) + math.tanh(x)
    buf.data[4] = math.erf(x) + math.cbrt(8.0) + math.expm1(x) + math.log1p(x)
    buf.data[5] = math.gamma(4.0) + math.lgamma(4.0)
    buf.data[6] = math.degrees(math.radians(45.0))
    buf.data[7] = clamp(5.0, 2.5, 4.3)
    buf.data[8] = math.log(8.0, 2.0)
"""
    uast = python_source_to_uast(src)
    cpp = CppEmitter().emit_module(uast)
    assert "std::asinh(" in cpp
    assert "std::acosh(std::max(1.0," in cpp
    assert "std::atanh(std::clamp(" in cpp
    assert "std::tgamma(" in cpp
    assert "invar::clamp(" in cpp

    bridge = DynamicCABIBridge(tmp_path / "cabi_math")
    res = bridge.run(
        cpp_code=cpp,
        initial_state=[0.0] * 10,
        dt=0.01,
        steps=1,
        entry="execute_step",
    )
    assert res["finite"] is True
    assert res["state"][0] == pytest.approx(math.asinh(0.5))
    assert res["state"][1] == pytest.approx(math.acosh(2.0))
    assert res["state"][2] == pytest.approx(math.atanh(0.25))
    assert res["state"][4] == pytest.approx(math.erf(0.5) + 2.0 + math.expm1(0.5) + math.log1p(0.5))
    assert res["state"][5] == pytest.approx(math.gamma(4.0) + math.lgamma(4.0))
    assert res["state"][6] == pytest.approx(45.0)
    assert res["state"][7] == pytest.approx(4.3)
    assert res["state"][8] == pytest.approx(3.0)

    # INLINE_INTRINSIC safety net for math.asinh
    healed = apply_healing_directives(
        "def f(x):\n    return math.asinh(x)\n",
        [{"op": "INLINE_INTRINSIC", "name": "math.asinh"}],
    )
    assert "math.log" in healed.code and "math.sqrt" in healed.code


def test_chained_bounds_general_upper_bound_and_exact_name_bridge():
    from invar_agent.research.state_machine import (
        _auto_populate_variable_map, _eval_invariant, _guard_contract_initial_state)
    from invar_agent.schemas.research import VerificationContract

    # The metric's exact declared name grounds the bridge, not a token overlap.
    kernel = (
        "import math\n\n"
        "def simulation_step(buf, dt):\n"
        "    eta_bv = math.asinh(0.4)\n"
        "    V_cell = 4.65 + 0.01 * eta_bv\n"  # deliberately above 4.3 upper bound
        "    t = buf.data[24]\n"
        "    temp_rise = 8.4\n"
        "    buf.data[15] = V_cell\n"
        "    buf.data[18] = max(0.0, buf.data[18] + 1e-10 * dt)\n"
        "    buf.data[24] = t + dt\n"
        "    buf.data[20] = temp_rise\n"
    )
    vc = VerificationContract.model_validate({
        "contract_id": "c1_dfn_p2d_sei",
        "target_backend": "cpp",
        "initial_state": [0.0] * 24,
        "dt": 1.0,
        "steps": 5,
        "invariants": [
            {"variable_name": "voltage_bounds", "assertion": "2.5 <= state[15] <= 4.3", "tolerance": 1e-9},
            {"variable_name": "sei_monotonic", "assertion": "state[18] >= 0.0 and state[18] <= telemetry['final_max']", "tolerance": 1e-12},
            {"variable_name": "temp_rise", "assertion": "state[20] <= 15.0", "tolerance": 1e-9},
        ],
    })
    bp = ResearchBlueprint(
        blueprint_id="dfn_test",
        title="DFN Test",
        falsifiable_hypothesis="H",
        search_queries=["dfn"],
        code_template=kernel,
        variable_map={},
        contracts=[vc],
    )
    san = sanitize_intrinsics(kernel, contracts=[vc], all_contracts=[vc], variable_map=bp.variable_map)
    san2 = sanitize_intrinsics(san.code, contracts=[vc], all_contracts=[vc], variable_map=bp.variable_map)
    assert san2.changed is False
    _auto_populate_variable_map(bp, san.code)
    assert bp.variable_map.get("voltage_bounds") == 15
    assert bp.variable_map.get("temp_rise") == 20

    init, _ = _guard_contract_initial_state(san.code, vc)
    assert 2.5 <= init[15] <= 4.3
    assert init[20] <= 15.0

    class _Buf:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(self.data)

    ns = {}
    exec(compile(san.code, "<dfn_bounds>", "exec"), ns)
    b = _Buf(init + [0.0] * 8)
    for _ in range(1800):
        ns["simulation_step"](b, 1.0)
    assert b.data[20] == pytest.approx(8.4)
    assert b.data[24] == pytest.approx(1800.0)
    assert b.data[20] <= 15.0


def test_zero_cross_run_commit_bleed_and_katex_ui_and_subscripts(tmp_path):
    from pathlib import Path
    from invar_agent.research.broadcaster import ResearchEventBroadcaster, ResearchRunManager
    from invar_agent.schemas.research import VerificationContract

    bc = ResearchEventBroadcaster()
    mgr = ResearchRunManager(tmp_path, bc)

    def _bp(bid: str):
        return ResearchBlueprint(
            blueprint_id=bid,
            title=bid,
            falsifiable_hypothesis="f(x) conservation holds",
            search_queries=["test"],
            code_template=(
                "def simulation_step(buf, dt):\n"
                "    buf.data[4] = 1.0\n"
            ),
            contracts=[
                VerificationContract.model_validate({
                    "contract_id": "c1_check",
                    "target_backend": "cpp",
                    "initial_state": [0.0] * 8,
                    "dt": 0.01,
                    "steps": 2,
                    "invariants": [{"variable_name": "ok", "assertion": "abs(state[4] - 1.0) <= 1e-9", "tolerance": 1e-9}],
                })
            ],
            target_research_stages=1,
        )

    mgr.request_plan(_bp("run_one"))
    assert mgr.approve()
    uid1 = mgr.active_run_uid
    mgr._thread.join(timeout=30)

    # Simulate stale late publish from uid1 after run 2 has started
    mgr.request_plan(_bp("run_two"))
    assert mgr.approve()
    uid2 = mgr.active_run_uid
    assert uid1 != uid2
    bc.publish("accelerator_log", "commit", {"bundle_hash": "deadbeef" * 8}, run_uid=uid1)
    assert mgr.committed_bundles == [] or all(b != "deadbeef" * 8 for b in mgr.committed_bundles)
    q = bc.subscribe(run_uid=uid2)
    mgr._thread.join(timeout=30)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert events
    assert all(e.get("run_uid") == uid2 for e in events)
    assert not any(e.get("bundle_hash") == "deadbeef" * 8 for e in events)

    # UI checks: per-run log card + KaTeX output: 'html' + hidden mathml + subscript bracing
    index_html = (Path(__file__).resolve().parents[1] / "invar_agent" / "static" / "index.html").read_text(encoding="utf-8")
    assert "output: 'html'" in index_html
    assert ".katex-mathml { display: none !important; }" in index_html
    assert "research-log-${uid}" in index_html

    norm = normalize_report_latex("$R_film(t) = \\delta_SEI(t) / \\kappa_SEI$")
    assert r"R_{\mathrm{film}}" in norm
    assert r"\delta_{\mathrm{SEI}}" in norm
    assert r"\kappa_{\mathrm{SEI}}" in norm


def test_ratio_product_or_disjunction_and_witness_cpp_cabi_and_fraction_extractor(tmp_path):
    from invar_agent.research.extractor import _parse_number, extract_parameters_from_text
    from invar_agent.research.state_machine import _eval_invariant, _guard_contract_initial_state
    from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
    from invar_agent.schemas.research import VerificationContract
    from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
    from invar_agent.transpiler.frontend import python_source_to_uast

    # 1. Fraction literals parsed cleanly without truncating at '/'
    assert _parse_number("100/1089") == pytest.approx(100.0 / 1089.0, rel=1e-12)
    extracted = extract_parameters_from_text(
        "S_shunt = 100/1089, eta_A = 10/33, Delta_h_vis = 45/64 Hz.",
        source="user_prompt",
    )
    by_name = {p.name: p.si_value for p in extracted}
    assert by_name.get("s_shunt") == pytest.approx(100.0 / 1089.0, rel=1e-12)
    assert by_name.get("eta") == pytest.approx(10.0 / 33.0, rel=1e-12)
    assert by_name.get("delta_h_vis") == pytest.approx(45.0 / 64.0, rel=1e-12)

    # 2. Ratio inequality, product positivity, `or` disjunction, and uncovered witness + unassigned slots
    kernel = (
        "import math\n\n"
        "def simulation_step(buf, dt):\n"
        "    # Stage 2 violates SNR < 1: sigma_sql = 1.7147e-22, snr_sql = 4.9448e20\n"
        "    buf.data[8] = 1.7147e-22\n"
        "    buf.data[9] = 4.9448e+20\n"
        "    # Stage 3 violates >= 10 disjunction when squeezed noise dominates\n"
        "    buf.data[18] = 1.0\n"
        "    buf.data[20] = 25.0\n"
        "    buf.data[21] = 100.0\n"
        "    # Deliberately omit assigning buf.data[19] and buf.data[22]\n"
    )
    contracts = [
        VerificationContract.model_validate({
            "contract_id": "c2_sql_classical_snr",
            "target_backend": "cpp",
            "initial_state": [0.0] * 24,
            "dt": 0.01,
            "steps": 3,
            "invariants": [
                {
                    "variable_name": "classical_SNR_below_unity",
                    "assertion": "state[8] * state[9] > 0.0 and state[9] / math.sqrt(state[8]) < 1.0",
                    "tolerance": 1e-12,
                },
            ],
        }),
        VerificationContract.model_validate({
            "contract_id": "c3_tmsv_snr_recovery",
            "target_backend": "cpp",
            "initial_state": [0.0] * 24,
            "dt": 0.01,
            "steps": 3,
            "invariants": [
                {
                    "variable_name": "quantum_SNR_recovered",
                    "assertion": "state[18] / math.sqrt(state[20] + state[21]) >= 10.0 or state[18] / math.sqrt(state[20]) >= 10.0",
                    "tolerance": 1e-12,
                },
                {
                    "variable_name": "witness_nonlinear_margin",
                    "assertion": "state[19] * state[19] + state[22] >= 2.0",
                    "tolerance": 1e-12,
                },
            ],
        }),
    ]
    san = sanitize_intrinsics(kernel, contracts=contracts, all_contracts=contracts)
    san2 = sanitize_intrinsics(san.code, contracts=contracts, all_contracts=contracts)
    assert san2.changed is False

    # Missing nonlinear witnesses remain errors rather than fabricated outputs.
    for vc in contracts:
        _init, warnings = _guard_contract_initial_state(san.code, vc, all_contracts=contracts)
        if vc.contract_id == 'c3_tmsv_snr_recovery':
            assert any('state[19]' in warning for warning in warnings)
            assert any('state[22]' in warning for warning in warnings)

    # Compile and execute via real C++20 C-ABI bridge
    uast = python_source_to_uast(san.code)
    cpp = CppEmitter().emit_module(uast)
    bridge = DynamicCABIBridge(tmp_path / "cabi_ratio_witness")
    for vc in contracts:
        init, _w = _guard_contract_initial_state(san.code, vc, all_contracts=contracts)
        res = bridge.run(
            cpp_code=cpp,
            initial_state=list(init) + [0.0] * 4,
            dt=0.01,
            steps=3,
            entry="execute_step",
        )
        assert res["finite"] is True
        # Physical sigma_sql (1.7147e-22) preserved when positive
        assert res["state"][8] == pytest.approx(1.7147e-22, rel=1e-9)
        for inv in vc.invariants:
            assert not _eval_invariant(inv.assertion, res["state"], {}, {"tol": inv.tolerance}), (
                f"{vc.contract_id}.{inv.variable_name} failed: {inv.assertion}, state={res['state'][:24]}"
            )


def test_tier0_tuple_loop_tuple_return_builtins_bridge_order_and_cpp20_execution(tmp_path):
    from invar_agent.research.state_machine import _eval_invariant, _guard_contract_initial_state
    from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
    from invar_agent.schemas.research import VerificationContract
    from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
    from invar_agent.transpiler.frontend import python_source_to_uast

    # Mimic nh3_floquet_microkinetics tuple/list for-loops, tuple-returning helper,
    # multi-arg min/max/sum, and de-collided TOF_classical bridge after state[6] <= 1e-3.
    kernel = (
        "import math\n\n"
        "def rate_classical(t):\n"
        "    k = math.exp(-3.0 / max(t, 1.0)) * 1e-4\n"
        "    ea = 1.6e-19 * 0.95\n"
        "    return k, ea\n\n"
        "def simulation_step(buf, dt):\n"
        "    theta_free = 0.70\n"
        "    theta_n = 0.15\n"
        "    theta_h = 0.10\n"
        "    theta_nh3 = 0.05\n"
        "    eps = 1e-12\n"
        "    s = 0.0\n"
        "    for v in (theta_free, theta_n, theta_h, theta_nh3):\n"
        "        if v > eps:\n"
        "            s += v * math.log(v)\n"
        "    k_n2, ea_n2 = rate_classical(300.0)\n"
        "    min_v = min(theta_free, theta_n, theta_h, theta_nh3)\n"
        "    tot = sum((theta_free, theta_n, theta_h, theta_nh3))\n"
        "    tof_cl = 1.0e13\n"  # deliberately violates Stage 1 state[6] <= 1e-3 before clamping
        "    tof_enh = 5.0 + 0.0 * k_n2 + 0.0 * ea_n2 + 0.0 * min_v + 0.0 * s + 0.0 * tot\n"
        "    buf.data[4] = theta_free\n"
        "    buf.data[5] = theta_n\n"
        "    buf.data[6] = tof_cl\n"
        "    buf.data[7] = tof_enh\n"
        "    buf.data[8] = 45.0 / 64.0\n"
        "    buf.data[9] = 0.95\n"
    )
    contracts = [
        VerificationContract.model_validate({
            "contract_id": "c1_classical_lhhw",
            "target_backend": "cpp",
            "initial_state": [0.0] * 24,
            "dt": 0.01,
            "steps": 3,
            "invariants": [
                {"variable_name": "tof_classical", "assertion": "state[6] <= 1e-3", "tolerance": 1e-12},
                {"variable_name": "theta_sum", "assertion": "abs(state[4] + state[5] - 0.85) <= 1e-9", "tolerance": 1e-9},
            ],
        }),
        VerificationContract.model_validate({
            "contract_id": "c2_floquet_virasoro_tof",
            "target_backend": "cpp",
            "initial_state": [0.0] * 24,
            "dt": 0.01,
            "steps": 3,
            "invariants": [
                {"variable_name": "delta_h_vis", "assertion": "abs(state[8] - 0.703125) <= 1e-12", "tolerance": 1e-12},
                {"variable_name": "tof_enhanced_exceeds_classical", "assertion": "state[7] > state[16] and state[7] >= 1.0", "tolerance": 1e-12},
                {"variable_name": "eta_selectivity", "assertion": "0.90 <= state[9] <= 1.0", "tolerance": 1e-12},
            ],
        }),
    ]
    san = sanitize_intrinsics(
        kernel,
        contracts=contracts,
        all_contracts=contracts,
        variable_map={"tof_classical": 6, "tof_classical_c2": 16, "tof_enhanced": 7},
    )
    assert san.changed
    assert "for v in" not in san.code
    assert "rate_classical__0" in san.code and "rate_classical__1" in san.code
    san2 = sanitize_intrinsics(
        san.code,
        contracts=contracts,
        all_contracts=contracts,
        variable_map={"tof_classical": 6, "tof_classical_c2": 16, "tof_enhanced": 7},
    )
    assert san2.changed is False

    # Compile and execute via C++20 C-ABI
    uast = python_source_to_uast(san.code)
    cpp = CppEmitter().emit_module(uast)
    bridge = DynamicCABIBridge(tmp_path / "cabi_tuple_unroll")
    for vc in contracts:
        init, _w = _guard_contract_initial_state(san.code, vc, all_contracts=contracts)
        res = bridge.run(
            cpp_code=cpp,
            initial_state=list(init) + [0.0] * 4,
            dt=0.01,
            steps=3,
            entry="execute_step",
        )
        assert res["finite"] is True
        assert res["state"][7] == pytest.approx(5.0)

    # Direct CppEmitter range-for fallback check on an unsanitized scalar tuple loop
    direct_src = (
        "def simulation_step(buf, dt):\n"
        "    s = 0.0\n"
        "    for v in (buf.data[0], 0.25, 0.5):\n"
        "        s += v\n"
        "    buf.data[4] = s\n"
    )
    direct_cpp = CppEmitter().emit_module(python_source_to_uast(direct_src))
    assert "for (double v : {" in direct_cpp

    # Multi-slot linear sum conservation with evolving dynamical slot state[1]
    simplex_src = (
        "def simulation_step(buf, dt):\n"
        "    buf.data[1] = 0.05\n"
        "    buf.data[24] = 0.1\n"
        "    buf.data[25] = 0.1\n"
        "    buf.data[26] = 0.7\n"
    )
    vc_simplex = VerificationContract.model_validate({
        "contract_id": "c3_thermo_selectivity",
        "target_backend": "cpp",
        "initial_state": [0.4, 0.1, 0.2, 0.3] + [0.0] * 24,
        "dt": 0.01,
        "steps": 3,
        "invariants": [
            {"variable_name": "coverage_simplex_stage3", "assertion": "abs(state[24] + state[1] + state[25] + state[26] - 1.0) <= 1e-12", "tolerance": 1e-12},
        ],
    })
    san_sx = sanitize_intrinsics(simplex_src, contracts=[vc_simplex], all_contracts=[vc_simplex])
    # ARCH-001: sanitize_intrinsics never injects silent simplex clamping.
    assert san_sx.changed is False
    res_sx = DynamicCABIBridge(tmp_path / "cabi_msum").run(
        cpp_code=CppEmitter().emit_module(python_source_to_uast(san_sx.code)),
        initial_state=list(vc_simplex.initial_state),
        dt=0.01,
        steps=3,
        entry="execute_step",
    )
    assert res_sx["finite"] is True


def test_latex_normalizer_no_double_subscripts_no_bare_latex_and_table_wrapping():
    import re
    from invar_agent.latex_normalizer import normalize_report_latex
    from invar_agent.research.extractor import ExtractedEquation, StructuredResearchEvidence, render_evidence_table

    hyp = (
        "**Hypothesis:** Coupling delta_g_PEGS(t) = G * M_dot_moment(t) / R_fault^3 with "
        "sigma_eff^2 = sigma_SQL^2 * (exp(-2 * r_squeeze) + S*1e-3), overcoming SNR_SQL < 1 "
        "so SNR_TMSV >= 10 with r_squeeze = 2.50. Legacy bare "
        r"\delta_g_{\mathrm{PEGS}}(t)=G \cdot M_dot_{\mathrm{moment}}(t)/R_{\mathrm{fault}}^3 "
        r"and \mathrm{SNR}_{\mathrm{SQL}}<1 and SNR_{\mathrm{TMSV}} \ge 10."
    )
    norm = normalize_report_latex(hyp)
    assert "S1e-3" not in norm
    assert r"\cdot 10^{-3}" in norm
    assert r"\delta_{g,\mathrm{PEGS}}" in norm
    assert r"M_{\mathrm{dot},\mathrm{moment}}" in norm
    # Zero double subscripts anywhere in math segments
    for seg in norm.split("$")[1::2]:
        assert not re.search(r"_(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|[A-Za-z0-9])\s*_", seg), seg
    # Zero bare LaTeX commands outside $...$
    outside = "".join(norm.split("$")[0::2])
    for forbidden in (r"\mathrm", r"\delta", r"\sigma", r"\ge", r"\le", r"\cdot", "_{", "^{"):
        assert forbidden not in outside, f"Bare {forbidden} leaked outside $...$: {outside}"

    ev = StructuredResearchEvidence(
        equations=[
            ExtractedEquation(
                lhs="eta_D",
                rhs_latex=r"\frac{23}{33}",
                rhs_python_hint="23.0 / 33.0",
                description="Dark partition",
                source="SHBT",
            ),
            ExtractedEquation(
                lhs="S",
                rhs_latex=r"\left(\frac{\eta_A}{\eta_D + \eta_A}\right)^2",
                rhs_python_hint="(eta_A / (eta_D + eta_A)) ** 2",
                description="Shunt factor",
                source="SHBT",
            ),
        ]
    )
    tbl = render_evidence_table(ev)
    assert "| $\\frac{23}{33}$ |" in tbl
    assert "| $\\left(\\frac{\\eta_A}{\\eta_D + \\eta_A}\\right)^2$ |" in tbl
    tbl_norm = normalize_report_latex(tbl)
    assert r"$\frac{23}{33}$" in tbl_norm


def test_initial_k_folding_introspection_desugaring_heal_fallback_and_norm_latex(tmp_path):
    import asyncio
    from invar_agent.latex_normalizer import normalize_report_latex
    from invar_agent.research.broadcaster import ResearchEventBroadcaster
    from invar_agent.research.state_machine import _eval_invariant, _guard_contract_initial_state, ResearchStateMachine
    from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
    from invar_agent.schemas.research import ResearchBlueprint, VerificationContract
    from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
    from invar_agent.transpiler.frontend import python_source_to_uast

    # 1. initial[k] folding on upper/lower/strict/witness bounds (shbt_grmhd_siphon)
    grmhd_kernel = (
        "import math\n\n"
        "def simulation_step(buf, dt):\n"
        "    buf.data[16] = 1.5\n"     # violates < initial[3] (1.0)
        "    buf.data[17] = 0.30119\n" # violates <= 1e-16 + 1e-3*initial[3] (0.001)
        "    buf.data[18] = 1e-5\n"    # violates <= initial[18] + 1e-30 (1e-30)
    )
    contracts = [
        VerificationContract.model_validate({
            "contract_id": "c1_base",
            "target_backend": "cpp",
            "initial_state": [0.0, 0.0, 0.0, 1.0] + [0.0] * 20,
            "dt": 0.01,
            "steps": 3,
            "invariants": [{"variable_name": "ok", "assertion": "state[4] >= 0.0", "tolerance": 1e-12}],
        }),
        VerificationContract.model_validate({
            "contract_id": "c2_ccz4_memory",
            "target_backend": "cpp",
            "initial_state": [0.0, 0.0, 0.0, 1.0] + [0.0] * 20,
            "dt": 0.01,
            "steps": 3,
            "invariants": [
                {"variable_name": "Theta_inf", "assertion": "state[17] <= 1e-16 + 1e-3*initial[3]", "tolerance": 1e-12},
                {"variable_name": "b16", "assertion": "state[16] < initial[3] and state[16] >= 0.0", "tolerance": 1e-12},
                {"variable_name": "b18", "assertion": "state[18] <= initial[18] + 1e-30", "tolerance": 1e-12},
            ],
        }),
    ]
    san = sanitize_intrinsics(grmhd_kernel, contracts=contracts, all_contracts=contracts)
    assert sanitize_intrinsics(san.code, contracts=contracts, all_contracts=contracts).changed is False

    bridge = DynamicCABIBridge(tmp_path / "cabi_initk")
    cpp = CppEmitter().emit_module(python_source_to_uast(san.code))
    init2, _ = _guard_contract_initial_state(san.code, contracts[1], all_contracts=contracts)
    res = bridge.run(cpp, list(init2), dt=0.01, steps=3, entry="execute_step")
    assert res["finite"] is True

    # 2. Tier-0 desugaring of hasattr/getattr/setattr/isinstance/print and custom buf.<attr>
    intro_kernel = (
        "import math\n\n"
        "def simulation_step(buf, dt):\n"
        "    if hasattr(buf, 'dt') and hasattr(math, 'sqrt') and isinstance(dt, float):\n"
        "        h = getattr(buf, 'dt', 0.01)\n"
        "    else:\n"
        "        h = 0.01\n"
        "    if not hasattr(buf, '_init'):\n"
        "        buf._init = 1.0\n"
        "        setattr(buf, '_acc', 2.5)\n"
        "    else:\n"
        "        buf._acc += 0.5\n"
        "    v_acc = getattr(buf, '_acc', 0.0)\n"
        "    print('debug step')\n"
        "    buf.data[4] = v_acc + 0.0 * h\n"
    )
    vc_intro = VerificationContract.model_validate({
        "contract_id": "c1_intro",
        "target_backend": "cpp",
        "initial_state": [0.0] * 8,
        "dt": 0.01,
        "steps": 3,
        "invariants": [{"variable_name": "acc", "assertion": "state[4] >= 3.0", "tolerance": 1e-12}],
    })
    san_in = sanitize_intrinsics(intro_kernel, contracts=[vc_intro])
    assert san_in.changed
    for bad_tok in ("hasattr", "getattr", "setattr", "isinstance", "print(", "._init", "._acc"):
        assert bad_tok not in san_in.code, san_in.code
    assert sanitize_intrinsics(san_in.code, contracts=[vc_intro]).changed is False

    init_in, _ = _guard_contract_initial_state(san_in.code, vc_intro)
    assert len(init_in) >= 64
    res_in = bridge.run(
        CppEmitter().emit_module(python_source_to_uast(san_in.code)),
        init_in,
        dt=0.01,
        steps=3,
        entry="execute_step",
    )
    assert res_in["finite"] is True
    # Step 0 sets 2.5, step 1 adds 0.5 -> 3.0, step 2 adds 0.5 -> 3.5
    assert res_in["state"][4] == pytest.approx(3.5)

    # 3. Healing must never terminate early on transpiler/syntax errors when directives return fixable=False
    async def _mock_fast(prompt_txt):
        return '{"fixable": false, "unfixable_reason": "no directive handles transpiler err", "directives": []}'

    async def _mock_planner(prompt_txt):
        return "```python\ndef simulation_step(buf, dt):\n    buf.data[4] = 3.5\n```"

    sm = ResearchStateMachine(tmp_path / "sm_heal", None, llm_planner=_mock_planner, llm_fast=_mock_fast)
    sm.blueprint = ResearchBlueprint(
        blueprint_id="bp_heal",
        title="Heal Fallback Test",
        falsifiable_hypothesis="state[4] >= 3.0 holds",
        search_queries=["q"],
        code_template="",
        contracts=[vc_intro],
    )
    heal_res = asyncio.run(sm._heal_and_refine(
        stage=1,
        round_idx=1,
        objective="obj",
        candidate_code="import math\n\ndef simulation_step(buf, dt):\n    x = unknown_builtin(buf.data[0])\n    buf.data[4] = 3.5 + 0.0 * x\n    if buf.data[4] < 3.0:\n        buf.data[4] = 3.0\n",
        contracts=[vc_intro],
        failure="transpiler: call to 'unknown_builtin' is not allowed in sandbox kernels",
        attempt=2,
        max_attempts=3,
    ))
    assert heal_res["terminated"] is False
    assert "unknown_builtin" not in heal_res["code"]

    # 4. LaTeX normalizer Fix 4: ||h_res||_inf and |Delta_x_CoM| <= 1e-15 / |Delta sigma_r^2/sigma_r^2| <= 1e-6
    latex_txt = "Checks ||h_res||_inf <= 1.0e-120, ||Theta||_inf <= 1.0e-16, |Delta_x_CoM| <= 1e-15, and |Delta sigma_r^2/sigma_r^2| <= 1e-6."
    norm = normalize_report_latex(latex_txt)
    assert r"$\|h_{\mathrm{res}}\|_{\infty} \le 10^{-120}$" in norm
    assert r"$\|\Theta\|_{\infty} \le 10^{-16}$" in norm
    assert r"$|\Delta_{x,\mathrm{CoM}}| \le 10^{-15}$" in norm
    assert r"$|\Delta \sigma_r^2/\sigma_r^2| \le 10^{-6}$" in norm
    assert "$_inf" not in norm


def test_helper_inlining_unconditional_slot0_3_enforcement_no_fibonacci_finance_and_ui_telemetry(tmp_path):
    from pathlib import Path
    from invar_agent.research.extractor import (
        QuantitativeInsightExtractor,
        classify_registry_domains,
        recommend_execution_strategy,
    )
    from invar_agent.research.state_machine import _eval_invariant, _guard_contract_initial_state
    from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
    from invar_agent.schemas.research import VerificationContract
    from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
    from invar_agent.transpiler.frontend import python_source_to_uast

    # 1. Helper step-function inlining (stage_a/stage_b/stage_c) + unconditional state[0..5]
    # enforcement across 3 contracts / 18 invariants (shbt_pycno_homochiral pattern).
    pycno_kernel = (
        "import math\n\n"
        "def stage_a(buf, dt):\n"
        "    eta_d = 23.0 / 33.0\n"
        "    eta_a = 10.0 / 33.0\n"
        "    r_pyc = 1.0e-120\n"
        "    buf.data[0] = r_pyc\n"
        "    buf.data[1] = eta_d\n"
        "    buf.data[2] = eta_a\n"
        "    buf.data[3] = 0.0\n"
        "    buf.data[4] = (eta_a / (eta_d + eta_a)) ** 2\n"
        "    buf.data[5] = 0.0\n\n"
        "def stage_b(buf, dt):\n"
        "    s_instanton = 256.0\n"
        "    s_pes = 82.0\n"
        "    mu_debye = 16.5\n"
        "    m_eff = 2.486\n"
        "    p_brems = 5.0e-16\n"
        "    branch_ratio = 1.2e-72\n"
        "    buf.data[6] = s_instanton\n"
        "    buf.data[7] = s_pes\n"
        "    buf.data[8] = mu_debye\n"
        "    buf.data[9] = m_eff\n"
        "    buf.data[10] = p_brems\n"
        "    buf.data[11] = branch_ratio\n\n"
        "def stage_c(buf, dt):\n"
        "    eta_d = 23.0 / 33.0\n"
        "    # Deliberately collide with stage_a's state[0..3] and leave c3 slots 12..17 to be populated/conditioned\n"
        "    buf.data[0] = 1.001e-120\n"
        "    buf.data[3] = 0.0053474936\n"
        "    t_lock_us = 11.40\n"
        "    q_internal = 0.0\n"
        "    v_chem = max(1e-12, buf.data[13] * math.exp(-0.5 * dt))\n"
        "    buf.data[12] = 1.0\n"
        "    buf.data[13] = v_chem\n"
        "    buf.data[14] = t_lock_us\n"
        "    buf.data[15] = q_internal\n"
        "    buf.data[16] = 0.0\n"
        "    buf.data[17] = 1.80e-6\n\n"
        "def simulation_step(buf, dt):\n"
        "    stage_a(buf, dt)\n"
        "    stage_b(buf, dt)\n"
        "    stage_c(buf, dt)\n"
    )
    contracts = [
        VerificationContract.model_validate({
            "contract_id": "c1_pycno_ledger",
            "target_backend": "cpp",
            "initial_state": [1.0e-120, 23.0 / 33.0, 10.0 / 33.0, 0.0, 0.09182736, 0.0] + [0.0] * 18,
            "dt": 0.01,
            "steps": 4,
            "invariants": [
                {"variable_name": "R_pyc_clamp", "assertion": "state[0] <= 1.0e-120 + 1e-135", "tolerance": 1e-135},
                {"variable_name": "R_pyc_nonneg", "assertion": "state[0] >= 0.0", "tolerance": 1e-135},
                {"variable_name": "eta_partition", "assertion": "abs(state[1] + state[2] - 1.0) <= 1e-12", "tolerance": 1e-12},
                {"variable_name": "framing_defect", "assertion": "abs(state[3]) <= 1e-15", "tolerance": 1e-15},
                {"variable_name": "shunt_factor_s", "assertion": "state[4] > 0.0 and state[4] <= 0.25", "tolerance": 1e-12},
                {"variable_name": "phi_nu_vis_zero", "assertion": "abs(state[5]) <= 1e-15", "tolerance": 1e-15},
            ],
        }),
        VerificationContract.model_validate({
            "contract_id": "c2_instanton_brems",
            "target_backend": "cpp",
            "initial_state": [0.0] * 6 + [256.0, 82.0, 16.5, 2.486, 5.0e-16, 1.2e-72] + [0.0] * 12,
            "dt": 0.01,
            "steps": 4,
            "invariants": [
                {"variable_name": "S_instanton_floor", "assertion": "state[6] >= 250.0", "tolerance": 1e-12},
                {"variable_name": "S_PES_floor", "assertion": "state[7] >= 81.50", "tolerance": 1e-12},
                {"variable_name": "mu_Debye_floor", "assertion": "state[8] >= 15.0", "tolerance": 1e-12},
                {"variable_name": "M_eff_mev", "assertion": "abs(state[9] - 2.486) <= 1e-6", "tolerance": 1e-6},
                {"variable_name": "P_brems_cap", "assertion": "state[10] >= 0.0 and state[10] <= 1.0e-15", "tolerance": 1e-20},
                {"variable_name": "branch_ratio_cap", "assertion": "state[11] >= 0.0 and state[11] <= 1.62e-71", "tolerance": 1e-80},
            ],
        }),
        VerificationContract.model_validate({
            "contract_id": "c3_chiral_lock",
            "target_backend": "cpp",
            "initial_state": [0.0] * 12 + [1.0, 1.0, 11.40, 0.0, 0.0, 1.80e-6] + [0.0] * 6,
            "dt": 0.01,
            "steps": 4,
            "invariants": [
                {"variable_name": "ee_homochiral", "assertion": "abs(state[12] - 1.0) <= 1e-6", "tolerance": 1e-6},
                {"variable_name": "V_chem_nonneg", "assertion": "state[13] >= 0.0 and state[13] <= initial[13]", "tolerance": 1e-12},
                {"variable_name": "t_lock_us", "assertion": "state[14] > 0.0 and state[14] <= 11.40", "tolerance": 1e-9},
                {"variable_name": "Q_internal_zero", "assertion": "abs(state[15]) <= 1e-15", "tolerance": 1e-15},
                {"variable_name": "delta_BL_EM", "assertion": "abs(state[16]) <= 1e-15", "tolerance": 1e-15},
                {"variable_name": "sigma_phi_urad", "assertion": "state[17] >= 0.0 and state[17] <= 1.83e-6", "tolerance": 1e-12},
            ],
        }),
    ]
    san = sanitize_intrinsics(pycno_kernel, contracts=contracts, all_contracts=contracts)
    assert san.changed
    assert "def stage_a(" not in san.code and "def stage_b(" not in san.code and "def stage_c(" not in san.code
    assert sanitize_intrinsics(san.code, contracts=contracts, all_contracts=contracts).changed is False

    bridge = DynamicCABIBridge(tmp_path / "cabi_pycno")
    cpp = CppEmitter().emit_module(python_source_to_uast(san.code))
    for vc in contracts:
        init_v, _ = _guard_contract_initial_state(san.code, vc, all_contracts=contracts)
        res = bridge.run(cpp, list(init_v), dt=0.01, steps=4, entry="execute_step")
        assert res["finite"] is True

    # ARCH-001: signed negative drift is preserved faithfully (NOT clamped to 0.0).
    neg_kernel = "def stage_n(buf, dt):\n    buf.data[3] = -0.0053474936\n\ndef simulation_step(buf, dt):\n    stage_n(buf, dt)\n"
    vc_neg = VerificationContract.model_validate({
        "contract_id": "c_neg_abs3",
        "target_backend": "cpp",
        "initial_state": [0.0] * 16,
        "dt": 0.01,
        "steps": 2,
        "invariants": [{"variable_name": "framing_defect", "assertion": "abs(state[3]) <= 1e-15", "tolerance": 1e-15}],
    })
    san_neg = sanitize_intrinsics(neg_kernel, contracts=[vc_neg])
    assert sanitize_intrinsics(san_neg.code, contracts=[vc_neg]).changed is False
    res_neg = bridge.run(CppEmitter().emit_module(python_source_to_uast(san_neg.code)), list(vc_neg.initial_state), dt=0.01, steps=2, entry="execute_step")
    assert res_neg["state"][3] == pytest.approx(-0.0053474936)

    # 2. "Fibonacci braids" / "Fibonacci anyons" must NOT trigger discrete_finance or rust_hin
    braid_prompt = (
        "Couple su(2)_26 + su(3)_8 + so(10)_312 affine weight router with coset weight shift "
        "compensated by 124 Fibonacci braids and Fibonacci anyons so framing defect is zero."
    )
    reg_doms = classify_registry_domains(braid_prompt)
    assert "discrete_finance" not in reg_doms, reg_doms
    ev_braid = QuantitativeInsightExtractor(use_registry=True).extract(braid_prompt)
    assert "discrete_finance" not in (ev_braid.domains or []), ev_braid.domains
    strat = recommend_execution_strategy(braid_prompt)
    assert strat["target_backend"] != "rust_hin", strat

    # 3. Chat UI telemetry wraps obs_* keys inside collapsed <details> instead of raw flex header dump
    idx_html = (Path(__file__).resolve().parents[1] / "invar_agent" / "static" / "index.html").read_text(encoding="utf-8")
    assert "Observables (${obsEntries.length})" in idx_html
    assert "className = 'telemetry-obs'" in idx_html
    assert ".telemetry details" in idx_html
    assert "max-height:140px; overflow-y:auto" in idx_html

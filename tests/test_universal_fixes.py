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
        "across an open bipartite Stinespring boundary register partitioned into 10:23 under "
        "symplectic Heegaard-Floer boundary relabeling."
    )
    res = resolve_shbt_solvability(prompt)
    assert res.should_activate is True
    repos = {r.repo for r in res.matched_repos}
    assert {"sys1own/shbt-precision", "sys1own/shbt-cf", "sys1own/shbt-warp", "sys1own/shbt-swarm"}.issubset(repos)

    std = resolve_shbt_solvability("Simulate a 1D damped harmonic oscillator in SI units")
    assert std.should_activate is False


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

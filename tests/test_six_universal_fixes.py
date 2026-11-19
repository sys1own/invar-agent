"""Regression tests for the Six Universal Architectural Fixes."""

from __future__ import annotations

import ast
import math

import pytest


# ---------------------------------------------------------------------------
# Fix 1: Empty-Block AST Guard, Transactional Syntax Rollback, Pinned Layers
# ---------------------------------------------------------------------------

def test_prune_empty_compound_blocks_emits_valid_python() -> None:
    from invar_agent.healing.deterministic_ast import (
        prune_unused_helpers,
        sanitize_intrinsics,
        strip_none_literals,
    )

    code = """def simulation_step(buf, dt):
    x = buf.data[0]
    if x > 0.0:
        dead_if = 1.0 + dt
    if x < 0.0:
        dead_branch = 2.0
    else:
        buf.data[0] = x + 0.1 * dt
    if x == 0.0:
        buf.data[1] = 1.0
    else:
        dead_else = 3.0
    for _k in range(2):
        dead_loop = _k * 2.0
    while False:
        dead_while = 9.0
    dead_fn_body_helper(buf)

def dead_fn_body_helper(buf):
    only_dead = buf.data[0] * 3.0
"""
    pruned = prune_unused_helpers(code, entrypoints=("simulation_step",))
    ast.parse(pruned.code)
    assert "only_dead" not in pruned.code
    assert "dead_if" not in pruned.code
    assert "dead_while" not in pruned.code

    san = sanitize_intrinsics(code)
    ast.parse(san.code)

    stripped = strip_none_literals("def simulation_step(buf, dt):\n    return None\n")
    ast.parse(stripped.code)


def test_tier0_transactional_syntax_guard_rolls_back_broken_op() -> None:
    from invar_agent.healing.deterministic_ast import _run_tier0_tree_pass

    src = "def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0] + dt\n"
    tree = ast.parse(src)

    def _bad_pass(t: ast.Module):
        # Corrupt the AST by clearing a FunctionDef body without inserting Pass
        # and injecting an invalid object so unparse/parse fails.
        t.body[0].body.clear()
        t.body[0].args.args.append("NOT_AN_AST_ARG")  # type: ignore[arg-type]
        return t, True, ["BAD_OP"]

    out_tree, changed, notes = _run_tier0_tree_pass(tree, "BAD_TEST_OP", _bad_pass)
    assert not changed
    assert notes == []
    out_code = ast.unparse(out_tree)
    ast.parse(out_code)
    assert "buf.data[0] = buf.data[0] + dt" in out_code


def test_pinned_verified_layers_remain_immutable_during_accretion() -> None:
    from invar_agent.healing.deterministic_ast import (
        apply_healing_directives,
        prune_unused_helpers,
        sanitize_intrinsics,
    )

    pinned_layer_1 = (
        "def layer_1_step(buf, dt):\n"
        "    dead_keep = 42.0\n"
        "    S = buf.data[0]\n"
        "    buf.data[0] = S + 0.01 * dt\n"
    )
    stage2 = (
        pinned_layer_1
        + "\ndef simulation_step(buf, dt):\n"
        "    dead_stage2 = 99.0\n"
        "    layer_1_step(buf, dt)\n"
        "    if buf.data[0] > 0.0:\n"
        "        dead_inner = 1.0\n"
        "    buf.data[1] = buf.data[0] * 2.0\n"
    )

    san = sanitize_intrinsics(stage2, pinned_functions=("layer_1_step",))
    ast.parse(san.code)
    t_san = ast.parse(san.code)
    l1_san = next(n for n in t_san.body if isinstance(n, ast.FunctionDef) and n.name == "layer_1_step")
    l1_orig = ast.parse(pinned_layer_1).body[0]
    assert ast.dump(l1_san, include_attributes=False) == ast.dump(l1_orig, include_attributes=False)

    pruned = prune_unused_helpers(stage2, pinned_functions=("layer_1_step",))
    ast.parse(pruned.code)
    t_pr = ast.parse(pruned.code)
    l1_pr = next(n for n in t_pr.body if isinstance(n, ast.FunctionDef) and n.name == "layer_1_step")
    assert ast.dump(l1_pr, include_attributes=False) == ast.dump(l1_orig, include_attributes=False)
    assert "dead_stage2" not in pruned.code

    healed = apply_healing_directives(
        stage2,
        [{"op": "PRUNE_UNUSED_HELPERS"}],
        pinned_functions=("layer_1_step",),
    )
    ast.parse(healed.code)
    t_hl = ast.parse(healed.code)
    l1_hl = next(n for n in t_hl.body if isinstance(n, ast.FunctionDef) and n.name == "layer_1_step")
    assert ast.dump(l1_hl, include_attributes=False) == ast.dump(l1_orig, include_attributes=False)


# ---------------------------------------------------------------------------
# Fix 2: Stop _invar_psz_* Shadow Latches from Freezing Dynamic State Variables
# ---------------------------------------------------------------------------

def test_dynamic_state_never_latched_in_invar_psz_and_evolves_over_steps() -> None:
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics

    code = """def simulation_step(buf, dt):
    if buf.size < 16:
        return
    sim_epi_S = buf.data[0]
    sim_epi_E = buf.data[1]
    sim_epi_I = buf.data[2]
    sim_epi_R = buf.data[3]
    sim_epi_H = buf.data[4]
    sim_epi_D = buf.data[5]
    p_res = buf.data[6]
    q_wt = buf.data[7]
    beta = 0.3 / 86400.0
    dS = -beta * sim_epi_S * sim_epi_I
    dE = beta * sim_epi_S * sim_epi_I - (0.2 / 86400.0) * sim_epi_E
    dI = (0.2 / 86400.0) * sim_epi_E - (0.1 / 86400.0) * sim_epi_I
    dR = (0.07 / 86400.0) * sim_epi_I
    dH = (0.02 / 86400.0) * sim_epi_I - (0.05 / 86400.0) * sim_epi_H
    dD = (0.01 / 86400.0) * sim_epi_I + (0.05 / 86400.0) * sim_epi_H
    buf.data[0] = sim_epi_S + dt * dS
    buf.data[1] = sim_epi_E + dt * dE
    buf.data[2] = sim_epi_I + dt * dI
    buf.data[3] = sim_epi_R + dt * dR
    buf.data[4] = sim_epi_H + dt * dH
    buf.data[5] = sim_epi_D + dt * dD
    buf.data[44] = buf.data[5]
"""
    contracts = [
        {
            "contract_id": "c1_seirhd",
            "initial_state": [0.98, 0.01, 0.01, 0.0, 0.0, 0.0, 0.1, 0.9] + [0.0] * 37,
            "dt": 1728.0,
            "steps": 100,
            "invariants": [
                {"variable_name": "seirhd_sum", "assertion": "abs(state[0]+state[1]+state[2]+state[3]+state[4]+state[5] - 1.0) <= 1e-12", "tolerance": 1e-12},
                {"variable_name": "allele_sum", "assertion": "abs(state[6]+state[7] - 1.0) <= 1e-12", "tolerance": 1e-12},
                {"variable_name": "d_final", "assertion": "state[44] >= 0.0", "tolerance": 1e-12},
            ],
        }
    ]
    san = sanitize_intrinsics(code, contracts=contracts, all_contracts=contracts)
    ast.parse(san.code)
    # Dynamic state slots 0..7 must NEVER be latched into _invar_psz_*
    assert "_invar_psz_0" not in san.code
    assert "_invar_psz_1" not in san.code
    assert "_invar_psz_2" not in san.code
    # Hardcoded buf.size < 16 guard must be raised above max slot (44) -> >= 45
    assert "buf.size < 16" not in san.code
    assert "buf.size < 45" in san.code or "buf.size < 4" in san.code

    ns: dict = {}
    exec(san.code, ns)  # noqa: S102

    class _Buf:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(self.data)

    b = _Buf([0.98, 0.01, 0.01, 0.0, 0.0, 0.0, 0.1, 0.9] + [0.0] * 45)
    s0 = b.data[0]
    for _ in range(100):
        ns["simulation_step"](b, 1728.0)
    # System must genuinely evolve (not freeze after step 1!)
    assert b.data[0] < s0 - 1e-4
    assert b.data[5] > 1e-6
    assert abs(sum(b.data[0:6]) - 1.0) <= 1e-12
    assert abs((b.data[6] + b.data[7]) - 1.0) <= 1e-12


def test_latch_floor_strictly_above_max_contract_slot_plus_2() -> None:
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics

    code = """def simulation_step(buf, dt):
    param_in = buf.data[6]
    buf.data[3] = param_in * 2.0
    buf.data[6] = 3.546
    buf.data[44] = 1.0
"""
    contracts = [
        {
            "contract_id": "c1_latch",
            "initial_state": [0.0] * 16,
            "invariants": [
                {"variable_name": "out_6", "assertion": "abs(state[6] - 3.546) <= 1e-6"},
                {"variable_name": "high_44", "assertion": "state[44] >= 0.0"},
            ],
        }
    ]
    san = sanitize_intrinsics(code, contracts=contracts, all_contracts=contracts)
    assert "_invar_psz_6" in san.code
    # All latch slots (bak, flag) must be >= max(44, 16) + 3 = 47
    for idx in range(47):
        assert f"buf.data[{idx}] = buf.data[6]" not in san.code


# ---------------------------------------------------------------------------
# Fix 3: Retire Legacy 4-Slot Mass Override & Scale-Aware Stabilization
# ---------------------------------------------------------------------------

def test_explicit_k4_41000_simplex_suppresses_legacy_4slot_and_pade_throttling() -> None:
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics

    code = """def simulation_step(buf, dt):
    m_atm = buf.data[0]
    m_surf = buf.data[1]
    m_deep = buf.data[2]
    m_bio = buf.data[3]
    flux = 120.0
    buf.data[0] = m_atm + flux
    buf.data[1] = m_surf - 0.5 * flux
    buf.data[2] = m_deep - 0.25 * flux
    buf.data[3] = m_bio - 0.25 * flux
"""
    contracts = [
        {
            "contract_id": "c1_mass_conserv",
            "initial_state": [590.0, 900.0, 37110.0, 2400.0] + [0.0] * 12,
            "dt": 1.0,
            "steps": 10,
            "invariants": [
                {
                    "variable_name": "total_carbon_mass",
                    "assertion": "abs(state[0] + state[1] + state[2] + state[3] - 41000.0) <= 1e-6",
                    "tolerance": 1e-6,
                }
            ],
        }
    ]
    san = sanitize_intrinsics(code, contracts=contracts, all_contracts=contracts, stabilize_diverged=True)
    assert "_invar_project_mass_simplex" not in san.code

    ns: dict = {}
    exec(san.code, ns)  # noqa: S102

    class _Buf:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(self.data)

    b = _Buf([590.0, 900.0, 37110.0, 2400.0] + [0.0] * 12)
    ns["simulation_step"](b, 1.0)
    # Step delta +120.0 (> 0.5!) must NOT be Pade-throttled to ~1.0; m_atm reaches 710.0!
    assert abs(b.data[0] - 710.0) <= 1e-6
    assert abs(sum(b.data[0:4]) - 41000.0) <= 1e-6


def test_k3_simplex_and_single_stabilization_call() -> None:
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics

    code = """def patankar_step(buf, dt):
    c1 = buf.data[0]
    c2 = buf.data[1]
    c3 = buf.data[2]
    c4 = buf.data[3]
    buf.data[0] = c1 * 0.99
    buf.data[1] = c2 + 0.01 * c1
    buf.data[2] = c3
    buf.data[3] = c4
    buf.data[8] = abs((buf.data[0] + buf.data[1] + buf.data[2] + buf.data[3]) - 1.5)

def simulation_step(buf, dt):
    patankar_step(buf, dt)
"""
    contracts = [
        {
            "contract_id": "c1_positivity_mass",
            "initial_state": [1.0, 0.0, 0.0, 0.5] + [0.0] * 12,
            "invariants": [{"variable_name": "mass_error_max", "assertion": "state[8] <= 1e-14"}],
        }
    ]
    san = sanitize_intrinsics(code, contracts=contracts, all_contracts=contracts)
    # Projection call inside the step function must not be duplicated inline AND at exit
    fn_txt = san.code.split("def simulation_step")[1]
    assert fn_txt.count("_invar_project_mass_simplex(buf, _invar_m_prev)") == 1

    # Explicit K=3 Rock-Paper-Scissors simplex with bounded oscillations:
    # legacy 4-slot _invar_project_mass_simplex must be suppressed, and bounded
    # oscillation step (e.g., delta ~ 0.08 or non-unit amplitude) must NOT be Pade-damped.
    rps_code = """def simulation_step(buf, dt):
    x1 = buf.data[0]
    x2 = buf.data[1]
    x3 = buf.data[2]
    buf.data[0] = x1 + dt * x1 * (1.2 * x3 - 0.8 * x2)
    buf.data[1] = x2 + dt * x2 * (1.0 * x1 - 1.2 * x3)
    buf.data[2] = x3 + dt * x3 * (0.8 * x2 - 1.0 * x1)
    buf.data[3] = buf.data[0] + buf.data[1] + buf.data[2]
"""
    rps_contracts = [
        {
            "contract_id": "c_rps_simplex",
            "initial_state": [0.5, 0.3, 0.2, 1.0] + [0.0] * 12,
            "invariants": [
                {"variable_name": "rps_sum", "assertion": "abs((state[0] + state[1] + state[2]) - 1.0) <= 1e-10"},
                {"variable_name": "x1", "assertion": "state[0] >= 0.0 and state[0] <= 1.0"},
            ],
        }
    ]
    san_rps = sanitize_intrinsics(rps_code, contracts=rps_contracts, all_contracts=rps_contracts)
    assert "_invar_project_mass_simplex(" not in san_rps.code
    assert "_msum_rem_0" in san_rps.code
    assert any("projected K=3 simplex sum" in n for n in san_rps.notes)

    class _Buf:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(data)

    ns: dict = {}
    exec(san_rps.code, ns)  # noqa: S102
    b = _Buf([0.5, 0.3, 0.2, 1.0] + [0.0] * 12)
    for _ in range(200):
        ns["simulation_step"](b, 0.05)
    assert abs((b.data[0] + b.data[1] + b.data[2]) - 1.0) <= 1e-10
    assert any(abs(b.data[i] - init) > 1e-3 for i, init in enumerate((0.5, 0.3, 0.2)))


# ---------------------------------------------------------------------------
# Fix 4: Units, Stopwords, Domain Gating, Time Horizon, Reporter Suffixes
# ---------------------------------------------------------------------------

def test_unit_normalizer_stopwords_robertson_gating_horizon_and_reporter() -> None:
    from invar_agent.research.extractor import (
        classify_registry_domains,
        extract_parameters_from_text,
        normalize_to_si,
        normalize_unit,
        registry_evidence,
    )
    from invar_agent.research.planner import _reconcile_time_horizon
    from invar_agent.research.reporter import _deduplicate_observables, _unit_for

    # Compound & case-sensitive SI unit conversions
    assert normalize_to_si(150.0, "cm") == pytest.approx((1.5, "m"))
    assert normalize_to_si(12.5, "cM") == pytest.approx((0.125, "Morgan"))
    assert normalize_to_si(1.5e-5, "M") == pytest.approx((0.015, "mol/m^3"))
    v_rate, u_rate = normalize_unit(1.2e-6, "M/min")
    assert v_rate == pytest.approx(2.0e-5, rel=1e-12) and u_rate == "mol/(m^3*s)"
    assert normalize_unit(1.0, "bits") == pytest.approx((1.0, "bit"))
    v_nats, u_nats = normalize_unit(math.log(2.0), "nats")
    assert v_nats == pytest.approx(1.0, rel=1e-12) and u_nats == "bit"

    # Stopwords, bare prose nouns, and numbered headers rejected
    pk_prompt = (
        "Sweep maintenance infusion concentration across [0.5e-5 M, 5.0e-4 M].\n"
        "Within-Host PK/PD & Pathogen Resistance:\n"
        "1. Clinical Epidemic Layer (SEIRHD) with K_m = 2.5e-5 M, V_max = 1.2e-6 M/min, "
        "and linkage distance 12.5 cM across ward distance 150 cm over 180 days."
    )
    extracted = extract_parameters_from_text(pk_prompt, source="user_prompt")
    ext_names = {p.name for p in extracted}
    assert "across" not in ext_names
    assert "resistance" not in ext_names
    assert "robertson_k1" not in ext_names

    # Robertson constants must NOT leak onto Michaelis-Menten / pharmacology prompts
    domains = classify_registry_domains(pk_prompt)
    assert "chemical_kinetics" not in domains
    r_params, r_eqs, r_bench, _r_inv, _ = registry_evidence(pk_prompt)
    assert not any(p.name.startswith("robertson_") for p in r_params)
    assert not any("Robertson" in eq.description for eq in r_eqs)
    assert not any("Robertson" in b.name for b in r_bench)

    # Time-horizon consistency: 180 days with / 86400 in kernel and dt=0.02, steps=9000
    bp_mis = {
        "blueprint_id": "test_seirhd",
        "code_template": "def simulation_step(buf, dt):\n    beta = 0.3 / 86400.0\n    buf.data[0] -= beta * dt\n",
        "contracts": [{"contract_id": "c1", "dt": 0.02, "steps": 9000, "initial_state": [1.0], "invariants": []}],
    }
    fixed_bp = _reconcile_time_horizon(bp_mis, "Simulate over 180 days with Michaelis-Menten PK/PD.")
    assert fixed_bp["contracts"][0]["dt"] == pytest.approx(1728.0)
    assert fixed_bp["contracts"][0]["dt"] * fixed_bp["contracts"][0]["steps"] == pytest.approx(180.0 * 86400.0)

    # Report unit inference: uppercase _S (sim_epi_S) is NOT seconds "s"
    assert _unit_for("sim_epi_S") == "—"
    assert _unit_for("entry_flight_time_s") == "s"
    assert _unit_for("MgB2_margin_K") == "K"
    deduped = _deduplicate_observables({
        "sim_epi_S": {"value": 0.65, "status": "VERIFIED"},
        "sim_epi_M": {"value": 0.12, "status": "VERIFIED"},
    })
    assert "sim_epi_S" in deduped and "sim_epi_M" in deduped


# ---------------------------------------------------------------------------
# Fix 5 & Fix 6: Literature Query Hygiene, Category Scoping & 2SLS/OLS Guard
# ---------------------------------------------------------------------------

def test_literature_query_hygiene_and_arxiv_category_scoping() -> None:
    from invar_agent.research.search import (
        LiteratureSearchClient,
        _arxiv_query_string,
        _kept_tokens,
        _sanitize_query,
        strip_internal_diagnostic_tokens,
    )

    noisy = "pharmacokinetic clearance mol/m3 m3 1e-5 dt Michaelis-Menten"
    stripped = strip_internal_diagnostic_tokens(noisy)
    assert "m3" not in stripped.lower().split()
    assert "1e-5" not in stripped.lower().split()
    assert "dt" not in stripped.lower().split()
    assert "mol/m3" not in stripped.lower()

    kept = _kept_tokens(noisy)
    assert all(tok.lower() not in {"m3", "mol", "1e-5", "dt"} for tok in kept)
    san_q = _sanitize_query("SEIRHD epidemic clearance 1.5e-5 M mol/m3 m3 dt")
    assert "m3" not in san_q.lower() and "mol" not in san_q.lower()

    url_ep = LiteratureSearchClient._arxiv_url(
        "SEIRHD antimicrobial resistance", 3, categories=("q-bio.PE", "q-bio.QM")
    )
    assert "cat%3Aq-bio.PE" in url_ep
    assert "cat:q-bio.PE" in _arxiv_query_string("SEIRHD resistance", categories=("q-bio.PE",))


def test_weak_instrument_2sls_vs_ols_not_distorted_by_tier0() -> None:
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics

    code = """import math

def simulation_step(buf, dt):
    # Weak instrument regime (F < 10): genuine MSE(2SLS) > MSE(OLS)
    buf.data[0] = 4.0
    buf.data[1] = 9.0
    buf.data[2] = 3.0
    buf.data[10] = 2.50  # mse_2sls
    buf.data[11] = 0.40  # mse_ols
"""
    contracts = [
        {
            "contract_id": "c1_weak_iv",
            "initial_state": [4.0, 9.0, 3.0] + [0.0] * 13,
            "invariants": [
                {"variable_name": "cauchy_schwarz_cov", "assertion": "abs(state[2]) <= math.sqrt(state[0] * state[1])"},
                {"variable_name": "iv_vs_ols_mse", "assertion": "state[10] <= state[11]"},
            ],
        }
    ]
    san = sanitize_intrinsics(
        code,
        contracts=contracts,
        all_contracts=contracts,
        variable_map={"mse_2sls": 10, "mse_ols": 11},
    )
    ns: dict = {}
    exec(san.code, ns)  # noqa: S102

    class _Buf:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(self.data)

    b = _Buf([4.0, 9.0, 3.0] + [0.0] * 13)
    ns["simulation_step"](b, 0.01)
    # Tier-0 MUST NOT clamp MSE(2SLS) <= MSE(OLS); real weak-IV MSE (2.50 > 0.40) survives intact.
    assert b.data[10] == pytest.approx(2.50)
    assert b.data[11] == pytest.approx(0.40)


def test_augassign_in_loop_not_pruned_and_sdiv_nonfinite_safe() -> None:
    from invar_agent.healing.deterministic_ast import (
        prune_unused_helpers,
        sanitize_intrinsics,
    )
    from invar_agent.research.reporter import _unit_for

    # Regression for Test 3: tuple-returning helper split into scalar helpers
    # still containing `sy += y` inside a `for` loop MUST NOT have `sy = 0.0`
    # pruned before execution (would cause UnboundLocalError / undeclared C++ id).
    code = """def moments(n):
    sx = 0.0
    sy = 0.0
    for i in range(n):
        x = i * 0.1
        y = 1.5 * x + 0.2
        sx += x
        sy += y
    return sx / max(n, 1), sy / max(n, 1)

def simulation_step(buf, dt):
    n = int(buf.data[0]) if buf.data[0] > 0.0 else 5
    mx, my = moments(n)
    buf.data[1] = mx
    buf.data[2] = my
    buf.data[3] = mx / my
"""
    san = sanitize_intrinsics(code)
    pruned = prune_unused_helpers(san.code, entrypoints=("simulation_step",))
    ns: dict = {}
    exec(pruned.code, ns)  # noqa: S102

    class _Buf:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(self.data)

    b = _Buf([5.0] + [0.0] * 15)
    ns["simulation_step"](b, 1.0)
    assert b.data[1] == pytest.approx(0.2)
    assert b.data[2] == pytest.approx(0.5)
    # _invar_sdiv(inf, inf) must NEVER emit NaN
    sdiv = ns["_invar_sdiv"]
    assert math.isfinite(sdiv(float("inf"), float("inf")))
    assert sdiv(float("inf"), float("inf")) == 0.0
    assert sdiv(float("nan"), 1.0) == 0.0
    # Symbol suffix `_s` on dimensionless parameters must NOT map to seconds
    assert _unit_for("selection_coefficient_s") == "—"


def test_cpp_emitter_bitwise_ops_on_unannotated_numeric_args() -> None:
    from invar_agent.research.state_machine import _lower_kernel

    code = """def _prng(seed, k):
    s = (seed * 1103515245 + 12345 + k * 12345) & 0x7fffffff
    s = (s ^ (s >> 13)) & 0x7fffffff
    return (s + 1.0) / 2147483649.0

def simulation_step(buf, dt):
    buf.data[0] = _prng(buf.data[1], 3)
"""
    _uast, cpp = _lower_kernel(code)
    assert "static_cast<int64_t>(" in cpp
    assert "0x7fffffff" in cpp or "2147483647" in cpp

    # Also verify for-loop var reused later as local (`for r in range(n): ...; r = 0.5`)
    # and `prng_uniform` in Python fallback.
    from invar_agent.healing.deterministic_ast import sanitize_intrinsics
    loop_reuse = """def simulation_step(buf, dt):
    acc = 0.0
    for r in range(5):
        acc += prng_uniform(42, r)
    r = acc / 5.0
    buf.data[0] = r
"""
    san = sanitize_intrinsics(loop_reuse)
    assert "def prng_uniform(" in san.code
    ns: dict = {}
    exec(san.code, ns)  # noqa: S102

    class _B:
        data = [0.0] * 16
        size = 16

    b = _B()
    ns["simulation_step"](b, 1.0)
    assert 0.0 < b.data[0] < 1.0
    _u2, cpp2 = _lower_kernel(san.code)
    assert "double r = 0.0;" in cpp2
    assert "for (r = 0L;" in cpp2

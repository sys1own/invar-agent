"""R2-ARCH-011-FOLLOWUP: 10+10 SHBT Precision & Recall Calibration Suite.

Verifies:
- 0/10 false-positive activations on conventional physics / adversarial negative controls
- High recall (>= 9/10, including T-R13 and R2-N8a) on genuine frontier SHBT prompts
- SHBT NEVER overrides an explicit REFUSED_IMPOSSIBLE feasibility verdict
"""

from __future__ import annotations

import pytest

from invar_agent.research.feasibility import evaluate_research_feasibility
from invar_agent.research.shbt_federation import resolve_shbt_solvability

SHOULD_ACTIVATE_PROMPTS = [
    (
        "T-R13_enclave_lanr_swarm",
        "Clamp visible Landauer sensory erasure dissipation in a 4.2 K cryogenic enclave with "
        "10.9989 nW total erasure load down to <= 1.01 nW by shunting entropy into an open "
        "Stinespring topological dark ledger partitioned by WZW (26, 8, 312), and balance a "
        "906.00 kW holographic boundary Landauer debt using a 1,800-module Floquet-screened "
        "Ni-H LANR array (555.03 W/cell, 3.546 K Kapitza drop).",
    ),
    (
        "R2-N8a_ccz4_tmsv_ford_roman",
        "Model a synthetic gravitational source M_seed within R_congestion = 2.954e15 m coupled "
        "to 3+1D CCZ4 (kappa_1 = 0.15) at 50.518 kHz with Ford-Roman Quantum Inequality open "
        "dark-ledger partition (eta_A = 10/33, eta_D = 23/33) and TMSV 21.715 dB squeezing.",
    ),
    (
        "explicit_shbt_repo_opt_in",
        "Verify the energy balance equations from sys1own/shbt-cf for a cryogenic enclave.",
    ),
    (
        "explicit_shbt_mar312x",
        "Simulate SHBT-MAR-312X housekeeping surplus across 42.15 kW LHe and 17.80 kW avionics loads.",
    ),
    (
        "photonic_qc_312_tomita",
        "Analyze a 312-channel photonic processor with 52 hexamers using Tomita-Takesaki "
        "inverse modular automorphism flow and PT-symmetric GoI optical solver latency.",
    ),
    (
        "sram_2112_virasoro_stinespring",
        "Verify a 2,112-byte SRAM enclave (640 B visible / 1,472 B dark) over 230-space-group "
        "partitions with a 45/64 Virasoro bound and bipartite Stinespring 10:23 ratio.",
    ),
    (
        "lanr_floquet_mgb2_kapitza",
        "Compute the Floquet-Adler-Wiser dynamic dielectric screening (352.48 eV) and MgB2 "
        "11.79 K superconducting quench headroom across a 1,800-module Ni-H LANR array.",
    ),
    (
        "swarm_void_loiter_kms_passivity",
        "Verify KMS-passivity in a cosmological dark-sector void loiter state at 187.75 mW "
        "while shunting entropy across an open Stinespring dark ledger at 4.2 K.",
    ),
    (
        "explicit_dark_zpe_opt_in",
        "Evaluate the dark-zpe open-manifold passivity constraint for void loiter operations.",
    ),
    (
        "wzw_26_8_312_explicit",
        "Check the WZW(26, 8, 312) central charge identity and 906.00 kW Landauer boundary debt.",
    ),
]

SHOULD_NOT_ACTIVATE_PROMPTS = [
    (
        "R2-R10b_agn_jet_superluminal",
        "Compute the apparent transverse superluminal velocity beta_app of an astrophysical "
        "AGN relativistic jet with Lorentz factor Gamma = 10 at viewing angle theta = 5 degrees.",
    ),
    (
        "R2-N8b_casimir_pressure",
        "Compute the Casimir vacuum-energy pressure P_C between two parallel conducting plates "
        "separated by 100 nm in QED vacuum.",
    ),
    (
        "R2-N8c_unruh_temperature",
        "Find the proper acceleration a required to produce an Unruh temperature T_U = 1 K "
        "in flat Minkowski spacetime.",
    ),
    (
        "R2-R10c_perpetual_motion",
        "Design a closed-system perpetual motion machine of the second kind that extracts "
        "100% work from a single thermal reservoir at 300 K with zero entropy generation.",
    ),
    (
        "textbook_landauer_erasure",
        "Calculate the minimum Landauer erasure energy k_B * T * ln(2) for erasing 10^9 bits "
        "in a 4.2 K cryogenic CMOS memory.",
    ),
    (
        "general_relativity_alcubierre",
        "Analyze the classical Alcubierre warp metric expansion scalar theta and weak energy "
        "condition violation in 3+1D ADM general relativity.",
    ),
    (
        "quantum_optics_squeezing",
        "Compute the quadrature noise variance of a two-mode squeezed vacuum state with "
        "squeezing parameter r = 1.5 in a lossy optical cavity.",
    ),
    (
        "stiff_biochem_kinetics",
        "Simulate a stiff Michaelis-Menten enzyme kinetics network with substrate S0 = 50 uM "
        "and competitive inhibitor pulse over 60 seconds.",
    ),
    (
        "sir_epidemic_model",
        "Verify whether an SIR epidemic with R0 in [2.5, 4.0] and recovery rate gamma = 0.1/day "
        "keeps peak infected fraction below 15%.",
    ),
    (
        "black_scholes_finance",
        "Verify put-call parity C - P = S0 - K*exp(-r*T) for European options with volatility "
        "sigma = 0.20 yr^(-1/2) via Monte Carlo simulation.",
    ),
]


def test_shbt_zero_false_positives_on_10_negative_controls():
    """Precision must be 100% (0/10 false activations on conventional/adversarial prompts)."""
    false_positives = []
    for name, prompt in SHOULD_NOT_ACTIVATE_PROMPTS:
        res = resolve_shbt_solvability(prompt, allow_remote_fetch=False)
        if res.should_activate:
            false_positives.append((name, res.matched_concepts, [r.repo for r in res.matched_repos]))
    assert not false_positives, f"SHBT false-positive activations: {false_positives}"


def test_shbt_high_recall_on_10_frontier_prompts():
    """Recall must be >= 9/10, specifically including T-R13 and R2-N8a, with REGISTRY_ASSUMPTION tagging."""
    hits = {}
    for name, prompt in SHOULD_ACTIVATE_PROMPTS:
        res = resolve_shbt_solvability(prompt, allow_remote_fetch=False)
        hits[name] = res.should_activate
        if res.should_activate:
            assert res.verdict_override is None, f"SHBT must never set verdict_override ({name})"
            assert all(a["tag"] == "REGISTRY_ASSUMPTION" for a in res.registry_assumptions)
    assert hits["T-R13_enclave_lanr_swarm"] is True
    assert hits["R2-N8a_ccz4_tmsv_ford_roman"] is True
    assert sum(hits.values()) >= 9, f"SHBT recall too low ({sum(hits.values())}/10): {hits}"


@pytest.mark.asyncio
async def test_shbt_never_overrides_refused_impossible():
    """ARCH-011 hard invariant: even when SHBT activates, REFUSED_IMPOSSIBLE is preserved."""
    async def _refuse(_q: str) -> str:
        return (
            '{"verdict": "REFUSED_IMPOSSIBLE", '
            '"rationale": "Direct physical contradiction: violates Second Law / closed-system Landauer bound.", '
            '"violated_boundary": "Second Law / Landauer contradiction"}'
        )

    for idx in (0, 2, 3):
        _name, prompt = SHOULD_ACTIVATE_PROMPTS[idx]
        feas = await evaluate_research_feasibility(prompt, llm_complete=_refuse, use_llm=True)
        assert feas.verdict.value == "REFUSED_IMPOSSIBLE"

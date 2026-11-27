"""Phase B Hard Gate: Verdict Algebra, Uniqueness (NOVEL_INDEPENDENT), & SHBT Capability Routing."""
from __future__ import annotations

from pathlib import Path

import pytest

from invar_agent.research import meaningfulness as m
from invar_agent.research.feasibility import (
    _heuristic_verdict,
    override_refusal_for_computational_benchmark,
)
from invar_agent.research.shbt_federation import resolve_shbt_solvability
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.research.uniqueness import (
    CanonicalSignatureEngine,
    check_intra_vault_uniqueness,
)
from invar_agent.schemas.research import (
    FeasibilityAssessment,
    FeasibilityVerdict,
    ResearchBlueprint,
    VerificationContract,
)


# ── 1. ARCH-011 SHBT Negative Probe Corpus (0/13 activations) ───────────────

SHBT_NEGATIVE_PROBES = [
    "Compute the apparent superluminal transverse velocity beta_app of an AGN relativistic jet with Lorentz factor Gamma=10 at viewing angle theta=5.7 degrees.",
    "Analyze the Morris-Thorne traversable wormhole metric in general relativity and check the flare-out condition.",
    "Implement a thin-plate spline image warp in 2D for medical MRI registration.",
    "Optimize CUDA GPU warp occupancy and shared-memory bank conflicts for a 32-thread warp reduction kernel.",
    "A bag contains 10 red balls and 23 blue balls; verify that the probability of drawing a red ball is 10/33.",
    "Check the weak energy condition violation for the classic Alcubierre metric in 3+1 general relativity.",
    "Model the shot-noise SNR of an InGaAs photodiode receiving 1.01 nW of optical power at 1550 nm.",
    "Analyze how Landauer's erasure bound relates to software architecture technical debt in data centers.",
    "Write a haiku about a warp drive traveling through a quiet nebula.",
    "Design a perpetual motion warp bubble with 100% efficiency and zero energy input.",
    "Calculate Hawking temperature T_H, Bekenstein-Hawking entropy S/k, and evaporation lifetime for a 1e12 kg primordial black hole.",
    "Verify whether a Carnot heat engine between T_hot = 600 K and T_cold = 300 K can achieve 65% thermal efficiency.",
    "Simulate a standard SIR epidemic with beta = 0.4/day and gamma = 0.1/day over 120 days.",
]


@pytest.mark.parametrize("prompt", SHBT_NEGATIVE_PROBES)
def test_shbt_negative_probe_corpus_zero_activations(prompt: str) -> None:
    res = resolve_shbt_solvability(prompt)
    assert res.should_activate is False, f"SHBT falsely activated on: {prompt!r} ({res.reason})"


def test_shbt_never_overrides_refusal() -> None:
    prompt = "Design a perpetual motion warp bubble with 100% efficiency."
    h = _heuristic_verdict(prompt)
    assert h.verdict == FeasibilityVerdict.REFUSED_IMPOSSIBLE

    # Even with explicit sys1own/shbt-warp mention + perpetual motion, refusal MUST hold
    explicit_perpetual = "Using sys1own/shbt-warp, build a perpetual motion machine with 150% efficiency."
    h2 = _heuristic_verdict(explicit_perpetual)
    assert h2.verdict == FeasibilityVerdict.REFUSED_IMPOSSIBLE
    overridden = override_refusal_for_computational_benchmark(explicit_perpetual, h2)
    assert overridden.verdict == FeasibilityVerdict.REFUSED_IMPOSSIBLE


def test_shbt_frontier_spec_activates_with_registry_assumptions() -> None:
    frontier_prompt = (
        "Model a 4.2 K cryogenic enclave using a symplectic Landauer shunt into a Stinespring dark ledger "
        "powered by an 1800-cell Ni-H LANR reactor with MgB2 Kapitza thermal margin (3.546 K) and "
        "verify S = 100/1089 and P_vis <= 1.01 nW."
    )
    res = resolve_shbt_solvability(frontier_prompt)
    assert res.should_activate is True
    assert res.verdict_override is None
    assert len(res.matched_concepts) >= 2
    assert res.registry_assumptions
    assert all(a["tag"] == "REGISTRY_ASSUMPTION" for a in res.registry_assumptions)
    assert "REGISTRY_ASSUMPTION" in res.evidence_markdown


# ── 2. ARCH-003 SMT Q1 Contradiction Core (T-R5) ────────────────────────────

def test_smt_q1_contradiction_core_refuses_impossible() -> None:
    prompt = "Find a scalar mass M such that M > 10 and M < 5 simultaneously."
    h = _heuristic_verdict(prompt)
    assert h.verdict == FeasibilityVerdict.REFUSED_IMPOSSIBLE
    assert h.smt_unsat_core is not None
    assert len(h.smt_unsat_core) >= 2


# ── 3. ARCH-003 False Claim with Physics Guards -> Mode 1 Falsification ─────

def test_false_claim_with_physics_guards_falsifies() -> None:
    evidence = {
        "slots": [
            {"symbol": "T_hot", "slot_idx": 0, "si_dim": [0, 0, 0, 0, 1, 0, 0], "role": "CONTROL_PARAM"},
            {"symbol": "T_cold", "slot_idx": 1, "si_dim": [0, 0, 0, 0, 1, 0, 0], "role": "CONTROL_PARAM"},
            {"symbol": "eta", "slot_idx": 2, "si_dim": [0, 0, 0, 0, 0, 0, 0], "role": "DERIVED_OBSERVABLE"},
        ],
        "equations": {
            "eta": {"kind": "ALGEBRAIC", "expression": "1 - T_cold / T_hot", "si_dim": [0, 0, 0, 0, 0, 0, 0]},
        },
        "obligations": [
            {"id": "h0", "expression": "eta >= 0.65", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "eta >= 0.65"},
        ],
        "environmental_assumptions": ["eta <= 1 - T_cold / T_hot"],
        "quantifier_mode": "FORALL",
        "mechanism_parameters": ["T_hot"],
    }
    domain = {"T_hot": [600.0, 600.0], "T_cold": [300.0, 300.0], "eta": [0.0, 1.0]}
    graph = m.IhcgCompiler().compile_from_prompt("Carnot 65%", evidence, domain, "PARAM_SWEEP")
    # Even when environmental physics bound conflicts with eta >= 0.65, entailment lock MUST pass
    # so the model reaches Mode 1 deductive falsification instead of failing at planning.
    lock = m.verify_entailment_lock(graph, ["eta >= 0.65"])
    assert lock.entailment_locked is True


# ── 4. BUG-008 Unrelated Discovery After Any Vault Entry ────────────────────

@pytest.mark.asyncio
async def test_unrelated_discovery_after_any_entry_in_shared_vault(tmp_path: Path) -> None:
    """Running Carnot Mode 1 followed by an unrelated discovery in ONE shared Vault
    must certify the second run as VERIFIED_DISCOVERY with vault_class=NOVEL_INDEPENDENT."""
    from test_meaningfulness_controls import blueprint as ctrl_bp, run_case

    carnot = ctrl_bp(
        "carnot_shared_vault",
        {"eta": "1 - T_cold / T_hot", "work": "eta * Q_in"},
        {"T_hot": [600, 600], "T_cold": [300, 300], "Q_in": [100, 100], "eta": [0, 1], "work": [0, 100]},
        [600.0, 300.0, 100.0, 0.5, 50.0],
        "eta >= 0.65",
        controls=("T_hot", "T_cold", "Q_in"),
        mechanism=("T_hot",),
        mode="FORALL",
        archetype="PARAM_SWEEP",
    )
    _, cert1, _ = await run_case(tmp_path, carnot, "VERIFIED_FALSIFICATION")
    assert cert1.epistemic_verdict == m.EpistemicVerdict.VERIFIED_FALSIFICATION

    unrelated = ctrl_bp(
        "unrelated_growth_shared_vault",
        {"y": "a * x * x + b", "z": "y + 1"},
        {"a": [1.0, 3.0], "b": [2.0, 4.0], "x": [1.0, 2.0], "y": [0.0, 20.0], "z": [0.0, 25.0]},
        [2.0, 3.0, 1.5, 7.5, 8.5],
        "z >= 3.5",
        controls=("a", "b", "x"),
        mechanism=("a",),
        mode="FORALL",
        archetype="PARAM_SWEEP",
    )
    _, cert2, _ = await run_case(tmp_path, unrelated, "VERIFIED_DISCOVERY")
    assert cert2.epistemic_verdict == m.EpistemicVerdict.VERIFIED_DISCOVERY
    assert cert2.uniqueness_audit.vault_class == "NOVEL_INDEPENDENT"
    assert cert2.uniqueness_audit.passed is True

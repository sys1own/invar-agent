"""Phase F Property-Based Tests (Section 9.2 of Audit Report)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from invar_agent.research.epistemic import incomplete_certificate
from invar_agent.research.meaningfulness import solve_query
from invar_agent.research.shbt_federation import resolve_shbt_solvability
from invar_agent.research.vault import (
    REQUIRED_VAULT_ARTIFACTS,
    ResearchVaultManager,
    VaultIntegrityError,
    verify_vault_manifest,
)
from invar_agent.schemas.research import ResearchBlueprint, VerificationContract


# ── Property 1: Random interval claims falsification implies valid witness ───

@settings(max_examples=20, deadline=None)
@given(
    lo=st.floats(min_value=-10.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    width=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
    thresh=st.floats(min_value=-15.0, max_value=20.0, allow_nan=False, allow_infinity=False),
)
def test_property_interval_claim_witness_soundness(lo: float, width: float, thresh: float) -> None:
    hi = lo + width
    # Query: exists x in [lo, hi] violating (x >= thresh), i.e. x < thresh
    res = solve_query(f"x < {thresh!r}", {"x": (lo, hi)}, witness=True, timeout_ms=500)
    if res.status == "SAT":
        assert res.witness is not None and "x" in res.witness
        xw = float(res.witness["x"])
        assert lo - 1e-9 <= xw <= hi + 1e-9
        assert xw < thresh + 1e-9
    elif res.status == "UNSAT":
        assert lo >= thresh - 1e-9


# ── Property 2: Single-byte tamper on ANY vault artifact is ALWAYS rejected ──

@settings(max_examples=12, deadline=None)
@given(
    artifact_idx=st.integers(min_value=0, max_value=len(REQUIRED_VAULT_ARTIFACTS)),
    delta=st.integers(min_value=1, max_value=255),
)
def test_property_single_byte_vault_tamper_rejected(tmp_path_factory, artifact_idx: int, delta: int) -> None:
    tmp_path = tmp_path_factory.mktemp("vault_prop")
    vault = ResearchVaultManager(tmp_path)
    bp = ResearchBlueprint(
        blueprint_id="prop_tamper",
        title="Prop Tamper",
        research_prompt="Test",
        falsifiable_hypothesis="x >= 0",
        search_queries=["test"],
        variable_map={"x": 0},
        code_template="def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0]\n",
        contracts=[
            VerificationContract(
                contract_id="c1",
                target_backend="cpp",
                initial_state=[1.0],
                dt=0.1,
                steps=5,
                invariants=[{"variable_name": "x", "assertion": "state[0] >= 0", "tolerance": 1e-6}],
            )
        ],
    )
    cert = incomplete_certificate("prop_tamper", "x >= 0", "TEST_ONLY")
    vault.register_run(
        run_id="prop_tamper",
        title="Prop Tamper",
        blueprint=bp,
        report_md="# report\n",
        status="INCONCLUSIVE_UNVERIFIED",
        meaningfulness_certificate=cert,
    )
    run_dir = vault.run_dir("prop_tamper")
    all_files = (*REQUIRED_VAULT_ARTIFACTS, "manifest.json")
    target = run_dir / all_files[artifact_idx]
    orig = target.read_bytes()
    flipped = bytearray(orig)
    pos = len(flipped) // 2
    flipped[pos] = (flipped[pos] + delta) % 256
    target.write_bytes(bytes(flipped))
    with pytest.raises(VaultIntegrityError):
        verify_vault_manifest(run_dir)


# ── Property 3: Random non-opt-in prompts NEVER activate SHBT ───────────────

_SAFE_WORDS = st.sampled_from([
    "warp", "superluminal", "wormhole", "alcubierre", "10/33", "23/33",
    "1.01 nW", "Landauer", "technical debt", "GPU", "photodiode",
    "black hole", "Hawking", "AGN jet", "Carnot", "SIR epidemic",
    "haiku", "perpetual motion", "efficiency", "entropy", "diffusion",
    "Navier-Stokes", "quantum", "oscillator", "reactor", "enzyme",
])


@settings(max_examples=40, deadline=None)
@given(words=st.lists(_SAFE_WORDS, min_size=1, max_size=8))
def test_property_random_non_optin_prompts_never_activate_shbt(words: list[str]) -> None:
    prompt = " ".join(words)
    res = resolve_shbt_solvability(prompt)
    assert res.should_activate is False, f"SHBT falsely activated on random prompt: {prompt!r}"

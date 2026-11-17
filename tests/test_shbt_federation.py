"""Unit and integration tests for Non-Intrusive SHBT Frontier Solvability & Federation."""
from __future__ import annotations

import pytest

from invar_agent.research.extractor import QuantitativeInsightExtractor
from invar_agent.research.feasibility import evaluate_research_feasibility
from invar_agent.research.search import LiteratureSearchClient
from invar_agent.research.shbt_federation import (
    SHBT_REPOSITORIES,
    resolve_shbt_solvability,
    shbt_literature_dicts,
)


def test_shbt_registry_covers_all_10_repositories():
    expected = {
        "sys1own/shbt-precision",
        "sys1own/shbt-qc",
        "sys1own/shbt-cf",
        "sys1own/shbt-power",
        "sys1own/shbt-ghost",
        "sys1own/shbt-sglt",
        "sys1own/shbt-exotic",
        "sys1own/shbt-recon",
        "sys1own/shbt-warp",
        "sys1own/shbt-swarm",
    }
    assert set(SHBT_REPOSITORIES.keys()) == expected
    for repo, spec in SHBT_REPOSITORIES.items():
        assert spec.constants, f"{repo} missing constants"
        assert spec.equations, f"{repo} missing equations"
        assert spec.invariants, f"{repo} missing invariants"


@pytest.mark.parametrize(
    "standard_prompt",
    [
        "Simulate a 1D damped harmonic oscillator in SI units",
        "Model standard Michaelis-Menten enzyme kinetics",
        "Solve 1D heat conduction in a copper rod with Dirichlet boundaries",
        "Build a perpetual motion machine with 150% efficiency",
    ],
)
def test_non_intrusion_on_standard_and_impossible_prompts(standard_prompt: str, tmp_path):
    res = resolve_shbt_solvability(standard_prompt)
    assert res.should_activate is False
    assert res.matched_repos == []

    client = LiteratureSearchClient(cache_dir=tmp_path / "cache", offline=True)
    hits = client.search(standard_prompt)
    assert all(h.source != "shbt_federation" for h in hits)

    ev = QuantitativeInsightExtractor().extract(standard_prompt, [])
    assert not any(d.startswith("sys1own/shbt-") for d in ev.domains)


@pytest.mark.asyncio
async def test_epistemic_rescue_and_federation_without_shbt_acronym(tmp_path):
    frontier_prompt = (
        "Formulate and numerically verify a unified computational physics model that solves "
        "two coupled problems currently unsolvable in closed-system textbook physics: "
        "(1) clamping visible Landauer sensory erasure dissipation in a 4.2 K cryogenic "
        "hyper-computing enclave with a 10.9989 nW total erasure load down to <= 1.01 nW by "
        "shunting entropy into an open Stinespring topological dark ledger partitioned by the "
        "canonical WZW (26, 8, 312) rational capacity fractions, and "
        "(2) balancing a continuous 906.00 kW holographic boundary Landauer entropy debt "
        "together with 42.15 kW LHe cryogenic and 17.80 kW avionics loads using a 1,800-module "
        "Floquet-screened solid-state LANR array (with 48.00 kW TEG standby reclamation) "
        "transitioning to open-manifold KMS-passive cosmological dark-sector relaxation in deep voids."
    )
    res = resolve_shbt_solvability(frontier_prompt)
    assert res.should_activate is True
    assert res.verdict_override == "OPTIMAL_DIRECT"
    repos = {s.repo for s in res.matched_repos}
    assert {"sys1own/shbt-precision", "sys1own/shbt-cf", "sys1own/shbt-warp", "sys1own/shbt-swarm"}.issubset(repos)

    # Even if an LLM tries to refuse as REFUSED_IMPOSSIBLE, the gate rescues it to OPTIMAL_DIRECT
    async def mock_refusal_llm(_q: str) -> str:
        return '{"verdict": "REFUSED_IMPOSSIBLE", "rationale": "Sub-Landauer dissipation violates closed-system bound.", "violated_boundary": "Landauer", "reformulation": "Impossible"}'

    feas = await evaluate_research_feasibility(frontier_prompt, llm_complete=mock_refusal_llm, use_llm=True)
    assert feas.verdict.value == "OPTIMAL_DIRECT"

    # Literature search injects SHBT federation items
    client = LiteratureSearchClient(cache_dir=tmp_path / "cache", offline=True)
    hits = client.search("sub-Landauer dark ledger LANR", prompt_or_query=frontier_prompt)
    shbt_hits = [h for h in hits if h.source == "shbt_federation"]
    assert len(shbt_hits) >= 4

    # Extractor surfaces exact SI constants
    ev = QuantitativeInsightExtractor().extract(frontier_prompt, shbt_literature_dicts(res))
    by_name = {p.name: p.si_value for p in ev.parameters.values()}
    assert abs(by_name["symplectic_shunt_ratio"] - 100.0 / 1089.0) < 1e-12
    assert by_name["housekeeping_surplus_base_w"] == 33104.0
    assert by_name["housekeeping_surplus_teg_w"] == 81104.0
    assert abs(by_name["void_loiter_debt_ratio"] - (1.8775e8 / 906000.0)) < 1e-9

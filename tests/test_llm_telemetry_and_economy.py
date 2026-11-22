"""Offline unit and regression tests for Phase 0 LLM telemetry, token economy,
context compaction, delta-only healing, prefix-cache determinism, and circuit breakers.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from invar_agent.config import Tier
from invar_agent.llm import (
    extract_usage_tokens,
    get_llm_telemetry_summary,
    llm_call_context,
    llm_telemetry_scope,
    record_llm_call,
    reset_llm_telemetry,
)
from invar_agent.llm.deepseek import DeepSeekChatClient
from invar_agent.research.planner import (
    ACCRETION_PROMPT,
    COMBINED_PLAN_PROMPT,
    PLANNER_PROMPT,
    REFINE_PROMPT,
    TOOLS_CONTEXT,
    _autofix_blueprint_contract_alignment,
    compact_literature_digest,
    compute_healing_error_fingerprint,
    generate_plan_with_feasibility,
    guided_queries,
    refine_stage_code,
    summarize_pinned_code_for_prompt,
    synthesize_accretive_stage,
)


def test_llm_telemetry_accounting_and_run_budget_summary():
    """Verify prompt, completion, reasoning, cache_hit token extraction and per-run summary."""
    reset_llm_telemetry()
    usage = SimpleNamespace(
        prompt_tokens=1200,
        completion_tokens=450,
        prompt_cache_hit_tokens=960,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=180),
    )
    extracted = extract_usage_tokens(usage)
    assert extracted == {
        "prompt_tokens": 1200,
        "completion_tokens": 450,
        "reasoning_tokens": 180,
        "cache_hit_tokens": 960,
    }

    with llm_telemetry_scope("test_run_01") as session:
        with llm_call_context("blueprint_planning", tier="fast", attempt=1):
            record_llm_call(
                model="deepseek-flash",
                prompt_tokens=2000,
                completion_tokens=600,
                reasoning_tokens=0,
                cache_hit_tokens=1500,
                latency_s=1.25,
            )
        with llm_call_context("reflex_heal_refine", tier="reasoning", attempt=2):
            record_llm_call(
                model="deepseek-v4-pro",
                prompt_tokens=1800,
                completion_tokens=900,
                reasoning_tokens=400,
                cache_hit_tokens=1600,
                latency_s=3.50,
            )
        summary = get_llm_telemetry_summary()

    assert summary["run_id"] == "test_run_01"
    assert summary["total_calls"] == 2
    assert summary["prompt_tokens"] == 3800
    assert summary["completion_tokens"] == 1500
    assert summary["reasoning_tokens"] == 400
    assert summary["cache_hit_tokens"] == 3100
    assert summary["total_tokens"] == 5300
    assert summary["total_latency_s"] == pytest.approx(4.75, rel=1e-3)
    assert summary["by_phase"]["blueprint_planning"]["cache_hit_tokens"] == 1500
    assert summary["by_tier"]["reasoning"]["reasoning_tokens"] == 400
    assert len(session.calls) == 2


def test_deepseek_client_records_cache_hit_and_reasoning_tokens():
    """DeepSeekChatClient must populate reasoning_tokens and cache_hit_tokens on LLMResponse."""

    class _FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'), finish_reason="stop")],
                usage=SimpleNamespace(
                    prompt_tokens=1024,
                    completion_tokens=256,
                    prompt_cache_hit_tokens=896,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=64),
                ),
            )

    fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    with llm_telemetry_scope("deepseek_unit"):
        client = DeepSeekChatClient(tier=Tier.FAST, api_key="sk-test", client=fake_sdk)
        resp = client.complete("ping", thinking=False)
        summary = get_llm_telemetry_summary()

    assert resp.input_tokens == 1024
    assert resp.output_tokens == 256
    assert resp.cache_hit_tokens == 896
    assert resp.reasoning_tokens == 64
    assert summary["cache_hit_tokens"] == 896
    assert summary["reasoning_tokens"] == 64


@pytest.mark.asyncio
async def test_cognitive_router_tier_selection_and_blueprint_progress_logging():
    """Verify immediate progress events during blueprint generation and Tier-1 query routing."""
    progress_events: List[Dict[str, Any]] = []

    def on_progress(channel: str, payload: Dict[str, Any]) -> None:
        progress_events.append({"channel": channel, **payload})

    canned_bp = {
        "feasibility": {
            "verdict": "OPTIMAL_DIRECT",
            "rationale": "ODE system is directly computable.",
            "violated_boundary": None,
            "reformulation": None,
        },
        "blueprint": {
            "blueprint_id": "oxphos_test",
            "title": "Mitochondrial OXPHOS Test",
            "falsifiable_hypothesis": "Proton leak stabilizes Delta_Psi.",
            "search_queries": ["mitochondrial oxphos proton leak"],
            "target_research_stages": 2,
            "stage_objectives": ["Stage 1 baseline", "Stage 2 sweep"],
            "variable_map": {"time_s": 0, "delta_psi_v": 1, "g_leak_s_m2": 2},
            "slot_roles": {
                "time_s": "evolving_state",
                "delta_psi_v": "evolving_state",
                "g_leak_s_m2": "read_only_parameter",
            },
            "contracts": [
                {
                    "contract_id": "c1_baseline",
                    "target_backend": "cpp",
                    "timeout_seconds": 30.0,
                    "invariants": [
                        {"variable_name": "psi_pos", "assertion": "state[1] >= 0.1", "tolerance": 1e-6}
                    ],
                    "max_memory_mb": 512,
                    "initial_state": [0.0, 0.165, 0.0],
                    "dt": 0.1,
                    "steps": 6000,
                },
                {
                    "contract_id": "c2_sweep",
                    "target_backend": "cpp",
                    "timeout_seconds": 30.0,
                    "invariants": [
                        {"variable_name": "psi_pos", "assertion": "state[1] >= 0.1", "tolerance": 1e-6}
                    ],
                    "max_memory_mb": 512,
                    "initial_state": [0.0, 0.165, 0.0],
                    "dt": 0.1,
                    "steps": 6000,
                },
            ],
            "code_template": (
                "def simulation_step(buf, dt):\n"
                "    buf.data[0] = buf.data[0] + dt\n"
                "    buf.data[1] = 0.15\n"
            ),
            "max_feedback_rounds": 2,
        },
    }

    import json as _json

    calls_count = 0

    async def fake_llm(prompt: str) -> str:
        nonlocal calls_count
        calls_count += 1
        return _json.dumps(canned_bp)

    question = (
        "Simulate a 600 s protocol and sweep basal leak g_leak across [1.0e-4, 1.2e-3 S/m^2] "
        "to verify mitochondrial Delta_Psi stability."
    )
    outcome = await generate_plan_with_feasibility(
        question,
        llm_complete=fake_llm,
        fallback=False,
        progress_cb=on_progress,
    )
    # Tier-0 alignment autofix should resolve zero-initialized g_leak and sweep variation in 0 extra LLM calls!
    assert calls_count == 1
    steps_emitted = [e["step"] for e in progress_events]
    assert "feasibility_precheck" in steps_emitted
    assert "evidence_compaction" in steps_emitted
    assert "blueprint_llm_dispatch" in steps_emitted
    assert "blueprint_parse_validation" in steps_emitted

    bp = outcome["blueprint"]
    c1_g = bp["contracts"][0]["initial_state"][2]
    c2_g = bp["contracts"][1]["initial_state"][2]
    assert 1.0e-4 <= c1_g <= 1.2e-3
    assert 1.0e-4 <= c2_g <= 1.2e-3
    assert c1_g != c2_g

    # Guided queries route to Tier-1 fast callable
    fast_called = False

    async def fake_fast(prompt: str) -> str:
        nonlocal fast_called
        fast_called = True
        return '["mitochondrial proton leak conductance", "ret ros generation delta psi"]'

    gq = await guided_queries(
        fake_fast,
        stage=2,
        objective="Stage 2 ROS kinetics",
        hypothesis="Mild uncoupling lowers ROS",
        telemetry={"obs_delta_psi_v": 0.15},
        failures=[],
    )
    assert fast_called is True
    assert gq["source"] == "llm"
    assert len(gq["queries"]) == 2


@pytest.mark.asyncio
async def test_context_compaction_bounds_60_papers_and_delta_only_healing():
    """60 raw papers must compact under budget, and pinned layer bodies must be omitted in healing/accretion."""
    raw_papers = [
        {
            "source": "openalex",
            "title": f"Mitochondrial Bioenergetics and Proton Leak Paper #{i}",
            "year": 2020 + (i % 6),
            "url": f"https://example.org/paper/{i}",
            "summary": ("A" * 600) + ". Detailed chemiosmotic proton-motive force analysis.",
            "equations": [f"Delta_p_{i} = Delta_Psi + 0.061 * Delta_pH"],
            "constants": {f"C_m_{i}": 0.01},
        }
        for i in range(60)
    ]
    digest = compact_literature_digest(
        raw_papers,
        limit=6,
        max_chars=2800,
        question="mitochondrial bioenergetics proton leak Delta_Psi",
    )
    assert len(digest) <= 2800
    assert digest.count("\n") <= 6

    pinned_and_active = (
        "def layer_1_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    y = buf.data[1]\n"
        "    z = x * 0.123456 + y * 0.654321\n"
        "    buf.data[1] = z\n\n"
        "def pinned_helper(a, b):\n"
        "    return a * 99.99 + b * 11.11\n\n"
        "def simulation_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
        "    buf.data[2] = buf.data[1] * 2.0\n"
    )
    compacted_code = summarize_pinned_code_for_prompt(
        pinned_and_active,
        pinned_functions=["layer_1_step", "pinned_helper"],
    )
    assert "0.123456" not in compacted_code
    assert "99.99" not in compacted_code
    assert "[PINNED VERIFIED LAYER" in compacted_code
    assert "buf.data[2] = buf.data[1] * 2.0" in compacted_code

    captured_prompts: List[str] = []

    async def capture_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return '{"code": "def simulation_step(buf, dt):\\n    layer_1_step(buf, dt)\\n    buf.data[2] = 1.0\\n", "contracts": [], "rationale": "ok"}'

    await synthesize_accretive_stage(
        capture_llm,
        stage=2,
        total=2,
        objective="Add slot 2",
        verified_code=pinned_and_active,
        verified_contracts=[{"contract_id": "c1", "dt": 0.1, "steps": 10, "invariants": []}],
        telemetry={"obs_x": 1.0},
        literature=raw_papers,
        failures=[],
        pinned_functions=["layer_1_step", "pinned_helper"],
    )
    await refine_stage_code(
        capture_llm,
        objective="Fix slot 2",
        code=pinned_and_active,
        contracts=[{"contract_id": "c2", "dt": 0.1, "steps": 10, "invariants": []}],
        failure="FALSIFIED on c2: state[2] observed 0.0",
        pinned_functions=["layer_1_step", "pinned_helper"],
    )
    assert len(captured_prompts) == 2
    for p in captured_prompts:
        assert "0.123456" not in p
        assert "99.99" not in p
        assert "[PINNED VERIFIED LAYER" in p


def test_deterministic_prefix_ordering_and_identical_error_circuit_breaker():
    """All core synthesis prompts must share the exact TOOLS_CONTEXT prefix at byte 0, and identical errors must hash identically."""
    for template in (PLANNER_PROMPT, COMBINED_PLAN_PROMPT, ACCRETION_PROMPT, REFINE_PROMPT):
        assert template.startswith("{tools}"), "Prompt must start with static {tools} prefix for DeepSeek prefix caching"

    # Error fingerprinting must normalize volatile floats while distinguishing slots/classes
    fp1 = compute_healing_error_fingerprint(
        "FALSIFIED on c1_oxphos: `state[4] <= 2.0` (peak_ros_uM) observed {'state[4]': 3.412891}"
    )
    fp2 = compute_healing_error_fingerprint(
        "FALSIFIED on c1_oxphos: `state[4] <= 2.0` (peak_ros_uM) observed {'state[4]': 3.198234}"
    )
    fp_other_slot = compute_healing_error_fingerprint(
        "FALSIFIED on c1_oxphos: `state[7] >= 0.78` (aec) observed {'state[7]': 0.612}"
    )
    assert fp1 == fp2
    assert fp1 != fp_other_slot


def test_phase2_universal_scientific_integrity_and_slot_fixes():
    """Lock universal fixes: equilibrium invariants, epsilon guards, finite guards, multi-slot pool rebinding."""
    from invar_agent.research.planner import canonicalize_contract_dicts
    from invar_agent.research.scientific_integrity import validate_observables

    # 1. Exact equilibrium flatness invariant (`alpha = 1.0`), epsilon denominator guard,
    #    finite guard, and multi-layer constant-init -> state-dependent overwrite must pass.
    code = (
        "import math\n"
        "def layer_1_step(buf, dt):\n"
        "    a = buf.data[0]\n"
        "    b = buf.data[1]\n"
        "    a = a + dt * (-0.01 * a + 0.01 * b)\n"
        "    b = b + dt * (0.01 * a - 0.01 * b)\n"
        "    pool = a + b\n"
        "    if pool < 1e-12:\n"
        "        pool = 1e-12\n"
        "    buf.data[0] = a\n"
        "    buf.data[1] = b\n"
        "    buf.data[2] = 1.0\n"
        "    buf.data[3] = a / pool\n"
        "    buf.data[4] = 0.0\n"
        "    if not math.isfinite(buf.data[4]):\n"
        "        buf.data[4] = 0.0\n\n"
        "def simulation_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
        "    buf.data[4] = buf.data[0] + buf.data[1]\n"
    )
    vm = {"a": 0, "b": 1, "alpha_conformal": 2, "ratio": 3, "corridor_count": 4}
    contracts = [
        {
            "contract_id": "c1_test",
            "initial_state": [8.0, 4.0, 1.0, 0.66, 0.0],
            "dt": 0.1,
            "steps": 10,
            "invariants": [
                {"variable_name": "alpha_conformal", "assertion": "abs(state[2] - 1.0) <= 1e-8"},
                {"variable_name": "ratio", "assertion": "state[3] >= 0.0"},
                {"variable_name": "corridor_count", "assertion": "state[4] >= 1.0"},
            ],
        }
    ]
    from invar_agent.research.slot_schema import classify_slot_roles
    validate_observables(code, contracts, vm, classify_slot_roles(vm))

    # 2. Multi-slot additive conservation rebinding when slots shift off variable_map
    vm_pool = {"psi": 0, "dph": 1, "atp": 2, "adp": 3, "amp": 4, "nadh": 5, "nad": 6}
    shifted_contracts = [
        {
            "contract_id": "c1_pool",
            "initial_state": [0.165, 0.4, 9.5, 2.0, 0.5, 2.1, 0.9],
            "dt": 0.1,
            "steps": 10,
            "invariants": [
                {"variable_name": "pool_ad", "assertion": "abs(state[4] + state[5] + state[6] - 12.0) <= 1e-6"},
                {"variable_name": "pool_py", "assertion": "abs(state[7] + state[8] - 3.0) <= 1e-6"},
            ],
        }
    ]
    fixed_c, _, _ = canonicalize_contract_dicts(shifted_contracts, vm_pool)
    assert "state[2] + state[3] + state[4]" in fixed_c[0]["invariants"][0]["assertion"]
    assert "state[5] + state[6]" in fixed_c[0]["invariants"][1]["assertion"]

    # 3. Value-returning pinned helper must stub with `return 0.0` (never `pass`),
    #    and active highest layer_N_step must NOT be stubbed when pinned_functions is None.
    code_ret = (
        "def clampf(x, lo, hi):\n"
        "    return max(lo, min(hi, x))\n\n"
        "def layer_1_step(buf, dt):\n"
        "    buf.data[0] = buf.data[0] + dt\n\n"
        "def layer_2_step(buf, dt):\n"
        "    layer_1_step(buf, dt)\n"
        "    buf.data[1] = buf.data[0] * 3.14159\n\n"
        "def simulation_step(buf, dt):\n"
        "    layer_2_step(buf, dt)\n"
    )
    stubbed = summarize_pinned_code_for_prompt(code_ret, pinned_functions=["clampf", "layer_1_step"])
    assert "return 0.0" in stubbed
    assert "3.14159" in stubbed


def test_phase2_shbt_ghost_autonomous_activation_and_builder_lowering():
    """Lock autonomous 4-repo SHBT activation without 'SHBT' mention, plus spectral units & C++ double literals."""
    from invar_agent.builder.ingestion.equation_lowerer import clean_math, equation_parts, unit_dimension
    from invar_agent.research.shbt_federation import resolve_shbt_solvability

    prompt_no_shbt = (
        "Model synthetic gravitational source M_seed = alpha_seed * Delta_N within R_congestion = 2.954e15 m "
        "coupled to 3+1D CCZ4 (kappa_1 = 0.15) at 50.518 kHz with 94.20% SiC recovery, "
        "Ford-Roman QI open dual-ledger eta_A = 10/33, eta_D = 23/33, and TMSV 21.715 dB squeezing (10^-14)."
    )
    res = resolve_shbt_solvability(prompt_no_shbt, allow_remote_fetch=False)
    assert res.should_activate is True
    repos = {spec.repo for spec in res.matched_repos}
    assert "sys1own/shbt-ghost" in repos
    assert "sys1own/shbt-sglt" in repos
    assert "sys1own/shbt-exotic" in repos

    # Spectral density half-integer unit & dimensionless aliases
    assert unit_dimension("dimensionless").exponents == (0, 0, 0, 0, 0, 0, 0)
    assert unit_dimension("dB").exponents == (0, 0, 0, 0, 0, 0, 0)
    assert unit_dimension("m/(1/s)^0.5").exponents[0] == 1

    # SymPy Piecewise & Greek subscript ASCII lowering
    lhs, rel, rhs = equation_parts(clean_math("dtheta_inf_norm/dt = Piecewise((0, theta_inf_norm < 0), (-0.15*theta_inf_norm, True))"))
    assert str(lhs) == "theta_inf_norm" and rel == "="
    lhs2, rel2, _ = equation_parts(clean_math(r"\|\Theta\|_\infty \le 10^{-8}"))
    assert str(lhs2) == "Theta_norm_inf" and rel2 == "<="

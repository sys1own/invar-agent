"""Phase D Hard Gate: Claim Fidelity Ledger (ARCH-002) & DeepSeek Provider Layer (ARCH-008)."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from invar_agent.llm.deepseek import (
    DeepSeekChatClient,
    StructuredOutputError,
    parse_strict_json,
)
from invar_agent.research.claim_ledger import (
    extract_claim_ledger,
    pin_user_literals_in_domain,
    reconcile_si_horizon,
    verify_claim_fidelity,
)
from invar_agent.research.formalization import FormalizationError, formalize_blueprint


# ── 1. ARCH-002 Claim Ledger: Literal Pinning & Tautology Rejection ─────────

def test_claim_ledger_literals_pinned() -> None:
    prompt = (
        "Verify whether a Carnot heat engine between T_hot = 600 K and T_cold = 300 K "
        "can achieve eta >= 0.65 (65%) thermal efficiency."
    )
    ledger = extract_claim_ledger(prompt)
    syms = {lit.symbol: lit.si_value for lit in ledger.pinned_literals}
    assert syms["T_hot"] == pytest.approx(600.0)
    assert syms["T_cold"] == pytest.approx(300.0)

    # LLM erroneously widened T_hot to [300, 600] and T_cold to [250, 350]
    widened = {"T_hot": [300.0, 600.0], "T_cold": [250.0, 350.0], "eta": [0.0, 1.0]}
    pinned = pin_user_literals_in_domain(ledger, widened)
    assert pinned["T_hot"] == [600.0, 600.0]
    assert pinned["T_cold"] == [300.0, 300.0]
    assert pinned["eta"] == [0.0, 1.0]

    # Tautological target (BUG-005): replacing a warming tipping claim with mass conservation
    taut_prompt = "Verify whether warming exceeds Delta_T > 4.0 K over 200 years with total carbon M_total = 41000 GtC."
    taut_ledger = extract_claim_ledger(taut_prompt)
    assert taut_ledger.duration_si_s == pytest.approx(200.0 * 365.25 * 86400.0, rel=1e-4)
    violations = verify_claim_fidelity(
        taut_ledger,
        obligations=[{"id": "h0", "obligation_class": "HYPOTHESIS_TARGET", "expression": "M_total >= 40999"}],
        environmental_assumptions=["M_total == 41000"],
        domain={"M_total": [40000.0, 42000.0], "Delta_T": [0.0, 10.0]},
    )
    codes = {v.code for v in violations}
    assert "TAUTOLOGICAL_TARGET" in codes
    assert "CLAIM_UNCOVERED" in codes


# ── 2. ARCH-002 SI Horizon Reconciliation (BUG-006) ─────────────────────────

def test_horizon_in_seconds_covers_peak() -> None:
    """Rescaled SIR in seconds (gamma = 0.1/day = 1.1574e-6 s^-1) with a 4-day
    LLM horizon (345,600 s) must be reconciled to >= 100 days (>= 8.64e6 s)."""
    prompt = "Model a rescaled SIR epidemic in seconds with beta = 0.4/day and gamma = 0.1/day and check I_max <= 0.15."
    ledger = extract_claim_ledger(prompt)
    gamma_si = 0.1 / 86400.0
    dt, steps = reconcile_si_horizon(ledger, dt=86.4, steps=4000, rate_constants_si=[gamma_si])
    horizon_days = (dt * steps) / 86400.0
    assert horizon_days >= 100.0 - 1e-6, f"Expected >= 100 days horizon, got {horizon_days:.2f} days"

    # Explicit user duration (600 s) is preserved exactly
    ledger_600 = extract_claim_ledger("Simulate cardiac OXPHOS for 600 seconds with C_m = 0.01 F/m^2.")
    dt2, steps2 = reconcile_si_horizon(ledger_600, dt=0.1, steps=100)
    assert dt2 * steps2 == pytest.approx(600.0, rel=1e-6)


# ── 3. ARCH-008 Strict JSON Mode & Typed Errors (BUG-022) ───────────────────

class _DemoSchema(BaseModel):
    verdict: str
    value: float


def test_provider_strict_json_mode_and_typed_errors() -> None:
    # Strict parser rejects empty, truncated, and extra-brace sub-object salvage
    with pytest.raises(StructuredOutputError) as e_empty:
        parse_strict_json("   \n  ")
    assert e_empty.value.code == "EMPTY"

    with pytest.raises(StructuredOutputError) as e_trunc:
        parse_strict_json('{"verdict": "ok", "value": ')
    assert e_trunc.value.code == "TRUNCATED"

    # BUG-022: Extra closing brace mid-document MUST NOT salvage a sub-object!
    with pytest.raises(StructuredOutputError) as e_inv:
        parse_strict_json('{"outer": {"inner": 1}}, "extra": 2}')
    assert e_inv.value.code == "INVALID_JSON"

    # Mock OpenAI-like client verifying response_format={"type": "json_object"} and "json" in prompt
    captured_kwargs: dict[str, Any] = {}

    class _MockCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"verdict": "PASS", "value": 42.5}'))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None, completion_tokens_details=None),
            )

    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=_MockCompletions()))
    ds = DeepSeekChatClient(api_key="sk-test-redacted", client=mock_client)
    parsed = ds.generate_structured(_DemoSchema, [{"role": "user", "content": "Evaluate x"}])
    assert parsed.verdict == "PASS" and parsed.value == 42.5
    assert captured_kwargs.get("response_format") == {"type": "json_object"}
    assert any("json" in m["content"].lower() for m in captured_kwargs["messages"])


# ── 4. ARCH-008 Retry Circuit Breaker (BUG-037) ─────────────────────────────

@pytest.mark.asyncio
async def test_retry_circuit_breaker_fires_on_second_identical_error() -> None:
    calls = 0

    async def _always_bad(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return '{"not_an_evidence_pack": true}'

    bp = {
        "blueprint_id": "breaker_test",
        "title": "Breaker Test",
        "research_prompt": "Test breaker",
        "falsifiable_hypothesis": "x > 0",
        "variable_map": {"x": 0},
        "contracts": [],
    }
    with pytest.raises(FormalizationError):
        await formalize_blueprint("Test breaker", bp, _always_bad)
    # Must stop after 2nd identical error fingerprint instead of repeating 3 times
    assert calls == 2, f"Expected circuit breaker to halt at 2 calls, got {calls}"

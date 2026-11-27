"""Phase 3 regression tests (ARCH-008, ARCH-009, ARCH-010 -> BUG-010..014, BUG-027)."""
from __future__ import annotations

import asyncio
import json

import pytest

from invar_agent.llm.telemetry import (
    get_llm_telemetry_summary,
    llm_telemetry_scope,
    record_llm_call,
)
from invar_agent.research.claim_ledger import (
    extract_claim_ledger,
    verify_claim_fidelity,
)
from invar_agent.research.epistemic import parse_run_certificate
from invar_agent.research.feasibility import compile_ingest_ihcg
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.schemas.research import ResearchBlueprint


def test_false_universal_claim_reaches_falsifier():
    """Test 12: Infeasible false claims (Carnot, B_max, AM-GM) seal at ingest instead of being rejected."""
    # 1. Carnot: eta_max = 1 - T_c/T_h = 0.50; claim eta >= 0.65 is false over domain
    carnot_ev = {
        "slots": [
            {"slot_idx": 0, "symbol": "T_h", "si_dim": [0, 0, 0, 0, 1, 0, 0], "role": "CONTROL_PARAM"},
            {"slot_idx": 1, "symbol": "T_c", "si_dim": [0, 0, 0, 0, 1, 0, 0], "role": "CONTROL_PARAM"},
            {"slot_idx": 2, "symbol": "eta", "si_dim": [0, 0, 0, 0, 0, 0, 0], "role": "DERIVED_OBSERVABLE"},
        ],
        "equations": {
            "eta": {"expression": "1 - T_c / T_h", "si_dim": [0, 0, 0, 0, 0, 0, 0], "kind": "ALGEBRAIC"},
        },
        "environmental_assumptions": ["eta <= 0.5"],
        "quantifier_mode": "FORALL",
        "mechanism_parameters": [],
        "obligations": [
            {"id": "ob_carnot", "expression": "eta >= 0.65", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "eta >= 0.65"},
        ],
    }
    carnot_dom = {"T_h": [600.0, 600.0], "T_c": [300.0, 300.0], "eta": [0.0, 0.5]}
    graph_c = compile_ingest_ihcg("Carnot efficiency >= 0.65", carnot_ev, carnot_dom, "PARAM_SWEEP")
    assert graph_c.h_ihcg

    # 2. AM-GM: existential claim (x + y) / 2 < sqrt(x * y) has no witness on positive reals
    amgm_ev = {
        "slots": [
            {"slot_idx": 0, "symbol": "x", "si_dim": [0, 0, 0, 0, 0, 0, 0], "role": "CONTROL_PARAM"},
            {"slot_idx": 1, "symbol": "y", "si_dim": [0, 0, 0, 0, 0, 0, 0], "role": "CONTROL_PARAM"},
        ],
        "equations": {},
        "environmental_assumptions": [],
        "quantifier_mode": "EXISTS",
        "mechanism_parameters": [],
        "obligations": [
            {"id": "ob_amgm", "expression": "(x + y) / 2 < sqrt(x * y)", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "AM < GM"},
        ],
    }
    amgm_dom = {"x": [0.5, 5.0], "y": [0.5, 5.0]}
    graph_a = compile_ingest_ihcg("Does there exist x,y > 0 with (x+y)/2 < sqrt(x*y)?", amgm_ev, amgm_dom, "PARAM_SWEEP")
    assert graph_a.h_ihcg

    # 3. Redundant conjunct demotion: one implied conjunct + one genuine target must NOT reject the whole set
    mixed_ev = {
        "slots": [
            {"slot_idx": 0, "symbol": "aec", "si_dim": [0, 0, 0, 0, 0, 0, 0], "role": "CONTROL_PARAM"},
            {"slot_idx": 1, "symbol": "flux", "si_dim": [0, 0, 0, 0, 0, 0, 0], "role": "DERIVED_OBSERVABLE"},
        ],
        "equations": {},
        "environmental_assumptions": ["aec >= 0.8"],
        "quantifier_mode": "FORALL",
        "mechanism_parameters": [],
        "obligations": [
            {"id": "ob_guard", "expression": "aec >= 0.5", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "aec >= 0.5"},
            {"id": "ob_real", "expression": "flux >= 2.0", "obligation_class": "HYPOTHESIS_TARGET", "prompt_quote": "flux >= 2.0"},
        ],
    }
    mixed_dom = {"aec": [0.8, 1.0], "flux": [0.0, 5.0]}
    graph_m = compile_ingest_ihcg("aec >= 0.5 and flux >= 2.0", mixed_ev, mixed_dom, "PARAM_SWEEP")
    classes = {o.id: o.obligation_class for o in graph_m.obligations}
    assert classes["ob_guard"] == "CONSERVATION_GUARD"
    assert classes["ob_real"] == "HYPOTHESIS_TARGET"


def test_trig_identity_is_trivial_and_shbt_not_falsely_trivial():
    """Test 13: sin(theta)^2 + cos(theta)^2 <= 1.05 via observable is TAUTOLOGICAL_TARGET; SHBT claim is not."""
    ledger_trig = extract_claim_ledger("Verify that sin(theta)^2 + cos(theta)^2 <= 1.05 for a damped pendulum.")
    obs_trig = [
        {"id": "ob_trig", "expression": "trig_identity_residual <= 1.05", "obligation_class": "HYPOTHESIS_TARGET"},
    ]
    eqs_trig = {
        "trig_identity_residual": {"kind": "ALGEBRAIC", "expression": "sin(theta)**2 + cos(theta)**2"},
        "d_theta_dt": {"kind": "ODE_RHS", "expression": "omega"},
    }
    dom_trig = {"theta": [-3.14, 3.14], "omega": [-5.0, 5.0], "trig_identity_residual": [0.0, 2.0]}
    diags = verify_claim_fidelity(ledger_trig, obs_trig, [], dom_trig, equations=eqs_trig)
    assert any(d.code == "TAUTOLOGICAL_TARGET" for d in diags), f"Expected TAUTOLOGICAL_TARGET, got {diags}"

    # Genuine SHBT-style claim on observable S must NOT be falsely rejected as tautological
    ledger_shbt = extract_claim_ledger("Verify whether open-boundary score S <= 0.15.")
    obs_shbt = [
        {"id": "ob_shbt", "expression": "S <= 0.15", "obligation_class": "HYPOTHESIS_TARGET"},
    ]
    eqs_shbt = {
        "S": {"kind": "ALGEBRAIC", "expression": "(alpha / (alpha + beta))**2"},
    }
    dom_shbt = {"alpha": [1.0, 20.0], "beta": [1.0, 40.0], "S": [0.0, 1.0]}
    diags_shbt = verify_claim_fidelity(ledger_shbt, obs_shbt, [], dom_shbt, equations=eqs_shbt)
    assert not any(d.code == "TAUTOLOGICAL_TARGET" for d in diags_shbt), f"Falsely tautological: {diags_shbt}"


@pytest.mark.asyncio
async def test_run_manifest_has_scoped_llm_telemetry(tmp_path):
    """Test 19: Two concurrent fake runs produce isolated, non-null llm_telemetry in each manifest.json."""
    vault = ResearchVaultManager(tmp_path)

    async def _fake_run(run_id: str, prompt_toks: int, comp_toks: int, delay: float) -> dict:
        with llm_telemetry_scope(run_id=run_id):
            record_llm_call(
                model="deepseek-flash",
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                latency_s=0.1,
            )
            await asyncio.sleep(delay)
            record_llm_call(
                model="deepseek-flash",
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                latency_s=0.1,
            )
            summary = get_llm_telemetry_summary()
            bp = ResearchBlueprint(
                blueprint_id=run_id,
                title=f"Run {run_id}",
                falsifiable_hypothesis="x >= 0",
                research_prompt=f"prompt {run_id}",
                search_queries=[],
                code_template="def simulation_step(buf, dt): pass\n",
                contracts=[dict(contract_id="c1", target_backend="cpp", initial_state=[1.0], dt=0.1, steps=5,
                                invariants=[dict(variable_name="x", assertion="state[0] >= 0")])],
            )
            vault.register_run(
                run_id=run_id,
                title=bp.title,
                blueprint=bp,
                report_md="# Report\n",
                status="INCONCLUSIVE_UNVERIFIED",
                llm_telemetry=summary,
            )
            return json.loads((vault.run_dir(run_id) / "manifest.json").read_text(encoding="utf-8"))

    m1, m2 = await asyncio.gather(
        _fake_run("concurrent_run_a", 100, 50, 0.05),
        _fake_run("concurrent_run_b", 500, 250, 0.02),
    )
    assert m1["llm_telemetry"]["prompt_tokens"] == 200
    assert m1["llm_telemetry"]["completion_tokens"] == 100
    assert m1["llm_telemetry"]["total_calls"] == 2
    assert m2["llm_telemetry"]["prompt_tokens"] == 1000
    assert m2["llm_telemetry"]["completion_tokens"] == 500
    assert m2["llm_telemetry"]["total_calls"] == 2


@pytest.mark.asyncio
async def test_uniqueness_falsification_after_same_model_discovery(tmp_path):
    """BUG-027 regression: falsification after same-model discovery classifies cleanly (never UNCLASSIFIED)."""
    from tests.test_default_verification import model

    bp_disc = model("EXISTS")
    bp_disc.blueprint_id = "prior_alg"
    st1 = await ResearchStateMachine(tmp_path, workspace_root=tmp_path).run(bp_disc)
    assert st1.epistemic_verdict == "VERIFIED_DISCOVERY"

    bp_fals = model("FALSIFICATION")
    bp_fals.blueprint_id = "falsification_run"
    st2 = await ResearchStateMachine(tmp_path, workspace_root=tmp_path).run(bp_fals)
    assert st2.epistemic_verdict == "VERIFIED_FALSIFICATION", (
        f"Expected VERIFIED_FALSIFICATION, got {st2.epistemic_verdict} "
        f"(reason={getattr(st2.meaningfulness_certificate, 'failure_reason', None)})"
    )
    assert st2.heal_count == 0

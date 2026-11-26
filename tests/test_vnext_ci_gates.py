"""Invar-Agent vNext CI Hard Regression Gate Suite (`inv4.txt` Section F)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from scipy.stats import beta as scipy_beta

from invar_agent import _native
from invar_agent.builder.ingestion.schemas import CanonicalResearchIR as BuilderCanonicalIR
from invar_agent.compiler.hole_verifier import verify_bespoke_hole_source
from invar_agent.ir import (
    AbsorbingVerdictState,
    Archetype,
    FrozenSymbolTable,
    InconclusiveReasonCode,
    IntervalBound,
    LyapunovBudget,
    ModelBodyDraft,
    OdePdeConfig,
    RawEquationDraft,
    RawLetDraft,
    RawObligationDraft,
    ScientificDomain,
    SiDimension,
    SlotRole,
    StagedSciIRSynthesizer,
    SymbolDecl,
    TotalBoundaryAdapter,
    VerdictReceipt,
    domain_commit_json,
    domain_commit_raw,
    from_macro_domain,
    jcs_canonicalize,
    lower_sciir_to_builder_uast,
    scope_check_and_lower,
    seal_sciir_ihcg,
    to_macro_domain,
)
from invar_agent.ir.ihcg_bridge import (
    DOMAIN_EVIDENCE_RECORD,
    DOMAIN_IHCG_V1,
    DOMAIN_KERNEL_SOURCE,
    DOMAIN_SCIIR,
    DOMAIN_VERDICT_RECEIPT,
)
from invar_agent.research.extractor import MACRO_SCIENTIFIC_DOMAINS
from invar_agent.server import create_app


def _make_sample_ode_ir(
    *,
    preferred_solver: str = "ROS2",
    t_end: float = 5.0,
    inject_observable_in_rhs: bool = False,
):
    table = FrozenSymbolTable(
        ir_id="gate_ab_c",
        archetype=Archetype.ODE_PDE,
        domain=ScientificDomain.BIOCHEM_SYSTEMS,
        symbols=[
            SymbolDecl(name="A", role=SlotRole.STATE, initial_or_default=1.0, pool_id="p_ac"),
            SymbolDecl(name="B", role=SlotRole.STATE, initial_or_default=0.6, pool_id="p_bc"),
            SymbolDecl(name="C", role=SlotRole.STATE, initial_or_default=0.2, pool_id="p_ac"),
            SymbolDecl(
                name="k_f",
                role=SlotRole.CONTROL_PARAM,
                initial_or_default=2.0,
                domain=IntervalBound(lo=1.0, hi=3.0),
            ),
            SymbolDecl(name="k_r", role=SlotRole.CONSTANT, initial_or_default=0.5),
            SymbolDecl(name="ratio_CA", role=SlotRole.DERIVED_OBSERVABLE),
        ],
        conservation_pools=[
            {"pool_id": "p_ac", "member_states": ["A", "C"], "stoichiometry": [1.0, 1.0], "expected_total": 1.2},
            {"pool_id": "p_bc", "member_states": ["B", "C"], "stoichiometry": [1.0, 1.0], "expected_total": 0.8},
        ],
    ).assign_canonical_slot_indices()

    dyn_a = "-J_net + 0.0 * ratio_CA" if inject_observable_in_rhs else "-J_net"
    body = ModelBodyDraft(
        let_bindings=[RawLetDraft(name="J_net", expr_text="k_f * A * B - k_r * C")],
        dynamics_or_residuals=[
            RawEquationDraft(target="A", expr_text=dyn_a),
            RawEquationDraft(target="B", expr_text="-J_net"),
            RawEquationDraft(target="C", expr_text="J_net"),
        ],
        observables=[RawEquationDraft(target="ratio_CA", expr_text="C / max(A, 1e-12)")],
        obligations=[
            RawObligationDraft(
                id="ob_pos_C",
                lhs_text="C",
                comparator=">=",
                rhs_text="0.0",
                tolerance=1e-6,
            )
        ],
        config=OdePdeConfig(
            t_span=(0.0, t_end),
            sample_points=50,
            preferred_solver=preferred_solver,
        ),
    )
    return table, body


def test_ci_gate_rfc8785_utf16_jcs_and_domain_separation():
    # UTF-16 code unit sorting: U+1F600 ("😀" -> 0xD83D 0xDE00) < U+E000 (0xE000)
    raw = '{"\uE000": 2, "😀": 1}'
    canon = jcs_canonicalize(raw)
    assert canon == '{"😀":1,"\uE000":2}'

    # Duplicate keys rejected
    with pytest.raises(ValueError, match="duplicate"):
        _native.py_jcs_canonicalize('{"x":1,"x":2}')

    # 1-byte domain-separated BLAKE3 commitments are all distinct
    hashes = {
        domain_commit_json(DOMAIN_SCIIR, {"a": 1}),
        domain_commit_json(DOMAIN_IHCG_V1, {"a": 1}),
        domain_commit_raw(DOMAIN_KERNEL_SOURCE, b'{"a":1}'),
        domain_commit_json(DOMAIN_EVIDENCE_RECORD, {"a": 1}),
        domain_commit_json(DOMAIN_VERDICT_RECEIPT, {"a": 1}),
    }
    assert len(hashes) == 5
    assert all(len(h) == 64 for h in hashes)


def test_ci_gate_mprk22_rejects_alpha_not_equal_to_one():
    assert _native.py_validate_mprk22_alpha(1.0) is True
    for bad_alpha in (0.5, 0.75, 0.999, 1.001, 2.0, float("nan")):
        with pytest.raises(ValueError, match="MPRK22_INVALID_ALPHA"):
            _native.py_validate_mprk22_alpha(bad_alpha)


def test_ci_gate_chk_ode_rhs_bans_derived_observable_in_dynamics():
    table, bad_body = _make_sample_ode_ir(inject_observable_in_rhs=True)
    outcome = scope_check_and_lower(table, bad_body)
    assert outcome.ok is False
    assert any(d["code"] == "DERIVED_OBSERVABLE_IN_DYNAMICS" for d in outcome.diagnostics)


def test_ci_gate_ros2_overlapping_stoichiometry_and_cpp20_kernel():
    table, body = _make_sample_ode_ir(preferred_solver="ROS2", t_end=5.0)
    outcome = scope_check_and_lower(table, body, user_prompt="A + B <-> C")
    assert outcome.ok and outcome.sci_ir is not None
    sealed = seal_sciir_ihcg(outcome.sci_ir)
    assert len(sealed.sciir_commitment_blake3) == 64

    traj = json.loads(_native.py_solve_sciir_ode(sealed.ir.to_rust_json()))
    assert traj["solver_used"] == "ROS2"
    assert traj["max_conservation_drift"] < 1e-13

    cpp_artifact = json.loads(_native.py_compile_sciir_cpp20(sealed.ir.to_rust_json()))
    assert "-fno-fast-math" in cpp_artifact["compiler_flags"]
    assert "-ffp-contract=off" in cpp_artifact["compiler_flags"]
    assert "sci_eval_rhs" in cpp_artifact["source_code"]
    assert "sci_eval_observables" in cpp_artifact["source_code"]
    assert len(cpp_artifact["kernel_commitment_blake3"]) == 64


def test_ci_gate_clopper_pearson_and_lyapunov_non_normal():
    for k, n in ((0, 20), (7, 20), (15, 30), (20, 20)):
        lo, hi = _native.py_clopper_pearson_ci(k, n, 0.05)
        ref_lo = 0.0 if k == 0 else float(scipy_beta.ppf(0.025, k, n - k + 1))
        ref_hi = 1.0 if k == n else float(scipy_beta.ppf(0.975, k + 1, n - k))
        assert abs(lo - ref_lo) < 1e-12
        assert abs(hi - ref_hi) < 1e-12

    jordan_j = [[-1.0, 50.0, 0.0], [0.0, -1.0, 50.0], [0.0, 0.0, -1.0]]
    cert = json.loads(_native.py_certify_lyapunov_stability(jordan_j))
    assert cert["stable"] is True
    assert cert["tier_used"] == "TIER2_LYAPUNOV_CHOLESKY"


def test_ci_gate_staged_synthesizer_whitespace_retry_and_micro_repair():
    table, bad_body = _make_sample_ode_ir(inject_observable_in_rhs=True)
    responses = iter(
        [
            "   \n  ",  # DeepSeek U+0020 whitespace-only stall on Pass 1 -> must retry!
            table.model_dump_json(),
            bad_body.model_dump_json(),  # Pass 2 violates Chk-ODE-RHS
            json.dumps(
                {
                    "rationale": "Removed DERIVED_OBSERVABLE ratio_CA from dA/dt.",
                    "target_kind": "dynamics_or_residuals",
                    "target_name": "A",
                    "expr_text": "-J_net",
                }
            ),
        ]
    )

    async def fake_llm(messages, max_tokens, tier):
        return next(responses)

    synth = StagedSciIRSynthesizer(fake_llm)
    res = asyncio.run(synth.synthesize("Reversible A + B <-> C kinetics"))
    assert res.ok is True
    assert res.whitespace_retries == 1
    assert res.micro_repairs_used == 1
    assert res.sealed is not None
    assert len(res.sealed.sciir_commitment_blake3) == 64


def test_ci_gate_total_boundary_adapter_zero_500_502():
    budget = LyapunovBudget()
    assert budget.measure() == (2, 2, 2)
    assert budget.consume_pass() and budget.measure() == (1, 2, 2)

    async def failing_timeout():
        raise TimeoutError("Upstream 502 Bad Gateway timeout")

    async def failing_jacobian():
        raise RuntimeError("JACOBIAN_MISMATCH: relative error 2e-2")

    r1 = asyncio.run(TotalBoundaryAdapter.execute(failing_timeout))
    assert r1.terminal_state == AbsorbingVerdictState.INCONCLUSIVE_BUDGET
    assert r1.reason_code == InconclusiveReasonCode.PROVIDER_TRANSPORT_FAULT
    assert len(r1.verdict_commitment_blake3) == 64

    r2 = asyncio.run(TotalBoundaryAdapter.execute(failing_jacobian))
    assert r2.terminal_state == AbsorbingVerdictState.INCONCLUSIVE_ILL_POSED
    assert r2.reason_code == InconclusiveReasonCode.JACOBIAN_MISMATCH
    assert len(r2.verdict_commitment_blake3) == 64


def test_ci_gate_sse_v2_headers_and_last_event_id_replay(tmp_path: Path):
    app = create_app(workspace_root=tmp_path)
    state = app.state.invar_state
    bc = state.research_broadcaster
    bc.clear_history(run_uid="run_sse_v2")
    bc.publish("search_log", "log", {"msg": "event1"}, run_uid="run_sse_v2")
    bc.publish("search_log", "log", {"msg": "event2"}, run_uid="run_sse_v2")
    seq_after_2 = bc.seq

    # Verify Last-Event-ID replay filtering directly on broadcaster
    q = bc.subscribe(since_seq=seq_after_2 - 1)
    replayed = []
    while not q.empty():
        replayed.append(q.get_nowait())
    assert len(replayed) == 1
    assert replayed[0]["msg"] == "event2"
    bc.unsubscribe(q)

    # Verify SSE v2.0 HTTP headers on /api/v1/research/stream
    client = TestClient(app)

    def _publish_terminal():
        import time

        time.sleep(0.05)
        bc.publish("goi_stream", "terminal_verdict", {"terminal_state": "VERIFIED"}, run_uid="run_sse_v2")

    import threading

    t = threading.Thread(target=_publish_terminal, daemon=True)
    t.start()
    with client.stream("GET", "/api/v1/research/stream", headers={"Last-Event-ID": str(seq_after_2)}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        assert "no-cache" in resp.headers.get("cache-control", "")
        assert resp.headers.get("x-accel-buffering") == "no"
        chunks = list(resp.iter_lines())
    t.join(timeout=2.0)
    joined = "\n".join(chunks)
    assert "id: " in joined
    assert "terminal_verdict" in joined


def test_ci_gate_hole_verifier_size_and_safety():
    hv_path = Path("invar_agent/compiler/hole_verifier.py")
    assert hv_path.stat().st_size <= 50 * 1024

    good = """
import math
def kernel_hole(y, p):
    total = 0.0
    for i in range(4):
        total += math.exp(-p[0] * y[i])
    return total
"""
    assert verify_bespoke_hole_source(good).ok is True

    bad_samples = [
        "import os\ndef f(y): return os.system('echo hi')",
        "def f(y):\n    while True:\n        pass",
        "def f(y):\n    y[0] = 99.0\n    return y[0]",
        "def f(y): return eval('1+1')",
        "def f(y): return y.__class__.__subclasses__()",
    ]
    for bad in bad_samples:
        rep = verify_bespoke_hole_source(bad)
        assert rep.ok is False


def test_ci_gate_builder_unification_and_macro_domain_bijection():
    # Lossless 8-domain bijection
    for dom in ScientificDomain:
        macro = to_macro_domain(dom)
        assert macro in MACRO_SCIENTIFIC_DOMAINS
        assert from_macro_domain(macro) == dom

    table, body = _make_sample_ode_ir()
    outcome = scope_check_and_lower(table, body)
    assert outcome.ok and outcome.sci_ir is not None
    sealed = seal_sciir_ihcg(outcome.sci_ir)

    uast = lower_sciir_to_builder_uast(sealed)
    assert uast["sciir_commitment_blake3"] == sealed.sciir_commitment_blake3

    builder_ir = BuilderCanonicalIR.from_sci_ir(sealed)
    assert builder_ir.sciir_commitment_blake3 == sealed.sciir_commitment_blake3
    assert builder_ir.ir_blake3_digest == sealed.sciir_commitment_blake3


def test_ci_gate_pass2_structural_normalization_for_non_ode_and_zero_tol():
    table = FrozenSymbolTable(
        ir_id="carnot_sweep",
        archetype=Archetype.PARAM_SWEEP,
        domain=ScientificDomain.PHYSICS_THERMO,
        symbols=[
            SymbolDecl(
                name="T_C",
                role=SlotRole.CONTROL_PARAM,
                initial_or_default=300.0,
                domain=IntervalBound(lo=250.0, hi=350.0),
            ),
            SymbolDecl(name="T_H", role=SlotRole.CONSTANT, initial_or_default=600.0),
            SymbolDecl(name="eta", role=SlotRole.DERIVED_OBSERVABLE),
        ],
    ).assign_canonical_slot_indices()
    raw_pass2 = {
        "rationale": "Carnot efficiency sweep",
        "let_bindings": [],
        "dynamics_or_residuals": [],
        "observables": [{"target": "eta", "expr_text": "1.0 - T_C / T_H"}],
        "obligations": [
            {
                "id": "ob_carnot",
                "quantifier": "FORALL",
                "lhs_text": "eta",
                "comparator": ">=",
                "rhs_text": "0.75",
                "tolerance": 0.0,
                "epistemic_origin": "USER_PROMPT",
                "source_text": "eta >= 0.75",
            }
        ],
        # Spurious ode_pde keys accidentally copied from exemplar into param_sweep
        "config": {
            "kind": "param_sweep",
            "t_span": [0.0, 1.0],
            "rtol": 1e-7,
            "points_per_axis": 11,
        },
    }
    norm = StagedSciIRSynthesizer._normalize_pass2_structural_fields(raw_pass2, table)
    body = ModelBodyDraft.model_validate(norm)
    assert body.config.kind == "param_sweep"
    assert body.obligations[0].tolerance == 1e-9
    out = scope_check_and_lower(table, body)
    assert out.ok and out.sci_ir is not None
    sweep = json.loads(_native.py_solve_sciir_sweep(out.sci_ir.to_rust_json()))
    assert sweep["all_obligations_satisfied"] is False


"""Phase 0 regression & property tests (ARCH-001, ARCH-002 -> BUG-001, BUG-002, BUG-003)."""
from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, strategies as st

from invar_agent.healing.deterministic_ast import (
    KernelTransform,
    apply_healing_directives,
    sanitize_intrinsics,
)
from invar_agent.research.epistemic import (
    compute_evidence_status,
    incomplete_certificate,
)
from invar_agent.research.report import build_research_report
from invar_agent.research.reporter import generate_comprehensive_research_report
from invar_agent.research.vault import ResearchVaultManager, _key_metrics
from invar_agent.schemas.research import (
    ResearchBlueprint,
    SimulationResult,
    VerificationContract,
)


def _exec_steps(code: str, initial: list[float], dt: float, steps: int) -> list[float]:
    ns: dict = {}
    exec(compile(code, "<kernel>", "exec"), ns)  # noqa: S102
    buf = SimpleNamespace(data=list(initial) + [0.0] * max(0, 64 - len(initial)), size=64)
    for _ in range(steps):
        ns["simulation_step"](buf, dt)
    return buf.data[: len(initial)]


@settings(max_examples=30, deadline=None)
@given(
    var0=st.sampled_from(["theta", "x", "q", "pos"]),
    var1=st.sampled_from(["omega", "kappa", "gamma", "g", "damping", "nu", "stiffness", "v"]),
    x0=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    v0=st.sampled_from([0.0, 0.25, -0.5, 1.2]),
    coeff=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    damp=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    dt=st.sampled_from([0.005, 0.01, 0.02]),
)
def test_sanitizer_never_changes_semantics(var0, var1, x0, v0, coeff, damp, dt):
    """Test 1: sanitize_intrinsics preserves exact per-step dynamics for arbitrary names."""
    code = (
        "import math\n"
        "def simulation_step(buf, dt):\n"
        f"    {var0} = buf.data[0]\n"
        f"    {var1} = buf.data[1]\n"
        f"    d0 = {var1}\n"
        f"    d1 = {coeff!r} * math.sin({var0}) - {damp!r} * {var1}\n"
        f"    {var0} = {var0} + dt * d0\n"
        f"    {var1} = {var1} + dt * d1\n"
        f"    buf.data[0] = {var0}\n"
        f"    buf.data[1] = {var1}\n"
    )
    san = sanitize_intrinsics(code, variable_map={var0: 0, var1: 1})
    for t in san.transforms:
        assert isinstance(t, KernelTransform)
        assert t.semantics_preserving is True
    orig = _exec_steps(code, [x0, v0], dt, steps=15)
    after = _exec_steps(san.code, [x0, v0], dt, steps=15)
    for a, b in zip(orig, after):
        assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), f"{orig} != {after}"


@settings(max_examples=25, deadline=None)
@given(
    bound=st.floats(min_value=1.2, max_value=5.0, allow_nan=False, allow_infinity=False),
    growth=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    preserve=st.booleans(),
)
def test_no_transform_makes_false_invariant_true(bound, growth, preserve):
    """Test 2: A violating trajectory stays violating after sanitize_intrinsics and healing directives."""
    code = (
        "def simulation_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        f"    x = x + dt * ({growth!r} * x)\n"
        "    buf.data[0] = x\n"
    )
    contracts = [
        {
            "contract_id": "c1",
            "invariants": [{"variable_name": "x", "assertion": f"state[0] <= {bound!r}", "tolerance": 0.0}],
            "initial_state": [1.0],
            "dt": 0.1,
            "steps": 40,
            "target_backend": "cpp",
        }
    ]
    orig_x = _exec_steps(code, [1.0], 0.1, 40)[0]
    assert orig_x > bound

    san = sanitize_intrinsics(
        code,
        contracts=contracts,
        all_contracts=contracts,
        variable_map={"x": 0},
        preserve_dynamics=preserve,
    )
    san_x = _exec_steps(san.code, [1.0], 0.1, 40)[0]
    assert san_x > bound
    assert math.isclose(san_x, orig_x, rel_tol=1e-11, abs_tol=1e-11)

    healed = apply_healing_directives(
        code,
        [
            {"op": "CLAMP_UPPER_BOUND", "slot": 0, "bound": bound},
            {"op": "CONDITION_SLOT", "slot": 0, "bound": bound},
            {"op": "ENFORCE_CONSERVATION_PROJECTION", "slot": 0},
            {"op": "STABILIZE_DYNAMICAL_STATE", "slot": 0},
        ],
        contracts=contracts,
        variable_map={"x": 0},
    )
    healed_x = _exec_steps(healed.code, [1.0], 0.1, 40)[0]
    assert healed_x > bound
    assert math.isclose(healed_x, orig_x, rel_tol=1e-11, abs_tol=1e-11)


@pytest.mark.asyncio
async def test_executed_digest_matches_archived_module(tmp_path):
    """Test 3: Executed bytes digest equals archived vault unified_module.py digest; mismatch fails closed."""
    from invar_agent.research.state_machine import ResearchRunState, ResearchStateMachine

    code = (
        "import math\n"
        "def simulation_step(buf, dt):\n"
        "    theta = buf.data[0]\n"
        "    omega = buf.data[1]\n"
        "    buf.data[0] = theta + dt * omega\n"
        "    buf.data[1] = omega + dt * (-9.81 * math.sin(theta) - 0.1 * omega)\n"
    )
    bp = ResearchBlueprint(
        blueprint_id="digest_match_test",
        title="Digest Match",
        falsifiable_hypothesis="theta stays bounded",
        research_prompt="pendulum",
        search_queries=[],
        code_template=code,
        unified_module_code=code,
        variable_map={"theta": 0, "omega": 1},
        contracts=[
            VerificationContract(
                contract_id="c1",
                target_backend="cpp",
                initial_state=[0.5, 0.0],
                dt=0.01,
                steps=10,
                invariants=[{"variable_name": "theta", "assertion": "abs(state[0]) <= 2.0"}],
            )
        ],
    )
    sm = ResearchStateMachine(tmp_path, workspace_root=tmp_path)
    state = ResearchRunState(
        blueprint=bp,
        status="VERIFIED",
        epistemic_verdict="VERIFIED_DISCOVERY",
        executed_module_digest=hashlib.sha256(code.encode("utf-8")).hexdigest(),
    )
    await sm._emit_report(state)
    vault = ResearchVaultManager(tmp_path)
    archived = (vault.run_dir("digest_match_test") / "unified_module.py").read_text(encoding="utf-8")
    assert hashlib.sha256(archived.encode("utf-8")).hexdigest() == state.executed_module_digest

    # Now simulate a mismatch between executed_module_digest and archived unified_module_code -> must fail closed.
    bp2 = bp.model_copy(update={"blueprint_id": "digest_mismatch_test", "unified_module_code": code + "# tampered\n"})
    state2 = ResearchRunState(
        blueprint=bp2,
        status="VERIFIED",
        epistemic_verdict="VERIFIED_DISCOVERY",
        executed_module_digest=hashlib.sha256(code.encode("utf-8")).hexdigest(),
    )
    await sm._emit_report(state2)
    assert state2.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED"
    assert state2.meaningfulness_certificate.failure_reason == "EXECUTED_ARCHIVE_DIGEST_MISMATCH"


def test_metrics_never_certified_without_verified_verdict(tmp_path):
    """Test 4: Metrics are never CERTIFIED and reports never say Verified without a VERIFIED_* verdict."""
    bp = ResearchBlueprint(
        blueprint_id="unverified_metrics_test",
        title="Damped Pendulum",
        falsifiable_hypothesis="theta <= 1",
        research_prompt="pendulum",
        search_queries=[],
        code_template="def simulation_step(buf, dt): pass\n",
        variable_map={"theta": 0, "omega": 1, "state_0": 0},
        contracts=[
            VerificationContract(
                contract_id="c1",
                target_backend="cpp",
                initial_state=[0.5, 0.0],
                dt=0.01,
                steps=10,
                invariants=[{"variable_name": "theta", "assertion": "state[0] <= 1.0"}],
            )
        ],
    )
    telemetry = {"obs_theta": 0.2755, "obs_omega": 0.3716, "obs_state_0": 0.2755}
    for verdict in ("INCONCLUSIVE_UNVERIFIED", "FAILED", "REFUSED_IMPOSSIBLE", ""):
        km = _key_metrics(bp, telemetry, epistemic_verdict=verdict)
        assert "state_0" not in km  # placeholder state_N dropped
        for name, entry in km.items():
            assert "verified" not in entry
            assert entry["evidence_status"] != "CERTIFIED"
            assert entry["evidence_status"] == "OBSERVED_UNVERIFIED"

    # Certified only when epistemic_verdict starts with VERIFIED_
    km_ok = _key_metrics(bp, telemetry, epistemic_verdict="VERIFIED_DISCOVERY")
    assert km_ok["theta"]["evidence_status"] == "CERTIFIED"
    assert compute_evidence_status("VERIFIED_DISCOVERY", None) == "NOT_COMPUTED"
    assert compute_evidence_status("VERIFIED_DISCOVERY", 1.0, violated=True) == "VIOLATED"

    # Reports on INCONCLUSIVE_UNVERIFIED must never contain "Verified" headings or VERIFIED stage rows
    sim_res = SimulationResult(
        blueprint_id="unverified_metrics_test",
        bundle_hash="a" * 64,
        contract_id="c1",
        stage=1,
        status="VERIFIED",
        execution_time_ms=1.2,
        telemetry=telemetry,
    )
    rep_md = generate_comprehensive_research_report(
        bp,
        telemetry,
        [],
        results=[sim_res],
        status="INCONCLUSIVE_UNVERIFIED",
        epistemic_verdict="INCONCLUSIVE_UNVERIFIED",
    )
    assert "Verified Numerical Findings" not in rep_md
    assert "| `c1` | VERIFIED |" not in rep_md
    assert "INVARIANTS_HELD" in rep_md
    assert "OBSERVED_UNVERIFIED" in rep_md

    cert = incomplete_certificate("unverified_metrics_test", "theta <= 1", "DEMO_INC")
    st_obj = SimpleNamespace(
        blueprint=bp,
        results=[sim_res],
        literature=[],
        status="INCONCLUSIVE_UNVERIFIED",
        epistemic_verdict="INCONCLUSIVE_UNVERIFIED",
        checkpoint_path=None,
        meaningfulness_certificate=cert,
        preserved_vs_enforced={},
    )
    rep2 = build_research_report(st_obj)
    assert "# Verified Quantitative Research Report" not in rep2
    assert "OBSERVED_UNVERIFIED" in rep2
    assert "| `c1` | VERIFIED |" not in rep2

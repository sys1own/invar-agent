"""Phase 1 regression tests (ARCH-003, ARCH-004 -> BUG-004, BUG-005)."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from invar_agent.builder.circuit_breaker import BuildBreakerBudget
from invar_agent.builder.emitters.base import EmitterRegistry, ModuleEmission
from invar_agent.builder.ingestion.pipeline import CanonicalResearchIngestionPipeline, seal_ir
from invar_agent.builder.ingestion.schemas import IngestionRequest
from invar_agent.builder.state_machine import BuildRequest, BuilderStateMachine
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.runtime.isolation import (
    SolverTimeoutError,
    Z3_GLOBAL_LOCK,
    run_on_z3_thread,
)
from invar_agent.schemas.research import ResearchBlueprint
from invar_agent.server import create_app


def _make_real_verified_vault_entry(ws: Path, run_id: str = "fx_rk4_ode") -> str:
    """Run the real ResearchStateMachine (zero stubs) to produce a verified IHCG vault entry."""
    prompt = (
        "Domain: drive in [0,2]; rate in [1,1]; x in [0,3]; z in [0,3]\n"
        "drive: SI [0,0,0,0,0,0,0]\nrate: SI [0,0,-1,0,0,0,0]\n"
        "x: SI [0,0,0,0,0,0,0]\nz: SI [0,0,0,0,0,0,0]\n"
        "Evolution: dx/dt = rate*(drive-x)\nEvolution: dz/dt = rate*(x-z)\n"
        "Claim: x > 0.25\nMechanism: drive\nQuantifier: EXISTS\nArchetype: ODE_PDE\n"
    )
    lines = [
        "def simulation_step(buf, dt):",
        "    drive = buf.data[0]",
        "    rate = buf.data[1]",
        "    x = buf.data[2]",
        "    z = buf.data[3]",
    ]
    for stage in range(1, 5):
        factor = "dt" if stage == 4 else "dt/2"
        x = "x" if stage == 1 else f"(x+{factor}*a{stage-1})"
        z = "z" if stage == 1 else f"(z+{factor}*b{stage-1})"
        lines.extend([f"    a{stage} = rate*(drive-{x})", f"    b{stage} = rate*({x}-{z})"])
    lines.extend([
        "    buf.data[2] = x+dt*(a1+2*a2+2*a3+a4)/6",
        "    buf.data[3] = z+dt*(b1+2*b2+2*b3+b4)/6",
    ])
    bp = ResearchBlueprint(
        blueprint_id=run_id,
        title="Coupled relaxation",
        falsifiable_hypothesis="x > 0.25",
        research_prompt=prompt,
        search_queries=[],
        code_template="\n".join(lines) + "\n",
        variable_map={"drive": 0, "rate": 1, "x": 2, "z": 3},
        slot_roles={
            "drive": "read_only_parameter",
            "rate": "read_only_parameter",
            "x": "evolving_state",
            "z": "evolving_state",
        },
        contracts=[
            dict(
                contract_id="relaxation",
                target_backend="cpp",
                initial_state=[1.5, 1, 0, 0],
                dt=0.125,
                steps=4,
                invariants=[dict(variable_name="x", assertion="state[2] > 0.25")],
            )
        ],
        max_feedback_rounds=1,
    )
    state = asyncio.run(ResearchStateMachine(ws, workspace_root=ws).run(bp))
    assert state.epistemic_verdict.startswith("VERIFIED_"), f"Expected VERIFIED_*, got {state.epistemic_verdict}"
    return run_id


def test_z3_lock_inversion_assertion_and_timeout():
    """ARCH-003: Submitting to Z3 actor while holding Z3_GLOBAL_LOCK raises; deadline raises SOLVER_TIMEOUT."""
    with Z3_GLOBAL_LOCK:
        with pytest.raises(RuntimeError, match="Z3_LOCK_INVERSION"):
            run_on_z3_thread(lambda: 42)

    # Normal lock-free call succeeds
    assert run_on_z3_thread(lambda: 42) == 42

    # Deadline expiry raises SolverTimeoutError
    with pytest.raises(SolverTimeoutError, match="SOLVER_TIMEOUT"):
        run_on_z3_thread(lambda: time.sleep(0.3), deadline=time.monotonic() + 0.02)
    # Drain the worker before exiting
    assert run_on_z3_thread(lambda: 7) == 7


@pytest.mark.timeout(30)
def test_builder_vault_ingest_over_asgi_with_real_certificate(tmp_path):
    """Test 5: Vault-bound Builder ingest over ASGI never deadlocks and server remains responsive."""
    run_id = _make_real_verified_vault_entry(tmp_path)
    app = create_app(workspace_root=tmp_path)
    with TestClient(app) as client:
        # 1. Vault-bound Builder ingest over ASGI (previously deadlocked on _FORMAL_LOCK)
        res = client.post(
            "/api/v1/builder/ingest",
            data={"vault_ids": json.dumps([run_id])},
        )
        assert res.status_code == 200, res.text
        ir_data = res.json()
        assert ir_data["ir_id"]

        # 2. Verify /api/status exposes solver.queue_age_s
        st = client.get("/api/status")
        assert st.status_code == 200
        body = st.json()
        assert "solver" in body
        assert "queue_age_s" in body["solver"]
        assert isinstance(body["solver"]["queue_age_s"], (int, float))

        # 3. Follow with a Z3-dependent override + research/vault request on the same app
        ov = client.patch(
            f"/api/v1/builder/ir/{ir_data['ir_id']}/override",
            json={"approve": True},
        )
        assert ov.status_code == 200, ov.text

        vlist = client.get("/api/v1/research/vault")
        assert vlist.status_code == 200


@pytest.mark.serial
@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_build_wall_clock_is_preemptive(tmp_path):
    """UPG-018: A sleeping emitter running in StepWorker is preempted and killed by max_wall_clock_sec."""
    from invar_agent.runtime.killable import is_pid_alive
    from tests._slow_emitters import slow_emitters

    run_id = await asyncio.to_thread(_make_real_verified_vault_entry, tmp_path, "fx_rk4_sleep")
    ir = await asyncio.to_thread(
        CanonicalResearchIngestionPipeline(tmp_path).ingest,
        IngestionRequest(vault_run_ids=[run_id]),
    )
    ir.user_approved = True
    seal_ir(ir)

    budget = BuildBreakerBudget(max_wall_clock_sec=5.0)
    machine = BuilderStateMachine(tmp_path, emitters=slow_emitters(), budget=budget)
    t0 = time.monotonic()
    result = await machine.run(BuildRequest(build_id="sleep_budget_test", ir=ir))
    elapsed = time.monotonic() - t0
    assert result.status == "ABORTED_CRITICAL"
    assert result.error_code == "BUILD_WALL_CLOCK_EXCEEDED" or "BUILD_WALL_CLOCK_EXCEEDED" in (result.error or "")
    assert elapsed < 12.0, f"Expected preemptive abort in < 12s, took {elapsed:.2f}s"
    assert machine._step_worker is None
    if getattr(machine, "_last_worker_pid", None):
        assert not is_pid_alive(machine._last_worker_pid)

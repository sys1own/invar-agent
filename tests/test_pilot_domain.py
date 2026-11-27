"""UPG-007 (BUG-009): Certified domain & resolution pilot protocol via DynamicCABIBridge."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from invar_agent.ir.lower_uast import emit_python_kernel_from_sciir
from invar_agent.ir.schema import (
    Archetype,
    EquationSpec,
    IntervalBound,
    SciIR,
    ScientificDomain,
    SiDimension,
    SlotRole,
    SymbolDecl,
)
from invar_agent.research.pilot_domain import PilotDomainError, propose_domains
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge


def _harmonic_ir() -> SciIR:
    # Undamped harmonic oscillator x'' = -omega^2 * x with omega=10 (x(0)=5, v(0)=0)
    return SciIR(
        ir_id="osc_pilot",
        domain=list(ScientificDomain)[0],
        user_prompt_digest="0" * 64,
        archetype=Archetype.ODE_PDE,
        symbols=[
            SymbolDecl(
                name="x",
                display_symbol="x",
                role=SlotRole.STATE,
                slot_index=0,
                dimension=SiDimension(exponents=[(0, 1)] * 7, unit_label="1"),
                initial_or_default=5.0,
            ),
            SymbolDecl(
                name="v",
                display_symbol="v",
                role=SlotRole.STATE,
                slot_index=1,
                dimension=SiDimension(exponents=[(0, 1)] * 7, unit_label="1"),
                initial_or_default=0.0,
            ),
        ],
        dynamics_or_residuals=[
            EquationSpec(target="x", expr={"kind": "sym", "name": "v", "role": "STATE", "slot_index": 1}),
            EquationSpec(
                target="v",
                expr={
                    "kind": "binary",
                    "op": "mul",
                    "lhs": {"kind": "const", "value": -100.0},
                    "rhs": {"kind": "sym", "name": "x", "role": "STATE", "slot_index": 0},
                },
            ),
        ],
    )


def test_upg007_pilot_derives_oscillator_envelope_and_stiffness_steps(tmp_path: Path):
    ir = _harmonic_ir()
    code = emit_python_kernel_from_sciir(ir)
    contract = SimpleNamespace(dt=0.05, steps=20, initial_state=[5.0, 0.0], timeout_seconds=5.0)
    bridge = DynamicCABIBridge(tmp_path)

    res = propose_domains(code, ir, contract, bridge)
    # Envelope must contain [-5, 5] for x and padded box strictly encloses it
    assert res.envelopes["x"][0] <= -4.9 and res.envelopes["x"][1] >= 4.9
    assert res.omega["x"][0] < -5.0 and res.omega["x"][1] > 5.0
    assert res.spectral_radius >= 9.9
    # dt * rho <= 0.2 => dt <= 0.02 => steps >= 50 on T=1.0
    assert res.steps >= 50
    assert any(entry["source"] == "COMPILER_PILOT" and entry["symbol"] == "x" for entry in res.ledger_entries)


def test_upg007_pilot_sign_preservation_and_singular_exclusion(tmp_path: Path):
    # Positive decaying state dT/dt = -0.1 * T with T(0) = 10.0 and observable log(T)
    ir = SciIR(
        ir_id="pos_log_pilot",
        domain=list(ScientificDomain)[0],
        user_prompt_digest="0" * 64,
        archetype=Archetype.ODE_PDE,
        symbols=[
            SymbolDecl(
                name="T",
                display_symbol="T",
                role=SlotRole.STATE,
                slot_index=0,
                dimension=SiDimension(exponents=[(0, 1)] * 7, unit_label="1"),
                initial_or_default=10.0,
            ),
            SymbolDecl(
                name="L",
                display_symbol="L",
                role=SlotRole.DERIVED_OBSERVABLE,
                slot_index=1,
                dimension=SiDimension(exponents=[(0, 1)] * 7, unit_label="1"),
                initial_or_default=0.0,
            ),
        ],
        dynamics_or_residuals=[
            EquationSpec(
                target="T",
                expr={
                    "kind": "binary",
                    "op": "mul",
                    "lhs": {"kind": "const", "value": -0.1},
                    "rhs": {"kind": "sym", "name": "T", "role": "STATE", "slot_index": 0},
                },
            )
        ],
        observables=[
            EquationSpec(
                target="L",
                expr={"kind": "unary", "op": "log", "arg": {"kind": "sym", "name": "T", "role": "STATE", "slot_index": 0}},
            )
        ],
    )
    code = emit_python_kernel_from_sciir(ir)
    contract = SimpleNamespace(dt=0.1, steps=50, initial_state=[10.0, 2.302585], timeout_seconds=5.0)
    bridge = DynamicCABIBridge(tmp_path)

    res = propose_domains(code, ir, contract, bridge)
    assert res.omega["T"][0] > 0.0, f"Positive state floor must stay strictly positive: {res.omega['T']}"


def test_upg007_pilot_diverged_on_blowup(tmp_path: Path):
    ir = SciIR(
        ir_id="blowup_pilot",
        domain=list(ScientificDomain)[0],
        user_prompt_digest="0" * 64,
        archetype=Archetype.ODE_PDE,
        symbols=[
            SymbolDecl(
                name="y",
                display_symbol="y",
                role=SlotRole.STATE,
                slot_index=0,
                dimension=SiDimension(exponents=[(0, 1)] * 7, unit_label="1"),
                initial_or_default=10.0,
            ),
        ],
        dynamics_or_residuals=[
            EquationSpec(
                target="y",
                expr={
                    "kind": "binary",
                    "op": "pow",
                    "lhs": {"kind": "sym", "name": "y", "role": "STATE", "slot_index": 0},
                    "rhs": {"kind": "const", "value": 4.0},
                },
            )
        ],
    )
    code = emit_python_kernel_from_sciir(ir)
    contract = SimpleNamespace(dt=0.1, steps=20, initial_state=[10.0], timeout_seconds=5.0)
    bridge = DynamicCABIBridge(tmp_path)

    with pytest.raises(PilotDomainError, match="PILOT_DIVERGED"):
        propose_domains(code, ir, contract, bridge)

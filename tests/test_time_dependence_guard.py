"""UPG-006 / NF-03: Non-autonomous ODE_PDE autonomization (`dt/dt = 1`) verification."""
from __future__ import annotations

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


def test_nf03_nonautonomous_ode_autonomizes_and_integrates_exact_half():
    ir = SciIR(
        ir_id="nonaut_t_sq",
        domain=list(ScientificDomain)[0],
        user_prompt_digest="0" * 64,
        archetype=Archetype.ODE_PDE,
        symbols=[
            SymbolDecl(
                name="t",
                display_symbol="t",
                role=SlotRole.INDEPENDENT_VAR,
                slot_index=0,
                dimension=SiDimension(exponents=[(0, 1)] * 7, unit_label="1"),
                initial_or_default=0.0,
                domain=IntervalBound(lo=0.0, hi=1.0),
            ),
            SymbolDecl(
                name="x",
                display_symbol="x",
                role=SlotRole.STATE,
                slot_index=1,
                dimension=SiDimension(exponents=[(0, 1)] * 7, unit_label="1"),
                initial_or_default=0.0,
                domain=IntervalBound(lo=0.0, hi=1.0),
            ),
        ],
        dynamics_or_residuals=[
            EquationSpec(
                target="x",
                expr={"kind": "sym", "name": "t", "role": "INDEPENDENT_VAR", "slot_index": 0},
            )
        ],
    )
    code = emit_python_kernel_from_sciir(ir)
    assert "_k1_t = 1.0" in code
    ns: dict = {}
    exec(code, ns)

    class _Buf:
        def __init__(self, data):
            self.data = list(data)

    buf = _Buf([0.0, 0.0])
    dt = 0.01
    for _ in range(100):
        ns["simulation_step"](buf, dt)
    assert abs(buf.data[0] - 1.0) < 1e-12
    assert abs(buf.data[1] - 0.5) < 1e-12

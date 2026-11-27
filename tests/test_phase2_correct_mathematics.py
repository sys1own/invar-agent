"""Phase 2 regression & property tests (ARCH-005, ARCH-006, ARCH-007 -> BUG-006..009, BUG-029)."""
from __future__ import annotations

import importlib
import math
import sys
from types import SimpleNamespace

import pytest
import sympy as sp
from hypothesis import given, settings, strategies as st
from scipy.integrate import solve_ivp

from invar_agent.builder.ingestion.pipeline import seal_ir
from invar_agent.builder.ingestion.schemas import (
    CanonicalResearchIR,
    EquationRole,
    ExtractedConstant,
    ExtractedEquation,
    SIDimensionVectorModel,
    SourceSpan,
    StateVariableSpec,
)
from invar_agent.builder.state_machine import BuildRequest, BuilderStateMachine
from invar_agent.ir.function_registry import (
    formal_gate_arities,
    formal_gate_vocabulary,
    lower_nary_binary_tree,
    prompt_function_vocabulary,
)
from invar_agent.ir.lower_uast import (
    _render_expr_to_python,
    emit_python_kernel_from_sciir,
)
from invar_agent.ir.prompt_templates import PASS2_SYSTEM_BLOCK_A
from invar_agent.ir.schema import (
    Archetype,
    EquationSpec,
    ExprBinary,
    ExprSym,
    ExprUnary,
    OdePdeConfig,
    ParamSweepConfig,
    SciIR,
    ScientificDomain,
    SiDimension,
    SlotRole,
    SymbolDecl,
)
from invar_agent.research.default_verification import _prove_rk4_assignments
from invar_agent.research.meaningfulness import _parse


def _make_algebraic_ir(ir_id: str, title: str, lhs: str, rhs: str, constants: list[tuple[str, float]]) -> CanonicalResearchIR:
    prov = SourceSpan(file="test.md", location="line:1", quote=f"{lhs} = {rhs}")
    ir = CanonicalResearchIR(
        ir_id=ir_id,
        source_mode="upload",
        title=title,
        constants=[
            ExtractedConstant(
                constant_id=f"c_{name}",
                name=name,
                symbol=name,
                si_value=val,
                raw_value=str(val),
                raw_unit="1",
                dimension=SIDimensionVectorModel(),
                provenance=prov,
            )
            for name, val in constants
        ],
        state_variables=[
            StateVariableSpec(
                symbol=lhs,
                slot_index=0,
                initial_value=0.0,
                dimension=SIDimensionVectorModel(),
                provenance=prov,
            )
        ],
        equations=[
            ExtractedEquation(
                equation_id=f"eq_{lhs}",
                role=EquationRole.ALGEBRAIC_CLOSURE,
                lhs_symbol=lhs,
                raw_latex=f"{lhs} = {rhs}",
                canonical_latex=f"{lhs} = {rhs}",
                sympy_expr=rhs,
                hin_homogeneous=True,
                z3_satisfiable=True,
                provenance=prov,
            )
        ],
        user_approved=True,
    )
    return seal_ir(ir)


@pytest.mark.asyncio
async def test_algebraic_program_invariant_to_horizon(tmp_path):
    """Test 8: AlgebraicMap outputs (F = S*exp(r*T) and Erlang-C W) are strictly invariant to horizon."""
    # 1. Forward price F = S * exp(r * T)
    ir = _make_algebraic_ir(
        "alg_forward_ir",
        "Forward Price",
        "F",
        "S * exp(r * T)",
        [("S", 100.0), ("r", 0.05), ("T", 1.0)],
    )
    machine = BuilderStateMachine(tmp_path)
    res = await machine.run(BuildRequest(build_id="alg_forward_test", ir=ir))
    assert res.status in ("COMPLETED", "COMPLETED_WITH_FALLBACKS"), f"Build failed: {res.error_code} {res.error}"

    src_dir = str(machine.workspace / "src")
    sys.path.insert(0, src_dir)
    try:
        pkg = next(p.name for p in (machine.workspace / "src").iterdir() if p.is_dir())
        sdk = importlib.import_module(f"{pkg}.sdk")
        expected_F = 100.0 * math.exp(0.05 * 1.0)
        ev = sdk.evaluate()
        assert math.isclose(ev["F"], expected_F, rel_tol=1e-10)
        sim1 = sdk.simulate(sdk.SimulationConfig(dt=0.1, steps=10))   # t = 1.0
        sim3 = sdk.simulate(sdk.SimulationConfig(dt=0.3, steps=10))   # t = 3.0
        assert math.isclose(sim1.trajectories["F"][-1], expected_F, rel_tol=1e-10)
        assert math.isclose(sim3.trajectories["F"][-1], expected_F, rel_tol=1e-10)
        # Rust parity
        sim_rs = sdk.simulate(sdk.SimulationConfig(dt=0.1, steps=5, backend="rust"))
        assert math.isclose(sim_rs.trajectories["F"][-1], expected_F, rel_tol=1e-10)
    finally:
        if src_dir in sys.path:
            sys.path.remove(src_dir)

    # 2. Erlang-C analytical golden value (c=3, lam=2, mu=1 -> a=2, rho=2/3 -> W = 4/9)
    ir2 = _make_algebraic_ir(
        "alg_erlang_ir",
        "Erlang C Queue",
        "W",
        "((lam/mu)**3 / (6.0 * (1.0 - lam/(c*mu)))) / (1.0 + (lam/mu) + (lam/mu)**2/2.0 + ((lam/mu)**3 / (6.0 * (1.0 - lam/(c*mu))))) / (c*mu - lam)",
        [("c", 3.0), ("mu", 1.0), ("lam", 2.0)],
    )
    machine2 = BuilderStateMachine(tmp_path)
    res2 = await machine2.run(BuildRequest(build_id="alg_erlang_test", ir=ir2))
    assert res2.status in ("COMPLETED", "COMPLETED_WITH_FALLBACKS"), f"Build failed: {res2.error_code} {res2.error}"
    src_dir2 = str(machine2.workspace / "src")
    sys.path.insert(0, src_dir2)
    try:
        pkg2 = next(p.name for p in (machine2.workspace / "src").iterdir() if p.is_dir())
        sdk2 = importlib.import_module(f"{pkg2}.sdk")
        ev2 = sdk2.evaluate()
        assert math.isclose(ev2["W"], 4.0 / 9.0, rel_tol=1e-10)
        sim_t1 = sdk2.simulate(sdk2.SimulationConfig(dt=0.2, steps=5))
        sim_t5 = sdk2.simulate(sdk2.SimulationConfig(dt=1.0, steps=5))
        assert math.isclose(sim_t1.trajectories["W"][-1], 4.0 / 9.0, rel_tol=1e-10)
        assert math.isclose(sim_t5.trajectories["W"][-1], 4.0 / 9.0, rel_tol=1e-10)
    finally:
        if src_dir2 in sys.path:
            sys.path.remove(src_dir2)


@settings(max_examples=30, deadline=None)
@given(
    x_val=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    y_val=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
)
def test_sciir_lowering_executable_equals_formal(x_val, y_val):
    """Test 9: Executable and formal SciIR lowering agree across every node kind; unknown node raises."""
    nodes = [
        {"kind": "unary", "op": "step", "arg": {"kind": "sym", "name": "x"}},
        {"kind": "unary", "op": "tanh", "arg": {"kind": "sym", "name": "x"}},
        {"kind": "unary", "op": "ln", "arg": {"kind": "sym", "name": "y"}},
        {"kind": "call", "func": "clamp", "args": [{"kind": "sym", "name": "x"}, {"kind": "const", "value": -1.0}, {"kind": "const", "value": 1.0}]},
        {"kind": "call", "func": "min", "args": [{"kind": "sym", "name": "x"}, {"kind": "sym", "name": "y"}, {"kind": "const", "value": 0.5}]},
        {"kind": "if_then_else", "cond_op": "ge", "lhs": {"kind": "sym", "name": "x"}, "rhs": {"kind": "const", "value": 0.0},
         "then_branch": {"kind": "sym", "name": "y"}, "else_branch": {"kind": "unary", "op": "neg", "arg": {"kind": "sym", "name": "y"}}},
    ]
    env_exec = {"x": x_val, "y": y_val, "math": math, "min": min, "max": max, "abs": abs}
    env_form = {
        "x": x_val, "y": y_val, "min": min, "max": max, "abs": abs,
        "tanh": math.tanh, "log": math.log, "exp": math.exp, "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt,
    }
    for node in nodes:
        py_s = _render_expr_to_python(node, {"x", "y"}, for_ihcg=False)
        ih_s = _render_expr_to_python(node, {"x", "y"}, for_ihcg=True)
        # Formal parser must accept the formal expression
        _parse(ih_s)
        v_exec = eval(py_s, {"__builtins__": {}}, env_exec)  # noqa: S307
        v_form = eval(ih_s, {"__builtins__": {}}, env_form)  # noqa: S307
        assert math.isclose(v_exec, v_form, rel_tol=1e-12, abs_tol=1e-12), f"{py_s} ({v_exec}) != {ih_s} ({v_form})"

    # Rand nodes must NOT lower to constant 0.5
    for dist in ("uniform01", "normal01", "exponential1"):
        r_s = _render_expr_to_python({"kind": "rand", "dist": dist, "draw_stream_id": 3}, set())
        assert r_s != "0.5"
        assert "splitmix64" in r_s

    # Unknown node kind MUST raise ValueError (never silently return "0.0")
    with pytest.raises(ValueError, match="Unsupported SciIR"):
        _render_expr_to_python({"kind": "totally_unknown_node_xyz"}, set())


def test_staged_ode_kernel_passes_scheme_proof_and_running_max():
    """Test 10: Staged ODE kernel passes _prove_rk4_assignments, rejects archetype mismatch, and matches SciPy running max."""
    # Consecutive first-order kinetics A -> B -> C (k1=0.8, k2=0.2)
    ir = SciIR(
        ir_id="kinetics_abc",
        user_prompt_digest="0" * 64,
        archetype=Archetype.ODE_PDE,
        domain=ScientificDomain.BIOCHEM_SYSTEMS,
        symbols=[
            SymbolDecl(name="A", role=SlotRole.STATE, slot_index=0, dimension=SiDimension(), initial_or_default=1.0),
            SymbolDecl(name="B", role=SlotRole.STATE, slot_index=1, dimension=SiDimension(), initial_or_default=0.0),
            SymbolDecl(name="k1", role=SlotRole.CONSTANT, slot_index=2, dimension=SiDimension(), initial_or_default=0.8),
            SymbolDecl(name="k2", role=SlotRole.CONSTANT, slot_index=3, dimension=SiDimension(), initial_or_default=0.2),
            SymbolDecl(name="B_max", role=SlotRole.DERIVED_OBSERVABLE, slot_index=4, dimension=SiDimension(), initial_or_default=0.0),
        ],
        dynamics_or_residuals=[
            EquationSpec(target="A", expr=ExprUnary(op="neg", arg=ExprBinary(op="mul", lhs=ExprSym(name="k1", role=SlotRole.CONSTANT), rhs=ExprSym(name="A", role=SlotRole.STATE)))),
            EquationSpec(target="B", expr=ExprBinary(op="sub",
                lhs=ExprBinary(op="mul", lhs=ExprSym(name="k1", role=SlotRole.CONSTANT), rhs=ExprSym(name="A", role=SlotRole.STATE)),
                rhs=ExprBinary(op="mul", lhs=ExprSym(name="k2", role=SlotRole.CONSTANT), rhs=ExprSym(name="B", role=SlotRole.STATE)))),
        ],
        observables=[
            EquationSpec(target="B_max", expr=ExprSym(name="B", role=SlotRole.STATE), reduction="TRAJECTORY_MAX"),
        ],
        config=OdePdeConfig(t_span=(0.0, 20.0), sample_points=400),
    )
    kernel = emit_python_kernel_from_sciir(ir)

    # 1. Verify _prove_rk4_assignments accepts the emitted RK4 kernel!
    A_s, B_s, k1_s, k2_s = sp.symbols("A B k1 k2")
    assert _prove_rk4_assignments(
        kernel,
        {"A": 0, "B": 1, "k1": 2, "k2": 3, "B_max": 4},
        {"A": -k1_s * A_s, "B": k1_s * A_s - k2_s * B_s},
    ) is True

    # 2. Run emitted kernel and compare running-max B_max against SciPy solve_ivp
    ns: dict = {}
    exec(compile(kernel, "<kinetics>", "exec"), ns)  # noqa: S102
    buf = SimpleNamespace(data=[1.0, 0.0, 0.8, 0.2, 0.0] + [0.0] * 16, size=21)
    dt = 20.0 / 400
    for _ in range(400):
        ns["simulation_step"](buf, dt)
    sol = solve_ivp(lambda t, y: [-0.8 * y[0], 0.8 * y[0] - 0.2 * y[1]], [0.0, 20.0], [1.0, 0.0], max_step=0.05)
    scipy_b_max = float(max(sol.y[1]))
    assert math.isclose(buf.data[4], scipy_b_max, rel_tol=1e-3), f"B_max={buf.data[4]} vs SciPy={scipy_b_max}"

    # 3. Reject archetype/equation mismatch (PARAM_SWEEP with STATE dynamics)
    bad_ir = SciIR(
        ir_id="bad_arch",
        user_prompt_digest="0" * 64,
        archetype=Archetype.PARAM_SWEEP,
        domain=ScientificDomain.BIOCHEM_SYSTEMS,
        symbols=ir.symbols[:2],
        dynamics_or_residuals=ir.dynamics_or_residuals[:1],
        config=ParamSweepConfig(),
    )
    with pytest.raises(ValueError, match="ARCHETYPE_EQUATION_MISMATCH"):
        emit_python_kernel_from_sciir(bad_ir)


def test_prompt_vocabulary_equals_gate_vocabulary():
    """Test 11: Prompt advertised function vocabulary equals the formal gate allow-list."""
    prompt_vocab = set(prompt_function_vocabulary())
    gate_vocab = set(formal_gate_vocabulary())
    assert prompt_vocab == gate_vocab
    for fn in prompt_vocab:
        assert f"`{fn}`" in PASS2_SYSTEM_BLOCK_A
        assert fn in formal_gate_arities()

    # Verify newly harmonized functions parse cleanly through the formal gate
    _parse("tanh(x) + ln(y) + clamp(x, 0.0, 1.0) + step(x) + min(x, y, 1.0, 2.0)")
    assert lower_nary_binary_tree("min", ["a", "b", "c", "d"]) == "min(min(a, b), min(c, d))"

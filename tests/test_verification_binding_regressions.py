"""Domain-certified helper specialization and exact identity proof regressions."""
import ast
import copy
import pytest
import sympy as sp

from invar_agent.healing.deterministic_ast import _SAFE_DIV_SRC
from invar_agent.research.default_verification import (
    UnsupportedVerification, _equivalent, _normalize_guarded_divisions,
)
from invar_agent.research import meaningfulness as m
from invar_agent.research.epistemic import compile_blueprint_ihcg
from phase3b_cases import sir_model, oregonator_model, allen_eggers_model
from invar_agent.research.evolution import _SymbolicKernel
from invar_agent.research.default_verification import _prove_rk4_assignments
from invar_agent.research.uniqueness import CanonicalSignatureEngine, si_error_budgets
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge
from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
from invar_agent.transpiler.frontend import python_source_to_uast
from invar_agent.research.stochastic_equilibrium import _Roundoff


def test_polynomial_identity_with_shared_opaque_rational_subexpression():
    assert _equivalent("b+r/(1+x*x)-(t*b+(1-t)*(b+r/(1+x*x)))","t*r/(1+x*x)")
    assert not _equivalent("r/(1+x*x)","r/(1+x*x)+0.001")
    assert _equivalent("0","0")


def test_guarded_division_specialization_preserves_only_domain_safe_arithmetic():
    source = "import math\n"+_SAFE_DIV_SRC+"\ndef simulation_step(buf,dt):\n    buf.data[2]=_invar_sdiv(buf.data[0],buf.data[1])\n"
    proof = _normalize_guarded_divisions(source,dict(x=0,y=1,z=2),dict(x=(1,2),y=(1,2),z=(0,2)),.1)
    assert not any(isinstance(n,(ast.If,ast.IfExp)) for n in ast.walk(ast.parse(proof)))
    assert _normalize_guarded_divisions("def simulation_step(buf,dt):\n    pass",{}, {},.1)
    with pytest.raises(UnsupportedVerification,match="ROUNDING_DENOMINATOR_NOT_SEPARATED"):
        _normalize_guarded_divisions(source,dict(x=0,y=1,z=2),dict(x=(1,2),y=(-1,1),z=(0,2)),.1)
    with pytest.raises(UnsupportedVerification,match="GUARDED_DIVISION_IMPLEMENTATION_UNBOUND"):
        _normalize_guarded_divisions(source.replace("return a / b","return a * b"),dict(x=0,y=1,z=2),dict(x=(1,2),y=(1,2),z=(0,2)),.1)


@pytest.mark.parametrize("formula,status",[("exp(x) > 0","SAT"),("exp(x) < 0","UNSAT"),("implies(x > 3,x > 0)","SAT")])
def test_fixed_coordinates_preserve_function_identifiers(formula,status):
    assert m.solve_query(formula,{"x":[1,1]}).status == status


def test_typed_temporal_reductions_and_stage_extension_integrity():
    bp = sir_model()
    graph = compile_blueprint_ihcg(bp)
    broken = copy.deepcopy(bp.ihcg_evidence["evidence"])
    broken["equations"]["Ipeak"]["expression"] = "I+1"
    with pytest.raises(ValueError,match="trajectory reductions"):
        m.IhcgCompiler().compile_from_prompt(bp.research_prompt,broken,bp.ihcg_evidence["domain"],"ODE_PDE")
    broken = copy.deepcopy(bp.ihcg_evidence["evidence"])
    broken["equations"]["dI"]["expression"] = "0"
    with pytest.raises(ValueError,match="foundation equations"):
        m.IhcgCompiler().extend_stage(graph,bp.research_prompt,broken,bp.ihcg_evidence["domain"],"ODE_PDE")
    assert m.solve_query("True",{}).status == "SAT"
    with pytest.raises(ValueError,match="unit scales"):
        m.SlotSpec(slot_idx=0,symbol="x",role="CONTROL_PARAM",si_dim=[0]*7,unit_scale=(0,1))


def test_large_executable_expression_dags_are_proved_without_expanding_source_ast():
    x = sp.Symbol("x")
    terms = sp.symbols("v0:300")
    left = x*sp.Add(*terms)
    right = sp.Add(*(x*t for t in terms))
    assert _equivalent(left,right)
    assert not _equivalent(left,right+1)


def test_roundoff_tracks_delegated_library_and_integer_power_calls():
    code = "import math\ndef stage(buf,dt):\n    buf.data[1] = math.exp(buf.data[0])\n    buf.data[2] = buf.data[0]**3\ndef simulation_step(buf,dt):\n    stage(buf,dt)\n"
    report = _Roundoff(code,dict(x=0,e=1,cube=2),dict(x=(1,2),e=(0,10),cube=(0,10)),.1)
    assert report.values[1][0][0] < 3 < report.values[1][0][1]
    assert report.values[2][0][0] <= 1 and report.values[2][0][1] >= 8
    assert all(report.values[i][1] > 0 for i in (1,2))


@pytest.mark.parametrize("status",["SAT","UNSAT"])
def test_output_definition_elimination_preserves_output_bounds(status):
    domain = dict(x=[0,1],z=[0,10] if status == "SAT" else [-10,-1])
    result = m.solve_query("z == exp(-exp(x))",domain,witness=True)
    assert result.status == status
    with pytest.raises(ValueError,match="deterministic caps"):
        m.solve_query("z == exp(-exp(x))",domain,timeout_ms=0)


def test_scaled_transcendental_cds_and_uncertainty_coordinates():
    graph = compile_blueprint_ihcg(allen_eggers_model("si_original"))
    scaled = compile_blueprint_ihcg(allen_eggers_model("si_scaled",kilometers=True))
    engine = CanonicalSignatureEngine()
    left = engine.compute_cds(graph,{},m.EpistemicVerdict.VERIFIED_DISCOVERY)
    right = engine.compute_cds(scaled,{},m.EpistemicVerdict.VERIFIED_DISCOVERY)
    assert left.model_hash == right.model_hash and left.claim_hash == right.claim_hash
    assert left.pi_domain.coordinate_system == right.pi_domain.coordinate_system
    assert si_error_budgets(scaled,1e-9,2e-9) == pytest.approx(si_error_budgets(graph,1e-6,2e-6))


def test_cubic_formal_rhs_and_rk4_protocol_identity():
    bp = oregonator_model("cubic_binding",cubic=True)
    graph = compile_blueprint_ihcg(bp)
    source = _SymbolicKernel(bp.code_template,bp.variable_map)
    rhs = {e.state_symbol:sp.sympify(e.expression,rational=True) for e in graph.equations.values()}
    derivatives,_ = source.derivatives(set(rhs))
    for name,expression in rhs.items(): assert _equivalent(derivatives[name],expression)
    assert _prove_rk4_assignments(bp.code_template,bp.variable_map,rhs)
    assert not _prove_rk4_assignments("def simulation_step(buf,dt):\n    pass\n",bp.variable_map,rhs)
    assert not _prove_rk4_assignments("",{},{})
    assert not _prove_rk4_assignments("def broken(",bp.variable_map,rhs)


def test_rk4_proof_accepts_renamed_wires_and_cse_helper_returns():
    bp = oregonator_model("renamed_wires",cubic=True)
    graph = compile_blueprint_ihcg(bp)
    rhs = {e.state_symbol:sp.sympify(e.expression,rational=True) for e in graph.equations.values()}
    tree = ast.parse(bp.code_template)
    names = sorted({n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)})
    renamed = dict(zip(names,(f"wire_{i}" for i in range(len(names)))))
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node): return ast.copy_location(ast.Name(id=renamed.get(node.id,node.id),ctx=node.ctx),node)
    assert _prove_rk4_assignments(ast.unparse(Rename().visit(tree)),bp.variable_map,rhs)
    helper = "def field(a,b,c,s,q,w,rate):\n    cross = a*b\n    return rate*s*(b+a-q*a*a-cross)-0.01*rate*a*a*a,rate*(c-b-b*a)/s,rate*w*(a-c)\n"
    tree = ast.parse(bp.code_template)
    body = tree.body[0].body
    for stage in range(1,5):
        equations = [n for n in body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith(f"k{stage}_")]
        stage_state = {name:sp.Symbol(name) if stage == 1 else sp.Symbol(name)+sp.Symbol("dt")*sp.Symbol(f"k{stage-1}_{name}")/(2 if stage < 4 else 1) for name in rhs}
        call = ast.parse(f"{','.join(f'k{stage}_{name}' for name in rhs)} = field({','.join(str(stage_state[name]) for name in rhs)},s,q,w,rate)").body[0]
        index = body.index(equations[0])
        for node in equations: body.remove(node)
        body.insert(index,call)
    assert _prove_rk4_assignments(helper+ast.unparse(tree),bp.variable_map,rhs)


@pytest.mark.parametrize("fault",["midpoint","weight","missing","unknown","early_return","tuple_loop","loop_else","whole_buffer","keywords"])
def test_rk4_assignment_proof_rejects_unmatched_or_corrupt_steps(fault):
    bp = oregonator_model("corrupt_step",cubic=True)
    rhs = {e.state_symbol:sp.sympify(e.expression,rational=True) for e in compile_blueprint_ihcg(bp).equations.values()}
    tree = ast.parse(bp.code_template)
    body = tree.body[0].body
    if fault == "missing": body[:] = [n for n in body if not isinstance(n,ast.Assign) or not isinstance(n.targets[0],ast.Name) or n.targets[0].id != "k3_y"]
    elif fault == "early_return": body.insert(0,ast.Return())
    elif fault == "unknown": body.extend(ast.parse("buf.data[0] = unknown_function(buf.data[0])").body)
    elif fault == "tuple_loop": body.extend(ast.parse("for a,b in [(0,1)]:\n    buf.data[0] = 0\n").body)
    elif fault == "loop_else": body.extend(ast.parse("for a in range(1):\n    pass\nelse:\n    buf.data[0] = 0\n").body)
    elif fault == "whole_buffer": body.extend(ast.parse("buf.data = [0]*7").body)
    elif fault == "keywords":
        tree.body.extend(ast.parse("def passthrough(x):\n    return x\n").body)
        body.extend(ast.parse("buf.data[0] = passthrough(x=0)").body)
    elif fault == "weight":
        node = next(n for n in body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id == "x" and isinstance(n.value,ast.BinOp))
        node.value = ast.parse(ast.unparse(node.value).replace("2 * k2_x","3 * k2_x"),mode="eval").body
    else:
        node = next(n for n in body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id == "k2_x")
        node.value = ast.parse(ast.unparse(node.value).replace("dt / 2","dt / 3"),mode="eval").body
    assert not _prove_rk4_assignments(ast.unparse(ast.fix_missing_locations(tree)),bp.variable_map,rhs)


def test_rk4_proof_does_not_accept_unproved_stage_identities(monkeypatch):
    bp = oregonator_model("unproved_stage",cubic=True)
    rhs = {e.state_symbol:sp.sympify(e.expression,rational=True) for e in compile_blueprint_ihcg(bp).equations.values()}
    monkeypatch.setattr("invar_agent.research.default_verification._equivalent",lambda *args,**kwargs:False)
    assert not _prove_rk4_assignments(bp.code_template,bp.variable_map,rhs)


@pytest.mark.parametrize("reference",[False,True])
def test_selected_trajectory_capture_does_not_expand_physical_state(tmp_path,reference):
    source = "def simulation_step(buf,dt):\n    buf.data[0] = buf.data[0]+dt\n"
    bridge = DynamicCABIBridge(tmp_path)
    run = bridge.run_python if reference else bridge.run
    code = source if reference else CppEmitter().emit_module(python_source_to_uast(source))
    result = run(code,[1.0],dt=.5,steps=3,trace_slots=(0,))
    assert result["state"] == [2.5]
    assert result["trace_values"] == [[1.5],[2.0],[2.5]]
    assert run(code,[1.0],dt=.5,steps=0,trace_slots=(0,))["trace_values"] == []
    assert run(code,[1.0],dt=.5,steps=1)["trace_values"] == []
    for slots in ((1,),(-1,),(False,)):
        with pytest.raises(ValueError,match="trace"):
            run(code,[1.0],dt=.5,steps=1,trace_slots=slots)

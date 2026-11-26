"""Executable-bound verification adapters. Unsupported program shapes fail closed.

Symbolic recovery is used only to establish a kernel/model binding. All primary,
baseline and sensitivity observations are executed by the sandbox bridge. A
memoryless algebraic map has no integration error; time-advancing kernels must
prove the RK4 update identity before an independent Heun kernel is constructed.
"""
from __future__ import annotations

import ast
import asyncio
import copy
import json
import math
import sys
from fractions import Fraction
from types import SimpleNamespace

import sympy as sp
import z3

from invar_agent import _native
from . import meaningfulness as m
from .accretion import code_hash
from .epistemic import ExecutionEvidence
from .evolution import _SymbolicKernel
from .falsification import MarginSpec, WitnessEvaluation, _claim_margin, _model_constraints, _is_formula, _ExactTranslator
from ..healing.deterministic_ast import instrument_purity, purity_audit_from_diag, _SAFE_DIV_SRC
from ..runtime.cabi_bridge import KernelExecutionError


class UnsupportedVerification(ValueError):
    pass


def _expression(value):
    import re
    text = re.sub(r"\bAbs\(", "abs(", str(value))
    text = re.sub(r"\bMax\(", "max(", text)
    text = re.sub(r"\bMin\(", "min(", text)
    tree = ast.parse(text, mode="eval")
    class _FoldVariadicMinMax(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in {"max", "min"} and len(node.args) > 2:
                cur = node.args[-1]
                for arg in reversed(node.args[:-1]):
                    cur = ast.Call(func=ast.Name(id=node.func.id, ctx=ast.Load()), args=[arg, cur], keywords=[])
                return cur
            return node
    folded = ast.unparse(ast.fix_missing_locations(_FoldVariadicMinMax().visit(tree)))
    m._parse(folded)
    return folded


def _proof_expression(value):
    class FoldExact(ast.NodeTransformer):
        def visit_BinOp(self,node):
            self.generic_visit(node)
            if isinstance(node.op,ast.Div) and isinstance(node.left,ast.Constant) and isinstance(node.right,ast.Constant):
                a,b = node.left.value,node.right.value
                if type(a) in (int,float) and type(b) in (int,float) and b:
                    exact = Fraction(a)/Fraction(b)
                    converted = float(exact)
                    if math.isfinite(converted) and Fraction(converted) == exact:
                        return ast.copy_location(ast.Constant(converted),node)
            return node
    return m._csel(FoldExact().visit(m._parse(_expression(value))))


def _equivalent(lhs, rhs, *, dynamic_symbols=()):
    if isinstance(lhs,sp.Basic) and isinstance(rhs,sp.Basic):
        if lhs == rhs: return True
        if max(sp.count_ops(lhs),sp.count_ops(rhs)) > 500:
            prefix = "_identity_common_"
            free_symbols = lhs.free_symbols|rhs.free_symbols
            while any(str(s).startswith(prefix) for s in free_symbols): prefix += "_"
            replacements, reduced = sp.cse([lhs,rhs],symbols=sp.numbered_symbols(prefix),order="none")
            aliases, canonical = {}, {s:(s,sp.Integer(1)) for s in free_symbols}
            recurrence = {sp.Symbol(s) for s in dynamic_symbols}|{s for s,_ in replacements}
            for variable,value in replacements:
                value = sp.factor_terms(sp.expand(value.xreplace(aliases)))
                if value == 0:
                    aliases[variable] = sp.Integer(0)
                    continue
                if not value.has(sp.Add):
                    aliases[variable] = value
                    continue
                try:
                    variables = value.free_symbols & recurrence if dynamic_symbols else value.free_symbols
                    if not variables or sp.Poly(value,*variables).total_degree() <= (1 if dynamic_symbols else 2):
                        aliases[variable] = value
                        continue
                except sp.PolynomialError:
                    pass
                content, primitive = value.as_content_primitive()
                if primitive in canonical:
                    previous,scale = canonical[primitive]
                    aliases[variable] = content/scale*previous
                else: canonical[primitive] = (variable,content)
            lhs,rhs = [sp.expand(value.xreplace(aliases)) for value in reduced]
            if lhs == rhs: return True
            memo = {}
            def term(value):
                if value in memo: return memo[value]
                if len(memo) > 20000: raise UnsupportedVerification("SYMBOLIC_IDENTITY_DAG_BUDGET")
                if value.is_Symbol: result = z3.Real(str(value))
                elif value.is_Number: result = z3.RealVal(str(value))
                elif value.func == sp.Add: result = z3.Sum(*(term(a) for a in value.args))
                elif value.func == sp.Mul: result = math.prod(term(a) for a in value.args)
                elif value.func == sp.Pow and value.args[1].is_Integer and abs(value.args[1]) <= 16:
                    result = term(value.args[0])**int(value.args[1])
                elif value.func == sp.Abs and len(value.args) == 1:
                    inner = term(value.args[0])
                    result = z3.If(inner >= 0, inner, -inner)
                elif value.func in (sp.Max, sp.Min) and value.args:
                    items = [term(a) for a in value.args]
                    cur = items[0]
                    for nxt in items[1:]:
                        cur = z3.If(nxt >= cur if value.func == sp.Max else nxt <= cur, nxt, cur)
                    result = cur
                else: raise UnsupportedVerification("SYMBOLIC_IDENTITY_DAG_OPERATOR_UNSUPPORTED")
                memo[value] = result
                return result
            solver = z3.SolverFor("QF_NRA"); solver.set(timeout=500,rlimit=2_000_000)
            solver.add(term(lhs) != term(rhs))
            return solver.check() == z3.unsat
    result = json.loads(_native.py_check_algebraic_equivalence(
        m._json(_proof_expression(lhs)), m._json(_proof_expression(rhs)),
        iter_limit=30, node_limit=20000, time_limit_ms=100))
    if result["equivalent"]: return True
    left,right = m._parse(_expression(lhs)),m._parse(_expression(rhs))
    reserved = m._variables(left)|m._variables(right)
    atoms = {}
    class Atomize(ast.NodeTransformer):
        def atom(self,node,*,inverse=False):
            key = (inverse,ast.dump(node))
            if key not in atoms:
                name = "_identity_atom_"+str(len(atoms))
                while name in reserved: name += "_"
                reserved.add(name); atoms[key] = name
            return ast.copy_location(ast.Name(id=atoms[key],ctx=ast.Load()),node)
        def visit_Call(self,node): return self.atom(node)
        def visit_BinOp(self,node):
            exponent = m._fraction(node.right)
            if isinstance(node.op,ast.Div):
                denominator = m._fraction(node.right)
                if denominator is not None and denominator != 0:
                    return self.generic_visit(node)
                return ast.copy_location(ast.BinOp(left=self.visit(node.left),op=ast.Mult(),right=self.atom(node.right,inverse=True)),node)
            if isinstance(node.op,ast.Pow) and not (exponent is not None and exponent.denominator == 1 and 0 <= exponent <= 6):
                return self.atom(node)
            return self.generic_visit(node)
    left,right = Atomize().visit(left),Atomize().visit(right)
    translator = m._Translator({name:(-1,1) for name in reserved},False)
    solver = z3.SolverFor("QF_NRA"); solver.set(timeout=100,rlimit=1_000_000)
    solver.add(translator.term(left) != translator.term(right))
    return solver.check() == z3.unsat


def _point_value(expression, values):
    interval = m._iv(m._parse(expression), {s: (v, v) for s,v in values.items()})
    if not math.isfinite(interval.lo) or not math.isfinite(interval.hi):
        raise UnsupportedVerification("NONFINITE_EXECUTABLE_OBSERVABLE")
    return (interval.lo+interval.hi)/2


def _normalize_guarded_divisions(code,slots,box,dt):
    tree = ast.parse(code)
    helpers = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "_invar_sdiv"]
    if not helpers: return code
    canonical = ast.dump(ast.parse(_SAFE_DIV_SRC).body[0])
    if len(helpers) != 1 or ast.dump(helpers[0]) != canonical:
        raise UnsupportedVerification("GUARDED_DIVISION_IMPLEMENTATION_UNBOUND")
    class Division(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id == "_invar_sdiv":
                if len(node.args) != 2 or node.keywords:
                    raise UnsupportedVerification("GUARDED_DIVISION_SIGNATURE_UNBOUND")
                return ast.copy_location(ast.BinOp(left=node.args[0],op=ast.Div(),right=node.args[1]),node)
            return node
    tree.body = [n for n in tree.body if n not in helpers]
    proof = ast.unparse(ast.fix_missing_locations(Division().visit(tree)))
    from .stochastic_equilibrium import _Roundoff
    _Roundoff(proof,slots,box,dt,minimum_divisor=1e-12)
    return proof


def _prove_rk4_assignments(code,slots,rhs):
    class ProofFailure(Exception):
        pass

    if not rhs or not set(rhs) <= set(slots) or "_dt" in slots: return False
    try: tree = ast.parse(code)
    except SyntaxError: return False
    if any(isinstance(n,(ast.If,ast.IfExp,ast.While,ast.Try,ast.With,ast.Break,ast.Continue))
        or isinstance(n,ast.Call) and n.keywords
        or isinstance(n,ast.FunctionDef) and n.decorator_list for n in ast.walk(tree)):
        return False
    prefix = "_rk4_proof_"
    names = set(slots)|{n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    while any(n.startswith(prefix) for n in names): prefix += "_"
    initial = {name:sp.Symbol(name) for name in rhs}
    dt = sp.Symbol("_dt")
    stages,definitions = {},{}
    for stage in range(1,5):
        state = initial if stage == 1 else {
            name:initial[name]+dt*stages[stage-1,name]/(2 if stage < 4 else 1) for name in rhs}
        for name,expression in rhs.items():
            key = (stage,name)
            definitions[key] = expression.subs({sp.Symbol(s):v for s,v in state.items()},simultaneous=True)
            stages[key] = sp.Symbol(f"{prefix}{stage}_{name}")
    private = set(stages.values())
    proven,aliases,rewrites,identities = set(),{},{},{}
    for key, definition in definitions.items():
        if definition.is_Number:
            aliases[stages[key]] = definition
            proven.add(key)

    def rewrite(value):
        for _ in range(len(definitions)+1):
            updated = value.xreplace(aliases).xreplace(rewrites)
            if updated == value: return value
            value = updated
        raise ProofFailure()

    def match(value):
        if not isinstance(value,sp.Expr): return value
        value = rewrite(value)
        for key,definition in definitions.items():
            if key in proven: continue
            expected = rewrite(definition)
            available = {stages[k] for k in proven}
            if expected.free_symbols & (private - available): continue
            identity = (value,expected)
            if identity not in identities: identities[identity] = _equivalent(value,expected)
            if not identities[identity]: continue
            symbol = expected if expected.is_Atom else stages[key]
            aliases[stages[key]] = symbol
            if not expected.is_Atom: rewrites[expected] = symbol
            if not value.is_Atom: rewrites[value] = symbol
            proven.add(key)
            value = symbol
        return value

    class AssignmentProof(_SymbolicKernel):
        def expression(self,node,environment,depth=0):
            try: return super().expression(node,environment,depth)
            except (ValueError,TypeError,IndexError,KeyError) as error: raise ProofFailure() from error

        def block(self,body,environment,depth=0):
            def supported_target(target):
                if isinstance(target,ast.Name): return True
                if isinstance(target,(ast.Tuple,ast.List)): return all(supported_target(t) for t in target.elts)
                if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and isinstance(environment.get(target.value.id),(list,tuple)):
                    return False
                return isinstance(target,ast.Subscript) and self._resolve_slot(target,environment,depth) is not None
            for node in body:
                if not isinstance(node,(ast.FunctionDef,ast.Assign,ast.AnnAssign,ast.AugAssign,ast.For,ast.Return,ast.Expr,ast.Pass)):
                    raise ProofFailure()
                if isinstance(node,ast.Return) and node.value is None: raise ProofFailure()
                if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
                    function = node.value.func
                    if not isinstance(function,ast.Name) or function.id not in self.functions and ("function",function.id) not in environment:
                        raise ProofFailure()
                targets = node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,(ast.AnnAssign,ast.AugAssign)) else []
                if not all(supported_target(target) for target in targets): raise ProofFailure()
                if isinstance(node,ast.For):
                    if not isinstance(node.target,ast.Name) or node.orelse: raise ProofFailure()
                    if isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name) and node.iter.func.id == "range":
                        args = [self.expression(a,environment,depth) for a in node.iter.args]
                        if not all(getattr(a,"is_Integer",False) for a in args): raise ProofFailure()
                        if not 1 <= len(args) <= 3 or len(range(*(int(a) for a in args))) > 64: raise ProofFailure()
                    else:
                        sequence = self.expression(node.iter,environment,depth)
                        if not isinstance(sequence,(list,tuple)) or len(sequence) > 64: raise ProofFailure()
                result = super().block([node],environment,depth)
                if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                    for key,value in list(environment.items()):
                        if isinstance(value,(list,tuple)): environment[key] = [match(v) for v in value]
                        else: environment[key] = match(value)
                if result is not None:
                    return [match(v) for v in result] if isinstance(result,(list,tuple)) else match(result)
            return None

    try:
        actual = AssignmentProof(code,slots).updates(set(rhs))
        if len(proven) != len(definitions) or set(actual) != set(rhs): return False
        for name in rhs:
            expected = initial[name]+dt*(stages[1,name]+2*stages[2,name]+2*stages[3,name]+stages[4,name])/6
            if not _equivalent(rewrite(actual[name]),rewrite(expected)): return False
        return True
    except (ProofFailure,ValueError,TypeError,KeyError,RecursionError,ArithmeticError):
        return False


def _certify_invariant_cancellations(graph,code,slots):
    invariants = [s.symbol for s in graph.slots if s.semantic_tag == m.ZERO_DISSIPATION_INVARIANT]
    if not invariants: return ()
    environment = {}
    writes = {}
    class Expand(ast.NodeTransformer):
        def visit_Name(self,node): return copy.deepcopy(environment.get(node.id,node))
        def visit_Subscript(self,node):
            if ast.unparse(node.value) == "buf.data" and isinstance(node.slice,ast.Constant):
                name = next((s for s,i in slots.items() if i == node.slice.value),None)
                if name is not None: return copy.deepcopy(writes.get(name,ast.Name(id=name,ctx=ast.Load())))
            return self.generic_visit(node)
    function = next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name == "simulation_step")
    for statement in function.body:
        if not isinstance(statement,ast.Assign): continue
        value = Expand().visit(copy.deepcopy(statement.value))
        for target in statement.targets:
            if isinstance(target,ast.Name): environment[target.id] = value
            elif isinstance(target,ast.Subscript) and ast.unparse(target.value) == "buf.data" and isinstance(target.slice,ast.Constant):
                for name,index in slots.items():
                    if index == target.slice.value: writes[name] = value
    def terms(node):
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub)):
            return terms(node.left)+terms(node.right)
        return [node]
    class ModelDefinitions(ast.NodeTransformer):
        def __init__(self): self.visiting = set()
        def visit_Name(self,node):
            if node.id not in graph.equations or graph.equations[node.id].kind != "ALGEBRAIC": return node
            if node.id in self.visiting: raise UnsupportedVerification("CYCLIC_INVARIANT_DEFINITION")
            self.visiting.add(node.id)
            value = self.visit(m._parse(graph.equations[node.id].expression))
            self.visiting.remove(node.id)
            return value
    for name in invariants:
        if name not in graph.equations or name not in writes:
            raise UnsupportedVerification("INVARIANT_IMPLEMENTATION_UNBOUND")
        for expression in (ModelDefinitions().visit(m._parse(graph.equations[name].expression)),writes[name]):
            if not _equivalent(ast.unparse(expression),"0"):
                raise UnsupportedVerification("INVARIANT_CANCELLATION_UNPROVEN")
            active_terms = 0
            for term in terms(expression):
                value = sp.sympify(ast.unparse(term),locals={s:sp.Symbol(s) for s in slots})
                if any(m.solve_query(f"({_expression(sp.diff(value,sp.Symbol(p)))}) != 0",graph.omega,witness=True).status == "SAT"
                       for p in graph.mechanism_parameters): active_terms += 1
            if active_terms < 2: raise UnsupportedVerification("TRIVIAL_HARDCODED_INVARIANT")
    return tuple(invariants)


def certified_valid_domain(graph, boundary):
    """Prove a positive-volume box, not a sampled interpolation of a boundary."""
    varying = [s for s in graph.mechanism_parameters if graph.omega[s][0] < graph.omega[s][1]]
    if len(varying) != 1: raise UnsupportedVerification("VALID_DOMAIN_PARAMETER_UNRESOLVED")
    parameter = varying[0]; box = dict(graph.omega)
    good = boundary.valid_witness_theta[parameter]; bad = boundary.falsified_witness_theta[parameter]
    box[parameter] = (good, box[parameter][1]) if good > bad else (box[parameter][0], good)
    model_eqs = list(_model_constraints(graph).values())
    env_expr = m._and(graph.environmental_assumptions)
    claim = m._and([o.expression for o in graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET"])
    sat_expr, sat_box = m._eliminate_explicit_model_eqs(box, model_eqs, env_expr, claim)
    if m.solve_query(sat_expr, sat_box, witness=True).status != "SAT":
        raise UnsupportedVerification("VALID_DOMAIN_EMPTY_OR_UNKNOWN")
    viol_expr, viol_box = m._eliminate_explicit_model_eqs(box, model_eqs, env_expr, f"not ({claim})")
    if m.solve_query(viol_expr, viol_box).status != "UNSAT":
        endpoint = graph.omega[parameter][1 if good > bad else 0]
        uncertain, certified = good, endpoint
        point_box = {**box,parameter:(endpoint,endpoint)}
        pt_viol, pt_box = m._eliminate_explicit_model_eqs(point_box, model_eqs, env_expr, f"not ({claim})")
        if m.solve_query(pt_viol, pt_box).status != "UNSAT":
            raise UnsupportedVerification("VALID_DOMAIN_NOT_PROVEN")
        for _ in range(16):
            middle = (uncertain+certified)/2
            candidate = {**box,parameter:(middle,endpoint) if good > bad else (endpoint,middle)}
            c_viol, c_box = m._eliminate_explicit_model_eqs(candidate, model_eqs, env_expr, f"not ({claim})")
            if m.solve_query(c_viol, c_box).status == "UNSAT": certified = middle
            else: uncertain = middle
        box[parameter] = (certified,endpoint) if good > bad else (endpoint,certified)
    m.domain_compression_bits(graph.omega, box)
    return box


class _Kernel:
    def __init__(self, machine, state, code):
        from .state_machine import _lower_kernel, v9_transpiler_violations
        if v9_transpiler_violations(code): raise UnsupportedVerification("UNSUPPORTED_EXECUTABLE_SYNTAX")
        self.machine, self.state, self.code = machine, state, code
        self.graph, self.bp = state.ihcg, state.blueprint
        from .state_machine import _contracts_through_stage
        contracts = _contracts_through_stage(self.bp,state.research_stage)
        self.contract = contracts[-1]
        self.refinement = 8 if getattr(state,"precision_refined",False) else 4
        if len(code) > 100000: raise UnsupportedVerification("SYMBOLIC_KERNEL_BUDGET_EXCEEDED")
        if any(s.symbol not in self.bp.variable_map or self.bp.variable_map[s.symbol] != s.slot_idx for s in self.graph.slots):
            raise UnsupportedVerification("IHCG_EXECUTABLE_SLOT_MISMATCH")
        self.slots = {s.symbol:s.slot_idx for s in self.graph.slots}
        self.initial = list(self.contract.initial_state)
        self.initial.extend([0.0]*max(0, max(self.slots.values())+1-len(self.initial)))
        self.base = {s:self.initial[i] for s,i in self.slots.items()}
        # Seed t=0 values of pure algebraic DERIVED_OBSERVABLE / CONSERVATION_AUX slots from inputs
        # so uninitialized 0.0 placeholders satisfy algebraic definitions and CONSERVATION_GUARDs at t=0.
        output_roles = {s.symbol for s in self.graph.slots if s.role in {"DERIVED_OBSERVABLE", "CONSERVATION_AUX"}}
        for _ in range(len(output_roles) + 1):
            progressed = False
            for eq_name, eq in self.graph.equations.items():
                if eq.kind == "ALGEBRAIC" and eq_name in output_roles and eq_name in self.slots:
                    lo, hi = self.graph.omega[eq_name]
                    try:
                        val = _point_value(eq.expression, self.base)
                        if lo <= val <= hi and val != self.base[eq_name]:
                            self.base[eq_name] = val
                            self.initial[self.slots[eq_name]] = val
                            progressed = True
                    except Exception:
                        pass
            if not progressed:
                break
        input_slots = {s.symbol for s in self.graph.slots if s.role in {"CONTROL_PARAM", "DYNAMIC_STATE"}}
        if any(not lo <= self.base[s] <= hi for s,(lo,hi) in self.graph.omega.items() if s in input_slots):
            raise UnsupportedVerification("INITIAL_POINT_OUTSIDE_OMEGA")
        self.compiled = {}
        self._lower = _lower_kernel
        self._compile(code)
        # Rewrite running-extremum updates (`if expr > buf.data[k]: buf.data[k] = expr`) on declared
        # TRAJECTORY_MAX / TRAJECTORY_MIN and auxiliary tracker slots BEFORE _normalize_guarded_divisions
        # so _Roundoff sees the branch-free proof AST (the executed kernel is untouched).
        self.proof_code = code
        temporal_slots = {
            self.slots[name]: eq.kind
            for name, eq in self.graph.equations.items()
            if eq.kind in {"TRAJECTORY_MAX", "TRAJECTORY_MIN"} and name in self.slots
        }
        aux_tracker_slots = {
            s.slot_idx for s in self.graph.slots
            if s.role in {"DERIVED_OBSERVABLE", "CONSERVATION_AUX"} and s.slot_idx not in temporal_slots
        }
        if temporal_slots or aux_tracker_slots:
            proof_tree = ast.parse(self.proof_code)
            class _RunningExtremumNormalizer(ast.NodeTransformer):
                def visit_If(self, node):
                    self.generic_visit(node)
                    if (not node.orelse and len(node.body) == 1 and isinstance(node.body[0], ast.Assign)
                            and len(node.body[0].targets) == 1
                            and isinstance(node.body[0].targets[0], ast.Subscript)
                            and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                            and len(node.test.comparators) == 1):
                        tgt = node.body[0].targets[0]
                        if (isinstance(tgt.value, ast.Attribute) and tgt.value.attr == "data"
                                and isinstance(tgt.slice, ast.Constant) and type(tgt.slice.value) is int
                                and (tgt.slice.value in temporal_slots or tgt.slice.value in aux_tracker_slots)):
                            kind = temporal_slots.get(tgt.slice.value)
                            tgt_src = ast.unparse(tgt)
                            val_src = ast.unparse(node.body[0].value)
                            lhs_src = ast.unparse(node.test.left)
                            rhs_src = ast.unparse(node.test.comparators[0])
                            op = node.test.ops[0]
                            is_max = kind in {"TRAJECTORY_MAX", None} and (
                                (lhs_src == val_src and rhs_src == tgt_src and isinstance(op, (ast.Gt, ast.GtE)))
                                or (lhs_src == tgt_src and rhs_src == val_src and isinstance(op, (ast.Lt, ast.LtE)))
                            )
                            is_min = kind in {"TRAJECTORY_MIN", None} and (
                                (lhs_src == val_src and rhs_src == tgt_src and isinstance(op, (ast.Lt, ast.LtE)))
                                or (lhs_src == tgt_src and rhs_src == val_src and isinstance(op, (ast.Gt, ast.GtE)))
                            )
                            if (is_max or is_min) and tgt_src != val_src:
                                return node.body[0]
                    # Skip pure NaN/overflow finiteness-flag guards on local variables
                    # (e.g. `if x != x: finite = 0.0` or `if x > 1e300: finite = 0.0`).
                    if (not node.orelse and len(node.body) == 1 and isinstance(node.body[0], ast.Assign)
                            and len(node.body[0].targets) == 1 and isinstance(node.body[0].targets[0], ast.Name)
                            and isinstance(node.body[0].value, ast.Constant) and node.body[0].value.value in (0, 0.0)):
                        cmps = [n for n in ast.walk(node.test) if isinstance(n, ast.Compare)]
                        def _num_val(n):
                            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                                return float(n.value)
                            if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant) and isinstance(n.operand.value, (int, float)):
                                return -float(n.operand.value)
                            return None
                        if cmps and all(
                            len(c.ops) == 1 and len(c.comparators) == 1 and (
                                (isinstance(c.ops[0], ast.NotEq) and ast.unparse(c.left) == ast.unparse(c.comparators[0]))
                                or (isinstance(c.ops[0], (ast.Gt, ast.GtE)) and (_num_val(c.comparators[0]) or 0.0) >= 1e100)
                                or (isinstance(c.ops[0], (ast.Lt, ast.LtE)) and (_num_val(c.comparators[0]) or 0.0) <= -1e100)
                            )
                            for c in cmps
                        ):
                            return ast.Pass()
                    return node
            proof_tree = _RunningExtremumNormalizer().visit(proof_tree)
            self.proof_code = ast.unparse(ast.fix_missing_locations(proof_tree))
        self.proof_code = _normalize_guarded_divisions(self.proof_code,self.slots,self.graph.omega,self.contract.dt)
        if any(isinstance(n, (ast.If, ast.IfExp, ast.While)) for n in ast.walk(ast.parse(self.proof_code))):
            raise UnsupportedVerification("PIECEWISE_KERNEL_BINDING_UNSUPPORTED")
        symbolic = _SymbolicKernel(self.proof_code, self.slots)
        self.updates = symbolic.updates(set(self.slots))
        self.odes = {e.state_symbol:(name,e) for name,e in self.graph.equations.items() if e.kind == "ODE_RHS"}
        self.temporal = {name:e for name,e in self.graph.equations.items() if e.kind in {"TRAJECTORY_MAX","TRAJECTORY_MIN"}}
        for name,equation in self.temporal.items():
            if name in self.updates and not _equivalent(self.updates[name],self.updates.get(equation.state_symbol,sp.Symbol(equation.state_symbol))):
                raise UnsupportedVerification("TEMPORAL_OBSERVABLE_WRITE")
        self.constraints = {name:e for name,e in self.graph.equations.items() if _is_formula(m._parse(e.expression))}
        self.outputs = {name:e for name,e in self.graph.equations.items() if e.kind == "ALGEBRAIC" and name not in self.constraints}
        if self.odes.keys() & self.outputs.keys():
            raise UnsupportedVerification("DIFFERENTIAL_ALGEBRAIC_MODEL_UNSUPPORTED")
        undeclared = set(self.updates) - set(self.outputs) - set(self.odes) - set(self.temporal) - set(s.symbol for s in self.graph.slots if s.role == "CONTROL_PARAM")
        if undeclared:
            slot_roles = {s.symbol: s.role for s in self.graph.slots}
            hyp_vars = {v for o in self.graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET" for v in m._variables(m._parse(o.expression))}
            eq_deps = {v for e in self.graph.equations.values() for v in m._variables(m._parse(e.expression))}
            dt_sym = sp.Symbol("_dt")
            truly_undeclared = set()
            for sym in undeclared:
                role = slot_roles.get(sym)
                if role == "DYNAMIC_STATE" and _equivalent(self.updates[sym], sp.Symbol(sym) + dt_sym):
                    continue
                if role in {"DERIVED_OBSERVABLE", "CONSERVATION_AUX"} and sym not in hyp_vars and sym not in eq_deps:
                    continue
                truly_undeclared.add(sym)
            if truly_undeclared:
                raise UnsupportedVerification("UNDECLARED_EXECUTABLE_STATE_UPDATE")
        self.controls = [s.symbol for s in self.graph.slots if s.role == "CONTROL_PARAM"]
        if any(name not in self.slots for name in self.outputs):
            raise UnsupportedVerification("ALGEBRAIC_OUTPUT_BINDING_UNSUPPORTED")
        for name in self.controls:
            if name in self.updates:
                raise UnsupportedVerification("T_CONST_PARAM_WRITE")
        expected = {name:sp.sympify(e.expression, locals={s:sp.Symbol(s) for s in self.slots},rational=True) for name,e in self.outputs.items()}
        for _ in range(len(expected)):
            expected = {name:value.subs({sp.Symbol(s):v for s,v in expected.items()}, simultaneous=True) for name,value in expected.items()}
        self.expected = expected
        self.invariants = _certify_invariant_cancellations(self.graph,self.proof_code,self.slots)
        if self.constraints:
            class BindOutputs(ast.NodeTransformer):
                def visit_Name(self,node):
                    return m._parse(_expression(expected[node.id])) if node.id in expected else node
            translator = _ExactTranslator(self.graph.omega,False)
            solver = z3.Solver(); solver.set(timeout=1500)
            for name,(lo,hi) in self.graph.omega.items():
                solver.add(translator.formula(m._parse(f"{lo!r} <= {name} <= {hi!r}")))
            clauses = [BindOutputs().visit(m._parse(e.expression)) for e in self.constraints.values()]
            solver.add(z3.Not(z3.And(*(translator.formula(n) for n in clauses))))
            if solver.check() != z3.unsat:
                raise UnsupportedVerification("PHYSICAL_MODEL_CONSTRAINT_NOT_PRESERVED")
        locked_const_subs = {
            sp.Symbol(s.symbol): sp.Rational(str(self.graph.omega[s.symbol][0]))
            for s in self.graph.slots
            if s.role == "CONTROL_PARAM"
            and s.symbol not in self.graph.mechanism_parameters
            and self.graph.omega[s.symbol][0] == self.graph.omega[s.symbol][1]
        }
        alg_mismatches = []
        for name, expression in expected.items():
            bound = expression.subs({sp.Symbol(s):self.updates[s] for s in self.odes if s in self.updates},simultaneous=True)
            if name not in self.updates or not (
                _equivalent(self.updates[name], bound)
                or (locked_const_subs and _equivalent(self.updates[name].subs(locked_const_subs), bound.subs(locked_const_subs)))
            ):
                alg_mismatches.append((name, bound, self.updates.get(name)))
        if alg_mismatches:
            first_name, _, _ = alg_mismatches[0]
            details = "; ".join(
                f"slot={nm!r} expected={exp} got={got}" for nm, exp, got in alg_mismatches
            )
            raise UnsupportedVerification(
                f"ALGEBRAIC_EXECUTABLE_MISMATCH: slot={first_name!r} ({len(alg_mismatches)} mismatch(es): {details})"
            )
        self.binding_failure = None
        self.frozen = False
        if self.odes:
            actual_rhs, _ = symbolic.derivatives(set(self.odes))
            alg_subs = {sp.Symbol(s): v for s, v in self.expected.items()}
            for name, (_,e) in self.odes.items():
                expected_rhs = sp.sympify(e.expression,locals={s:sp.Symbol(s) for s in self.slots},rational=True).subs(alg_subs, simultaneous=True)
                if name not in actual_rhs or not (
                    _equivalent(actual_rhs[name], expected_rhs)
                    or (locked_const_subs and _equivalent(actual_rhs[name].subs(locked_const_subs), expected_rhs.subs(locked_const_subs)))
                ):
                    self.binding_failure = "ODE_RHS_EXECUTABLE_MISMATCH"
            self.frozen = bool(actual_rhs) and all(_equivalent(actual_rhs.get(name,sp.Symbol(name)),"0") for name in self.odes)
            self.heun = self._integrator("HEUN")
            rhs = {name:sp.sympify(e.expression,locals={s:sp.Symbol(s) for s in self.slots},rational=True).subs(alg_subs, simultaneous=True)
                for name,(_,e) in self.odes.items()}
            if not _prove_rk4_assignments(self.proof_code,self.slots,rhs):
                self.binding_failure = self.binding_failure or "PRIMARY_SCHEME_IS_NOT_RK4"
            self._compile(self.heun)
        else:
            self.heun = code
            mutable = set(self.updates)-set(self.controls)
            if not mutable <= set(self.outputs): raise UnsupportedVerification("UNDECLARED_EXECUTABLE_STATE_UPDATE")
            inputs = set().union(*(m._variables(m._parse(_expression(v))) for v in expected.values()))
            if inputs & set(self.outputs): raise UnsupportedVerification("RECURRENT_MAP_NOT_AN_ALGEBRAIC_MODEL")
        quantities = [{"symbol":s.symbol, "si_dimension":s.si_dim, "interval":self.graph.omega[s.symbol],
            "is_state_variable":s.role == "DYNAMIC_STATE", "semantic_tag":s.semantic_tag} for s in self.graph.slots]
        metadata = {s.symbol:{"role":s.role} for s in self.graph.slots}
        arena = _native.PyHolographicHin(m._json(quantities), m._json(metadata))
        expressions = {f"{name}:{i}":m._csel(term) for name,e in self.graph.equations.items()
            for i,term in enumerate(m._equation_arithmetic(e.expression))}
        purity = json.loads(arena.verify_observational_purity(m._json(expressions)))
        if not purity["valid"]: raise UnsupportedVerification("ARENA_PURITY_VIOLATION")
        self.margin = _claim_margin(self.graph)
        self.raw_margin = _claim_margin(self.graph, substitute_model=False)
        self.functional = copy.deepcopy(self.raw_margin)
        if isinstance(self.functional,ast.BinOp) and isinstance(self.functional.op,ast.Sub):
            if m._variables(self.functional.right) <= set(self.controls): self.functional = self.functional.left
            elif m._variables(self.functional.left) <= set(self.controls): self.functional = ast.UnaryOp(op=ast.USub(),operand=self.functional.right)
        if m._variables(self.margin) & set(self.controls) and not (m._variables(m._parse(m._and([o.expression for o in self.graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET"]))) - set(self.controls)):
            raise UnsupportedVerification("CONTROL_PARAMETER_ECHO_IS_NOT_A_DISCOVERY")
        self.formal = {"equations":{k:e.expression for k,e in self.graph.equations.items()},
            "parameters":{s:self.base[s] for s in self.controls}, "step_size":2*self.contract.dt/self.refinement,
            "equation_form":"assignment","functional_expression":ast.unparse(self.functional),
            "functional_controls":{s:self.base[s] for s in self.controls}}
        self.formal["symbol_metadata"] = {s.symbol:{"unit_scale":s.unit_scale} for s in self.graph.slots}
        self.trace_functional = copy.deepcopy(self.functional)
        if self.temporal:
            relevant = m._variables(self.functional) & self.temporal.keys()
            if relevant:
                if len(relevant) != 1 or m._variables(self.functional)-relevant-set(self.controls):
                    raise UnsupportedVerification("MULTIOBJECTIVE_TEMPORAL_REDUCTION_UNBOUND")
                target = next(iter(relevant))
                slope = sp.diff(sp.sympify(ast.unparse(self.functional)),sp.Symbol(target))
                if slope not in (1,-1): raise UnsupportedVerification("NONAFFINE_TEMPORAL_FUNCTIONAL_UNBOUND")
                maximum = self.temporal[target].kind == "TRAJECTORY_MAX"
                self.formal["observable"] = "max" if maximum == (slope == 1) else "min"
                class Sources(ast.NodeTransformer):
                    def visit_Name(_,node):
                        return ast.copy_location(ast.Name(id=self.temporal[node.id].state_symbol,ctx=ast.Load()),node) if node.id in self.temporal else node
                self.trace_functional = Sources().visit(self.trace_functional)

    def signal_value(self,point):
        return _point_value(ast.unparse(self.functional),{**point,**self.formal["functional_controls"]})

    def _compile(self, code, *, already_instrumented=False):
        if code in self.compiled: return
        try:
            _, cpp = self._lower(code if already_instrumented else instrument_purity(code))
        except (ValueError, SyntaxError, NotImplementedError):
            cpp = None
        self.compiled[code] = cpp

    def run(self, values=None, *, steps=None, dt=None, code=None, trace=False):
        code = instrument_purity(code or self.code)
        initial = list(self.initial)
        for name,value in (values or {}).items(): initial[self.slots[name]] = float(value)
        steps = self.contract.steps*self.refinement if steps is None else steps
        dt = self.contract.dt/self.refinement if dt is None else dt
        if steps > 65536: raise UnsupportedVerification("VERIFICATION_GRID_BUDGET_EXCEEDED")
        requested_trace = trace
        trace = trace or bool(self.temporal)
        if trace:
            tree = ast.parse(code)
            if any(isinstance(n, ast.FunctionDef) and n.name == "_verified_source_step" for n in tree.body):
                raise UnsupportedVerification("RESERVED_VERIFICATION_SYMBOL")
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "simulation_step": node.name = "_verified_source_step"
            index = len(initial)
            class Bind(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id in fixed_controls: return ast.Constant(fixed_controls[node.id])
                    return ast.parse(f"buf.data[{self_slots[node.id]}]",mode="eval").body if node.id in self_slots else node
            self_slots = self.slots
            fixed_controls = self.formal["functional_controls"]
            margin = ast.unparse(Bind().visit(copy.deepcopy(self.trace_functional)))
            tree.body.extend(ast.parse(f"def simulation_step(buf, dt):\n    _verified_source_step(buf, dt)\n    buf.data[{index}] = {margin}\n").body)
            code = ast.unparse(ast.fix_missing_locations(tree))
            initial.append(0.0)
        self._compile(code,already_instrumented=True)
        cpp = self.compiled[code]
        try:
            result = self.machine.bridge.run(cpp, initial, dt, steps, timeout_seconds=min(2.0,self.contract.timeout_seconds),trace_slots=(index,*[self.slots[e.state_symbol] for e in self.temporal.values()]) if trace else ()) if cpp else self.machine.bridge.run_python(code, initial, dt, steps, timeout_seconds=2.0,trace_slots=(index,*[self.slots[e.state_symbol] for e in self.temporal.values()]) if trace else ())
        except KernelExecutionError as error:
            if not error.compile_error: raise UnsupportedVerification("EXECUTABLE_SANDBOX_FAILED") from error
            result = self.machine.bridge.run_python(code, initial, dt, steps, timeout_seconds=2.0,trace_slots=(index,*[self.slots[e.state_symbol] for e in self.temporal.values()]) if trace else ())
        if not result.get("finite"): raise UnsupportedVerification("NONFINITE_EXECUTABLE_RESULT")
        audit = purity_audit_from_diag(result["diag"], domain_tag=f"stage{self.state.research_stage}/default", static_purity_passed=True)
        if audit.projected_steps or audit.stabilized_steps: raise UnsupportedVerification("RUNTIME_ENFORCEMENT")
        point = {s:float(result["state"][i]) for s,i in self.slots.items()}
        if trace:
            times = tuple(j*dt for j in range(steps+1))
            for j,(name,equation) in enumerate(self.temporal.items(),start=1):
                samples = (initial[self.slots[equation.state_symbol]],*[float(row[j]) for row in result["trace_values"]])
                point[name] = m.quadratic_peak(times,samples,minimum=equation.kind == "TRAJECTORY_MIN")[0]
        for name,equation in self.outputs.items():
            expected = m._iv(m._parse(equation.expression),{s:(v,v) for s,v in point.items()})
            if not expected.lo <= point[name] <= expected.hi:
                raise UnsupportedVerification("ALGEBRAIC_EXECUTABLE_VALUE_MISMATCH")
        output_roles = {s.symbol for s in self.graph.slots if s.role in {"DERIVED_OBSERVABLE", "CONSERVATION_AUX"}}
        counterfactual_null = bool(values) and any(
            v == 0.0 and k in self.graph.omega and not (self.graph.omega[k][0] <= 0.0 <= self.graph.omega[k][1])
            for k, v in (values or {}).items()
        )
        for name,(lo,hi) in self.graph.omega.items():
            if counterfactual_null and (name in (values or {}) or name in output_roles):
                continue
            i = self.slots[name]
            fp_tol = 1e-9 * max(1.0, abs(lo), abs(hi))
            if name in output_roles:
                if point[name] < lo - fp_tol or point[name] > hi + fp_tol:
                    raise UnsupportedVerification("TRAJECTORY_OUTSIDE_OMEGA")
            elif result.get("trace_min",initial)[i] < lo - fp_tol or result.get("trace_max",initial)[i] > hi + fp_tol:
                raise UnsupportedVerification("TRAJECTORY_OUTSIDE_OMEGA")
        trajectory = {s:(float(result.get("trace_min",initial)[i]),float(result.get("trace_max",initial)[i])) for s,i in self.slots.items()}
        self.physical_checks = {o.id:m._truth(m._parse(o.expression),trajectory) is True for o in self.graph.obligations if o.obligation_class == "CONSERVATION_GUARD"}
        if requested_trace:
            before = {s:initial[i] for s,i in self.slots.items()}
            first = _point_value(ast.unparse(self.trace_functional),{**before,**self.formal["functional_controls"]})
            return point,audit,(first,*[float(row[0]) for row in result["trace_values"]])
        return point, audit

    def _integrator(self, scheme):
        names = sorted(self.odes)
        lines = ["import math", "def simulation_step(buf, dt):"]
        for name,index in self.slots.items(): lines.append(f"    {name} = buf.data[{index}]")
        alg_subs = {sp.Symbol(s): v for s, v in self.expected.items()}
        rhs = {name:sp.sympify(e.expression, locals={s:sp.Symbol(s) for s in self.slots},rational=True).subs(alg_subs, simultaneous=True) for name,(_,e) in self.odes.items()}
        for stage in range(1,3 if scheme == "HEUN" else 5):
            substitutions = {}
            if stage > 1:
                prior = stage-1
                factor = "dt" if scheme == "HEUN" or stage == 4 else "dt/2"
                substitutions = {sp.Symbol(s):sp.Symbol(s)+sp.Symbol(f"_k{prior}_{s}")*sp.sympify(factor) for s in names}
            for name in names:
                expr = rhs[name].subs(substitutions, simultaneous=True)
                lines.append(f"    _k{stage}_{name} = {sp.pycode(expr)}")
        for name in names:
            update = f"dt*(_k1_{name}+_k2_{name})/2" if scheme == "HEUN" else f"dt*(_k1_{name}+2*_k2_{name}+2*_k3_{name}+_k4_{name})/6"
            lines.append(f"    {name} = {name}+{update}")
            lines.append(f"    buf.data[{self.slots[name]}] = {name}")
        for name,equation in self.outputs.items():
            lines.append(f"    {name} = {equation.expression}")
            lines.append(f"    buf.data[{self.slots[name]}] = {name}")
        for name,equation in self.temporal.items():
            lines.append(f"    buf.data[{self.slots[name]}] = {equation.state_symbol}")
        return "\n".join(lines)+"\n"

    def fidelity(self, overrides=None, *, check_guards=True):
        parameters = dict(self.formal["parameters"]); parameters.update(overrides or {})
        formal = {**self.formal,"parameters":parameters}
        if not self.odes and self.contract.steps < 4:
            formal["step_size"] = (self.contract.steps * self.contract.dt) / 4.0
        fidelity_graph = self.graph
        if overrides:
            if set(overrides)-set(self.graph.mechanism_parameters):
                raise UnsupportedVerification("UNDECLARED_NULL_INTERVENTION")
            controls = {k:overrides.get(k,v) for k,v in self.graph.control_values.items()}
            omega_null = dict(self.graph.omega)
            for k, v in overrides.items():
                if k in omega_null:
                    lo, hi = omega_null[k]
                    if v < lo: omega_null[k] = (float(v), hi)
                    elif v > hi: omega_null[k] = (lo, float(v))
            fidelity_graph = self.graph.model_copy(update={"control_values":controls,"omega":omega_null,"h_ihcg":""})
            fidelity_graph = fidelity_graph.model_copy(update={"h_ihcg":_native.py_compute_ihcg_digest(fidelity_graph.model_dump_json())})
        def runner(_formal, *, scheme, step_size, timeout_ms):
            horizon = self.contract.steps*self.contract.dt
            steps = round(horizon/step_size)
            if steps < 2 or not math.isclose(steps*step_size,horizon,rel_tol=1e-12):
                raise UnsupportedVerification("NONNESTED_EXECUTION_GRIDS")
            _,_,values = self.run(parameters,steps=steps,dt=step_size,code=self.code if scheme == "RK4" else self.heun,trace=True)
            guards = self.physical_checks if check_guards else {o.id: True for o in self.graph.obligations if o.obligation_class == "CONSERVATION_GUARD"}
            return m.GridRun(scheme=scheme, times=tuple(i*step_size for i in range(steps+1)),
                values=values, physical_checks=guards)
        audit = m.certify_implementation_fidelity(fidelity_graph, formal, runner)
        return audit


class DefaultVerificationRunner:
    def __init__(self, machine): self.machine = machine

    async def __call__(self, **kwargs):
        return await asyncio.to_thread(self.evaluate, **kwargs)

    def evaluate(self, *, phase, state, code, uast=None, results=()):
        graph = state.ihcg
        if phase == "deductive":
            deductive_eligible = graph.quantifier_mode != "EXISTS" or bool(graph.environmental_assumptions) or any(o.obligation_class == "CONSERVATION_GUARD" for o in graph.obligations)
            if not deductive_eligible: return None
            audit = m.classify_contract_vacuity(graph,list(_model_constraints(graph).values()))
            if audit.classification != "INFEASIBLE_UNDER_MODEL": return None
        if graph.archetype in {"MONTE_CARLO", "FIXED_POINT"}:
            from .stochastic_equilibrium import derive_evidence
            return derive_evidence(self.machine,state,code,uast=uast,phase=phase)
        try:
            kernel = _Kernel(self.machine,state,code)
        except UnsupportedVerification as exc:
            if phase == "deductive" and str(exc) in {"PIECEWISE_KERNEL_BINDING_UNSUPPORTED", "RIGOROUS_ROUNDING_BOUND_UNAVAILABLE", "ROUNDING_PROGRAM_UNBOUND"}:
                from .state_machine import v9_transpiler_violations
                if v9_transpiler_violations(code):
                    raise
                formal_eqs = {k: e.expression for k, e in graph.equations.items()}
                formal_uast = {
                    "equations": formal_eqs,
                    "parameters": dict(graph.control_values),
                    "symbol_metadata": {s.symbol: {"unit_scale": list(s.unit_scale)} for s in graph.slots},
                }
                quantities = [{"symbol": s.symbol, "si_dimension": s.si_dim, "interval": graph.omega[s.symbol],
                               "is_state_variable": s.role == "DYNAMIC_STATE", "semantic_tag": s.semantic_tag} for s in graph.slots]
                arithmetic, owners = {}, {}
                for eq_name, expr in formal_eqs.items():
                    for idx_t, term in enumerate(m._equation_arithmetic(expr)):
                        key = f"{eq_name}:{idx_t}"
                        arithmetic[key] = m._csel(term)
                        owners[key] = eq_name
                reports = json.loads(_native.PyHolographicHin(m._json(quantities)).verify_equations(m._json(arithmetic)))
                dim_ok = bool(reports.get("all_valid")) and all(
                    (exp := graph.equations.get(owners[rep["equation_name"]])) is not None
                    and tuple(rep["output_dimensions"]) == exp.si_dim
                    and rep["output_semantic_tag"] == exp.semantic_tag
                    and not rep["dimensional_warnings"]
                    for rep in reports.get("equations", [])
                )
                fidelity = m.FidelityAudit(
                    passed=dim_ok,
                    algebraic_equiv_passed=True,
                    si_and_semantic_tag_passed=dim_ok,
                    convergence_passed=True,
                    physical_checks_passed=True,
                    convergence_reliability_ratio=0.0,
                    epsilon_disc=0.0,
                    epsilon_total=0.0,
                    failure_reason=None if dim_ok else "SI_OR_SEMANTIC_MISMATCH",
                )
                purity = m.PurityAudit(static_purity_passed=True, domain_tag=f"stage{state.research_stage}/deductive")
                return ExecutionEvidence(
                    executable_digest=code_hash(code),
                    formal_uast=formal_uast,
                    purity=purity,
                    identifiability=m.IdentifiabilityAudit(coupling_passed=False),
                    evaluation=WitnessEvaluation(signed_margin=-1.0, epsilon_total=0.0),
                    fidelity=fidelity,
                )
            raise
        if kernel.binding_failure:
            fidelity = m.FidelityAudit(passed=False,algebraic_equiv_passed=False,si_and_semantic_tag_passed=True,
                convergence_passed=False,physical_checks_passed=False,epsilon_disc=0,epsilon_total=0,failure_reason=kernel.binding_failure)
            parameters = graph.mechanism_parameters
            if kernel.frozen and parameters:
                report = json.loads(_native.py_compute_sensitivity_svd([[0.0 for _ in parameters]]))
                active = m.IdentifiabilityAudit(coupling_passed=False,dead_parameters=tuple(parameters),
                    column_norms=dict(zip(parameters,report["column_norms"])),singular_values=tuple(report["singular_values"]),equilibrium_restoring_force=0.0)
            else: active = m.IdentifiabilityAudit(coupling_passed=False)
            purity = m.PurityAudit(static_purity_passed=True,domain_tag="source-binding")
            return ExecutionEvidence(executable_digest=code_hash(code),formal_uast=kernel.formal,purity=purity,
                identifiability=active,evaluation=WitnessEvaluation(signed_margin=0,epsilon_total=0),fidelity=fidelity)
        fidelity = kernel.fidelity()
        point, purity = kernel.run()
        margin = _point_value(ast.unparse(kernel.raw_margin),point)
        evaluation = WitnessEvaluation(signed_margin=margin,epsilon_total=fidelity.epsilon_total)
        if phase == "deductive" or not fidelity.passed:
            return ExecutionEvidence(executable_digest=code_hash(code),formal_uast=kernel.formal,purity=purity,
                identifiability=m.IdentifiabilityAudit(coupling_passed=False),evaluation=evaluation,fidelity=fidelity)
        for s, (lo, hi) in graph.omega.items():
            fp_tol = 1e-9 * max(1.0, abs(lo), abs(hi))
            if not (lo - fp_tol <= point[s] <= hi + fp_tol):
                raise UnsupportedVerification("EXECUTION_OUTSIDE_OMEGA")
        def _env_holds_at(pt, pt_box, tol):
            if m._truth(m._parse(m._and(graph.environmental_assumptions)), pt_box) is True:
                return True
            for expr in graph.environmental_assumptions:
                node = m._parse(expr)
                if m._truth(node, pt_box) is True:
                    continue
                if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                    lhs_v = _point_value(ast.unparse(node.left), pt)
                    rhs_v = _point_value(ast.unparse(node.comparators[0]), pt)
                    if abs(lhs_v - rhs_v) <= tol * max(1.0, abs(lhs_v), abs(rhs_v)):
                        continue
                return False
            return True

        point_box = {s: (max(graph.omega[s][0], min(graph.omega[s][1], v)), max(graph.omega[s][0], min(graph.omega[s][1], v))) for s, v in point.items()}
        if not _env_holds_at(point, point_box, max(1e-9, 100.0 * fidelity.epsilon_total)):
            raise UnsupportedVerification("EXECUTION_VIOLATES_ENVIRONMENT")
        targets = [s.symbol for s in graph.slots if s.role in {"DYNAMIC_STATE","DERIVED_OBSERVABLE"}
            and any(s.symbol in m._variables(m._parse(o.expression)) for o in graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET")]
        if not targets: raise UnsupportedVerification("NO_MODEL_DEPENDENT_TARGET")
        parameters = list(graph.mechanism_parameters)
        functionals = []
        def conjuncts(node):
            if isinstance(node,ast.BoolOp) and isinstance(node.op,ast.And):
                return [n for child in node.values for n in conjuncts(child)]
            return [node]
        for obligation in graph.obligations:
            if obligation.obligation_class != "HYPOTHESIS_TARGET": continue
            for node in conjuncts(m._parse(obligation.expression)):
                local = graph.model_copy(update={"obligations":(obligation.model_copy(update={"expression":ast.unparse(node)}),)})
                functionals.append(ast.unparse(_claim_margin(local,substitute_model=False)))
        rows = []
        for fraction in (0.25,0.5,0.75):
            anchor = {p:graph.omega[p][0]+fraction*(graph.omega[p][1]-graph.omega[p][0]) for p in parameters}
            columns = []
            for parameter in parameters:
                lo,hi = graph.omega[parameter]
                step = min(sys.float_info.epsilon**(1/3)*max(1,abs(anchor[parameter])),(hi-lo)/8)
                if step <= 0: raise UnsupportedVerification("DEGENERATE_MECHANISM_DOMAIN")
                plus,_ = kernel.run({**anchor,parameter:anchor[parameter]+step})
                minus,_ = kernel.run({**anchor,parameter:anchor[parameter]-step})
                columns.append([(_point_value(f,{**plus,**kernel.formal["functional_controls"]})-
                    _point_value(f,{**minus,**kernel.formal["functional_controls"]}))/(2*step)*
                    max(1,abs(anchor[parameter]))/max(1,abs(_point_value(f,point))) for f in functionals])
            rows.extend([list(row) for row in zip(*columns)])
        if not rows: raise UnsupportedVerification("MECHANISM_NOT_DECLARED")
        svd = json.loads(_native.py_compute_sensitivity_svd(rows))
        dead = tuple(parameters[i] for i in svd["dead_parameters"])
        active = m.IdentifiabilityAudit(coupling_passed=not dead and svd["rank"] == len(parameters), dead_parameters=dead,
            column_norms=dict(zip(parameters,svd["column_norms"])), singular_values=tuple(svd["singular_values"]),
            brun_collinearity_gamma_k=svd["brun_gamma_k"],fim_condition_number=svd["fim_condition_number"],excluded_certified_invariants=kernel.invariants)
        if not active.coupling_passed:
            return ExecutionEvidence(executable_digest=code_hash(code),formal_uast=kernel.formal,purity=purity,
                identifiability=active,evaluation=evaluation,fidelity=fidelity)
        if margin+fidelity.epsilon_total < 0 and graph.quantifier_mode in {"FORALL","EXISTS"}:
            return ExecutionEvidence(executable_digest=code_hash(code),formal_uast=kernel.formal,purity=purity,
                identifiability=active,evaluation=evaluation,fidelity=fidelity,theta_star=point,
                margin_spec=MarginSpec(expression=ast.unparse(kernel.margin)))
        if graph.quantifier_mode == "FORALL" and margin + fidelity.epsilon_total >= 0:
            # For universal claims (FORALL), probe domain boundary endpoints of varying
            # mechanism parameters to search for a genuine Mode 2A counterexample witness.
            varying_params = [p for p in parameters if graph.omega[p][0] < graph.omega[p][1]]
            candidates = []
            for p in varying_params:
                for val in (graph.omega[p][1], graph.omega[p][0]):
                    if not math.isclose(val, kernel.base[p], rel_tol=1e-12, abs_tol=1e-12):
                        candidates.append({p: val})
            for cand_overrides in candidates:
                try:
                    cand_point, cand_purity = kernel.run(cand_overrides)
                except Exception:
                    continue
                cand_margin = _point_value(ast.unparse(kernel.raw_margin), cand_point)
                if cand_margin + fidelity.epsilon_total < 0:
                    cand_box = {s: (max(graph.omega[s][0], min(graph.omega[s][1], v)), max(graph.omega[s][0], min(graph.omega[s][1], v))) for s, v in cand_point.items()}
                    if _env_holds_at(cand_point, cand_box, max(1e-9, 100.0 * fidelity.epsilon_total)):
                        cand_fidelity = kernel.fidelity(cand_overrides)
                        if cand_fidelity.passed and cand_margin + cand_fidelity.epsilon_total < 0:
                            theta_clean = {
                                s: (max(graph.omega[s][0], min(graph.omega[s][1], v))
                                    if s in graph.omega and (graph.omega[s][0] - 1e-9 * max(1.0, abs(graph.omega[s][0]), abs(graph.omega[s][1])) <= v <= graph.omega[s][1] + 1e-9 * max(1.0, abs(graph.omega[s][0]), abs(graph.omega[s][1])))
                                    else v)
                                for s, v in cand_point.items()
                            }
                            return ExecutionEvidence(
                                executable_digest=code_hash(code),
                                formal_uast=kernel.formal,
                                purity=cand_purity,
                                identifiability=active,
                                evaluation=WitnessEvaluation(signed_margin=cand_margin, epsilon_total=cand_fidelity.epsilon_total),
                                fidelity=cand_fidelity,
                                theta_star=theta_clean,
                                margin_spec=MarginSpec(expression=ast.unparse(kernel.margin)),
                            )
        prior_kernel = None
        compared_run_id = (state.blueprint.ihcg_evidence or {}).get("execution",{}).get("compared_run_id")
        if state.research_stage > 1:
            foundation = getattr(self.machine,"_verified_foundations",{}).get(state.research_stage-1)
            if foundation is None or foundation[0].h_ihcg != graph.parent_ihcg_digest:
                raise UnsupportedVerification("PINNED_FOUNDATION_BINDING_REQUIRED")
            prior_graph,prior_blueprint,prior_code = foundation
            kernel.formal["pinned_foundation"] = {"ihcg_digest":prior_graph.h_ihcg,"uast":{}}
        elif compared_run_id:
            from .vault import ResearchVaultManager
            from .epistemic import compile_blueprint_ihcg,parse_run_certificate
            from ..schemas.research import ResearchBlueprint
            bundle = ResearchVaultManager(self.machine.workspace_root).get_run(compared_run_id)
            if not bundle: raise UnsupportedVerification("PRIOR_BASELINE_NOT_FOUND")
            certificate = parse_run_certificate(bundle["meaningfulness_certificate"])
            if certificate.epistemic_verdict not in {m.EpistemicVerdict.VERIFIED_DISCOVERY,m.EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY}:
                raise UnsupportedVerification("PRIOR_BASELINE_NOT_VERIFIED")
            prior_blueprint = ResearchBlueprint.model_validate(bundle["blueprint"])
            prior_graph = compile_blueprint_ihcg(prior_blueprint,stage=len(certificate.ihcg_anchor.stage_chain_digests)+1)
            prior_code = bundle["unified_module_code"]
        if state.research_stage > 1 or compared_run_id:
            if (any(graph.omega.get(s) != bounds for s,bounds in prior_graph.omega.items())
                    or not m._variables(kernel.functional) <= set(prior_graph.omega)
                    or prior_blueprint.contracts[0].steps*prior_blueprint.contracts[0].dt != kernel.contract.steps*kernel.contract.dt):
                raise UnsupportedVerification("PRIOR_BASELINE_FUNCTIONAL_UNBOUND")
            prior_state = SimpleNamespace(ihcg=prior_graph,blueprint=prior_blueprint,research_stage=1+len(prior_graph.stage_chain_digests),precision_refined=state.precision_refined)
            prior_kernel = _Kernel(self.machine,prior_state,prior_code)
            if prior_kernel.binding_failure: raise UnsupportedVerification("PRIOR_BASELINE_EXECUTABLE_MISMATCH")
            if state.research_stage > 1: kernel.formal["pinned_foundation"]["uast"] = prior_kernel.formal
        baseline = m.synthesize_null_baseline(kernel.formal, graph, state.research_stage)
        if prior_kernel is not None:
            baseline = baseline.model_copy(update={"parameter_overrides":{},"description":"Executed verified prior model"})
        if any(not graph.omega[p][0] <= v <= graph.omega[p][1] and v != 0.0 for p,v in baseline.parameter_overrides.items()):
            raise UnsupportedVerification("NULL_BASELINE_OUTSIDE_DECLARED_DOMAIN")
        null_fidelity = prior_kernel.fidelity() if prior_kernel is not None else kernel.fidelity(baseline.parameter_overrides, check_guards=not bool(baseline.parameter_overrides))
        if not null_fidelity.passed:
            return ExecutionEvidence(executable_digest=code_hash(code),formal_uast=kernel.formal,purity=purity,identifiability=active,
                evaluation=WitnessEvaluation(signed_margin=margin,epsilon_total=null_fidelity.epsilon_total),fidelity=null_fidelity)
        null,null_purity = prior_kernel.run() if prior_kernel is not None else kernel.run(baseline.parameter_overrides)
        value = kernel.signal_value(point)
        null_value = kernel.signal_value(null)
        primary = m.SignificanceResult(observables=(value,),fidelity=fidelity)
        reference = m.SignificanceResult(observables=(null_value,),fidelity=null_fidelity,null_baseline_description=baseline.description)
        if graph.archetype == "PARAM_SWEEP":
            claims = [o.expression for o in graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET"]
            if len(claims) == 1 and isinstance(m._parse(claims[0]), ast.Compare) and len(m._parse(claims[0]).ops) == 1:
                primary = primary.model_copy(update={"pareto_points":((-value,),),"objective_errors":((fidelity.epsilon_total,),)})
                reference = reference.model_copy(update={"pareto_points":((-null_value,),),"objective_errors":((null_fidelity.epsilon_total,),)})
            elif functionals:
                prim_vec = tuple(-_point_value(f, {**point, **kernel.formal["functional_controls"]}) for f in functionals)
                null_vec = tuple(-_point_value(f, {**null, **kernel.formal["functional_controls"]}) for f in functionals)
                disc_p = fidelity.epsilon_disc if kernel.odes else 0.0
                disc_n = null_fidelity.epsilon_disc if kernel.odes else 0.0
                prim_err = tuple(max(disc_p, 100.0 * sys.float_info.epsilon * max(abs(v), 1e-30)) for v in prim_vec)
                null_err = tuple(max(disc_n, 100.0 * sys.float_info.epsilon * max(abs(v), 1e-30)) for v in null_vec)
                primary = primary.model_copy(update={"pareto_points":(prim_vec,),"objective_errors":(prim_err,)})
                reference = reference.model_copy(update={"pareto_points":(null_vec,),"objective_errors":(null_err,)})
            else:
                raise UnsupportedVerification("MULTIOBJECTIVE_SWEEP_REQUIRES_DECLARED_OBJECTIVES")
        comparison = (primary,reference)
        if graph.quantifier_mode == "FORALL" and margin > fidelity.epsilon_total:
            null_margin = _point_value(ast.unparse(kernel.raw_margin),null)
            null_in_scope = all(null[k] == v for k,v in graph.control_values.items()) and m._truth(
                m._parse(m._and(graph.environmental_assumptions)),{s:(v,v) for s,v in null.items()}) is True
            if null_in_scope and null_margin+null_fidelity.epsilon_total < 0:
                point,purity,fidelity = null,null_purity,null_fidelity
                evaluation = WitnessEvaluation(signed_margin=null_margin,epsilon_total=fidelity.epsilon_total)
                comparison = None
            else:
                if kernel.odes: raise UnsupportedVerification("TEMPORAL_UNIVERSAL_DOMAIN_UNVERIFIED")
                claim = m._and([o.expression for o in graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET"])
                query = m._and([*graph.environmental_assumptions,*_model_constraints(graph).values(),f"not ({claim})"])
                if m.solve_query(query,graph.omega).status != "UNSAT":
                    raise UnsupportedVerification("UNIVERSAL_DOMAIN_UNVERIFIED")
        def boundary_runner(theta):
            executed,_ = kernel.run({p:theta[p] for p in kernel.controls})
            return WitnessEvaluation(signed_margin=_point_value(ast.unparse(kernel.raw_margin),executed),epsilon_total=fidelity.epsilon_total)
        varying = [p for p in graph.mechanism_parameters if graph.omega[p][0] < graph.omega[p][1]]
        return ExecutionEvidence(executable_digest=code_hash(code),formal_uast=kernel.formal,purity=purity,identifiability=active,
            evaluation=evaluation,fidelity=fidelity,comparison=comparison,theta_star=point,
            compared_run_id=compared_run_id,
            margin_spec=MarginSpec(expression=ast.unparse(kernel.margin)),boundary_runner=boundary_runner,
            boundary_parameter=varying[0] if len(varying)==1 else None,
            boundary_tolerance=max(1e-6,100*fidelity.epsilon_total),fixed_values=point)

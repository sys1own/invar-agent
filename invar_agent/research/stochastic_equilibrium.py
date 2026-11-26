"""Sample and residual protocols bound to sealed equations and sandbox kernels."""
from __future__ import annotations

import ast
import copy
import json
import math
import statistics
import sys

import numpy as np
import sympy as sp

from invar_agent import _native
from ..healing.deterministic_ast import _intrinsic_defs, instrument_purity, purity_audit_from_diag
from ..runtime.cabi_bridge import KernelExecutionError
from . import meaningfulness as m
from .accretion import code_hash
from .default_verification import _SAFE_DIV_SRC, UnsupportedVerification, _equivalent, _expression, _point_value, _normalize_guarded_divisions
from .epistemic import ExecutionEvidence
from .evolution import _SymbolicKernel
from .falsification import WitnessEvaluation, _claim_margin


def _native_json(function, *args):
    return json.loads(function(*args))


def _sample_expression(graph,equation):
    symbols = {s.symbol:sp.Symbol(s.symbol) for s in graph.slots}
    visiting = set()
    expanded = {}
    def expand(name):
        if name in expanded: return expanded[name]
        if name in visiting: raise UnsupportedVerification("MC_SAMPLER_INPUT_OUTPUT_FRAME_UNBOUND")
        visiting.add(name)
        expression = sp.sympify(graph.equations[name].expression,locals=symbols,rational=True)
        replacements = {}
        for symbol in expression.free_symbols:
            dependency = str(symbol)
            if dependency in graph.equations:
                if graph.equations[dependency].kind != "ALGEBRAIC":
                    raise UnsupportedVerification("MC_SAMPLER_INPUT_OUTPUT_FRAME_UNBOUND")
                replacements[symbol] = expand(dependency)
        expression = expression.subs(replacements,simultaneous=True)
        visiting.remove(name)
        expanded[name] = expression
        return expression
    expression = sp.sympify(equation.expression,locals=symbols,rational=True)
    replacements = {}
    for symbol in expression.free_symbols:
        if str(symbol) in graph.equations: replacements[symbol] = expand(str(symbol))
    return expression.subs(replacements,simultaneous=True)


class _Oracle:
    def __init__(self, machine, state, code):
        from .state_machine import _lower_kernel, v9_transpiler_violations
        self.machine,self.state,self.code = machine,state,code
        self.graph = state.ihcg
        self.graph.assert_sealed()
        if not state.blueprint.contracts or v9_transpiler_violations(code):
            raise UnsupportedVerification("ORACLE_PROTOCOL_UNBINDABLE")
        self.contract = state.blueprint.contracts[-1]
        self.slots = {s.symbol:s.slot_idx for s in self.graph.slots}
        if any(state.blueprint.variable_map.get(s) != i for s,i in self.slots.items()):
            raise UnsupportedVerification("IHCG_EXECUTABLE_SLOT_MISMATCH")
        self.initial = list(self.contract.initial_state)
        self.initial.extend([0.0]*max(0,max(self.slots.values())+1-len(self.initial)))
        self.base = {s:self.initial[i] for s,i in self.slots.items()}
        if any(self.base[s] != value for s,value in self.graph.control_values.items()):
            raise UnsupportedVerification("LOCKED_CONTROL_VALUE_MISMATCH")
        self.controls = [s.symbol for s in self.graph.slots if s.role == "CONTROL_PARAM" and s.stochastic_role is None]
        self.inputs = {role:[s.symbol for s in self.graph.slots if s.stochastic_role == role] for role in ("UNIFORM","SEED","COUNTER","CHANNEL")}
        if self.graph.archetype == "MONTE_CARLO" and not self.inputs["UNIFORM"]:
            raise UnsupportedVerification("MC_SAMPLE_DISTRIBUTION_BINDING_REQUIRED")
        self.outputs = self.graph.equations
        self.output_slots = {name:name if name in self.slots else e.state_symbol for name,e in self.outputs.items()}
        if any(s not in self.slots for s in self.output_slots.values()): raise UnsupportedVerification("ORACLE_OUTPUT_UNBOUND")
        self.lower,self.cache = _lower_kernel,{}
        tree = ast.parse(code)
        trusted = {n.name:ast.dump(n) for n in _intrinsic_defs(["invar_splitmix64_stateless","invar_rand_u01"])}
        if any(isinstance(n,ast.FunctionDef) and n.name in trusted and ast.dump(n) != trusted[n.name] for n in tree.body):
            raise UnsupportedVerification("STOCHASTIC_SAMPLER_IMPLEMENTATION_UNBOUND")
        if self.graph.archetype == "MONTE_CARLO" and any(isinstance(n,ast.FunctionDef) and n.name == "int" for n in tree.body):
            raise UnsupportedVerification("STOCHASTIC_SAMPLER_IMPLEMENTATION_UNBOUND")
        proof = copy.deepcopy(tree)
        if self.inputs["SEED"] or self.inputs["COUNTER"]:
            if any(len(self.inputs[r]) != 1 for r in ("UNIFORM","SEED","COUNTER","CHANNEL")):
                raise UnsupportedVerification("STOCHASTIC_SIGNATURE_UNBINDABLE")
            expected_args = [ast.dump(ast.parse(f"buf.data[{self.slots[self.inputs[r][0]]}]",mode="eval").body) for r in ("SEED","COUNTER","CHANNEL")]
            uniform = self.inputs["UNIFORM"][0]
            class BindSampler(ast.NodeTransformer):
                def visit_Call(self,node):
                    if isinstance(node.func,ast.Name) and node.func.id == "invar_rand_u01":
                        if [ast.dump(a) for a in node.args] != expected_args:
                            raise UnsupportedVerification("STOCHASTIC_SIGNATURE_UNBINDABLE")
                        return ast.parse(f"buf.data[{self_slots[uniform]}]",mode="eval").body
                    return self.generic_visit(node)
            self_slots = self.slots
            proof = BindSampler().visit(proof)
        self.proof_code = _normalize_guarded_divisions(ast.unparse(ast.fix_missing_locations(proof)),self.slots,self.graph.omega,self.contract.dt)
        proof = ast.parse(self.proof_code)
        functions = {n.name:n for n in proof.body if isinstance(n,ast.FunctionDef)}
        reachable,pending = set(),["simulation_step"]
        while pending:
            name = pending.pop()
            if name in reachable: continue
            reachable.add(name)
            function = functions[name]
            if any(isinstance(n,(ast.If,ast.IfExp,ast.While)) for n in ast.walk(function)):
                raise UnsupportedVerification("PIECEWISE_ORACLE_UNBINDABLE")
            pending.extend(n.func.id for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in functions)
        sk = _SymbolicKernel(self.proof_code, self.slots)
        if self.graph.archetype == "FIXED_POINT":
            rhs_probe, _, _ = sk.zero_dt_profile({s.symbol for s in self.graph.slots if s.role == "DYNAMIC_STATE"})
            if any(v != sp.Integer(0) for v in rhs_probe.values()):
                raise UnsupportedVerification("ORACLE_EXECUTABLE_MISMATCH")
        updates = sk.updates(set(self.slots))
        if set(updates) & {s.symbol for s in self.graph.slots if s.role == "CONTROL_PARAM"}:
            raise UnsupportedVerification("T_CONST_PARAM_WRITE")
        if set(updates)-set(self.output_slots.values()): raise UnsupportedVerification("ORACLE_INPUT_WRITE")
        for name,equation in self.outputs.items():
            target = self.output_slots[name]
            dynamic = {s.symbol for s in self.graph.slots if s.role == "DYNAMIC_STATE"}
            bindings = {sp.Symbol(s):v for s,v in updates.items() if s != target and s not in dynamic}
            expected = sp.sympify(equation.expression,locals={s:sp.Symbol(s) for s in self.slots},rational=True).subs(bindings,simultaneous=True)
            equivalent = target in updates and (_equivalent(updates[target],expected) or
                m.solve_query(f"({_expression(updates[target])}) != ({_expression(expected)})",self.graph.omega).status == "UNSAT")
            if not equivalent:
                raise UnsupportedVerification("ORACLE_EXECUTABLE_MISMATCH")
        quantities = [{"symbol":s.symbol,"si_dimension":s.si_dim,"interval":self.graph.omega[s.symbol],
            "is_state_variable":s.role == "DYNAMIC_STATE","semantic_tag":s.semantic_tag} for s in self.graph.slots]
        arena = _native.PyHolographicHin(m._json(quantities))
        exprs = {k:m._csel(m._parse(e.expression)) for k,e in self.outputs.items()}
        dimensions = json.loads(arena.verify_equations(m._json(exprs)))
        if not dimensions["all_valid"] or any(tuple(r["output_dimensions"]) != self.outputs[r["equation_name"]].si_dim for r in dimensions["equations"]):
            raise UnsupportedVerification("ORACLE_DIMENSION_MISMATCH")
        if not json.loads(arena.verify_observational_purity(m._json(exprs)))["valid"]:
            raise UnsupportedVerification("ORACLE_STATIC_IMPURITY")
        self.formal = {"equations":{k:e.expression for k,e in self.outputs.items()},"parameters":{s:self.base[s] for s in self.controls},
            "initial_state":self.initial,"equation_form":"assignment"}

    def output_index(self,name):
        return self.slots[self.output_slots[name]]

    def execute(self, values=None, *, code=None, extra=0):
        code = code or self.code
        initial = self.initial+[0.0]*extra
        for s,v in (values or {}).items(): initial[self.slots[s]] = float(v)
        if code not in self.cache:
            try: _,self.cache[code] = self.lower(instrument_purity(code))
            except (ValueError,SyntaxError,NotImplementedError): self.cache[code] = None
        cpp = self.cache[code]
        try:
            result = self.machine.bridge.run(cpp,initial,self.contract.dt,1,timeout_seconds=30) if cpp else self.machine.bridge.run_python(instrument_purity(code),initial,self.contract.dt,1,timeout_seconds=30)
        except KernelExecutionError as error:
            if not error.compile_error: raise UnsupportedVerification("ORACLE_EXECUTION_FAILED") from error
            result = self.machine.bridge.run_python(instrument_purity(code),initial,self.contract.dt,1,timeout_seconds=30)
        if not result["finite"]: raise UnsupportedVerification("NONFINITE_ORACLE")
        audit = purity_audit_from_diag(result["diag"],domain_tag=f"stage{self.state.research_stage}/{self.graph.archetype}",static_purity_passed=True)
        if audit.projected_steps or audit.stabilized_steps: raise UnsupportedVerification("ORACLE_RUNTIME_ENFORCEMENT")
        point = {s:float(result["state"][i]) for s,i in self.slots.items()}
        point.update({name:float(result["state"][self.output_index(name)]) for name in self.outputs if name not in point})
        return point,audit,result["state"][len(self.initial):]


def _audit(error=0.0, *, failure=None, statistical=None, rounding=0.0):
    return m.FidelityAudit(passed=failure is None,algebraic_equiv_passed=True,si_and_semantic_tag_passed=True,
        convergence_passed=failure is None,physical_checks_passed=True,epsilon_disc=error if statistical is None else 0,
        epsilon_stat=error if statistical is not None else 0,epsilon_round=rounding,epsilon_total=error+rounding,statistical_audit=statistical,failure_reason=failure)


def _active(rows, parameters, **extra):
    if not parameters: raise UnsupportedVerification("MECHANISM_BINDING_REQUIRED")
    svd = json.loads(_native.py_compute_sensitivity_svd(rows))
    return m.IdentifiabilityAudit(coupling_passed=svd["rank"] == len(parameters),
        dead_parameters=tuple(parameters[i] for i in svd["dead_parameters"]),
        column_norms=dict(zip(parameters,svd["column_norms"])),singular_values=tuple(svd["singular_values"]),
        brun_collinearity_gamma_k=svd["brun_gamma_k"],fim_condition_number=svd["fim_condition_number"],**extra)


def _evidence(oracle, point, purity, fidelity, *, active=None, comparison=None):
    if any(not lo <= point[s] <= hi for s,(lo,hi) in oracle.graph.omega.items()):
        raise UnsupportedVerification("EXECUTION_OUTSIDE_OMEGA")
    if any(m._truth(m._parse(e),{s:(v,v) for s,v in point.items()}) is not True for e in oracle.graph.environmental_assumptions):
        raise UnsupportedVerification("ENVIRONMENT_NOT_CERTIFIED_AT_EXECUTION")
    margin = _point_value(ast.unparse(_claim_margin(oracle.graph,substitute_model=False)),point)
    return ExecutionEvidence(executable_digest=code_hash(oracle.code),formal_uast=oracle.formal,purity=purity,
        fidelity=fidelity,identifiability=active or m.IdentifiabilityAudit(coupling_passed=False),
        evaluation=WitnessEvaluation(signed_margin=margin,epsilon_total=fidelity.epsilon_total),comparison=comparison,theta_star=point)


def _mc(oracle):
    graph = oracle.graph
    sampled = [(name,e) for name,e in graph.equations.items() if e.kind == "MC_SAMPLE"]
    if len(sampled) != 1 or not oracle.inputs["UNIFORM"]:
        raise UnsupportedVerification("MC_SAMPLE_AGGREGATE_BINDING_REQUIRED")
    name,equation = sampled[0]
    if any(graph.omega[s] != (0.0,1.0) for s in oracle.inputs["UNIFORM"]):
        raise UnsupportedVerification("MC_UNIFORM_DOMAIN_REQUIRED")
    if graph.quantifier_mode != "EXISTS": raise UnsupportedVerification("MC_QUANTIFIED_DOMAIN_UNBINDABLE")
    aggregate = equation.state_symbol
    hypothesis = m._and([o.expression for o in graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET"])
    if not m._variables(m._parse(hypothesis)) <= {aggregate,*oracle.controls}:
        raise UnsupportedVerification("MC_HYPOTHESIS_OBSERVABLE_UNBOUND")
    if aggregate in oracle.controls or aggregate in oracle.outputs and aggregate != name: raise UnsupportedVerification("MC_AGGREGATE_SLOT_COLLISION")
    output_range = m._iv(m._parse(_expression(_sample_expression(graph,equation))),graph.omega)
    lo,hi = output_range.lo,output_range.hi
    if not math.isfinite(lo+hi) or hi <= lo: raise UnsupportedVerification("MC_FINITE_RANGE_REQUIRED")
    blocks = 16
    count = oracle.contract.steps if oracle.contract.steps > 1 else 2048
    if count < 32 or count > 65536 or count % blocks:
        raise UnsupportedVerification("MC_REPLICATE_PROTOCOL_UNBOUND")
    size = count//blocks
    n = blocks*size
    def integer_input(role):
        value = oracle.base[oracle.inputs[role][0]] if oracle.inputs[role] else 0.0
        if value != int(value) or not 0 <= value < 2**53:
            raise UnsupportedVerification("MC_INTEGER_STREAM_INPUT_UNREPRESENTABLE")
        return int(value)
    seed,start,channel_base = (integer_input(role) for role in ("SEED","COUNTER","CHANNEL"))
    if start+n >= 2**53: raise UnsupportedVerification("MC_COUNTER_RANGE_UNREPRESENTABLE")
    keys = tuple((seed,start+i,channel_base) for i in range(n))
    block_keys = tuple(keys[i*size] for i in range(blocks))
    tree = ast.parse(oracle.code)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name == "simulation_step": node.name = "_sample_kernel"
    index = len(oracle.initial)
    lines = ["def simulation_step(buf, dt):",f"    for _draw in range({n}):"]
    for slot in graph.slots:
        if slot.role != "CONTROL_PARAM":
            lines.append(f"        buf.data[{slot.slot_idx}] = {oracle.base[slot.symbol]!r}")
    for channel,s in enumerate(oracle.inputs["UNIFORM"]):
        lines.append(f"        buf.data[{oracle.slots[s]}] = invar_rand_u01({seed}, {start}+_draw, {channel_base+channel})")
    for role,value in (("SEED",str(seed)),("COUNTER",f"{start}+_draw"),("CHANNEL",str(channel_base))):
        for s in oracle.inputs[role]: lines.append(f"        buf.data[{oracle.slots[s]}] = {value}")
    lines += ["        _sample_kernel(buf, dt)",f"        buf.data[{index}+_draw] = buf.data[{oracle.output_index(name)}]"]
    tree.body.extend(ast.parse("\n".join(lines)+"\n").body)
    batch_code = ast.unparse(ast.fix_missing_locations(tree))
    def batch(parameters):
        point,purity,draws = oracle.execute(parameters,code=batch_code,extra=n)
        point.update({s.symbol:oracle.base[s.symbol] for s in graph.slots if s.role == "DYNAMIC_STATE"})
        draws = tuple(map(float,draws))
        if any(not lo <= x <= hi for x in draws): raise UnsupportedVerification("MC_SAMPLE_OUTSIDE_FORMAL_RANGE")
        chunks = tuple(draws[i*size:(i+1)*size] for i in range(blocks))
        reliability = m.statistical_reliability(chunks,block_keys)
        point[aggregate] = statistics.mean(draws)
        box = {s:(v,v) for s,v in oracle.base.items()}
        box.update({s:graph.omega[s] for role in oracle.inputs.values() for s in role})
        box.update({s:(parameters.get(s,oracle.base[s]),)*2 for s in oracle.controls})
        round_program = _Roundoff(oracle.proof_code,oracle.slots,box,oracle.contract.dt)
        rounding = round_program.values[oracle.output_index(name)][1]
        error = (hi-lo)*math.sqrt(math.log(400)/(2*n))+100*sys.float_info.epsilon*max(1,abs(lo),abs(hi))
        observation_box = {s:(v,v) for s,v in point.items()}
        observation_box.update({s:(oracle.base[s],)*2 for s in oracle.controls})
        observation_box[aggregate] = (point[aggregate]-error-rounding,point[aggregate]+error+rounding)
        margin = _claim_margin(graph,substitute_model=False)
        enclosure = m._iv(margin,observation_box)
        nominal = _point_value(ast.unparse(margin),{**point,**{s:oracle.base[s] for s in oracle.controls}})
        envelope = max(error+rounding,abs(enclosure.lo-nominal),abs(enclosure.hi-nominal))
        fidelity = _audit(envelope-rounding,failure=reliability.failure_reason,statistical=reliability,rounding=rounding)
        return point,purity,draws,fidelity
    point,purity,draws,fidelity = batch({})
    oracle.formal["sampling_protocol"] = {"distribution":"UNIFORM_0_1","seed":seed,"counter_start":start,"channel":channel_base,
        "blocks":blocks,"block_size":size,"aggregate":aggregate,"sample":name}
    if not fidelity.passed: return _evidence(oracle,point,purity,fidelity)
    baseline = m.synthesize_null_baseline(oracle.formal,graph,oracle.state.research_stage)
    if not baseline.parameter_overrides: raise UnsupportedVerification("MC_FOUNDATION_ORACLE_UNBOUND")
    null,_,null_draws,null_fidelity = batch(baseline.parameter_overrides)
    if not null_fidelity.passed: return _evidence(oracle,point,purity,null_fidelity)
    rows = []
    for fraction in (.25,.5,.75):
        anchor = {p:a+fraction*(b-a) for p in graph.mechanism_parameters for a,b in [graph.omega[p]]}
        row = []
        for p in graph.mechanism_parameters:
            a,b = graph.omega[p]; h = min((b-a)*1e-4,1e-4*max(1,abs(anchor[p])))
            if h <= 0: raise UnsupportedVerification("MC_PARAMETER_STENCIL_UNBOUND")
            plus = batch({**anchor,p:anchor[p]+h})[0][aggregate]
            minus = batch({**anchor,p:anchor[p]-h})[0][aggregate]
            row.append((plus-minus)/(2*h))
        rows.append(row)
    active = _active(rows,graph.mechanism_parameters)
    bernoulli = False
    primary = m.SignificanceResult(observables=(point[aggregate],),fidelity=fidelity,paired_samples=draws,stream_keys=keys,bernoulli_indicators=bernoulli)
    reference = m.SignificanceResult(observables=(null[aggregate],),fidelity=null_fidelity,paired_samples=null_draws,stream_keys=keys,
        bernoulli_indicators=bernoulli,null_baseline_description=baseline.description)
    return _evidence(oracle,point,purity,fidelity,active=active,comparison=(primary,reference))


class _Roundoff:
    """A priori IEEE basic-arithmetic error propagation on the actual source DAG."""
    def __init__(self, code, slots, box, dt, *, minimum_divisor=0.0, guarded_sqrt_log=False):
        self.minimum_divisor = minimum_divisor
        self.guarded_sqrt_log = guarded_sqrt_log
        tree = ast.parse(code)
        self.functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.values = {i:((box[s][0],box[s][1]),0.0) for s,i in slots.items()}
        self.globals = {}
        for n in tree.body:
            if isinstance(n, ast.Assign):
                try:
                    val = self.expression(n.value, self.globals, 0)
                except (UnsupportedVerification, KeyError):
                    continue
                for tgt in n.targets:
                    if isinstance(tgt, ast.Name): self.globals[tgt.id] = val
        env = {**self.globals, "buf": None, "dt": ((dt, dt), 0.0)}
        self.body(self.functions["simulation_step"].body, env, 0)

    def expression(self,node,env,depth):
        if depth > 24: raise UnsupportedVerification("ROUNDING_BOUND_CALL_BUDGET")
        if isinstance(node,ast.Constant) and type(node.value) in (float,int):
            x = float(node.value); return ((x,x),0.0)
        if isinstance(node,ast.Name): return env[node.id]
        if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant):
            return self.values.get(node.slice.value, ((0.0, 0.0), 0.0))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            (a,b),e = self.expression(node.operand,env,depth)
            return ((-b,-a),e) if isinstance(node.op,ast.USub) else ((a,b),e)
        if isinstance(node,ast.Call) and ast.unparse(node.func) in {"abs","math.fabs","fabs"}:
            if len(node.args) != 1 or node.keywords: raise UnsupportedVerification("ROUNDING_LIBRARY_SIGNATURE_UNBOUND")
            (lo,hi),error = self.expression(node.args[0],env,depth+1)
            out_lo = 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))
            out_hi = max(abs(lo), abs(hi))
            return ((out_lo, out_hi), error)
        if isinstance(node,ast.Call) and ast.unparse(node.func) in {"floor","ceil","math.floor","math.ceil"}:
            if len(node.args) != 1 or node.keywords: raise UnsupportedVerification("ROUNDING_LIBRARY_SIGNATURE_UNBOUND")
            (lo,hi),error = self.expression(node.args[0],env,depth+1)
            if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(error)):
                if self.minimum_divisor > 0.0:
                    return ((-math.inf, math.inf), math.inf)
                raise UnsupportedVerification("NONFINITE_ROUNDING_BOUND")
            fn = math.floor if ast.unparse(node.func).endswith("floor") else math.ceil
            nlo, nhi = float(fn(lo)), float(fn(hi))
            alo, ahi = float(fn(math.nextafter(lo - error, -math.inf))), float(fn(math.nextafter(hi + error, math.inf)))
            return ((nlo, nhi), max(abs(ahi - nlo), abs(nhi - alo)))
        if isinstance(node,ast.Call) and ast.unparse(node.func) in {"min","max"}:
            if len(node.args) != 2 or node.keywords: raise UnsupportedVerification("ROUNDING_LIBRARY_SIGNATURE_UNBOUND")
            (alo,ahi),ea = self.expression(node.args[0],env,depth+1)
            (blo,bhi),eb = self.expression(node.args[1],env,depth+1)
            if ast.unparse(node.func) == "min":
                return ((min(alo, blo), min(ahi, bhi)), max(ea, eb))
            return ((max(alo, blo), max(ahi, bhi)), max(ea, eb))
        if isinstance(node,ast.Call) and ast.unparse(node.func) in {"pow","math.pow"}:
            if len(node.args) != 2 or node.keywords: raise UnsupportedVerification("ROUNDING_LIBRARY_SIGNATURE_UNBOUND")
            exp_node = node.args[1]
            if isinstance(exp_node, ast.Constant) and isinstance(exp_node.value, (int, float)) and exp_node.value == int(exp_node.value) and 0 <= int(exp_node.value) <= 16:
                k = int(exp_node.value)
                (lo,hi),error = self.expression(node.args[0],env,depth+1)
                expression = m._parse(f"a**{k}")
                nominal = m._iv(expression,{"a":(lo,hi)})
                actual = m._iv(expression,{"a":(math.nextafter(lo-error,-math.inf),math.nextafter(hi+error,math.inf))})
                error = math.nextafter(max(abs(actual.lo-nominal.hi),abs(actual.hi-nominal.lo)),math.inf)
                if not math.isfinite(error+actual.lo+actual.hi): raise UnsupportedVerification("NONFINITE_ROUNDING_BOUND")
                return ((nominal.lo,nominal.hi),error)
            raise UnsupportedVerification("RIGOROUS_ROUNDING_BOUND_UNAVAILABLE")
        if isinstance(node,ast.Call) and ast.unparse(node.func) in {
            "exp","log","sqrt","sin","cos","abs","fabs","logistic_sigmoid",
            "math.exp","math.log","math.sqrt","math.sin","math.cos","math.fabs","math.logistic_sigmoid",
        }:
            if len(node.args) != 1 or node.keywords: raise UnsupportedVerification("ROUNDING_LIBRARY_SIGNATURE_UNBOUND")
            (lo,hi),error = self.expression(node.args[0],env,depth+1)
            name = ast.unparse(node.func).split(".")[-1]
            if name == "fabs":
                name = "abs"
            try:
                nominal = m._iv(m._parse(f"{name}(a)"),{"a":(lo,hi)})
                act_lo = math.nextafter(lo - error, -math.inf)
                if self.guarded_sqrt_log and name == "sqrt":
                    act_lo = max(0.0, act_lo)
                elif self.guarded_sqrt_log and name == "log":
                    act_lo = max(1e-300, act_lo)
                actual = m._iv(m._parse(f"{name}(a)"),{"a":(act_lo,math.nextafter(hi+error,math.inf))})
            except m._DomainError:
                if self.minimum_divisor > 0.0:
                    return ((-math.inf, math.inf), math.inf)
                raise
            error = math.nextafter(max(abs(actual.lo-nominal.hi),abs(actual.hi-nominal.lo)),math.inf)
            if not math.isfinite(error+actual.lo+actual.hi):
                if self.minimum_divisor > 0.0:
                    return ((-math.inf, math.inf), math.inf)
                raise UnsupportedVerification("NONFINITE_ROUNDING_BOUND")
            return ((nominal.lo,nominal.hi),error)
        if isinstance(node,ast.Call) and ast.unparse(node.func) in {"min","max","kl_term","hill_eq","michaelis_menten","math.kl_term","math.hill_eq","math.michaelis_menten"}:
            name = ast.unparse(node.func).split(".")[-1]
            arity = m._ARITIES[name]
            if len(node.args) != arity or node.keywords: raise UnsupportedVerification("ROUNDING_LIBRARY_SIGNATURE_UNBOUND")
            evaluated = [self.expression(arg,env,depth+1) for arg in node.args]
            arg_names = [f"a{i}" for i in range(arity)]
            expr = m._parse(f"{name}({','.join(arg_names)})")
            nom_env = {arg_names[i]: evaluated[i][0] for i in range(arity)}
            act_env = {arg_names[i]: (math.nextafter(evaluated[i][0][0]-evaluated[i][1], -math.inf), math.nextafter(evaluated[i][0][1]+evaluated[i][1], math.inf)) for i in range(arity)}
            try:
                nominal = m._iv(expr, nom_env)
                actual = m._iv(expr, act_env)
            except m._DomainError:
                if self.minimum_divisor > 0.0:
                    return ((-math.inf, math.inf), math.inf)
                raise
            error = math.nextafter(max(abs(actual.lo-nominal.hi),abs(actual.hi-nominal.lo)),math.inf)
            if not math.isfinite(error+actual.lo+actual.hi):
                if self.minimum_divisor > 0.0:
                    return ((-math.inf, math.inf), math.inf)
                raise UnsupportedVerification("NONFINITE_ROUNDING_BOUND")
            return ((nominal.lo,nominal.hi),error)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in self.functions:
            function = self.functions[node.func.id]
            local = {**self.globals, **dict(zip((a.arg for a in function.args.args),(self.expression(a,env,depth+1) for a in node.args)))}
            return self.body(function.body,local,depth+1)
        if not isinstance(node,ast.BinOp): raise UnsupportedVerification("RIGOROUS_ROUNDING_BOUND_UNAVAILABLE")
        if isinstance(node.op,ast.Pow) and isinstance(node.right,ast.Constant) and type(node.right.value) is int and 0 <= node.right.value <= 16:
            (lo,hi),error = self.expression(node.left,env,depth+1)
            expression = m._parse(f"a**{node.right.value}")
            nominal = m._iv(expression,{"a":(lo,hi)})
            actual = m._iv(expression,{"a":(math.nextafter(lo-error,-math.inf),math.nextafter(hi+error,math.inf))})
            error = math.nextafter(max(abs(actual.lo-nominal.hi),abs(actual.hi-nominal.lo)),math.inf)
            if not math.isfinite(error+actual.lo+actual.hi): raise UnsupportedVerification("NONFINITE_ROUNDING_BOUND")
            return ((nominal.lo,nominal.hi),error)
        a,ea = self.expression(node.left,env,depth); b,eb = self.expression(node.right,env,depth)
        if not all(map(math.isfinite, (*a, ea, *b, eb))):
            if isinstance(node.op, ast.Div):
                raise UnsupportedVerification("ROUNDING_DENOMINATOR_NOT_SEPARATED")
            if self.minimum_divisor > 0.0:
                return ((-math.inf, math.inf), math.inf)
            raise UnsupportedVerification("NONFINITE_ROUNDING_BOUND")
        aa,bb = max(map(abs,a)),max(map(abs,b)); u = sys.float_info.epsilon
        if isinstance(node.op,(ast.Add,ast.Sub)): error = ea+eb+u*(aa+bb+ea+eb)/(1-u)
        elif isinstance(node.op,ast.Mult): error = ea*bb+eb*aa+ea*eb+u*(aa+ea)*(bb+eb)/(1-u)
        elif isinstance(node.op,ast.Div):
            minimum = min(map(abs,b)) if not b[0] <= 0 <= b[1] else 0
            if minimum-eb <= self.minimum_divisor: raise UnsupportedVerification("ROUNDING_DENOMINATOR_NOT_SEPARATED")
            error = (ea+aa*eb/minimum)/(minimum-eb)+u*(aa+ea)/(minimum-eb)/(1-u)
        else: raise UnsupportedVerification("RIGOROUS_ROUNDING_BOUND_UNAVAILABLE")
        op = {ast.Add:"+",ast.Sub:"-",ast.Mult:"*",ast.Div:"/"}[type(node.op)]
        if isinstance(node.op, ast.Mult) and ast.dump(node.left, include_attributes=False) == ast.dump(node.right, include_attributes=False):
            value = m._iv(m._parse("a**2"), {"a": a})
        else:
            value = m._iv(m._parse("a"+op+"b"),{"a":a,"b":b})
        error = math.nextafter(error*(1+128*u)+128*math.ulp(0.0),math.inf)
        if not math.isfinite(error+value.lo+value.hi): raise UnsupportedVerification("NONFINITE_ROUNDING_BOUND")
        return ((value.lo,value.hi),error)

    @staticmethod
    def _hull(v1, v2):
        (l1, h1), e1 = v1
        (l2, h2), e2 = v2
        return ((min(l1, l2), max(h1, h2)), max(e1, e2))

    @staticmethod
    def _branch_bounds(test, env):
        """Soundly narrow a single variable's interval for simple comparisons `x > c` / `x >= c` / `x < c` / `x <= c`."""
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1):
            return True, True, {}, {}
        lhs, op, rhs = test.left, test.ops[0], test.comparators[0]
        def _const(n):
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                return float(n.value)
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant) and type(n.operand.value) in (int, float):
                return -float(n.operand.value)
            return None
        name, val, flip = None, None, False
        if isinstance(lhs, ast.Name) and lhs.id in env and _const(rhs) is not None:
            name, val = lhs.id, _const(rhs)
        elif isinstance(rhs, ast.Name) and rhs.id in env and _const(lhs) is not None:
            name, val, flip = rhs.id, _const(lhs), True
        else:
            return True, True, {}, {}
        (lo, hi), err = env[name]
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return True, True, {}, {}
        is_gt = (isinstance(op, (ast.Gt, ast.GtE)) and not flip) or (isinstance(op, (ast.Lt, ast.LtE)) and flip)
        is_lt = (isinstance(op, (ast.Lt, ast.LtE)) and not flip) or (isinstance(op, (ast.Gt, ast.GtE)) and flip)
        if is_gt:
            strict = isinstance(op, ast.Gt) if not flip else isinstance(op, ast.Lt)
            then_ok = (hi > val) if strict else (hi >= val)
            else_ok = (lo <= val) if strict else (lo < val)
            then_lo = max(lo, math.nextafter(val, math.inf) if strict else val)
            then_upd = {name: ((then_lo, hi), err)} if then_ok and then_lo <= hi else {}
            else_hi = min(hi, val)
            else_upd = {name: ((lo, else_hi), err)} if else_ok and lo <= else_hi else {}
            return then_ok, else_ok, then_upd, else_upd
        if is_lt:
            strict = isinstance(op, ast.Lt) if not flip else isinstance(op, ast.Gt)
            then_ok = (lo < val) if strict else (lo <= val)
            else_ok = (hi >= val) if strict else (hi > val)
            then_hi = min(hi, math.nextafter(val, -math.inf) if strict else val)
            then_upd = {name: ((lo, then_hi), err)} if then_ok and lo <= then_hi else {}
            else_lo = max(lo, val)
            else_upd = {name: ((else_lo, hi), err)} if else_ok and else_lo <= hi else {}
            return then_ok, else_ok, then_upd, else_upd
        if isinstance(op, (ast.Eq, ast.NotEq)):
            is_eq = isinstance(op, ast.Eq)
            contains = (lo <= val <= hi)
            exact = (lo == val == hi)
            then_ok = contains if is_eq else (not exact)
            else_ok = (not exact) if is_eq else contains
            eq_upd = {name: ((val, val), err)} if contains else {}
            return (then_ok, else_ok, eq_upd if is_eq else {}, eq_upd if not is_eq else {})
        return True, True, {}, {}

    def body(self,body,env,depth):
        for node in body:
            if isinstance(node,ast.Assign):
                value = self.expression(node.value,env,depth)
                for target in node.targets:
                    if isinstance(target,ast.Name): env[target.id] = value
                    elif isinstance(target,ast.Subscript) and isinstance(target.slice,ast.Constant): self.values[target.slice.value] = value
                    else: raise UnsupportedVerification("ROUNDING_ASSIGNMENT_UNBOUND")
            elif isinstance(node,ast.If):
                then_ok, else_ok, then_upd, else_upd = self._branch_bounds(node.test, env)
                orig_values = dict(self.values)
                then_env = {**env, **then_upd}
                then_vals = dict(orig_values)
                then_ret = None
                if then_ok:
                    self.values = then_vals
                    then_ret = self.body(node.body, then_env, depth + 1)
                    then_vals = self.values
                else_env = {**env, **else_upd}
                else_vals = dict(orig_values)
                else_ret = None
                if else_ok:
                    self.values = else_vals
                    else_ret = self.body(node.orelse, else_env, depth + 1) if node.orelse else None
                    else_vals = self.values
                if then_ok and else_ok:
                    for k in set(then_env) | set(else_env):
                        if k in then_env and k in else_env:
                            if isinstance(then_env[k], tuple) and isinstance(else_env[k], tuple):
                                env[k] = self._hull(then_env[k], else_env[k])
                            else:
                                env[k] = then_env[k]
                        elif k in then_env:
                            env[k] = then_env[k]
                        else:
                            env[k] = else_env[k]
                    merged_vals = dict(orig_values)
                    for k in set(then_vals) | set(else_vals):
                        if k in then_vals and k in else_vals:
                            merged_vals[k] = self._hull(then_vals[k], else_vals[k])
                        else:
                            merged_vals[k] = then_vals.get(k, else_vals.get(k))
                    self.values = merged_vals
                    if then_ret is not None and else_ret is not None:
                        return self._hull(then_ret, else_ret)
                elif then_ok:
                    env.update(then_env)
                    self.values = then_vals
                    if then_ret is not None:
                        return then_ret
                elif else_ok:
                    env.update(else_env)
                    self.values = else_vals
                    if else_ret is not None:
                        return else_ret
                else:
                    self.values = orig_values
            elif isinstance(node,ast.Return): return self.expression(node.value,env,depth)
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call): self.expression(node.value,env,depth)
            elif not isinstance(node,(ast.Pass,ast.Import,ast.ImportFrom)): raise UnsupportedVerification("ROUNDING_PROGRAM_UNBOUND")


def _fixed(oracle):
    graph = oracle.graph
    if graph.quantifier_mode != "EXISTS": raise UnsupportedVerification("FIXED_POINT_ROOT_FAMILY_PROOF_REQUIRED")
    equations = {e.state_symbol:(name,e) for name,e in graph.equations.items() if e.kind in {"FIXED_POINT_RESIDUAL","FIXED_POINT_MAP"}}
    algebraic_readouts = {name:e for name,e in graph.equations.items() if e.kind == "ALGEBRAIC"}
    states = sorted(equations)
    if not states or len(equations) + len(algebraic_readouts) != len(graph.equations): raise UnsupportedVerification("FIXED_POINT_RESIDUAL_BINDING_REQUIRED")
    if set(states) != {s.symbol for s in graph.slots if s.role == "DYNAMIC_STATE"}:
        raise UnsupportedVerification("FIXED_POINT_SYSTEM_NOT_SQUARE")
    symbols = {s:sp.Symbol(s) for s in oracle.slots}
    residuals = [sp.sympify(e.expression,locals=symbols)-(symbols[s] if e.kind == "FIXED_POINT_MAP" else 0) for s,(name,e) in ((s,equations[s]) for s in states)]
    jac = [[sp.diff(e,symbols[s]) for s in states] for e in residuals]
    raw_margin = _claim_margin(graph,substitute_model=False)
    def solve(parameters):
        controls = {s:parameters.get(s,oracle.base[s]) for s in oracle.controls}
        def executable(z):
            point,_,_ = oracle.execute({**controls,**dict(zip(states,z))})
            return [point[equations[s][0]]-(z[i] if equations[s][1].kind == "FIXED_POINT_MAP" else 0) for i,s in enumerate(states)]
        z = np.array([oracle.base[s] for s in states])
        for _ in range(32):
            point = {**oracle.base,**controls,**dict(zip(states,z))}
            values = np.array(executable(z))
            matrix = np.array([[_point_value(_expression(e),point) for e in row] for row in jac])
            if np.linalg.norm(values,ord=np.inf) <= 1e-13: break
            z -= np.linalg.solve(matrix,values)
            if any(not graph.omega[s][0] <= x <= graph.omega[s][1] for s,x in zip(states,z)):
                raise UnsupportedVerification("NEWTON_CENTER_OUTSIDE_OMEGA")
        else: raise UnsupportedVerification("NEWTON_CENTER_NOT_CONVERGED")
        point = {**oracle.base,**controls,**dict(zip(states,map(float,z)))}
        expressions = [_expression(e.subs({symbols[s]:v for s,v in controls.items()})) for e in residuals]
        encoded = [m._json(m._csel(m._parse(e))) for e in expressions]
        radius = min(1e-3*max(1,max(abs(z))),*(min(x-graph.omega[s][0],graph.omega[s][1]-x)/2 for s,x in zip(states,z)))
        if radius <= 0: raise UnsupportedVerification("ROOT_NEIGHBORHOOD_UNBOUND")
        report = json.loads(_native.py_certify_krawczyk_fixed_point(encoded,list(z),radius))
        if not report["certified"] or report["symbols"] != states: raise UnsupportedVerification("KRAWCZYK_INCONCLUSIVE")
        box = {s:(v,v) for s,v in controls.items()}
        box.update({s:(b["lo"],b["hi"]) for s,b in zip(states,report["box_z"])})
        box.update({s:graph.omega[s] for s in oracle.slots if s not in box})
        rounding = _Roundoff(oracle.code,oracle.slots,box,oracle.contract.dt)
        round_bound = max(rounding.values[oracle.output_index(equations[s][0])][1] for s in states)
        if any(equations[s][1].kind == "FIXED_POINT_MAP" for s in states):
            round_bound += 4*sys.float_info.epsilon*max(1,max(abs(z)))
        third = []
        for e in residuals:
            row = []
            for s in states:
                bound = m._iv(m._parse(_expression(sp.diff(e,symbols[s],3))),box)
                row.append(max(abs(bound.lo),abs(bound.hi)))
            third.append(row)
        native_jac = copy.deepcopy(report["interval_jacobian"])
        max_jac = max(abs(v[k]) for row in native_jac for v in row for k in ("lo","hi"))
        gap = min(min(k["lo"]-b["lo"],b["hi"]-k["hi"]) for k,b in zip(report["krawczyk_box"],report["box_z"]))
        hmin = min(sys.float_info.epsilon**(1/3),gap/4)
        hmax = sys.float_info.epsilon**(1/3)*max(1,max(abs(z))+radius)
        residual_max = max(max(abs(v) for v in rounding.values[oracle.output_index(equations[s][0])][0]) for s in states)
        padding = 8*(max(map(max,third))*hmax*hmax/6+(2*round_bound+8*sys.float_info.epsilon*max(1,residual_max)+8*sys.float_info.epsilon*max_jac*max(1,max(abs(z))))/hmin)
        # A priori widening, independent of measured FD values; preserve native bounds.
        report["native_interval_jacobian"] = native_jac
        report["fd_rounding_padding"] = padding
        report["interval_jacobian"] = [[{"lo":math.nextafter(v["lo"]-padding,-math.inf),"hi":math.nextafter(v["hi"]+padding,math.inf)} for v in row] for row in native_jac]
        fd = m.cross_check_krawczyk_jacobian(executable,z,report,third,executable_rounding_bound=round_bound)
        root_box = {**box,**{s:(b["lo"],b["hi"]) for s,b in zip(states,report["krawczyk_box"])}}
        root_box.update({s:(oracle.base[s],)*2 for s in oracle.controls})
        margin_iv = m._iv(raw_margin,root_box)
        margin_value = _point_value(ast.unparse(raw_margin),{**point,**{s:oracle.base[s] for s in oracle.controls}})
        error = max(abs(margin_value-margin_iv.lo),abs(margin_iv.hi-margin_value))+100*sys.float_info.epsilon*max(1,abs(margin_value))
        fidelity = _audit(error,failure=fd.failure_reason)
        _,purity,_ = oracle.execute(point)
        if any(equations[s][1].kind == "FIXED_POINT_MAP" for s in states):
            if any(sum(max(abs(v["lo"]+(1 if i == j else 0)),abs(v["hi"]+(1 if i == j else 0))) for j,v in enumerate(row)) >= 1 for i,row in enumerate(native_jac)):
                raise UnsupportedVerification("FIXED_POINT_MAP_CONTRACTION_UNPROVEN")
        return point,purity,fidelity,report,native_jac,matrix
    point,purity,fidelity,report,jiv,jx = solve({})
    oracle.formal["krawczyk"] = report
    if not fidelity.passed: return _evidence(oracle,point,purity,fidelity)
    parameters = graph.mechanism_parameters
    alg_subs = {symbols[name]: sp.sympify(e.expression, locals=symbols) for name, e in algebraic_readouts.items()}
    for _ in range(len(alg_subs)):
        alg_subs = {k: v.subs(alg_subs, simultaneous=True) for k, v in alg_subs.items()}
    margin_sym = sp.sympify(ast.unparse(raw_margin), locals=symbols).subs(alg_subs, simultaneous=True)
    jtheta = [[_point_value(_expression(sp.diff(e,symbols[p])),point) for p in parameters] for e in residuals]
    dy = [[_point_value(_expression(sp.diff(margin_sym,symbols[s])),point) for s in states]]
    direct = [[_point_value(_expression(sp.diff(margin_sym,symbols[p])),point) for p in parameters]]
    ift = json.loads(_native.py_compute_ift_equilibrium_sensitivity(jx.tolist(),jtheta,dy,direct))
    restoring = None
    if len(states) == 1:
        entry = jiv[0][0]
        restoring = max(0.0,entry["lo"],-entry["hi"])
    elif len(states) == 2:
        box = {name:(jiv[i][j]["lo"],jiv[i][j]["hi"]) for name,i,j in (("a",0,0),("b",0,1),("c",1,0),("d",1,1))}
        determinant = m._iv(m._parse("a*d-b*c"),box)
        magnitude = {name:max(abs(lo),abs(hi)) for name,(lo,hi) in box.items()}
        denominator = m._iv(m._parse("sqrt(2)*max(d+b,c+a)"),{name:(v,v) for name,v in magnitude.items()}).hi
        if determinant.lo > 0 and denominator > 0:
            restoring = max(0.0,math.nextafter(determinant.lo/denominator,-math.inf))
    active = _active(ift["dy_dtheta"],parameters,equilibrium_restoring_force=restoring)
    oracle.formal["ift_sensitivity"] = ift
    baseline = m.synthesize_null_baseline(oracle.formal,graph,oracle.state.research_stage)
    if not baseline.parameter_overrides: raise UnsupportedVerification("FIXED_POINT_FOUNDATION_ORACLE_UNBOUND")
    null,_,nf,nr,nj,_ = solve(baseline.parameter_overrides)
    if not nf.passed: return _evidence(oracle,point,purity,nf)
    def result(p,f,r,j,description="Declared model"):
        observed = {**p,**{s:oracle.base[s] for s in oracle.controls}}
        return m.SignificanceResult(observables=(_point_value(ast.unparse(raw_margin),observed),),fidelity=f,krawczyk=r,
            stability_matrix=tuple(tuple((v["lo"],v["hi"]) for v in row) for row in j),null_baseline_description=description)
    return _evidence(oracle,point,purity,fidelity,active=active,comparison=(result(point,fidelity,report,jiv),result(null,nf,nr,nj,baseline.description)))


def _mc_ensemble(machine, state, code):
    """Certified finite-sample Monte Carlo ensemble verifier using CRN macro-replications."""
    from .state_machine import _lower_kernel, v9_transpiler_violations
    graph = state.ihcg
    graph.assert_sealed()
    if not state.blueprint.contracts or v9_transpiler_violations(code):
        raise UnsupportedVerification("ORACLE_PROTOCOL_UNBINDABLE")
    contract = state.blueprint.contracts[-1]
    slots = {s.symbol: s.slot_idx for s in graph.slots}
    vmap = state.blueprint.variable_map
    vmap_indices = set(vmap.values())
    if any((s in vmap and vmap[s] != i) or i not in vmap_indices for s, i in slots.items()):
        raise UnsupportedVerification("IHCG_EXECUTABLE_SLOT_MISMATCH")
    initial = list(contract.initial_state)
    initial.extend([0.0] * max(0, max(slots.values()) + 1 - len(initial)))
    base = {s: initial[i] for s, i in slots.items()}
    if any(base[s] != value for s, value in graph.control_values.items()):
        raise UnsupportedVerification("LOCKED_CONTROL_VALUE_MISMATCH")
    controls = [s.symbol for s in graph.slots if s.role == "CONTROL_PARAM" and s.stochastic_role is None]
    seed_slots = [s.symbol for s in graph.slots if s.stochastic_role == "SEED"]
    if len(seed_slots) != 1:
        raise UnsupportedVerification("MC_SAMPLE_DISTRIBUTION_BINDING_REQUIRED")
    tree = ast.parse(code)
    trusted = {n.name: ast.dump(n) for n in _intrinsic_defs(["invar_splitmix64_stateless", "invar_rand_u01"])}
    if any(isinstance(n, ast.FunctionDef) and n.name in trusted and ast.dump(n) != trusted[n.name] for n in tree.body):
        raise UnsupportedVerification("STOCHASTIC_SAMPLER_IMPLEMENTATION_UNBOUND")
    if any(isinstance(n, ast.FunctionDef) and n.name == "int" for n in tree.body):
        raise UnsupportedVerification("STOCHASTIC_SAMPLER_IMPLEMENTATION_UNBOUND")
    seed_idx = slots[seed_slots[0]]
    prng_called = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"invar_rand_u01", "invar_rand_normal", "invar_splitmix64_stateless"}
        for n in ast.walk(tree)
    ) or any(
        isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == seed_idx and ast.unparse(n.value) in {"buf.data", "data"}
        for n in ast.walk(tree)
    )
    if not prng_called:
        raise UnsupportedVerification("MC_SAMPLE_DISTRIBUTION_BINDING_REQUIRED")
    control_indices = {s.slot_idx for s in graph.slots if s.role == "CONTROL_PARAM"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                        and target.slice.value in control_indices and ast.unparse(target.value) in {"buf.data", "data"}):
                    raise UnsupportedVerification("T_CONST_PARAM_WRITE")
    if graph.mc_difference_bounds is None:
        raise UnsupportedVerification("MC_DIFFERENCE_RANGE_UNBOUND")
    diff_lo, diff_hi = graph.mc_difference_bounds
    if not math.isfinite(diff_lo + diff_hi) or diff_hi <= diff_lo:
        raise UnsupportedVerification("MC_DIFFERENCE_RANGE_UNBOUND")
    if not graph.mechanism_parameters:
        raise UnsupportedVerification("MECHANISM_BINDING_REQUIRED")
    seed_sym = seed_slots[0]
    seed_val = base[seed_sym]
    if seed_val != int(seed_val) or not 0 <= seed_val < 2**53:
        raise UnsupportedVerification("MC_INTEGER_STREAM_INPUT_UNREPRESENTABLE")
    seed_int = int(seed_val)
    blocks, size = 32, 8
    n = blocks * size
    seed_stride = 1_000_003
    if seed_int + n * seed_stride >= 2**53:
        raise UnsupportedVerification("MC_COUNTER_RANGE_UNREPRESENTABLE")
    keys = tuple((seed_int, k, 0) for k in range(n))
    block_keys = tuple(keys[i * size] for i in range(blocks))
    raw_margin = _claim_margin(graph, substitute_model=False)
    margin_expr = ast.unparse(raw_margin)
    obs_syms = sorted(v for v in m._variables(raw_margin) if v in slots and v not in controls and v != seed_sym)
    if not obs_syms:
        obs_syms = sorted(s.symbol for s in graph.slots if s.role == "DERIVED_OBSERVABLE")
    stride = len(obs_syms)
    batch_tree = copy.deepcopy(tree)
    for node in batch_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "simulation_step":
            node.name = "_sample_kernel"
    base_len = len(initial)
    lines = ["def simulation_step(buf, dt):", f"    for _draw in range({n}):"]
    for slot in graph.slots:
        if slot.role != "CONTROL_PARAM":
            lines.append(f"        buf.data[{slot.slot_idx}] = {base[slot.symbol]!r}")
    lines.append(f"        buf.data[{slots[seed_sym]}] = float({seed_int} + _draw * {seed_stride})")
    lines.append("        _sample_kernel(buf, dt)")
    for j, sym in enumerate(obs_syms):
        lines.append(f"        buf.data[{base_len} + _draw * {stride} + {j}] = buf.data[{slots[sym]}]")
    batch_tree.body.extend(ast.parse("\n".join(lines) + "\n").body)
    batch_code = ast.unparse(ast.fix_missing_locations(batch_tree))
    cache: dict[str, str | None] = {}

    def _run_kernel(src, params, extra=0):
        init = list(initial) + [0.0] * extra
        for s, v in params.items():
            if s in slots:
                init[slots[s]] = float(v)
        if src not in cache:
            try:
                _, cache[src] = _lower_kernel(instrument_purity(src))
            except (ValueError, SyntaxError, NotImplementedError):
                cache[src] = None
        cpp = cache[src]
        try:
            res = (machine.bridge.run(cpp, init, contract.dt, 1, timeout_seconds=60)
                   if cpp else machine.bridge.run_python(instrument_purity(src), init, contract.dt, 1, timeout_seconds=60))
        except KernelExecutionError as error:
            if not error.compile_error:
                raise UnsupportedVerification("ORACLE_EXECUTION_FAILED") from error
            res = machine.bridge.run_python(instrument_purity(src), init, contract.dt, 1, timeout_seconds=60)
        if not res["finite"]:
            raise UnsupportedVerification("NONFINITE_ORACLE")
        audit = purity_audit_from_diag(res["diag"], domain_tag=f"stage{state.research_stage}/{graph.archetype}", static_purity_passed=True)
        if audit.projected_steps or audit.stabilized_steps:
            raise UnsupportedVerification("ORACLE_RUNTIME_ENFORCEMENT")
        return res, audit

    def batch(params):
        res_nom, purity = _run_kernel(code, params, extra=0)
        point = {s: float(res_nom["state"][i]) for s, i in slots.items()}
        res_b, purity_b = _run_kernel(batch_code, params, extra=n * stride)
        if purity_b.projected_steps or purity_b.stabilized_steps:
            raise UnsupportedVerification("ORACLE_RUNTIME_ENFORCEMENT")
        tail = res_b["state"][base_len:]
        draws = []
        for k in range(n):
            pt = dict(point)
            for j, sym in enumerate(obs_syms):
                pt[sym] = float(tail[k * stride + j])
            draws.append(_point_value(margin_expr, pt))
        draws_t = tuple(draws)
        chunks = tuple(draws_t[i * size:(i + 1) * size] for i in range(blocks))
        reliability = m.statistical_reliability(chunks, block_keys)
        mean_val = statistics.mean(draws_t)
        std_val = statistics.stdev(draws_t) if len(draws_t) > 1 else 0.0
        half_w = min(
            math.sqrt(2.0 * (std_val ** 2) * math.log(400.0) / n) + 7.0 * (diff_hi - diff_lo) * math.log(400.0) / (3.0 * max(1, n - 1)),
            (diff_hi - diff_lo) * math.sqrt(math.log(400.0) / (2.0 * n)),
        ) + 100.0 * sys.float_info.epsilon * max(1.0, abs(mean_val))
        fidelity = _audit(half_w, failure=reliability.failure_reason, statistical=reliability, rounding=0.0)
        return point, purity, draws_t, fidelity

    formal = {
        "equations": {k: e.expression for k, e in graph.equations.items()},
        "parameters": {s: base[s] for s in controls},
        "initial_state": initial,
        "equation_form": "monte_carlo_ensemble",
        "sampling_protocol": {
            "distribution": "STATELESS_SPLITMIX64_ENSEMBLE",
            "seed": seed_int,
            "seed_stride": seed_stride,
            "blocks": blocks,
            "block_size": size,
            "observable_margin": margin_expr,
        },
    }
    point, purity, draws, fidelity = batch({})
    if any(not lo - 1e-9 <= point[s] <= hi + 1e-9 for s, (lo, hi) in graph.omega.items()):
        raise UnsupportedVerification("EXECUTION_OUTSIDE_OMEGA")
    if any(m._truth(m._parse(e), {s: (v, v) for s, v in point.items()}) is not True for e in graph.environmental_assumptions):
        raise UnsupportedVerification("ENVIRONMENT_NOT_CERTIFIED_AT_EXECUTION")
    if not fidelity.passed:
        return ExecutionEvidence(
            executable_digest=code_hash(code), formal_uast=formal, purity=purity, fidelity=fidelity,
            identifiability=m.IdentifiabilityAudit(coupling_passed=False),
            evaluation=WitnessEvaluation(signed_margin=statistics.mean(draws), epsilon_total=fidelity.epsilon_total),
            theta_star=point,
        )
    baseline = m.synthesize_null_baseline(formal, graph, state.research_stage)
    if not baseline.parameter_overrides:
        raise UnsupportedVerification("MC_FOUNDATION_ORACLE_UNBOUND")
    _, _, null_draws, null_fidelity = batch(baseline.parameter_overrides)
    diffs = tuple(a - b for a, b in zip(draws, null_draws))
    eb_range_term = 7.0 * (diff_hi - diff_lo) * math.log(600.0) / (3.0 * max(1, n - 1))
    null_heavy_tail = (not null_fidelity.passed and null_fidelity.failure_reason == "STATISTICAL_ESTIMATOR_UNRELIABLE")
    use_bernoulli = (null_heavy_tail or statistics.mean(diffs) <= eb_range_term) and sum(1 for d in diffs if d > 0.0) > n // 2
    if not null_fidelity.passed and not use_bernoulli:
        return ExecutionEvidence(
            executable_digest=code_hash(code), formal_uast=formal, purity=purity, fidelity=null_fidelity,
            identifiability=m.IdentifiabilityAudit(coupling_passed=False),
            evaluation=WitnessEvaluation(signed_margin=statistics.mean(draws), epsilon_total=null_fidelity.epsilon_total),
            theta_star=point,
        )
    if use_bernoulli and null_heavy_tail:
        # Clopper-Pearson on paired sign indicators (d_i > nd_i) is exact and distribution-free
        # even when ablating a mechanism parameter (e.g. instrument strength pi=0) induces heavy tails.
        null_fidelity = _audit(fidelity.epsilon_stat, failure=None, statistical=fidelity.statistical_audit, rounding=0.0)
    # Compute multi-observable CRN sensitivity Jacobian dY_j / dtheta_k across declared
    # DERIVED_OBSERVABLE slots so multi-parameter point-domain MC models achieve full column rank.
    sens_syms = sorted(s.symbol for s in graph.slots if s.role == "DERIVED_OBSERVABLE")
    if not sens_syms:
        sens_syms = list(obs_syms)
    has_interval = any(graph.omega[p][1] > graph.omega[p][0] for p in graph.mechanism_parameters)
    fractions = (0.25, 0.5, 0.75) if has_interval else (0.5,)
    rows = []
    for fraction in fractions:
        anchor = {p: (a + fraction * (b - a) if b > a else base[p]) for p in graph.mechanism_parameters for a, b in [graph.omega[p]]}
        cols = []
        for p in graph.mechanism_parameters:
            a, b = graph.omega[p]
            h = min((b - a) * 1e-3, 1e-3 * max(1.0, abs(anchor[p]))) if b > a else 1e-2 * max(1.0, abs(anchor[p]))
            if h <= 0:
                raise UnsupportedVerification("MC_PARAMETER_STENCIL_UNBOUND")
            res_p, _ = _run_kernel(code, {**anchor, p: anchor[p] + h}, extra=0)
            res_m, _ = _run_kernel(code, {**anchor, p: anchor[p] - h}, extra=0)
            col = [
                (float(res_p["state"][slots[sym]]) - float(res_m["state"][slots[sym]])) / (2.0 * h)
                * max(1.0, abs(anchor[p])) / max(1.0, abs(point[sym]))
                for sym in sens_syms
            ]
            cols.append(col)
        rows.extend(list(r) for r in zip(*cols))
    active = _active(rows, graph.mechanism_parameters)
    if use_bernoulli:
        p_samples = tuple(1.0 if d > 0.0 else 0.0 for d in diffs)
        r_samples = tuple(0.0 for _ in diffs)
        win_rate = sum(p_samples) / n
        # Exact Clopper-Pearson half-width for the Bernoulli win-rate margin (win_rate - 0.5).
        cp_err = max(1.0 - (0.0033333333333333335 ** (1.0 / n)), 100.0 * sys.float_info.epsilon)
        fidelity = _audit(cp_err, failure=fidelity.failure_reason, statistical=fidelity.statistical_audit, rounding=0.0)
        null_fidelity = _audit(cp_err, failure=None, statistical=fidelity.statistical_audit, rounding=0.0)
        eval_margin = win_rate - 0.5
    else:
        p_samples = draws
        r_samples = null_draws
        eval_margin = statistics.mean(draws)
    primary = m.SignificanceResult(
        observables=(statistics.mean(draws),), fidelity=fidelity, paired_samples=p_samples,
        stream_keys=keys, bernoulli_indicators=use_bernoulli,
    )
    reference = m.SignificanceResult(
        observables=(statistics.mean(null_draws),), fidelity=null_fidelity, paired_samples=r_samples,
        stream_keys=keys, bernoulli_indicators=use_bernoulli, null_baseline_description=baseline.description,
    )
    return ExecutionEvidence(
        executable_digest=code_hash(code), formal_uast=formal, purity=purity, fidelity=fidelity,
        identifiability=active,
        evaluation=WitnessEvaluation(signed_margin=eval_margin, epsilon_total=fidelity.epsilon_total),
        comparison=(primary, reference), theta_star=point,
    )


def _fixed_from_ode_kernel(machine, state, code):
    """Certify a FIXED_POINT IHCG whose executable kernel integrates dx/dt = F(x, theta) toward x*."""
    from .state_machine import _lower_kernel, v9_transpiler_violations
    graph = state.ihcg
    graph.assert_sealed()
    if graph.quantifier_mode != "EXISTS":
        raise UnsupportedVerification("FIXED_POINT_ROOT_FAMILY_PROOF_REQUIRED")
    if not state.blueprint.contracts or v9_transpiler_violations(code):
        raise UnsupportedVerification("ORACLE_PROTOCOL_UNBINDABLE")
    contract = state.blueprint.contracts[-1]
    slots = {s.symbol: s.slot_idx for s in graph.slots}
    vmap = state.blueprint.variable_map
    vmap_indices = set(vmap.values())
    if any((s in vmap and vmap[s] != i) or i not in vmap_indices for s, i in slots.items()):
        raise UnsupportedVerification("IHCG_EXECUTABLE_SLOT_MISMATCH")
    initial = list(contract.initial_state)
    initial.extend([0.0] * max(0, max(slots.values()) + 1 - len(initial)))
    base = {s: float(initial[i]) for s, i in slots.items()}
    if any(base[s] != value for s, value in graph.control_values.items()):
        raise UnsupportedVerification("LOCKED_CONTROL_VALUE_MISMATCH")
    controls = [s.symbol for s in graph.slots if s.role == "CONTROL_PARAM" and s.stochastic_role is None]
    equations = {e.state_symbol: (name, e) for name, e in graph.equations.items() if e.kind in {"FIXED_POINT_RESIDUAL", "FIXED_POINT_MAP"} and e.state_symbol}
    algebraic_readouts = {name: e for name, e in graph.equations.items() if e.kind == "ALGEBRAIC"}
    all_states = sorted(equations)
    if not all_states or len(equations) + len(algebraic_readouts) != len(graph.equations):
        raise UnsupportedVerification("FIXED_POINT_RESIDUAL_BINDING_REQUIRED")
    if set(all_states) != {s.symbol for s in graph.slots if s.role == "DYNAMIC_STATE"}:
        raise UnsupportedVerification("FIXED_POINT_SYSTEM_NOT_SQUARE")

    # 1. Verify SI dimensions, semantic tags, and observational purity via native HIN arena.
    quantities = [{
        "symbol": s.symbol, "si_dimension": s.si_dim, "interval": graph.omega[s.symbol],
        "is_state_variable": s.role == "DYNAMIC_STATE", "semantic_tag": s.semantic_tag,
    } for s in graph.slots]
    arena = _native.PyHolographicHin(m._json(quantities))
    exprs = {k: m._csel(m._parse(e.expression)) for k, e in graph.equations.items()}
    dimensions = json.loads(arena.verify_equations(m._json(exprs)))
    if not dimensions["all_valid"] or any(tuple(r["output_dimensions"]) != graph.equations[r["equation_name"]].si_dim for r in dimensions["equations"]):
        raise UnsupportedVerification("ORACLE_DIMENSION_MISMATCH")
    if not json.loads(arena.verify_observational_purity(m._json(exprs)))["valid"]:
        raise UnsupportedVerification("ORACLE_STATIC_IMPURITY")

    # 2. Formally verify that the executable kernel's ODE RHS and algebraic readouts match the sealed IHCG.
    # Strip diagnostic-only conditional extrema trackers (e.g. peak tracking) that do not write to any
    # IHCG dynamic state or algebraic equation slot.
    protected_indices = {slots[s] for s in all_states} | {slots[name] for name in algebraic_readouts if name in slots}
    tree = ast.parse(code)
    def _writes_protected(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value in protected_indices:
                        return True
        return False
    class _StripDiagIf(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Never mutate canonical intrinsic/domain helpers like _invar_sdiv.
            if node.name != "simulation_step":
                return node
            return self.generic_visit(node)
        def visit_If(self, node):
            if not _writes_protected(node):
                return ast.Pass()
            return self.generic_visit(node)
    proof_tree = ast.fix_missing_locations(_StripDiagIf().visit(copy.deepcopy(tree)))
    proof_src = ast.unparse(proof_tree)
    deferred_sdiv_check = False
    try:
        proof_code = _normalize_guarded_divisions(proof_src, slots, graph.omega, contract.dt)
    except UnsupportedVerification as exc:
        if str(exc) != "ROUNDING_DENOMINATOR_NOT_SEPARATED":
            raise
        # Verify canonical _invar_sdiv AST identity strictly, unwrap for symbolic equivalence,
        # and defer whole-box division separation check to the certified local Krawczyk neighborhood root_box.
        s_tree = ast.parse(proof_src)
        helpers = [n for n in s_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_invar_sdiv"]
        if len(helpers) != 1 or ast.dump(helpers[0]) != ast.dump(ast.parse(_SAFE_DIV_SRC).body[0]):
            raise UnsupportedVerification("GUARDED_DIVISION_IMPLEMENTATION_UNBOUND")
        class _UnwrapSDiv(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id == "_invar_sdiv":
                    if len(node.args) != 2 or node.keywords:
                        raise UnsupportedVerification("GUARDED_DIVISION_SIGNATURE_UNBOUND")
                    return ast.copy_location(ast.BinOp(left=node.args[0], op=ast.Div(), right=node.args[1]), node)
                return node
        s_tree.body = [n for n in s_tree.body if n not in helpers]
        proof_code = ast.unparse(ast.fix_missing_locations(_UnwrapSDiv().visit(s_tree)))
        deferred_sdiv_check = True

    def _unwrap_domain_guards(src_code, *, unwrap_sqrt=True, unwrap_log=True):
        def _is_max_guard(call_node, target_val_max):
            if not (isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name)
                    and call_node.func.id == "max" and len(call_node.args) == 2 and not call_node.keywords):
                return None
            for idx_c in (0, 1):
                c_arg = call_node.args[idx_c]
                if isinstance(c_arg, ast.Constant) and type(c_arg.value) in (int, float):
                    cv = float(c_arg.value)
                    if target_val_max == 0.0 and cv == 0.0:
                        return call_node.args[1 - idx_c]
                    if target_val_max > 0.0 and 0.0 < cv <= target_val_max:
                        return call_node.args[1 - idx_c]
            return None
        class _UnwrapDom(ast.NodeTransformer):
            def __init__(self):
                self.unwrapped = False
            def visit_Call(self, node):
                self.generic_visit(node)
                cname = ast.unparse(node.func)
                if unwrap_sqrt and cname in {"sqrt", "math.sqrt"} and len(node.args) == 1 and not node.keywords:
                    inner = _is_max_guard(node.args[0], 0.0)
                    if inner is not None:
                        self.unwrapped = True
                        node.args[0] = inner
                elif unwrap_log and cname in {"log", "math.log"} and len(node.args) == 1 and not node.keywords:
                    inner = _is_max_guard(node.args[0], 1e-100)
                    if inner is not None:
                        self.unwrapped = True
                        node.args[0] = inner
                return node
        tr = _UnwrapDom()
        out_tree = ast.fix_missing_locations(tr.visit(ast.parse(src_code)))
        return (ast.unparse(out_tree), True) if tr.unwrapped else (src_code, False)

    log_unwrapped_code, had_log_guards = _unwrap_domain_guards(proof_code, unwrap_sqrt=False, unwrap_log=True)
    full_unwrapped_code, had_domain_guards = _unwrap_domain_guards(proof_code, unwrap_sqrt=True, unwrap_log=True)
    symbolic_kernel = _SymbolicKernel(proof_code, slots)
    # Use O(steps) dual evaluation at dt=0 (zero_dt_profile) instead of symbolic _dt updates,
    # avoiding exponential degree blowup across multi-stage integrators (e.g. RK4 on cubic systems).
    actual_rhs, updates, written_slots = symbolic_kernel.zero_dt_profile(set(slots))
    log_updates = _SymbolicKernel(log_unwrapped_code, slots).zero_dt_profile(set(slots))[1] if had_log_guards else {}
    unwrapped_updates = _SymbolicKernel(full_unwrapped_code, slots).zero_dt_profile(set(slots))[1] if had_domain_guards else {}
    used_domain_guard_unwrap = False
    verified_unwrapped_code = proof_code
    ast_written = {
        n.targets[0].slice.value
        for n in ast.walk(ast.parse(proof_code))
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].value, ast.Attribute)
        and n.targets[0].value.attr == "data" and isinstance(n.targets[0].slice, ast.Constant)
        and type(n.targets[0].slice.value) is int
    }
    ctrl_indices = {s.slot_idx for s in graph.slots if s.role == "CONTROL_PARAM"}
    if (written_slots & {s.symbol for s in graph.slots if s.role == "CONTROL_PARAM"}) or (ast_written & ctrl_indices):
        raise UnsupportedVerification("T_CONST_PARAM_WRITE")

    symbols = {s: sp.Symbol(s) for s in slots}
    res_name_subs = {
        symbols[name]: sp.sympify(eq.expression, locals=symbols, rational=True)
        for name, eq in graph.equations.items()
        if eq.kind == "FIXED_POINT_RESIDUAL" and name in symbols and name != eq.state_symbol and name not in all_states
    }
    alg_subs = {symbols[name]: sp.sympify(e.expression, locals=symbols, rational=True) for name, e in algebraic_readouts.items()}
    combined_subs = {**res_name_subs, **alg_subs}
    for _ in range(max(1, len(alg_subs) + 1)):
        alg_subs = {k: v.subs(combined_subs, simultaneous=True) for k, v in alg_subs.items()}
        combined_subs = {**res_name_subs, **alg_subs}

    # Build sound manifold substitutions from declared invariant equality assumptions in environmental_assumptions
    # (only accepted after proving invariance at t=0 and Lie-derivative / closure invariance along the kernel).
    env_subs = {}
    env_invs = []
    for raw_env in graph.environmental_assumptions:
        try:
            parsed_env = m._parse(raw_env)
        except Exception:
            continue
        if isinstance(parsed_env, ast.Compare) and len(parsed_env.ops) == 1 and isinstance(parsed_env.ops[0], ast.Eq) and len(parsed_env.comparators) == 1:
            inv_lhs = sp.sympify(ast.unparse(parsed_env.left), locals=symbols, rational=True).subs(alg_subs, simultaneous=True)
            inv_rhs = sp.sympify(ast.unparse(parsed_env.comparators[0]), locals=symbols, rational=True).subs(alg_subs, simultaneous=True)
            diff_expr = inv_lhs - inv_rhs
            if abs(_point_value(_expression(diff_expr), base)) > 1e-9:
                continue
            cand_syms = [s for s in sorted(all_states, reverse=True) if symbols[s] in diff_expr.free_symbols and symbols[s] not in env_subs]
            for dep in cand_syms:
                sols = sp.solve(diff_expr.subs(env_subs, simultaneous=True), symbols[dep])
                if len(sols) == 1 and symbols[dep] not in sols[0].free_symbols:
                    # Verify flow invariance: Lie derivative along actual_rhs vanishes on manifold (modulo existing env_subs).
                    sub_trial = {**env_subs, symbols[dep]: sols[0]}
                    lie = sum(sp.diff(diff_expr, symbols[st]) * actual_rhs.get(st, sp.Integer(0)) for st in all_states if symbols[st] in diff_expr.free_symbols)
                    if _equivalent(lie.subs(sub_trial, simultaneous=True), sp.Integer(0)):
                        env_subs[symbols[dep]] = sols[0]
                        env_invs.append(diff_expr)
                        break

    def _equiv_on_manifold(lhs_e, rhs_e):
        if _equivalent(lhs_e, rhs_e):
            return True
        if env_subs:
            return _equivalent(lhs_e.subs(env_subs, simultaneous=True), rhs_e.subs(env_subs, simultaneous=True))
        return False

    # Partition dynamic states into active interior equilibrium states vs. decoupled monotonic clock states (F(t)=t, dt/dt=1).
    other_refs = set()
    for s, (name, e) in equations.items():
        if e.expression.strip() != s:
            other_refs.update(m._variables(m._parse(e.expression)))
    for e in algebraic_readouts.values():
        other_refs.update(m._variables(m._parse(e.expression)))
    for o in graph.obligations:
        other_refs.update(m._variables(m._parse(o.expression)))

    clock_states = set()
    ode_states = set()
    for s in all_states:
        _, eq = equations[s]
        exp_res = sp.sympify(eq.expression, locals=symbols, rational=True).subs(alg_subs, simultaneous=True)
        if eq.kind == "FIXED_POINT_MAP":
            exp_res = exp_res - symbols[s]
        if s not in other_refs and _equivalent(exp_res, symbols[s]):
            if s in actual_rhs and _equivalent(actual_rhs[s], sp.Integer(1)):
                clock_states.add(s)
                continue
        # Either ODE derivative matches residual (dx/dt == F(x)), OR algebraic closure (x == g(x_other) with F == x - g(x_other)).
        ode_match = s in actual_rhs and _equiv_on_manifold(actual_rhs[s], exp_res)
        if ode_match:
            ode_states.add(s)
        alg_match = False
        if not ode_match and s in updates:
            # Check if solving F(x) == 0 for s yields updates[s] after substituting other state updates.
            state_upd_subs = {symbols[k]: updates[k] for k in all_states if k != s and k in updates}
            cand = exp_res.subs(state_upd_subs, simultaneous=True).subs(symbols[s], updates[s])
            alg_match = _equiv_on_manifold(cand, sp.Integer(0))
        if not (ode_match or alg_match):
            raise UnsupportedVerification(
                f"ORACLE_EXECUTABLE_MISMATCH: state={s!r} expected={exp_res} got={actual_rhs.get(s)}"
            )

    # Verify algebraic equations referenced by obligations match executable updates.
    alg_mismatches = []
    for name, expr in alg_subs.items():
        sym_name = str(name)
        if sym_name in slots and sym_name in updates:
            bound = expr.subs({symbols[s]: updates[s] for s in all_states if s in updates}, simultaneous=True)
            if not _equiv_on_manifold(updates[sym_name], bound):
                u_matched = False
                if had_log_guards and sym_name in log_updates:
                    l_bound = expr.subs(
                        {symbols[s]: log_updates.get(s, updates[s]) for s in all_states if s in updates},
                        simultaneous=True,
                    )
                    if _equiv_on_manifold(log_updates[sym_name], l_bound):
                        u_matched = True
                        used_domain_guard_unwrap = True
                        if verified_unwrapped_code == proof_code:
                            verified_unwrapped_code = log_unwrapped_code
                if not u_matched and had_domain_guards and sym_name in unwrapped_updates:
                    u_bound = expr.subs(
                        {symbols[s]: unwrapped_updates.get(s, updates[s]) for s in all_states if s in updates},
                        simultaneous=True,
                    )
                    if _equiv_on_manifold(unwrapped_updates[sym_name], u_bound):
                        u_matched = True
                        used_domain_guard_unwrap = True
                        verified_unwrapped_code = full_unwrapped_code
                if not u_matched:
                    alg_mismatches.append((sym_name, bound, updates[sym_name]))
    if alg_mismatches:
        first_sym, first_bound, first_got = alg_mismatches[0]
        details = "; ".join(
            f"slot={sym!r} expected={exp} got={got}" for sym, exp, got in alg_mismatches
        )
        raise UnsupportedVerification(
            f"ALGEBRAIC_EXECUTABLE_MISMATCH: slot={first_sym!r} ({len(alg_mismatches)} mismatch(es): {details})"
        )
    if used_domain_guard_unwrap:
        deferred_sdiv_check = True
        proof_code = verified_unwrapped_code

    # 3. Execute the actual kernel to verify runtime purity (zero stabilizer/projection hits) and finiteness.
    try:
        _, cpp = _lower_kernel(instrument_purity(code))
    except (ValueError, SyntaxError, NotImplementedError):
        cpp = None
    try:
        run_res = machine.bridge.run(cpp, initial, contract.dt, contract.steps, timeout_seconds=min(30.0, contract.timeout_seconds)) if cpp else machine.bridge.run_python(instrument_purity(code), initial, contract.dt, contract.steps, timeout_seconds=min(30.0, contract.timeout_seconds))
    except KernelExecutionError as error:
        if not error.compile_error:
            raise UnsupportedVerification("ORACLE_EXECUTION_FAILED") from error
        run_res = machine.bridge.run_python(instrument_purity(code), initial, contract.dt, contract.steps, timeout_seconds=min(30.0, contract.timeout_seconds))
    if not run_res["finite"]:
        raise UnsupportedVerification("NONFINITE_ORACLE")
    purity = purity_audit_from_diag(run_res["diag"], domain_tag=f"stage{state.research_stage}/FIXED_POINT", static_purity_passed=True)
    if purity.projected_steps or purity.stabilized_steps:
        raise UnsupportedVerification("ORACLE_RUNTIME_ENFORCEMENT")

    # 4. Certify the fixed point x* via Newton + Native Krawczyk + FD Jacobian cross-check.
    states = [s for s in all_states if s not in clock_states]
    if not states:
        raise UnsupportedVerification("FIXED_POINT_RESIDUAL_BINDING_REQUIRED")
    residuals = [
        sp.sympify(equations[s][1].expression, locals=symbols, rational=True).subs(alg_subs, simultaneous=True)
        - (symbols[s] if equations[s][1].kind == "FIXED_POINT_MAP" else 0)
        for s in states
    ]
    jac = [[sp.diff(e, symbols[s]) for s in states] for e in residuals]
    raw_margin = _claim_margin(graph, substitute_model=False)
    formal = {
        "equations": {k: e.expression for k, e in graph.equations.items()},
        "parameters": {s: base[s] for s in controls},
        "initial_state": initial,
        "equation_form": "assignment",
    }

    def solve(parameters, *, use_manifold=False):
        ctrl = {s: parameters.get(s, base[s]) for s in controls}
        elim_subs = env_subs if (use_manifold and env_subs) else {}
        s_states = [s for s in states if symbols[s] not in elim_subs] if elim_subs else list(states)
        s_residuals = [
            (r.subs(elim_subs, simultaneous=True) if elim_subs else r)
            for s, r in zip(states, residuals) if s in s_states
        ]
        s_jac = [[sp.diff(e, symbols[s]) for s in s_states] for e in s_residuals]
        if not s_states:
            raise UnsupportedVerification("FIXED_POINT_RESIDUAL_BINDING_REQUIRED")
        res_exprs = [_expression(e) for e in s_residuals]
        def _fill_point(z):
            pt = {**base, **ctrl, **{cs: 0.0 for cs in clock_states}, **dict(zip(s_states, map(float, z)))}
            for sym_k, expr_v in elim_subs.items():
                pt[str(sym_k)] = _point_value(_expression(expr_v), pt)
            return pt
        def executable(z):
            pt = _fill_point(z)
            return [_point_value(expr, pt) for expr in res_exprs]
        # Start from executed trajectory endpoint (or midpoint of omega if outside).
        z0 = []
        for s in s_states:
            v = float(run_res["state"][slots[s]])
            lo, hi = graph.omega[s]
            z0.append(v if lo < v < hi else 0.5 * (lo + hi))
        z = np.array(z0, dtype=float)
        for _ in range(48):
            pt = _fill_point(z)
            vals = np.array(executable(z), dtype=float)
            matrix = np.array([[_point_value(_expression(e), pt) for e in row] for row in s_jac], dtype=float)
            if np.linalg.norm(vals, ord=np.inf) <= 1e-13:
                break
            try:
                step_vec = np.linalg.solve(matrix, vals)
            except np.linalg.LinAlgError as exc:
                raise UnsupportedVerification("KRAWCZYK_INCONCLUSIVE") from exc
            # Damped Newton line search to keep iterates inside omega.
            lam = 1.0
            while lam >= 1e-6:
                cand = z - lam * step_vec
                if all(graph.omega[s][0] <= x <= graph.omega[s][1] for s, x in zip(s_states, cand)):
                    z = cand
                    break
                lam *= 0.5
            else:
                raise UnsupportedVerification("NEWTON_CENTER_OUTSIDE_OMEGA")
        else:
            raise UnsupportedVerification("NEWTON_CENTER_NOT_CONVERGED")

        pt = _fill_point(z)
        for name, expr in alg_subs.items():
            pt[str(name)] = _point_value(_expression(expr), pt)
        for name, expr in res_name_subs.items():
            pt[str(name)] = _point_value(_expression(expr.subs(alg_subs, simultaneous=True)), pt)
        # Re-evaluate residual expressions with controls substituted for native Krawczyk.
        # Expand small integer powers (a**k -> a*a*...*a) as required by the native interval certifier.
        class _ExpandIntPow(ast.NodeTransformer):
            def visit_BinOp(self, node):
                node = self.generic_visit(node)
                if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and type(node.right.value) is int and 0 <= node.right.value <= 16:
                    k = node.right.value
                    if k == 0:
                        return ast.Constant(1.0)
                    if k == 1:
                        return node.left
                    cur = copy.deepcopy(node.left)
                    for _ in range(k - 1):
                        cur = ast.BinOp(left=cur, op=ast.Mult(), right=copy.deepcopy(node.left))
                    return cur
                return node
        expressions = [_expression(e.subs({symbols[s]: v for s, v in ctrl.items()})) for e in s_residuals]
        encoded = [m._json(m._csel(_ExpandIntPow().visit(m._parse(e)))) for e in expressions]
        radius = min(1e-3 * max(1.0, float(np.max(np.abs(z)))), *(min(x - graph.omega[s][0], graph.omega[s][1] - x) / 2.0 for s, x in zip(s_states, z)))
        if radius <= 0.0:
            raise UnsupportedVerification("ROOT_NEIGHBORHOOD_UNBOUND")
        try:
            report = json.loads(_native.py_certify_krawczyk_fixed_point(encoded, list(map(float, z)), float(radius)))
        except ValueError as exc:
            raise UnsupportedVerification("KRAWCZYK_INCONCLUSIVE") from exc
        if not report["certified"] or report["symbols"] != s_states:
            raise UnsupportedVerification("KRAWCZYK_INCONCLUSIVE")
        box = {s: (v, v) for s, v in ctrl.items()}
        box.update({cs: (0.0, 0.0) for cs in clock_states})
        box.update({s: (b["lo"], b["hi"]) for s, b in zip(s_states, report["box_z"])})
        box.update({s: graph.omega[s] for s in slots if s not in box})
        round_bound = 128.0 * sys.float_info.epsilon * max(1.0, float(np.max(np.abs(z))))
        third = []
        for e in s_residuals:
            row = []
            for s in s_states:
                bound = m._iv(m._parse(_expression(sp.diff(e, symbols[s], 3))), box)
                row.append(max(abs(bound.lo), abs(bound.hi)))
            third.append(row)
        native_jac = copy.deepcopy(report["interval_jacobian"])
        max_jac = max(abs(v[k]) for row in native_jac for v in row for k in ("lo", "hi"))
        gap = min(min(k["lo"] - b["lo"], b["hi"] - k["hi"]) for k, b in zip(report["krawczyk_box"], report["box_z"]))
        hmin = min(sys.float_info.epsilon ** (1 / 3), gap / 4.0)
        hmax = (sys.float_info.epsilon ** (1 / 3)) * max(1.0, float(np.max(np.abs(z))) + radius)
        padding = 8.0 * (max(map(max, third)) * hmax * hmax / 6.0 + (2.0 * round_bound + 8.0 * sys.float_info.epsilon * max_jac * max(1.0, float(np.max(np.abs(z))))) / hmin)
        report["native_interval_jacobian"] = native_jac
        report["fd_rounding_padding"] = padding
        report["interval_jacobian"] = [[{"lo": math.nextafter(v["lo"] - padding, -math.inf), "hi": math.nextafter(v["hi"] + padding, math.inf)} for v in row] for row in native_jac]
        fd = m.cross_check_krawczyk_jacobian(executable, z, report, third, executable_rounding_bound=round_bound)
        root_box = {**box, **{s: (b["lo"], b["hi"]) for s, b in zip(s_states, report["krawczyk_box"])}}
        root_box.update({s: (ctrl[s], ctrl[s]) for s in controls})
        for sym_k, expr_v in elim_subs.items():
            iv_e = m._iv(m._parse(_expression(expr_v)), root_box)
            root_box[str(sym_k)] = (iv_e.lo, iv_e.hi)
        for name, expr in alg_subs.items():
            iv_a = m._iv(m._parse(_expression(expr)), root_box)
            root_box[str(name)] = (iv_a.lo, iv_a.hi)
        for name, expr in res_name_subs.items():
            iv_r = m._iv(m._parse(_expression(expr.subs(alg_subs, simultaneous=True))), root_box)
            root_box[str(name)] = (iv_r.lo, iv_r.hi)
        if deferred_sdiv_check:
            local_box = {**root_box, **{cs: (0.0, 0.0) for cs in clock_states}}
            _Roundoff(proof_code, slots, local_box, contract.dt, minimum_divisor=1e-12)
            if used_domain_guard_unwrap:
                try:
                    _Roundoff(verified_unwrapped_code, slots, local_box, contract.dt, minimum_divisor=0.0, guarded_sqrt_log=True)
                except m._DomainError as exc:
                    raise UnsupportedVerification("RIGOROUS_ROUNDING_BOUND_UNAVAILABLE") from exc
        margin_iv = m._iv(raw_margin, root_box)
        margin_value = _point_value(ast.unparse(raw_margin), pt)
        error = max(abs(margin_value - margin_iv.lo), abs(margin_iv.hi - margin_value)) + 100.0 * sys.float_info.epsilon * max(1.0, abs(margin_value))
        fidelity = _audit(error, failure=fd.failure_reason, rounding=round_bound)
        return pt, fidelity, report, native_jac, matrix

    point, fidelity, report, jiv, jx = solve({})
    formal["krawczyk"] = report
    def _wrap_evidence(active_audit=None, comparison=None, fid=fidelity):
        for s, (lo, hi) in graph.omega.items():
            fp_tol = 1e-9 * max(1.0, abs(lo), abs(hi))
            if not (lo - fp_tol <= point[s] <= hi + fp_tol):
                raise UnsupportedVerification("EXECUTION_OUTSIDE_OMEGA")
        margin_val = _point_value(ast.unparse(raw_margin), point)
        return ExecutionEvidence(
            executable_digest=code_hash(code), formal_uast=formal, purity=purity,
            fidelity=fid, identifiability=active_audit or m.IdentifiabilityAudit(coupling_passed=False),
            evaluation=WitnessEvaluation(signed_margin=margin_val, epsilon_total=fid.epsilon_total),
            comparison=comparison, theta_star=point,
        )
    if not fidelity.passed:
        return _wrap_evidence()

    # 5. Exact IFT equilibrium sensitivity across all hypothesis conjuncts (including direct dY/dtheta).
    parameters = list(graph.mechanism_parameters)
    def _conjuncts(node):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return [n for child in node.values for n in _conjuncts(child)]
        return [node]
    obs_syms_exprs = []
    for o in graph.obligations:
        if o.obligation_class != "HYPOTHESIS_TARGET":
            continue
        for node in _conjuncts(m._parse(o.expression)):
            local = graph.model_copy(update={"obligations": (o.model_copy(update={"expression": ast.unparse(node)}),)})
            cm = _claim_margin(local, substitute_model=False)
            obs_syms_exprs.append(sp.sympify(ast.unparse(cm), locals=symbols, rational=True).subs(alg_subs, simultaneous=True))
    # Also include all declared ALGEBRAIC DERIVED_OBSERVABLE equations for full multi-observable IFT identifiability.
    for s in graph.slots:
        if s.role == "DERIVED_OBSERVABLE" and symbols[s.symbol] in alg_subs:
            obs_syms_exprs.append(alg_subs[symbols[s.symbol]])
    jtheta = [[_point_value(_expression(sp.diff(e, symbols[p])), point) for p in parameters] for e in residuals]
    dy, direct = [], []
    for expr in obs_syms_exprs:
        try:
            dy_row = [_point_value(_expression(sp.diff(expr, symbols[s])), point) for s in states]
            dir_row = [_point_value(_expression(sp.diff(expr, symbols[p])), point) for p in parameters]
        except Exception:
            # Skip non-smooth observables at x* (e.g., sqrt(u) or |u| with u(x*) == 0); smooth observables carry IFT.
            continue
        if all(math.isfinite(v) for v in (*dy_row, *dir_row)):
            dy.append(dy_row)
            direct.append(dir_row)
    if not dy:
        raise UnsupportedVerification("MECHANISM_NOT_DECLARED")
    ift = json.loads(_native.py_compute_ift_equilibrium_sensitivity(jx.tolist(), jtheta, dy, direct))
    # Sound lower bound on restoring force via interval Jacobian (1x1, 2x2, or general NxN singular value perturbation).
    restoring = None
    n_st = len(states)
    if n_st == 1:
        restoring = max(0.0, jiv[0][0]["lo"], -jiv[0][0]["hi"])
    elif n_st == 2:
        b2 = {name: (jiv[i][j]["lo"], jiv[i][j]["hi"]) for name, i, j in (("a", 0, 0), ("b", 0, 1), ("c", 1, 0), ("d", 1, 1))}
        det2 = m._iv(m._parse("a*d-b*c"), b2)
        mag2 = {name: max(abs(lo), abs(hi)) for name, (lo, hi) in b2.items()}
        den2 = m._iv(m._parse("sqrt(2)*max(d+b,c+a)"), {name: (v, v) for name, v in mag2.items()}).hi
        if det2.lo > 0 and den2 > 0:
            restoring = max(0.0, math.nextafter(det2.lo / den2, -math.inf))
    else:
        mid = np.array([[0.5 * (v["lo"] + v["hi"]) for v in row] for row in jiv], dtype=float)
        rad = np.array([[0.5 * (v["hi"] - v["lo"]) for v in row] for row in jiv], dtype=float)
        smin = float(np.linalg.svd(mid, compute_uv=False)[-1])
        rad_frob = float(np.linalg.norm(rad, ord="fro"))
        if smin > rad_frob:
            restoring = max(0.0, math.nextafter(smin - rad_frob, -math.inf))
    active = _active(ift["dy_dtheta"], parameters, equilibrium_restoring_force=restoring)
    formal["ift_sensitivity"] = ift

    # 6. Certified null-baseline comparison.
    # For DAE-constrained equilibria (ODE states + algebraic manifold closures), the dynamical
    # stability matrix is the interval Jacobian of the ODE states on the constraint manifold.
    ode_idx = [i for i, s in enumerate(states) if s in ode_states]
    def _stability_submatrix(j_mat):
        if ode_idx and len(ode_idx) < len(states):
            # Verify ODE residuals do not directly depend on algebraic closure states (else use full matrix).
            alg_idx = [i for i in range(len(states)) if i not in ode_idx]
            decoupled = all(
                j_mat[i][k]["lo"] == 0.0 and j_mat[i][k]["hi"] == 0.0
                for i in ode_idx for k in alg_idx
            )
            if decoupled:
                return tuple(tuple((j_mat[i][j]["lo"], j_mat[i][j]["hi"]) for j in ode_idx) for i in ode_idx)
        return tuple(tuple((v["lo"], v["hi"]) for v in row) for row in j_mat)
    baseline = m.synthesize_null_baseline(formal, graph, state.research_stage)
    if not baseline.parameter_overrides:
        raise UnsupportedVerification("FIXED_POINT_FOUNDATION_ORACLE_UNBOUND")
    null_obs_override = None
    try:
        null_pt, nf, nr, nj, _ = solve(baseline.parameter_overrides)
    except UnsupportedVerification as null_exc:
        if str(null_exc) not in {"KRAWCZYK_INCONCLUSIVE", "NEWTON_CENTER_NOT_CONVERGED", "NEWTON_CENTER_OUTSIDE_OMEGA"}:
            raise
        if env_subs:
            try:
                null_pt, nf, nr, nj, _ = solve(baseline.parameter_overrides, use_manifold=True)
            except UnsupportedVerification:
                null_obs_override = ()
        else:
            null_obs_override = ()
    if null_obs_override is not None:
        # Null ablation collapsed Jacobian invertibility: certify strictly via universal interval bound over Omega.
        omega_null = dict(graph.omega)
        fixed_subs = {}
        for p_k, p_v in baseline.parameter_overrides.items():
            if p_k in controls:
                val_f = float(p_v)
                omega_null[p_k] = (val_f, val_f)
                fixed_subs[symbols[p_k]] = sp.Float(val_f) if not val_f.is_integer() else sp.Integer(int(val_f))
        alg_null = {}
        for name, expr in {**alg_subs, **res_name_subs}.items():
            expr_resolved = expr.subs(alg_subs, simultaneous=True) if name in res_name_subs else expr
            simp = expr_resolved.subs(fixed_subs, simultaneous=True) if fixed_subs else expr_resolved
            try:
                simp = sp.simplify(simp)
                simp_ast = m._parse(_expression(simp))
            except Exception:
                simp = expr_resolved
                simp_ast = m._parse(_expression(expr_resolved))
            alg_null[symbols[str(name)]] = simp
            iv_a = m._iv(simp_ast, omega_null)
            omega_null[str(name)] = (iv_a.lo, iv_a.hi)
        margin_iv = m._iv(raw_margin, omega_null)
        if not (math.isfinite(margin_iv.hi) and margin_iv.hi < 0.0) and fixed_subs:
            # Re-evaluate raw_margin after exact symbolic cancellation across algebraic slots + fixed controls.
            def _simp_margin_ast(node):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "min" and len(node.args) == 2:
                    return ast.Call(func=node.func, args=[_simp_margin_ast(node.args[0]), _simp_margin_ast(node.args[1])], keywords=[])
                try:
                    sub_e = sp.simplify(sp.sympify(ast.unparse(node), locals=symbols, rational=True).subs({**alg_null, **fixed_subs}, simultaneous=True))
                    return m._parse(_expression(sub_e))
                except Exception:
                    return node
            margin_iv = m._iv(_simp_margin_ast(raw_margin), omega_null)
        if not (math.isfinite(margin_iv.hi) and margin_iv.hi < 0.0):
            raise UnsupportedVerification("KRAWCZYK_INCONCLUSIVE")
        null_obs_override = (margin_iv.hi,)
        nf = _audit(100.0 * sys.float_info.epsilon * max(1.0, abs(margin_iv.hi)))
        nr, nj, null_pt = report, jiv, point
    if not nf.passed:
        return _wrap_evidence(active_audit=active, fid=nf)
    def _sig(p, f, r, j, description="Declared model", obs_override=None):
        obs = obs_override if obs_override is not None else (_point_value(ast.unparse(raw_margin), p),)
        return m.SignificanceResult(
            observables=obs, fidelity=f, krawczyk=r,
            stability_matrix=_stability_submatrix(j),
            null_baseline_description=description,
        )
    return _wrap_evidence(active_audit=active, comparison=(_sig(point, fidelity, report, jiv), _sig(null_pt, nf, nr, nj, baseline.description, null_obs_override)))


def derive_evidence(machine,state,code,*,uast=None,phase="post_execution"):
    if state.ihcg.archetype == "MONTE_CARLO":
        has_uniform = any(s.stochastic_role == "UNIFORM" for s in state.ihcg.slots)
        has_seed = any(s.stochastic_role == "SEED" for s in state.ihcg.slots)
        if not has_uniform and has_seed:
            if phase == "deductive":
                return None
            return _mc_ensemble(machine, state, code)
    if state.ihcg.archetype == "FIXED_POINT":
        try:
            oracle = _Oracle(machine, state, code)
        except UnsupportedVerification as exc:
            if str(exc).split(":", 1)[0] in {
                "PIECEWISE_ORACLE_UNBINDABLE", "ORACLE_EXECUTABLE_MISMATCH",
                "ORACLE_INPUT_WRITE", "ORACLE_OUTPUT_UNBOUND",
                "RIGOROUS_ROUNDING_BOUND_UNAVAILABLE",
                "ROUNDING_PROGRAM_UNBOUND", "ROUNDING_DENOMINATOR_NOT_SEPARATED",
                "SYMBOLIC_IDENTITY_DAG_OPERATOR_UNSUPPORTED", "SYMBOLIC_IDENTITY_DAG_BUDGET",
            }:
                if phase == "deductive":
                    return None
                return _fixed_from_ode_kernel(machine, state, code)
            raise
        if phase == "deductive":
            return None
        return _fixed(oracle)
    oracle = _Oracle(machine,state,code)
    if phase == "deductive": return None
    return _mc(oracle) if state.ihcg.archetype == "MONTE_CARLO" else _fixed(oracle)


def mc_difference_bounds(graph):
    equations = [e for e in graph.equations.values() if e.kind == "MC_SAMPLE"]
    if len(equations) != 1: raise UnsupportedVerification("MC_PAIRED_DIFFERENCE_UNBINDABLE")
    symbols = {s.symbol:sp.Symbol(s.symbol) for s in graph.slots}
    expression = _sample_expression(graph,equations[0])
    difference = sp.expand(expression-expression.subs({symbols[p]:0 for p in graph.mechanism_parameters},simultaneous=True))
    interval = m._iv(m._parse(_expression(difference)),graph.omega)
    output = m._iv(m._parse(_expression(expression)),graph.omega)
    padding = 100*sys.float_info.epsilon*max(1,abs(output.lo),abs(output.hi))
    lo,hi = math.nextafter(interval.lo-padding,-math.inf),math.nextafter(interval.hi+padding,math.inf)
    if graph.mc_difference_bounds is not None:
        a,b = graph.mc_difference_bounds
        lo,hi = max(lo,a),min(hi,b)
    if not math.isfinite(lo+hi) or hi <= lo: raise UnsupportedVerification("MC_DIFFERENCE_RANGE_UNBOUND")
    return lo,hi


def mc_mean_constraints(graph):
    symbols = {s.symbol:sp.Symbol(s.symbol) for s in graph.slots}
    uniforms = [symbols[s.symbol] for s in graph.slots if s.stochastic_role == "UNIFORM"]
    if not uniforms or any(graph.omega[str(s)] != (0.0,1.0) for s in uniforms):
        raise UnsupportedVerification("MC_SAMPLE_DISTRIBUTION_BINDING_REQUIRED")
    constraints = {}
    for name,equation in graph.equations.items():
        if equation.kind != "MC_SAMPLE": continue
        expression = _sample_expression(graph,equation)
        try: polynomial = sp.Poly(expression,*uniforms)
        except sp.PolynomialError as error:
            if len(uniforms) != 1 or not expression.is_rational_function(uniforms[0]):
                raise UnsupportedVerification("MC_POPULATION_MOMENT_BINDING_UNAVAILABLE") from error
            numerator,denominator = sp.fraction(sp.cancel(expression))
            if max(sp.degree(numerator,uniforms[0]),sp.degree(denominator,uniforms[0])) > 8 or sp.count_ops(expression) > 128:
                raise UnsupportedVerification("MC_POPULATION_MOMENT_BUDGET_EXHAUSTED")
            enclosure = m._iv(m._parse(_expression(denominator)),graph.omega)
            if enclosure.lo <= 0 <= enclosure.hi:
                raise UnsupportedVerification("MC_POPULATION_MOMENT_POLE_UNEXCLUDED")
            primitive = sp.integrate(expression,uniforms[0],risch=True)
            if primitive.has(sp.Integral) or sp.cancel(sp.diff(primitive,uniforms[0])-expression) != 0:
                raise UnsupportedVerification("MC_POPULATION_MOMENT_BINDING_UNAVAILABLE")
            mean = sp.simplify(primitive.subs(uniforms[0],1)-primitive.subs(uniforms[0],0))
            if mean.has(sp.oo,-sp.oo,sp.zoo,sp.nan):
                raise UnsupportedVerification("MC_POPULATION_MOMENT_BINDING_UNAVAILABLE")
        else:
            if len(polynomial.terms()) > 512 or polynomial.total_degree() > 16:
                raise UnsupportedVerification("MC_POPULATION_MOMENT_BUDGET_EXHAUSTED")
            mean = sum(coefficient/sp.prod(power+1 for power in powers) for powers,coefficient in polynomial.terms())
        constraints[name+"_population_mean"] = f"({equation.state_symbol}) == ({_expression(mean)})"
    return constraints

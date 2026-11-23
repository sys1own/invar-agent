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
from .default_verification import UnsupportedVerification, _equivalent, _expression, _point_value, _normalize_guarded_divisions
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
        if len(state.blueprint.contracts) != 1 or v9_transpiler_violations(code):
            raise UnsupportedVerification("ORACLE_PROTOCOL_UNBINDABLE")
        self.contract = state.blueprint.contracts[0]
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
        updates = _SymbolicKernel(self.proof_code,self.slots).updates(set(self.slots))
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
            result = self.machine.bridge.run(cpp,initial,self.contract.dt,1,timeout_seconds=2) if cpp else self.machine.bridge.run_python(instrument_purity(code),initial,self.contract.dt,1,timeout_seconds=2)
        except KernelExecutionError as error:
            if not error.compile_error: raise UnsupportedVerification("ORACLE_EXECUTION_FAILED") from error
            result = self.machine.bridge.run_python(instrument_purity(code),initial,self.contract.dt,1,timeout_seconds=2)
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
    def __init__(self, code, slots, box, dt, *, minimum_divisor=0.0):
        self.minimum_divisor = minimum_divisor
        self.functions = {n.name:n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
        self.values = {i:((box[s][0],box[s][1]),0.0) for s,i in slots.items()}
        self.body(self.functions["simulation_step"].body,{"buf":None,"dt":((dt,dt),0.0)},0)

    def expression(self,node,env,depth):
        if depth > 24: raise UnsupportedVerification("ROUNDING_BOUND_CALL_BUDGET")
        if isinstance(node,ast.Constant) and type(node.value) in (float,int):
            x = float(node.value); return ((x,x),0.0)
        if isinstance(node,ast.Name): return env[node.id]
        if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant): return self.values[node.slice.value]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            (a,b),e = self.expression(node.operand,env,depth)
            return ((-b,-a),e) if isinstance(node.op,ast.USub) else ((a,b),e)
        if isinstance(node,ast.Call) and ast.unparse(node.func) in {"exp","log","sqrt","sin","cos","math.exp","math.log","math.sqrt","math.sin","math.cos"}:
            if len(node.args) != 1 or node.keywords: raise UnsupportedVerification("ROUNDING_LIBRARY_SIGNATURE_UNBOUND")
            (lo,hi),error = self.expression(node.args[0],env,depth+1)
            name = ast.unparse(node.func).split(".")[-1]
            nominal = m._iv(m._parse(f"{name}(a)"),{"a":(lo,hi)})
            actual = m._iv(m._parse(f"{name}(a)"),{"a":(math.nextafter(lo-error,-math.inf),math.nextafter(hi+error,math.inf))})
            error = math.nextafter(max(abs(actual.lo-nominal.hi),abs(actual.hi-nominal.lo)),math.inf)
            if not math.isfinite(error+actual.lo+actual.hi): raise UnsupportedVerification("NONFINITE_ROUNDING_BOUND")
            return ((nominal.lo,nominal.hi),error)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in self.functions:
            function = self.functions[node.func.id]
            local = dict(zip((a.arg for a in function.args.args),(self.expression(a,env,depth+1) for a in node.args)))
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
        aa,bb = max(map(abs,a)),max(map(abs,b)); u = sys.float_info.epsilon
        if isinstance(node.op,(ast.Add,ast.Sub)): error = ea+eb+u*(aa+bb+ea+eb)/(1-u)
        elif isinstance(node.op,ast.Mult): error = ea*bb+eb*aa+ea*eb+u*(aa+ea)*(bb+eb)/(1-u)
        elif isinstance(node.op,ast.Div):
            minimum = min(map(abs,b)) if not b[0] <= 0 <= b[1] else 0
            if minimum-eb <= self.minimum_divisor: raise UnsupportedVerification("ROUNDING_DENOMINATOR_NOT_SEPARATED")
            error = (ea+aa*eb/minimum)/(minimum-eb)+u*(aa+ea)/(minimum-eb)/(1-u)
        else: raise UnsupportedVerification("RIGOROUS_ROUNDING_BOUND_UNAVAILABLE")
        op = {ast.Add:"+",ast.Sub:"-",ast.Mult:"*",ast.Div:"/"}[type(node.op)]
        value = m._iv(m._parse("a"+op+"b"),{"a":a,"b":b})
        error = math.nextafter(error*(1+128*u)+128*math.ulp(0.0),math.inf)
        if not math.isfinite(error+value.lo+value.hi): raise UnsupportedVerification("NONFINITE_ROUNDING_BOUND")
        return ((value.lo,value.hi),error)

    def body(self,body,env,depth):
        for node in body:
            if isinstance(node,ast.Assign):
                value = self.expression(node.value,env,depth)
                for target in node.targets:
                    if isinstance(target,ast.Name): env[target.id] = value
                    elif isinstance(target,ast.Subscript) and isinstance(target.slice,ast.Constant): self.values[target.slice.value] = value
                    else: raise UnsupportedVerification("ROUNDING_ASSIGNMENT_UNBOUND")
            elif isinstance(node,ast.Return): return self.expression(node.value,env,depth)
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call): self.expression(node.value,env,depth)
            elif not isinstance(node,(ast.Pass,ast.Import,ast.ImportFrom)): raise UnsupportedVerification("ROUNDING_PROGRAM_UNBOUND")


def _fixed(oracle):
    graph = oracle.graph
    if graph.quantifier_mode != "EXISTS": raise UnsupportedVerification("FIXED_POINT_ROOT_FAMILY_PROOF_REQUIRED")
    equations = {e.state_symbol:(name,e) for name,e in graph.equations.items() if e.kind in {"FIXED_POINT_RESIDUAL","FIXED_POINT_MAP"}}
    states = sorted(equations)
    if not states or len(equations) != len(graph.equations): raise UnsupportedVerification("FIXED_POINT_RESIDUAL_BINDING_REQUIRED")
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
    jtheta = [[_point_value(_expression(sp.diff(e,symbols[p])),point) for p in parameters] for e in residuals]
    dy = [[_point_value(_expression(sp.diff(sp.sympify(ast.unparse(raw_margin),locals=symbols),symbols[s])),point) for s in states]]
    direct = [[0.0 for p in parameters]]
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


def derive_evidence(machine,state,code,*,uast=None,phase="post_execution"):
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

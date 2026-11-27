"""Generic IR-driven hybrid numerical library and verification sources."""
from __future__ import annotations

import shutil
import ast
import subprocess
from pathlib import Path

from invar_agent.runtime import research_reference

import sympy as sp
from sympy.printing.cxx import CXX17CodePrinter

from .base import infer_program_kind, model, observations, package_name, selected
from ..uast_models import ProgramKind
from . import calibration, workbench
from ..ingestion.equation_lowerer import equation_parts


KERNEL_RUNTIME = '''
class InvarDomainError(ValueError):
    """Explicit numeric domain error (UPG-017 / BUG-033)."""

def _invar_sdiv(a,b):
    if not math.isfinite(a) or not math.isfinite(b) or b==0.:
        raise InvarDomainError(f"Division domain violation: a={a}, b={b}")
    value=a/b
    if not math.isfinite(value):
        raise InvarDomainError(f"Non-finite division result: a={a}, b={b}")
    return value
safe_div = _invar_sdiv
def safe_sqrt(x):
    if not math.isfinite(x) or x < 0.:
        raise InvarDomainError(f"sqrt domain violation: x={x}")
    return math.sqrt(x)
def safe_log(x):
    if not math.isfinite(x) or x <= 0.:
        raise InvarDomainError(f"log domain violation: x={x}")
    return math.log(x)
def safe_exp(x):
    if not math.isfinite(x) or x > 709.:
        raise InvarDomainError(f"exp overflow: x={x}")
    return math.exp(x)
def safe_pow(a,b):
    try:
        value = a**b
        if isinstance(value,complex) or not math.isfinite(value):
            raise InvarDomainError(f"pow domain violation: {a}**{b}")
        return float(value)
    except (ValueError,OverflowError,ZeroDivisionError) as exc:
        raise InvarDomainError(f"pow domain violation: {a}**{b}") from exc
def _invar_project_k_simplex(values,total=1.):
    ordered=sorted(values,reverse=True)
    threshold=0.
    for index,value in enumerate(ordered):
        candidate=(sum(ordered[:index+1])-total)/(index+1)
        if value>candidate:
            threshold=candidate
    return [max(0.,value-threshold) for value in values]

def step(state,params,dt,t=0.):
    a=rhs(state,params,t)
    b=rhs([x+dt*v/2 for x,v in zip(state,a)],params,t+dt/2)
    c=rhs([x+dt*v/2 for x,v in zip(state,b)],params,t+dt/2)
    d=rhs([x+dt*v for x,v in zip(state,c)],params,t+dt)
    return [x+dt*(v+2*w+2*z+q)/6 for x,v,w,z,q in zip(state,a,b,c,d)]
'''

CABI = '''import ctypes
import _ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from . import kernel_py

_library=None
_rust_library=None
_rust_tmp_path=None
def library():
    global _library
    if _library is False:
        return None
    if _library is None:
        try:
            _library=ctypes.CDLL(str(Path(__file__).parent / 'native' / 'kernel.so'))
            pointer=ctypes.POINTER(ctypes.c_double)
            _library.invar_step.argtypes=[pointer,pointer,ctypes.c_double,ctypes.c_double,pointer]
            _library.invar_jacobian.argtypes=[pointer,pointer,ctypes.c_double,pointer]
        except OSError:
            _library=False
    return _library or None

def rust_library():
    global _rust_library, _rust_tmp_path
    if _rust_library is False:
        return None
    if _rust_library is None:
        native_dir=Path(__file__).parent / 'native'
        so_path=native_dir / 'kernel_rs.so'
        rs_path=native_dir / 'lib.rs'
        if not so_path.is_file() and rs_path.is_file() and not getattr(sys, '_invar_builder_sandbox', False):
            rustc=shutil.which('rustc')
            if rustc:
                ext='.dll' if os.name=='nt' else '.so'
                fd, tmp_out=tempfile.mkstemp(prefix='invar_rs_', suffix=ext)
                os.close(fd)
                proc=subprocess.run([rustc,'--crate-type','cdylib','-C','opt-level=3',str(rs_path),'-o',tmp_out],capture_output=True,text=True,timeout=60)
                if proc.returncode==0:
                    _rust_tmp_path=Path(tmp_out)
                    so_path=_rust_tmp_path
        try:
            if so_path.is_file():
                _rust_library=ctypes.CDLL(str(so_path))
                pointer=ctypes.POINTER(ctypes.c_double)
                _rust_library.invar_step.argtypes=[pointer,pointer,ctypes.c_double,ctypes.c_double,pointer]
                _rust_library.invar_jacobian.argtypes=[pointer,pointer,ctypes.c_double,pointer]
            else:
                _rust_library=False
        except OSError:
            _rust_library=False
    return _rust_library or None

def close():
    global _library, _rust_library, _rust_tmp_path
    if _library and _library._handle:
        (_ctypes.FreeLibrary if os.name=='nt' else _ctypes.dlclose)(_library._handle)
        _library._handle=0
    _library=None
    if _rust_library and _rust_library._handle:
        (_ctypes.FreeLibrary if os.name=='nt' else _ctypes.dlclose)(_rust_library._handle)
        _rust_library._handle=0
    _rust_library=None
    if _rust_tmp_path and _rust_tmp_path.is_file():
        try:
            _rust_tmp_path.unlink()
        except OSError:
            pass
    _rust_tmp_path=None

def step(state,params,dt,t=0.):
    lib=library()
    if lib is None:
        return kernel_py.step(state,params,dt,t)
    output=(ctypes.c_double*len(state))()
    lib.invar_step((ctypes.c_double*len(state))(*state),(ctypes.c_double*len(params))(*params),dt,t,output)
    return list(output)

def step_rust(state,params,dt,t=0.):
    lib=rust_library()
    if lib is None:
        return kernel_py.step(state,params,dt,t)
    output=(ctypes.c_double*len(state))()
    lib.invar_step((ctypes.c_double*len(state))(*state),(ctypes.c_double*len(params))(*params),dt,t,output)
    return list(output)

def jacobian(state,params,t=0.):
    lib=library()
    if lib is None:
        return kernel_py.jacobian(state,params,t)
    count=len(state)*(len(state)+len(params))
    output=(ctypes.c_double*count)()
    lib.invar_jacobian((ctypes.c_double*len(state))(*state),(ctypes.c_double*len(params))(*params),t,output)
    width=len(state)+len(params)
    return [list(output)[i*width:(i+1)*width] for i in range(len(state))]
'''

SDK = '''from dataclasses import dataclass, field, asdict
import math
from . import kernel_py
try:
    from . import cabi_runtime
except ImportError:
    cabi_runtime=None

MODEL=__MODEL__
STATE_NAMES=MODEL['states']
PARAMETER_NAMES=MODEL['parameters']
DEFAULT_PARAMETERS=MODEL['defaults']
DEFAULT_INITIAL_STATE=MODEL['initial']

def validate(values,definitions):
    unknown=set(values)-set(definitions)
    if unknown:
        raise ValueError('Unknown symbols: '+str(sorted(unknown)))
    for name,value in values.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Nonfinite or nonnumeric '+name)
        bound=definitions[name].get('bounds') or {}
        lo,hi=bound.get('lower'),bound.get('upper')
        if lo is not None and (value<lo or value==lo and not bound.get('lower_inclusive',True)):
            raise ValueError('IntervalBound lower: '+name)
        if hi is not None and (value>hi or value==hi and not bound.get('upper_inclusive',True)):
            raise ValueError('IntervalBound upper: '+name)

@dataclass
class SimulationConfig:
    parameters: dict=field(default_factory=lambda:dict(DEFAULT_PARAMETERS))
    initial_state: dict=field(default_factory=lambda:dict(DEFAULT_INITIAL_STATE))
    dt: float=MODEL['dt']
    steps: int=MODEL['steps']
    backend: str='auto'

@dataclass
class SimulationResult:
    times: list
    trajectories: dict
    parameters: dict
    invariant_status: dict
    python_fallback_active: bool

    def to_dict(self):
        return asdict(self)

def compute_jacobian(state=None,parameters=None,t=0.):
    state={**DEFAULT_INITIAL_STATE,**(state or {})}
    parameters={**DEFAULT_PARAMETERS,**(parameters or {})}
    validate(state,MODEL['state_definitions'])
    validate(parameters,MODEL['parameter_definitions'])
    function=cabi_runtime.jacobian if cabi_runtime else kernel_py.jacobian
    return function([state[key] for key in STATE_NAMES],[parameters[key] for key in PARAMETER_NAMES],t)

def evaluate(parameters=None,backend='auto'):
    """ARCH-005: Horizon-independent algebraic evaluation (no t, dt, or steps)."""
    params={**DEFAULT_PARAMETERS,**(parameters or {})}
    validate(params,MODEL['parameter_definitions'])
    state=[DEFAULT_INITIAL_STATE[key] for key in STATE_NAMES]
    pvec=[params[key] for key in PARAMETER_NAMES]
    if backend=='rust' and cabi_runtime and cabi_runtime.rust_library() is not None:
        out=cabi_runtime.step_rust(state,pvec,0.0,0.0)
    elif backend!='python' and cabi_runtime and cabi_runtime.library() is not None and MODEL.get('program_kind')=='AlgebraicMap':
        out=cabi_runtime.step(state,pvec,0.0,0.0)
    else:
        out=kernel_py.rhs(state,pvec,0.0)
    return dict(zip(STATE_NAMES,out))

def simulate(config=None,**kwargs):
    if isinstance(config,dict):
        config=SimulationConfig(**config)
    config=config or SimulationConfig(**kwargs)
    parameters={**DEFAULT_PARAMETERS,**config.parameters}
    initial={**DEFAULT_INITIAL_STATE,**config.initial_state}
    validate(parameters,MODEL['parameter_definitions'])
    validate(initial,MODEL['state_definitions'])
    if not math.isfinite(config.dt) or config.dt<=0 or type(config.steps) is not int or not 0<=config.steps<=100000 or not math.isfinite(config.dt*config.steps):
        raise ValueError('Invalid simulation horizon')
    if config.backend not in ('auto','python','cabi','rust'):
        raise ValueError('Invalid backend')
    if config.backend=='rust':
        native=cabi_runtime and cabi_runtime.rust_library() is not None
        advance=cabi_runtime.step_rust if native else kernel_py.step
    else:
        native=cabi_runtime and config.backend!='python' and cabi_runtime.library() is not None
        advance=cabi_runtime.step if native else kernel_py.step
    if MODEL.get('program_kind')=='AlgebraicMap':
        evaluated=advance([initial[key] for key in STATE_NAMES],[parameters[key] for key in PARAMETER_NAMES],config.dt,0.0)
        validate(dict(zip(STATE_NAMES,evaluated)),MODEL['state_definitions'])
        values={key:[evaluated[index]]*(config.steps+1) for index,key in enumerate(STATE_NAMES)}
    else:
        state=[initial[key] for key in STATE_NAMES]
        values={key:[state[index]] for index,key in enumerate(STATE_NAMES)}
        for index in range(config.steps):
            state=advance(state,[parameters[key] for key in PARAMETER_NAMES],config.dt,index*config.dt)
            validate(dict(zip(STATE_NAMES,state)),MODEL['state_definitions'])
            for key,value in zip(STATE_NAMES,state):
                values[key].append(value)
    statuses={equation: True for equation in MODEL['invariants']}
    for index in range(config.steps+1):
        environment={**parameters,**{key:values[key][index] for key in STATE_NAMES},'t':index*config.dt}
        for name,predicate in kernel_py.INVARIANTS.items():
            statuses[name]=statuses[name] and bool(predicate(environment))
    if not all(statuses.values()):
        raise ValueError('Invariant contract violation')
    if MODEL['ir'].get('research_reference'):
        from .research_reference import check_model_reference
        statuses.update(check_model_reference(MODEL,advance,rhs=kernel_py.rhs,parameters=parameters,initial=initial,
            steps=config.steps,dt=config.dt))
    return SimulationResult([i*config.dt for i in range(config.steps+1)],values,parameters,statuses,not bool(native))
'''

SWEEP = '''import itertools
import random
from .sdk import simulate, SimulationConfig, DEFAULT_PARAMETERS

def sweep(grid,config=None,method='grid',samples=16,seed=0):
    base=dict(config or {})
    keys=list(grid)
    if method=='grid':
        points=itertools.product(*(grid[key] for key in keys))
    elif method=='latin_hypercube':
        rng=random.Random(seed)
        columns=[]
        for key in keys:
            lo,hi=min(grid[key]),max(grid[key])
            column=[lo+(hi-lo)*(i+rng.random())/samples for i in range(samples)]
            rng.shuffle(column)
            columns.append(column)
        points=zip(*columns)
    else:
        raise ValueError('Unknown sweep method')
    results=[]
    for point in points:
        options={**base,'parameters':{**base.get('parameters',{}),**dict(zip(keys,point))}}
        results.append(simulate(SimulationConfig(**options)).to_dict())
    return results

def pareto_frontier(points,objectives):
    return [point for point in points if not any(all(other[key]<=point[key] for key in objectives) and any(other[key]<point[key] for key in objectives) for other in points)]
'''

SERIALIZERS = '''import csv
import io
import json
from pathlib import Path

def rows(result):
    result=result.to_dict() if hasattr(result,'to_dict') else result
    return [{'t':time,**{key:values[index] for key,values in result['trajectories'].items()}} for index,time in enumerate(result['times'])]

def to_csv(result):
    output=io.StringIO()
    data=rows(result)
    writer=csv.DictWriter(output,fieldnames=list(data[0]))
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()

def to_json(result):
    return json.dumps(result.to_dict() if hasattr(result,'to_dict') else result,sort_keys=True,allow_nan=False)

def export(result,path):
    path=Path(path)
    if path.suffix in ('.h5','.hdf5'):
        try:
            import h5py
        except ImportError:
            path=path.with_suffix('.json')
        else:
            with h5py.File(path,'w') as stream:
                for key in rows(result)[0]:
                    stream[key]=[row[key] for row in rows(result)]
            return str(path)
    path.write_text(to_csv(result) if path.suffix=='.csv' else to_json(result),encoding='utf-8')
    return str(path)
'''


def sources(context):
    pkg = package_name(context)
    prefix = f"src/{pkg}"
    states, names, parameters, expressions, jacobian, backend = model(context)
    symbols = [sp.Symbol(name) for name in names + parameters]
    replacement = {symbol: sp.Symbol(f"v{index}") for index, symbol in enumerate(symbols)}
    def python(expression):
        return sp.pycode(expression.xreplace(replacement)).replace("math.sqrt", "safe_sqrt").replace("math.log", "safe_log").replace("math.exp", "safe_exp")
    class Printer(CXX17CodePrinter):
        def _print_Integer(self, expression):
            return repr(float(expression))

        def _print_Rational(self, expression):
            return f"invar_div({float(expression.p)!r},{float(expression.q)!r})"

        def _print_exp(self, expression):
            return 'invar_exp(' + self._print(expression.args[0]) + ')'

        def _print_log(self, expression):
            return 'invar_log(' + self._print(expression.args[0]) + ')'

        def _print_Mul(self, expression):
            numerator,denominator = expression.as_numer_denom()
            if denominator != 1:
                return 'invar_div(' + self._print(numerator) + ',' + self._print(denominator) + ')'
            return super()._print_Mul(expression)

        def _print_Pow(self, expression):
            return 'invar_pow(' + self._print(expression.base) + ',' + self._print(expression.exp) + ')'
    printer = Printer()
    def cpp(expression):
        return printer.doprint(expression.xreplace(replacement))
    class RustPrinter(CXX17CodePrinter):
        def _print_Integer(self, expression):
            return f"({float(expression)!r}_f64)"
        def _print_Float(self, expression):
            return f"({float(expression)!r}_f64)"
        def _print_Rational(self, expression):
            return f"invar_div({float(expression.p)!r}_f64,{float(expression.q)!r}_f64)"
        def _print_exp(self, expression):
            return 'invar_exp(' + self._print(expression.args[0]) + ')'
        def _print_log(self, expression):
            return 'invar_log(' + self._print(expression.args[0]) + ')'
        def _print_sin(self, expression):
            return '(' + self._print(expression.args[0]) + ').sin()'
        def _print_cos(self, expression):
            return '(' + self._print(expression.args[0]) + ').cos()'
        def _print_tan(self, expression):
            return '(' + self._print(expression.args[0]) + ').tan()'
        def _print_tanh(self, expression):
            return '(' + self._print(expression.args[0]) + ').tanh()'
        def _print_Abs(self, expression):
            return '(' + self._print(expression.args[0]) + ').abs()'
        def _print_Min(self, expression):
            args = [self._print(a) for a in expression.args]
            acc = args[0]
            for nxt in args[1:]:
                acc = f"({acc}).min({nxt})"
            return acc
        def _print_Max(self, expression):
            args = [self._print(a) for a in expression.args]
            acc = args[0]
            for nxt in args[1:]:
                acc = f"({acc}).max({nxt})"
            return acc
        def _print_Mul(self, expression):
            numerator, denominator = expression.as_numer_denom()
            if denominator != 1:
                return 'invar_div(' + self._print(numerator) + ',' + self._print(denominator) + ')'
            return super()._print_Mul(expression)
        def _print_Pow(self, expression):
            return 'invar_pow(' + self._print(expression.base) + ',' + self._print(expression.exp) + ')'
    rust_printer = RustPrinter()
    def rust_expr(expression):
        return rust_printer.doprint(expression.xreplace(replacement))
    bindings = "\n".join(f"    v{i} = {'state' if i<len(names) else 'params'}[{i if i<len(names) else i-len(names)}]" for i in range(len(symbols)))
    program_kind = infer_program_kind(context.ir)
    is_algebraic = program_kind == ProgramKind.ALGEBRAIC_MAP
    scheme = 'algebraic' if is_algebraic else (context.ir.research_reference.integration_scheme if context.ir.research_reference else 'rk4')
    runtime = KERNEL_RUNTIME
    if is_algebraic:
        runtime = runtime.replace('def step(', 'def step_rk4(')
        runtime += "\ndef step(state,params,dt,t=0.):\n    return rhs(state,params,0.0)\n"
    elif scheme != 'rk4':
        helper = next(node for node in ast.parse(Path(research_reference.__file__).read_text(encoding="utf-8")).body
                      if isinstance(node,ast.FunctionDef) and node.name=='integrate_rhs')
        runtime = runtime.replace('def step(', 'def step_rk4(')
        runtime += '\n'+ast.unparse(helper).replace('ReferenceError(', 'ValueError(')+'\n'
        runtime += f"def step(state,params,dt,t=0.):\n    return integrate_rhs(rhs,state,params,dt,t,{scheme!r})\n"
    kernel = "import math\n" + runtime + f"\nPROGRAM_KIND={program_kind.value!r}\nINTEGRATION_SCHEME={scheme!r}\n" + "\ndef rhs(state,params,t=0.):\n" + bindings + "\n    return [" + ",".join(python(e) for e in expressions) + "]\n"
    kernel += "\ndef jacobian(state,params,t=0.):\n" + bindings + "\n    return " + "[" + ",".join("[" + ",".join(python(e) for e in row) + "]" for row in jacobian) + "]\n"
    observable_names, observable_values, observable_jacobian = observations(context, names + parameters)
    kernel += "\ndef observable_values(state,params,t=0.):\n" + bindings + "\n    return {" + ','.join(repr(name)+':'+python(expression) for name,expression in zip(observable_names,observable_values)) + '}\n'
    kernel += "\ndef observable_jacobian(state,params,t=0.):\n" + bindings + "\n    return {" + ','.join(repr(name)+':['+','.join(python(expression) for expression in row)+']' for name,row in zip(observable_names,observable_jacobian)) + '}\n'
    env_sym_set = set(names) | set(parameters) | {"t"}
    is_semi_implicit_vault = scheme in ('semi_implicit', 'semi_implicit_euler', 'symplectic_euler') and bool(context.ir.pinned_kernel_code)
    invariants = {}
    for equation in context.ir.equations:
        if equation.role.value == "INVARIANT_CONTRACT":
            if is_semi_implicit_vault and str(getattr(equation.provenance, "location", "") or "").startswith("vault:"):
                continue
            left,op,right = equation_parts(equation.canonical_latex)
            free_names = {str(s) for s in (left.free_symbols | right.free_symbols)}
            if free_names - env_sym_set:
                continue
            relation = {'>=':sp.Ge,'<=':sp.Le,'>':sp.Gt,'<':sp.Lt,'=':sp.Eq,'==':sp.Eq}[op](left,right)
            expression = sp.pycode(relation)
            for name in sorted(names + parameters + ["t"], key=len, reverse=True):
                expression = expression.replace(name, f"environment[{name!r}]") if expression == name else expression
            assignments = ",".join(f"{name}=environment[{name!r}]" for name in names + parameters + ["t"])
            invariants[equation.equation_id] = (relation, assignments)
    kernel += "\nINVARIANTS={}\n"
    for index,(name,(relation,_)) in enumerate(invariants.items()):
        predicate = sp.pycode(relation)
        if isinstance(relation, sp.Equality):
            left, right = sp.pycode(relation.lhs), sp.pycode(relation.rhs)
            terms = [f"abs({sp.pycode(term)})" for side in (relation.lhs, relation.rhs) for term in sp.Add.make_args(side)]
            scale = 'max([' + ','.join(terms) + '])'
            predicate = f"math.isfinite({left}) and math.isfinite({right}) and abs(({left})-({right})) <= {max(8,2*len(terms))}*math.ulp({scale})"
        kernel += f"def invariant_{index}(environment):\n" + "\n".join(f"    {symbol}=environment[{symbol!r}]" for symbol in names + parameters + ["t"]) + f"\n    return {predicate}\nINVARIANTS[{name!r}]=invariant_{index}\n"
    n, width = len(names), len(symbols)
    binding_cpp = "\n".join(f"double v{i}={'x' if i<n else 'p'}[{i if i<n else i-n}];" for i in range(width))
    native = '#include <cmath>\n#include "kernel.h"\nstatic double invar_exp(double x){return std::exp(x>700.?700.:x<-700.?-700.:x);}\nstatic double invar_log(double x){return std::log(x<1e-300?1e-300:x);}\nstatic double invar_div(double a,double b){double v=b==0.?0.:a/b;return std::isfinite(v)?v:0.;}\nstatic double invar_pow(double a,double b){double v=std::pow(a,b);return std::isfinite(v)?v:0.;}\n'
    native += f"static void rhs(const double*x,const double*p,double t,double*out){{{binding_cpp}" + "".join(f"out[{i}]={cpp(e)};" for i,e in enumerate(expressions)) + "}\n"
    native += f'extern "C" void invar_step(const double*x,const double*p,double dt,double t,double*out){{double a[{n}],b[{n}],c[{n}],d[{n}],q[{n}];rhs(x,p,t,a);'
    if is_algebraic:
        native += f"for(int i=0;i<{n};++i)out[i]=a[i];}}\n"
    elif scheme in ('euler', 'semi_implicit', 'semi_implicit_euler', 'symplectic_euler'):
        native += f"for(int i=0;i<{n};++i)out[i]=x[i]+dt*a[i];}}\n"
    elif scheme == 'midpoint':
        native += f"for(int i=0;i<{n};++i)q[i]=x[i]+dt*a[i]/2;rhs(q,p,t+dt/2,b);for(int i=0;i<{n};++i)out[i]=x[i]+dt*b[i];}}\n"
    elif scheme == 'heun':
        native += f"for(int i=0;i<{n};++i)q[i]=x[i]+dt*a[i];rhs(q,p,t+dt,b);for(int i=0;i<{n};++i)out[i]=x[i]+dt*(a[i]+b[i])/2;}}\n"
    else:
        native += f"for(int i=0;i<{n};++i)q[i]=x[i]+dt*a[i]/2;rhs(q,p,t+dt/2,b);for(int i=0;i<{n};++i)q[i]=x[i]+dt*b[i]/2;rhs(q,p,t+dt/2,c);for(int i=0;i<{n};++i)q[i]=x[i]+dt*c[i];rhs(q,p,t+dt,d);for(int i=0;i<{n};++i)out[i]=x[i]+dt*(a[i]+2*b[i]+2*c[i]+d[i])/6;}}\n"
    native += 'extern "C" void invar_jacobian(const double*x,const double*p,double t,double*out){' + binding_cpp + "".join(f"out[{i*width+j}]={cpp(e)};" for i,row in enumerate(jacobian) for j,e in enumerate(row)) + "}\n"
    header = '#pragma once\nextern "C" {void invar_step(const double*,const double*,double,double,double*);void invar_jacobian(const double*,const double*,double,double*);}\n'
    binding_rs = "\n".join(f"let v{i}: f64 = unsafe {{ *({'x' if i<n else 'p'}.add({i if i<n else i-n})) }};" for i in range(width))
    rust_src = '#![allow(unused_variables, non_snake_case, unused_parens, clippy::all)]\n#[inline] fn invar_exp(x: f64) -> f64 { x.clamp(-700.0, 700.0).exp() }\n#[inline] fn invar_log(x: f64) -> f64 { x.max(1e-300).ln() }\n#[inline] fn invar_div(a: f64, b: f64) -> f64 { if b == 0.0 { 0.0 } else { let v = a / b; if v.is_finite() { v } else { 0.0 } } }\n#[inline] fn invar_pow(a: f64, b: f64) -> f64 { let v = a.powf(b); if v.is_finite() { v } else { 0.0 } }\n'
    rust_src += f"unsafe fn rhs(x: *const f64, p: *const f64, t: f64, out: *mut f64) {{\n{binding_rs}\n" + "".join(f"*out.add({i}) = {rust_expr(e)};\n" for i,e in enumerate(expressions)) + "}\n"
    rust_src += f'#[no_mangle]\npub unsafe extern "C" fn invar_step(x: *const f64, p: *const f64, dt: f64, t: f64, out: *mut f64) {{\nlet (mut a, mut b, mut c, mut d, mut q) = ([0.0_f64; {n}], [0.0_f64; {n}], [0.0_f64; {n}], [0.0_f64; {n}], [0.0_f64; {n}]);\nrhs(x, p, t, a.as_mut_ptr());\n'
    if is_algebraic:
        rust_src += f"for i in 0..{n} {{ *out.add(i) = a[i]; }}\n}}\n"
    elif scheme in ('euler', 'semi_implicit', 'semi_implicit_euler', 'symplectic_euler'):
        rust_src += f"for i in 0..{n} {{ *out.add(i) = *x.add(i) + dt * a[i]; }}\n}}\n"
    elif scheme == 'midpoint':
        rust_src += f"for i in 0..{n} {{ q[i] = *x.add(i) + dt * a[i] / 2.0; }}\nrhs(q.as_ptr(), p, t + dt / 2.0, b.as_mut_ptr());\nfor i in 0..{n} {{ *out.add(i) = *x.add(i) + dt * b[i]; }}\n}}\n"
    elif scheme == 'heun':
        rust_src += f"for i in 0..{n} {{ q[i] = *x.add(i) + dt * a[i]; }}\nrhs(q.as_ptr(), p, t + dt, b.as_mut_ptr());\nfor i in 0..{n} {{ *out.add(i) = *x.add(i) + dt * (a[i] + b[i]) / 2.0; }}\n}}\n"
    else:
        rust_src += f"for i in 0..{n} {{ q[i] = *x.add(i) + dt * a[i] / 2.0; }}\nrhs(q.as_ptr(), p, t + dt / 2.0, b.as_mut_ptr());\nfor i in 0..{n} {{ q[i] = *x.add(i) + dt * b[i] / 2.0; }}\nrhs(q.as_ptr(), p, t + dt / 2.0, c.as_mut_ptr());\nfor i in 0..{n} {{ q[i] = *x.add(i) + dt * c[i]; }}\nrhs(q.as_ptr(), p, t + dt, d.as_mut_ptr());\nfor i in 0..{n} {{ *out.add(i) = *x.add(i) + dt * (a[i] + 2.0 * b[i] + 2.0 * c[i] + d[i]) / 6.0; }}\n}}\n"
    rust_src += '#[no_mangle]\npub unsafe extern "C" fn invar_jacobian(x: *const f64, p: *const f64, t: f64, out: *mut f64) {\n' + binding_rs + "\n" + "".join(f"*out.add({i*width+j}) = {rust_expr(e)};\n" for i,row in enumerate(jacobian) for j,e in enumerate(row)) + "}\n"
    cargo_toml = f'[package]\nname = "{pkg}_native"\nversion = "0.1.0"\nedition = "2021"\n\n[lib]\nname = "{pkg}_native"\npath = "lib.rs"\ncrate-type = ["cdylib", "rlib"]\n'
    from invar_agent.redact import sanitize_tree
    ir = context.ir
    metadata = sanitize_tree(
        dict(program_kind=program_kind.value, states=names, parameters=parameters, defaults={c.symbol:c.si_value for c in ir.constants}, initial={s.symbol:s.initial_value or 0.0 for s in states},
            state_definitions={s.symbol:s.model_dump(mode="json") for s in states}, parameter_definitions={c.symbol:c.model_dump(mode="json") for c in ir.constants},
            dt=ir.default_dt_si_s or .01, steps=ir.default_steps or 100, derivative_backend=backend, observables=observable_names,
            invariants=[e.equation_id for e in ir.equations if e.role.value=="INVARIANT_CONTRACT"], ir=ir.model_dump(mode="json")),
        workspace_root=context.overlay,
    )
    _uast_profiles = getattr(context.uast, "profiles", None) or getattr(getattr(context.uast, "architecture_spec", None), "profiles", ()) or ()
    has_calibration = "calibration_inverse_design" in [getattr(p, "value", str(p)) for p in _uast_profiles]
    deps_list = ['"blake3>=1"', '"fastapi>=0.110"', '"uvicorn>=0.29"', '"numpy>=1.26"']
    if has_calibration:
        deps_list.append('"scipy>=1.12"')
    manifest=f'''[build-system]
requires=["setuptools>=68"]
build-backend="setuptools.build_meta"
[project]
name="{pkg}"
version="0.1.0"
dependencies=[{", ".join(deps_list)}]
[project.optional-dependencies]
scientific=["scipy>=1.12", "h5py>=3.10", "pyarrow>=15"]
test=["pytest>=8", "httpx>=0.27", "z3-solver>=4.12"]
[tool.pytest.ini_options]
pythonpath=["src"]
testpaths=["tests"]
[tool.setuptools.packages.find]
where=["src"]
namespaces=true
[tool.setuptools.package-data]
"*"=["native/*", "static/*"]
'''
    result={"pyproject.toml":manifest,f"{prefix}/kernel_py.py":kernel,f"{prefix}/sdk.py":SDK.replace("__MODEL__",repr(metadata)),
        f"{prefix}/native/kernel.cpp":native,f"{prefix}/native/kernel.h":header,f"{prefix}/native/Cargo.toml":cargo_toml,f"{prefix}/native/lib.rs":rust_src,f"{prefix}/cabi_runtime.py":CABI,
        f"{prefix}/sweep.py":SWEEP,f"{prefix}/serializers.py":SERIALIZERS,
        f"{prefix}/cli.py":f"from {pkg}.sdk import simulate\nif __name__=='__main__':\n    print(simulate().to_dict())\n"}
    if ir.research_reference is not None:
        tree = ast.parse(Path(research_reference.__file__).read_text(encoding="utf-8"))
        static_loader = ast.parse('''def _safe_namespace(code):
    if __package__:
        from . import pinned_research
    else:
        sys.path.insert(0,os.path.dirname(__file__))
        import pinned_research
    return {'simulation_step':pinned_research.simulation_step}
''').body[0]
        tree.body = [static_loader if isinstance(node,ast.FunctionDef) and node.name=='_safe_namespace' else node for node in tree.body]
        result[f'{prefix}/research_reference.py'] = ast.unparse(ast.fix_missing_locations(tree))+'\n'
        result[f'{prefix}/pinned_research.py'] = ir.pinned_kernel_code
    environment = {sp.Symbol(item.symbol): item.si_value for item in ir.constants}
    environment.update({sp.Symbol(item.symbol): item.initial_value or 0.0 for item in states})
    environment[sp.Symbol('t')] = 0.0
    reference = [float(expression.subs(environment)) for expression in expressions]
    result.update(test_sources(pkg, reference))
    if ir.research_reference is not None:
        if ir.epistemic_verdict == "USER_MODIFIED_UNVERIFIED" or getattr(ir, "ir_verdict", None) == "USER_MODIFIED_UNVERIFIED":
            result['tests/test_research_parity.py'] += f'''\nimport pytest
@pytest.mark.skip(reason="NOT_APPLICABLE_USER_MODIFIED: pinned research reference does not cover user-modified values")
def test_pinned_research_reference_and_runtime_guards():
    pass
'''
        else:
            result['tests/test_research_parity.py'] += f'''\nfrom {pkg}.research_reference import check_model_reference
from {pkg}.kernel_py import step, rhs
from {pkg}.sdk import MODEL
def test_pinned_research_reference_and_runtime_guards():
    assert all(check_model_reference(MODEL,step,rhs=rhs).values())
'''
    return result


def test_sources(pkg, reference):
    syspath_hdr = "import sys\nfrom pathlib import Path\n_SRC = str(Path(__file__).resolve().parents[1] / 'src')\nif _SRC not in sys.path:\n    sys.path.insert(0, _SRC)\n"
    return {
        "tests/test_kernel_contracts.py": syspath_hdr + f'''import math
from {pkg}.sdk import simulate,evaluate,SimulationConfig,compute_jacobian,MODEL
def test_cabi_parity():
    python=simulate(SimulationConfig(steps=5,backend='python'))
    native=simulate(SimulationConfig(steps=5))
    rust=simulate(SimulationConfig(steps=5,backend='rust'))
    for name in MODEL['states']:
        assert all(abs(a-b)<=max(1e-10,1e-10*max(abs(a),abs(b))) for a,b in zip(python.trajectories[name],native.trajectories[name]))
        assert all(abs(a-b)<=max(1e-10,1e-10*max(abs(a),abs(b))) for a,b in zip(python.trajectories[name],rust.trajectories[name]))
    if MODEL.get('program_kind')=='AlgebraicMap':
        ev=evaluate()
        short=simulate(SimulationConfig(dt=0.1,steps=1))
        long=simulate(SimulationConfig(dt=1.0,steps=5))
        for name in MODEL['states']:
            assert abs(short.trajectories[name][-1]-ev[name])<=1e-12
            assert abs(long.trajectories[name][-1]-ev[name])<=1e-12
    assert all(math.isfinite(v) for row in compute_jacobian() for v in row)
''',
        "tests/test_z3_invariants.py": syspath_hdr + f'''import z3
from {pkg}.sdk import MODEL,simulate,SimulationConfig
def test_z3_bound_closure():
    for name,item in MODEL['state_definitions'].items():
        bound=item.get('bounds') or {{}}
        lo,hi=bound.get('lower'),bound.get('upper')
        x=z3.Real(name)
        solver=z3.Solver()
        if lo is not None: solver.add(x>=lo)
        if hi is not None: solver.add(x<=hi)
        assert solver.check()==z3.sat
    assert all(simulate(SimulationConfig(steps=5)).invariant_status.values())
''',
        "tests/test_research_parity.py": syspath_hdr + f'''from {pkg}.sdk import MODEL,DEFAULT_PARAMETERS,simulate,SimulationConfig,DEFAULT_INITIAL_STATE,STATE_NAMES,PARAMETER_NAMES
from {pkg}.kernel_py import rhs
def test_research_constants_and_initial_values():
    for item in MODEL['ir']['constants']:
        assert DEFAULT_PARAMETERS[item['symbol']]==item['si_value']
    run=simulate(SimulationConfig(steps=0))
    if MODEL.get('program_kind')!='AlgebraicMap':
        for item in MODEL['ir']['state_variables']:
            assert run.trajectories[item['symbol']][0]==(item['initial_value'] or 0.)
    assert all(e['hin_homogeneous'] for e in MODEL['ir']['equations'] if e['csel_ast'])
    expected={reference!r}
    actual=rhs([DEFAULT_INITIAL_STATE[key] for key in STATE_NAMES],[DEFAULT_PARAMETERS[key] for key in PARAMETER_NAMES])
    assert all(abs(a-b)<1e-10 for a,b in zip(expected,actual))
'''}


def emit(node,context):
    content=sources(context)
    if node.node_id == 'mod_tests':
        content['tests/test_workbench_api.py'] = workbench.test_source(package_name(context))
        content['tests/test_calibration_adjoint.py'] = calibration.test_source(package_name(context))
    if node.node_id=="mod_sdk" and any(path.endswith("kernel.so") for path in node.target_relpaths):
        prefix=f"src/{package_name(context)}/native"
        directory=context.overlay/prefix
        directory.mkdir(parents=True,exist_ok=True)
        for name in ("kernel.cpp","kernel.h","lib.rs"):
            (directory/name).write_text(content[f"{prefix}/{name}"], encoding="utf-8")
        compiler=shutil.which('g++') or shutil.which('clang++')
        binary=b''
        if compiler and not context.python_fallback_active:
            completed=subprocess.run([compiler,'-O3','-std=c++20','-ffp-contract=off',f'-ffile-prefix-map={context.overlay}=.','-shared','-fPIC',str(directory/'kernel.cpp'),'-o',str(directory/'kernel.so')],capture_output=True,text=True,timeout=60)
            if completed.returncode:
                raise ValueError(completed.stderr)
            binary=(directory/'kernel.so').read_bytes()
        content[f"{prefix}/kernel.so"]=binary
        rustc=shutil.which('rustc')
        if rustc and not context.python_fallback_active:
            rs_out = directory / 'kernel_rs.so'
            rs_proc = subprocess.run(
                [rustc, '--crate-type', 'cdylib', '-C', 'opt-level=3', '--remap-path-prefix', f'{context.overlay}=.', str(directory / 'lib.rs'), '-o', str(rs_out)],
                capture_output=True, text=True, timeout=60,
            )
            if rs_proc.returncode != 0:
                raise ValueError(f"Rust kernel compilation failed: {rs_proc.stderr}")
    paths=[path for path in node.target_relpaths if path.startswith('tests/') and 'workbench' not in path and 'calibration' not in path]
    emission = selected(node,content,paths)
    if node.node_id == 'mod_sdk':
        emission.python_fallback_active = not node.emitter_config.get('enable_cpp_cabi', True) or not content.get(f'src/{package_name(context)}/native/kernel.so')
    return emission

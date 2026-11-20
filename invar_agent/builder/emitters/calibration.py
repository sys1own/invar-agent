"""Bounded generic ODE calibration using exact variational sensitivities."""
from .base import package_name, selected
from . import hybrid_compute


LOADER = '''import csv
import math
from pathlib import Path

def load_data(path,si_scales=None):
    path=Path(path)
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        rows=pq.read_table(path).to_pylist()
    elif path.suffix in ('.h5','.hdf5'):
        import h5py
        with h5py.File(path) as stream:
            rows=[{key:stream[key][i] for key in stream} for i in range(len(next(iter(stream.values()))))]
    else:
        with path.open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream,delimiter='\\t' if path.suffix=='.tsv' else ','))
    if not rows:
        raise ValueError('Empty calibration dataset')
    result=[]
    for row in rows:
        values={key:float(value)*(si_scales or {}).get(key,1.) for key,value in row.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError('Nonfinite observation')
        result.append(values)
    return result
'''

FITTER = '''import math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from ..sdk import MODEL,DEFAULT_PARAMETERS,DEFAULT_INITIAL_STATE,STATE_NAMES,PARAMETER_NAMES,validate
from .. import kernel_py
try:
    from .. import cabi_runtime
except ImportError:
    cabi_runtime=None
from .data_loader import load_data

def _prediction(rows,theta,names,observables,mode,initial):
    parameters={**DEFAULT_PARAMETERS,**dict(zip(names,theta))}
    validate(parameters,MODEL['parameter_definitions'])
    indices=[PARAMETER_NAMES.index(name) for name in names]
    n=len(STATE_NAMES)
    state=np.array([initial[key] for key in STATE_NAMES],dtype=float)
    sensitivity=np.zeros((n,len(names)))
    predictions=[]
    derivatives=[]
    current=0.
    def flow(y,s,t):
        values=[parameters[key] for key in PARAMETER_NAMES]
        matrix=np.asarray((cabi_runtime.jacobian if cabi_runtime else kernel_py.jacobian)(y,values,t))
        return np.asarray(kernel_py.rhs(y,values,t)),matrix[:,:n]@s+matrix[:,[n+i for i in indices]]
    for row in rows:
        if mode=='trajectory':
            target=row['t']
            if target<current:
                raise ValueError('Observation times must be nondecreasing')
            while current<target-1e-14:
                dt=min(MODEL['dt'],target-current)
                a,sa=flow(state,sensitivity,current)
                scheme=getattr(kernel_py,'INTEGRATION_SCHEME','rk4')
                if scheme=='euler':
                    state=state+dt*a
                    sensitivity=sensitivity+dt*sa
                elif scheme=='heun':
                    b,sb=flow(state+dt*a,sensitivity+dt*sa,current+dt)
                    state=state+dt*(a+b)/2
                    sensitivity=sensitivity+dt*(sa+sb)/2
                else:
                    b,sb=flow(state+dt*a/2,sensitivity+dt*sa/2,current+dt/2)
                    if scheme=='midpoint':
                        state=state+dt*b
                        sensitivity=sensitivity+dt*sb
                    else:
                        c,sc=flow(state+dt*b/2,sensitivity+dt*sb/2,current+dt/2)
                        d,sd=flow(state+dt*c,sensitivity+dt*sc,current+dt)
                        state=state+dt*(a+2*b+2*c+d)/6
                        sensitivity=sensitivity+dt*(sa+2*sb+2*sc+sd)/6
                current+=dt
            p=[parameters[key] for key in PARAMETER_NAMES]
            algebraic=kernel_py.observable_values(state,p,current)
            gradients=kernel_py.observable_jacobian(state,p,current)
            values=[]
            jac=[]
            for name in observables:
                if name in STATE_NAMES:
                    index=STATE_NAMES.index(name)
                    values.append(state[index]); jac.append(sensitivity[index])
                else:
                    gradient=np.asarray(gradients[name])
                    values.append(algebraic[name]); jac.append(gradient[:n]@sensitivity+gradient[[n+i for i in indices]])
        else:
            local={**parameters,**{key:value for key,value in row.items() if key in PARAMETER_NAMES and key not in names}}
            y=[row.get(key,initial[key]) for key in STATE_NAMES]
            p=[local[key] for key in PARAMETER_NAMES]
            rhs=kernel_py.rhs(y,p)
            matrix=np.asarray(kernel_py.jacobian(y,p))
            algebraic=kernel_py.observable_values(y,p)
            gradients=kernel_py.observable_jacobian(y,p)
            values=[]
            jac=[]
            for name in observables:
                if name in algebraic:
                    values.append(algebraic[name]); jac.append(np.asarray(gradients[name])[[n+i for i in indices]])
                else:
                    if name not in STATE_NAMES and n!=1:
                        raise ValueError('Ambiguous rate output: '+name)
                    index=STATE_NAMES.index(name) if name in STATE_NAMES else 0
                    values.append(rhs[index]); jac.append(matrix[index,[n+i for i in indices]])
        for name,value,gradient in zip(observables,values,jac):
            sigma=row.get('sigma_'+name,row.get('sigma',1.))
            if sigma<=0:
                raise ValueError('Observation sigma must be positive')
            predictions.append((value-row[name])/sigma)
            derivatives.append(np.divide(gradient,sigma))
    return np.asarray(predictions),np.asarray(derivatives)

def fit(data,initial_guess=None,parameters=None,observables=None,mode='auto',loss='linear',initial_state=None):
    rows=load_data(data) if isinstance(data,(str,Path)) else list(data)
    if not rows:
        raise ValueError('Empty calibration dataset')
    if not all(math.isfinite(float(value)) for row in rows for value in row.values()):
        raise ValueError('Nonfinite observation')
    mode=('trajectory' if 't' in rows[0] else 'rate') if mode=='auto' else mode
    if mode not in ('trajectory','rate') or loss not in ('linear','soft_l1','huber','cauchy'):
        raise ValueError('Unsupported calibration mode/loss')
    names=list(parameters or [key for key in PARAMETER_NAMES if MODEL['parameter_definitions'][key]['is_tunable_parameter'] and key not in rows[0]])
    if not names or set(names)-set(PARAMETER_NAMES):
        raise ValueError('No valid calibration parameters')
    observations=list(observables or ([key for key in STATE_NAMES+MODEL['observables'] if key in rows[0]] if mode=='trajectory' else [key for key in rows[0] if key not in set(STATE_NAMES+PARAMETER_NAMES+['sigma']) and not key.startswith('sigma_')]))
    if not observations or any(name not in row for row in rows for name in observations):
        raise ValueError('No valid observed outputs')
    initial={**DEFAULT_INITIAL_STATE,**(initial_state or {})}
    validate(initial,MODEL['state_definitions'])
    lower=[]
    upper=[]
    for name in names:
        bound=MODEL['parameter_definitions'][name].get('bounds') or {}
        lo,hi=bound.get('lower'),bound.get('upper')
        lower.append(-np.inf if lo is None else lo if bound.get('lower_inclusive',True) else np.nextafter(lo,np.inf))
        upper.append(np.inf if hi is None else hi if bound.get('upper_inclusive',True) else np.nextafter(hi,-np.inf))
    guess=[(initial_guess or {}).get(name,DEFAULT_PARAMETERS[name]) for name in names]
    def evaluate(theta):
        return _prediction(rows,theta,names,observations,mode,initial)
    result=least_squares(lambda theta:evaluate(theta)[0],guess,jac=lambda theta:evaluate(theta)[1],bounds=(lower,upper),method='trf',x_scale='jac',loss=loss,ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=1000)
    _,singular,vh=np.linalg.svd(result.jac,full_matrices=False)
    cutoff=np.finfo(float).eps*max(result.jac.shape)*(singular[0] if len(singular) else 0.)
    keep=singular>cutoff
    rank=int(sum(keep))
    variance=2*result.cost/max(1,len(result.fun)-rank)
    inverse=np.zeros_like(singular)
    inverse[keep]=np.reciprocal(singular[keep])
    covariance=(vh.T*np.square(inverse))@vh*variance
    stderr=np.sqrt(np.maximum(0,np.diag(covariance)))
    condition=float(singular[0]/singular[-1]) if rank==len(names) and len(singular) else math.inf
    return {'parameters':dict(zip(names,result.x.tolist())),'stderr':dict(zip(names,stderr.tolist())),'three_sigma':dict(zip(names,(3*stderr).tolist())),
        'covariance':covariance.tolist(),'condition_number':condition,'rank':rank,'rank_deficient':rank<len(names),'cost':float(result.cost),'success':bool(result.success),'jacobian':result.jac.tolist(),'residuals':result.fun.tolist(),'nfev':result.nfev}

def residuals_and_jacobian(data,parameters,observables=None,mode='trajectory',initial_state=None):
    rows=load_data(data) if isinstance(data,str) else list(data)
    names=list(parameters)
    observations=observables or [name for name in STATE_NAMES if name in rows[0]]
    return _prediction(rows,list(parameters.values()),names,observations,mode,{**DEFAULT_INITIAL_STATE,**(initial_state or {})})
'''


def test_source(pkg):
    return f'''import numpy as np
from scipy.optimize._numdiff import approx_derivative
from {pkg}.sdk import DEFAULT_PARAMETERS,STATE_NAMES,simulate,SimulationConfig
from {pkg}.calibration.fitter import residuals_and_jacobian
def test_analytical_adjoint_matches_three_point():
    result=simulate(SimulationConfig(steps=3,backend='python'))
    rows=[{{'t':time,**{{key:values[index] for key,values in result.trajectories.items()}}}} for index,time in enumerate(result.times)]
    names=list(DEFAULT_PARAMETERS)
    if not names: return
    theta=np.array(list(DEFAULT_PARAMETERS.values()))
    _,analytical=residuals_and_jacobian(rows,dict(zip(names,theta)))
    finite=approx_derivative(lambda values:residuals_and_jacobian(rows,dict(zip(names,values)))[0],theta,method='3-point',rel_step=1e-4)
    assert np.max(np.abs(analytical-finite))<1e-7
'''


def emit(node,context):
    prefix=f"src/{package_name(context)}"
    return selected(node,{f"{prefix}/serializers.py":hybrid_compute.SERIALIZERS,f"{prefix}/calibration/data_loader.py":LOADER,f"{prefix}/calibration/fitter.py":FITTER})

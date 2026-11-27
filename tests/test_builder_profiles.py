from __future__ import annotations

import asyncio
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from invar_agent.builder.emitters.receipts import BuildReceipt
from invar_agent.builder.ingestion.pipeline import CanonicalResearchIngestionPipeline, seal_ir, reverify_ir, apply_ir_override
from invar_agent.builder.ingestion.schemas import CanonicalResearchIR, IngestionRequest, IntervalBoundModel, IRReviewOverride
from invar_agent.builder.planner import compile_builder_uast
from invar_agent.builder.state_machine import BuilderStateMachine, BuildRequest
from invar_agent.builder.uast_models import BuilderArchitectureSpec, BuilderProfile
from invar_agent.builder.research_bridge import build_research_instrument
from test_builder_state_machine import offline


def run_python(root, code=None, args=()):
    environment={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'}
    command=[sys.executable,'-B',*(['-c',code] if code is not None else args)]
    return subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True,timeout=180)


@pytest.fixture(scope='module')
def instrument(tmp_path_factory):
    root=tmp_path_factory.mktemp('four_profiles')
    paper=root/'paper.tex'
    paper.write_text(r'''\section{Dimensionless Michaelis--Menten substrate depletion}
\begin{equation}\dot{x}=-v*x/(k+x)\end{equation}
\begin{equation}x >= 0\end{equation}
\begin{tabular}{lll}
Symbol & Value & Unit \\
v & 2e-5 & 1/s \\
k & 0.025 & 1 \\
\end{tabular}
''', encoding="utf-8")
    dataset=root/'observations.csv'
    rng=np.random.default_rng(7)
    x=.1
    rows=[]
    def rhs(y): return -2e-5*y/(.025+y)
    for index in range(501):
        if index%10==0:
            noise=rng.normal(0,2e-5)
            rows.extend({'t':index*10.,'x':x+sign*noise,'sigma':2e-5} for sign in (-1,1))
        a=rhs(x); b=rhs(x+5*a); c=rhs(x+5*b); d=rhs(x+10*c)
        x+=10*(a+2*b+2*c+d)/6
    with dataset.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['t','x','sigma']); writer.writeheader(); writer.writerows(rows)
    ir=CanonicalResearchIngestionPipeline(root).ingest(IngestionRequest(upload_paths=[str(paper),str(dataset)],title='Michaelis Menten'))
    assert not [flag for flag in ir.disambiguation_flags if not flag.resolved], ir.disambiguation_flags
    ir.state_variables[0].initial_value=.1
    ir.state_variables[0].bounds=IntervalBoundModel(lower=0,upper=1)
    for constant in ir.constants:
        constant.bounds=IntervalBoundModel(lower=1e-8 if constant.symbol=='v' else 1e-4,upper=1e-3 if constant.symbol=='v' else 1.)
    ir.user_approved=True; ir.default_dt_si_s=10.; ir.default_steps=20
    reverify_ir(ir); seal_ir(ir)
    spec=BuilderArchitectureSpec(project_name='enzyme_lab',profiles=list(BuilderProfile))
    graph=compile_builder_uast(ir,spec)
    state=asyncio.run(BuilderStateMachine(root).run(BuildRequest(build_id='all_profiles',ir=ir,uast=graph)))
    assert state.status in {'COMPLETED','COMPLETED_WITH_FALLBACKS'}, (state.error,state.events[-8:])
    return Path(state.workspace),state,dataset


def test_compose_all_four_profiles_michaelis_menten(instrument):
    root,state,dataset=instrument
    assert state.total_llm_tokens_used==0
    assert state.receipt['verification']['pytest_exit_code']==0
    assert state.receipt['verification']['pytest_passed']>=5
    assert BuildReceipt.model_validate(state.receipt).fallbacks[0].fallback_verified
    result=run_python(root,args=['-m','pytest','-q','-p','no:cacheprovider'])
    assert result.returncode==0,result.stdout+result.stderr
    destination=Path('/home/ubuntu/.devin-files/invar-m5-example')
    if Path('/home/ubuntu').exists():
        if destination.exists(): shutil.rmtree(destination)
        shutil.copytree(root,destination,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))


def test_profile1_cabi_vs_python_parity(instrument):
    root,_,_=instrument
    result=run_python(root,"""from enzyme_lab.sdk import *
from enzyme_lab import cabi_runtime
python=simulate(SimulationConfig(steps=100,backend='python'))
native=simulate(SimulationConfig(steps=100))
assert max(abs(a-b) for a,b in zip(python.trajectories['x'],native.trajectories['x']))<1e-10
assert MODEL['derivative_backend']=='native_adjoint'
assert not native.python_fallback_active
cabi_runtime.close()
print('C-ABI/Python parity <= 1e-10; native adjoint active')
""")
    assert result.returncode==0,result.stdout+result.stderr


def test_profile2_workbench_bounds_and_export(instrument):
    root,_,_=instrument
    result=run_python(root,"""from fastapi.testclient import TestClient
from enzyme_lab.workbench.app import app
with TestClient(app) as client:
 assert client.get('/api/schema').status_code==200
 assert client.post('/api/simulate',json={'parameters':{'v':-1}}).status_code==422
 assert client.post('/api/simulate',json={'steps':4}).json()['trajectories']['x']
 assert len(client.post('/api/sweep',json={'grid':{'v':[1e-5,2e-5]},'config':{'steps':2}}).json())==2
 assert client.post('/api/export/csv',json={'steps':2}).text.startswith('t,x')
 assert client.post('/api/export/json',json={'steps':2}).json()['times']
 assert client.post('/api/export/hdf5',json={'steps':2}).status_code==200
 assert 'data:' in client.get('/api/sweep/stream',params={'grid':'{\"v\":[0.00002]}'}).text
""")
    assert result.returncode==0,result.stdout+result.stderr


def test_profile3_executable_paper_and_receipt_tamper(instrument,tmp_path):
    root,_,_=instrument
    result=run_python(root,args=['-m','enzyme_lab.audit.cli','run','--output',str(tmp_path/'report')])
    assert result.returncode==0,result.stdout+result.stderr
    for name in ('reproducibility_report.md','reproducibility_report.html','reproducibility_table.tex'):
        assert (tmp_path/'report'/name).stat().st_size>20
    clean=run_python(root,args=['-m','enzyme_lab.audit.verify_receipt'])
    assert clean.returncode==0,clean.stdout+clean.stderr
    copied=tmp_path/'tampered'; shutil.copytree(root,copied)
    sdk=copied/'src/enzyme_lab/sdk.py'; sdk.write_bytes(sdk.read_bytes()+b' ')
    tampered=run_python(copied,args=['-m','enzyme_lab.audit.verify_receipt'])
    assert tampered.returncode!=0 and 'sdk.py' in tampered.stderr


def test_profile4_calibration_adjoint_jacobian(instrument):
    root,_,dataset=instrument
    code=f'''import numpy as np
from scipy.optimize._numdiff import approx_derivative
from enzyme_lab.calibration.fitter import fit,residuals_and_jacobian
from enzyme_lab.calibration.data_loader import load_data
rows=load_data({str(dataset)!r})
theta=np.array([2e-5,.025])
def evaluate(values): return residuals_and_jacobian(rows,dict(zip(['v','k'],values)))
actual=evaluate(theta)[1]
finite=approx_derivative(lambda values:evaluate(values)[0],theta,method='3-point',rel_step=1e-5)
assert np.linalg.norm(actual-finite)/np.linalg.norm(finite)<1e-7
result=fit(rows,initial_guess={{'v':1.5e-5,'k':.04}})
assert result['success'] and result['rank']==2
for key,truth in [('v',2e-5),('k',.025)]:
 assert abs(result['parameters'][key]-truth)<3*result['stderr'][key],result
print(result)
'''
    result=run_python(root,code)
    assert result.returncode==0,result.stdout+result.stderr


def test_generic_two_state_system_and_python_only_profile_subset(tmp_path,monkeypatch):
    paper=tmp_path/'transfer.tex'
    paper.write_text(r'''\begin{equation}\dot{a}=-r*a\end{equation}
\begin{equation}\dot{b}=r*a-q*b\end{equation}
\begin{tabular}{lll}
Symbol & Value & Unit \\
r & 0.3 & 1/s \\
q & 0.7 & 1/s \\
\end{tabular}
''', encoding="utf-8")
    ir=CanonicalResearchIngestionPipeline(tmp_path).ingest(IngestionRequest(upload_paths=[str(paper)]))
    for state in ir.state_variables:
        state.initial_value=.2 if state.symbol=='a' else .4
        state.bounds=IntervalBoundModel(lower=0,upper=2)
    for parameter in ir.constants:
        parameter.bounds=IntervalBoundModel(lower=.01,upper=2)
    ir.default_dt_si_s=.001; ir.default_steps=100; ir.user_approved=True
    ir=apply_ir_override(ir,IRReviewOverride(state_symbol_order=['a','b'],approve=True))
    original=shutil.which
    monkeypatch.setattr(shutil,'which',lambda command:None if command in {'g++','clang++'} else original(command))
    state=asyncio.run(build_research_instrument(tmp_path,ir=ir,profiles=[BuilderProfile.HYBRID_COMPUTE_LIBRARY,BuilderProfile.CALIBRATION_INVERSE_DESIGN],build_id='generic',
        leaf_slots=[{'slot_id':'bespoke_data_adapter','intent_summary':'Adapt scientific rows using the frozen contract'}]))
    assert state.status=='COMPLETED_WITH_FALLBACKS',state.error
    assert state.python_fallback_active and state.total_llm_tokens_used==0
    assert state.fallbacks[0]['fallback_emitter_id']=='fb_csv_json_adapter'
    root=Path(state.workspace)
    result=run_python(root,f'''import math
from {state.receipt['project_id']}.sdk import simulate,SimulationConfig,MODEL
run=simulate(SimulationConfig(steps=100))
t=.1
assert len(run.trajectories)==2 and run.python_fallback_active
assert abs(run.trajectories['a'][-1]-.2*math.exp(-.3*t))<1e-10
expected=.4*math.exp(-.7*t)+.3*.2/(.7-.3)*(math.exp(-.3*t)-math.exp(-.7*t))
assert abs(run.trajectories['b'][-1]-expected)<1e-10
''')
    assert result.returncode==0,result.stdout+result.stderr


def test_calibration_empty_nonfinite_and_rank_deficient_inputs(instrument):
    root,_,_=instrument
    result=run_python(root,'''import math
from enzyme_lab.calibration.fitter import fit
for rows in ([],[{'t':0.,'x':math.nan}]):
 try: fit(rows)
 except ValueError: pass
 else: raise AssertionError('Invalid data accepted')
result=fit([{'t':0.,'x':.1},{'t':0.,'x':.1}])
assert result['rank_deficient'] and result['rank']==0 and math.isinf(result['condition_number'])
''')
    assert result.returncode==0,result.stdout+result.stderr


def test_uploaded_initial_conditions_drive_conservation_build(tmp_path):
    paper = tmp_path/'closed.tex'
    paper.write_text(r'''\begin{equation}\dot{x}=-k*x\end{equation}
\begin{equation}\dot{y}=k*x\end{equation}
\begin{equation}x+y == 1\end{equation}
\begin{equation}x(0)=1\end{equation}
\begin{equation}y(0)=0\end{equation}
\begin{equation}x >= 0\end{equation}
\begin{equation}y >= 0\end{equation}
\begin{tabular}{lll}
Symbol & Value & Unit \\
k & 0.1 & 1/s \\
\end{tabular}
''', encoding="utf-8")
    ir = CanonicalResearchIngestionPipeline(tmp_path).ingest(IngestionRequest(upload_paths=[str(paper)]))
    ir = apply_ir_override(ir, IRReviewOverride(state_symbol_order=['x', 'y'], default_steps=20))
    state = asyncio.run(build_research_instrument(tmp_path, ir=ir,
        profiles=[BuilderProfile.HYBRID_COMPUTE_LIBRARY, BuilderProfile.EXECUTABLE_PAPER_AUDIT], build_id='closed'))
    assert state.status == 'COMPLETED', state.error
    assert state.total_llm_tokens_used == 0
    result = run_python(Path(state.workspace), '''from instrument.sdk import simulate, SimulationConfig
run=simulate(SimulationConfig(steps=20))
assert run.trajectories['x'][0] == 1
assert run.trajectories['y'][0] == 0
assert 0 < run.trajectories['x'][-1] < 1
assert run.trajectories['y'][-1] > 0
assert all(run.invariant_status.values())
from scientific_instrument.kernel_py import INVARIANTS
predicate=INVARIANTS['equation:2']
assert not predicate({'x': .5, 'y': .5+1e-8, 'k': .1, 't': 0.})
assert not predicate({'x': float('inf'), 'y': 0., 'k': .1, 't': 0.})
'''.replace('from instrument.sdk', f"from {state.receipt['project_id']}.sdk"))
    assert result.returncode == 0, result.stdout+result.stderr


def test_cached_receipt_retains_verified_fallback_and_backend(instrument):
    root,previous,_=instrument
    build_root=root.parent
    ir_json=json.loads((build_root/'canonical_research_ir.json').read_text(encoding="utf-8"))
    ir=CanonicalResearchIR.model_validate(ir_json)
    spec=BuilderArchitectureSpec(project_name='enzyme_lab',profiles=list(BuilderProfile))
    workspace_root=build_root.parents[2]
    state=asyncio.run(BuilderStateMachine(workspace_root).run(BuildRequest(build_id='all_profiles',ir=ir,uast=compile_builder_uast(ir,spec))))
    assert state.status=='COMPLETED_WITH_FALLBACKS',state.error
    assert state.fallbacks==previous.fallbacks and state.cabi_compiled
    assert state.receipt['verification']['pytest_exit_code']==0
    assert sum(event['event']=='CACHE_HIT_SKIPPED' for event in state.events)==11


def test_rate_calibration_uses_generic_ir_algebraic_observables(tmp_path):
    paper=tmp_path/'flux.tex'
    paper.write_text(r'''\begin{equation}flux=v*x/(k+x)\end{equation}
\begin{equation}\dot{x}=-flux\end{equation}
\begin{tabular}{lll}
Symbol & Value & Unit \\
v & 2e-5 & 1/s \\
k & 0.025 & 1 \\
\end{tabular}
''', encoding="utf-8")
    ir=CanonicalResearchIngestionPipeline(tmp_path).ingest(IngestionRequest(upload_paths=[str(paper)]))
    ir.state_variables[0].initial_value=.1
    ir.state_variables[0].bounds=IntervalBoundModel(lower=0,upper=1)
    for parameter in ir.constants:
        parameter.bounds=IntervalBoundModel(lower=1e-8,upper=.01 if parameter.symbol=='v' else 1.)
    ir.user_approved=True; ir.default_dt_si_s=1.; ir.default_steps=10
    reverify_ir(ir); seal_ir(ir)
    spec=BuilderArchitectureSpec(project_name='flux_model',profiles=[BuilderProfile.CALIBRATION_INVERSE_DESIGN])
    state=asyncio.run(BuilderStateMachine(tmp_path).run(BuildRequest(build_id='flux',ir=ir,uast=compile_builder_uast(ir,spec))))
    assert state.status=='COMPLETED',state.error
    result=run_python(Path(state.workspace),'''import numpy as np
from flux_model.calibration.fitter import fit
rows=[]
for x in np.linspace(.001,.2,30):
 value=2e-5*x/(.025+x)
 rows.extend([{'x':x,'flux':value-1e-8,'sigma':1e-8},{'x':x,'flux':value+1e-8,'sigma':1e-8}])
result=fit(rows,initial_guess={'v':1.5e-5,'k':.04})
assert result['success']
assert abs(result['parameters']['v']-2e-5)<1e-10
assert abs(result['parameters']['k']-.025)<1e-6
''')
    assert result.returncode==0,result.stdout+result.stderr

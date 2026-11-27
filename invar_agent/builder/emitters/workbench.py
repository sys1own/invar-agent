"""Standalone nonblocking scientific HTTP workbench."""
from .base import package_name, selected


APP = '''import json
import itertools
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool
from ..sdk import MODEL, SimulationConfig, simulate
from ..sweep import sweep
from ..serializers import to_csv, to_json

app=FastAPI(title='Scientific Workbench')


class SweepRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    grid: Dict[str, List[float]] = {}
    config: Optional[Dict[str, Any]] = None
    method: str = 'grid'
    samples: int = 16


@app.get('/api/schema')
def schema():
    return MODEL

@app.get('/')
def index():
    return HTMLResponse((Path(__file__).parent/'static'/'index.html').read_text(encoding='utf-8'))

async def execute(payload):
    try:
        return await run_in_threadpool(simulate,SimulationConfig(**payload))
    except (ValueError,TypeError,OverflowError) as error:
        raise HTTPException(422,str(error)) from error

@app.post('/api/simulate')
async def simulation(payload: dict):
    return (await execute(payload)).to_dict()

@app.post('/api/sweep')
async def parameter_sweep(payload: SweepRequest):
    known_params = set(MODEL.get('parameter_definitions', {}).keys())
    unknown = set(payload.grid.keys()) - known_params
    if unknown:
        raise HTTPException(422, f'Unknown sweep parameter(s): {sorted(unknown)}')
    keys = list(payload.grid)
    def _run_grid():
        out = []
        for point in itertools.product(*(payload.grid[k] for k in keys)):
            pt_params = dict(zip(keys, point))
            try:
                res = sweep({k: [v] for k, v in pt_params.items()}, payload.config, 'grid', 1)[0]
                out.append({'ok': True, 'parameters': res['parameters'], 'result': res, 'trajectories': res['trajectories'], 'violations': []})
            except Exception as err:
                out.append({'ok': False, 'parameters': pt_params, 'result': None, 'trajectories': {}, 'violations': [str(err)]})
        return out
    try:
        return await run_in_threadpool(_run_grid)
    except (ValueError,TypeError) as error:
        raise HTTPException(422,str(error)) from error

@app.get('/api/sweep/stream')
async def sweep_stream(grid: str='{}'):
    try:
        options=json.loads(grid)
        if not isinstance(options, dict):
            raise ValueError('grid must be a JSON object')
        keys=list(options)
        known_params = set(MODEL.get('parameter_definitions', {}).keys())
        unknown = set(keys) - known_params
        if unknown:
            raise ValueError(f'Unknown sweep parameter(s): {sorted(unknown)}')
        count=math.prod(len(options[key]) for key in keys) if keys else 0
        if count>4096:
            raise ValueError('Sweep limit exceeded')
    except (ValueError,TypeError) as error:
        raise HTTPException(422,str(error)) from error
    async def events():
        for index,point in enumerate(itertools.product(*(options[key] for key in keys))):
            pt = {key:[value] for key,value in zip(keys,point)}
            try:
                results=await run_in_threadpool(sweep,pt)
                yield 'data: '+json.dumps({'index':index,'total':count,'ok':True,'result':results[0],'violations':[]})+'\\n\\n'
            except Exception as err:
                yield 'data: '+json.dumps({'index':index,'total':count,'ok':False,'parameters':dict(zip(keys,point)),'violations':[str(err)]})+'\\n\\n'
    return StreamingResponse(events(),media_type='text/event-stream')

@app.post('/api/export/{kind}')
async def export(kind: str,payload: dict):
    result=await execute(payload)
    if kind=='csv':
        return Response(to_csv(result),media_type='text/csv')
    if kind=='json':
        return Response(to_json(result),media_type='application/json')
    if kind=='hdf5':
        try:
            import h5py
            import io
            buffer=io.BytesIO()
            with h5py.File(buffer,'w') as stream:
                stream['t']=result.times
                for name,values in result.trajectories.items(): stream[name]=values
            return Response(buffer.getvalue(),media_type='application/x-hdf5')
        except ImportError:
            return Response(to_json(result),media_type='application/json',headers={'X-Invar-Fallback':'json'})
    raise HTTPException(422,'Unknown export format')

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8000)
'''

HTML = '''<!doctype html><html><head><meta charset="utf-8"><title>Scientific Workbench</title>
<style>body{font:16px system-ui;max-width:1100px;margin:3em auto;background:#101b29;color:#e3f2ff}label{display:flex;align-items:center;gap:12px;margin:0.8em 0}svg{background:#0b1420;border:1px solid #2b415a;width:100%;height:340px}button{padding:0.7em 1.2em;margin-right:8px;cursor:pointer}.ok{color:#5f9}.err{color:#f87171;font-weight:600}.legend{display:flex;gap:16px;margin:8px 0;font-size:13px}.readout{font-family:monospace;min-width:72px;color:#93c5fd}</style></head>
<body><h1>Scientific Workbench</h1><section id="controls"></section><button id="run">Simulate</button><button id="sweep">2D sweep</button><button id="csv">CSV</button><button id="json">JSON</button><div id="status" role="status"></div><div id="legend" class="legend"></div><svg id="plot" viewBox="0 0 800 340"></svg><pre id="results"></pre>
<script>
let model,config=()=>({parameters:Object.fromEntries([...document.querySelectorAll('input[type=range]')].map(e=>[e.name,Number(e.value)]))});
const COLORS=['#60a5fa','#f87171','#34d399','#c084fc','#fbbf24'];
async function post(path,data){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw Error(await r.text());return r}
function setStatus(text,ok){const el=document.getElementById('status');el.textContent=text;el.className=ok?'ok':'err'}
fetch('/api/schema').then(r=>r.json()).then(m=>{model=m;for(let [name,d] of Object.entries(m.parameter_definitions)){let label=document.createElement('label');let txt=document.createElement('span');txt.textContent=name+' ['+d.dimension.canonical_unit+']';let input=document.createElement('input');input.type='range';input.name=name;input.min=d.bounds?.lower??0;input.max=d.bounds?.upper??Math.max(1,d.si_value*2);input.step='any';input.value=d.si_value;let out=document.createElement('span');out.className='readout';out.textContent=Number(input.value).toPrecision(4);input.oninput=()=>{out.textContent=Number(input.value).toPrecision(4)};label.append(txt,input,out);document.getElementById('controls').append(label)}});
document.getElementById('run').onclick=async()=>{try{let data=await(await post('/api/simulate',config())).json(),svg=document.getElementById('plot'),leg=document.getElementById('legend');svg.replaceChildren();leg.replaceChildren();let ax=document.createElementNS('http://www.w3.org/2000/svg','path');ax.setAttribute('d','M 40 20 L 40 300 L 780 300');ax.setAttribute('stroke','#475569');ax.setAttribute('fill','none');svg.append(ax);let color=0;for(let [name,values] of Object.entries(data.trajectories)){let c=COLORS[color++%COLORS.length],scale=Math.max(...values.map(Math.abs),1e-30),p=document.createElementNS('http://www.w3.org/2000/svg','polyline');p.setAttribute('points',values.map((v,i)=>`${40+740*i/Math.max(1,values.length-1)},${300-260*v/scale}`).join(' '));p.setAttribute('fill','none');p.setAttribute('stroke',c);p.setAttribute('stroke-width','2');p.setAttribute('aria-label',name);svg.append(p);let item=document.createElement('span');item.style.color=c;item.textContent='● '+name+' (final='+values[values.length-1].toPrecision(4)+')';leg.append(item)}setStatus(Object.entries(data.invariant_status).map(([k,v])=>(v?'● ':'✖ ')+k).join(' ')||'● finite and within bounds',true)}catch(e){setStatus(e.message,false)}};
document.getElementById('sweep').onclick=async()=>{try{let cur=config().parameters;let grid=Object.fromEntries(Object.entries(model.parameter_definitions).slice(0,2).map(([n,d])=>{let base=cur[n]??d.si_value;let lo=d.bounds?.lower??base*0.5;let hi=d.bounds?.upper??base*1.5;let span=Math.max(1e-6,(hi-lo)*0.25);let a=Math.max(lo,base-span),b=Math.min(hi,base+span);return [n,[a,base,b]]}));let data=await(await post('/api/sweep',{grid,config:config()})).json(),keys=Object.keys(grid),svg=document.getElementById('plot'),leg=document.getElementById('legend');svg.replaceChildren();leg.innerHTML='<span style="color:#34d399">● Valid</span><span style="color:#f87171">● Violated</span>';let ax=document.createElementNS('http://www.w3.org/2000/svg','path');ax.setAttribute('d','M 40 20 L 40 300 L 780 300');ax.setAttribute('stroke','#475569');ax.setAttribute('fill','none');svg.append(ax);let okCount=0;for(let pt of data){let coordinates=keys.map(k=>(pt.parameters[k]-Math.min(...grid[k]))/(Math.max(...grid[k])-Math.min(...grid[k])||1)),circle=document.createElementNS('http://www.w3.org/2000/svg','circle'),title=document.createElementNS('http://www.w3.org/2000/svg','title');circle.setAttribute('cx',60+700*coordinates[0]);circle.setAttribute('cy',280-240*(coordinates[1]??.5));circle.setAttribute('r',10);circle.setAttribute('fill',pt.ok?'#34d399':'#f87171');if(pt.ok)okCount++;title.textContent=JSON.stringify(pt.ok?{parameters:pt.parameters,final:Object.fromEntries(Object.entries(pt.trajectories).map(([k,v])=>[k,v[v.length-1]]))}:{parameters:pt.parameters,violations:pt.violations});circle.append(title);svg.append(circle)}setStatus(`2D sweep: ${okCount}/${data.length} valid points`,okCount>0);document.getElementById('results').textContent=JSON.stringify({axes:keys,total:data.length,valid:okCount,points:data.map(p=>({ok:p.ok,parameters:p.parameters,violations:p.violations}))},null,2)}catch(e){setStatus(e.message,false)}};
for(let kind of ['csv','json'])document.getElementById(kind).onclick=async()=>{try{let blob=await(await post('/api/export/'+kind,config())).blob(),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='simulation.'+kind;a.click();URL.revokeObjectURL(a.href)}catch(e){setStatus(e.message,false)}};
</script></body></html>'''


def test_source(pkg):
    return f'''from fastapi.testclient import TestClient
from {pkg}.workbench.app import app
from {pkg}.sdk import MODEL
def test_workbench_bounds_exports():
    with TestClient(app) as client:
        assert client.get('/api/schema').status_code==200
        assert client.post('/api/simulate',json={{'steps':2}}).status_code==200
        assert client.post('/api/export/csv',json={{'steps':2}}).text.startswith('t,')
        assert client.post('/api/export/json',json={{'steps':2}}).json()['times']
        for name,item in MODEL['parameter_definitions'].items():
            bounds=item.get('bounds') or {{}}
            if bounds.get('lower') is not None:
                assert client.post('/api/simulate',json={{'parameters':{{name:bounds['lower']-1}}}}).status_code==422
'''


def emit(node,context):
    prefix=f"src/{package_name(context)}/workbench"
    return selected(node,{f"{prefix}/app.py":APP,f"{prefix}/static/index.html":HTML})

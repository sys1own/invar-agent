"""Standalone nonblocking scientific HTTP workbench."""
from .base import package_name, selected


APP = '''import json
import itertools
import math
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool
from ..sdk import MODEL, SimulationConfig, simulate
from ..sweep import sweep
from ..serializers import to_csv, to_json

app=FastAPI(title='Scientific Workbench')

@app.get('/api/schema')
def schema():
    return MODEL

@app.get('/')
def index():
    return HTMLResponse((Path(__file__).parent/'static'/'index.html').read_text())

async def execute(payload):
    try:
        return await run_in_threadpool(simulate,SimulationConfig(**payload))
    except (ValueError,TypeError,OverflowError) as error:
        raise HTTPException(422,str(error)) from error

@app.post('/api/simulate')
async def simulation(payload: dict):
    return (await execute(payload)).to_dict()

@app.post('/api/sweep')
async def parameter_sweep(payload: dict):
    try:
        return await run_in_threadpool(sweep,payload.get('grid',{}),payload.get('config'),payload.get('method','grid'),payload.get('samples',16))
    except (ValueError,TypeError) as error:
        raise HTTPException(422,str(error)) from error

@app.get('/api/sweep/stream')
async def sweep_stream(grid: str='{}'):
    try:
        options=json.loads(grid)
        keys=list(options)
        count=math.prod(len(options[key]) for key in keys)
        if count>4096:
            raise ValueError('Sweep limit exceeded')
    except (ValueError,TypeError) as error:
        raise HTTPException(422,str(error)) from error
    async def events():
        for index,point in enumerate(itertools.product(*(options[key] for key in keys))):
            results=await run_in_threadpool(sweep,{key:[value] for key,value in zip(keys,point)})
            yield 'data: '+json.dumps({'index':index,'total':count,'result':results[0]})+'\\n\\n'
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
<style>body{font:16px system-ui;max-width:1100px;margin:3em auto;background:#101b29;color:#e3f2ff}label{display:block;margin:1em}svg{background:white;width:100%;height:320px}button{padding:1em}.ok{color:#5f9}</style></head>
<body><h1>Scientific Workbench</h1><section id="controls"></section><button id="run">Simulate</button><button id="sweep">2D sweep</button><button id="csv">CSV</button><button id="json">JSON</button><div id="status" role="status"></div><svg id="plot" viewBox="0 0 800 320"></svg><pre id="results"></pre>
<script>
let model,config=()=>({parameters:Object.fromEntries([...document.querySelectorAll('input')].map(e=>[e.name,Number(e.value)]))});
async function post(path,data){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw Error(await r.text());return r}
fetch('/api/schema').then(r=>r.json()).then(m=>{model=m;for(let [name,d] of Object.entries(m.parameter_definitions)){let label=document.createElement('label');label.textContent=name+' ['+d.dimension.canonical_unit+'] ';let input=document.createElement('input');input.type='range';input.name=name;input.min=d.bounds?.lower??0;input.max=d.bounds?.upper??Math.max(1,d.si_value*2);input.step='any';input.value=d.si_value;label.append(input);document.getElementById('controls').append(label)}});
document.getElementById('run').onclick=async()=>{try{let data=await(await post('/api/simulate',config())).json(),svg=document.getElementById('plot');svg.replaceChildren();let color=0;for(let [name,values] of Object.entries(data.trajectories)){let scale=Math.max(...values.map(Math.abs),1e-30),p=document.createElementNS('http://www.w3.org/2000/svg','polyline');p.setAttribute('points',values.map((v,i)=>`${20+760*i/Math.max(1,values.length-1)},${290-260*v/scale}`).join(' '));p.setAttribute('fill','none');p.setAttribute('stroke',['blue','red','green','purple'][color++%4]);p.setAttribute('aria-label',name);svg.append(p)}document.getElementById('status').textContent=Object.entries(data.invariant_status).map(([k,v])=>(v?'● ':'✖ ')+k).join(' ')||'● finite and within bounds';document.getElementById('status').className='ok'}catch(e){document.getElementById('status').textContent=e.message}};
document.getElementById('sweep').onclick=async()=>{try{let grid=Object.fromEntries(Object.entries(model.parameter_definitions).slice(0,2).map(([n,d])=>[n,[d.bounds?.lower??d.si_value*.5,d.bounds?.upper??d.si_value*1.5]])),data=await(await post('/api/sweep',{grid,config:config()})).json(),keys=Object.keys(grid),svg=document.getElementById('plot');svg.replaceChildren();for(let result of data){let coordinates=keys.map(k=>(result.parameters[k]-Math.min(...grid[k]))/(Math.max(...grid[k])-Math.min(...grid[k])||1)),circle=document.createElementNS('http://www.w3.org/2000/svg','circle'),title=document.createElementNS('http://www.w3.org/2000/svg','title');circle.setAttribute('cx',40+720*coordinates[0]);circle.setAttribute('cy',280-240*(coordinates[1]??.5));circle.setAttribute('r',12);circle.setAttribute('fill','teal');title.textContent=JSON.stringify({parameters:result.parameters,final:Object.fromEntries(Object.entries(result.trajectories).map(([k,v])=>[k,v[v.length-1]]))});circle.append(title);svg.append(circle)}document.getElementById('results').textContent=JSON.stringify({axes:keys,runs:data.length},null,2)}catch(e){document.getElementById('status').textContent=e.message}};
for(let kind of ['csv','json'])document.getElementById(kind).onclick=async()=>{let blob=await(await post('/api/export/'+kind,config())).blob(),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='simulation.'+kind;a.click();URL.revokeObjectURL(a.href)};
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

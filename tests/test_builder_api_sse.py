"""Hermetic HTTP lifecycle tests through the real Builder and shared broadcaster."""
import io
import json
import os
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from invar_agent.builder.uast_models import BuilderProfile


PAPER = r'''\begin{equation}\dot{x}=-k*x\end{equation}
\begin{equation}x >= 0\end{equation}
\begin{tabular}{lll}
Symbol & Value & Unit \\
k & 0.1 & 1/s \\
\end{tabular}
'''


@pytest.fixture
def builder_client(tmp_path, monkeypatch):
    import invar_agent.server as server
    from invar_agent.config import GENERIC_KEY_ENV, PROVIDER_KEY_ENVS
    for key in set(PROVIDER_KEY_ENVS.values()) | {GENERIC_KEY_ENV}:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('INVAR_RESEARCH_SEARCH_LIVE','0')
    monkeypatch.setattr(server,'load_persisted_state',lambda:{})
    monkeypatch.setattr(server,'_STATE',None)
    app=server.create_app(tmp_path)
    with TestClient(app) as client:
        yield client


def prepare_plan(client, profiles=None):
    response=client.post('/api/v1/builder/ingest',files={'files':('paper.tex',PAPER,'text/plain')},data={'allow_llm_disambiguation':'false'})
    assert response.status_code==200,response.text
    original=response.json()
    response=client.patch('/api/v1/builder/ir/'+original['ir_id']+'/override',json={'approve':True,'default_steps':12,
        'constant_patches':[{'constant_id':original['constants'][0]['constant_id'],'bounds':{'lower':0.,'upper':1.}}]})
    assert response.status_code==200,response.text
    ir=response.json()
    assert ir['ir_blake3_digest']!=original['ir_blake3_digest']
    response=client.post('/api/v1/builder/plan',json={'ir_id':ir['ir_id'],'profiles':profiles or [profile.value for profile in BuilderProfile],
        'project_name':'api_instrument','leaf_slots':['domain_visualization']})
    assert response.status_code==200,response.text
    return ir,response.json()


def wait_build(client,build_id):
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        response=client.get('/api/v1/builder/status/'+build_id)
        assert response.status_code==200,response.text
        result=response.json()
        if result['status']!='RUNNING':
            assert result['status'] in {'COMPLETED','COMPLETED_WITH_FALLBACKS'},result
            return result
        time.sleep(.05)
    pytest.fail('Builder did not finish')


def test_ingest_override_plan_execute_export_flow(builder_client):
    client=builder_client
    ir,plan=prepare_plan(client)
    assert ir['source_mode']=='upload' and ir['llm_disambiguation_tokens_used']==0
    assert all(item['hin_homogeneous'] and item['z3_satisfiable'] for item in ir['equations'])
    assert client.get('/api/v1/builder/ir/'+ir['ir_id']).json()==ir
    assert plan['spectral_radius']==0 and len(plan['frozen_his_digests'])==11
    bad=client.patch('/api/v1/builder/ir/'+ir['ir_id']+'/override',json={'constant_patches':[{'constant_id':ir['constants'][0]['constant_id'],'si_value':2.}]})
    assert bad.status_code==422 and 'Z3_SPEC_UNSAT' in bad.text
    assert client.get('/api/v1/builder/ir/'+ir['ir_id']).json()['ir_blake3_digest']==ir['ir_blake3_digest']
    response=client.post('/api/v1/builder/execute',json={'plan_id':plan['plan_id']})
    assert response.status_code==202,response.text
    build_id=response.json()['build_id']
    result=wait_build(client,build_id)
    assert result['receipt']['total_llm_tokens_used']==0
    archive=client.get('/api/v1/builder/export/'+build_id+'.zip')
    assert archive.status_code==200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        assert 'provenance/build_receipt.json' in bundle.namelist()
        assert len(bundle.read('src/api_instrument/sdk.py'))>100
        assert json.loads(bundle.read('provenance/build_receipt.json'))['merkle_root_blake3']==result['receipt']['merkle_root_blake3']
    response=client.post('/api/v1/builder/instrument/'+build_id+'/launch',json={})
    assert response.status_code==200,response.text
    assert response.json()['health_ok'] and response.json()['url'].startswith('http://127.0.0.1:')
    assert client.get('/api/v1/builder/instrument/'+build_id).json()['running']
    assert client.delete('/api/v1/builder/instrument/'+build_id+'/launch').status_code==200
    assert not client.get('/api/v1/builder/instrument/'+build_id).json()['running']
    response=client.post('/api/v1/builder/rebuild/'+build_id,json={'modified_slot_prompts':{'domain_visualization':'A revised domain visualization'}})
    assert response.status_code==202,response.text
    update=response.json()
    assert set(update['rebuild_nodes'])=={'slot_custom_viz','mod_receipt'}
    assert len(update['cached_nodes'])==9 and update['smw_capacitance_det']==1.
    wait_build(client,update['build_id'])


def test_rebuild_sse_does_not_replay_previous_execution_terminal(builder_client):
    client=builder_client
    _,plan=prepare_plan(client)
    build_id=client.post('/api/v1/builder/execute',json={'plan_id':plan['plan_id']}).json()['build_id']
    wait_build(client,build_id)
    first=client.get('/api/v1/builder/stream',params={'build_id':build_id})
    prior=[json.loads(line[6:]) for line in first.text.splitlines() if line.startswith('data: ')]
    response=client.post('/api/v1/builder/rebuild/'+build_id,json={'modified_slot_prompts':{'domain_visualization':'Accessible SVG legend'}})
    assert response.status_code==202,response.text
    wait_build(client,build_id)
    stream=client.get('/api/v1/builder/stream',params={'build_id':build_id})
    events=[json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith('data: ')]
    assert all(event['seq']>max(item['seq'] for item in prior) for event in events)
    assert set(response.json()['cached_nodes'])=={event['module_id'] for event in events if event['event']=='CACHE_HIT_SKIPPED'}
    assert sum(event['event']=='builder.completed' and event['channel']=='log' for event in events)==1


def test_upload_over_50mb_returns_413(builder_client,monkeypatch):
    response=builder_client.post('/api/v1/builder/ingest',files={'files':('huge.md',b'x'*(50*1024*1024+1),'text/markdown')})
    assert response.status_code==413,response.text
    from invar_agent.builder import api_router
    monkeypatch.setattr(api_router,'UPLOAD_LIMIT',8)
    response=builder_client.post('/api/v1/builder/ingest',files=[('files',('a.md',b'12345')),('files',('b.md',b'67890'))])
    assert response.status_code==413


def test_ingest_research_picker_prefix_is_normalized_and_validated(builder_client, monkeypatch):
    from invar_agent.builder.ingestion.pipeline import CanonicalResearchIngestionPipeline
    from invar_agent.builder.ingestion.schemas import CanonicalResearchIR

    seen = []

    def ingest(self, request):
        seen.extend(request.vault_run_ids)
        return CanonicalResearchIR(ir_id='prefix_test', source_mode='vault', title='Research reference', vault_run_ids=request.vault_run_ids)

    monkeypatch.setattr(CanonicalResearchIngestionPipeline, 'ingest', ingest)
    response = builder_client.post('/api/v1/builder/ingest', data={'vault_ids': '["@research:real_run"]'})
    assert response.status_code == 200, response.text
    assert seen == ['real_run']
    assert response.json()['source_mode'] == 'vault'
    for item in ('@research:../escape', '@research:', '@research:@research:real_run'):
        response = builder_client.post('/api/v1/builder/ingest', data={'vault_ids': json.dumps([item])})
        assert response.status_code == 422
    assert seen == ['real_run']


def test_sse_builder_events_schema_and_channel_compat(builder_client):
    _,plan=prepare_plan(builder_client,profiles=['hybrid_compute_library','executable_paper_audit'])
    build_id=builder_client.post('/api/v1/builder/execute',json={'plan_id':plan['plan_id']}).json()['build_id']
    wait_build(builder_client,build_id)
    response=builder_client.get('/api/v1/builder/stream',params={'build_id':build_id})
    assert response.status_code==200
    events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert {'goi_stream','accelerator_log','log'}<={item['channel'] for item in events}
    assert all(item['event_version']=='2.0' and item['mode']=='builder' for item in events)
    assert events[-1]['event']=='builder.completed' and len(events[-1]['merkle_root_blake3'])==64
    alias=builder_client.get('/api/v1/builder/stream',params={'build_id':build_id,'channel':'research_log'})
    assert 'search_log' in alias.text and 'builder.completed' in alias.text


def test_builder_api_rejects_empty_unknown_and_unsafe_requests(builder_client):
    client=builder_client
    assert client.post('/api/v1/builder/ingest').status_code==422
    assert client.get('/api/v1/builder/ir/missing').status_code==404
    assert client.get('/api/v1/builder/status/missing').status_code==404
    assert client.post('/api/v1/builder/ingest',data={'vault_ids':'{}'}).status_code==422
    assert client.post('/api/v1/builder/execute',json={'plan_id':'../escape'}).status_code==422
    source=io.BytesIO()
    with zipfile.ZipFile(source,'w') as archive:
        archive.writestr('../escape.tex',PAPER)
    assert client.post('/api/v1/builder/ingest',files={'files':('unsafe.zip',source.getvalue())}).status_code==400


def test_changed_ir_rebuild_forks_immutable_build_and_rejects_stale_plan(builder_client):
    client=builder_client
    ir,plan=prepare_plan(client,profiles=['hybrid_compute_library'])
    original_id=client.post('/api/v1/builder/execute',json={'plan_id':plan['plan_id']}).json()['build_id']
    original=wait_build(client,original_id)
    updated=client.patch('/api/v1/builder/ir/'+ir['ir_id']+'/override',json={'constant_patches':[{'constant_id':ir['constants'][0]['constant_id'],'si_value':.2,'bounds':{'lower':0.,'upper':2.}}]})
    assert updated.status_code==200,updated.text
    assert client.post('/api/v1/builder/execute',json={'plan_id':plan['plan_id']}).status_code==409
    response=client.post('/api/v1/builder/rebuild/'+original_id,json={'updated_ir_id':ir['ir_id']})
    assert response.status_code==202,response.text
    rebuilt=response.json()
    assert rebuilt['build_id']!=original_id and not rebuilt['cached_nodes']
    result=wait_build(client,rebuilt['build_id'])
    assert result['receipt']['canonical_ir_digest']==updated.json()['ir_blake3_digest']
    assert client.get('/api/v1/builder/status/'+original_id).json()['receipt']==original['receipt']
    updated=client.patch('/api/v1/builder/ir/'+ir['ir_id']+'/override',json={'constant_patches':[{'constant_id':ir['constants'][0]['constant_id'],'si_value':.3}]})
    assert updated.status_code==200
    response=client.post('/api/v1/builder/rebuild/'+rebuilt['build_id'],json={'updated_ir_id':ir['ir_id']})
    assert response.status_code==202,response.text
    cached=response.json()
    assert 'slot_custom_viz' in cached['cached_nodes']
    assert {'mod_sdk','mod_tests'} <= set(cached['rebuild_nodes'])
    result=wait_build(client,cached['build_id'])
    assert len(result['fallbacks'])==1 and result['receipt']['canonical_ir_digest']==updated.json()['ir_blake3_digest']


def test_instrument_process_terminates_on_app_shutdown(builder_client,tmp_path,monkeypatch):
    import invar_agent.server as server
    monkeypatch.setattr(server,'_STATE',None)
    app=server.create_app(tmp_path/'shutdown')
    with TestClient(app) as client:
        _,plan=prepare_plan(client,profiles=['interactive_workbench'])
        build_id=client.post('/api/v1/builder/execute',json={'plan_id':plan['plan_id']}).json()['build_id']
        wait_build(client,build_id)
        launched=client.post('/api/v1/builder/instrument/'+build_id+'/launch',json={})
        assert launched.status_code==200,launched.text
        pid=launched.json()['pid']
    with pytest.raises((ProcessLookupError, OSError, SystemError)):
        os.kill(pid,0)
    assert not app.state.builder_service.instruments


def test_ir_rebuild_preserves_last_leaf_prompt_and_exact_body_cone(builder_client):
    client=builder_client
    ir,plan=prepare_plan(client,profiles=['hybrid_compute_library','interactive_workbench'])
    build_id=client.post('/api/v1/builder/execute',json={'plan_id':plan['plan_id']}).json()['build_id']
    wait_build(client,build_id)
    leaf=client.post('/api/v1/builder/rebuild/'+build_id,json={'modified_slot_prompts':{'domain_visualization':'Use accessible purple phase-plane trajectories.'}})
    assert leaf.status_code==202,leaf.text
    wait_build(client,leaf.json()['build_id'])
    updated=client.patch('/api/v1/builder/ir/'+ir['ir_id']+'/override',json={'constant_patches':[{'constant_id':ir['constants'][0]['constant_id'],'si_value':.2}]})
    assert updated.status_code==200,updated.text
    rebuilt=client.post('/api/v1/builder/rebuild/'+build_id,json={'updated_ir_id':ir['ir_id']})
    assert rebuilt.status_code==202,rebuilt.text
    assert set(rebuilt.json()['rebuild_nodes'])=={'kernel_core','mod_sdk','mod_tests','mod_receipt'}
    assert 'slot_custom_viz' in rebuilt.json()['cached_nodes']
    result=wait_build(client,rebuilt.json()['build_id'])
    assert result['status']=='COMPLETED_WITH_FALLBACKS' and len(result['fallbacks'])==1
    assert set(rebuilt.json()['cached_nodes']) <= {event['module_id'] for event in result['events'] if event['event']=='CACHE_HIT_SKIPPED'}


def test_builder_client_uses_current_server_provider_and_key(builder_client,monkeypatch):
    from invar_agent.builder import api_router
    import invar_agent.server as server
    calls=[]
    sentinel=object()
    def factory(**options):
        calls.append(options)
        return sentinel
    monkeypatch.setattr(api_router,'get_llm_client',factory)
    server._STATE.provider='deepseek'
    server._STATE.api_key='test-only-not-a-secret'
    prepare_plan(builder_client,profiles=['hybrid_compute_library'])
    assert len(calls)==1
    assert calls[0]['provider']=='deepseek' and calls[0]['api_key']=='test-only-not-a-secret'

"""Locks additive routing, native symbols and legacy research entry points."""
import inspect

from test_builder_api_sse import builder_client, prepare_plan, wait_build


def test_research_planner_and_vault_routes_unchanged(builder_client,monkeypatch):
    client=builder_client
    for path in ('/api/v1/research/status','/api/v1/research/vault'):
        assert client.get(path).status_code==200
    assert client.get('/api/v1/research/vault/export-all-zip').status_code==404
    response=client.post('/api/v1/research/vault/upload',json={'files':[{'filename':'paper.md','content':'# First order decay\nA verified research source.'}]})
    assert response.status_code==200,response.text
    assert client.get('/api/v1/research/vault/export-all-zip').status_code==200
    response=client.post('/api/v1/research/plan',json={'question':'Study first-order decay'})
    assert response.status_code==200 and 'blueprint' in response.json()
    response=client.post('/api/v1/research/plan-prompt',json={'question':'Study first-order decay'})
    assert response.status_code==400,response.text
    import invar_agent.server as server
    manager=server._STATE.ensure_research_manager()
    monkeypatch.setattr(manager,'approve',lambda:True)
    response=client.post('/api/v1/research/run',json={'question':'Study first-order decay'})
    assert response.status_code==200 and response.json()['started'],response.text
    paths=client.get('/openapi.json').json()['paths']
    assert '/api/v1/research/run' in paths and '/api/v1/research/stream' in paths


def test_native_and_legacy_research_entrypoints_retained():
    from invar_agent import _native
    from invar_agent.builder import research_bridge
    from invar_agent.research.state_machine import ResearchStateMachine
    for name in ('PyGoiSolver','PyHinEngine','PyDeterministicHealer','PyHisRegister','UASTKind','UASTGraph','HinEngine',
                 'GoIWavefrontScheduler','GoIProofNet','GoIWavefrontSolverNative','GraphPolyglotMaterializer',
                 'ASTRewritePatch','PreWriteHealer','PyHolographicHin','PyAdjointHin','csel_egraph_optimize',
                 'desugar_math_to_csel','resolve_symbol_fuzzy','validate_builder_uast','heal_builder_tier0'):
        assert hasattr(_native,name),name
    for name in ('classify_software_archetype','resolve_relevant_research_runs','compile_research_build_packet',
                 'render_packet_prompt_block','generate_research_parity_test_code','write_research_build_artifacts',
                 'run_research_parity_tests','bind_research_context'):
        assert callable(getattr(research_bridge,name))
    assert list(inspect.signature(ResearchStateMachine.run).parameters)==['self','blueprint']


def test_standard_builder_does_not_activate_shbt_federation(builder_client,monkeypatch):
    import builtins
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert 'shbt' not in name.lower() and 'federation' not in name.lower(),name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    _,plan=prepare_plan(builder_client,profiles=['hybrid_compute_library'])
    build_id=builder_client.post('/api/v1/builder/execute',json={'plan_id':plan['plan_id']}).json()['build_id']
    assert wait_build(builder_client,build_id)['total_llm_tokens_used']==0


def test_legacy_server_still_runs_without_optional_builder_extras(builder_client,monkeypatch):
    import builtins
    import invar_agent.server as server
    from fastapi.testclient import TestClient
    original=builtins.__import__
    def missing_extra(name,*args,**kwargs):
        if name=='invar_agent.builder.api_router':
            raise ModuleNotFoundError("No module named 'sympy'",name='sympy')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',missing_extra)
    app=server.create_app()
    monkeypatch.setattr(builtins,'__import__',original)
    with TestClient(app) as client:
        assert client.get('/api/v1/research/status').status_code==200
        assert client.post('/api/v1/builder/plan',json={}).status_code==503


def test_builder_execute_preserves_research_event_replay(builder_client):
    _,plan=prepare_plan(builder_client,profiles=['hybrid_compute_library'])
    broadcaster=builder_client.app.state.builder_service.server.research_broadcaster
    broadcaster.publish('search_log','research_marker',{'mode':'research','message':'Preserve replay'},run_uid='res_marker_1')
    response=builder_client.post('/api/v1/builder/execute',json={'plan_id':plan['plan_id']})
    assert response.status_code==202,response.text
    subscription=broadcaster.subscribe()
    events=[]
    try:
        while not subscription.empty():
            events.append(subscription.get_nowait())
    finally:
        broadcaster.unsubscribe(subscription)
    assert any(event['event']=='research_marker' and event['mode']=='research' for event in events)
    wait_build(builder_client,response.json()['build_id'])


def test_builder_action_controls_use_scoped_console_styles(builder_client):
    html=builder_client.get('/').text
    assert '#view-builder button, #builder-files::file-selector-button' in html
    assert '#view-builder button:hover:not(:disabled)' in html
    assert '#view-builder button:disabled' in html
    assert '#view-builder button:focus-visible' in html
    for name in ('builder-files','builder-ingest','builder-plan','builder-execute','builder-launch'):
        assert 'id="'+name+'"' in html

"""Additive Builder lifecycle API and loopback-only instrument supervisor."""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from invar_agent.config import Tier
from invar_agent.llm.clients import get_llm_client
from invar_agent.llm.errors import redact_secrets
from .circuit_breaker import BuildBreakerBudget
from .his_cache import HISCache
from .ingestion.archive_guard import IngestionError
from .ingestion.pipeline import CanonicalResearchIngestionPipeline, apply_ir_override
from .ingestion.schemas import CanonicalResearchIR, IngestionRequest, IRReviewOverride, StrictModel
from .planner import BuilderPlanner
from .state_machine import BuildRequest, BuilderStateMachine, _FORMAL_LOCK, _atomic_json, _link_directory
from .uast_models import BuilderArchitectureSpec, BuilderProfile, BuilderUAST, LLMExtensionSlotRequest
from .wavefront_scheduler import WavefrontScheduler


UPLOAD_LIMIT = 50 * 1024 * 1024


class BuilderUploadLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/api/v1/builder/ingest':
            return await self.app(scope, receive, send)
        received = 0
        limit = UPLOAD_LIMIT + 1024 * 1024
        headers = dict(scope.get('headers', []))
        if int(headers.get(b'content-length', b'0')) > limit:
            return await JSONResponse({'detail': 'Upload exceeds 50 MB'}, 413)(scope, receive, send)

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > limit:
                raise HTTPException(413, 'Upload exceeds 50 MB')
            return message
        await self.app(scope, bounded_receive, send)


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value):
        raise HTTPException(422, 'Invalid identifier')
    return value


class BuilderPlanRequest(StrictModel):
    ir_id: str
    build_goal: str = ''
    prompt: str = ''
    project_name: str | None = None
    profiles: list[BuilderProfile] | None = None
    enable_cpp_cabi: bool = True
    enable_adjoint_jacobian: bool = True
    leaf_slots: list[str | LLMExtensionSlotRequest] = Field(default_factory=list)


class BuilderExecuteRequest(StrictModel):
    plan_id: str
    output_workspace_name: str | None = None
    budget_override: dict[str, int | float] | None = None


class IncrementalRebuildRequest(StrictModel):
    updated_ir_id: str | None = None
    updated_plan_id: str | None = None
    modified_slot_prompts: dict[str, str] = Field(default_factory=dict)


class InstrumentLaunch(StrictModel):
    profile: str = 'interactive_workbench'
    port: int | None = Field(default=None, ge=1024, le=65535)


class BuilderService:
    def __init__(self, state):
        self.server = state
        self.root = state.workspace_root / '.invar' / 'builder'
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.instruments = {}
        self.lock = threading.RLock()

    def client(self):
        if not self.server.api_key:
            return None
        return get_llm_client(provider=self.server.provider, tier=Tier.FAST,
            api_key=self.server.api_key, override=self.server.effective_tier_overrides().get(Tier.FAST.value))

    def read(self, kind, key):
        path = self.root / kind / (identifier(key) + '.json')
        if not path.is_file():
            raise HTTPException(404, kind + ' not found')
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, kind, key, value):
        _atomic_json(self.root / kind / (identifier(key) + '.json'), value)

    def ir(self, key):
        return CanonicalResearchIR.model_validate(self.read('ir', key))

    def bind_ir_artifacts(self, uast, ir):
        for node in uast.nodes:
            if node.kind == 'deterministic_module' and node.module_role in {'python_sdk','pytest_z3_suite'}:
                node.emitter_config['canonical_ir_digest'] = ir.ir_blake3_digest
        return uast

    def emit(self, channel, event, data):
        from ..research.broadcaster import GLOBAL_BUILDER_SCOPE
        data = {key: value for key, value in data.items() if key not in {'event', 'channel', 'seq', 'run_uid', 'scope'}}
        build_uid = f"builder:{data['build_id']}" if data.get('build_id') else GLOBAL_BUILDER_SCOPE
        self.server.research_broadcaster.publish(channel, event, {'event_version':'2.0', 'mode':'builder', **data}, run_uid=build_uid)
        if channel == 'log':
            self.server.research_broadcaster.publish('search_log', event, {'event_version':'2.0', 'mode':'builder', **data}, run_uid=build_uid)

    def progress(self, channel, payload):
        original = payload['event']
        job = self.jobs.get(payload.get('build_id'))
        if job and getattr(job['machine'], 'state', None):
            machine = job['machine']
            if machine._plan and not job.get('restored_fallbacks'):
                known = {record['module_id'] for record in machine.state.fallbacks}
                slots = {node.node_id:node for node in machine.uast.nodes if node.kind=='llm_extension_slot'}
                for record in job.get('inherited_fallbacks',[]):
                    module = record['module_id']
                    if module in machine._plan.cached_nodes and module not in known and module in slots and slots[module].frozen_his_digest == record['frozen_his_digest']:
                        machine.state.fallbacks.append(record.copy())
                job['restored_fallbacks'] = True
            committed = machine.state.committed_waves
            for index in range(job.get('committed_count',0),len(committed)):
                self.emit('goi_stream','builder.wavefront_committed',{'build_id':payload['build_id'],
                    'wavefront_idx':index,'committed_modules':committed[index]})
            job['committed_count'] = len(committed)
            payload = {**payload,'wavefront_idx':len(committed),'elapsed_ms':round((time.monotonic()-machine._started)*1000,2),
                'status':payload.get('status',payload.get('stage'))}
        if original in {'builder_completed','builder_failed'}:
            return
        event = {'builder_stage':'builder.stage_transition', 'healing_decision':'builder.healing_decision',
            'DEGRADED_DETERMINISTIC_FALLBACK':'builder.fallback_pending', 'builder_completed':'builder.completed',
            'builder_failed':'builder.failed'}.get(original, original)
        if original == 'healing_decision':
            channel = 'accelerator_log'
            payload = {**payload,'round_idx':payload.get('round_index')}
        self.emit(channel, event, payload)
        if payload.get('stage') == 'SMT_Concolic_Verify':
            self.emit('accelerator_log', 'builder.verification_gate', {**payload, 'status':'VERIFYING',
                'hin_ok':None,'z3_ok':None,'cabi_ok':None,'pytest_exit_code':None,'progress_metric':None})

    def build_root(self, build_id):
        return self.server.workspace_root / '.invar' / 'builds' / identifier(build_id)

    def status(self, build_id):
        identifier(build_id)
        with self.lock:
            job = self.jobs.get(build_id)
            if job and job['thread'].is_alive():
                machine = job['machine']
                value = machine.state.model_dump(mode='json') if getattr(machine, 'state', None) else {'build_id':build_id, 'status':'RUNNING'}
                value['current_wavefront'] = len(value.get('committed_waves', []))
                value['module_statuses'] = {item.get('module_id'):item.get('stage') for item in value.get('events',[]) if item.get('module_id')}
                return {**value, 'status':'RUNNING'}
        return self.read('status', build_id)

    def start(self, plan, build_id=None, budget_override=None):
        build_id = identifier(build_id or uuid4().hex)
        with self.lock:
            if build_id in self.jobs and self.jobs[build_id]['thread'].is_alive():
                raise HTTPException(409, 'Build is already running')
            overrides = budget_override or {}
            allowed = {'max_tier0_per_slot':2, 'max_tier1_per_slot':2, 'max_tier2_per_slot':2,
                'max_global_tier2_repairs':6, 'max_global_repair_tokens':24000, 'max_wall_clock_sec':300}
            if any(key not in allowed or value < 0 or value > allowed[key] for key,value in overrides.items()):
                raise HTTPException(422, 'Budget overrides may only lower builder limits')
            machine = BuilderStateMachine(self.server.workspace_root, llm_client=self.client(), progress_cb=self.progress,
                budget=BuildBreakerBudget(**overrides))
            ir = self.ir(plan['ir_id'])
            if ir.ir_blake3_digest != plan['ir_blake3_digest']:
                raise HTTPException(409,'Research IR changed; create a new verified plan')
            request = BuildRequest(build_id=build_id, ir=ir, uast=BuilderUAST.model_validate(plan['uast']))
            self.save('build_plans', build_id, plan)
            self.save('status', build_id, {'build_id':build_id, 'status':'RUNNING'})

            def execute():
                try:
                    self.emit('goi_stream','builder.wavefront_plan',{'build_id':build_id,**plan['wavefront_schedule']})
                    result = asyncio.run(machine.run(request))
                    value = result.model_dump(mode='json')
                    value['current_wavefront'] = len(result.committed_waves)
                    value['module_statuses'] = {module:'COMMITTED' for wave in result.committed_waves for module in wave}
                    for record in result.fallbacks:
                        value['module_statuses'][record['module_id']] = 'DEGRADED_DETERMINISTIC_FALLBACK'
                        self.emit('log','builder.fallback_substituted',{'build_id':build_id,**record,
                            'fault_fingerprint':record['fault_fingerprint_digest'],'status':'DEGRADED_DETERMINISTIC_FALLBACK'})
                    for module in value['module_statuses']:
                        self.emit('accelerator_log','builder.verification_gate',{'build_id':build_id,'module_id':module,
                            'status':value['module_statuses'][module],'hin_ok':None,'z3_ok':None,
                            'cabi_ok':(True if (result.cabi_compiled and module == 'mod_sdk') else None),
                            'pytest_exit_code':(result.pytest_exit_code if module == 'mod_tests' else None),
                            'progress_metric':{'failed_test_count':result.pytest_failed}})
                    self.save('status', build_id, value)
                    self.emit('log', 'builder.completed' if not result.error else 'builder.failed', {
                        'build_id':build_id, 'status':result.status, 'merkle_root_blake3':result.receipt.get('merkle_root_blake3'),
                        'fallbacks_count':len(result.fallbacks), 'total_llm_tokens':result.total_llm_tokens_used})
                except Exception as error:
                    self.save('status', build_id, {'build_id':build_id, 'status':'ABORTED_CRITICAL', 'error':redact_secrets(str(error))})
                    self.emit('log', 'builder.failed', {'build_id':build_id, 'status':'ABORTED_CRITICAL'})
            thread = threading.Thread(target=execute, name='builder-'+build_id, daemon=True)
            self.jobs[build_id] = {'machine':machine, 'thread':thread, 'inherited_fallbacks':plan.get('inherited_fallbacks',[]),
                'start_seq':self.server.research_broadcaster.seq}
            thread.start()
        return build_id

    def stop_instrument(self, build_id):
        with self.lock:
            item = self.instruments.pop(build_id, None)
        if item:
            process = item['process']
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
            item['log'].close()

    def close(self):
        for build_id in list(self.instruments):
            self.stop_instrument(build_id)


def create_builder_router(state):
    service = BuilderService(state)
    router = APIRouter(prefix='/api/v1/builder', tags=['builder'])

    @router.post('/ingest')
    async def ingest(files: list[UploadFile] = File(default=[]), vault_ids: str = Form(default='[]'),
                     spec_text: str | None = Form(default=None),
                     allow_llm_disambiguation: bool = Form(default=True)):
        stage = service.root / 'uploads' / uuid4().hex
        stage.mkdir(parents=True)
        paths = []
        total = 0
        try:
            ids = json.loads(vault_ids)
            if not isinstance(ids,list) or not all(isinstance(item,str) for item in ids):
                raise HTTPException(422, 'vault_ids must be a JSON list of IDs')
            ids = [item.removeprefix('@research:') for item in ids]
            for item in ids:
                identifier(item)
            if spec_text and spec_text.strip():
                spec_path = stage / 'inline_spec.md'
                spec_path.write_text(spec_text, encoding='utf-8')
                paths.append(str(spec_path))
            for index, upload in enumerate(files):
                name = Path(upload.filename or 'upload').name
                path = stage / (str(index)+'_'+name)
                with path.open('wb') as output:
                    while chunk := await upload.read(1024*1024):
                        total += len(chunk)
                        if total > UPLOAD_LIMIT:
                            raise HTTPException(413, 'Upload exceeds 50 MB')
                        output.write(chunk)
                paths.append(str(path))
            if not paths and not ids:
                raise HTTPException(422, 'Supply uploads, spec_text, or vault_ids')
            pipeline = CanonicalResearchIngestionPipeline(state.workspace_root,
                llm_client=service.client() if allow_llm_disambiguation else None)
            def extract():
                return pipeline.ingest(IngestionRequest(upload_paths=paths,vault_run_ids=ids))
            ir = await run_in_threadpool(extract)
            service.save('ir',ir.ir_id,ir.model_dump(mode='json'))
            service.emit('log','builder.ingest_complete',{'build_id':None,'ir_id':ir.ir_id,'source_mode':ir.source_mode,
                'equations_count':len(ir.equations),'constants_count':len(ir.constants),
                'disambiguation_flags_count':len(ir.disambiguation_flags),'llm_tokens':ir.llm_disambiguation_tokens_used})
            return ir
        except IngestionError as error:
            shutil.rmtree(stage,ignore_errors=True)
            raise HTTPException(400 if error.code.startswith('SANDBOX_ZIP') or 'TYPE' in error.code else 422,
                {'code':error.code,'message':str(error),'details':error.details}) from error
        except (ValueError,TypeError) as error:
            shutil.rmtree(stage,ignore_errors=True)
            raise HTTPException(422,str(error)) from error
        except TimeoutError as error:
            shutil.rmtree(stage,ignore_errors=True)
            raise HTTPException(422,{'code':'SOLVER_TIMEOUT','message':str(error)}) from error
        except RuntimeError as error:
            if getattr(error, 'code', None) == 'SOLVER_BUSY':
                shutil.rmtree(stage,ignore_errors=True)
                raise HTTPException(503,{'code':'SOLVER_BUSY','message':str(error)}) from error
            raise
        except HTTPException:
            shutil.rmtree(stage,ignore_errors=True)
            raise
        finally:
            for upload in files:
                await upload.close()

    @router.get('/ir/{ir_id}')
    def preview(ir_id: str):
        return service.ir(ir_id)

    @router.patch('/ir/{ir_id}/override')
    def override(ir_id: str, request: IRReviewOverride):
        try:
            ir = apply_ir_override(service.ir(ir_id),request)
        except IngestionError as error:
            raise HTTPException(422,{'code':error.code,'message':str(error),'details':error.details}) from error
        except TimeoutError as error:
            raise HTTPException(422,{'code':'SOLVER_TIMEOUT','message':str(error)}) from error
        except RuntimeError as error:
            if getattr(error, 'code', None) == 'SOLVER_BUSY':
                raise HTTPException(503,{'code':'SOLVER_BUSY','message':str(error)}) from error
            raise
        service.save('ir',ir_id,ir.model_dump(mode='json'))
        return ir

    @router.post('/plan')
    def plan(request: BuilderPlanRequest):
        try:
            ir = service.ir(request.ir_id)
            planner = BuilderPlanner(service.client())
            spec = planner.translate_intent(ir,request.prompt or request.build_goal,profiles=request.profiles,project_name=request.project_name)
            slots = [LLMExtensionSlotRequest(slot_id=item,intent_summary=request.prompt or request.build_goal or 'Implement the requested scientific leaf contract') if isinstance(item,str) else item for item in request.leaf_slots]
            spec = BuilderArchitectureSpec.model_validate({**spec.model_dump(mode='json'), 'enable_cpp_cabi':request.enable_cpp_cabi,
                'enable_adjoint_jacobian':request.enable_adjoint_jacobian,
                'llm_extension_slots':[item.model_dump(mode='json') for item in (slots or spec.llm_extension_slots)]})
            uast = service.bind_ir_artifacts(planner.compile(ir,spec),ir)
            schedule = WavefrontScheduler().plan(uast)
            result = {'plan_id':uuid4().hex,'ir_id':ir.ir_id,'ir_blake3_digest':ir.ir_blake3_digest,'arch_spec':spec.model_dump(mode='json'),
                'uast':uast.model_dump(mode='json'),'wavefront_schedule':schedule.model_dump(mode='json'),
                'spectral_radius':schedule.spectral_radius,'wavefronts':schedule.wavefronts,
                'frozen_his_digests':{node.node_id:getattr(node,'frozen_his_digest',getattr(node,'his_digest',None)) for node in uast.nodes}}
            service.save('plans',result['plan_id'],result)
            service.emit('goi_stream','builder.wavefront_plan',{'build_id':None,**schedule.model_dump(mode='json')})
            return result
        except ValueError as error:
            raise HTTPException(422,str(error)) from error

    @router.post('/execute',status_code=202)
    def execute(request: BuilderExecuteRequest):
        if request.output_workspace_name:
            identifier(request.output_workspace_name)
        build_id = service.start(service.read('plans',request.plan_id),budget_override=request.budget_override)
        return {'build_id':build_id,'status':'RUNNING','sse_channels':['goi_stream','accelerator_log','log','search_log']}

    @router.get('/status/{build_id}')
    def status(build_id: str):
        return service.status(build_id)

    @router.post('/rebuild/{build_id}',status_code=202)
    def rebuild(build_id: str, request: IncrementalRebuildRequest):
        if service.status(build_id)['status']=='RUNNING':
            raise HTTPException(409,'Build is already running')
        value = service.read('plans',request.updated_plan_id) if request.updated_plan_id else service.read('build_plans',build_id)
        if request.updated_ir_id:
            prompts = {node['node_id']:node['intent_summary'] for node in value['uast']['nodes']
                       if node['kind']=='llm_extension_slot'} if not request.updated_plan_id else {}
            ir = service.ir(request.updated_ir_id)
            value['ir_id'] = ir.ir_id
            value['ir_blake3_digest'] = ir.ir_blake3_digest
            value['uast'] = service.bind_ir_artifacts(BuilderPlanner().compile(ir,BuilderArchitectureSpec.model_validate(value['arch_spec'])),ir).model_dump(mode='json')
            for node in value['uast']['nodes']:
                if node['kind']=='llm_extension_slot' and node['node_id'] in prompts:
                    node['intent_summary'] = prompts[node['node_id']]
        uast = BuilderUAST.model_validate(value['uast'])
        for key,prompt in request.modified_slot_prompts.items():
            matches = [node for node in uast.nodes if node.kind=='llm_extension_slot' and key in {node.node_id,node.slot_kind}]
            if len(matches)!=1:
                raise HTTPException(422,'Unknown or ambiguous leaf slot: '+key)
            matches[0].intent_summary = prompt
        value['uast'] = uast.model_dump(mode='json')
        old_root = service.build_root(build_id)
        previous = HISCache(state.workspace_root,build_id,old_root/'workspace').load()
        schedule = WavefrontScheduler(old_root/'workspace').plan(uast,previous)
        value['wavefront_schedule'] = schedule.model_dump(mode='json')
        prior_ir = json.loads((old_root/'canonical_research_ir.json').read_text(encoding="utf-8"))
        if service.ir(value['ir_id']).ir_blake3_digest != prior_ir['ir_blake3_digest']:
            value['inherited_fallbacks'] = service.status(build_id).get('fallbacks',[])
            next_id = uuid4().hex
            new_root = service.build_root(next_id)
            new_root.mkdir(parents=True)
            shutil.copytree((old_root/'workspace').resolve(),new_root/'seed')
            _link_directory(new_root/'workspace', new_root/'seed')
            if previous:
                previous.build_id=next_id; previous.artifact_root=str((new_root/'workspace').resolve())
                HISCache(state.workspace_root,next_id,new_root/'workspace').save(previous)
            build_id=next_id
        service.emit('goi_stream','builder.wavefront_plan',{'build_id':build_id,**schedule.model_dump(mode='json')})
        service.stop_instrument(build_id)
        service.start(value,build_id)
        return {'build_id':build_id,**schedule.model_dump(mode='json')}

    @router.get('/export/{build_id}.zip')
    def export(build_id: str):
        from .export_utils import ExportBlockedError, build_verified_export_zip

        status = service.status(build_id)
        root = service.build_root(build_id)
        workspace = (root/'workspace').resolve()
        if not workspace.is_dir() or not workspace.is_relative_to(root.resolve()):
            if status.get('status') not in {'COMPLETED', 'COMPLETED_WITH_FALLBACKS'}:
                raise HTTPException(
                    409,
                    f"Cannot export build with status {status.get('status')!r}; a completed build is required",
                )
            raise HTTPException(404,'Committed workspace not found')
        cache_manifest = HISCache(state.workspace_root, build_id, workspace).load()
        archive = root / ('export-'+uuid4().hex+'.zip')
        try:
            build_verified_export_zip(
                workspace,
                archive,
                build_status=status.get('status'),
                cache_manifest=cache_manifest,
            )
        except ExportBlockedError as exc:
            raise HTTPException(exc.status_code, exc.message) from exc
        def chunks():
            try:
                with archive.open('rb') as stream:
                    while chunk:=stream.read(1024*1024):
                        yield chunk
            finally:
                archive.unlink(missing_ok=True)
        return StreamingResponse(chunks(),media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="{build_id}.zip"'})

    @router.post('/instrument/{build_id}/launch')
    def launch(build_id: str, request: InstrumentLaunch = InstrumentLaunch()):
        if request.profile!='interactive_workbench':
            raise HTTPException(422,'Only interactive_workbench can be launched')
        if service.status(build_id)['status'] not in {'COMPLETED','COMPLETED_WITH_FALLBACKS'}:
            raise HTTPException(409,'A completed build is required')
        with service.lock:
            existing = service.instruments.get(build_id)
            if existing and existing['process'].poll() is None:
                return {'url':existing['url'],'pid':existing['process'].pid,'health_ok':True}
            workspace=(service.build_root(build_id)/'workspace').resolve()
            plan=service.read('build_plans',build_id)
            package=plan['uast']['project_id']
            if not (workspace/'src'/package/'workbench/app.py').is_file():
                raise HTTPException(422,'Workbench profile was not emitted')
            listener=socket.socket()
            try:
                listener.bind(('127.0.0.1',request.port or 0)); listener.listen(128)
            except OSError as error:
                listener.close()
                raise HTTPException(409,'Loopback port unavailable') from error
            port=listener.getsockname()[1]
            log=(service.build_root(build_id)/'instrument.log').open('ab')
            environment={key:os.environ[key] for key in ('PATH','SYSTEMROOT','WINDIR') if key in os.environ}
            environment.update(PYTHONPATH=str(workspace/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            try:
                if os.name == 'nt':
                    listener.close()
                    process=subprocess.Popen([sys.executable,'-m','uvicorn',package+'.workbench.app:app','--host','127.0.0.1','--port',str(port)],
                        cwd=workspace,env=environment,stdout=log,stderr=log)
                else:
                    process=subprocess.Popen([sys.executable,'-m','uvicorn',package+'.workbench.app:app','--fd',str(listener.fileno())],
                        pass_fds=(listener.fileno(),),cwd=workspace,env=environment,stdout=log,stderr=log)
            except Exception:
                log.close(); raise
            finally:
                listener.close()
            url=f'http://127.0.0.1:{port}'
            service.instruments[build_id]={'process':process,'url':url,'log':log}
        deadline=time.monotonic()+15
        while time.monotonic()<deadline and process.poll() is None:
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.2) as connection:
                    connection.sendall(b'GET /api/schema HTTP/1.0\r\nHost: localhost\r\n\r\n')
                    if b'200' in connection.recv(128).split(b'\r\n')[0]:
                        return {'url':url,'pid':process.pid,'health_ok':True}
            except OSError:
                pass
            time.sleep(.05)
        service.stop_instrument(build_id)
        raise HTTPException(503,'Workbench failed to start; see instrument.log')

    @router.delete('/instrument/{build_id}/launch')
    @router.delete('/instrument/{build_id}')
    @router.post('/instrument/{build_id}/teardown')
    def stop(build_id: str):
        identifier(build_id); service.stop_instrument(build_id)
        return {'build_id':build_id,'status':'STOPPED'}

    @router.get('/instrument/{build_id}')
    def inspect_instrument(build_id: str):
        status=service.status(build_id)
        plan=service.read('build_plans',build_id)
        item=service.instruments.get(build_id)
        running=bool(item and item['process'].poll() is None)
        return {'build_id':build_id,'running':running,'url':item['url'] if running else None,'pid':item['process'].pid if running else None,
            'project_id':plan['uast']['project_id'],'profiles':plan['arch_spec']['profiles'],
            'python_fallback_active':status.get('python_fallback_active',False)}

    @router.get('/stream')
    def stream(build_id: str | None = None, channel: str | None = None):
        if build_id:
            identifier(build_id)
        channel='search_log' if channel=='research_log' else channel
        broadcaster=state.research_broadcaster
        terminal_statuses = {'COMPLETED', 'COMPLETED_WITH_FALLBACKS', 'ABORTED_CRITICAL', 'FAILED'}
        build_uid = f"builder:{build_id}" if build_id else None
        subscription=broadcaster.subscribe(run_uid=build_uid)

        def _synth_terminal(st: dict) -> dict:
            status_val = st.get('status', 'FAILED')
            evt = 'builder.completed' if status_val in {'COMPLETED', 'COMPLETED_WITH_FALLBACKS'} else 'builder.failed'
            return {
                'channel': channel or 'log',
                'event': evt,
                'event_version': '2.0',
                'mode': 'builder',
                'build_id': build_id,
                'status': status_val,
                'seq': broadcaster.seq,
            }

        def events():
            try:
                saw_terminal = False
                while True:
                    try:
                        payload = subscription.get_nowait()
                    except queue.Empty:
                        break
                    if payload.get('mode')!='builder' or build_id and payload.get('build_id')!=build_id:
                        continue
                    job = service.jobs.get(build_id) if build_id else None
                    if job and payload.get('seq',0) <= job['start_seq']:
                        continue
                    if channel and payload['channel']!=channel:
                        continue
                    yield 'data: '+json.dumps(payload,default=str)+'\n\n'
                    if build_id and payload['event'] in {'builder.completed','builder.failed'}:
                        saw_terminal = True
                        break
                if saw_terminal:
                    return
                if build_id:
                    try:
                        st = service.status(build_id)
                    except HTTPException:
                        st = None
                    if st and st.get('status') in terminal_statuses:
                        yield 'data: '+json.dumps(_synth_terminal(st),default=str)+'\n\n'
                        return
                while True:
                    try:
                        payload=subscription.get(timeout=1)
                    except queue.Empty:
                        if build_id:
                            try:
                                st = service.status(build_id)
                            except HTTPException:
                                st = None
                            if st and st.get('status') in terminal_statuses:
                                yield 'data: '+json.dumps(_synth_terminal(st),default=str)+'\n\n'
                                break
                        yield ': keep-alive\n\n'; continue
                    if payload.get('mode')!='builder' or build_id and payload.get('build_id')!=build_id:
                        continue
                    job = service.jobs.get(build_id) if build_id else None
                    if job and payload.get('seq',0) <= job['start_seq']:
                        continue
                    if channel and payload['channel']!=channel:
                        continue
                    yield 'data: '+json.dumps(payload,default=str)+'\n\n'
                    if build_id and payload['event'] in {'builder.completed','builder.failed'}:
                        break
            finally:
                broadcaster.unsubscribe(subscription)
        return StreamingResponse(events(),media_type='text/event-stream')

    return router,service

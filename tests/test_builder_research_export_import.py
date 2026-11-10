"""Research-to-Builder bridge, user research uploads and zip export/import.

Fully hermetic: no live LLM calls (LLM clients are mocked or absent), all
endpoints driven through FastAPI's TestClient, parity tests executed by a real
local pytest subprocess.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from invar_agent import server
from invar_agent.builder.research_bridge import (
    ARCHETYPES,
    PARITY_TEST_RELPATH,
    PROVENANCE_FILENAME,
    bind_research_context,
    classify_software_archetype,
    compile_research_build_packet,
    resolve_relevant_research_runs,
    run_research_parity_tests,
    write_research_build_artifacts,
)
from invar_agent.context_compactor import build_research_vault_context
from invar_agent.copilot.agent import CopilotAgent
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.schemas.research import (
    InvariantConstraint,
    ResearchBlueprint,
    VerificationContract,
)

RUN_ID = "newton_cooling_probe"

KERNEL = '''
import math

K_COOL = 0.05
T_AMB = 300.0


def simulation_step(buf, dt):
    t = buf.data[0]
    buf.data[0] = t + dt * (-K_COOL * (t - T_AMB))
    buf.data[1] = buf.data[1] + dt
'''

BUILT_SIMULATOR = KERNEL + '''

RESEARCH_CONSTANTS = {"K_COOL": 0.05, "T_AMB": 300.0}
'''

LATEX_MD = r"""# Tungsten Carbide Cooling Study

Newton's law of cooling for the probe shell:
<math>\frac{dT}{dt} = -k (T - T_{\text{amb}})</math>

Thermal conductivity kappa = 175 W/(m*K) and density rho = 15.6 g/cm^3.

| name | expected | unit |
|---|---|---|
| steady_state_temp | 300.0 | K |
| half_life_s | 13.86 | s |
"""

COOLING_MODEL = '''
K_COOL = 0.05
T_AMB = 300.0


def simulation_step(buf, dt):
    buf.data[0] = buf.data[0] + dt * (-K_COOL * (buf.data[0] - T_AMB))
'''

_LLM_ENVS = ("GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_API",
             "OPENROUTER_API_KEY", "INVAR_API_KEY", "INVAR_LLM_PROVIDER")


def _final_temp(steps: int = 100, dt: float = 0.1) -> float:
    t = 400.0
    for _ in range(steps):
        t = t + dt * (-0.05 * (t - 300.0))
    return t


def _register_verified_run(ws: Path, run_id: str = RUN_ID,
                           title: str = "Newton cooling of a reentry probe") -> None:
    bp = ResearchBlueprint(
        blueprint_id=run_id, title=title,
        falsifiable_hypothesis="The probe relaxes to ambient temperature.",
        search_queries=["newton cooling"], code_template=KERNEL, unified_module_code=KERNEL,
        contracts=[VerificationContract(
            contract_id="c1_cooling", target_backend="cpp",
            invariants=[
                InvariantConstraint(variable_name="T", assertion="state[0] >= 300.0"),
                InvariantConstraint(variable_name="T", assertion="state[0] <= 400.0"),
            ],
            initial_state=[400.0, 0.0], dt=0.1, steps=100)],
        variable_map={"final_temp_k": 0},
        extracted_constants={"k_cool": 0.05, "t_amb": 300.0})
    ResearchVaultManager(ws).register_run(
        run_id=run_id, title=bp.title, blueprint=bp, report_md="# Newton cooling report",
        status="VERIFIED", telemetry={"obs_final_temp_k": _final_temp()})


@pytest.fixture()
def client(tmp_path, monkeypatch):
    for var in _LLM_ENVS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("INVAR_CONFIG_PATH", str(tmp_path / "invar-config.json"))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(server, "_STATE", None)
    return TestClient(server.create_app(str(ws))), ws


def _sse_events(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


# ---------------------------------------------------------------------------


def test_archetype_and_research_resolution(tmp_path):
    assert set(ARCHETYPES) == {
        "SCIENTIFIC_SDK_LIBRARY", "CLI_SIMULATOR_SWEEP", "INTERACTIVE_WEB_SIMULATOR",
        "INVERSE_DESIGN_OPTIMIZER", "MULTI_SOLVER_BENCHMARK", "TELEMETRY_DIGITAL_TWIN",
        "STANDALONE_NATIVE_CRATE"}
    for spec in ARCHETYPES.values():
        assert spec.recommended_file_layout and spec.architecture_guidelines
        assert spec.verification_requirements

    cases = {
        "Build a command-line tool with a parameter sweep over entry angles":
            "CLI_SIMULATOR_SWEEP",
        "Build an interactive web dashboard with sliders to explore the model":
            "INTERACTIVE_WEB_SIMULATOR",
        "Optimize the shield thickness to minimize mass (inverse design)":
            "INVERSE_DESIGN_OPTIMIZER",
        "Benchmark RK4 versus symplectic Verlet solvers and compare accuracy":
            "MULTI_SOLVER_BENCHMARK",
        "Build a digital twin that ingests live telemetry streams":
            "TELEMETRY_DIGITAL_TWIN",
        "Package the kernel as a standalone Rust crate with no Python":
            "STANDALONE_NATIVE_CRATE",
        "Build a Python library for the model": "SCIENTIFIC_SDK_LIBRARY",
    }
    for prompt, expected in cases.items():
        assert classify_software_archetype(prompt).archetype_id == expected, prompt

    ws = tmp_path / "ws"
    ws.mkdir()
    _register_verified_run(ws)
    _register_verified_run(ws, "lorenz_attractor_chaos", title="Lorenz attractor chaos study")

    explicit = resolve_relevant_research_runs(ws, f"Build a CLI for @research:{RUN_ID}")
    assert [r["run_id"] for r in explicit] == [RUN_ID]
    assert explicit[0]["resolution_tier"] == "explicit"
    attached = resolve_relevant_research_runs(ws, "Build a library", [RUN_ID])
    assert [r["run_id"] for r in attached] == [RUN_ID]

    keyword = resolve_relevant_research_runs(ws, "Build a probe cooling simulator for newton cooling")
    assert keyword and keyword[0]["run_id"] == RUN_ID
    assert keyword[0]["resolution_tier"] == "keyword"

    implicit = resolve_relevant_research_runs(ws, "Turn this into an SDK from our completed research")
    assert len(implicit) == 1 and implicit[0]["resolution_tier"] == "implicit"

    # Unrelated greenfield guard: no research is forced into unrelated builds.
    assert resolve_relevant_research_runs(ws, "Build a markdown blog generator with tags") == []
    assert bind_research_context(ws, "Build a todo list REST API") is None


def test_builder_direct_and_plan_research_bridge(client, monkeypatch):
    tc, ws = client
    _register_verified_run(ws)

    packet = compile_research_build_packet(
        f"Build a CLI parameter sweep for @research:{RUN_ID}",
        resolve_relevant_research_runs(ws, f"@research:{RUN_ID}"), workspace_dir=ws)
    assert packet.archetype.archetype_id == "CLI_SIMULATOR_SWEEP"
    assert packet.run_ids == [RUN_ID]
    block = packet.prompt_block
    assert block.startswith("<research_to_software_specification>")
    assert block.rstrip().endswith("</research_to_software_specification>")
    assert "K_COOL" in block and "simulation_step(buf, dt)" in block
    assert PARITY_TEST_RELPATH in block
    names = {c.python_name: c.value for c in packet.constants}
    assert names["K_COOL"] == pytest.approx(0.05) and names["T_AMB"] == pytest.approx(300.0)
    assert packet.contracts and packet.contracts[0].contract_id == "c1_cooling"

    # Compactor injects the research-to-software block for the Builder only.
    builder_ctx = build_research_vault_context(ws, [RUN_ID], user_query="build", agent_mode="builder")
    assert "<research_to_software_specification>" in builder_ctx
    research_ctx = build_research_vault_context(ws, [RUN_ID], user_query="x", agent_mode="research")
    assert "<research_to_software_specification>" not in research_ctx

    # -- Plan Mode (mocked Tier-2 planner): card carries packet + attached ids.
    plan_json = json.dumps({"display_text": "## Plan\nCLI over the cooling kernel.",
                            "clean_prompt": "Build a CLI sweep tool for the cooling model."})

    class _FakeLLM:
        model = "mock-planner"

        def __init__(self):
            self.calls = []

        def generate(self, messages, temperature=0.2, max_tokens=4096):
            self.calls.append({"messages": messages, "max_tokens": max_tokens})
            return "```json\n" + plan_json + "\n```"

    fake = _FakeLLM()
    monkeypatch.setattr(CopilotAgent, "_llm_client", lambda self: fake)
    resp = tc.post("/api/copilot/chat", json={
        "message": "Plan a CLI parameter sweep tool for the cooling research",
        "history": [], "attached_research_ids": [RUN_ID]})
    assert resp.status_code == 200
    data = resp.json()
    assert fake.calls, "planner must be the mocked client"
    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "<research_to_software_specification>" in system_prompt
    action = data["action"]
    assert action["type"] == "SUGGEST_BUILD_PROMPT"
    params = action["params"]
    assert params["attached_research_ids"] == [RUN_ID]
    assert params["archetype"] == "CLI_SIMULATOR_SWEEP"
    assert "<research_to_software_specification>" in params["prompt"]
    assert f"@research:{RUN_ID}" in params["prompt"]
    assert data["attached_research_ids"] == [RUN_ID]
    assert data["clean_prompt"] == params["prompt"]

    # Offline Plan Mode path (no provider configured) binds research too.
    monkeypatch.setattr(CopilotAgent, "_llm_client", lambda self: None)
    offline = tc.post("/api/copilot/chat", json={
        "message": f"Build a CLI sweep tool for @research:{RUN_ID}", "history": []}).json()
    assert offline["action"]["type"] == "SUGGEST_BUILD_PROMPT"
    assert offline["action"]["params"]["attached_research_ids"] == [RUN_ID]

    # -- Direct Mode: bound context SSE event, parity test executed, provenance.
    (ws / "simulator.py").write_text(BUILT_SIMULATOR, encoding="utf-8")
    resp = tc.post("/api/copilot/run-build", json={
        "prompt": "Build a CLI parameter sweep tool for the cooling model",
        "attached_research_ids": [RUN_ID], "execution_mode": "direct"})
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    bound = [e for e in events if e.get("type") == "activity"
             and e.get("step") == "research_context_bound"]
    assert bound, "research_context_bound activity must be emitted"
    assert bound[0]["stage"] == "BUILDER"
    assert bound[0]["run_ids"] == [RUN_ID]
    assert bound[0]["archetype"] == "CLI_SIMULATOR_SWEEP"
    assert f"Bound research context ['{RUN_ID}'] (Archetype: CLI_SIMULATOR_SWEEP)" == bound[0]["message"]
    parity = [e for e in events if e.get("type") == "activity"
              and e.get("step") == "research_parity_result"]
    assert parity, "parity test result activity must be emitted"
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[-1]["report"]["research_parity"]["passed"] is True, \
        done[-1]["report"]["research_parity"]["output_tail"]

    assert (ws / PARITY_TEST_RELPATH).is_file()
    provenance = json.loads((ws / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert provenance["archetype"] == "CLI_SIMULATOR_SWEEP"
    assert [r["run_id"] for r in provenance["research_runs"]] == [RUN_ID]

    # A built module violating a research constant fails parity deterministically.
    (ws / "simulator.py").write_text(
        BUILT_SIMULATOR.replace('"K_COOL": 0.05', '"K_COOL": 0.5'), encoding="utf-8")
    failing = run_research_parity_tests(ws)
    assert failing["ran"] and failing["passed"] is False
    assert "K_COOL" in failing["output"]


def test_user_research_upload_and_extraction(client):
    tc, ws = client
    csv_text = "name,value,unit\npeak_temp_k,400.0,K\n"
    resp = tc.post("/api/v1/research/vault/upload", json={
        "title": "Tungsten Carbide Cooling Study",
        "files": [
            {"filename": "paper.md", "content": LATEX_MD},
            {"filename": "benchmarks.csv", "content": csv_text},
            {"filename": "cooling_model.py",
             "content": base64.b64encode(COOLING_MODEL.encode()).decode(),
             "encoding": "base64"},
        ]})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "success"
    run_id = data["run_id"]
    meta = data["metadata"]
    assert meta["origin"] == "user_upload"
    assert meta["status"] == "UPLOADED_READY"
    assert meta["verdict"] == "OPTIMAL_DIRECT"

    constants = meta["extracted_constants"]
    kappa = [v for k, v in constants.items() if "kappa" in k]
    rho = [v for k, v in constants.items() if "rho" in k]
    assert kappa and kappa[0] == pytest.approx(175.0)
    assert rho and rho[0] == pytest.approx(15600.0)

    eqs = " | ".join(meta["governing_equations"])
    assert "dT/dt" in eqs and "T_amb" in eqs

    metrics = meta["key_metrics"]
    names = " ".join(metrics)
    assert "steady_state_temp" in names and "half_life_s" in names and "peak_temp_k" in names

    detail = tc.get(f"/api/v1/research/vault/{run_id}").json()
    assert "def simulation_step" in detail["unified_module_code"]
    assert "K_COOL = 0.05" in detail["unified_module_code"]
    run_dir = ws / ".invar" / "research_vault" / run_id
    assert (run_dir / "uploaded_sources" / "paper.md").is_file()
    assert (run_dir / "uploaded_sources" / "cooling_model.py").is_file()
    assert (run_dir / "report.md").is_file()
    listed = tc.get("/api/v1/research/vault").json()["runs"]
    assert any(r["run_id"] == run_id and r.get("origin") == "user_upload" for r in listed)

    # Multipart upload (alias endpoint); no .py kernel -> synthesized baseline.
    tex = r"The decay law is $\frac{dN}{dt} = -\lambda N$ with lambda = 0.1 1/s."
    resp = tc.post("/api/v1/research/vault/import",
                   files=[("file", ("decay.tex", tex.encode(), "text/plain"))])
    assert resp.status_code == 200, resp.text
    tex_meta = resp.json()["metadata"]
    tex_detail = ResearchVaultManager(ws).get_run(tex_meta["run_id"])
    compile(tex_detail["unified_module_code"], "unified_module.py", "exec")
    assert "def simulation_step" in tex_detail["unified_module_code"]

    # Uploaded runs are usable by the Builder (bridge) ...
    packet = bind_research_context(ws, f"Build a Python library for @research:{run_id}")
    assert packet is not None and packet.run_ids == [run_id]
    assert any(abs(c.value - 15600.0) < 1e-6 for c in packet.constants), [(c.python_name, c.value) for c in packet.constants]
    # ... and by the Research Agent, which is told to verify the upload.
    ctx = build_research_vault_context(ws, [run_id], user_query="verify", agent_mode="research")
    assert "<uploaded_research_verification_directive>" in ctx

    # UPLOADED_READY -> VERIFIED once a seeded research loop verifies it.
    manager = server._STATE.ensure_research_manager()

    class _State:
        status = "VERIFIED"
        aborted = False
        stage = "Adjoint_Commit_or_Rollback"

        def snapshot(self):
            return {}

    class _Machine:
        async def run(self, bp):
            return _State()

    asyncio.run(manager._execute(
        _Machine(), ResearchBlueprint.model_validate(_minimal_blueprint()), [run_id]))
    assert ResearchVaultManager(ws).get_run(run_id)["metadata"]["status"] == "VERIFIED"

    # Bad uploads are rejected cleanly.
    assert tc.post("/api/v1/research/vault/upload", json={"files": []}).status_code == 400
    bad = tc.post("/api/v1/research/vault/upload", json={"files": [
        {"filename": "x.md", "content": "!!!not-base64!!!", "encoding": "base64"}]})
    assert bad.status_code == 400


def _minimal_blueprint() -> dict:
    return {"blueprint_id": "bp_seed", "title": "Seeded upload verification", "falsifiable_hypothesis": "h",
            "search_queries": ["q"], "code_template": KERNEL,
            "contracts": [{"contract_id": "c1", "target_backend": "cpp",
                           "invariants": [{"variable_name": "T",
                                           "assertion": "state[0] >= 0.0"}],
                           "initial_state": [400.0, 0.0], "dt": 0.1, "steps": 10}]}


def test_research_zip_export_and_import_roundtrip(client):
    tc, ws = client
    _register_verified_run(ws)
    up = tc.post("/api/v1/research/vault/upload", json={
        "files": [{"filename": "paper.md", "content": LATEX_MD}]}).json()
    upload_id = up["run_id"]

    resp = tc.get(f"/api/v1/research/vault/{RUN_ID}/export-zip")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert f'filename="{RUN_ID}.invar-research.zip"' in resp.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = set(zf.namelist())
    for required in ("metadata.json", "report.md", "unified_module.py", "blueprint.json"):
        assert f"{RUN_ID}/{required}" in names
    original = ResearchVaultManager(ws).get_run(RUN_ID)

    assert tc.get("/api/v1/research/vault/does_not_exist/export-zip").status_code == 404

    bundle = tc.get("/api/v1/research/vault/export-all-zip")
    assert bundle.status_code == 200
    assert 'filename="invar_research_vault_bundle.zip"' in bundle.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as zf:
        tops = {n.split("/")[0] for n in zf.namelist()}
    assert {RUN_ID, upload_id} <= tops

    # Delete and restore via multipart upload of the .invar-research.zip.
    assert tc.delete(f"/api/v1/research/vault/{RUN_ID}").status_code == 200
    assert ResearchVaultManager(ws).get_run(RUN_ID) is None
    restored = tc.post("/api/v1/research/vault/upload", files=[
        ("file", (f"{RUN_ID}.invar-research.zip", resp.content, "application/zip"))])
    assert restored.status_code == 200, restored.text
    body = restored.json()
    assert body["status"] == "success" and body["run_id"] == RUN_ID and body["restored"]
    again = ResearchVaultManager(ws).get_run(RUN_ID)
    assert again["unified_module_code"] == original["unified_module_code"]
    assert again["report_md"] == original["report_md"]
    assert again["metadata"]["status"] == "VERIFIED"
    assert again["metadata"]["key_metrics"] == original["metadata"]["key_metrics"]
    # Restored research is immediately usable by the Builder bridge.
    assert bind_research_context(ws, f"Build a library for @research:{RUN_ID}").run_ids == [RUN_ID]

    # Zip-slip members inside a research bundle are never written outside the vault.
    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("metadata.json", json.dumps({"run_id": "evil_run", "title": "e"}))
        zf.writestr("report.md", "# e")
        zf.writestr("../../escaped.txt", "pwned")
    tc.post("/api/v1/research/vault/upload", files=[
        ("file", ("evil.invar-research.zip", evil.getvalue(), "application/zip"))])
    assert not (ws / "escaped.txt").exists()
    assert not (ws.parent / "escaped.txt").exists()
    assert not (ws / ".invar" / "escaped.txt").exists()


def test_workspace_software_zip_export_and_import(client, tmp_path):
    tc, ws = client
    (ws / "cooling").mkdir()
    (ws / "cooling" / "__init__.py").write_text("", encoding="utf-8")
    (ws / "cooling" / "kernel.py").write_text(BUILT_SIMULATOR, encoding="utf-8")
    (ws / "README.md").write_text("# cooling\n", encoding="utf-8")
    (ws / "cooling" / "__pycache__").mkdir()
    (ws / "cooling" / "__pycache__" / "kernel.cpython-310.pyc").write_bytes(b"\0")
    (ws / ".git").mkdir()
    (ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (ws / ".venv").mkdir()
    (ws / ".venv" / "pyvenv.cfg").write_text("home=/x\n", encoding="utf-8")
    _register_verified_run(ws)

    resp = tc.get("/api/v1/workspace/export-zip")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert 'filename="invar_software_workspace.zip"' in resp.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("INVAR_BUILD_MANIFEST.json"))
    assert {"cooling/__init__.py", "cooling/kernel.py", "README.md",
            "INVAR_BUILD_MANIFEST.json"} <= names
    for name in names:
        assert not name.startswith((".invar", ".git", ".venv"))
        assert "__pycache__" not in name and not name.endswith(".pyc")
    assert manifest["file_count"] == 3
    assert "cooling/kernel.py" in manifest["files"]

    # Import into a fresh workspace (multipart).
    server._STATE = None
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    tc2 = TestClient(server.create_app(str(ws2)))
    imp = tc2.post("/api/v1/workspace/import-zip", files=[
        ("file", ("invar_software_workspace.zip", resp.content, "application/zip"))])
    assert imp.status_code == 200, imp.text
    body = imp.json()
    assert body["status"] == "imported"
    assert body["file_count"] == 3
    assert sorted(body["files_imported"]) == ["README.md", "cooling/__init__.py",
                                              "cooling/kernel.py"]
    assert (ws2 / "cooling" / "kernel.py").read_text(encoding="utf-8") == BUILT_SIMULATOR
    assert not (ws2 / "INVAR_BUILD_MANIFEST.json").exists()
    tree = tc2.get("/api/files").json()
    assert "kernel.py" in json.dumps(tree) and "cooling" in json.dumps(tree)

    # .git / .invar members are ignored; zip-slip members are rejected with 400.
    hidden = io.BytesIO()
    with zipfile.ZipFile(hidden, "w") as zf:
        zf.writestr(".git/config", "[core]")
        zf.writestr(".invar/research_vault/x/metadata.json", "{}")
        zf.writestr("src/app.py", "print('ok')\n")
    ok = tc2.post("/api/v1/workspace/import-zip", files=[
        ("file", ("w.zip", hidden.getvalue(), "application/zip"))]).json()
    assert ok["files_imported"] == ["src/app.py"]
    assert not (ws2 / ".git").exists() and not (ws2 / ".invar" / "research_vault" / "x").exists()

    for evil_name in ("../evil.py", "/etc/evil.py", "pkg/../../evil.py"):
        evil = io.BytesIO()
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr(evil_name, "boom")
        bad = tc2.post("/api/v1/workspace/import-zip", files=[
            ("file", ("evil.zip", evil.getvalue(), "application/zip"))])
        assert bad.status_code == 400, evil_name
        assert "zip-slip" in bad.json()["detail"]
    assert not (tmp_path / "evil.py").exists()

    assert tc2.post("/api/v1/workspace/import-zip", files=[
        ("file", ("x.zip", b"not a zip", "application/zip"))]).status_code == 400

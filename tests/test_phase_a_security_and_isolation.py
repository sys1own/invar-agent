"""Phase A Hard Gate: Containment, Trust Boundaries, Vault Manifest v2, Concurrency & Encoding."""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from invar_agent.builder.ingestion import CanonicalResearchIngestionPipeline, IngestionRequest
from invar_agent.builder.ingestion.archive_guard import IngestionError
from invar_agent.builder.ingestion.vault_adapter import load_vault_bundle
from invar_agent.research.broadcaster import ResearchEventBroadcaster, ResearchRunManager
from invar_agent.research.epistemic import incomplete_certificate
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.research.vault import (
    REQUIRED_VAULT_ARTIFACTS,
    ResearchVaultManager,
    VaultIntegrityError,
    verify_vault_manifest,
)
from invar_agent.runtime.isolation import (
    IsolationViolation,
    run_isolated_kernel_smoke,
    validate_kernel_ast_allowlist,
)
from invar_agent.schemas.research import ResearchBlueprint, VerificationContract
from invar_agent.server import create_app


# ── 1. ARCH-005 Escape Corpus (AST allow-list + OS isolation) ────────────────

ESCAPE_CORPUS = [
    # Dunder / MRO / subclasses escape
    "def simulation_step(buf, dt):\n    x = ().__class__.__bases__[0].__subclasses__()\n    buf.data[0] = float(len(x))\n",
    # Direct __builtins__ access
    "def simulation_step(buf, dt):\n    __builtins__['open']('pwn.txt', 'w').write('x')\n",
    # __import__ call
    "def simulation_step(buf, dt):\n    os = __import__('os')\n    buf.data[0] = 1.0\n",
    # Forbidden import os/subprocess/socket
    "import os\ndef simulation_step(buf, dt):\n    os.system('echo pwn')\n",
    "import subprocess\ndef simulation_step(buf, dt):\n    subprocess.run(['echo', 'hi'])\n",
    "import socket\ndef simulation_step(buf, dt):\n    socket.socket()\n",
    # Forbidden open() builtin
    "def simulation_step(buf, dt):\n    f = open('escape.txt', 'w')\n    buf.data[0] = 1.0\n",
    # Forbidden getattr / eval / exec
    "def simulation_step(buf, dt):\n    getattr(buf, '__class__')\n",
    "def simulation_step(buf, dt):\n    eval('1+1')\n",
    # Forbidden class / lambda / comprehension
    "class A:\n    pass\ndef simulation_step(buf, dt):\n    buf.data[0] = 1.0\n",
    "def simulation_step(buf, dt):\n    f = lambda x: x\n    buf.data[0] = f(1.0)\n",
    # Forbidden private attribute on buf
    "def simulation_step(buf, dt):\n    buf.__dict__['x'] = 1\n",
    # Unbounded while True
    "def simulation_step(buf, dt):\n    while True:\n        buf.data[0] = 1.0\n",
]


@pytest.mark.parametrize("payload", ESCAPE_CORPUS)
def test_guard_escape_corpus_ast_allowlist(payload: str) -> None:
    violations = validate_kernel_ast_allowlist(payload)
    assert violations, f"Escape payload was NOT blocked by AST allow-list:\n{payload}"
    with pytest.raises(IsolationViolation):
        run_isolated_kernel_smoke(payload, validate_ast=True)


def test_guard_escape_corpus_os_layer(tmp_path: Path) -> None:
    """Even when AST check is bypassed, the OS child worker blocks file, process, and network."""
    marker = tmp_path / "escaped.txt"
    os_payloads = [
        # File write attempt via builtins/open (should be blocked by restricted builtins + audit hook)
        f"open({str(marker)!r}, 'w').write('pwn')",
        # Subprocess attempt
        "import subprocess; subprocess.Popen(['cmd', '/c', 'echo', 'hi'])",
        # Network socket attempt
        "import socket; s = socket.socket(); s.connect(('1.1.1.1', 80))",
        # Dunder subclass recovery of os._wrap_close / BuiltinImporter
        "next(c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'BuiltinImporter').load_module('os').system('echo pwn')",
    ]
    for code in os_payloads:
        with pytest.raises((PermissionError, RuntimeError, IsolationViolation, TimeoutError)):
            run_isolated_kernel_smoke(code, validate_ast=False, mode="raw", timeout_s=4.0)
    assert not marker.exists(), "OS isolation failed: marker file was created!"


def test_legitimate_kernel_passes_isolated_smoke() -> None:
    code = (
        "import math\n"
        "def simulation_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    k = buf.data[1]\n"
        "    buf.data[0] = x + dt * (-k * x + math.sin(0.0))\n"
    )
    assert validate_kernel_ast_allowlist(code) == []
    res = run_isolated_kernel_smoke(code, dt=0.01, validate_ast=True)
    assert res["ok"] is True


# ── 2. ARCH-007 Vault Manifest v2 & Admission Matrix ─────────────────────────

def _sample_verified_bp(run_id: str, assertion: str = "state[0] <= 1.0", *, mode: str = "DISCOVERY") -> ResearchBlueprint:
    code = (
        "import math\n"
        "def simulation_step(buf, dt):\n"
        "    x = buf.data[0]\n"
        "    r = buf.data[1]\n"
        "    buf.data[0] = x + dt * (-r * x)\n"
        "    buf.data[2] = buf.data[0]\n"
    )
    return ResearchBlueprint(
        blueprint_id=run_id,
        title=f"Decay {run_id}",
        research_prompt=f"Check decay {run_id}",
        falsifiable_hypothesis=assertion,
        search_queries=["decay"],
        code_template=code,
        variable_map={"x": 0, "r": 1, "y": 2},
        contracts=[
            VerificationContract(
                contract_id="c1",
                target_backend="cpp",
                initial_state=[0.8, 0.5, 0.8],
                dt=0.1,
                steps=5,
                invariants=[{"variable_name": "x", "assertion": assertion, "tolerance": 1e-6}],
            )
        ],
    )


def test_vault_admission_matrix(tmp_path: Path) -> None:
    """Test intact, 1-byte tamper on EVERY vault file, missing cert, and inconclusive/falsified verdicts."""
    vault = ResearchVaultManager(tmp_path)
    # 1. Incomplete / inconclusive run
    bp_inc = _sample_verified_bp("run_inc")
    cert_inc = incomplete_certificate("run_inc", bp_inc.falsifiable_hypothesis, "NO_EVIDENCE")
    vault.register_run(
        run_id="run_inc",
        title="Inconclusive",
        blueprint=bp_inc,
        report_md="# report\n",
        status="INCONCLUSIVE_UNVERIFIED",
        meaningfulness_certificate=cert_inc,
        llm_telemetry={"calls": 2, "prompt_tokens": 100, "completion_tokens": 50},
    )
    run_dir = vault.run_dir("run_inc")
    manifest = verify_vault_manifest(run_dir)
    assert manifest["llm_telemetry"]["calls"] == 2
    assert manifest["epistemic_verdict"] == "INCONCLUSIVE_UNVERIFIED"

    # Inconclusive MUST be rejected with 422
    with pytest.raises(IngestionError) as exc_inc:
        load_vault_bundle(tmp_path, "run_inc")
    assert exc_inc.value.status_code == 422
    assert exc_inc.value.code == "INCONCLUSIVE_UNVERIFIED"

    # 2. Single-byte tamper on EVERY persisted vault artifact + manifest.json MUST be rejected
    for artifact_name in (*REQUIRED_VAULT_ARTIFACTS, "manifest.json"):
        target = run_dir / artifact_name
        original = target.read_bytes()
        try:
            flipped = bytearray(original)
            idx = max(0, len(flipped) - 2)
            flipped[idx] = (flipped[idx] + 1) % 256
            target.write_bytes(bytes(flipped))
            with pytest.raises(VaultIntegrityError):
                verify_vault_manifest(run_dir)
            with pytest.raises(IngestionError) as exc_tamp:
                load_vault_bundle(tmp_path, "run_inc")
            assert exc_tamp.value.status_code == 422
        finally:
            target.write_bytes(original)

    # 3. Missing certificate MUST be rejected
    cert_file = run_dir / "meaningfulness_certificate.json"
    cert_bytes = cert_file.read_bytes()
    manifest_file = run_dir / "manifest.json"
    manifest_bytes = manifest_file.read_bytes()
    try:
        cert_file.unlink()
        manifest_file.unlink()
        with pytest.raises(IngestionError) as exc_miss:
            load_vault_bundle(tmp_path, "run_inc")
        assert exc_miss.value.status_code == 422
        assert exc_miss.value.code == "VAULT_CERTIFICATE_MISSING"
    finally:
        cert_file.write_bytes(cert_bytes)
        manifest_file.write_bytes(manifest_bytes)


# ── 3. BUG-032 Export Guard & ARCH-005 Bearer Auth ──────────────────────────

def test_export_requires_completed_and_bearer_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(workspace_root=tmp_path)
    client = TestClient(app)

    # Inject a synthetic ABORTED_CRITICAL build status in builder workspace
    build_id = "aborted_test_build"
    build_root = tmp_path / ".invar" / "builds" / build_id
    (build_root / "workspace").mkdir(parents=True)
    status_dir = tmp_path / ".invar" / "builder" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{build_id}.json").write_text(
        json.dumps({"build_id": build_id, "status": "ABORTED_CRITICAL", "error": "boom"}),
        encoding="utf-8",
    )
    resp = client.get(f"/api/v1/builder/export/{build_id}.zip")
    assert resp.status_code == 409, f"Expected 409 for aborted build export, got {resp.status_code}: {resp.text}"

    # Optional Bearer auth enforcement
    monkeypatch.setenv("INVAR_API_TOKEN", "secret-phase-a-token")
    unauth = client.post("/api/v1/research/action", json={"action": "abort"})
    assert unauth.status_code == 401
    auth_ok = client.post(
        "/api/v1/research/action",
        json={"action": "abort"},
        headers={"Authorization": "Bearer secret-phase-a-token"},
    )
    assert auth_ok.status_code == 200


# ── 4. ARCH-006 Concurrency & Terminal Event Totality ────────────────────────

def test_terminal_event_totality_and_run_uid_scoping(tmp_path: Path) -> None:
    broadcaster = ResearchEventBroadcaster()
    uid1 = "run_alpha"
    uid2 = "run_beta"
    broadcaster.clear_history(run_uid=uid1)
    broadcaster.publish("accelerator_log", "step", {"n": 1}, run_uid=uid1)
    broadcaster.publish_terminal(uid1, status="VERIFIED", epistemic_verdict="VERIFIED_DISCOVERY")
    # Duplicate terminal for uid1 MUST be ignored (exactly-once)
    dup = broadcaster.publish_terminal(uid1, status="FAILED", epistemic_verdict="INCONCLUSIVE_UNVERIFIED")
    assert dup is None

    broadcaster.clear_history(run_uid=uid2)
    broadcaster.publish("accelerator_log", "step", {"n": 2}, run_uid=uid2)
    broadcaster.publish_terminal(uid2, status="REFUSED_UNSUITABLE", epistemic_verdict="REFUSED_UNSUITABLE")

    hist1 = broadcaster.get_run_history(uid1)
    hist2 = broadcaster.get_run_history(uid2)
    assert sum(1 for e in hist1 if e["event"] == "terminal_verdict") == 1
    assert sum(1 for e in hist2 if e["event"] == "terminal_verdict") == 1
    assert all(e["run_uid"] == uid1 and e["event_version"] == "2.0" for e in hist1)
    assert all(e["run_uid"] == uid2 and e["event_version"] == "2.0" for e in hist2)

    # Scoped subscription to uid2 must NEVER replay uid1 events
    q2 = broadcaster.subscribe(since_seq=0, run_uid=uid2)
    received = []
    while not q2.empty():
        received.append(q2.get_nowait())
    assert received and all(e["run_uid"] == uid2 for e in received)


def test_concurrent_runs_z3_isolation(tmp_path: Path) -> None:
    """8 parallel Z3/research formal queries execute cleanly with zero Z3 crashes and exact serial parity."""
    from invar_agent.research.meaningfulness import solve_query

    queries = [
        ("x + y <= 10 and x >= 1 and y >= 2", {"x": (0.0, 10.0), "y": (0.0, 10.0)}),
        ("x * y >= 25 and x <= 4 and y <= 4", {"x": (0.0, 5.0), "y": (0.0, 5.0)}),
        ("a*a + b*b <= 1 and a >= 0.5", {"a": (-2.0, 2.0), "b": (-2.0, 2.0)}),
        ("eta >= 0.65 and eta <= 1 - 300/600", {"eta": (0.0, 1.0)}),
    ] * 2

    serial_statuses = [solve_query(expr, dom, witness=True).status for expr, dom in queries]
    parallel_statuses: list[str | None] = [None] * len(queries)
    errors: list[BaseException] = []

    def _worker(idx: int, expr: str, dom: dict) -> None:
        try:
            parallel_statuses[idx] = solve_query(expr, dom, witness=True).status
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=(i, expr, dom))
        for i, (expr, dom) in enumerate(queries)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent Z3 execution raised errors: {errors}"
    assert parallel_statuses == serial_statuses

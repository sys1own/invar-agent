"""Web server and export tests: FastAPI endpoints and curated exports."""
from __future__ import annotations

import io
import json
import sys
import textwrap
import zipfile

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import invar_agent.server as server
from invar_agent.server import create_app


def _sample_workspace(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    (tmp_path / "conftest.py").write_text("")
    return tmp_path


@pytest.fixture()
def client(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    _sample_workspace(ws)
    monkeypatch.setattr(server, "_STATE", None)
    app = create_app(str(ws))
    return TestClient(app), ws


def test_index_and_status(client):
    tc, _ = client
    resp = tc.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "invar" in html.lower()
    # No legacy accelerate menu / .aero bloat.
    assert "Accelerate" not in html
    assert ".aero" not in html

    status = tc.get("/api/status").json()
    assert status["kernel"] == "invar-agent"
    assert "native_core_status" in status
    assert "holographic_boundary_register" in status
    assert status["holographic_boundary_register"]["algorithm"].startswith("BLAKE3")
    assert "router_budget" in status


def test_workspace_ingest_and_execute(client):
    tc, ws = client
    ingest = tc.post("/api/workspace/ingest", json={"path": str(ws)}).json()
    assert ingest["file_count"] >= 2

    node = {
        "node_id": "wf1", "wavefront_generation": 0, "assigned_tier": "TIER_1_FLASH",
        "description": "extend calc", "read_set_paths": ["test_calc.py"], "write_set_paths": ["calc.py"],
        "target_ast_symbols": ["calc.add"],
        "output_ports": [{"port_name": "out", "symbol_contract": "calc.add"}],
        "verification_contract": {},
    }
    patch = {
        "file_path": "calc.py", "operation": "INSERT_AFTER_SYMBOL", "target_symbol": "add",
        "search_block": "", "replacement_code": "def sub(a, b):\n    return a - b\n",
    }
    exec_resp = tc.post("/api/agent/execute", json={
        "objective": "add sub helper",
        "nodes": [node],
        "patches_by_node": {"wf1": [patch]},
        "sandbox_cmd": [sys.executable, "-m", "pytest", "-q"],
    })
    assert exec_resp.status_code == 200
    report = exec_resp.json()
    assert report["nodes_total"] == 1
    assert report["nodes_committed"] == 1
    assert "def sub" in (ws / "calc.py").read_text()


def test_overlay_rollback_endpoint(client):
    tc, ws = client
    tc.post("/api/workspace/ingest", json={"path": str(ws)})
    resp = tc.post("/api/overlay/rollback", json={})
    # No active transaction -> 404 is acceptable and documented.
    assert resp.status_code in (200, 404)


def test_exports(client):
    tc, ws = client
    tc.post("/api/workspace/ingest", json={"path": str(ws)})

    z = tc.get("/api/export/workspace_zip")
    assert z.status_code == 200
    assert z.content[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        names = zf.namelist()
        assert "calc.py" in names

    bundle = tc.get("/api/export/patch_adjoint_bundle")
    assert bundle.status_code == 200
    data = json.loads(bundle.content)
    assert data["schema"] == "invar-agent/patch-adjoint-bundle/v1"
    assert "patches" in data and "adjoint_patches" in data

    cert = tc.get("/api/export/verification_certificate")
    assert cert.status_code == 200
    cdata = json.loads(cert.content)
    assert cdata["schema"] == "invar-agent/verification-certificate/v1"
    assert "nodes_committed" in cdata

    bad = tc.get("/api/export/nope")
    assert bad.status_code == 404

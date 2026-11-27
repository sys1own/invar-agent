"""UPG-017 (BUG-031..034) + UPG-019 (BUG-036): Dataset schemas, fresh-unzip pytest, InvarDomainError, Build Explorer."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from invar_agent.builder.dataset_schema import infer_dataset_schema, verify_fresh_unzip_pytest
from invar_agent.builder.emitters.hybrid_compute import KERNEL_RUNTIME
from invar_agent.server import create_app


def test_upg017_infer_dataset_schema_and_flags():
    schema, flags = infer_dataset_schema("run.csv", ["time_s", "T (K)", "V_out (V)"])
    assert flags == []
    assert schema.time_column == "time_s" and schema.time_unit == "s"
    assert [c.unit for c in schema.columns] == ["s", "K", "V"]

    _s2, flags2 = infer_dataset_schema("bad.csv", ["x (not_a_real_unit_xyz)"])
    assert any("DATA_SCHEMA_REQUIRED" in f for f in flags2)


def test_upg017_kernel_runtime_raises_invar_domain_error_not_silent_zero():
    ns = {"math": __import__("math")}
    exec(KERNEL_RUNTIME, ns)
    InvarDomainError = ns["InvarDomainError"]
    with pytest.raises(InvarDomainError):
        ns["safe_div"](1.0, 0.0)
    with pytest.raises(InvarDomainError):
        ns["safe_sqrt"](-1.0)
    with pytest.raises(InvarDomainError):
        ns["safe_log"](0.0)
    assert ns["safe_sqrt"](4.0) == 2.0


def test_upg017_fresh_unzip_pytest_proof(tmp_path: Path):
    zpath = tmp_path / "pkg.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(
            "pyproject.toml",
            '[tool.pytest.ini_options]\npythonpath = ["src"]\ntestpaths = ["tests"]\n',
        )
        zf.writestr("src/demo_pkg/__init__.py", "def add(a, b): return a + b\n")
        zf.writestr("tests/test_smoke.py", "from demo_pkg import add\ndef test_ok(): assert add(2, 3) == 5\n")
    res = verify_fresh_unzip_pytest(zpath, timeout_s=30.0)
    assert res["ok"] is True
    assert res["export_pytest_exit_code"] == 0
    assert res["export_pytest_passed"] == 1


def test_upg019_build_explorer_tree_file_and_path_traversal_blocked(tmp_path: Path):
    app = create_app(workspace_root=tmp_path)
    build_id = "testbuild01"
    ws = tmp_path / ".invar" / "builds" / build_id / "workspace"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "main.py").write_text("x = 42\n", encoding="utf-8")
    (ws / "bin.dat").write_bytes(b"\x00\x01\x02")
    (ws / ".invar").mkdir(parents=True)
    (ws / ".invar" / "secret.txt").write_text("hidden", encoding="utf-8")

    client = TestClient(app)
    r_tree = client.get(f"/api/v1/builder/builds/{build_id}/tree")
    assert r_tree.status_code == 200
    paths = [f["path"] for f in r_tree.json()["files"]]
    assert "src/main.py" in paths
    assert not any(".invar" in p for p in paths)

    r_file = client.get(f"/api/v1/builder/builds/{build_id}/file", params={"path": "src/main.py"})
    assert r_file.status_code == 200
    assert r_file.json()["content"].strip() == "x = 42"

    # Traversal blocked -> 400 PATH_OUTSIDE_BUILD
    r_bad = client.get(f"/api/v1/builder/builds/{build_id}/file", params={"path": "../outside.txt"})
    assert r_bad.status_code == 400

    # Binary rejected -> 415
    r_bin = client.get(f"/api/v1/builder/builds/{build_id}/file", params={"path": "bin.dat"})
    assert r_bin.status_code == 415

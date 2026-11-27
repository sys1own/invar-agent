"""Phase 4 regression tests (ARCH-012, ARCH-013, ARCH-014 -> BUG-018..022)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from invar_agent.builder.export_utils import (
    ExportBlockedError,
    build_verified_export_zip,
    scan_archive_for_leaks,
)
from invar_agent.builder.ingestion.archive_guard import IngestionError
from invar_agent.builder.ingestion.equation_lowerer import equation_parts
from invar_agent.builder.ingestion.parsers import parse_document
from invar_agent.builder.ingestion.pipeline import (
    CanonicalResearchIngestionPipeline,
    apply_ir_override,
)
from invar_agent.builder.ingestion.schemas import (
    ConstantOverridePatch,
    IRReviewOverride,
    IngestionRequest,
)
from invar_agent.cli import main as cli_main
from tests.test_phase1_lockfree_and_bounded import _make_real_verified_vault_entry


def test_manifest_required_and_override_outside_domain_downgrades(tmp_path):
    """Test 14: Deleted manifest.json fails closed; out-of-domain override downgrades certificate."""
    run_id = _make_real_verified_vault_entry(tmp_path, "fx_rk4_trust")
    vault_dir = tmp_path / ".invar" / "research_vault" / run_id

    # 1. Valid manifest ingests as POSITIVE_EVIDENCE
    ir = CanonicalResearchIngestionPipeline(tmp_path).ingest(
        IngestionRequest(vault_run_ids=[run_id])
    )
    assert ir.research_import_role == "POSITIVE_EVIDENCE"
    assert str(ir.epistemic_verdict or "").startswith("VERIFIED_")

    # 2. Override a constant outside certified_valid_domain -> MUST downgrade to USER_MODIFIED_UNVERIFIED
    drive_const = next(c for c in ir.constants if c.symbol == "drive")
    ov_ir = apply_ir_override(
        ir,
        IRReviewOverride(
            constant_patches=[ConstantOverridePatch(constant_id=drive_const.constant_id, si_value=50.0)],
            approve=True,
        ),
    )
    assert ov_ir.research_import_role == "USER_MODIFIED_UNVERIFIED"
    assert ov_ir.epistemic_verdict == "USER_MODIFIED_UNVERIFIED"

    # 3. Deleting manifest.json MUST fail closed with VAULT_MANIFEST_MISSING
    (vault_dir / "manifest.json").unlink()
    with pytest.raises(IngestionError) as exc_info:
        CanonicalResearchIngestionPipeline(tmp_path).ingest(
            IngestionRequest(vault_run_ids=[run_id])
        )
    assert exc_info.value.code == "VAULT_MANIFEST_MISSING"


def test_export_has_no_absolute_paths_or_pyc(tmp_path):
    """Test 15: Export zip excludes __pycache__/*.pyc and fails closed on Windows/POSIX host paths or secrets."""
    ws = tmp_path / "ws"
    (ws / "src" / "pkg").mkdir(parents=True)
    (ws / "src" / "pkg" / "sdk.py").write_text("MODEL = {'uploaded_files': ['spec.md']}\n", encoding="utf-8")
    (ws / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (ws / "src" / "pkg" / "__pycache__" / "pinned_research.cpython-312.pyc").write_bytes(b"\x00\x01pyc")

    out_zip = tmp_path / "clean.zip"
    build_verified_export_zip(ws, out_zip, build_status="COMPLETED")
    assert out_zip.is_file()
    with zipfile.ZipFile(out_zip, "r") as zf:
        names = zf.namelist()
        assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    assert scan_archive_for_leaks(out_zip) == []

    # Inject Windows host path -> must fail closed and delete archive
    (ws / "src" / "pkg" / "leak_win.py").write_text("P = r'C:\\Users\\alice\\secret\\spec.md'\n", encoding="utf-8")
    win_zip = tmp_path / "win_leak.zip"
    with pytest.raises(ExportBlockedError) as exc_win:
        build_verified_export_zip(ws, win_zip, build_status="COMPLETED")
    assert exc_win.value.code == "EXPORT_TRUST_BOUNDARY_LEAK"
    assert not win_zip.exists()
    (ws / "src" / "pkg" / "leak_win.py").unlink()

    # Inject POSIX host path -> must fail closed and delete archive
    (ws / "src" / "pkg" / "leak_posix.py").write_text("P = '/home/alice/project/spec.md'\n", encoding="utf-8")
    posix_zip = tmp_path / "posix_leak.zip"
    with pytest.raises(ExportBlockedError) as exc_posix:
        build_verified_export_zip(ws, posix_zip, build_status="COMPLETED")
    assert exc_posix.value.code == "EXPORT_TRUST_BOUNDARY_LEAK"
    assert not posix_zip.exists()


def test_cli_builder_ingest_approve_plan_execute_export(tmp_path):
    """Test 16: CLI builder ingest -> approve -> plan -> execute -> export works; failed build export exits non-zero with no zip."""
    run_id = _make_real_verified_vault_entry(tmp_path, "fx_rk4_cli")
    ir_json = tmp_path / "ir.json"
    plan_json = tmp_path / "plan.json"
    out_zip = tmp_path / "cli_export.zip"

    # 1. ingest
    rc = cli_main(["builder", "ingest", str(tmp_path), "--vault-id", run_id, "--out", str(ir_json)])
    assert rc == 0 and ir_json.is_file()

    # 2. approve
    rc = cli_main(["builder", "approve", str(tmp_path), str(ir_json)])
    assert rc == 0

    # 3. plan (previously AttributeError: 'BuilderPlanner' object has no attribute 'plan')
    rc = cli_main(["builder", "plan", str(tmp_path), str(ir_json), "--out", str(plan_json)])
    assert rc == 0 and plan_json.is_file()

    # 4. execute
    rc = cli_main(["builder", "execute", str(tmp_path), str(ir_json), str(plan_json), "--build-id", "cli_b1"])
    assert rc == 0

    # 5. export completed build -> succeeds with zero leaks
    rc = cli_main(["builder", "export", str(tmp_path), "cli_b1", "--out", str(out_zip)])
    assert rc == 0 and out_zip.is_file()
    assert scan_archive_for_leaks(out_zip) == []

    # 6. export failed/aborted build -> non-zero exit and NO zip file written
    failed_dir = tmp_path / ".invar" / "builds" / "cli_failed"
    (failed_dir / "workspace").mkdir(parents=True)
    (failed_dir / "build_receipt.json").write_text(json.dumps({"status": "ABORTED_CRITICAL"}), encoding="utf-8")
    failed_zip = tmp_path / "should_not_exist.zip"
    rc = cli_main(["builder", "export", str(tmp_path), "cli_failed", "--out", str(failed_zip)])
    assert rc != 0
    assert not failed_zip.exists()


def test_parser_table_punctuation_keywords_sigma_implicit_ode(tmp_path):
    """Test 17: Parser handles trailing punctuation, lambda keyword, sigma, implicit products, and ODE approval."""
    # 1. Trailing punctuation on unit: `tau = 20 s.` -> unit `s`, NOT `1`
    doc_file = tmp_path / "punct.md"
    doc_file.write_text("The time constant is tau = 20 s.\n", encoding="utf-8")
    parsed = parse_document(doc_file)
    tau_c = next(c for c in parsed.constants if c.symbol == "tau")
    assert tau_c.unit == "s", f"Expected unit 's', got {tau_c.unit!r}"

    # 2. Reserved keyword `lambda` parsed cleanly without SympifyError and restored as Symbol('lambda')
    lhs_l, rel_l, rhs_l = equation_parts("a = lambda / mu")
    assert rel_l == "="
    assert "lambda" in {str(s) for s in rhs_l.free_symbols}

    # 3. `sigma` without backslash stays a single symbol (never split into s*i*g*m*a)
    _, _, rhs_s = equation_parts(r"d_1 = \frac{\ln(S/K) + (r + \frac{1}{2} sigma^2) T}{sigma \sqrt{T}}")
    free_names = {str(s) for s in rhs_s.free_symbols}
    assert "sigma" in free_names
    assert not ({"s", "i", "g", "m", "a"} <= free_names)

    # 4. Implicit products `(...) P_0` and `(...) \exp(...)`
    _, _, rhs_imp1 = equation_parts(r"C = (a + b) P_0")
    assert {"a", "b", "P_0"} <= {str(s) for s in rhs_imp1.free_symbols}
    _, _, rhs_imp2 = equation_parts(r"V = (a + b) \exp(-r T)")
    assert {"a", "b", "r", "T"} <= {str(s) for s in rhs_imp2.free_symbols}

    # 5. Standalone ODE ingest -> approval round trip without HIN_SI_DIM_MISMATCH
    ode_file = tmp_path / "decay.md"
    ode_file.write_text(
        "# Thermal Decay\n\n"
        "Parameters:\n"
        "- k = 0.05 1/s\n\n"
        "Equations:\n"
        "$$\\frac{dx}{dt} = -k x$$\n",
        encoding="utf-8",
    )
    ir = CanonicalResearchIngestionPipeline(tmp_path).ingest(
        IngestionRequest(upload_paths=[str(ode_file)])
    )
    for flag in ir.disambiguation_flags:
        flag.resolved = True
    # Must approve cleanly without HIN_SI_DIM_MISMATCH
    approved = apply_ir_override(ir, IRReviewOverride(approve=True))
    assert approved.user_approved is True

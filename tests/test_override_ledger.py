"""HOTFIX-B / UPG-013 (BUG-005, NF-02) tests: provenance-safe override ledger."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from invar_agent.builder.ingestion.pipeline import (
    CanonicalResearchIngestionPipeline,
    apply_ir_override,
    seal_ir,
)
from invar_agent.builder.ingestion.schemas import (
    CanonicalResearchIR,
    ConstantOverridePatch,
    IRReviewOverride,
    IngestionRequest,
    SymbolResolution,
)
from invar_agent.builder.research_bridge import validate_research_ir
from invar_agent.builder.state_machine import BuildRequest, BuilderStateMachine
from invar_agent.builder.uast_models import BuilderArchitectureSpec
from invar_agent.builder.planner import BuilderPlanner
from invar_agent.research import meaningfulness as _m
from invar_agent.research.vault import write_vault_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "vault_reference"


def _make_verified_vault_run(tmp_path: Path, monkeypatch, *, certified_valid_domain=None) -> str:
    run_id = "example_r4_ov"
    root = tmp_path / ".invar" / "research_vault" / run_id
    shutil.copytree(FIXTURE, root)
    code = (root / "unified_module.py").read_text(encoding="utf-8")
    code = code.replace(
        "    if not math.isfinite(buf.data[0]) or buf.data[0] > 0.001:\n        buf.data[0] = 0.001\n",
        "",
    )
    (root / "unified_module.py").write_text(code, encoding="utf-8")
    bp = json.loads((root / "blueprint.json").read_text(encoding="utf-8"))
    for contract in bp["contracts"]:
        contract["steps"] = 10
    (root / "blueprint.json").write_text(json.dumps(bp, indent=2), encoding="utf-8")
    cert_payload = {
        "run_id": run_id,
        "epistemic_verdict": "VERIFIED_DISCOVERY",
        "quantifier_mode": "FORALL",
    }
    if certified_valid_domain is not None:
        cert_payload["certified_valid_domain"] = certified_valid_domain
    (root / "meaningfulness_certificate.json").write_text(
        json.dumps(cert_payload),
        encoding="utf-8",
    )
    write_vault_manifest(
        root,
        run_id=run_id,
        epistemic_verdict="VERIFIED_DISCOVERY",
        status="VERIFIED_DISCOVERY",
    )

    def _fake_validate(raw, module_code=None, bp=None):
        data = dict(raw) if isinstance(raw, dict) else dict(cert_payload)
        return SimpleNamespace(
            run_id=data.get("run_id", run_id),
            epistemic_verdict=_m.EpistemicVerdict.VERIFIED_DISCOVERY,
            certified_valid_domain=data.get("certified_valid_domain"),
            model_dump=lambda mode="json": dict(data),
        )

    monkeypatch.setattr("invar_agent.research.epistemic.validate_certificate_artifacts", _fake_validate)
    monkeypatch.setattr("invar_agent.research.epistemic.parse_run_certificate", _fake_validate)
    return run_id


def test_zero_drift_serialization_omits_empty_override_fields():
    ir = CanonicalResearchIR(ir_id="zero_drift", source_mode="upload", title="Zero Drift")
    dumped = ir.model_dump(mode="json")
    assert "source_certificates" not in dumped
    assert "ir_verdict" not in dumped
    assert "override_ledger" not in dumped


def test_nf02_symbol_resolution_override_downgrades_verdict(tmp_path, monkeypatch):
    run_id = _make_verified_vault_run(tmp_path, monkeypatch)
    pipe = CanonicalResearchIngestionPipeline(tmp_path)
    ir = pipe.ingest(IngestionRequest(vault_run_ids=[run_id], title="NF-02 test"))
    assert ir.epistemic_verdict == "VERIFIED_DISCOVERY"
    assert ir.research_import_role == "POSITIVE_EVIDENCE"
    assert len(ir.constants) >= 1
    first_const = ir.constants[0]

    # NF-02 attack: change certified constant via symbol_resolutions FREE_PARAMETER
    overridden = apply_ir_override(
        ir,
        IRReviewOverride(
            symbol_resolutions=[
                SymbolResolution(
                    symbol=first_const.symbol,
                    role="FREE_PARAMETER",
                    si_value=float(first_const.si_value) + 6.0,
                )
            ],
            approve=True,
        ),
    )
    assert overridden.epistemic_verdict == "USER_MODIFIED_UNVERIFIED"
    assert overridden.ir_verdict == "USER_MODIFIED_UNVERIFIED"
    assert overridden.research_import_role == "USER_MODIFIED_UNVERIFIED"
    assert len(overridden.override_ledger) >= 1
    assert any(e.invalidating for e in overridden.override_ledger)
    # Must pass validate_research_ir (BUG-005 fixed)
    validated = validate_research_ir(overridden)
    assert validated.epistemic_verdict == "USER_MODIFIED_UNVERIFIED"


def test_irreversible_downgrade_and_tamper_rejection(tmp_path, monkeypatch):
    run_id = _make_verified_vault_run(tmp_path, monkeypatch)
    pipe = CanonicalResearchIngestionPipeline(tmp_path)
    ir = pipe.ingest(IngestionRequest(vault_run_ids=[run_id], title="Tamper test"))
    first_const = ir.constants[0]
    orig_val = float(first_const.si_value)

    ov1 = apply_ir_override(
        ir,
        IRReviewOverride(
            constant_patches=[ConstantOverridePatch(constant_id=first_const.constant_id, si_value=orig_val + 5.0)],
            approve=True,
        ),
    )
    assert ov1.epistemic_verdict == "USER_MODIFIED_UNVERIFIED"

    # Restoring the old value must NOT restore VERIFIED_DISCOVERY
    ov2 = apply_ir_override(
        ov1,
        IRReviewOverride(
            constant_patches=[ConstantOverridePatch(constant_id=first_const.constant_id, si_value=orig_val)],
            approve=True,
        ),
    )
    assert ov2.epistemic_verdict == "USER_MODIFIED_UNVERIFIED"
    assert ov2.ir_verdict == "USER_MODIFIED_UNVERIFIED"

    # Tampering: flipping invalidating=False or restoring verdict fails validate_research_ir
    tampered = ov1.model_copy(deep=True)
    tampered.override_ledger[0].invalidating = False
    seal_ir(tampered)
    with pytest.raises(ValueError, match="OVERRIDE_LEDGER_TAMPERED"):
        validate_research_ir(tampered)

    tampered2 = ov1.model_copy(deep=True)
    tampered2.epistemic_verdict = "VERIFIED_DISCOVERY"
    tampered2.research_import_role = "POSITIVE_EVIDENCE"
    seal_ir(tampered2)
    with pytest.raises(ValueError, match="OVERRIDE_LEDGER_TAMPERED"):
        validate_research_ir(tampered2)


async def test_bug005_override_then_build_succeeds_with_unverified_receipt(tmp_path, monkeypatch):
    run_id = _make_verified_vault_run(tmp_path, monkeypatch)
    pipe = CanonicalResearchIngestionPipeline(tmp_path)
    ir = pipe.ingest(IngestionRequest(vault_run_ids=[run_id], title="BUG-005 rebuild test"))
    first_const = ir.constants[0]

    ov = apply_ir_override(
        ir,
        IRReviewOverride(
            constant_patches=[ConstantOverridePatch(constant_id=first_const.constant_id, si_value=float(first_const.si_value) * 1.25)],
            approve=True,
        ),
    )
    assert ov.epistemic_verdict == "USER_MODIFIED_UNVERIFIED"

    arch = BuilderArchitectureSpec(
        project_name="ov_test_pkg",
        profiles=["hybrid_compute_library"],
        enable_cpp_cabi=False,
        enable_adjoint_jacobian=False,
    )
    uast = BuilderPlanner().compile(ov, arch)
    sm = BuilderStateMachine(tmp_path)
    state = await sm.run(
        BuildRequest(
            build_id="build_ov_test",
            ir=ov,
            uast=uast,
            architecture_spec=arch,
            auto_approve_clean_ir=True,
        )
    )
    assert state.status in {"COMPLETED", "COMPLETED_WITH_FALLBACKS"}, f"Build failed: {state.error_code} {state.error}"
    assert state.receipt is not None
    ver = state.receipt["verification"]
    assert ver.get("research_parity") == "NOT_APPLICABLE_USER_MODIFIED"
    assert ver.get("evidence", {}).get("ir_verdict") == "USER_MODIFIED_UNVERIFIED"
    assert ver.get("evidence", {}).get("invalidating_entries", 0) >= 1

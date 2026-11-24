"""Only receipt-checked, persisted prior discoveries count as duplicates."""
import pytest

from trap_control_cases import allen_eggers_model,oregonator_model,run_case


@pytest.mark.asyncio
async def test_5a_oregonator_renamed_permuted_refactored_clone(tmp_path):
    _,original,_ = await run_case(tmp_path,oregonator_model("oregonator_original"),"VERIFIED_DISCOVERY")
    _,clone,_ = await run_case(tmp_path,oregonator_model("trap5a_oregonator_clone",renamed=True),"INCONCLUSIVE_UNVERIFIED")
    assert clone.uniqueness_audit.cds_model_hash == original.uniqueness_audit.cds_model_hash
    assert clone.uniqueness_audit.vault_class == "EXACT_ALGEBRAIC_DUPLICATE"


@pytest.mark.asyncio
async def test_5b_km_rescaled_allen_eggers_clone(tmp_path):
    _,original,_ = await run_case(tmp_path,allen_eggers_model("allen_eggers_original"),"VERIFIED_DISCOVERY")
    _,clone,_ = await run_case(tmp_path,allen_eggers_model("trap5b_allen_eggers_km",kilometers=True),"INCONCLUSIVE_UNVERIFIED")
    assert clone.uniqueness_audit.cds_model_hash == original.uniqueness_audit.cds_model_hash
    assert clone.uniqueness_audit.vault_class == "EXACT_ALGEBRAIC_DUPLICATE"


@pytest.mark.asyncio
async def test_5c_cubic_oregonator_extension_compared_with_executed_prior(tmp_path):
    _,original,_ = await run_case(tmp_path,oregonator_model("oregonator_prior"),"VERIFIED_DISCOVERY")
    bp = oregonator_model("trap5c_oregonator_cubic",cubic=True)
    bp.ihcg_evidence["execution"] = {"compared_run_id":"oregonator_prior"}
    _,extension,_ = await run_case(tmp_path,bp,"VERIFIED_DISCOVERY")
    assert extension.uniqueness_audit.cds_model_hash != original.uniqueness_audit.cds_model_hash
    assert extension.uniqueness_audit.vault_class == "VALID_COUPLED_EXTENSION"
    assert extension.uniqueness_audit.related_run_id == "@research:oregonator_prior"
    assert extension.significance_audit.passed
    assert extension.significance_audit.null_baseline_description == "Executed verified prior model"

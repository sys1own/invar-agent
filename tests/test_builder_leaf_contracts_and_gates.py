"""UPG-016 (BUG-024, 025, 026, NF-13): Leaf slot contracts, degenerate-output rejection, and SMT scope honesty."""
from __future__ import annotations

import pytest

from invar_agent.builder.ingestion.schemas import (
    EquationRole,
    ExtractedEquation,
    SourceSpan,
)
from invar_agent.builder.leaf_contracts import (
    emit_slot_pytest_source,
    validate_leaf_slot_contract,
    verify_leaf_outputs_against_contract,
)
from invar_agent.builder.uast_models import (
    LLMExtensionSlotRequest,
    SlotExample,
    SlotOutputSpec,
)


def test_upg016_functional_slot_requires_contract_narrative_exempt():
    bad_slot = LLMExtensionSlotRequest(
        slot_id="slot_data_adapter",
        slot_kind="bespoke_data_adapter",
        intent_summary="parse checkpoints",
        fallback_emitter_id="fb_csv_json_adapter",
    )
    with pytest.raises(ValueError, match="LEAF_CONTRACT_REQUIRED"):
        validate_leaf_slot_contract(bad_slot)

    doc_slot = LLMExtensionSlotRequest(
        slot_id="slot_narrative",
        slot_kind="narrative_documentation",
        intent_summary="write readme",
        fallback_emitter_id="fb_markdown_narrative_doc",
    )
    validate_leaf_slot_contract(doc_slot)  # exempt


def test_upg016_leaf_contract_rejects_degenerate_constant_999_and_emits_pytest():
    slot = LLMExtensionSlotRequest(
        slot_id="slot_regime_adapter",
        slot_kind="bespoke_data_adapter",
        intent_summary="classify thermal regime",
        fallback_emitter_id="fb_csv_json_adapter",
        output_spec=[
            SlotOutputSpec(name="E_vis_J", scalar_type="f64", range=(0.0, 1e6)),
            SlotOutputSpec(name="regime", scalar_type="enum", enum=["CRYO", "AMBIENT"]),
        ],
        postconditions=["E_vis_J >= 0"],
        examples=[
            SlotExample(inputs={"T_K": 4.2}, expected={"E_vis_J": 4.2, "regime": "CRYO"}),
            SlotExample(inputs={"T_K": 300.0}, expected={"E_vis_J": 300.0, "regime": "AMBIENT"}),
        ],
    )

    # Constant degenerate function returning 999 fails example check and degenerate check
    def bad_const(_inp):
        return {"E_vis_J": 999.0, "regime": "CRYO"}

    with pytest.raises(ValueError, match="LEAF_EXAMPLE_MISMATCH"):
        verify_leaf_outputs_against_contract(bad_const, slot)

    # Degenerate constant with postcondition-only slot fails LEAF_DEGENERATE_OUTPUT
    pc_only = slot.model_copy(update={"examples": []})
    with pytest.raises(ValueError, match="LEAF_DEGENERATE_OUTPUT"):
        verify_leaf_outputs_against_contract(
            bad_const,
            pc_only,
            extra_probe_inputs=[{"T_K": 4.2}, {"T_K": 77.0}, {"T_K": 300.0}],
        )

    # Valid implementation passes
    def good_fn(inp):
        t = float(inp["T_K"])
        return {"E_vis_J": t, "regime": "CRYO" if t < 50.0 else "AMBIENT"}

    res = verify_leaf_outputs_against_contract(good_fn, slot)
    assert res["passed"] is True and res["checked_cases"] == 2

    test_src = emit_slot_pytest_source(slot, "pkg.slot_regime_adapter", "run_slot")
    assert "def test_slot_regime_adapter_contract_examples" in test_src


def test_nf13_extracted_equation_smt_scope_zero_drift():
    prov = SourceSpan(file="vault:r1/blueprint.json", location="L1", quote="x = 1")
    eq_default = ExtractedEquation(
        equation_id="e0",
        raw_latex="x = 1",
        canonical_latex="x = 1",
        role=EquationRole.ALGEBRAIC_CLOSURE,
        provenance=prov,
    )
    assert "smt_scope" not in eq_default.model_dump(mode="json")

    eq_asserted = eq_default.model_copy(update={"smt_scope": "ASSERTED", "z3_satisfiable": True})
    assert eq_asserted.model_dump(mode="json")["smt_scope"] == "ASSERTED"

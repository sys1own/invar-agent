"""Deterministic lowering from sealed `SciIR` (`CanonicalResearchIR`) to Builder Studio UAST."""
from __future__ import annotations

from typing import Any

from invar_agent.ir.ihcg_bridge import SealedSciIR, seal_sciir_ihcg
from invar_agent.ir.schema import SciIR, SlotRole


def lower_sciir_to_builder_uast(
    sealed_or_ir: SealedSciIR | SciIR,
    *,
    target_languages: list[str] | None = None,
) -> dict[str, Any]:
    """Deterministically lower a sealed `SciIR` into a Builder Studio UAST descriptor
    preserving the exact `0x01 || JCS(SciIR)` BLAKE3 commitment."""
    sealed = (
        sealed_or_ir
        if isinstance(sealed_or_ir, SealedSciIR)
        else seal_sciir_ihcg(sealed_or_ir)
    )
    ir = sealed.ir
    langs = list(target_languages or ir.builder_studio.target_languages or ["python"])

    state_symbols = [s for s in ir.symbols if s.role == SlotRole.STATE]
    param_symbols = [s for s in ir.symbols if s.role == SlotRole.CONTROL_PARAM]
    const_symbols = [s for s in ir.symbols if s.role == SlotRole.CONSTANT]
    obs_symbols = [s for s in ir.symbols if s.role == SlotRole.DERIVED_OBSERVABLE]

    return {
        "schema_version": "invar.builder.uast.v2",
        "ir_id": ir.ir_id,
        "sciir_commitment_blake3": sealed.sciir_commitment_blake3,
        "ihcg_v1_commitment_blake3": sealed.ihcg_v1_commitment_blake3,
        "archetype": ir.archetype.value,
        "domain": ir.domain.value,
        "macro_domain": ir.to_macro_domain(),
        "target_languages": langs,
        "export_workbench_ui": ir.builder_studio.export_workbench_ui,
        "buffer_layout": {
            "n_states": len(state_symbols),
            "n_params": len(param_symbols),
            "n_constants": len(const_symbols),
            "n_observables": len(obs_symbols),
            "states": [
                {"name": s.name, "slot": s.slot_index, "initial": s.initial_or_default, "unit": s.dimension.unit_label}
                for s in state_symbols
            ],
            "params": [
                {
                    "name": s.name,
                    "slot": s.slot_index,
                    "default": s.initial_or_default,
                    "domain": s.domain.model_dump() if s.domain else None,
                    "unit": s.dimension.unit_label,
                }
                for s in param_symbols
            ],
            "constants": [
                {"name": s.name, "slot": s.slot_index, "value": s.initial_or_default, "unit": s.dimension.unit_label}
                for s in const_symbols
            ],
            "observables": [
                {"name": s.name, "slot": s.slot_index, "unit": s.dimension.unit_label}
                for s in obs_symbols
            ],
        },
        "equations": {
            "let_bindings": [lb.model_dump(mode="json") for lb in ir.let_bindings],
            "dynamics_or_residuals": [eq.model_dump(mode="json") for eq in ir.dynamics_or_residuals],
            "observables": [obs.model_dump(mode="json") for obs in ir.observables],
        },
        "obligations": [ob.model_dump(mode="json") for ob in ir.obligations],
        "conservation_pools": [p.model_dump(mode="json") for p in ir.conservation_pools],
    }

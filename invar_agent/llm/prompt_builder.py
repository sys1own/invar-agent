"""Telemetry prompt builder: envelopes with overlay, SMT, and HIN telemetry."""
from __future__ import annotations

import json
from typing import Any, Optional


class TelemetryPromptBuilder:
    """Assembles structured telemetry blocks interpolated into system prompts."""

    def build_envelope(
        self,
        node: Any,
        stage_outputs: Optional[dict[str, Any]] = None,
        boundary_digest: str = "",
        hin_summary: Optional[dict[str, Any]] = None,
        symbol_index: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        envelope: dict[str, Any] = {
            "node_id": getattr(node, "node_id", ""),
            "assigned_tier": getattr(node, "assigned_tier", "TIER_1_FLASH"),
            "holographic_boundary_digest": boundary_digest,
            "hin_interaction_summary": hin_summary or {},
            "workspace_symbol_index": symbol_index or {},
            "stage_outputs": stage_outputs or {},
        }
        contract = getattr(node, "verification_contract", None)
        if contract is not None:
            envelope["verification_contract"] = contract.model_dump()
        return envelope

    def inject_overlay_telemetry(self, envelope: dict[str, Any], overlay_state: dict[str, Any]) -> dict[str, Any]:
        envelope["overlay_state"] = {
            "staged_files": sorted((overlay_state.get("staged_files") or {}).keys()),
            "staged_patches": len(overlay_state.get("staged_patches") or []),
            "pre_holographic_boundary": overlay_state.get("pre_holographic_boundary", ""),
            "post_holographic_boundary": overlay_state.get("post_holographic_boundary", ""),
        }
        return envelope

    def inject_smt_obligations(self, envelope: dict[str, Any], obligations: list[dict[str, Any]]) -> dict[str, Any]:
        envelope["smt_obligations"] = obligations
        return envelope

    def format_for_llm(self, envelope: dict[str, Any]) -> str:
        return json.dumps(envelope, indent=2, default=str)

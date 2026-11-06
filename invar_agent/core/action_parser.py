"""Action parsing: strict JSON envelope parsing for planner and executor outputs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from invar_agent.contracts import ASTSurgicalPatch, PortSpec, TaskWavefrontNode, VerificationContract


@dataclass
class ParsedAgentAction:
    action_type: str
    payload: dict[str, Any]
    raw_text: str = ""


@dataclass
class WavefrontPlanEnvelope:
    plan_id: str = ""
    architectural_rationale: str = ""
    global_holographic_invariant: str = ""
    nodes: list[TaskWavefrontNode] = field(default_factory=list)


@dataclass
class WavefrontActionEnvelope:
    node_id: str = ""
    execution_summary: str = ""
    patches: list[ASTSurgicalPatch] = field(default_factory=list)
    claimed_postconditions: list[str] = field(default_factory=list)
    chiasmus_attestations: list[dict[str, Any]] = field(default_factory=list)
    sandbox_commands: list[dict[str, Any]] = field(default_factory=list)


class StrictActionParser:
    """Parses LLM JSON envelopes into typed kernel contracts."""

    def parse_wavefront_plan(self, text: str) -> WavefrontPlanEnvelope:
        data = self._extract_json_object(text)
        nodes: list[TaskWavefrontNode] = []
        for raw in data.get("wavefront_nodes", []) or []:
            contract_raw = raw.get("verification_contract", {}) or {}
            contract = VerificationContract(
                foge_guard_pre=list(contract_raw.get("foge_guard_pre", []) or []),
                smt_preconditions=list(contract_raw.get("smt_preconditions", []) or []),
                smt_postconditions=list(contract_raw.get("smt_postconditions", []) or []),
                chiasmus_pairs=[tuple(p) for p in (contract_raw.get("chiasmus_pairs", []) or [])],
                concolic_targets=list(contract_raw.get("concolic_targets", []) or []),
                sandbox_verification_cmd=contract_raw.get("sandbox_verification_cmd"),
                smt_timeout_ms=int(contract_raw.get("smt_timeout_ms", 5000) or 5000),
            )
            input_ports = [
                PortSpec(port_name=str(p.get("port_name", "")),
                         source_node_id=p.get("source_node_id"),
                         symbol_contract=str(p.get("symbol_contract", "")))
                for p in (raw.get("input_ports", []) or [])
            ]
            output_ports = [
                PortSpec(port_name=str(p.get("port_name", "")),
                         symbol_contract=str(p.get("symbol_contract", "")))
                for p in (raw.get("output_ports", []) or [])
            ]
            nodes.append(TaskWavefrontNode(
                node_id=str(raw.get("node_id", f"node_{len(nodes)}")),
                wavefront_generation=int(raw.get("wavefront_generation", 0) or 0),
                assigned_tier=str(raw.get("assigned_tier", "TIER_1_FLASH")),
                description=str(raw.get("description", "")),
                input_ports=input_ports,
                output_ports=output_ports,
                read_set_paths=[str(p) for p in (raw.get("read_set_paths", []) or [])],
                write_set_paths=[str(p) for p in (raw.get("write_set_paths", []) or [])],
                target_ast_symbols=[str(s) for s in (raw.get("target_ast_symbols", []) or [])],
                modifies_holographic_boundary=bool(raw.get("modifies_holographic_boundary", False)),
                expected_boundary_delta=[str(s) for s in (raw.get("expected_boundary_delta", []) or [])],
                verification_contract=contract,
            ))
        return WavefrontPlanEnvelope(
            plan_id=str(data.get("plan_id", "")),
            architectural_rationale=str(data.get("architectural_rationale", "")),
            global_holographic_invariant=str(data.get("global_holographic_invariant", "")),
            nodes=nodes,
        )

    def parse_surgical_patch(self, text: str) -> WavefrontActionEnvelope:
        data = self._extract_json_object(text)
        patches: list[ASTSurgicalPatch] = []
        for raw in data.get("ast_patches", []) or []:
            try:
                patches.append(ASTSurgicalPatch(
                    file_path=str(raw.get("file_path", "")),
                    operation=str(raw.get("operation", "REPLACE_EXACT_SPAN")),
                    target_symbol=raw.get("target_symbol"),
                    search_block=str(raw.get("search_block", "") or ""),
                    replacement_code=str(raw.get("replacement_code", "") or ""),
                ))
            except Exception:  # noqa: BLE001 - skip malformed patch entries
                continue
        return WavefrontActionEnvelope(
            node_id=str(data.get("node_id", "")),
            execution_summary=str(data.get("execution_summary", "")),
            patches=patches,
            claimed_postconditions=[str(s) for s in (data.get("claimed_postconditions_satisfied", []) or [])],
            chiasmus_attestations=list(data.get("chiasmus_symmetry_attestation", []) or []),
            sandbox_commands=list(data.get("sandbox_commands", []) or []),
        )

    def parse_actions(self, text: str) -> list[ParsedAgentAction]:
        data = self._extract_json_object(text)
        actions: list[ParsedAgentAction] = []
        for item in data.get("actions", []) or []:
            if isinstance(item, dict):
                actions.append(ParsedAgentAction(
                    action_type=str(item.get("type", item.get("action", "UNKNOWN"))),
                    payload=item,
                    raw_text=json.dumps(item),
                ))
        if not actions and data:
            actions.append(ParsedAgentAction(action_type="ENVELOPE", payload=data, raw_text=text))
        return actions

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        candidate = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
        if fence:
            candidate = fence.group(1)
        else:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start : end + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM output is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("LLM output JSON must be an object")
        return data

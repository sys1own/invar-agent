"""Healing context builder: minimal fault slices and diagnostic payloads."""
from __future__ import annotations

from typing import Any, Optional

from .error_classifier import ClassifiedErrorPayload


class HealingContextBuilder:
    """Builds compact, LLM-ready diagnostic contexts for healing tiers."""

    def __init__(self, max_slice_symbols: int = 12) -> None:
        self.max_slice_symbols = max_slice_symbols

    def extract_minimal_fault_slice(self, hin_graph, fault_node_id: str) -> list[str]:
        """Backward causal slice from the fault through the HIN graph."""
        if hin_graph is None:
            return []
        try:
            return list(hin_graph.trace_fault_propagation(fault_node_id, max_hops=self.max_slice_symbols))[: self.max_slice_symbols]
        except Exception:  # noqa: BLE001
            return [fault_node_id]

    def build_diagnostic_prompt_payload(
        self,
        classified: ClassifiedErrorPayload,
        sources: dict[str, str],
        fault_slice: Optional[list[str]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": classified.category.value,
            "error_code": classified.error_code,
            "diagnostic": classified.diagnostic_msg,
            "failing_file": classified.failing_file,
            "failing_line": classified.failing_line,
            "stack_trace_frames": classified.stack_trace_frames[-5:],
            "fault_slice": fault_slice or [],
            "sources": {
                path: self._trim_source(src)
                for path, src in sources.items()
            },
        }
        if extra:
            payload.update(extra)
        return payload

    def _trim_source(self, source: str, max_chars: int = 6000) -> str:
        if len(source) <= max_chars:
            return source
        return source[: max_chars // 2] + "\n# ... [truncated] ...\n" + source[-max_chars // 2 :]

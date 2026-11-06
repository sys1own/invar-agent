"""Surgical LLM healer: generates and validates minimal AST repair patches."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from invar_agent.contracts import ASTSurgicalPatch

from .error_classifier import ClassifiedErrorPayload


@dataclass
class LLMRepairResult:
    success: bool
    patched_sources: dict[str, str] = field(default_factory=dict)
    patches: list[ASTSurgicalPatch] = field(default_factory=list)
    rationale: str = ""
    tokens_used: int = 0
    unrecoverable: bool = False


class SurgicalLLMHealer:
    """Tier-1 surgical healer backed by an injectable LLM client."""

    def __init__(self, client=None, system_prompt: str = "") -> None:
        self.client = client
        self.system_prompt = system_prompt

    def generate_ast_repair_patch(
        self,
        classified: ClassifiedErrorPayload,
        sources: dict[str, str],
        prompt_payload: Optional[dict[str, Any]] = None,
        max_tokens: int = 4096,
    ) -> LLMRepairResult:
        if self.client is None:
            return LLMRepairResult(success=False, rationale="no LLM client configured")
        prompt = self._build_prompt(classified, sources, prompt_payload)
        try:
            response = self.client.complete(prompt, system=self.system_prompt or None, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            return LLMRepairResult(success=False, rationale=f"LLM call failed: {e}")
        content = response.content if hasattr(response, "content") else str(response)
        tokens = getattr(response, "output_tokens", 0) or max(1, len(content) // 4)
        return self._parse_response(content, sources, tokens)

    def validate_healed_candidate(
        self,
        original_sources: dict[str, str],
        patched_sources: dict[str, str],
    ) -> bool:
        """Syntax validity + no empty-file regressions."""
        from ..guards.syntax_guard import MultiLangSyntaxGuard

        guard = MultiLangSyntaxGuard()
        for path, content in patched_sources.items():
            if path.endswith(".py"):
                res = guard.verify_python_cst(content)
                if not res.valid:
                    return False
        # Must not delete entire files.
        for path, original in original_sources.items():
            patched = patched_sources.get(path)
            if patched is None and original.strip():
                return False
        return True

    # -- internals -----------------------------------------------------------------

    def _build_prompt(self, classified: ClassifiedErrorPayload, sources: dict[str, str],
                      payload: Optional[dict[str, Any]]) -> str:
        base = {
            "category": classified.category.value,
            "error_code": classified.error_code,
            "diagnostic": classified.diagnostic_msg,
            "failing_file": classified.failing_file,
            "failing_line": classified.failing_line,
        }
        if payload:
            base.update(payload)
        # Include only the failing file (plus small context) to bound tokens.
        files: dict[str, str] = {}
        if classified.failing_file and classified.failing_file in sources:
            files[classified.failing_file] = sources[classified.failing_file]
        else:
            files = dict(list(sources.items())[:3])
        envelope = {"healing_telemetry": base, "sources": files}
        return json.dumps(envelope, indent=2, default=str)

    def _parse_response(self, content: str, sources: dict[str, str], tokens: int) -> LLMRepairResult:
        data = self._extract_json(content)
        if data is None:
            return LLMRepairResult(success=False, rationale="unparseable LLM response", tokens_used=tokens)
        if data.get("unrecoverable"):
            return LLMRepairResult(success=False, unrecoverable=True,
                                   rationale=str(data.get("architectural_rationale", "unrecoverable")),
                                   tokens_used=tokens)
        patches_raw = data.get("ast_patches") or []
        patches: list[ASTSurgicalPatch] = []
        patched_sources = dict(sources)
        # Full-file replacements take precedence.
        replacements = data.get("replacements") or {}
        if isinstance(replacements, dict):
            for path, code in replacements.items():
                if isinstance(code, str):
                    patched_sources[path] = code
        for raw in patches_raw:
            if not isinstance(raw, dict):
                continue
            try:
                patch = ASTSurgicalPatch(
                    file_path=str(raw.get("file_path", "")),
                    operation=str(raw.get("operation", "REPLACE_EXACT_SPAN")),
                    target_symbol=raw.get("target_symbol"),
                    search_block=str(raw.get("search_block", "")),
                    replacement_code=str(raw.get("replacement_code", "")),
                )
            except Exception:  # noqa: BLE001
                continue
            patches.append(patch)
            # Apply optimistically to produce patched sources for validation.
            target = patched_sources.get(patch.file_path, "")
            from ..overlay.patch import apply_single_patch

            result = apply_single_patch(target, patch)
            if result.applied:
                patched_sources[patch.file_path] = result.source
        if not patches and not replacements:
            return LLMRepairResult(success=False, rationale="no patches in response", tokens_used=tokens)
        if not self.validate_healed_candidate(sources, patched_sources):
            return LLMRepairResult(success=False, rationale="healed candidate failed validation",
                                   patches=patches, tokens_used=tokens)
        return LLMRepairResult(
            success=True,
            patched_sources=patched_sources,
            patches=patches,
            rationale=str(data.get("root_cause_analysis", data.get("execution_summary", "")))[:500],
            tokens_used=tokens,
        )

    def _extract_json(self, content: str) -> Optional[dict[str, Any]]:
        text = content.strip()
        # Fenced code block.
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

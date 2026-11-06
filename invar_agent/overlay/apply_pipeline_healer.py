"""Healing hooks for the apply pipeline: malformed hunk repair and retry."""
from __future__ import annotations

import re
from typing import Optional

from invar_agent.contracts import ASTSurgicalPatch


class ApplyPipelineHealer:
    """Repairs structurally malformed patches before they reach the guards."""

    def repair_malformed_hunk(self, patch: ASTSurgicalPatch) -> ASTSurgicalPatch:
        """Normalizes replacement code: trailing newline, indent realignment."""
        code = patch.replacement_code
        if not code.endswith("\n"):
            code = code + "\n"
        code = self.realign_ast_indentation(code)
        return patch.model_copy(update={"replacement_code": code})

    def realign_ast_indentation(self, code: str) -> str:
        """Fixes tabs/spaces mixtures and dedents over-indented blocks by one level."""
        lines = code.splitlines()
        if not lines:
            return code
        # Detect dominant indent unit.
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        if not indents:
            return code
        min_indent = min(indents)
        # If the whole block is indented (e.g. LLM emitted body-level indent),
        # dedent by the minimum common indent so top-level defs start at column 0.
        out: list[str] = []
        for line in lines:
            if line.strip():
                current = len(line) - len(line.lstrip())
                strip = min(current, min_indent)
                out.append(line[strip:])
            else:
                out.append(line)
        # Convert tabs to 4 spaces.
        return "\n".join(l.replace("\t", "    ") for l in out) + ("\n" if code.endswith("\n") else "")

    def heal_with_retry(
        self,
        patch: ASTSurgicalPatch,
        validator,
        max_attempts: int = 3,
    ) -> tuple[Optional[ASTSurgicalPatch], list[str]]:
        """Attempts deterministic repairs until the buffer validates."""
        attempts: list[str] = []
        current = patch
        for _ in range(max_attempts):
            current = self.repair_malformed_hunk(current)
            violations = validator.validate_staged_buffer(patch.file_path, current.replacement_code)
            if not violations:
                return current, attempts
            attempts.append("; ".join(f"{v.rule}@{v.line}" for v in violations[:3]))
            # Try stripping trailing whitespace-only lines as a last resort.
            stripped = current.replacement_code.rstrip() + "\n"
            if stripped != current.replacement_code:
                current = current.model_copy(update={"replacement_code": stripped})
            else:
                break
        return None, attempts

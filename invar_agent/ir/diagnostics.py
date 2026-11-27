"""UPG-004: Structured lowering diagnostics and typed LoweringError."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class Diagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    field: str
    message: str
    symbol: Optional[str] = None
    hint: str = ""


class LoweringError(ValueError):
    """Typed compiler/lowering error carrying structured `Diagnostic` objects."""

    def __init__(self, diagnostics: List[Diagnostic]) -> None:
        self.diagnostics = list(diagnostics)
        summary = "; ".join(f"{d.code} ({d.field}): {d.message}" for d in self.diagnostics)
        super().__init__(summary or "LOWERING_FAILED")

"""Unified, deterministic Research-to-Builder ingestion."""
from .pipeline import CanonicalResearchIngestionPipeline, apply_ir_override
from .schemas import CanonicalResearchIR, IngestionRequest, IRReviewOverride

__all__ = ["CanonicalResearchIngestionPipeline", "CanonicalResearchIR", "IngestionRequest",
           "IRReviewOverride", "apply_ir_override"]

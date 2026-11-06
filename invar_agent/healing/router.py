"""Healing tier router: novelty scoring and strategy selection."""
from __future__ import annotations

import blake3
from typing import Optional

from .error_classifier import FailureCategory


class HealingTierRouter:
    """Decides Tier-0 reflex vs Tier-1 flash vs Tier-2 pro for a failure."""

    def __init__(self, reflex_confidence_threshold: float = 0.9) -> None:
        self.threshold = reflex_confidence_threshold
        self._seen_signatures: dict[str, int] = {}

    def compute_failure_novelty(self, category: FailureCategory, diagnostic: str) -> float:
        """0.0 = seen before (reflex-friendly), 1.0 = novel (escalate)."""
        key = f"{category.value}|{' '.join(diagnostic.split())[:160]}"
        digest = blake3.blake3(key.encode()).hexdigest()
        count = self._seen_signatures.get(digest, 0)
        self._seen_signatures[digest] = count + 1
        if count == 0:
            return 0.6  # first sighting: moderately novel
        if count == 1:
            return 0.3
        return 0.1

    def select_healing_strategy(
        self,
        category: FailureCategory,
        diagnostic: str,
        reflex_available: bool,
        reflex_confidence: float = 1.0,
        tier_1_failures: int = 0,
        modifies_holographic_boundary: bool = False,
    ) -> str:
        novelty = self.compute_failure_novelty(category, diagnostic)
        if reflex_available and reflex_confidence >= self.threshold and tier_1_failures == 0 and not modifies_holographic_boundary:
            return "TIER_0_REFLEX"
        if tier_1_failures >= 2 or modifies_holographic_boundary or novelty > 0.75:
            return "TIER_2_PRO"
        return "TIER_1_FLASH"

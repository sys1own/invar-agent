"""invar-agent: Mathematically Verified Autonomous Agent Kernel and Web Console."""
from __future__ import annotations

__version__ = "0.1.0"

from .contracts import (
    AgentStateStage,
    ASTSurgicalPatch,
    HealingVerdict,
    OverlayTransaction,
    PortSpec,
    ShieldVerificationReport,
    TaskWavefrontNode,
    VerificationContract,
)

__all__ = [
    "AgentStateStage", "ASTSurgicalPatch", "HealingVerdict", "OverlayTransaction",
    "PortSpec", "ShieldVerificationReport", "TaskWavefrontNode", "VerificationContract",
    "VerifiedAutonomousAgent", "KernelConfig",
]


class KernelConfig:
    """Top-level kernel configuration."""

    def __init__(
        self,
        workspace_root: str = ".",
        pro_model: str = "gemini-3.1-pro-preview",
        flash_model: str = "gemini-3.8-flash",
        max_pro_tokens: int = 250_000,
        max_flash_tokens: int = 1_000_000,
    ) -> None:
        self.workspace_root = workspace_root
        self.pro_model = pro_model
        self.flash_model = flash_model
        self.max_pro_tokens = max_pro_tokens
        self.max_flash_tokens = max_flash_tokens


def __getattr__(name: str):
    if name == "VerifiedAutonomousAgent":
        from .core.agent import VerifiedAutonomousAgent

        return VerifiedAutonomousAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

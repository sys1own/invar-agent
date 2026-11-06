"""Precision Shield: FOGE + Z3 SMT + Concolic + Chiasmus + Holographic + Rust gates."""
from .adjoint import AdjointComputationResult, AdjointStateOperator
from .chiasmus import ChiasticFrame, ChiasmusResult, ChiasmusSymmetryVerifier, ChiasmusViolation
from .concolic import ConcolicExecutor, ConcolicFinding, ConcolicReport, PathCondition, PathConstraintCollector, SymbolicShadowValue, negate_and_solve_branch
from .contract_synth import ContractSynthesizer
from .foge import FOGEResult, FirstOrderGuardedExecutor, PreConditionResult
from .holographic import HolographicVerifier, extract_boundary_signature
from .rust_shield import RustShieldReport, RustSourceShield
from .shield import PrecisionShieldOrchestrator, ShieldContext
from .smt_engine import RelationalInvariant, SymbolicVerificationEngine, SymbolicVerdict
from .smt_solver import SMTResult, SMTStatus, SMTSolverBridge
from .spec import FormalSpec, FormalVerificationObligation, InvariantClause, ObligationStatus, SpecVerifier

__all__ = [
    "AdjointComputationResult", "AdjointStateOperator",
    "ChiasticFrame", "ChiasmusResult", "ChiasmusSymmetryVerifier", "ChiasmusViolation",
    "ConcolicExecutor", "ConcolicFinding", "ConcolicReport", "PathCondition", "PathConstraintCollector",
    "SymbolicShadowValue", "negate_and_solve_branch",
    "ContractSynthesizer",
    "FOGEResult", "FirstOrderGuardedExecutor", "PreConditionResult",
    "HolographicVerifier", "extract_boundary_signature",
    "RustShieldReport", "RustSourceShield",
    "PrecisionShieldOrchestrator", "ShieldContext",
    "RelationalInvariant", "SymbolicVerificationEngine", "SymbolicVerdict",
    "SMTResult", "SMTStatus", "SMTSolverBridge",
    "FormalSpec", "FormalVerificationObligation", "InvariantClause", "ObligationStatus", "SpecVerifier",
]

"""Declarative ResearchComputeUAST schemas, CSEL parser, and deterministic lowering compiler."""

from .compiler import (
    CompiledUASTResult,
    analyze_uast_with_hin,
    apply_perturbation_delta,
    compile_uast_to_python,
)
from .csel import (
    CSEL_INTRINSICS,
    CselSyntaxError,
    collect_csel_symbols,
    csel_ast_to_python,
    csel_ast_to_str,
    parse_csel_to_ast,
)
from .schemas import (
    CounterexampleDiagnosticCard,
    InvariantContractSpec,
    OntologyAndParameterSpec,
    OperatorMutation,
    PhysicalQuantity,
    ResearchComputeUAST,
    SymbolicGovernanceMap,
    UASTPerturbationDelta,
)

__all__ = [
    "CSEL_INTRINSICS",
    "CompiledUASTResult",
    "CounterexampleDiagnosticCard",
    "CselSyntaxError",
    "InvariantContractSpec",
    "OntologyAndParameterSpec",
    "OperatorMutation",
    "PhysicalQuantity",
    "ResearchComputeUAST",
    "SymbolicGovernanceMap",
    "UASTPerturbationDelta",
    "analyze_uast_with_hin",
    "apply_perturbation_delta",
    "collect_csel_symbols",
    "compile_uast_to_python",
    "csel_ast_to_python",
    "csel_ast_to_str",
    "parse_csel_to_ast",
]

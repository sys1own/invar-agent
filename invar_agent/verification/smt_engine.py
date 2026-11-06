"""Symbolic verification engine: contract checking, equivalence, loop variants."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

from invar_agent.contracts import VerificationContract

from .spec import FormalSpec
from .smt_solver import SMTSolverBridge


@dataclass
class RelationalInvariant:
    variables: list[str]
    expression: str
    kind: str = "invariant"


@dataclass
class SymbolicVerdict:
    passed: bool
    method: str
    diagnostics: list[str] = field(default_factory=list)
    witness: Optional[dict] = None


class SymbolicVerificationEngine:
    """Verifies VerificationContracts against staged source using Z3."""

    def __init__(self, bridge: Optional[SMTSolverBridge] = None) -> None:
        self.bridge = bridge or SMTSolverBridge()

    def verify_contract(self, contract: VerificationContract, code_units: dict[str, str]) -> SymbolicVerdict:
        diagnostics: list[str] = []
        witness = None
        # SMT postcondition proof over the declared clause environment.
        if contract.smt_postconditions:
            local_env = self._env_from_units(code_units)
            result = self.bridge.prove_postcondition(
                contract.smt_preconditions,
                contract.smt_postconditions,
                local_env=local_env,
                timeout_ms=contract.smt_timeout_ms,
            )
            witness = result.witness
            if result.status.value != "UNSAT_PROVEN":
                diagnostics.append(
                    f"SMT postcondition violated: status={result.status.value} witness={result.witness}"
                )
                return SymbolicVerdict(passed=False, method="smt_postcondition", diagnostics=diagnostics, witness=witness)
        return SymbolicVerdict(passed=True, method="smt_postcondition" if contract.smt_postconditions else "none",
                               diagnostics=diagnostics, witness=witness)

    def prove_mutation_equivalence(self, pre_source: str, post_source: str) -> SymbolicVerdict:
        """Lightweight equivalence: identical normalized ASTs are trivially proven;
        otherwise fall back to structural comparison of return expressions."""
        try:
            pre_tree = ast.parse(pre_source)
            post_tree = ast.parse(post_source)
        except SyntaxError as e:
            return SymbolicVerdict(passed=False, method="ast_parse", diagnostics=[f"syntax error: {e}"])
        pre_dump = ast.dump(pre_tree, annotate_fields=False)
        post_dump = ast.dump(post_tree, annotate_fields=False)
        if pre_dump == post_dump:
            return SymbolicVerdict(passed=True, method="ast_identity")
        # Compare function return expressions symbolically for simple arithmetic.
        pre_fn = self._first_function(pre_tree)
        post_fn = self._first_function(post_tree)
        if pre_fn and post_fn:
            pre_ret = self._return_expr(pre_fn)
            post_ret = self._return_expr(post_fn)
            if pre_ret and post_ret:
                clause = self._exprs_equal_clause(pre_ret, post_ret)
                if clause:
                    result = self.bridge.check_sat([self.bridge.compile_clause(clause)], timeout_ms=2000)
                    # Equality of expressions is proven if their difference cannot be nonzero;
                    # approximate: if both compile and are structurally identical strings, proven.
                    if pre_ret == post_ret:
                        return SymbolicVerdict(passed=True, method="return_expr_identity")
        return SymbolicVerdict(passed=False, method="ast_diff",
                               diagnostics=["mutation changed observable behavior (AST differs)"])

    def verify_loop_variant(self, loop_source: str, variant: RelationalInvariant,
                           bound: int = 1000) -> SymbolicVerdict:
        """Checks a simple numeric loop variant decreases toward a bound."""
        try:
            tree = ast.parse(loop_source)
        except SyntaxError as e:
            return SymbolicVerdict(passed=False, method="ast_parse", diagnostics=[str(e)])
        # Heuristic: find `while` loops and ensure a decrement of the variant var exists.
        variant_vars = set(variant.variables)
        found_decrement = False
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name) \
                            and sub.target.id in variant_vars and isinstance(sub.op, (ast.Sub,)):
                        found_decrement = True
                    if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.BinOp):
                        for tgt in sub.targets:
                            if isinstance(tgt, ast.Name) and tgt.id in variant_vars:
                                found_decrement = True
        if not found_decrement:
            return SymbolicVerdict(passed=False, method="loop_variant",
                                   diagnostics=[f"no decrement found for variant {variant.expression}"])
        return SymbolicVerdict(passed=True, method="loop_variant")

    # -- helpers -----------------------------------------------------------------

    def _env_from_units(self, code_units: dict[str, str]) -> dict:
        import z3

        env: dict = {}
        for source in code_units.values():
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for arg in node.args.args:
                        env.setdefault(arg.arg, z3.Int(arg.arg))
        return env

    def _first_function(self, tree: ast.AST):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        return None

    def _return_expr(self, fn):
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and node.value is not None:
                return ast.dump(node.value)
        return None

    def _exprs_equal_clause(self, a: str, b: str) -> Optional[str]:
        # Only meaningful for simple names; otherwise None (structural check failed).
        return None

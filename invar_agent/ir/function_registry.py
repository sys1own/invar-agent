"""ARCH-007: Single canonical math function registry.

This single table is the sole authority for:
1. Prompt vocabulary advertised to the SciIR synthesizer (`ir/prompt_templates.py`),
2. Formal gate allow-list and arities (`research/meaningfulness.py`),
3. SciIR -> UAST / Python / C++ / Rust expression lowering (`ir/lower_uast.py`).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MathFunctionSpec:
    name: str
    min_arity: int
    max_arity: Optional[int]  # None => variadic >= min_arity
    py_name: str
    cpp_name: str
    rust_name: str
    formal_supported: bool = True
    prompt_advertised: bool = True
    alias_of: Optional[str] = None
    is_stochastic: bool = False


FUNCTION_REGISTRY: Dict[str, MathFunctionSpec] = {
    "exp": MathFunctionSpec("exp", 1, 1, "math.exp", "std::exp", "f64::exp"),
    "log": MathFunctionSpec("log", 1, 1, "math.log", "std::log", "f64::ln"),
    "ln": MathFunctionSpec("ln", 1, 1, "math.log", "std::log", "f64::ln", alias_of="log"),
    "sqrt": MathFunctionSpec("sqrt", 1, 1, "math.sqrt", "std::sqrt", "f64::sqrt"),
    "abs": MathFunctionSpec("abs", 1, 1, "abs", "std::abs", "f64::abs"),
    "sin": MathFunctionSpec("sin", 1, 1, "math.sin", "std::sin", "f64::sin"),
    "cos": MathFunctionSpec("cos", 1, 1, "math.cos", "std::cos", "f64::cos"),
    "tanh": MathFunctionSpec("tanh", 1, 1, "math.tanh", "std::tanh", "f64::tanh"),
    "step": MathFunctionSpec("step", 1, 1, "step", "invar_step_fn", "invar_step_fn"),
    "min": MathFunctionSpec("min", 2, None, "min", "std::min", "f64::min"),
    "max": MathFunctionSpec("max", 2, None, "max", "std::max", "f64::max"),
    "clamp": MathFunctionSpec("clamp", 3, 3, "clamp", "std::clamp", "f64::clamp"),
    "pow": MathFunctionSpec("pow", 2, 2, "pow", "std::pow", "f64::powf"),
    "logistic_sigmoid": MathFunctionSpec("logistic_sigmoid", 1, 1, "logistic_sigmoid", "logistic_sigmoid", "logistic_sigmoid"),
    "kl_term": MathFunctionSpec("kl_term", 2, 2, "kl_term", "kl_term", "kl_term"),
    "hill_eq": MathFunctionSpec("hill_eq", 3, 3, "hill_eq", "hill_eq", "hill_eq"),
    "michaelis_menten": MathFunctionSpec("michaelis_menten", 3, 3, "michaelis_menten", "michaelis_menten", "michaelis_menten"),
    "implies": MathFunctionSpec("implies", 2, 2, "implies", "implies", "implies"),
    # Seeded stochastic primitives (allowed in MONTE_CARLO / StochasticSimulation)
    "rand_uniform01": MathFunctionSpec(
        "rand_uniform01", 0, 1, "_splitmix64_uniform01", "invar_rand_uniform01", "invar_rand_uniform01",
        formal_supported=True, prompt_advertised=True, is_stochastic=True,
    ),
    "rand_normal01": MathFunctionSpec(
        "rand_normal01", 0, 1, "_splitmix64_normal01", "invar_rand_normal01", "invar_rand_normal01",
        formal_supported=True, prompt_advertised=True, is_stochastic=True,
    ),
    "rand_exponential1": MathFunctionSpec(
        "rand_exponential1", 0, 1, "_splitmix64_exponential1", "invar_rand_exponential1", "invar_rand_exponential1",
        formal_supported=True, prompt_advertised=True, is_stochastic=True,
    ),
}

BUILTIN_CONSTANTS: Tuple[str, ...] = ("pi", "e_euler")


def prompt_function_vocabulary() -> List[str]:
    """Exact ordered list of built-in functions advertised in synthesis prompts."""
    return sorted(
        name for name, spec in FUNCTION_REGISTRY.items()
        if spec.prompt_advertised
    )


def formal_gate_vocabulary() -> List[str]:
    """Exact ordered list of functions accepted by the formal verification gate."""
    return sorted(
        name for name, spec in FUNCTION_REGISTRY.items()
        if spec.formal_supported
    )


def formal_gate_arities() -> Dict[str, Tuple[int, Optional[int]]]:
    """Map of formal-gate function name -> (min_arity, max_arity_or_None)."""
    return {
        name: (spec.min_arity, spec.max_arity)
        for name, spec in FUNCTION_REGISTRY.items()
        if spec.formal_supported
    }


def lower_nary_binary_tree(fn_name: str, args: Sequence[str]) -> str:
    """Lower n-ary ``min``/``max`` into a balanced binary tree."""
    items = list(args)
    if len(items) < 2:
        raise ValueError(f"{fn_name} requires at least 2 arguments, got {len(items)}")
    while len(items) > 1:
        nxt: List[str] = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                nxt.append(f"{fn_name}({items[i]}, {items[i + 1]})")
            else:
                nxt.append(items[i])
        items = nxt
    return items[0]


def lower_nary_ast_tree(fn_name: str, args: Sequence[ast.expr]) -> ast.expr:
    """AST equivalent of ``lower_nary_binary_tree`` for balanced binary min/max."""
    items = list(args)
    if len(items) < 2:
        raise ValueError(f"{fn_name} requires at least 2 arguments")
    while len(items) > 1:
        nxt: List[ast.expr] = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                nxt.append(
                    ast.Call(
                        func=ast.Name(id=fn_name, ctx=ast.Load()),
                        args=[items[i], items[i + 1]],
                        keywords=[],
                    )
                )
            else:
                nxt.append(items[i])
        items = nxt
    return items[0]


class FormalFunctionNormalizer(ast.NodeTransformer):
    """Rewrite registered convenience functions (ln, clamp, n-ary min/max) into canonical formal AST."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.keywords:
            return node
        name = node.func.id
        if name == "ln" and len(node.args) == 1:
            return ast.copy_location(
                ast.Call(func=ast.Name(id="log", ctx=ast.Load()), args=list(node.args), keywords=[]),
                node,
            )
        if name == "clamp" and len(node.args) == 3:
            x_arg, lo_arg, hi_arg = node.args
            inner = ast.Call(func=ast.Name(id="max", ctx=ast.Load()), args=[lo_arg, x_arg], keywords=[])
            return ast.copy_location(
                ast.Call(func=ast.Name(id="min", ctx=ast.Load()), args=[hi_arg, inner], keywords=[]),
                node,
            )
        if name in ("min", "max") and len(node.args) > 2:
            return ast.copy_location(lower_nary_ast_tree(name, node.args), node)
        return node

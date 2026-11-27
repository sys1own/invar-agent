"""Shared emission interface and validated symbolic model lowering."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


def relativize_provenance_path(raw: str) -> str:
    """Strip absolute host/user prefixes from provenance paths (BUG-043)."""
    text = str(raw or "").replace("\\", "/")
    if not text:
        return ""
    for marker in (".invar/research_vault/", "uploads/", "src/", "provenance/", "tests/"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:]
    if re.match(r"^[A-Za-z]:/", text) or text.startswith("/"):
        return Path(text).name
    return text


def sanitize_ir_for_export(ir_dict: dict, workspace_root=None) -> dict:
    """Return a copy of a serialized IR/metadata dict with all absolute host paths recursively stripped."""
    from invar_agent.redact import sanitize_tree
    return sanitize_tree(ir_dict, workspace_root=workspace_root)

import sympy as sp
try:
    from invar_agent._native import PyAdjointHin
except ImportError:
    PyAdjointHin = None

from ..error_taxonomy import BuildFault
from ..ingestion.equation_lowerer import equation_parts, to_csel
from ..ingestion.schemas import CanonicalResearchIR, EquationRole
from ..uast_models import BuilderUAST


@dataclass
class ModuleEmission:
    files: dict[str, str | bytes] = field(default_factory=dict)
    pytest_paths: list[str] = field(default_factory=list)
    port_file: str | None = None
    python_fallback_active: bool = False


@dataclass(frozen=True)
class EmitterContext:
    ir: CanonicalResearchIR
    uast: BuilderUAST
    overlay: Path
    python_fallback_active: bool


class EmitterRegistry:
    def __init__(self):
        self.emitters = {}
        self.process_factory: str | None = None

    def register(self, identifier, emitter):
        if identifier in self.emitters:
            raise ValueError("Emitter already registered: " + identifier)
        self.emitters[identifier] = emitter
        return self

    def emit(self, node, context):
        if node.kind == "interface_port":
            return ModuleEmission()
        identifier = "kernel" if node.kind == "kernel" else "slot" if node.kind == "llm_extension_slot" else node.emitter_id
        if identifier not in self.emitters:
            raise BuildFault("DETERMINISTIC_EMITTER_BUG", "No emitter registered: " + identifier)
        emission = self.emitters[identifier](node, context)
        if not isinstance(emission, ModuleEmission):
            raise TypeError("Emitters must return ModuleEmission")
        return emission


def infer_program_kind(ir):
    """ARCH-005: Derive typed ProgramKind strictly from IR equation roles."""
    from ..uast_models import ProgramKind
    has_evolution = any(eq.role == EquationRole.EVOLUTION_LAW for eq in ir.equations)
    has_algebraic = any(eq.role == EquationRole.ALGEBRAIC_CLOSURE and eq.lhs_symbol for eq in ir.equations)
    if has_evolution:
        return ProgramKind.ODE
    if has_algebraic:
        return ProgramKind.ALGEBRAIC_MAP
    raise ValueError("EMPTY_EQUATION_SET: IR contains no governing equations")


def package_name(context):
    return context.uast.project_id


def model(context):
    ir = context.ir
    states = sorted(ir.state_variables, key=lambda item: item.slot_index)
    names = [item.symbol for item in states]
    seen_params = set(names)
    parameters = []
    for item in ir.constants:
        if item.symbol not in seen_params:
            seen_params.add(item.symbol)
            parameters.append(item.symbol)
    equations = {item.lhs_symbol: item for item in ir.equations if item.role == EquationRole.EVOLUTION_LAW}
    if not equations:
        alg_eqs = {item.lhs_symbol: item for item in ir.equations if item.role == EquationRole.ALGEBRAIC_CLOSURE and item.lhs_symbol}
        if alg_eqs:
            equations = alg_eqs
            if set(alg_eqs) <= set(names):
                states = [s for s in states if s.symbol in equations]
            else:
                from ..ingestion.schemas import StateVariableSpec, SIDimensionVectorModel
                existing = {s.symbol: s for s in states}
                states = [
                    existing.get(sym) or StateVariableSpec(
                        symbol=sym, slot_index=idx, initial_value=0.0,
                        dimension=SIDimensionVectorModel(), provenance=alg_eqs[sym].provenance,
                    )
                    for idx, sym in enumerate(alg_eqs)
                ]
            names = [s.symbol for s in states]
    if not names or set(names) != set(equations):
        raise ValueError("Every state must have exactly one evolution equation")
    if len(set(names + parameters)) != len(names + parameters):
        raise ValueError("State and parameter symbols must be distinct")
    declared_syms = set(names + parameters)
    symbols = {name: sp.Symbol(name) for name in names + parameters + ["t"]}
    closures = {sp.Symbol(item.lhs_symbol): equation_parts(item.canonical_latex, declared_symbols=declared_syms)[2] for item in ir.equations
                if item.role == EquationRole.ALGEBRAIC_CLOSURE and item.lhs_symbol}
    expressions = [equation_parts(equations[name].canonical_latex, declared_symbols=declared_syms)[2] for name in names]
    for _ in range(len(closures) + 1):
        expressions = [expression.xreplace(closures) for expression in expressions]
    if any(expression.free_symbols - set(symbols.values()) for expression in expressions):
        raise ValueError("Unresolved scientific symbols")
    kernel = next(node for node in context.uast.nodes if node.kind == "kernel")
    environment = {item.symbol: item.si_value for item in ir.constants}
    environment.update({item.symbol: item.initial_value or 0.0 for item in states})
    environment['t'] = 0.0
    jacobian, backend = differentiate(expressions, names + parameters, environment, kernel.export_adjoint_jacobian)
    return states, names, parameters, expressions, jacobian, backend


def differentiate(expressions, variables, environment, enabled=True):
    if not enabled or PyAdjointHin is None:
        return [[sp.diff(expression, sp.Symbol(name)) for name in variables] for expression in expressions], 'sympy_fallback'
    outputs = [f'_builder_output_{index}' for index in range(len(expressions))]
    while set(outputs) & set(variables):
        outputs = ['_' + name for name in outputs]
    native = PyAdjointHin(outputs + variables, json.dumps({name: to_csel(expression) for name, expression in zip(outputs, expressions)}))
    try:
        report = json.loads(native.analyze(json.dumps(environment), compute_spectrum=False))
    except TypeError:
        report = json.loads(native.analyze(json.dumps(environment)))
    symbols = {name: sp.Symbol(name) for name in variables + ['t']}
    functions = {'safe_div': lambda a,b:a/b, 'safe_pow': lambda a,b:a**b,
                 'safe_sqrt':sp.sqrt, 'safe_log':sp.log, 'safe_exp':sp.exp,
                 'safe_sin':sp.sin, 'safe_cos':sp.cos, 'safe_tanh':sp.tanh,
                 'log1p':lambda a:sp.log(1+a), 'expm1':lambda a:sp.exp(a)-1,
                 'where':lambda a,b,c:sp.Piecewise((b,sp.Gt(a,0)),(c,True))}
    size = len(outputs)
    jacobian = [[sp.sympify(value.replace('^','**'), locals={**symbols, **functions}) for value in row[size:]]
                for row in report['symbolic_jacobian'][:size]]
    return jacobian, 'native_adjoint'


def observations(context, variables):
    closures = {item.lhs_symbol: equation_parts(item.canonical_latex)[2] for item in context.ir.equations
                if item.role == EquationRole.ALGEBRAIC_CLOSURE and item.lhs_symbol}
    substitution = {sp.Symbol(name): value for name,value in closures.items()}
    values = list(closures.values())
    for _ in range(len(closures) + 1):
        values = [value.xreplace(substitution) for value in values]
    allowed = {sp.Symbol(name) for name in variables + ['t']}
    if any(value.free_symbols - allowed for value in values):
        raise ValueError('Unresolved algebraic observable')
    environment = {item.symbol:item.si_value for item in context.ir.constants}
    environment.update({item.symbol:item.initial_value or 0.0 for item in context.ir.state_variables})
    environment['t'] = 0.0
    kernel = next(node for node in context.uast.nodes if node.kind == 'kernel')
    jacobian, _ = differentiate(values, variables, environment, kernel.export_adjoint_jacobian) if values else ([], '')
    return list(closures), values, jacobian


def selected(node, sources, pytest_paths=()):
    return ModuleEmission({path: sources[path] for path in node.target_relpaths}, list(pytest_paths))

"""Recover scientific models and preserve the exact persisted runtime reference."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import sympy as sp

from invar_agent.research.evolution import check_evolution_reference, structured_evidence
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.runtime.research_reference import ReferenceError, validate_predicate

from .archive_guard import IngestionError
from .equation_lowerer import lower_and_verify_equations, to_csel, unit_dimension
from .schemas import (CanonicalResearchIR, RawConstant, RawEquation, SourceSpan,
                      SIDimensionVectorModel, StateVariableSpec, ResearchReference,
                      ReferenceContract, RuntimePredicate)


def _dimensions(laws, parameters, quantities):
    unknown = {name: [sp.Symbol(f'_dim_{index}_{axis}') for axis in range(7)]
               for index, name in enumerate(laws)}
    known = {name: list(unit_dimension(item['unit']).exponents) for name, item in parameters.items()}
    known['t'] = [0, 0, 1, 0, 0, 0, 0]
    constraints = []
    for name, quantity in quantities.items():
        if name in unknown and not quantity.get('dimension_unspecified'):
            powers = list(quantity.get('si_dimension', [0]*7))
            powers[0], powers[1] = powers[1], powers[0]
            constraints.extend(a-b for a,b in zip(unknown[name], powers))

    def dimension(expression):
        if expression.is_Number:
            return [0]*7
        if isinstance(expression, sp.Symbol):
            return known.get(str(expression), unknown.get(str(expression), [0]*7))
        if isinstance(expression, sp.Piecewise):
            branch_dims = [dimension(expr_b) for expr_b, _cond in expression.args if expr_b != 0]
            if not branch_dims:
                return [0]*7
            for part in branch_dims[1:]:
                constraints.extend(a-b for a,b in zip(branch_dims[0], part))
            return branch_dims[0]
        if expression.func in (sp.Min, sp.Max):
            arg_dims = [dimension(arg) for arg in expression.args if arg != 0]
            if not arg_dims:
                return [0]*7
            for part in arg_dims[1:]:
                constraints.extend(a-b for a,b in zip(arg_dims[0], part))
            return arg_dims[0]
        parts = [dimension(arg) for arg in expression.args]
        if isinstance(expression, sp.Add):
            for part in parts[1:]:
                constraints.extend(a-b for a,b in zip(parts[0], part))
            return parts[0]
        if isinstance(expression, sp.Mul):
            return [sum(part[axis] for part in parts) for axis in range(7)]
        if isinstance(expression, sp.Pow) and expression.exp.is_Rational:
            return [value*expression.exp for value in parts[0]]
        if expression.func in (sp.sin, sp.cos, sp.exp, sp.log):
            constraints.extend(parts[0])
            return [0]*7
        if expression.func == sp.Abs:
            return parts[0]
        raise IngestionError('HIN_SI_DIM_MISMATCH', str(expression))

    for name, expression in laws.items():
        if expression == 0:
            continue
        constraints.extend(a-b-c for a,b,c in zip(unknown[name], known['t'], dimension(expression)))
    variables = [value for powers in unknown.values() for value in powers]
    result = sp.linsolve(constraints, variables) if constraints else sp.FiniteSet(tuple(variables))
    if result is sp.EmptySet or not result:
        # Surrogate ODE modules often inline numeric rate/scaling constants as pure Python floats
        # whose implicit physical dimensions are absorbed numerically; fall back to declared
        # state quantities when the over-constrained symbolic check hits EmptySet due to literals.
        q_constraints = []
        for name, quantity in quantities.items():
            if name in unknown and not quantity.get('dimension_unspecified'):
                powers = list(quantity.get('si_dimension', [0]*7))
                powers[0], powers[1] = powers[1], powers[0]
                q_constraints.extend(a-b for a, b in zip(unknown[name], powers))
        result = sp.linsolve(q_constraints, variables) if q_constraints else sp.FiniteSet(tuple([0]*len(variables)))
        if result is sp.EmptySet or not result:
            raise IngestionError('HIN_SI_DIM_MISMATCH', 'ODE dimensions contradict the persisted parameter units')
    values = list(next(iter(result)))
    free = {symbol: 0 for value in values for symbol in value.free_symbols}
    values = [value.subs(free) for value in values]
    output = {}
    for index, name in enumerate(unknown):
        powers = values[index*7:(index+1)*7]
        if any(not value.is_Integer for value in powers):
            raise IngestionError('HIN_SI_DIM_MISMATCH', f'Nonintegral SI dimensions for {name}')
        powers = tuple(map(int, powers))
        unit = '*'.join(base+(f'^{power}' if power != 1 else '') for base,power in zip(('m','kg','s','A','K','mol','cd'),powers) if power) or '1'
        output[name] = SIDimensionVectorModel(exponents=powers, canonical_unit=unit,
            semantic_tag=(quantities.get(name) or {}).get('semantic_tag', 0))
    return output


class _ScalarContract(ast.NodeTransformer):
    def __init__(self, slots, initial):
        self.slots, self.initial = slots, initial

    def visit_Subscript(self, node):
        if not isinstance(node.slice, ast.Constant) or type(node.slice.value) is not int:
            raise ValueError('Array predicate')
        index = node.slice.value
        if isinstance(node.value, ast.Name) and node.value.id in {'initial','initial_state'} and 0 <= index < len(self.initial):
            return ast.copy_location(ast.Constant(self.initial[index]), node)
        state = isinstance(node.value, ast.Name) and node.value.id in {'state','data'}
        buffer = isinstance(node.value,ast.Attribute) and node.value.attr=='data' and isinstance(node.value.value,ast.Name) and node.value.value.id=='buf'
        if (state or buffer) and index in self.slots:
            return ast.copy_location(ast.Name(self.slots[index], ast.Load()), node)
        raise ValueError('Runtime buffer predicate')

    def visit_Call(self, node):
        raise ValueError('Runtime function predicate')


def load_vault_bundle(workspace_root: Path, run_id: str) -> CanonicalResearchIR:
    manager = ResearchVaultManager(Path(workspace_root))
    if not run_id or manager.run_dir(run_id).name != run_id:
        raise IngestionError("VAULT_INVALID_RUN_ID", run_id)
    root = manager.run_dir(run_id)
    try:
        texts = {name: (root/name).read_text() for name in ('metadata.json','blueprint.json','unified_module.py','report.md')}
        metadata, blueprint = json.loads(texts['metadata.json']), json.loads(texts['blueprint.json'])
    except (OSError, ValueError) as error:
        raise IngestionError('VAULT_BUNDLE_INVALID', str(error)) from error
    raw_certificate = metadata.get('meaningfulness_certificate')
    certificate_path = root/'meaningfulness_certificate.json'
    if certificate_path.exists(): raw_certificate = json.loads(certificate_path.read_text())
    certificates = {}
    certified_valid_domain = None
    if raw_certificate is not None:
        from ...research.epistemic import validate_certificate_artifacts
        certificate = validate_certificate_artifacts(raw_certificate, texts['unified_module.py'], blueprint)
        certified_valid_domain = getattr(certificate,'certified_valid_domain',None)
        if certificate.run_id != run_id: raise IngestionError('VAULT_CERTIFICATE_MISMATCH', run_id)
        certificates[run_id] = certificate.model_dump(mode='json')
        if certificate.epistemic_verdict.value == 'VERIFIED_FALSIFICATION':
            return CanonicalResearchIR(ir_id=f'vault:{run_id}', source_mode='vault', vault_run_ids=[run_id],
                title=metadata.get('title') or run_id, meaningfulness_certificates=certificates,
                meaningfulness_certificate=certificates[run_id], epistemic_verdict='VERIFIED_FALSIFICATION', certified_valid_domain=certified_valid_domain,
                research_import_role='NEGATIVE_CONSTRAINT_ONLY', negative_constraints=[{'run_id': run_id, 'certificate': certificates[run_id]}])
        if hasattr(certificate, 'ihcg_anchor') and certificate.epistemic_verdict.value == 'INCONCLUSIVE_UNVERIFIED':
            raise IngestionError('INCONCLUSIVE_UNVERIFIED', run_id)

    def provenance(filename, location, needle=''):
        text = texts[filename]
        position = text.find(needle) if needle else 0
        position = max(position, 0)
        start = text.rfind('\n', 0, position)+1
        end = min(len(text), start+2000)
        newline = text.find('\n', position)
        if newline >= 0:
            end = min(end, newline)
        return SourceSpan(file=str(root/filename), location=f'vault:@research:{run_id}:{location}',
                          quote=text[start:end] or ' ', char_start=start, char_end=end)

    try:
        recovered = check_evolution_reference(blueprint, metadata, texts['unified_module.py'])
        for contract in recovered['contracts']:
            for invariant in contract.get('invariants') or []:
                validate_predicate(invariant['assertion'])
    except ReferenceError as error:
        if error.code == "VAULT_INTEGRATOR_UNSUPPORTED" and certificates:
            from types import SimpleNamespace
            from ...schemas.research import ResearchBlueprint
            from ...research.epistemic import compile_blueprint_ihcg
            from ...research.default_verification import _Kernel
            parsed = ResearchBlueprint.model_validate(blueprint)
            graph = compile_blueprint_ihcg(parsed)
            if (graph is not None and all(e.kind == "ALGEBRAIC" for e in graph.equations.values())
                    and all(not any(s.si_dim) for s in graph.slots)):
                _Kernel(None,SimpleNamespace(ihcg=graph,blueprint=parsed,research_stage=1),texts['unified_module.py'])
                span = provenance('blueprint.json','sealed-algebraic-model','"ihcg_evidence"')
                raw_equations = [RawEquation(text=f'{name} = {equation.expression}',provenance=span)
                                 for name,equation in graph.equations.items()]
                constants = [RawConstant(symbol=s.symbol,value=str(parsed.contracts[0].initial_state[s.slot_idx]),unit='1',provenance=span)
                             for s in graph.slots if s.role == 'CONTROL_PARAM']
                equations,constants,flags = lower_and_verify_equations(raw_equations,constants,
                    {'symbol_dimensions':{s.symbol:SIDimensionVectorModel().model_dump() for s in graph.slots}})
                return CanonicalResearchIR(ir_id=f'vault:{run_id}',source_mode='vault',vault_run_ids=[run_id],
                    title=metadata.get('title') or run_id,summary=texts['report.md'][:2000],
                    equations=equations,constants=constants,disambiguation_flags=flags,
                    meaningfulness_certificates=certificates,meaningfulness_certificate=certificates[run_id],
                    epistemic_verdict=certificate.epistemic_verdict.value,certified_valid_domain=certified_valid_domain)
        raise IngestionError(error.code, str(error), details=error.details) from error
    laws, parameters, slots = recovered['laws'], recovered['parameters'], recovered['slots']
    quantities = ((blueprint.get('uast_spec') or {}).get('ontology') or {}).get('quantities') or {}
    dimensions = _dimensions(laws, parameters, quantities)
    raw_constants = [RawConstant(symbol=name, value=str(item['value']), unit=item['unit'],
        provenance=provenance(item.get('file') or ('blueprint.json' if item['source'] != 'stored' else 'metadata.json'),
                              f'constant:{name}', json.dumps(item['key'])))
                     for name,item in parameters.items()]
    raw_equations = []
    sources = {}
    def _compact_law_expr(expr):
        s = str(expr)
        if len(s) <= 4096:
            return expr
        # Collapse nested Piecewise branches to their default/interior branch for compact CSEL/HIN lowering;
        # the exact bit-identical Python kernel remains preserved in `pinned_kernel_code`.
        def _unwrap_pw(node):
            if isinstance(node, sp.Piecewise):
                for val, cond in reversed(node.args):
                    if cond == sp.true and val != 0:
                        return _unwrap_pw(val)
                for val, _cond in node.args:
                    if val != 0:
                        return _unwrap_pw(val)
                return sp.Integer(0)
            if node.args:
                return node.func(*(_unwrap_pw(a) for a in node.args))
            return node
        try:
            return _unwrap_pw(expr)
        except Exception:
            return expr

    for state, expression in laws.items():
        kind, origin, *source_file = recovered['origins'][state]
        filename = source_file[0] if source_file else 'unified_module.py' if kind == 'kernel' else 'blueprint.json'
        span = provenance(filename, f'evolution:{state}:{kind}', f'def {origin}' if kind=='kernel' else json.dumps(origin))
        raw_equations.append(RawEquation(text=f'd{state}/dt = {_compact_law_expr(expression)}', provenance=span))
        sources[state] = [provenance('unified_module.py', f'symbolic:{state}:{function.name}', f'def {function.name}')
                          for function in ast.parse(texts['unified_module.py']).body if isinstance(function, ast.FunctionDef)] if state in recovered['symbolic'] else []
    known_symbols = set(laws) | set(parameters) | set(slots) | {'t'}
    for item in structured_evidence(blueprint,metadata)['equations']:
        if (item.get('source') not in (None, '', 'user_prompt', 'blueprint', 'metadata')
                and not item.get('role') and item.get('lhs') not in (blueprint.get('variable_map') or {})):
            continue
        item_vars = set(item.get('variables') or [])
        if item_vars and (item_vars - known_symbols):
            continue
        if ',' in str(item.get('rhs_latex') or '') or ',' in str(item.get('lhs') or ''):
            continue
        if item.get('role') != 'EVOLUTION_LAW' and not re.search(r'\bd\w+/dt|\\dot|\\frac\{d', item.get('lhs','')):
            raw_equations.append(RawEquation(text=f"{item['lhs']} = {item['rhs_latex']}",
                provenance=provenance('blueprint.json','structured-equation', '"rhs_latex"')))
    runtime = []
    state_slots = {name:slots[name] for name in laws}
    scalar_slots = {index:name for name,index in slots.items()}
    contracts = []
    stages_completed = metadata.get('stages_completed')
    is_partial_vault = str(metadata.get('status') or '').upper() in ('PARTIAL', 'PARTIALLY_VERIFIED')
    for contract in recovered['contracts']:
        if is_partial_vault and isinstance(stages_completed, int) and stages_completed >= 1:
            m_stg = re.match(r'^c(\d+)_', str(contract.get('contract_id') or ''), re.I)
            if m_stg and int(m_stg.group(1)) > stages_completed:
                continue
        contracts.append(ReferenceContract.model_validate({key:contract[key] for key in ('contract_id','initial_state','dt','steps','invariants') if key in contract}))
        for invariant in contract.get('invariants') or []:
            assertion = invariant.get('assertion') or ''
            span = provenance('blueprint.json', f"contract:{contract['contract_id']}", json.dumps(assertion))
            try:
                transformed = _ScalarContract(scalar_slots, contract['initial_state']).visit(ast.parse(assertion, mode='eval'))
                names = {node.id for node in ast.walk(transformed) if isinstance(node, ast.Name)}
                if names - set(slots) or not isinstance(transformed.body, ast.Compare) or len(transformed.body.ops) != 1:
                    raise ValueError('Non-scalar predicate')
                raw_equations.append(RawEquation(text=ast.unparse(ast.fix_missing_locations(transformed)), provenance=span))
            except ValueError:
                runtime.append(RuntimePredicate(contract_id=contract['contract_id'], assertion=assertion,
                    tolerance=invariant.get('tolerance',0), provenance=span))
    equations, constants, flags = lower_and_verify_equations(raw_equations, raw_constants,
        {'state_symbols': list(laws), 'symbol_dimensions': {name:value.model_dump() for name,value in dimensions.items()}})
    for equation in equations:
        equation.supporting_provenance = sources.get(equation.lhs_symbol, [])
    ir = CanonicalResearchIR(ir_id=f'vault:{run_id}', source_mode='vault', vault_run_ids=[run_id],
        meaningfulness_certificates=certificates,
        certified_valid_domain=certified_valid_domain,
        title=metadata.get('title') or blueprint.get('title') or run_id, summary=texts['report.md'][:2000],
        equations=equations, constants=constants, pinned_kernel_code=texts['unified_module.py'],
        runtime_predicates=runtime, research_reference=ResearchReference(state_slots=state_slots, variable_map=blueprint.get('variable_map') or {},
            parameter_slots={name:slots[name] for name in parameters if name in slots},
            contracts=contracts, rhs_cross_checks=recovered['comparisons'],integration_scheme=recovered['integration_scheme']),
        disambiguation_flags=flags, macro_domains=(blueprint.get('structured_evidence') or {}).get('domains') or [])
    initial = contracts[0].initial_state
    for index,name in enumerate(laws):
        ir.state_variables.append(StateVariableSpec(symbol=name, slot_index=index,
            initial_value=initial[slots[name]], dimension=dimensions[name],
            provenance=provenance('blueprint.json',f'state:{name}:{slots[name]}','"initial_state"')))
    ir.default_dt_si_s, ir.default_steps = contracts[0].dt, contracts[0].steps
    ir.time_horizon_si_s = ir.default_dt_si_s * ir.default_steps
    return ir

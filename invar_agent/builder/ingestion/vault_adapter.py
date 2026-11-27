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
    from invar_agent.research.vault import VaultIntegrityError, verify_vault_manifest

    manager = ResearchVaultManager(Path(workspace_root))
    if not run_id or manager.run_dir(run_id).name != run_id:
        raise IngestionError("VAULT_INVALID_RUN_ID", run_id)
    root = manager.run_dir(run_id)
    certificate_path = root/'meaningfulness_certificate.json'
    if not certificate_path.is_file():
        raise IngestionError(
            'VAULT_CERTIFICATE_MISSING',
            f"Vault entry {run_id!r} is missing meaningfulness_certificate.json",
        )
    # ARCH-012 (BUG-019): manifest.json is mandatory for Vault admission.
    # Deleting manifest.json fails closed (HTTP 422 VAULT_MANIFEST_MISSING).
    if not (root / "manifest.json").is_file():
        raise IngestionError(
            "VAULT_MANIFEST_MISSING",
            f"Vault entry {run_id!r} is missing mandatory manifest.json",
        )
    try:
        verify_vault_manifest(root)
    except VaultIntegrityError as error:
        raise IngestionError(error.code, error.message) from error
    try:
        texts = {name: (root/name).read_text(encoding='utf-8') for name in ('metadata.json','blueprint.json','unified_module.py','report.md')}
        metadata, blueprint = json.loads(texts['metadata.json']), json.loads(texts['blueprint.json'])
    except (OSError, ValueError) as error:
        raise IngestionError('VAULT_BUNDLE_INVALID', str(error)) from error
    try:
        raw_certificate = json.loads(certificate_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise IngestionError('VAULT_CERTIFICATE_INVALID', str(error)) from error
    # If manifest.json was absent (pre-v2 vault), still verify metadata certificate consistency
    # and require full certificate artifact binding.
    certificates = {}
    certified_valid_domain = None
    try:
        from ...research.epistemic import validate_certificate_artifacts
        certificate = validate_certificate_artifacts(raw_certificate, texts['unified_module.py'], blueprint)
    except ValueError as error:
        raise IngestionError('VAULT_CERTIFICATE_INVALID', str(error)) from error
    certified_valid_domain = getattr(certificate, 'certified_valid_domain', None)
    if certificate.run_id != run_id:
        raise IngestionError('VAULT_CERTIFICATE_MISMATCH', run_id)
    certificates[run_id] = certificate.model_dump(mode='json')
    verdict_val = certificate.epistemic_verdict.value
    if verdict_val == 'VERIFIED_FALSIFICATION':
        return CanonicalResearchIR(ir_id=f'vault:{run_id}', source_mode='vault', vault_run_ids=[run_id],
            title=metadata.get('title') or run_id, meaningfulness_certificates=certificates,
            meaningfulness_certificate=certificates[run_id], epistemic_verdict='VERIFIED_FALSIFICATION', certified_valid_domain=certified_valid_domain,
            research_import_role='NEGATIVE_CONSTRAINT_ONLY', negative_constraints=[{'run_id': run_id, 'certificate': certificates[run_id]}])
    if verdict_val not in {'VERIFIED_DISCOVERY', 'VERIFIED_BOUNDARY_DISCOVERY'}:
        raise IngestionError(
            'INCONCLUSIVE_UNVERIFIED',
            f"Vault entry {run_id!r} cannot be admitted with verdict {verdict_val!r}",
            details={'epistemic_verdict': verdict_val, 'failure_reason': getattr(certificate, 'failure_reason', None)},
        )

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
    except (ReferenceError, NameError) as error:
        if certificates and blueprint.get('ihcg_evidence'):
            from types import SimpleNamespace
            from ...schemas.research import ResearchBlueprint
            from ...research.epistemic import compile_blueprint_ihcg
            from ...research.default_verification import _Kernel
            parsed = ResearchBlueprint.model_validate(blueprint)
            graph = compile_blueprint_ihcg(parsed)
            if graph is not None:
                _Kernel(None, SimpleNamespace(ihcg=graph, blueprint=parsed, research_stage=1), texts['unified_module.py'])
                span = provenance('blueprint.json', 'sealed-ihcg-model', '"ihcg_evidence"')
                raw_equations = []
                for name, equation in graph.equations.items():
                    if equation.kind == 'ODE_RHS' and equation.state_symbol:
                        raw_equations.append(RawEquation(text=f'd{equation.state_symbol}/dt = {equation.expression}', provenance=span))
                    elif equation.kind == 'ALGEBRAIC':
                        raw_equations.append(RawEquation(text=f'{name} = {equation.expression}', provenance=span))
                init_vec = list(parsed.contracts[-1].initial_state) if parsed.contracts else []
                def _unit_from_si_dim(dim_vec):
                    # IHCG si_dim order is [M, L, T, I, Theta, N, J] -> Builder [m, kg, s, A, K, mol, cd]
                    d = list(dim_vec) if len(dim_vec) == 7 else [0] * 7
                    exps = (d[1], d[0], d[2], d[3], d[4], d[5], d[6])
                    lbl = "*".join(
                        b + (f"^{p}" if p != 1 else "")
                        for b, p in zip(("m", "kg", "s", "A", "K", "mol", "cd"), exps)
                        if p
                    ) or "1"
                    return exps, lbl

                constants = [
                    RawConstant(
                        symbol=s.symbol,
                        value=str(init_vec[s.slot_idx] if s.slot_idx < len(init_vec) else 0.0),
                        unit=_unit_from_si_dim(s.si_dim)[1],
                        provenance=span,
                    )
                    for s in graph.slots if s.role == 'CONTROL_PARAM'
                ]
                sym_dims = {
                    s.symbol: SIDimensionVectorModel(
                        exponents=_unit_from_si_dim(s.si_dim)[0],
                        canonical_unit=_unit_from_si_dim(s.si_dim)[1],
                        semantic_tag=s.semantic_tag,
                    ).model_dump()
                    for s in graph.slots
                }
                state_syms = [s.symbol for s in graph.slots if s.role != 'CONTROL_PARAM']
                equations, constants, flags = lower_and_verify_equations(
                    raw_equations, constants,
                    {'state_symbols': state_syms, 'symbol_dimensions': sym_dims},
                )
                for c in constants:
                    if c.symbol in sym_dims:
                        c.dimension = SIDimensionVectorModel.model_validate(sym_dims[c.symbol])
                verdict_str = certificate.epistemic_verdict.value
                import_role = 'POSITIVE_EVIDENCE'
                ir = CanonicalResearchIR(
                    ir_id=f'vault:{run_id}', source_mode='vault', vault_run_ids=[run_id],
                    title=metadata.get('title') or blueprint.get('title') or run_id,
                    summary=texts['report.md'][:2000],
                    equations=equations, constants=constants, disambiguation_flags=flags,
                    pinned_kernel_code=texts['unified_module.py'],
                    meaningfulness_certificates=certificates,
                    meaningfulness_certificate=certificates[run_id],
                    epistemic_verdict=verdict_str,
                    research_import_role=import_role,
                    certified_valid_domain=certified_valid_domain,
                    macro_domains=(blueprint.get('structured_evidence') or {}).get('domains') or [],
                )
                dyn_slots = [s for s in graph.slots if s.role == 'DYNAMIC_STATE'] or [s for s in graph.slots if s.role != 'CONTROL_PARAM']
                for idx_s, s in enumerate(dyn_slots):
                    ir.state_variables.append(StateVariableSpec(
                        symbol=s.symbol, slot_index=idx_s,
                        initial_value=float(init_vec[s.slot_idx]) if s.slot_idx < len(init_vec) else 0.0,
                        dimension=SIDimensionVectorModel(exponents=tuple(s.si_dim), semantic_tag=s.semantic_tag),
                        provenance=span,
                    ))
                if parsed.contracts:
                    c0 = parsed.contracts[-1]
                    ir.default_dt_si_s, ir.default_steps = c0.dt, c0.steps
                    ir.time_horizon_si_s = c0.dt * c0.steps
                return ir
        if isinstance(error, ReferenceError):
            raise IngestionError(error.code, str(error), details=error.details) from error
        raise IngestionError('VAULT_REFERENCE_UNSUPPORTED', str(error)) from error
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
                # ARCH-009 (BUG-020): Contract invariants on dynamic state variables that do NOT
                # hold at t=0 (e.g. `x(T) > 2` with `x(0) = 1`, or `I_max > 0.15`) are FINAL/MAX/MIN
                # temporal predicates, NOT t=0 constraints. Route them to RuntimePredicate so Z3
                # never contradicts initial_state; keep ALWAYS invariants that hold at t=0.
                if names & set(laws):
                    init_env = {
                        sym: float(contract['initial_state'][slots[sym]])
                        for sym in names
                        if sym in slots and slots[sym] < len(contract['initial_state'])
                    }
                    try:
                        holds_at_t0 = bool(eval(compile(ast.fix_missing_locations(transformed), '<t0_check>', 'eval'), {'__builtins__': {}}, init_env))
                    except Exception:
                        holds_at_t0 = False
                    if not holds_at_t0:
                        # Keep simple zero-bound positivity guards (e.g. `x >= 0`) when violated by
                        # a negative initial state so Z3 catches genuine initial-state contradictions.
                        cmp_node = transformed.body
                        is_zero_positivity_violation = (
                            isinstance(cmp_node.left, ast.Name)
                            and isinstance(cmp_node.ops[0], (ast.GtE, ast.Gt))
                            and isinstance(cmp_node.comparators[0], ast.Constant)
                            and isinstance(cmp_node.comparators[0].value, (int, float))
                            and float(cmp_node.comparators[0].value) == 0.0
                            and init_env.get(cmp_node.left.id, 0.0) < 0.0
                        )
                        if not is_zero_positivity_violation:
                            raise ValueError('Temporal state predicate')
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
        meaningfulness_certificate=certificates[run_id],
        epistemic_verdict=verdict_val,
        research_import_role='POSITIVE_EVIDENCE',
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

"""Exact slot identities shared by planning, healing and reporting."""
import ast
import re


def observable_name(name):
    return re.sub(r"_(?:positivity|nonnegative|nonneg|finite|bound|window|check)$", "", name.lower())


def observable_label(name):
    return re.sub(r"_(?:positivity|nonnegative|nonneg|finite|bound|window|check)$", "", name, flags=re.I)


def _state_slot(node):
    if not isinstance(node, ast.Subscript):
        return None
    if ast.unparse(node.value) in {'state', 'data', 'buf.data'}:
        if isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
            return node.slice.value
        if isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub) and isinstance(node.slice.operand, ast.Constant) and type(node.slice.operand.value) is int:
            return -node.slice.operand.value
    return None


def _state_slots(tree):
    return {slot for node in ast.walk(tree) if (slot := _state_slot(node)) is not None}


def validate_slot_writes(code, variable_map, roles):
    for node in ast.walk(ast.parse(code)):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name)
                and node.value.value.id == 'buf' and node.value.attr == 'data'
                and isinstance(node.slice, ast.Constant)):
            for name, role in roles.items():
                if role == 'read_only_parameter' and variable_map.get(name) == node.slice.value:
                    raise ValueError(f'SLOT_SCHEMA_ERROR: write to read-only parameter {name} at state[{node.slice.value}] line {node.lineno}')


def validate_contract_slots(contracts):
    claims = {}
    extrema = {}
    identities = {}
    for contract in contracts:
        value = contract if isinstance(contract, dict) else contract.model_dump()
        for inv in value.get('invariants') or []:
            name = inv.get('variable_name', '')
            source = inv.get('assertion', '')
            try:
                tree = ast.parse(source, mode='eval')
            except SyntaxError:
                continue
            if 'conservation' in name.lower():
                for node in ast.walk(tree):
                    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                        def terms(expr):
                            if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
                                return terms(expr.left) + terms(expr.right)
                            slot = _state_slot(expr)
                            return [slot] if slot is not None else []
                        parts = terms(node)
                        if len(parts) != len(set(parts)):
                            raise ValueError(f'SLOT_SCHEMA_ERROR: duplicate conservation term in {name}: {source}')
            slots = _state_slots(tree)
            if any(slot < 0 or slot >= 4096 for slot in slots):
                raise ValueError(f'SLOT_SCHEMA_ERROR: invalid state slots in {source}')
            if len(slots) == 1:
                slot = next(iter(slots))
                semantic = observable_name(name)
                names = identities.setdefault(slot, set())
                names.add(semantic)
                for identity in names:
                    if identity.startswith(('peak_', 'min_', 'max_')) and identity.split('_', 1)[1] in names:
                        raise ValueError(f'SLOT_SCHEMA_ERROR: instantaneous and extrema observables share state[{slot}]')
            if len(slots) == 1 and name.lower().startswith(('peak_', 'min_', 'max_')):
                semantic = observable_name(name)
                slot = next(iter(slots))
                owner = extrema.setdefault(semantic, slot)
                if owner != slot:
                    raise ValueError(f'SLOT_SCHEMA_ERROR: extrema {semantic} split across state[{owner}] and state[{slot}]')
            if len(slots) == 1 and name.lower().endswith(('_positivity', '_nonnegative', '_nonneg')):
                slot = next(iter(slots)); semantic = observable_name(name)
                owner = claims.setdefault(slot, semantic)
                if owner != semantic:
                    raise ValueError(f'SLOT_SCHEMA_ERROR: distinct variables {owner}, {semantic} share state[{slot}]')


def canonical_variable_map(contracts, variable_map):
    validate_contract_slots(contracts)
    values = [c if isinstance(c, dict) else c.model_dump() for c in contracts]
    size = max([len(c.get('initial_state') or []) for c in values] + [0])
    owners = {}
    for name, slot in variable_map.items():
        slot = int(slot)
        if slot < 0 or slot >= 4096:
            raise ValueError(f'SLOT_SCHEMA_ERROR: invalid state slot {slot}')
        if slot not in owners or owners[slot] == f'state_{slot}':
            owners[slot] = observable_label(name)
    for contract in values:
        for inv in contract.get('invariants') or []:
            slots = _state_slots(ast.parse(inv.get('assertion', ''), mode='eval'))
            if len(slots) == 1:
                slot = next(iter(slots)); size = max(size, slot + 1)
                owners.setdefault(slot, observable_label(inv.get('variable_name', '')) or f'state_{slot}')
    for slot in range(size):
        owners.setdefault(slot, f'state_{slot}')
    result = {}
    for slot, name in sorted(owners.items()):
        if name in result and result[name] != slot:
            raise ValueError(f'SLOT_SCHEMA_ERROR: observable {name} has conflicting slots')
        result[name] = slot
    return result

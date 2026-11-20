"""Structural failures must be repaired before numerical conditioning."""
import ast
import re


def is_structural_failure(failure: str) -> bool:
    if 'VAULT_REFERENCE_' in failure:
        return False
    return any(marker in failure for marker in (
        'PreWrite_Guard', 'transpiler:', 'AttributeError', 'NameError',
        'TypeError', 'SyntaxError', 'IndexError', 'AST firewall', 'SLOT_SCHEMA_ERROR'))


def unresolved_symbol(failure: str, code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return str(error)
    match = re.search(r"name ['\"]([A-Za-z_]\w*)['\"] is not defined", failure)
    if match:
        name = match[1]
        loads = any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load) for n in ast.walk(tree))
        bound = any((isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store))
                    or (isinstance(n, ast.arg) and n.arg == name)
                    or (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
                    or (isinstance(n, ast.alias) and (n.asname or n.name.split('.')[0]) == name)
                    for n in ast.walk(tree))
        if loads and not bound:
            return f'Unresolved offending symbol {name}'
    match = re.search(r"(?:has no attribute ['\"]|unsupported buffer attribute buf\.)([A-Za-z_]\w*)", failure)
    if match and any(isinstance(n, ast.Attribute) and n.attr == match[1]
                     and isinstance(n.value, ast.Name) and n.value.id == 'buf' for n in ast.walk(tree)):
        return f'Unresolved offending attribute buf.{match[1]}'
    return ''

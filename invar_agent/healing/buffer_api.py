"""Scalar buffer clock lowering without extending the public C ABI."""
import ast
import re


BUFFER_API = "SimulationBuffer exposes ONLY buf.data[i] and buf.size; dt is a function argument. Use a dedicated state slot for elapsed time/step counting, never buf.dt or buf.steps."
STEP_ALIASES = frozenset({"steps", "step", "step_idx", "step_index", "iter", "iteration", "n_steps"})
TIME_ALIASES = frozenset({"t", "time", "elapsed", "sim_time"})


def lower_buffer_clock(tree, contracts=()):
    aliases = STEP_ALIASES | TIME_ALIASES
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == "buf"
             and n.attr in aliases]
    if not reads:
        return tree, False, []
    slots = [int(m) for m in re.findall(r"(?:data|state|initial)\[(\d+)\]", ast.unparse(tree))]
    for contract in contracts:
        value = contract if isinstance(contract, dict) else contract.model_dump()
        slots.append(len(value.get("initial_state") or []))
        slots.extend(int(m) for m in re.findall(r"(?:data|state|initial)\[(\d+)\]", str(value)))
    slot = max(slots + [0]) + 2
    for contract in contracts:
        initial = contract.get('initial_state') if isinstance(contract, dict) else contract.initial_state
        if isinstance(initial, list) and len(initial) <= slot:
            initial.extend([0.0] * (slot + 1 - len(initial)))

    class Clock(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'buf' and t.attr in aliases for t in node.targets):
                return ast.copy_location(ast.Pass(), node)
            return self.generic_visit(node)

        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Attribute) and isinstance(node.target.value, ast.Name) and node.target.value.id == 'buf' and node.target.attr in aliases:
                return ast.copy_location(ast.Pass(), node)
            return self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "buf" and node.attr in aliases:
                source = f"buf.data[{slot}]"
                if node.attr in TIME_ALIASES:
                    source = f"({source} * dt)"
                return ast.copy_location(ast.parse(source, mode="eval").body, node)
            return self.generic_visit(node)

    tree = Clock().visit(tree)
    increment = f"buf.data[{slot}] = buf.data[{slot}] + 1.0"

    class Exit(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return node

        def visit_Return(self, node):
            if node.value is not None:
                return [ast.Assign(targets=[ast.Name(id='_invar_return_value', ctx=ast.Store())], value=node.value),
                        ast.parse(increment).body[0], ast.Return(value=ast.Name(id='_invar_return_value', ctx=ast.Load()))]
            return [ast.parse(increment).body[0], node]

    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == "simulation_step":
            body = ast.parse(f"_invar_step_slot = {slot}\nif buf.size <= _invar_step_slot:\n    return").body
            for statement in fn.body:
                result = Exit().visit(statement)
                body.extend(result if isinstance(result, list) else [result])
            fn.body = body
            if not isinstance(fn.body[-1], ast.Return):
                fn.body.append(ast.parse(increment).body[0])
    ast.fix_missing_locations(tree)
    return tree, True, [f"lowered buffer clock to internal state[{slot}]"]

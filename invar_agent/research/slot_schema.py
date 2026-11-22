"""Exact slot identities and unified slot-role classifier shared by planning, healing, guards and reporting."""
from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple


ROLE_STATE = "STATE"
ROLE_CONTROL_PARAM = "CONTROL_PARAM"
ROLE_DIAGNOSTIC = "DIAGNOSTIC"
ROLE_TIME_STEP = "TIME_STEP"

_TIME_STEP_EXACT = {
    "t", "time", "time_s", "sim_time", "sim_time_s", "elapsed", "elapsed_s",
    "elapsed_time", "elapsed_time_s", "step_count", "steps", "step", "clock",
}
_TIME_STEP_RE = re.compile(r"^(?:sim_|elapsed_|current_)?time(?:_[a-z0-9]+)?$", re.I)
_DIAGNOSTIC_NAME_RE = re.compile(
    r"(?:^(?:peak|min|max|mean|avg|sum|total|corridor|ratio|error|err|drift|"
    r"residual|margin|violation|mismatch|found|flag|feasible|status|score|"
    r"index|aec|energy_charge|observ(?:ed|able))_|"
    r"_(?:peak|min|max|ratio|residual|error|err|drift|band|window|corridor|"
    r"range|margin|diagnostic|outcome|threshold|found|flag|charge|index)$|"
    r"^(?:aec|energy_charge|energy_drift|corridor_ok|corridor_found)$)",
    re.I,
)
_CONTROL_PARAM_NAME_RE = re.compile(
    r"^(?:g_|k_|c_m$|r_gas$|f_far$|t_k$|param_|ctrl_|control_|dt_|step_size)",
    re.I,
)
_BAND_NAME_RE = re.compile(r"(?:^|_)(?:band|window|corridor|range)(?:_|$)", re.I)
_UPPER_CEIL_NAME_RE = re.compile(r"(?:^|_)(?:bound|max|limit|ceiling|cap)(?:_|$)", re.I)
_EXTREMA_PREFIX_RE = re.compile(r"^(peak|max|min)_(.+)$", re.I)


def to_blueprint_role(role: str) -> Literal["read_only_parameter", "evolving_state", "diagnostic"]:
    """Map unified 4-role classifier label to ResearchBlueprint.slot_roles Literal."""
    if role == ROLE_CONTROL_PARAM or role == "read_only_parameter":
        return "read_only_parameter"
    if role == ROLE_DIAGNOSTIC or role == "diagnostic":
        return "diagnostic"
    return "evolving_state"


def from_blueprint_role(role: Optional[str]) -> Optional[str]:
    """Map ResearchBlueprint.slot_roles Literal to unified 4-role label."""
    if not role:
        return None
    low = str(role).strip().upper()
    if low in (ROLE_STATE, ROLE_CONTROL_PARAM, ROLE_DIAGNOSTIC, ROLE_TIME_STEP):
        return low
    if low == "READ_ONLY_PARAMETER":
        return ROLE_CONTROL_PARAM
    if low == "DIAGNOSTIC":
        return ROLE_DIAGNOSTIC
    if low == "EVOLVING_STATE":
        return ROLE_STATE
    return None


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


def _analyze_code_slot_flow(code_or_tree: Any) -> Dict[str, Any]:
    """Lightweight AST dataflow over buf.data slots (no circular imports)."""
    reads: Set[int] = set()
    writes: Set[int] = set()
    clocks: Set[int] = set()
    extrema: Set[int] = set()
    graph: Dict[int, Set[int]] = {}
    if not code_or_tree:
        return {"reads": reads, "writes": writes, "clocks": clocks,
                "extrema": extrema, "physical": set(), "derived": set(), "dependencies": graph}
    try:
        tree = ast.parse(code_or_tree) if isinstance(code_or_tree, str) else code_or_tree
    except SyntaxError:
        return {"reads": reads, "writes": writes, "clocks": clocks,
                "extrema": extrema, "physical": set(), "derived": set(), "dependencies": graph}

    functions = {n.name: n for n in getattr(tree, "body", []) if isinstance(n, ast.FunctionDef)}
    reachable: Set[str] = set()
    if "simulation_step" in functions:
        queue = ["simulation_step"]
        while queue:
            fn_name = queue.pop()
            if fn_name in reachable or fn_name not in functions:
                continue
            reachable.add(fn_name)
            queue.extend(
                c.func.id for c in ast.walk(functions[fn_name])
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            )
    else:
        reachable = set(functions.keys())

    all_writes: Set[int] = set()
    has_dynamic_write = False
    for all_fn in functions.values():
        for n in ast.walk(all_fn):
            tgts = n.targets if isinstance(n, ast.Assign) else ([n.target] if isinstance(n, ast.AugAssign) else ())
            for tgt in tgts:
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Attribute) and tgt.value.attr == "data":
                    s_idx = _state_slot(tgt)
                    if s_idx is not None:
                        all_writes.add(s_idx)
                    else:
                        has_dynamic_write = True

    scopes: Dict[str, Dict[str, Set[int]]] = {fn_name: {} for fn_name in reachable}
    return_deps: Dict[str, List[Set[int]]] = {fn_name: [] for fn_name in reachable}
    for _pass in range(len(reachable) + 2):
        any_scope_change = False
        for fn_name in reachable:
            fn = functions[fn_name]
            bindings = scopes[fn_name]

            def _deps(expr: ast.AST, _b=bindings) -> Set[int]:
                out = _state_slots(expr)
                for node in ast.walk(expr):
                    if isinstance(node, ast.Name):
                        out |= _b.get(node.id, set())
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in return_deps:
                        for rset in return_deps[node.func.id]:
                            out |= rset
                return out

            assignments = [
                n if isinstance(n, ast.Assign) else ast.Assign(
                    targets=[n.target],
                    value=ast.BinOp(left=n.target, op=n.op, right=n.value),
                )
                for n in ast.walk(fn)
                if isinstance(n, (ast.Assign, ast.AugAssign))
            ]

            def _bind_target(target: ast.AST, rhs_node: ast.AST) -> bool:
                if isinstance(target, ast.Name):
                    old = bindings.get(target.id, set())
                    val = old | _deps(rhs_node)
                    if val != old:
                        bindings[target.id] = val
                        return True
                    return False
                if isinstance(target, (ast.Tuple, ast.List)):
                    if isinstance(rhs_node, (ast.Tuple, ast.List)) and len(target.elts) == len(rhs_node.elts):
                        any_c = False
                        for sub_t, sub_v in zip(target.elts, rhs_node.elts):
                            if _bind_target(sub_t, sub_v):
                                any_c = True
                        return any_c
                    if isinstance(rhs_node, ast.Call) and isinstance(rhs_node.func, ast.Name) and rhs_node.func.id in return_deps:
                        rlist = return_deps[rhs_node.func.id]
                        any_c = False
                        for idx, sub_t in enumerate(target.elts):
                            if isinstance(sub_t, ast.Name):
                                old = bindings.get(sub_t.id, set())
                                rset = rlist[idx] if idx < len(rlist) else (set().union(*rlist) if rlist else _deps(rhs_node))
                                val = old | rset
                                if val != old:
                                    bindings[sub_t.id] = val
                                    any_c = True
                        return any_c
                return False

            for _ in range(len(assignments) + 1):
                changed = False
                for stmt in assignments:
                    for target in stmt.targets:
                        if _bind_target(target, stmt.value):
                            changed = True
                            any_scope_change = True
                if not changed:
                    break

            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in scopes:
                    callee_fn = functions[node.func.id]
                    callee_scope = scopes[node.func.id]
                    for param, arg_expr in zip(callee_fn.args.args, node.args):
                        if param.arg == "buf":
                            continue
                        old_p = callee_scope.get(param.arg, set())
                        new_p = old_p | _deps(arg_expr)
                        if new_p != old_p:
                            callee_scope[param.arg] = new_p
                            any_scope_change = True
                elif isinstance(node, ast.Return) and node.value is not None:
                    if isinstance(node.value, (ast.Tuple, ast.List)):
                        cur_ret = [_deps(elt) for elt in node.value.elts]
                    else:
                        cur_ret = [_deps(node.value)]
                    prev_ret = return_deps.get(fn_name) or []
                    if len(cur_ret) > len(prev_ret):
                        prev_ret = prev_ret + [set() for _ in range(len(cur_ret) - len(prev_ret))]
                    merged_ret = [a | b for a, b in zip(prev_ret, cur_ret)] + prev_ret[len(cur_ret):]
                    if merged_ret != return_deps.get(fn_name):
                        return_deps[fn_name] = merged_ret
                        any_scope_change = True
        if not any_scope_change:
            break

    for fn_name in reachable:
        fn = functions[fn_name]
        bindings = scopes[fn_name]

        def _deps(expr: ast.AST, _b=bindings) -> Set[int]:
            out = _state_slots(expr)
            for node in ast.walk(expr):
                if isinstance(node, ast.Name):
                    out |= _b.get(node.id, set())
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in return_deps:
                    for rset in return_deps[node.func.id]:
                        out |= rset
            return out

        assignments = [
            n if isinstance(n, ast.Assign) else ast.Assign(
                targets=[n.target],
                value=ast.BinOp(left=n.target, op=n.op, right=n.value),
            )
            for n in ast.walk(fn)
            if isinstance(n, (ast.Assign, ast.AugAssign))
        ]
        step_slots = {
            n.value.value for n in assignments
            if isinstance(n.value, ast.Constant) and type(n.value.value) is int
            and any(isinstance(t, ast.Name) and t.id == "_invar_step_slot" for t in n.targets)
        }
        if_extrema_vars: Set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
                if isinstance(node.test.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                    cmp_names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                    for child in node.body:
                        if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name) and isinstance(child.value, ast.Name):
                            if (
                                child.targets[0].id in cmp_names
                                and child.value.id in cmp_names
                                and child.targets[0].id != child.value.id
                                and bool(bindings.get(child.value.id))
                                and not child.value.id.isupper()
                            ):
                                if_extrema_vars.add(child.targets[0].id)
        reads |= _state_slots(fn)
        for stmt in assignments:
            for target in stmt.targets:
                slot = _state_slot(target)
                if slot is None:
                    continue
                writes.add(slot)
                deps = _deps(stmt.value)
                graph.setdefault(slot, set()).update(deps)
                if isinstance(stmt.value, ast.Name) and stmt.value.id in if_extrema_vars and slot in deps and (deps - {slot}):
                    extrema.add(slot)
                if (
                    isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id in {"min", "max"}
                    and len(stmt.value.args) >= 2
                    and not any(isinstance(arg, ast.Constant) for arg in stmt.value.args)
                ):
                    arg_deps = [_deps(arg) for arg in stmt.value.args]
                    if any(d == {slot} for d in arg_deps) and any(d - {slot} for d in arg_deps):
                        extrema.add(slot)
                expr = stmt.value
                if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)):
                    if isinstance(expr.right, ast.Name) and expr.right.id == "dt" and _deps(expr.left) == {slot}:
                        clocks.add(slot)
                    if slot in step_slots and isinstance(expr.right, ast.Constant) and expr.right.value == 1:
                        clocks.add(slot)
        # Detect running extrema written via `if expr > buf.data[d]: buf.data[d] = expr`
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and len(node.body) == 1 and not node.orelse:
                stmt = node.body[0]
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    d_slot = _state_slot(stmt.targets[0])
                    if d_slot is not None and d_slot in _deps(node.test) and (_deps(stmt.value) - {d_slot}):
                        extrema.add(d_slot)

    physical: Set[int] = set()
    for start in writes:
        pending = list(graph.get(start, ()))
        seen: Set[int] = set()
        while pending:
            slot = pending.pop()
            if slot == start:
                physical.add(start)
                break
            if slot not in seen:
                seen.add(slot)
                pending.extend(graph.get(slot, ()))
    physical -= extrema
    # Conservation-complement species (e.g. C = K - A - B or summands in conserved pool sum = s1 + s2 + s3)
    # are physical STATE variables rather than derived DIAGNOSTIC metrics.
    if isinstance(tree, ast.Module):
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_invar_"):
                continue
            loc_exprs: Dict[str, ast.AST] = {}
            slot_aliases: Dict[str, int] = {}
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    t0 = node.targets[0]
                    if isinstance(t0, ast.Name):
                        # Preserve non-constant defining expressions over clamp resets like `x = 0.0`
                        if t0.id not in loc_exprs or not isinstance(node.value, ast.Constant):
                            loc_exprs[t0.id] = node.value
                    else:
                        ts = _state_slot(t0)
                        if ts is not None and isinstance(node.value, ast.Name):
                            slot_aliases[node.value.id] = ts
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                tgt_s = _state_slot(node.targets[0])
                if tgt_s is None or tgt_s in clocks or tgt_s in extrema:
                    continue
                rhs = node.value
                for _ in range(3):
                    if isinstance(rhs, ast.Name) and rhs.id in loc_exprs:
                        rhs = loc_exprs[rhs.id]
                    else:
                        break
                # Case 1: algebraic complement `K - state_a (- state_b ...)`
                if isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.Sub):
                    if (graph.get(tgt_s, set()) & physical) and not any(
                        isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in {"abs", "min", "max"}
                        for c in ast.walk(rhs)
                    ):
                        left = rhs.left
                        while isinstance(left, ast.BinOp) and isinstance(left.op, ast.Sub):
                            left = left.left
                        if (
                            (isinstance(left, ast.Constant) and isinstance(left.value, (int, float)) and left.value > 0)
                            or (isinstance(left, ast.Name) and left.id.isupper())
                        ):
                            physical.add(tgt_s)
                # Case 2: sum diagnostic `buf.data[sum_slot] = s1 + s2 + ...` promotes physical co-summands
                if isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.Add) and tgt_s not in physical:
                    add_terms: Set[int] = set()
                    stack = [rhs]
                    seen_names: Set[str] = set()
                    pure_add = True
                    while stack:
                        cur = stack.pop()
                        if isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Add):
                            stack.extend((cur.left, cur.right))
                        else:
                            cs = _state_slot(cur)
                            if cs is not None:
                                add_terms.add(cs)
                            elif isinstance(cur, ast.Name) and cur.id in slot_aliases:
                                add_terms.add(slot_aliases[cur.id])
                            elif isinstance(cur, ast.Name) and cur.id in loc_exprs and cur.id not in seen_names:
                                seen_names.add(cur.id)
                                stack.append(loc_exprs[cur.id])
                            else:
                                pure_add = False
                    if pure_add and len(add_terms) >= 2 and (add_terms & physical):
                        for term_s in add_terms:
                            if term_s not in clocks and term_s not in extrema:
                                physical.add(term_s)
    return {
        "reads": reads,
        "writes": writes,
        "all_writes": all_writes,
        "has_dynamic_write": has_dynamic_write,
        "clocks": clocks,
        "extrema": extrema,
        "physical": physical,
        "derived": writes - physical,
        "dependencies": graph,
    }


def _is_time_step_name(name: str) -> bool:
    low = (name or "").strip().lower()
    if low in _TIME_STEP_EXACT:
        return True
    if low in {"time_constant", "dt", "delta_t", "time_step", "timestep"}:
        return False
    return bool(_TIME_STEP_RE.match(low))


def _is_diagnostic_name(name: str) -> bool:
    return bool(_DIAGNOSTIC_NAME_RE.search((name or "").strip()))


def classify_variable_roles(
    variable_map: Optional[Dict[str, int]],
    code_or_tree: Any = "",
    contracts: Sequence[Any] = (),
    *,
    extracted_constants: Optional[Dict[str, float]] = None,
    explicit_roles: Optional[Dict[str, str]] = None,
    physical_slots: Sequence[int] = (),
) -> Dict[str, str]:
    """Single shared slot-role classifier consumed by Planner, Tier-0, and PreWrite_Guard.

    Returns ``{variable_name: role}`` where ``role`` is one of:
    ``STATE``, ``CONTROL_PARAM``, ``DIAGNOSTIC``, ``TIME_STEP``.
    """
    vm = dict(variable_map or {})
    explicit = dict(explicit_roles or {})
    ext_consts = {str(k).lower() for k in (extracted_constants or {})}
    flow = _analyze_code_slot_flow(code_or_tree)
    phys_hint = set(physical_slots)
    has_code = bool(flow["reads"] or flow["writes"])

    # Also check contract invariants for conservation / extrema hints
    extrema_inv_slots: Set[int] = set()
    for c in contracts or ():
        c_dict = c if isinstance(c, dict) else (c.model_dump() if hasattr(c, "model_dump") else {})
        for inv in c_dict.get("invariants") or ():
            iname = str(inv.get("variable_name") or "")
            if _EXTREMA_PREFIX_RE.match(iname):
                try:
                    inv_s = _state_slots(ast.parse(str(inv.get("assertion") or ""), mode="eval"))
                    if len(inv_s) == 1:
                        extrema_inv_slots |= inv_s
                    else:
                        extrema_inv_slots |= (inv_s - flow["physical"])
                except SyntaxError:
                    pass

    roles: Dict[str, str] = {}
    for name, raw_slot in vm.items():
        slot = int(raw_slot)
        exp = from_blueprint_role(explicit.get(name))

        # 1. Time step / simulation clock takes top precedence
        if slot in flow["clocks"] or _is_time_step_name(name) or exp == ROLE_TIME_STEP:
            roles[name] = ROLE_TIME_STEP
            continue

        # 1b. Code-verified evolving physical state variable (non-extrema, non-diagnostic name)
        if (
            has_code
            and slot in (flow["physical"] | phys_hint)
            and slot not in flow["extrema"]
            and not _is_diagnostic_name(name)
        ):
            roles[name] = ROLE_STATE
            continue

        # 2. Extrema or pure derived dataflow or explicit diagnostic
        is_derived_in_code = (
            has_code
            and slot in (flow["derived"] - phys_hint)
            and (bool(flow["dependencies"].get(slot)) or slot in flow["extrema"])
        )
        if exp == ROLE_DIAGNOSTIC or _is_diagnostic_name(name) or slot in flow["extrema"] or slot in extrema_inv_slots or is_derived_in_code:
            roles[name] = ROLE_DIAGNOSTIC
            continue

        # 3. Read-only control parameter (never written in kernel or variants)
        if has_code:
            all_written = flow.get("all_writes", flow["writes"])
            if (
                slot in flow["reads"]
                and slot not in all_written
                and (not flow.get("has_dynamic_write") or exp == ROLE_CONTROL_PARAM or _CONTROL_PARAM_NAME_RE.match(name) or name.lower() in ext_consts)
            ):
                roles[name] = ROLE_CONTROL_PARAM
                continue
            if exp == ROLE_CONTROL_PARAM and slot not in all_written:
                roles[name] = ROLE_CONTROL_PARAM
                continue
        else:
            if exp == ROLE_CONTROL_PARAM or _CONTROL_PARAM_NAME_RE.match(name) or name.lower() in ext_consts:
                roles[name] = ROLE_CONTROL_PARAM
                continue

        # 4. Evolving physical state
        roles[name] = ROLE_STATE
    return roles


def classify_slot_roles(
    variable_map: Optional[Dict[str, int]],
    code_or_tree: Any = "",
    contracts: Sequence[Any] = (),
    *,
    extracted_constants: Optional[Dict[str, float]] = None,
    explicit_roles: Optional[Dict[str, str]] = None,
    physical_slots: Sequence[int] = (),
) -> Dict[int, str]:
    """Return ``{slot_idx: role}`` from the shared :func:`classify_variable_roles`."""
    vm = dict(variable_map or {})
    var_roles = classify_variable_roles(
        vm, code_or_tree, contracts,
        extracted_constants=extracted_constants,
        explicit_roles=explicit_roles,
        physical_slots=physical_slots,
    )
    flow = _analyze_code_slot_flow(code_or_tree)
    phys_hint = set(physical_slots)
    priority = {ROLE_TIME_STEP: 4, ROLE_DIAGNOSTIC: 3, ROLE_CONTROL_PARAM: 2, ROLE_STATE: 1}
    by_slot: Dict[int, str] = {}
    for name, raw_slot in vm.items():
        slot = int(raw_slot)
        role = var_roles[name]
        prev = by_slot.get(slot)
        if prev is None or priority[role] > priority[prev]:
            by_slot[slot] = role

    # Also classify unmapped code/contract slots deterministically via AST flow
    for slot in flow["clocks"]:
        by_slot.setdefault(slot, ROLE_TIME_STEP)
    for slot in flow["extrema"]:
        by_slot.setdefault(slot, ROLE_DIAGNOSTIC)
    for slot in (flow["derived"] - phys_hint):
        if flow["dependencies"].get(slot):
            by_slot.setdefault(slot, ROLE_DIAGNOSTIC)
    for slot in (flow["reads"] - flow["writes"]):
        by_slot.setdefault(slot, ROLE_CONTROL_PARAM)
    for slot in flow["physical"]:
        by_slot.setdefault(slot, ROLE_STATE)
    return by_slot


def compile_initial_state_from_values(
    variable_map: Optional[Dict[str, int]],
    initial_values: Optional[Dict[str, float]],
    base_initial_state: Optional[Sequence[float]] = None,
) -> List[float]:
    """Deterministically compile ``initial_state`` by iterating ``variable_map`` indices 0..N-1."""
    vm = {str(k): int(v) for k, v in (variable_map or {}).items() if int(v) >= 0}
    init_vals = {str(k): float(v) for k, v in (initial_values or {}).items()}
    max_vm = max(vm.values(), default=-1)
    size = max(max_vm + 1, len(base_initial_state or ()))
    if size <= 0:
        return []
    out = [0.0] * size
    if base_initial_state:
        for i, val in enumerate(base_initial_state):
            if i < size:
                out[i] = float(val)
    if not init_vals or not vm:
        return out
    lower_vals = {k.lower(): v for k, v in init_vals.items()}
    stem_vals = {observable_name(k): v for k, v in init_vals.items()}
    by_slot: Dict[int, List[str]] = {}
    for name, idx in vm.items():
        by_slot.setdefault(idx, []).append(name)
    for idx in range(size):
        for name in by_slot.get(idx, ()):
            if name in init_vals:
                out[idx] = float(init_vals[name])
                break
            if name.lower() in lower_vals:
                out[idx] = float(lower_vals[name.lower()])
                break
            stem = observable_name(name)
            if stem in stem_vals:
                out[idx] = float(stem_vals[stem])
                break
    return out


def validate_slot_writes(code, variable_map, roles):
    slot_roles = classify_slot_roles(variable_map, code, explicit_roles=roles)
    name_by_slot: Dict[int, str] = {}
    for name, slot in (variable_map or {}).items():
        name_by_slot.setdefault(int(slot), name)
    for node in ast.walk(ast.parse(code)):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name)
                and node.value.value.id == 'buf' and node.value.attr == 'data'
                and isinstance(node.slice, ast.Constant)):
            slot = node.slice.value
            if not isinstance(slot, int):
                continue
            for name, role in (roles or {}).items():
                if role in ('read_only_parameter', ROLE_CONTROL_PARAM) and (variable_map or {}).get(name) == slot:
                    raise ValueError(f'SLOT_SCHEMA_ERROR: write to read-only parameter {name} at state[{slot}] line {node.lineno}')
            if slot_roles.get(slot) == ROLE_CONTROL_PARAM and name_by_slot.get(slot) in (roles or {}):
                pname = name_by_slot.get(slot, f'state_{slot}')
                raise ValueError(f'SLOT_SCHEMA_ERROR: write to read-only parameter {pname} at state[{slot}] line {node.lineno}')


def validate_contract_slots(contracts, variable_map: Optional[Dict[str, int]] = None):
    claims = {}
    extrema = {}
    identities = {}
    vm = dict(variable_map or {})
    vm_by_slot: Dict[int, List[str]] = {}
    for k, v in vm.items():
        vm_by_slot.setdefault(int(v), []).append(str(k))
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
                # If variable_map distinguishes instantaneous slot vs extrema slot,
                # reject extrema invariants (`peak_*`, `max_*`, `min_*`) pointing to the instantaneous slot.
                m_ext = _EXTREMA_PREFIX_RE.match(semantic)
                if m_ext and vm:
                    ext_prefix, ext_stem = m_ext.group(1).lower(), m_ext.group(2).lower()
                    slot_names = [n.lower() for n in vm_by_slot.get(slot, ())]
                    has_ext_in_slot = any(n.startswith(("peak_", "max_", "min_")) for n in slot_names)
                    if slot_names and not has_ext_in_slot:
                        stem_root = ext_stem.split("_")[0]
                        matching_ext = [
                            (vm_name, vm_slot) for vm_name, vm_slot in vm.items()
                            if vm_name.lower().startswith((f"{ext_prefix}_", "peak_", "max_", "min_"))
                            and stem_root in vm_name.lower()
                            and int(vm_slot) != slot
                        ]
                        if matching_ext:
                            ext_name, ext_slot = matching_ext[0]
                            raise ValueError(
                                f"SLOT_SCHEMA_ERROR: extrema invariant '{name}' references instantaneous slot "
                                f"state[{slot}] ({'/'.join(vm_by_slot.get(slot, ()))}) instead of extrema slot "
                                f"state[{ext_slot}] ({ext_name})"
                            )
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
                    _MODIFIERS = {
                        'semi', 'implicit', 'explicit', 'rk4', 'euler', 'midpoint', 'heun',
                        'patankar', 'stiff', 'baseline', 'corridor', 'reperfusion', 'ischemia',
                        'adaptive', 'initial', 'final', 'verified', 'positivity', 'nonnegative', 'nonneg',
                    }
                    def _core_tokens(label: str) -> set[str]:
                        return {t for t in label.lower().split('_') if t and t not in _MODIFIERS}
                    if not (_core_tokens(owner) & _core_tokens(semantic)):
                        raise ValueError(f'SLOT_SCHEMA_ERROR: distinct variables {owner}, {semantic} share state[{slot}]')


_NONZERO_T0_RE = re.compile(
    r"(?:\bt_?0\s*=\s*|\binitial\s+time\s*(?:=|of|is|at)\s*|\bstarting\s+at\s+t\s*=\s*)([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.I,
)
_STRICT_POS_SWEEP_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]{1,32})\s+(?:across|in|over|from)\s*\[\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.I,
)


def audit_contract_alignment(
    contracts: Sequence[Any],
    variable_map: Optional[Dict[str, int]],
    slot_roles: Optional[Dict[str, str]] = None,
    *,
    prompt: str = "",
    code: str = "",
) -> None:
    """Pre-execution alignment audit for TIME_STEP, CONTROL_PARAM, and extrema invariants."""
    vm = dict(variable_map or {})
    validate_contract_slots(contracts, variable_map=vm)
    var_roles = classify_variable_roles(vm, code, contracts, explicit_roles=slot_roles)

    # 1. Check whether prompt explicitly sets a nonzero t_0
    explicit_nonzero_t0 = False
    for m in _NONZERO_T0_RE.finditer(prompt or ""):
        try:
            if abs(float(m.group(1))) > 1e-12:
                explicit_nonzero_t0 = True
                break
        except ValueError:
            pass

    # 2. Extract strictly positive prompt-specified control parameter ranges [lo, hi] with lo > 0
    positive_param_ranges: List[Tuple[str, float, float]] = []
    for m in _STRICT_POS_SWEEP_RE.finditer(prompt or ""):
        try:
            lo = float(m.group(2))
            hi = float(m.group(3))
        except ValueError:
            continue
        if lo > 0.0 and hi > lo:
            positive_param_ranges.append((m.group(1).lower(), lo, hi))

    has_internal_sweep = bool(
        code and re.search(r"\bfor\s+\w+\s+in\s+range\(", code)
        and any(p_name in code.lower() for p_name, _, _ in positive_param_ranges)
    )

    for contract in contracts or ():
        c = contract if isinstance(contract, dict) else contract.model_dump()
        init = c.get("initial_state") or []
        if not init:
            continue
        cid = str(c.get("contract_id") or "")
        for name, raw_slot in vm.items():
            slot = int(raw_slot)
            if slot < 0 or slot >= len(init):
                continue
            val = float(init[slot])
            role = var_roles.get(name)
            if role == ROLE_TIME_STEP and not explicit_nonzero_t0 and abs(val) > 1e-12:
                raise ValueError(
                    f"SLOT_ALIGNMENT_ERROR: contract '{cid}' initializes TIME_STEP slot "
                    f"'{name}' (state[{slot}]) to {val!r} instead of 0.0 "
                    "(possible initial_state/variable_map positional shift)"
                )
            if role == ROLE_CONTROL_PARAM and abs(val) <= 1e-15 and not has_internal_sweep:
                nlow = name.lower()
                for p_stem, lo, hi in positive_param_ranges:
                    if nlow == p_stem or nlow.startswith(p_stem + "_"):
                        raise ValueError(
                            f"SLOT_ALIGNMENT_ERROR: contract '{cid}' initializes strictly positive "
                            f"CONTROL_PARAM '{name}' (state[{slot}]) to 0.0, outside prompt range "
                            f"[{lo:g}, {hi:g}] (possible initial_state/variable_map positional shift)"
                        )


def _bound_directions_in_assertion(assertion: str) -> Tuple[bool, bool]:
    """Return ``(has_lower, has_upper)`` finite numeric bounds for an assertion."""
    try:
        tree = ast.parse(assertion, mode="eval").body
    except SyntaxError:
        return False, False
    has_lower = False
    has_upper = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        lhs = node.left
        for op, rhs in zip(node.ops, node.comparators):
            # abs(x - c) <= w or abs(x - c) < w encodes a two-sided band
            if (isinstance(op, (ast.Lt, ast.LtE))
                    and isinstance(lhs, ast.Call) and isinstance(lhs.func, ast.Name) and lhs.func.id == "abs"
                    and _state_slots(lhs)):
                has_lower = True
                has_upper = True
            elif (isinstance(op, (ast.Gt, ast.GtE))
                    and isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id == "abs"
                    and _state_slots(rhs)):
                has_lower = True
                has_upper = True
            else:
                l_slots = _state_slots(lhs)
                r_slots = _state_slots(rhs)
                if l_slots and not r_slots:
                    if isinstance(op, (ast.Gt, ast.GtE)):
                        has_lower = True
                    elif isinstance(op, (ast.Lt, ast.LtE)):
                        has_upper = True
                    elif isinstance(op, ast.Eq):
                        has_lower = True
                        has_upper = True
                elif r_slots and not l_slots:
                    if isinstance(op, (ast.Lt, ast.LtE)):
                        has_lower = True
                    elif isinstance(op, (ast.Gt, ast.GtE)):
                        has_upper = True
                    elif isinstance(op, ast.Eq):
                        has_lower = True
                        has_upper = True
                elif l_slots and r_slots:
                    if isinstance(op, (ast.Lt, ast.LtE)):
                        has_upper = True
                    elif isinstance(op, (ast.Gt, ast.GtE)):
                        has_lower = True
            lhs = rhs
    return has_lower, has_upper


def validate_contract_faithfulness(
    contracts: Sequence[Any],
    variable_map: Optional[Dict[str, int]] = None,
    prompt: str = "",
    code: str = "",
    *,
    require_sweep_coverage: bool = False,
) -> None:
    """Reject hollowed two-sided bands (`state[i] >= 0.0`), hollow upper ceilings, and degenerate sweeps."""
    vm = dict(variable_map or {})
    for contract in contracts or ():
        c = contract if isinstance(contract, dict) else contract.model_dump()
        cid = str(c.get("contract_id") or "")
        for inv in c.get("invariants") or ():
            name = str(inv.get("variable_name") or "")
            assertion = str(inv.get("assertion") or "")
            if _BAND_NAME_RE.search(name):
                has_lo, has_hi = _bound_directions_in_assertion(assertion)
                if not (has_lo and has_hi):
                    raise ValueError(
                        f"CONTRACT_FAITHFULNESS_ERROR: two-sided band/window invariant "
                        f"'{name}' in '{cid}' degraded to one-sided/trivial assertion "
                        f"'{assertion}'; encode both finite lower and upper bounds."
                    )
            elif (_UPPER_CEIL_NAME_RE.search(name)
                  and not name.lower().startswith(("min_", "lower_"))
                  and not name.lower().endswith(("_nonneg", "_nonnegative", "_positivity", "_finite"))):
                _has_lo, has_hi = _bound_directions_in_assertion(assertion)
                if not has_hi:
                    raise ValueError(
                        f"CONTRACT_FAITHFULNESS_ERROR: upper-ceiling invariant "
                        f"'{name}' in '{cid}' lacks a finite upper bound in '{assertion}'."
                    )

    if require_sweep_coverage and prompt:
        sweep_params: List[str] = []
        for m in _STRICT_POS_SWEEP_RE.finditer(prompt):
            sweep_params.append(m.group(1).lower())
        if not sweep_params and re.search(r"\bsweep\b", prompt, re.I):
            for name, role in classify_variable_roles(vm, code, contracts).items():
                if role == ROLE_CONTROL_PARAM:
                    sweep_params.append(name.lower())
        if sweep_params and len(contracts or ()) >= 2:
            roles = classify_variable_roles(vm, code, contracts)
            ctrl_slots = [
                int(slot) for name, slot in vm.items()
                if roles.get(name) == ROLE_CONTROL_PARAM
                and any(name.lower() == sp or name.lower().startswith(sp + "_") for sp in sweep_params)
            ]
            if ctrl_slots:
                vectors = [
                    tuple(
                        round(float((c if isinstance(c, dict) else c.model_dump()).get("initial_state", [])[s]), 12)
                        if s < len((c if isinstance(c, dict) else c.model_dump()).get("initial_state") or [])
                        else 0.0
                        for s in ctrl_slots
                    )
                    for c in contracts
                ]
                has_code_sweep = bool(
                    code and re.search(r"\bfor\s+\w+\s+in\s+range\(", code)
                    and any(sp in code.lower() for sp in sweep_params)
                )
                if len(set(vectors)) <= 1 and not has_code_sweep:
                    raise ValueError(
                        "CONTRACT_FAITHFULNESS_ERROR: prompt requests a parameter sweep, "
                        "but all contracts duplicate identical CONTROL_PARAM initial_state values "
                        "with no sweep loop in the kernel."
                    )


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

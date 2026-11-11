"""Deterministic AST sanitizer and directive applier for research kernels.

Tier-0 healing is intentionally conservative: it only applies semantics-
closing rewrites for scalar math intrinsics, math-domain guards, and the
contract-directed monotonicity/drift repair.  LLM healing never emits code;
it emits :class:`HealingDirective` dictionaries which are applied here.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MATH_FUNCS = {
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",
    "cosh", "tanh", "exp", "log", "log10", "log2", "log1p", "expm1", "ceil",
    "floor", "pow", "fabs", "fmod", "hypot", "copysign", "trunc", "isfinite",
    "isnan", "isinf", "radians", "degrees",
}
MATH_CONSTANTS = {"pi", "e", "tau"}
INTRINSIC_ALIASES = {
    "clamp": "clamp", "clip": "clip",
    "sign": "sign", "sgn": "sgn",
    "sq": "sq", "lerp": "lerp", "safe_div": "safe_div",
}
NP_MATH_FUNCS = MATH_FUNCS | {"abs", "clip", "sign", "sgn"}
MONOTONE_NAME_TOKENS = ("entropy_drift", "entropy_monotonicity",
                        "free_energy_drift", "lyapunov_drift")


@dataclass(frozen=True)
class SanitizerResult:
    code: str
    notes: List[str]
    changed: bool


def _defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _call_name(func: ast.expr) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return None


def _has_math_import(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == "math" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "math":
            return True
    return False


def _is_none_const(node: Optional[ast.AST]) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


class _NoneLiteralTransformer(ast.NodeTransformer):
    """Lower Python ``None`` to the numeric C-ABI dialect before transpile."""

    def __init__(self) -> None:
        self.changed = False
        self.notes: set[str] = set()
        self._void_stack: List[bool] = []

    def _has_value_return(self, fn: ast.FunctionDef) -> bool:
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and node.value is not None \
                    and not _is_none_const(node.value):
                return True
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._void_stack.append(not self._has_value_return(node))
        try:
            self.generic_visit(node)
        finally:
            self._void_stack.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if _is_none_const(node.value):
            self.changed = True
            self.notes.add("return None lowered")
            node.value = None if self._void_stack and self._void_stack[-1] else _num(0.0)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if _is_none_const(node.value):
            self.changed = True
            self.notes.add("None assignment lowered to 0.0")
            node.value = _num(0.0)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        if node.value is not None and _is_none_const(node.value):
            self.changed = True
            self.notes.add("None assignment lowered to 0.0")
            node.value = _num(0.0)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1 and len(node.comparators) == 1:
            op = node.ops[0]
            if isinstance(op, (ast.Is, ast.IsNot)) and (
                    _is_none_const(node.left) or _is_none_const(node.comparators[0])):
                self.changed = True
                self.notes.add("is None comparison lowered")
                value = isinstance(op, ast.IsNot)
                return ast.copy_location(ast.Constant(value=value), node)
        return node


def _strip_none_literals_tree(tree: ast.Module) -> Tuple[ast.Module, bool, List[str]]:
    transformer = _NoneLiteralTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    return tree, transformer.changed, sorted(transformer.notes)


def _ensure_math_import(tree: ast.Module) -> bool:
    if _has_math_import(tree):
        return False
    idx = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            idx = i + 1
    tree.body.insert(idx, ast.Import(names=[ast.alias(name="math")]))
    ast.fix_missing_locations(tree)
    return True


def _intrinsic_defs(used: Iterable[str]) -> List[ast.FunctionDef]:
    """AST helper definitions for scalar intrinsics missing from builtins."""
    src = {
        "clamp": "def clamp(x, lo, hi):\n    return min(max(x, lo), hi)\n",
        "clip": "def clip(x, lo, hi):\n    return clamp(x, lo, hi)\n",
        "sign": "def sign(x):\n    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)\n",
        "sgn": "def sgn(x):\n    return sign(x)\n",
        "sq": "def sq(x):\n    return x * x\n",
        "lerp": "def lerp(a, b, t):\n    return a + t * (b - a)\n",
        "safe_div": "def safe_div(a, b):\n    return a / (b if abs(b) > 1e-30 else 1e-30)\n",
    }
    out: List[ast.FunctionDef] = []
    for name in sorted(set(used) & set(src)):
        out.extend(ast.parse(src[name]).body)
    return out


def _is_guarded_log(arg: ast.expr) -> bool:
    return isinstance(arg, ast.Call) and _call_name(arg.func) in {"max", "math.fmax"}


def _is_guarded_sqrt(arg: ast.expr) -> bool:
    return isinstance(arg, ast.Call) and _call_name(arg.func) in {"max", "math.fmax"}


def _is_guarded_unit_interval(arg: ast.expr) -> bool:
    if not isinstance(arg, ast.Call) or _call_name(arg.func) != "min":
        return False
    return len(arg.args) >= 1


def _num(value: float) -> ast.Constant:
    return ast.Constant(value=value)


def _domain_guarded_call(node: ast.Call, eps: float) -> Optional[ast.Call]:
    """Return a domain-safe replacement for math.log/sqrt/asin/acos calls."""
    name = _call_name(node.func)
    if name not in {"math.log", "math.sqrt", "math.asin", "math.acos"} or not node.args:
        return None
    arg = node.args[0]
    if name == "math.log":
        if _is_guarded_log(arg):
            return None
        guarded = ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                           args=[arg, _num(eps)], keywords=[])
    elif name == "math.sqrt":
        if _is_guarded_sqrt(arg):
            return None
        guarded = ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                           args=[arg, _num(0.0)], keywords=[])
    else:
        if _is_guarded_unit_interval(arg):
            return None
        inner = ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                         args=[arg, _num(-1.0)], keywords=[])
        guarded = ast.Call(func=ast.Name(id="min", ctx=ast.Load()),
                           args=[inner, _num(1.0)], keywords=[])
    node.args[0] = guarded
    return node


class _IntrinsicTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.defined: set[str] = set()
        self.used_intrinsics: set[str] = set()
        self.used_math: set[str] = set()
        self.used_np: set[str] = set()
        self.notes: set[str] = set()
        self.guard_eps = 1e-300

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = _call_name(node.func)
        if name is None:
            return node
        base, _, attr = name.partition(".")
        if base in {"np", "numpy"} and attr:
            self.used_np.add(attr)
            if attr == "clip":
                self.used_intrinsics.add("clip")
                node.func = ast.Name(id="clip", ctx=ast.Load())
            elif attr in {"sign", "sgn", "sq", "lerp", "safe_div"}:
                self.used_intrinsics.add(attr)
                node.func = ast.Name(id=attr, ctx=ast.Load())
            elif attr == "abs":
                node.func = ast.Name(id="abs", ctx=ast.Load())
            elif attr in MATH_FUNCS:
                self.used_math.add(attr)
                node.func = ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()),
                                          attr=attr, ctx=ast.Load())
            else:
                self.notes.add(f"unsupported np.{attr} left unchanged")
            guarded = _domain_guarded_call(node, self.guard_eps)
            return guarded if guarded is not None else node
        if "." not in name:
            if name in MATH_FUNCS and name not in self.defined and name not in {"abs", "min", "max", "sum", "len", "range", "int", "float"}:
                self.used_math.add(name)
                node.func = ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()),
                                          attr=name, ctx=ast.Load())
                name = f"math.{name}"
            elif name in INTRINSIC_ALIASES and name not in self.defined:
                self.used_intrinsics.add(name)
        guarded = _domain_guarded_call(node, self.guard_eps)
        return guarded if guarded is not None else node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in MATH_CONSTANTS \
                and node.id not in self.defined and node.id not in {"pi", "e", "tau"}:
            # Constants are rewritten only when not shadowed; pi/e/tau are
            # handled by the sandbox namespace and math.* attributes.
            pass
        if isinstance(node.ctx, ast.Load) and node.id in {"pi", "e"} \
                and node.id not in self.defined:
            # Bare pi/e are common LLM mistakes; math.pi/math.e is canonical.
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()),
                              attr=node.id, ctx=ast.Load()), node)
        return node


def _insert_intrinsic_helpers(tree: ast.Module, names: Iterable[str]) -> bool:
    defs = _intrinsic_defs(names)
    if not defs:
        return False
    idx = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            idx = i + 1
    tree.body[idx:idx] = defs
    ast.fix_missing_locations(tree)
    return True


def sanitize_intrinsics(code: str, *, guard_domains: bool = True,
                        eps: float = 1e-300) -> SanitizerResult:
    """Rewrite scalar intrinsic/math-namespace hazards before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    original = ast.dump(tree, annotate_fields=False, include_attributes=False)
    tree, none_changed, none_notes = _strip_none_literals_tree(tree)
    transformer = _IntrinsicTransformer()
    transformer.defined = _defined_names(tree)
    transformer.guard_eps = float(eps)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    changed = ast.dump(tree, annotate_fields=False, include_attributes=False) != original
    notes = sorted(set(none_notes) | set(transformer.notes))
    if none_changed:
        changed = True
    if transformer.used_math or transformer.used_np or transformer.used_intrinsics:
        if _ensure_math_import(tree):
            notes.append("injected import math")
        changed = True
    if _insert_intrinsic_helpers(tree, transformer.used_intrinsics):
        notes.append("injected scalar intrinsics: " + ", ".join(sorted(transformer.used_intrinsics)))
        changed = True
    if guard_domains:
        # Second pass guards math.log/sqrt/asin/acos introduced above.
        before = ast.dump(tree, annotate_fields=False, include_attributes=False)
        tree = _DomainGuardTransformer(eps=float(eps)).visit(tree)
        ast.fix_missing_locations(tree)
        if ast.dump(tree, annotate_fields=False, include_attributes=False) != before:
            notes.append("guarded math domains (log/sqrt/asin/acos)")
            changed = True
    out = ast.unparse(tree) + ("\n" if code.endswith("\n") or not code.endswith("\n") else "")
    return SanitizerResult(out, notes, changed)


class _DomainGuardTransformer(ast.NodeTransformer):
    def __init__(self, eps: float) -> None:
        self.eps = float(eps)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        guarded = _domain_guarded_call(node, self.eps)
        return guarded if guarded is not None else node


def _target_slot(node: ast.AST) -> Optional[int]:
    if not isinstance(node, ast.Subscript):
        return None
    sl = node.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
        return int(sl.value)
    return None


def _expr_free_vars(expr: ast.AST) -> List[str]:
    names = sorted({n.id for n in ast.walk(expr)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})
    return names


def _eval_numeric_expr(expr: ast.AST, env: Dict[str, float]) -> Optional[float]:
    try:
        value = eval(compile(ast.Expression(expr), "<egg_guard>", "eval"),  # noqa: S307
                     {"__builtins__": {}}, dict(env))
    except Exception:  # noqa: BLE001
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def numeric_expr_equivalent(left: ast.AST, right: ast.AST, *,
                            tolerance: float = 1e-12,
                            points: int = 3) -> bool:
    """Reject algebraic rewrites that change numeric evaluation.

    Free variables are assigned three deterministic non-zero pseudo-random
    values; any absolute deviation above ``tolerance`` rejects the rewrite.
    """
    vars_ = sorted(set(_expr_free_vars(left)) | set(_expr_free_vars(right)))
    for point in range(max(1, int(points))):
        env = {name: 0.37 + 0.13 * idx + 0.031 * point + 0.017 * idx * point
               for idx, name in enumerate(vars_)}
        a = _eval_numeric_expr(left, env)
        b = _eval_numeric_expr(right, env)
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
            return False
        if abs(a - b) > float(tolerance):
            return False
    return True


def strip_none_literals(code: str) -> SanitizerResult:
    """Public Tier-0 entry point for ``STRIP_NONE_LITERALS`` directives."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    original = ast.dump(tree, annotate_fields=False, include_attributes=False)
    tree, changed, notes = _strip_none_literals_tree(tree)
    if not changed:
        return SanitizerResult(code, [], False)
    return SanitizerResult(ast.unparse(tree) + "\n", notes, True)


def _is_state_target(node: ast.AST, slot: int) -> bool:
    if _target_slot(node) != slot:
        return False
    value = node.value  # type: ignore[attr-defined]
    if isinstance(value, ast.Name) and value.id in {"buf", "state", "data"}:
        return True
    if isinstance(value, ast.Attribute) and value.attr == "data" \
            and isinstance(value.value, ast.Name) and value.value.id == "buf":
        return True
    return False


def _var_kind(name: str) -> str:
    low = name.lower()
    if any(tok in low for tok in ("before", "prev", "s0", "old", "initial")):
        return "before"
    if any(tok in low for tok in ("after", "next", "s1", "new")):
        return "after"
    return ""


def _abs_diff(node: ast.AST) -> Optional[Tuple[str, str]]:
    if not isinstance(node, ast.Call):
        return None
    if _call_name(node.func) not in {"abs", "math.fabs"} or len(node.args) != 1:
        return None
    arg = node.args[0]
    if not isinstance(arg, ast.BinOp) or not isinstance(arg.op, ast.Sub):
        return None
    if not isinstance(arg.left, ast.Name) or not isinstance(arg.right, ast.Name):
        return None
    return arg.left.id, arg.right.id


def _monotone_expr(before: str, after: str, direction: str) -> ast.Call:
    if direction == "non_decreasing":
        left, right = before, after
    else:
        left, right = after, before
    diff = ast.BinOp(left=ast.Name(id=left, ctx=ast.Load()), op=ast.Sub(),
                     right=ast.Name(id=right, ctx=ast.Load()))
    return ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                    args=[_num(0.0), diff], keywords=[])


class _MonotoneRewrite(ast.NodeTransformer):
    def __init__(self, slot: int, direction: str,
                 before_var: Optional[str], after_var: Optional[str]) -> None:
        self.slot = int(slot)
        self.direction = direction if direction in {"non_increasing", "non_decreasing"} else "non_increasing"
        self.before_var = before_var
        self.after_var = after_var
        self.changed = False

    def _replacement(self, node: ast.Call) -> Optional[ast.Call]:
        pair = _abs_diff(node)
        if pair is None:
            return None
        a, b = pair
        before, after = self.before_var, self.after_var
        if before is None or after is None:
            ka, kb = _var_kind(a), _var_kind(b)
            if ka == "before" and kb == "after":
                before, after = a, b
            elif kb == "before" and ka == "after":
                before, after = b, a
            elif self.direction == "non_increasing":
                # Fallback heuristic: *_after/*_s1 is the post-step value.
                before, after = (b, a) if _var_kind(b) == "before" or _var_kind(a) == "after" else (a, b)
            else:
                before, after = (a, b) if _var_kind(a) == "before" or _var_kind(b) == "after" else (b, a)
        if not before or not after:
            return None
        self.changed = True
        return _monotone_expr(before, after, self.direction)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        repl = self._replacement(node)
        return repl if repl is not None else node


def rewrite_monotone_drift(code: str, *, slot: int, direction: str = "non_increasing",
                           before_var: Optional[str] = None,
                           after_var: Optional[str] = None) -> SanitizerResult:
    """Rewrite ``abs(after-before)`` drift accumulators to one-sided violations."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    original = ast.dump(tree, annotate_fields=False, include_attributes=False)

    class _SlotTransformer(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            self.generic_visit(node)
            if any(_is_state_target(t, slot) for t in node.targets):
                rw = _MonotoneRewrite(slot, direction, before_var, after_var)
                node.value = rw.visit(node.value)
                self.changed = self.changed or rw.changed
            return node

        def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
            self.generic_visit(node)
            if _is_state_target(node.target, slot) and isinstance(node.op, ast.Add):
                rw = _MonotoneRewrite(slot, direction, before_var, after_var)
                node.value = rw.visit(node.value)
                self.changed = self.changed or rw.changed
            return node

        changed = False

    transformer = _SlotTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    changed = transformer.changed or ast.dump(tree, annotate_fields=False,
                                              include_attributes=False) != original
    notes = [f"rewrote monotone drift state[{slot}] ({direction})"] if transformer.changed else []
    return SanitizerResult(ast.unparse(tree) + "\n", notes, changed)


def infer_monotone_drift_directives(contracts: Sequence[Any], code: str) -> List[Dict[str, Any]]:
    """Infer Tier-0 drift directives from falsifiable contract metadata."""
    out: List[Dict[str, Any]] = []
    for contract in contracts or []:
        for inv in getattr(contract, "invariants", None) or []:
            name = str(getattr(inv, "variable_name", "") or "").lower()
            if not any(tok in name for tok in MONOTONE_NAME_TOKENS):
                continue
            if float(getattr(inv, "tolerance", 1.0) or 1.0) > 1e-8:
                continue
            text = f"{getattr(inv, 'variable_name', '')} {getattr(inv, 'assertion', '')}"
            m = re.search(r"(?:state|buf(?:\.data)?|data)\[(\d+)\]\s*<=\s*([0-9.eE+-]+)", text)
            if not m:
                continue
            slot, bound = int(m.group(1)), float(m.group(2))
            if bound > 1e-6:
                continue
            direction = "non_increasing"
            if "entropy" in name and "relative" not in name and "free_energy" not in name:
                direction = "non_decreasing"
            if rewrite_monotone_drift(code, slot=slot, direction=direction).changed:
                out.append({"op": "REWRITE_MONOTONE_DRIFT", "slot": slot,
                            "direction": direction})
    return out




def _slot_index(node: ast.AST) -> Optional[int]:
    return _target_slot(node)


def _state_slot_expr(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    base = node.value
    name = ""
    if isinstance(base, ast.Name):
        name = base.id
    elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
        name = f"{base.value.id}.{base.attr}"
    return name in {"state", "data", "buf.data", "buf"} and _target_slot(node) is not None


def _choose_prev_slot(code: str, slot: int, contracts: Sequence[Any] = ()) -> Optional[int]:
    used = {int(m.group(1)) for m in re.finditer(r"(?:state|buf\.data|data)\[(\d+)\]", code)}
    used.add(slot)
    sizes = [len(getattr(contract, "initial_state", None) or [])
             for contract in contracts or []]
    sizes = [n for n in sizes if n]
    limit = max(sizes) if sizes else 16
    for cand in (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16):
        if cand > slot and cand < limit and cand not in used:
            return cand
    return None


def _drift_self_pattern(body: List[ast.stmt], slot: int):
    for i, stmt in enumerate(body[:-1]):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 \
                or not isinstance(stmt.targets[0], ast.Name) or not isinstance(stmt.value, ast.BinOp) \
                or not isinstance(stmt.value.op, ast.Sub):
            continue
        diff = stmt.targets[0].id
        left, right = stmt.value.left, stmt.value.right
        if _state_slot_expr(right) and _slot_index(right) == slot and isinstance(left, ast.Name):
            obs = left.id
            expr = f"{obs} - buf.data[{slot}]"
        elif _state_slot_expr(left) and _slot_index(left) == slot and isinstance(right, ast.Name):
            obs = right.id
            expr = f"buf.data[{slot}] - {obs}"
        else:
            continue
        nxt = body[i + 1]
        if not isinstance(nxt, ast.If):
            continue
        test = nxt.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            continue
        if not (isinstance(test.left, ast.Name) and test.left.id == diff
                and isinstance(test.ops[0], (ast.Gt, ast.GtE))):
            continue
        assigns_slot = False
        for sub in ast.walk(nxt):
            if isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if _state_slot_expr(target) and _slot_index(target) == slot:
                        assigns_slot = True
        if not assigns_slot:
            continue
        return i, diff, obs, expr
    return None


def repair_drift_self_comparison(code: str, *, slot: int, prev_slot: int,
                                 contracts: Sequence[Any] = ()) -> SanitizerResult:
    """Repair ``diff = obs - buf.data[error_slot]`` self-comparisons."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SanitizerResult(code, [], False)
    changed = False
    notes: List[str] = []
    var = f"_step_drift_{slot}"
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        body = fn.body
        found = _drift_self_pattern(body, int(slot))
        if found is None:
            continue
        i, diff, obs, expr = found
        expr = expr.replace(f"buf.data[{slot}]", f"buf.data[{prev_slot}]")
        expr = expr.replace(f"state[{slot}]", f"state[{prev_slot}]")
        expr = expr.replace(f"data[{slot}]", f"data[{prev_slot}]")
        snippet = (
            "if buf.steps > 0:\n"
            f"    {var} = max(0.0, {expr})\n"
            f"    if {var} > buf.data[{slot}]:\n"
            f"        buf.data[{slot}] = {var}\n"
            f"buf.data[{prev_slot}] = {obs}\n"
        )
        repl = ast.parse(snippet).body
        # Keep any unrelated statements between the diff assignment and guard.
        body[i:i + 2] = repl
        ast.fix_missing_locations(tree)
        changed = True
        notes.append(f"repaired drift self-comparison state[{slot}] via buf.data[{prev_slot}]")
        break
    if not changed:
        return SanitizerResult(code, [], False)
    return SanitizerResult(ast.unparse(tree) + "\n", notes, True)


def infer_drift_self_comparison_directives(contracts: Sequence[Any],
                                           code: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for contract in contracts or []:
        for inv in getattr(contract, "invariants", None) or []:
            name = str(getattr(inv, "variable_name", "") or "").lower()
            assertion = str(getattr(inv, "assertion", "") or "")
            if not any(tok in name for tok in ("drift", "error", "err", "residual", "violation", "mismatch")):
                continue
            if float(getattr(inv, "tolerance", 1.0) or 1.0) > 1e-6:
                continue
            m = re.search(r"(?:state|buf(?:\.data)?|data)\[(\d+)\]\s*<=\s*([0-9.eE+-]+)",
                          f"{name} {assertion}")
            if not m:
                continue
            slot = int(m.group(1))
            bound = float(m.group(2))
            if bound > 1e-6 or slot in seen:
                continue
            seen.add(slot)
            prev = _choose_prev_slot(code, slot, contracts)
            if prev is None:
                continue
            if repair_drift_self_comparison(code, slot=slot, prev_slot=prev,
                                            contracts=contracts).changed:
                out.append({"op": "REPAIR_DRIFT_SELF_COMPARISON", "slot": slot,
                            "prev_slot": prev})
    return out


def _wrap_substep(tree: ast.Module, substeps: int) -> bool:
    n = max(1, int(substeps or 1))
    if n == 1:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "simulation_step" and node.args.args:
            dt_arg = node.args.args[1].arg if len(node.args.args) > 1 else "dt"
            loop = ast.For(
                target=ast.Name(id="__healing_substep", ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                              args=[ast.Constant(value=n)], keywords=[]),
                body=node.body, orelse=[], type_comment=None)
            scale = ast.Assign(
                targets=[ast.Name(id=dt_arg, ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id=dt_arg, ctx=ast.Load()), op=ast.Div(),
                                right=ast.Constant(value=float(n))),
                type_comment=None)
            node.body = [scale, loop]
            ast.fix_missing_locations(node)
            return True
    return False


def _scale_coefficient(tree: ast.Module, var: str, factor: float) -> bool:
    factor = float(factor)
    if factor == 1.0:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "simulation_step":
            node.body.insert(0, ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id=var, ctx=ast.Load()), op=ast.Mult(),
                                right=ast.Constant(value=factor)),
                type_comment=None))
            ast.fix_missing_locations(node)
            return True
    return False


def apply_healing_directives(code: str, directives: Iterable[Dict[str, Any]],
                             *, contracts: Sequence[Any] = ()) -> SanitizerResult:
    """Apply LLM-emitted structured directives deterministically."""
    notes: List[str] = []
    current = code
    # Always run the intrinsic/domain sanitizer first (Tier-0, zero tokens).
    san = sanitize_intrinsics(current)
    if san.changed:
        current = san.code
        notes.extend(san.notes)
    try:
        tree = ast.parse(current)
    except SyntaxError:
        return SanitizerResult(current, notes, bool(notes))
    before_dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    for raw in directives or []:
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("op") or "")
        if op == "INLINE_INTRINSIC":
            name = str(raw.get("name") or "")
            if name in INTRINSIC_ALIASES:
                patched = _insert_intrinsic_helpers(tree, [name])
                if patched:
                    notes.append(f"INLINE_INTRINSIC({name})")
        elif op == "GUARD_MATH_DOMAINS":
            eps = float(raw.get("eps", 1e-300) or 1e-300)
            guarded = _DomainGuardTransformer(eps=eps).visit(tree)
            if guarded is not tree:
                notes.append("GUARD_MATH_DOMAINS")
            tree = guarded
        elif op == "REWRITE_MONOTONE_DRIFT":
            slot = int(raw.get("slot", -1))
            if slot >= 0:
                # Re-render through the dedicated rewriter for precise notes.
                rw = rewrite_monotone_drift(ast.unparse(tree) + "\n", slot=slot,
                                            direction=str(raw.get("direction") or "non_increasing"),
                                            before_var=raw.get("before_var"),
                                            after_var=raw.get("after_var"))
                if rw.changed:
                    tree = ast.parse(rw.code)
                    notes.extend(rw.notes)
        elif op == "WRAP_SUBSTEP":
            if _wrap_substep(tree, int(raw.get("substeps", 1) or 1)):
                notes.append(f"WRAP_SUBSTEP({int(raw.get('substeps', 1) or 1)})")
        elif op == "SCALE_COEFFICIENT":
            var = str(raw.get("var") or "")
            if var and _scale_coefficient(tree, var, float(raw.get("factor", 1.0) or 1.0)):
                notes.append(f"SCALE_COEFFICIENT({var},{float(raw.get('factor', 1.0) or 1.0)})")
        elif op == "REPAIR_DRIFT_SELF_COMPARISON":
            slot = int(raw.get("slot", -1))
            prev = int(raw.get("prev_slot", -1))
            if slot >= 0 and prev >= 0:
                rw = repair_drift_self_comparison(ast.unparse(tree) + "\n", slot=slot,
                                                 prev_slot=prev, contracts=contracts)
                if rw.changed:
                    tree = ast.parse(rw.code)
                    notes.extend(rw.notes)
        elif op == "STRIP_NONE_LITERALS":
            stripped = strip_none_literals(ast.unparse(tree) + "\n")
            if stripped.changed:
                tree = ast.parse(stripped.code)
                notes.append("STRIP_NONE_LITERALS")
        elif op == "CLAMP_STATE_NORM":
            notes.append("CLAMP_STATE_NORM recorded (no scalar AST rewrite)")
        elif op == "BIND_SLOT_TO_VAR":
            notes.append(f"BIND_SLOT_TO_VAR({raw.get('slot')},{raw.get('var')})")
        else:
            notes.append(f"ignored unknown directive {op or '<missing>'}")
    # Contract-inferred deterministic drift repairs run without an LLM.
    current_code = ast.unparse(tree) + "\n"
    inferred = infer_monotone_drift_directives(contracts, current_code)
    inferred += infer_drift_self_comparison_directives(contracts, current_code)
    for d in inferred:
        if d.get("op") == "REPAIR_DRIFT_SELF_COMPARISON":
            rw = repair_drift_self_comparison(ast.unparse(tree) + "\n", slot=int(d["slot"]),
                                             prev_slot=int(d["prev_slot"]), contracts=contracts)
        else:
            rw = rewrite_monotone_drift(ast.unparse(tree) + "\n", slot=int(d["slot"]),
                                        direction=str(d.get("direction") or "non_increasing"))
        if rw.changed:
            tree = ast.parse(rw.code)
            notes.extend(rw.notes)
    ast.fix_missing_locations(tree)
    after_dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    ast_changed = after_dump != before_dump
    changed = bool(san.changed or ast_changed)
    out = ast.unparse(tree) + "\n"
    return SanitizerResult(out if changed else code, sorted(set(notes)), changed)


__all__ = [
    "SanitizerResult", "sanitize_intrinsics", "rewrite_monotone_drift",
    "infer_monotone_drift_directives", "apply_healing_directives",
    "strip_none_literals", "numeric_expr_equivalent",
    "repair_drift_self_comparison", "infer_drift_self_comparison_directives",
    "MATH_FUNCS", "INTRINSIC_ALIASES",
]

"""Deterministic AST sanitizer and directive applier for research kernels.

Tier-0 healing is intentionally conservative: it only applies semantics-
closing rewrites for scalar math intrinsics, math-domain guards, and the
contract-directed monotonicity/drift repair.  LLM healing never emits code;
it emits :class:`HealingDirective` dictionaries which are applied here.
"""
from __future__ import annotations

import ast
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
    transformer = _IntrinsicTransformer()
    transformer.defined = _defined_names(tree)
    transformer.guard_eps = float(eps)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    changed = ast.dump(tree, annotate_fields=False, include_attributes=False) != original
    notes = sorted(transformer.notes)
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
        elif op == "CLAMP_STATE_NORM":
            notes.append("CLAMP_STATE_NORM recorded (no scalar AST rewrite)")
        elif op == "BIND_SLOT_TO_VAR":
            notes.append(f"BIND_SLOT_TO_VAR({raw.get('slot')},{raw.get('var')})")
        else:
            notes.append(f"ignored unknown directive {op or '<missing>'}")
    # Contract-inferred deterministic drift repairs run without an LLM.
    inferred = infer_monotone_drift_directives(contracts, ast.unparse(tree) + "\n")
    for d in inferred:
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
    "MATH_FUNCS", "INTRINSIC_ALIASES",
]

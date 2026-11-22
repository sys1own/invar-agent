"""Cumulative blueprint evolution: unified-module accretion and pinning.

A *unified module* is the concatenation of all verified layers plus the
active candidate layer. Each verified stage ``k`` is pinned by renaming its
driver ``simulation_step`` to ``layer_k_step`` so the compiled unified
module exports ``execute_layer_k_step`` alongside the new
``execute_step`` entry point. Sandbox_Execute can therefore re-verify every
pinned layer *inside the same compiled artefact* as the new logic.

The merger validator enforces additive/coupled integration:

- pinned functions can never be overwritten or deleted by a candidate
  (overwrites are reverted to the verified version and reported);
- the candidate must define ``simulation_step(buf, dt)``;
- the candidate driver must (transitively) call at least one pinned
  function, otherwise the stage is rejected as *uncoupled*.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

DRIVER = "simulation_step"


class AccretionError(ValueError):
    """Candidate stage violates the additive/coupled integration contract."""


@dataclass
class MergeReport:
    code: str
    pinned_functions: List[str]
    new_functions: List[str]
    reverted_overwrites: List[str] = field(default_factory=list)
    restored_missing: List[str] = field(default_factory=list)
    coupled_calls: List[str] = field(default_factory=list)


def layer_driver_name(stage: int) -> str:
    return f"layer_{stage}_step"


def code_hash(code: str) -> str:
    return hashlib.blake2b(code.encode(), digest_size=32).hexdigest()


def _functions(tree: ast.Module) -> Dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _module_constants(tree: ast.Module) -> Dict[str, ast.stmt]:
    out: Dict[str, ast.stmt] = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = n
    return out


def _imports(tree: ast.Module) -> List[ast.stmt]:
    return [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]


def _calls(fn: ast.FunctionDef) -> Set[str]:
    return {c.func.id for c in ast.walk(fn)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}


def _norm(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=False, include_attributes=False)


class _Rename(ast.NodeTransformer):
    def __init__(self, old: str, new: str):
        self.old, self.new = old, new

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        if node.name == self.old:
            node.name = self.new
        return node

    def visit_Name(self, node: ast.Name):
        if node.id == self.old:
            node.id = self.new
        return node


def pin_layer_code(candidate_layer_code: str, stage: int) -> str:
    """Pin the stage driver as ``layer_<stage>_step``.

    Candidates already structured as ``layer_<stage>_step`` + a thin
    ``simulation_step`` delegate are pinned by dropping the delegate; renaming
    the delegate onto the existing layer name would create a recursive stub.
    """
    tree = ast.parse(candidate_layer_code)
    target = layer_driver_name(stage)
    has_target = any(isinstance(n, ast.FunctionDef) and n.name == target
                     for n in tree.body)
    has_driver = any(isinstance(n, ast.FunctionDef) and n.name == DRIVER
                     for n in tree.body)
    if has_target and has_driver:
        tree.body = [n for n in tree.body
                     if not (isinstance(n, ast.FunctionDef) and n.name == DRIVER)]
    else:
        tree = _Rename(DRIVER, target).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def unified_from_layers(pinned_code: str, latest_stage: int) -> str:
    """Unified module whose driver delegates to the latest pinned layer."""
    driver = (f"\n\ndef {DRIVER}(buf, dt):\n"
              f"    {layer_driver_name(latest_stage)}(buf, dt)\n")
    return pinned_code.rstrip() + driver


def strip_driver(code: str) -> str:
    """Remove ``simulation_step`` from a module (keeps pinned layer code)."""
    tree = ast.parse(code)
    tree.body = [n for n in tree.body
                 if not (isinstance(n, ast.FunctionDef) and n.name == DRIVER)]
    return ast.unparse(tree) + "\n" if tree.body else ""


def merge_accretive_module(pinned_code: str, candidate_code: str,
                           require_coupling: bool = True,
                           allow_helper_overrides: bool = False) -> MergeReport:
    """Merge a stage-k candidate into the pinned (verified) module.

    ``pinned_code`` contains only pinned layers (no ``simulation_step``).
    Returns the unified module source plus a report of every repair made.
    """
    try:
        cand_tree = ast.parse(candidate_code)
    except SyntaxError as exc:
        raise AccretionError(f"candidate is not valid Python: {exc}") from exc
    pinned_tree = ast.parse(pinned_code) if pinned_code.strip() else ast.Module(body=[], type_ignores=[])
    pinned_fns = _functions(pinned_tree)
    pinned_consts = _module_constants(pinned_tree)
    cand_fns = _functions(cand_tree)
    cand_consts = _module_constants(cand_tree)
    if DRIVER not in cand_fns:
        raise AccretionError(f"candidate must define {DRIVER}(buf, dt)")

    reverted, restored, new_fns = [], [], []
    overridden_helpers: Set[str] = set()
    for name, fn in cand_fns.items():
        if name in pinned_fns:
            if _norm(fn) != _norm(pinned_fns[name]):
                is_layer_driver = name.startswith("layer_") and name.endswith("_step")
                if allow_helper_overrides and not is_layer_driver:
                    overridden_helpers.add(name)
                    new_fns.append(name)
                else:
                    reverted.append(name)
        elif name != DRIVER:
            new_fns.append(name)
    for name in pinned_fns:
        if name not in cand_fns:
            restored.append(name)

    for name, node in cand_consts.items():
        if name in pinned_consts and _norm(node) != _norm(pinned_consts[name]):
            reverted.append(f"constant {name}")

    intrinsic_prefixes = ("_invar_", "safe_")
    intrinsic_names = {"div_eps", "smooth_clamp"}

    def _is_intrinsic(fn_name: str) -> bool:
        return fn_name.startswith(intrinsic_prefixes) or fn_name in intrinsic_names

    body: List[ast.stmt] = []
    seen_imports: Set[str] = set()
    for imp in _imports(pinned_tree) + _imports(cand_tree):
        key = _norm(imp)
        if key not in seen_imports:
            seen_imports.add(key)
            body.append(imp)
    seen_intrinsics: Set[str] = set()
    for k, fn in pinned_fns.items():
        if k not in overridden_helpers and _is_intrinsic(k) and k not in seen_intrinsics:
            seen_intrinsics.add(k)
            body.append(fn)
    for n in new_fns:
        if _is_intrinsic(n) and n not in seen_intrinsics:
            seen_intrinsics.add(n)
            body.append(cand_fns[n])
    body.extend(pinned_consts.values())
    body.extend(n for k, n in cand_consts.items() if k not in pinned_consts)
    body.extend(fn for k, fn in pinned_fns.items() if k not in overridden_helpers and not _is_intrinsic(k))
    body.extend(cand_fns[n] for n in new_fns if not _is_intrinsic(n))
    body.append(cand_fns[DRIVER])

    coupled: List[str] = []
    if pinned_fns:
        reachable: Set[str] = set()
        frontier = [DRIVER]
        all_fns = {**{n: cand_fns[n] for n in new_fns}, DRIVER: cand_fns[DRIVER]}
        while frontier:
            cur = frontier.pop()
            if cur in reachable:
                continue
            reachable.add(cur)
            fn = all_fns.get(cur)
            if fn is None:
                continue
            for callee in _calls(fn):
                if callee in pinned_fns:
                    coupled.append(callee)
                elif callee in all_fns:
                    frontier.append(callee)
        coupled = sorted(set(coupled))
        if require_coupling and not coupled:
            raise AccretionError(
                "uncoupled stage: the new simulation_step does not call any verified "
                f"layer function ({', '.join(sorted(pinned_fns))}); extend and couple "
                "the verified module instead of replacing it")

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    return MergeReport(
        code=ast.unparse(module) + "\n",
        pinned_functions=sorted(pinned_fns),
        new_functions=new_fns,
        reverted_overwrites=reverted,
        restored_missing=restored,
        coupled_calls=coupled,
    )


def candidate_layer_code(unified_code: str, pinned_code: str) -> str:
    """Extract the stage delta (new functions/constants + driver) from a unified module."""
    uni = ast.parse(unified_code)
    pinned = ast.parse(pinned_code) if pinned_code.strip() else ast.Module(body=[], type_ignores=[])
    pinned_fns = set(_functions(pinned))
    pinned_consts = set(_module_constants(pinned))
    keep: List[ast.stmt] = []
    for n in uni.body:
        if isinstance(n, ast.FunctionDef) and n.name in pinned_fns:
            continue
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in pinned_consts:
            continue
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            continue
        keep.append(n)
    module = ast.Module(body=keep, type_ignores=[])
    return ast.unparse(module) + "\n"


def is_noop_layer_wrapper(code: str, stage: int) -> tuple[bool, str]:
    """Return ``(True, reason)`` if Stage ``k > 1`` driver is a bare forwarding wrapper.

    A synthesized ``layer_{stage}_step(buf, dt)`` (or ``simulation_step``) that
    only delegates to ``layer_{prev}_step(buf, dt)`` with no new assignments,
    loops, conditionals, or solver statements is rejected so ``Reflex_Heal``
    forces Stage ``stage`` to implement its contracted logic.
    """
    if int(stage or 1) <= 1 or not (code or "").strip():
        return (False, "")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return (False, "")

    target_name = layer_driver_name(int(stage))
    fn_node: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == target_name:
            fn_node = node
            break
    if fn_node is None:
        return (False, "")

    def _is_trivial_clamp_if(stmt: ast.stmt) -> bool:
        if not isinstance(stmt, ast.If) or stmt.orelse or len(stmt.body) != 1:
            return False
        body_stmt = stmt.body[0]
        if not isinstance(body_stmt, ast.Assign) or len(body_stmt.targets) != 1:
            return False
        tgt_s = ast.unparse(body_stmt.targets[0])
        if not (tgt_s.startswith("buf.data[") and isinstance(body_stmt.value, ast.Constant)):
            return False
        test_s = ast.unparse(stmt.test)
        return tgt_s in test_s

    meaningful: list[ast.stmt] = []
    for idx, stmt in enumerate(fn_node.body):
        if idx == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if isinstance(stmt, ast.Pass) or _is_trivial_clamp_if(stmt):
            continue
        if isinstance(stmt, ast.Return) and (
            stmt.value is None
            or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)
        ):
            continue
        meaningful.append(stmt)

    if not meaningful:
        return (
            True,
            f"NOOP_ACCRETION_LAYER: {target_name} has an empty body; implement Stage {stage}'s contracted solver/sweep logic rather than a no-op stub.",
        )

    def _forwarding_callee(stmt: ast.stmt) -> str | None:
        call: ast.Call | None = None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        if call is None or not isinstance(call.func, ast.Name):
            return None
        name = call.func.id
        if (
            name != target_name
            and (
                name == "simulation_step"
                or (name.startswith("layer_") and name.endswith("_step") and name[6:-5].isdigit())
            )
        ):
            return name
        return None

    callees = [_forwarding_callee(s) for s in meaningful]
    if all(c is not None for c in callees):
        callee = callees[0]
        return (
            True,
            f"NOOP_ACCRETION_LAYER: {target_name} is a bare forwarding wrapper calling {callee}(...) with zero new computational statements; implement Stage {stage}'s contracted numerical method or parameter sweep.",
        )
    # Also reject forwarding wrappers that only add redundant writes to slots already written
    # by the forwarded layer without introducing any loops, helper calls, branches, or new slots.
    first_callee = next((c for c in callees if c is not None), None)
    if first_callee is not None:
        callee_fn = next(
            (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == first_callee),
            None,
        )
        callee_slots = set()
        if callee_fn is not None:
            for sub in ast.walk(callee_fn):
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        ts = ast.unparse(tgt)
                        if ts.startswith("buf.data[") and ts.endswith("]"):
                            callee_slots.add(ts)
        non_fwd = [s for s, c in zip(meaningful, callees) if c is None]
        has_rich_control = any(
            isinstance(n, (ast.For, ast.While, ast.If, ast.FunctionDef, ast.Call))
            for st in non_fwd
            for n in ast.walk(st)
        ) or any(
            isinstance(n, ast.Name) and n.id == "dt"
            for st in non_fwd
            for n in ast.walk(st)
        )
        if not has_rich_control:
            new_slots = set()
            for st in non_fwd:
                if isinstance(st, ast.Assign):
                    for tgt in st.targets:
                        ts = ast.unparse(tgt)
                        if ts.startswith("buf.data[") and ts.endswith("]"):
                            new_slots.add(ts)
            if new_slots and new_slots <= callee_slots:
                return (
                    True,
                    f"NOOP_ACCRETION_LAYER: {target_name} only forwards to {first_callee}(...) and redundantly rewrites existing slots {sorted(new_slots)}; implement genuinely new Stage {stage} computations.",
                )
    return (False, "")


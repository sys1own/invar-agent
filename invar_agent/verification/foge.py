"""First-Order Guarded Execution (FOGE): pre-invocation predicate gating."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class FOGEResult:
    allowed: bool
    predicate: str
    detail: str = ""


@dataclass
class PreConditionResult:
    predicate: str
    passed: bool
    detail: str = ""


class FirstOrderGuardedExecutor:
    """Evaluates declarative first-order pre-conditions over a context mapping."""

    def __init__(self, workspace_root: Optional[str] = None) -> None:
        self.workspace_root = workspace_root or os.getcwd()
        self._custom: dict[str, Callable[[list[str], dict[str, Any]], bool]] = {}

    def register_predicate(self, name: str, fn: Callable[[list[str], dict[str, Any]], bool]) -> None:
        self._custom[name] = fn

    # -- predicate parsing -------------------------------------------------------

    @staticmethod
    def _split_args(inner: str) -> list[str]:
        args: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in inner:
            if ch in "([{" :
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        tail = "".join(current).strip()
        if tail:
            args.append(tail)
        return args

    @staticmethod
    def _unquote(arg: str) -> str:
        arg = arg.strip()
        if len(arg) >= 2 and arg[0] in "'\"" and arg[-1] == arg[0]:
            return arg[1:-1]
        return arg

    def evaluate_predicate(self, predicate: str, context: Optional[dict[str, Any]] = None) -> PreConditionResult:
        ctx = context or {}
        predicate = predicate.strip()
        negated = predicate.startswith("not ")
        body = predicate[4:].strip() if negated else predicate
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", body, re.DOTALL)
        if m:
            name, argstr = m.group(1), m.group(2)
            args = [self._unquote(a) for a in self._split_args(argstr)] if argstr.strip() else []
            passed = self._eval_call(name, args, ctx)
            detail = f"{name}({', '.join(args)})"
        elif body in ("true", "always_true"):
            passed, detail = True, "true"
        elif body in ("false", "always_false"):
            passed, detail = False, body
        else:
            # Bare context key lookup.
            passed, detail = bool(ctx.get(body, False)), f"ctx[{body!r}]"
        if negated:
            passed = not passed
        return PreConditionResult(predicate=predicate, passed=passed, detail=detail)

    def _eval_call(self, name: str, args: list[str], ctx: dict[str, Any]) -> bool:
        if name in self._custom:
            return bool(self._custom[name](args, ctx))
        root = self.workspace_root
        if name == "file_exists":
            path = self._resolve(args[0], ctx)
            return os.path.exists(path)
        if name == "path_exists":
            return os.path.exists(self._resolve(args[0], ctx))
        if name == "non_empty":
            key = self._unquote(args[0])
            val = ctx.get(key, "")
            if isinstance(val, (list, dict, str)):
                return len(val) > 0
            return val is not None
        if name == "symbol_defined":
            key = self._unquote(args[0])
            return key in ctx
        if name == "hash_matches":
            path = self._resolve(args[0], ctx)
            expected = self._unquote(args[1]) if len(args) > 1 else ""
            try:
                import blake3

                with open(path, "rb") as fh:
                    actual = blake3.blake3(fh.read()).hexdigest()
                return actual == expected.lower()
            except OSError:
                return False
        if name == "boundary_unchanged":
            return bool(ctx.get("holographic_parity_valid", False))
        if name == "and":
            return all(self.evaluate_predicate(a, ctx).passed for a in args)
        if name == "or":
            return any(self.evaluate_predicate(a, ctx).passed for a in args)
        # Unknown predicate: fail closed.
        return False

    def _resolve(self, path: str, ctx: dict[str, Any]) -> str:
        path = self._unquote(path)
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_root, path)

    def gate_tool_invocation(
        self,
        action_name: str,
        preconditions: list[str],
        context: Optional[dict[str, Any]] = None,
    ) -> FOGEResult:
        """All pre-conditions must hold before a tool/mutation may be invoked."""
        failures: list[str] = []
        for pre in preconditions:
            res = self.evaluate_predicate(pre, context)
            if not res.passed:
                failures.append(f"{pre} ({res.detail})")
        if failures:
            return FOGEResult(
                allowed=False,
                predicate=" && ".join(preconditions),
                detail=f"Blocked {action_name}: " + "; ".join(failures),
            )
        return FOGEResult(allowed=True, predicate=" && ".join(preconditions), detail=f"{action_name} admitted")

    def run_guarded(self, action: Callable[[], Any], action_name: str, preconditions: list[str],
                    context: Optional[dict[str, Any]] = None) -> tuple[bool, Any]:
        gate = self.gate_tool_invocation(action_name, preconditions, context)
        if not gate.allowed:
            return False, gate.detail
        return True, action()

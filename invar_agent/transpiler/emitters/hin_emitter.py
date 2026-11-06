"""Lower UAST to a HIN arena JSON description for the native Tier-1 engine.

The arena uses the node/port schema consumed by ``HinEngine.build_from_json``:
each node has an id, a kind, and named ports; wires connect port pairs.
Numeric kernels are lowered to ``number``/``add``/``sub``/``mul``/``div``
nodes; anything else raises :class:`EmitterError` so the caller can fall
back to the C++ backend.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .base import EmitterError

_BINOP_KIND = {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod"}


class HinEmitter:
    def __init__(self) -> None:
        self.nodes: List[Dict[str, Any]] = []
        self._counter = 0

    def _emit(self, node: Optional[Dict[str, Any]]) -> str:
        """Emit an expression node; returns the node id holding the result."""
        if node is None:
            raise EmitterError("missing expression")
        t = node.get("type")
        if t == "literal":
            value = node.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise EmitterError(f"non-numeric literal {value!r}")
            nid = self._add("number", {"value": f"{float(value):.17g}"})
            return nid
        if t == "reference":
            # Bindings become input ports named after the variable.
            return self._add("input", {"name": str(node.get("name"))})
        if t == "binop":
            op = node.get("op")
            kind = _BINOP_KIND.get(op)
            if kind is None:
                raise EmitterError(f"HIN emitter does not support binop {op!r}")
            left = self._emit(node.get("left"))
            right = self._emit(node.get("right"))
            return self._add(kind, {"left": left, "right": right})
        if t == "unaryop":
            op = node.get("op")
            if op != "-":
                raise EmitterError(f"HIN emitter does not support unaryop {op!r}")
            operand = self._emit(node.get("operand"))
            zero = self._emit({"type": "literal", "value": 0})
            return self._add("sub", {"left": zero, "right": operand})
        if t == "call":
            fn = node.get("function") or {}
            name = fn.get("name") if fn.get("type") == "reference" else None
            args = node.get("arguments") or []
            math_fn = {"sqrt": "sqrt", "sin": "sin", "cos": "cos", "exp": "exp",
                       "abs": "abs", "log": "log"}
            if name and name.startswith("math."):
                base = name.split(".", 1)[1]
                if base in math_fn and len(args) == 1:
                    arg = self._emit(args[0])
                    return self._add(math_fn[base], {"arg": arg})
            raise EmitterError(f"HIN emitter does not support call {name!r}")
        if t == "compare":
            raise EmitterError("HIN emitter does not support compare")
        raise EmitterError(f"HIN emitter does not support expression {t!r}")

    def _add(self, kind: str, attrs: Dict[str, Any]) -> str:
        nid = f"n{self._counter}"
        self._counter += 1
        self.nodes.append({"id": nid, "kind": kind, "attrs": attrs})
        return nid

    def emit_module(self, uast: Dict[str, Any]) -> str:
        """Lower a module's top-level numeric bindings into arena JSON."""
        self.nodes = []
        self._counter = 0
        results: Dict[str, str] = {}
        for child in uast.get("children") or []:
            if not isinstance(child, dict):
                continue
            if child.get("type") == "binding" and isinstance(child.get("name"), str):
                try:
                    results[child["name"]] = self._emit(child.get("value"))
                except EmitterError:
                    continue
        arena = {
            "nodes": self.nodes,
            "outputs": {name: nid for name, nid in results.items()},
        }
        return json.dumps(arena)

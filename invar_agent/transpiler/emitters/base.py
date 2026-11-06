"""Shared UAST expression/statement rendering helpers for polyglot emitters."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class EmitterError(ValueError):
    pass


def render_literal(value: Any, lang: str) -> str:
    if value is None:
        return "nullptr" if lang == "cpp" else "None"
    if value is True:
        return "true" if lang == "cpp" else "true"
    if value is False:
        return "false" if lang == "cpp" else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NAN" if lang == "cpp" else "float('nan')"
        if math.isinf(value):
            return ("INFINITY" if value > 0 else "-INFINITY") if lang == "cpp" else repr(value)
        text = repr(value)
        if lang == "cpp" and "." not in text and "e" not in text and "E" not in text:
            text += ".0"
        return text
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"' if lang == "cpp" else repr(value)
    raise EmitterError(f"cannot render literal {value!r}")


def qual_name(ref_name: str, lang: str) -> str:
    """Map Python-ish dotted names to target-language call names."""
    if "." not in ref_name:
        return ref_name
    base, _, attr = ref_name.rpartition(".")
    if base == "math":
        mapping = {"sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan",
                   "exp": "exp", "log": "log", "log10": "log10", "ceil": "ceil",
                   "floor": "floor", "pow": "pow", "fabs": "fabs", "fmod": "fmod"}
        fn = mapping.get(attr, attr)
        return f"std::{fn}" if lang == "rust" else fn
    if base == "np" or base == "numpy":
        mapping = {"sqrt": "sqrt", "sin": "sin", "cos": "cos", "exp": "exp",
                   "abs": "abs", "array": "array"}
        fn = mapping.get(attr, attr)
        return f"np.{fn}" if lang == "cpp" else f"numpy.{fn}"
    if lang == "cpp":
        return f"{base}.{attr}"  # C++ has no dot calls; caller handles attribute nodes
    return ref_name


def node_name(node: Optional[Dict[str, Any]]) -> Optional[str]:
    if isinstance(node, dict) and node.get("type") == "reference":
        return node.get("name")
    return None

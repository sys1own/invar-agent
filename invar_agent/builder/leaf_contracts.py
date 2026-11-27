"""UPG-016 (BUG-024, 025, 026): Per-slot leaf contracts, degenerate-output rejection, and slot test emission."""
from __future__ import annotations

import ast
import json
import math
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from invar_agent.builder.uast_models import LLMExtensionSlotRequest, SlotExample, SlotOutputSpec

FUNCTIONAL_SLOT_KINDS = frozenset({"bespoke_data_adapter", "domain_visualization", "workflow_glue"})


def validate_leaf_slot_contract(slot: LLMExtensionSlotRequest) -> None:
    """UPG-016 step 2: Require >= 1 postcondition OR >= 2 examples with distinct expected outputs
    for functional slots (`narrative_documentation` exempt). Raises `ValueError('LEAF_CONTRACT_REQUIRED')`.
    """
    if slot.slot_kind not in FUNCTIONAL_SLOT_KINDS:
        return
    if slot.postconditions:
        return
    if len(slot.examples) >= 2:
        distinct = {json.dumps(ex.expected, sort_keys=True) for ex in slot.examples}
        if len(distinct) >= 2:
            return
    raise ValueError(
        f"LEAF_CONTRACT_REQUIRED: functional slot {slot.slot_id!r} ({slot.slot_kind}) "
        "requires >= 1 postcondition or >= 2 examples with distinct expected outputs"
    )


def _check_scalar_spec(spec: SlotOutputSpec, val: Any) -> None:
    if spec.scalar_type == "f64":
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(float(val)):
            raise ValueError(f"LEAF_OUTPUT_TYPE_MISMATCH: {spec.name} expected finite f64, got {val!r}")
        if spec.range is not None:
            lo, hi = spec.range
            if not (lo <= float(val) <= hi):
                raise ValueError(f"LEAF_OUTPUT_RANGE_VIOLATION: {spec.name}={val} outside [{lo}, {hi}]")
    elif spec.scalar_type == "i64":
        if not isinstance(val, int) or isinstance(val, bool):
            raise ValueError(f"LEAF_OUTPUT_TYPE_MISMATCH: {spec.name} expected i64, got {val!r}")
        if spec.range is not None:
            lo, hi = spec.range
            if not (lo <= val <= hi):
                raise ValueError(f"LEAF_OUTPUT_RANGE_VIOLATION: {spec.name}={val} outside [{lo}, {hi}]")
    elif spec.scalar_type == "bool":
        if not isinstance(val, bool):
            raise ValueError(f"LEAF_OUTPUT_TYPE_MISMATCH: {spec.name} expected bool, got {val!r}")
    elif spec.scalar_type == "str":
        if not isinstance(val, str):
            raise ValueError(f"LEAF_OUTPUT_TYPE_MISMATCH: {spec.name} expected str, got {val!r}")
    elif spec.scalar_type == "enum":
        if not isinstance(val, str) or (spec.enum is not None and val not in spec.enum):
            raise ValueError(f"LEAF_OUTPUT_ENUM_VIOLATION: {spec.name}={val!r} not in {spec.enum!r}")


def _eval_postcondition(expr: str, env: Mapping[str, Any]) -> bool:
    tree = ast.parse(expr, mode="eval")
    allowed = (
        ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call,
        ast.Constant, ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Lt, ast.LtE, ast.Gt,
        ast.GtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Tuple, ast.List,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"LEAF_POSTCONDITION_UNSAFE: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in {"abs", "len", "min", "max", "isfinite"}:
                raise ValueError(f"LEAF_POSTCONDITION_UNSAFE: call {ast.unparse(node.func)}")
    ns: Dict[str, Any] = {"__builtins__": {}, "abs": abs, "len": len, "min": min, "max": max, "isfinite": math.isfinite}
    ns.update(env)
    return bool(eval(compile(tree, "<postcondition>", "eval"), ns, {}))


def verify_leaf_outputs_against_contract(
    fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    slot: LLMExtensionSlotRequest,
    *,
    extra_probe_inputs: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """UPG-016 step 3: Validate types, enums, ranges, postconditions, examples, and
    reject constant/degenerate outputs across distinct inputs (`LEAF_DEGENERATE_OUTPUT`).
    """
    validate_leaf_slot_contract(slot)
    spec_map = {s.name: s for s in slot.output_spec}

    observed_outputs: List[Tuple[str, str]] = []  # (canonical_input, canonical_output)

    def _check_one(inp: Mapping[str, Any], expected: Mapping[str, Any] | None = None, tol: float = 1e-9) -> None:
        out = dict(fn(inp))
        for sname, spec in spec_map.items():
            if sname not in out:
                raise ValueError(f"LEAF_OUTPUT_MISSING: {sname!r}")
            _check_scalar_spec(spec, out[sname])
        if expected is not None:
            for k, exp_v in expected.items():
                if k not in out:
                    raise ValueError(f"LEAF_EXAMPLE_MISMATCH: missing key {k!r}")
                got_v = out[k]
                if isinstance(exp_v, (int, float)) and not isinstance(exp_v, bool):
                    if abs(float(got_v) - float(exp_v)) > tol:
                        raise ValueError(f"LEAF_EXAMPLE_MISMATCH: {k}={got_v} != {exp_v} (tol={tol})")
                elif got_v != exp_v:
                    raise ValueError(f"LEAF_EXAMPLE_MISMATCH: {k}={got_v!r} != {exp_v!r}")
        env = {**dict(inp), **out}
        for pc in slot.postconditions:
            if not _eval_postcondition(pc, env):
                raise ValueError(f"LEAF_POSTCONDITION_FAILED: {pc!r} failed on {env!r}")
        observed_outputs.append((json.dumps(dict(inp), sort_keys=True), json.dumps(out, sort_keys=True)))

    for ex in slot.examples:
        _check_one(ex.inputs, ex.expected, ex.tolerance)
    for pinp in extra_probe_inputs:
        _check_one(pinp, None)

    # Degenerate constant output detection
    distinct_inputs = {i for i, _ in observed_outputs}
    distinct_outputs = {o for _, o in observed_outputs}
    if (len(slot.examples) >= 2 or len(distinct_inputs) >= 3) and len(distinct_inputs) >= 2 and len(distinct_outputs) == 1:
        raise ValueError(
            f"LEAF_DEGENERATE_OUTPUT: slot {slot.slot_id!r} produced identical output across {len(distinct_inputs)} distinct inputs"
        )

    return {"passed": True, "checked_cases": len(observed_outputs)}


def emit_slot_pytest_source(slot: LLMExtensionSlotRequest, module_import: str, fn_name: str) -> str:
    """Emit `tests/test_slot_{slot_id}.py` running all contract examples in the exported package."""
    cases = [
        {"inputs": ex.inputs, "expected": ex.expected, "tolerance": ex.tolerance}
        for ex in slot.examples
    ]
    return (
        f'"""Auto-generated contract tests for {slot.slot_id}."""\n'
        "from __future__ import annotations\n"
        f"from {module_import} import {fn_name}\n\n"
        f"EXAMPLES = {cases!r}\n\n"
        f"def test_{slot.slot_id}_contract_examples():\n"
        "    for ex in EXAMPLES:\n"
        f"        got = {fn_name}(ex['inputs'])\n"
        "        for k, exp in ex['expected'].items():\n"
        "            assert k in got\n"
        "            if isinstance(exp, (int, float)) and not isinstance(exp, bool):\n"
        "                assert abs(float(got[k]) - float(exp)) <= ex['tolerance']\n"
        "            else:\n"
        "                assert got[k] == exp\n"
    )

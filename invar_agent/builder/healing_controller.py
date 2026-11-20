"""Zero-token repairs and strictly leaf-scoped, budgeted source proposals."""
from __future__ import annotations

import ast
import json
import math
import os
import re
from dataclasses import dataclass
from uuid import uuid4

from invar_agent.healing.deterministic_ast import sanitize_builder_module
from invar_agent.llm.errors import complete_for_builder, redact_secrets

from .circuit_breaker import BuildBreakerBudget
from .error_taxonomy import BuildFault
from .his_cache import interface_digest, port_signature
from .leaf_policy import LEAF_SYSTEM_PROMPT
from .uast_models import builder_backend, canonical_json


@dataclass
class HealingResult:
    source: str
    tokens_used: int = 0
    frozen_his_digest: str | None = None


class HealingController:
    def __init__(self, llm_client=None, *, diagnostics_dir=None):
        self.llm_client = llm_client
        self.diagnostics_dir = diagnostics_dir

    def tier0(self, source: str, module_kind="llm_extension_slot") -> HealingResult:
        try:
            result = json.loads(builder_backend().heal_builder_tier0(source, module_kind))
        except ValueError as exc:
            raise BuildFault("PY_SYNTAX_ERROR", str(exc)) from exc
        code = result["code"]
        sanitized = sanitize_builder_module(code)
        ast.parse(sanitized.code)
        return HealingResult(sanitized.code)

    def causal_slice(self, uast, module_id):
        return json.loads(builder_backend().extract_hin_causal_slice(uast.model_dump_json(), module_id, 3))

    def tier1(self, source, port, *, error_code="HIS_CONTRACT_DRIFT", details=None) -> HealingResult:
        details = details or {}
        if interface_digest(port_signature(port)) != port.his_digest:
            raise BuildFault("HIS_CONTRACT_DRIFT", "Frozen port itself changed")
        tree = ast.parse(source)
        function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == port.port_name), None)
        if function is None or function.args.vararg or function.args.kwarg or function.args.posonlyargs or function.args.kwonlyargs:
            raise BuildFault("HIS_CONTRACT_DRIFT", "No unambiguous structural adapter exists")
        expected = [field.name for field in port.inputs]
        current = [arg.arg for arg in function.args.args]
        if set(expected) != set(current):
            raise BuildFault("HIS_CONTRACT_DRIFT", "Structural adaptation cannot invent missing input semantics")
        original = "_invar_original_" + port.port_name
        if any(isinstance(node, ast.FunctionDef) and node.name == original for node in tree.body):
            raise BuildFault("HIS_CONTRACT_DRIFT", "Structural adapter already installed")
        function.name = original
        arguments = ", ".join(f"{name}={name}" for name in current)
        body = [f"def {port.port_name}({', '.join(expected)}):", f"    result = {original}({arguments})"]
        if len(port.outputs) == 1:
            name = port.outputs[0].name
            body += ["    if not isinstance(result, dict):", f"        result = {{{name!r}: getattr(result, {name!r}) if hasattr(result, {name!r}) else result}}"]
        else:
            names = [field.name for field in port.outputs]
            body += ["    if isinstance(result, (list, tuple)):", f"        result = dict(zip({names!r}, result))",
                     "    elif not isinstance(result, dict):", f"        result = {{name: getattr(result, name) for name in {names!r}}}"]
        if error_code == "HIN_SI_DIM_MISMATCH":
            if details.get("source_dimension") != details.get("target_dimension") or details.get("source_dimension") is None:
                raise BuildFault("HIN_SI_DIM_MISMATCH", "Incommensurable dimensions cannot be coerced")
            scale = float(details.get("unit_scale", 1))
            if not math.isfinite(scale) or scale <= 0:
                raise BuildFault("HIN_SI_DIM_MISMATCH", "Invalid SI coercion scale")
            for field in port.outputs:
                if field.scalar_type == "f64":
                    body.append(f"    result[{field.name!r}] *= {scale!r}")
        if error_code == "HIN_SEMANTIC_TAG_MISMATCH":
            raise BuildFault(error_code, "Semantic tags require an authorized scientific correction")
        if error_code == "BOUND_CLAMP_VIOLATION":
            for field in port.outputs:
                if field.bounds and field.scalar_type in {"f64", "i64"}:
                    lo, hi = field.bounds.lower, field.bounds.upper
                    if lo is not None:
                        lo = lo if field.bounds.lower_inclusive else math.nextafter(lo, math.inf)
                        body.append(f"    result[{field.name!r}] = max({lo!r}, result[{field.name!r}])")
                    if hi is not None:
                        hi = hi if field.bounds.upper_inclusive else math.nextafter(hi, -math.inf)
                        body.append(f"    result[{field.name!r}] = min({hi!r}, result[{field.name!r}])")
        body.append("    return result")
        code = ast.unparse(tree) + "\n\n" + "\n".join(body) + "\n"
        ast.parse(code)
        return HealingResult(code, frozen_his_digest=port.his_digest)

    def tier2(self, source, node, port, uast, diagnostic, budget: BuildBreakerBudget) -> HealingResult:
        if node.kind != "llm_extension_slot":
            raise BuildFault("TIER2_FORBIDDEN_FOR_DETERMINISTIC_MODULE", "Tier-2 is restricted to extension slots")
        if self.llm_client is None:
            raise BuildFault("LLM_AUTH_ERROR", "No leaf synthesis client is configured")
        payload = {"entry_point_signature": f"def {port.port_name}({', '.join(field.name for field in port.inputs)}):",
            "required_output_keys": [field.name for field in port.outputs],
            "intent": node.intent_summary,
            "hin_causal_slice": self.causal_slice(uast, node.node_id),
            "frozen_interface": {"schema": port.model_dump(mode="json"), "frozen_his_digest": node.frozen_his_digest},
            "diagnostic": str(diagnostic)[-4000:], "source": source[:16000]}
        prompt = canonical_json(payload)
        remaining = budget.max_global_repair_tokens - budget.used_repair_tokens
        estimate = len(prompt.encode()) + 512
        if remaining <= estimate:
            raise BuildFault("LLM_QUOTA_EXHAUSTED", "Repair token reservation would exceed global budget")
        response = complete_for_builder(self.llm_client, prompt, system=LEAF_SYSTEM_PROMPT,
            max_tokens=min(4096, remaining - estimate), temperature=0.0)
        tokens = max(0, response.input_tokens) + max(0, response.output_tokens)
        budget.used_repair_tokens += tokens
        if self.diagnostics_dir is not None:
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self.diagnostics_dir / (uuid4().hex + ".json")
            payload = dict(module_id=node.node_id, input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                raw_response=redact_secrets(response.content, api_key=getattr(self.llm_client, "api_key", None)))
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
                json.dump(payload, stream, ensure_ascii=True, indent=2)
        if budget.used_repair_tokens > budget.max_global_repair_tokens:
            raise BuildFault("LLM_QUOTA_EXHAUSTED", "Provider exceeded reserved repair tokens")
        code = response.content.strip()
        fenced = re.fullmatch(r"```(?:python|py)?[ \t]*\r?\n([\s\S]*?)\r?\n```[ \t]*", code, re.IGNORECASE)
        if fenced:
            code = fenced.group(1).strip()
        return HealingResult(code + "\n", tokens_used=tokens)

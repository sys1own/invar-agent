"""Research Agent planner: LLM-assisted blueprint synthesis with pydantic validation."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional

from ..schemas.research import InvariantConstraint, ResearchBlueprint, VerificationContract

PLANNER_PROMPT = """You are the Research Agent planner for a mathematically verified autonomous agent.
Given a research question, produce ONE falsifiable research blueprint as strict JSON with keys:
blueprint_id (snake_case), title (5-120 chars), falsifiable_hypothesis,
search_queries (3-6 concrete search queries), contracts (list of one contract with:
contract_id, target_backend in [rust_hin, cpp, multi_variant], timeout_seconds,
invariants (list of {variable_name, assertion, tolerance}), max_memory_mb),
code_template (Python source defining `def simulation_step(buf, dt):` operating ONLY on
buf.data[i] element reads/writes, buf.size and buf.dt, with numeric literals and the operators
+ - * / ** and comparisons; loops must be `for i in range(...)`. FORBIDDEN: rebinding buf.data,
list/dict/set literals or comprehensions, function calls other than len/range/abs/min/max,
imports, attribute access other than buf.data/buf.size/buf.dt, and while loops),
max_feedback_rounds (1-5).
Invariants must reference telemetry keys the sandbox actually emits, e.g.
telemetry['final_max'], telemetry['final_min'], telemetry['steps'], telemetry['execution_time_ms'].
No markdown fences. Output JSON only.

Research question: {question}
"""

_FALLBACK_BLUEPRINT = {
    "blueprint_id": "kdv_soliton_scaling",
    "title": "KdV soliton amplitude-velocity scaling c = 2A",
    "falsifiable_hypothesis": "For the Korteweg-de Vries equation, a soliton of amplitude A propagates with velocity c = 2A.",
    "search_queries": [
        "Korteweg-de Vries soliton amplitude velocity relation c=2A",
        "KdV equation numerical simulation soliton",
        "Soliton amplitude velocity scaling law verification",
    ],
    "contracts": [{
        "contract_id": "kdv_contract",
        "target_backend": "multi_variant",
        "timeout_seconds": 30.0,
        "invariants": [
            {"variable_name": "energy", "assertion": "telemetry['final_max'] > 0.0", "tolerance": 1e-6},
            {"variable_name": "bounded", "assertion": "telemetry['final_max'] < 1e6", "tolerance": 0.0},
        ],
        "max_memory_mb": 1024,
    }],
    "code_template": (
        "def simulation_step(buf, dt):\n"
        "    n = buf.size\n"
        "    for i in range(1, n - 1):\n"
        "        u = buf.data[i]\n"
        "        buf.data[i] = u + dt * (-6.0 * u * (buf.data[i + 1] - buf.data[i - 1]))\n"
    ),
    "max_feedback_rounds": 3,
}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_blueprint(payload: Dict[str, Any]) -> ResearchBlueprint:
    return ResearchBlueprint.model_validate(payload)


async def generate_research_plan(question: str,
                                 llm_complete: Optional[Callable[[str], Any]] = None,
                                 fallback: bool = True) -> Dict[str, Any]:
    """Tier-2 REASONING plan synthesis; falls back to a deterministic blueprint."""
    if llm_complete is not None:
        try:
            # The prompt embeds a JSON example with literal braces, so
            # str.format() cannot be used; substitute the placeholder directly.
            raw = await llm_complete(PLANNER_PROMPT.replace("{question}", question))
            data = _extract_json(str(raw))
            if data is not None:
                bp = parse_blueprint(data)
                return {"blueprint": bp.model_dump(), "source": "llm"}
        except Exception:  # noqa: BLE001 - fall through to fallback
            pass
    if fallback:
        return {"blueprint": _FALLBACK_BLUEPRINT, "source": "fallback"}
    raise RuntimeError("research plan synthesis failed")

"""Research Agent planner: LLM-assisted blueprint synthesis with pydantic validation.

Prompts encode complete tool mastery: backend selection heuristics, the
strict ``v9_transpiler_friendly`` Python subset accepted by the
UAST -> C++/HIN transpilers, the sandbox observables available to
invariants, and cumulative (accretive) multi-stage integration rules.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from ..schemas.research import ResearchBlueprint

# -- shared tool-mastery context ----------------------------------------------

TOOLS_CONTEXT = """## Your execution tools (know them exactly)
- Sandbox_Execute is AIR-GAPPED: no network, no disk, no imports. Your Python is
  transpiled to C++20 (-O3 -std=c++20 -fPIC -shared), compiled, and executed in an
  isolated subprocess with CPU/memory limits. Nothing else runs your code.
- Backends (contract.target_backend):
  * "cpp": C-ABI kernel. Use for heavy array loops, ODE/PDE time stepping (RK4,
    leapfrog/velocity-Verlet, finite differences), matrix ops. Default choice.
  * "rust_hin": additionally reduces the module UAST in the native Rust Lafont
    HIN arena (structural normalisation check, zero-heap). Use for scalar/vector
    recurrences, discrete maps, algebraic invariant checks.
  * "multi_variant": compares competing algorithms. Define the reference
    `simulation_step(buf, dt)` plus variants `simulation_step_<name>(buf, dt)`.
    Every variant runs on the same initial_state/dt/steps; telemetry adds
    variants['<name>']['residual'] (L2 distance from the reference final state)
    and variants['<name>']['ms']. Use for Pareto comparisons of approximations.
- Z3 SMT checks that each contract's invariants are mutually satisfiable BEFORE
  execution; contradictory invariants are refused. Invariants are then checked
  concolically against the real numeric outputs.

## v9_transpiler_friendly Python (STRICT)
ALLOWED: top-level pure functions; `def simulation_step(buf, dt):` driver that
mutates buf.data[i] in place (buf.data is a flat list[float] of length buf.size,
buf.dt is the step); scalar params/locals of type int/float/bool; local lists via
`[0.0] * n` or `[a, b, c]`; 2-D via `[[0.0] * n for _ in range(m)]` only;
`for i in range(stop)` / `range(start, stop, step)`; `while cond:` (an iteration
guard is injected automatically); if/elif/else; ternary `a if c else b`;
simultaneous assignment `x, v = x + dt * v, v - dt * x`; augmented assignment;
`import math` and math.sqrt/sin/cos/tan/exp/log/pow/fabs/atan2/hypot/floor/...,
math.pi/math.e; abs/min/max/int/float/len(buf.data)/len(local_list); intrinsics
prng_uniform(seed, k) -> [0,1) (deterministic counter PRNG), clamp(x, lo, hi), sign(x).
Helper functions take scalars (and optionally `buf` as FIRST parameter) and return
a float/int or nothing. Module-level scalar constants are allowed.
FORBIDDEN (rejected by PreWrite_Guard / transpiler): classes/dataclasses,
try/except, with, yield, async/await, lambda, nested functions/closures,
global/nonlocal, tuples other than simultaneous assignment, dict/set, list
comprehensions (except the 2-D idiom), strings, print, import of anything except
math (no os/sys/socket/random/re/json/numpy), returning lists, passing lists to
helpers, rebinding buf.data, attribute access other than buf.data/buf.size/buf.dt,
`while True` without break.

## Sandbox observables usable in invariant assertions (Python expressions)
state[i] (final buffer), initial[i] (initial buffer), trace_min[i] / trace_max[i]
(min/max of slot i over the whole trajectory), len(state),
telemetry['final_min'|'final_max'|'final_mean'|'final_l2'|'initial_l2'|
'max_abs_change'|'finite'|'steps'|'execution_time_ms'],
variants['<name>']['residual'|'ms'] (multi_variant only), plus abs/min/max/sum/
all/any/sqrt/exp/log. Store derived diagnostics (energy, drift, amplitude,
residual) in dedicated buffer slots inside simulation_step so invariants can
read them, e.g. slot 2 = current energy, slot 3 = initial energy (set once when
buf.data[3] == 0.0), slot 4 = max |E - E0| / E0 so far.
Invariants must be falsifiable and physically meaningful; never write vacuous
invariants (like x == x) and never loosen tolerances to force a pass."""

PLANNER_PROMPT = """You are the Research Agent planner of a mathematically verified computational
research engine. Produce ONE falsifiable research blueprint for the request.

{tools}

## Output: strict JSON object with keys
blueprint_id (snake_case), title (5-120 chars), falsifiable_hypothesis,
search_queries (3-6 concrete literature queries for arXiv/OpenAlex),
target_research_stages (1-4; use 2-3 when the request describes a progression,
e.g. "start by ... then integrate ..."), stage_objectives (one short objective per
stage, stage 1 first), contracts (stage-1 contracts: list of {{contract_id,
target_backend, timeout_seconds, invariants: [{{variable_name, assertion,
tolerance}}], max_memory_mb, initial_state: [floats], dt, steps}}),
code_template (the STAGE-1 module only, v9_transpiler_friendly Python),
max_feedback_rounds (2-4).
Keep initial_state small (<= 16 slots) and steps <= 20000.
No markdown fences. Output JSON only.

## Literature context (may be empty)
{literature}

Research request: {question}
"""

ACCRETION_PROMPT = """You are extending a VERIFIED computational research module (cumulative
blueprint evolution, stage {stage} of {total}).

{tools}

## Rules for accretive integration (MANDATORY)
- The verified module below is pinned. Copy every verified function VERBATIM
  (same name, same body) or omit it (it is re-inserted automatically); NEVER
  modify or rename verified functions or constants.
- Add NEW helper functions for the new physics/algorithm and write a NEW
  `simulation_step(buf, dt)` driver that CALLS the verified functions (e.g. the
  verified force/energy helpers or `layer_{prev}_step(buf, dt)`) so the new logic
  executes integrated with the verified foundation. An uncoupled driver is rejected.
- Verified layers are re-verified with their original contracts inside the same
  compiled module, so do not change the meaning of shared buffer slots used by
  verified layers; use new slots (append beyond the existing ones) for new state.
- Add cross-coupling invariants that test the combined system (e.g. bounded
  limit cycle amplitude, phase-space bounds, reduction to the verified model when
  the new coupling is zero).

## Stage objective
{objective}

## Verified module (stages 1..{prev})
```python
{verified_code}
```

## Verified contracts (still enforced)
{verified_contracts}

## Telemetry of the last verified stage
{telemetry}

## Literature retrieved for this stage
{literature}

## Previous failed attempts for this stage (fix these)
{failures}

## Output: strict JSON object with keys
code (the full unified module: verified functions + new functions + new
simulation_step), contracts (NEW contracts for this stage, same schema as before,
initial_state long enough for all slots), rationale (one paragraph).
No markdown fences. Output JSON only.
"""

REFINE_PROMPT = """Your research kernel failed verification. Fix the CODE ONLY; the contracts
and invariants are fixed and must not be weakened.

{tools}

## Stage objective
{objective}

## Failing module
```python
{code}
```

## Contracts
{contracts}

## Counterexample / diagnostics
{failure}

Output strict JSON: {{"code": "<fixed full module>", "rationale": "..."}}. No fences.
"""

QUERY_PROMPT = """You steer literature search for a computational research loop.
Stage {stage} objective: {objective}
Hypothesis: {hypothesis}
Numerical telemetry from the previous stage: {telemetry}
Failures so far: {failures}
Write 3 precise literature search queries (arXiv/OpenAlex style, <= 12 words each)
that target the higher-order effects, couplings, closures or algorithmic fixes the
telemetry points to. Output a JSON list of strings only."""

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
        "initial_state": [1.0] * 16,
        "dt": 0.01,
        "steps": 10,
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


def _extract_json(text: str) -> Optional[Any]:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    candidates = []
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            candidates.append((start, text[start:end + 1]))
    for _, chunk in sorted(candidates):
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return None


def parse_blueprint(payload: Dict[str, Any]) -> ResearchBlueprint:
    return ResearchBlueprint.model_validate(payload)


def _fill(template: str, **values: Any) -> str:
    # Prompts embed literal JSON braces; substitute named placeholders only.
    out = template.replace("{{", "\x00").replace("}}", "\x01")
    for key, val in values.items():
        out = out.replace("{" + key + "}", str(val))
    return out.replace("\x00", "{").replace("\x01", "}")


def format_literature(items: List[Dict[str, Any]], limit: int = 8) -> str:
    if not items:
        return "(none retrieved)"
    lines = []
    for it in items[:limit]:
        year = f" ({it.get('year')})" if it.get("year") else ""
        summary = (it.get("summary") or "")[:400]
        lines.append(f"- [{it.get('source')}] {it.get('title')}{year}: {summary}")
    return "\n".join(lines)


async def generate_research_plan(question: str,
                                 llm_complete: Optional[Callable[[str], Any]] = None,
                                 fallback: bool = True,
                                 literature: Optional[List[Dict[str, Any]]] = None,
                                 ) -> Dict[str, Any]:
    """Tier-2 REASONING plan synthesis; falls back to a deterministic blueprint."""
    error: Optional[str] = None
    if llm_complete is not None:
        try:
            prompt = _fill(PLANNER_PROMPT, tools=TOOLS_CONTEXT,
                           literature=format_literature(literature or []),
                           question=question)
            raw = await llm_complete(prompt)
            data = _extract_json(str(raw))
            if isinstance(data, dict):
                bp = parse_blueprint(data)
                return {"blueprint": bp.model_dump(), "source": "llm"}
            error = "planner returned no JSON object"
        except Exception as exc:  # noqa: BLE001 - fall through to fallback
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
    if fallback:
        out: Dict[str, Any] = {"blueprint": _FALLBACK_BLUEPRINT, "source": "fallback"}
        if error:
            out["planner_error"] = error
        return out
    raise RuntimeError(f"research plan synthesis failed: {error}")


async def synthesize_accretive_stage(llm_complete: Callable[[str], Any], *,
                                     stage: int, total: int, objective: str,
                                     verified_code: str, verified_contracts: List[Dict[str, Any]],
                                     telemetry: Dict[str, Any],
                                     literature: List[Dict[str, Any]],
                                     failures: List[str]) -> Dict[str, Any]:
    """Tier-2: synthesize the stage-k unified module + new contracts."""
    prompt = _fill(
        ACCRETION_PROMPT, tools=TOOLS_CONTEXT, stage=stage, total=total, prev=stage - 1,
        objective=objective, verified_code=verified_code.strip(),
        verified_contracts=json.dumps(verified_contracts, indent=1)[:6000],
        telemetry=json.dumps(telemetry, default=str)[:3000],
        literature=format_literature(literature),
        failures="\n".join(f"- {f}" for f in failures[-3:]) or "(none)")
    raw = await llm_complete(prompt)
    data = _extract_json(str(raw))
    if not isinstance(data, dict) or not isinstance(data.get("code"), str):
        raise ValueError("accretion synthesis returned no {code, contracts} JSON")
    if not isinstance(data.get("contracts"), list) or not data["contracts"]:
        raise ValueError("accretion synthesis returned no contracts")
    return data


async def refine_stage_code(llm_complete: Callable[[str], Any], *, objective: str,
                            code: str, contracts: List[Dict[str, Any]],
                            failure: str) -> Optional[str]:
    """Tier-2: repair failing kernel code using the counterexample (contracts fixed)."""
    prompt = _fill(REFINE_PROMPT, tools=TOOLS_CONTEXT, objective=objective, code=code.strip(),
                   contracts=json.dumps(contracts, indent=1)[:6000], failure=failure[:3000])
    raw = await llm_complete(prompt)
    data = _extract_json(str(raw))
    if isinstance(data, dict) and isinstance(data.get("code"), str) and data["code"].strip():
        return data["code"]
    return None


def heuristic_guided_queries(objective: str, hypothesis: str,
                             telemetry: Dict[str, Any], failures: List[str]) -> List[str]:
    """Deterministic telemetry-guided query formulation (offline fallback)."""
    topic = re.sub(r"\s+", " ", objective or hypothesis).strip()[:90]
    queries = [topic]
    flat: Dict[str, float] = {}
    for k, v in (telemetry or {}).items():
        if isinstance(v, (int, float)):
            flat[k] = float(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    flat[f"{k}.{kk}"] = float(vv)
    joined = " ".join(failures).lower()
    if any("drift" in k for k in flat) or "drift" in joined or "energy" in joined:
        queries.append(f"{topic} symplectic integrator energy drift")
    if "diverged" in joined or "non-finite" in joined or flat.get("finite", 1.0) == 0.0:
        queries.append(f"{topic} numerical stability time step CFL")
    if "falsified" in joined:
        queries.append(f"{topic} counterexample regime breakdown")
    if any("residual" in k for k in flat):
        queries.append(f"{topic} convergence order error residual comparison")
    if len(queries) < 3:
        queries.append(f"{topic} governing equations parameters")
    seen, out = set(), []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:4]


async def guided_queries(llm_fast: Optional[Callable[[str], Any]], *, stage: int,
                         objective: str, hypothesis: str, telemetry: Dict[str, Any],
                         failures: List[str]) -> Dict[str, Any]:
    """Tier-1 FAST: telemetry-guided literature queries for the next stage."""
    if llm_fast is not None:
        try:
            raw = await llm_fast(_fill(
                QUERY_PROMPT, stage=stage, objective=objective, hypothesis=hypothesis,
                telemetry=json.dumps(telemetry, default=str)[:2000],
                failures="; ".join(failures[-3:]) or "none"))
            data = _extract_json(str(raw))
            if isinstance(data, list):
                qs = [str(q).strip()[:160] for q in data if str(q).strip()]
                if qs:
                    return {"queries": qs[:4], "source": "llm"}
        except Exception:  # noqa: BLE001 - deterministic fallback
            pass
    return {"queries": heuristic_guided_queries(objective, hypothesis, telemetry, failures),
            "source": "heuristic"}

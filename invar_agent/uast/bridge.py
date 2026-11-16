"""Bridge between Declarative ResearchComputeUAST and the 9-stage Research State Machine."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..schemas.research import (
    InvariantConstraint,
    ResearchBlueprint,
    VerificationContract,
)
from .compiler import (
    CompiledUASTResult,
    analyze_uast_with_hin,
    apply_perturbation_delta,
    compile_uast_to_python,
)
from .healer import (
    _loads_repaired_json,
    _strip_json_fences,
    build_counterexample_diagnostic_card,
    coerce_llm_json_to_uast,
    format_single_hole_repair_prompt,
    tier0_heal_uast,
)
from .schemas import (
    OperatorMutation,
    ResearchComputeUAST,
    UASTPerturbationDelta,
)

UAST_PLANNER_PROMPT = """You are a Computational Physics & Mathematical Modeling Agent.
You MUST NOT write any Python code, loops, or solver implementations.
Instead, output ONLY a declarative `ResearchComputeUAST` JSON object specifying:
1. `ontology.quantities`: every state variable and physical parameter/constant with:
   - `symbol` (valid identifier),
   - `si_dimension`: 7-element list `[Mass, Length, Time, Current, Temperature, Amount, Luminosity]`,
   - `interval`: `[min_val, max_val]` (use `-1e30` / `1e30` or `["-inf", "inf"]` if unbounded),
   - `is_state_variable`: `true` for ODE state variables, `false` for parameters/constants,
   - `initial_value`: initial value at t=0 (for state variables),
   - `default_value`: numeric SI value (for parameters/constants),
   - `unit`: SI unit string.
2. `governance.equations`: dict mapping each state variable symbol `x_i` to its RHS CSEL expression `dx_i/dt = f_i(x, p, t)` using ONLY declared symbols and CSEL operators (`+`, `-`, `*`, `/`, `^`, `safe_div`, `safe_sqrt`, `safe_log`, `safe_exp`, `safe_pow`, `safe_sin`, `safe_cos`, `safe_tanh`, `safe_acos`, `safe_asin`, `smooth_clamp`, `where`, `abs`, `min`, `max`, `log1p`, `expm1`).
3. `governance.observables`: dict mapping named output observables with units (e.g. `"total_energy_joules"`) to CSEL expressions over state variables and parameters.
4. `invariants`: `{"conserved_quantities": [...], "monotonic_increasing": [...], "custom_smt_assertions": [...]}`.
5. `dt` (float > 0), `steps` (int >= 1), `target_backend` (`"multi_variant"`).

Literature & Extracted SI Constants:
{literature}

Research Question:
{question}

Output ONLY valid JSON matching the `ResearchComputeUAST` schema (no markdown commentary, no Python code)."""

UAST_DELTA_PROMPT = """You are performing Cumulative Blueprint Evolution on a verified `ResearchComputeUAST`.
You MUST NOT rewrite the whole model or write any Python code.
Output ONLY a `UASTPerturbationDelta` JSON object with:
- `add_quantities`: new parameters/constants/state variables `{"symbol": {"symbol": "...", "si_dimension": [0,0,0,0,0,0,0], "default_value": 1.0, "initial_value": 0.0, "is_state_variable": false}}`
- `add_terms`: dict of `state_var -> CSEL_additive_term` to splice onto existing RHS equations
- `modify_terms`: list of `{"target_node_id": "state_var", "new_csel_expression": "..."}` for replacing a specific RHS
- `add_observables`: dict of `observable_name -> CSEL_expression`

Current Verified UAST:
{current_uast}

Stage Objective:
{objective}

Literature Context:
{literature}

Output ONLY valid JSON matching `UASTPerturbationDelta`."""


def _translate_invariant_to_slots(
    expr: str,
    slot_map: Dict[str, int],
) -> Optional[str]:
    """Map symbolic invariant references (e.g. `x >= -10`) to `state[idx]` for sandbox/Z3."""
    if not expr or not expr.strip():
        return None
    out = expr.strip()
    # Replace longest symbols first to avoid partial collisions
    for sym in sorted(slot_map.keys(), key=len, reverse=True):
        idx = slot_map[sym]
        out = re.sub(rf"\b{re.escape(sym)}\b", f"state[{idx}]", out)
    if "state[" not in out and "telemetry[" not in out:
        return None
    if not any(op in out for op in ("==", "!=", "<=", ">=", "<", ">")):
        return None
    return out


def uast_to_blueprint(
    compiled: CompiledUASTResult,
    *,
    blueprint_id: str = "uast_research_model",
    target_research_stages: int = 1,
    stage_objectives: Optional[List[str]] = None,
    search_queries: Optional[List[str]] = None,
) -> ResearchBlueprint:
    """Convert a CompiledUASTResult into a ResearchBlueprint for the 9-stage state machine."""
    uast = compiled.uast
    full_slot_map: Dict[str, int] = {}
    full_slot_map.update(compiled.state_slot_map)
    full_slot_map.update(compiled.observable_slot_map)

    invariants: List[InvariantConstraint] = [
        InvariantConstraint(
            variable_name="finite_state",
            assertion="telemetry['finite'] == 1.0",
            tolerance=1e-9,
        )
    ]
    for raw_assert in uast.invariants.custom_smt_assertions:
        mapped = _translate_invariant_to_slots(raw_assert, full_slot_map)
        if mapped:
            invariants.append(
                InvariantConstraint(
                    variable_name="custom_smt",
                    assertion=mapped,
                    tolerance=1e-6,
                )
            )
    for i, sv in enumerate(compiled.state_slot_map):
        invariants.append(
            InvariantConstraint(
                variable_name=f"{sv}_bounded",
                assertion=f"abs(state[{i}]) >= 0.0",
                tolerance=1e-9,
            )
        )

    contract = VerificationContract(
        contract_id="c1_uast_verified_kernel",
        target_backend=uast.target_backend,
        timeout_seconds=30.0,
        invariants=invariants,
        initial_state=compiled.initial_state,
        dt=uast.dt,
        steps=uast.steps,
    )

    variable_map = dict(compiled.observable_slot_map)
    if not variable_map:
        variable_map = dict(compiled.state_slot_map)

    return ResearchBlueprint(
        blueprint_id=blueprint_id,
        title=uast.title[:120] if len(uast.title) > 120 else uast.title,
        falsifiable_hypothesis=uast.hypothesis,
        search_queries=search_queries or [uast.title[:80]],
        contracts=[contract],
        code_template=compiled.python_code,
        target_research_stages=target_research_stages,
        stage_objectives=stage_objectives or [],
        variable_map=variable_map,
        extracted_constants={
            sym: q.default_value
            for sym, q in uast.ontology.quantities.items()
            if not q.is_state_variable
        },
        uast_spec=uast.model_dump(),
    )


async def verify_and_heal_uast_with_reprompt(
    uast: ResearchComputeUAST,
    *,
    llm_complete: Optional[Callable[[str], Any]] = None,
    extracted_constants: Optional[Dict[str, float]] = None,
    max_repairs: int = 2,
) -> Tuple[ResearchComputeUAST, CompiledUASTResult, List[str]]:
    """Run Tier-0 Healer -> Holographic HIN -> Single-Hole Re-Prompter -> Deterministic Compiler."""
    heal_logs: List[str] = []
    current, logs = tier0_heal_uast(uast, extracted_constants=extracted_constants)
    heal_logs.extend(logs)

    for _ in range(max_repairs):
        hin_report, adjoint_report, _ = analyze_uast_with_hin(current)
        if hin_report.get("all_valid", True):
            break
        fault_id = hin_report.get("fault_node_id")
        if not fault_id or llm_complete is None:
            # Tier-0 fallback: auto-inject remaining unknown symbols if no LLM available
            current, extra_logs = tier0_heal_uast(
                current,
                extracted_constants=extracted_constants,
                auto_inject_unknown=True,
            )
            heal_logs.extend(extra_logs)
            break

        card = build_counterexample_diagnostic_card(
            current,
            fault_id,
            hin_report=hin_report,
            adjoint_report=adjoint_report,
        )
        prompt = format_single_hole_repair_prompt(card)
        raw_reply = await llm_complete(prompt)
        try:
            patch_data = _loads_repaired_json(str(raw_reply))
            mut = OperatorMutation.model_validate(patch_data)
            current.governance.equations[mut.target_node_id] = mut.new_csel_expression
            current, extra_logs = tier0_heal_uast(current, extracted_constants=extracted_constants)
            heal_logs.extend(extra_logs)
            heal_logs.append(f"single_hole_reprompt_repaired:{mut.target_node_id}")
        except Exception as exc:  # noqa: BLE001
            heal_logs.append(f"single_hole_reprompt_parse_failed:{exc}")
            current, extra_logs = tier0_heal_uast(
                current,
                extracted_constants=extracted_constants,
                auto_inject_unknown=True,
            )
            heal_logs.extend(extra_logs)
            break

    compiled = compile_uast_to_python(current)
    return current, compiled, heal_logs


async def generate_uast_research_plan(
    question: str,
    *,
    llm_complete: Callable[[str], Any],
    literature: Optional[List[Dict[str, Any]]] = None,
    extracted_constants: Optional[Dict[str, float]] = None,
    target_research_stages: int = 1,
    stage_objectives: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a ResearchBlueprint via the Declarative ResearchComputeUAST pipeline."""
    lit_lines = []
    for item in (literature or [])[:6]:
        lit_lines.append(f"- {item.get('title', '')}: {str(item.get('summary', ''))[:300]}")
    if extracted_constants:
        lit_lines.append(f"Extracted SI constants: {json.dumps(extracted_constants)}")
    lit_text = "\n".join(lit_lines) if lit_lines else "(none)"

    prompt = UAST_PLANNER_PROMPT.replace("{literature}", lit_text).replace("{question}", question)
    raw = await llm_complete(prompt)
    uast = coerce_llm_json_to_uast(str(raw), default_title=question[:80])
    healed_uast, compiled, heal_logs = await verify_and_heal_uast_with_reprompt(
        uast,
        llm_complete=llm_complete,
        extracted_constants=extracted_constants,
    )
    bp = uast_to_blueprint(
        compiled,
        target_research_stages=target_research_stages,
        stage_objectives=stage_objectives,
    )
    return {
        "blueprint": bp.model_dump(),
        "uast": healed_uast.model_dump(),
        "compiled": compiled,
        "heal_logs": heal_logs,
        "source": "uast_llm",
    }


async def evolve_uast_stage(
    base_uast_dict: Dict[str, Any],
    *,
    objective: str,
    llm_complete: Callable[[str], Any],
    literature: Optional[List[Dict[str, Any]]] = None,
    extracted_constants: Optional[Dict[str, float]] = None,
) -> Tuple[ResearchComputeUAST, CompiledUASTResult, List[str]]:
    """Cumulative Blueprint Evolution via UASTPerturbationDelta splicing."""
    base_uast = ResearchComputeUAST.model_validate(base_uast_dict)
    lit_text = "\n".join(
        f"- {it.get('title', '')}: {str(it.get('summary', ''))[:250]}"
        for it in (literature or [])[:5]
    ) or "(none)"
    prompt = (
        UAST_DELTA_PROMPT
        .replace("{current_uast}", json.dumps(base_uast.model_dump(), indent=2))
        .replace("{objective}", objective)
        .replace("{literature}", lit_text)
    )
    raw = await llm_complete(prompt)
    delta_data = _loads_repaired_json(str(raw))
    delta = UASTPerturbationDelta.model_validate(delta_data)
    spliced = apply_perturbation_delta(base_uast, delta)
    return await verify_and_heal_uast_with_reprompt(
        spliced,
        llm_complete=llm_complete,
        extracted_constants=extracted_constants,
    )

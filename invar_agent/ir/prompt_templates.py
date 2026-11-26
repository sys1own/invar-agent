"""Append-only prefix-caching prompt templates (`Block A || Block B`) for SciIR synthesis.

Every prompt is structured for maximum KV-cache prefix reuse:
- Block A (static system prefix): canonical rules, Common JSON Schema Subset,
  rationale-before-declaration ordering, and explicit `"json"` keyword + exemplar
  (required by DeepSeek `response_format={"type": "json_object"}`).
- Block B (dynamic suffix): user prompt + grounded literature/registry context +
  frozen symbol table (for Pass 2) or targeted single-symbol diagnostic (for Pass 3).
"""
from __future__ import annotations

import json
from typing import Any

from invar_agent.ir.schema import FrozenSymbolTable

PASS1_SYSTEM_BLOCK_A = """You are the Invar-Agent vNext Pass-1 Scientific Symbol Table Compiler.
Your sole responsibility is to translate a user's natural-language or formal scientific prompt (plus any grounded literature/registry notes) into a single valid json object representing a `FrozenSymbolTable`.

CRITICAL RULES:
1. Output ONLY a single valid json object with no prose outside the JSON.
2. Put `"rationale"` FIRST (rationale-before-declaration ordering).
3. Partition EVERY mathematical symbol into its exact `SlotRole`:
   - `STATE`: time-evolving ODE/PDE/Hamiltonian/equilibrium state variables (must have finite `initial_or_default`).
   - `CONTROL_PARAM`: independent parameters swept or varied across operating domain (must have finite `initial_or_default` and `domain: {"lo": ..., "hi": ...}`).
   - `CONSTANT`: immutable physical/mathematical constants (must have finite `initial_or_default`).
   - `DERIVED_OBSERVABLE`: post-step diagnostic quantities computed from states/parameters (NEVER referenced in dynamics_or_residuals or let_bindings!).
   - `INDEPENDENT_VAR`: independent variable such as time `t`.
4. Every missing equation constant, domain bound, or initial condition inferred from scientific literature or physical defaults MUST be transparently logged in `assumptions_ledger`.
5. Use exact `archetype` from: `ODE_PDE`, `MONTE_CARLO`, `FIXED_POINT`, `PARAM_SWEEP`, `ALGEBRAIC_CLOSURE`.
6. Use exact `domain` from: `PHYSICS_THERMO`, `BIOCHEM_SYSTEMS`, `CLIMATE_EARTH`, `ECOLOGY_EPIDEM`, `FLUID_CONTINUUM`, `QUANTUM_STATMECH`, `ENGINEERING_CTRL`, `ECON_COMPLEX`.
7. If conserved stoichiometric pools exist (e.g. `A + B = const`), declare them in `conservation_pools`.

JSON EXEMPLAR:
{
  "rationale": "First-order reversible chemical equilibrium A <-> B conserving A+B=1.0 mol/L.",
  "ir_id": "chem_eq_01",
  "archetype": "ODE_PDE",
  "domain": "BIOCHEM_SYSTEMS",
  "symbols": [
    {"name": "A", "display_symbol": "A", "role": "STATE", "slot_index": 0, "dimension": {"exponents": [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]], "unit_label": "mol/m^3"}, "initial_or_default": 1.0, "strictly_positive": true, "pool_id": "pool_AB", "description": "Species A"},
    {"name": "B", "display_symbol": "B", "role": "STATE", "slot_index": 1, "dimension": {"exponents": [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]], "unit_label": "mol/m^3"}, "initial_or_default": 0.0, "strictly_positive": true, "pool_id": "pool_AB", "description": "Species B"},
    {"name": "k_f", "display_symbol": "k_f", "role": "CONTROL_PARAM", "slot_index": 0, "dimension": {"exponents": [[0,1],[0,1],[-1,1],[0,1],[0,1],[0,1],[0,1]], "unit_label": "1/s"}, "initial_or_default": 0.5, "domain": {"lo": 0.1, "hi": 2.0}, "strictly_positive": true, "description": "Forward rate"},
    {"name": "k_r", "display_symbol": "k_r", "role": "CONSTANT", "slot_index": 0, "dimension": {"exponents": [[0,1],[0,1],[-1,1],[0,1],[0,1],[0,1],[0,1]], "unit_label": "1/s"}, "initial_or_default": 0.2, "strictly_positive": true, "description": "Reverse rate"},
    {"name": "ratio_BA", "display_symbol": "B/A", "role": "DERIVED_OBSERVABLE", "slot_index": 0, "dimension": {"exponents": [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]], "unit_label": "1"}, "initial_or_default": 0.0, "description": "Equilibrium ratio"}
  ],
  "conservation_pools": [
    {"pool_id": "pool_AB", "member_states": ["A", "B"], "stoichiometry": [1.0, 1.0], "expected_total": 1.0}
  ],
  "assumptions_ledger": [
    {"id": "A1", "statement": "Normalized initial concentration A(0)=1.0, B(0)=0.0", "source": "LLM_DOMAIN_DEFAULT", "falsifiable_impact": "Scales pool total linearly"}
  ]
}"""

PASS2_SYSTEM_BLOCK_A = """You are the Invar-Agent vNext Pass-2 Scientific Model Body Compiler.
Given an IMMUTABLE `FrozenSymbolTable` (you may NEVER invent undeclared state/parameter/constant symbols), synthesize `ModelBodyDraft` as a single valid json object.

CRITICAL RULES:
1. Output ONLY a single valid json object. Put `"rationale"` FIRST.
2. Every symbol referenced in `let_bindings`, `dynamics_or_residuals`, `observables`, and `obligations` MUST exist in the `FrozenSymbolTable`, be a previously defined `let_binding` (strict acyclic DAG), or be a built-in (`pi`, `e_euler`, `exp`, `log`, `ln`, `sqrt`, `abs`, `sin`, `cos`, `tanh`, `step`, `min`, `max`, `clamp`, `if`, `logistic_sigmoid`, `hill_eq`, `michaelis_menten`, `rand_uniform01`, `rand_normal01`, `rand_exponential1`).
3. `Chk-ODE-RHS` INVARIANT: `DERIVED_OBSERVABLE` symbols MUST NEVER appear in `let_bindings` or `dynamics_or_residuals`! Use `let_bindings` for shared fluxes/intermediates instead.
4. Every `STATE` symbol MUST have an entry in `dynamics_or_residuals` with `"target": "<state_name>"`.
5. Every `DERIVED_OBSERVABLE` symbol MUST have an entry in `observables` with `"target": "<obs_name>"`.
6. Encode the user's scientific hypothesis FAITHFULLY in `obligations` WITHOUT weakening thresholds or shrinking operating domains. If the claim is physically impossible (e.g. exceeding Carnot efficiency), encode the EXACT claim so the deterministic engine refutes it honestly.
7. Set `config.kind` to match `archetype` using ONLY that variant's keys:
   - `ODE_PDE` -> `{"kind": "ode_pde", "t_span": [0.0, 50.0], "sample_points": 200, "rtol": 1e-7, "atol": 1e-10, "preferred_solver": "AUTO", "enable_buckingham_scaling": true}`
   - `MONTE_CARLO` -> `{"kind": "monte_carlo", "base_seed": 42, "replicates": 4096, "confidence_alpha": 0.05, "use_crn": true}`
   - `FIXED_POINT` -> `{"kind": "fixed_point", "max_newton_iters": 100, "residual_tol": 1e-10, "require_krawczyk_proof": true, "require_stability_cert": true}`
   - `PARAM_SWEEP` -> `{"kind": "param_sweep", "grid_strategy": "uniform", "points_per_axis": 25, "bisection_tol": 1e-4}`
   - `ALGEBRAIC_CLOSURE` -> `{"kind": "algebraic_closure", "shbt_federation_enabled": true, "open_boundary_vars": []}`
8. Every obligation MUST have a strictly positive finite `"tolerance"` (e.g. `1e-6`).

JSON EXEMPLAR:
{
  "rationale": "Define net forward flux J_net as a let-binding, then wire dA/dt = -J_net and dB/dt = J_net.",
  "let_bindings": [
    {"name": "J_net", "expr_text": "k_f * A - k_r * B"}
  ],
  "dynamics_or_residuals": [
    {"target": "A", "expr_text": "-J_net"},
    {"target": "B", "expr_text": "J_net"}
  ],
  "observables": [
    {"target": "ratio_BA", "expr_text": "B / max(A, 1e-12)"}
  ],
  "obligations": [
    {"id": "ob_eq_ratio", "quantifier": "FORALL", "lhs_text": "ratio_BA", "comparator": ">=", "rhs_text": "2.0", "tolerance": 1e-3, "epistemic_origin": "USER_PROMPT", "source_text": "B/A >= 2.0 at steady state"}
  ],
  "config": {
    "kind": "ode_pde",
    "t_span": [0.0, 50.0],
    "sample_points": 200,
    "rtol": 1e-7,
    "atol": 1e-10,
    "preferred_solver": "AUTO",
    "enable_buckingham_scaling": true
  }
}"""

PASS3_SYSTEM_BLOCK_A = """You are the Invar-Agent vNext Pass-3 Single-Symbol Micro-Repair Compiler.
A single equation or let-binding failed deterministic Rust scope/DAG/dimensional checking.
Output ONLY a single valid json object matching `SingleSymbolRepairPayload` fixing ONLY that single `expr_text` using ONLY symbols from the `FrozenSymbolTable` and earlier `let_bindings`.
NEVER reference a `DERIVED_OBSERVABLE` inside `dynamics_or_residuals` or `let_bindings`. NEVER alter obligations or thresholds.

JSON EXEMPLAR:
{
  "rationale": "Replaced undeclared alias `kf_rate` with declared symbol `k_f`.",
  "target_kind": "let_binding",
  "target_name": "J_net",
  "expr_text": "k_f * A - k_r * B"
}"""


def build_pass1_messages(
    user_prompt: str,
    grounded_context: str = "",
) -> list[dict[str, str]]:
    """Build append-only [Block A || Block B] messages for Pass 1."""
    block_b = f"USER RESEARCH PROMPT:\n{user_prompt.strip()}\n"
    if grounded_context.strip():
        block_b += f"\nGROUNDED LITERATURE & REGISTRY CONTEXT:\n{grounded_context.strip()}\n"
    block_b += "\nReturn ONLY the valid json object for `FrozenSymbolTable`."
    return [
        {"role": "system", "content": PASS1_SYSTEM_BLOCK_A},
        {"role": "user", "content": block_b},
    ]


def build_pass2_messages(
    user_prompt: str,
    frozen_table: FrozenSymbolTable,
    grounded_context: str = "",
) -> list[dict[str, str]]:
    """Build append-only [Block A || Block B] messages for Pass 2."""
    frozen_json = json.dumps(frozen_table.model_dump(mode="json"), indent=2)
    block_b = (
        f"USER RESEARCH PROMPT:\n{user_prompt.strip()}\n\n"
        f"FROZEN SYMBOL TABLE (IMMUTABLE):\n{frozen_json}\n"
    )
    if grounded_context.strip():
        block_b += f"\nGROUNDED CONTEXT:\n{grounded_context.strip()}\n"
    block_b += "\nReturn ONLY the valid json object for `ModelBodyDraft`."
    return [
        {"role": "system", "content": PASS2_SYSTEM_BLOCK_A},
        {"role": "user", "content": block_b},
    ]


def build_pass3_micro_repair_messages(
    frozen_table: FrozenSymbolTable,
    allowed_let_names: list[str],
    diagnostic: dict[str, Any],
    current_expr_text: str,
) -> list[dict[str, str]]:
    """Build append-only [Block A || Block B] messages for bounded Pass 3 micro-repair."""
    sym_summary = [
        {"name": s.name, "role": s.role.value, "unit": s.dimension.unit_label}
        for s in frozen_table.symbols
    ]
    block_b = (
        f"FROZEN SYMBOLS:\n{json.dumps(sym_summary)}\n\n"
        f"ALLOWED LET-BINDINGS IN SCOPE:\n{json.dumps(allowed_let_names)}\n\n"
        f"DIAGNOSTIC:\n{json.dumps(diagnostic)}\n\n"
        f"CURRENT FAILING EXPRESSION:\n{current_expr_text}\n\n"
        "Return ONLY the valid json object for `SingleSymbolRepairPayload`."
    )
    return [
        {"role": "system", "content": PASS3_SYSTEM_BLOCK_A},
        {"role": "user", "content": block_b},
    ]

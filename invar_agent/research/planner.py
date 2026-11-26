"""Research Agent planner: LLM-assisted blueprint synthesis with pydantic validation.

Prompts encode complete tool mastery: backend selection heuristics, the
strict ``v9_transpiler_friendly`` Python subset accepted by the
UAST -> C++/HIN transpilers, the sandbox observables available to
invariants, and cumulative (accretive) multi-stage integration rules.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..schemas.research import FeasibilityAssessment, FeasibilityVerdict, ResearchBlueprint
from .feasibility import truncate_at_sentence
from .slot_schema import observable_name
from .scientific_integrity import SCIENTIFIC_CONTRACT

# -- shared tool-mastery context ----------------------------------------------

TOOLS_CONTEXT = SCIENTIFIC_CONTRACT + "\n\n" + """## Your execution tools (know them exactly)
## Formal evidence at ingest (Four-Namespace Blueprint AST)
Translate ANY clear research intent (even 1-2 sentences of casual English with zero equations or units) into a complete, falsifiable, SI-normalized Blueprint AST grounded in retrieved literature and registries.
Log every added equation, SI constant, operating domain, initial condition, or null baseline in `assumptions_ledger`.
CRITICAL ANTI-GAMING RULE: If the user asks whether a specific claim or threshold holds (including potentially false claims like "Can a reversible heat engine between 300 K and 600 K reach 65% efficiency?"), preserve the user's EXACT claimed predicate faithfully in `obligations` so the deterministic engine can prove OR formally falsify it — NEVER silently weaken or invert a false claim!
Use four cleanly separated namespaces:
- `typed_slots`: mutable simulation buffer slots ONLY (DYNAMIC_STATE, CONTROL_PARAM, DERIVED_OBSERVABLE, CONSERVATION_AUX, TIME_STEP).
- `constants`: immutable typed SI physical/model constants (`R_gas`, `F_far`, `T_K`, `C_m`, pool totals, etc.). Constants are NEVER required to exist in `typed_slots`.
- `derived_expressions`: pure algebraic definitions (`delta_p`, `aec_ratio`, etc.) over `typed_slots` and `constants`. Derived expressions are NEVER required to exist in `typed_slots` unless promoted.
- `ihcg_evidence`: `{"evidence":{"slots":[...],"equations":{},"environmental_assumptions":[],"obligations":[{"id":"h","expression":"<formal claim>","obligation_class":"HYPOTHESIS_TARGET","prompt_quote":"<optional provenance quote>"}],"quantifier_mode":"FORALL","mechanism_parameters":[],"assumptions_ledger":[]},"domain":{},"archetype":"ODE_PDE"}`.
Keep physical symbols, variable_map indices, domains, code and contracts consistent. A claimed bound must apply to the computed physical observable, not an input containing the desired answer. Retain strictness and quantifiers exactly; an execution tolerance must not weaken a sealed inequality.
For a single bounded study use one stage. Later stages require explicit sealed IHCG extensions; do not manufacture stages just to rephrase a refutation.
Formal JSON shapes are strict:
- domain maps EVERY free slot symbol to a numeric [lower, upper] array. Fixed values use [v,v].
- Every equation has expression and si_dim (seven integer SI exponents). ALGEBRAIC is the default kind; ODE_RHS and FIXED_POINT_MAP/RESIDUAL require state_symbol bound to a DYNAMIC_STATE.
- archetype is ODE_PDE, MONTE_CARLO, FIXED_POINT or PARAM_SWEEP.
- environmental_assumptions and obligations[].expression are formal Boolean expressions, not English prose; quantification belongs in quantifier_mode (FORALL, EXISTS, BOUNDARY).
Use CONTROL_PARAM for held inputs and DERIVED_OBSERVABLE for pure readouts.
An evolution equation must include `kind:"ODE_RHS"` and `state_symbol` naming the evolving state; its SI dimensions are state dimensions minus time.
For equilibrium equations use kind FIXED_POINT_RESIDUAL (g(z)=0) or FIXED_POINT_MAP (z=F(z)), with state_symbol naming the corresponding state.
For bounded Monte Carlo samplers use kind MC_SAMPLE and state_symbol naming the population-mean observable. Mark immutable stochastic input slots with stochastic_role UNIFORM, SEED, COUNTER, or CHANNEL; uniform inputs require domain [0,1]. Bind invar_rand_u01(seed,counter,channel) to those exact inputs.
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
The blueprint variable_map must identify every initial-state slot bijectively by
physical symbol. Include slot_roles keyed by the same symbols: read_only_parameter,
evolving_state, or diagnostic. Time-varying ODE states MUST be `evolving_state`, and
derived instantaneous rates/fluxes/slopes/currents/extrema initialized to 0.0 MUST
be `diagnostic` — NEVER mark ODE states or computed fluxes/rates as `read_only_parameter`.
Never store diagnostics in input parameter slots.
ALLOWED: top-level pure functions; `def simulation_step(buf, dt):` driver that
mutates buf.data[i] in place (buf.data is a flat list[float] of length buf.size,
ONLY buf.data[i] and buf.size exist; dt is the function argument, never buf.dt,
buf.steps or other buffer attributes); scalar params/locals of type int/float/bool; local lists via
`[0.0] * n` or `[a, b, c]`; 2-D via `[[0.0] * n for _ in range(m)]` only;
`for i in range(stop)` / `range(start, stop, step)`; `while cond:` (an iteration
guard is injected automatically); if/elif/else; ternary `a if c else b`;
simultaneous assignment `x, v = x + dt * v, v - dt * x`; augmented assignment;
`import math` and math.sqrt/sin/cos/tan/exp/log/pow/fabs/atan2/hypot/floor/...,
math.pi/math.e; abs/min/max/int/float/len(buf.data)/len(local_list); intrinsics
prng_uniform(seed, k) -> [0,1) (deterministic counter PRNG),
invar_rand_u01(seed, step, channel) -> [1e-15, 1-1e-15],
invar_rand_normal(seed, step, channel) -> N(0,1),
invar_rand_poisson(seed, step, lam), invar_rand_binomial(seed, step, n, p),
welford_mean_update(mean, x, k), welford_m2_update(m2, mean_old, mean_new, x),
welford_cov_update(c_xy, mean_x_old, x, mean_y_new, y),
norm_cdf(x), norm_ppf(p), logistic_sigmoid(x), logit(p),
hill_eq(c, ec50, n), michaelis_menten(s, vmax, km),
kl_term(p, q), shannon_term(p), gammainc_reg(a, x), betainc_reg(a, b, x),
clamp(x, lo, hi), sign(x).
STRICT REAL-SCALAR RULE: NEVER use Python complex literals (`1j`), `complex()`, or `cmath`.
Represent all complex quantities (such as off-diagonal density matrix coherences $\\rho_{01}$)
strictly as separate real and imaginary `float` variables (`rho01_re`, `rho01_im`).
Helper functions take scalars (and optionally `buf` as FIRST parameter) and return
a float/int or nothing. Module-level scalar constants are allowed.
FORBIDDEN (rejected by PreWrite_Guard / transpiler): classes/dataclasses,
try/except, with, yield, async/await, lambda, nested functions/closures,
global/nonlocal, tuples other than simultaneous assignment, dict/set, list
comprehensions (except the 2-D idiom), strings, print, import of anything except
math (no os/sys/socket/random/re/json/numpy), complex literals/`complex()`/`cmath`,
returning lists, passing lists to
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
LIMIT-CYCLE / EXTREMA / PERIOD TRACKING: when tracking limit cycles, extrema, or
periods, use scalar state variables inside `buf.data` (e.g., `prev_val =
buf.data[0]`, `cycle_count = buf.data[10]`, `last_peak_t = buf.data[11]`). NEVER
use dynamic Python lists (`list.append`), `while` searches across history, or
variable-length arrays, as they cannot lower to C++ C-ABI.
VARIABLE STYLE: prefer plain scalar variables and indexable arrays over
nested Python tuples. Tuple unpacking (`a, b = b, a`), sequence unpacking
(`q, p, s = state`) and tuple literals are supported by the transpiler, but
simple scalars keep kernels readable and portable.
KERNEL HYGIENE (hard rules): (1) NO comment monologues — at most 5 short
comment lines across the entire file; never write `# Wait...` derivations
inside code comments. (2) NO `# placeholder` slots — every state[k]
(buf.data[k]) referenced in ANY contract invariant (roundtrip_error,
entropy_monotonicity, trace_error_max, min_eigenvalue, purity, ...) MUST be
explicitly computed and assigned to buf.data[k] inside simulation_step on
every step. (3) Error/drift metrics (roundtrip_error, trace_error,
entropy_drift, energy_drift, residuals) must be RECOMPUTED each step and
start at 0.0 — never left at an arbitrary initial value.
Invariants must be falsifiable and physically meaningful; never write vacuous
invariants (like x == x) and never loosen tolerances to force a pass.
CRITICAL INVARIANT RULE: Do NOT guess ungrounded arbitrary narrow intervals when no
scale or bound exists. ALWAYS faithfully encode: (a) explicit target windows,
bands, corridors, and ceilings stated by the user (`*_band`, `*_window`,
`*_corridor`, `*_bound`, `*_max`, `*_limit` MUST encode their finite lower AND
upper bounds — NEVER degrade a two-sided band or upper ceiling to `state[i] >= 0.0`),
(b) physical positivity, finiteness, and realistic physical plausibility envelopes
(e.g. keeping intensive potentials, temperatures, or concentrations within
physically plausible finite ceilings so numerical blowups are falsified and
repaired rather than passing silently), (c) conserved pool sums and monotone/extrema
relations (`peak_*`/`max_*`/`min_*` invariants MUST reference the matching extrema
slot in `variable_map`, never the instantaneous state slot), and (d) exact analytical
identities within a generous tolerance. When a parameter sweep across control ranges
is requested, instantiate distinct control-parameter values (`initial_values`) across
stages/contracts (or an explicit sweep kernel) rather than copying identical
`initial_state` vectors. Always emit `initial_values` keyed by canonical
`variable_map` names (with `time_s`/`t` starting at `0.0`) alongside `initial_state`.

## Domain observables & variable_map (MANDATORY for quantitative problems)
- The blueprint MUST include `variable_map`: {"<metric_name_with_units>": <buf.data index>, ...}
  mapping EVERY numerical quantity the user asked for (e.g.
  "magnetic_field_B_tesla", "lorentz_factor_gamma", "drag_force_newtons",
  "thermal_heat_flux_w_m2", "total_thermal_power_watts",
  "optimal_shield_thickness_m", "shield_mass_tons") to the buffer slot holding
  its FINAL computed value. Metric names must carry their units.
- Inside simulation_step, compute the requested physical quantities and store
  them in those slots (helpers may take `buf` as first parameter and return a
  float; they may also return nothing and mutate buf.data directly).
- initial_state MUST encode the actual physical state variables of the
  surrogate model (positions, momenta, fields, temperatures, ...). NEVER emit a
  lazy repeating vector such as [1, 1, 1, ...] with a dummy loop: the sandbox
  must compute real domain quantities from real initial conditions.
- For REFORMULATED_SURROGATE verdicts, implement exactly the surrogate
  equations named in the feasibility reformulation (0D/1D reductions, scaling
  laws, boundary conditions) and record the surrogate assumptions in the
  hypothesis and contract rationales.

## Observable accuracy rules (MANDATORY — unit and peak-tracking bugs falsify reports)
- RUNNING MAXIMUM for peaks: any observable named `max_*` or `peak_*` (e.g.
  `max_heat_flux_mw_m2`, `peak_g_gE`) MUST be updated inside simulation_step
  with a running-maximum guard — `if q > buf.data[idx]: buf.data[idx] = q` —
  NEVER by overwriting `buf.data[idx] = q` unconditionally every step (that
  replaces the true peak with the final ground-impact value).
- EXPLICIT SI PREFIX CONVERSIONS: the correlation you integrate produces SI
  base units (W/m², J/m², m/s, m/s², m). The observable NAME declares the
  display unit, so convert BEFORE writing the slot:
  * `_mw_m2` (MW/m²): divide W/m² by `1e6` (`q_w_m2 / 1e6`);
  * `_mj_m2` (MJ/m²): divide J/m² by `1e6`;
  * `_km` / `_km_s`: multiply m / (m/s) by `0.001` to km / (km/s);
  * `_g` / `_ge` / `_gE` (Earth g): divide m/s² by `9.80665`;
  * `_m_s`: already m/s, no conversion.
  Sanity bound: peak convective heat flux in MW/m² must stay below the
  free-stream kinetic-energy flux 0.5 * rho0 * v0^3 / 1e6 (≈ a few hundred
  MW/m² for planetary entry) — add an invariant enforcing it.
- VERBATIM USER PARAMETERS: preserve every numerical parameter from the user's
  prompt exactly. If the user says "5.75 km/s", initial_state velocity MUST be
  `5750.0` (m/s) — convert units exactly once, never round (5.75 km/s is NOT
  5800 m/s). Same for masses, altitudes, radii, densities and angles.
- MULTI-STAGE ACCRETION: when the request or expert prompt describes a staged
  model or multiple distinct verification regimes (e.g. baseline trace/positivity,
  stiff entropy monotonicity, and closed-limit chiasmus reversibility), set
  `target_research_stages: 3` (or `>= 2` matching the contract stage prefixes)
  with one `stage_objectives` entry per stage, and name each contract's stage
  clearly (prefix contract_ids with `c1_` / `c2_` / `c3_` or `stage1_` /
  `stage2_` / `stage3_`) so every stage builds on the pinned verified layer.
- PHYSICAL TIME-HORIZON CLOSURE (CRITICAL): `dt * steps` MUST match the
  requested simulation duration in the EXACT same time unit as the kernel's
  rate constants. If rate constants are in SI `1/s` (e.g. divided by `86400.0`
  or `3.15576e7`) and the prompt specifies 180 days, then `dt` MUST be in seconds
  so `dt * steps = 180 * 86400 = 15552000.0 s` (e.g. `dt = 1728.0, steps = 9000`,
  NEVER `dt = 0.02, steps = 9000`).
- TIME-UNIT & CFL STABILITY RULE (CRITICAL): EVERY rate constant (transfer,
  clearance, mortality, reaction, relaxation) in `code_template` MUST be
  expressed in the SAME time unit as `dt` (convert `1/day` rates by `/ 86400.0`,
  `1/min` by `/ 60.0` whenever `dt` is in seconds!) and satisfy `dt * k < 0.5`.
  For any stiff/fast linear compartment (`dt * k > 0.25`), ALWAYS use an
  unconditionally positive semi-implicit / Patankar update
  `C_new = (C + dt * inflow) / (1.0 + dt * outflow_rate)` instead of explicit
  Euler `C + dt * (-outflow_rate * C + inflow)` which explodes on large `dt`.
- BASELINE PLAUSIBILITY & THERMODYNAMIC FLUX SCALING (Stage 1 CRITICAL):
  Under baseline/control=0 conditions in Stage 1, include explicit physical
  plausibility envelopes on primary intensive state variables (e.g. potentials,
  temperatures, concentrations, energy ratios) so uncalibrated source rates
  causing 2x+ unphysical drift are caught in Stage 1 BEFORE `layer_1_step` is
  pinned. Any driving flux coupled to an intensive potential/gradient MUST
  include thermodynamic back-pressure / reversal saturation consistent with SI
  flux-to-conductance units so baseline stays in the physiological/physical
  envelope while moderate dissipation sweeps remain energetically viable. In
  capacitive/potential ODEs (`C * dX/dt = +J_source - J_sink - J_leak - J_control`),
  verify sign polarity (`+J_source` charges/increases `X`, `-J_sink - J_leak`
  discharges/lowers `X`) so `X` stabilizes inside the physiological envelope
  rather than collapsing to a lower clamp. Always store electrical potentials
  in SI Volts (`V`, matching `structured_evidence` `si_value`, e.g. `0.165` V
  for `165 mV`, corridor `[0.115, 0.155]` V) rather than `mV`, recompute
  derived energy ratios (e.g. `aec = (atp + 0.5 * adp) / pool_total`) AFTER any
  simplex/pool conservation projection on every step, and allocate distinct
  slots for instantaneous vs. running-maximum (`peak_x = max(peak_x, x)`) metrics.
- NO MULTI-TUPLE RHS/STATE HELPERS & MANDATORY CLOCK/CONTROL SLOTS (CRITICAL):
  NEVER define helper functions returning 3+ element tuples (such as `def rhs(...)`
  returning an 8-tuple of derivatives or `def write_state(...)` returning a 7-tuple),
  because tuple scalarization clones the entire helper function once per tuple element.
  Instead, compute all state updates directly inside `simulation_step(buf, dt)` (or
  `layer_k_step(buf, dt)`). Always assign dedicated `variable_map` slots for `"time_s"`
  and every swept control parameter, read them from `buf.data[slot]` at the top of
  `simulation_step` (NEVER hardcode `t = 0.0` or fixed control constants inside
  `simulation_step`), and advance `buf.data[time_slot] = t + dt` on every step.
- CORRIDOR / EXISTENTIAL REGIME DISCOVERY SWEEPS (Stage 3 CRITICAL):
  When a prompt asks to identify whether a non-empty parameter corridor
  `[p_lo, p_hi]` simultaneously satisfies multiple target bounds (e.g. low
  hazard, preserved energy ratio, and an interior intensive window), NEVER place
  every narrow interior target constraint as a hard PASS invariant on ALL
  extreme low/mid/high endpoint probes simultaneously (excessive dissipation
  at high control MUST push intensive state below the interior band in any
  physically coupled model). Instead:
  (a) extreme endpoint probes enforce safety, finiteness, conservation,
      positivity, and regime sensitivity (e.g. peak hazard or excessive collapse);
  (b) Stage `k` corridor discovery either implements a deterministic multi-point
      grid search inside `layer_k_step` (with bounded sub-steps, total work <=
      20000 steps) outputting `corridor_feasible_count` and best corridor state
      and asserting `corridor_feasible_count >= 1`, OR tests a calibrated interior
      operating point inside the feasible corridor against the full target triad.
- DISJOINT PARAMETER & OBSERVABLE SLOTS: static parameters (e.g. `V_max`,
  `K_m`, `EC_50`, `E_max`, `beta_0`) and derived outputs (e.g. `occ`, `E_eff`,
  `R_eff`) MUST occupy separate indices in `buf.data[...]` and `variable_map`;
  NEVER write a derived output back into a static parameter slot.
- MONTE CARLO ENSEMBLE STEPPING (CRITICAL): NEVER multiply an outer contract
  step count (`steps >= 50`) by an inner Monte Carlo replication loop (`M >= 50`)
  inside `simulation_step` (O(steps * M * N) causes >120s sandbox timeouts). If
  `simulation_step` already runs `M` replications internally, set `steps = 1`
  (single-shot ensemble contract); otherwise accumulate 1 replication per step
  with `steps = M`. Also keep `O(N)` sample-accumulation loops directly inside
  the step/accumulation function rather than returning a 4+ tuple from an
  `O(N)` helper (tuple scalarization multiplies runtime by tuple arity).
- SI CONCENTRATION & UNIT CONVERSIONS (CRITICAL): `1 M = 1 mol/L = 1000 mol/m^3`;
  `1 mM = 1 mmol/L = 1.0 mol/m^3` (numerically identical: e.g. `12.0 mM = 12.0 mol/m^3`,
  `3.0 mM = 3.0 mol/m^3` — NEVER divide `mM` by `1000` when converting to `mol/m^3`!);
  `1 uM = 1e-3 mol/m^3` (when a threshold is stated in `uM`, keep the observable in `uM`
  scale, e.g. `peak_ros_uM < 2.0`, matching its `variable_map` name);
  `1 uF/cm^2 = 0.01 F/m^2`."""

_FORMAL_EXAMPLE_PROMPT = "All quantities are dimensionless. x in [0,1]; y in [0,2]. Model: y = 2*x. Claim: y >= x."
_FORMAL_EXAMPLE_PACK = {
    "evidence": {
        "slots": [
            {"slot_idx": 0, "symbol": "x", "si_dim": [0]*7, "role": "CONTROL_PARAM"},
            {"slot_idx": 1, "symbol": "y", "si_dim": [0]*7, "role": "DERIVED_OBSERVABLE"},
        ],
        "equations": {"y": {"expression": "2*x", "si_dim": [0]*7}},
        "environmental_assumptions": ["x >= 0"],
        "obligations": [{"id": "h", "expression": "y >= x", "obligation_class": "HYPOTHESIS_TARGET",
            "prompt_quote": "y >= x", "prompt_span": [_FORMAL_EXAMPLE_PROMPT.index("y >= x"), _FORMAL_EXAMPLE_PROMPT.index("y >= x")+len("y >= x")]}],
        "quantifier_mode": "FORALL", "mechanism_parameters": ["x"],
    },
    "domain": {"x": [0, 1], "y": [0, 2]}, "archetype": "ODE_PDE",
}
TOOLS_CONTEXT += ("\n\n## Worked formal-evidence example (schema only)\n"
    "This unrelated scalar example illustrates encoding, not a certified discovery. "
    "Use its schema, never its symbols, numbers or claim in place of the user's task.\n"
    "Example original research_prompt: " + json.dumps(_FORMAL_EXAMPLE_PROMPT) + "\n"
    "Example ihcg_evidence:\n```json\n" + json.dumps(_FORMAL_EXAMPLE_PACK, separators=(",", ":")) + "\n```\n")

PLANNER_PROMPT = """{tools}

## Task — Research Agent Planner
You are the Research Agent planner of a mathematically verified computational
research engine. Produce ONE falsifiable research blueprint for the request.

## Output: strict JSON object with keys
blueprint_id (lowercase snake_case, under 32 chars, e.g. 'mars_entry_probe'),
title (concise human-readable, under 80 chars, e.g. 'Mars Atmospheric Entry &
Stagnation Heating'), falsifiable_hypothesis,
search_queries (3-6 concise technical keyword queries of 2-4 terms each for
arXiv/OpenAlex; NEVER full sentences, and never use filler words like
'vs', 'versus', 'exact', 'using', 'between' or 'comparison'),
target_research_stages (1-4; use 3 when the request describes 3+ verification
regimes or a multi-stage progression, e.g. "start by ... then integrate ..."),
stage_objectives (one short objective per stage, stage 1 first), contracts
(staged contracts across stages 1..target_research_stages, prefixed c1_...,
c2_..., c3_...: list of {{contract_id, target_backend, timeout_seconds,
invariants: [{{variable_name, assertion, tolerance}}], max_memory_mb,
initial_values: {{"<variable_map_name>": float}}, initial_state: [floats], dt, steps}}),
code_template (the unified module implementing all staged contracts, v9_transpiler_friendly Python),
variable_map ({{"<metric_with_units>": <buf.data index>}} for every numerical
quantity the user requested), max_feedback_rounds (2-4). Across stages each named
metric keeps ONE unique buf.data slot; never reuse an error/drift accumulator to
store a previous observable; compute val_before at the top and val_after at the
bottom and update violations with `viol = max(0.0, val_after - val_before)`.
Keep initial_state small (<= 16 slots unless the surrogate genuinely needs more,
max 64) and steps <= 20000. initial_state must be the physical state of the
surrogate model, never a repeating dummy vector. Keep code_template concise
(<= 80 lines of scalar Python implementing the primary reference integrator in
simulation_step; do not embed duplicate Euler/RK4 state copies inside simulation_step).
Use a single canonical, DISJOINT `buf.data[k]` slot layout across all contracts
and sub-models (never reuse the same slot index for different physical observables).
Each variable/invariant has a dedicated global `buf.data[idx]` slot. Write every
computed quantity to its exact assigned `buf.data[idx]` slot on every step. NEVER
branch on `buf.size` or `len(buf.data)` (e.g. `if buf.size > 10:`), and NEVER
overwrite input parameter slots (`N_cells`, `P_cell`, etc.) with output observables.
Implement exact mathematical formulas cleanly without ad-hoc smoothing epsilons (e.g. NEVER write `math.log(x + 1e-12)` or `math.sqrt(x + 1e-12)` for `log(x)`/`sqrt(x)`) and use closed-form analytic derivatives/Jacobians instead of finite-difference steps (`h = 1e-5`).
No markdown fences, no <think> blocks. Output JSON only.

## Feasibility context (authoritative)
{feasibility}

## Literature context (may be empty)
{literature}

Research request: {question}
"""

ACCRETION_PROMPT = """{tools}

## Task — Cumulative Blueprint Evolution (Accretive Stage Integration)
You are extending a VERIFIED computational research module (cumulative
blueprint evolution, stage {stage} of {total}).

## Rules for accretive integration (MANDATORY)
- The verified module below is pinned. Pinned functions are shown as signature + slot
  contract stubs (or full code) and are re-inserted automatically by the merger;
  NEVER modify or rename verified functions or constants.
- Add NEW helper functions for the new physics/algorithm and write a NEW
  `simulation_step(buf, dt)` driver that CALLS the verified functions (e.g. the
  verified force/energy helpers or `layer_{prev}_step(buf, dt)`) so the new logic
  executes integrated with the verified foundation. An uncoupled driver is rejected.
- NON-NO-OP STAGE LAYER RULE (CRITICAL): NEVER emit a bare forwarding `simulation_step(buf, dt)`
  whose body is just `layer_{prev}_step(buf, dt)` (with or without positivity clamps).
  Stage {stage}'s `simulation_step(buf, dt)` MUST execute genuinely new computational
  statements addressing Stage {stage}'s objective (e.g. evaluating corridor/regime
  margins or control feedback, integrating Stage {stage}'s extended dynamics, or
  computing comparative stepper/error diagnostics into Stage {stage}'s `buf.data` slots)
  alongside calling the pinned foundation.
- DO NOT DOUBLE-INCREMENT TIME: `layer_{prev}_step(buf, dt)` ALREADY advances the time
  slot (e.g. `buf.data[0]`) by `dt`. Never add a second `buf.data[0] = buf.data[0] + dt`
  after calling `layer_{prev}_step(buf, dt)`.
- Verified layers are re-verified with their original contracts inside the same
  compiled module, so do not change the meaning of shared buffer slots used by
  verified layers; use new slots (append beyond the existing ones) for new state.
- Canonical slot map (MANDATORY): every named observable/metric has exactly one
  buf.data slot across ALL stages. Each variable/invariant has a dedicated global
  `buf.data[idx]` slot. Write every computed quantity to its exact assigned
  `buf.data[idx]` slot on every step. NEVER branch on `buf.size` or
  `len(buf.data)`, and NEVER overwrite input parameter slots (`N_cells`,
  `P_cell`, etc.) with output observables. Never reuse an error/drift accumulator slot to
  store a previous observable. For step-to-step drift compute val_before at the
  top and val_after at the bottom, then `viol = max(0.0, val_after - val_before)`;
  update the error slot only when `viol` is larger. Helper calls must receive the
  identical coordinate representation in forward and time-reversed steps.
- `None` literals are scalarized by Tier-0; still prefer explicit `0.0` returns.
- Add cross-coupling invariants that test the combined system (e.g. bounded
  limit cycle amplitude, phase-space bounds, reduction to the verified model when
  the new coupling is zero).
- When tracking limit cycles, extrema, or periods, use scalar state variables
  inside `buf.data` (e.g., `prev_val = buf.data[0]`, `cycle_count = buf.data[10]`,
  `last_peak_t = buf.data[11]`). NEVER use dynamic Python lists (`list.append`),
  `while` searches across history, or variable-length arrays, as they cannot
  lower to C++ C-ABI.

## Output: strict JSON object with keys
code (new functions + new simulation_step calling `layer_{prev}_step(buf, dt)`; pinned
functions are restored automatically), contracts (NEW contracts for this stage, same
schema as before, initial_state long enough for all slots), rationale (one paragraph).
No markdown fences. Output JSON only.

## Stage objective
{objective}

## Verified pinned signatures & slot contracts (stages 1..{prev})
```python
{verified_code}
```

## Canonical slot map (reuse exactly; do not renumber)
{canonical_slots}

## Verified contracts summary (still enforced)
{verified_contracts}

## Planned contracts & required slots for Stage {stage} (MUST compute every listed slot)
{stage_contracts}

## Telemetry of the last verified stage
{telemetry}

## Literature retrieved for this stage
{literature}

## Previous failed attempts for this stage (fix these)
{failures}
"""

REFINE_PROMPT = """{tools}

## Task — Kernel Verification Repair (Contracts Fixed)
Your research kernel failed verification. Fix the CODE ONLY; the contracts
and invariants are fixed and must not be weakened.

CRITICAL — v9_transpiler_friendly subset (violations are rejected by PreWrite_Guard):
use ONLY pure module-level functions, math.* calls, if/elif/else, for/while loops,
return statements and plain float arithmetic. FORBIDDEN: tuples (no tuple literals,
no `a, b = ...` unpacking, no multi-value returns), classes, exceptions, dicts,
sets, comprehensions, nested function definitions, string operations, imports
other than `import math`. If the failure mentions "unsupported construct: tuples",
eliminate EVERY tuple from the module.
Each variable/invariant has a dedicated global `buf.data[idx]` slot. Write every
computed quantity to its exact assigned `buf.data[idx]` slot on every step. NEVER
branch on `buf.size` or `len(buf.data)` (e.g. `if buf.size > 10:`), and NEVER
overwrite input parameter slots (`N_cells`, `P_cell`, etc.) with output observables.
Pinned functions marked `[PINNED VERIFIED LAYER]` are restored automatically by the
merger; focus your repair on the active driver and unpinned functions.
Output strict JSON: {{"code": "<fixed module>", "rationale": "..."}}. No fences.

## Stage objective
{objective}

## Failing module (pinned layer bodies omitted for token economy)
```python
{code}
```

## Contracts
{contracts}

## Counterexample / diagnostics
{failure}
"""

HEALING_DIRECTIVE_PROMPT = """You are the directive-only healing controller for a verified research kernel.
You MUST NOT output Python, C++, or any source code. Output ONE strict JSON object:
{"fixable": true|false, "unfixable_reason": "...", "directives": [...]}.
Allowed directive ops: INLINE_INTRINSIC {name}, GUARD_MATH_DOMAINS {eps},
REWRITE_MONOTONE_DRIFT {slot,direction,before_var,after_var}, WRAP_SUBSTEP {substeps},
SCALE_COEFFICIENT {var,factor}, CLAMP_STATE_NORM {slots,max_norm},
BIND_SLOT_TO_VAR {slot,var}, STRIP_NONE_LITERALS {}, REPAIR_DRIFT_SELF_COMPARISON {slot,prev_slot},
LOWER_COMPLEX_TO_REAL {}, PRUNE_UNUSED_HELPERS {}.
No markdown fences. Output JSON only.

Stage objective: {objective}
Failing module hash/context (never rewrite it):
{code}
Contracts:
{contracts}
Failure/counterexample:
{failure}
"""

QUERY_PROMPT = """You steer literature search for a computational research loop.
Write 3 precise literature search queries (arXiv/OpenAlex style, <= 12 words each)
that target the higher-order effects, couplings, closures or algorithmic fixes the
telemetry points to. Output a JSON list of strings only.

Stage {stage} objective: {objective}
Hypothesis: {hypothesis}
Numerical telemetry from the previous stage: {telemetry}
Failures so far: {failures}"""

# -- combined feasibility + blueprint (single Tier-2 call, Task 4 latency fix) --

COMBINED_PLAN_PROMPT = """{tools}

## Task — Epistemic Feasibility Gate & Research Blueprint Planner
You are the Epistemic Feasibility Gate AND the Research Agent planner
of a mathematically verified computational research engine, in ONE pass. Produce
ONE strict JSON object with two keys: "feasibility" and "blueprint".

## Part 1 — feasibility (gate)
Classify the request into EXACTLY ONE verdict:
- OPTIMAL_DIRECT: quantitative, mathematical, physical, dynamical or algorithmic
  research falsifiable by simulation, INCLUDING requests that ask to formally
  test, analyze, or audit a falsifiable hypothesis against governing laws (even
  if you expect the hypothesis to be refuted/falsified — NEVER refuse a
  consistent testable hypothesis; emit OPTIMAL_DIRECT and keep the hypothesis
  intact so the formal falsification certifier can prove VERIFIED_FALSIFICATION).
- REFORMULATED_SURROGATE: too large for full-scale local execution but admits a
  rigorous reduced-order / 0D-1D asymptotic / scaling-law surrogate.
- REFUSED_IMPOSSIBLE: ONLY when the request's own environmental assumptions are
  mutually contradictory (e.g. requiring a single parameter to satisfy T >= 500 K
  and T <= 300 K simultaneously), or asks to build/fabricate a perpetual-motion
  device rather than formally test a hypothesis.
- REFUSED_UNSUITABLE: subjective/qualitative/historical/clinical/literary tasks
  without falsifiable governing equations, or requiring forbidden sandbox ops.
For REFORMULATED_SURROGATE, the "reformulation" field MUST name the exact
surrogate equations, scaling laws, boundary conditions and observables to
synthesise (e.g. relativistic Lorentz factor, Larmor-radius magnetic deflection,
kinetic-energy flux over a scoop cross-section, radiative/ablative thermal
balance, stopping-range thickness optimisation).
feasibility object keys: verdict, rationale, violated_boundary, reformulation.

## Part 2 — blueprint (only if NOT refused)
Produce ONE falsifiable research blueprint:
blueprint_id (lowercase snake_case, under 32 chars, e.g. 'mars_entry_probe'),
title (concise human-readable, under 80 chars, e.g. 'Mars Atmospheric Entry &
Stagnation Heating'), falsifiable_hypothesis,
search_queries (3-6 concise technical keyword queries of 2-4 terms each for
arXiv/OpenAlex; no full sentences, no 'vs'/'exact'/'using' filler),
target_research_stages (1-4; use 3 when the request describes 3+ verification
regimes or a progression),
stage_objectives (one short objective per stage, stage 1 first), contracts
(staged contracts across stages 1..target_research_stages, prefixed c1_...,
c2_..., c3_...: list of {{contract_id, target_backend, timeout_seconds,
invariants: [{{variable_name, assertion, tolerance}}], max_memory_mb,
initial_values: {{"<variable_map_name>": float}}, initial_state: [floats], dt, steps}}), code_template (the unified module
implementing all staged contracts, v9_transpiler_friendly Python), variable_map ({{"<metric_with_units>":
<buf.data index>}} for every numerical quantity the user requested),
max_feedback_rounds (2-4).
Keep initial_state small (<= 16 slots unless the surrogate genuinely needs more,
max 64) and steps <= 20000. initial_state must encode the physical state of the
surrogate model, never a repeating dummy vector. Keep code_template concise
(<= 80 lines of scalar Python implementing the primary reference integrator in
simulation_step; do not embed duplicate Euler/RK4 state copies inside simulation_step).
Implement exact mathematical formulas cleanly without ad-hoc smoothing epsilons (NEVER write `math.log(x + 1e-12)` or `math.sqrt(x + 1e-12)` for `log(x)`/`sqrt(x)`) and use closed-form analytic derivatives/Jacobians instead of finite-difference steps (`h = 1e-5`).
If the verdict is REFORMULATED_SURROGATE, implement exactly the surrogate equations named in the
reformulation.

## Executable-research requirements (ALL mandatory for quantitative problems)
1. **Governing equations & scaling laws**: name the exact ODE/PDE or scaling
   relations (e.g. Allen-Eggers aerothermodynamic entry relations, Sutton-Graves
   radiative heating, Newtonian drag F = 0.5 rho v^2 Cd A) and how they couple.
2. **Initial conditions & numerical grid**: specify mass, entry velocity,
   atmospheric scale height, density profile, time step dt and integration
   duration as concrete numbers inside the contracts/initial_state.
3. **Falsifiable verification invariants with tolerances**: every contract
   invariant must be a physically meaningful bound the simulation can violate
   (e.g. peak acceleration <= 50 g, heat flux >= 0, total heat load finite,
   energy monotonicity) with explicit tolerance values — never vacuous.
   CRITICAL INVARIANT RULE: Do NOT guess narrow arbitrary numeric lower/upper
   intervals (e.g. `15.0 <= state[7] <= 70.0`) before the ODE is integrated.
   Only write invariants for: (a) explicit bounds stated by the user (e.g.
   `state[3] < 40.0`), (b) physical positivity and finiteness
   (`state[i] >= 0.0`, `math.isfinite(state[i])`), (c) monotonic conservation
   laws (`state[1] <= initial_state[1]`), or (d) exact analytical identities
   within a generous relative tolerance (>= 15% for exponential atmosphere
   entry approximations).
4. **Backend recommendation**: prefer "cpp" (C-ABI) for ODE/PDE time stepping
   and array workloads; use "rust_hin" for scalar recurrences/algebraic checks;
   "multi_variant" only for genuine algorithm comparisons.
5. **Named telemetry observables**: variable_map must name every requested
   output with units (e.g. peak_g, peak_alt_km, max_heat_flux_mw_m2,
   total_heat_load_mj_m2) mapped to buffer slots the module writes.
6. **v9_transpiler_friendly code**: code_template uses ONLY the allowed subset
   (pure functions, simulation_step(buf, dt), math.* only, no classes,
   exceptions, dynamic imports, dicts/sets/tuples/comprehensions, strings).
   Each variable/invariant MUST have a dedicated globally DISJOINT `buf.data[idx]`
   slot across ALL contracts `c1..cN`. Write every computed quantity to its exact
   assigned `buf.data[idx]` slot on every step. NEVER branch on `buf.size` or
   `len(buf.data)` (e.g. `if buf.size > 10:`), and NEVER overwrite input
   parameter slots (`N_cells`, `P_cell`, etc.) with output observables.

## Few-shot exemplar (aerothermodynamics entry)
Request: "600 kg probe, 5.8 km/s entry, 1.4 m spherical-segment aeroshell
(nose radius 0.7 m) — compute peak g, peak altitude, peak heat flux and total
heat load."
Blueprint sketch: title "Martian aerothermodynamic entry of 600 kg probe";
hypothesis "A 1.4 m diameter 70 deg spherical-segment aeroshell survives Martian
entry with peak deceleration below 50 g and radiative heat flux consistent with
Sutton-Graves scaling"; contracts: cpp backend, dt=0.05 s, steps<=20000,
initial_state [alt_km=125.0, v_kms=5.8, mass_kg=600.0, nose_radius_m=0.7,
scale_height_km=11.1, rho0_kg_m3=0.02, Cd=1.5, sigma=5.67e-8, epsilon=0.85],
invariants ["telemetry['final_max'] >= 0.0" boundedness, "state[8] <= 50.0"
peak g bound with tolerance 1e-3, "state[10] >= 0.0" non-negative heat flux];
variable_map {"peak_g": 8, "peak_alt_km": 9, "max_heat_flux_mw_m2": 10,
"total_heat_load_mj_m2": 11}; code_template integrates m dv/dt = -0.5 rho v^2
Cd A - m g, dh/dt = -v sin(gamma), rho = rho0 exp(-h/H), Sutton-Graves q =
k sqrt(rho/Rn) v^3 with RK4 in the allowed subset.

Output strict JSON only: {{"feasibility": {{...}}, "blueprint": {{...}}}}.
No markdown fences, no <think> blocks.

## Literature context (may be empty)
{literature}

Research request: {question}
"""

# -- Plan Mode: expert deep-research prompt (never raw blueprint JSON) --------

EXPERT_PROMPT_PLANNER_PROMPT = """You are the Expert Research Prompt Planner of a mathematically
verified computational research engine. Turn the user's request into an
EXPERT DEEP RESEARCH PROMPT that another Research Agent instance will execute
verbatim. Also classify feasibility in the same pass.

(No tool schema needed: you write a natural-language brief, not code. Keep the
brief concise — at most ~900 words — while covering all four sections below.)

## Feasibility gate (same pass)
Classify the request into EXACTLY ONE verdict:
- OPTIMAL_DIRECT, REFORMULATED_SURROGATE, REFUSED_IMPOSSIBLE, REFUSED_UNSUITABLE
(OPTIMAL_DIRECT = fully specified computable model; REFORMULATED_SURROGATE =
complex real-world problem that becomes computable via a rigorous 0D/1D
reduced-order surrogate — this is the CORRECT verdict for sophisticated
engineering/physics optimization requests such as relativistic spacecraft
shielding, drag, thermal or materials problems; REFUSED_IMPOSSIBLE = violates
conservation laws or physical upper bounds, e.g. COP > 1, efficiency > 1,
perpetual motion, net energy from nothing; REFUSED_UNSUITABLE = no falsifiable
mathematical model exists at all, e.g. romantic literature analysis).
NEVER refuse a quantitative physics/engineering request merely because it
needs a surrogate model or spans multiple physical effects — reformulate it.

## Output: strict JSON only
{{"feasibility": {{"verdict": ..., "rationale": ..., "violated_boundary": ...,
"reformulation": ...}}, "expert_prompt": "..."}}
If the verdict is REFUSED_IMPOSSIBLE or REFUSED_UNSUITABLE, set expert_prompt to
"" (empty). Otherwise expert_prompt must be a complete, authoritative, editable
markdown research brief with EXACTLY these sections:
1. **Objective & Quantitative Targets** — the exact numerical values, optimal
   parameters and physical/mathematical relationships the Research Agent must
   compute and return (with units).
2. **Surrogate & Governing Physics Formulation** — if REFORMULATED_SURROGATE,
   the exact 0D/1D/analytical-numerical surrogate equations, scaling laws and
   boundary conditions to synthesise; otherwise the governing equations.
3. **Literature Retrieval Directives** — the specific empirical constants,
   material limits and governing equations to extract via live literature
   search (arXiv / OpenAlex / Wikipedia).
4. **Computational Synthesis & Verification Protocol** — how to structure
   Stage 1 (foundational physics/baseline model) and Stage 2+ (coupled
   thermal/material/optimization accretion) in the v9_transpiler_friendly
   compute synthesizer, which invariants to verify in Z3/Sandbox, and which
   output variables to report via variable_map.
The expert_prompt must NEVER contain raw ResearchBlueprint JSON
(no initial_state lists, no code_template) — it is a natural-language brief.
No markdown fences, no <think> blocks. Output JSON only.

## Executable-research requirements (the brief MUST make these explicit)
The Research Agent executes the brief verbatim, so the expert_prompt must spell
out ALL of the following (use the aerothermodynamics entry problem below as
the style exemplar):
1. **Governing equations & scaling laws** — exact ODE/PDE relations or scaling
   laws (Allen-Eggers entry relations, Sutton-Graves heating, Newtonian drag)
   and how they couple; name constants and where literature must supply them.
2. **Initial conditions & numerical grid** — concrete mass, entry speed,
   atmospheric scale height, density, dt and integration duration.
3. **Falsifiable verification invariants with tolerances** — e.g. peak
   deceleration <= 50 g, heat flux >= 0, cumulative heat load finite; each with
   an explicit tolerance.
4. **Backend recommendation** — cpp C-ABI for ODE/PDE stepping, rust_hin for
   scalar/algebraic checks, multi_variant only for real comparisons.
5. **Named telemetry observables** — e.g. peak_g, peak_alt_km,
   max_heat_flux_mw_m2, total_heat_load_mj_m2, each with units.
6. **v9_transpiler_friendly synthesis** — pure functions + simulation_step(buf,
   dt) only; NO classes, exceptions, dynamic imports, dicts/sets/tuples/comprehensions
   or non-math imports.
7. **Verbatim parameter preservation** — copy every numerical parameter from
   the user's prompt EXACTLY (e.g. "5.75 km/s" stays 5.75 km/s = 5750.0 m/s,
   never rounded to 5800 m/s), and state each peak/maximum observable
   (`max_*`/`peak_*`) as a RUNNING MAXIMUM tracked inside simulation_step with
   explicit SI-prefix unit conversion (W/m² -> MW/m² divides by 1e6, etc.).

## Few-shot exemplar (aerothermodynamics entry)
User: "Research the aerothermodynamic entry environment for a 600 kg planetary
probe entering the Martian atmosphere at 5.8 km/s with a 1.4-meter diameter
spherical-segment aeroshell (nose radius 0.7 m). Determine peak deceleration,
peak altitude, peak heat flux, and total integrated heat load."
Expert brief (abridged): Objective — compute peak deceleration [g], peak
altitude [km], Sutton-Graves peak heat flux [MW/m^2] and cumulative heat load
[MJ/m^2] for the stated entry state. Physics — Newtonian drag F = 0.5 rho v^2
Cd A with Cd from literature, m dv/dt = -F/m - m g sin(gamma), dh/dt = -v
sin(gamma), rho(h) = rho0 exp(-h/H) with Martian scale height H ~ 11.1 km and
surface density rho0 ~ 0.02 kg/m^3 (verify via literature search); Sutton-Graves
q = k sqrt(rho/Rn) v^3 with emissivity/absorptivity from literature; RK4
integration. Initial conditions — m0 = 600 kg, v0 = 5800 m/s, h0 = 125 km,
Rn = 0.7 m, A = pi Rn^2-equivalent aeroshell area, dt = 0.05 s until impact or
20000 steps. Invariants — peak deceleration <= 50 g (tol 1e-3), heat flux
>= 0, heat load finite, mass constant, trajectory monotonic descent after peak
altitude. Backend — cpp C-ABI. Observables — peak_g, peak_alt_km,
max_heat_flux_mw_m2, total_heat_load_mj_m2. Literature — fetch Martian
atmospheric density/scale-height profiles, Sutton-Graves constants,
spherical-segment aeroshell drag coefficients.

## Literature context (may be empty)
{literature}

Research request: {question}
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


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)
_WHOLE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", re.S)


def _iter_balanced(text: str, open_c: str, close_c: str, limit: int = 12):
    """Yield balanced open_c..close_c chunks, string-aware, left to right."""
    i, n, yielded = 0, len(text), 0
    while yielded < limit:
        start = text.find(open_c, i)
        if start == -1:
            return
        depth, in_str, esc, j = 0, False, False, start
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == open_c:
                    depth += 1
                elif ch == close_c:
                    depth -= 1
                    if depth == 0:
                        yield text[start:j + 1]
                        yielded += 1
                        break
            j += 1
        i = j + 1 if depth == 0 else start + 1


def _extract_json(text: str) -> Optional[Any]:
    """Robust JSON extraction: strip think-blocks, unwrap fences, then use a
    string-aware balanced-brace matcher so trailing prose after the JSON
    object does not break parsing."""
    text = (text or "").strip()
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(document, (dict, list)) and document != {}:
            from ..latex_normalizer import repair_json_strings
            return repair_json_strings(document)
    # Reasoning models may leak chain-of-thought wrappers; strip them first.
    text = _THINK_BLOCK.sub("", text).strip()
    if not text:
        return None
    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    candidates = [c for c in candidates if c] + [text]
    first_valid: Optional[Any] = None
    best_dict: Optional[Dict[str, Any]] = None
    best_rank = 0
    for candidate in candidates:
        for open_c, close_c in (("{", "}"), ("[", "]")):
            for chunk in _iter_balanced(candidate, open_c, close_c):
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    try:
                        parsed = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', chunk))
                    except json.JSONDecodeError:
                        continue
                if first_valid is None and parsed != {}:
                    first_valid = parsed
                if isinstance(parsed, dict) and parsed:
                    blueprint = parsed.get("blueprint")
                    rank = (4 if isinstance(blueprint, dict) and (blueprint.get("blueprint_id") or blueprint.get("contracts")) else
                            3 if any(k in parsed for k in ("blueprint_id", "contracts", "code", "directives")) else
                            2 if "feasibility" in parsed else 1)
                    if rank > best_rank or (rank == best_rank == 1 and len(parsed) > len(best_dict)):
                        best_dict = parsed
                        best_rank = rank
    chosen = best_dict if best_dict is not None else first_valid
    if chosen is not None:
        from ..latex_normalizer import repair_json_strings
        return repair_json_strings(chosen)
    return None


def _clean_prompt_text(raw: Any) -> str:
    """Think-block-stripped, fence-unwrapped plain text for the Markdown
    fallback path (no JSON object present in the LLM response)."""
    text = _THINK_BLOCK.sub("", str(raw or "")).strip()
    whole = _WHOLE_FENCE_RE.match(text)
    if whole:
        text = whole.group(1).strip()
    return text


# -- schema-drift normalisation ------------------------------------------------

_BLUEPRINT_FIELDS = {
    "blueprint_id", "title", "falsifiable_hypothesis", "search_queries",
    "contracts", "code_template", "max_feedback_rounds", "verified_layers",
    "target_research_stages", "stage_objectives", "current_stage",
    "evolution_history", "unified_module_code", "feasibility", "variable_map",
    "slot_roles", "typed_slots", "structured_evidence", "extracted_constants",
    "ihcg_evidence", "research_prompt", "constants", "derived_expressions",
    "assumptions_ledger", "formal_spec",
}
_CONTRACT_FIELDS = {
    "contract_id", "target_backend", "timeout_seconds", "invariants",
    "max_memory_mb", "initial_values", "initial_state", "dt", "steps",
}
_INVARIANT_FIELDS = {"variable_name", "assertion", "tolerance"}
_BACKENDS = {"rust_hin", "cpp", "cuda", "multi_variant"}


def _as_float_list(value: Any) -> Optional[List[float]]:
    """Coerce a numeric list (int/float/str entries) to floats; None if not a list."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return None
    out: List[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def _sanitize_invariant_assertion(expr: str) -> Optional[str]:
    """Deterministically normalize an LLM-emitted invariant assertion into valid Python eval syntax."""
    import ast as _ast

    s = str(expr or "").strip()
    if not s:
        return None
    s = s.replace("&&", " and ").replace("||", " or ").replace("^", "**")
    s = s.rstrip(".;, ")
    # Strip trailing parenthetical label like `state[4] <= 1e-14 (trace_error_max)`
    s = re.sub(r"\s+\([A-Za-z_][A-Za-z0-9_ ]*\)$", "", s).strip()
    try:
        _ast.parse(s, mode="eval")
        return s
    except SyntaxError:
        # Try extracting a clean comparison over state/initial/trace_min/trace_max/telemetry
        m = re.search(
            r"((?:abs\s*\([^)]+\)|(?:state|initial|trace_min|trace_max|telemetry)\[[^\]]+\])\s*(?:<=|>=|<|>|==|!=)\s*[A-Za-z0-9_.\[\]'\"+\-*/() ]+)",
            s,
        )
        if m:
            cand = m.group(1).strip().rstrip(".;, ")
            try:
                _ast.parse(cand, mode="eval")
                return cand
            except SyntaxError:
                pass
    return None


def _coerce_contract(raw: Any, index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    contract: Dict[str, Any] = {k: raw[k] for k in _CONTRACT_FIELDS if k in raw}
    contract["contract_id"] = str(contract.get("contract_id") or f"contract_{index + 1}")[:120]
    backend = str(contract.get("target_backend") or "cpp").strip().lower()
    contract["target_backend"] = backend if backend in _BACKENDS else "cpp"
    try:
        contract["timeout_seconds"] = max(1.0, min(300.0, float(contract.get("timeout_seconds", 30.0))))
        raw_dt = float(contract.get("dt", 0.01))
        contract["dt"] = max(1e-12, min(1e8, raw_dt if raw_dt > 0.0 else 0.01))
        contract["steps"] = max(1, min(100000, int(contract.get("steps", 1000))))
        contract["max_memory_mb"] = max(64, min(8192, int(contract.get("max_memory_mb", 1024))))
    except (TypeError, ValueError):
        return None
    raw_iv = contract.get("initial_values")
    if isinstance(raw_iv, dict):
        iv_clean: Dict[str, float] = {}
        for k, v in raw_iv.items():
            if len(iv_clean) >= 4096:
                break
            ks = str(k).strip()[:120]
            if not ks:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv == fv and abs(fv) != float("inf"):
                iv_clean[ks] = fv
        if iv_clean:
            contract["initial_values"] = iv_clean
        else:
            contract.pop("initial_values", None)
    else:
        contract.pop("initial_values", None)
    initial = _as_float_list(contract.get("initial_state"))
    if initial is not None:
        contract["initial_state"] = initial[:4096]
    else:
        contract.pop("initial_state", None)
    invariants = []
    for inv in contract.get("invariants") or []:
        if not isinstance(inv, dict):
            continue
        cleaned = {k: inv[k] for k in _INVARIANT_FIELDS if k in inv}
        cleaned["variable_name"] = str(cleaned.get("variable_name") or "value")[:120]
        raw_assert = str(cleaned.get("assertion") or "")[:2000]
        sanitized = _sanitize_invariant_assertion(raw_assert)
        cleaned["assertion"] = sanitized if sanitized else "telemetry['finite'] == 1.0"
        try:
            cleaned["tolerance"] = float(cleaned.get("tolerance", 1e-6))
        except (TypeError, ValueError):
            cleaned["tolerance"] = 1e-6
        if cleaned["assertion"]:
            invariants.append(cleaned)
    if not invariants:
        return None
    contract["invariants"] = invariants
    return contract


def _sanitize_raw_blueprint_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-validation normalisation of a raw LLM blueprint dict.

    Handles missing optional fields, trims excessive strings and coerces
    integer/float lists so pydantic validation never crashes on long titles,
    unsafe identifiers or minor type drift.
    """
    if not isinstance(raw, dict):
        return {}
    out = dict(raw)
    # Identifiers: slugify instead of failing on spaces/special characters.
    for key, fallback in (("blueprint_id", "blueprint_unnamed"),):
        if key in out:
            from invar_agent.schemas.research import _slugify
            out[key] = _slugify(out.get(key), fallback=fallback)
    # Title: collapse whitespace and truncate before the schema validator runs.
    if "title" in out and out.get("title") is not None:
        title = " ".join(str(out["title"]).strip().split())
        if len(title) > 120:
            title = title[:117] + "..."
        out["title"] = title or "Untitled Research Blueprint"
    # Numeric list fields: coerce int/float/string entries to floats.
    if "initial_state" in out:
        coerced = _as_float_list(out.get("initial_state"))
        if coerced is not None:
            out["initial_state"] = coerced[:4096]
        else:
            out.pop("initial_state", None)
    return out



_CANON_SLOT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:state|buf\.data|data)\[(\d+)\]")
_CANON_REWRITE_RE = re.compile(
    r"(?P<prefix>\b(?:state|initial|trace_min|trace_max|buf\.data|data))\[(?P<slot>\d+)\]")
_CANON_OBSERVABLE_TOKENS = (
    "error", "err", "drift", "purity", "entropy", "energy", "eigen", "roundtrip",
    "trace", "max", "min", "observable", "metric", "loss", "fidelity",
    "infidelity", "distance", "residual", "violation", "bound",
)
_CANON_STAGE_RE = re.compile(r"^(?:c|stage)(\d+)(?:[a-z](?=[_-])|[_-])", re.IGNORECASE)


def _canon_stage_index(contract_id: str) -> int:
    m = _CANON_STAGE_RE.match(contract_id or "")
    return int(m.group(1)) if m else 1


def _canon_observable_name(name: str, variable_map: Optional[Dict[str, int]] = None) -> bool:
    low = str(name or "").lower().strip()
    if not low or low in {"value", "state", "buf", "data", "finite", "finiteness", "finite_all"}:
        return False
    return True


def _canon_slots_in_text(text: str) -> List[int]:
    return [int(m.group(1)) for m in _CANON_SLOT_RE.finditer(text or "")]


def _canon_rewrite_slots(text: str, old: int, new: int) -> str:
    def repl(m: "re.Match[str]") -> str:
        if int(m.group("slot")) == int(old):
            return f"{m.group('prefix')}[{int(new)}]"
        return m.group(0)
    return _CANON_REWRITE_RE.sub(repl, text or "")


def _canon_alloc_slot(assigned: set, reserved: set) -> int:
    for cand in range(16, 4096):
        if cand not in assigned and cand not in reserved:
            return cand
    return max(assigned | reserved | {15}) + 1


_CANON_SEMANTIC_SUFFIXES = (
    "_finite", "_positivity", "_nonnegative", "_window",
    "_trace_rho2", "_bound", "_limit", "_max", "_min", "_current", "_val",
    "_check", "_error", "_err", "_drift",
)
# Concept buckets: every single-slot inequality constrained to machine
# precision (|tol| <= 1e-6) is classified by domain root tokens so the SAME
# physical quantity measured under different stage-specific names
# (``energy_drift_max`` vs ``unified_energy_drift``) shares one canonical slot.
_CONCEPT_BUCKETS: Tuple[Tuple[str, Tuple[str, ...], str, Tuple[str, ...]], ...] = (
    ("ENERGY_DRIFT", ("energy", "hamiltonian", "h_energy"), "le",
     ("drift", "err", "error", "residual", "bound", "violation")),
    ("ENTROPY_MONOTONE_LOWER", ("entropy", "s_entropy", "lyapunov"), "ge",
     ("monotone", "step", "min", "prod")),
    ("ENTROPY_DRIFT_UPPER", ("entropy", "s_entropy", "lyapunov", "spohn"), "le",
     ("drift", "monotone", "increase", "err", "error", "residual", "max")),
    ("TRACE_DRIFT", ("trace",), "le",
     ("drift", "err", "error", "max", "residual", "hold", "conservation")),
    ("ROUNDTRIP_ERROR", ("roundtrip", "revers", "chiasmus"), "le",
     ("roundtrip", "revers", "chiasmus", "err", "error", "drift")),
    ("UNIT_UPPER_BOUND", ("purity", "closed_energy", "norm"), "le",
     ("purity", "closed_energy", "norm", "bound", "upper", "max", "hold")),
    ("POSITIVITY_MIN_EIG", ("positivity", "eigenvalue", "lambda_min"), "ge",
     ("positivity", "eigenvalue", "lambda_min", "min")),
)


def concept_bucket_key(name: str, assertion: str
                       ) -> Optional[Tuple[str, str, str, int]]:
    """Canonical physical-concept key ``(bucket, direction, stem, slot)``.

    Only single-slot machine-precision inequalities qualify (``state[k] <= tol``
    or ``state[k] >= -tol`` with ``abs(tol) <= 1e-6``). Returns ``None`` for
    everything else so ordinary observables keep the legacy semantic key.
    """
    low = str(name or "").lower().strip()
    if not low:
        return None
    slots = _canon_slots_in_text(assertion or "")
    distinct_slots = sorted(set(slots))
    if len(distinct_slots) != 1 or len(slots) != 1:
        return None
    slot = distinct_slots[0]
    if slot in (0, 1, 2, 3):
        return None
    bound = re.search(r"(?:state|buf\.data|data)\[\d+\]\s*(?:<=|>=|<|>)\s*([0-9.eE+-]+)",
                      assertion or "")
    if bound is None:
        return None
    try:
        val = float(bound.group(1))
        is_zero_tol = abs(val) <= 1e-6
        is_unit_tol = abs(val - 1.0) <= 1e-6
        if not (is_zero_tol or is_unit_tol):
            return None
    except ValueError:
        return None
    direction = _canon_bound_direction(assertion)
    if direction not in {"le", "ge"}:
        return None
    stem = _canon_semantic_stem(low)
    for bucket, roots, bucket_dir, tokens in _CONCEPT_BUCKETS:
        if bucket_dir != direction:
            continue
        if bucket == "UNIT_UPPER_BOUND" and not is_unit_tol:
            continue
        if bucket != "UNIT_UPPER_BOUND" and not is_zero_tol:
            continue
        if bucket == "POSITIVITY_MIN_EIG" and any(
                bad in low for bad in ("violation", "rk4", "euler")):
            continue
        if not any(tok in low or tok in stem for tok in roots):
            continue
        if not any(tok in low for tok in tokens):
            continue
        return bucket, direction, stem, slot
    return None


_CONCEPT_REMAP_RE = re.compile(
    r"(?<![A-Za-z_])(?P<prefix>state|buf\.data|buf|data)\[(?P<slot>\d+)\]")


def concept_bucket_slot_map(contracts: List[Dict[str, Any]]
                            ) -> Tuple[Dict[int, int], List[str]]:
    """Map later-stage concept slots onto the first stage's slot per bucket.

    Contracts are visited in stage order; the first slot claimed for a
    concept bucket stays canonical and every later slot in the same bucket is
    remapped to it. Returns ``(remap, notes)`` where ``remap`` maps an old slot
    index to its canonical slot index (only entries that actually change).
    """
    ordered = sorted(
        enumerate(contracts or []),
        key=lambda kv: (_canon_stage_index(str(kv[1].get("contract_id") or "")), kv[0]))
    bucket_slot: Dict[Tuple[str, str], int] = {}
    remap: Dict[int, int] = {}
    notes: List[str] = []
    for _idx, contract in ordered:
        if not isinstance(contract, dict):
            continue
        for inv in contract.get("invariants") or []:
            if not isinstance(inv, dict):
                continue
            key = concept_bucket_key(str(inv.get("variable_name") or ""),
                                     str(inv.get("assertion") or ""))
            if key is None:
                continue
            bucket, direction, _stem, slot = key
            if slot in (0, 1, 2, 3):
                continue
            claim = (bucket, direction, _stem) if bucket == "POSITIVITY_MIN_EIG" else (bucket, direction)
            canonical = bucket_slot.setdefault(claim, slot)
            if canonical in (0, 1, 2, 3):
                continue
            if slot != canonical:
                remap[slot] = canonical
                notes.append(
                    f"concept slot unified: {inv.get('variable_name')} "
                    f"state[{slot}] -> state[{canonical}] ({bucket})")
    return remap, notes


def apply_concept_bucket_remap(contracts: List[Dict[str, Any]],
                               remap: Dict[int, int]) -> List[Dict[str, Any]]:
    """Rewrite every ``state[k]``/``buf.data[k]`` reference per ``remap``."""
    if not remap:
        return contracts

    def rewrite(text: str) -> str:
        slots = set(_canon_slots_in_text(text))
        if len({remap.get(slot, slot) for slot in slots}) < len(slots):
            raise ValueError(f'SLOT_SCHEMA_ERROR: remap would merge distinct operands: {text}')
        def repl(m: "re.Match[str]") -> str:
            slot = int(m.group("slot"))
            return f"{m.group('prefix')}[{remap.get(slot, slot)}]"
        return _CONCEPT_REMAP_RE.sub(repl, text or "")

    updates = []
    for contract in contracts or []:
        if not isinstance(contract, dict):
            continue
        for inv in contract.get("invariants") or []:
            if isinstance(inv, dict) and "assertion" in inv:
                updates.append((inv, rewrite(str(inv.get("assertion") or ""))))
    for inv, assertion in updates:
        inv['assertion'] = assertion
    return contracts
_CANON_INITIAL_STATE_RE = re.compile(r"\binitial_state\[(\d+)\]")


def _canon_semantic_stem(name: str) -> str:
    low = str(name or "").lower().strip()
    if not low:
        return ""
    changed = True
    while changed:
        changed = False
        for prefix in ("max_", "min_"):
            if low.startswith(prefix) and len(low) > len(prefix):
                low = low[len(prefix):]
                changed = True
        for suffix in _CANON_SEMANTIC_SUFFIXES:
            if low.endswith(suffix) and len(low) > len(suffix):
                low = low[:-len(suffix)]
                changed = True
    return low


def _canon_bound_direction(assertion: str) -> str:
    m = re.search(r"(?:state|buf\.data|data)\[\d+\]\s*(<=|>=|<|>)", assertion or "")
    if not m:
        return ""
    return "le" if m.group(1) in ("<=", "<") else "ge"


def _canon_semantic_key(name: str, assertion: str) -> tuple[str, str]:
    if name.lower().startswith(('peak_', 'min_', 'max_')):
        return (observable_name(name), "extremum")
    return (_canon_semantic_stem(name) or str(name or "").lower(),
            _canon_bound_direction(assertion))


def _materialize_initial_state_bounds(contracts: List[Dict[str, Any]]) -> None:
    """Replace ``initial_state[k]`` in assertions with its numeric literal."""
    for contract in contracts or []:
        if not isinstance(contract, dict):
            continue
        initial = contract.get("initial_state") or []
        for inv in contract.get("invariants") or []:
            if not isinstance(inv, dict):
                continue
            assertion = str(inv.get("assertion") or "")
            if "initial_state[" not in assertion:
                continue

            def repl(match: "re.Match[str]") -> str:
                idx = int(match.group(1))
                if 0 <= idx < len(initial):
                    try:
                        return format(float(initial[idx]), ".12g")
                    except (TypeError, ValueError):
                        pass
                return match.group(0)

            inv["assertion"] = _CANON_INITIAL_STATE_RE.sub(repl, assertion)


def canonicalize_contract_dicts(
        contracts: List[Dict[str, Any]],
        variable_map: Optional[Dict[str, int]] = None,
        prior_contracts: Optional[List[Dict[str, Any]]] = None,
        *,
        preserve_mapped_slots: bool = False,
        ) -> tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    """Deterministically give every named observable one unique state slot.

    Works on planner/accretion contract dicts before pydantic validation.
    Earlier stages win slot ownership; later duplicate names are rewritten to
    the canonical slot and conflicting names receive the next free high slot.
    """
    notes: List[str] = []
    contracts = copy.deepcopy(contracts or [])
    vm = {str(k): int(v) for k, v in (variable_map or {}).items() if str(k).strip()}
    if vm:
        from .slot_schema import observable_name
        _qual_re_c = re.compile(
            r"_(?:plausible|target|envelope|ceiling|floor|bounds?|window|final|baseline|initial|check|limit|corridor|steady|range|valid|ok|horizon|s|v|mv|mm|um|mol|m3|k|m|hz|db|s_m2|mmhg|units|frac|j_m2)$",
            re.I,
        )
        def _strip_qual_c(s_name: str) -> str:
            base = observable_name(s_name)
            prev = None
            while base and base != prev:
                prev = base
                base = _qual_re_c.sub("", base)
            return base or observable_name(s_name)

        _slot_to_vnames_c: Dict[int, List[str]] = {}
        _sem_to_vslot_c: Dict[str, int] = {}
        for _vn, _vs in vm.items():
            _slot_to_vnames_c.setdefault(int(_vs), []).append(str(_vn))
            _sem_to_vslot_c.setdefault(observable_name(str(_vn)), int(_vs))
            _sem_to_vslot_c.setdefault(_strip_qual_c(str(_vn)), int(_vs))
        for _c_item in contracts:
            if not isinstance(_c_item, dict):
                continue
            for _inv_item in (_c_item.get("invariants") or []):
                if not isinstance(_inv_item, dict):
                    continue
                _inm = str(_inv_item.get("variable_name") or "").strip()
                _ias = str(_inv_item.get("assertion") or "")
                _islots = sorted(set(_canon_slots_in_text(_ias)))
                if len(_islots) >= 2:
                    # Multi-slot additive conservation alignment: if an invariant asserts
                    # abs(state[i] + state[j] (+ ...) - target) <= tol and references any
                    # unmapped slot (not in variable_map) or slots whose initial_state sum
                    # disagrees with `target`, rebind to the matching contiguous mapped
                    # slot tuple in `variable_map` whose initial_state values sum to `target`.
                    _sum_m = re.search(
                        r"abs\(\s*\(?\s*((?:state|buf\.data|data)\[\d+\](?:\s*\+\s*(?:state|buf\.data|data)\[\d+\])+)\s*\)?\s*-\s*\(?\s*([0-9.eE+\-\s*/()]+?)\s*\)?\s*\)\s*<=?",
                        _ias,
                    )
                    if _sum_m:
                        try:
                            _sum_target = float(eval(_sum_m.group(2), {"__builtins__": {}}, {}))
                        except Exception:
                            _sum_target = None
                        _c_init_l = _c_item.get("initial_state") if isinstance(_c_item.get("initial_state"), list) else []
                        if _sum_target is not None and _sum_target > 0.0 and _c_init_l:
                            _k = len(_islots)
                            _has_unmapped = any(s not in _slot_to_vnames_c for s in _islots)
                            _cur_sum = sum(float(_c_init_l[s]) for s in _islots if s < len(_c_init_l))
                            _cur_ok = (not _has_unmapped) and abs(_cur_sum - _sum_target) <= 1e-3 * max(1.0, abs(_sum_target))
                            if not _cur_ok:
                                _mapped_sorted = sorted(s for s in _slot_to_vnames_c if s < len(_c_init_l))
                                _best_tuple = None
                                for _start_i in range(len(_mapped_sorted) - _k + 1):
                                    _cand = _mapped_sorted[_start_i:_start_i + _k]
                                    if _cand == _islots:
                                        continue
                                    _cand_sum = sum(float(_c_init_l[s]) for s in _cand)
                                    if abs(_cand_sum - _sum_target) <= 1e-3 * max(1.0, abs(_sum_target)):
                                        _best_tuple = _cand
                                        break
                                if _best_tuple is not None:
                                    _new_ias = _ias
                                    for _pos, (_old_s, _new_s) in enumerate(zip(_islots, _best_tuple)):
                                        _new_ias = re.sub(
                                            rf"\b(state|buf\.data|data)\[{_old_s}\]",
                                            rf"\1[__POOL_TMP_{_pos}__]",
                                            _new_ias,
                                        )
                                    for _pos, _new_s in enumerate(_best_tuple):
                                        _new_ias = _new_ias.replace(f"[__POOL_TMP_{_pos}__]", f"[{_new_s}]")
                                    _inv_item["assertion"] = _new_ias
                                    notes.append(f"rebound multi-slot conservation {_inm}: {_islots} -> {_best_tuple}")
                    continue
                if len(_islots) != 1:
                    continue
                _s_cur = _islots[0]
                _target_vs = vm.get(_inm)
                if _target_vs is None:
                    _target_vs = _sem_to_vslot_c.get(observable_name(_inm))
                if _target_vs is None:
                    _target_vs = _sem_to_vslot_c.get(_strip_qual_c(_inm))
                if _target_vs is None:
                    continue
                _target_vsi = int(_target_vs)
                if _target_vsi != _s_cur:
                    _cur_owners = _slot_to_vnames_c.get(_s_cur, [])
                    if _inm in vm or not _cur_owners or all(_strip_qual_c(o) != _strip_qual_c(_inm) for o in _cur_owners):
                        _inv_item["assertion"] = _canon_rewrite_slots(_ias, _s_cur, _target_vsi)
                        notes.append(f"rebound {_inm}: state[{_s_cur}] -> state[{_target_vsi}]")
    prior = copy.deepcopy(prior_contracts or [])
    all_contracts = prior + contracts
    _materialize_initial_state_bounds(all_contracts)
    order = sorted(range(len(all_contracts)),
                   key=lambda i: (_canon_stage_index(str(all_contracts[i].get("contract_id") or "")), i))

    reserved: set = set()
    for c in all_contracts:
        for inv in (c.get("invariants") or []):
            if isinstance(inv, dict):
                reserved.update(_canon_slots_in_text(str(inv.get("assertion") or "")))
    reserved.update(int(v) for v in vm.values())

    name_slot: Dict[str, int] = {}
    name_semantic: Dict[str, Tuple[str, str]] = {}
    semantic_slot: Dict[Tuple[str, str], int] = {}
    slot_owner: Dict[int, str] = {}
    slot_owner_contract: Dict[int, str] = {}
    slot_eq_target: Dict[int, float] = {}
    slot_sum_target: Dict[int, float] = {}
    slot_lower: Dict[int, float] = {}
    slot_upper: Dict[int, float] = {}
    slot_init_val: Dict[int, float] = {}
    init_reserved_owner: Dict[int, str] = {}
    init_reserved_val: Dict[int, float] = {}
    name_order: List[str] = []
    remaps: List[tuple[Dict[str, Any], Dict[str, Any], int, int]] = []

    def _extract_eq_target(expr: str) -> Optional[float]:
        m = re.search(
            r"abs\(\s*(?:state|buf\.data|data)\[\d+\]\s*-\s*\(?\s*([0-9.eE+\-\s*/()]+?)\s*\)?\s*\)\s*"
            r"(?:/\s*\(?\s*[0-9.eE+\-\s*/()]+?\s*\)?\s*)?<=?",
            expr or "",
        )
        if not m:
            return None
        try:
            val = eval(m.group(1), {"__builtins__": {}}, {})
            return float(val)
        except Exception:
            return None

    def _extract_sum_target(expr: str) -> Optional[float]:
        m = re.search(
            r"abs\(\s*\(?\s*(?:state|buf\.data|data)\[\d+\](?:\s*\+\s*(?:state|buf\.data|data)\[\d+\])+\s*\)?\s*-\s*\(?\s*([0-9.eE+\-\s*/()]+?)\s*\)?\s*\)\s*<=?",
            expr or "",
        )
        if not m:
            return None
        try:
            val = eval(m.group(1), {"__builtins__": {}}, {})
            return float(val)
        except Exception:
            return None

    def _extract_interval(expr: str, slot_idx: int) -> Tuple[Optional[float], Optional[float]]:
        lo: Optional[float] = None
        hi: Optional[float] = None
        pat_slot = rf"(?:state|buf\.data|data|trace_min|trace_max)\[{slot_idx}\]"
        for m in re.finditer(pat_slot + r"\s*(>=?|<=?)\s*\(?\s*([0-9.eE+\-\s*/()]+?)\s*\)?(?:\s*(?:$|&&|\|\||and\b|or\b|[<>]))", expr or ""):
            op, rhs = m.group(1), m.group(2)
            try:
                v = float(eval(rhs, {"__builtins__": {}}, {}))
            except Exception:
                continue
            if op in (">", ">="):
                lo = v if lo is None else max(lo, v)
            else:
                hi = v if hi is None else min(hi, v)
        return lo, hi

    def _has_cross_contract_clash(
        slot_idx: int,
        c_id: str,
        name: str,
        assertion: str,
        init_v: Optional[float],
    ) -> bool:
        prev_cid = slot_owner_contract.get(slot_idx)
        prev_owner = slot_owner.get(slot_idx)
        if prev_cid is None:
            init_cid = init_reserved_owner.get(slot_idx)
            init_rval = init_reserved_val.get(slot_idx)
            if init_cid is None or init_cid == c_id or init_rval is None:
                return False
            # Earlier contract initialized slot_idx to a non-zero value without
            # claiming it in an invariant; detect incompatible initial_state.
            # Final-time bounds on an evolving variable (e.g. initialized at 0.165,
            # relaxing into [0.115, 0.155]) do not clash when initial_state agrees
            # or when the slot is explicitly mapped in variable_map.
            if slot_idx in {int(v) for v in vm.values()} and (name not in vm or vm[name] == slot_idx):
                return False
            diff_init = (
                init_v is None
                or abs(float(init_v) - init_rval) > 1e-6 * max(1.0, abs(init_rval), abs(float(init_v)))
            )
            return bool(diff_init)
        if prev_cid == c_id or prev_owner == name:
            return False
        prev_iv = slot_init_val.get(slot_idx)
        same_init = (
            init_v is None
            or prev_iv is None
            or abs(float(init_v) - float(prev_iv)) <= 1e-6 * max(1.0, abs(float(prev_iv)), abs(float(init_v)))
        )
        explicit_vm_slots = {int(v) for v in vm.values()}
        if (
            slot_idx in explicit_vm_slots
            and (name not in vm or vm[name] == slot_idx)
            and (prev_owner not in vm or vm[prev_owner] == slot_idx)
        ):
            eq_t_cur = _extract_eq_target(assertion)
            prev_t_cur = slot_eq_target.get(slot_idx)
            sum_t_cur = _extract_sum_target(assertion)
            prev_sum_cur = slot_sum_target.get(slot_idx)
            eq_clash = (
                eq_t_cur is not None
                and prev_t_cur is not None
                and abs(prev_t_cur - eq_t_cur) > 1e-3 * max(1.0, abs(prev_t_cur), abs(eq_t_cur))
            )
            sum_clash = (
                sum_t_cur is not None
                and prev_sum_cur is not None
                and abs(prev_sum_cur - sum_t_cur) > 1e-3 * max(1.0, abs(prev_sum_cur), abs(sum_t_cur))
            )
            lo_cur, hi_cur = _extract_interval(assertion, slot_idx)
            prev_lo_cur = slot_lower.get(slot_idx)
            prev_hi_cur = slot_upper.get(slot_idx)
            target_bound_clash = (
                (eq_t_cur is not None and prev_hi_cur is not None and eq_t_cur > prev_hi_cur + 1e-6)
                or (eq_t_cur is not None and prev_lo_cur is not None and eq_t_cur < prev_lo_cur - 1e-6)
                or (prev_t_cur is not None and hi_cur is not None and prev_t_cur > hi_cur + 1e-6)
                or (prev_t_cur is not None and lo_cur is not None and prev_t_cur < lo_cur - 1e-6)
            )
            if not eq_clash and not sum_clash and not target_bound_clash:
                # Both contracts inspect the exact same model slot declared in `variable_map`
                # under compatible equality/sum targets (even if testing different initial states
                # or sub-intervals). Never clone it to an uncomputed phantom slot outside `variable_map`.
                return False
        if preserve_mapped_slots and slot_idx in set(vm.values()):
            return False
        prev_t = slot_eq_target.get(slot_idx)
        prev_sum = slot_sum_target.get(slot_idx)
        prev_lo = slot_lower.get(slot_idx)
        prev_hi = slot_upper.get(slot_idx)
        eq_target = _extract_eq_target(assertion)
        sum_target = _extract_sum_target(assertion)
        lo_new, hi_new = _extract_interval(assertion, slot_idx)
        if slot_idx not in (0, 1, 2, 3):
            # If the invariant references asingle slot (and neither is part of a multi-slot sum
            # conservation law), distinct observable names in different contracts must not share a slot
            # unless they map to the same explicit variable_map slot or share the same semantic stem.
            inv_slots = set(_canon_slots_in_text(assertion))
            is_multi_slot_inv = len(inv_slots) > 1
            same_vm_slot = (vm.get(name) == slot_idx and vm.get(prev_owner or "") == slot_idx)
            same_stem = (
                bool(_canon_semantic_stem(name))
                and _canon_semantic_stem(name) == _canon_semantic_stem(prev_owner or "")
            )
            if not is_multi_slot_inv and prev_sum is None and sum_target is None and not same_vm_slot and not same_stem:
                return True
            # Allow multi-contract checks on the SAME slot when initial values,
            # sum targets, equality targets, and intervals are mutually compatible.
            init_clash_hi = (init_v is not None and prev_iv is not None and abs(init_v - prev_iv) > 1e-6 * max(1.0, abs(init_v), abs(prev_iv)))
            target_clash_hi = (
                (eq_target is not None and prev_t is not None and abs(prev_t - eq_target) > 1e-3 * max(1.0, abs(prev_t), abs(eq_target)))
                or (eq_target is not None and prev_hi is not None and eq_target > prev_hi + 1e-6)
                or (eq_target is not None and prev_lo is not None and eq_target < prev_lo - 1e-6)
                or (prev_t is not None and hi_new is not None and prev_t > hi_new + 1e-6)
                or (prev_t is not None and lo_new is not None and prev_t < lo_new - 1e-6)
            )
            sum_clash_hi = (
                sum_target is not None and prev_sum is not None and abs(sum_target - prev_sum) > 1e-3 * max(1.0, abs(sum_target), abs(prev_sum))
            )
            interval_clash_hi = (
                (lo_new is not None and prev_hi is not None and lo_new > prev_hi + 1e-9)
                or (hi_new is not None and prev_lo is not None and hi_new < prev_lo - 1e-9)
                or (prev_hi is not None and hi_new is not None and (prev_hi <= 1e-6) != (hi_new <= 1e-6))
            )
            return bool(init_clash_hi or target_clash_hi or sum_clash_hi or interval_clash_hi)
        init_clash = (init_v is not None and prev_iv is not None and abs(init_v - prev_iv) > 1e-6)
        target_clash = (
            (eq_target is not None and (prev_t is None or abs(prev_t - eq_target) > 1e-3))
            or (prev_t is not None and eq_target is None and (
                (hi_new is not None and hi_new < prev_t - 1e-6)
                or (lo_new is not None and lo_new > prev_t + 1e-6)
                or True
            ))
        )
        partition_clash = (prev_sum is not None and sum_target is None) or (
            sum_target is not None and prev_sum is not None and abs(sum_target - prev_sum) > 1e-3
        ) or (sum_target is not None and prev_sum is None)
        interval_clash = (
            (lo_new is not None and prev_hi is not None and lo_new > prev_hi + 1e-9)
            or (hi_new is not None and prev_lo is not None and hi_new < prev_lo - 1e-9)
            or (prev_hi is not None and hi_new is not None and (prev_hi <= 1e-6) != (hi_new <= 1e-6))
        )
        return bool(init_clash or target_clash or partition_clash or interval_clash)

    def _record_slot_claim(slot_idx: int, c_id: str, name: str, assertion: str, init_v: Optional[float]) -> None:
        slot_owner.setdefault(slot_idx, name)
        slot_owner_contract.setdefault(slot_idx, c_id)
        eq_t = _extract_eq_target(assertion)
        sum_t = _extract_sum_target(assertion)
        lo, hi = _extract_interval(assertion, slot_idx)
        if eq_t is not None:
            slot_eq_target.setdefault(slot_idx, eq_t)
        if sum_t is not None:
            slot_sum_target.setdefault(slot_idx, sum_t)
        if lo is not None:
            slot_lower[slot_idx] = max(slot_lower.get(slot_idx, lo), lo)
        if hi is not None:
            slot_upper[slot_idx] = min(slot_upper.get(slot_idx, hi), hi)
        if init_v is not None:
            slot_init_val.setdefault(slot_idx, init_v)

    def assign_for_inv(c: Dict[str, Any], inv: Dict[str, Any], record: bool) -> None:
        c_id = str(c.get("contract_id") or "")
        name = str(inv.get("variable_name") or "").strip()
        if not _canon_observable_name(name, vm):
            return
        assertion = str(inv.get("assertion") or "")
        slots = _canon_slots_in_text(assertion)
        distinct_slots = sorted(set(slots))
        c_init = c.get("initial_state") if isinstance(c.get("initial_state"), list) else []
        if slots and len(distinct_slots) != 1:
            for s in distinct_slots:
                init_s = float(c_init[s]) if s < len(c_init) else None
                if _has_cross_contract_clash(s, c_id, name, assertion, init_s):
                    alloc_used = set(name_slot.values()) | set(semantic_slot.values()) | set(slot_owner)
                    new_s = _canon_alloc_slot(alloc_used, reserved)
                    reserved.add(new_s)
                    if record:
                        remaps.append((c, inv, s, new_s))
                    _record_slot_claim(new_s, c_id, name, _canon_rewrite_slots(assertion, s, new_s), init_s)
                    notes.append(f"canonical multi-slot conflict: {name} state[{s}] -> state[{new_s}]")
                else:
                    _record_slot_claim(s, c_id, name, assertion, init_s)
            return
        desired = distinct_slots[0] if distinct_slots else vm.get(name)
        if desired is None:
            return
        desired = int(desired)
        init_v = float(c_init[desired]) if desired < len(c_init) else None
        if desired in (0, 1, 2, 3):
            if _has_cross_contract_clash(desired, c_id, name, assertion, init_v):
                pass  # Clashing cross-contract usage on 0..3 -> remap below
            else:
                _record_slot_claim(desired, c_id, name, assertion, init_v)
                return
        key = _canon_semantic_key(name, assertion)
        name_semantic.setdefault(name, key)
        canonical = semantic_slot.get(key)
        if canonical is not None and _has_cross_contract_clash(int(canonical), c_id, name, assertion, init_v):
            canonical = None
        if canonical is None:
            if _has_cross_contract_clash(desired, c_id, name, assertion, init_v):
                alloc_used = set(name_slot.values()) | set(semantic_slot.values()) | set(slot_owner)
                canonical = _canon_alloc_slot(alloc_used, reserved)
                reserved.add(int(canonical))
                notes.append(f"canonical slot conflict: {name} state[{desired}] -> state[{canonical}]")
            else:
                canonical = desired
            semantic_slot[key] = canonical
            _record_slot_claim(int(canonical), c_id, name, assertion, init_v)
            if name not in name_order:
                name_order.append(name)
        name_slot[name] = int(canonical)
        if desired != canonical and record:
            remaps.append((c, inv, desired, canonical))

    for idx in order:
        c = all_contracts[idx]
        is_prior = idx < len(prior)
        for inv in (c.get("invariants") or []):
            if isinstance(inv, dict):
                assign_for_inv(c, inv, record=not is_prior)
        c_id_done = str(c.get("contract_id") or "")
        for s_i, val in enumerate(c.get("initial_state") or []):
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if fval != 0.0:
                init_reserved_owner.setdefault(s_i, c_id_done)
                init_reserved_val.setdefault(s_i, fval)
                slot_init_val.setdefault(s_i, fval)
                reserved.add(s_i)

    for c, inv, old, new in remaps:
        if new in (0, 1, 2, 3):
            continue
        inv["assertion"] = _canon_rewrite_slots(str(inv.get("assertion") or ""), old, new)
        c_init = c.get("initial_state")
        if isinstance(c_init, list):
            was_missing = new >= len(c_init)
            while len(c_init) <= new:
                c_init.append(0.0)
            if old < len(c_init) and (was_missing or c_init[new] == 0.0) and c_init[old] != 0.0:
                c_init[new] = c_init[old]
        notes.append(f"canonicalized {inv.get('variable_name')}: state[{old}] -> state[{new}]")
    # Restore earlier stage's non-zero value at remapped-away `old` slots so Stage k
    # regression against c1..c_{k-1} still sees Stage 1's initial parameter/state.
    for c, _inv, old, new in remaps:
        if new in (0, 1, 2, 3):
            continue
        c_init = c.get("initial_state")
        if isinstance(c_init, list) and old < len(c_init) and old in init_reserved_val:
            c_init[old] = init_reserved_val[old]

    # Build a unified initial_state layout across c1..cN so every contract carries
    # valid initial values for both Stage 1's slots and Stage k's de-collided slots.
    ordered_contracts = [all_contracts[idx] for idx in order if isinstance(all_contracts[idx], dict)]
    orig_lens = {id(c_ord): len(c_ord.get("initial_state") or []) for c_ord in ordered_contracts}
    if any(isinstance(c.get("initial_state"), list) for c in ordered_contracts):
        unified_len = max((len(c.get("initial_state") or []) for c in ordered_contracts), default=0)
        if unified_len > 0:
            merged_init: List[float] = [0.0] * unified_len
            for s_i in range(unified_len):
                # Earliest contract defining a non-zero initial_state[s_i] sets canonical value.
                for c_ord in ordered_contracts:
                    ci = c_ord.get("initial_state")
                    if isinstance(ci, list) and s_i < len(ci):
                        try:
                            fv = float(ci[s_i])
                        except (TypeError, ValueError):
                            fv = 0.0
                        if fv != 0.0:
                            merged_init[s_i] = fv
                            break
            from .slot_schema import ROLE_CONTROL_PARAM, ROLE_TIME_STEP, classify_slot_roles
            slot_roles = classify_slot_roles(vm) if vm else {}
            for c_ord in ordered_contracts:
                ci = c_ord.get("initial_state")
                if not isinstance(ci, list):
                    continue
                orig_l = orig_lens.get(id(c_ord), 0)
                has_iv = isinstance(c_ord.get("initial_values"), dict) and bool(c_ord.get("initial_values"))
                while len(ci) < unified_len:
                    ci.append(0.0)
                start_idx = orig_l if (has_iv or bool(vm)) else 0
                for s_i in range(start_idx, unified_len):
                    if ci[s_i] == 0.0 and merged_init[s_i] != 0.0:
                        if slot_roles.get(s_i) in (ROLE_CONTROL_PARAM, ROLE_TIME_STEP):
                            continue
                        ci[s_i] = merged_init[s_i]

    # Canonicalize explicit variable_map entries against contract-owned names.
    assigned_names = set(name_slot)
    assigned_slots = set(name_slot.values()) | set(semantic_slot.values())
    new_vm: Dict[str, int] = {}
    for name, idx in vm.items():
        if int(idx) in (0, 1, 2, 3) and name not in name_slot:
            new_vm[name] = int(idx)
            continue
        canonical = name_slot.get(name)
        if canonical is None:
            key = name_semantic.get(name)
            if key is not None:
                canonical = semantic_slot.get(key)
        if canonical is None:
            canonical = int(idx)
            slot_owner.setdefault(int(canonical), name)
            assigned_names.add(name)
        canonical = int(canonical)
        name_slot.setdefault(name, canonical)
        new_vm[name] = canonical
        assigned_slots.add(canonical)
    vm = new_vm

    # Cross-stage physical-concept slot unification runs LAST so the same
    # conserved/monotone quantity measured under different stage names shares
    # the first stage's slot (e.g. ``unified_energy_drift`` state[16] -> the
    # ``energy_drift_max`` slot of the earlier stage) without the conflict
    # resolver above re-allocating it to a fresh slot.
    concept_remap, concept_notes = concept_bucket_slot_map(all_contracts)
    if concept_remap:
        apply_concept_bucket_remap(all_contracts, concept_remap)
        vm = {k: concept_remap.get(v, v) for k, v in vm.items()}
        notes.extend(concept_notes)

    if not vm:
        for name in name_order:
            if _canon_observable_name(name, {}):
                vm[name] = int(name_slot[name])
                if len(vm) >= 64:
                    break
    if not vm:
        # Last-resort deterministic map so reports never see an empty map.
        first: Dict[str, int] = {}
        for c in contracts:
            for inv in (c.get("invariants") or []):
                if not isinstance(inv, dict):
                    continue
                nm = str(inv.get("variable_name") or "").strip()
                distinct_slots = sorted(set(_canon_slots_in_text(str(inv.get("assertion") or ""))))
                if nm and len(distinct_slots) == 1 and nm not in first:
                    first[nm] = distinct_slots[0]
        vm = dict(list(first.items())[:64]) or {f"state_{i}": i for i in range(4)}

    # Compile initial_state deterministically from named initial_values when present,
    # and repair single-slot extrema invariants accidentally bound to instantaneous slots.
    try:
        from .slot_schema import compile_initial_state_from_values, observable_name
        stem_to_slot: Dict[str, int] = {}
        extrema_stem_to_slot: Dict[str, int] = {}
        for vname, vslot in vm.items():
            sem = observable_name(vname)
            stem_to_slot.setdefault(sem, int(vslot))
            if sem.startswith(("peak_", "max_", "min_")):
                extrema_stem_to_slot.setdefault(sem, int(vslot))
        for c in contracts:
            if not isinstance(c, dict):
                continue
            iv = c.get("initial_values")
            if isinstance(iv, dict) and iv and vm:
                c["initial_state"] = compile_initial_state_from_values(
                    vm, iv, c.get("initial_state")
                )
            for inv in (c.get("invariants") or []):
                if not isinstance(inv, dict):
                    continue
                iname = observable_name(str(inv.get("variable_name") or ""))
                if not iname.startswith(("peak_", "max_", "min_")):
                    continue
                base_stem = iname.split("_", 1)[1]
                # Find matching extrema slot in variable_map (exact stem or prefix match)
                target_ext_slot = extrema_stem_to_slot.get(iname)
                if target_ext_slot is None:
                    prefix = iname.split("_", 1)[0] + "_"
                    base_tok = base_stem.split("_")[0]
                    for esem, eslot in extrema_stem_to_slot.items():
                        if esem.startswith(prefix) and (
                            esem.split("_", 1)[1] == base_stem
                            or esem.split("_", 1)[1].startswith(base_tok)
                        ):
                            target_ext_slot = eslot
                            break
                if target_ext_slot is None:
                    continue
                slots_in_assert = sorted(set(_canon_slots_in_text(str(inv.get("assertion") or ""))))
                if len(slots_in_assert) == 1 and slots_in_assert[0] != target_ext_slot:
                    bad_slot = slots_in_assert[0]
                    inv["assertion"] = re.sub(
                        rf"\bstate\s*\[\s*{bad_slot}\s*\]",
                        f"state[{target_ext_slot}]",
                        str(inv.get("assertion") or ""),
                    )
                    notes.append(
                        f"rebound extrema invariant '{inv.get('variable_name')}' "
                        f"from state[{bad_slot}] to extrema slot state[{target_ext_slot}]"
                    )
    except Exception:  # noqa: BLE001
        pass
    return contracts, vm, notes


def canonical_slot_map_text(contracts: List[Dict[str, Any]],
                            variable_map: Optional[Dict[str, int]] = None) -> str:
    """Human-readable Canonical Global State Slot Table for synthesis prompts."""
    canon_contracts, vm, _ = canonicalize_contract_dicts(contracts, variable_map)
    slot_info: Dict[int, Dict[str, Any]] = {}
    for name, idx in sorted(vm.items()):
        slot_info.setdefault(int(idx), {"names": [], "init": None, "bounds": []})
        if name not in slot_info[int(idx)]["names"]:
            slot_info[int(idx)]["names"].append(name)
    for c in canon_contracts:
        init = c.get("initial_state") if isinstance(c.get("initial_state"), list) else []
        for inv in (c.get("invariants") or []):
            if not isinstance(inv, dict):
                continue
            nm = str(inv.get("variable_name") or "").strip()
            expr = str(inv.get("assertion") or "").strip()
            for s in sorted(set(_canon_slots_in_text(expr))):
                entry = slot_info.setdefault(s, {"names": [], "init": None, "bounds": []})
                if nm and nm not in entry["names"]:
                    entry["names"].append(nm)
                if entry["init"] is None and s < len(init):
                    entry["init"] = init[s]
                if expr and expr not in entry["bounds"]:
                    entry["bounds"].append(expr)
    if not slot_info:
        return "(no named observables)"
    lines = []
    for idx in sorted(slot_info):
        info = slot_info[idx]
        nm = "/".join(info["names"][:2]) or f"slot_{idx}"
        iv = info["init"] if info["init"] is not None else 0.0
        bnds = "; ".join(info["bounds"][:2]) or "unconstrained"
        lines.append(f"- `state[{idx}]`: `{nm}` (initial={iv}, invariants: {bnds})")
    return "\n".join(lines)


def normalize_blueprint_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce minor LLM schema drift instead of discarding the whole response.

    Drops unknown keys (the pydantic models forbid extras), coerces int/float
    and str/list mismatches on optional fields, and repairs contract shapes.
    Required fields (blueprint_id, title, hypothesis, code_template, ...) are
    left for pydantic validation to reject with a precise error.
    """
    data = _sanitize_raw_blueprint_dict(data)
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Any] = {k: data[k] for k in _BLUEPRINT_FIELDS if k in data}
    for key in ("blueprint_id", "title", "falsifiable_hypothesis", "code_template"):
        if key in out and out[key] is not None:
            out[key] = str(out[key])
    if "code_template" in out and out["code_template"] is None:
        out["code_template"] = "def simulation_step(buf, dt):\n    pass\n"
    for key in ("search_queries", "stage_objectives"):
        if key in out:
            val = out[key]
            if isinstance(val, str):
                val = [val]
            if isinstance(val, list):
                out[key] = [str(x)[:300] for x in val if str(x).strip()][:12]
            else:
                out.pop(key, None)
    for key, lo, hi in (("target_research_stages", 1, 10), ("max_feedback_rounds", 1, 10)):
        if key in out:
            try:
                out[key] = max(lo, min(hi, int(out[key])))
            except (TypeError, ValueError):
                out.pop(key, None)
    if "current_stage" in out:
        try:
            out["current_stage"] = max(1, min(10, int(out["current_stage"])))
        except (TypeError, ValueError):
            out.pop("current_stage", None)
    contracts = out.get("contracts")
    if contracts is None:
        for alt_k in ("verification_contracts", "stage_contracts"):
            if alt_k in data and data[alt_k] is not None:
                contracts = data[alt_k]
                break
    if isinstance(contracts, dict):
        contracts = [contracts]
    if isinstance(contracts, list):
        cleaned = [c for c in (_coerce_contract(raw, i)
                               for i, raw in enumerate(contracts)) if c]
        if cleaned:
            out["contracts"] = cleaned[:8]
        else:
            out.pop("contracts", None)
    variable_map = out.get("variable_map")
    if isinstance(variable_map, dict):
        vm: Dict[str, int] = {}
        for name, idx in list(variable_map.items())[:4096]:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if i >= 0 and str(name).strip():
                vm[str(name).strip()[:120]] = i
        out["variable_map"] = vm
    elif variable_map is not None:
        out.pop("variable_map", None)
    raw_roles = out.get("slot_roles")
    if isinstance(raw_roles, dict):
        from .slot_schema import _CONTROL_PARAM_NAME_RE, from_blueprint_role, to_blueprint_role

        _rate_diag_re = re.compile(
            r"(?:_(?:rate|flux|load|slope|current)(?:_[a-z0-9]+)*$|^d[A-Z0-9_]+_slope)",
            re.I,
        )
        vm_slots = out.get("variable_map") or {}
        contracts_list = out.get("contracts") or []
        roles_clean: Dict[str, str] = {}
        for k, v in list(raw_roles.items())[:4096]:
            ks = str(k).strip()[:120]
            if not ks:
                continue
            b_role = to_blueprint_role(from_blueprint_role(str(v).strip()))
            if (
                b_role == "read_only_parameter"
                and _rate_diag_re.search(ks)
                and not _CONTROL_PARAM_NAME_RE.match(ks)
            ):
                # Check if initial values for this slot are zero/unset across contracts
                slot_idx = vm_slots.get(ks)
                inits: List[float] = []
                for c in contracts_list:
                    if not isinstance(c, dict):
                        continue
                    iv = c.get("initial_values") or {}
                    if isinstance(iv, dict) and ks in iv:
                        try:
                            inits.append(float(iv[ks]))
                            continue
                        except (TypeError, ValueError):
                            pass
                    init_list = c.get("initial_state") or []
                    if slot_idx is not None and 0 <= int(slot_idx) < len(init_list):
                        try:
                            inits.append(float(init_list[int(slot_idx)]))
                        except (TypeError, ValueError):
                            pass
                if not inits or all(abs(val) <= 1e-15 for val in inits):
                    b_role = "diagnostic"
            roles_clean[ks] = b_role
        out["slot_roles"] = roles_clean
    elif raw_roles is not None:
        out.pop("slot_roles", None)
    if isinstance(out.get("contracts"), list):
        if isinstance(out.get("code_template"), str) and out["code_template"].strip():
            _vm_pre = out.get("variable_map") if isinstance(out.get("variable_map"), dict) else {}
            if _vm_pre:
                from .slot_schema import _analyze_code_slot_flow, observable_name
                _qual_re = re.compile(
                    r"_(?:plausible|target|envelope|ceiling|floor|bounds?|window|final|baseline|initial|check|limit|corridor|steady|range|valid|ok|horizon|s|v|mv|mm|um|mol|m3|k|m|hz|db|s_m2|mmhg|units|frac|j_m2)$",
                    re.I,
                )
                def _strip_qual(s_name: str) -> str:
                    base = observable_name(s_name)
                    prev = None
                    while base and base != prev:
                        prev = base
                        base = _qual_re.sub("", base)
                    return base or observable_name(s_name)

                _flow_pre = _analyze_code_slot_flow(out["code_template"])
                _written_pre = _flow_pre.get("all_writes", _flow_pre["writes"])
                _slot_to_vnames: Dict[int, List[str]] = {}
                _sem_to_vslot: Dict[str, int] = {}
                for _vn, _vs in _vm_pre.items():
                    try:
                        _vsi = int(_vs)
                    except (TypeError, ValueError):
                        continue
                    _slot_to_vnames.setdefault(_vsi, []).append(str(_vn))
                    _sem_to_vslot.setdefault(observable_name(str(_vn)), _vsi)
                    _sem_to_vslot.setdefault(_strip_qual(str(_vn)), _vsi)
                for _c_pre in out["contracts"]:
                    if not isinstance(_c_pre, dict):
                        continue
                    for _inv_pre in (_c_pre.get("invariants") or []):
                        if not isinstance(_inv_pre, dict):
                            continue
                        _inm = str(_inv_pre.get("variable_name") or "").strip()
                        _ias = str(_inv_pre.get("assertion") or "")
                        _islots = sorted(set(_canon_slots_in_text(_ias)))
                        if len(_islots) != 1:
                            continue
                        _s_cur = _islots[0]
                        _target_vs = _vm_pre.get(_inm)
                        if _target_vs is None:
                            _target_vs = _sem_to_vslot.get(observable_name(_inm))
                        if _target_vs is None:
                            _target_vs = _sem_to_vslot.get(_strip_qual(_inm))
                        if _target_vs is None:
                            continue
                        _target_vsi = int(_target_vs)
                        if _target_vsi != _s_cur and _target_vsi in _written_pre:
                            _cur_owners = _slot_to_vnames.get(_s_cur, [])
                            if (_cur_owners and all(_strip_qual(o) != _strip_qual(_inm) for o in _cur_owners)) or _s_cur not in _written_pre:
                                _inv_pre["assertion"] = _canon_rewrite_slots(_ias, _s_cur, _target_vsi)
            from ..healing.deterministic_ast import resolve_slot_collisions
            new_code, new_contracts, _ = resolve_slot_collisions(
                out["code_template"], out["contracts"], out["contracts"]
            )
            out["code_template"] = new_code
            out["contracts"] = new_contracts
        out["contracts"], out["variable_map"], _ = canonicalize_contract_dicts(
            out["contracts"], out.get("variable_map"), preserve_mapped_slots=bool(out.get("typed_slots")))
        # If the LLM emitted >= 3 contracts all tagged stage 1 (e.g. stage1_...
        # or c1_...) while requesting multi-stage verification, partition them
        # across c1_, c2_, c3_ so each research stage has dedicated contracts.
        c_stages = {
            _canon_stage_index(str(c.get("contract_id") or ""))
            for c in out["contracts"] if isinstance(c, dict)
        }
        dict_contracts = [c for c in out["contracts"] if isinstance(c, dict)]
        if (
            len(dict_contracts) >= 3
            and c_stages == {1}
            and int(out.get("target_research_stages") or 1) >= 2
        ):
            closed_re = re.compile(r"(closed|revers|chiasmus|gamma_?0)", re.IGNORECASE)
            stage3_idxs = [
                idx for idx, c in enumerate(dict_contracts)
                if closed_re.search(str(c.get("contract_id") or ""))
            ]
            if not stage3_idxs or len(stage3_idxs) == len(dict_contracts):
                stage3_idxs = [len(dict_contracts) - 1]
            rem_idxs = [idx for idx in range(len(dict_contracts)) if idx not in stage3_idxs]
            split = max(1, len(rem_idxs) // 2)
            stage1_idxs = set(rem_idxs[:split])
            stage2_idxs = set(rem_idxs[split:])
            for idx, c in enumerate(dict_contracts):
                cid = str(c.get("contract_id") or f"contract_{idx + 1}")
                stem = _CANON_STAGE_RE.sub("", cid) or cid
                s_num = 1 if idx in stage1_idxs else (2 if idx in stage2_idxs else 3)
                c["contract_id"] = f"c{s_num}_{stem}"
            c_stages = {1, 2, 3}
        # If contracts carry explicit c3_/stage3_ prefixes while the LLM left
        # target_research_stages at 2, align target_research_stages to 3 so
        # Stage 3 contracts are executed rather than orphaned.
        if max(c_stages, default=1) >= 3 and int(out.get("target_research_stages") or 1) == 2:
            if {1, 2}.issubset(c_stages):
                out["target_research_stages"] = 3
                objs = list(out.get("stage_objectives") or [])
                while len(objs) < 3:
                    s_idx = len(objs) + 1
                    c_ids = [
                        str(c.get("contract_id")) for c in out["contracts"]
                        if isinstance(c, dict) and _canon_stage_index(str(c.get("contract_id") or "")) >= s_idx
                    ]
                    objs.append(f"Stage {s_idx}: Verify contracts {', '.join(c_ids)}")
                out["stage_objectives"] = objs
    if isinstance(out.get("typed_slots"), list):
        _slot_allowed = {
            "symbol", "slot_idx", "si_dim", "role", "semantic_tag", "unit_scale",
            "is_falsification_target", "stochastic_role", "value", "prompt_symbol",
            "si_unit", "initial_value", "bounds",
        }
        cleaned_slots = []
        for s in out["typed_slots"]:
            if isinstance(s, dict):
                s_clean = {k: v for k, v in s.items() if k in _slot_allowed}
                if "si_unit" not in s_clean and isinstance(s.get("unit"), str):
                    s_clean["si_unit"] = s["unit"]
                cleaned_slots.append(s_clean)
            else:
                cleaned_slots.append(s)
        out["typed_slots"] = cleaned_slots
    if "assumptions_ledger" in out:
        from ..schemas.research import coerce_assumption_entry
        raw_ledger = out["assumptions_ledger"]
        if isinstance(raw_ledger, list):
            out["assumptions_ledger"] = [
                c for c in (coerce_assumption_entry(item) for item in raw_ledger) if c is not None
            ]
        elif isinstance(raw_ledger, dict):
            coerced = coerce_assumption_entry(raw_ledger)
            out["assumptions_ledger"] = [coerced] if coerced else []
        else:
            out.pop("assumptions_ledger", None)
    return out


_HORIZON_FACTORS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
    "wk": 604800.0, "week": 604800.0, "weeks": 604800.0,
    "yr": 3.15576e7, "yrs": 3.15576e7, "year": 3.15576e7, "years": 3.15576e7,
}
_HORIZON_RE = re.compile(
    r"(?:over|for|across|during|until|horizon(?:\s+of)?|duration(?:\s+of)?|simulate)\s+"
    r"(?:a\s+|the\s+)?(\d+(?:\.\d+)?)[\s-]*"
    r"(s|secs?|seconds?|minutes?|mins?|hours?|hrs?|days?|weeks?|years?|yrs?)\b",
    re.I,
)
_UNIT_DIVISOR_HINTS = {
    60.0: ("60.0", "/ 60", "/60"),
    3600.0: ("3600",),
    86400.0: ("86400",),
    604800.0: ("604800",),
    3.15576e7: ("3.15576e7", "31557600", "31536000", "3.1536e7", "86400"),
}


def _reconcile_time_horizon(
    bp_dict: Dict[str, Any],
    question: str,
    evidence: Any = None,
) -> Dict[str, Any]:
    """Ensure ``dt * steps`` uses the same time unit as the kernel's rate constants."""
    if not isinstance(bp_dict, dict) or not question:
        return bp_dict
    matches = _HORIZON_RE.findall(question)
    if not matches:
        return bp_dict
    horizons: List[Tuple[float, float]] = []
    for val_s, unit_s in matches:
        try:
            raw_v = float(val_s)
        except ValueError:
            continue
        fac = _HORIZON_FACTORS.get(unit_s.lower())
        if fac and raw_v > 0.0:
            horizons.append((raw_v, fac))
    if not horizons:
        return bp_dict
    # Use the longest physical horizon specified in the prompt.
    raw_val, unit_factor = max(horizons, key=lambda pair: pair[0] * pair[1])
    code = str(bp_dict.get("code_template") or "")
    has_si_rate_conv = any(tok in code for tok in _UNIT_DIVISOR_HINTS.get(unit_factor, ()))
    if not has_si_rate_conv and evidence is not None:
        for p in (getattr(evidence, "parameters", None) or {}).values():
            if getattr(p, "si_unit", "") in ("1/s", "mol/(m^3*s)", "kg/s") and getattr(p, "raw_unit", "") != getattr(p, "si_unit", ""):
                has_si_rate_conv = True
                break
    out = copy.deepcopy(bp_dict)
    for c in out.get("contracts") or []:
        if not isinstance(c, dict):
            continue
        try:
            dt = float(c.get("dt", 0.0))
            steps = int(c.get("steps", 0))
        except (TypeError, ValueError):
            continue
        if dt <= 0.0 or steps <= 0:
            continue
        sim_t = dt * steps
        # Only rescale when dt * steps matches the raw (unconverted) prompt horizon
        # while the kernel itself uses SI 1/s rates (off by ~unit_factor).
        if 0.5 * raw_val <= sim_t <= 2.0 * raw_val and unit_factor >= 60.0:
            c["dt"] = min(1e8, dt * unit_factor)
        target = raw_val * unit_factor
        dt = float(c['dt'])
        steps = max(1, int(math.ceil(target / dt)))
        if steps > 100000:
            raise ValueError('SCIENTIFIC_INTEGRITY: requested horizon exceeds the step budget at the chosen dt')
        c['steps'] = steps
        c['dt'] = target / steps
    return out


def parse_blueprint(payload: Dict[str, Any]) -> ResearchBlueprint:
    from . import meaningfulness as m
    from .formal_evidence import attach_formal_evidence
    from .ihcg_derivation import derive_ihcg_evidence
    blueprint = ResearchBlueprint.model_validate(normalize_blueprint_payload(payload))
    if blueprint.typed_slots:
        for idx, raw_slot in enumerate(blueprint.typed_slots):
            slot_dict = raw_slot.model_dump(exclude_none=True) if hasattr(raw_slot, "model_dump") else dict(raw_slot)
            check = {k: v for k, v in slot_dict.items() if k not in {"value", "prompt_symbol", "si_unit", "initial_value", "bounds"}}
            if check.get("role") == "TIME_STEP":
                check["role"] = "DYNAMIC_STATE"
            if check.get("slot_idx") is None:
                sym = check.get("symbol")
                resolved_idx = blueprint.variable_map.get(sym)
                if resolved_idx is None:
                    raise ValueError(
                        f"typed_slots[{idx}] symbol {sym!r} has no slot_idx and is missing from "
                        f"variable_map (keys={sorted(blueprint.variable_map)})"
                    )
                check["slot_idx"] = resolved_idx
            spec_obj = m.SlotSpec.model_validate_json(json.dumps(check, allow_nan=False))
            if spec_obj.stochastic_role and spec_obj.role != "CONTROL_PARAM":
                raise ValueError(
                    f"typed_slots[{idx}] ({spec_obj.symbol!r}): stochastic_role={spec_obj.stochastic_role!r} "
                    f"requires role='CONTROL_PARAM' (got role={spec_obj.role!r})"
                )
        mapped_indices = {v for v in blueprint.variable_map.values() if isinstance(v, int)}
        if mapped_indices:
            for contract in blueprint.contracts:
                for inv in contract.invariants:
                    for idx_str in re.findall(r"(?:state|buf\.data|data)\[(\d+)\]", str(inv.assertion or "")):
                        if int(idx_str) not in mapped_indices:
                            raise ValueError(
                                f"contract {contract.contract_id} invariant references state[{idx_str}] "
                                f"which is not in variable_map (mapped_indices={sorted(mapped_indices)})"
                            )
    if blueprint.ihcg_evidence is not None:
        pack = derive_ihcg_evidence(blueprint.research_prompt or blueprint.falsifiable_hypothesis, blueprint.model_dump(), blueprint.ihcg_evidence)
        blueprint = blueprint.model_copy(update={'ihcg_evidence': pack})
    return attach_formal_evidence(blueprint)


def _parse_planned_blueprint(payload, question, *, require_evidence=False):
    from . import meaningfulness as m
    blueprint = parse_blueprint({**payload, "research_prompt": question})
    pack = blueprint.ihcg_evidence
    if pack is None:
        if require_evidence:
            raise ValueError("FORMAL_EVIDENCE_MISSING: planned blueprint requires compiled ihcg_evidence")
    else:
        graph = m.IhcgCompiler().compile_from_prompt(question, **{k: v for k, v in pack.items() if k not in {"extensions", "execution", "notes"}})
        for extension in pack.get("extensions", ()):
            graph = m.IhcgCompiler().extend_stage(graph, question, extension["evidence"], extension["domain"], extension["archetype"])
        from .epistemic import compile_blueprint_ihcg
        compile_blueprint_ihcg(blueprint)
    return blueprint


def _fill(template: str, **values: Any) -> str:
    # Prompts embed literal JSON braces; substitute named placeholders only.
    out = template.replace("{{", "\x00").replace("}}", "\x01")
    for key, val in values.items():
        out = out.replace("{" + key + "}", str(val))
    return out.replace("\x00", "{").replace("\x01", "}")


def _blueprint_tools_context(fallback):
    if fallback: return TOOLS_CONTEXT
    head, rest = TOOLS_CONTEXT.split('## Formal evidence at ingest', 1)
    return head + '- Sandbox_Execute' + rest.split('- Sandbox_Execute', 1)[1].split('## Worked formal-evidence example', 1)[0] + '\nDo not emit ihcg_evidence. A separate evidence-only formalization call follows this blueprint call. Translate ANY clear research intent (even 1-2 sentences of casual English with zero equations or units) into a complete, falsifiable, SI-normalized Blueprint grounded in retrieved literature and registries, logging every added equation/constant/domain/initial-condition/baseline in `assumptions_ledger`. CRITICAL ANTI-GAMING RULE: preserve the user\'s exact claimed predicate faithfully (never silently weaken or invert a potentially false claim). Separate mutable simulation buffer slots (`typed_slots`), immutable SI/model constants (`constants`), and pure algebraic helper definitions (`derived_expressions`). Include `typed_slots`: an array of `{symbol, si_dim: seven integer SI exponents, role: CONTROL_PARAM|DYNAMIC_STATE|DERIVED_OBSERVABLE|CONSERVATION_AUX|TIME_STEP}`, with optional `si_unit`, `initial_value`, `bounds`, `semantic_tag` (MUST be `0` unless using a reserved tag `251`–`254`; NEVER number `semantic_tag` sequentially like slot indices), `stochastic_role` (UNIFORM|SEED|COUNTER|CHANNEL for MONTE_CARLO slots), `unit_scale` ([positive_int_num, positive_int_den], default [1, 1], never 0). Keep `assumptions_ledger` concise (3–6 key entries). Keep variable_map and typed_slots compact (at most 24 slots; the formal verifier enforces a hard cap of 32 bounded variables). NEVER write to any CONTROL_PARAM slot inside simulation_step (any computed parameter such as beta = R0 * gamma must have role="DERIVED_OBSERVABLE"). Indices are derived from variable_map. Contract invariants MUST reference only state[i] / buf.data[i] bound in variable_map (never telemetry["finite"], which Sandbox_Execute already enforces unconditionally). For ODE/dynamical kernels, write pure branch-free smooth RK4 updates with ZERO defensive clamping (`if x < 0: x = 0`, `max(0.0, ...)`) and ZERO `if t == 0:` guards (running peak `if x > buf.data[k]: buf.data[k] = x` on a TRAJECTORY_MAX slot is allowed). For fixed-point equilibrium problems (`FIXED_POINT`), write a pure branch-free `simulation_step` that evaluates the equilibrium residual vector `g(x)` (and any algebraic readouts) directly at `buf.data`. Primary contract assertions must retain the original hypothesis expressed directly on the governing physical/mathematical observable slot (e.g. `state[eta_idx] >= 0.65`, NEVER a proxy counter like `feasible_count >= 1.0`), not its negation even when you expect falsification. Do not include mutually contradictory assertions or put the desired answer in an input slot.\n'


async def _finish_planned_blueprint(payload, question, *, fallback, complete, repair_complete=None):
    if not fallback:
        import inspect as _inspect
        from .formalization import formalize_blueprint
        core = parse_blueprint({**payload, 'ihcg_evidence': None, 'research_prompt': question})
        immutable = core.model_dump()
        immutable['ihcg_evidence'] = payload.get('ihcg_evidence')
        if repair_complete is not None and 'repair_complete' in _inspect.signature(formalize_blueprint).parameters:
            return await formalize_blueprint(question, immutable, complete, repair_complete=repair_complete)
        return await formalize_blueprint(question, immutable, complete)
    return _parse_planned_blueprint(payload, question, require_evidence=False)


def compact_literature_digest(
    items: List[Dict[str, Any]],
    limit: int = 6,
    max_chars: int = 2800,
    *,
    question: str = "",
    title: str = "",
    hypothesis: str = "",
) -> str:
    """Bounded, relevance-ranked evidence digest (top-K citations, equations, SI constants)."""
    if not items:
        return "(none retrieved)"
    try:
        from .report import rank_literature_by_relevance
        ranked = rank_literature_by_relevance(
            items, prompt=question, title=title, hypothesis=hypothesis
        )
    except Exception:  # noqa: BLE001
        ranked = list(items)
    lines: List[str] = []
    total_chars = 0
    for it in ranked[:limit]:
        if not isinstance(it, dict):
            continue
        year = f" ({it.get('year')})" if it.get("year") else ""
        summary = truncate_at_sentence(str(it.get("summary") or ""), 220)
        eqs = [str(e) for e in (it.get("equations") or [])[:2] if str(e).strip()]
        consts = it.get("constants") or {}
        const_bits = [f"{k}={v}" for k, v in list(consts.items())[:4]] if isinstance(consts, dict) else []
        extra_bits: List[str] = []
        if eqs:
            extra_bits.append("eqs: " + "; ".join(eqs)[:160])
        if const_bits:
            extra_bits.append("consts: " + ", ".join(const_bits)[:120])
        extra_str = f" [{'; '.join(extra_bits)}]" if extra_bits else ""
        line = f"- [{it.get('source')}] {it.get('title')}{year}: {summary}{extra_str}"
        if total_chars + len(line) + 1 > max_chars and lines:
            break
        lines.append(line[:max_chars])
        total_chars += len(line) + 1
    return "\n".join(lines) if lines else "(none retrieved)"


def format_literature(
    items: List[Dict[str, Any]],
    limit: int = 6,
    max_chars: int = 2800,
    *,
    question: str = "",
) -> str:
    return compact_literature_digest(items, limit=limit, max_chars=max_chars, question=question)


def summarize_pinned_code_for_prompt(
    code: str,
    pinned_functions: Optional[List[str]] = None,
) -> str:
    """Delta-only code view: replace already-pinned function bodies with signature + slot stubs.

    Pinned functions (`layer_1_step..layer_{k-1}_step` and explicitly listed
    `pinned_functions`) are immutable and automatically restored by the accretion
    merger, so omitting their full bodies saves thousands of tokens on Stage k > 1
    accretion and healing retries.
    """
    import ast as _ast

    raw = (code or "").strip()
    if not raw:
        return ""
    try:
        tree = _ast.parse(raw)
    except SyntaxError:
        return raw[:8000]

    pinned_set = set(pinned_functions or ())
    if pinned_functions is None:
        # Auto-detect prior layer_*_step functions only when caller did not
        # provide an explicit pinned_functions list; never stub the active
        # highest layer if simulation_step is a thin wrapper calling it.
        layer_nums = []
        for node in tree.body:
            if isinstance(node, _ast.FunctionDef) and node.name.startswith("layer_") and node.name.endswith("_step"):
                mid = node.name[6:-5]
                if mid.isdigit():
                    layer_nums.append((int(mid), node.name))
        if layer_nums:
            max_num = max(n for n, _ in layer_nums)
            has_non_wrapper_sim = any(
                isinstance(n, _ast.FunctionDef)
                and n.name == "simulation_step"
                and len(n.body) > 2
                for n in tree.body
            )
            for num, lname in layer_nums:
                if num < max_num or has_non_wrapper_sim:
                    pinned_set.add(lname)

    if not pinned_set:
        return raw

    out_stmts: List[str] = []
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name in pinned_set:
            reads: List[int] = []
            writes: List[int] = []
            has_value_return = False
            for sub in _ast.walk(node):
                if isinstance(sub, _ast.Return) and sub.value is not None:
                    has_value_return = True
                if (
                    isinstance(sub, _ast.Subscript)
                    and isinstance(sub.value, _ast.Attribute)
                    and sub.value.attr == "data"
                    and isinstance(sub.slice, _ast.Constant)
                    and isinstance(sub.slice.value, int)
                ):
                    idx = int(sub.slice.value)
                    if isinstance(sub.ctx, _ast.Store):
                        if idx not in writes:
                            writes.append(idx)
                    else:
                        if idx not in reads:
                            reads.append(idx)
            args_str = ", ".join(a.arg for a in node.args.args)
            reads_s = sorted(reads)[:16]
            writes_s = sorted(writes)[:16]
            stub_stmt = "return 0.0" if has_value_return else "pass"
            out_stmts.append(
                f"def {node.name}({args_str}):\n"
                f"    # [PINNED VERIFIED LAYER — body omitted; restored automatically; "
                f"reads slots {reads_s}, writes slots {writes_s}]\n"
                f"    {stub_stmt}"
            )
        else:
            out_stmts.append(_ast.unparse(node))
    return "\n\n".join(out_stmts)


def summarize_contracts_for_prompt(contracts: List[Dict[str, Any]], max_chars: int = 4500) -> str:
    """Compact deterministic JSON serialization of contracts for LLM prompts."""
    slim: List[Dict[str, Any]] = []
    for c in contracts or []:
        if not isinstance(c, dict):
            c = c.model_dump() if hasattr(c, "model_dump") else {}
        entry: Dict[str, Any] = {
            "contract_id": c.get("contract_id"),
            "dt": c.get("dt"),
            "steps": c.get("steps"),
            "invariants": [
                {
                    "variable_name": inv.get("variable_name"),
                    "assertion": inv.get("assertion"),
                    "tolerance": inv.get("tolerance", 1e-6),
                }
                for inv in (c.get("invariants") or [])
                if isinstance(inv, dict)
            ],
        }
        if c.get("initial_values"):
            entry["initial_values"] = c.get("initial_values")
        elif c.get("initial_state"):
            entry["initial_state"] = list(c.get("initial_state") or [])[:24]
        slim.append(entry)
    return json.dumps(slim, sort_keys=True, separators=(",", ":"))[:max_chars]


def compute_healing_error_fingerprint(failure: str, code: str = "") -> str:
    """Canonical error fingerprint `(error_class + normalized_message + offending_symbol/line)`."""
    import hashlib as _hashlib

    raw = str(failure or "").strip()
    low = raw.lower()
    err_class = "GENERIC_FAILURE"
    for candidate in (
        "prewrite_guard",
        "scientific_integrity",
        "slot_schema_error",
        "slot_alignment_error",
        "contract_faithfulness_error",
        "noop_accretion_layer",
        "vault_reference",
        "research_ode_reference_failed",
        "transpiler",
        "syntaxerror",
        "nameerror",
        "unboundlocalerror",
        "typeerror",
        "zerodivisionerror",
        "diverged",
        "falsified",
    ):
        if candidate in low:
            err_class = candidate.upper()
            break

    # Extract offending symbol, variable name, or state[k] slot
    slots = sorted(set(re.findall(r"(?:state|buf\.data|data)\[(\d+)\]", raw)))
    symbols = sorted(set(re.findall(r"'([A-Za-z_][A-Za-z0-9_]{1,48})'", raw)))[:4]
    # Normalize volatile floating-point numbers so minor telemetry jitter on the
    # same broken invariant or syntax error hashes to the same fingerprint
    first_line = raw.splitlines()[0] if raw else ""
    norm_msg = re.sub(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", "<NUM>", first_line.lower())
    norm_msg = re.sub(r"\s+", " ", norm_msg).strip()[:240]
    key = f"{err_class}|{norm_msg}|slots={','.join(slots[:6])}|syms={','.join(symbols)}"
    return _hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()


def _autofix_blueprint_contract_alignment(
    bp_dict: Dict[str, Any],
    question: str,
) -> Dict[str, Any]:
    """Tier-0 deterministic repair of blueprint contract alignment & sweep coverage (0 tokens)."""
    if not isinstance(bp_dict, dict):
        return bp_dict
    contracts = bp_dict.get("contracts")
    vm = bp_dict.get("variable_map")
    if not isinstance(contracts, list) or not contracts or not isinstance(vm, dict) or not vm:
        return bp_dict

    from .slot_schema import (
        ROLE_CONTROL_PARAM,
        ROLE_TIME_STEP,
        _BAND_NAME_RE,
        _NONZERO_T0_RE,
        _STRICT_POS_SWEEP_RE,
        _UPPER_CEIL_NAME_RE,
        _bound_directions_in_assertion,
        classify_variable_roles,
        compile_initial_state_from_values,
        observable_name,
    )

    code = str(bp_dict.get("code_template") or "")
    slot_roles = bp_dict.get("slot_roles") if isinstance(bp_dict.get("slot_roles"), dict) else {}
    var_roles = classify_variable_roles(vm, code, contracts, explicit_roles=slot_roles)

    explicit_nonzero_t0 = False
    for m in _NONZERO_T0_RE.finditer(question or ""):
        try:
            if abs(float(m.group(1))) > 1e-12:
                explicit_nonzero_t0 = True
                break
        except ValueError:
            pass

    param_ranges: Dict[str, Tuple[float, float]] = {}
    for m in _STRICT_POS_SWEEP_RE.finditer(question or ""):
        try:
            p_key = m.group(1).lower()
            lo, hi = float(m.group(2)), float(m.group(3))
            if hi >= lo:
                param_ranges[p_key] = (lo, hi)
        except ValueError:
            continue

    n_contracts = len(contracts)
    for c_idx, c in enumerate(contracts):
        if not isinstance(c, dict):
            continue
        init = list(c.get("initial_state") or [])
        max_slot = max((int(s) for s in vm.values() if isinstance(s, int)), default=-1)
        while len(init) <= max_slot:
            init.append(0.0)
        iv = dict(c.get("initial_values") or {})
        frac = (c_idx + 1) / (n_contracts + 1) if n_contracts > 1 else 0.5
        for name, raw_slot in vm.items():
            slot = int(raw_slot)
            role = var_roles.get(name)
            cur_val = float(iv.get(name, init[slot] if slot < len(init) else 0.0))
            if role == ROLE_TIME_STEP and not explicit_nonzero_t0:
                cur_val = 0.0
            elif role == ROLE_CONTROL_PARAM:
                nlow = name.lower()
                for p_stem, (lo, hi) in param_ranges.items():
                    if nlow == p_stem or nlow.startswith(p_stem + "_"):
                        if (lo > 0.0 and abs(cur_val) <= 1e-15) or (n_contracts >= 2 and c_idx > 0 and "initial_values" not in c):
                            cur_val = lo + frac * (hi - lo)
                        break
            iv[name] = cur_val
            if 0 <= slot < len(init):
                init[slot] = cur_val
        c["initial_values"] = iv
        c["initial_state"] = compile_initial_state_from_values(vm, iv, init)

    # Ensure parameter sweep variation across contracts when prompt requests a sweep
    if n_contracts >= 2 and param_ranges:
        for p_stem, (lo, hi) in param_ranges.items():
            if hi <= lo:
                continue
            matching = [
                (name, int(slot))
                for name, slot in vm.items()
                if var_roles.get(name) == ROLE_CONTROL_PARAM
                and (name.lower() == p_stem or name.lower().startswith(p_stem + "_"))
            ]
            for name, slot in matching:
                vals = [
                    round(float((c.get("initial_state") or [0.0] * (slot + 1))[slot]), 12)
                    for c in contracts
                    if isinstance(c, dict) and slot < len(c.get("initial_state") or [])
                ]
                if len(set(vals)) <= 1:
                    for c_idx, c in enumerate(contracts):
                        if not isinstance(c, dict):
                            continue
                        frac = (c_idx + 1) / (n_contracts + 1)
                        new_v = lo + frac * (hi - lo)
                        if isinstance(c.get("initial_values"), dict):
                            c["initial_values"][name] = new_v
                        if isinstance(c.get("initial_state"), list) and slot < len(c["initial_state"]):
                            c["initial_state"][slot] = new_v

    # Repair one-sided band/ceiling invariant names or rebind extrema slots
    extrema_map: Dict[str, int] = {
        observable_name(k): int(v)
        for k, v in vm.items()
        if observable_name(k).startswith(("peak_", "max_", "min_"))
    }
    for c in contracts:
        if not isinstance(c, dict):
            continue
        for inv in c.get("invariants") or []:
            if not isinstance(inv, dict):
                continue
            iname = str(inv.get("variable_name") or "")
            assertion = str(inv.get("assertion") or "")
            has_lo, has_hi = _bound_directions_in_assertion(assertion)
            if _BAND_NAME_RE.search(iname) and not (has_lo and has_hi):
                # Rename one-sided assertion so it does not claim to be a two-sided band
                inv["variable_name"] = re.sub(
                    r"(?:^|_)(?:band|window|corridor|range)(?=_|$)",
                    "_positivity" if has_lo else "_check",
                    iname,
                    flags=re.I,
                ).strip("_") or "state_check"
            elif (
                _UPPER_CEIL_NAME_RE.search(iname)
                and not iname.lower().startswith(("min_", "lower_"))
                and not iname.lower().endswith(("_nonneg", "_nonnegative", "_positivity", "_finite"))
                and not has_hi
            ):
                inv["variable_name"] = re.sub(
                    r"(?:^|_)(?:bound|max|limit|ceiling|cap)(?=_|$)",
                    "_positivity" if has_lo else "_check",
                    iname,
                    flags=re.I,
                ).strip("_") or "state_positivity"
            sem = observable_name(str(inv.get("variable_name") or ""))
            if sem in extrema_map:
                target_slot = extrema_map[sem]
                slots_in_a = sorted(set(_canon_slots_in_text(assertion)))
                if len(slots_in_a) == 1 and slots_in_a[0] != target_slot:
                    inv["assertion"] = _canon_rewrite_slots(assertion, slots_in_a[0], target_slot)
    return bp_dict


async def _emit_plan_progress(
    progress_cb: Optional[Callable[[str, Dict[str, Any]], Any]],
    step: str,
    message: str,
    **extra: Any,
) -> None:
    """Emit immediate, flushed console & SSE progress logs during blueprint planning."""
    import inspect as _inspect
    import sys as _sys

    line = f"[PLAN_PROGRESS:{step}] {message}"
    try:
        _sys.stderr.write(line + "\n")
        _sys.stderr.flush()
    except Exception:
        pass
    if progress_cb is not None:
        payload = {"event": "activity", "stage": "PLAN", "step": step, "message": message, **extra}
        try:
            res = progress_cb("search_log", payload)
            if _inspect.isawaitable(res):
                await res
        except Exception:
            pass


def _feasibility_context(feasibility: Any) -> str:
    """Render a FeasibilityAssessment (or raw dict) for planner prompts."""
    if feasibility is None:
        return "(not yet assessed — classify implicitly from the request)"
    if isinstance(feasibility, dict):
        verdict = feasibility.get("verdict", "")
        rationale = feasibility.get("rationale", "")
        reformulation = feasibility.get("reformulation", "")
    else:
        verdict = getattr(feasibility, "verdict", "")
        rationale = getattr(feasibility, "rationale", "")
        reformulation = getattr(feasibility, "reformulation", "")
    lines = [f"verdict: {verdict}", f"rationale: {rationale}"]
    if reformulation:
        lines.append(f"surrogate/reformulation: {reformulation}")
    return "\n".join(lines)


async def generate_research_plan(question: str,
                                 llm_complete: Optional[Callable[[str], Any]] = None,
                                 fallback: bool = True,
                                 literature: Optional[List[Dict[str, Any]]] = None,
                                 feasibility: Any = None,
                                 extra_context: str = "",
                                 llm_formalizer: Optional[Callable[[str], Any]] = None,
                                 llm_fast: Optional[Callable[[str], Any]] = None,
                                 ) -> Dict[str, Any]:
    """Tier-2 REASONING plan synthesis; falls back to a deterministic blueprint.

    ``fallback`` must be False whenever an LLM provider is configured: the canned
    KdV stub is restricted to explicit offline (no-key) unit-test mode.
    """
    from ..llm.telemetry import llm_call_context

    error: Optional[str] = None
    evidence = _compute_structured_evidence(question, literature)
    if llm_complete is not None:
        try:
            prompt = _fill(PLANNER_PROMPT, tools=_blueprint_tools_context(fallback),
                           feasibility=_feasibility_context(feasibility),
                           literature=compact_literature_digest(literature or [], question=question),
                           question=question)
            prompt += _evidence_prompt_block(evidence)
            if extra_context:
                prompt += "\n\n" + extra_context
            with llm_call_context("blueprint_planning", attempt=1):
                raw = await llm_complete(prompt)
            data = _extract_json(str(raw))
            if isinstance(data, dict):
                try:
                    bp = await _finish_planned_blueprint(data, question, fallback=fallback, complete=llm_formalizer or llm_complete, repair_complete=llm_fast or llm_formalizer)
                except ValueError as exc:
                    from .formalization import FormalizationError
                    if isinstance(exc, FormalizationError): raise
                    if fallback: raise
                    with llm_call_context("blueprint_schema_repair", attempt=2):
                        repaired = await llm_complete(prompt+"\n\nBLUEPRINT SCHEMA VALIDATION ERROR: "+str(exc)
                            +"\nReturn the complete corrected blueprint JSON without ihcg_evidence; a separate evidence-only call follows. Never fabricate absent physical information. Previous JSON:\n"+json.dumps(data))
                    bp = await _finish_planned_blueprint(_extract_json(str(repaired)), question, fallback=False, complete=llm_formalizer or llm_complete, repair_complete=llm_fast or llm_formalizer)
                bp_dict = _reconcile_time_horizon(bp.model_dump(), question, evidence)
                bp_dict = _autofix_blueprint_contract_alignment(bp_dict, question)
                return {"blueprint": _attach_evidence(bp_dict, evidence),
                        "source": "llm", "structured_evidence": evidence.model_dump() if evidence else None}
            error = "planner returned no JSON object"
        except Exception as exc:  # noqa: BLE001 - fall through to fallback
            from .formalization import FormalizationError
            if isinstance(exc, FormalizationError) and not fallback: raise
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
    if fallback:
        out: Dict[str, Any] = {"blueprint": _attach_evidence(dict(_FALLBACK_BLUEPRINT), evidence),
                               "source": "fallback"}
        if evidence is not None:
            out["structured_evidence"] = evidence.model_dump()
        if error:
            out["planner_error"] = error
        return out
    raise RuntimeError(f"research plan synthesis failed: {error}")


def _coerce_feasibility(data: Any) -> Optional[FeasibilityAssessment]:
    """Build a FeasibilityAssessment from a combined-prompt payload fragment.

    Verdicts are coerced case-insensitively (LLM JSON often lowercases them);
    the schema validator nulls refusal fields on feasible verdicts.
    """
    if not isinstance(data, dict):
        return None
    try:
        verdict = FeasibilityVerdict(str(data.get("verdict") or ""))
    except ValueError:
        return None
    reformulation = str(data.get("reformulation") or "").strip()
    if not reformulation and verdict != FeasibilityVerdict.OPTIMAL_DIRECT:
        reformulation = ("Reformulate as a falsifiable quantitative question "
                         "with explicit governing equations and measurable "
                         "invariants that a deterministic sandbox can verify.")
    try:
        return FeasibilityAssessment(
            verdict=verdict,
            rationale=truncate_at_sentence(str(data.get("rationale") or ""), 1200) or "LLM assessment",
            violated_boundary=str(data.get("violated_boundary") or "")[:400] or None,
            reformulation=reformulation[:600] or None,
            assessed_by="llm")
    except Exception:  # noqa: BLE001
        return None


async def generate_plan_with_feasibility(question: str,
                                         llm_complete: Optional[Callable[[str], Any]] = None,
                                         fallback: bool = True,
                                         literature: Optional[List[Dict[str, Any]]] = None,
                                         assume_feasible: bool = False,
                                         extra_context: str = "",
                                         progress_cb: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
                                         llm_formalizer: Optional[Callable[[str], Any]] = None,
                                         llm_fast: Optional[Callable[[str], Any]] = None,
                                         ) -> Dict[str, Any]:
    """Feasibility and blueprint synthesis, then bounded evidence-only formalization.

    Returns {"feasibility": FeasibilityAssessment|None, "blueprint": dict|None,
    "source": "llm"|"fallback"|"heuristic", ...}. When the verdict is a refusal,
    ``blueprint`` is None. With ``fallback=False`` (live provider configured) any
    synthesis failure raises instead of returning the canned KdV stub.
    """
    from ..llm.telemetry import llm_call_context
    from .feasibility import (
        _heuristic_verdict,
        is_computational_dynamical_benchmark,
        override_refusal_for_computational_benchmark,
    )

    heuristic = _heuristic_verdict(question)
    await _emit_plan_progress(
        progress_cb,
        "feasibility_precheck",
        f"Heuristic feasibility pre-check: verdict={heuristic.verdict}",
        verdict=str(heuristic.verdict),
    )
    bypass = assume_feasible and llm_complete is not None
    if heuristic.verdict.startswith("REFUSED") and not bypass:
        # Fast path: clear-cut refusals never reach the LLM.
        return {"feasibility": heuristic, "blueprint": None, "source": "heuristic"}
    if llm_complete is None:
        if fallback:
            return {"feasibility": heuristic,
                    "blueprint": dict(_FALLBACK_BLUEPRINT), "source": "fallback"}
        raise RuntimeError("research plan synthesis failed: no LLM planner configured")

    compacted_lit = compact_literature_digest(literature or [], question=question)
    evidence = _compute_structured_evidence(question, literature)
    await _emit_plan_progress(
        progress_cb,
        "evidence_compaction",
        f"Compacted {len(literature or [])} literature sources into {len(compacted_lit)} chars.",
        raw_sources=len(literature or []),
        compacted_chars=len(compacted_lit),
    )
    prompt = _fill(COMBINED_PLAN_PROMPT, tools=_blueprint_tools_context(fallback),
                   literature=compacted_lit,
                   question=question)
    prompt += _evidence_prompt_block(evidence)
    from .shbt_federation import resolve_shbt_solvability
    shbt_res = resolve_shbt_solvability(question)
    if shbt_res.should_activate and shbt_res.evidence_markdown:
        prompt += (
            "\n\n"
            + shbt_res.evidence_markdown
            + "\n\nAUTHORITATIVE SHBT SOLVABILITY DIRECTIVE: This frontier physics "
            "problem is SOLVABLE in open-boundary physics via the SHBT Frontier Knowledge "
            "Federation (`verdict`: `OPTIMAL_DIRECT`). Do NOT output `REFUSED_IMPOSSIBLE` "
            "and NEVER set `blueprint` to null; you MUST emit both `feasibility` and the "
            "complete `blueprint` JSON object using the verified SHBT equations and SI constants above."
        )
    if extra_context:
        prompt += "\n\n" + extra_context

    est_tokens = max(1, len(prompt) // 4)
    await _emit_plan_progress(
        progress_cb,
        "blueprint_llm_dispatch",
        f"Dispatching combined feasibility + blueprint synthesis (~{est_tokens} est. input tokens)...",
        estimated_input_tokens=est_tokens,
    )
    seen_error_fingerprints: set = set()
    with llm_call_context("blueprint_planning", attempt=1):
        raw = await llm_complete(prompt)
    data = _extract_json(str(raw))
    if not isinstance(data, dict) and not fallback:
        fp = compute_healing_error_fingerprint("combined planner returned no JSON object", str(raw)[:400])
        seen_error_fingerprints.add(fp)
        await _emit_plan_progress(
            progress_cb,
            "blueprint_json_repair",
            "Initial response lacked valid JSON object; running bounded JSON repair pass...",
        )
        with llm_call_context("blueprint_json_repair", attempt=2):
            raw_json_retry = await llm_complete(
                prompt
                + "\n\nCRITICAL: Your previous response did not contain a complete, valid JSON object. "
                "Emit ONLY one complete valid JSON object matching the required schema."
            )
        data = _extract_json(str(raw_json_retry))
    if not isinstance(data, dict):
        if fallback:
            return {"feasibility": heuristic,
                    "blueprint": dict(_FALLBACK_BLUEPRINT), "source": "fallback",
                    "planner_error": "combined planner returned no JSON object"}
        raise RuntimeError("research plan synthesis failed: combined planner returned no JSON object")
    llm_feas = _coerce_feasibility(data.get("feasibility"))
    feasibility = heuristic if shbt_res.should_activate else (llm_feas or heuristic)
    llm_refused = bool(llm_feas and llm_feas.verdict.startswith("REFUSED"))
    if feasibility.verdict.startswith("REFUSED") or (shbt_res.should_activate and (llm_refused or not isinstance(data.get("blueprint"), dict))):
        if shbt_res.should_activate:
            feasibility = heuristic
            if not isinstance(data.get("blueprint"), dict) and not (isinstance(data.get("contracts"), list) and data.get("title")):
                shbt_prompt = (
                    prompt
                    + "\n\nIMPORTANT OVERRIDE: This frontier physics problem is SOLVABLE "
                    "via the SHBT Frontier Knowledge Federation (`OPTIMAL_DIRECT`). Do NOT "
                    "refuse and NEVER set `blueprint` to null. Output a complete valid `blueprint` "
                    "JSON object using the exact SHBT equations, SI constants, and invariants provided above."
                )
                with llm_call_context("blueprint_shbt_rescue", attempt=2):
                    raw2 = await llm_complete(shbt_prompt)
                data2 = _extract_json(str(raw2))
                if isinstance(data2, dict) and isinstance(data2.get("blueprint"), dict):
                    data = data2
                elif isinstance(data2, dict) and isinstance(data2.get("contracts"), list) and data2.get("title"):
                    data = data2
        elif is_computational_dynamical_benchmark(question):
            feasibility = override_refusal_for_computational_benchmark(question, feasibility)
            if feasibility.verdict in (FeasibilityVerdict.REFUSED_IMPOSSIBLE, FeasibilityVerdict.REFUSED_UNSUITABLE):
                return {"feasibility": feasibility, "blueprint": None, "source": "llm"}
            if not isinstance(data.get("blueprint"), dict) and not (isinstance(data.get("contracts"), list) and data.get("title")):
                surrogate_prompt = (
                    prompt
                    + "\n\nIMPORTANT OVERRIDE: The feasibility verdict is REFORMULATED_SURROGATE. "
                    "Do NOT refuse. You MUST output a complete valid blueprint JSON using "
                    "symmetric/compatible surrogate coupling parameters so all stated invariants "
                    "are mathematically consistent and falsifiable."
                )
                with llm_call_context("blueprint_surrogate_rescue", attempt=2):
                    raw2 = await llm_complete(surrogate_prompt)
                data2 = _extract_json(str(raw2))
                if isinstance(data2, dict) and isinstance(data2.get("blueprint"), dict):
                    data = data2
        else:
            return {"feasibility": feasibility, "blueprint": None, "source": "llm"}
    bp_raw = data.get("blueprint")
    if not isinstance(bp_raw, dict) and isinstance(data.get("contracts"), list) and data.get("title"):
        bp_raw = data
    if not isinstance(bp_raw, dict):
        if fallback:
            return {"feasibility": feasibility,
                    "blueprint": dict(_FALLBACK_BLUEPRINT), "source": "fallback",
                    "planner_error": "combined planner returned no blueprint object"}
        raise RuntimeError("research plan synthesis failed: no blueprint object in response")
    try:
        bp = await _finish_planned_blueprint(bp_raw, question, fallback=fallback, complete=llm_formalizer or llm_complete, repair_complete=llm_fast or llm_formalizer)
    except Exception as exc:
        from .formalization import FormalizationError
        if isinstance(exc, FormalizationError): raise
        if fallback:
            return {"feasibility": feasibility,
                    "blueprint": dict(_FALLBACK_BLUEPRINT), "source": "fallback",
                    "planner_error": str(exc)[:300]}
        fp = compute_healing_error_fingerprint(str(exc))
        if fp in seen_error_fingerprints:
            raise
        seen_error_fingerprints.add(fp)
        repair_prompt = (
            prompt
            + "\n\nCRITICAL BLUEPRINT VALIDATION ERROR on your previous JSON output: "
            f"{str(exc)}. "
            "You MUST include a non-empty `contracts` list (with `c1_`, `c2_`, `c3_` contracts, "
            "each containing `contract_id`, `target_backend`: 'cpp', `initial_state`, `dt`, `steps`, "
            "and a non-empty `invariants` list), plus `blueprint_id`, `title`, `falsifiable_hypothesis`, "
            "`search_queries`, `variable_map`, and `code_template`."
            " Do not emit `ihcg_evidence`; a separate evidence-only call follows. "
            "Correct the structural blueprint, including executable/contract bindings; never invent missing "
            "physical declarations or weaken a claim to make it pass. Previous blueprint JSON:\n"
            + json.dumps(bp_raw)
        )
        with llm_call_context("blueprint_schema_repair", attempt=2):
            raw_retry2 = await llm_complete(repair_prompt)
        data_retry2 = _extract_json(str(raw_retry2))
        bp_raw2 = None
        if isinstance(data_retry2, dict):
            bp_raw2 = data_retry2.get("blueprint")
            if not isinstance(bp_raw2, dict) and isinstance(data_retry2.get("contracts"), list) and data_retry2.get("title"):
                bp_raw2 = data_retry2
        if not isinstance(bp_raw2, dict):
            raise
        bp = await _finish_planned_blueprint(bp_raw2, question, fallback=False, complete=llm_formalizer or llm_complete, repair_complete=llm_fast or llm_formalizer)
    bp_dict = _reconcile_time_horizon(bp.model_dump(), question, evidence)
    bp_dict["research_prompt"] = question
    # Tier-0 deterministic contract alignment autofix BEFORE running validation checks (0 tokens)
    bp_dict = _autofix_blueprint_contract_alignment(bp_dict, question)
    try:
        from .slot_schema import audit_contract_alignment, validate_contract_faithfulness
        _plan_issues: List[str] = []
        for fn in (
            lambda: audit_contract_alignment(
                bp_dict.get("contracts") or [],
                bp_dict.get("variable_map") or {},
                prompt=question,
                code=str(bp_dict.get("code_template") or ""),
            ),
            lambda: validate_contract_faithfulness(
                bp_dict.get("contracts") or [],
                bp_dict.get("variable_map") or {},
                prompt=question,
                code=str(bp_dict.get("code_template") or ""),
                require_sweep_coverage=True,
            ),
        ):
            try:
                fn()
            except ValueError as err:
                _plan_issues.append(str(err))
        if _plan_issues and llm_complete is not None:
            issue_fp = compute_healing_error_fingerprint("; ".join(_plan_issues))
            if issue_fp not in seen_error_fingerprints:
                seen_error_fingerprints.add(issue_fp)
                fix_prompt = (
                    f"{prompt}\n\n"
                    "VALIDATION FAILURE ON YOUR PREVIOUS BLUEPRINT JSON:\n- "
                    + "\n- ".join(_plan_issues)
                    + "\n\nFix ALL validation issues and emit the complete corrected JSON object:\n"
                    "1) Every contract MUST include `initial_values` keyed by exact `variable_map` names "
                    "(TIME_STEP initialized to 0.0, strictly positive CONTROL_PARAMs initialized cleanly inside their ranges).\n"
                    "2) Every `*_band`, `*_window`, `*_corridor`, `*_bound`, `*_max`, `*_limit` invariant MUST encode its finite upper (and lower) bounds — NEVER collapse to `state[i] >= 0.0`.\n"
                    "3) Every `peak_*`/`max_*`/`min_*` invariant MUST reference the extrema slot in `variable_map`.\n"
                    "4) When the prompt requests a parameter sweep, vary CONTROL_PARAM `initial_values` across stages or include an explicit sweep loop in `code_template`."
                )
                try:
                    with llm_call_context("blueprint_alignment_repair", attempt=2):
                        raw_fix = await llm_complete(fix_prompt)
                    data_fix = _extract_json(str(raw_fix))
                    bp_fix = None
                    if isinstance(data_fix, dict):
                        bp_fix = data_fix.get("blueprint")
                        if not isinstance(bp_fix, dict) and isinstance(data_fix.get("contracts"), list) and data_fix.get("title"):
                            bp_fix = data_fix
                    if isinstance(bp_fix, dict):
                        bp2 = await _finish_planned_blueprint(bp_fix, question, fallback=fallback, complete=llm_formalizer or llm_complete, repair_complete=llm_fast or llm_formalizer)
                        bp_dict = _reconcile_time_horizon(bp2.model_dump(), question, evidence)
                        bp_dict["research_prompt"] = question
                        bp_dict = _autofix_blueprint_contract_alignment(bp_dict, question)
                except Exception as error:  # noqa: BLE001
                    from .formalization import FormalizationError
                    if isinstance(error, FormalizationError): raise
    except Exception as error:  # noqa: BLE001
        from .formalization import FormalizationError
        if isinstance(error, FormalizationError): raise
    await _emit_plan_progress(
        progress_cb,
        "blueprint_parse_validation",
        f"Blueprint '{bp_dict.get('blueprint_id')}' parsed and validated ({len(bp_dict.get('contracts') or [])} contracts).",
        blueprint_id=bp_dict.get("blueprint_id"),
        contracts_count=len(bp_dict.get("contracts") or []),
    )
    return {"feasibility": feasibility,
            "blueprint": _attach_evidence(bp_dict, evidence),
            "source": "llm",
            "structured_evidence": evidence.model_dump() if evidence else None}


async def generate_expert_research_prompt(question: str,
                                          llm_complete: Optional[Callable[[str], Any]] = None,
                                          literature: Optional[List[Dict[str, Any]]] = None,
                                          extra_context: str = "",
                                          ) -> Dict[str, Any]:
    """Plan Mode: one Tier-2 call -> feasibility + editable expert research prompt.

    Returns {"feasibility": FeasibilityAssessment, "expert_prompt": str,
    "source": "llm"|"llm_text"|"heuristic"}. Raises RuntimeError when the LLM
    call itself fails or returns an empty response; plain-Markdown answers
    without a JSON object are wrapped as the expert prompt (never a canned
    stub, never a 502 for parse drift).
    """
    from ..llm.telemetry import llm_call_context
    from .feasibility import (
        _heuristic_verdict,
        is_computational_dynamical_benchmark,
        override_refusal_for_computational_benchmark,
    )

    heuristic = _heuristic_verdict(question)
    if heuristic.verdict.startswith("REFUSED"):
        return {"feasibility": heuristic, "expert_prompt": "", "source": "heuristic"}
    if llm_complete is None:
        raise RuntimeError("expert prompt synthesis failed: no LLM planner configured")
    prompt = _fill(EXPERT_PROMPT_PLANNER_PROMPT, tools="",
                   literature=compact_literature_digest(literature or [], question=question),
                   question=question)
    if extra_context:
        prompt += "\n\n" + extra_context
    with llm_call_context("expert_prompt_planning", attempt=1):
        raw = await llm_complete(prompt)
    text = _clean_prompt_text(raw)
    data = _extract_json(text)
    if not isinstance(data, dict):
        # Safe fallback wrapper: the model answered in plain Markdown (or
        # emitted unparseable JSON). Wrap the cleaned text as the expert
        # prompt instead of failing the request with a 502.
        if not text:
            raise RuntimeError("expert prompt synthesis failed: empty LLM response")
        return {"feasibility": heuristic, "expert_prompt": text[:8000],
                "source": "llm_text"}
    feasibility = _coerce_feasibility(data.get("feasibility")) or heuristic
    expert_prompt = str(data.get("expert_prompt") or "").strip()
    if feasibility.verdict.startswith("REFUSED"):
        if is_computational_dynamical_benchmark(question):
            feasibility = override_refusal_for_computational_benchmark(question, feasibility)
            if not expert_prompt:
                surrogate_prompt = (
                    prompt
                    + "\n\nIMPORTANT OVERRIDE: The feasibility verdict is REFORMULATED_SURROGATE. "
                    "Do NOT refuse. Emit the complete expert_prompt using symmetric/compatible "
                    "surrogate coupling parameters."
                )
                with llm_call_context("expert_prompt_surrogate_rescue", attempt=2):
                    raw2 = await llm_complete(surrogate_prompt)
                text2 = _clean_prompt_text(raw2)
                data2 = _extract_json(text2)
                if isinstance(data2, dict) and str(data2.get("expert_prompt") or "").strip():
                    expert_prompt = str(data2["expert_prompt"]).strip()
                elif text2:
                    expert_prompt = text2[:8000]
        else:
            return {"feasibility": feasibility, "expert_prompt": "", "source": "llm"}
    if not expert_prompt:
        # JSON parsed but carried no prompt: fall back to any non-JSON prose
        # the model produced rather than erroring out.
        if not text:
            raise RuntimeError("expert prompt synthesis failed: empty expert_prompt")
        return {"feasibility": feasibility, "expert_prompt": text[:8000],
                "source": "llm_text"}
    return {"feasibility": feasibility, "expert_prompt": expert_prompt, "source": "llm"}


async def synthesize_accretive_stage(llm_complete: Callable[[str], Any], *,
                                     stage: int, total: int, objective: str,
                                     verified_code: str, verified_contracts: List[Dict[str, Any]],
                                     telemetry: Dict[str, Any],
                                     literature: List[Dict[str, Any]],
                                     failures: List[str],
                                     pinned_functions: Optional[List[str]] = None,
                                     stage_contracts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Synthesize the stage-k delta module + new contracts with delta-only pinned stubs."""
    from ..llm.telemetry import llm_call_context

    compact_code = summarize_pinned_code_for_prompt(verified_code.strip(), pinned_functions)
    stage_contracts_text = (
        summarize_contracts_for_prompt(stage_contracts, max_chars=3000)
        if stage_contracts
        else "(none pre-planned; emit stage-specific contracts)"
    )
    prompt = _fill(
        ACCRETION_PROMPT, tools=TOOLS_CONTEXT, stage=stage, total=total, prev=stage - 1,
        objective=objective, verified_code=compact_code,
        verified_contracts=summarize_contracts_for_prompt(verified_contracts, max_chars=3500),
        stage_contracts=stage_contracts_text,
        canonical_slots=canonical_slot_map_text(verified_contracts + list(stage_contracts or [])),
        telemetry=json.dumps(telemetry, sort_keys=True, default=str)[:2000],
        literature=compact_literature_digest(literature, limit=4, max_chars=1800, question=objective),
        failures="\n".join(f"- {f}" for f in failures[-3:]) or "(none)")
    with llm_call_context(f"stage_{stage}_accretion_synthesis", attempt=len(failures) + 1):
        raw = await llm_complete(prompt)
    data = _extract_json(str(raw))
    if not isinstance(data, dict) or not isinstance(data.get("code"), str):
        code_block = ""
        text = str(raw or "")
        for m in _FENCE_RE.finditer(text):
            blk = m.group(1).strip()
            if "def " in blk and ("simulation_step" in blk or f"layer_{stage}_step" in blk or "buf" in blk):
                code_block = blk
                break
        if not code_block:
            stripped = text.strip()
            if stripped.startswith(("def ", "import ", "#")) and ("simulation_step" in stripped or f"layer_{stage}_step" in stripped):
                code_block = stripped
        if not code_block:
            raise ValueError("accretion synthesis returned no {code, contracts} JSON")
        data = {"code": code_block, "contracts": []}
    if not isinstance(data.get("contracts"), list) or not data["contracts"]:
        data["contracts"] = []
    if data["contracts"]:
        data["contracts"], data["variable_map"], _ = canonicalize_contract_dicts(
            data["contracts"], data.get("variable_map"), prior_contracts=verified_contracts)
    return data


async def request_healing_directives(llm_complete: Callable[[str], Any], *,
                                     objective: str, code: str,
                                     contracts: List[Dict[str, Any]],
                                     failure: str,
                                     pinned_functions: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Ask the LLM for structured healing directives only (never source code)."""
    import asyncio
    import inspect
    from ..llm.telemetry import llm_call_context

    compact_code = summarize_pinned_code_for_prompt(code.strip(), pinned_functions)[:8000]
    prompt = _fill(HEALING_DIRECTIVE_PROMPT, objective=objective[:1500],
                   code=compact_code,
                   contracts=summarize_contracts_for_prompt(contracts, max_chars=3000),
                   failure=str(failure)[:2500])
    try:
        with llm_call_context("reflex_heal_directives"):
            result = llm_complete(prompt)
            raw = (await asyncio.wait_for(result, timeout=60.0)
                   if inspect.isawaitable(result) else result)
    except Exception:  # noqa: BLE001 - directive call is best-effort
        return None
    data = _extract_json(str(raw))
    if not isinstance(data, dict):
        return None
    # Directive plans must not smuggle source repairs through a `code` key.
    data.pop("code", None)
    data.pop("replacements", None)
    data.pop("ast_patches", None)
    return data


async def refine_stage_code(llm_complete: Callable[[str], Any], *, objective: str,
                            code: str, contracts: List[Dict[str, Any]],
                            failure: str,
                            pinned_functions: Optional[List[str]] = None) -> Optional[str]:
    """Repair failing kernel code using the counterexample (contracts fixed, pinned bodies omitted)."""
    from ..llm.telemetry import llm_call_context

    slot_table = canonical_slot_map_text(contracts)
    compact_code = summarize_pinned_code_for_prompt(code.strip(), pinned_functions)
    prompt = _fill(REFINE_PROMPT, tools=TOOLS_CONTEXT,
                   objective=f"{objective}\n\n## Canonical Global State Slot Table\n{slot_table}",
                   code=compact_code,
                   contracts=summarize_contracts_for_prompt(contracts, max_chars=4000),
                   failure=failure[:2500])
    with llm_call_context("reflex_heal_refine"):
        raw = await llm_complete(prompt)
    data = _extract_json(str(raw))
    if isinstance(data, dict) and isinstance(data.get("code"), str) and data["code"].strip():
        return data["code"]
    # Fallback: reasoning models often ignore the JSON contract and return the
    # repaired module as a fenced/raw Python block. Accept it instead of
    # silently retrying the unchanged broken code (the 0-second healing bug).
    text = str(raw or "")
    for m in _FENCE_RE.finditer(text):
        block = m.group(1).strip()
        if "def " in block and ("simulation_step" in block or "layer_" in block or "buf" in block):
            return block
    stripped = text.strip()
    if stripped.startswith(("def ", "import ", "#")) and ("simulation_step" in stripped or "layer_" in stripped):
        return stripped
    return None


def _compute_structured_evidence(question: str,
                                 literature: Optional[List[Dict[str, Any]]]
                                 ) -> Optional[Any]:
    """Zero-LLM deterministic extraction over the question + literature hits."""
    try:
        from .extractor import QuantitativeInsightExtractor
        return QuantitativeInsightExtractor().extract(question, literature or [])
    except Exception:  # noqa: BLE001 - evidence is best-effort
        return None


def _evidence_prompt_block(evidence: Any) -> str:
    table = getattr(evidence, "summary_markdown_table", "") or ""
    if not table:
        return ""
    return ("\n\n## Extracted Literature Evidence (SI-normalized, deterministic)\n"
            + table
            + "\n\nUse these EXACT SI-normalized constants and Python-ready equations. "
              "Never invent dummy [1, 1, 1] initial states or guess units: bind every "
              "physical constant from the table above (or state it as a named "
              "assumption in the hypothesis).")


def _attach_evidence(bp_dict: Dict[str, Any], evidence: Any) -> Dict[str, Any]:
    """Attach structured_evidence + extracted_constants onto a blueprint dict."""
    if evidence is None or not isinstance(bp_dict, dict):
        return bp_dict
    from .extractor import evidence_si_constants
    out = dict(bp_dict)
    out["structured_evidence"] = evidence.model_dump()
    out["extracted_constants"] = evidence_si_constants(evidence)
    return out


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
    from ..llm.telemetry import llm_call_context

    if llm_fast is not None:
        try:
            with llm_call_context(f"stage_{stage}_query_gen", tier="fast", attempt=1):
                raw = await llm_fast(_fill(
                    QUERY_PROMPT, stage=stage, objective=objective, hypothesis=hypothesis,
                    telemetry=json.dumps(telemetry, sort_keys=True, default=str)[:1500],
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

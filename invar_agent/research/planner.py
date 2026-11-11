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

from ..schemas.research import FeasibilityAssessment, FeasibilityVerdict, ResearchBlueprint
from .feasibility import truncate_at_sentence

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
LIMIT-CYCLE / EXTREMA / PERIOD TRACKING: when tracking limit cycles, extrema, or
periods, use scalar state variables inside `buf.data` (e.g., `prev_val =
buf.data[0]`, `cycle_count = buf.data[10]`, `last_peak_t = buf.data[11]`). NEVER
use dynamic Python lists (`list.append`), `while` searches across history, or
variable-length arrays, as they cannot lower to C++ C-ABI.
VARIABLE STYLE: prefer plain scalar variables and indexable arrays over
nested Python tuples. Tuple unpacking (`a, b = b, a`), sequence unpacking
(`q, p, s = state`) and tuple literals are supported by the transpiler, but
simple scalars keep kernels readable and portable.
Invariants must be falsifiable and physically meaningful; never write vacuous
invariants (like x == x) and never loosen tolerances to force a pass.
CRITICAL INVARIANT RULE: Do NOT guess narrow arbitrary numeric lower/upper
intervals (e.g. `15.0 <= state[7] <= 70.0`) before the ODE is integrated. Only
write invariants for: (a) explicit bounds stated by the user (e.g.
`state[3] < 40.0`), (b) physical positivity and finiteness (`state[i] >= 0.0`,
`math.isfinite(state[i])`), (c) monotonic conservation laws
(`state[1] <= initial_state[1]`), or (d) exact analytical identities within a
generous relative tolerance (>= 15% for exponential atmosphere entry
approximations).

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
  model (e.g. "Stage 1 baseline, then Stage 2 accretion"), set
  `target_research_stages >= 2` with one `stage_objectives` entry per stage,
  and name each contract's stage clearly (prefix contract_ids with `c1_` /
  `c2_` or `stage1_` / `stage2_`) so Stage 2 builds on the pinned Stage 1
  layer instead of silently collapsing into one stage."""

PLANNER_PROMPT = """You are the Research Agent planner of a mathematically verified computational
research engine. Produce ONE falsifiable research blueprint for the request.

{tools}

## Output: strict JSON object with keys
blueprint_id (lowercase snake_case, under 32 chars, e.g. 'mars_entry_probe'),
title (concise human-readable, under 80 chars, e.g. 'Mars Atmospheric Entry &
Stagnation Heating'), falsifiable_hypothesis,
search_queries (3-6 concise technical keyword queries of 2-4 terms each for
arXiv/OpenAlex; NEVER full sentences, and never use filler words like
'vs', 'versus', 'exact', 'using', 'between' or 'comparison'),
target_research_stages (1-4; use 2-3 when the request describes a progression,
e.g. "start by ... then integrate ..."), stage_objectives (one short objective per
stage, stage 1 first), contracts (stage-1 contracts: list of {{contract_id,
target_backend, timeout_seconds, invariants: [{{variable_name, assertion,
tolerance}}], max_memory_mb, initial_state: [floats], dt, steps}}),
code_template (the STAGE-1 module only, v9_transpiler_friendly Python),
variable_map ({{"<metric_with_units>": <buf.data index>}} for every numerical
quantity the user requested), max_feedback_rounds (2-4).
Keep initial_state small (<= 16 slots unless the surrogate genuinely needs more,
max 64) and steps <= 20000. initial_state must be the physical state of the
surrogate model, never a repeating dummy vector.
No markdown fences, no <think> blocks. Output JSON only.

## Feasibility context (authoritative)
{feasibility}

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
- When tracking limit cycles, extrema, or periods, use scalar state variables
  inside `buf.data` (e.g., `prev_val = buf.data[0]`, `cycle_count = buf.data[10]`,
  `last_peak_t = buf.data[11]`). NEVER use dynamic Python lists (`list.append`),
  `while` searches across history, or variable-length arrays, as they cannot
  lower to C++ C-ABI.

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

CRITICAL — v9_transpiler_friendly subset (violations are rejected by PreWrite_Guard):
use ONLY pure module-level functions, math.* calls, if/elif/else, for/while loops,
return statements and plain float arithmetic. FORBIDDEN: tuples (no tuple literals,
no `a, b = ...` unpacking, no multi-value returns), classes, exceptions, dicts,
sets, comprehensions, nested function definitions, string operations, imports
other than `import math`. If the failure mentions "unsupported construct: tuples",
eliminate EVERY tuple from the module.

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

# -- combined feasibility + blueprint (single Tier-2 call, Task 4 latency fix) --

COMBINED_PLAN_PROMPT = """You are the Epistemic Feasibility Gate AND the Research Agent planner
of a mathematically verified computational research engine, in ONE pass. Produce
ONE strict JSON object with two keys: "feasibility" and "blueprint".

{tools}

## Part 1 — feasibility (gate)
Classify the request into EXACTLY ONE verdict:
- OPTIMAL_DIRECT: quantitative, mathematical, physical, dynamical or algorithmic
  research falsifiable by simulation.
- REFORMULATED_SURROGATE: too large for full-scale local execution but admits a
  rigorous reduced-order / 0D-1D asymptotic / scaling-law surrogate.
- REFUSED_IMPOSSIBLE: mathematically, physically or logically impossible
  (conservation-law violations, undecidable, contradictory invariants).
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
target_research_stages (1-4; use 2-3 when the request describes a progression),
stage_objectives (one short objective per stage, stage 1 first), contracts
(stage-1 contracts: list of {{contract_id, target_backend, timeout_seconds,
invariants: [{{variable_name, assertion, tolerance}}], max_memory_mb,
initial_state: [floats], dt, steps}}), code_template (the STAGE-1 module only,
v9_transpiler_friendly Python), variable_map ({{"<metric_with_units>":
<buf.data index>}} for every numerical quantity the user requested),
max_feedback_rounds (2-4).
Keep initial_state small (<= 16 slots unless the surrogate genuinely needs more,
max 64) and steps <= 20000. initial_state must encode the physical state of the
surrogate model, never a repeating dummy vector. If the verdict is
REFORMULATED_SURROGATE, implement exactly the surrogate equations named in the
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
        i = start + 1


def _extract_json(text: str) -> Optional[Any]:
    """Robust JSON extraction: strip think-blocks, unwrap fences, then use a
    string-aware balanced-brace matcher so trailing prose after the JSON
    object does not break parsing."""
    text = (text or "").strip()
    # Reasoning models may leak chain-of-thought wrappers; strip them first.
    text = _THINK_BLOCK.sub("", text).strip()
    if not text:
        return None
    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    candidates = [c for c in candidates if c] + [text]
    for candidate in candidates:
        for open_c, close_c in (("{", "}"), ("[", "]")):
            for chunk in _iter_balanced(candidate, open_c, close_c):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue
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
    "structured_evidence", "extracted_constants",
}
_CONTRACT_FIELDS = {
    "contract_id", "target_backend", "timeout_seconds", "invariants",
    "max_memory_mb", "initial_state", "dt", "steps",
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


def _coerce_contract(raw: Any, index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    contract: Dict[str, Any] = {k: raw[k] for k in _CONTRACT_FIELDS if k in raw}
    contract["contract_id"] = str(contract.get("contract_id") or f"contract_{index + 1}")[:120]
    backend = str(contract.get("target_backend") or "cpp").strip().lower()
    contract["target_backend"] = backend if backend in _BACKENDS else "cpp"
    try:
        contract["timeout_seconds"] = float(contract.get("timeout_seconds", 30.0))
        contract["dt"] = float(contract.get("dt", 0.01))
        contract["steps"] = int(contract.get("steps", 1000))
        contract["max_memory_mb"] = int(contract.get("max_memory_mb", 1024))
    except (TypeError, ValueError):
        return None
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
        # Assertions are executable expressions: never slice them mid-token
        # (a [:500] cut produced `state[12] <= initial[12] + 1e-06 * initial[`).
        cleaned["assertion"] = str(cleaned.get("assertion") or "")[:2000]
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
        for name, idx in list(variable_map.items())[:64]:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if i >= 0 and str(name).strip():
                vm[str(name).strip()[:120]] = i
        out["variable_map"] = vm
    elif variable_map is not None:
        out.pop("variable_map", None)
    return out


def parse_blueprint(payload: Dict[str, Any]) -> ResearchBlueprint:
    return ResearchBlueprint.model_validate(normalize_blueprint_payload(payload))


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
                                 ) -> Dict[str, Any]:
    """Tier-2 REASONING plan synthesis; falls back to a deterministic blueprint.

    ``fallback`` must be False whenever an LLM provider is configured: the canned
    KdV stub is restricted to explicit offline (no-key) unit-test mode.
    """
    error: Optional[str] = None
    evidence = _compute_structured_evidence(question, literature)
    if llm_complete is not None:
        try:
            prompt = _fill(PLANNER_PROMPT, tools=TOOLS_CONTEXT,
                           feasibility=_feasibility_context(feasibility),
                           literature=format_literature(literature or []),
                           question=question)
            prompt += _evidence_prompt_block(evidence)
            if extra_context:
                prompt += "\n\n" + extra_context
            raw = await llm_complete(prompt)
            data = _extract_json(str(raw))
            if isinstance(data, dict):
                bp = parse_blueprint(data)
                return {"blueprint": _attach_evidence(bp.model_dump(), evidence),
                        "source": "llm", "structured_evidence": evidence.model_dump() if evidence else None}
            error = "planner returned no JSON object"
        except Exception as exc:  # noqa: BLE001 - fall through to fallback
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
                                         ) -> Dict[str, Any]:
    """Single Tier-2 call: feasibility gate + blueprint (Task 4 latency fix).

    Returns {"feasibility": FeasibilityAssessment|None, "blueprint": dict|None,
    "source": "llm"|"fallback"|"heuristic", ...}. When the verdict is a refusal,
    ``blueprint`` is None. With ``fallback=False`` (live provider configured) any
    synthesis failure raises instead of returning the canned KdV stub.

    ``assume_feasible=True`` is set when Plan Mode already approved this prompt
    (OPTIMAL_DIRECT / REFORMULATED_SURROGATE): the pre-flight substring
    heuristics must not reject the approved expert prompt on re-submission;
    the LLM feasibility assessment inside the combined call still decides.
    """
    from .feasibility import _heuristic_verdict

    heuristic = _heuristic_verdict(question)
    bypass = assume_feasible and llm_complete is not None
    if heuristic.verdict.startswith("REFUSED") and not bypass:
        # Fast path: clear-cut refusals never reach the LLM.
        return {"feasibility": heuristic, "blueprint": None, "source": "heuristic"}
    if llm_complete is None:
        if fallback:
            return {"feasibility": heuristic,
                    "blueprint": dict(_FALLBACK_BLUEPRINT), "source": "fallback"}
        raise RuntimeError("research plan synthesis failed: no LLM planner configured")
    prompt = _fill(COMBINED_PLAN_PROMPT, tools=TOOLS_CONTEXT,
                   literature=format_literature(literature or []),
                   question=question)
    evidence = _compute_structured_evidence(question, literature)
    prompt += _evidence_prompt_block(evidence)
    if extra_context:
        prompt += "\n\n" + extra_context
    raw = await llm_complete(prompt)
    data = _extract_json(str(raw))
    if not isinstance(data, dict):
        if fallback:
            return {"feasibility": heuristic,
                    "blueprint": dict(_FALLBACK_BLUEPRINT), "source": "fallback",
                    "planner_error": "combined planner returned no JSON object"}
        raise RuntimeError("research plan synthesis failed: combined planner returned no JSON object")
    feasibility = _coerce_feasibility(data.get("feasibility")) or heuristic
    if feasibility.verdict.startswith("REFUSED"):
        return {"feasibility": feasibility, "blueprint": None, "source": "llm"}
    bp_raw = data.get("blueprint")
    if not isinstance(bp_raw, dict):
        if fallback:
            return {"feasibility": feasibility,
                    "blueprint": dict(_FALLBACK_BLUEPRINT), "source": "fallback",
                    "planner_error": "combined planner returned no blueprint object"}
        raise RuntimeError("research plan synthesis failed: no blueprint object in response")
    bp = parse_blueprint(bp_raw)
    return {"feasibility": feasibility,
            "blueprint": _attach_evidence(bp.model_dump(), evidence),
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
    from .feasibility import _heuristic_verdict

    heuristic = _heuristic_verdict(question)
    if heuristic.verdict.startswith("REFUSED"):
        return {"feasibility": heuristic, "expert_prompt": "", "source": "heuristic"}
    if llm_complete is None:
        raise RuntimeError("expert prompt synthesis failed: no LLM planner configured")
    prompt = _fill(EXPERT_PROMPT_PLANNER_PROMPT, tools="",
                   literature=format_literature(literature or []),
                   question=question)
    if extra_context:
        prompt += "\n\n" + extra_context
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

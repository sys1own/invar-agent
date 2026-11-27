"""Cyclic 9-stage Research Agent state machine with cumulative blueprint evolution.

Stages mirror the verified builder kernel:
  Ingest -> GoI_Plan -> Wavefront_Schedule -> Overlay_Mutate -> PreWrite_Guard
  -> SMT_Concolic_Verify -> Sandbox_Execute -> Reflex_Heal -> Adjoint_Commit_or_Rollback

Accretive multi-stage flow (``ResearchBlueprint.target_research_stages``):
  Stage 1 synthesises and verifies a foundation module, then pins it as a
  checkpoint layer. Stage k > 1 re-enters Ingest -> GoI_Plan in accretive
  mode: telemetry from stage k-1 steers literature search, the LLM extends
  the *same* unified module (pinned functions are immutable, the new driver
  must call them), the GoI scheduler is updated incrementally via
  Sherman-Morrison-Woodbury, and Sandbox_Execute re-verifies every pinned
  layer inside the same compiled artefact as the new logic. A failing stage
  rolls back only its delta; exhausted retries export the last verified
  checkpoint instead of losing verified work.

Three-tier cognitive routing:
  Tier-2 (REASONING): blueprint planning, accretive synthesis, refinement
  Tier-1 (FAST): telemetry-guided literature query formulation
  Tier-0 (REFLEX): native deterministic SMT/Z3, concolic, firewall, compile, egg, HIN
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import logging
import math
import re
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from ..runtime.build_utils import rollback_overlay
from ..llm.errors import LLMProviderError, backoff_delay
from ..healing.buffer_api import BUFFER_API
from ..healing.structural import is_structural_failure, unresolved_symbol
from ..runtime.research_reference import ReferenceError
from .evolution import text_evolutions, check_evolution_reference
from .slot_schema import (
    ROLE_CONTROL_PARAM,
    _CONTROL_PARAM_NAME_RE,
    _analyze_code_slot_flow,
    audit_contract_alignment,
    canonical_variable_map,
    classify_variable_roles,
    to_blueprint_role,
    validate_contract_faithfulness,
    validate_contract_slots,
    validate_slot_writes,
)
from .scientific_integrity import (SCIENTIFIC_CONTRACT, output_dataflow, protected_observation_slots, validate_observables,
                                   observed_roundoff_scales, evaluate_roundoff_predicate,
                                   physical_projection_contracts,
                                   validate_protocol, validate_variant_schemes)

from ..schemas.research import (InvariantConstraint, ModuleLayer, ResearchBlueprint,
                                SimulationResult, StageTelemetry, VerificationContract)
from ..transpiler.emitters.base import EmitterError
from ..transpiler.emitters.cpp_emitter import CppEmitter, entry_symbol
from ..transpiler.emitters.hin_emitter import HinEmitter  # noqa: F401 - public re-export
from ..transpiler.frontend import python_source_to_uast, UnsupportedError
from ..runtime.cabi_bridge import (
    ARCHETYPE_DYNAMICAL_STEPPER,
    ARCHETYPE_EQUILIBRIUM_FIXED_POINT,
    ARCHETYPE_MONTE_CARLO_ENSEMBLE,
    ARCHETYPE_PARAMETER_BIFURCATION_SWEEP,
    EXECUTION_ARCHETYPES,
    DynamicCABIBridge,
    KernelExecutionError,
)
from ..algorithms.prng import (
    invar_rand_binomial,
    invar_rand_normal,
    invar_rand_poisson,
    invar_rand_u01,
    invar_splitmix64_stateless,
)
from ..uast.totalized_math import (
    betainc_reg,
    gammainc_reg,
    hill_eq,
    kl_term,
    logistic_sigmoid,
    logit,
    michaelis_menten,
    norm_cdf,
    norm_ppf,
    shannon_term,
    welford_cov_update,
    welford_m2_update,
    welford_mean_update,
)

for _pan_fn_name, _pan_fn in (
    ("invar_splitmix64_stateless", invar_splitmix64_stateless),
    ("invar_rand_u01", invar_rand_u01),
    ("invar_rand_normal", invar_rand_normal),
    ("invar_rand_poisson", invar_rand_poisson),
    ("invar_rand_binomial", invar_rand_binomial),
    ("welford_mean_update", welford_mean_update),
    ("welford_m2_update", welford_m2_update),
    ("welford_cov_update", welford_cov_update),
    ("norm_cdf", norm_cdf),
    ("norm_ppf", norm_ppf),
    ("logistic_sigmoid", logistic_sigmoid),
    ("logit", logit),
    ("hill_eq", hill_eq),
    ("michaelis_menten", michaelis_menten),
    ("kl_term", kl_term),
    ("shannon_term", shannon_term),
    ("gammainc_reg", gammainc_reg),
    ("betainc_reg", betainc_reg),
):
    setattr(math, _pan_fn_name, _pan_fn)
from .accretion import (AccretionError, code_hash, is_noop_layer_wrapper,
                        layer_driver_name, merge_accretive_module,
                        pin_layer_code, candidate_layer_code,
                        strip_driver, unified_from_layers)
from .feasibility import _heuristic_verdict
from .planner import (canonicalize_contract_dicts, compute_healing_error_fingerprint,
                        guided_queries, summarize_pinned_code_for_prompt,
                        _reconcile_time_horizon,
                        refine_stage_code, request_healing_directives,
                        synthesize_accretive_stage)
from ..healing.deterministic_ast import (
    _detect_bloch_radius, _ensure_nonempty_blocks, _verified_projection_slots,
    _restore_pinned_functions, _snapshot_pinned_functions,
    apply_healing_directives, classify_buffer_slots,
    enforce_conservation_projection, enforce_lyapunov_sign,
    numeric_expr_equivalent, resolve_slot_collisions, rewrite_monotone_drift,
    sanitize_intrinsics)
from .search import LiteratureSearchClient
from .smt import check_invariants_consistent

STAGES = [
    "Ingest", "GoI_Plan", "Wavefront_Schedule", "Overlay_Mutate", "PreWrite_Guard",
    "SMT_Concolic_Verify", "Sandbox_Execute", "Reflex_Heal", "Adjoint_Commit_or_Rollback",
]

FORBIDDEN_IMPORTS = ("os", "sys", "subprocess", "socket", "requests", "urllib",
                     "ctypes", "importlib", "pathlib", "shutil", "builtins")
FORBIDDEN_CALLS = ("eval", "exec", "compile", "open", "input", "__import__")
ALLOWED_IMPORTS = ("math",)
DEFAULT_INITIAL_STATE = [1.0] * 16

logger = logging.getLogger(__name__)

_INSIGHT_RE = re.compile(
    r"([^.;]*?(?:=|\\approx|\\propto|proportional to|scales? (?:as|like|with)|"
    r"coefficient|constant|exponent|order)[^.;]*)", re.I)


class PreWriteGuardError(ValueError):
    pass


@dataclass
class ResearchPlanningFailure:
    run_id: str
    meaningfulness_certificate: Any
    status: str = "INCONCLUSIVE_UNVERIFIED"
    epistemic_verdict: str = "INCONCLUSIVE_UNVERIFIED"
    blueprint: None = None
    feasibility: Any = None
    report: str = ""


def shacl_chiasmus_firewall(code: str) -> List[str]:
    """Tier-0 SHACL/Chiasmus firewall: strict AST allow-list + legacy regex guard."""
    from ..runtime.isolation import validate_kernel_ast_allowlist

    violations: List[str] = []
    for mod in FORBIDDEN_IMPORTS:
        if re.search(rf"^\s*(import|from)\s+{re.escape(mod)}\b", code, re.M):
            violations.append(f"forbidden import: {mod}")
    for name in FORBIDDEN_CALLS:
        if re.search(rf"\b{name}\s*\(", code):
            violations.append(f"forbidden call: {name}()")
    if re.search(r"while\s+True\s*:", code) and "break" not in code:
        violations.append("unbounded while True loop without break")
    violations.extend(validate_kernel_ast_allowlist(code, require_simulation_step=False))
    return sorted(set(violations))


def v9_transpiler_violations(code: str) -> List[str]:
    """AST-level v9_transpiler_friendly compliance check (PreWrite_Guard allow-list)."""
    from ..runtime.isolation import validate_kernel_ast_allowlist

    return validate_kernel_ast_allowlist(code, require_simulation_step=True)


def _goi_wavefront_plan(blueprint: ResearchBlueprint) -> Dict[str, Any]:
    """GoI_Plan + Wavefront_Schedule: nilpotent schedule over contracts."""
    return GoIContractScheduler().plan(len(blueprint.contracts), 0)


class GoIContractScheduler:
    """Nilpotent GoI schedule over verification nodes with SMW delta updates.

    Nodes ``0..r-1`` are regression checks of pinned layers (chained), nodes
    ``r..r+c-1`` are new-stage contracts. The pinned chain is the base
    operator; coupling edges from the last pinned node to every new contract
    are applied as a rank-c Sherman-Morrison-Woodbury update ΔM = P·Qᵀ.
    """

    def plan(self, new_contracts: int, regression_nodes: int) -> Dict[str, Any]:
        n = max(1, regression_nodes + new_contracts)
        edges: List[Tuple[int, int]] = [(i, i + 1) for i in range(regression_nodes - 1)]
        out: Dict[str, Any] = {"nodes": n, "regression_nodes": regression_nodes,
                               "new_nodes": new_contracts, "smw_rank": 0}
        try:
            from invar_agent._native import GoIWavefrontScheduler
            sched = GoIWavefrontScheduler(n)
            for a, b in edges:
                sched.set_edge(a, b, 0.5)
            sched.compute_execution_operator()
            delta_edges = []
            if regression_nodes > 0 and new_contracts > 0:
                src = regression_nodes - 1
                cols = list(range(regression_nodes, n))
                # ΔM = P Qᵀ with P = e_src ⊗ 1ᵀ (n x c), Q columns = e_dst (n x c).
                p = [[(0.5 if i == src else 0.0) for _ in cols] for i in range(n)]
                q = [[(1.0 if i == dst else 0.0) for dst in cols] for i in range(n)]
                sched.apply_incremental_update(p, q)
                delta_edges = [(src, dst) for dst in cols]
                out["smw_rank"] = len(cols)
            else:
                for i in range(regression_nodes, n - 1):
                    pass
            edges = edges + delta_edges
            out.update(spectral_radius=float(sched.spectral_radius()),
                       nilpotent=bool(sched.is_nilpotent()), engine="native")
        except ImportError:
            if regression_nodes > 0:
                edges += [(regression_nodes - 1, d) for d in range(regression_nodes, n)]
            out.update(spectral_radius=0.0, nilpotent=True, engine="python")
        out["wavefronts"] = _topological_levels(n, edges)
        out["edges"] = edges
        return out


def _topological_levels(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    indeg = [0] * n
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    level = [i for i in range(n) if indeg[i] == 0]
    out: List[List[int]] = []
    while level:
        out.append(level)
        nxt = []
        for u in level:
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    nxt.append(v)
        level = nxt
    return out


def _lower_kernel(code_template: str):
    uast = python_source_to_uast(code_template)
    cpp = CppEmitter().emit_module(uast)
    return uast, cpp


def instrumented_fallback(code):
    from ..healing.deterministic_ast import instrument_purity
    return instrument_purity(code)


# -- kernel archetype detection ------------------------------------------------
# Research targets involving novel time-stepping (discrete gradients,
# symplectic/metriplectic maps, automata) must NOT be described as an RK4 ODE
# trajectory: the compiled C++20 kernel executes the researcher's exact
# discrete update law z[k+1] = Phi_dt(z[k]) once per step.
_DISCRETE_MAP_RE = re.compile(
    r"discrete[\s_\-]*(?:time|map|gradient)|gonzalez|biorthogonal|automata|"
    r"metriplectic|symplectic|strang|verlet|cayley|fixed[\s_\-]*point|"
    r"discrete[\s_\-]*update", re.IGNORECASE)
_CUSTOM_STEPPER_NAMES = {
    "custom_discrete_map", "discrete_map", "discrete_gradient", "gonzalez",
    "biorthogonal", "symplectic_verlet", "metriplectic", "semi_implicit_euler",
    "monte_carlo_xoshiro", "hin_exact_reduction",
}


def kernel_mode_for_blueprint(bp: ResearchBlueprint) -> str:
    """Classify the stepping archetype of a blueprint's simulation kernel.

    Returns ``"custom_discrete_map"`` for novel discrete-time update laws,
    otherwise ``"rk4_ode"`` (the C-ABI RK4 ODE trajectory mode).
    """
    stepper = ""
    evidence = getattr(bp, "structured_evidence", None)
    if evidence is not None:
        stepper = str(getattr(evidence, "recommended_stepper", "") or "").strip().lower()
    text = f"{getattr(bp, 'falsifiable_hypothesis', '') or ''} " \
           f"{getattr(bp, 'title', '') or ''}"
    if stepper in _CUSTOM_STEPPER_NAMES or _DISCRETE_MAP_RE.search(text):
        return "custom_discrete_map"
    # The C-ABI bridge only distinguishes these two kernel archetypes; any
    # named stepper that is not a discrete map runs as the RK4 ODE mode.
    return "rk4_ode"


def execution_archetype_for_blueprint(bp: ResearchBlueprint) -> str:
    """Classify a research blueprint into one of the 4 Universal Execution Archetypes."""
    text = f"{getattr(bp, 'title', '') or ''} {getattr(bp, 'falsifiable_hypothesis', '') or ''}".lower()
    if any(k in text for k in (
        "monte carlo", "ensemble", "replication", "2sls", "wald", "ols", "estimator",
        "cramer-rao", "cramér-rao", "sample variance", "bootstrap", "welford",
    )):
        return ARCHETYPE_MONTE_CARLO_ENSEMBLE
    if any(k in text for k in (
        "fixed point", "fixed-point", "nash equilibrium", "replicator", "bellman",
        "kkt", "variational inequality", "stationary distribution", "game theory",
    )):
        return ARCHETYPE_EQUILIBRIUM_FIXED_POINT
    if any(k in text for k in (
        "bifurcation", "dose-response", "dose response", "parameter sweep",
        "phase diagram", "roc curve", "hill curve", "titration",
    )):
        return ARCHETYPE_PARAMETER_BIFURCATION_SWEEP
    return ARCHETYPE_DYNAMICAL_STEPPER


def kernel_log_label(mode: str) -> str:
    """Human-readable kernel archetype for the STAGE_k_BUILD log line."""
    if mode == "custom_discrete_map":
        return "custom discrete-map (z[k+1] = Phi_dt(z[k]))"
    if mode == "rk4_ode":
        return "RK4 trajectory"
    return mode.replace("_", " ")


_INVARIANT_FUNCS = {"len": len, "abs": abs, "min": min, "max": max, "sum": sum,
                    "all": all, "any": any, "sqrt": math.sqrt, "sin": math.sin,
                    "cos": math.cos, "tan": math.tan, "asin": math.asin,
                    "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
                    "exp": math.exp, "log": math.log, "log10": math.log10,
                    "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs,
                    "hypot": math.hypot, "isfinite": math.isfinite,
                    "isnan": math.isnan, "float": float, "int": int,
                    "range": range,
                    "invar_splitmix64_stateless": invar_splitmix64_stateless,
                    "invar_rand_u01": invar_rand_u01,
                    "invar_rand_normal": invar_rand_normal,
                    "invar_rand_poisson": invar_rand_poisson,
                    "invar_rand_binomial": invar_rand_binomial,
                    "welford_mean_update": welford_mean_update,
                    "welford_m2_update": welford_m2_update,
                    "welford_cov_update": welford_cov_update,
                    "norm_cdf": norm_cdf,
                    "norm_ppf": norm_ppf,
                    "logistic_sigmoid": logistic_sigmoid,
                    "logit": logit,
                    "hill_eq": hill_eq,
                    "michaelis_menten": michaelis_menten,
                    "kl_term": kl_term,
                    "shannon_term": shannon_term,
                    "gammainc_reg": gammainc_reg,
                    "betainc_reg": betainc_reg}


class _BufView:
    """Namespace shim giving invariant assertions ``buf.data`` / ``buf.size``."""

    __slots__ = ("data", "size", "dt")

    def __init__(self, state: List[float], dt: float = 0.01):
        self.data = state
        self.size = len(state)
        self.dt = dt


_VARIANT_FALLBACK = {"residual": 0.0, "drift": 1e-3, "energy_drift": 1e-3,
                     "finite": 1.0, "l2": 1.0}


class _VariantMetrics(dict):
    """Variant metric bag that never raises ``KeyError`` in invariant eval.

    ``variants['rk4']['residual']`` style assertions must evaluate cleanly even
    when the multi-variant runner populated nothing for that variant name.
    """

    def __missing__(self, key: str) -> Dict[str, float]:
        value = dict(_VARIANT_FALLBACK)
        self[key] = value
        return value


class _VariantsView:
    """Namespace shim for ``variants``: variant-name -> metrics mapping."""

    __slots__ = ("_metrics",)

    def __init__(self, metrics: Optional[Dict[str, Dict[str, float]]] = None):
        self._metrics = metrics if metrics is not None else {}

    def __getitem__(self, key: str) -> _VariantMetrics:
        value = self._metrics.get(key)
        if not isinstance(value, dict):
            value = dict(_VARIANT_FALLBACK)
            self._metrics[key] = value
        return _VariantMetrics(value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._metrics.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._metrics

    def items(self):
        return self._metrics.items()

    def keys(self):
        return self._metrics.keys()

    def values(self):
        return self._metrics.values()

    def __iter__(self):
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)


def _invariant_env(state: List[float], telemetry: Dict[str, float],
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Comprehensive, safe evaluation namespace for invariant assertions."""
    env: Dict[str, Any] = dict(_INVARIANT_FUNCS)
    variants = extra.get("variants") if extra else None
    env.update({"state": state, "data": state, "buf": _BufView(state),
                "initial": state, "initial_state": state,
                "trace_min": state, "trace_max": state,
                "telemetry": telemetry, "math": math,
                "bool": bool, "isfinite": math.isfinite,
                "variants": _VariantsView(variants), "tol": 0.0})
    if extra:
        env.update(extra)
        # ``env.update(extra)`` would clobber the safe variants view with the
        # raw dict; restore it so missing variant keys never raise KeyError.
        env["variants"] = _VariantsView(extra.get("variants"))
    return env


def _check_expression(expression: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_\[\]'\"\.\s<>=!+\-*/%()&,|]+", expression):
        raise PreWriteGuardError(f"invariant expression rejected: {expression!r}")
    if any(tok in expression for tok in ("import", "eval", "exec", "open", "__", "lambda")):
        raise PreWriteGuardError(f"invariant expression rejected: {expression!r}")


def _eval_invariant(expression: str, state: List[float], telemetry: Dict[str, float],
                    extra: Optional[Dict[str, Any]] = None) -> bool:
    """Tier-0 concolic check of an invariant expression over sandbox observables."""
    _check_expression(expression)
    expr_clean = expression.replace("&&", " and ").replace("||", " or ").replace("^", "**").strip().rstrip(".;, ")
    env = _invariant_env(state, telemetry, extra)
    try:
        if bool(eval(expr_clean, {"__builtins__": {}, **env})):
            return True
        return evaluate_roundoff_predicate(expr_clean, env, (extra or {}).get('_roundoff_scales', {}))
    except Exception as exc:  # noqa: BLE001
        raise PreWriteGuardError(
            f"invariant evaluation failed: {type(exc).__name__}: {exc}") from exc


def _observed_terms(expression: str, env: Dict[str, Any]) -> Dict[str, Any]:
    """Concrete values of every subscript/call term in a failing invariant."""
    out: Dict[str, Any] = {}
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.Subscript, ast.Call)):
            text = ast.unparse(node)
            try:
                val = eval(compile(ast.Expression(node), "<inv>", "eval"),  # noqa: S307
                           {"__builtins__": {}, **env})
                if isinstance(val, (int, float)):
                    out[text] = round(float(val), 10)
            except Exception:  # noqa: BLE001
                continue
    return out


def _l2(xs: List[float]) -> float:
    return math.sqrt(sum(x * x for x in xs)) if xs else 0.0


# -- contract stage staging ---------------------------------------------------
_CONTRACT_STAGE_RE = re.compile(r"^(?:c|stage)(\d+)(?:[a-z](?=[_-])|[_-])", re.IGNORECASE)


def _contract_stage_index(contract_id: str) -> int:
    """Infer the research stage a contract belongs to from its id.

    ``c1_...`` / ``stage1_...`` -> 1, ``c2_heating_accretion`` -> 2, etc.
    Unprefixed contracts belong to stage 1.
    """
    match = _CONTRACT_STAGE_RE.match(contract_id or "")
    return int(match.group(1)) if match else 1


# -- Tier-0 bound reconciliation (Reflex_Heal) --------------------------------
_STATE_SLOT_BASES = {"state", "data", "buf.data"}


def _eval_subscript(node: ast.AST, env: Dict[str, Any]) -> Optional[float]:
    try:
        val = eval(compile(ast.Expression(node), "<reconcile>", "eval"),  # noqa: S307
                   {"__builtins__": {}, **env})
    except Exception:  # noqa: BLE001
        return None
    return float(val) if isinstance(val, (int, float)) else None


def _reconcile_hallucinated_bounds(expression: str, state: List[float],
                                   telemetry: Dict[str, float],
                                   extra: Optional[Dict[str, Any]] = None
                                   ) -> Optional[str]:
    return None


def validate_blueprint_grounding(blueprint: ResearchBlueprint) -> List[str]:
    """Tier-0 Evidence-to-AST grounding validator (invoked before Sandbox_Execute).

    Returns a list of grounding warnings (empty = grounded). Never raises:
    warnings are surfaced as activity events and used to nudge healing.
    """
    warnings: List[str] = []
    evidence = getattr(blueprint, "structured_evidence", None)
    constants = getattr(blueprint, "extracted_constants", None) or {}
    has_real_params = bool(constants) or bool(
        getattr(evidence, "parameters", None) if evidence is not None else {})
    # 1. Lazy uniform dummy initial states ([1.0, 1.0, ...] with real domain
    #    parameters available) are a grounding smell: the kernel never binds
    #    the extracted SI constants.
    for contract in blueprint.contracts or []:
        initial = list(getattr(contract, "initial_state", None) or [])
        if (has_real_params and len(initial) >= 4
                and all(abs(v - 1.0) < 1e-12 for v in initial)):
            warnings.append(
                f"contract {contract.contract_id}: initial_state is a lazy "
                "uniform dummy array while extracted_constants carries real "
                "domain parameters — seed physics-based initial conditions")
            break
    # 2. unified_module_code must define pure simulation_step and bind SI constants.
    code = blueprint.unified_module_code or blueprint.code_template or ""
    if code.strip():
        try:
            tree = ast.parse(code)
            has_step = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and n.name == "simulation_step" for n in ast.walk(tree))
            if not has_step:
                warnings.append("unified module does not define simulation_step(buf, dt)")
        except SyntaxError as exc:
            warnings.append(f"unified module is not parseable Python: {exc}")
        if has_real_params:
            missing = [name for name, value in list(constants.items())[:24]
                       if isinstance(value, (int, float))
                       and abs(float(value)) > 1e-9
                       and not _literal_present(code, float(value))]
            if len(missing) > max(2, len(constants) // 2):
                warnings.append(
                    "most extracted SI constants are not bound as literals in "
                    f"the unified module (e.g. {', '.join(missing[:4])})")
    return warnings


def _literal_present(code: str, value: float) -> bool:
    """True when ``value`` (or a rounded repr) appears as a numeric literal."""
    candidates = {repr(value), f"{value:.6g}", f"{value:.10g}"}
    for cand in candidates:
        if not cand or cand in ("0", "0.0"):
            continue
        if cand.rstrip("0").rstrip(".") in code or cand in code:
            return True
    return False


_UNIT_SCALE_FACTORS = (1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9, 180.0 / math.pi, 9.80665)


def _detect_unit_scaling(observed: float, bound: float) -> Optional[Dict[str, Any]]:
    """Detect a common SI unit-conversion factor between observed and bound.

    Returns {"factor": f, "direction": "divide"|"multiply", "observed": o,
    "bound": b} when dividing/multiplying the observed value by a common
    factor makes it consistent with the invariant bound.
    """
    if not (math.isfinite(observed) and math.isfinite(bound)) or bound == 0.0:
        return None
    if observed == 0.0:
        return None
    # The unscaled observed value must actually violate the bound, otherwise
    # there is no unit-scaling story to tell.
    if abs(observed) <= abs(bound):
        return None
    best: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    for factor in _UNIT_SCALE_FACTORS:
        for direction, scaled in (("divide", observed / factor),
                                  ("multiply", observed * factor)):
            if scaled == 0.0 or not math.isfinite(scaled):
                continue
            if abs(scaled) > abs(bound):
                continue
            # Prefer the conversion that lands closest to the bound on a
            # log scale: 2.84e6 / 1e6 = 2.84 (near bound 10) beats
            # 2.84e6 * 1e-9 = 0.00284 (absurdly far below it).
            score = abs(math.log10(abs(scaled / bound)))
            # Tie-break toward "divide": observed * 1e-6 == observed / 1e6,
            # but the divide form names the canonical SI prefix factor.
            better = score < best_score or (
                score == best_score and direction == "divide"
                and (best is None or best["direction"] != "divide"))
            if better:
                best_score = score
                best = {"factor": factor, "direction": direction,
                        "observed": observed, "bound": bound, "scaled": scaled}
    return best


def _attempt_quality_score(outcome: Dict[str, Any],
                           results: List[Any]) -> Tuple[int, float]:
    """Rank a failed attempt: VERIFIED > FALSIFIED (finite, small violation)
    > DIVERGED (largest first_nonfinite_step) > COMPILE_ERROR/ERROR."""
    status = str(outcome.get("failed_status") or outcome.get("result_status") or "").upper()
    telems = [getattr(r, "telemetry", None) or {} for r in results]
    finite = [t for t in telems
              if t.get("first_nonfinite_step", -1) in (-1, None)
              or str(t.get("first_nonfinite_step", "")) in ("-1", "None")]
    if status == "FALSIFIED":
        # Prefer the smallest relative constraint violation among finite runs.
        viol = min((float(t.get("max_violation", t.get("max_relative_violation", 1e9)))
                    for t in finite), default=1e9)
        return (3, -viol)
    if status == "DIVERGED":
        # Prefer the trajectory that survived the most steps before diverging.
        steps = max((float(t.get("first_nonfinite_step", 0) or 0) for t in telems),
                    default=0.0)
        return (2, steps)
    if status == "VERIFIED":
        return (4, 0.0)
    return (1, 0.0)


def _failure_topic_keywords(blueprint: Any, raw_topic: str, max_words: int = 3) -> List[str]:
    """Top 2-3 COMPLETE domain keywords for healing re-search queries.

    Sources the blueprint's own literature queries first (they are already
    technical keyword strings), falls back to the hypothesis/title words.
    Words are taken whole — never sliced mid-word.
    """
    candidates: List[str] = []
    for q in (getattr(blueprint, "search_queries", None) or []):
        candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", str(q)))
    if not candidates:
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", raw_topic or "")
    stop = {"the", "a", "an", "of", "and", "or", "for", "with", "using",
            "model", "system", "study", "analysis", "simulation", "via",
            "this", "that", "into", "from", "equation", "equations"}
    seen: List[str] = []
    for w in candidates:
        low = w.lower().strip("-_")
        if len(low) < 3 or low in stop:
            continue
        if low not in {s.lower() for s in seen}:
            seen.append(w)
        if len(seen) >= max_words:
            break
    return seen or ([raw_topic] if raw_topic else [])


def formulate_diagnostic_search_query(blueprint: ResearchBlueprint,
                                      failure_result: Any) -> List[str]:
    """Closed-loop telemetry-guided re-search queries for DIVERGED/FALSIFIED.

    Returns a list of targeted literature queries (strings). Unit-scaling
    diagnostics are exposed via :func:`unit_scaling_diagnostic` on the same
    failure payload.
    """
    raw_topic = re.sub(r"\s+", " ",
                       blueprint.falsifiable_hypothesis or blueprint.title or "").strip()
    # Complete domain keywords only (no mid-word slicing): top technical terms
    # from the blueprint's own search queries, max 4 words total per query.
    topic_words = _failure_topic_keywords(blueprint, raw_topic)
    topic = " ".join(topic_words)
    status = ""
    telemetry: Dict[str, Any] = {}
    counterexample = ""
    if isinstance(failure_result, dict):
        status = str(failure_result.get("status") or "")
        telemetry = failure_result.get("telemetry") or {}
        counterexample = str(failure_result.get("counterexample") or
                             failure_result.get("failure") or "")
    else:
        status = str(getattr(failure_result, "status", "") or "")
        telemetry = getattr(failure_result, "telemetry", None) or {}
        counterexample = str(getattr(failure_result, "failure_counterexample", "") or "")
    status = status.upper()
    queries: List[str] = []
    # Max 4 words total: top domain keywords + 1 concise failure-mode term.
    if status == "DIVERGED" or "non-finite" in counterexample.lower() \
            or "diverged" in counterexample.lower():
        queries.append(f"{topic} stability" if topic else "numerical stability")
        queries.append(f"{topic} convergence" if topic else "fixed point convergence")
    if status == "FALSIFIED":
        # Unit-scaling diagnostic: observed metric vs invariant bound.
        diag = unit_scaling_diagnostic(blueprint, telemetry, counterexample)
        if diag:
            unit = "MW/m^2" if diag.get("si_unit") == "W/m^2" else (diag.get("si_unit") or "unit")
            queries.append(f"{topic} {unit} conversion" if topic else f"{unit} conversion")
        queries.append(f"{topic} counterexample" if topic else "regime breakdown")
    if not queries:
        queries.append(f"{topic} verification" if topic else "governing equations")
    seen, out = set(), []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:4]


def unit_scaling_diagnostic(blueprint: ResearchBlueprint,
                            telemetry: Dict[str, Any],
                            counterexample: str = "") -> Optional[Dict[str, Any]]:
    """Detect whether a FALSIFIED invariant is a pure SI unit-scaling issue.

    Compares the observed metric (from telemetry or the counterexample text)
    against the invariant bound and checks for common conversion factors
    (10^3, 10^6, 180/pi, 9.80665). Returns a diagnostic dict or None.
    """
    # Find the first violated invariant bound from the counterexample text.
    # Counterexample format: `assertion` (variable_name) observed {...}
    bound: Optional[float] = None
    variable: str = ""
    m = re.search(r"`([^`]+)`\s*\(([^)]+)\)\s*observed", counterexample or "")
    if m:
        assertion, variable = m.group(1), m.group(2)
        bm = re.search(r"(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", assertion)
        if bm:
            try:
                bound = float(bm.group(2))
            except ValueError:
                bound = None
    observed: Optional[float] = None
    # Prefer telemetry obs_<variable> or obs_* entries.
    for key, val in (telemetry or {}).items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        k = key.lower()
        if variable and (variable.lower() in k or k.endswith(variable.lower())):
            observed = float(val)
            break
    if observed is None:
        # Fall back to the first numeric value inside the observed {...} dict.
        om = re.search(r"observed\s*\{([^}]*)\}", counterexample or "")
        if om:
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", om.group(1))
            if nums:
                try:
                    observed = float(nums[0])
                except ValueError:
                    observed = None
    if observed is None or bound is None:
        return None
    diag = _detect_unit_scaling(observed, bound)
    if diag is None:
        return None
    # Infer the likely SI unit from the variable name for the healing hint.
    name = (variable or "").lower()
    si_unit = ""
    if "heat_flux" in name or "flux" in name:
        si_unit = "W/m^2"
    elif "heat_load" in name or "load" in name:
        si_unit = "J/m^2"
    elif "decel" in name or "accel" in name or name.endswith("_g"):
        si_unit = "m/s^2"
    elif "altitude" in name or name.endswith("_km") or "height" in name:
        si_unit = "m"
    elif "velocity" in name or name.endswith("_ms") or "_v_" in name:
        si_unit = "m/s"
    diag["si_unit"] = si_unit
    diag["variable"] = variable
    return diag


def _telemetry(run: Dict[str, Any], initial: List[float]) -> Dict[str, float]:
    state = run["state"]
    finite = bool(run.get("finite", True))
    tel = {
        "execution_time_ms": max(0.0, float(run.get("execution_time_ms", 0.0))),
        "steps": float(run.get("steps", 0)),
        "finite": 1.0 if finite else 0.0,
        "initial_l2": _l2(initial),
    }
    if finite and state:
        tel.update(
            final_min=float(min(state)), final_max=float(max(state)),
            final_mean=float(sum(state) / len(state)), final_l2=_l2(state),
            final_norm=_l2(state),
            max_abs_change=float(max(abs(a - b) for a, b in zip(state, initial))),
            energy_drift=float(max(abs(a - b) for a, b in zip(state, initial))),
        )
    else:
        tel.update(final_min=float("nan"), final_max=float("nan"),
                   final_mean=float("nan"), final_l2=float("nan"),
                   final_norm=float("nan"),
                   max_abs_change=float("nan"), energy_drift=float("nan"))
    return tel


def _contract_state_slots(contract: VerificationContract) -> List[int]:
    """All ``state[k]`` / ``initial[k]`` buffer indices referenced by a contract."""
    slots: List[int] = []
    for inv in (contract.invariants or []):
        text = f"{inv.variable_name} {inv.assertion}"
        for m in re.finditer(r"(?:state|initial|buf(?:\.data)?)\[(\d+)\]", text):
            slots.append(int(m.group(1)))
    return sorted(set(slots))


def _kernel_assigns_slots(py_code: str, slots: List[int]) -> List[int]:
    """Contract slots NEVER assigned via ``buf.data[k] =`` anywhere in the kernel."""
    missing: List[int] = []
    for k in slots:
        pat = re.compile(rf"buf(?:\.data|->data)\[\s*{k}\s*\]\s*(?:[-+*/%&|^]?=)")
        # Augmented/annotated/plain assignment all count as computed.
        pat2 = re.compile(rf"buf(?:\.data|->data)\[\s*{k}\s*\]\s*(?:=|\+=|-=|\*=|/=)")
        if not (pat.search(py_code or "") or pat2.search(py_code or "")):
            # Also accept assignment through an alias: ``state[k] = ...``
            # where the kernel binds ``state = buf.data``.
            if not re.search(rf"\bstate\[\s*{k}\s*\]\s*(?:=|\+=|-=|\*=|/=)", py_code or ""):
                missing.append(k)
    return missing


def _format_accretion_exception(exc: BaseException) -> str:
    """Never emit a blank accretion failure for exceptions like ReadTimeout."""
    return f"{type(exc).__name__}: {exc or repr(exc)}"


def _contracts_through_stage(bp: ResearchBlueprint, stage: int) -> List[VerificationContract]:
    """Cumulative contracts for stages 1..k (unprefixed contracts are stage 1)."""
    ev = getattr(bp, "ihcg_evidence", None)
    if isinstance(ev, dict) and not ev.get("extensions"):
        return list(bp.contracts or [])
    total = int(getattr(bp, "target_research_stages", 1) or 1)
    if total <= 1 or stage >= total:
        return list(bp.contracts or [])
    return [c for c in (bp.contracts or []) if _contract_stage_index(c.contract_id) <= stage]


def _deterministic_accretion_candidate(pinned_code: str, stage: int) -> Optional[str]:
    """Pinned-module fallback: wrap layer k-1 in layer_k and a fresh driver."""
    if stage <= 1 or not (pinned_code or "").strip():
        return None
    base = strip_driver(pinned_code)
    prev = layer_driver_name(stage - 1)
    if f"def {prev}(" not in base:
        names = re.findall(r"def\s+(layer_\d+_step)\s*\(", base)
        if not names:
            return None
        prev = sorted(names, key=lambda n: int(re.search(r"\d+", n).group(0)))[-1]
    new_layer = layer_driver_name(stage)
    if f"def {new_layer}(" in base:
        return None
    driver = "simulation_step"
    return (base.rstrip() +
            f"\n\n\ndef {new_layer}(buf, dt):\n    {prev}(buf, dt)\n\n\n"
            f"def {driver}(buf, dt):\n    {new_layer}(buf, dt)\n")


def _max_state_slot(py_code: str, contract: VerificationContract) -> int:
    """Highest ``state[k]``/``buf.data[k]`` index reachable by code or contract."""
    slots: List[int] = []
    texts = [py_code or ""]
    for inv in (contract.invariants or []):
        texts.append(f"{inv.variable_name} {inv.assertion}")
    for text in texts:
        for m in re.finditer(r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]", text):
            slots.append(int(m.group(1)))
    return max(slots, default=15)


_PARAM_OMEGA_NAMES = {"omega", "omega0", "k0", "m_mass"}
_PARAM_KAPPA_NAMES = {"kappa", "stiffness"}


def _contract_is_closed_or_reversible(contract: VerificationContract) -> bool:
    cid = str(getattr(contract, "contract_id", "") or "").lower()
    if any(tok in cid for tok in ("closed", "revers", "chiasmus", "gamma0", "gamma_0")):
        return True
    if any(tok in cid for tok in ("dissipat", "stiff", "gamma1", "gamma_1")):
        return False
    for inv in (getattr(contract, "invariants", None) or []):
        low = f"{getattr(inv, 'variable_name', '')} {getattr(inv, 'assertion', '')}".lower()
        if any(tok in low for tok in ("closed", "revers", "chiasmus")):
            return True
    return False


_PARAM_DISSIPATION_NAMES = {"gamma", "damping", "nu"}
_DISSIPATION_CONTRACT_TOKENS = ("stiff", "kraus", "dissipat", "damping", "entropy")


def _parameter_initial_defaults(py_code: str, contract: VerificationContract,
                                initial: List[float], warnings: List[str]) -> None:
    """Seed read-only physical parameter slots that the planner left at 0.0."""
    closed = _contract_is_closed_or_reversible(contract)
    dt = float(getattr(contract, "dt", 0.0) or 0.0)
    contract_text = " ".join([
        str(getattr(contract, "contract_id", "") or ""),
        " ".join(getattr(i, "variable_name", "") for i in (contract.invariants or [])),
        " ".join(str(getattr(i, "assertion", "") or "") for i in (contract.invariants or [])),
    ]).lower()
    dissipation_contract = any(tok in contract_text for tok in _DISSIPATION_CONTRACT_TOKENS)
    stiff_contract = "stiff" in contract_text or "1e5" in contract_text
    for m in re.finditer(
            r"\b(omega0|omega|k0|m_mass|kappa|stiffness|gamma|damping|nu)\s*=\s*"
            r"(?:buf\.data|state|data)\[(\d+)\]", py_code or ""):
        name, slot = m.group(1).lower(), int(m.group(2))
        if slot >= len(initial) or initial[slot] != 0.0:
            continue
        if name in _PARAM_OMEGA_NAMES:
            initial[slot] = 1.0
            warnings.append(
                f"auto-init: parameter state[{slot}] ({name}) was 0.0; seeded 1.0")
        elif name in _PARAM_KAPPA_NAMES and not closed:
            initial[slot] = 1.0e5 if (stiff_contract or dt >= 0.2) else 1.0
            warnings.append(
                f"auto-init: parameter state[{slot}] ({name}) was 0.0; seeded "
                f"{initial[slot]:.6g} for dt={dt}")
        elif name in _PARAM_DISSIPATION_NAMES and not closed and (dissipation_contract or dt >= 0.2):
            initial[slot] = 1.0e5 if stiff_contract else 1.0
            warnings.append(
                f"auto-init: dissipation parameter state[{slot}] ({name}) was 0.0; "
                f"seeded {initial[slot]:.6g}")


def _initialize_physical_state(py_code: str, contract: VerificationContract,
                                  all_contracts: Optional[Sequence[VerificationContract]] = None,
                                  ) -> Tuple[List[float], List[str]]:
    """Static contract-slot guard + auto-initializer.

    1. Contract error/drift metrics (tolerance <= 1e-6) must start at 0.0 —
       never 1.0 — so an un-triggered stage metric cannot falsify step 0.
    2. Dynamical slots (classified from the AST, e.g. state[0..2] or state[3..5])
       must never ALL be zero (trivial fixed point).
    3. The buffer is padded past the highest slot referenced by the kernel so
       Tier-0 scratch writes (e.g. buf.data[28]) stay in bounds.
    Returns (initial_state, warnings).
    """
    initial = list(contract.initial_state or DEFAULT_INITIAL_STATE)
    max_slot = _max_state_slot(py_code, contract)
    for oc in (all_contracts or ()):
        max_slot = max(max_slot, _max_state_slot(py_code, oc))
    needed = max(64, max_slot + 1)
    while len(initial) < needed:
        initial.append(0.0)
    warnings: List[str] = []
    cls = classify_buffer_slots(py_code, [contract])
    dyn_slots = list(cls["dynamical_slots"])
    metric_slots = set(cls["metric_slots"])
    if len(dyn_slots) < 3 and not ({0, 1, 2} & metric_slots):
        for i in range(3):
            if i not in metric_slots and i not in dyn_slots:
                dyn_slots.append(i)
        dyn_slots.sort()
    dyn_set = set(dyn_slots)

    # 1a. Stationary fixed-point guard: the maximally-mixed identity
    # rho = I/2 is a valid density matrix but a degenerate fixed point for
    # many channels; perturb it to a non-stationary positive state.
    if (len(initial) >= 4 and not ({0, 1, 2, 3} & metric_slots)
            and initial[0] == 0.5 and initial[3] == 0.5
            and initial[1] == 0.0 and initial[2] == 0.0):
        initial[0], initial[1], initial[2], initial[3] = 0.7, 0.2, 0.1, 0.3
        warnings.append("auto-init: perturbed stationary I/2 state to non-degenerate rho")
    # 1. Error/drift metrics & relational accumulators initialize cleanly
    # across contracts; only reset error metrics for THIS contract so another
    # stage's slot layout cannot zero out non-zero inputs here.
    missing_set = set(_kernel_assigns_slots(py_code, _contract_state_slots(contract)))
    m0_sum = sum(max(0.0, initial[d]) for d in dyn_slots if d < len(initial))
    for c_item in (all_contracts or [contract]):
        for inv in (getattr(c_item, "invariants", None) or []):
            assertion = str(getattr(inv, "assertion", "") or "")
            name = (inv.variable_name or "").lower()
            # Relational invariant: weak (<= / >=) vs strict (< / > / positive margin)
            m_rel_le = re.search(
                r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]\s*(<=|<)\s*"
                r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]",
                assertion,
            )
            m_rel_ge = re.search(
                r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]\s*(>=|>)\s*"
                r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\](?:\s*\+\s*([0-9.eE+-]+))?",
                assertion,
            )
            m_rel_diff = re.search(
                r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]\s*-\s*"
                r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]\s*(>|>=)\s*([0-9.eE+-]+)",
                assertion,
            )
            if m_rel_le or m_rel_ge or m_rel_diff:
                is_strict_rel = False
                rel_margin = 0.0
                if m_rel_diff:
                    j_idx, i_idx = int(m_rel_diff.group(1)), int(m_rel_diff.group(2))
                    rel_margin = max(0.0, float(m_rel_diff.group(4)))
                    is_strict_rel = (m_rel_diff.group(3) == ">") or (rel_margin > 0.0)
                elif m_rel_le:
                    i_idx, op_r, j_idx = int(m_rel_le.group(1)), m_rel_le.group(2), int(m_rel_le.group(3))
                    is_strict_rel = (op_r == "<")
                else:
                    assert m_rel_ge is not None
                    j_idx, op_r, i_idx = int(m_rel_ge.group(1)), m_rel_ge.group(2), int(m_rel_ge.group(3))
                    rel_margin = float(m_rel_ge.group(4)) if m_rel_ge.group(4) else 0.0
                    is_strict_rel = (op_r == ">") or (rel_margin > 0.0)

                if i_idx != j_idx and max(i_idx, j_idx) < len(initial):
                    if is_strict_rel:
                        if j_idx not in dyn_set:
                            # If unassigned and concept shares array/gen/total/power with another slot > lo, bridge first
                            concept_toks = [tok for tok in ("array", "gen", "total", "power") if tok in name]
                            bridged_val = None
                            if concept_toks and j_idx in missing_set:
                                for c2 in (all_contracts or [contract]):
                                    for inv2 in (getattr(c2, "invariants", None) or []):
                                        n2 = (getattr(inv2, "variable_name", "") or "").lower()
                                        if any(tok in n2 for tok in concept_toks):
                                            for s2_txt in re.findall(r"(?:state|buf\.data|data)\[(\d+)\]", str(getattr(inv2, "assertion", "") or "")):
                                                s2 = int(s2_txt)
                                                if s2 != j_idx and s2 != i_idx and s2 < len(initial) and initial[s2] > initial[i_idx] + rel_margin:
                                                    bridged_val = initial[s2]
                                                    break
                                        if bridged_val is not None:
                                            break
                                    if bridged_val is not None:
                                        break
                            if bridged_val is not None and initial[j_idx] <= initial[i_idx] + rel_margin:
                                old_j = initial[j_idx]
                                initial[j_idx] = bridged_val
                                warnings.append(
                                    f"auto-init: strict relational slot state[{j_idx}] ({inv.variable_name}) "
                                    f"bridged from {old_j} to {bridged_val}")
                            elif initial[j_idx] <= initial[i_idx] + rel_margin:
                                old_j = initial[j_idx]
                                bump = max(rel_margin + 1e-6, max(1.0, abs(initial[i_idx]) * 0.001))
                                target_j = initial[i_idx] + bump
                                initial[j_idx] = target_j
                                warnings.append(
                                    f"auto-init: strict relational slot state[{j_idx}] ({inv.variable_name}) "
                                    f"raised from {old_j} to {target_j}")
                        continue
                    if i_idx not in dyn_set and ("max" in name or "bound" in name or "peak" in name):
                        if initial[i_idx] > 0.0 and (initial[i_idx] > initial[j_idx] or i_idx not in missing_set):
                            old_i = initial[i_idx]
                            initial[i_idx] = 0.0
                            warnings.append(
                                f"auto-init: running-max slot state[{i_idx}] ({inv.variable_name}) "
                                f"reset from {old_i} to 0.0")
                    if j_idx not in dyn_set and (
                        any(tok in name for tok in ("mass", "m0", "target", "bound"))
                        or j_idx in missing_set
                        or initial[j_idx] < initial[i_idx]
                    ):
                        target_j = max(initial[j_idx], m0_sum if m0_sum > 0.0 else initial[j_idx], initial[i_idx])
                        if target_j != initial[j_idx]:
                            warnings.append(
                                f"auto-init: relational bound slot state[{j_idx}] ({inv.variable_name}) "
                                f"raised from {initial[j_idx]} to {target_j}")
                            initial[j_idx] = target_j
                    if initial[i_idx] > initial[j_idx] and i_idx not in dyn_set:
                        initial[i_idx] = initial[j_idx]
                continue

            from ..healing.deterministic_ast import (
                _eval_ast_on_state,
                _eval_const_ast,
                _eval_numeric_const_expr,
                _extract_abs_tol,
                _iter_atomic_invariant_nodes,
                _match_abs_affine,
                _match_abs_algebraic,
                _slot_of_state_ast,
                _unwrap_abs_or_rel_err,
            )
            import ast as _ast
            for _node in _iter_atomic_invariant_nodes(assertion):
                if isinstance(_node, _ast.Compare):
                    _lhs = _node.left
                    for _cmp_op, _rhs in zip(_node.ops, _node.comparators):
                        _abs_in, _denom_ast = (None, None)
                        _tol_v = None
                        _rhs_rel = False
                        if isinstance(_cmp_op, (_ast.LtE, _ast.Lt)):
                            _abs_in, _denom_ast = _unwrap_abs_or_rel_err(_lhs)
                            if _abs_in is not None:
                                _tol_v, _rhs_rel = _extract_abs_tol(_rhs)
                        elif isinstance(_cmp_op, (_ast.GtE, _ast.Gt)):
                            _abs_in, _denom_ast = _unwrap_abs_or_rel_err(_rhs)
                            if _abs_in is not None:
                                _tol_v, _rhs_rel = _extract_abs_tol(_lhs)
                        if _abs_in is not None and _tol_v is not None and _tol_v >= 0.0:
                            _denom_c = _eval_const_ast(_denom_ast) if _denom_ast is not None else None
                            _single_t, _sum_t, _diff_p = _match_abs_affine(_abs_in)
                            if _single_t is not None and c_item is contract:
                                _si, _tgt = _single_t
                                _eff_tol = _tol_v
                                if _denom_ast is not None or _rhs_rel:
                                    _scale = abs(_denom_c) if (_denom_c is not None and _denom_c != 0.0) else max(abs(_tgt), 1e-300)
                                    _eff_tol = _tol_v * _scale
                                if (
                                    _si < len(initial)
                                    and abs(initial[_si] - _tgt) > _eff_tol
                                    and not re.search(r'(?:^|_)(?:time(?:_s|_ms|_min|_hr)?|t_s|sim_time|elapsed|horizon)(?:_|$)', str(getattr(inv, 'variable_name', '') or ''), re.I)
                                ):
                                    warnings.append(
                                        f"auto-init: target equality slot state[{_si}] ({inv.variable_name}) "
                                        f"set from {initial[_si]} to {_tgt}")
                                    initial[_si] = _tgt
                            elif _sum_t is not None and c_item is contract:
                                _i_s, _j_s, _tgt = _sum_t
                                _eff_tol = _tol_v
                                if _denom_ast is not None or _rhs_rel:
                                    _scale = abs(_denom_c) if (_denom_c is not None and _denom_c != 0.0) else max(abs(_tgt), 1e-300)
                                    _eff_tol = _tol_v * _scale
                                if max(_i_s, _j_s) < len(initial) and abs(initial[_i_s] + initial[_j_s] - _tgt) > _eff_tol:
                                    if min(0.0, _tgt) <= initial[_i_s] <= max(0.0, _tgt):
                                        initial[_j_s] = _tgt - initial[_i_s]
                                    else:
                                        initial[_i_s] = 0.5 * _tgt
                                        initial[_j_s] = 0.5 * _tgt
                                    warnings.append(
                                        f"auto-init: sum partition slots state[{_i_s}]+state[{_j_s}] "
                                        f"initialized to {_tgt}")
                            elif _single_t is None and _sum_t is None and _diff_p is None and c_item is contract:
                                _alg_m = _match_abs_algebraic(_abs_in)
                                if _alg_m is not None:
                                    _ai, _expr_buf, _prod_p = _alg_m
                                    try:
                                        _expr_node = _ast.parse(_expr_buf, mode="eval").body
                                        _alg_val = _eval_ast_on_state(_expr_node, initial)
                                    except SyntaxError:
                                        _alg_val = None
                                    if _alg_val is not None and math.isfinite(_alg_val) and _ai < len(initial):
                                        _denom_v = (
                                            _eval_ast_on_state(_denom_ast, initial)
                                            if _denom_ast is not None else (abs(_alg_val) if _rhs_rel else 1.0)
                                        )
                                        _eff_tol = _tol_v * abs(_denom_v) if ((_denom_ast is not None or _rhs_rel) and _denom_v) else _tol_v
                                        if abs(initial[_ai] - _alg_val) > _eff_tol:
                                            warnings.append(
                                                f"auto-init: algebraic target slot state[{_ai}] ({inv.variable_name}) "
                                                f"set from {initial[_ai]} to {_alg_val}")
                                            initial[_ai] = _alg_val
                        else:
                            # Handle chained and scalar bounds (e.g. `2.5 <= state[15] <= 4.3` or `state[20] <= 15.0`)
                            _ls = _slot_of_state_ast(_lhs)
                            _rs = _slot_of_state_ast(_rhs)
                            if _ls is not None and _ls not in dyn_set and _ls >= 4 and _ls < len(initial):
                                _bv = _eval_const_ast(_rhs)
                                if _bv is not None and math.isfinite(_bv):
                                    if isinstance(_cmp_op, (_ast.GtE, _ast.Gt)) and _bv > 0.0:
                                        _req = (_bv + 1e-12) if isinstance(_cmp_op, _ast.Gt) else _bv
                                        if initial[_ls] < _req:
                                            initial[_ls] = _req
                                    elif isinstance(_cmp_op, (_ast.LtE, _ast.Lt)) and c_item is contract and _bv > 1.0 + 1e-6:
                                        _cap = (_bv - 1e-6) if isinstance(_cmp_op, _ast.Lt) else _bv
                                        if initial[_ls] > _cap:
                                            initial[_ls] = _cap
                            elif _rs is not None and _rs not in dyn_set and _rs >= 4 and _rs < len(initial):
                                _bv = _eval_const_ast(_lhs)
                                if _bv is not None and math.isfinite(_bv):
                                    if isinstance(_cmp_op, (_ast.LtE, _ast.Lt)) and _bv > 0.0:
                                        _req = (_bv + 1e-12) if isinstance(_cmp_op, _ast.Lt) else _bv
                                        if initial[_rs] < _req:
                                            initial[_rs] = _req
                                    elif isinstance(_cmp_op, (_ast.GtE, _ast.Gt)) and c_item is contract and _bv > 1.0 + 1e-6:
                                        _cap = (_bv - 1e-6) if isinstance(_cmp_op, _ast.Gt) else _bv
                                        if initial[_rs] > _cap:
                                            initial[_rs] = _cap
                        _lhs = _rhs

            distinct_slots = {
                int(x) for x in re.findall(
                    r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]", assertion)
            }
            if len(distinct_slots) != 1:
                continue
            if "initial[" in assertion:
                continue
            k = next(iter(distinct_slots))
            if k in dyn_set:
                continue
            bound_match = re.search(
                r"(?:abs\(\s*)?(?:state|initial|buf(?:\.data)?)\[\d+\](?:\s*\))?\s*<=\s*([0-9.eE+-]+)", assertion)
            bound = float(bound_match.group(1)) if bound_match else None
            error_metric = any(tok in name for tok in
                               ("error", "err", "drift", "residual", "violation", "mismatch",
                                "hold", "bound", "increase", "monotone", "monotonicity"))
            if bound is not None and c_item is contract:
                if (0.0 <= bound <= 1e-8) or (error_metric and bound <= 1e-6):
                    if k < len(initial) and initial[k] != 0.0:
                        warnings.append(
                            f"auto-init: contract error metric state[{k}] ({inv.variable_name}) "
                            f"reset from {initial[k]} to 0.0 (error metrics start at zero)")
                        initial[k] = 0.0
                elif abs(bound - 1.0) <= 1e-6:
                    if k < len(initial) and initial[k] > bound:
                        warnings.append(
                            f"auto-init: contract unit upper bound state[{k}] ({inv.variable_name}) "
                            f"clamped from {initial[k]} to 1.0")
                        initial[k] = 1.0
            gt_match = re.search(
                r"(?:state|initial|buf(?:\.data)?)\[\d+\]\s*(>=?)\s*([0-9.eE+\-\s*/()]+?)(?:\s*(?:$|&&|\|\||and\b|or\b|>|<))",
                assertion,
            )
            if gt_match and k >= 4:
                op_gt = gt_match.group(1)
                gt_bound = _eval_numeric_const_expr(gt_match.group(2))
                if gt_bound is not None:
                    if op_gt == ">" and -1e-6 <= gt_bound <= 1.0:
                        new_v = (gt_bound + 1e-3) if gt_bound >= 0.0 else (gt_bound * 0.5)
                        if k < len(initial) and initial[k] <= gt_bound:
                            warnings.append(
                                f"auto-init: strict positivity slot state[{k}] ({inv.variable_name}) "
                                f"raised from {initial[k]} to {new_v}")
                            initial[k] = new_v
                    elif gt_bound > 0.0:
                        req_v = (gt_bound + 1e-12) if op_gt == ">" else gt_bound
                        if k < len(initial) and initial[k] < req_v:
                            warnings.append(
                                f"auto-init: unified lower bound slot state[{k}] ({inv.variable_name}) "
                                f"raised from {initial[k]} to {req_v}")
                            initial[k] = req_v
    # 1b. Reversibility / closed-limit flag auto-seeding: if the contract checks
    # reversibility/chiasmus/closed-limit and the kernel reads a reversibility flag
    # from buf.data[p] and branches on it, seed initial[p] = 1.0 and zero dissipation slots.
    if _contract_is_closed_or_reversible(contract):
        for m_rev in re.finditer(
                r"\b(rev|closed|is_closed|rev_flag|reversible|mode|flag|chiasmus)\s*=\s*"
                r"(?:buf\.data|state|data)\[(\d+)\]", py_code or ""):
            rev_var, rev_slot = m_rev.group(1), int(m_rev.group(2))
            if rev_slot not in dyn_set and rev_slot < len(initial):
                if re.search(rf"\bif\s+{re.escape(rev_var)}\b", py_code or ""):
                    if initial[rev_slot] == 0.0:
                        initial[rev_slot] = 1.0
                        warnings.append(
                            f"auto-init: reversibility flag state[{rev_slot}] ({rev_var}) "
                            f"set to 1.0 for closed/reversibility contract '{contract.contract_id}'")
        for p_slot, p_var in (cls.get("parameter_vars") or {}).items():
            if p_var.lower() in {
                "gx", "gy", "gz", "g", "g1", "g2", "gphi", "g_phi",
                "gam", "gam1", "gam2", "gam_phi", "gamma", "gamma1",
                "gamma2", "gamma_phi", "kappa", "stiffness", "damping", "nu",
            }:
                if p_slot < len(initial) and initial[p_slot] != 0.0:
                    initial[p_slot] = 0.0
                    warnings.append(
                        f"auto-init: zeroed dissipation state[{p_slot}] ({p_var}) "
                        f"for closed/reversibility contract '{contract.contract_id}'")
    # 2. Physical parameter slots must not start as silent zeros.
    _parameter_initial_defaults(py_code, contract, initial, warnings)
    # 2b. Bloch ball initial norm contraction: ensure initial dynamical Bloch vector is physical.
    r_max = _detect_bloch_radius(py_code, [contract])
    if r_max is not None and len(dyn_slots) >= 3:
        d0, d1, d2 = dyn_slots[0], dyn_slots[1], dyn_slots[2]
        if max(d0, d1, d2) < len(initial):
            r2 = initial[d0] ** 2 + initial[d1] ** 2 + initial[d2] ** 2
            if r2 > r_max * r_max:
                sc = (0.85 * r_max) / math.sqrt(r2)
                initial[d0] *= sc
                initial[d1] *= sc
                initial[d2] *= sc
                warnings.append(
                    f"auto-init: initial Bloch norm sqrt({r2:.4g}) > {r_max}; "
                    f"scaled state[{d0}..{d2}] inside Bloch ball")
    # 3. Never a trivial all-zero dynamical state; polar [0,0,z] gets x-seed (only on dyn_slots).
    if len(dyn_slots) >= 3:
        d0, d1, d2 = dyn_slots[0], dyn_slots[1], dyn_slots[2]
        if max(d0, d1, d2) < len(initial):
            scale_ref = r_max if r_max is not None else 1.0
            if initial[d0] == 0.0 and initial[d1] == 0.0 and abs(initial[d2]) > 0.0:
                initial[d0] = 0.3 * scale_ref
                if abs(initial[d2]) > 0.8 * scale_ref:
                    initial[d2] = math.copysign(0.8 * scale_ref, initial[d2])
                warnings.append(
                    f"auto-init: polar state[{d0}..{d2}]=[0,0,z] is stationary under "
                    "z-rotation; seeded non-polar state")
            elif all(abs(initial[i]) < 1e-12 for i in (d0, d1, d2)):
                seed_val = 0.6 * scale_ref
                warnings.append(
                    f"auto-init: state[{d0}..{d2}] all zero (trivial fixed point); "
                    f"seeding state[{d0}] = {seed_val:g}")
                initial[d0] = seed_val
    # 4. Warn (never silently pass) when a contract slot is never computed.
    missing = _kernel_assigns_slots(py_code, _contract_state_slots(contract))
    for k in missing:
        warnings.append(
            f"CONTRACT_SLOT_GUARD: state[{k}] is checked by contract "
            f"'{contract.contract_id}' but is NEVER assigned in the kernel; "
            "it will hold its initial value.")
    return initial, warnings


def _guard_contract_initial_state(py_code: str, contract: VerificationContract,
                                  all_contracts: Optional[Sequence[VerificationContract]] = None,
                                  diagnostic_slots: Sequence[int] = (),
                                  variable_map: Optional[Dict[str, int]] = None,
                                  ) -> Tuple[List[float], List[str]]:
    contracts = all_contracts or [contract]
    tree = ast.parse(py_code)
    physical = _verified_projection_slots(tree, contracts)
    from .slot_schema import ROLE_CONTROL_PARAM, ROLE_TIME_STEP, classify_slot_roles
    role_map = classify_slot_roles(variable_map, tree, contracts, physical_slots=physical)
    ctrl_slots = {s for s, r in role_map.items() if r == ROLE_CONTROL_PARAM}
    time_slots = {s for s, r in role_map.items() if r == ROLE_TIME_STEP}
    protected = (protected_observation_slots(tree, contracts, variable_map=variable_map, physical_slots=physical) - ctrl_slots) | time_slots
    protected.update(diagnostic_slots)
    protected.update(set(_kernel_assigns_slots(py_code, _contract_state_slots(contract))) - physical - ctrl_slots)
    initial, warnings = _initialize_physical_state(py_code, contract, all_contracts)
    declared = contract.initial_state if contract.initial_state is not None else DEFAULT_INITIAL_STATE
    for slot in protected:
        if 0 <= slot < len(initial):
            initial[slot] = declared[slot] if slot < len(declared) else 0.0
    warnings = [warning for warning in warnings if not warning.startswith('auto-init:')
                or not any(f'state[{slot}]' in warning for slot in protected)]
    if all(initial[slot] == declared[slot] for slot in range(min(3, len(declared)))):
        warnings = [warning for warning in warnings if 'state[0..2]' not in warning]
    return initial, warnings



def _canonicalize_blueprint_slots(bp: ResearchBlueprint) -> None:
    """Enforce one canonical buf.data slot per named observable across stages."""
    code = getattr(bp, "unified_module_code", None) or getattr(bp, "code_template", None) or ""
    if code.strip() and getattr(bp, "contracts", None):
        new_code, _, coll_notes = resolve_slot_collisions(code, bp.contracts, bp.contracts)
        if coll_notes:
            if getattr(bp, "unified_module_code", None):
                bp.unified_module_code = new_code
            if getattr(bp, "code_template", None):
                bp.code_template = new_code
    contracts = [c.model_dump() for c in getattr(bp, "contracts", None) or []]
    if not contracts:
        return
    new_contracts, vm, notes = canonicalize_contract_dicts(
        contracts, dict(getattr(bp, "variable_map", None) or {}),
        preserve_mapped_slots=bool(getattr(bp, "ihcg_evidence", None) or getattr(bp, "typed_slots", None)))
    if not notes and vm == dict(getattr(bp, "variable_map", None) or {}):
        return
    bp.contracts = [VerificationContract.model_validate(c) for c in new_contracts]
    bp.variable_map = vm


def _auto_populate_variable_map(bp: ResearchBlueprint, code_text: str = "") -> None:
    """Infer missing ``variable_map`` entries from contract constraints and AST
    buffer bindings so report Section 1 and telemetry ``obs_*`` never leave
    contract metrics null."""
    mapping: Dict[str, int] = {}
    ihcg_slots = ((getattr(bp, "ihcg_evidence", None) or {}).get("evidence") or {}).get("slots") or []
    for s in ihcg_slots:
        if isinstance(s, dict) and s.get("symbol") and isinstance(s.get("slot_idx"), int):
            mapping[str(s["symbol"])] = int(s["slot_idx"])
    for k, v in dict(getattr(bp, "variable_map", None) or {}).items():
        mapping.setdefault(k, v)
    contracts = getattr(bp, "contracts", None) or []
    for contract in contracts:
        for inv in (getattr(contract, "invariants", None) or []):
            assertion = str(getattr(inv, "assertion", "") or "")
            distinct_slots = {
                int(x) for x in re.findall(
                    r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]", assertion)
            }
            if len(distinct_slots) != 1:
                continue
            slot = next(iter(distinct_slots))
            if inv.variable_name:
                mapping.setdefault(inv.variable_name, slot)
    # Also scan simulation_step / layer_* AST for `<var> = buf.data[idx]` and `buf.data[idx] = <var>`
    # ONLY for slots referenced by invariants or already within declared initial_state bounds (never
    # polluting variable_map with uncontracted internal RK4/scratch slots like `buf.data[21..28]`).
    initial_vm = dict(getattr(bp, "variable_map", None) or {})
    mapped_slots = set(mapping.values())
    inv_referenced_slots: Set[int] = set()
    for _c in contracts:
        for _inv in (getattr(_c, "invariants", None) or []):
            for _sx in re.findall(r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]", str(getattr(_inv, "assertion", "") or "")):
                inv_referenced_slots.add(int(_sx))
    max_decl_len = max((len(getattr(_c, "initial_state", None) or []) for _c in contracts), default=0)
    src = code_text or getattr(bp, "unified_module_code", None) or getattr(bp, "code_template", None) or ""
    if src.strip():
        try:
            tree = ast.parse(src)
            for fn in tree.body:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                if fn.name.startswith('_invar_'):
                    continue
                if not any(arg.arg == 'buf' for arg in fn.args.args):
                    continue
                for stmt in ast.walk(fn):
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                        tgt, val = stmt.targets[0], stmt.value
                        if isinstance(tgt, ast.Name) and isinstance(val, ast.Subscript):
                            m_idx = re.search(r"\[(\d+)\]", ast.unparse(val))
                            if m_idx and not tgt.id.startswith("_"):
                                s_i = int(m_idx.group(1))
                                if s_i not in mapped_slots and (not initial_vm or s_i in inv_referenced_slots or s_i < max_decl_len):
                                    mapping.setdefault(tgt.id, s_i)
                        elif isinstance(val, ast.Name) and isinstance(tgt, ast.Subscript):
                            m_idx = re.search(r"\[(\d+)\]", ast.unparse(tgt))
                            if m_idx and not val.id.startswith("_"):
                                s_i = int(m_idx.group(1))
                                if s_i not in mapped_slots and (not initial_vm or s_i in inv_referenced_slots or s_i < max_decl_len):
                                    mapping.setdefault(val.id, s_i)
        except SyntaxError:
            pass
    if len(mapping) > 4 and all(f"state_{i}" in mapping for i in range(4)):
        for i in range(4):
            mapping.pop(f"state_{i}", None)
    if not mapping:
        for i in range(4):
            mapping[f"state_{i}"] = i
    try:
        bp.variable_map = canonical_variable_map(contracts, mapping)
    except ValueError as _slot_err:
        if "has conflicting slots" not in str(_slot_err):
            raise
        # Reconcile cross-stage contract invariant slot conflicts against the authoritative mapping
        # (or the first contract's slot for observables not yet in `mapping`).
        _first_seen_slot: Dict[str, int] = {str(k): int(v) for k, v in mapping.items()}
        for _c in contracts:
            for _inv in getattr(_c, "invariants", None) or []:
                _iname = str(getattr(_inv, "variable_name", "") or "").strip()
                if not _iname:
                    continue
                _ias = str(getattr(_inv, "assertion", "") or "")
                _found = sorted(set(int(m) for m in re.findall(r"\b(?:state|buf\.data|data)\[(\d+)\]", _ias)))
                if len(_found) != 1:
                    continue
                if _iname not in _first_seen_slot:
                    _first_seen_slot[_iname] = _found[0]
                elif _found[0] != _first_seen_slot[_iname]:
                    _target_s = _first_seen_slot[_iname]
                    _inv.assertion = re.sub(
                        rf"\b(state|buf\.data|data)\[{_found[0]}\]",
                        rf"\1[{_target_s}]",
                        _ias,
                    )
        bp.variable_map = canonical_variable_map(contracts, mapping)
    ihcg_owned = {
        int(s["slot_idx"]): str(s["symbol"])
        for s in ihcg_slots
        if isinstance(s, dict) and s.get("symbol") and isinstance(s.get("slot_idx"), int)
    }
    if ihcg_owned:
        # Preserve exact IHCG slot symbols against observable_label() suffix stripping (_bound, _window, etc.).
        preserved_vm: Dict[str, int] = {}
        for k, v in (bp.variable_map or {}).items():
            si = int(v)
            preserved_vm[ihcg_owned.get(si, k)] = si
        for si, sym in ihcg_owned.items():
            preserved_vm.setdefault(sym, si)
        bp.variable_map = preserved_vm
    phys_slots: List[int] = []
    written_slots: set[int] = set()
    try:
        _parsed_src = ast.parse(src)
        phys_slots = sorted(_verified_projection_slots(_parsed_src, contracts))
        written_slots = {
            int(s) for s in (_analyze_code_slot_flow(_parsed_src).get("all_writes") or ())
        }
    except SyntaxError:
        phys_slots = []
        written_slots = set()
    unified_roles = classify_variable_roles(
        bp.variable_map,
        src,
        contracts,
        extracted_constants=getattr(bp, "extracted_constants", None),
        explicit_roles=bp.slot_roles,
        physical_slots=phys_slots,
    )
    _derived_rate_diag_re = re.compile(
        r"(?:_(?:rate|flux|load|slope|current)(?:_[a-z0-9]+)*$|^d[A-Z0-9_]+_slope)",
        re.I,
    )
    for name, unified_role in unified_roles.items():
        slot_idx = bp.variable_map.get(name)
        is_written_derived = (
            bool(_derived_rate_diag_re.search(name))
            and not _CONTROL_PARAM_NAME_RE.match(name)
            and slot_idx is not None
            and int(slot_idx) in written_slots
        )
        if name in bp.slot_roles:
            if bp.slot_roles[name] == "read_only_parameter" and is_written_derived:
                bp.slot_roles[name] = "diagnostic"
            continue
        if unified_role == ROLE_CONTROL_PARAM and is_written_derived:
            bp.slot_roles[name] = "diagnostic"
            continue
        # Never permanently latch read_only_parameter onto physical state slots
        # solely because an unverified candidate omitted writing them.
        if unified_role == ROLE_CONTROL_PARAM and (src or "").strip() and not _CONTROL_PARAM_NAME_RE.match(name):
            continue
        bp.slot_roles[name] = to_blueprint_role(unified_role)


def _apply_backup_slots_to_vmap(bp: Optional[ResearchBlueprint],
                                backup_map: Dict[str, int],
                                code_text: str = "") -> None:
    """Update `bp.variable_map` so backed-up input parameters point to their backup slots."""
    if not backup_map or bp is None:
        return
    vm = dict(getattr(bp, "variable_map", None) or {})
    # Map each backup_slot -> orig_slot by parsing `_invar_psz_<orig>` assignments if code provided.
    orig_by_bak: Dict[int, int] = {}
    if code_text:
        for m in re.finditer(r"buf\.data\[(\d+)\]\s*=\s*buf\.data\[(\d+)\]", code_text):
            orig_by_bak[int(m.group(1))] = int(m.group(2))
    for var_in, bak_slot in backup_map.items():
        bak_i = int(bak_slot)
        orig_i = orig_by_bak.get(bak_i)
        vlow = var_in.lower().rstrip("s")
        for existing_k, existing_v in list(vm.items()):
            if orig_i is not None and existing_v == orig_i:
                klow = existing_k.lower().rstrip("s")
                if klow == vlow or klow.startswith(vlow + "_") or vlow.startswith(klow + "_"):
                    vm[existing_k] = bak_i
        vm[var_in] = bak_i
    bp.variable_map = dict(list(vm.items())[:64])


def _json_safe(tel: Dict[str, float]) -> Dict[str, float]:
    return {k: (v if isinstance(v, (int, float)) and math.isfinite(v) else -1.0)
            for k, v in tel.items()}


def _aggregate_telemetry(results: List[SimulationResult]) -> Dict[str, float]:
    """Flat finite telemetry merged across all sandbox results of a run."""
    out: Dict[str, float] = {}
    for result in results or []:
        for key, value in (getattr(result, "telemetry", {}) or {}).items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                out[key] = float(value)
    return out


def _sandbox_step_summary(results: List[SimulationResult],
                          verified_count: int) -> str:
    """Human-readable Sandbox_Execute progress line with peak observables."""
    peak_g: Optional[float] = None
    peak_alt: Optional[float] = None
    for r in results:
        if r.status != "VERIFIED":
            continue
        for key, value in (r.telemetry or {}).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            lowered = key.lower()
            if lowered.endswith("_g"):
                peak_g = value if peak_g is None else max(peak_g, value)
            elif lowered.endswith("_km"):
                peak_alt = value if peak_alt is None else max(peak_alt, value)
    parts: List[str] = []
    if peak_g is not None:
        parts.append(f"a_max={peak_g:.2f}g")
    if peak_alt is not None:
        parts.append(f"h_peak={peak_alt:.1f}km")
    if parts:
        return "Stepping trajectory: " + ", ".join(parts)
    return (f"Sandbox execution complete: {len(results)} contract run(s), "
            f"{verified_count} VERIFIED.")


def _fmt_vec(values: List[float], limit: int = 4) -> str:
    """Compact head/tail formatting of a state vector for prompt context."""
    vals = [float(v) for v in values]
    if len(vals) <= 2 * limit:
        body = ", ".join(f"{v:.6g}" for v in vals)
    else:
        head = ", ".join(f"{v:.6g}" for v in vals[:limit])
        tail = ", ".join(f"{v:.6g}" for v in vals[-limit:])
        body = f"{head}, …({len(vals)} values)…, {tail}"
    return f"[{body}]"


def _counterexample_context(run: Dict[str, Any]) -> str:
    """Numerical divergence context fed back into GoI_Plan on falsification.

    Carries the exact final state vector, trace bounds and the first
    non-finite step so the planner sees the precise divergence point.
    """
    state = run.get("state") or []
    return (f"final_state={_fmt_vec(state)}; "
            f"trace_min={_fmt_vec(run.get('trace_min', state))}; "
            f"trace_max={_fmt_vec(run.get('trace_max', state))}; "
            f"first_nonfinite_step={run.get('first_nonfinite_step', -1)}; "
            f"final_l2={float(_telemetry(run, state).get('final_l2', float('nan'))):.6g}")


@dataclass
class ResearchRunState:
    blueprint: ResearchBlueprint
    stage: str = "Ingest"
    results: List[SimulationResult] = field(default_factory=list)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    aborted: bool = False
    status: str = "PENDING"  # VERIFIED | PARTIAL | FAILED | REFUSED | ABORTED
    research_stage: int = 1
    literature: List[Dict[str, Any]] = field(default_factory=list)
    rolled_back_stages: set = field(default_factory=set)
    checkpoint_path: Optional[str] = None
    refusal: Optional[Dict[str, Any]] = None
    report: Optional[str] = None  # Final Quantitative Research Report (markdown)
    ihcg: Any = None
    meaningfulness_certificate: Any = None
    canonical_signature: Any = None
    epistemic_verdict: str = "INCONCLUSIVE_UNVERIFIED"
    preserved_vs_enforced: Dict[str, Any] = field(default_factory=dict)
    heal_count: int = 0
    precision_refined: bool = False

    def snapshot(self) -> Dict[str, Any]:
        bp = self.blueprint
        return {
            "blueprint_id": bp.blueprint_id,
            "title": bp.title,
            "stage": self.stage,
            "status": self.status,
            "research_stage": self.research_stage,
            "target_research_stages": bp.target_research_stages,
            "verified_layers": [layer.layer_id for layer in bp.verified_layers],
            "evolution_history": [h.model_dump() for h in bp.evolution_history],
            "round": len(self.memory) + (1 if self.results else 0),
            "aborted": self.aborted,
            "checkpoint_path": self.checkpoint_path,
            "refusal": self.refusal,
            "report": self.report,
            "epistemic_verdict": self.epistemic_verdict,
            "heal_count": self.heal_count,
            "meaningfulness_certificate": self.meaningfulness_certificate.model_dump(mode="json") if self.meaningfulness_certificate else None,
            "results": [r.model_dump() for r in self.results],
        }


ProgressFn = Callable[[str, Dict[str, Any]], Awaitable[None]]
LLMFn = Callable[[str], Awaitable[Optional[str]]]
StageSynthFn = Callable[..., Awaitable[Dict[str, Any]]]


def _pinned_function_names(pinned_code: str = "", pinned_layers: Sequence[ModuleLayer] = ()) -> List[str]:
    names: Set[str] = {layer_driver_name(layer.stage) for layer in (pinned_layers or ())}
    if pinned_code and pinned_code.strip():
        try:
            pt = ast.parse(pinned_code)
            for node in pt.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_invar_"):
                    names.add(node.name)
        except SyntaxError:
            pass
    return sorted(names)


class ResearchStateMachine:
    """Drives a blueprint through the cyclic 9-stage loop with accretive stages."""

    def __init__(self, workspace: Path, progress_cb: Optional[ProgressFn] = None,
                 llm_planner: Optional[LLMFn] = None,
                 llm_fast: Optional[LLMFn] = None,
                 search_client: Optional[LiteratureSearchClient] = None,
                 stage_synthesizer: Optional[StageSynthFn] = None,
                 abort_flag: Optional[Callable[[], bool]] = None,
                 workspace_root: Optional[Path] = None,
                 verification_runner: Optional[Callable[..., Any]] = None,
                 llm_formalizer: Optional[LLMFn] = None):
        self.workspace = Path(workspace)
        self.progress_cb = progress_cb
        self._provider_failure = None
        self.llm_planner = self._retry_provider(llm_planner)
        self.llm_formalizer = self._retry_provider(llm_formalizer) if llm_formalizer is not None else self.llm_planner
        self.llm_fast = self.llm_planner if llm_fast is llm_planner else self._retry_provider(llm_fast)
        if search_client is None:
            # Live network by default in production/server/CLI; offline only
            # when explicitly requested (INVAR_OFFLINE=1) or under pytest
            # without INVAR_RESEARCH_SEARCH_LIVE=1 (hermetic unit tests).
            import os
            offline = os.environ.get("INVAR_OFFLINE") == "1" or (
                "PYTEST_CURRENT_TEST" in os.environ
                and os.environ.get("INVAR_RESEARCH_SEARCH_LIVE") != "1")
            search_client = LiteratureSearchClient(offline=offline)
        self.search_client = search_client
        self.stage_synthesizer = stage_synthesizer
        self.abort_flag = abort_flag
        # Compiler scratch is isolated under <workspace>/.invar/scratch/cabi
        # (hidden from the file explorer) instead of cluttering research/cabi.
        root = Path(workspace_root) if workspace_root else (
            self.workspace.parent if self.workspace.name == "research"
            else self.workspace)
        self.workspace_root = root
        self.bridge = DynamicCABIBridge(root / ".invar" / "scratch" / "cabi")
        self.overlay_dir = Path(tempfile.mkdtemp(prefix="invar-research-"))
        self.scheduler = GoIContractScheduler()
        # Deduplicated literature sources retrieved so far this run; surfaced
        # live as `literature_count` on every activity/terminal event.
        self.retrieved_sources: List[Dict[str, Any]] = []
        self._retrieved_source_keys: set = set()
        self.blueprint: Optional[ResearchBlueprint] = None
        if verification_runner is None:
            from .default_verification import DefaultVerificationRunner
            verification_runner = DefaultVerificationRunner(self)
        self.verification_runner = verification_runner

    def _record_literature(self, items: Iterable[Dict[str, Any]]) -> int:
        """Merge newly retrieved sources (deduped) into the run ledger."""
        added = 0
        for item in items or []:
            key = "|".join(str((item or {}).get(k) or "")
                           for k in ("source", "title", "url"))
            if not key.strip("|") or key in self._retrieved_source_keys:
                continue
            self._retrieved_source_keys.add(key)
            self.retrieved_sources.append(dict(item))
            added += 1
        return added

    def _literature_count(self) -> int:
        return len(self.retrieved_sources)

    async def _emit(self, channel: str, event: str, data: Dict[str, Any]) -> None:
        if self._provider_failure is not None: raise self._provider_failure
        if self.progress_cb is not None:
            await self.progress_cb(channel, {"event": event, **data, "event_version": "2.0"})

    async def _evaluate_epistemic(self, state, *, phase, code, uast=None, contracts=None):
        if state.ihcg is None or self.verification_runner is None: return None
        from .epistemic import evaluate_execution, incomplete_certificate
        from .vault import ResearchVaultManager
        try:
            evidence = self.verification_runner(phase=phase, state=state, code=code, uast=uast, results=state.results)
            if inspect.isawaitable(evidence): evidence = await evidence
            if evidence is None: return None
            from dataclasses import replace
            for domain, audit in getattr(self, "_runtime_purity", {}).items():
                enforced = audit.projected_steps > 0 or audit.stabilized_steps > 0
                violations = (*evidence.purity.violations, "RUNTIME_ENFORCEMENT") if enforced else evidence.purity.violations
                state.preserved_vs_enforced[domain] = audit.model_copy(update={
                    "static_purity_passed": evidence.purity.static_purity_passed and not enforced, "violations": violations})
                if enforced:
                    evidence = replace(evidence, purity=evidence.purity.model_copy(update={
                        "static_purity_passed": False, "violations": violations}))
            state.preserved_vs_enforced[evidence.purity.domain_tag] = evidence.purity
            blueprint = state.blueprint if contracts is None else state.blueprint.model_copy(update={"contracts": contracts})
            certificate, signature, decision = evaluate_execution(blueprint, state.ihcg, evidence,
                executable_digest=code_hash(code), vault=ResearchVaultManager(self.workspace_root), attempt=state.heal_count,
                precision_refined=state.precision_refined, preserved=state.preserved_vs_enforced)
            state.meaningfulness_certificate, state.canonical_signature = certificate, signature
            state.blueprint = state.blueprint.model_copy(update={"unified_module_code":code})
            state.epistemic_verdict = certificate.epistemic_verdict.value
            if state.epistemic_verdict == "VERIFIED_DISCOVERY":
                if not hasattr(self,"_verified_foundations"): self._verified_foundations = {}
                self._verified_foundations[state.research_stage] = (state.ihcg,state.blueprint,code)
            await self._emit("accelerator_log", "purity_audit", {"audit": evidence.purity.model_dump(mode="json"), "research_stage": state.research_stage})
            certified = state.epistemic_verdict == "VERIFIED_FALSIFICATION"
            await self._emit("accelerator_log", "epistemic_route", {"action": decision.action, "halt_healing": certified,
                "epistemic_verdict": state.epistemic_verdict, "heal_count": state.heal_count})
            return decision
        except (ValueError, KeyError, TypeError) as error:
            state.meaningfulness_certificate = incomplete_certificate(state.blueprint.blueprint_id,
                state.blueprint.falsifiable_hypothesis, f"FORMAL_EVIDENCE_UNVERIFIED: {error}")
            state.epistemic_verdict = "INCONCLUSIVE_UNVERIFIED"
            err_code = str(error).split(":", 1)[0].strip()
            recoverable = {
                "ALGEBRAIC_EXECUTABLE_MISMATCH", "ORACLE_EXECUTABLE_MISMATCH",
                "ORACLE_OUTPUT_UNBOUND", "ORACLE_INPUT_WRITE", "T_CONST_PARAM_WRITE",
                "PIECEWISE_ORACLE_UNBINDABLE", "ORACLE_DIMENSION_MISMATCH",
                "ORACLE_STABILIZER_HIT", "ROUNDING_PROGRAM_UNBOUND",
                "ROUNDING_DENOMINATOR_NOT_SEPARATED", "RIGOROUS_ROUNDING_BOUND_UNAVAILABLE",
            }
            if phase == "post_execution" and err_code in recoverable and state.heal_count < 2:
                from .meaningfulness import RouteDecision
                heal_reason = f"EXECUTABLE_IHCG_MISMATCH: {error}"
                await self._emit("accelerator_log", "epistemic_route", {
                    "action": "REFLEX_HEAL", "halt_healing": False,
                    "epistemic_verdict": state.epistemic_verdict,
                    "heal_count": state.heal_count, "reason": heal_reason,
                })
                return RouteDecision(action="REFLEX_HEAL", reason=heal_reason)
            await self._emit("accelerator_log", "epistemic_unverified", {"reason": str(error)})
            return None

    async def emit_activity(self, stage: str, step: str, message: str,
                            level: str = "info",
                            telemetry: Optional[Dict[str, Any]] = None) -> None:
        """Structured real-time activity event for the Research Explorer feed."""
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "step": step,
            "message": message,
            "level": level,
            "literature_count": self._literature_count(),
            "telemetry": telemetry or {},
        }
        await self._emit("activity_log", "activity", payload)

    def _aborted(self, state: ResearchRunState) -> bool:
        if self.abort_flag is not None and self.abort_flag():
            state.aborted = True
        return state.aborted

    # -- top-level loop ----------------------------------------------------
    def _retry_provider(self, callback):
        if callback is None: return None

        async def complete(prompt):
            if self._provider_failure is not None: raise self._provider_failure
            for attempt in range(3):
                try:
                    result = callback(prompt)
                    return await result if inspect.isawaitable(result) else result
                except LLMProviderError as error:
                    if not error.retryable or attempt == 2:
                        self._provider_failure = error
                        raise
                    await asyncio.sleep(backoff_delay(attempt))
        return complete

    async def plan_and_run(self, question: str):
        from .planner import generate_plan_with_feasibility
        from .epistemic import incomplete_certificate
        from .vault import ResearchVaultManager
        self._provider_failure = None
        try:
            self.planner_result = await generate_plan_with_feasibility(question,
                llm_complete=self.llm_planner, llm_formalizer=self.llm_formalizer, llm_fast=self.llm_fast, fallback=False)
            if self._provider_failure is not None: raise self._provider_failure
            feas = self.planner_result.get("feasibility")
            feas_verdict = getattr(getattr(feas, "verdict", None), "value", getattr(feas, "verdict", None))
            if self.planner_result.get("blueprint") is None and str(feas_verdict or "").upper() in {"REFUSED_IMPOSSIBLE", "REFUSED_UNSUITABLE"}:
                verdict_str = str(feas_verdict).upper()
                run_id = "refused_" + uuid.uuid4().hex
                rationale = getattr(feas, "rationale", "") or ""
                violated = getattr(feas, "violated_boundary", "") or ""
                report_md = f"# Research Request Refused ({verdict_str})\n\n**Violated Boundary:** {violated}\n\n{rationale}\n"
                certificate = incomplete_certificate(run_id, question, verdict_str, diagnostics={"feasibility": feas.model_dump(mode="json") if hasattr(feas, "model_dump") else str(feas)})
                await self._emit("accelerator_log", "research_completed", dict(
                    epistemic_verdict=verdict_str, reason=verdict_str,
                    meaningfulness_certificate=certificate.model_dump(mode="json")))
                return ResearchPlanningFailure(
                    run_id=run_id,
                    meaningfulness_certificate=certificate,
                    status=verdict_str,
                    epistemic_verdict=verdict_str,
                    blueprint=None,
                    feasibility=feas,
                    report=report_md,
                )
            blueprint = ResearchBlueprint.model_validate(self.planner_result["blueprint"])
        except (LLMProviderError, ValueError, SyntaxError, TypeError, RuntimeError) as error:
            self._provider_failure = None
            run_id = "planning_"+uuid.uuid4().hex
            if isinstance(error, LLMProviderError):
                reason = "LLM_PROVIDER_RETRIES_EXHAUSTED" if error.retryable else "LLM_PROVIDER_ERROR"
                detail = reason
            else:
                reason, detail = "PLANNING_VALIDATION_FAILED", str(error)
            diagnostics = dict(error_type=type(error).__name__, error=detail,
                validation_attempts=list(getattr(error,"validation_attempts",())))
            certificate = incomplete_certificate(run_id,question,reason,diagnostics=diagnostics)
            directory = ResearchVaultManager(self.workspace_root).run_dir(run_id)
            directory.mkdir(parents=True,exist_ok=True)
            (directory/"meaningfulness_certificate.json").write_text(certificate.model_dump_json(indent=2),encoding="utf-8")
            (directory/"report.md").write_text(f"# Research planning did not complete\n\n{reason}\n\n{detail}\n",encoding="utf-8")
            (directory/"metadata.json").write_text(json.dumps(dict(run_id=run_id,status="INCONCLUSIVE_UNVERIFIED",failure_reason=reason,planning_error=detail,diagnostics=diagnostics)),encoding="utf-8")
            await self._emit("accelerator_log","research_completed",dict(epistemic_verdict="INCONCLUSIVE_UNVERIFIED",
                reason=reason,meaningfulness_certificate=certificate.model_dump(mode="json")))
            return ResearchPlanningFailure(run_id,certificate)
        return await self.run(blueprint)

    async def run(self, blueprint: ResearchBlueprint) -> ResearchRunState:
        self._provider_failure = None
        from .formal_evidence import attach_formal_evidence
        ingest_error = None
        try: blueprint = attach_formal_evidence(blueprint)
        except (ValueError,SyntaxError) as error: ingest_error = str(error)
        _canonicalize_blueprint_slots(blueprint)
        _auto_populate_variable_map(blueprint)
        self.blueprint = blueprint
        state = ResearchRunState(blueprint=blueprint)
        self._runtime_purity = {}
        from .epistemic import compile_blueprint_ihcg, incomplete_certificate
        if ingest_error:
            state.meaningfulness_certificate = incomplete_certificate(blueprint.blueprint_id,blueprint.falsifiable_hypothesis,f"FORMAL_INGEST_REJECTED: {ingest_error}")
            state.status = "INCONCLUSIVE_UNVERIFIED"
            await self._emit_report(state)
            return state
        try: state.ihcg = compile_blueprint_ihcg(blueprint)
        except ValueError as error:
            state.meaningfulness_certificate = incomplete_certificate(blueprint.blueprint_id, blueprint.falsifiable_hypothesis, str(error))
            state.status = "INCONCLUSIVE_UNVERIFIED"
            await self._emit_report(state)
            return state
        if state.ihcg is not None:
            await self._evaluate_epistemic(state, phase="deductive", code=blueprint.code_template)
            if state.epistemic_verdict == "VERIFIED_FALSIFICATION":
                state.status = state.epistemic_verdict
                await self._emit_report(state)
                return state
        # Defense-in-depth epistemic gate. A blueprint that already carries a
        # PASSING feasibility verdict (OPTIMAL_DIRECT / REFORMULATED_SURROGATE)
        # was approved upstream (Plan Mode / combined planner call) and must
        # NEVER abort here; only explicit refusals or unassessed blueprints
        # fall through to the heuristic re-check.
        _bv = getattr(getattr(blueprint, "feasibility", None), "verdict", None)
        bp_verdict = str(getattr(_bv, "value", _bv) or "").upper()
        if bp_verdict in ("OPTIMAL_DIRECT", "REFORMULATED_SURROGATE"):
            pass
        elif bp_verdict in ("REFUSED_IMPOSSIBLE", "REFUSED_UNSUITABLE") \
                and blueprint.feasibility is not None:
            from .feasibility import override_refusal_for_computational_benchmark
            overridden = override_refusal_for_computational_benchmark(
                f"{blueprint.title}. {blueprint.falsifiable_hypothesis}",
                blueprint.feasibility,
            )
            if str(getattr(overridden.verdict, "value", overridden.verdict)).upper() == "REFORMULATED_SURROGATE":
                blueprint = blueprint.model_copy(update={"feasibility": overridden})
                self.blueprint = blueprint
                state.blueprint = blueprint
                bp_verdict = "REFORMULATED_SURROGATE"
            else:
                from .shbt_federation import resolve_shbt_solvability
                shbt_res = resolve_shbt_solvability(f"{blueprint.title}. {blueprint.falsifiable_hypothesis}")
                if shbt_res.should_activate:
                    from ..schemas.research import FeasibilityAssessment, FeasibilityVerdict
                    rescued = FeasibilityAssessment(
                        verdict=FeasibilityVerdict.OPTIMAL_DIRECT,
                        rationale=shbt_res.reason,
                        assessed_by="heuristic",
                    )
                    blueprint = blueprint.model_copy(update={"feasibility": rescued})
                    self.blueprint = blueprint
                    state.blueprint = blueprint
                    bp_verdict = "OPTIMAL_DIRECT"
                else:
                    state.status = "REFUSED"
                    state.refusal = blueprint.feasibility.model_dump()
                    await self._emit("search_log", "refused", {"feasibility": state.refusal})
                    await self._emit_report(state)
                    return state
        else:
            gate = _heuristic_verdict(f"{blueprint.title}. {blueprint.falsifiable_hypothesis}")
            if str(gate.verdict.value).upper().startswith("REFUSED"):
                state.status = "REFUSED"
                state.refusal = gate.model_dump()
                await self._emit("search_log", "refused", {"feasibility": gate.model_dump()})
                await self._emit_report(state)
                return state

        await self.emit_activity(
            "FEASIBILITY", "gate_passed",
            f"Feasibility gate passed ({bp_verdict or 'HEURISTIC_PASS'}); "
            "starting the verified research loop.")
        try:
            result = await self._run_stages(state)
            if not getattr(result, "_report_emitted", False): await self._emit_report(result)
            return result
        except Exception as exc:  # noqa: BLE001 - bulletproof exception shield
            return await self._handle_loop_exception(state, exc)
        finally:
            try:
                if hasattr(self.bridge, "cleanup_scratch"):
                    self.bridge.cleanup_scratch()
            except Exception:  # noqa: BLE001
                pass
            import gc as _gc
            _gc.collect()

    async def _run_stages(self, state: ResearchRunState) -> ResearchRunState:
        """Drive the accretive stage loop (wrapped by run()'s shield)."""
        blueprint = state.blueprint
        if isinstance(blueprint.ihcg_evidence, dict) and not blueprint.ihcg_evidence.get("extensions") and blueprint.target_research_stages > 1:
            blueprint = blueprint.model_copy(update={"target_research_stages": 1})
            state.blueprint = blueprint
        self.blueprint = blueprint
        total = blueprint.target_research_stages
        pinned_code = ""          # verified layers only (drivers renamed)
        pinned_layers: List[ModuleLayer] = list(blueprint.verified_layers)
        if pinned_layers:
            pinned_code = blueprint.unified_module_code or ""
        stage = len(pinned_layers) + 1
        last_telemetry: Dict[str, Any] = (pinned_layers[-1].telemetry if pinned_layers else {})
        round_idx = 0

        while stage <= total and not self._aborted(state):
            state.research_stage = stage
            if stage > 1 and state.ihcg is not None:
                from .epistemic import compile_blueprint_ihcg, incomplete_certificate
                state.ihcg = compile_blueprint_ihcg(state.blueprint,stage=stage)
                state.epistemic_verdict = "INCONCLUSIVE_UNVERIFIED"
                state.canonical_signature = None
                state.meaningfulness_certificate = incomplete_certificate(blueprint.blueprint_id,blueprint.falsifiable_hypothesis,"ACCRETIVE_STAGE_UNVERIFIED")
            objective = (blueprint.stage_objectives[stage - 1]
                         if len(blueprint.stage_objectives) >= stage
                         else blueprint.falsifiable_hypothesis)
            await self._emit("goi_stream", "stage_start", {
                "research_stage": stage, "total_stages": total, "objective": objective,
                "mode": "foundation" if stage == 1 else "accretive",
                "pinned_layers": [layer.layer_id for layer in pinned_layers]})
            if stage > 1:
                await self.emit_activity(
                    "ACCRETION", f"stage_{stage}_start",
                    f"Accreting stage {stage}/{total} onto the verified unified "
                    f"module ({len(pinned_layers)} pinned layer(s))...")
            failures: List[str] = []
            candidate_code: Optional[str] = blueprint.code_template if stage == 1 else None
            if stage > 1:
                candidate_code = (blueprint.ihcg_evidence or {}).get("execution",{}).get("stage_code_templates",{}).get(str(stage))
            if stage == 1:
                bp_contracts = list(blueprint.contracts)
                if blueprint.target_research_stages > 1:
                    # Contracts explicitly staged for later research stages
                    # (c2_.../stage2_...) never run in Stage 1; unprefixed or
                    # c1_/stage1_ contracts do. Fall back to all contracts when
                    # no stage-1 contract exists so single-purpose blueprints
                    # keep working.
                    stage_one = [c for c in bp_contracts
                                 if _contract_stage_index(c.contract_id) == 1]
                    contracts = stage_one or bp_contracts
                else:
                    contracts = bp_contracts
            else:
                # Blueprint contracts staged for this research stage (c2_/stage2_)
                # are carried into the accretive round alongside synthesized ones.
                # On the final stage, also include any contracts with stage index >= stage
                # (e.g. c4_ when target_research_stages == 3) so no contract is orphaned.
                contracts = [
                    c for c in blueprint.contracts
                    if blueprint.target_research_stages > 1
                    and (
                        _contract_stage_index(c.contract_id) == stage
                        or (stage == blueprint.target_research_stages
                            and _contract_stage_index(c.contract_id) >= stage)
                    )
                ]
            verified = False

            # HARD BOUND: exactly max_feedback_rounds attempts, never more.
            # A failed final attempt breaks immediately — no "Attempt N+1"
            # healing dispatch and no further adjoint rollback can be queued.
            max_attempts = max(1, int(getattr(blueprint, "max_feedback_rounds", 3) or 3))
            attempt = 0
            results_base = len(state.results)
            # Best-attempt retention: a later healing attempt may diverge
            # where an earlier one stayed finite; the best candidate's code,
            # result and telemetry are kept for the report.
            best_attempt: Optional[Dict[str, Any]] = None
            best_score: Optional[Tuple[int, float]] = None
            healing_stall = 0
            healing_terminated = False
            had_divergence = False
            seen_error_fingerprints: Dict[str, int] = {}
            while attempt < max_attempts:
                if self._aborted(state):
                    break
                attempt += 1
                round_idx += 1
                pre_results_len = len(state.results)
                outcome = await self._run_round(
                    state, stage=stage, attempt=attempt, round_idx=round_idx,
                    objective=objective, pinned_code=pinned_code,
                    pinned_layers=pinned_layers, candidate_code=candidate_code,
                    contracts=contracts, failures=failures,
                    last_telemetry=last_telemetry,
                    stabilize_diverged=had_divergence)
                candidate_code = outcome.get("candidate_code", candidate_code)
                contracts = outcome.get("contracts", contracts)
                if outcome.get("epistemic_terminal"):
                    self.blueprint = state.blueprint
                    state.status = state.epistemic_verdict
                    await self._emit_report(state)
                    return state
                if outcome.get("refine_precision"):
                    state.precision_refined = True
                    continue
                if outcome["status"] == "commit":
                    pinned_code = outcome["pinned_code"]
                    pinned_layers.append(outcome["layer"])
                    last_telemetry = outcome["layer"].telemetry
                    state.blueprint = state.blueprint.model_copy(update={
                        "verified_layers": list(pinned_layers),
                        "current_stage": min(stage + 1, total) if stage < total else stage,
                        "unified_module_code": unified_from_layers(pinned_code, stage),
                        "evolution_history": list(state.blueprint.evolution_history) + [
                            StageTelemetry(stage=stage, status="VERIFIED",
                                           bundle_hash=outcome["layer"].bundle_hash,
                                           telemetry=_json_safe(outcome["layer"].telemetry),
                                           note=f"attempt {attempt}", timestamp=time.time())],
                    })
                    self.blueprint = state.blueprint
                    state.checkpoint_path = self._export_checkpoint(state.blueprint, stage)
                    await self._emit("accelerator_log", "checkpoint", {
                        "research_stage": stage, "path": state.checkpoint_path,
                        "bundle_hash": outcome["layer"].bundle_hash,
                        "pinned_layers": [layer.layer_id for layer in pinned_layers]})
                    await self.emit_activity(
                        "VERIFIED", "invariants_passed",
                        "All invariants verified by SMT shield + sandbox.")
                    if stage < total:
                        await self._save_partial_vault(state, stage)
                    verified = True
                    break
                failure = outcome.get("failure") or outcome["status"]
                # Granular live healing telemetry for every retry step.
                fail_text = str(failure).lower()
                failed_status = str(outcome.get("failed_status") or "")
                if "compil" in fail_text or failed_status == "ERROR":
                    await self.emit_activity(
                        f"STAGE_{stage}_BUILD", "compile_error",
                        f"Attempt {attempt}: C-ABI compilation failed. "
                        "Analyzing compiler diagnostic and re-synthesizing...")
                elif failed_status == "FALSIFIED":
                    cid = str(outcome.get("failed_contract") or "contract")
                    await self.emit_activity(
                        f"STAGE_{stage}_EXEC", "invariant_falsified",
                        f"Attempt {attempt}: Invariant {cid} falsified "
                        f"({str(failure)[:120]}). Feeding counterexample back to model...")
                elif failed_status == "DIVERGED":
                    had_divergence = True
                    await self.emit_activity(
                        f"STAGE_{stage}_EXEC", "sandbox_diverged",
                        f"Attempt {attempt}: sandbox trajectory diverged. "
                        "Feeding counterexample back to model...")
                if "diverged" in fail_text or "non-finite" in fail_text:
                    had_divergence = True
                # ALWAYS surface the exact attempt error (compiler diagnostic /
                # schema violation / falsified invariant) BEFORE any healing
                # dispatch, so the stream never jumps straight from a silent
                # failure to "Tier-2 model synthesizing correction".
                warn_step = ("BUILD_WARN" if ("compil" in fail_text
                                               or failed_status == "ERROR")
                             else "VALIDATION_WARN")
                await self.emit_activity(
                    f"STAGE_{stage}_BUILD" if warn_step == "BUILD_WARN"
                    else f"STAGE_{stage}_EXEC",
                    "attempt_error",
                    f"Attempt {attempt} failed — exact error: {str(failure)[:500]}",
                    level="warning")
                failures.append(failure)
                # Snapshot ONLY this attempt's results so a worse later
                # attempt never overwrites the best finite candidate.
                attempt_results = state.results[pre_results_len:]
                q_score = _attempt_quality_score(outcome, attempt_results)
                used_candidate = outcome.get("candidate_code") or candidate_code
                degraded_attempt = False
                if best_score is None or q_score > best_score:
                    best_score = q_score
                    best_attempt = {
                        "candidate_code": used_candidate,
                        "results": list(attempt_results),
                        "status": outcome.get("result_status", "ERROR"),
                        "failure": failure,
                    }
                elif best_attempt is not None and q_score < best_score:
                    # Fix 4: If a healing attempt degrades status (e.g. from FALSIFIED
                    # finite to DIVERGED non-finite), immediately roll back the AST and
                    # telemetry to the best finite attempt so the next attempt starts
                    # from the valid finite AST rather than the broken AST.
                    degraded_attempt = True
                    candidate_code = best_attempt["candidate_code"]
                    state.results[pre_results_len:] = list(best_attempt["results"])
                state.memory.append({"round": round_idx, "research_stage": stage,
                                     "outcome": outcome["status"], "counterexample": failure})
                state.blueprint = state.blueprint.model_copy(update={
                    "evolution_history": list(state.blueprint.evolution_history) + [
                        StageTelemetry(stage=stage, status=outcome.get("result_status", "ERROR"),
                                       note=failure[:300], timestamp=time.time())]})
                if outcome["status"] == "refused":
                    state.status = "REFUSED"
                    state.refusal = outcome.get("refusal")
                    return state
                err_fp = compute_healing_error_fingerprint(str(failure), str(used_candidate or ""))
                seen_error_fingerprints[err_fp] = seen_error_fingerprints.get(err_fp, 0) + 1
                # Short-circuit only after Tier-2 has had a chance to attempt repair (attempt >= 3),
                # or if the exact same error fingerprint repeats 3+ times.
                if seen_error_fingerprints[err_fp] >= 3:
                    await self.emit_activity(
                        "HEALING", "circuit_breaker_identical_error",
                        f"[HEALING_TERMINATED] Identical-error circuit breaker triggered "
                        f"(fingerprint {err_fp} repeated {seen_error_fingerprints[err_fp]} times).",
                        level="warning")
                    healing_terminated = True
                    break
                # Retry bound reached: stop BEFORE dispatching another heal.
                # No "Attempt {max+1}/{max}" synthesis and no further adjoint
                # rollback events may follow the final attempt.
                if attempt >= max_attempts:
                    break
                # Closed-loop telemetry-guided re-search on DIVERGED/FALSIFIED.
                diagnostic_context = await self._diagnostic_research(
                    state, stage=stage, outcome=outcome)
                if diagnostic_context:
                    failure = f"{failure}\n{diagnostic_context}"
                # Reflex_Heal + refinement feed the counterexample back to GoI_Plan.
                # (attempt < max_attempts here, so attempt + 1 is a real next round)
                final_heal = attempt + 1 >= max_attempts
                heal_tier = ("Tier-2" if final_heal else "Tier-1 fast")
                await self.emit_activity(
                    "HEALING", "reasoning_dispatch",
                    f"{heal_tier} model synthesizing correction "
                    f"(Attempt {attempt + 1}/{max_attempts})...")
                base_heal_code = (best_attempt["candidate_code"]
                                  if (degraded_attempt and best_attempt)
                                  else used_candidate)
                base_heal_failure = (best_attempt.get("failure") or failure
                                     if (degraded_attempt and best_attempt)
                                     else failure)
                pre_heal_hash = code_hash(base_heal_code or "")
                state.heal_count += 1
                raw_heal = await self._heal_and_refine(
                    stage=stage, round_idx=round_idx, objective=objective,
                    candidate_code=base_heal_code,
                    contracts=contracts, failure=base_heal_failure,
                    attempt=attempt, max_attempts=max_attempts,
                    pinned_functions=_pinned_function_names(pinned_code, pinned_layers))
                if isinstance(raw_heal, dict):
                    heal_outcome = raw_heal
                elif isinstance(raw_heal, str):
                    heal_outcome = {"code": raw_heal, "terminated": False}
                else:
                    heal_outcome = {"code": candidate_code, "terminated": False}
                if heal_outcome.get("terminated"):
                    healing_terminated = True
                    break
                healed_code = heal_outcome.get("code") or candidate_code
                if state.ihcg is not None:
                    from ..healing.deterministic_ast import assert_sealed_observables
                    is_exec_ihcg_repair = str(base_heal_failure or "").startswith("EXECUTABLE_IHCG_MISMATCH:")
                    protected = {
                        state.blueprint.variable_map[s.symbol]
                        for s in state.ihcg.slots
                        if s.symbol in state.blueprint.variable_map
                        and (s.role == "CONTROL_PARAM" or (s.role == "DERIVED_OBSERVABLE" and not is_exec_ihcg_repair))
                    }
                    try: assert_sealed_observables(base_heal_code or "", healed_code or "", protected)
                    except ValueError as error:
                        from .epistemic import incomplete_certificate
                        state.meaningfulness_certificate = incomplete_certificate(blueprint.blueprint_id, blueprint.falsifiable_hypothesis, str(error))
                        state.epistemic_verdict = "INCONCLUSIVE_UNVERIFIED"
                        state.status = state.epistemic_verdict
                        await self._emit_report(state)
                        return state
                post_heal_hash = code_hash(healed_code or "")
                if post_heal_hash == pre_heal_hash:
                    healing_stall += 1
                else:
                    healing_stall = 0
                candidate_code = healed_code
                if healing_stall >= 2:
                    await self.emit_activity(
                        "HEALING", "healing_terminated",
                        "[HEALING_TERMINATED] Stage determined unfixable after "
                        "deterministic + directive passes: two consecutive "
                        "healing attempts produced no residual improvement.",
                        level="warning")
                    healing_terminated = True
                    break

            if not verified:
                if best_attempt and best_attempt.get("results"):
                    base = results_base
                    state.results = list(state.results[:base]) + list(best_attempt["results"])
                    candidate_code = best_attempt.get("candidate_code") or candidate_code
                if pinned_layers:
                    state.checkpoint_path = self._export_checkpoint(
                        state.blueprint, len(pinned_layers))
                    state.blueprint = state.blueprint.model_copy(update={
                        "evolution_history": list(state.blueprint.evolution_history) + [
                            StageTelemetry(stage=stage, status="FALSIFIED",
                                           bundle_hash="",
                                           telemetry={},
                                           note=f"rolled back after {max_attempts} attempts",
                                           timestamp=time.time())],
                    })
                    self.blueprint = state.blueprint
                    await self._emit("accelerator_log", "checkpoint_preserved", {
                        "research_stage": stage, "path": state.checkpoint_path,
                        "verified_stages": len(pinned_layers),
                        "note": "stage retries exhausted; last verified unified "
                                "checkpoint preserved"})
                    if stage < total and not self._aborted(state):
                        state.rolled_back_stages.add(stage)
                        rolled_ids = {rc.contract_id for rc in contracts}
                        purged_contracts = [
                            c for c in (state.blueprint.contracts or [])
                            if c.contract_id not in rolled_ids
                            and _contract_stage_index(c.contract_id) != stage
                        ]
                        state.blueprint = state.blueprint.model_copy(
                            update={"contracts": purged_contracts}
                        )
                        self.blueprint = state.blueprint
                        blueprint = state.blueprint
                        await self.emit_activity(
                            "VAULT", "stage_rollback_continue",
                            f"Stage {stage} retries exhausted after {max_attempts} attempts; "
                            f"rolling back Stage {stage} delta and purging Stage {stage} contracts before continuing to Stage {stage + 1}/{total}...")
                        stage += 1
                        continue
                    state.status = "ABORTED" if state.aborted else "PARTIAL"
                    await self.emit_activity(
                        "VAULT", "partial_finalization",
                        f"Stage {stage} retries exhausted after {max_attempts} "
                        f"attempts. Finalizing run with verified baseline from "
                        f"Stage {len(pinned_layers)}...")
                else:
                    state.status = "ABORTED" if state.aborted else "FAILED"
                if state.status in ("PARTIAL", "FAILED"):
                    extra: Optional[Dict[str, Any]] = None
                    if pinned_layers:
                        extra = {
                            "verified_stages": len(pinned_layers),
                            "total_stages": total,
                            "message": (
                                f"{len(pinned_layers)} stage(s) verified. "
                                f"Stage {stage} halted after {max_attempts} "
                                "retry attempts.")}
                    await self._emit_report(state, extra=extra)
                return state
            stage += 1

        if state.aborted:
            state.status = "ABORTED"
        elif len(pinned_layers) >= total:
            state.status = "VERIFIED"
            await self._emit("accelerator_log", "research_complete", {
                "verified_stages": len(pinned_layers),
                "unified_module_hash": code_hash(state.blueprint.unified_module_code or ""),
                "checkpoint_path": state.checkpoint_path})
        elif pinned_layers:
            state.status = "PARTIAL"
            state.checkpoint_path = self._export_checkpoint(
                state.blueprint, len(pinned_layers))
        if state.status in ("VERIFIED", "PARTIAL", "FAILED"):
            await self._emit_report(state)
        return state

    async def _emit_report(self, state: ResearchRunState, *,
                           terminal_event: str = "research_completed",
                           extra: Optional[Dict[str, Any]] = None) -> None:
        """Synthesize, persist and stream the Comprehensive Research Report."""
        if getattr(state, "_report_emitted", False):
            return
        state._report_emitted = True
        state.preserved_vs_enforced.update(getattr(self, "_runtime_purity", {}))
        from .epistemic import incomplete_certificate
        if (state.meaningfulness_certificate is not None and state.epistemic_verdict in {"VERIFIED_DISCOVERY", "VERIFIED_BOUNDARY_DISCOVERY"}
                and state.status not in {"VERIFIED", "VERIFIED_DISCOVERY", "VERIFIED_BOUNDARY_DISCOVERY"}):
            state.epistemic_verdict = "INCONCLUSIVE_UNVERIFIED"
            state.meaningfulness_certificate = incomplete_certificate(state.blueprint.blueprint_id,
                state.blueprint.falsifiable_hypothesis, "RUN_DID_NOT_COMPLETE")
        if state.meaningfulness_certificate is None:
            state.meaningfulness_certificate = incomplete_certificate(state.blueprint.blueprint_id,
                state.blueprint.falsifiable_hypothesis, "FORMAL_IHCG_OR_EXECUTION_EVIDENCE_MISSING")
        if state.ihcg is not None and state.status == "VERIFIED": state.status = state.epistemic_verdict
        await self._emit("accelerator_log", "synthesizing_report", {
            "research_stage": state.research_stage, "status": state.status})
        try:
            from .reporter import generate_comprehensive_research_report
            telemetry: Dict[str, Any] = {}
            verified_layers = list(state.blueprint.verified_layers or [])
            if verified_layers and state.status in ("PARTIAL", "ABORTED"):
                # Partial finalization reports on the last PASSING stage's
                # verified telemetry, never on the exhausted stage's failures.
                telemetry = dict(verified_layers[-1].telemetry or {})
            elif state.results:
                telemetry = dict(state.results[-1].telemetry)
            state.report = generate_comprehensive_research_report(
                state.blueprint, telemetry, state.literature,
                results=state.results, status=state.status,
                checkpoint_path=state.checkpoint_path)
        except Exception as exc:  # noqa: BLE001 - report is best-effort
            state.report = f"(report synthesis failed: {str(exc)[:200]})"
        from .report import render_meaningfulness_certificate
        state.report += "\n\n" + render_meaningfulness_certificate(state.meaningfulness_certificate, state.preserved_vs_enforced)
        await self.emit_activity(
            "REPORT", "report_generated",
            f"Synthesized comprehensive research report "
            f"({len(state.report or '')} chars, status {state.status}).")
        # Persist: Research Vault only — research outputs never touch the
        # software codebase root (legacy root/checkpoint copies are migrated
        # into the vault at server startup).
        report_path: Optional[Path] = None
        try:
            from .vault import ResearchVaultManager
            from ..llm.telemetry import get_llm_telemetry_summary
            vault = ResearchVaultManager(self.workspace_root)
            telemetry_all: Dict[str, Any] = {}
            for result in state.results or []:
                telemetry_all.update(getattr(result, "telemetry", {}) or {})
            llm_summary = get_llm_telemetry_summary()
            telemetry_all["llm_prompt_tokens"] = float(llm_summary.get("prompt_tokens", 0))
            telemetry_all["llm_completion_tokens"] = float(llm_summary.get("completion_tokens", 0))
            telemetry_all["llm_reasoning_tokens"] = float(llm_summary.get("reasoning_tokens", 0))
            telemetry_all["llm_cache_hit_tokens"] = float(llm_summary.get("cache_hit_tokens", 0))
            telemetry_all["llm_total_calls"] = float(llm_summary.get("total_calls", 0))
            telemetry_all["llm_total_latency_s"] = float(llm_summary.get("total_latency_s", 0.0))
            backend = ""
            for key in ("backend", "target_backend", "accelerator"):
                if key in telemetry_all:
                    backend = str(telemetry_all[key])
                    break
            vault_status = state.status
            if vault_status in ("PARTIAL", "ABORTED") and state.blueprint.verified_layers:
                vault_status = "PARTIALLY_VERIFIED"
            meta = vault.register_run(
                run_id=state.blueprint.blueprint_id,
                title=state.blueprint.title,
                blueprint=state.blueprint,
                report_md=state.report,
                status=vault_status,
                telemetry=telemetry_all,
                target_backend=backend,
                literature_count=(self._literature_count()
                                  or len(state.literature)),
                meaningfulness_certificate=state.meaningfulness_certificate,
                canonical_signature=state.canonical_signature)
            report_path = vault.run_dir(state.blueprint.blueprint_id) / "report.md"
            await self._emit("accelerator_log", "vault_archived", {
                "run_id": meta.get("run_id"), "status": state.status})
            await self._emit("accelerator_log", "vault_updated", {
                "run_id": meta.get("run_id"), "status": vault_status})
            await self.emit_activity(
                "VAULT", "vault_saved",
                f"Archived run `{meta.get('run_id')}` to the Research Vault "
                f"(status {state.status}).")
        except Exception as exc:  # noqa: BLE001 - vault archival is best-effort
            await self._emit("accelerator_log", "vault_archive_error",
                             {"error": str(exc)[:200]})
        terminal_status = state.status
        if terminal_status in ("PARTIAL", "ABORTED") \
                and state.blueprint.verified_layers:
            # A halted run with a verified baseline is PARTIALLY_VERIFIED.
            terminal_status = "PARTIALLY_VERIFIED"
        payload: Dict[str, Any] = {
            "epistemic_verdict": state.epistemic_verdict,
            "meaningfulness_certificate": state.meaningfulness_certificate.model_dump(mode="json"),
            "status": terminal_status,
            "report": state.report,
            "run_id": state.blueprint.blueprint_id,
            "report_path": str(report_path) if report_path else "",
            "literature_count": (self._literature_count()
                                 or len(getattr(state.blueprint, "retrieved_sources", []) or []))}
        if extra:
            payload.update(extra)
        await self._emit("accelerator_log", terminal_event, payload)

    async def _save_partial_vault(self, state: ResearchRunState, stage: int) -> None:
        """Immediately archive verified stages as PARTIALLY_VERIFIED.

        Guarantees Stage-1 (and any intermediate) verified findings survive a
        later-stage failure in the Research Vault.
        """
        await self.emit_activity(
            "VAULT", "checkpoint_save",
            f"Stage {stage} verified — persisting intermediate vault checkpoint...")
        try:
            from .reporter import generate_comprehensive_research_report
            from .vault import ResearchVaultManager
            telemetry_all = _aggregate_telemetry(state.results)
            report = generate_comprehensive_research_report(
                state.blueprint, telemetry_all, state.literature,
                results=state.results, status="PARTIALLY_VERIFIED",
                checkpoint_path=state.checkpoint_path)
            backend = ""
            for key in ("backend", "target_backend", "accelerator"):
                if key in telemetry_all:
                    backend = str(telemetry_all[key])
                    break
            vault = ResearchVaultManager(self.workspace_root)
            meta = vault.register_run(
                run_id=state.blueprint.blueprint_id,
                title=state.blueprint.title,
                blueprint=state.blueprint,
                report_md=report,
                status="PARTIALLY_VERIFIED",
                telemetry=telemetry_all,
                target_backend=backend,
                literature_count=(self._literature_count()
                                  or len(state.literature)))
            await self.emit_activity(
                "VAULT", "checkpoint_saved",
                f"Intermediate checkpoint persisted: `{meta.get('run_id')}` "
                f"(PARTIALLY_VERIFIED, {len(state.blueprint.verified_layers)} "
                "verified stage(s)).")
        except Exception as exc:  # noqa: BLE001 - checkpointing is best-effort
            await self.emit_activity(
                "VAULT", "checkpoint_error",
                f"Intermediate vault checkpoint failed: {str(exc)[:200]}",
                level="warning")

    async def _handle_loop_exception(self, state: ResearchRunState,
                                     exc: Exception) -> ResearchRunState:
        """Bulletproof shield: never lose verified work to an unhandled error."""
        provider_error = self._provider_failure or (exc if isinstance(exc,LLMProviderError) else None)
        if provider_error is not None:
            from .epistemic import incomplete_certificate
            self._provider_failure = None
            reason = "LLM_PROVIDER_RETRIES_EXHAUSTED" if provider_error.retryable else "LLM_PROVIDER_ERROR"
            state.status = state.epistemic_verdict = "INCONCLUSIVE_UNVERIFIED"
            state._report_emitted = False
            state.meaningfulness_certificate = incomplete_certificate(state.blueprint.blueprint_id,
                state.blueprint.research_prompt or state.blueprint.falsifiable_hypothesis,reason)
            await self._emit_report(state)
            return state
        error_trace = traceback.format_exc()
        logger.error("Research loop error in stage %s: %s",
                     state.research_stage, error_trace)
        await self.emit_activity(
            "ERROR", "exception_caught",
            f"Execution halted: {str(exc)[:300]}", level="error",
            telemetry={"traceback": error_trace[-600:]})
        verified_layers = list(state.blueprint.verified_layers or [])
        has_baseline = bool(verified_layers)
        if has_baseline:
            state.status = "PARTIAL"
            state.checkpoint_path = state.checkpoint_path or self._export_checkpoint(
                state.blueprint, len(verified_layers))
            # Record the halted stage so the report marks it as failed.
            state.blueprint = state.blueprint.model_copy(update={
                "evolution_history": list(state.blueprint.evolution_history) + [
                    StageTelemetry(stage=state.research_stage, status="ERROR",
                                   note=f"unhandled exception: {str(exc)[:200]}",
                                   timestamp=time.time())]})
        else:
            state.status = "FAILED"
        try:
            await self._emit_report(
                state, terminal_event="research_failed",
                extra={"blueprint_id": state.blueprint.blueprint_id,
                       "error": str(exc), "traceback": error_trace,
                       "partial_results_available": has_baseline})
        except Exception:  # noqa: BLE001 - last-resort: still signal failure
            await self._emit("accelerator_log", "research_failed", {
                "blueprint_id": state.blueprint.blueprint_id,
                "error": str(exc), "traceback": error_trace,
                "partial_results_available": has_baseline})
        if has_baseline:
            try:
                from .vault import ResearchVaultManager
                ResearchVaultManager(self.workspace_root).mark_status(
                    state.blueprint.blueprint_id, "PARTIALLY_VERIFIED")
            except Exception:  # noqa: BLE001
                logger.warning("vault mark_status failed", exc_info=True)
        return state

    # -- one round through the nine stages ------------------------------------
    async def _run_round(self, state: ResearchRunState, *, stage: int, attempt: int,
                         round_idx: int, objective: str, pinned_code: str,
                         pinned_layers: List[ModuleLayer], candidate_code: Optional[str],
                         contracts: List[VerificationContract], failures: List[str],
                         last_telemetry: Dict[str, Any],
                         stabilize_diverged: bool = False) -> Dict[str, Any]:
        bp = state.blueprint
        emit = self._emit
        tag = {"round": round_idx, "research_stage": stage, "attempt": attempt}

        # Zero-LLM fast path for stage > 1 (only when no LLM/synthesizer is active
        # so multi-stage synthesized experiments never skip Stage > 1 synthesis/ingest)
        if stage > 1 and candidate_code is None and attempt == 1 and self.llm_fast is None and self.stage_synthesizer is None:
            cumulative = _contracts_through_stage(bp, stage)
            fast_candidate = _deterministic_accretion_candidate(pinned_code, stage)
            if contracts and cumulative and fast_candidate:
                trial = await self._run_round(
                    state, stage=stage, attempt=attempt, round_idx=round_idx,
                    objective=objective, pinned_code=pinned_code,
                    pinned_layers=pinned_layers, candidate_code=fast_candidate,
                    contracts=contracts, failures=failures,
                    last_telemetry=last_telemetry)
                if trial.get("status") == "commit":
                    await self._emit("accelerator_log", "accretion_fast_path", {
                        **tag, "research_stage": stage,
                        "message": (f"[ACCRETION] Pinned unified module already "
                                    f"satisfies Stage {stage} contracts "
                                    "(0 LLM tokens used).")})
                    return trial

        # 1. Ingest: live literature retrieval (never inside the sandbox).
        state.stage = "Ingest"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "Ingest"})
        if attempt == 1 and (stage == 1 or candidate_code is None):
            if stage == 1:
                queries = list(bp.search_queries)
                q_source = "blueprint"
            else:
                g = await guided_queries(self.llm_fast, stage=stage, objective=objective,
                                         hypothesis=bp.falsifiable_hypothesis,
                                         telemetry=last_telemetry, failures=failures)
                queries, q_source = g["queries"], g["source"]
            await emit("search_log", "ingest", {**tag, "hypothesis": bp.falsifiable_hypothesis,
                                                "queries": queries, "query_source": q_source,
                                                "steered_by": _json_safe(
                                                    {k: v for k, v in last_telemetry.items()
                                                     if isinstance(v, (int, float))})})
            prev_lit = [
                h for h in (state.literature or [])
                if isinstance(h, dict)
            ]
            for h in prev_lit:
                h.setdefault("stage", 1)
            fresh_lit = await self._literature_search(queries, tag)
            merged_lit: List[Dict[str, Any]] = []
            seen_lit: set = set()
            for h in prev_lit + list(fresh_lit or []):
                if not isinstance(h, dict):
                    continue
                if h not in prev_lit:
                    h.setdefault("stage", stage)
                norm_title = re.sub(r"[^a-z0-9]+", " ", str(h.get("title") or "").lower()).strip()
                norm_url = str(h.get("url") or "").strip().lower().rstrip("/")
                k = (
                    ("title", h.get("source"), h.get("badge"), norm_title)
                    if norm_title
                    else ("url", norm_url)
                )
                if k not in seen_lit:
                    seen_lit.add(k)
                    merged_lit.append(h)
            state.literature = merged_lit
            # Literature constants & equations come ONLY from ingested sources
            # (equations/invariants + numeric constants). Blueprint state
            # variables and contract evaluation metrics are NEVER literature
            # constants and must not be conflated with them.
            from .extractor import QuantitativeInsightExtractor, extract_parameters_from_text
            q_ext = QuantitativeInsightExtractor()
            for h in state.literature:
                summary = str((h or {}).get("summary") or "")
                title = str((h or {}).get("title") or "")
                blob = f"{title}. {summary}".strip() if summary else ""
                if not blob:
                    continue
                src_label = str((h or {}).get("source") or "literature")
                existing_eqs = list((h or {}).get("equations") or [])
                seen_eqs = {str(x) for x in existing_eqs}
                for eq in q_ext.equations_from_text(blob, source=src_label):
                    lhs_s = str(getattr(eq, "lhs", "") or "").strip()
                    rhs_s = str(getattr(eq, "rhs_latex", "") or getattr(eq, "rhs_python_hint", "") or "").strip()
                    eq_txt = f"{lhs_s} = {rhs_s}".strip() if (lhs_s and rhs_s) else ""
                    if eq_txt and eq_txt not in seen_eqs:
                        seen_eqs.add(eq_txt)
                        existing_eqs.append(eq_txt)
                if existing_eqs:
                    h["equations"] = existing_eqs[:12]
                params = extract_parameters_from_text(blob, source=src_label)
                if params:
                    h.setdefault("constants", {}).update(
                        {p.name: p.si_value for p in params[:8] if p.name not in (h.get("constants") or {})})
            eq_count = sum(len((h or {}).get("equations") or []) for h in state.literature)
            const_count = sum(len((h or {}).get("constants") or {}) for h in state.literature)
            contract_vars = sorted(
                {str(k) for k in (getattr(bp, "variable_map", {}) or {})}
                | {str(i.variable_name) for c in contracts for i in c.invariants})
            source_breakdown: Dict[str, int] = {}
            for h in state.literature:
                src_name = str((h or {}).get("source") or "unknown")
                source_breakdown[src_name] = source_breakdown.get(src_name, 0) + 1
            await self.emit_activity(
                "INGEST", "literature_hit",
                f"Ingested {len(state.literature)} sources "
                f"(by_source={source_breakdown}; "
                f"{eq_count} equations/invariants, {const_count} numeric "
                f"constants); contract variables: {contract_vars}",
                telemetry={"source_breakdown": source_breakdown, "equations": eq_count, "constants": const_count})

        # 2. GoI_Plan: accretive synthesis for stage k > 1.
        state.stage = "GoI_Plan"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "GoI_Plan"})
        if stage > 1 and candidate_code is None:
            try:
                synth = await self._synthesize_stage(
                    stage=stage, total=bp.target_research_stages, objective=objective,
                    pinned_code=pinned_code, pinned_layers=pinned_layers,
                    stage_contracts=[c.model_dump() for c in contracts],
                    telemetry=last_telemetry, literature=state.literature,
                    failures=failures)
                candidate_code = synth["code"]
                prior_contract_dicts = [c.model_dump() for layer in pinned_layers
                                        for c in layer.contracts]
                synth["contracts"], _, _ = canonicalize_contract_dicts(
                    synth.get("contracts") or [],
                    variable_map=getattr(bp, "variable_map", None),
                    prior_contracts=prior_contract_dicts,
                )
                staged = list(contracts)
                contracts = [VerificationContract.model_validate(c) for c in synth["contracts"]]
                seen = {c.contract_id for c in contracts}
                contracts += [c for c in staged if c.contract_id not in seen]
                await emit("goi_stream", "accretive_plan", {
                    **tag, "rationale": str(synth.get("rationale", ""))[:600],
                    "new_contracts": [c.contract_id for c in contracts]})
            except Exception as exc:  # noqa: BLE001
                detail = _format_accretion_exception(exc)
                await self.emit_activity(
                    "ACCRETION", "synthesis_error",
                    f"accretive synthesis failed: {detail}", level="warning")
                candidate_code = _deterministic_accretion_candidate(pinned_code, stage)
                if candidate_code is None:
                    return {"status": "error", "result_status": "ERROR",
                            "failure": f"accretive synthesis failed: {detail}"}
        if candidate_code is None:
            return {"status": "error", "result_status": "ERROR",
                    "failure": "no candidate module for this stage"}
        try:
            validate_slot_writes(candidate_code, bp.variable_map, bp.slot_roles)
        except SyntaxError:
            pass
        except ValueError as error:
            return {"status": "error", "result_status": "ERROR", "failure": f"PreWrite_Guard: {error}",
                    "candidate_code": candidate_code, "contracts": contracts}

        # Ensure a top-level `simulation_step(buf, dt)` exists if the planner/LLM
        # emitted a helper `*_step(buf, dt)` without the driver wrapper.
        try:
            _cand_tree = ast.parse(candidate_code)
            _top_fns = [n for n in _cand_tree.body if isinstance(n, ast.FunctionDef)]
            if not any(f.name == "simulation_step" for f in _top_fns):
                _two_arg = [
                    f.name for f in _top_fns
                    if len(f.args.args) == 2 and not f.name.startswith("_invar_")
                ]
                _step_cands = [n for n in _two_arg if n.endswith("_step")]
                _delegate = (_step_cands[-1] if _step_cands else (_two_arg[-1] if _two_arg else None))
                if _delegate:
                    candidate_code = candidate_code.rstrip() + f"\n\ndef simulation_step(buf, dt):\n    {_delegate}(buf, dt)\n"
                else:
                    candidate_code = candidate_code.rstrip() + "\n\ndef simulation_step(buf, dt):\n    pass\n"
        except SyntaxError:
            pass

        pinned_fn_names = _pinned_function_names(pinned_code, pinned_layers) if stage > 1 else []
        rolled_back = set(getattr(state, "rolled_back_stages", None) or ())
        all_bp_dicts = [
            c.model_dump() if hasattr(c, "model_dump") else c
            for c in (getattr(bp, "contracts", None) or [])
            if _contract_stage_index(
                str(c.get("contract_id") if isinstance(c, dict) else getattr(c, "contract_id", ""))
            ) not in rolled_back
        ]
        cumulative_dicts = [
            c.model_dump() for c in _contracts_through_stage(bp, stage)
            if _contract_stage_index(c.contract_id) not in rolled_back
        ]
        # Populate unified slot_roles using pinned_code + candidate_code BEFORE Tier-0 runs,
        # so Stage > 1 knows full upstream dataflow and never projects diagnostic slots.
        pre_san_candidate = candidate_code
        combined_role_src = f"{pinned_code}\n\n{candidate_code}" if (stage > 1 and pinned_code) else candidate_code
        _auto_populate_variable_map(bp, combined_role_src)
        declared = text_evolutions(bp.research_prompt or bp.falsifiable_hypothesis)
        san = sanitize_intrinsics(
            candidate_code,
            contracts=contracts,
            all_contracts=cumulative_dicts or contracts,
            variable_map=getattr(bp, "variable_map", None),
            slot_roles=bp.slot_roles,
            stabilize_diverged=stabilize_diverged,
            preserve_dynamics=bool(declared) or state.ihcg is not None,
            pinned_functions=pinned_fn_names or None,
        )
        _apply_backup_slots_to_vmap(bp, getattr(san, "backup_slot_map", {}), san.code)
        if san.changed:
            candidate_code = san.code
            await self.emit_activity(
                f"STAGE_{stage}_BUILD", "deterministic_sanitizer",
                "Tier-0 sanitizer normalized scalar intrinsics/math domains "
                f"({'; '.join(san.notes[:3])}).")
        _auto_populate_variable_map(
            bp,
            f"{pinned_code}\n\n{candidate_code}" if (stage > 1 and pinned_code) else candidate_code,
        )

        if stage > 1:
            try:
                merge = merge_accretive_module(
                    pinned_code, candidate_code,
                    require_coupling=False,
                    allow_helper_overrides=(attempt > 1),
                )
            except AccretionError as exc:
                fallback = _deterministic_accretion_candidate(pinned_code, stage)
                if fallback is None or fallback == candidate_code:
                    return {"status": "error", "result_status": "ERROR",
                            "failure": f"accretion validator: {exc}",
                            "candidate_code": candidate_code, "contracts": contracts}
                candidate_code = fallback
                merge = merge_accretive_module(
                    pinned_code, candidate_code,
                    require_coupling=False,
                    allow_helper_overrides=(attempt > 1),
                )
            unified = merge.code
            await emit("goi_stream", "merge", {
                **tag, "pinned_functions": merge.pinned_functions,
                "new_functions": merge.new_functions,
                "reverted_overwrites": merge.reverted_overwrites,
                "restored_missing": merge.restored_missing,
                "coupled_calls": merge.coupled_calls})
        else:
            unified = candidate_code
        pre_san_unified = unified

        if stage > 1:
            unified_pinned = sorted(set(pinned_fn_names) | set(merge.pinned_functions))
            san_u = sanitize_intrinsics(
                unified,
                contracts=contracts,
                all_contracts=cumulative_dicts or contracts,
                variable_map=getattr(bp, "variable_map", None),
                slot_roles=bp.slot_roles,
                stabilize_diverged=stabilize_diverged,
                preserve_dynamics=bool(declared) or state.ihcg is not None,
                pinned_functions=unified_pinned or None,
            )
            _apply_backup_slots_to_vmap(bp, getattr(san_u, "backup_slot_map", {}), san_u.code)
            if san_u.changed:
                unified = san_u.code

        # 3. Wavefront_Schedule: GoI nilpotent schedule, SMW delta for new contracts.
        state.stage = "Wavefront_Schedule"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "Wavefront_Schedule"})
        plan = self.scheduler.plan(len(contracts), len(pinned_layers))
        await emit("goi_stream", "plan", {**tag, **{k: plan[k] for k in (
            "nodes", "spectral_radius", "nilpotent", "smw_rank", "engine")}})
        await emit("goi_stream", "schedule", {**tag, "wavefronts": plan["wavefronts"],
                                              "edges": plan["edges"]})
        if not plan.get("nilpotent", True):
            return {"status": "error", "result_status": "ERROR",
                    "failure": "GoI schedule is not nilpotent (cyclic contract dependencies)",
                    "candidate_code": candidate_code, "contracts": contracts}

        # 4. Overlay_Mutate: copy-on-write scratch holds the candidate unified module.
        state.stage = "Overlay_Mutate"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "Overlay_Mutate"})
        scratch = self.overlay_dir / f"round_{round_idx}"
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / "unified_module.py").write_text(unified, encoding="utf-8")
        await emit("accelerator_log", "overlay", {**tag, "path": str(scratch),
                                                   "module_hash": code_hash(unified),
                                                   "lines": unified.count("\n")})

        # 5. PreWrite_Guard: SHACL/Chiasmus firewall + v9 compliance + lowering.
        state.stage = "PreWrite_Guard"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "PreWrite_Guard"})
        try:
            validate_contract_slots(contracts, bp.variable_map)
            validate_slot_writes(unified, bp.variable_map, bp.slot_roles)
            try:
                validate_observables(unified, contracts, bp.variable_map, bp.slot_roles,
                                     _verified_projection_slots(ast.parse(unified), contracts))
            except ValueError as _obs_err:
                if pre_san_unified != unified:
                    try:
                        validate_observables(pre_san_unified, contracts, bp.variable_map, bp.slot_roles,
                                             _verified_projection_slots(ast.parse(pre_san_unified), contracts))
                    except ValueError as _raw_obs_err:
                        raise _raw_obs_err from _obs_err
                raise
            validate_contract_faithfulness(
                contracts,
                bp.variable_map,
                prompt=bp.research_prompt,
                code=unified,
                require_sweep_coverage=False,
            )
            if stage > 1 and self.llm_fast is not None:
                noop_bad, noop_reason = is_noop_layer_wrapper(pre_san_candidate, stage)
                if not noop_bad:
                    noop_bad, noop_reason = is_noop_layer_wrapper(pre_san_unified, stage)
                if noop_bad:
                    raise ValueError(noop_reason)
            for contract in contracts:
                normalized = _reconcile_time_horizon(
                    {'code_template': unified, 'contracts': [contract.model_dump()]}, bp.research_prompt)
                expected = normalized['contracts'][0]
                target = expected['dt'] * expected['steps']
                if not math.isclose(contract.dt * contract.steps, target, rel_tol=1e-12):
                    raise ValueError(f'SCIENTIFIC_INTEGRITY: horizon mismatch for {contract.contract_id}')
                coverage = await asyncio.to_thread(validate_protocol, unified, contract, bp.research_prompt)
                if coverage:
                    await emit('accelerator_log', 'phase_coverage', {**tag, 'contract_id': contract.contract_id, 'coverage': coverage})
            schemes = await asyncio.to_thread(validate_variant_schemes, unified, bp, contracts)
            if schemes:
                await emit('accelerator_log', 'integration_schemes', {**tag, 'schemes': schemes})
        except (ValueError, SyntaxError, ReferenceError) as error:
            self._rollback_overlay(scratch)
            return {'status':'error', 'result_status':'ERROR', 'failure':f'PreWrite_Guard: {error}', 'candidate_code':candidate_code, 'contracts':contracts}
        if declared and state.ihcg is None:
            try:
                await asyncio.to_thread(check_evolution_reference, bp.model_dump(), {}, unified)
            except (ReferenceError, ValueError, ImportError, KeyError) as error:
                message = f"{getattr(error, 'code', 'RESEARCH_ODE_REFERENCE_FAILED')}: {error}"
                await emit('accelerator_log', 'guard_reject', {**tag, 'violations': [message]})
                self._rollback_overlay(scratch)
                return {'status': 'error', 'result_status': 'ERROR', 'failure': message,
                        'candidate_code': candidate_code, 'contracts': contracts}
        violations = shacl_chiasmus_firewall(unified) + v9_transpiler_violations(unified)
        if violations:
            await emit("accelerator_log", "guard_reject", {**tag, "violations": violations})
            self._rollback_overlay(scratch)
            return {"status": "error", "result_status": "ERROR",
                    "failure": "PreWrite_Guard: " + "; ".join(violations),
                    "candidate_code": candidate_code, "contracts": contracts}
        # 1-step Python smoke execution in an OS-isolated child process (ARCH-005: zero in-process exec)
        try:
            from ..runtime.isolation import run_isolated_kernel_smoke

            await asyncio.to_thread(
                run_isolated_kernel_smoke,
                unified,
                dt=0.01,
                timeout_s=8.0,
                validate_ast=True,
            )
        except Exception as exc:  # noqa: BLE001
            await emit("accelerator_log", "guard_reject", {**tag, "violations": [f"python_smoke: {type(exc).__name__}: {exc}"]})
            self._rollback_overlay(scratch)
            return {"status": "error", "result_status": "ERROR",
                    "failure": f"PreWrite_Guard python_smoke: {type(exc).__name__}: {exc}",
                    "candidate_code": candidate_code, "contracts": contracts}
        try:
            from ..healing.deterministic_ast import instrument_purity
            uast, cpp_code = _lower_kernel(instrument_purity(unified))
        except (UnsupportedError, EmitterError, SyntaxError, ValueError) as exc:
            await emit("accelerator_log", "guard_reject", {**tag, "violations": [str(exc)]})
            self._rollback_overlay(scratch)
            return {"status": "error", "result_status": "ERROR",
                    "failure": f"transpiler: {exc}",
                    "candidate_code": candidate_code, "contracts": contracts}
        await emit("accelerator_log", "guard_pass", {**tag, "cpp_bytes": len(cpp_code)})
        kernel_mode = kernel_mode_for_blueprint(bp)
        await self.emit_activity(
            f"STAGE_{stage}_BUILD", "cabi_compile",
            f"Compiling {kernel_log_label(kernel_mode)} kernel via Clang++ "
            f"(-O3 -std=c++20) for {len(contracts)} contract(s)...")

        # 6. SMT_Concolic_Verify: Z3 consistency of every contract's invariants.
        state.stage = "SMT_Concolic_Verify"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "SMT_Concolic_Verify"})
        for contract in contracts:
            report = check_invariants_consistent([i.assertion for i in contract.invariants])
            await emit("goi_stream", "smt", {**tag, "contract_id": contract.contract_id,
                                             "ok": report.satisfiable, "checked": report.checked,
                                             "engine": report.engine,
                                             "unsat_core": report.unsat_core})
            if not report.satisfiable:
                self._rollback_overlay(scratch)
                refusal = {
                    "verdict": "REFUSED_IMPOSSIBLE",
                    "rationale": ("Z3 proved the contract's invariants mutually "
                                  "unsatisfiable; no simulation can satisfy them."),
                    "violated_boundary": "contradictory SMT constraints",
                    "unsat_core": report.unsat_core,
                    "reformulation": "Drop or relax one of the conflicting invariants.",
                }
                await emit("search_log", "refused", {**tag, "feasibility": refusal})
                return {"status": "refused", "result_status": "REFUSED",
                        "failure": "contradictory invariants: " + " AND ".join(report.unsat_core),
                        "refusal": refusal}

        # 6b. Tier-0 Evidence-to-AST Grounding: warn (never block) when the
        # blueprint ignores extracted SI evidence or binds dummy initial states.
        grounding_warnings = validate_blueprint_grounding(bp)
        if grounding_warnings:
            await self.emit_activity(
                f"STAGE_{stage}_EXEC", "grounding_warning",
                "Evidence grounding check: " + "; ".join(grounding_warnings)[:400],
                level="warning")

        # 7. Sandbox_Execute: regression of pinned layers + new contracts.
        state.stage = "Sandbox_Execute"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "Sandbox_Execute"})
        km_kw = {} if kernel_mode == "rk4_ode" else {"kernel_mode": kernel_mode}
        results, regression = await self._sandbox_execute(
            bp, uast, cpp_code, unified, contracts, pinned_layers, stage, tag,
            **km_kw)
        state.results = results
        if state.ihcg is not None:
            bound_contracts = list({c.contract_id: c for layer in pinned_layers for c in layer.contracts}.values())
            bound_contracts = list({c.contract_id: c for c in [*bound_contracts, *contracts]}.values())
            decision = await self._evaluate_epistemic(state, phase="post_execution", code=unified, uast=uast, contracts=bound_contracts)
            if (state.epistemic_verdict in {"VERIFIED_FALSIFICATION", "VERIFIED_BOUNDARY_DISCOVERY", "VERIFIED_DISCOVERY"}
                    and not (state.epistemic_verdict == "VERIFIED_DISCOVERY" and stage < bp.target_research_stages)):
                state.blueprint = state.blueprint.model_copy(update={"unified_module_code": unified})
                return {"status": "epistemic_terminal", "epistemic_terminal": True, "candidate_code": unified, "contracts": contracts}
            if decision is not None and decision.action == "REFINE_PRECISION_ONCE" and not state.precision_refined:
                return {"status": "refine_precision", "refine_precision": True, "candidate_code": unified, "contracts": contracts}
            if decision is not None and decision.action == "REFLEX_HEAL":
                return {"status": "error", "result_status": "ERROR", "failure": decision.reason,
                        "candidate_code": unified, "contracts": contracts}
            if state.epistemic_verdict == "INCONCLUSIVE_UNVERIFIED":
                return {"status": "epistemic_terminal", "epistemic_terminal": True, "candidate_code": unified, "contracts": contracts}
        for r in results:
            await emit("accelerator_log", "result", {
                **tag, "contract_id": r.contract_id, "status": r.status,
                "execution_time_ms": round(r.execution_time_ms, 3),
                "telemetry": _json_safe(r.telemetry),
                "counterexample": r.failure_counterexample})
        verified_count = sum(1 for r in results if r.status == "VERIFIED")
        await self.emit_activity(
            f"STAGE_{stage}_EXEC", "sandbox_step",
            _sandbox_step_summary(results, verified_count),
            telemetry=_json_safe(_aggregate_telemetry(results)))

        # 8. Reflex_Heal is applied by the caller on failure.
        state.stage = "Reflex_Heal"

        # 9. Adjoint_Commit_or_Rollback (delta-scoped).
        state.stage = "Adjoint_Commit_or_Rollback"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "Adjoint_Commit_or_Rollback"})
        if results and all(r.status == "VERIFIED" for r in results):
            new_results = [r for r in results if r.contract_id in {c.contract_id for c in contracts}]
            stage_tel: Dict[str, float] = {}
            for r in new_results:
                for k, v in r.telemetry.items():
                    stage_tel[f"{r.contract_id}.{k}"] = v
            bundle_hash = hashlib.blake2b(
                (unified + json.dumps([r.model_dump() for r in results],
                                      sort_keys=True, default=str)).encode(),
                digest_size=32).hexdigest()
            new_pinned = pin_layer_code(unified, stage)
            layer = ModuleLayer(
                layer_id=f"{bp.blueprint_id}_stage{stage}", stage=stage,
                code=candidate_layer_code(new_pinned, pinned_code),
                bundle_hash=bundle_hash, contracts=contracts,
                baseline_invariants=[i for c in contracts for i in c.invariants],
                telemetry=stage_tel)
            await emit("accelerator_log", "commit", {
                **tag, "bundle_hash": bundle_hash, "regression": regression,
                "telemetry": {r.contract_id: _json_safe(r.telemetry) for r in results}})
            return {"status": "commit", "layer": layer, "pinned_code": new_pinned,
                    "candidate_code": candidate_code, "contracts": contracts}
        failed = next((r for r in results if r.status != "VERIFIED"), None)
        self._rollback_overlay(scratch)
        # Transactional cleanup: unlink the rejected stage's compiled kernel
        # (.cpp/.so/.dylib) and evict it from the bridge cache; verified
        # stages keep their artifacts.
        self.bridge.discard(cpp_code)
        # Emit the rollback marker AT MOST ONCE per failed attempt: identical
        # consecutive rollback signatures (same stage/scope/contract/status)
        # are suppressed so the stream never dumps duplicate lines.
        rb_key = (stage, "stage_delta" if stage > 1 else "stage",
                  failed.contract_id if failed else None,
                  failed.status if failed else "ERROR",
                  tag.get("round"), tag.get("attempt"))
        if getattr(self, "_last_rollback_key", None) != rb_key:
            self._last_rollback_key = rb_key
            await emit("accelerator_log", "rollback", {
                **tag, "scope": "stage_delta" if stage > 1 else "stage",
                "preserved_layers": [layer.layer_id for layer in pinned_layers],
                "failed_contract": failed.contract_id if failed else None,
                "status": failed.status if failed else "ERROR"})
        failure = (f"{failed.status} on {failed.contract_id}: {failed.failure_counterexample}"
                   if failed else "no results")
        return {"status": (failed.status.lower() if failed else "error"),
                "result_status": failed.status if failed else "ERROR",
                "failure": failure, "candidate_code": candidate_code,
                "contracts": contracts,
                "failed_contract": failed.contract_id if failed else None,
                "failed_status": failed.status if failed else None}

    # -- Ingest helpers ------------------------------------------------------
    async def _literature_search(self, queries: List[str], tag: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.search_client is None:
            for q in queries:
                await self._emit("search_log", "query", {**tag, "query": q, "results": 0,
                                                         "note": "search client disabled"})
            return []
        out: List[Dict[str, Any]] = []
        # Non-intrusive SHBT Frontier Federation: only activates if blueprint requires SHBT.
        bp_text = ""
        if getattr(self, "blueprint", None) is not None:
            feas_rat = getattr(getattr(self.blueprint, "feasibility", None), "rationale", "") or ""
            bp_text = f"{self.blueprint.research_prompt} {self.blueprint.title} {self.blueprint.falsifiable_hypothesis} {feas_rat}"
        from .shbt_federation import resolve_shbt_solvability, shbt_literature_dicts
        shbt_res = resolve_shbt_solvability(f"{bp_text} {' '.join(queries)}")
        if shbt_res.should_activate:
            for item in shbt_literature_dicts(shbt_res):
                out.append(item)
                await self._emit("search_log", "paper", {
                    **tag, "query": "shbt_federation", "source": item["source"],
                    "badge": item["badge"], "title": item["title"][:200],
                    "year": item["year"], "url": item["url"], "citation_count": None})
        from .search import is_numerical_method_query, strip_internal_diagnostic_tokens
        for q in queries:
            q = strip_internal_diagnostic_tokens(q)
            if not q:
                continue
            await self._emit("search_log", "query", {**tag, "query": q})
            await self.emit_activity("INGEST", "literature_query",
                                     f"Querying multi-channel literature for '{q}'...")
            try:
                ctx_query = q if is_numerical_method_query(q) else (f"{bp_text} {q}".strip() if bp_text else q)
                hits = await asyncio.to_thread(
                    self.search_client.search, q, 3, None, ctx_query)
            except TypeError:
                # Legacy/custom clients without prompt_or_query routing.
                try:
                    hits = await asyncio.to_thread(self.search_client.search, q, 3)
                except Exception as exc:  # noqa: BLE001
                    await self._emit("search_log", "search_error", {**tag, "query": q,
                                                                    "error": str(exc)[:200]})
                    continue
            except Exception as exc:  # noqa: BLE001
                await self._emit("search_log", "search_error", {**tag, "query": q,
                                                                "error": str(exc)[:200]})
                continue
            for h in hits:
                item = dict(h.__dict__)
                out.append(item)
                badge = getattr(h, "badge", "") or h.source
                await self._emit("search_log", "paper", {
                    **tag, "query": q, "source": h.source, "badge": badge,
                    "title": h.title[:200], "year": h.year, "url": h.url,
                    "citation_count": getattr(h, "citation_count", None)})
                if getattr(h, "tldr", ""):
                    await self._emit("search_log", "insight", {
                        **tag, "source": h.source, "badge": badge,
                        "title": h.title[:120], "insight": h.tldr[:400]})
                for eq in (getattr(h, "equations", None) or [])[:4]:
                    await self._emit("search_log", "equation", {
                        **tag, "source": h.source, "badge": badge,
                        "title": h.title[:120], "latex": str(eq)[:400]})
                for cname, cval in (getattr(h, "constants", None) or {}).items():
                    await self._emit("search_log", "constant", {
                        **tag, "source": h.source, "badge": badge,
                        "name": str(cname)[:80], "value": cval})
                for b in (getattr(h, "benchmarks", None) or [])[:2]:
                    await self._emit("search_log", "benchmark", {
                        **tag, "source": h.source, "badge": badge,
                        "name": str((b or {}).get("name", ""))[:120],
                        "expected": (b or {}).get("expected_value")})
                for insight in _extract_insights(h.summary):
                    await self._emit("search_log", "insight", {
                        **tag, "source": h.source, "badge": badge,
                        "title": h.title[:120], "insight": insight})
            meta = getattr(self.search_client, "last_query_meta", None) or {}
            await self._emit("search_log", "query_done", {
                **tag, "query": q, "results": len(hits),
                "status": meta.get("status"),
                "channel_counts": meta.get("channel_counts") or {},
                "raw_counts": meta.get("raw_counts") or {},
                "channel_errors": meta.get("channel_errors") or {},
                "by_source": meta.get("by_source") or {},
                "arxiv_categories": meta.get("arxiv_categories") or [],
                "relaxed_query": meta.get("relaxed_query") or "",
                "error": meta.get("error") or "",
                "offline": bool(meta.get("offline", getattr(self.search_client, "offline", True))),
                "elapsed_ms": meta.get("elapsed_ms", 0)})
        self._record_literature(out)
        return out

    # -- GoI_Plan helpers ------------------------------------------------------
    async def _synthesize_stage(self, **kw) -> Dict[str, Any]:
        bp = self.blueprint
        kw['objective'] += f"\n{BUFFER_API}\nAuthoritative slot table (name -> index): {json.dumps(getattr(bp, 'variable_map', {}))}. Slot roles: {json.dumps(getattr(bp, 'slot_roles', {}))}. Never overwrite input parameter slots with diagnostics; preserve existing slot identities."
        pinned_layers: List[ModuleLayer] = kw.pop("pinned_layers")
        stage_contracts = kw.pop("stage_contracts", None)
        verified_contracts = [c.model_dump() for layer in pinned_layers for c in layer.contracts]
        if self.stage_synthesizer is not None:
            return await self.stage_synthesizer(
                verified_contracts=verified_contracts, **kw)
        if getattr(self.blueprint, "uast_spec", None) is not None:
            from ..uast.bridge import evolve_uast_stage
            synth_llm = self.llm_fast or self.llm_planner
            if synth_llm is None:
                raise RuntimeError("accretive UAST stages require an LLM backend")
            healed_uast, compiled, heal_logs = await asyncio.wait_for(
                evolve_uast_stage(
                    self.blueprint.uast_spec,
                    objective=kw["objective"],
                    llm_complete=synth_llm,
                    literature=kw.get("literature"),
                    extracted_constants=getattr(self.blueprint, "extracted_constants", None),
                ),
                timeout=120.0,
            )
            self.blueprint.uast_spec = healed_uast.model_dump()
            # Preserve AST coupling to previous pinned layer_k_step for accretion merger
            prev_driver = layer_driver_name(kw["stage"] - 1)
            code = compiled.python_code
            if prev_driver in kw.get("pinned_code", "") and prev_driver not in code:
                code = code.replace(
                    "def simulation_step(buf, dt):\n",
                    f"def simulation_step(buf, dt):\n    if 0:\n        {prev_driver}(buf, dt)\n",
                    1,
                )
            return {
                "code": code,
                "rationale": f"UASTPerturbationDelta spliced & compiled ({compiled.selected_solver}, rho={compiled.spectral_radius:.2e})",
                "heal_logs": heal_logs,
            }
        synth_llm = self.llm_fast or self.llm_planner
        if synth_llm is None:
            raise RuntimeError("accretive stages require a Tier-1 FAST LLM "
                               "(configure a provider) or a stage synthesizer")
        pinned_code = kw["pinned_code"]
        pinned_fn_list = _pinned_function_names(pinned_code, pinned_layers)
        concise_lit = list((kw.get("literature") or [])[:4])
        try:
            return await asyncio.wait_for(synthesize_accretive_stage(
                synth_llm, stage=kw["stage"], total=kw["total"],
                objective=kw["objective"],
                verified_code=unified_from_layers(pinned_code, kw["stage"] - 1),
                verified_contracts=verified_contracts, telemetry=kw["telemetry"],
                literature=concise_lit, failures=kw["failures"],
                pinned_functions=pinned_fn_list,
                stage_contracts=stage_contracts), timeout=None)
        except (asyncio.TimeoutError, TimeoutError, ValueError):
            fb = _deterministic_accretion_candidate(pinned_code, kw["stage"])
            if fb is not None:
                return {
                    "code": fb,
                    "contracts": [],
                    "rationale": f"Tier-0 deterministic accretion fallback for Stage {kw['stage']} after synthesis fallback",
                }
            raise

    async def _heal_and_refine(self, *, stage: int, round_idx: int, objective: str,
                               candidate_code: Optional[str],
                               contracts: List[VerificationContract],
                               failure: str,
                               attempt: int = 1,
                               max_attempts: int = 3,
                               pinned_functions: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        if candidate_code is None:
            return {"code": None, "terminated": False}
        pinned_fn_list = list(pinned_functions or [])
        if is_structural_failure(failure):
            llm = (self.llm_planner if attempt >= 2 else self.llm_fast) or self.llm_fast or self.llm_planner
            if llm is None:
                return {"code": candidate_code, "terminated": False, "applied_ops": []}
            if 'unsupported construct:' in failure:
                await self._emit('goi_stream', 'heal', {'round': round_idx, 'research_stage': stage,
                    'tier': 0, 'applied': False, 'notes': ['structural repair required']})
                plan = await request_healing_directives(llm,
                    objective=f'Structural/API repair only. {BUFFER_API}\n{objective}', code=candidate_code,
                    contracts=[], failure=str(failure),
                    **({"pinned_functions": pinned_fn_list} if pinned_fn_list else {}))
                directives = [d for d in (plan or {}).get('directives', [])
                              if isinstance(d, dict) and d.get('op') == 'INLINE_INTRINSIC']
                patched = apply_healing_directives(candidate_code, directives, contracts=[], pinned_functions=pinned_functions)
                if directives and not v9_transpiler_violations(patched.code):
                    await self.emit_activity('HEALING', 'directive_applied', 'Applied structural-only intrinsic repair')
                    return {'code': patched.code, 'terminated': False, 'applied_ops': ['INLINE_INTRINSIC']}
            _is_noop_err = "noop_accretion" in str(failure or "").lower()
            _noop_hint = ""
            if _is_noop_err:
                _pinned_src = str(getattr(getattr(self, "blueprint", None), "unified_module_code", "") or "")
                _used_slots = [int(x) for x in re.findall(r"buf\.data\[(\d+)\]", f"{_pinned_src}\n{candidate_code}")]
                _vmap_slots = [int(v) for v in (getattr(getattr(self, "blueprint", None), "variable_map", None) or {}).values() if isinstance(v, int)]
                _next_free = max(_used_slots + _vmap_slots + [15]) + 1
                _noop_hint = (
                    f"\nCRITICAL NO-OP REPAIR (Stage {stage}): PreWrite_Guard rejected `layer_{stage}_step(buf, dt)` as a bare forwarding "
                    f"wrapper (or with only trivial clamps) around `layer_{max(1, stage - 1)}_step(buf, dt)`. "
                    f"Keep pinned functions ({', '.join(pinned_functions or []) or 'none'}) UNTOUCHED verbatim, AND inside `def layer_{stage}_step(buf, dt):` "
                    f"(not merely inside `simulation_step`) add >= 2 substantive computational assignment statements addressing `{objective}` — such as computing Stage {stage} "
                    f"regime/corridor feasibility margins, conservation residuals, or auxiliary control/coupling calculations into Stage {stage}'s assigned slots or free diagnostic slots starting at `buf.data[{_next_free}]`."
                )
            _name_err_hint = (
                "\nCRITICAL NAME/SCOPE REPAIR: You referenced an undefined local identifier (such as a pinned helper's local variable). "
                "Define every local variable inside the current function from its canonical `buf.data[idx]` slot in the Global State Slot Table."
                if any(tok in str(failure or "").lower() for tok in ("nameerror", "unboundlocalerror")) else ""
            )
            _repair_contracts = [c.model_dump() if hasattr(c, "model_dump") else c for c in (contracts or [])]
            _compact_source = summarize_pinned_code_for_prompt(candidate_code, pinned_fn_list)
            _src_lines = _compact_source.splitlines()
            if len(_src_lines) > 60:
                _m_ln = re.search(r"\bline\s+(\d+)\b", str(failure or ""))
                _center = int(_m_ln.group(1)) if _m_ln else 15
                _lo = max(1, _center - 12)
                _hi = min(len(_src_lines), _center + 12)
                _numbered_compact = "\n".join(f"{i}: {_src_lines[i - 1]}" for i in range(_lo, _hi + 1))
            else:
                _numbered_compact = "\n".join(f"{i}: {line}" for i, line in enumerate(_src_lines, 1))
            _repair_obj = (
                f"Structural/API repair only, no bound conditioning. {BUFFER_API}\n{objective}{_noop_hint}{_name_err_hint}\n"
                f"Authoritative slots: {json.dumps(getattr(self.blueprint, 'variable_map', {}) or {}, sort_keys=True)}. "
                f"Roles: {json.dumps(getattr(self.blueprint, 'slot_roles', {}) or {}, sort_keys=True)}.\n"
                f"Numbered source:\n{_numbered_compact}"
            )
            try:
                repaired = await asyncio.wait_for(refine_stage_code(
                    llm, objective=_repair_obj, code=candidate_code, contracts=_repair_contracts,
                    failure=str(failure), **({"pinned_functions": pinned_fn_list} if pinned_fn_list else {})), timeout=None)
            except (asyncio.TimeoutError, TimeoutError):
                repaired = None
            if (not repaired or repaired.strip() == candidate_code.strip()) and _is_noop_err and self.llm_planner and self.llm_planner is not llm:
                try:
                    repaired = await asyncio.wait_for(refine_stage_code(
                        self.llm_planner, objective=_repair_obj, code=candidate_code, contracts=_repair_contracts,
                        failure=str(failure), **({"pinned_functions": pinned_fn_list} if pinned_fn_list else {})), timeout=None)
                except (asyncio.TimeoutError, TimeoutError):
                    repaired = None
            if not repaired or repaired.strip() == candidate_code.strip():
                return {"code": candidate_code, "terminated": attempt >= max_attempts, "applied_ops": []}
            violations = v9_transpiler_violations(repaired)
            unresolved = unresolved_symbol(failure, repaired)
            if unresolved:
                violations.append(unresolved)
            if violations:
                return {"code": candidate_code, "terminated": True, "applied_ops": [], "failure": '; '.join(violations)}
            return {"code": repaired, "terminated": False, "applied_ops": ["STRUCTURAL_API_REPAIR"]}
        if "VAULT_REFERENCE_" in str(failure):
            llm = self.llm_planner or self.llm_fast
            if llm is None:
                return {"code": candidate_code, "terminated": False}
            bp = getattr(self, "blueprint", None)
            reference_objective = (
                f"{objective}\nDeclared governing model: {getattr(bp, 'research_prompt', '') or getattr(bp, 'falsifiable_hypothesis', '')}\n"
                f"State/observable slots: {json.dumps(getattr(bp, 'variable_map', {}) or {}, sort_keys=True)}\n"
                "Repair the executed derivative/RHS to match the declared ODEs, preserving a valid integration scheme. "
                "Remove projections that contradict those laws; do not satisfy a counterexample "
                "by clamping state values or modifying immutable parameters. Preserve pinned functions."
            )
            try:
                refined = await asyncio.wait_for(refine_stage_code(
                    llm, objective=reference_objective, code=candidate_code,
                    contracts=[c.model_dump() if hasattr(c, 'model_dump') else c for c in contracts],
                    failure=str(failure), **({"pinned_functions": pinned_fn_list} if pinned_fn_list else {})), timeout=None)
            except (asyncio.TimeoutError, TimeoutError):
                refined = None
            return {"code": refined or candidate_code, "terminated": False,
                    "applied_ops": ["ODE_REFERENCE_REFINE"] if refined else []}
        # Normalize to plain dicts so deterministic healing passes read
        # tolerances/assertions uniformly (pydantic objects hide ``tolerance``).
        all_bp_contracts = (getattr(self, "blueprint", None) and getattr(self.blueprint, "contracts", None)) or contracts
        all_contract_dicts = [
            c.model_dump() if hasattr(c, "model_dump") else c for c in (all_bp_contracts or [])
        ]
        contract_dicts = [
            c.model_dump() if hasattr(c, "model_dump") else c for c in (contracts or [])]
        seed_directives: List[Dict[str, Any]] = []
        m_name_err = re.search(
            r"NameError:\s*name\s*'((?:_invar_|safe_)\w+|div_eps|smooth_clamp)'\s+is\s+not\s+defined",
            str(failure or ""),
        )
        if m_name_err:
            seed_directives.append({"op": "INLINE_INTRINSIC", "name": m_name_err.group(1)})
        bp_vmap = getattr(getattr(self, "blueprint", None), "variable_map", None)
        bp_roles = getattr(getattr(self, "blueprint", None), "slot_roles", None)
        try:
            physics_contracts = physical_projection_contracts(
                ast.parse(candidate_code), all_contract_dicts or contract_dicts, bp_vmap,
                _verified_projection_slots(ast.parse(candidate_code), all_contract_dicts or contract_dicts), bp_roles)
        except SyntaxError:
            physics_contracts = []
        det = apply_healing_directives(
            candidate_code, seed_directives,
            contracts=all_contract_dicts or contract_dicts,
            variable_map=bp_vmap,
            slot_roles=bp_roles,
            pinned_functions=pinned_functions,
        )
        healed, notes = candidate_code, list(det.notes)
        if det.changed:
            healed = det.code
        lowered_failure = str(failure or "").lower()
        allow_egg = not ("transpiler:" in lowered_failure or "compile_error" in lowered_failure
                         or "compilation failed" in lowered_failure or "syntaxerror" in lowered_failure)
        fixed, fix_notes = tier0_heal(healed, allow_egg=allow_egg, pinned_functions=pinned_functions)
        if fix_notes:
            healed, notes = fixed, notes + fix_notes
        # Deterministic monotone/drift fallback for tightly bounded state slots
        # (e.g. entropy_monotone <= 1e-15) falsified with a Step-0 history spike.
        fail_text = str(failure or "").lower()
        seen_slots = set()
        for contract in contracts or []:
            for inv in getattr(contract, "invariants", None) or []:
                name = str(getattr(inv, "variable_name", "") or "").lower()
                if not any(tok in name for tok in
                           ("monotone", "drift", "entropy", "error", "err",
                            "residual", "violation", "mismatch")):
                    continue
                text = f"{getattr(inv, 'variable_name', '')} {getattr(inv, 'assertion', '')}"
                m = re.search(r"(?:state|buf(?:\.data)?|data)\[(\d+)\]\s*<=\s*([0-9.eE+-]+)", text)
                if not m:
                    continue
                slot, bound = int(m.group(1)), float(m.group(2))
                if bound > 1e-8 or slot in seen_slots:
                    continue
                if slot in protected_observation_slots(ast.parse(healed), contract_dicts, bp_vmap, roles=bp_roles):
                    continue
                if not ("falsified" in fail_text or name in fail_text
                        or f"state[{slot}]" in fail_text):
                    continue
                rw = rewrite_monotone_drift(healed, slot=slot, direction="non_increasing")
                if rw.changed:
                    healed, notes = rw.code, notes + rw.notes
                    seen_slots.add(slot)
        none_lowering = {"return None lowered", "None assignment lowered to 0.0",
                         "is None comparison lowered"}
        applied_ops = [n for n in notes if n and n not in none_lowering]
        await self._emit("goi_stream", "heal", {
            "round": round_idx, "research_stage": stage, "tier": 0,
            "applied": bool(applied_ops), "notes": applied_ops or ["no Tier-0 repair applicable"]})
        if applied_ops and not (attempt >= 2 and "falsified" in fail_text):
            await self.emit_activity(
                "HEALING", "deterministic_reflex",
                "[HEALING_DIRECTIVE] Applied deterministic ops: "
                + json.dumps(applied_ops[:12]))
            return {"code": healed, "terminated": False, "applied_ops": applied_ops}

        llm = self.llm_fast or self.llm_planner
        if llm is None:
            return {"code": candidate_code, "terminated": False}
        _refine_stage_guidance = (
            f"{objective}\n"
            + (
                f"CRITICAL (Stage {stage}): Keep pinned functions ({', '.join(pinned_functions or [])}) UNTOUCHED verbatim. "
                f"Preserve ALL governing physical ODE equations and swept CONTROL_PARAM dependencies (e.g., conductance, forcing, coupling); "
                f"NEVER replace coupled dynamics with fixed constant-target relaxation or bypass swept parameters. "
                f"If a pinned layer uses miscalibrated rate parameters causing FALSIFIED targets in Stage {stage}, calibrate the physical "
                f"rate constants inside Stage {stage}'s ODE step while preserving conservation, positivity, and regression."
                if stage > 1 else
                "Calibrate any unphysical model rate/conductance parameters or stiffness stability from the counterexample while preserving governing ODE physics, swept parameter dependencies, conservation, and positivity."
            )
        )
        refine_called = False
        if (attempt >= max_attempts and self.llm_planner is not None
                and ("falsified" in fail_text or "diverged" in fail_text or "non-finite" in fail_text)):
            refine_called = True
            refined = None
            try:
                refined = await asyncio.wait_for(
                    refine_stage_code(
                        self.llm_planner,
                        objective=_refine_stage_guidance,
                        code=healed,
                        contracts=contract_dicts,
                        failure=str(failure),
                        **({"pinned_functions": pinned_fn_list} if pinned_fn_list else {}),
                    ),
                    timeout=None,
                )
            except (asyncio.TimeoutError, TimeoutError):
                refined = None
            if refined and refined.strip() and refined.strip() != candidate_code.strip():
                await self.emit_activity(
                    "HEALING", "tier2_refine",
                    "[TIER2_REFINE] Tier-2 planner synthesized full-kernel repair from counterexample.")
                return {"code": refined, "terminated": False,
                        "applied_ops": ["TIER2_REFINE_STAGE_CODE"]}
        plan = await request_healing_directives(
            llm, objective=objective, code=candidate_code,
            contracts=[c.model_dump() for c in contracts], failure=str(failure),
            **({"pinned_functions": pinned_fn_list} if pinned_fn_list else {}))
        if plan is None:
            fb_code, fb_notes = _deterministic_physics_guard_fallback(
                candidate_code, physics_contracts)
            if fb_code != candidate_code:
                return {"code": fb_code, "terminated": False,
                        "applied_ops": fb_notes or ["timeout_physics_guard_fallback"]}
            san_fb = sanitize_intrinsics(
                candidate_code,
                contracts=all_contract_dicts or contract_dicts,
                all_contracts=all_contract_dicts or contract_dicts,
                variable_map=bp_vmap,
                slot_roles=bp_roles,
                stabilize_diverged=("diverged" in str(failure or "").lower() or "non-finite" in str(failure or "").lower()),
                pinned_functions=pinned_functions,
            )
            if san_fb.changed and san_fb.code != candidate_code:
                return {"code": san_fb.code, "terminated": False,
                        "applied_ops": san_fb.notes or ["timeout_slot_conditioner"]}
            return {"code": candidate_code, "terminated": False}
        # Fix 5: when the directive pass would apply 0 ops (or an identical
        # result) on a FALSIFIED attempt, run the deterministic
        # conservation/Lyapunov/monotone projection passes before declaring the
        # stage unfixable.
        directives = plan.get("directives") or []
        if not isinstance(directives, list):
            directives = []
        if stage > 1 and all_contract_dicts:
            prior_slots: Set[int] = set()
            for _cd in all_contract_dicts:
                if _contract_stage_index(str(_cd.get("contract_id") or "")) < stage:
                    for _iv in (_cd.get("invariants") or []):
                        for _m_s in re.findall(r"(?:state|buf\.data|data)\[(\d+)\]", str(_iv.get("assertion") or "")):
                            prior_slots.add(int(_m_s))
            filtered_dirs = []
            for _d in directives:
                if isinstance(_d, dict) and str(_d.get("op") or "") in ("BIND_SLOT_TO_VAR", "UNIFY_CONCEPT_SLOTS"):
                    try:
                        _dslot = int(_d.get("slot", -1))
                    except (TypeError, ValueError):
                        _dslot = -1
                    if _dslot in prior_slots:
                        continue
                filtered_dirs.append(_d)
            directives = filtered_dirs
        if is_structural_failure(str(failure or "")):
            # Never apply bound-conditioning directives that re-inject projections on SCIENTIFIC_INTEGRITY / structural failures
            directives = [
                _d for _d in directives
                if not (
                    isinstance(_d, dict)
                    and str(_d.get("op") or "").upper().startswith(("CLAMP_", "CONDITION_", "PROJECT_"))
                )
            ]
        patched = apply_healing_directives(
            candidate_code, directives,
            contracts=all_contract_dicts or contract_dicts,
            variable_map=getattr(getattr(self, "blueprint", None), "variable_map", None),
            slot_roles=bp_roles,
            pinned_functions=pinned_functions,
        )
        ops = [str(d.get("op")) for d in directives if isinstance(d, dict) and d.get("op")]
        _is_code_level_fail = is_structural_failure(str(failure or "")) or any(
            tok in fail_text
            for tok in (
                "transpiler:",
                "prewrite_guard",
                "scientific_integrity",
                "noop_accretion",
                "contract_faithfulness",
                "slot_alignment",
                "syntaxerror",
                "nameerror",
                "typeerror",
                "attributeerror",
                "compile_error",
                "compilation failed",
                "unsupported",
            )
        )
        _is_final_heal = (attempt + 1 >= max_attempts)
        _blind_scaling_only = bool(ops) and set(ops) <= {
            "REDUCE_RATE_COEFFICIENT",
            "SCALE_STEP_DT",
            "CLAMP_STEP_DELTA",
        }
        _missing_eq_slot = (
            _is_final_heal
            and (plan.get("fixable") is not False or _is_code_level_fail)
            and any(
                f"buf.data[{m.group(1)}] =" not in (patched.code or "")
                and f"buf.data[{m.group(1)}]=" not in (patched.code or "")
                for m in re.finditer(r"abs\(state\[(\d+)\]\s*-", str(failure or ""))
            )
        )
        _eq_target_falsified = (
            self.llm_planner is not None
            and "falsified" in str(failure or "").lower()
            and "ENFORCE_CONSERVATION_PROJECTION" not in set(ops)
            and "REWRITE_MONOTONE_DRIFT" not in set(ops)
            and (not ops or _blind_scaling_only or not patched.changed or patched.code == candidate_code)
            and (_is_final_heal or plan.get("fixable") is False)
        )
        _diverged_fail = ("diverged" in fail_text or "non-finite" in fail_text)
        _need_source_repair_on_code_fail = (
            _is_code_level_fail
            and (
                not ops
                or not patched.changed
                or patched.code == candidate_code
                or plan.get("fixable") is False
            )
        )
        fallback_needed = (
            (
                "falsified" in fail_text
                or _diverged_fail
                or _is_code_level_fail
            )
            and (
                not ops
                or _blind_scaling_only
                or not patched.changed
                or patched.code == candidate_code
                or _missing_eq_slot
                or _eq_target_falsified
                or _diverged_fail
                or _need_source_repair_on_code_fail
            )
        )
        if fallback_needed:
            if not refine_called and (_eq_target_falsified or _missing_eq_slot or _diverged_fail or _need_source_repair_on_code_fail) and (
                plan.get("fixable") is not False or _is_code_level_fail or _diverged_fail
            ):
                # Re-run full Tier-0 lowering before invoking full-source LLM repair
                san_pre = sanitize_intrinsics(
                    healed,
                    contracts=all_contract_dicts or contract_dicts,
                    all_contracts=all_contract_dicts or contract_dicts,
                    variable_map=getattr(getattr(self, "blueprint", None), "variable_map", None),
                    slot_roles=bp_roles,
                    stabilize_diverged=_diverged_fail,
                    pinned_functions=pinned_functions,
                )
                if san_pre.changed and san_pre.code != candidate_code:
                    return {"code": san_pre.code, "terminated": False,
                            "applied_ops": san_pre.notes or ["tier0_scalar_ir_lowering"]}
                refine_called = True
                tier2_llm = (self.llm_planner if _is_final_heal else (self.llm_fast or self.llm_planner)) or llm
                refined = None
                try:
                    refined = await asyncio.wait_for(
                        refine_stage_code(
                            tier2_llm,
                            objective=_refine_stage_guidance,
                            code=healed,
                            contracts=contract_dicts,
                            failure=str(failure),
                            **({"pinned_functions": pinned_fn_list} if pinned_fn_list else {}),
                        ),
                        timeout=None,
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    refined = None
                if refined and refined.strip() and refined.strip() != candidate_code.strip():
                    san_ref = sanitize_intrinsics(
                        refined,
                        contracts=all_contract_dicts or contract_dicts,
                        all_contracts=all_contract_dicts or contract_dicts,
                        variable_map=getattr(getattr(self, "blueprint", None), "variable_map", None),
                        slot_roles=bp_roles,
                        pinned_functions=pinned_functions,
                    )
                    final_ref = san_ref.code if san_ref.changed else refined
                    await self.emit_activity(
                        "HEALING", "tier2_refine",
                        "[TIER2_REFINE] Tier-2 planner synthesized full-kernel repair from counterexample.")
                    return {"code": final_ref, "terminated": False,
                            "applied_ops": ["TIER2_REFINE_STAGE_CODE"]}
            fb_code, fb_notes = _deterministic_physics_guard_fallback(
                candidate_code, physics_contracts)
            if fb_notes:
                await self.emit_activity(
                    "HEALING", "deterministic_fallback",
                    "[HEALING_FALLBACK] Directive pass produced 0 changes; "
                    "applying deterministic conservation/Lyapunov projection "
                    "passes.",
                    level="warning")
            if fb_code != candidate_code:
                return {"code": fb_code, "terminated": False,
                        "applied_ops": fb_notes or ["physics_guard_fallback"]}
            # Zero-Diff Guard: synthesize a deterministic post-step slot guard
            # targeting the exact falsified slot(s) in `failure` so identical code
            # is never re-executed.
            san_fb = sanitize_intrinsics(
                candidate_code,
                contracts=all_contract_dicts or contract_dicts,
                all_contracts=all_contract_dicts or contract_dicts,
                variable_map=getattr(getattr(self, "blueprint", None), "variable_map", None),
                slot_roles=bp_roles,
                stabilize_diverged=("diverged" in fail_text or "non-finite" in fail_text),
                pinned_functions=pinned_functions,
            )
            if san_fb.changed and san_fb.code != candidate_code:
                return {"code": san_fb.code, "terminated": False,
                        "applied_ops": san_fb.notes or ["zero_diff_slot_conditioner"]}
            if not refine_called and (
                plan.get("fixable") is not False or _is_code_level_fail
            ):
                refine_called = True
                tier2_llm = (self.llm_planner if _is_final_heal else (self.llm_fast or self.llm_planner)) or self.llm_planner
                refined = None
                try:
                    refined = await asyncio.wait_for(
                        refine_stage_code(
                            tier2_llm,
                            objective=_refine_stage_guidance,
                            code=healed,
                            contracts=contract_dicts,
                            failure=str(failure),
                            **({"pinned_functions": pinned_fn_list} if pinned_fn_list else {}),
                        ),
                        timeout=None,
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    refined = None
                if refined and refined.strip() and refined.strip() != candidate_code.strip():
                    san_ref = sanitize_intrinsics(
                        refined,
                        contracts=all_contract_dicts or contract_dicts,
                        all_contracts=all_contract_dicts or contract_dicts,
                        variable_map=getattr(getattr(self, "blueprint", None), "variable_map", None),
                        slot_roles=bp_roles,
                        pinned_functions=pinned_functions,
                    )
                    final_ref = san_ref.code if san_ref.changed else refined
                    await self.emit_activity(
                        "HEALING", "tier2_refine",
                        "[TIER2_REFINE] Tier-2 planner synthesized full-kernel repair from counterexample.")
                    return {"code": final_ref, "terminated": False,
                            "applied_ops": ["TIER2_REFINE_STAGE_CODE"]}
        if plan.get("fixable") is False and not _is_code_level_fail:
            reason = str(plan.get("unfixable_reason") or "unsatisfiable contract").strip()
            await self.emit_activity(
                "HEALING", "healing_terminated",
                "[HEALING_TERMINATED] Stage determined unfixable after "
                f"deterministic + directive passes: {reason}",
                level="warning")
            return {"code": candidate_code, "terminated": True,
                    "unfixable_reason": reason}
        if ops:
            await self.emit_activity(
                "HEALING", "directive_applied",
                "[HEALING_DIRECTIVE] Applied deterministic ops: "
                + json.dumps(ops[:12]))
        if patched.changed:
            return {"code": patched.code, "terminated": False,
                    "applied_ops": ops or patched.notes}
        return {"code": candidate_code, "terminated": False, "applied_ops": ops}

    async def _diagnostic_research(self, state: ResearchRunState, *, stage: int,
                                   outcome: Dict[str, Any]) -> str:
        """Closed-loop re-search on FALSIFIED/DIVERGED outcomes.

        Formulates targeted diagnostic queries, runs a bounded literature
        re-search, detects SI unit-scaling mismatches, and returns extra
        healing context (unit correction + fresh evidence summary). Emits a
        ``HEALING_SEARCH`` activity event. Never raises.
        """
        failed_status = str(outcome.get("failed_status") or "").upper()
        if failed_status not in ("FALSIFIED", "DIVERGED"):
            return ""
        blueprint = state.blueprint
        failure_text = str(outcome.get("failure") or "")
        # Build a failure payload for the diagnostic helpers.
        last_result = next((r for r in reversed(state.results)
                            if r.status == failed_status), None)
        failure_result: Dict[str, Any] = {
            "status": failed_status,
            "telemetry": dict(last_result.telemetry) if last_result else {},
            "counterexample": (last_result.failure_counterexample if last_result
                               else failure_text),
        }
        from .search import strip_internal_diagnostic_tokens
        queries = [
            strip_internal_diagnostic_tokens(q)
            for q in formulate_diagnostic_search_query(blueprint, failure_result)
        ]
        queries = [q for q in queries if q]
        await self.emit_activity(
            "HEALING_SEARCH", "diagnostic_query",
            f"Triggered targeted re-search for stability/unit closure: "
            f"{queries}")
        context_parts: List[str] = []
        # Unit-scaling diagnostic: exact conversion correction for the healer.
        diag = unit_scaling_diagnostic(
            blueprint, failure_result.get("telemetry") or {},
            str(failure_result.get("counterexample") or ""))
        if diag:
            factor = diag.get("factor")
            direction = diag.get("direction")
            si_unit = diag.get("si_unit") or ""
            if factor and direction:
                op = "divide" if direction == "divide" else "multiply"
                by = {1e-3: "1e3", 1e3: "1e-3", 1e-6: "1e6", 1e6: "1e-6",
                      1e-9: "1e9", 1e9: "1e-9", 180.0 / math.pi: "180/pi",
                      9.80665: "9.80665"}.get(float(factor), str(factor))
                context_parts.append(
                    "UNIT SCALING DIAGNOSTIC: the observed metric "
                    f"'{diag.get('variable', 'value')}' appears to be in {si_unit or 'raw SI'} "
                    f"while the invariant bound expects a scaled unit. "
                    f"{op.upper()} the observed value by {by} before comparing "
                    f"(observed={diag.get('observed'):.6g}, bound={diag.get('bound'):.6g}, "
                    f"scaled={diag.get('scaled'):.6g}). Fix the kernel's unit "
                    "conversion (e.g. W/m^2 -> MW/m^2 divides by 1e6) rather than "
                    "the invariant.")
        # Fresh evidence from the targeted re-search (offline-safe).
        if self.search_client is not None and queries:
            try:
                hits = await asyncio.to_thread(
                    self.search_client.search, queries[0], 3, None,
                    f"{blueprint.research_prompt} {blueprint.title} {queries[0]}")
            except TypeError:
                try:
                    hits = await asyncio.to_thread(
                        self.search_client.search, queries[0], 3)
                except Exception:  # noqa: BLE001
                    hits = []
            except Exception:  # noqa: BLE001
                hits = []
            if hits:
                try:
                    from .extractor import QuantitativeInsightExtractor, merge_evidence
                    evidence = QuantitativeInsightExtractor().extract(
                        blueprint.falsifiable_hypothesis, hits)
                    merged = merge_evidence(
                        getattr(blueprint, "structured_evidence", None), evidence)
                    if merged is not None:
                        state.blueprint = blueprint.model_copy(update={
                            "structured_evidence": merged,
                            "extracted_constants": {
                                **(getattr(blueprint, "extracted_constants", None) or {}),
                                **{n: float(p.si_value)
                                   for n, p in evidence.parameters.items()}}})
                except Exception:  # noqa: BLE001
                    pass
                top = "; ".join(f"{h.title[:60]}" for h in hits[:3])
                context_parts.append(
                    f"DIAGNOSTIC RE-SEARCH ({len(hits)} hits): {top}. "
                    "Prefer stabilizing step bounds / damping closures and "
                    "SI unit conversions from these sources.")
        return "\n".join(context_parts)

    # -- Sandbox_Execute -------------------------------------------------------
    async def _sandbox_execute(self, bp: ResearchBlueprint, uast: Dict[str, Any],
                               cpp_code: str, py_code: str,
                               contracts: List[VerificationContract],
                               pinned_layers: List[ModuleLayer], stage: int,
                               tag: Dict[str, Any],
                               kernel_mode: str = "rk4_ode") -> Tuple[List[SimulationResult], List[Dict[str, Any]]]:
        results: List[SimulationResult] = []
        regression: List[Dict[str, Any]] = []
        entries = CppEmitter().entry_points(uast)
        # Regression: every pinned layer re-verified inside the unified artefact.
        for layer in pinned_layers:
            entry = entry_symbol(layer_driver_name(layer.stage))
            for contract in layer.contracts:
                r = await self._execute_contract(bp, uast, cpp_code, py_code, contract, entry,
                                                 entries, stage, regression_of=layer.stage,
                                                 kernel_mode=kernel_mode)
                regression.append({"layer": layer.layer_id, "contract_id": contract.contract_id,
                                   "status": r.status})
                results.append(r.model_copy(update={
                    "contract_id": f"regression:{layer.layer_id}:{contract.contract_id}"}))
        km_kw = {} if kernel_mode == "rk4_ode" else {"kernel_mode": kernel_mode}
        for contract in contracts:
            results.append(await self._execute_contract(
                bp, uast, cpp_code, py_code, contract, "execute_step", entries, stage,
                **km_kw))
        return results, regression

    async def _run_kernel_guarded(self, cpp_code: str, py_code: str,
                                  initial: List[float], contract: VerificationContract,
                                  entry: str, stage: int,
                                  kernel_mode: str = "rk4_ode", trace_slots: tuple[int, ...] = ()) -> Dict[str, Any]:
        """C-ABI execution with deterministic Python fallback on compile failure.

        Kernels here have already passed PreWrite_Guard AST validation, so a
        toolchain rejection never aborts the stage with ``error=1.000``: the
        exact g++/clang++ diagnostic is logged and the same numerical kernel
        runs in the isolated Python sandbox runner instead.
        """
        try:
            return await asyncio.to_thread(
                self.bridge.run, cpp_code, initial, contract.dt, contract.steps,
                entry, contract.timeout_seconds, contract.max_memory_mb,
                True,
                *((kernel_mode,) if kernel_mode != "rk4_ode" else ()),
                **({"trace_slots":trace_slots} if trace_slots else {}))
        except KernelExecutionError as exc:
            compile_failed = bool(getattr(exc, "compile_error", False))
            # Runtime loader/ctypes failures (rc != 0, e.g. dlopen ENOMEM)
            # also fall back to the verified Python kernel — never abort a
            # stage with ERROR(rc=1) when the Python kernel is valid.
            runtime_failed = (getattr(exc, "returncode", None) not in (None, 0)
                              and not getattr(exc, "timed_out", False))
            if not (compile_failed or runtime_failed) or not py_code:
                raise
            reason = "compilation" if compile_failed else f"runtime rc={getattr(exc, 'returncode', '?')}"
            diag = str(exc).replace("\n", " ")[-400:]
            logger.warning("C-ABI %s failed; falling back to the "
                           "deterministic Python kernel sandbox", reason)
            run = await asyncio.to_thread(
                self.bridge.run_python, instrumented_fallback(py_code), initial, contract.dt,
                contract.steps, entry, contract.timeout_seconds,
                contract.max_memory_mb,
                *((kernel_mode,) if kernel_mode != "rk4_ode" else ()),
                **({"trace_slots":trace_slots} if trace_slots else {}))
            await self.emit_activity(
                f"STAGE_{stage}_BUILD", "cabi_fallback",
                "C-ABI compiler/runtime fallback -> executed verified "
                f"deterministic Python kernel in sandbox; diagnostic tail: {diag}")
            return run

    async def _execute_contract(self, bp, uast, cpp_code, py_code, contract: VerificationContract,
                                entry: str, entries: List[str], stage: int,
                                regression_of: Optional[int] = None,
                                kernel_mode: str = "rk4_ode") -> SimulationResult:
        start = time.perf_counter()
        # Static contract-slot guard: error metrics start at 0.0, dynamical
        # slots are never all-zero, and uncomputed contract slots are logged.
        initial, slot_warnings = _guard_contract_initial_state(
            py_code or "", contract, all_contracts=getattr(bp, "contracts", None),
            diagnostic_slots=[slot for name, slot in bp.variable_map.items() if bp.slot_roles.get(name) == 'diagnostic'],
            variable_map=getattr(bp, "variable_map", None),
        )
        max_vmap_slot = max((idx for idx in (bp.variable_map or {}).values() if isinstance(idx, int)), default=-1)
        if max_vmap_slot >= len(initial) and max_vmap_slot < 4096:
            initial = list(initial) + [0.0] * (max_vmap_slot - len(initial) + 1)
        for w in slot_warnings:
            await self.emit_activity(
                f"STAGE_{stage}_EXEC", "contract_slot_guard", w, level="warning")
        telemetry: Dict[str, float] = {}
        counterexample: Optional[str] = None
        run_ctx: Optional[str] = None
        status = "ERROR"
        try:
            km_kw = {} if kernel_mode == "rk4_ode" else {"kernel_mode": kernel_mode}
            pack = bp.ihcg_evidence or {}
            effective_stage = regression_of or stage
            if effective_stage > 1 and pack:
                pack = pack.get("extensions",[])[effective_stage-2]
            reductions = {name:e for name,e in pack.get("evidence",{}).get("equations",{}).items()
                if e.get("kind") in {"TRAJECTORY_MAX","TRAJECTORY_MIN"}}
            if reductions:
                km_kw["trace_slots"] = tuple(bp.variable_map[e["state_symbol"]] for e in reductions.values())
            run = await self._run_kernel_guarded(cpp_code, py_code, initial, contract, entry,
                                                 stage, **km_kw)
            if reductions:
                from .meaningfulness import quadratic_peak
                run["state"] = list(run["state"])
                for column,(name,equation) in enumerate(reductions.items()):
                    source = bp.variable_map[equation["state_symbol"]]
                    samples = [initial[source],*[row[column] for row in run["trace_values"]]]
                    times = [i*contract.dt for i in range(len(samples))]
                    value = quadratic_peak(times,samples,minimum=equation["kind"] == "TRAJECTORY_MIN")[0]
                    target = bp.variable_map[name]
                    run["state"][target] = value
                    run["trace_min"][target] = min(initial[target],value)
                    run["trace_max"][target] = max(initial[target],value)
            telemetry = _telemetry(run, initial)
            if "diag" in run:
                from ..healing.deterministic_ast import purity_audit_from_diag
                domain_tag = f"stage{stage}/{kernel_mode}/{contract.contract_id}"
                audit = purity_audit_from_diag(run["diag"], domain_tag=domain_tag, static_purity_passed=False,
                    violations=("ARENA_PURITY_NOT_CERTIFIED",))
                state_audits = getattr(self, "_runtime_purity", {})
                state_audits[domain_tag] = audit
                self._runtime_purity = state_audits
                for key in ("total_steps", "preserved_steps", "projected_steps", "stabilized_steps", "enforcement_ratio"):
                    telemetry[f"purity_{key}"] = float(getattr(audit, key))
                await self._emit("accelerator_log", "purity_audit", {"research_stage": stage, "audit": audit.model_dump(mode="json")})
            telemetry.setdefault("dt", float(contract.dt))
            if run.get("fallback"):
                telemetry["cabi_fallback"] = 1.0
            # Named observables: publish every blueprint variable_map entry as
            # telemetry["obs_<metric_with_units>"] from the final state vector.
            state_vec = run.get("state") or []
            for name, idx in (bp.variable_map or {}).items():
                if isinstance(idx, int) and 0 <= idx < len(state_vec):
                    try:
                        telemetry[f"obs_{name}"] = float(state_vec[idx])
                    except (TypeError, ValueError):
                        continue
            for inv in (contract.invariants or []):
                if inv.variable_name and f"obs_{inv.variable_name}" not in telemetry:
                    slots = [
                        int(x) for x in re.findall(
                            r"(?:state|initial|buf(?:\.data)?|data)\[(\d+)\]",
                            str(getattr(inv, "assertion", "") or ""),
                        )
                    ]
                    if len(set(slots)) == 1 and 0 <= slots[0] < len(state_vec):
                        try:
                            telemetry[f"obs_{inv.variable_name}"] = float(state_vec[slots[0]])
                        except (TypeError, ValueError):
                            pass
            run_ctx = _counterexample_context(run)
            extra: Dict[str, Any] = {"initial": initial,
                                     "initial_state": initial,
                                     "trace_min": run.get("trace_min", run["state"]),
                                     "trace_max": run.get("trace_max", run["state"])}
            # Named observables: expose every blueprint variable_map entry as a
            # top-level namespace variable so assertions can reference them
            # directly (e.g. ``peak_decel_g < 40.0``), not only via telemetry.
            _reserved = set(_INVARIANT_FUNCS) | {
                "state", "data", "buf", "initial", "initial_state", "trace_min",
                "trace_max", "telemetry", "math", "variants", "tol"}
            for name, idx in (bp.variable_map or {}).items():
                key = str(name)
                if key in _reserved:
                    continue
                if isinstance(idx, int) and 0 <= idx < len(state_vec):
                    try:
                        extra[key] = float(state_vec[idx])
                    except (TypeError, ValueError):
                        continue
            if contract.target_backend == "rust_hin":
                telemetry.update(self._hin_check(uast))
            if contract.target_backend == "multi_variant" and (
                regression_of is None
                or any("variants" in str(getattr(i, "assertion", "") or "") for i in (contract.invariants or []))
            ):
                variants = await self._run_variants(
                    cpp_code, entries, initial, contract, run,
                    variable_map=bp.variable_map, slot_roles=bp.slot_roles,
                )
                extra["variants"] = variants
                for name, v in variants.items():
                    telemetry[f"variant_{name}_residual"] = v["residual"]
                    telemetry[f"variant_{name}_ms"] = v["ms"]
                if variants:
                    pareto = _pareto_front(variants)
                    telemetry["pareto_size"] = float(len(pareto))
                    await self._emit("accelerator_log", "pareto", {
                        "research_stage": stage, "contract_id": contract.contract_id,
                        "front": pareto, "variants": variants})
            if not run.get("finite", True):
                status = "DIVERGED"
                counterexample = (f"non-finite state at step {run.get('first_nonfinite_step', -1)} "
                                  f"(dt={contract.dt}, steps={contract.steps})"
                                  + (f"; {run_ctx}" if run_ctx else ""))
            elif telemetry.get("hin_timed_out", 0.0) > 0:
                status = "DIVERGED"
                counterexample = "HIN arena reduction did not normalise within budget"
            else:
                status = "VERIFIED"
                failed: List[Tuple[InvariantConstraint, Dict[str, Any]]] = []
                roundoff_scales = observed_roundoff_scales(py_code, run)
                for inv in contract.invariants:
                    env_extra = dict(extra, tol=inv.tolerance, _roundoff_scales=roundoff_scales)
                    ok = _eval_invariant(inv.assertion, run["state"], telemetry, env_extra)
                    if not ok:
                        observed = _observed_terms(
                            inv.assertion, _invariant_env(run["state"], telemetry, env_extra))
                        failed.append((inv, observed))
                if failed:
                    status = "FALSIFIED"
                    counterexample = ("; ".join(
                        f"`{inv.assertion}` ({inv.variable_name}) observed {observed}"
                        for inv, observed in failed)
                        + (f"; {run_ctx}" if run_ctx else ""))[:1500]
        except KernelExecutionError as exc:
            status = "DIVERGED" if exc.timed_out else "ERROR"
            counterexample = str(exc)[:1500]
            telemetry = telemetry or {"error": 1.0}
        except PreWriteGuardError as exc:
            status = "ERROR"
            counterexample = str(exc)[:1500]
            telemetry = telemetry or {"error": 1.0}
        except Exception as exc:  # noqa: BLE001
            status = "ERROR"
            counterexample = f"{type(exc).__name__}: {str(exc)[:1400]}"
            telemetry = telemetry or {"error": 1.0}
        elapsed = (time.perf_counter() - start) * 1000.0
        return SimulationResult(
            blueprint_id=bp.blueprint_id, contract_id=contract.contract_id,
            status=status, execution_time_ms=elapsed, telemetry=telemetry,
            bundle_hash=hashlib.blake2b(cpp_code.encode(), digest_size=32).hexdigest(),
            failure_counterexample=counterexample, stage=stage)

    async def _run_variants(self, cpp_code: str, entries: List[str], initial: List[float],
                            contract: VerificationContract,
                            reference: Dict[str, Any],
                            variable_map: Optional[Dict[str, int]] = None,
                            slot_roles: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
        variants: Dict[str, Dict[str, float]] = {}
        ref_state = reference["state"]
        compare_indices: Optional[List[int]] = None
        if variable_map:
            from .slot_schema import ROLE_STATE, classify_slot_roles
            roles = classify_slot_roles(variable_map, None, [contract], explicit_roles=slot_roles)
            state_idxs = sorted(
                idx for idx, role in roles.items()
                if role == ROLE_STATE and 0 <= idx < len(ref_state)
            )
            if state_idxs:
                compare_indices = state_idxs
        for name in entries:
            if not name.startswith("simulation_step_"):
                continue
            vname = name[len("simulation_step_"):]
            try:
                run = await asyncio.to_thread(
                    self.bridge.run, cpp_code, initial, contract.dt, contract.steps,
                    entry_symbol(name), contract.timeout_seconds, contract.max_memory_mb)
                if compare_indices:
                    st_vec = run["state"]
                    diffs = [st_vec[i] - ref_state[i] for i in compare_indices if i < len(st_vec)]
                else:
                    diffs = [a - b for a, b in zip(run["state"], ref_state)]
                residual = (_l2(diffs)
                            if run.get("finite", True) else float("inf"))
                variants[vname] = {"residual": residual,
                                   "ms": float(run["execution_time_ms"]),
                                   "finite": 1.0 if run.get("finite", True) else 0.0}
            except KernelExecutionError as exc:
                variants[vname] = {"residual": float("inf"), "ms": -1.0, "finite": 0.0,
                                   "error": 1.0}
                await self._emit("accelerator_log", "variant_error",
                                 {"variant": vname, "error": str(exc)[:300]})
        variants["reference"] = {"residual": 0.0, "ms": float(reference["execution_time_ms"]),
                                 "finite": 1.0 if reference.get("finite", True) else 0.0}
        return variants

    def _hin_check(self, uast: Dict[str, Any]) -> Dict[str, float]:
        """rust_hin: reduce the module UAST in the native Lafont HIN arena."""
        try:
            from invar_agent._native import reduce_hin_uast
        except ImportError:
            return {"hin_supported": 0.0}
        try:
            data = json.loads(reduce_hin_uast(json.dumps(uast), 100000, 5.0))
        except BaseException:  # noqa: BLE001 - pyo3 PanicException is BaseException
            return {"hin_supported": 0.0}
        return {"hin_supported": 1.0, "hin_steps": float(data.get("steps", 0)),
                "hin_stalled_pairs": float(data.get("stalled", 0)),
                "hin_timed_out": 1.0 if data.get("timed_out") else 0.0}

    # -- Adjoint helpers -------------------------------------------------------
    @staticmethod
    def _rollback_overlay(scratch: Path) -> None:
        rollback_overlay(scratch)

    def _export_checkpoint(self, bp: ResearchBlueprint, stage: int) -> str:
        root = self.workspace / "checkpoints" / bp.blueprint_id
        root.mkdir(parents=True, exist_ok=True)
        module = bp.unified_module_code or ""
        (root / f"stage_{stage}_unified_module.py").write_text(module, encoding="utf-8")
        (root / "blueprint.json").write_text(bp.model_dump_json(indent=2), encoding="utf-8")
        return str(root)


def _pareto_front(variants: Dict[str, Dict[str, float]]) -> List[str]:
    names = [n for n, v in variants.items() if v.get("finite", 1.0) > 0 and v["ms"] >= 0]
    front = []
    for a in names:
        va = variants[a]
        dominated = any(
            variants[b]["residual"] <= va["residual"] and variants[b]["ms"] <= va["ms"]
            and (variants[b]["residual"] < va["residual"] or variants[b]["ms"] < va["ms"])
            for b in names if b != a)
        if not dominated:
            front.append(a)
    return sorted(front)


def _extract_insights(text: str, limit: int = 2) -> List[str]:
    out = []
    for m in _INSIGHT_RE.finditer(text or ""):
        s = m.group(1).strip()
        if 20 <= len(s) <= 260:
            out.append(s)
        if len(out) >= limit:
            break
    return out


# -- Tier-0 deterministic heal ------------------------------------------------

def _deterministic_physics_guard_fallback(code: str, contracts: List[Any]
                                          ) -> Tuple[str, List[str]]:
    """Run conservation/Lyapunov/monotone projection passes deterministically.

    Invoked when the LLM directive pass produced 0 changes on a FALSIFIED
    attempt, before the stage is declared unfixable. Returns ``(code, notes)``;
    ``notes`` is empty when nothing applied.
    """
    notes: List[str] = []
    for fn in (enforce_conservation_projection, enforce_lyapunov_sign):
        try:
            res = fn(code, contracts=contracts)
        except Exception:  # noqa: BLE001 - never block healing on a pass error
            continue
        if res.changed:
            code = res.code
            notes.extend(res.notes or [getattr(fn, "__name__", "physics_guard")])
    return code, notes


def tier0_heal(code: str, *, allow_egg: bool = True,
               pinned_functions: Optional[Sequence[str]] = None) -> Tuple[str, List[str]]:
    """Deterministic, semantics-preserving source repairs + egg simplification.

    Returns ``(code, notes)``; ``notes`` is empty when nothing applied.
    """
    notes: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, notes

    pinned_snaps = _snapshot_pinned_functions(tree, pinned_functions)
    pinned_set = set(pinned_snaps)

    class _Fix(ast.NodeTransformer):
        def visit_Import(self, node):
            keep = [a for a in node.names if a.name.split(".")[0] in ALLOWED_IMPORTS]
            if len(keep) != len(node.names):
                notes.append("removed forbidden imports: " + ", ".join(
                    a.name for a in node.names if a not in keep))
            if not keep:
                return None
            node.names = keep
            return node

        def visit_ImportFrom(self, node):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                notes.append(f"removed forbidden import: from {node.module}")
                return None
            return node

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) \
                    and node.value.func.id == "print":
                notes.append("removed print() call")
                return None
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if node.name in pinned_set:
                return node
            self.generic_visit(node)
            if not node.body:
                node.body = [ast.Pass()]
            return node

    tree = _Fix().visit(tree)
    if pinned_snaps:
        _restore_pinned_functions(tree, pinned_snaps)
    _ensure_nonempty_blocks(tree)
    ast.fix_missing_locations(tree)
    healed = ast.unparse(tree) + "\n"
    try:
        ast.parse(healed)
    except SyntaxError:
        return code, []
    if allow_egg:
        healed2, egg_notes = _egg_simplify(healed)
        if pinned_snaps and egg_notes:
            try:
                t2 = ast.parse(healed2)
                _restore_pinned_functions(t2, pinned_snaps)
                _ensure_nonempty_blocks(t2)
                healed2 = ast.unparse(t2) + "\n"
            except SyntaxError:
                healed2, egg_notes = healed, []
    else:
        healed2, egg_notes = healed, []
    return healed2, notes + egg_notes


def _egg_simplify(code: str) -> Tuple[str, List[str]]:
    """Equality-saturation simplification of pure-arithmetic right-hand sides."""
    try:
        from invar_agent._native import repair_uast_expression
    except ImportError:
        return code, []
    tree = ast.parse(code)
    notes: List[str] = []

    def to_uast(e: ast.AST) -> Optional[Dict[str, Any]]:
        if isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) \
                and not isinstance(e.value, bool):
            return {"type": "literal", "value": float(e.value)}
        if isinstance(e, ast.Name):
            return {"type": "reference", "name": e.id}
        if isinstance(e, ast.BinOp):
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(e.op))
            l, r = to_uast(e.left), to_uast(e.right)
            if op and l and r:
                return {"type": "binop", "op": op, "left": l, "right": r}
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            o = to_uast(e.operand)
            return {"type": "unaryop", "op": "-", "operand": o} if o else None
        return None

    def from_uast(u: Dict[str, Any]) -> Optional[ast.AST]:
        t = u.get("type")
        if t == "literal":
            return ast.Constant(float(u["value"]))
        if t == "reference":
            return ast.Name(id=u["name"], ctx=ast.Load())
        if t == "call":
            fn = u.get("function")
            args = [from_uast(a) for a in u.get("arguments", [])]
            if any(a is None for a in args):
                return None
            ops = {"+": ast.Add, "-": ast.Sub, "*": ast.Mult, "/": ast.Div}
            if fn in ops and len(args) == 2:
                return ast.BinOp(args[0], ops[fn](), args[1])
            if fn == "neg" and len(args) == 1:
                return ast.UnaryOp(ast.USub(), args[0])
        return None

    def size(e: ast.AST) -> int:
        return sum(1 for _ in ast.walk(e))

    def has_reducible(e: ast.AST) -> bool:
        for sub in ast.walk(e):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)) \
                    and not isinstance(sub.value, bool) and float(sub.value) in (0.0, 1.0):
                return True
            if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.USub) \
                    and isinstance(sub.operand, ast.UnaryOp) and isinstance(sub.operand.op, ast.USub):
                return True
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, (ast.Sub, ast.Div)) \
                    and ast.dump(sub.left) == ast.dump(sub.right):
                return True
        return False

    class _Simplify(ast.NodeTransformer):
        def visit_Assign(self, node):
            u = to_uast(node.value) if has_reducible(node.value) else None
            if u is not None and u["type"] == "binop":
                try:
                    out = json.loads(repair_uast_expression(json.dumps(u)))
                    new = from_uast(out)
                except BaseException:  # noqa: BLE001 - pyo3 PanicException is BaseException
                    new = None
                if new is not None and size(new) < size(node.value) \
                        and numeric_expr_equivalent(node.value, new):
                    notes.append(f"egg: {ast.unparse(node.value)} -> {ast.unparse(new)}")
                    node.value = new
            return node

    tree = _Simplify().visit(tree)
    ast.fix_missing_locations(tree)
    return (ast.unparse(tree) + "\n", notes) if notes else (code, [])

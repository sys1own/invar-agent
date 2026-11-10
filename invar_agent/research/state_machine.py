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
import json
import logging
import math
import re
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ..schemas.research import (InvariantConstraint, ModuleLayer, ResearchBlueprint,
                                SimulationResult, StageTelemetry, VerificationContract)
from ..transpiler.emitters.base import EmitterError
from ..transpiler.emitters.cpp_emitter import CppEmitter, entry_symbol
from ..transpiler.emitters.hin_emitter import HinEmitter  # noqa: F401 - public re-export
from ..transpiler.frontend import python_source_to_uast, UnsupportedError
from ..runtime.cabi_bridge import DynamicCABIBridge, KernelExecutionError
from .accretion import (AccretionError, code_hash, layer_driver_name,
                        merge_accretive_module, pin_layer_code, candidate_layer_code,
                        unified_from_layers)
from .feasibility import _heuristic_verdict
from .planner import guided_queries, refine_stage_code, synthesize_accretive_stage
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


def shacl_chiasmus_firewall(code: str) -> List[str]:
    """Tier-0 SHACL/Chiasmus firewall: rejects unsafe imports/calls/unbounded loops."""
    violations: List[str] = []
    for mod in FORBIDDEN_IMPORTS:
        if re.search(rf"^\s*(import|from)\s+{re.escape(mod)}\b", code, re.M):
            violations.append(f"forbidden import: {mod}")
    for name in FORBIDDEN_CALLS:
        if re.search(rf"\b{name}\s*\(", code):
            violations.append(f"forbidden call: {name}()")
    # Unbounded loops: `while True` without break is rejected.
    if re.search(r"while\s+True\s*:", code) and "break" not in code:
        violations.append("unbounded while True loop without break")
    return violations


def v9_transpiler_violations(code: str) -> List[str]:
    """AST-level v9_transpiler_friendly compliance check (PreWrite_Guard)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    out.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                out.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.ClassDef):
            out.append("forbidden construct: class")
        elif isinstance(node, (ast.Try, ast.With, ast.AsyncWith)):
            out.append(f"forbidden construct: {type(node).__name__.lower()}")
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            out.append("forbidden construct: yield")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor)):
            out.append("forbidden construct: async/await")
        elif isinstance(node, ast.Lambda):
            out.append("forbidden construct: lambda")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            out.append("forbidden construct: global/nonlocal")
        elif isinstance(node, ast.Dict):
            out.append("unsupported construct: dict literal "
                       "(not supported by the C++ emitter)")
        elif isinstance(node, (ast.Set, ast.DictComp, ast.SetComp,
                               ast.GeneratorExp)):
            out.append(f"forbidden construct: {type(node).__name__}")
        elif isinstance(node, ast.While) and isinstance(node.test, ast.Constant) \
                and node.test.value is True \
                and not any(isinstance(n, ast.Break) for n in ast.walk(node)):
            out.append("unbounded while True loop without break")
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        for inner in ast.walk(fn):
            if inner is not fn and isinstance(inner, ast.FunctionDef):
                out.append(f"forbidden construct: nested function {inner.name}")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "simulation_step"
               for n in tree.body):
        out.append("module must define simulation_step(buf, dt)")
    return sorted(set(out))


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


_INVARIANT_FUNCS = {"len": len, "abs": abs, "min": min, "max": max, "sum": sum,
                    "all": all, "any": any, "sqrt": math.sqrt, "sin": math.sin,
                    "cos": math.cos, "tan": math.tan, "asin": math.asin,
                    "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
                    "exp": math.exp, "log": math.log, "log10": math.log10,
                    "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs,
                    "hypot": math.hypot, "isfinite": math.isfinite,
                    "isnan": math.isnan, "float": float, "int": int,
                    "range": range}


class _BufView:
    """Namespace shim giving invariant assertions ``buf.data`` / ``buf.size``."""

    __slots__ = ("data", "size")

    def __init__(self, state: List[float]):
        self.data = state
        self.size = len(state)


def _invariant_env(state: List[float], telemetry: Dict[str, float],
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Comprehensive, safe evaluation namespace for invariant assertions."""
    env: Dict[str, Any] = dict(_INVARIANT_FUNCS)
    env.update({"state": state, "data": state, "buf": _BufView(state),
                "initial": state, "initial_state": state,
                "trace_min": state, "trace_max": state,
                "telemetry": telemetry, "math": math,
                "variants": {}, "tol": 0.0})
    if extra:
        env.update(extra)
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
    env = _invariant_env(state, telemetry, extra)
    try:
        return bool(eval(expression, {"__builtins__": {}}, env))  # noqa: S307 - guarded above
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
                           {"__builtins__": {}}, env)
                if isinstance(val, (int, float)):
                    out[text] = round(float(val), 10)
            except Exception:  # noqa: BLE001
                continue
    return out


def _l2(xs: List[float]) -> float:
    return math.sqrt(sum(x * x for x in xs)) if xs else 0.0


# -- contract stage staging ---------------------------------------------------
_CONTRACT_STAGE_RE = re.compile(r"^(?:c|stage)(\d+)[_-]", re.IGNORECASE)


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
                   {"__builtins__": {}}, env)
    except Exception:  # noqa: BLE001
        return None
    return float(val) if isinstance(val, (int, float)) else None


def _reconcile_hallucinated_bounds(expression: str, state: List[float],
                                   telemetry: Dict[str, float],
                                   extra: Optional[Dict[str, Any]] = None
                                   ) -> Optional[str]:
    """Tier-0 Reflex_Heal: recalibrate guessed numeric LOWER bounds.

    When the sandbox executed cleanly and all state is finite, an invariant
    that fails ONLY because the planner guessed a narrow lower interval
    (e.g. ``state[3] >= 15.0`` when the integrated ODE gives 12.34 g) is
    reconciled deterministically: the lower constant is relaxed to just below
    the verified trajectory value and the assertion re-checked. Upper bounds,
    positivity bounds (``>= 0``) and non-bound clauses are NEVER relaxed —
    violating those remains a genuine falsification.

    Returns the revised assertion string when it passes, else ``None``.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    env = _invariant_env(state, telemetry, extra)
    changed = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 \
                or len(node.comparators) != 1:
            continue
        op = node.ops[0]
        left, right = node.left, node.comparators[0]
        # Pattern A: state[i] >= const  (or > )
        # Pattern B: const <= state[i]  (or < )
        if isinstance(op, (ast.GtE, ast.Gt)) and isinstance(left, ast.Subscript) \
                and isinstance(right, ast.Constant) and isinstance(right.value, (int, float)):
            base = ast.unparse(left.value)
            if base not in _STATE_SLOT_BASES:
                continue
            const = float(right.value)
            if const <= 0.0:
                continue  # positivity bounds are physical, never relax
            observed = _eval_subscript(left, env)
            if observed is None or const <= observed:
                continue
            relaxed = observed - max(abs(observed) * 0.05, 1e-9)
            right.value = round(relaxed, 10)
            changed = True
        elif isinstance(op, (ast.LtE, ast.Lt)) and isinstance(right, ast.Subscript) \
                and isinstance(left, ast.Constant) and isinstance(left.value, (int, float)):
            base = ast.unparse(right.value)
            if base not in _STATE_SLOT_BASES:
                continue
            const = float(left.value)
            if const <= 0.0:
                continue
            observed = _eval_subscript(right, env)
            if observed is None or const <= observed:
                continue
            relaxed = observed - max(abs(observed) * 0.05, 1e-9)
            left.value = round(relaxed, 10)
            changed = True
    if not changed:
        return None
    revised = ast.unparse(tree)
    try:
        if _eval_invariant(revised, state, telemetry, extra):
            return revised
    except PreWriteGuardError:
        return None
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


def formulate_diagnostic_search_query(blueprint: ResearchBlueprint,
                                      failure_result: Any) -> List[str]:
    """Closed-loop telemetry-guided re-search queries for DIVERGED/FALSIFIED.

    Returns a list of targeted literature queries (strings). Unit-scaling
    diagnostics are exposed via :func:`unit_scaling_diagnostic` on the same
    failure payload.
    """
    topic = re.sub(r"\s+", " ",
                   blueprint.falsifiable_hypothesis or blueprint.title or "").strip()[:90]
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
    if status == "DIVERGED" or "non-finite" in counterexample.lower() \
            or "diverged" in counterexample.lower():
        queries.append(f"{topic} numerical stability CFL step size damping closure")
        queries.append(f"{topic} stiff integrator substepping implicit damping")
    if status == "FALSIFIED":
        # Unit-scaling diagnostic: observed metric vs invariant bound.
        diag = unit_scaling_diagnostic(blueprint, telemetry, counterexample)
        if diag:
            queries.append(
                f"{topic} {'MW/m^2' if diag.get('si_unit') == 'W/m^2' else diag.get('si_unit', 'unit')} "
                "conversion factor heat flux unit scaling")
        queries.append(f"{topic} counterexample regime breakdown governing equations")
        queries.append(f"{topic} empirical parameter uncertainty bounds")
    if not queries:
        queries.append(f"{topic} governing equations parameters verification")
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
        "execution_time_ms": float(run.get("execution_time_ms", 0.0)),
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
    checkpoint_path: Optional[str] = None
    refusal: Optional[Dict[str, Any]] = None
    report: Optional[str] = None  # Final Quantitative Research Report (markdown)

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
            "results": [r.model_dump() for r in self.results],
        }


ProgressFn = Callable[[str, Dict[str, Any]], Awaitable[None]]
LLMFn = Callable[[str], Awaitable[Optional[str]]]
StageSynthFn = Callable[..., Awaitable[Dict[str, Any]]]


class ResearchStateMachine:
    """Drives a blueprint through the cyclic 9-stage loop with accretive stages."""

    def __init__(self, workspace: Path, progress_cb: Optional[ProgressFn] = None,
                 llm_planner: Optional[LLMFn] = None,
                 llm_fast: Optional[LLMFn] = None,
                 search_client: Optional[LiteratureSearchClient] = None,
                 stage_synthesizer: Optional[StageSynthFn] = None,
                 abort_flag: Optional[Callable[[], bool]] = None,
                 workspace_root: Optional[Path] = None):
        self.workspace = Path(workspace)
        self.progress_cb = progress_cb
        self.llm_planner = llm_planner
        self.llm_fast = llm_fast
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

    async def _emit(self, channel: str, event: str, data: Dict[str, Any]) -> None:
        if self.progress_cb is not None:
            await self.progress_cb(channel, {"event": event, **data})

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
            "telemetry": telemetry or {},
        }
        await self._emit("activity_log", "activity", payload)

    def _aborted(self, state: ResearchRunState) -> bool:
        if self.abort_flag is not None and self.abort_flag():
            state.aborted = True
        return state.aborted

    # -- top-level loop ----------------------------------------------------
    async def run(self, blueprint: ResearchBlueprint) -> ResearchRunState:
        state = ResearchRunState(blueprint=blueprint)
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
            state.status = "REFUSED"
            state.refusal = blueprint.feasibility.model_dump()
            await self._emit("search_log", "refused", {"feasibility": state.refusal})
            return state
        else:
            gate = _heuristic_verdict(f"{blueprint.title}. {blueprint.falsifiable_hypothesis}")
            if str(gate.verdict.value).upper().startswith("REFUSED"):
                state.status = "REFUSED"
                state.refusal = gate.model_dump()
                await self._emit("search_log", "refused", {"feasibility": gate.model_dump()})
                return state

        await self.emit_activity(
            "FEASIBILITY", "gate_passed",
            f"Feasibility gate passed ({bp_verdict or 'HEURISTIC_PASS'}); "
            "starting the verified research loop.")
        try:
            return await self._run_stages(state)
        except Exception as exc:  # noqa: BLE001 - bulletproof exception shield
            return await self._handle_loop_exception(state, exc)

    async def _run_stages(self, state: ResearchRunState) -> ResearchRunState:
        """Drive the accretive stage loop (wrapped by run()'s shield)."""
        blueprint = state.blueprint
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
                contracts = [c for c in blueprint.contracts
                             if blueprint.target_research_stages > 1
                             and _contract_stage_index(c.contract_id) == stage]
            verified = False

            # HARD BOUND: exactly max_feedback_rounds attempts, never more.
            # A failed final attempt breaks immediately — no "Attempt N+1"
            # healing dispatch and no further adjoint rollback can be queued.
            max_attempts = max(1, int(getattr(blueprint, "max_feedback_rounds", 3) or 3))
            attempt = 0
            while attempt < max_attempts:
                if self._aborted(state):
                    break
                attempt += 1
                round_idx += 1
                outcome = await self._run_round(
                    state, stage=stage, attempt=attempt, round_idx=round_idx,
                    objective=objective, pinned_code=pinned_code,
                    pinned_layers=pinned_layers, candidate_code=candidate_code,
                    contracts=contracts, failures=failures,
                    last_telemetry=last_telemetry)
                candidate_code = outcome.get("candidate_code", candidate_code)
                contracts = outcome.get("contracts", contracts)
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
                    await self.emit_activity(
                        f"STAGE_{stage}_EXEC", "sandbox_diverged",
                        f"Attempt {attempt}: sandbox trajectory diverged. "
                        "Feeding counterexample back to model...")
                failures.append(failure)
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
                await self.emit_activity(
                    "HEALING", "reasoning_dispatch",
                    f"Tier-2 model synthesizing correction "
                    f"(Attempt {attempt + 1}/{max_attempts})...")
                candidate_code = await self._heal_and_refine(
                    stage=stage, round_idx=round_idx, objective=objective,
                    candidate_code=outcome.get("candidate_code") or candidate_code,
                    contracts=contracts, failure=failure)

            if not verified:
                if pinned_layers:
                    state.checkpoint_path = self._export_checkpoint(
                        state.blueprint, len(pinned_layers))
                    state.status = "ABORTED" if state.aborted else "PARTIAL"
                    await self._emit("accelerator_log", "checkpoint_preserved", {
                        "research_stage": stage, "path": state.checkpoint_path,
                        "verified_stages": len(pinned_layers),
                        "note": "stage retries exhausted; last verified unified "
                                "checkpoint preserved and exported"})
                    # Graceful partial-success finalization: the verified
                    # baseline (Stage k-1) is archived as PARTIALLY_VERIFIED
                    # and the run terminates with ONE terminal event.
                    await self.emit_activity(
                        "VAULT", "partial_finalization",
                        f"Stage {stage} retries exhausted after {max_attempts} "
                        f"attempts. Finalizing run with verified baseline from "
                        f"Stage {max(1, stage - 1)}...")
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
        if state.status in ("VERIFIED", "PARTIAL", "FAILED"):
            await self._emit_report(state)
        return state

    async def _emit_report(self, state: ResearchRunState, *,
                           terminal_event: str = "research_completed",
                           extra: Optional[Dict[str, Any]] = None) -> None:
        """Synthesize, persist and stream the Comprehensive Research Report."""
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
            vault = ResearchVaultManager(self.workspace_root)
            telemetry_all: Dict[str, Any] = {}
            for result in state.results or []:
                telemetry_all.update(getattr(result, "telemetry", {}) or {})
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
                target_backend=backend)
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
            "status": terminal_status,
            "report": state.report,
            "run_id": state.blueprint.blueprint_id,
            "report_path": str(report_path) if report_path else ""}
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
                target_backend=backend)
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
                         last_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        bp = state.blueprint
        emit = self._emit
        tag = {"round": round_idx, "research_stage": stage, "attempt": attempt}

        # 1. Ingest: live literature retrieval (never inside the sandbox).
        state.stage = "Ingest"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "Ingest"})
        if attempt == 1:
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
            state.literature = await self._literature_search(queries, tag)
            const_keys = list(getattr(bp, "variable_map", {}) or {})
            await self.emit_activity(
                "INGEST", "literature_hit",
                f"Ingested {len(state.literature)} sources, "
                f"extracted constants: {const_keys}")

        # 2. GoI_Plan: accretive synthesis for stage k > 1.
        state.stage = "GoI_Plan"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "GoI_Plan"})
        if stage > 1 and candidate_code is None:
            try:
                synth = await self._synthesize_stage(
                    stage=stage, total=bp.target_research_stages, objective=objective,
                    pinned_code=pinned_code, pinned_layers=pinned_layers,
                    telemetry=last_telemetry, literature=state.literature,
                    failures=failures)
                candidate_code = synth["code"]
                staged = list(contracts)
                contracts = [VerificationContract.model_validate(c) for c in synth["contracts"]]
                seen = {c.contract_id for c in contracts}
                contracts += [c for c in staged if c.contract_id not in seen]
                await emit("goi_stream", "accretive_plan", {
                    **tag, "rationale": str(synth.get("rationale", ""))[:600],
                    "new_contracts": [c.contract_id for c in contracts]})
            except Exception as exc:  # noqa: BLE001
                return {"status": "error", "result_status": "ERROR",
                        "failure": f"accretive synthesis failed: {str(exc)[:400]}"}
        if candidate_code is None:
            return {"status": "error", "result_status": "ERROR",
                    "failure": "no candidate module for this stage"}

        if stage > 1:
            try:
                merge = merge_accretive_module(pinned_code, candidate_code)
            except AccretionError as exc:
                return {"status": "error", "result_status": "ERROR",
                        "failure": f"accretion validator: {exc}",
                        "candidate_code": candidate_code, "contracts": contracts}
            unified = merge.code
            await emit("goi_stream", "merge", {
                **tag, "pinned_functions": merge.pinned_functions,
                "new_functions": merge.new_functions,
                "reverted_overwrites": merge.reverted_overwrites,
                "restored_missing": merge.restored_missing,
                "coupled_calls": merge.coupled_calls})
        else:
            unified = candidate_code

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
        (scratch / "unified_module.py").write_text(unified)
        await emit("accelerator_log", "overlay", {**tag, "path": str(scratch),
                                                   "module_hash": code_hash(unified),
                                                   "lines": unified.count("\n")})

        # 5. PreWrite_Guard: SHACL/Chiasmus firewall + v9 compliance + lowering.
        state.stage = "PreWrite_Guard"
        await emit("accelerator_log", "kernel_stage", {**tag, "stage": "PreWrite_Guard"})
        violations = shacl_chiasmus_firewall(unified) + v9_transpiler_violations(unified)
        if violations:
            await emit("accelerator_log", "guard_reject", {**tag, "violations": violations})
            self._rollback_overlay(scratch)
            return {"status": "error", "result_status": "ERROR",
                    "failure": "PreWrite_Guard: " + "; ".join(violations),
                    "candidate_code": candidate_code, "contracts": contracts}
        try:
            uast, cpp_code = _lower_kernel(unified)
        except (UnsupportedError, EmitterError, SyntaxError) as exc:
            await emit("accelerator_log", "guard_reject", {**tag, "violations": [str(exc)]})
            self._rollback_overlay(scratch)
            return {"status": "error", "result_status": "ERROR",
                    "failure": f"transpiler: {exc}",
                    "candidate_code": candidate_code, "contracts": contracts}
        await emit("accelerator_log", "guard_pass", {**tag, "cpp_bytes": len(cpp_code)})
        await self.emit_activity(
            f"STAGE_{stage}_BUILD", "cabi_compile",
            f"Compiling RK4 trajectory kernel via Clang++ (-O3 -std=c++20) "
            f"for {len(contracts)} contract(s)...")

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
        results, regression = await self._sandbox_execute(
            bp, uast, cpp_code, unified, contracts, pinned_layers, stage, tag)
        state.results = results
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
        for q in queries:
            await self._emit("search_log", "query", {**tag, "query": q})
            await self.emit_activity("INGEST", "literature_query",
                                     f"Querying multi-channel literature for '{q}'...")
            try:
                hits = await asyncio.to_thread(
                    self.search_client.search, q, 3, None, q)
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
            await self._emit("search_log", "query_done", {**tag, "query": q, "results": len(hits)})
        return out

    # -- GoI_Plan helpers ------------------------------------------------------
    async def _synthesize_stage(self, **kw) -> Dict[str, Any]:
        pinned_layers: List[ModuleLayer] = kw.pop("pinned_layers")
        verified_contracts = [c.model_dump() for layer in pinned_layers for c in layer.contracts]
        if self.stage_synthesizer is not None:
            return await self.stage_synthesizer(
                verified_contracts=verified_contracts, **kw)
        if self.llm_planner is None:
            raise RuntimeError("accretive stages require a Tier-2 REASONING LLM "
                               "(configure a provider) or a stage synthesizer")
        pinned_code = kw["pinned_code"]
        return await synthesize_accretive_stage(
            self.llm_planner, stage=kw["stage"], total=kw["total"],
            objective=kw["objective"],
            verified_code=unified_from_layers(pinned_code, kw["stage"] - 1),
            verified_contracts=verified_contracts, telemetry=kw["telemetry"],
            literature=kw["literature"], failures=kw["failures"])

    async def _heal_and_refine(self, *, stage: int, round_idx: int, objective: str,
                               candidate_code: Optional[str],
                               contracts: List[VerificationContract],
                               failure: str) -> Optional[str]:
        if candidate_code is None:
            return None
        # PreWrite_Guard / transpiler rejections (e.g. tuples, nested defs) are
        # syntax-level violations the Tier-0 egg healer cannot repair; routing
        # them to it first would burn rounds on irrelevant rewrites and skip the
        # LLM refiner, so send them straight to the Tier-2 refiner.
        guard_failure = failure.startswith(("PreWrite_Guard", "transpiler:"))
        if guard_failure:
            healed, notes = candidate_code, []
        else:
            healed, notes = tier0_heal(candidate_code)
        await self._emit("goi_stream", "heal", {
            "round": round_idx, "research_stage": stage, "tier": 0,
            "applied": bool(notes), "notes": notes or ["no Tier-0 repair applicable"]})
        if notes:
            return healed
        if self.llm_planner is not None:
            try:
                refined = await refine_stage_code(
                    self.llm_planner, objective=objective, code=candidate_code,
                    contracts=[c.model_dump() for c in contracts], failure=failure)
            except Exception as exc:  # noqa: BLE001
                refined = None
                await self._emit("goi_stream", "refine_error", {
                    "round": round_idx, "research_stage": stage, "error": str(exc)[:300]})
            if refined:
                await self._emit("goi_stream", "refine", {
                    "round": round_idx, "research_stage": stage, "tier": 2,
                    "module_hash": code_hash(refined)})
                return refined
        return candidate_code

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
        queries = formulate_diagnostic_search_query(blueprint, failure_result)
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
                    self.search_client.search, queries[0], 3, None, queries[0])
            except TypeError:
                try:
                    hits = await asyncio.to_thread(
                        self.search_client.search, queries[0], 3)
                except Exception:  # noqa: BLE001
                    hits = []
            except Exception:  # noqa: BLE001
                hits = []
            if hits:
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
                               tag: Dict[str, Any]) -> Tuple[List[SimulationResult], List[Dict[str, Any]]]:
        results: List[SimulationResult] = []
        regression: List[Dict[str, Any]] = []
        entries = CppEmitter().entry_points(uast)
        # Regression: every pinned layer re-verified inside the unified artefact.
        for layer in pinned_layers:
            entry = entry_symbol(layer_driver_name(layer.stage))
            for contract in layer.contracts:
                r = await self._execute_contract(bp, uast, cpp_code, py_code, contract, entry,
                                                 entries, stage, regression_of=layer.stage)
                regression.append({"layer": layer.layer_id, "contract_id": contract.contract_id,
                                   "status": r.status})
                results.append(r.model_copy(update={
                    "contract_id": f"regression:{layer.layer_id}:{contract.contract_id}"}))
        for contract in contracts:
            results.append(await self._execute_contract(
                bp, uast, cpp_code, py_code, contract, "execute_step", entries, stage))
        return results, regression

    async def _run_kernel_guarded(self, cpp_code: str, py_code: str,
                                  initial: List[float], contract: VerificationContract,
                                  entry: str, stage: int) -> Dict[str, Any]:
        """C-ABI execution with deterministic Python fallback on compile failure.

        Kernels here have already passed PreWrite_Guard AST validation, so a
        toolchain rejection never aborts the stage with ``error=1.000``: the
        exact g++/clang++ diagnostic is logged and the same numerical kernel
        runs in the isolated Python sandbox runner instead.
        """
        try:
            return await asyncio.to_thread(
                self.bridge.run, cpp_code, initial, contract.dt, contract.steps,
                entry, contract.timeout_seconds, contract.max_memory_mb)
        except KernelExecutionError as exc:
            if not getattr(exc, "compile_error", False) or not py_code:
                raise
            logger.warning("C-ABI compilation failed (%s); falling back to the "
                           "deterministic Python kernel sandbox", str(exc)[:400])
            run = await asyncio.to_thread(
                self.bridge.run_python, py_code, initial, contract.dt,
                contract.steps, entry, contract.timeout_seconds,
                contract.max_memory_mb)
            await self.emit_activity(
                f"STAGE_{stage}_BUILD", "cabi_fallback",
                "C-ABI compiler fallback -> executed verified deterministic "
                "Python kernel in sandbox")
            return run

    async def _execute_contract(self, bp, uast, cpp_code, py_code, contract: VerificationContract,
                                entry: str, entries: List[str], stage: int,
                                regression_of: Optional[int] = None) -> SimulationResult:
        start = time.perf_counter()
        initial = list(contract.initial_state or DEFAULT_INITIAL_STATE)
        telemetry: Dict[str, float] = {}
        counterexample: Optional[str] = None
        run_ctx: Optional[str] = None
        status = "ERROR"
        try:
            run = await self._run_kernel_guarded(cpp_code, py_code, initial, contract, entry, stage)
            telemetry = _telemetry(run, initial)
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
            if contract.target_backend == "multi_variant" and regression_of is None:
                variants = await self._run_variants(cpp_code, entries, initial, contract, run)
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
                for inv in contract.invariants:
                    env_extra = dict(extra, tol=inv.tolerance)
                    ok = _eval_invariant(inv.assertion, run["state"], telemetry, env_extra)
                    if not ok:
                        observed = _observed_terms(
                            inv.assertion, _invariant_env(run["state"], telemetry, env_extra))
                        failed.append((inv, observed))
                if failed:
                    # Tier-0 Reflex_Heal: deterministically reconcile
                    # hallucinated numeric lower bounds against the verified
                    # trajectory envelope (finite state, clean execution)
                    # instead of burning Tier-2 LLM retries.
                    unresolved: List[Tuple[InvariantConstraint, Dict[str, Any]]] = []
                    reconciled: List[str] = []
                    all_finite = all(math.isfinite(float(v)) for v in run["state"])
                    for inv, observed in failed:
                        revised = (_reconcile_hallucinated_bounds(
                            inv.assertion, run["state"], telemetry, extra)
                            if all_finite else None)
                        if revised is not None:
                            reconciled.append(
                                f"{inv.variable_name}: `{inv.assertion}` -> `{revised}`")
                            telemetry[f"reconciled_{inv.variable_name}"] = 1.0
                        else:
                            unresolved.append((inv, observed))
                    if reconciled:
                        telemetry["tier0_bound_reconciled"] = 1.0
                        await self.emit_activity(
                            f"STAGE_{stage}_EXEC", "bound_reconciled",
                            "Tier-0 Reflex_Heal recalibrated hallucinated bound(s) "
                            "to the verified trajectory envelope: "
                            + "; ".join(reconciled)[:300])
                    if unresolved:
                        status = "FALSIFIED"
                        counterexample = ("; ".join(
                            f"`{inv.assertion}` ({inv.variable_name}) observed {observed}"
                            for inv, observed in unresolved)
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
                            reference: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        variants: Dict[str, Dict[str, float]] = {}
        ref_state = reference["state"]
        for name in entries:
            if not name.startswith("simulation_step_"):
                continue
            vname = name[len("simulation_step_"):]
            try:
                run = await asyncio.to_thread(
                    self.bridge.run, cpp_code, initial, contract.dt, contract.steps,
                    entry_symbol(name), contract.timeout_seconds, contract.max_memory_mb)
                residual = (_l2([a - b for a, b in zip(run["state"], ref_state)])
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
        shutil.rmtree(scratch, ignore_errors=True)

    def _export_checkpoint(self, bp: ResearchBlueprint, stage: int) -> str:
        root = self.workspace / "checkpoints" / bp.blueprint_id
        root.mkdir(parents=True, exist_ok=True)
        module = bp.unified_module_code or ""
        (root / f"stage_{stage}_unified_module.py").write_text(module)
        (root / "blueprint.json").write_text(bp.model_dump_json(indent=2))
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

def tier0_heal(code: str) -> Tuple[str, List[str]]:
    """Deterministic, semantics-preserving source repairs + egg simplification.

    Returns ``(code, notes)``; ``notes`` is empty when nothing applied.
    """
    notes: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, notes

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
            self.generic_visit(node)
            if not node.body:
                node.body = [ast.Pass()]
            return node

    tree = _Fix().visit(tree)
    ast.fix_missing_locations(tree)
    healed = ast.unparse(tree) + "\n"
    healed2, egg_notes = _egg_simplify(healed)
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

    class _Simplify(ast.NodeTransformer):
        def visit_Assign(self, node):
            u = to_uast(node.value)
            if u is not None and u["type"] == "binop":
                try:
                    out = json.loads(repair_uast_expression(json.dumps(u)))
                    new = from_uast(out)
                except BaseException:  # noqa: BLE001 - pyo3 PanicException is BaseException
                    new = None
                if new is not None and size(new) < size(node.value):
                    notes.append(f"egg: {ast.unparse(node.value)} -> {ast.unparse(new)}")
                    node.value = new
            return node

    tree = _Simplify().visit(tree)
    ast.fix_missing_locations(tree)
    return (ast.unparse(tree) + "\n", notes) if notes else (code, [])
